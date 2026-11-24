// gpu_prefill.h — batched-chunk prompt prefill for the gpu backend.
//
// Row-batched sibling of GpuGraph::emit_token: processes a chunk of T prompt
// positions through all 48 visits in ONE recording, all activations widened
// to T rows. Shares the GpuGraph's context, allocator, weights, pipelines
// and the persistent state (kvk_/kvv_ KV caches, hist_ conv state) so decode
// continues through the untouched single-token path. Semantics mirror
// src/model/graph.cpp op-for-op — see docs/PREFILL.md.
//
// Design invariants (same as GpuGraph): every execution-time decision is
// device data (router idx/w buffers, toks/taus host buffers); descriptor
// bindings are position-invariant so a single build pass materializes the
// desc-set stream replayed per chunk.
#pragma once

#include "gpu/gpu_graph.h"

#include "vk/cmd.h"
#include "vk/mem.h"
#include "vk/pipe.h"
#include "vk/vk.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ink::gpu {

class GpuPrefill {
 public:
  // g must be fully initialized. chunk = max rows per chunk (clamped 1..16).
  bool init(GpuGraph& g, int chunk, std::string* err);
  void shutdown();
  ~GpuPrefill() { shutdown(); }

  // Run tokens[0..P) as the prompt at positions 0..P-1. On success the
  // graph's kv/conv state, readback (taps, visit dumps, logits) and
  // logits() are identical to P sequential forward_token calls starting
  // from reset(); returns false on error (caller may fall back to the
  // sequential path). Fresh sequence only — call after g.reset().
  bool prefill(const int* tokens, int64_t P);

  const std::string& error() const { return err_; }
  bool ok() const { return ok_; }
  int chunk() const { return T_; }

  struct Stats {
    uint64_t submits = 0, dispatches = 0, copies = 0, barriers = 0;
    double record_ms = 0, wait_ms = 0;
  } stats;

 private:
  using Bnd = GpuGraph::Bnd;

  // Dual-mode emit helpers (own site stream, independent of GpuGraph's).
  const ink::vk::DescSet& dset(const ink::vk::ComputePipeline& p,
                               std::initializer_list<Bnd> bs);
  void op(const ink::vk::ComputePipeline& p, std::initializer_list<Bnd> bs,
          const void* pc, uint32_t pcb, uint32_t gx, uint32_t gy = 1,
          uint32_t gz = 1);
  void mv_seq(const GpuGraph::WB& w, Bnd x, const ink::vk::Buffer& o, int64_t M,
              int64_t K, int64_t xs, int t);
  void cp(const ink::vk::Buffer& s, VkDeviceSize so, const ink::vk::Buffer& d,
          VkDeviceSize dof, VkDeviceSize sz);
  void sync();
  Bnd h3b(int slot, int stream) const;  // row-major conv history slice

  void conv_seq_static(const ink::vk::Buffer& x, const ink::vk::Buffer& w,
                       int slot, int stream, int64_t C,
                       const ink::vk::Buffer& out, int t, bool last);
  void conv_seq_dyn(const ink::vk::Buffer& x, const ink::vk::Buffer& sw,
                    const ink::vk::Buffer& dyn, int slot, int stream,
                    int64_t C, const ink::vk::Buffer& out, int t, bool last);
  // pos0 = absolute position of row 0 (KV rows, attention masks);
  // base = position of row 0 relative to the toks/taus staging block
  // (== pos0 - block_start; identical to pos0 while P <= pmax_).
  void emit_visit_seq(int i, int p, int64_t pos0, int64_t base, int t,
                      bool last);
  void emit_chunk_seq(int64_t pos0, int64_t base, int t, bool last);

  GpuGraph* g_ = nullptr;
  std::string err_;
  bool ok_ = false;
  bool dbg_en_ = false;
  int T_ = 8;       // rows per chunk
  int64_t pmax_ = 1024;  // tokens capacity of the host staging buffer

  // host-side per-chunk counters (shared by all rows — same depth)
  int nblock_ = 0, pre_nblock_ = 0;

  // T-row activation buffers (all device-local f32)
  ink::vk::Buffer partial_seq_, block_seq_, pre_partial_seq_, pre_block_seq_,
      bank_seq_;
  ink::vk::Buffer reads_seq_, msgs_seq_, mgated_seq_, mlpegated_seq_,
      inbuf_seq_;
  ink::vk::Buffer q_seq_, kraw_seq_, vraw_seq_, kout_seq_, vout_seq_, r_seq_,
      tk_seq_, tv_seq_, rel_seq_, probs_seq_, grate_seq_, sink_seq_, o_seq_;
  ink::vk::Buffer attn_out_seq_, a_out_seq_, residual_seq_, m_in_seq_, mn_seq_,
      moe_out_seq_, mo_seq_;
  ink::vk::Buffer ridx_seq_, rw_seq_, gate_seq_, up_seq_;
  ink::vk::Buffer h3_seq_;        // [48*4][3*H] row-major conv history
  ink::vk::Buffer tail_blk_;      // [5*H] last-row block slice for the tail
  ink::vk::Buffer toks_tau_;      // HOST: [pmax]i32 toks ++ [pmax]f32 taus

  // seq pipelines (owned here); elementwise/single-row kernels reuse the
  // graph's pipes via the friend access.
  ink::vk::ComputePipeline p_emb_seq_, p_rms_seq_, p_mvf_seq_, p_mvb_seq_,
      p_mv8_seq_, p_cvs_seq_, p_cvd_seq_, p_rel_seq_, p_sco_seq_, p_smx_seq_,
      p_acc_seq_, p_ares_seq_, p_msg_seq_, p_rtr_seq_, p_moe_seq_, p_l2_seq_,
      p_log_seq_, p_nv4_seq_, p_mx8_seq_;

  std::vector<ink::vk::DescSet> sites_;
  size_t site_cursor_ = 0;
  bool building_ = false;
  ink::vk::Recorder* rec_ = nullptr;
};

}  // namespace ink::gpu
