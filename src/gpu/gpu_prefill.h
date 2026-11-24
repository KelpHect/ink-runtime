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
#include <functional>
#include <string>
#include <unordered_map>
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
    uint64_t uploads = 0;
    // per-helper host emit cost (ns)
    uint64_t op_ns = 0, cp_ns = 0, sync_ns = 0;
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
  // Visit is split like GpuGraph's emit_visit_a/b: _a covers the visit head
  // through the router plus a T*8 router-id readback into ids_host_;
  // _b is the MoE tail (slice copies for staged layers, expert GEMVs,
  // conv, residual, dumps). Between them the chunk driver closes the cb,
  // reads the ids and reopens — see prefill().
  void emit_visit_seq_a(int i, int p, int64_t pos0, int64_t base, int t,
                        bool last);
  void emit_visit_seq_b(int i, int p, int t, bool last);
  // One staging wave [sbase,sbase+sw) of a staged visit on the compute
  // queue: wave gate/up GEMVs + serial accums; at sbase+sw==8 the visit
  // tail (mlp conv, residual add, out dump). Called by _b in the build
  // pass and by the driver's interleaved submit loop at record time.
  void emit_moe_wave(int i, int p, int t, bool last, uint32_t sbase,
                    uint32_t sw);
  // 4 vkCmdCopyBuffer x T*sw slots for one wave [sbase,sbase+sw):
  // expert ids[r*8+s] -> staged slot r*sw+(s-sbase).
  void stage_seq_copies(const GpuGraph::Lw& L, int t, uint32_t sbase,
                        uint32_t sw);
  // on_stage: called between _a and _b for each host-resident, staging-live
  // visit (record mode only); submits the cb, fills stage_ids64_ from
  // ids_host_ and runs the interleaved transfer+compute wave submits.
  // Returns false on error (prefill aborts).
  void emit_chunk_seq(
      int64_t pos0, int64_t base, int t, bool last,
      const std::function<bool(int, int, int, bool)>& on_stage);

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

  // host-resident expert staging: T*8 slots x the max per-expert slice of
  // each tensor (slot index = row*8 + topk slot). ids_host_ carries the
  // router's per-row top-k ids back to the host at the cb boundary.
  ink::vk::Buffer stg13w_seq_, stg13s_seq_, stg2w_seq_, stg2s_seq_;
  ink::vk::Buffer ids_host_;
  int32_t stage_ids64_[16 * 8] = {};  // last readback of ridx_seq_ (T*8)
  // staged slots per row (wave width): alloc ladder tries 8,4,2,1 — the
  // first size that fits VRAM wins; 0/false when nothing fits and
  // host-resident layers keep the original *_seq host-read path.
  uint32_t stage_sw_ = 8;
  bool stage_seq_ok_ = false;
  // Wave-submit chain: one timeline semaphore serializes the alternating
  // transfer/copy-queue and compute-queue wave submits GPU-side (each
  // submit waits stg_ctr_ then signals ++stg_ctr_), so the host only
  // blocks once per staged visit — on the router-id readback fence.
  // VK_NULL_HANDLE when timeline semaphores are unavailable (sync path).
  VkSemaphore sem_stg_ = VK_NULL_HANDLE;
  uint64_t stg_ctr_ = 0;
  // merge_tail_: keep the last wave's compute cb open so the following
  // ops (next visit's _a + ids copy) record into it — one fewer submit per
  // staged visit. EXPERIMENTAL: currently produces a real divergence
  // (hidden_final degrades after decode starts) — left OFF.
  bool merge_tail_ = false;  // INK_MERGE_TAIL=1 re-enables

  // seq pipelines (owned here); elementwise/single-row kernels reuse the
  // graph's pipes via the friend access.
  ink::vk::ComputePipeline p_emb_seq_, p_rms_seq_, p_mvf_seq_, p_mvb_seq_,
      p_mv8_seq_, p_cvs_seq_, p_cvd_seq_, p_rel_seq_, p_sco_seq_, p_smx_seq_,
      p_acc_seq_, p_ares_seq_, p_msg_seq_, p_rtr_seq_, p_moe_seq_, p_l2_seq_,
      p_log_seq_, p_nv4_seq_, p_mx8_seq_,
      p_nv4st_seq_, p_mx8st_seq_, p_moest_seq_;  // staged-slot variants

  std::vector<ink::vk::DescSet> sites_;
  size_t site_cursor_ = 0;
  bool building_ = false;
  ink::vk::Recorder* rec_ = nullptr;

  // ---- INK_GPU_TS=1: GPU timestamps inside the recorded chunk cb ----
  // Same pattern as GpuGraph: one timestamp at cb start plus one after each
  // recorded op; labels are built once (first emit) and deltas summed across
  // submissions. Everything is inert unless the env var is set.
  void ts_mark(const char* name);
  void ts_collect();
  void ts_report() const;
  const char* ts_name(const ink::vk::ComputePipeline& p) const;
  struct TsSeg {
    std::string label;   // "<scope>/<kernel>" e.g. "v17/matvec_nvfp4_seq"
    double ms = 0;       // summed GPU time across collected submissions
  };
  bool ts_en_ = false;
  bool ts_off_ = false;  // suppress marks (copies on the transfer queue)
  VkQueryPool ts_pool_ = VK_NULL_HANDLE;
  uint32_t ts_cap_ = 0;    // pool slots allocated
  uint32_t ts_next_ = 0;   // next query slot in the current cb (slot0=start)
  uint32_t ts_segbase_ = 0;  // ts_segs_ index the current cb's marks land on
  double ts_period_ = 0;   // ns per timestamp tick
  uint64_t ts_runs_ = 0;   // submissions collected
  double ts_gpu_ms_ = 0;   // summed busy ms (seg deltas)
  std::vector<uint64_t> ts_raw_;  // query result scratch
  std::vector<TsSeg> ts_segs_;    // one per timestamp boundary
  std::unordered_map<const ink::vk::ComputePipeline*, const char*>
      pipe_names_;                     // pipe address -> kernel name
  std::unordered_map<std::string, int> ts_dup_;  // label dedup, first emit
  std::string cur_scope_;              // "v<vidx>" / "pre" / "fin"
};

}  // namespace ink::gpu
