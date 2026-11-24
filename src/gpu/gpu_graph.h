// gpu_graph.h — device-resident mirror of Graph::forward_token (P2a).
// All activations/caches/scratch live in Vulkan buffers on device; the whole
// token forward (48 visits) is recorded into ONE vk::Recorder and submitted
// once per token. Dense weights are uploaded device-local (native dtype, or
// decoded to fp32 for f32-structured kernels); expert tensors are mirrored
// host-visible and streamed over PCIe (same policy as VkBackend::host_buf).
// The CPU Graph stays the correctness reference; the golden gate arbitrates.
#pragma once

#include "io/config.h"
#include "io/safetensors.h"
#include "vk/cmd.h"
#include "vk/mem.h"
#include "vk/pipe.h"
#include "vk/vk.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ink::shaders {
struct GpuParamsRow;  // bindings.h — defined at include site in the .cpp
}

namespace ink::gpu {

class GpuGraph {
  // GpuPrefill (src/gpu/gpu_prefill.*) shares this object's context,
  // allocator, weights, pipelines and the kv/conv state buffers; it is the
  // row-batched (T-token chunk) sibling of emit_token for prompt prefill.
  friend class GpuPrefill;

 public:
  // ctx/alloc are caller-owned and must outlive the graph. ctx_cap = KV
  // capacity in tokens (the --ctx flag). shader_dir holds the .spv set.
  bool init(const ModelConfig& c, const WeightStore& w, ink::vk::Context& ctx,
            ink::vk::Allocator& alloc, const std::string& shader_dir,
            int64_t ctx_cap, std::string* err);
  void shutdown();
  ~GpuGraph() { shutdown(); }

  void reset();  // new sequence: clears per-slot counters + block depth

  const std::vector<float>& forward_token(int token, int64_t pos);
  const std::vector<float>& logits() const { return logits_host_; }

  // Golden-gate readback (valid until the next forward_token):
  //   tap(0..2) = visit-input hidden at vidx {5,23,41}; tap(3) = h_fin
  //   (final normed — spec: "id 48 = the final normed output").
  //   visit_dump(vidx, 0/1) = partial at visit input/output (48 visits).
  const float* tap(int i) const;
  const float* visit_dump(int vidx, int phase) const;
  const float* dbg(int row) const;  // visit0 stage dump rows (debug)
  // Raw bf16 unembed SSBO (model.llm.unembed.weight bytes) — shared
  // read-only with the MTP draft head so it needs no 883MB second copy.
  const ink::vk::Buffer& unembed_buffer() const { return unembw_; }

  struct Stats {
    uint64_t submits = 0, dispatches = 0, copies = 0, barriers = 0;
    uint64_t uploads = 0, upload_bytes = 0;
    double record_ms = 0, wait_ms = 0;
  } stats;

  const std::string& error() const { return err_; }

  // descriptor triple (buffer, byte off, byte size; sz==0 = whole buffer)
  struct Bnd {
    const ink::vk::Buffer* b;
    VkDeviceSize off, sz;
  };

 private:
  struct Pipe {
    ink::vk::ComputePipeline p;
    const char* name = "";  // spv name, for INK_GPU_TS labels
  };
  struct WB {  // matvec weight: uploaded buffer + dtype-picked pipeline
    const ink::vk::Buffer* b = nullptr;
    const Pipe* p = nullptr;
  };
  struct Lw {
    WB wq, wk, wv, wo, gate, sink, wr;                  // bf16 matvecs
    const ink::vk::Buffer *relproj = nullptr, *qclip = nullptr,
                          *rclip = nullptr;            // f32
    const ink::vk::Buffer *kconv_w = nullptr,
                          *vconv_w = nullptr;          // sliding f32 [C,4]
    WB kconv_wp, vconv_wp;                             // full bf16 [32,2304]
    const ink::vk::Buffer *kconv_sw = nullptr,
                          *vconv_sw = nullptr;         // full f32 [4,C]
    const ink::vk::Buffer* norms[3] = {};              // f32 gammas
    const ink::vk::Buffer *attn_sconv = nullptr, *mlp_sconv = nullptr,
                          *mlp_norm = nullptr, *router_w = nullptr,
                          *router_b = nullptr;
    const ink::vk::Buffer* ctrl[4][3] = {};  // [0]=down bf16, [1,2]=f32
    // expert tensors: host-visible mirrors (or dummy4_ when absent on mxfp8)
    const ink::vk::Buffer *w13 = nullptr, *w13s = nullptr, *w13a = nullptr,
                          *w13o = nullptr, *w2 = nullptr, *w2s = nullptr,
                          *w2a = nullptr, *w2o = nullptr;
    bool full = false, mxfp8 = false;
    // experts_dev: all four big expert tensors (w13/w13s/w2/w2s) landed
    // device-local at init -> the mslot shaders read them directly. When
    // false, decode stages the selected experts' slices into stg_*_ via
    // vkCmdCopyBuffer (bytes per slot = sl* below) and runs the *_stage
    // shader variants that index weights by slot instead of expert id.
    bool experts_dev = false;
    int64_t w13R = 0, w2R = 0;
    int64_t sl13w = 0, sl13s = 0, sl2w = 0, sl2s = 0;  // bytes per expert
  };

  bool load_pipe(const char* name, uint32_t bindings, uint32_t pc_bytes,
                 Pipe& out);
  const ink::vk::Buffer* dev_w(const TensorView& t);   // native-dtype upload
  const ink::vk::Buffer* dev_f32(const TensorView& t); // decode -> f32 upload
  const ink::vk::Buffer* host_w(const TensorView& t);  // host-visible mirror
  ink::vk::Buffer upload_dev(const void* src, VkDeviceSize bytes);
  const Pipe* mvpipe(DType dt) const;

  // Dual-mode emit helpers: with building_ = true they create the fixed
  // descriptor-set stream (all bindings are token-invariant); with
  // building_ = false they replay it by index into the Recorder. Host-side
  // counters (kv_len, conv filled, block depth) advance in BOTH modes so the
  // site sequence is identical.
  const ink::vk::DescSet& dset(const Pipe& p, std::initializer_list<Bnd> bs);
  void op(const Pipe& p, std::initializer_list<Bnd> bs, const void* pc,
          uint32_t pcb, uint32_t gx, uint32_t gy = 1,
          const char* tag = nullptr);
  void mv(const Pipe& p, Bnd w, Bnd x, Bnd o, int64_t M, int64_t K,
          const char* tag = nullptr);
  void cp(const ink::vk::Buffer& s, VkDeviceSize so, const ink::vk::Buffer& d,
          VkDeviceSize dof, VkDeviceSize sz, const char* tag = nullptr);
  void fill(const ink::vk::Buffer& d, VkDeviceSize off, VkDeviceSize sz,
            uint32_t data, const char* tag = nullptr);
  // cpr: one vkCmdCopyBuffer carrying a region array (per-tensor batches
  // for the staged expert-slice copies — fewer driver packets).
  void cpr(const ink::vk::Buffer& s, const ink::vk::Buffer& d,
           const VkBufferCopy* regs, uint32_t n, const char* tag = nullptr);
  void sync();  // full pipeline barrier (correctness-first)
  Bnd histb(int slot, int stream, int C) const;
  Bnd kvb(const ink::vk::Buffer& cache, int slot) const;

  void emit_token(int token, int64_t pos);
  // Record-once decode: walk emit_token once at init, recording per-segment
  // persistent command buffers (a segment boundary sits between a staged
  // visit's a() — ending in the router-id readback — and its b()). Every
  // recorded argument is token-invariant; per-token scalars ride in the
  // params_ rows the shaders read (see shaders/bindings.h PARAMS ROWS).
  bool record_segments();
  void write_params(int token, int64_t pos);  // params rows + counter advance
  // row stride honors minStorageBufferOffsetAlignment (>= 64B rows)
  Bnd prm_b(int slot) const {
    return {&params_, VkDeviceSize(slot) * prm_stride_, 64};
  }
  Bnd prm_head() const { return {&params_, VkDeviceSize(48) * prm_stride_, 64}; }
  ink::shaders::GpuParamsRow* prm_row(int i) const {
    return (ink::shaders::GpuParamsRow*)((char*)prm_ + size_t(i) * prm_stride_);
  }
  // Per-visit split (host-resident expert staging): emit_visit_a covers the
  // visit head through the router plus a 32B copy of ridx_ into the readback
  // ids region; emit_visit_b is the MoE tail (staged copies, fill, expert
  // GEMVs, moe_reduce, mlp conv, residual add, dump cp, counters). The flat
  // op() sequence is a(v) then b(v) per visit in BOTH build and record
  // walks — record just submits between them so the host can read the
  // freshly routed expert ids and program visit v's slice copies.
  void emit_visit_a(int i, int p, int64_t pos);
  // part: 0 = whole MoE tail; 1 = w13 phase (moe_zero + gate/up GEMVs —
  // consumes the w13 slices); 2 = w2 phase onward (accum..out_add — waits
  // on the w2 slices). Parts only differ for staged visits in the
  // two-phase path; every other call is part 0.
  void emit_visit_b(int i, int p, int part = 0);
  void emit_head(int token, int64_t pos);  // nblock reset + embed
  void emit_snap();                        // prelude snapshot (pre_* buffers)
  void emit_bank(int p);                   // pass-p loop-state restore
  void emit_fin();                         // final attn_res + norm + logits
  // Record-mode helper: 4 vkCmdCopyBuffer slices per top-k slot for one
  // host-resident layer (32 copies; no-ops while building_). part 0 = all
  // four tensors, 1 = w13 weight+scale only, 2 = w2 weight+scale only.
  void stage_copies(const Lw& L, int part);

  const ModelConfig* cfg_ = nullptr;
  ink::vk::Context* ctx_ = nullptr;
  ink::vk::Allocator* alloc_ = nullptr;
  std::string shader_dir_;
  std::string err_;
  bool ok_ = false;
  bool dbg_en_ = false;  // INK_DBG: emit stage-dump copies for visit5
  VkDeviceSize expert_vram_ = 7ull << 30;  // dev-resident expert budget (safe under driver reserve)

  int H_ = 0;
  int64_t V_ = 0;    // unpadded vocab
  int64_t cap_ = 0;  // KV capacity in tokens

  ink::vk::CmdPool cpool_, tpool_;
  ink::vk::Buffer staging_, dummy4_;

  // device state buffers
  ink::vk::Buffer partial_, block_, pre_partial_, pre_block_, bank_;
  ink::vk::Buffer reads_, msgs_, mgated_, inbuf_;
  ink::vk::Buffer q_, k_, v_, r_, tk_, tv_, rel_, probs_, grate_, sink_, o_;
  ink::vk::Buffer attn_out_, a_out_, residual_, m_in_, mn_, moe_out_, mo_;
  ink::vk::Buffer ridx_, rw_, gate_, up_, moe_rows_;
  ink::vk::Buffer kvk_, kvv_, hist_;
  ink::vk::Buffer h_, hfin_, logits_;
  ink::vk::Buffer readback_;  // HOST-visible: logits + taps + visit dumps
  // dev-local expert-slice staging for host-resident layers: 8 slots x the
  // max per-expert slice of each tensor (w13 weight/scale, w2 weight/scale)
  ink::vk::Buffer stg_w13w_, stg_w13s_, stg_w2w_, stg_w2s_;
  int32_t stage_ids_[8] = {};  // last readback of the router's top-k ids
  // false when the staging alloc failed (VRAM full): host-resident layers
  // then keep the original mslot host-read path instead of staging.
  bool stage_ok_ = false;

  // weight buffers (globals; per-layer weights live in lw_ as map pointers)
  ink::vk::Buffer embedw_, embn_, normw_, unembw_;
  ink::vk::Buffer attnq_, mlpq_, msgq_, outq_;
  WB unemb_mv_;
  uint32_t emb_flags_ = 0;
  std::unordered_map<const void*, ink::vk::Buffer> devw_, hostw_, f32w_;
  Lw lw_[24];

  Pipe p_rms_, p_mvf_, p_mvb_, p_mv8_, p_cvs_, p_cvd_, p_rel_, p_sco_, p_smx_,
      p_acc_, p_ares_, p_msg_, p_rtr_, p_moe_, p_mred_, p_moem_, p_nv4m_, p_mx8m_,
      p_sig_, p_cap_, p_add_, p_l2_,
      p_log_, p_emb_, p_nv4s_, p_mx8s_,
      p_nv4st_, p_mx8st_, p_moest_,  // *_stage variants (slot-indexed stg_*)
      p_kva_;                      // kv_append (device-driven kv dst offset)

  std::vector<ink::vk::DescSet> sites_;  // binding stream, built once
  size_t site_cursor_ = 0;
  bool building_ = false;
  ink::vk::Recorder* rec_ = nullptr;

  // ---- record-once token replay (params SSBO + persistent segments) ----
  ink::vk::Buffer params_;                 // host-coherent GpuParamsRow[49]
  void* prm_ = nullptr;                    // mapped rows (stride below)
  VkDeviceSize prm_stride_ = 64;           // >=64, SBO-aligned
  int slot_full_[48] = {};                 // cache slot -> is_full_attn(layer)
  uint32_t pstride_ = 0;                   // fixed probs row stride (cap_+8)
  uint32_t score_gy_ = 0;                  // fixed attn_scores grid.y
  std::vector<std::unique_ptr<ink::vk::Recorder>> segs_;   // persistent cbs
  std::vector<std::unique_ptr<ink::vk::Recorder>> trec13_, trec2_;
  std::vector<int> staged_layer_;  // segment boundary k -> staged visit's layer
  std::vector<int> staged_vidx_;   // boundary k -> visit index (sentinel slot)
  // Two-phase staging: w13 copies feed their own segment while the w2
  // copies keep streaming on the SDMA queue (needs a dedicated transfer
  // queue + timeline semaphore; otherwise single-phase as before).
  bool split_stg_ = false;
  VkFence seg_fence_ = VK_NULL_HANDLE;     // ids/logits readback fence (reused)
  VkSemaphore tl_sem_ = VK_NULL_HANDLE;    // transfer->compute ordering
  uint64_t tl_val_ = 0;                    // last value signaled on tl_sem_
  bool tl_en_ = false;                     // timeline semaphores available
  uint64_t rec_disp_ = 0, rec_cp_ = 0, rec_bar_ = 0;  // per-token stat deltas

  // ---- INK_GPU_TS=1: GPU timestamps inside the recorded token cb ----
  // One timestamp at cb start plus one after every op/mv/cp/fill. The op
  // sequence is token-invariant, so segment labels are built once (first
  // emit) and indexed by slot afterwards. Everything here is inert unless
  // the env var is set: no writes reach the command buffer otherwise.
  void ts_mark(const char* name, const char* tag);
  void ts_gap();  // cb-start mark in submits > 0; segment = inter-submit gap
  void ts_collect();
  void ts_report() const;
  struct TsSeg {
    std::string label;   // "<scope>/<role>" e.g. "v5/wk", "fin/unemb"
    uint8_t cls;         // 0 matvec,1 conv4,2 attn,3 moe,4 elementwise,5 copy
    uint8_t phase;       // 0 prelude,1-3 pass0-2,4 coda,5 final
    double ms = 0;       // summed GPU time across collected tokens
  };
  bool ts_en_ = false;
  bool ts_off_ = false;  // suppress marks (copies on the transfer queue)
  VkQueryPool ts_pool_ = VK_NULL_HANDLE;
  uint32_t ts_cap_ = 0;    // pool slots allocated
  uint32_t ts_next_ = 0;   // next query slot during an emit (slot0=cb start)
  double ts_period_ = 0;   // ns per timestamp tick
  uint64_t ts_tokens_ = 0;
  double ts_gpu_ms_ = 0;                    // summed busy ms (seg deltas)
  std::vector<uint64_t> ts_raw_;            // query result scratch
  std::vector<TsSeg> ts_segs_;              // one per timestamp boundary
  std::unordered_map<std::string, int> ts_dup_;  // label dedup, first emit
  std::string cur_scope_;                  // "v<vidx>" / "pre" / "bank" / "fin"
  uint8_t cur_phase_ = 0;

  // host-side counters — mirror ops::CacheSlot bookkeeping
  int64_t kv_len_[48] = {};
  int conv_filled_[48][4] = {};
  int nblock_ = 0, pre_nblock_ = 0;

  std::vector<float> logits_host_;
  float* rb_ = nullptr;   // mapped readback
  int64_t rb_vis_ = 0;    // float offset of the visit-dump region
  int64_t rb_dbg_ = 0;    // float offset of the visit0 debug region
  int64_t rb_ids_ = 0;    // float offset of the 8-int32 router-id region
  int64_t rb_sen_ = 0;    // float offset of the sentinel dwords (ids-ready
                        // spin-poll marks; one per visit + one for fin)
};

}  // namespace ink::gpu
