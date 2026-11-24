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
#include <string>
#include <unordered_map>
#include <vector>

namespace ink::gpu {

class GpuGraph {
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
  //   tap(0..2) = visit-input hidden at vidx {5,23,41}; tap(3) = h_fin.
  //   visit_dump(vidx, 0/1) = partial at visit input/output (48 visits).
  const float* tap(int i) const;
  const float* visit_dump(int vidx, int phase) const;
  const float* dbg(int row) const;  // visit0 stage dump rows (debug)

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
    int64_t w13R = 0, w2R = 0;
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
          uint32_t pcb, uint32_t gx, uint32_t gy = 1);
  void mv(const Pipe& p, Bnd w, Bnd x, Bnd o, int64_t M, int64_t K);
  void cp(const ink::vk::Buffer& s, VkDeviceSize so, const ink::vk::Buffer& d,
          VkDeviceSize dof, VkDeviceSize sz);
  void sync();  // full pipeline barrier (correctness-first)
  Bnd histb(int slot, int stream, int C) const;
  Bnd kvb(const ink::vk::Buffer& cache, int slot) const;

  void emit_token(int token, int64_t pos);
  void emit_visit(int i, int p, int64_t pos);

  const ModelConfig* cfg_ = nullptr;
  ink::vk::Context* ctx_ = nullptr;
  ink::vk::Allocator* alloc_ = nullptr;
  std::string shader_dir_;
  std::string err_;
  bool ok_ = false;
  bool dbg_en_ = false;  // INK_DBG: emit stage-dump copies for visit5

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
  ink::vk::Buffer ridx_, rw_, gate_, up_;
  ink::vk::Buffer kvk_, kvv_, hist_;
  ink::vk::Buffer h_, hfin_, logits_;
  ink::vk::Buffer readback_;  // HOST-visible: logits + taps + visit dumps

  // weight buffers (globals; per-layer weights live in lw_ as map pointers)
  ink::vk::Buffer embedw_, embn_, normw_, unembw_;
  ink::vk::Buffer attnq_, mlpq_, msgq_, outq_;
  WB unemb_mv_;
  uint32_t emb_flags_ = 0;
  std::unordered_map<const void*, ink::vk::Buffer> devw_, hostw_, f32w_;
  Lw lw_[24];

  Pipe p_rms_, p_mvf_, p_mvb_, p_mv8_, p_cvs_, p_cvd_, p_rel_, p_sco_, p_smx_,
      p_acc_, p_ares_, p_msg_, p_rtr_, p_moe_, p_sig_, p_cap_, p_add_, p_l2_,
      p_log_, p_emb_, p_nv4s_, p_mx8s_;

  std::vector<ink::vk::DescSet> sites_;  // binding stream, built once
  size_t site_cursor_ = 0;
  bool building_ = false;
  ink::vk::Recorder* rec_ = nullptr;

  // host-side counters — mirror ops::CacheSlot bookkeeping
  int64_t kv_len_[48] = {};
  int conv_filled_[48][4] = {};
  int nblock_ = 0, pre_nblock_ = 0;

  std::vector<float> logits_host_;
  float* rb_ = nullptr;   // mapped readback
  int64_t rb_vis_ = 0;    // float offset of the visit-dump region
  int64_t rb_dbg_ = 0;    // float offset of the visit0 debug region
};

}  // namespace ink::gpu
