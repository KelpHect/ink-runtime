// gpu_draft.h — GPU implementation of the MTP/DFlare speculative draft head
// (Inkling-2-Mini mtp.safetensors; math spec: research/07 §9).
//
// Standalone sibling of GpuGraph/GpuPrefill: owns its weights, pipelines,
// descriptor sets, ctx-KV cache and a per-propose command buffer. Inputs per
// block: the target's 4 tap vectors [4x2304] (visit-input hiddens at vidx
// {5,23,41} + final normed — GpuGraph::tap(0) exposes them contiguously),
// the anchor token id and its absolute position. Output: 8 drafted ids for
// positions anchor+1..anchor+7 (7 markov steps over mask slots 1..7, DFlash
// fill-in) + a spare argmax over the slot-7 base logits in ids[7].
//
// Per propose: one staging upload (~110KB), one recorded command buffer
// (ctx-KV append per layer + 3-layer non-causal block + 7 sequential markov
// steps chained on device), one submit_and_wait, and the 8 ids are read from
// a host-visible cell buffer — no mid-cb host round trips.
#pragma once

#include "io/config.h"
#include "io/safetensors.h"
#include "vk/cmd.h"
#include "vk/mem.h"
#include "vk/pipe.h"
#include "vk/vk.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ink::gpu {

class GpuDrafter {
 public:
  // ctx/alloc are caller-owned and must outlive the drafter. mtp is the open
  // mtp.safetensors file; embed/embn/unemb are the shared target tensors
  // (embed is only read host-side; unembed is uploaded once unless
  // shared_unemb points at an identical raw-bf16 SSBO — e.g. the graph's
  // own unembw_ — which saves ~883MB and is never freed here).
  // ctx_cap bounds the per-layer context-KV cache (clamped to MAXK-8).
  bool init(const ModelConfig& c, const SafetensorsFile& mtp,
            const TensorView& embed, const TensorView& embn,
            const TensorView& unemb, ink::vk::Context& ctx,
            ink::vk::Allocator& alloc, const std::string& shader_dir,
            int64_t ctx_cap, std::string* err,
            const ink::vk::Buffer* shared_unemb = nullptr);
  void shutdown();
  ~GpuDrafter() { shutdown(); }

  void reset();  // new sequence: clears the ctx-KV length

  // taps4 = 4*H contiguous fp32 tap vectors. Proposes ids for positions
  // anchor_pos+1 .. anchor_pos+8; empty vector on failure (see error()).
  std::vector<int> propose(const float* taps4, int anchor, int64_t anchor_pos);

  const std::string& error() const { return err_; }
  bool ok() const { return ok_; }
  int64_t ctx_len() const { return n_ctx_; }

  // Debug readbacks (valid until the next propose; only populated when
  // INK_DRAFT_DEBUG is set): fused ctx [4*H] and normed backbone out [8*H].
  const float* dbg_ctx() const { return dbg_map_; }
  const float* dbg_out() const { return dbg_map_ ? dbg_map_ + 4 * H_ : nullptr; }
  const float* dbg_x(int layer) const {  // residual after layer 0..2
    return dbg_map_ ? dbg_map_ + (12 + layer * 8) * H_ : nullptr;
  }
  const float* dbg_att0() const {  // layer-0 attention output [8*3072]
    return dbg_map_ ? dbg_map_ + 36 * H_ : nullptr;
  }
  const float* dbg_xin() const { return dbg_map_ ? dbg_map_ + 36 * H_ + 24576 : nullptr; }   // staged block input [8H]
  const float* dbg_xn0() const { return dbg_map_ ? dbg_map_ + 44 * H_ + 24576 : nullptr; }   // L0 prenorm [8H]
  const float* dbg_q0() const { return dbg_map_ ? dbg_map_ + 52 * H_ + 24576 : nullptr; }    // L0 q post norm+rope [8*3072]
  const float* dbg_k0() const { return dbg_map_ ? dbg_map_ + 52 * H_ + 49152 : nullptr; }    // L0 k post norm+rope [8*1024]
  const float* dbg_ctxk0() const { return dbg_map_ ? dbg_map_ + 52 * H_ + 57344 : nullptr; } // L0 ctx key row [1024]

  struct Stats {
    uint64_t submits = 0, dispatches = 0, copies = 0, barriers = 0;
    uint64_t uploads = 0, upload_bytes = 0;
    double wait_ms = 0, record_ms = 0;
  } stats;

  struct Bnd {
    const ink::vk::Buffer* b;
    VkDeviceSize off, sz;
  };

 private:
  struct Pipe {
    ink::vk::ComputePipeline p;
    const char* name = "";
  };

  bool load_pipe(const char* name, uint32_t bindings, uint32_t pc_bytes,
                 Pipe& out);
  ink::vk::Buffer upload_dev(const void* src, VkDeviceSize bytes);
  const ink::vk::Buffer* dev_w(const TensorView& t);    // raw dtype upload
  const ink::vk::Buffer* dev_f32(const TensorView& t);  // decode -> f32

  // Dual-mode emit (GpuGraph pattern): building_ = true materializes the
  // fixed descset stream in sites_; false replays it into rec_.
  const ink::vk::DescSet& dset(const Pipe& p, std::initializer_list<Bnd> bs);
  void op(const Pipe& p, std::initializer_list<Bnd> bs, const void* pc,
          uint32_t pcb, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);
  void mv(Bnd w, Bnd x, Bnd o, int64_t M, int64_t K);       // single GEMV
  void mv_seq(Bnd w, Bnd x, Bnd o, int64_t M, int64_t K,
              int64_t xs, int64_t Mstride, int t);          // T-row GEMV
  void cp(const ink::vk::Buffer& s, VkDeviceSize so,
          const ink::vk::Buffer& d, VkDeviceSize dof, VkDeviceSize sz);
  void fill(const ink::vk::Buffer& d, VkDeviceSize off, VkDeviceSize sz,
            uint32_t data);
  void sync();

  void emit_block(int64_t pos, int64_t n_ctx, bool appending);

  const ModelConfig* cfg_ = nullptr;
  ink::vk::Context* ctx_ = nullptr;
  ink::vk::Allocator* alloc_ = nullptr;
  std::string shader_dir_;
  std::string err_;
  bool ok_ = false;
  bool dbg_en_ = false;

  int H_ = 2304;
  int64_t V_ = 200058;    // unpadded vocab (argmax domain)
  int64_t VSTR_ = 200064; // padded f32 row stride (64B-aligned)
  int64_t ctxcap_ = 0;    // ctx-KV rows per layer (<= 4224)
  int64_t n_ctx_ = 0;     // appended so far

  ink::vk::CmdPool cpool_, tpool_;
  ink::vk::Buffer staging_, dummy4_;

  // device state / activations
  ink::vk::Buffer x_, xn_, q8_, kb8_, vb8_, att8_, ao8_;
  ink::vk::Buffer g8_, u8_, a8_, mo8_, out8_, ctxb_, taps_;
  ink::vk::Buffer kscr_, vscr_, ctxk_, ctxv_;
  ink::vk::Buffer base_, in_, proj_, state_, lat_, dvec_, logits_;
  ink::vk::Buffer pv_, pi_;                 // argmax partials
  ink::vk::Buffer did_, ctxpos_, dbg_;      // HOST-visible cells/positions/dump

  // weights (unemb_ may alias the graph's shared buffer: unemb_shared_)
  ink::vk::Buffer unemb_, w1_, rnn_, rnn_b_, hout_, houtb_;
  bool unemb_shared_ = false;
  ink::vk::Buffer alpha_, dgam_, normw_, inln_, pln_, qn_, kn_, knc_;
  ink::vk::Buffer wq_[3], wk_[3], wv_[3], wo_[3], wkc_[3], wvc_[3];
  ink::vk::Buffer wg_[3], wu_[3], wd_[3];
  std::vector<float> emb_g_, mask_;         // host copies (block init)
  const TensorView* embed_t_ = nullptr;     // target embed (row gather)

  Pipe p_mvb_, p_mvs_, p_rms_, p_rmss_, p_argp_, p_argf_, p_vadd_, p_mix_,
      p_rope_, p_attn_, p_swig_, p_resid_, p_pack_, p_gru_, p_logits_;

  std::vector<ink::vk::DescSet> sites_;
  size_t site_cursor_ = 0;
  bool building_ = false;
  ink::vk::Recorder* rec_ = nullptr;
  uint32_t* did_map_ = nullptr;   // mapped did_ cells (16-stride)
  uint32_t* cp_map_ = nullptr;    // mapped ctxpos_
  float* dbg_map_ = nullptr;
};

}  // namespace ink::gpu
