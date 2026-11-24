#pragma once

#include "ink/tensor.h"
#include "io/config.h"
#include "io/safetensors.h"

#include <vector>

namespace ink::ops {

float sigmoid(float x);
float softcap(float x, float cap);              // cap * tanh(x / cap)
void softmax(float* x, int n);                  // in-place, fp32, max-subtracted
int argmax(const float* x, int n);

// fp32 accumulate throughout (research/07 §12).
float rms_rstd(const float* x, int n, float eps);
void rmsnorm(const float* x, const TensorView& w, float* out, int n, float eps);
void rmsnorm_w(const float* x, const float* w, float* out, int n, float eps);
void rmsnorm_uw(const float* x, float* out, int n, float eps);  // no gamma

float dot(const float* a, const float* b, int64_t n);
// out[m] = sum_k W[m,k] * x[k]; W may be F32/BF16/F8E4M3/U8/I8. fp32 accum.
void matvec(const TensorView& W, const float* x, float* out, int64_t M, int64_t K);
// Controller down-proj: inputs AND output rounded through bf16 (07 §3).
void matvec_bf16io(const TensorView& W, const float* x, float* out, int64_t M,
                   int64_t K);

// v[h,:] *= min(1, thr / (||v[h,:]|| + 1e-6)); all fp32.
void l2_clip(float* v, int heads, int dim, float thr);
// v[h,d] *= scale[h,d] first, then the same cap.
void scaled_l2_clip(float* v, const TensorView& scale, int heads, int dim,
                    float thr);

// Last-3-inputs state for one causal depthwise conv4 stream.
struct ConvState4 {
  int C = 0;
  int filled = 0;                  // inputs seen so far (warm-start < 3)
  std::vector<float> s;            // s[c*3+i] = x[t-1-i], i in 0..2
  void init(int channels);
};
// Static taps w [C,1,4] in conv1d layout (tap 3 = current); out =
// (conv(x)+x)/sqrt(5). x may alias out.
void sconv4_static(const float* x, const TensorView& w, ConvState4& st,
                   float* out);
// Dynamic taps (full-attn layers): per-token dyn [(C/head_dim)*4] added to
// static_w [4,C]; out = (conv(x)+x)/sqrt(2*4+1) = /3. x may alias out.
void sconv4_dynamic(const float* x, const float* dyn, const TensorView& static_w,
                    int head_dim, ConvState4& st, float* out);

struct KVSlot {
  int C = 1024;                    // n_kv * head_dim
  int64_t len = 0;                 // tokens stored
  std::vector<float> k, v;         // [len][C] each, fp32
  void append(const float* k_in, const float* v_in);
};
// conv streams per cache slot: 0=K, 1=V, 2=attn-out, 3=mlp-out (07 §5).
struct CacheSlot {
  KVSlot kv;
  ConvState4 conv[4];
};

struct AttnW {
  TensorView wq, wk, wv, wo, gate, sink, wr, relproj, qclip, rclip;
  TensorView kconv_w, vconv_w;                                  // sliding
  TensorView kconv_static, kconv_wproj, vconv_static, vconv_wproj;  // full
};
// Full §4 pipeline for one token: projections, k/v convs, clips, KV append,
// rel-bias gather, sink-column softmax, output gate, wo. Writes out[hidden].
void attn_step(const ModelConfig& c, const AttnW& A, bool full,
               const float* xq, const float* xk, const float* xv,
               CacheSlot& cache, int64_t pos, float* out);

// Learned softmax read over block-boundary snapshots (07 §2).
// srcs = block_states ++ {partial}; query q [Q,D] fp32.
// rms-normalize each source (eps 1e-6), logits = q.xn / sqrt(D), softmax
// over the n sources, out[q] = sum p_n * xn_n.
void attn_res_read(const float* const* srcs, int nsrc, const float* q, int Q,
                   int D, float* out);

// Rank-64 message gate (07 §3): h = bf16(down @ bf16(cat(rnu(dest),
// rnu(msg)))); out = (1-tanh(forget@h))*dest + tanh(gain@h)*msg.
// Caller applies the sublayer norm afterwards.
void msg_gate(const TensorView& down, const TensorView& forget,
              const TensorView& gain, const float* dest, const float* msg,
              int D, int rank, float* out);

// Router (07 §6): scores = sigmoid(gate_w @ x) in fp32; topk on
// scores + bias (bias is selection-only); kept raw scores renormalized
// to sum 1 (+1e-20) then * route_scale.
void router_topk(const TensorView& gate_w, const TensorView& gate_b,
                 const float* x, int n_experts, int k, float route_scale,
                 int* idx, float* w);

struct ExpertW {
  // NVFP4: w13/w2 are U8 packed (2 E2M1/byte), scales F8E4M3, per-expert
  // factor amax/outer_scale_denominator. MXFP8: F8E4M3 + UE8M0 scales.
  TensorView w13, w13s, w13amax, w13osd;
  TensorView w2, w2s, w2amax, w2osd;
  bool mxfp8 = false;
};
struct MlpW {
  TensorView router_w, router_b;
  ExpertW experts;
};
// Router + top-k experts + softcapped SwiGLU, fp32 accumulate.
void moe_forward(const ModelConfig& c, const MlpW& m, const float* x,
                 float* out);

// ------------------------------------------------------------------
// CpuDraft — CPU reference for the MTP/DFlare speculative draft head
// (research/07 §9; GPU twin: src/gpu/gpu_draft.*). Same math, same
// bf16 rounding discipline, kept append-only at the bottom of ops.cpp.
// Per propose(): fuse taps -> per-layer ctx-KV append -> 3-layer
// non-causal block of 8 -> base logits via shared unembed -> 7 markov
// GRU steps -> 8 proposed ids.
struct DraftW {
  TensorView fuse_w, norm, mask_embed;                    // globals
  TensorView dgam[4];                                     // 3 layers + markov
  TensorView inln[3], pln[3], qn[3], kn[3], knc[3];
  TensorView wq[3], wk[3], wv[3], wo[3], wkc[3], wvc[3];
  TensorView wg[3], wu[3], wd[3];
  TensorView w1, rnn, rnn_b, hout, hout_b;                // markov head
};

struct CpuDraft {
  const ModelConfig* cfg = nullptr;
  DraftW w;
  static constexpr int64_t CTX_MAX = 4224;
  std::vector<float> ctxk[3], ctxv[3];  // [n][1024] f32
  std::vector<int64_t> ctxpos;
  int64_t n_ctx = 0;

  // embed/embn/unemb are the shared target tensors (from WeightStore).
  bool init(const ModelConfig& c, const ink::SafetensorsFile& mtp,
            const TensorView& embed, const TensorView& embn,
            const TensorView& unemb, std::string* err);
  void reset() { n_ctx = 0; }
  // taps4 = 4*H contiguous fp32 (visit5/23/41 inputs + final normed).
  // Proposes ids for positions anchor_pos+1 .. anchor_pos+7 (DFlash
  // fill-in); ids[7] is a spare argmax over slot-7 base logits.
  std::vector<int> propose(const float* taps4, int anchor, int64_t anchor_pos);

  // Debug captures from the last propose() (INK_DRAFT_DEBUG): fused ctx
  // [4*H], block output post-norm [8*H], slot-0 base-logit argmax context.
  // dbg_xall = residual x after each layer: [3][8*H]. dbg_xin = staged
  // block input [8H]; dbg_xn0/q0/k0 = L0 prenorm / q / k post norm+rope.
  std::vector<float> dbg_ctx, dbg_out8, dbg_base0, dbg_xall, dbg_att0,
      dbg_xin, dbg_xn0, dbg_q0, dbg_k0, dbg_ctxk0;

 private:
  TensorView embed_, embn_, unemb_;
};

}  // namespace ink::ops
