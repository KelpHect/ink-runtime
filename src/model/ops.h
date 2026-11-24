#pragma once

#include "ink/tensor.h"
#include "io/config.h"

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
// Static taps w [C,1,4]; out = (conv(x)+x)/sqrt(5). x may alias out.
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

}  // namespace ink::ops
