#pragma once

#include "ink/tensor.h"

namespace ink::ops {

// Compute backend seam: GPU backends implement matvec-family ops; every
// method returns false when it can't take the call (dtype/shape not covered,
// device busy) and the CPU reference path runs instead — the golden gate
// arbitrates each swap. Default: no backend = pure CPU (P0 reference).
class Backend {
 public:
  virtual ~Backend() = default;

  // out[m] = W[m,:] . x ; W storage dtype F32/BF16/F8E4M3/U8/I8.
  virtual bool matvec(const TensorView& W, const float* x, float* out,
                      int64_t M, int64_t K) {
    (void)W; (void)x; (void)out; (void)M; (void)K;
    return false;
  }

  // Expert dequant-dot: W row r of expert e, block scales S, optional
  // per-expert factor amax/osd (NVFP4) — mxfp8 selects the formula.
  virtual bool expert_dot(const TensorView& W, const TensorView& S,
                          const TensorView& amax, const TensorView& osd,
                          bool mxfp8, int e, int64_t r, const float* x,
                          int64_t K, float* out) {
    (void)W; (void)S; (void)amax; (void)osd; (void)mxfp8;
    (void)e; (void)r; (void)x; (void)K; (void)out;
    return false;
  }

  // Whole-expert FFN (batched): computes the full gate/up/swiglu/down for
  // expert e and accumulates `wscale * result` into out[0..D). Matches the
  // moe_forward per-expert semantics exactly:
  //   gate[m] = dot(w13[e,2m],   x),  up[m] = dot(w13[e,2m+1], x)
  //   a[m] = softcap(gate[m],gcap)*sigmoid(gate[m])*softcap(up[m],ucap)
  //   out[r] += wscale * dot(w2[e,r], a)
  virtual bool expert_ffn(const TensorView& w13, const TensorView& w13s,
                          const TensorView& w13amax, const TensorView& w13osd,
                          const TensorView& w2, const TensorView& w2s,
                          const TensorView& w2amax, const TensorView& w2osd,
                          bool mxfp8, int e, const float* x,
                          int64_t D, int64_t I, float gcap, float ucap,
                          float wscale, float* out) {
    (void)w13; (void)w13s; (void)w13amax; (void)w13osd;
    (void)w2; (void)w2s; (void)w2amax; (void)w2osd;
    (void)mxfp8; (void)e; (void)x; (void)D; (void)I;
    (void)gcap; (void)ucap; (void)wscale; (void)out;
    return false;
  }
};

// Process-wide backend; set once at startup (nullptr = CPU only).
Backend*& backend_slot();

}  // namespace ink::ops
