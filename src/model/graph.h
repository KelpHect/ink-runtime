#pragma once

#include "io/config.h"
#include "io/safetensors.h"
#include "ops.h"

#include <vector>

namespace ink {

struct LayerW {
  ops::AttnW attn;
  TensorView attn_norm;                 // sliding layers
  TensorView qnorm, knorm, vnorm;       // full layers (5,11,17,23)
  TensorView mlp_norm;                  // all layers
  TensorView attn_sconv, mlp_sconv;     // [2304,1,4]
  ops::MlpW mlp;
  // controllers[stream][mat]: stream 0=q(attn),1=k,2=v,3=mlp; mat 0=down,
  // 1=forget, 2=gain. Loop layers (6..17) only.
  TensorView ctrl[4][3];
  bool full = false, loop = false;
};

struct ModelW {
  TensorView embed, embed_norm, norm, unembed;
  TensorView attn_q[23];  // attn_res.attn_queries.{0..22} → layer i+1
  TensorView mlp_q[24];   // attn_res.mlp_queries.{0..23} → layer i
  TensorView output_q;
  // merged [12][4][2304] message queries: (q,k,v,mlp) from
  // middle_cycle_cross_pass_{,kv_}queries — merge order per 07 §1.
  std::vector<float> msg_q;
  LayerW L[24];
};

// Per-token position-independent state + persistent per-slot caches.
// block_states/partial are per-token; the 48 cache slots (KV + 4 conv
// streams each) persist across positions (07 §8 decode-time state).
struct RunState {
  std::vector<std::vector<float>> block;  // attn_res stack (<=4 entries)
  std::vector<float> partial;             // running residual [2304]
  std::vector<std::vector<float>> pre_block;
  std::vector<float> pre_partial;
  std::vector<float> bank0, bank1;        // prev-pass s12 + final hidden
  ops::CacheSlot cache[48];

  // per-visit scratch
  std::vector<float> reads;   // [3][2304]
  std::vector<float> msgs;    // [4][2304]
  std::vector<float> qbuf;    // [4][2304] decoded query rows
  std::vector<float> xq, xk, xv;
  std::vector<float> attn_out, a_out, residual, m_in, m_gated, moe_out, mo;
  std::vector<float> h_fin;
  std::vector<float> logits;  // [unpadded_vocab]
  std::vector<float> taps[4]; // MTP hidden taps {5,23,41} + final normed
};

// Fixed-shape forward skeleton (research/07 §8). CPU, fp32 math — the
// correctness reference. Single token per call; prefill = call per token.
class Graph {
 public:
  bool init(const ModelConfig& c, const WeightStore& w, std::string* err);
  void reset();  // clears all caches (new sequence)
  const std::vector<float>& forward_token(int token, int64_t pos);
  const RunState& state() const { return s_; }
  const ModelW& weights() const { return w_; }

 private:
  void visit(int i, int p, int64_t pos);
  const ModelConfig* cfg_;
  ModelW w_;
  RunState s_;
  int64_t H_;
};

}  // namespace ink
