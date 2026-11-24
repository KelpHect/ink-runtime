#include "ops.h"

#include "backend.h"
#include "io/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace ink::ops {

Backend*& backend_slot() {
  static Backend* b = nullptr;
  return b;
}

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
float softcap(float x, float cap) { return cap * std::tanh(x / cap); }

void softmax(float* x, int n) {
  float m = *std::max_element(x, x + n);
  float s = 0.f;
  for (int i = 0; i < n; i++) s += (x[i] = std::exp(x[i] - m));
  for (int i = 0; i < n; i++) x[i] /= s;
}

int argmax(const float* x, int n) {
  return int(std::max_element(x, x + n) - x);
}

float rms_rstd(const float* x, int n, float eps) {
  float ss = 0.f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  return 1.f / std::sqrt(ss / n + eps);
}

void rmsnorm_w(const float* x, const float* w, float* out, int n, float eps) {
  float r = rms_rstd(x, n, eps);
  for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

void rmsnorm(const float* x, const TensorView& w, float* out, int n, float eps) {
  float r = rms_rstd(x, n, eps);
  for (int i = 0; i < n; i++) out[i] = x[i] * r * w.f32(i);
}

void rmsnorm_uw(const float* x, float* out, int n, float eps) {
  float r = rms_rstd(x, n, eps);
  for (int i = 0; i < n; i++) out[i] = x[i] * r;
}

float dot(const float* a, const float* b, int64_t n) {
  float s = 0.f;
  for (int64_t i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

void matvec(const TensorView& W, const float* x, float* out, int64_t M,
            int64_t K) {
  // M <= W.shape[0] allowed: computes the first M rows (unpad slice).
  if (backend_slot() && backend_slot()->matvec(W, x, out, M, K)) return;
  if (W.shape[0] < M || W.shape[1] != K)
    throw std::runtime_error("matvec shape mismatch");
  for (int64_t m = 0; m < M; m++) {
    const std::byte* row = W.ptr(m, 0);
    float acc = 0.f;
    switch (W.dtype) {
      case DType::F32: {
        const float* w = reinterpret_cast<const float*>(row);
        for (int64_t k = 0; k < K; k++) acc += w[k] * x[k];
        break;
      }
      case DType::BF16: {
        const uint16_t* w = reinterpret_cast<const uint16_t*>(row);
        for (int64_t k = 0; k < K; k++) acc += bf16_to_f32(w[k]) * x[k];
        break;
      }
      case DType::F8E4M3: {
        const uint8_t* w = reinterpret_cast<const uint8_t*>(row);
        for (int64_t k = 0; k < K; k++) acc += f8e4m3_to_f32(w[k]) * x[k];
        break;
      }
      case DType::U8:
      case DType::U8_NVFP4: {
        const uint8_t* w = reinterpret_cast<const uint8_t*>(row);
        for (int64_t k = 0; k < K; k++) acc += float(w[k]) * x[k];
        break;
      }
      case DType::I8: {
        const int8_t* w = reinterpret_cast<const int8_t*>(row);
        for (int64_t k = 0; k < K; k++) acc += float(w[k]) * x[k];
        break;
      }
    }
    out[m] = acc;
  }
}

void matvec_bf16io(const TensorView& W, const float* x, float* out, int64_t M,
                   int64_t K) {
  std::vector<float> xb(K);
  for (int64_t k = 0; k < K; k++) xb[k] = bf16_round(x[k]);
  matvec(W, xb.data(), out, M, K);
  for (int64_t m = 0; m < M; m++) out[m] = bf16_round(out[m]);
}

void l2_clip(float* v, int heads, int dim, float thr) {
  for (int h = 0; h < heads; h++) {
    float* x = v + int64_t(h) * dim;
    float n = std::sqrt(dot(x, x, dim));
    float s = std::min(1.f, thr / (n + 1e-6f));
    for (int d = 0; d < dim; d++) x[d] *= s;
  }
}

void scaled_l2_clip(float* v, const TensorView& scale, int heads, int dim,
                    float thr) {
  for (int h = 0; h < heads; h++) {
    float* x = v + int64_t(h) * dim;
    for (int d = 0; d < dim; d++) x[d] *= scale.f32(h, d);
    float n = std::sqrt(dot(x, x, dim));
    float s = std::min(1.f, thr / (n + 1e-6f));
    for (int d = 0; d < dim; d++) x[d] *= s;
  }
}

void ConvState4::init(int channels) {
  C = channels;
  filled = 0;
  s.assign(size_t(C) * 3, 0.f);
}

void sconv4_static(const float* x, const TensorView& w, ConvState4& st,
                   float* out) {
  if (st.C == 0) st.init(int(w.shape[0]));
  const int C = st.C;
  constexpr float inv = 0.4472135954999579f;  // 1/sqrt(4+1)
  std::vector<float> y(C);
  for (int c = 0; c < C; c++) {
    // conv1d layout [C,1,4]: tap 3 = current token, tap 0 = 3 steps back
    float acc = w.f32(c, 0, 3) * x[c];
    for (int lag = 1; lag < 4; lag++) {
      float xin = (lag <= st.filled) ? st.s[size_t(c) * 3 + (lag - 1)] : 0.f;
      acc += w.f32(c, 0, 3 - lag) * xin;
    }
    y[c] = (acc + x[c]) * inv;
  }
  if (st.filled < 3) st.filled++;
  for (int c = 0; c < C; c++) {
    st.s[size_t(c) * 3 + 2] = st.s[size_t(c) * 3 + 1];
    st.s[size_t(c) * 3 + 1] = st.s[size_t(c) * 3 + 0];
    st.s[size_t(c) * 3 + 0] = x[c];
  }
  std::memcpy(out, y.data(), sizeof(float) * C);
}

void sconv4_dynamic(const float* x, const float* dyn, const TensorView& static_w,
                    int head_dim, ConvState4& st, float* out) {
  if (st.C == 0) st.init(int(static_w.shape[1]));
  const int C = st.C;
  constexpr float inv = 1.f / 3.f;  // 1/sqrt(2*4+1)
  std::vector<float> y(C);
  for (int c = 0; c < C; c++) {
    int h = c / head_dim;
    float acc = 0.f;
    for (int lag = 0; lag < 4; lag++) {
      // taps[h,ch,lag] = static_w[lag, c] + dyn[h*4+lag]
      float tap = static_w.f32(lag, c) + dyn[h * 4 + lag];
      float xin = lag == 0 ? x[c]
                  : (lag <= st.filled) ? st.s[size_t(c) * 3 + (lag - 1)]
                                       : 0.f;
      acc += tap * xin;
    }
    y[c] = (acc + x[c]) * inv;
  }
  if (st.filled < 3) st.filled++;
  for (int c = 0; c < C; c++) {
    st.s[size_t(c) * 3 + 2] = st.s[size_t(c) * 3 + 1];
    st.s[size_t(c) * 3 + 1] = st.s[size_t(c) * 3 + 0];
    st.s[size_t(c) * 3 + 0] = x[c];
  }
  std::memcpy(out, y.data(), sizeof(float) * C);
}

void KVSlot::append(const float* k_in, const float* v_in) {
  k.insert(k.end(), k_in, k_in + C);
  v.insert(v.end(), v_in, v_in + C);
  len++;
}

void attn_step(const ModelConfig& c, const AttnW& A, bool full,
               const float* xq, const float* xk, const float* xv,
               CacheSlot& cache, int64_t pos, float* out) {
  const int H = c.hidden_size;
  const int HQ = c.num_attention_heads, HK = c.num_key_value_heads;
  const int DH = c.head_dim, DR = c.d_rel;
  const int ext = c.rel_extent;

  std::vector<float> q(HQ * DH), k(HK * DH), v(HK * DH), r(HQ * DR);
  matvec(A.wq, xq, q.data(), HQ * DH, H);
  matvec(A.wk, xk, k.data(), HK * DH, H);
  matvec(A.wv, xv, v.data(), HK * DH, H);
  matvec(A.wr, xq, r.data(), HQ * DR, H);

  if (std::getenv("INK_DBG")) {
    static int pre = 0;
    if (pre++ == 5) {
      FILE* f = std::fopen("testdata/ink_dbg_cpu/dbgA_kpre.bin", "wb");
      if (f) {
        std::fwrite(k.data(), 4, size_t(HK * DH), f);
        std::fclose(f);
      }
    }
  }
  if (full) {
    // dynamic taps come from the sublayer input (x_k / x_v), not from k/v
    std::vector<float> tk(HK * 4), tv(HK * 4);
    matvec(A.kconv_wproj, xk, tk.data(), HK * 4, H);
    matvec(A.vconv_wproj, xv, tv.data(), HK * 4, H);
    sconv4_dynamic(k.data(), tk.data(), A.kconv_static, DH, cache.conv[0],
                   k.data());
    sconv4_dynamic(v.data(), tv.data(), A.vconv_static, DH, cache.conv[1],
                   v.data());
  } else {
    sconv4_static(k.data(), A.kconv_w, cache.conv[0], k.data());
    sconv4_static(v.data(), A.vconv_w, cache.conv[1], v.data());
  }

  scaled_l2_clip(q.data(), A.qclip, HQ, DH, float(c.qk_l2_clip_threshold));
  l2_clip(k.data(), HK, DH, float(c.qk_l2_clip_threshold));
  for (int h = 0; h < HK; h++)
    rmsnorm_uw(v.data() + h * DH, v.data() + h * DH, DH,
               float(c.rms_norm_eps));
  scaled_l2_clip(r.data(), A.rclip, HQ, DR, float(c.rel_l2_clip_threshold));

  if (full) {
    // log-scaling (context extension); sliding layers never apply it
    double tau =
        1.0 + c.log_scaling_alpha *
                  std::log(std::max(double(pos + 1) / c.log_scaling_n_floor, 1.0));
    for (auto& f : q) f = float(f * tau);
    for (auto& f : r) f = float(f * tau);
  }

  cache.kv.append(k.data(), v.data());
  const int64_t len = cache.kv.len;  // == pos+1 for sequential decode

  // rel-bias profiles: rel[h][d] = r_h . proj[:,d]
  std::vector<float> rel(size_t(HQ) * ext, 0.f);
  for (int h = 0; h < HQ; h++)
    for (int d = 0; d < ext; d++) {
      float acc = 0.f;
      for (int e = 0; e < DR; e++)
        acc += r[h * DR + e] * A.relproj.f32(e, d);
      rel[size_t(h) * ext + d] = acc;
    }

  std::vector<float> gate(HQ * DH);
  matvec(A.gate, xq, gate.data(), HQ * DH, H);
  for (auto& f : gate) f = sigmoid(f);

  std::vector<float> sink(HQ);
  matvec(A.sink, xq, sink.data(), HQ, H);  // per-head sink logits [24,2304]@[2304]

  std::vector<float> probs(size_t(len) + 1);
  std::vector<float> o(HQ * DH, 0.f);
  const int64_t j0 = full ? 0 : std::max<int64_t>(0, len - c.sliding_window_size);
  const float scale = 1.f / std::sqrt(float(DH));
  for (int h = 0; h < HQ; h++) {
    int g = h * HK / HQ;  // GQA groups of 3
    const float* qh = q.data() + h * DH;
    int n = 0;
    for (int64_t j = j0; j < len; j++) {
      const float* kj = cache.kv.k.data() + j * HK * DH + g * DH;
      float s = dot(qh, kj, DH) * scale;
      int64_t d = pos - j;  // cache index == absolute position
      if (d >= 0 && d < ext) s += rel[size_t(h) * ext + d];
      probs[n++] = s;
    }
    // attention sink: extra softmax column with zero V (dropped after softmax)
    probs[n++] = sink[h];
    softmax(probs.data(), n);
    float* oh = o.data() + h * DH;
    for (int64_t j = j0; j < len; j++) {
      float p = probs[j - j0];
      const float* vj = cache.kv.v.data() + j * HK * DH + g * DH;
      for (int dd = 0; dd < DH; dd++) oh[dd] += p * vj[dd];
    }
  }
  for (int i = 0; i < HQ * DH; i++) o[i] *= gate[i];
  if (std::getenv("INK_DBG")) {
    static int call = 0;
    if (call++ == 5) {  // visit5 == first full layer
      auto w = [](const char* nm, const float* x, int n) {
        char p[256];
        std::snprintf(p, sizeof p, "testdata/ink_dbg_cpu/%s.bin", nm);
        FILE* f = std::fopen(p, "wb");
        if (f) {
          std::fwrite(x, 4, size_t(n), f);
          std::fclose(f);
        }
      };
      w("dbgA_q", q.data(), HQ * DH);
      w("dbgA_k", k.data(), HK * DH);
      w("dbgA_v", v.data(), HK * DH);
      w("dbgA_o", o.data(), HQ * DH);
    }
  }
  matvec(A.wo, o.data(), out, H, HQ * DH);
}

void attn_res_read(const float* const* srcs, int nsrc, const float* q, int Q,
                   int D, float* out) {
  const float eps = 1e-6f;
  std::vector<float> xn(size_t(nsrc) * D);
  for (int s = 0; s < nsrc; s++)
    rmsnorm_uw(srcs[s], xn.data() + size_t(s) * D, D, eps);
  const float scale = 1.f / std::sqrt(float(D));
  std::vector<float> logits(nsrc);
  for (int qi = 0; qi < Q; qi++) {
    const float* qq = q + size_t(qi) * D;
    for (int s = 0; s < nsrc; s++)
      logits[s] = dot(qq, xn.data() + size_t(s) * D, D) * scale;
    softmax(logits.data(), nsrc);
    float* o = out + size_t(qi) * D;
    std::fill(o, o + D, 0.f);
    for (int s = 0; s < nsrc; s++) {
      float p = logits[s];
      const float* x = xn.data() + size_t(s) * D;
      for (int d = 0; d < D; d++) o[d] += p * x[d];
    }
  }
}

void msg_gate(const TensorView& down, const TensorView& forget,
              const TensorView& gain, const float* dest, const float* msg,
              int D, int rank, float* out) {
  const float eps = 1e-6f;
  std::vector<float> cat(2 * D);
  rmsnorm_uw(dest, cat.data(), D, eps);
  rmsnorm_uw(msg, cat.data() + D, D, eps);
  std::vector<float> h(rank);
  matvec_bf16io(down, cat.data(), h.data(), rank, 2 * D);
  for (int d = 0; d < D; d++) {
    float fr = 0.f, gn = 0.f;
    for (int r = 0; r < rank; r++) {
      fr += forget.f32(d, r) * h[r];
      gn += gain.f32(d, r) * h[r];
    }
    out[d] = (1.f - std::tanh(fr)) * dest[d] + std::tanh(gn) * msg[d];
  }
}

void router_topk(const TensorView& gate_w, const TensorView& gate_b,
                 const float* x, int n_experts, int k, float route_scale,
                 int* idx, float* w) {
  std::vector<float> logits(n_experts);
  matvec(gate_w, x, logits.data(), n_experts, gate_w.shape[1]);
  std::vector<float> scores(n_experts);
  for (int i = 0; i < n_experts; i++) scores[i] = sigmoid(logits[i]);
  std::vector<int> ord(n_experts);
  std::iota(ord.begin(), ord.end(), 0);
  // bias participates ONLY in selection, never in the kept weights
  std::partial_sort(ord.begin(), ord.begin() + k, ord.end(), [&](int a, int b) {
    return scores[a] + gate_b.f32(a) > scores[b] + gate_b.f32(b);
  });
  float sum = 0.f;
  for (int i = 0; i < k; i++) {
    idx[i] = ord[i];
    w[i] = scores[ord[i]];
    sum += w[i];
  }
  for (int i = 0; i < k; i++) w[i] = w[i] / (sum + 1e-20f) * route_scale;
}

// Dequant row dot for expert e, out-row r, over K inputs.
static float expert_dot(const TensorView& W, const TensorView& S,
                        const TensorView& amax, const TensorView& osd,
                        bool mxfp8, int e, int64_t r, const float* x, int64_t K) {
  float gpu_out = 0.f;
  if (backend_slot() &&
      backend_slot()->expert_dot(W, S, amax, osd, mxfp8, e, r, x, K,
                                 &gpu_out)) {
    return gpu_out;
  }
  const std::byte* wrow = W.ptr(e, r, 0);
  const std::byte* srow = S.ptr(e, r, 0);
  float acc = 0.f;
  if (mxfp8) {
    for (int64_t b = 0; b < K; b += 32) {
      float bs = ue8m0_to_f32(uint8_t(srow[b >> 5]));
      for (int j = 0; j < 32; j++)
        acc += f8e4m3_to_f32(uint8_t(wrow[b + j])) * bs * x[b + j];
    }
  } else {
    float factor = amax.f32(e) / osd.f32(0);
    for (int64_t b = 0; b < K; b += 16) {
      float bs = f8e4m3_to_f32(uint8_t(srow[b >> 4]));
      for (int j = 0; j < 16; j++)
        acc += TensorView::nvfp4_elem(wrow, b + j) * bs * x[b + j];
    }
    acc *= factor;
  }
  return acc;
}

void moe_forward(const ModelConfig& c, const MlpW& m, const float* x,
                 float* out) {
  const int D = c.hidden_size, I = c.intermediate_size;
  const int k = c.num_experts_per_tok;
  std::vector<int> idx(k);
  std::vector<float> w(k);
  router_topk(m.router_w, m.router_b, x, c.n_routed_experts, k,
              float(c.route_scale), idx.data(), w.data());
  if (std::getenv("INK_DBG")) {
    static bool once = false;
    if (!once) {
      once = true;
      std::fprintf(stderr, "router idx:");
      for (int i = 0; i < k; i++) std::fprintf(stderr, " %d", idx[i]);
      std::fprintf(stderr, " w:");
      for (int i = 0; i < k; i++) std::fprintf(stderr, " %.4f", w[i]);
      std::fprintf(stderr, "\n");
    }
  }
  if (const char* idf = std::getenv("INK_IDS")) {
    static FILE* f = std::fopen(idf, "a");
    static long call = 0;
    if (f) {
      std::fprintf(f, "%ld", call++);
      for (int i = 0; i < k; i++) std::fprintf(f, " %d", idx[i]);
      std::fprintf(f, "\n");
      std::fflush(f);
    }
  }

  const ExpertW& E = m.experts;
  std::vector<float> gate(I), up(I), a(I);
  std::fill(out, out + D, 0.f);
  for (int i = 0; i < k; i++) {
    int e = idx[i];
    if (backend_slot() &&
        backend_slot()->expert_ffn(E.w13, E.w13s, E.w13amax, E.w13osd, E.w2,
                                   E.w2s, E.w2amax, E.w2osd, E.mxfp8, e, x, D,
                                   I, float(c.mlp_gate_softcap),
                                   float(c.mlp_up_softcap), w[i], out))
      continue;
    // w13 out-rows are interleaved (gate,up) pairs along dim 1: row 2m is
    // gate_m, row 2m+1 is up_m (de-interleave = reshape [1152,2].T)
    for (int mm = 0; mm < I; mm++) {
      gate[mm] = expert_dot(E.w13, E.w13s, E.w13amax, E.w13osd, E.mxfp8, e,
                            2 * mm, x, D);
      up[mm] = expert_dot(E.w13, E.w13s, E.w13amax, E.w13osd, E.mxfp8, e,
                          2 * mm + 1, x, D);
    }
    for (int mm = 0; mm < I; mm++) {
      float g = gate[mm], u = up[mm];
      // sigmoid sees the RAW gate; softcap only bounds the linear factor
      a[mm] = softcap(g, float(c.mlp_gate_softcap)) * sigmoid(g) *
              softcap(u, float(c.mlp_up_softcap));
    }
    for (int r = 0; r < D; r++)
      out[r] += w[i] * expert_dot(E.w2, E.w2s, E.w2amax, E.w2osd, E.mxfp8, e,
                                  r, a.data(), I);
  }
}

// ==================================================================
// CpuDraft — CPU reference for the MTP/DFlare speculative draft head
// (research/07 §9). Mirrors src/gpu/gpu_draft.cpp emit_block exactly:
// fp32 accumulate, bf16 rounding on residual adds and markov
// intermediates. Kept append-only; the golden gate arbitrates.
// ==================================================================

namespace {

void draft_rope_head(float* v, int64_t pos, double theta) {
  // GPT-NeoX half-split: pairs (d, d+64), inv_freq[d] = theta^(-d/64).
  for (int d = 0; d < 64; d++) {
    const double ang = double(pos) * std::pow(theta, -double(d) / 64.0);
    const float c = float(std::cos(ang)), s = float(std::sin(ang));
    const float x1 = v[d], x2 = v[d + 64];
    v[d] = x1 * c - x2 * s;
    v[d + 64] = x2 * c + x1 * s;
  }
}

}  // namespace

bool CpuDraft::init(const ModelConfig& c, const SafetensorsFile& mtp,
                    const TensorView& embed, const TensorView& embn,
                    const TensorView& unemb, std::string* err) {
  cfg = &c;
  embed_ = embed;
  embn_ = embn;
  unemb_ = unemb;
  std::vector<std::string> missing;
  auto get = [&](const char* n, TensorView& t) {
    if (const TensorView* p = mtp.find(n))
      t = *p;
    else
      missing.push_back(n);
  };
  get("fuse_weight", w.fuse_w);
  get("norm.weight", w.norm);
  get("mask_embed", w.mask_embed);
  get("markov_head.dflare_norm.weight", w.dgam[3]);
  get("markov_head.markov_w1.weight", w.w1);
  get("markov_head.rnn_proj.weight", w.rnn);
  get("markov_head.rnn_proj.bias", w.rnn_b);
  get("markov_head.hidden_out.weight", w.hout);
  get("markov_head.hidden_out.bias", w.hout_b);
  for (int i = 0; i < 3; i++) {
    char buf[64];
    auto L = [&](const char* s, TensorView& t) {
      std::snprintf(buf, sizeof(buf), "layers.%d.%s", i, s);
      get(buf, t);
    };
    L("dflare_norm.weight", w.dgam[i]);
    L("input_layernorm.weight", w.inln[i]);
    L("post_attention_layernorm.weight", w.pln[i]);
    L("self_attn.q_norm.weight", w.qn[i]);
    L("self_attn.k_norm.weight", w.kn[i]);
    L("self_attn.k_norm_ctx.weight", w.knc[i]);
    L("self_attn.q_proj.weight", w.wq[i]);
    L("self_attn.k_proj.weight", w.wk[i]);
    L("self_attn.v_proj.weight", w.wv[i]);
    L("self_attn.o_proj.weight", w.wo[i]);
    L("self_attn.k_proj_ctx.weight", w.wkc[i]);
    L("self_attn.v_proj_ctx.weight", w.wvc[i]);
    L("mlp.gate_proj.weight", w.wg[i]);
    L("mlp.up_proj.weight", w.wu[i]);
    L("mlp.down_proj.weight", w.wd[i]);
  }
  if (!missing.empty()) {
    *err = "cpu draft: " + std::to_string(missing.size()) +
           " tensors missing, first: " + missing.front();
    return false;
  }
  reset();
  return true;
}

std::vector<int> CpuDraft::propose(const float* taps4, int anchor,
                                   int64_t pos) {
  const ModelConfig& c = *cfg;
  const int H = c.mtp.hidden_size;
  const int I = c.mtp.intermediate_size;
  const int HQ = c.mtp.num_attention_heads, HK = c.mtp.num_key_value_heads;
  const int DH = c.mtp.head_dim;
  const int64_t V = c.unpadded_vocab_size;
  const float eps = float(c.mtp.rms_norm_eps);
  const bool appending = n_ctx < CTX_MAX;
  const int64_t nctx = n_ctx + (appending ? 1 : 0);

  // ---- fused contexts (fp32 mix -> rmsnorm) ----
  float alpha[4][4];
  for (int r = 0; r < 4; r++) {
    float m = -1e30f;
    for (int j = 0; j < 4; j++) m = std::max(m, w.fuse_w.f32(r, j));
    float s = 0.f;
    for (int j = 0; j < 4; j++) s += (alpha[r][j] = std::exp(w.fuse_w.f32(r, j) - m));
    for (int j = 0; j < 4; j++) alpha[r][j] /= s;
  }
  std::vector<float> ctx(4 * H), mix(H);
  for (int cc = 0; cc < 4; cc++) {
    for (int i = 0; i < H; i++)
      mix[i] = alpha[cc][0] * taps4[i] + alpha[cc][1] * taps4[H + i] +
               alpha[cc][2] * taps4[2 * H + i] +
               alpha[cc][3] * taps4[3 * H + i];
    rmsnorm(mix.data(), w.dgam[cc], ctx.data() + cc * H, H, eps);
  }

  // ---- block input: row 0 = embed_norm(anchor), rows 1..7 = mask_embed ----
  std::vector<float> x(8 * H), xn(8 * H);
  {
    std::vector<float> e(H);
    const int64_t off = int64_t(anchor) * H;
    for (int i = 0; i < H; i++) e[i] = embed_.f32_linear(off + i);
    rmsnorm(e.data(), embn_, x.data(), H, eps);
    for (int j = 1; j < 8; j++)
      for (int i = 0; i < H; i++) x[j * H + i] = w.mask_embed.f32(i);
  }

  const bool dbg = std::getenv("INK_DRAFT_DEBUG") != nullptr;
  if (dbg) dbg_xin = x;
  std::vector<float> q(8 * HQ * DH), kb(8 * HK * DH), vb(8 * HK * DH);
  std::vector<float> att(8 * HQ * DH), scr(8 * H);
  for (int li = 0; li < 3; li++) {
    // ---- ctx-KV append (anchor's fused features at this layer) ----
    std::vector<float> kc(HK * DH), vc(HK * DH);
    matvec(w.wkc[li], ctx.data() + li * H, kc.data(), HK * DH, H);
    matvec(w.wvc[li], ctx.data() + li * H, vc.data(), HK * DH, H);
    for (int h = 0; h < HK; h++)
      rmsnorm(kc.data() + h * DH, w.knc[li], kc.data() + h * DH, DH, eps);
    for (int h = 0; h < HK; h++)
      draft_rope_head(kc.data() + h * DH, pos, c.mtp.rope_theta);
    if (dbg && li == 0) dbg_ctxk0 = kc;
    if (appending) {
      ctxk[li].insert(ctxk[li].end(), kc.begin(), kc.end());
      ctxv[li].insert(ctxv[li].end(), vc.begin(), vc.end());
      if (li == 0) ctxpos.push_back(pos);
    }

    // ---- block layer ----
    for (int j = 0; j < 8; j++)
      rmsnorm(x.data() + j * H, w.inln[li], xn.data() + j * H, H, eps);
    if (dbg && li == 0) dbg_xn0 = xn;
    for (int j = 0; j < 8; j++) {
      matvec(w.wq[li], xn.data() + j * H, q.data() + j * HQ * DH, HQ * DH, H);
      matvec(w.wk[li], xn.data() + j * H, kb.data() + j * HK * DH, HK * DH, H);
      matvec(w.wv[li], xn.data() + j * H, vb.data() + j * HK * DH, HK * DH, H);
    }
    for (int j = 0; j < 8; j++) {
      for (int h = 0; h < HQ; h++)
        rmsnorm(q.data() + (j * HQ + h) * DH, w.qn[li],
                q.data() + (j * HQ + h) * DH, DH, eps);
      for (int h = 0; h < HK; h++)
        rmsnorm(kb.data() + (j * HK + h) * DH, w.kn[li],
                kb.data() + (j * HK + h) * DH, DH, eps);
      for (int h = 0; h < HQ; h++)
        draft_rope_head(q.data() + (j * HQ + h) * DH, pos + j,
                        c.mtp.rope_theta);
      for (int h = 0; h < HK; h++)
        draft_rope_head(kb.data() + (j * HK + h) * DH, pos + j,
                        c.mtp.rope_theta);
    }
    if (dbg && li == 0) {
      dbg_q0 = q;
      dbg_k0 = kb;
    }
    // non-causal block attention over [ctx ++ block] keys, GQA g = h/3
    const bool local = std::find(c.mtp.local_layer_ids.begin(),
                                 c.mtp.local_layer_ids.end(),
                                 li) != c.mtp.local_layer_ids.end();
    const float scale = float(1.0 / std::sqrt(double(DH)));
    std::vector<float> s(nctx + 8);
    for (int j = 0; j < 8; j++) {
      const int64_t qpos = pos + j;
      for (int h = 0; h < HQ; h++) {
        const int g = h * HK / HQ;
        const float* qh = q.data() + (j * HQ + h) * DH;
        for (int64_t kk = 0; kk < nctx; kk++) {
          const int64_t kpos = ctxpos[size_t(kk)];
          const bool vis =
              !local || (kpos <= qpos &&
                         (qpos - kpos) < c.mtp.sliding_window_size);
          s[kk] = vis ? dot(qh, ctxk[li].data() + kk * HK * DH + g * DH, DH) *
                            scale
                      : -1e30f;
        }
        for (int kk = 0; kk < 8; kk++)
          s[nctx + kk] =
              dot(qh, kb.data() + (kk * HK + g) * DH, DH) * scale;
        softmax(s.data(), int(nctx + 8));
        float* oh = att.data() + (j * HQ + h) * DH;
        for (int d = 0; d < DH; d++) {
          float a = 0.f;
          for (int64_t kk = 0; kk < nctx; kk++)
            a += s[kk] * ctxv[li][size_t(kk) * HK * DH + g * DH + d];
          for (int kk = 0; kk < 8; kk++)
            a += s[nctx + kk] * vb[size_t(kk) * HK * DH + g * DH + d];
          oh[d] = a;
        }
      }
    }
    if (dbg && li == 0) dbg_att0 = att;
    for (int j = 0; j < 8; j++) {
      matvec(w.wo[li], att.data() + j * HQ * DH, scr.data() + j * H, H,
             HQ * DH);
      for (int i = 0; i < H; i++)
        x[j * H + i] = bf16_round(x[j * H + i] + scr[j * H + i]);
    }
    for (int j = 0; j < 8; j++)
      rmsnorm(x.data() + j * H, w.pln[li], xn.data() + j * H, H, eps);
    std::vector<float> gg(I), uu(I), aa(I);
    for (int j = 0; j < 8; j++) {
      matvec(w.wg[li], xn.data() + j * H, gg.data(), I, H);
      matvec(w.wu[li], xn.data() + j * H, uu.data(), I, H);
      for (int i = 0; i < I; i++)
        aa[i] = gg[i] * sigmoid(gg[i]) * uu[i];
      matvec(w.wd[li], aa.data(), scr.data() + j * H, H, I);
      for (int i = 0; i < H; i++)
        x[j * H + i] = bf16_round(x[j * H + i] + scr[j * H + i]);
    }
    if (dbg)
      dbg_xall.insert(dbg_xall.end(), x.begin(), x.end());
  }

  // ---- final norm + base logits ----
  std::vector<float> out8(8 * H);
  for (int j = 0; j < 8; j++)
    rmsnorm(x.data() + j * H, w.norm, out8.data() + j * H, H, eps);
  std::vector<float> base(8 * V);
  for (int j = 0; j < 8; j++)
    matvec(unemb_, out8.data() + j * H, base.data() + j * V, V, H);
  if (dbg) {
    dbg_ctx = ctx;
    dbg_out8 = out8;
    dbg_base0.assign(base.data(), base.data() + V);
  }

  // ---- markov head: 7 steps (DFlash fill-in layout) ----
  // Slot 0 is the anchor seed (unsampled); markov step i consumes slot
  // i+1's hidden/base — a mask slot at position pos+1+i predicting the
  // token AT that same position. ids[0..6] predict pos+1 .. pos+7;
  // ids[7] = spare argmax over slot-7 base logits.
  std::vector<int> ids(8, 0);
  std::vector<float> state(256, 0.f), lat(256), in(5120), proj(768),
      dvec(H);
  int prev = anchor;
  for (int i = 0; i < 7; i++) {
    for (int k = 0; k < 256; k++) in[k] = bf16_round(state[k]);
    for (int k = 0; k < 256; k++)
      in[256 + k] = w.w1.f32_linear(int64_t(prev) * 256 + k);  // already bf16
    for (int k = 0; k < H; k++)
      in[512 + k] = bf16_round(out8[(i + 1) * H + k]);  // fill-in slot i+1
    for (int k = 0; k < H; k++)
      in[512 + H + k] = bf16_round(ctx[3 * H + k]);
    matvec(w.rnn, in.data(), proj.data(), 768, 5120);
    for (int k = 0; k < 768; k++)
      proj[k] = bf16_round(proj[k] + w.rnn_b.f32(k));
    for (int k = 0; k < 256; k++) {
      const float gate = bf16_round(sigmoid(proj[k]));
      const float cand = bf16_round(std::tanh(proj[256 + k]));
      lat[k] = bf16_round(std::tanh(proj[512 + k]));
      state[k] = bf16_round(gate * state[k] + (1.f - gate) * cand);
    }
    matvec(w.hout, lat.data(), dvec.data(), H, 256);
    for (int d = 0; d < H; d++) dvec[d] += w.hout_b.f32(d);
    std::vector<float> tmp(V);
    matvec(unemb_, dvec.data(), tmp.data(), V, H);
    for (int64_t v = 0; v < V; v++)
      tmp[v] =
          30.f * std::tanh(
              bf16_round(bf16_round(base[(i + 1) * V + v]) + tmp[v]) / 30.f);
    ids[i] = argmax(tmp.data(), int(V));
    prev = ids[i];
  }
  ids[7] = argmax(base.data() + 7 * V, int(V));
  if (appending) n_ctx++;
  return ids;
}

}  // namespace ink::ops
