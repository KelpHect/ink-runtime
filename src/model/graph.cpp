#include "graph.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace ink {

namespace {
std::string lname(int i, const char* suffix) {
  return "model.llm.layers." + std::to_string(i) + "." + suffix;
}
// Decode a (possibly unaligned) TensorView into an fp32 buffer.
void load_f32(const TensorView& t, float* out) {
  int64_t n = t.numel();
  for (int64_t i = 0; i < n; i++) out[i] = t.f32_linear(i);
}
}  // namespace

bool Graph::init(const ModelConfig& c, const WeightStore& store,
                 std::string* err) {
  cfg_ = &c;
  H_ = c.hidden_size;
  if (c.num_hidden_layers != 24 ||
      c.loop.core_end - c.loop.core_start != 12) {
    *err = "fixed-shape graph supports exactly 24 layers with a 12-layer "
           "loop core (this checkpoint)";
    return false;
  }
  std::vector<std::string> missing;
  auto get = [&](const std::string& n, TensorView* t) -> bool {
    const TensorView* v = store.find(n);
    if (v) *t = *v;
    else missing.push_back(n);
    return v != nullptr;
  };

  get("model.llm.embed.weight", &w_.embed);
  get("model.llm.embed_norm.weight", &w_.embed_norm);
  get("model.llm.norm.weight", &w_.norm);
  get("model.llm.unembed.weight", &w_.unembed);
  get("model.llm.attn_res.output_query", &w_.output_q);
  for (int i = 0; i < 23; i++)
    get("model.llm.attn_res.attn_queries." + std::to_string(i), &w_.attn_q[i]);
  for (int i = 0; i < 24; i++)
    get("model.llm.attn_res.mlp_queries." + std::to_string(i), &w_.mlp_q[i]);

  // Merge message queries: first tensor = per-layer [q, mlp] row pairs,
  // kv tensor = [k, v] pairs; merged order is (q, k, v, mlp) per loop layer.
  TensorView mq, mkv;
  get("model.llm.middle_cycle_cross_pass_queries", &mq);
  get("model.llm.middle_cycle_cross_pass_kv_queries", &mkv);
  if (mq.data && mkv.data) {
    int ncore = c.loop.core_end - c.loop.core_start;
    w_.msg_q.resize(size_t(ncore) * 4 * H_);
    for (int li = 0; li < ncore; li++)
      for (int e = 0; e < H_; e++) {
        w_.msg_q[(size_t(li) * 4 + 0) * H_ + e] = mq.f32(2 * li, e);
        w_.msg_q[(size_t(li) * 4 + 1) * H_ + e] = mkv.f32(2 * li, e);
        w_.msg_q[(size_t(li) * 4 + 2) * H_ + e] = mkv.f32(2 * li + 1, e);
        w_.msg_q[(size_t(li) * 4 + 3) * H_ + e] = mq.f32(2 * li + 1, e);
      }
  }

  static const char* kCtrlName[4] = {"attn", "k", "v", "mlp"};
  static const char* kCtrlMat[3] = {"down_weight", "forget_weight",
                                    "gain_weight"};
  for (int i = 0; i < c.num_hidden_layers; i++) {
    LayerW& L = w_.L[i];
    L.full = c.is_full_attn(i);
    L.loop = c.is_loop(i);
    std::string p = lname(i, "attn.");
    get(p + "wq_du.weight", &L.attn.wq);
    get(p + "wk_dv.weight", &L.attn.wk);
    get(p + "wv_dv.weight", &L.attn.wv);
    get(p + "wo_ud.weight", &L.attn.wo);
    get(p + "gate_attn.weight", &L.attn.gate);
    get(p + "attn_sink_proj.weight", &L.attn.sink);
    get(p + "wr_du.weight", &L.attn.wr);
    get(p + "rel_logits_proj.proj", &L.attn.relproj);
    get(p + "q_l2_clip.scale", &L.attn.qclip);
    get(p + "rel_l2_clip.scale", &L.attn.rclip);
    if (L.full) {
      get(p + "k_sconv.static_w", &L.attn.kconv_static);
      get(p + "k_sconv.weight_proj", &L.attn.kconv_wproj);
      get(p + "v_sconv.static_w", &L.attn.vconv_static);
      get(p + "v_sconv.weight_proj", &L.attn.vconv_wproj);
      get(lname(i, "q_input_norm.weight"), &L.qnorm);
      get(lname(i, "k_input_norm.weight"), &L.knorm);
      get(lname(i, "v_input_norm.weight"), &L.vnorm);
    } else {
      get(p + "k_sconv.weight", &L.attn.kconv_w);
      get(p + "v_sconv.weight", &L.attn.vconv_w);
      get(lname(i, "attn_norm.weight"), &L.attn_norm);
    }
    get(lname(i, "attn_sconv.weight"), &L.attn_sconv);
    get(lname(i, "mlp_sconv.weight"), &L.mlp_sconv);
    get(lname(i, "mlp_norm.weight"), &L.mlp_norm);
    get(lname(i, "mlp.gate.weight"), &L.mlp.router_w);
    get(lname(i, "mlp.gate.bias"), &L.mlp.router_b);
    get(lname(i, "mlp.experts.w13_weight"), &L.mlp.experts.w13);
    get(lname(i, "mlp.experts.w13_weight_scale"), &L.mlp.experts.w13s);
    get(lname(i, "mlp.experts.w2_weight"), &L.mlp.experts.w2);
    get(lname(i, "mlp.experts.w2_weight_scale"), &L.mlp.experts.w2s);
    L.mlp.experts.mxfp8 = c.mxfp8_layers.count(i) != 0;
    if (!L.mlp.experts.mxfp8) {
      get(lname(i, "mlp.experts.w13_weight_amax"), &L.mlp.experts.w13amax);
      get(lname(i, "mlp.experts.w13_weight_outer_scale_denominator"),
          &L.mlp.experts.w13osd);
      get(lname(i, "mlp.experts.w2_weight_amax"), &L.mlp.experts.w2amax);
      get(lname(i, "mlp.experts.w2_weight_outer_scale_denominator"),
          &L.mlp.experts.w2osd);
      if (L.mlp.experts.w13.data &&
          L.mlp.experts.w13.dtype != DType::U8) {
        *err = "layer " + std::to_string(i) +
               ": expected U8-packed NVFP4 experts";
        return false;
      }
    }
    if (L.loop)
      for (int st = 0; st < 4; st++)
        for (int mt = 0; mt < 3; mt++)
          get(lname(i, "controllers.") + kCtrlName[st] + "." + kCtrlMat[mt],
              &L.ctrl[st][mt]);
  }

  if (!missing.empty()) {
    std::ostringstream os;
    os << missing.size() << " missing tensors, first: " << missing.front();
    *err = os.str();
    return false;
  }
  reset();
  return true;
}

void Graph::reset() {
  s_.block.clear();
  s_.partial.assign(H_, 0.f);
  s_.pre_block.clear();
  s_.pre_partial.clear();
  s_.bank0.clear();
  s_.bank1.clear();
  for (auto& c : s_.cache) {
    c.kv.k.clear();
    c.kv.v.clear();
    c.kv.len = 0;
    for (auto& cv : c.conv) cv = ops::ConvState4{};
  }
  const int H = int(H_);
  s_.reads.assign(3 * H, 0.f);
  s_.msgs.assign(4 * H, 0.f);
  s_.qbuf.assign(4 * H, 0.f);
  s_.xq.assign(H, 0.f);
  s_.xk.assign(H, 0.f);
  s_.xv.assign(H, 0.f);
  s_.attn_out.assign(H, 0.f);
  s_.a_out.assign(H, 0.f);
  s_.residual.assign(H, 0.f);
  s_.m_in.assign(H, 0.f);
  s_.m_gated.assign(H, 0.f);
  s_.moe_out.assign(H, 0.f);
  s_.mo.assign(H, 0.f);
  s_.h_fin.assign(H, 0.f);
  s_.logits.assign(cfg_->unpadded_vocab_size, 0.f);
  for (auto& t : s_.taps) t.clear();
}

// INK_DBG=1: dump visit0 stage intermediates to testdata/ink_dbg_cpu/dbg*.bin
// (mirrors GpuGraph's dbg_cp rows for cross-backend bisection).
static void dbg_dump(int row, const float* x, int n) {
  if (!std::getenv("INK_DBG")) return;
  static bool once[16] = {};
  if (once[row]) return;
  once[row] = true;
  char p[256];
  std::snprintf(p, sizeof p, "testdata/ink_dbg_cpu/dbg%d.bin", row);
  FILE* f = std::fopen(p, "wb");
  if (!f) return;
  std::fwrite(x, 4, size_t(n), f);
  std::fclose(f);
}

void Graph::visit(int i, int p, int64_t pos) {
  const ModelConfig& c = *cfg_;
  const int H = int(H_);
  LayerW& L = w_.L[i];
  ops::CacheSlot& cache = s_.cache[c.cache_slot(i, p)];
  const int vidx = c.visit_index(i, p);

  // MTP hidden taps: visit inputs at {5,23,41}
  if (vidx == 5) s_.taps[0] = s_.partial;
  else if (vidx == 23) s_.taps[1] = s_.partial;
  else if (vidx == 41) s_.taps[2] = s_.partial;
  if (visit_hook) visit_hook(vidx, 0, s_.partial);

  // --- messages from previous pass (loop layers, pass >= 1) ---
  const float* msgs = nullptr;
  if (p > 0) {
    const float* srcs[2] = {s_.bank0.data(), s_.bank1.data()};
    ops::attn_res_read(srcs, 2,
                       w_.msg_q.data() + size_t(i - c.loop.core_start) * 4 * H,
                       4, H, s_.msgs.data());
    msgs = s_.msgs.data();
  }

  const float* x_in = s_.partial.data();

  // --- attention input read (attn_res), queries indexed i-1 (trap: +1) ---
  int Q = 1;
  if (i > 0) {
    const TensorView& aq = w_.attn_q[i - 1];
    Q = (aq.nd == 2) ? 3 : 1;  // full layers store [3,2304] q/k/v queries
    load_f32(aq, s_.qbuf.data());
    std::vector<const float*> srcs;
    for (auto& b : s_.block) srcs.push_back(b.data());
    srcs.push_back(x_in);
    ops::attn_res_read(srcs.data(), int(srcs.size()), s_.qbuf.data(), Q, H,
                       s_.reads.data());
  } else {
    std::memcpy(s_.reads.data(), x_in, sizeof(float) * H);
  }
  if (i == 5)
    for (int st = 0; st < 3; st++)
      dbg_dump(st, s_.reads.data() + size_t(st) * H, H);

  // --- stream inputs: message gates (pass>0 loop), then sublayer norms ---
  float* in[3] = {s_.xq.data(), s_.xk.data(), s_.xv.data()};
  if (L.full) {
    const TensorView* norms[3] = {&L.qnorm, &L.knorm, &L.vnorm};
    for (int st = 0; st < 3; st++) {
      const float* dest = s_.reads.data() + size_t(st) * H;
      if (msgs)
        ops::msg_gate(L.ctrl[st][0], L.ctrl[st][1], L.ctrl[st][2], dest,
                      msgs + size_t(st) * H, H, c.loop.gate_rank,
                      s_.m_gated.data());
      ops::rmsnorm(msgs ? s_.m_gated.data() : dest, *norms[st], in[st], H,
                   float(c.rms_norm_eps));
    }
  } else {
    const float* dest = s_.reads.data();
    for (int st = 0; st < 3; st++) {
      if (msgs)
        ops::msg_gate(L.ctrl[st][0], L.ctrl[st][1], L.ctrl[st][2], dest,
                      msgs + size_t(st) * H, H, c.loop.gate_rank,
                      s_.m_gated.data());
      ops::rmsnorm(msgs ? s_.m_gated.data() : dest, L.attn_norm, in[st], H,
                   float(c.rms_norm_eps));
    }
  }

  if (i == 5)
    for (int st = 0; st < 3; st++) dbg_dump(3 + st, in[st], H);

  // --- attention + post-attn conv (stream 2) ---
  ops::attn_step(c, L.attn, L.full, in[0], in[1], in[2], cache, pos,
                 s_.attn_out.data());
  if (i == 5) dbg_dump(7, s_.attn_out.data(), H);
  ops::sconv4_static(s_.attn_out.data(), L.attn_sconv, cache.conv[2],
                     s_.a_out.data());

  // --- block boundary: push the layer INPUT, residual restarts ---
  if (i % c.n_layers_per_attn_res_block == 0) {
    s_.block.emplace_back(x_in, x_in + H);
    s_.residual = s_.a_out;
  } else {
    for (int d = 0; d < H; d++) s_.residual[d] = x_in[d] + s_.a_out[d];
  }
  if (i == 5) dbg_dump(8, s_.residual.data(), H);

  // --- mlp input read ---
  {
    load_f32(w_.mlp_q[i], s_.qbuf.data());
    std::vector<const float*> srcs;
    for (auto& b : s_.block) srcs.push_back(b.data());
    srcs.push_back(s_.residual.data());
    ops::attn_res_read(srcs.data(), int(srcs.size()), s_.qbuf.data(), 1, H,
                       s_.m_in.data());
  }
  if (i == 5) dbg_dump(9, s_.m_in.data(), H);
  if (msgs)
    ops::msg_gate(L.ctrl[3][0], L.ctrl[3][1], L.ctrl[3][2], s_.m_in.data(),
                  msgs + size_t(3) * H, H, c.loop.gate_rank, s_.m_gated.data());
  {
    std::vector<float> mn(H);
    ops::rmsnorm(msgs ? s_.m_gated.data() : s_.m_in.data(), L.mlp_norm,
                 mn.data(), H, float(c.rms_norm_eps));
    if (i == 5) dbg_dump(10, mn.data(), H);
    ops::moe_forward(c, L.mlp, mn.data(), s_.moe_out.data());
  }
  if (i == 5) dbg_dump(11, s_.moe_out.data(), H);
  ops::sconv4_static(s_.moe_out.data(), L.mlp_sconv, cache.conv[3],
                     s_.mo.data());
  if (i == 5) dbg_dump(12, s_.mo.data(), H);
  for (int d = 0; d < H; d++) s_.partial[d] = s_.residual[d] + s_.mo[d];
  if (visit_hook) visit_hook(vidx, 1, s_.partial);
}

const std::vector<float>& Graph::forward_token(int token, int64_t pos) {
  const ModelConfig& c = *cfg_;
  const int H = int(H_);
  const float eps = float(c.rms_norm_eps);

  // h0 = embed_norm(embed[token])
  {
    std::vector<float> e(H);
    const std::byte* row = w_.embed.ptr(token, 0);
    if (w_.embed.dtype == DType::BF16)
      for (int d = 0; d < H; d++)
        e[d] = bf16_to_f32(reinterpret_cast<const uint16_t*>(row)[d]);
    else
      for (int d = 0; d < H; d++) e[d] = w_.embed.f32(token, d);
    ops::rmsnorm(e.data(), w_.embed_norm, s_.partial.data(), H, eps);
  }

  s_.block.clear();

  for (int i = 0; i < c.loop.core_start; i++) visit(i, 0, pos);
  s_.pre_partial = s_.partial;
  s_.pre_block = s_.block;  // == [emb] for this config
  for (int p = 0; p < c.loop.num_passes; p++) {
    if (p > 0) {
      // bank = block_states[num_pre+1:] + prev partial; here exactly
      // [s12_prevpass, prev_final]
      s_.bank0 = s_.block.back();
      s_.bank1 = s_.partial;
      s_.partial = s_.pre_partial;
      s_.block = s_.pre_block;
    }
    for (int i = c.loop.core_start; i < c.loop.core_end; i++) visit(i, p, pos);
  }
  for (int i = c.loop.core_end; i < c.num_hidden_layers; i++)
    visit(i, 0, pos);

  // final learned read + final norm
  {
    load_f32(w_.output_q, s_.qbuf.data());
    std::vector<const float*> srcs;
    for (auto& b : s_.block) srcs.push_back(b.data());
    srcs.push_back(s_.partial.data());
    std::vector<float> h(H);
    ops::attn_res_read(srcs.data(), int(srcs.size()), s_.qbuf.data(), 1, H,
                       h.data());
    ops::rmsnorm(h.data(), w_.norm, s_.h_fin.data(), H, eps);
    s_.taps[3] = s_.h_fin;  // visit id 48
  }

  // logits = unembed[:unpadded] @ h, fp32 out, softcap 30
  ops::matvec(w_.unembed, s_.h_fin.data(), s_.logits.data(),
              c.unpadded_vocab_size, H);
  float cap = float(c.final_logit_softcapping);
  for (auto& l : s_.logits) l = cap * std::tanh(l / cap);
  return s_.logits;
}

}  // namespace ink
