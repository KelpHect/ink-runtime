// gpu_prefill.cpp — batched-chunk prompt prefill (row-batched emit_token).
//
// Processes T prompt positions per chunk through all 48 visits in one
// recording; every activation is a T-row mirror of the GpuGraph buffer.
// Shares the graph's context/allocator/weights/pipelines and persistent
// state (kvk_/kvv_, hist_) — after prefill, decode through
// GpuGraph::forward_token is byte-identical to P sequential calls.
// Op sequence mirrors Graph::forward_token / GpuGraph::emit_token exactly;
// see docs/PREFILL.md for the design.

#include "gpu/gpu_prefill.h"

#include "bindings.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace ink::gpu {
namespace sh = ink::shaders;
namespace clk = std::chrono;

namespace {
// PC layouts for the small kernels shared with the decode path.
struct PcN {
  uint32_t n;
};
struct PcCap {
  uint32_t n;
  float cap;
};
constexpr uint32_t GRP = 64;
inline uint32_t wg64(int64_t n) { return uint32_t((n + GRP - 1) / GRP); }
}  // namespace

static GpuGraph::Bnd BW(const ink::vk::Buffer& b) { return {&b, 0, 0}; }
static GpuGraph::Bnd BA(const ink::vk::Buffer& b, int64_t fe, int64_t n) {
  return {&b, VkDeviceSize(fe) * 4, VkDeviceSize(n) * 4};
}
// Rest-of-buffer slice at a float element offset: a fixed, chunk-invariant
// binding whose range always ends at the buffer end (never OOB).
static GpuGraph::Bnd REST(const ink::vk::Buffer& b, int64_t fe) {
  return {&b, VkDeviceSize(fe) * 4, b.size - VkDeviceSize(fe) * 4};
}

// ------------------------------------------------------- emit helpers

const ink::vk::DescSet& GpuPrefill::dset(const ink::vk::ComputePipeline& p,
                                         std::initializer_list<Bnd> bs) {
  if (!building_) return sites_[site_cursor_++];
  sites_.emplace_back();
  ink::vk::DescSet& ds = sites_.back();
  if (!vk::alloc_descset(*g_->ctx_, p.dsl, uint32_t(bs.size()), ds, err_))
    return ds;
  uint32_t i = 0;
  for (const Bnd& b : bs) {
    if (!vk::write_storage(*g_->ctx_, ds, i++, b.b->buf, b.off,
                           b.sz ? b.sz : b.b->size, err_))
      break;
  }
  return ds;
}

void GpuPrefill::op(const ink::vk::ComputePipeline& p,
                    std::initializer_list<Bnd> bs, const void* pc,
                    uint32_t pcb, uint32_t gx, uint32_t gy, uint32_t gz) {
  const vk::DescSet& ds = dset(p, bs);
  if (building_) return;
  rec_->bind_pipeline(p.pipe);
  rec_->bind_descset(p.layout, ds.set);
  if (pcb) rec_->push_constants(p.layout, pc, pcb);
  rec_->dispatch(gx, gy, gz);
  stats.dispatches++;
}

void GpuPrefill::cp(const ink::vk::Buffer& s, VkDeviceSize so,
                    const ink::vk::Buffer& d, VkDeviceSize dof,
                    VkDeviceSize sz) {
  if (building_) return;
  rec_->copy_buffer(s.buf, d.buf, sz, so, dof);
  stats.copies++;
}

void GpuPrefill::sync() {
  if (building_) return;
  rec_->barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
  stats.barriers++;
}

GpuPrefill::Bnd GpuPrefill::h3b(int slot, int stream) const {
  const int64_t H = g_->H_;
  return {&h3_seq_, VkDeviceSize((int64_t(slot) * 4 + stream) * 3 * H) * 4,
          VkDeviceSize(3 * H) * 4};
}

void GpuPrefill::mv_seq(const GpuGraph::WB& w, Bnd x,
                        const ink::vk::Buffer& o, int64_t M, int64_t K,
                        int64_t xs, int t) {
  const ink::vk::ComputePipeline* p = nullptr;
  if (w.p == &g_->p_mvf_) p = &p_mvf_seq_;
  else if (w.p == &g_->p_mvb_) p = &p_mvb_seq_;
  else if (w.p == &g_->p_mv8_) p = &p_mv8_seq_;
  if (!p) {
    if (err_.empty()) err_ = "gpu_prefill: matvec has no seq pipe for dtype";
    return;
  }
  sh::PcMatvecSeq pc{uint32_t(K), 0u, uint32_t(xs), uint32_t(M)};
  op(*p, {w.b ? Bnd{w.b, 0, 0} : Bnd{&g_->dummy4_, 0, 0}, x, BW(o)}, &pc,
     sizeof(pc), uint32_t(M), uint32_t(t));
}

// -------------------------------------------------------------- convs

void GpuPrefill::conv_seq_static(const ink::vk::Buffer& x,
                                 const ink::vk::Buffer& w, int slot, int stream,
                                 int64_t C, const ink::vk::Buffer& out, int t,
                                 bool last) {
  const uint32_t fl = sh::F_SEQ_STATE | (last ? sh::F_SEQ_EXPORT : 0u);
  {
    sh::PcConv4StaticSeq pc{uint32_t(C), uint32_t(t), 0u};
    op(p_cvs_seq_, {BW(x), BW(w), h3b(slot, stream), BW(out),
                    g_->histb(slot, stream, int(C))},
       &pc, sizeof(pc), wg64(C), uint32_t(t));
  }
  sync();  // WAR: state pass rewrites h3 after the compute pass reads it
  {
    sh::PcConv4StaticSeq pc{uint32_t(C), uint32_t(t), fl};
    op(p_cvs_seq_, {BW(x), BW(w), h3b(slot, stream), BW(out),
                    g_->histb(slot, stream, int(C))},
       &pc, sizeof(pc), wg64(C), 1);
  }
  sync();
}

void GpuPrefill::conv_seq_dyn(const ink::vk::Buffer& x,
                              const ink::vk::Buffer& sw,
                              const ink::vk::Buffer& dyn, int slot, int stream,
                              int64_t C, const ink::vk::Buffer& out, int t,
                              bool last) {
  const uint32_t fl = sh::F_SEQ_STATE | (last ? sh::F_SEQ_EXPORT : 0u);
  {
    sh::PcConv4DynSeq pc{uint32_t(C), 128u, uint32_t(t), 0u};
    op(p_cvd_seq_, {BW(x), BW(sw), BW(dyn), h3b(slot, stream), BW(out),
                    g_->histb(slot, stream, int(C))},
       &pc, sizeof(pc), wg64(C), uint32_t(t));
  }
  sync();
  {
    sh::PcConv4DynSeq pc{uint32_t(C), 128u, uint32_t(t), fl};
    op(p_cvd_seq_, {BW(x), BW(sw), BW(dyn), h3b(slot, stream), BW(out),
                    g_->histb(slot, stream, int(C))},
       &pc, sizeof(pc), wg64(C), 1);
  }
  sync();
}

// ------------------------------------------------------------------ visit

void GpuPrefill::emit_visit_seq(int i, int p, int64_t pos0, int64_t base,
                                int t, bool last) {
  const ModelConfig& c = *g_->cfg_;
  const int H = g_->H_;
  const int64_t Hb = int64_t(H) * 4;
  const int vidx = c.visit_index(i, p);
  const int slot = c.cache_slot(i, p);
  const GpuGraph::Lw& L = g_->lw_[i];
  const float eps = float(c.rms_norm_eps);
  const uint32_t rank = uint32_t(c.loop.gate_rank);
  const int64_t tl = int64_t(t);
  const int64_t lr = tl - 1;  // last row index (gate dumps)

  // last-row taps + visit-input dump (gate readback), only on the last chunk
  if (last) {
    if (vidx == 5 || vidx == 23 || vidx == 41) {
      const int j = vidx == 5 ? 0 : vidx == 23 ? 1 : 2;
      cp(partial_seq_, lr * Hb, g_->readback_,
         (g_->V_ + int64_t(j) * H) * 4, Hb);
    }
    cp(partial_seq_, lr * Hb, g_->readback_,
       (g_->rb_vis_ + int64_t(vidx * 2) * H) * 4, Hb);
  }
  const bool dbg = dbg_en_ && last && vidx == 5;
  auto dbg_row = [&](const vk::Buffer& s, int64_t row_elems, int row,
                     VkDeviceSize bytes) {
    if (dbg && !building_)
      cp(s, lr * row_elems * 4, g_->readback_,
         (g_->rb_dbg_ + int64_t(row) * H) * 4, bytes);
  };

  // messages from the previous pass (loop layers only)
  if (p > 0) {
    sh::PcAttnResSeq pc{2u, uint32_t(H), 4u, uint32_t(2 * H), uint32_t(4 * H)};
    op(p_ares_seq_,
       {BW(bank_seq_),
        BA(g_->msgq_, int64_t(i - c.loop.core_start) * 4 * H, 4 * H),
        BW(msgs_seq_)},
       &pc, sizeof(pc), uint32_t(tl));
    sync();
  }

  // attn_res read: block[0..nb0) ++ x_in (x_in appended at row nb0 per row)
  const int nb0 = nblock_;
  for (int64_t r = 0; r < tl; r++)
    cp(partial_seq_, r * Hb, block_seq_, (r * 5 + nb0) * Hb, Hb);
  if (i > 0) {
    sync();
    sh::PcAttnResSeq pc{uint32_t(nb0 + 1), uint32_t(H), L.full ? 3u : 1u,
                        uint32_t(5 * H), uint32_t(3 * H)};
    op(p_ares_seq_,
       {BW(block_seq_), BA(g_->attnq_, int64_t(i - 1) * 3 * H,
                           int64_t(pc.Q) * H),
        BW(reads_seq_)},
       &pc, sizeof(pc), uint32_t(tl));
  } else {
    for (int64_t r = 0; r < tl; r++)
      cp(partial_seq_, r * Hb, reads_seq_, r * 3 * Hb, Hb);
  }
  sync();

  // stream inputs: optional message gating, then per-stream norms.
  const bool uni = !L.full && p == 0;
  const int64_t xo[3] = {0, uni ? 0 : int64_t(H), uni ? 0 : int64_t(2 * H)};
  if (p > 0) {
    for (int st = 0; st < 3; st++) {
      sh::PcMsgGateSeq pc{uint32_t(H), rank, uint32_t(3 * H), uint32_t(4 * H),
                          uint32_t(3 * H)};
      op(p_msg_seq_,
         {BW(*L.ctrl[st][0]), BW(*L.ctrl[st][1]), BW(*L.ctrl[st][2]),
          REST(reads_seq_, (L.full ? st : 0) * H), REST(msgs_seq_, st * H),
          REST(mgated_seq_, st * H)},
         &pc, sizeof(pc), uint32_t(tl));
    }
    sync();
    if (L.full) {
      for (int st = 0; st < 3; st++) {
        sh::PcRmsnormSeq pc{uint32_t(H), eps, sh::F_USE_GAMMA,
                            uint32_t(3 * H), uint32_t(3 * H)};
        op(p_rms_seq_,
           {REST(mgated_seq_, st * H), BW(*L.norms[st]),
            REST(inbuf_seq_, st * H)},
           &pc, sizeof(pc), uint32_t(tl));
      }
    } else {
      // sliding: all three streams read dst row0 -> contiguous T*3 rows
      sh::PcRmsnormSeq pc{uint32_t(H), eps, sh::F_USE_GAMMA, uint32_t(H),
                          uint32_t(H)};
      op(p_rms_seq_, {BW(mgated_seq_), BW(*L.norms[0]), BW(inbuf_seq_)}, &pc,
         sizeof(pc), uint32_t(tl * 3));
    }
  } else {
    if (L.full) {
      for (int st = 0; st < 3; st++) {
        sh::PcRmsnormSeq pc{uint32_t(H), eps, sh::F_USE_GAMMA,
                            uint32_t(3 * H), uint32_t(3 * H)};
        op(p_rms_seq_,
           {REST(reads_seq_, st * H), BW(*L.norms[st]),
            REST(inbuf_seq_, st * H)},
           &pc, sizeof(pc), uint32_t(tl));
      }
    } else {
      sh::PcRmsnormSeq pc{uint32_t(H), eps, sh::F_USE_GAMMA, uint32_t(3 * H),
                          uint32_t(3 * H)};
      op(p_rms_seq_, {BW(reads_seq_), BW(*L.norms[0]), BW(inbuf_seq_)}, &pc,
         sizeof(pc), uint32_t(tl));
    }
  }
  sync();

  // ---------------- attention (ops::attn_step mirror) ----------------
  mv_seq(L.wq, REST(inbuf_seq_, xo[0]), q_seq_, 3072, H, 3 * H, t);
  mv_seq(L.wk, REST(inbuf_seq_, xo[1]), kraw_seq_, 1024, H, 3 * H, t);
  mv_seq(L.wv, REST(inbuf_seq_, xo[2]), vraw_seq_, 1024, H, 3 * H, t);
  mv_seq(L.wr, REST(inbuf_seq_, xo[0]), r_seq_, 384, H, 3 * H, t);
  sync();

  dbg_row(kraw_seq_, 1024, 14, 4096);  // pre-conv k (visit5)
  if (L.full) {
    mv_seq(L.kconv_wp, REST(inbuf_seq_, xo[1]), tk_seq_, 32, H, 3 * H, t);
    mv_seq(L.vconv_wp, REST(inbuf_seq_, xo[2]), tv_seq_, 32, H, 3 * H, t);
    sync();
    conv_seq_dyn(kraw_seq_, *L.kconv_sw, tk_seq_, slot, 0, 1024, kout_seq_, t,
                 last);
    conv_seq_dyn(vraw_seq_, *L.vconv_sw, tv_seq_, slot, 1, 1024, vout_seq_, t,
                 last);
  } else {
    conv_seq_static(kraw_seq_, *L.kconv_w, slot, 0, 1024, kout_seq_, t, last);
    conv_seq_static(vraw_seq_, *L.vconv_w, slot, 1, 1024, vout_seq_, t, last);
  }
  sync();

  {
    sh::PcL2ClipSeq cq{128u, float(c.qk_l2_clip_threshold), sh::F_USE_SCALE,
                       3072u};
    op(p_l2_seq_, {BW(q_seq_), BW(*L.qclip)}, &cq, sizeof(cq), 24,
       uint32_t(tl));
    sh::PcL2ClipSeq ck{128u, float(c.qk_l2_clip_threshold), 0u, 1024u};
    op(p_l2_seq_, {BW(kout_seq_), BW(g_->dummy4_)}, &ck, sizeof(ck), 8,
       uint32_t(tl));
    sh::PcL2ClipSeq cr{16u, float(c.rel_l2_clip_threshold), sh::F_USE_SCALE,
                       384u};
    op(p_l2_seq_, {BW(r_seq_), BW(*L.rclip)}, &cr, sizeof(cr), 24,
       uint32_t(tl));
    sh::PcRmsnormSeq cv{128u, eps, 0u, 128u, 128u};
    op(p_rms_seq_, {BW(vout_seq_), BW(g_->dummy4_), BW(vout_seq_)}, &cv,
       sizeof(cv), uint32_t(tl * 8));
  }
  sync();

  if (L.full) {
    // tau values are staged by chunk position (base + row), computed on the
    // host in double exactly like the CPU/GpuGraph paths.
    sh::PcLogScaleSeq pq{3072u, uint32_t(tl) * 3072u, uint32_t(base)};
    op(p_log_seq_, {BW(q_seq_), REST(toks_tau_, pmax_)}, &pq, sizeof(pq),
       wg64(tl * 3072));
    sh::PcLogScaleSeq pr{384u, uint32_t(tl) * 384u, uint32_t(base)};
    op(p_log_seq_, {BW(r_seq_), REST(toks_tau_, pmax_)}, &pr, sizeof(pr),
       wg64(tl * 384));
    sync();
  }

  // KV append: contiguous absolute rows pos0..pos0+t in the graph's caches
  const VkDeviceSize koff = VkDeviceSize(slot) * VkDeviceSize(g_->cap_) * 4096 +
                            VkDeviceSize(pos0) * 4096;
  cp(kout_seq_, 0, g_->kvk_, koff, VkDeviceSize(tl) * 4096);
  cp(vout_seq_, 0, g_->kvv_, koff, VkDeviceSize(tl) * 4096);
  {
    sh::PcRelBiasSeq pc{16u, 1024u, 24u * 1024u};
    op(p_rel_seq_, {BW(r_seq_), BW(*L.relproj), BW(rel_seq_)}, &pc,
       sizeof(pc), wg64(24 * 1024), uint32_t(tl));
  }
  mv_seq(L.gate, REST(inbuf_seq_, xo[0]), grate_seq_, 3072, H, 3 * H, t);
  mv_seq(L.sink, REST(inbuf_seq_, xo[0]), sink_seq_, 24, H, 3 * H, t);
  sync();

  const int64_t N = pos0 + tl;  // score row width (absolute cols)
  {
    sh::PcAttnScoresSeq pc{24u,
                           8u,
                           128u,
                           uint32_t(N),
                           uint32_t(pos0),
                           1024u,
                           uint32_t(N + 1),
                           uint32_t(c.sliding_window_size),
                           L.full ? sh::F_SEQ_FULL : 0u};
    op(p_sco_seq_, {BW(q_seq_), g_->kvb(g_->kvk_, slot), BW(rel_seq_),
                    BW(probs_seq_)},
       &pc, sizeof(pc), 24, wg64(N), uint32_t(tl));
  }
  sync();
  {
    sh::PcAttnSoftmaxSeq pc{uint32_t(N), uint32_t(N + 1), 24u};
    op(p_smx_seq_, {BW(probs_seq_), BW(sink_seq_)}, &pc, sizeof(pc), 24,
       uint32_t(tl));
  }
  sync();
  {
    sh::PcAttnAccumSeq pc{24u, 8u, 128u, uint32_t(N), uint32_t(N + 1)};
    op(p_acc_seq_, {BW(probs_seq_), g_->kvb(g_->kvv_, slot), BW(o_seq_)}, &pc,
       sizeof(pc), 24, uint32_t(tl));
  }
  sync();
  {
    PcN pc{uint32_t(tl) * 3072u};
    op(g_->p_sig_.p, {BW(o_seq_), BW(grate_seq_)}, &pc, sizeof(pc),
       wg64(tl * 3072));
  }
  sync();
  if (dbg && !building_) {
    cp(tk_seq_, lr * 32 * 4, g_->readback_, (g_->rb_dbg_ + 6 * int64_t(H)) * 4,
       32 * 4);
    cp(tv_seq_, lr * 32 * 4, g_->readback_,
       (g_->rb_dbg_ + 6 * int64_t(H)) * 4 + 128, 32 * 4);
  }
  dbg_row(q_seq_, 3072, 0, 3072 * 4);
  dbg_row(kout_seq_, 1024, 2, 4096);
  dbg_row(vout_seq_, 1024, 3, 4096);
  dbg_row(o_seq_, 3072, 4, 3072 * 4);
  mv_seq(L.wo, BW(o_seq_), attn_out_seq_, H, 3072, 3072, t);
  sync();
  dbg_row(attn_out_seq_, H, 7, Hb);

  conv_seq_static(attn_out_seq_, *L.attn_sconv, slot, 2, H, a_out_seq_, t,
                  last);
  sync();

  // block boundary push / residual
  const vk::Buffer* resid;
  if (i % c.n_layers_per_attn_res_block == 0) {
    nblock_++;
    resid = &a_out_seq_;
  } else {
    PcN pc{uint32_t(tl * H)};
    op(g_->p_add_.p, {BW(partial_seq_), BW(a_out_seq_), BW(residual_seq_)},
       &pc, sizeof(pc), wg64(tl * H));
    sync();
    resid = &residual_seq_;
  }
  for (int64_t r = 0; r < tl; r++)
    cp(*resid, r * Hb, block_seq_, (r * 5 + nblock_) * Hb, Hb);
  sync();
  dbg_row(*resid, H, 8, Hb);

  // mlp attn_res read: block[0..nblock) ++ residual
  {
    sh::PcAttnResSeq pc{uint32_t(nblock_ + 1), uint32_t(H), 1u,
                        uint32_t(5 * H), uint32_t(H)};
    op(p_ares_seq_,
       {BW(block_seq_), BA(g_->mlpq_, int64_t(i) * H, H), BW(m_in_seq_)},
       &pc, sizeof(pc), uint32_t(tl));
  }
  sync();
  dbg_row(m_in_seq_, H, 9, Hb);

  if (p > 0) {
    sh::PcMsgGateSeq pcg{uint32_t(H), rank, uint32_t(H), uint32_t(4 * H),
                         uint32_t(H)};
    op(p_msg_seq_,
       {BW(*L.ctrl[3][0]), BW(*L.ctrl[3][1]), BW(*L.ctrl[3][2]),
        BW(m_in_seq_), REST(msgs_seq_, 3 * H), BW(mlpegated_seq_)},
       &pcg, sizeof(pcg), uint32_t(tl));
    sync();
    sh::PcRmsnormSeq pc{uint32_t(H), eps, sh::F_USE_GAMMA, uint32_t(H),
                        uint32_t(H)};
    op(p_rms_seq_, {BW(mlpegated_seq_), BW(*L.mlp_norm), BW(mn_seq_)}, &pc,
       sizeof(pc), uint32_t(tl));
  } else {
    sh::PcRmsnormSeq pc{uint32_t(H), eps, sh::F_USE_GAMMA, uint32_t(H),
                        uint32_t(H)};
    op(p_rms_seq_, {BW(m_in_seq_), BW(*L.mlp_norm), BW(mn_seq_)}, &pc,
       sizeof(pc), uint32_t(tl));
  }
  sync();
  dbg_row(mn_seq_, H, 10, Hb);

  {
    sh::PcRouterSeq pc{128u, 8u, uint32_t(H), float(c.route_scale)};
    op(p_rtr_seq_,
       {BW(*L.router_w), BW(*L.router_b), BW(mn_seq_), BW(ridx_seq_),
        BW(rw_seq_)},
       &pc, sizeof(pc), uint32_t(tl));
  }
  sync();
  if (dbg && !building_) {
    cp(ridx_seq_, lr * 32, g_->readback_, (g_->rb_dbg_ + 13 * int64_t(H)) * 4,
       32);
    cp(rw_seq_, lr * 32, g_->readback_,
       (g_->rb_dbg_ + 13 * int64_t(H)) * 4 + 32, 32);
  }

  // MoE: zero the accumulator, per-slot gate/up + serial accumulate.
  if (!building_) rec_->fill_buffer(moe_out_seq_.buf, 0, VkDeviceSize(tl) * Hb, 0u);
  const int64_t I = c.intermediate_size;
  for (uint32_t s = 0; s < 8; s++) {
    if (L.mxfp8) {
      sh::PcMxfp8Seq pg{s, uint32_t(L.w13R), uint32_t(H), 0u, 2u, 1.f, 0u, 8u,
                        uint32_t(I)};
      op(p_mx8_seq_,
         {BW(*L.w13), BW(*L.w13s), BW(mn_seq_), BW(gate_seq_),
          BW(ridx_seq_)},
         &pg, sizeof(pg), uint32_t(I), uint32_t(tl));
      sh::PcMxfp8Seq pu{s, uint32_t(L.w13R), uint32_t(H), 1u, 2u, 1.f, 0u, 8u,
                        uint32_t(I)};
      op(p_mx8_seq_,
         {BW(*L.w13), BW(*L.w13s), BW(mn_seq_), BW(up_seq_), BW(ridx_seq_)},
         &pu, sizeof(pu), uint32_t(I), uint32_t(tl));
    } else {
      sh::PcNvfp4Seq pg{s, uint32_t(L.w13R), uint32_t(H), 0u, 2u, 1.f, 0u, 8u,
                        uint32_t(I)};
      op(p_nv4_seq_,
         {BW(*L.w13), BW(*L.w13s), BW(*L.w13a), BW(*L.w13o), BW(mn_seq_),
          BW(gate_seq_), BW(ridx_seq_)},
         &pg, sizeof(pg), uint32_t(I), uint32_t(tl));
      sh::PcNvfp4Seq pu{s, uint32_t(L.w13R), uint32_t(H), 1u, 2u, 1.f, 0u, 8u,
                        uint32_t(I)};
      op(p_nv4_seq_,
         {BW(*L.w13), BW(*L.w13s), BW(*L.w13a), BW(*L.w13o), BW(mn_seq_),
          BW(up_seq_), BW(ridx_seq_)},
         &pu, sizeof(pu), uint32_t(I), uint32_t(tl));
    }
  }
  sync();
  for (uint32_t s = 0; s < 8; s++) {
    sh::PcMoeAccumSeq pc{s,
                         uint32_t(L.w2R),
                         uint32_t(I),
                         L.mxfp8 ? 1u : 0u,
                         float(c.mlp_gate_softcap),
                         float(c.mlp_up_softcap),
                         0.f,
                         8u,
                         uint32_t(8 * I)};
    op(p_moe_seq_,
       {REST(gate_seq_, int64_t(s) * I), REST(up_seq_, int64_t(s) * I),
        BW(*L.w2), BW(*L.w2s), BW(*L.w2a), BW(*L.w2o), BW(moe_out_seq_),
        BW(ridx_seq_), BW(rw_seq_)},
       &pc, sizeof(pc), uint32_t(H), uint32_t(tl));
    sync();  // serial: each slot does += on moe_out
  }
  dbg_row(moe_out_seq_, H, 11, Hb);

  conv_seq_static(moe_out_seq_, *L.mlp_sconv, slot, 3, H, mo_seq_, t, last);
  sync();
  dbg_row(mo_seq_, H, 12, Hb);
  {
    PcN pc{uint32_t(tl * H)};
    op(g_->p_add_.p, {BW(*resid), BW(mo_seq_), BW(partial_seq_)}, &pc,
       sizeof(pc), wg64(tl * H));
  }
  sync();

  if (last)
    cp(partial_seq_, lr * Hb, g_->readback_,
       (g_->rb_vis_ + int64_t(vidx * 2 + 1) * H) * 4, Hb);
}

// ------------------------------------------------------------------ chunk

void GpuPrefill::emit_chunk_seq(int64_t pos0, int64_t base, int t, bool last) {
  const ModelConfig& c = *g_->cfg_;
  const int H = g_->H_;
  const int64_t Hb = int64_t(H) * 4;
  nblock_ = 0;
  site_cursor_ = 0;  // every chunk replays the site stream from the top

  {
    sh::PcEmbedSeq pc{uint32_t(base), uint32_t(H), float(c.rms_norm_eps),
                      g_->emb_flags_};
    op(p_emb_seq_,
       {BW(g_->embedw_), BW(toks_tau_), BW(g_->embn_), BW(partial_seq_)},
       &pc, sizeof(pc), uint32_t(t));
  }
  sync();

  for (int i = 0; i < c.loop.core_start; i++)
    emit_visit_seq(i, 0, pos0, base, t, last);

  // prelude snapshot: pre_partial = partial; pre_block = block
  cp(partial_seq_, 0, pre_partial_seq_, 0, VkDeviceSize(t) * Hb);
  pre_nblock_ = nblock_;
  for (int r = 0; r < t; r++)
    for (int b = 0; b < nblock_; b++)
      cp(block_seq_, (int64_t(r) * 5 + b) * Hb, pre_block_seq_,
         (int64_t(r) * 4 + b) * Hb, Hb);
  sync();

  for (int p = 0; p < c.loop.num_passes; p++) {
    if (p > 0) {
      // bank0 = block.back() (last loop-boundary block), bank1 = prev partial
      for (int r = 0; r < t; r++) {
        cp(block_seq_, (int64_t(r) * 5 + nblock_ - 1) * Hb, bank_seq_,
           int64_t(r) * 2 * Hb, Hb);
        cp(partial_seq_, int64_t(r) * Hb, bank_seq_,
           (int64_t(r) * 2 + 1) * Hb, Hb);
        cp(pre_partial_seq_, int64_t(r) * Hb, partial_seq_, int64_t(r) * Hb,
           Hb);
        for (int b = 0; b < pre_nblock_; b++)
          cp(pre_block_seq_, (int64_t(r) * 4 + b) * Hb, block_seq_,
             (int64_t(r) * 5 + b) * Hb, Hb);
      }
      nblock_ = pre_nblock_;
      sync();
    }
    for (int i = c.loop.core_start; i < c.loop.core_end; i++)
      emit_visit_seq(i, p, pos0, base, t, last);
  }
  for (int i = c.loop.core_end; i < c.num_hidden_layers; i++)
    emit_visit_seq(i, 0, pos0, base, t, last);

  if (!last) return;

  // ---- tail on the last row only (gate-compatible readback + logits) ----
  const int64_t lr = int64_t(t) - 1;
  cp(partial_seq_, lr * Hb, block_seq_, (lr * 5 + nblock_) * Hb, Hb);
  sync();
  cp(block_seq_, lr * 5 * Hb, tail_blk_, 0, 5 * Hb);  // fixed binding for ares
  sync();
  {
    sh::PcAttnResSeq pc{uint32_t(nblock_ + 1), uint32_t(H), 1u,
                        uint32_t(5 * H), uint32_t(H)};
    op(p_ares_seq_, {BW(tail_blk_), BW(g_->outq_), BW(g_->h_)}, &pc,
       sizeof(pc), 1);
  }
  sync();
  {
    sh::PcRmsnorm pc{uint32_t(H), float(c.rms_norm_eps), sh::F_USE_GAMMA};
    op(g_->p_rms_.p, {BW(g_->h_), BW(g_->normw_), BW(g_->hfin_)}, &pc,
       sizeof(pc), 1);
  }
  sync();
  cp(g_->hfin_, 0, g_->readback_, (g_->V_ + 3 * int64_t(H)) * 4, Hb);
  {
    // single-row unembed on the graph's own pipe (chunked at 65535 like mv())
    const GpuGraph::Pipe& up = *g_->unemb_mv_.p;
    const vk::DescSet& ds =
        dset(up.p, {BW(*g_->unemb_mv_.b), BW(g_->hfin_), BW(g_->logits_)});
    if (!building_) {
      rec_->bind_pipeline(up.p.pipe);
      rec_->bind_descset(up.p.layout, ds.set);
      const int64_t M = g_->V_;
      for (int64_t m0 = 0; m0 < M; m0 += 65535) {
        sh::PcMatvec pc{uint32_t(H), uint32_t(m0)};
        rec_->push_constants(up.p.layout, &pc, sizeof(pc));
        rec_->dispatch(uint32_t(std::min<int64_t>(M - m0, 65535)));
        stats.dispatches++;
      }
    }
  }
  sync();
  {
    PcCap pc{uint32_t(g_->V_), float(c.final_logit_softcapping)};
    op(g_->p_cap_.p, {BW(g_->logits_)}, &pc, sizeof(pc), wg64(g_->V_));
  }
  sync();
  cp(g_->logits_, 0, g_->readback_, 0, VkDeviceSize(g_->V_) * 4);
}

// -------------------------------------------------------------------- api

bool GpuPrefill::prefill(const int* tokens, int64_t P) {
  GpuGraph& g = *g_;
  const ModelConfig& c = *g.cfg_;
  if (P <= 0) return true;
  if (P > g.cap_) {
    err_ = "prefill: prompt exceeds kv capacity";
    return false;
  }
  int32_t* tp = static_cast<int32_t*>(g.alloc_->mapped(toks_tau_));
  float* taus = reinterpret_cast<float*>(tp + pmax_);
  if (!tp) {
    err_ = "prefill: staging buffer not mapped";
    return false;
  }

  vk::Recorder rec;
  if (!rec.begin(g.cpool_)) {
    err_ = "prefill: recorder begin failed: " + rec.error();
    return false;
  }
  rec_ = &rec;
  building_ = false;
  const auto t0 = clk::steady_clock::now();

  // zero the row-major conv history (cold start). Long prompts are split
  // into pmax_-token super-blocks, one submission each — state lives on
  // device across submits and the host staging is refilled per block.
  rec.fill_buffer(h3_seq_.buf, 0, h3_seq_.size, 0u);
  int64_t pos0 = 0;
  while (pos0 < P) {
    const int64_t blk = std::min<int64_t>(pmax_, P - pos0);
    const int64_t end = pos0 + blk;
    // stage this block's token ids + per-position log-scale taus (same
    // host-side double math as GpuGraph::emit_visit).
    for (int64_t i = 0; i < blk; i++) tp[i] = int32_t(tokens[pos0 + i]);
    for (int64_t i = 0; i < blk; i++) {
      const double arg =
          std::max(double(pos0 + i + 1) / c.log_scaling_n_floor, 1.0);
      taus[i] = float(1.0 + c.log_scaling_alpha * std::log(arg));
    }
    for (int64_t p0 = pos0; p0 < end;) {
      const int t = int(std::min<int64_t>(T_, end - p0));
      emit_chunk_seq(p0, p0 - pos0, t, p0 + t == P);
      p0 += t;
    }
    pos0 = end;
    const auto w0 = clk::steady_clock::now();
    if (!rec.submit_and_wait(g.ctx_->compute_queue)) {
      err_ = "prefill: submit failed: " + rec.error();
      rec_ = nullptr;
      rec.shutdown();
      return false;
    }
    stats.wait_ms +=
        clk::duration<double, std::milli>(clk::steady_clock::now() - w0)
            .count();
    stats.submits++;
    if (pos0 < P) {
      rec.shutdown();
      if (!rec.begin(g.cpool_)) {
        err_ = "prefill: recorder rebegin failed: " + rec.error();
        rec_ = nullptr;
        return false;
      }
      rec_ = &rec;
      building_ = false;
    }
  }
  stats.record_ms +=
      clk::duration<double, std::milli>(clk::steady_clock::now() - t0).count();
  rec_ = nullptr;
  rec.shutdown();

  // hand off host-side counters so decode sees identical state
  for (int s = 0; s < 48; s++) {
    g.kv_len_[s] = P;
    for (int st = 0; st < 4; st++)
      g.conv_filled_[s][st] = int(std::min<int64_t>(3, P));
  }
  g.nblock_ = 0;
  g.alloc_->invalidate(g.readback_);
  std::memcpy(g.logits_host_.data(), g.rb_, size_t(g.V_) * 4);
  return err_.empty();
}

// -------------------------------------------------------------------- init

bool GpuPrefill::init(GpuGraph& g, int chunk, std::string* err) {
  g_ = &g;
  const ModelConfig& c = *g.cfg_;
  T_ = std::max(1, std::min(16, chunk));
  const int64_t T = T_;
  const int64_t H = g.H_;
  auto dev = [&](int64_t bytes) {
    vk::Buffer b = g.alloc_->alloc_device_local(VkDeviceSize(bytes));
    if (!b.buf && err_.empty())
      err_ = "gpu_prefill: device alloc failed: " + g.alloc_->last_error;
    return b;
  };

  toks_tau_ = g.alloc_->alloc_host(VkDeviceSize(2 * pmax_) * 4);
  partial_seq_ = dev(T * H * 4);
  block_seq_ = dev(T * 5 * H * 4);
  pre_partial_seq_ = dev(T * H * 4);
  pre_block_seq_ = dev(T * 4 * H * 4);
  bank_seq_ = dev(T * 2 * H * 4);
  reads_seq_ = dev(T * 3 * H * 4);
  msgs_seq_ = dev(T * 4 * H * 4);
  mgated_seq_ = dev(T * 3 * H * 4);
  mlpegated_seq_ = dev(T * H * 4);
  inbuf_seq_ = dev(T * 3 * H * 4);
  q_seq_ = dev(T * 3072 * 4);
  kraw_seq_ = dev(T * 1024 * 4);
  vraw_seq_ = dev(T * 1024 * 4);
  kout_seq_ = dev(T * 1024 * 4);
  vout_seq_ = dev(T * 1024 * 4);
  r_seq_ = dev(T * 384 * 4);
  tk_seq_ = dev(T * 32 * 4);
  tv_seq_ = dev(T * 32 * 4);
  rel_seq_ = dev(T * 24 * 1024 * 4);
  probs_seq_ = dev(T * 24 * (g.cap_ + T + 8) * 4);
  grate_seq_ = dev(T * 3072 * 4);
  sink_seq_ = dev(T * 24 * 4);
  o_seq_ = dev(T * 3072 * 4);
  attn_out_seq_ = dev(T * H * 4);
  a_out_seq_ = dev(T * H * 4);
  residual_seq_ = dev(T * H * 4);
  m_in_seq_ = dev(T * H * 4);
  mn_seq_ = dev(T * H * 4);
  moe_out_seq_ = dev(T * H * 4);
  mo_seq_ = dev(T * H * 4);
  ridx_seq_ = dev(T * 8 * 4);
  rw_seq_ = dev(T * 8 * 4);
  gate_seq_ = dev(T * 8 * c.intermediate_size * 4);
  up_seq_ = dev(T * 8 * c.intermediate_size * 4);
  h3_seq_ = dev(48 * 4 * 3 * H * 4);
  tail_blk_ = dev(5 * H * 4);
  if (!err_.empty() || !toks_tau_.buf) {
    if (err_.empty()) err_ = "gpu_prefill: staging alloc failed";
    *err = err_;
    return false;
  }

  const std::string& sd = g.shader_dir_;
  auto lp = [&](const char* name, uint32_t bindings, uint32_t pc_bytes,
                ink::vk::ComputePipeline& out) {
    vk::PipelineDesc d;
    d.spv_path = sd + "/" + name + ".spv";
    d.storage_bindings = bindings;
    d.push_constant_bytes = pc_bytes;
    return vk::load_compute_pipeline(*g.ctx_, d, out, err_);
  };
  const bool okp =
      lp("embed_norm_seq", 4, sizeof(sh::PcEmbedSeq), p_emb_seq_) &&
      lp("rmsnorm_seq", 3, sizeof(sh::PcRmsnormSeq), p_rms_seq_) &&
      lp("matvec_f32_seq", 3, sizeof(sh::PcMatvecSeq), p_mvf_seq_) &&
      lp("matvec_bf16_seq", 3, sizeof(sh::PcMatvecSeq), p_mvb_seq_) &&
      lp("matvec_f8e4m3_seq", 3, sizeof(sh::PcMatvecSeq), p_mv8_seq_) &&
      lp("conv4_static_seq", 5, sizeof(sh::PcConv4StaticSeq), p_cvs_seq_) &&
      lp("conv4_dynamic_seq", 6, sizeof(sh::PcConv4DynSeq), p_cvd_seq_) &&
      lp("rel_bias_seq", 3, sizeof(sh::PcRelBiasSeq), p_rel_seq_) &&
      lp("attn_scores_seq", 4, sizeof(sh::PcAttnScoresSeq), p_sco_seq_) &&
      lp("attn_softmax_seq", 2, sizeof(sh::PcAttnSoftmaxSeq), p_smx_seq_) &&
      lp("attn_accum_seq", 3, sizeof(sh::PcAttnAccumSeq), p_acc_seq_) &&
      lp("attn_res_seq", 3, sizeof(sh::PcAttnResSeq), p_ares_seq_) &&
      lp("msg_gate_seq", 6, sizeof(sh::PcMsgGateSeq), p_msg_seq_) &&
      lp("router_seq", 5, sizeof(sh::PcRouterSeq), p_rtr_seq_) &&
      lp("moe_accum_seq", 9, sizeof(sh::PcMoeAccumSeq), p_moe_seq_) &&
      lp("l2_clip_seq", 2, sizeof(sh::PcL2ClipSeq), p_l2_seq_) &&
      lp("log_scale_seq", 2, sizeof(sh::PcLogScaleSeq), p_log_seq_) &&
      lp("matvec_nvfp4_seq", 7, sizeof(sh::PcNvfp4Seq), p_nv4_seq_) &&
      lp("matvec_mxfp8_seq", 5, sizeof(sh::PcMxfp8Seq), p_mx8_seq_);
  if (!okp) {
    *err = "gpu_prefill: pipeline load failed: " + err_;
    return false;
  }

  // every matvec WB the prefill batches must map to a seq pipe
  for (int i = 0; i < 24; i++) {
    const GpuGraph::Lw& L = g.lw_[i];
    for (const GpuGraph::WB* w :
         {&L.wq, &L.wk, &L.wv, &L.wo, &L.gate, &L.sink, &L.wr, &L.kconv_wp,
          &L.vconv_wp}) {
      if (!w->p) continue;
      if (w->p != &g.p_mvf_ && w->p != &g.p_mvb_ && w->p != &g.p_mv8_) {
        *err = "gpu_prefill: layer weight uses an unbatched matvec pipe";
        return false;
      }
    }
  }

  dbg_en_ = std::getenv("INK_DBG") != nullptr;
  building_ = true;
  emit_chunk_seq(0, 0, int(T), true);
  building_ = false;
  nblock_ = pre_nblock_ = 0;
  if (!err_.empty()) {
    *err = "gpu_prefill: descriptor build failed: " + err_;
    return false;
  }
  ok_ = true;
  return true;
}

void GpuPrefill::shutdown() {
  if (!g_ || !g_->ctx_ || !g_->ctx_->device) return;
  vkDeviceWaitIdle(g_->ctx_->device);
  vk::Buffer* bufs[] = {&toks_tau_,    &partial_seq_, &block_seq_,
                        &pre_partial_seq_, &pre_block_seq_, &bank_seq_,
                        &reads_seq_,   &msgs_seq_,    &mgated_seq_,
                        &mlpegated_seq_, &inbuf_seq_, &q_seq_,
                        &kraw_seq_,    &vraw_seq_,    &kout_seq_,
                        &vout_seq_,    &r_seq_,       &tk_seq_,
                        &tv_seq_,      &rel_seq_,     &probs_seq_,
                        &grate_seq_,   &sink_seq_,    &o_seq_,
                        &attn_out_seq_, &a_out_seq_,  &residual_seq_,
                        &m_in_seq_,    &mn_seq_,      &moe_out_seq_,
                        &mo_seq_,      &ridx_seq_,    &rw_seq_,
                        &gate_seq_,    &up_seq_,      &h3_seq_,
                        &tail_blk_};
  for (auto* b : bufs)
    if (b->buf) g_->alloc_->free(*b);
  for (auto& ds : sites_) ds.shutdown(*g_->ctx_);
  sites_.clear();
  ink::vk::ComputePipeline* pipes[] = {
      &p_emb_seq_, &p_rms_seq_, &p_mvf_seq_, &p_mvb_seq_, &p_mv8_seq_,
      &p_cvs_seq_, &p_cvd_seq_, &p_rel_seq_, &p_sco_seq_, &p_smx_seq_,
      &p_acc_seq_, &p_ares_seq_, &p_msg_seq_, &p_rtr_seq_, &p_moe_seq_,
      &p_l2_seq_, &p_log_seq_, &p_nv4_seq_, &p_mx8_seq_};
  for (auto* p : pipes) p->shutdown();
  ok_ = false;
  g_ = nullptr;
}

}  // namespace ink::gpu
