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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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
// PC layouts for the staged-slot *_seqs kernels: same as the *_seq structs
// plus the wave geometry (sw = staged slots per row, sbase = wave's first
// real slot). Defined here so bindings.h needs no edit for this change.
struct PcNvfp4Seqs {
  uint32_t e, R, K, row0, row_stride;
  float wscale;
  uint32_t flags, ktop, M, sw, sbase;
};
struct PcMxfp8Seqs {
  uint32_t e, R, K, row0, row_stride;
  float wscale;
  uint32_t flags, ktop, M, sw, sbase;
};
struct PcMoeAccumSeqs {
  uint32_t e, R, I, quant;
  float gcap, ucap, wscale;
  uint32_t ktop, gstride, sw, sbase;
};
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
  const auto q0 = clk::steady_clock::now();
  rec_->bind_pipeline(p.pipe);
  rec_->bind_descset(p.layout, ds.set);
  if (pcb) rec_->push_constants(p.layout, pc, pcb);
  rec_->dispatch(gx, gy, gz);
  stats.dispatches++;
  stats.op_ns += uint64_t(
      std::chrono::duration<double, std::nano>(clk::steady_clock::now() - q0)
          .count());
  ts_mark(ts_name(p));
}

void GpuPrefill::cp(const ink::vk::Buffer& s, VkDeviceSize so,
                    const ink::vk::Buffer& d, VkDeviceSize dof,
                    VkDeviceSize sz) {
  if (building_) return;
  const auto q0 = clk::steady_clock::now();
  rec_->copy_buffer(s.buf, d.buf, sz, so, dof);
  stats.copies++;
  stats.cp_ns += uint64_t(
      std::chrono::duration<double, std::nano>(clk::steady_clock::now() - q0)
          .count());
  ts_mark("copy");
}

void GpuPrefill::sync() {
  if (building_) return;
  const auto q0 = clk::steady_clock::now();
  rec_->barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
  stats.barriers++;
  stats.sync_ns += uint64_t(
      std::chrono::duration<double, std::nano>(clk::steady_clock::now() - q0)
          .count());
}

// ---- INK_GPU_TS: timestamp boundary after each recorded op ----

const char* GpuPrefill::ts_name(const ink::vk::ComputePipeline& p) const {
  const auto it = pipe_names_.find(&p);
  return it != pipe_names_.end() ? it->second : "op";
}

void GpuPrefill::ts_mark(const char* name) {
  if (!ts_en_ || ts_off_) return;
  const uint32_t slot = ts_next_++;
  if (slot >= ts_cap_) return;  // pool exhausted: count but don't write
  rec_->write_timestamp(ts_pool_, slot);
  // this mark's segment index inside the whole-run seg list
  const uint32_t idx = ts_segbase_ + (slot - 1);
  if (ts_segs_.size() == idx) {
    std::string base = cur_scope_;
    base += '/';
    base += name;
    int& seen = ts_dup_[base];
    std::string l = seen ? base + "." + std::to_string(seen) : base;
    seen++;
    ts_segs_.push_back({std::move(l), 0.0});
  }
}

void GpuPrefill::ts_collect() {
  const uint32_t n = std::min(ts_next_, ts_cap_);
  if (n < 2) return;
  VkResult r = vkGetQueryPoolResults(g_->ctx_->device, ts_pool_, 0, n,
                                     size_t(n) * sizeof(uint64_t),
                                     ts_raw_.data(), sizeof(uint64_t),
                                     VK_QUERY_RESULT_64_BIT);
  if (r != VK_SUCCESS) {
    std::fprintf(stderr, "gpu-ts: vkGetQueryPoolResults: %s\n",
                 vk::result_name(r));
    return;
  }
  double busy = 0;
  const uint32_t m =
      std::min<uint32_t>(n - 1, uint32_t(ts_segs_.size()) - ts_segbase_);
  for (uint32_t i = 0; i < m; i++) {
    const double ns = double(ts_raw_[i + 1] - ts_raw_[i]) * ts_period_;
    ts_segs_[ts_segbase_ + i].ms += ns;
    busy += ns;
  }
  ts_gpu_ms_ += busy;
  ts_runs_++;
  std::fprintf(stderr, "gpu-ts: prefill submit %llu gpu-busy %.2fms (%u segs)\n",
               (unsigned long long)ts_runs_, busy / 1e6, m);
}

void GpuPrefill::ts_report() const {
  if (!ts_en_ || ts_runs_ == 0 || ts_segs_.empty()) return;
  std::fprintf(stderr,
               "gpu-ts: ===== prefill aggregate over %llu submit(s) =====\n",
               (unsigned long long)ts_runs_);
  std::fprintf(stderr,
               "gpu-ts: gpu-busy %.2f ms total (record %.2f ms, wait %.2f ms, "
               "dispatches %llu, copies %llu)\n",
               ts_gpu_ms_ / 1e6, stats.record_ms, stats.wait_ms,
               (unsigned long long)stats.dispatches,
               (unsigned long long)stats.copies);
  std::fprintf(stderr,
               "gpu-ts: host emit: op %.1fms cp %.1fms sync %.1fms "
               "(barriers %llu)\n",
               stats.op_ns / 1e6, stats.cp_ns / 1e6, stats.sync_ns / 1e6,
               (unsigned long long)stats.barriers);
  // aggregate by kernel class (label after '/', dedup ".N" suffix stripped)
  // and by visit scope
  std::unordered_map<std::string, double> by_kernel, by_scope;
  for (const TsSeg& s : ts_segs_) {
    const size_t slash = s.label.find('/');
    std::string kern = s.label.substr(slash + 1);
    const size_t dot = kern.rfind('.');
    if (dot != std::string::npos &&
        kern.find_first_not_of("0123456789", dot + 1) == std::string::npos)
      kern.resize(dot);
    by_kernel[kern] += s.ms;
    by_scope[s.label.substr(0, slash)] += s.ms;
  }
  std::vector<std::pair<std::string, double>> kern(by_kernel.begin(),
                                                   by_kernel.end());
  std::sort(kern.begin(), kern.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::fprintf(stderr, "gpu-ts: -- by kernel (ms total) --\n");
  for (const auto& [l, ms] : kern) {
    std::fprintf(stderr, "gpu-ts:   %-24s %8.2f ms  %5.1f%%\n", l.c_str(),
                 ms / 1e6, 100.0 * ms / ts_gpu_ms_);
  }
  std::vector<std::pair<std::string, double>> scop(by_scope.begin(),
                                                   by_scope.end());
  std::sort(scop.begin(), scop.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::fprintf(stderr, "gpu-ts: -- by scope (ms total) --\n");
  for (const auto& [l, ms] : scop) {
    std::fprintf(stderr, "gpu-ts:   %-10s %8.2f ms  %5.1f%%\n", l.c_str(),
                 ms / 1e6, 100.0 * ms / ts_gpu_ms_);
  }
  std::vector<uint32_t> ord(ts_segs_.size());
  for (uint32_t i = 0; i < ord.size(); i++) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) {
    return ts_segs_[a].ms > ts_segs_[b].ms;
  });
  std::fprintf(stderr, "gpu-ts: -- top-25 stages (ms total) --\n");
  const uint32_t top = std::min<uint32_t>(25, uint32_t(ord.size()));
  for (uint32_t i = 0; i < top; i++) {
    const TsSeg& s = ts_segs_[ord[i]];
    std::fprintf(stderr, "gpu-ts:   %2u %-34s %8.2f ms  %4.1f%%\n", i + 1,
                 s.label.c_str(), s.ms / 1e6, 100.0 * s.ms / ts_gpu_ms_);
  }
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

void GpuPrefill::emit_visit_seq_a(int i, int p, int64_t pos0, int64_t base,
                                  int t, bool last) {
  const ModelConfig& c = *g_->cfg_;
  const int H = g_->H_;
  const int64_t Hb = int64_t(H) * 4;
  const int vidx = c.visit_index(i, p);
  const int slot = c.cache_slot(i, p);
  const GpuGraph::Lw& L = g_->lw_[i];
  cur_scope_ = "v" + std::to_string(vidx);  // INK_GPU_TS label scope
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
  if (!L.experts_dev && stage_seq_ok_)
    cp(ridx_seq_, 0, ids_host_, 0, VkDeviceSize(tl) * 8 * 4);
  if (dbg && !building_) {
    cp(ridx_seq_, lr * 32, g_->readback_, (g_->rb_dbg_ + 13 * int64_t(H)) * 4,
       32);
    cp(rw_seq_, lr * 32, g_->readback_,
       (g_->rb_dbg_ + 13 * int64_t(H)) * 4 + 32, 32);
  }
}

// --------------------------------------------------- visit tail (MoE->out)

// emit_visit_seq_b: the part of a visit that needs the router result on the
// host before it can be recorded — for host-resident layers the per-(row,
// slot) expert slices were DMA-copied into stg_*_seq_ at the cb boundary, so
// the *_seqs kernels index weights by slot (t*8+s) instead of expert id and
// read VRAM instead of streaming host memory over PCIe. Dev-resident layers
// keep the *_seq kernels bound to their device tensors.
// emit_moe_wave: ONE staging wave [sbase, sbase+sw) of a staged visit —
// the fill at sbase==0, the wave's gate/up GEMVs and serial accums, and at
// sbase+sw==8 the visit tail (mlp conv, residual add, out dump). Recorded
// on the compute queue; the wave's slice copies ran on the transfer queue
// right before (fenced by the driver) — or, in the build pass, inline.
void GpuPrefill::emit_moe_wave(int i, int p, int t, bool last,
                              uint32_t sbase, uint32_t sw) {
  const ModelConfig& c = *g_->cfg_;
  const int H = g_->H_;
  const int64_t Hb = int64_t(H) * 4;
  const int vidx = c.visit_index(i, p);
  const int slot = c.cache_slot(i, p);
  const GpuGraph::Lw& L = g_->lw_[i];
  const int64_t tl = int64_t(t);
  const int64_t lr = tl - 1;
  const bool dbg = dbg_en_ && last && vidx == 5;
  auto dbg_row = [&](const vk::Buffer& s, int64_t row_elems, int row,
                     VkDeviceSize bytes) {
    if (dbg && !building_)
      cp(s, lr * row_elems * 4, g_->readback_,
         (g_->rb_dbg_ + int64_t(row) * H) * 4, bytes);
  };

  if (sbase == 0 && !building_) {
    rec_->fill_buffer(moe_out_seq_.buf, 0, VkDeviceSize(tl) * Hb, 0u);
    ts_mark("fill");
  }
  const int64_t I = c.intermediate_size;
  for (uint32_t s = sbase; s < sbase + sw; s++) {
    if (L.mxfp8) {
      PcMxfp8Seqs pg{s, uint32_t(L.w13R), uint32_t(H), 0u, 2u,
                     1.f, 0u, 8u, uint32_t(I), sw, sbase};
      op(p_mx8st_seq_, {BW(stg13w_seq_), BW(stg13s_seq_), BW(mn_seq_),
                        BW(gate_seq_), BW(ridx_seq_)},
         &pg, sizeof(pg), uint32_t(I), uint32_t(tl));
      PcMxfp8Seqs pu{s, uint32_t(L.w13R), uint32_t(H), 1u, 2u,
                     1.f, 0u, 8u, uint32_t(I), sw, sbase};
      op(p_mx8st_seq_, {BW(stg13w_seq_), BW(stg13s_seq_), BW(mn_seq_),
                        BW(up_seq_), BW(ridx_seq_)},
         &pu, sizeof(pu), uint32_t(I), uint32_t(tl));
    } else {
      PcNvfp4Seqs pg{s, uint32_t(L.w13R), uint32_t(H), 0u, 2u,
                     1.f, 0u, 8u, uint32_t(I), sw, sbase};
      op(p_nv4st_seq_,
         {BW(stg13w_seq_), BW(stg13s_seq_), BW(*L.w13a), BW(*L.w13o),
          BW(mn_seq_), BW(gate_seq_), BW(ridx_seq_)},
         &pg, sizeof(pg), uint32_t(I), uint32_t(tl));
      PcNvfp4Seqs pu{s, uint32_t(L.w13R), uint32_t(H), 1u, 2u,
                     1.f, 0u, 8u, uint32_t(I), sw, sbase};
      op(p_nv4st_seq_,
         {BW(stg13w_seq_), BW(stg13s_seq_), BW(*L.w13a), BW(*L.w13o),
          BW(mn_seq_), BW(up_seq_), BW(ridx_seq_)},
         &pu, sizeof(pu), uint32_t(I), uint32_t(tl));
    }
  }
  sync();
  for (uint32_t s = sbase; s < sbase + sw; s++) {
    PcMoeAccumSeqs pc{s,
                      uint32_t(L.w2R),
                      uint32_t(I),
                      L.mxfp8 ? 1u : 0u,
                      float(c.mlp_gate_softcap),
                      float(c.mlp_up_softcap),
                      0.f,
                      8u,
                      uint32_t(8 * I),
                      sw,
                      sbase};
    op(p_moest_seq_,
       {REST(gate_seq_, int64_t(s) * I), REST(up_seq_, int64_t(s) * I),
        BW(stg2w_seq_), BW(stg2s_seq_), BW(*L.w2a), BW(*L.w2o),
        BW(moe_out_seq_), BW(ridx_seq_), BW(rw_seq_)},
       &pc, sizeof(pc), uint32_t(H), uint32_t(tl));
    sync();  // serial: each slot does += on moe_out
  }
  if (sbase + sw < 8) return;

  // ---- visit tail: mlp conv, residual add, out dump (last wave only) ----
  dbg_row(moe_out_seq_, H, 11, Hb);

  conv_seq_static(moe_out_seq_, *L.mlp_sconv, slot, 3, H, mo_seq_, t, last);
  sync();
  dbg_row(mo_seq_, H, 12, Hb);
  {
    const vk::Buffer* resid = (i % c.n_layers_per_attn_res_block == 0)
                                  ? &a_out_seq_
                                  : &residual_seq_;
    PcN pc{uint32_t(tl * H)};
    op(g_->p_add_.p, {BW(*resid), BW(mo_seq_), BW(partial_seq_)}, &pc,
       sizeof(pc), wg64(tl * H));
  }
  sync();

  if (last)
    cp(partial_seq_, lr * Hb, g_->readback_,
       (g_->rb_vis_ + int64_t(vidx * 2 + 1) * H) * 4, Hb);
}

void GpuPrefill::emit_visit_seq_b(int i, int p, int t, bool last) {
  const ModelConfig& c = *g_->cfg_;
  const int H = g_->H_;
  const int64_t Hb = int64_t(H) * 4;
  const int vidx = c.visit_index(i, p);
  const int slot = c.cache_slot(i, p);
  const GpuGraph::Lw& L = g_->lw_[i];
  const int64_t tl = int64_t(t);
  const int64_t lr = tl - 1;  // last row index (gate dumps)
  cur_scope_ = "v" + std::to_string(vidx);

  const bool staged = !L.experts_dev && stage_seq_ok_;

  const bool dbg = dbg_en_ && last && vidx == 5;
  auto dbg_row = [&](const vk::Buffer& s, int64_t row_elems, int row,
                     VkDeviceSize bytes) {
    if (dbg && !building_)
      cp(s, lr * row_elems * 4, g_->readback_,
         (g_->rb_dbg_ + int64_t(row) * H) * 4, bytes);
  };

  // residual source for the out add — same pick as _a's block push.
  const vk::Buffer* resid =
      (i % c.n_layers_per_attn_res_block == 0) ? &a_out_seq_ : &residual_seq_;
  // MoE: zero the accumulator, per-slot gate/up + serial accumulate.
  if (!building_) {
    rec_->fill_buffer(moe_out_seq_.buf, 0, VkDeviceSize(tl) * Hb, 0u);
    ts_mark("fill");
  }
  const int64_t I = c.intermediate_size;
  if (staged) {
    // Build-pass emission of the staged path: the same wave order the
    // record walk produces (copies are no-ops here; sites only for the
    // GEMV/accum dispatches).
    for (uint32_t sbase = 0; sbase < 8; sbase += stage_sw_) {
      stage_seq_copies(L, t, sbase, stage_sw_);
      emit_moe_wave(i, p, t, last, sbase, stage_sw_);
    }
    return;
  }
  for (uint32_t s = 0; s < 8; s++) {
    if (L.mxfp8) {
      sh::PcMxfp8Seq pg{s, uint32_t(L.w13R), uint32_t(H), 0u, 2u, 1.f, 0u,
                        8u, uint32_t(I)};
      op(p_mx8_seq_,
         {BW(*L.w13), BW(*L.w13s), BW(mn_seq_), BW(gate_seq_),
          BW(ridx_seq_)},
         &pg, sizeof(pg), uint32_t(I), uint32_t(tl));
      sh::PcMxfp8Seq pu{s, uint32_t(L.w13R), uint32_t(H), 1u, 2u, 1.f, 0u,
                        8u, uint32_t(I)};
      op(p_mx8_seq_,
         {BW(*L.w13), BW(*L.w13s), BW(mn_seq_), BW(up_seq_),
          BW(ridx_seq_)},
         &pu, sizeof(pu), uint32_t(I), uint32_t(tl));
    } else {
      sh::PcNvfp4Seq pg{s, uint32_t(L.w13R), uint32_t(H), 0u, 2u, 1.f, 0u,
                        8u, uint32_t(I)};
      op(p_nv4_seq_,
         {BW(*L.w13), BW(*L.w13s), BW(*L.w13a), BW(*L.w13o), BW(mn_seq_),
          BW(gate_seq_), BW(ridx_seq_)},
         &pg, sizeof(pg), uint32_t(I), uint32_t(tl));
      sh::PcNvfp4Seq pu{s, uint32_t(L.w13R), uint32_t(H), 1u, 2u, 1.f, 0u,
                        8u, uint32_t(I)};
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

// ------------------------------------------------- expert-slice staging

// T*sw slice copies for one staging wave [sbase, sbase+sw): the expert id
// stage_ids64_[r*8+s] (written by the host after _a's router-id readback) is
// copied into staged slot r*sw + (s-sbase). Slice/offset math is in BYTES,
// and the dst pack is slot-contiguous — the *_seqs kernels index
// (t*sw + s-sbase) directly, no id map needed.
void GpuPrefill::stage_seq_copies(const GpuGraph::Lw& L, int t,
                                  uint32_t sbase, uint32_t sw) {
  const int64_t E = g_->cfg_->n_routed_experts;
  for (int r = 0; r < t; r++) {
    for (uint32_t s = sbase; s < sbase + sw; s++) {
      int64_t e = stage_ids64_[r * 8 + s];
      if (e < 0 || e >= E) e = 0;  // clamp: a bad id must never index OOB
      const VkDeviceSize eo = VkDeviceSize(e);
      const VkDeviceSize so = VkDeviceSize(r * sw + (s - sbase));
      if (L.sl13w > 0)
        cp(*L.w13, eo * VkDeviceSize(L.sl13w), stg13w_seq_,
           so * VkDeviceSize(L.sl13w), VkDeviceSize(L.sl13w));
      if (L.sl13s > 0)
        cp(*L.w13s, eo * VkDeviceSize(L.sl13s), stg13s_seq_,
           so * VkDeviceSize(L.sl13s), VkDeviceSize(L.sl13s));
      if (L.sl2w > 0)
        cp(*L.w2, eo * VkDeviceSize(L.sl2w), stg2w_seq_,
           so * VkDeviceSize(L.sl2w), VkDeviceSize(L.sl2w));
      if (L.sl2s > 0)
        cp(*L.w2s, eo * VkDeviceSize(L.sl2s), stg2s_seq_,
           so * VkDeviceSize(L.sl2s), VkDeviceSize(L.sl2s));
    }
  }
}

// ------------------------------------------------------------------ chunk

void GpuPrefill::emit_chunk_seq(
    int64_t pos0, int64_t base, int t, bool last,
    const std::function<bool(int, int, int, bool)>& on_stage) {
  const ModelConfig& c = *g_->cfg_;
  const int H = g_->H_;
  const int64_t Hb = int64_t(H) * 4;
  nblock_ = 0;
  site_cursor_ = 0;  // every chunk replays the site stream from the top
  cur_scope_ = "pre";

  // Staged visits split at the router boundary: _a ends with the ids
  // readback; on_stage submits, reads the ids, then interleaves per-wave
  // transfer-queue slice copies with compute-queue wave GEMVs. Dev-resident
  // visits emit a+b flat; the build pass emits a+b flat too (sites only).
  bool okv = true;
  const auto run_visit = [&](int i, int p) {
    if (!okv) return;
    emit_visit_seq_a(i, p, pos0, base, t, last);
    const GpuGraph::Lw& L = g_->lw_[i];
    if (!L.experts_dev && stage_seq_ok_ && !building_) {
      if (on_stage && !on_stage(i, p, t, last)) {
        okv = false;
        return;
      }
      return;  // on_stage emitted the wave GEMVs + tail
    }
    emit_visit_seq_b(i, p, t, last);
  };

  {
    sh::PcEmbedSeq pc{uint32_t(base), uint32_t(H), float(c.rms_norm_eps),
                      g_->emb_flags_};
    op(p_emb_seq_,
       {BW(g_->embedw_), BW(toks_tau_), BW(g_->embn_), BW(partial_seq_)},
       &pc, sizeof(pc), uint32_t(t));
  }
  sync();

  for (int i = 0; i < c.loop.core_start; i++)
    run_visit(i, 0);

  // prelude snapshot: pre_partial = partial; pre_block = block
  cur_scope_ = "snap";
  cp(partial_seq_, 0, pre_partial_seq_, 0, VkDeviceSize(t) * Hb);
  pre_nblock_ = nblock_;
  for (int r = 0; r < t; r++)
    for (int b = 0; b < nblock_; b++)
      cp(block_seq_, (int64_t(r) * 5 + b) * Hb, pre_block_seq_,
         (int64_t(r) * 4 + b) * Hb, Hb);
  sync();

  for (int p = 0; p < c.loop.num_passes; p++) {
    if (p > 0) {
      cur_scope_ = "bank" + std::to_string(p);
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
      run_visit(i, p);
  }
  for (int i = c.loop.core_end; i < c.num_hidden_layers; i++)
    run_visit(i, 0);

  if (!okv || !last) return;

  // ---- tail on the last row only (gate-compatible readback + logits) ----
  cur_scope_ = "fin";
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
      ts_mark("unembed");
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

  // One Recorder per recorded cb: submitted cbs stay owned by their recorder
  // in done_ (the pool must not free them while in flight) and are released
  // — recorder->shutdown() frees cb + fence — once a later chain fence or
  // the final drain proves they finished. The current open recorder is the
  // plain pointer; done_ holds the ones already submitted.
  std::unique_ptr<vk::Recorder> rec, trec;
  std::vector<std::unique_ptr<vk::Recorder>> done_;
  const auto reclaim = [&] {
    for (auto& r : done_) r->shutdown();
    done_.clear();
  };
  rec = std::make_unique<vk::Recorder>();
  if (!rec->begin(g.cpool_)) {
    err_ = "prefill: recorder begin failed: " + rec->error();
    return false;
  }
  rec_ = rec.get();
  building_ = false;
  const auto t0 = clk::steady_clock::now();
  if (ts_en_) {
    rec_->reset_query_pool(ts_pool_, 0, ts_cap_);
    rec_->write_timestamp(ts_pool_, 0);  // slot 0 = cb start
    ts_next_ = 1;
    ts_segbase_ = uint32_t(ts_segs_.size());
  }

  // zero the row-major conv history (cold start). Long prompts are split
  // into pmax_-token super-blocks, one submission each — state lives on
  // device across submits and the host staging is refilled per block.
  rec->fill_buffer(h3_seq_.buf, 0, h3_seq_.size, 0u);
  ts_mark("fill");
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
    // Staged-visit boundary (record mode only). Async path: every cb joins
    // one timeline-sem chain (wait stg_ctr_, signal ++stg_ctr_) so the GPU
    // serializes copy/GEMV alternation across the copy and compute queues
    // — the host only blocks on the router-id fence, once per staged visit.
    // The LAST wave's compute cb stays open: following ops (non-staged
    // visits, the next staged visit's _a+ids copy) record straight into it,
    // saving a submit and a cb boundary per visit. When timeline semaphores
    // are unavailable (or INK_GPU_TS needs per-submit query reads), the
    // sync path submits/waits each wave instead.
    const bool async_waves = !ts_en_ && sem_stg_ != VK_NULL_HANDLE;
    const auto submit_open = [&](bool fence) -> bool {
      const auto w0 = clk::steady_clock::now();
      if (!rec->end()) {
        err_ = "prefill: submit end failed: " + rec->error();
        return false;
      }
      if (async_waves) {
        const uint64_t wv[1] = {stg_ctr_};
        const uint64_t sv[1] = {++stg_ctr_};
        VkFence f = fence ? g.ctx_->create_fence() : VK_NULL_HANDLE;
        if (fence && !f) {  // the ids readback below depends on this wait
          err_ = "prefill: fence create failed: " + g.ctx_->last_error;
          return false;
        }
        if (!g.ctx_->submit_timeline(g.ctx_->compute_queue, rec->cb(),
                                     &sem_stg_, wv, 1, &sem_stg_, sv, 1, f)) {
          err_ = "prefill: submit failed: " + g.ctx_->last_error;
          if (f) g.ctx_->destroy_fence(f);
          return false;
        }
        done_.push_back(std::move(rec));
        if (f) {
          g.ctx_->wait_fence(f);
          g.ctx_->destroy_fence(f);
          // the fence covers this cb's chain slot — every cb submitted
          // before it finished too, so their recordings can be freed now.
          reclaim();
        }
      } else if (!(fence ? rec->submit_and_wait(g.ctx_->compute_queue)
                         : rec->submit(g.ctx_->compute_queue))) {
        err_ = "prefill: submit failed: " + rec->error();
        return false;
      } else if (fence) {
        rec->shutdown();  // submit_and_wait → this cb already completed
      } else {
        done_.push_back(std::move(rec));  // freed by a later fence/drain
      }
      if (fence)
        stats.wait_ms +=
            clk::duration<double, std::milli>(clk::steady_clock::now() - w0)
                .count();
      stats.submits++;
      return true;
    };
    const std::function<bool(int, int, int, bool)> on_stage =
        [&](int i, int p, int t, bool last) -> bool {
      const GpuGraph::Lw& L = g.lw_[i];
      if (!submit_open(true)) return false;  // ids land here
      if (ts_en_) ts_collect();
      g.alloc_->invalidate(ids_host_);
      std::memcpy(stage_ids64_, g.alloc_->mapped(ids_host_),
                  size_t(t) * 8 * 4);
      for (uint32_t sbase = 0; sbase < 8; sbase += stage_sw_) {
        // transfer cb: this wave's slices (overlaps the compute cb record)
        trec = std::make_unique<vk::Recorder>();
        if (!trec->begin(g.tpool_)) {
          err_ = "prefill: transfer recorder begin failed: " + trec->error();
          return false;
        }
        rec_ = trec.get();
        ts_off_ = true;  // no timestamps on the transfer queue
        stage_seq_copies(L, t, sbase, stage_sw_);
        ts_off_ = false;
        rec_ = nullptr;
        if (async_waves) {
          if (!trec->end()) {
            err_ = "prefill: stage copy end failed: " + trec->error();
            return false;
          }
          const uint64_t wv[1] = {stg_ctr_};
          const uint64_t sv[1] = {++stg_ctr_};
          if (!g.ctx_->submit_timeline(g.ctx_->copy_queue(), trec->cb(),
                                       &sem_stg_, wv, 1, &sem_stg_, sv, 1)) {
            err_ = "prefill: stage copy submit failed: " + g.ctx_->last_error;
            return false;
          }
          done_.push_back(std::move(trec));
          stats.submits++;
        } else {
          if (!trec->submit(g.ctx_->copy_queue())) {
            err_ = "prefill: stage copy submit failed: " + trec->error();
            trec->shutdown();
            return false;
          }
        }
        // compute cb: this wave's GEMVs (+ tail on the last wave)
        rec = std::make_unique<vk::Recorder>();
        if (!rec->begin(g.cpool_)) {
          err_ = "prefill: recorder rebegin failed: " + rec->error();
          rec_ = nullptr;
          return false;
        }
        rec_ = rec.get();
        if (ts_en_) {
          rec_->reset_query_pool(ts_pool_, 0, ts_cap_);
          rec_->write_timestamp(ts_pool_, 0);
          ts_next_ = 1;
          ts_segbase_ = uint32_t(ts_segs_.size());
        }
        emit_moe_wave(i, p, t, last, sbase, stage_sw_);
        if (async_waves) {
          if (sbase + stage_sw_ < 8 || !merge_tail_) {
            rec_ = nullptr;
            if (!submit_open(false)) return false;
            continue;  // submitted cb is parked in done_ until its fence
          }
          return true;  // last wave cb stays open as the stream's tail
        }
        rec_ = nullptr;
        if (!trec->wait()) {
          err_ = "prefill: stage copy wait failed";
          trec->shutdown();
          rec->shutdown();
          return false;
        }
        trec->shutdown();
        const auto w1 = clk::steady_clock::now();
        if (!rec->submit_and_wait(g.ctx_->compute_queue)) {
          err_ = "prefill: wave submit failed: " + rec->error();
          rec->shutdown();
          return false;
        }
        stats.wait_ms +=
            clk::duration<double, std::milli>(clk::steady_clock::now() - w1)
                .count();
        stats.submits++;
        rec->shutdown();
        if (ts_en_) ts_collect();
      }
      rec = std::make_unique<vk::Recorder>();
      if (!rec->begin(g.cpool_)) {
        err_ = "prefill: recorder rebegin failed: " + rec->error();
        rec_ = nullptr;
        return false;
      }
      rec_ = rec.get();
      if (ts_en_) {
        rec_->reset_query_pool(ts_pool_, 0, ts_cap_);
        rec_->write_timestamp(ts_pool_, 0);
        ts_next_ = 1;
        ts_segbase_ = uint32_t(ts_segs_.size());
      }
      return true;
    };
    for (int64_t p0 = pos0; p0 < end;) {
      const int t = int(std::min<int64_t>(T_, end - p0));
      emit_chunk_seq(p0, p0 - pos0, t, p0 + t == P, on_stage);
      if (!err_.empty()) {
        rec_ = nullptr;
        // pending async cbs may be in flight — drain before freeing
        vkQueueWaitIdle(g.ctx_->copy_queue());
        vkQueueWaitIdle(g.ctx_->compute_queue);
        reclaim();
        if (trec) trec->shutdown();
        if (rec) rec->shutdown();
        return false;
      }
      p0 += t;
    }
    pos0 = end;
    rec_ = nullptr;
    if (!submit_open(true)) {
      rec_ = nullptr;
      if (rec) rec->shutdown();
      return false;
    }
    if (ts_en_) ts_collect();
    if (pos0 < P) {
      rec = std::make_unique<vk::Recorder>();
      if (!rec->begin(g.cpool_)) {
        err_ = "prefill: recorder rebegin failed: " + rec->error();
        rec_ = nullptr;
        return false;
      }
      rec_ = rec.get();
      building_ = false;
      if (ts_en_) {
        rec_->reset_query_pool(ts_pool_, 0, ts_cap_);
        rec_->write_timestamp(ts_pool_, 0);
        ts_next_ = 1;
        ts_segbase_ = uint32_t(ts_segs_.size());
      }
    }
  }
  stats.record_ms +=
      clk::duration<double, std::milli>(clk::steady_clock::now() - t0).count();
  rec_ = nullptr;
  // drain the async wave chain: the last wave cb may still be in flight
  if (sem_stg_ != VK_NULL_HANDLE && stg_ctr_ > 0)
    g.ctx_->timeline_wait(sem_stg_, stg_ctr_);
  reclaim();  // every submitted cb is provably done — free the recordings
  if (trec) trec->shutdown();
  if (rec) rec->shutdown();

  // hand off host-side counters so decode sees identical state
  for (int s = 0; s < 48; s++) {
    g.kv_len_[s] = P;
    for (int st = 0; st < 4; st++)
      g.conv_filled_[s][st] = int(std::min<int64_t>(3, P));
  }
  g.nblock_ = 0;
  g.alloc_->invalidate(g.readback_);
  std::memcpy(g.logits_host_.data(), g.rb_, size_t(g.V_) * 4);
  if (ts_en_) ts_report();  // printed here: shutdown runs after vkctx teardown
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
      lp("matvec_mxfp8_seq", 5, sizeof(sh::PcMxfp8Seq), p_mx8_seq_) &&
      lp("matvec_nvfp4_seqs", 7, sizeof(PcNvfp4Seqs), p_nv4st_seq_) &&
      lp("matvec_mxfp8_seqs", 5, sizeof(PcMxfp8Seqs), p_mx8st_seq_) &&
      lp("moe_accum_seqs", 9, sizeof(PcMoeAccumSeqs), p_moest_seq_);
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

  // kernel names for INK_GPU_TS labels (pipe address -> spv name)
  const std::pair<const ink::vk::ComputePipeline*, const char*> kNames[] = {
      {&p_emb_seq_, "embed_norm_seq"},   {&p_rms_seq_, "rmsnorm_seq"},
      {&p_mvf_seq_, "matvec_f32_seq"},   {&p_mvb_seq_, "matvec_bf16_seq"},
      {&p_mv8_seq_, "matvec_f8e4m3_seq"},{&p_cvs_seq_, "conv4_static_seq"},
      {&p_cvd_seq_, "conv4_dynamic_seq"},{&p_rel_seq_, "rel_bias_seq"},
      {&p_sco_seq_, "attn_scores_seq"},  {&p_smx_seq_, "attn_softmax_seq"},
      {&p_acc_seq_, "attn_accum_seq"},   {&p_ares_seq_, "attn_res_seq"},
      {&p_msg_seq_, "msg_gate_seq"},     {&p_rtr_seq_, "router_seq"},
      {&p_moe_seq_, "moe_accum_seq"},    {&p_l2_seq_, "l2_clip_seq"},
      {&p_log_seq_, "log_scale_seq"},    {&p_nv4_seq_, "matvec_nvfp4_seq"},
      {&p_mx8_seq_, "matvec_mxfp8_seq"}, {&g.p_sig_.p, "sigmoid_mul"},
      {&p_nv4st_seq_, "matvec_nvfp4_seqs"},
      {&p_mx8st_seq_, "matvec_mxfp8_seqs"},
      {&p_moest_seq_, "moe_accum_seqs"},
      {&g.p_add_.p, "vadd"},             {&g.p_rms_.p, "rmsnorm"},
      {&g.p_cap_.p, "softcap"},          {&g.unemb_mv_.p->p, "unembed"},
  };
  for (const auto& [p, n] : kNames) pipe_names_[p] = n;

  // INK_GPU_TS=1: timestamp pool covering every op boundary in the chunk cb.
  // Chunked prefill records ~1.3k ops per chunk; pool holds one prompt's
  // marks (P <= pmax_ per submit).
  if (const char* e = std::getenv("INK_GPU_TS");
      e && e[0] && std::strcmp(e, "0") != 0) {
    uint32_t valid_bits = 0;
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g.ctx_->phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(g.ctx_->phys, &nq, qf.data());
    if (g.ctx_->compute_family < nq)
      valid_bits = qf[g.ctx_->compute_family].timestampValidBits;
    if (valid_bits == 0 || g.ctx_->props.limits.timestampPeriod <= 0.f) {
      std::fprintf(stderr,
                   "gpu-ts: timestamps unsupported on the compute queue; "
                   "disabled\n");
    } else {
      VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
      qi.queryCount = 16384;
      if (vkCreateQueryPool(g.ctx_->device, &qi, nullptr, &ts_pool_) ==
          VK_SUCCESS) {
        ts_en_ = true;
        ts_cap_ = 16384;
        ts_period_ = g.ctx_->props.limits.timestampPeriod;
        ts_raw_.assign(ts_cap_, 0);
        std::fprintf(stderr,
                     "gpu-ts: prefill on (timestampPeriod %.3f ns, %u slots, "
                     "%u valid bits)\n",
                     ts_period_, ts_cap_, valid_bits);
      } else {
        std::fprintf(stderr, "gpu-ts: vkCreateQueryPool failed; disabled\n");
      }
    }
  }

  // router-id readback target + dev-local expert-slice staging (only when
  // some layer is host-resident): T*8 slots x the largest per-expert slice
  // of each tensor, packed slot-contiguous for the *_seqs kernels.
  ids_host_ = g.alloc_->alloc_host(VkDeviceSize(T) * 8 * 4);
  {
    VkDeviceSize hb = 0, hu = 0;
    if (g.alloc_->heap_budget(1, hb, hu))
      std::fprintf(stderr,
                   "gpu prefill: device heap budget %.0fMB, usage %.0fMB, "
                   "headroom %.0fMB\n",
                   hb / double(1 << 20), hu / double(1 << 20),
                   (hb - hu) / double(1 << 20));

    int64_t m13w = 0, m13s = 0, m2w = 0, m2s = 0;
    for (int i = 0; i < 24; i++) {
      const GpuGraph::Lw& L = g.lw_[i];
      if (L.experts_dev) continue;
      m13w = std::max(m13w, L.sl13w);
      m13s = std::max(m13s, L.sl13s);
      m2w = std::max(m2w, L.sl2w);
      m2s = std::max(m2s, L.sl2s);
    }
    if (m13w > 0 && ids_host_.buf) {
      // wave-width ladder: the biggest slot-column count whose 4 staging
      // buffers fit the current heap headroom wins (8 = all slots at once).
      for (const uint32_t sw : {8u, 4u, 2u, 1u}) {
        const std::string e0 = err_;  // staging is optional — restore
        stg13w_seq_ = dev(T * sw * m13w);
        stg13s_seq_ = dev(T * sw * m13s);
        stg2w_seq_ = dev(T * sw * m2w);
        stg2s_seq_ = dev(T * sw * m2s);
        stage_seq_ok_ = stg13w_seq_.buf && stg13s_seq_.buf &&
                        stg2w_seq_.buf && stg2s_seq_.buf;
        err_ = e0;
        if (stage_seq_ok_) {
          stage_sw_ = sw;
          if (g.ctx_->feats.timeline_semaphore)
            sem_stg_ = g.ctx_->create_timeline_semaphore(0);
          merge_tail_ = std::getenv("INK_MERGE_TAIL") != nullptr;
          std::fprintf(stderr,
                       "gpu prefill: expert staging on (%.0fMB, sw=%u over "
                       "%lld slots%s)\n",
                       double(T * sw) * (m13w + m13s + m2w + m2s) / (1 << 20),
                       sw, (long long)(T * sw),
                       sem_stg_ ? ", async waves" : "");
          break;
        }
        for (vk::Buffer* b :
             {&stg13w_seq_, &stg13s_seq_, &stg2w_seq_, &stg2s_seq_})
          if (b->buf) g.alloc_->free(*b);
      }
      if (!stage_seq_ok_)
        std::fprintf(stderr,
                     "gpu prefill: expert staging alloc failed at every wave "
                     "width (%.0fMB min); host-read path kept: %s\n",
                     double(T) * (m13w + m13s + m2w + m2s) / (1 << 20),
                     g.alloc_->last_error.c_str());
    }
  }

  dbg_en_ = std::getenv("INK_DBG") != nullptr;
  if (dbg_en_) {
    std::string host;
    for (int i = 0; i < 24; i++)
      if (!g.lw_[i].experts_dev) host += " " + std::to_string(i);
    std::fprintf(stderr,
                 "prefill: host-resident expert layers:%s (staging %s)\n",
                 host.empty() ? " none" : host.c_str(),
                 stage_seq_ok_ ? "on" : "off");
  }
  building_ = true;
  emit_chunk_seq(0, 0, int(T), true, {});
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
  if (sem_stg_ != VK_NULL_HANDLE) {
    g_->ctx_->destroy_semaphore(sem_stg_);
    sem_stg_ = VK_NULL_HANDLE;
  }
  if (ts_en_) {
    vkDestroyQueryPool(g_->ctx_->device, ts_pool_, nullptr);
    ts_pool_ = VK_NULL_HANDLE;
    ts_en_ = false;
  }
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
                        &tail_blk_,    &stg13w_seq_,  &stg13s_seq_,
                        &stg2w_seq_,   &stg2s_seq_,   &ids_host_};
  for (auto* b : bufs)
    if (b->buf) g_->alloc_->free(*b);
  for (auto& ds : sites_) ds.shutdown(*g_->ctx_);
  sites_.clear();
  ink::vk::ComputePipeline* pipes[] = {
      &p_emb_seq_, &p_rms_seq_, &p_mvf_seq_, &p_mvb_seq_, &p_mv8_seq_,
      &p_cvs_seq_, &p_cvd_seq_, &p_rel_seq_, &p_sco_seq_, &p_smx_seq_,
      &p_acc_seq_, &p_ares_seq_, &p_msg_seq_, &p_rtr_seq_, &p_moe_seq_,
      &p_l2_seq_, &p_log_seq_, &p_nv4_seq_, &p_mx8_seq_,
      &p_nv4st_seq_, &p_mx8st_seq_, &p_moest_seq_};
  for (auto* p : pipes) p->shutdown();
  ok_ = false;
  g_ = nullptr;
}

}  // namespace ink::gpu
