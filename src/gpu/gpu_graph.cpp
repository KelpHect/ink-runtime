// gpu_graph.cpp — device-resident mirror of Graph::forward_token.
//
// The whole 48-visit token forward is emitted into ONE command buffer and
// submitted once per token. That requires every decision the shader needs
// to be visible to the shader as DATA (storage buffers), because push
// constants and descriptor bindings are frozen at record time:
//   * expert routing: router output idx[8]/w[8] stays in device buffers;
//     the *_slot shader variants index them with pc.e = slot.
//   * everything else (token id, pos, len, filled, j0, tau) is host-known
//     at record time and rides in push constants.
// All descriptor bindings are token-invariant, so the ~2k descriptor sets
// are built ONCE at init by running the emit code in "build" mode and then
// replayed by index when recording each token.
//
// Buffer model (all device-local unless noted):
//   block_    [5,2304]  attn_res block stack (4 stack rows + 1 append row);
//                        every attn_res-style read appends its source by a
//                        recorded copy into row nblock so sources stay
//                        contiguous — matches ops.cpp srcs = block ++ x.
//   bank_     [2,2304]  message bank {block.back(), prev partial} (p>0)
//   kvk_/kvv_ [48][cap][1024] per-(layer,pass) K/V, absolute row indexing
//   hist_     [48*4][3*2304] shortconv shift registers (k,v,attn,mlp)
//   readback_ HOST: [logits V][taps 4H][visit dumps 96H] — D2H copies are
//             recorded inline at the exact points Graph's visit_hook fires.

#include "gpu/gpu_graph.h"

#include "bindings.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ink::gpu {
namespace sh = ink::shaders;
namespace clk = std::chrono;

namespace {
bool contiguous(const TensorView& t) {
  if (t.nd == 0) return true;
  int64_t e = 1;
  for (int i = t.nd - 1; i >= 0; i--) {
    if (t.shape[i] == 1) continue;
    if (t.stride[i] != e) return false;
    e *= t.shape[i];
  }
  return true;
}
// PC layouts for the small kernels not named as structs in bindings.h.
struct PcN {
  uint32_t n;
};
struct PcCap {
  uint32_t n;
  float cap;
};
struct PcClip {
  uint32_t dim;
  float thr;
  uint32_t flags;
};
struct PcTau {
  uint32_t n;
  float tau;
};
struct PcEmb {
  uint32_t token, n;
  float eps;
  uint32_t flags;
};
constexpr uint32_t GRP = 64;
inline uint32_t wg64(int64_t n) { return uint32_t((n + GRP - 1) / GRP); }
static std::string lname(int i, const char* s) {
  return "model.llm.layers." + std::to_string(i) + "." + s;
}
}  // namespace

// ---------------------------------------------------------------- helpers

GpuGraph::Bnd GpuGraph::histb(int slot, int stream, int C) const {
  return {&hist_, VkDeviceSize((int64_t(slot) * 4 + stream) * 3 * H_) * 4,
          VkDeviceSize(C) * 3 * 4};
}
GpuGraph::Bnd GpuGraph::kvb(const ink::vk::Buffer& cache, int slot) const {
  return {&cache, VkDeviceSize(slot) * VkDeviceSize(cap_) * 4096,
          VkDeviceSize(cap_) * 4096};
}
static GpuGraph::Bnd BW(const ink::vk::Buffer& b) { return {&b, 0, 0}; }
static GpuGraph::Bnd BA(const ink::vk::Buffer& b, int64_t fe, int64_t n) {
  return {&b, VkDeviceSize(fe) * 4, VkDeviceSize(n) * 4};
}

bool GpuGraph::load_pipe(const char* name, uint32_t bindings,
                         uint32_t pc_bytes, Pipe& out) {
  vk::PipelineDesc d;
  d.spv_path = shader_dir_ + "/" + name + ".spv";
  d.storage_bindings = bindings;
  d.push_constant_bytes = pc_bytes;
  out.name = name;
  return vk::load_compute_pipeline(*ctx_, d, out.p, err_);
}

const GpuGraph::Pipe* GpuGraph::mvpipe(DType dt) const {
  switch (dt) {
    case DType::F32: return &p_mvf_;
    case DType::BF16: return &p_mvb_;
    case DType::F8E4M3: return &p_mv8_;
    default: return nullptr;
  }
}

// ------------------------------------------------------- upload machinery

ink::vk::Buffer GpuGraph::upload_dev(const void* src, VkDeviceSize bytes) {
  vk::Buffer dst = alloc_->alloc_device_local(bytes);
  if (!dst.buf) {
    if (err_.empty()) err_ = "alloc_device_local failed: " + alloc_->last_error;
    return {};
  }
  const VkDeviceSize CH = 64ull << 20;  // staging slab; one submit per chunk
  if (staging_.size < CH) {
    alloc_->free(staging_);
    staging_ = alloc_->alloc_host(CH);
    if (!staging_.buf) {
      err_ = "staging alloc failed";
      return {};
    }
  }
  char* sp = (char*)alloc_->mapped(staging_);
  const char* s = (const char*)src;
  for (VkDeviceSize off = 0; off < bytes; off += CH) {
    const VkDeviceSize n = std::min(CH, bytes - off);
    std::memcpy(sp, s + off, size_t(n));
    vk::Recorder r;
    if (!r.begin(tpool_)) {
      err_ = r.error();
      return {};
    }
    r.copy_buffer(staging_.buf, dst.buf, n, 0, off);
    if (!r.submit_and_wait(ctx_->copy_queue())) {
      err_ = "upload submit failed: " + r.error();
      r.shutdown();
      return {};
    }
    r.shutdown();
  }
  stats.uploads++;
  stats.upload_bytes += bytes;
  return dst;
}

const ink::vk::Buffer* GpuGraph::dev_w(const TensorView& t) {
  if (!t.data) return nullptr;
  if (auto it = devw_.find(t.data); it != devw_.end()) return &it->second;
  if (!contiguous(t)) {
    err_ = "non-contiguous tensor upload";
    return nullptr;
  }
  vk::Buffer b = upload_dev(t.data, t.nbytes());
  if (!b.buf) return nullptr;
  auto [it, ok] = devw_.emplace(t.data, std::move(b));
  (void)ok;
  return &it->second;
}

const ink::vk::Buffer* GpuGraph::dev_f32(const TensorView& t) {
  if (!t.data) return nullptr;
  if (auto it = f32w_.find(t.data); it != f32w_.end()) return &it->second;
  if (!contiguous(t)) {
    err_ = "non-contiguous tensor (f32 decode)";
    return nullptr;
  }
  const int64_t n = t.numel();
  std::vector<float> tmp((size_t)n);
  for (int64_t i = 0; i < n; i++) tmp[size_t(i)] = t.f32_linear(i);
  vk::Buffer b = upload_dev(tmp.data(), VkDeviceSize(n) * 4);
  if (!b.buf) return nullptr;
  auto [it, ok] = f32w_.emplace(t.data, std::move(b));
  (void)ok;
  return &it->second;
}

const ink::vk::Buffer* GpuGraph::host_w(const TensorView& t) {
  if (!t.data) return nullptr;
  if (auto it = hostw_.find(t.data); it != hostw_.end()) return &it->second;
  if (!contiguous(t)) {
    err_ = "non-contiguous tensor (host mirror)";
    return nullptr;
  }
  vk::Buffer b = alloc_->alloc_host(t.nbytes());
  if (!b.buf) {
    err_ = "alloc_host failed: " + alloc_->last_error;
    return nullptr;
  }
  std::memcpy(alloc_->mapped(b), t.data, size_t(t.nbytes()));
  auto [it, ok] = hostw_.emplace(t.data, std::move(b));
  (void)ok;
  return &it->second;
}

// ----------------------------------------------------------- emit helpers

const ink::vk::DescSet& GpuGraph::dset(const Pipe& p,
                                       std::initializer_list<Bnd> bs) {
  if (!building_) return sites_[site_cursor_++];
  sites_.emplace_back();
  ink::vk::DescSet& ds = sites_.back();
  if (!vk::alloc_descset(*ctx_, p.p.dsl, uint32_t(bs.size()), ds, err_))
    return ds;
  uint32_t i = 0;
  for (const Bnd& b : bs) {
    if (!vk::write_storage(*ctx_, ds, i++, b.b->buf, b.off,
                           b.sz ? b.sz : b.b->size, err_))
      break;
  }
  return ds;
}

void GpuGraph::op(const Pipe& p, std::initializer_list<Bnd> bs, const void* pc,
                  uint32_t pcb, uint32_t gx, uint32_t gy, const char* tag) {
  const vk::DescSet& ds = dset(p, bs);
  if (building_) return;
  rec_->bind_pipeline(p.p.pipe);
  rec_->bind_descset(p.p.layout, ds.set);
  if (pcb) rec_->push_constants(p.p.layout, pc, pcb);
  rec_->dispatch(gx, gy);
  stats.dispatches++;
  ts_mark(p.name, tag);
}

void GpuGraph::mv(const Pipe& p, Bnd w, Bnd x, Bnd o, int64_t M, int64_t K,
                  const char* tag) {
  const vk::DescSet& ds = dset(p, {w, x, o});
  if (building_) return;
  rec_->bind_pipeline(p.p.pipe);
  rec_->bind_descset(p.p.layout, ds.set);
  for (int64_t m0 = 0; m0 < M; m0 += 65535) {
    sh::PcMatvec pc{uint32_t(K), uint32_t(m0)};
    rec_->push_constants(p.p.layout, &pc, sizeof(pc));
    rec_->dispatch(uint32_t(std::min<int64_t>(M - m0, 65535)));
    stats.dispatches++;
  }
  ts_mark(p.name, tag);
}

void GpuGraph::cp(const ink::vk::Buffer& s, VkDeviceSize so,
                  const ink::vk::Buffer& d, VkDeviceSize dof, VkDeviceSize sz,
                  const char* tag) {
  if (building_) return;
  rec_->copy_buffer(s.buf, d.buf, sz, so, dof);
  stats.copies++;
  ts_mark("copy", tag);
}

void GpuGraph::fill(const ink::vk::Buffer& d, VkDeviceSize off, VkDeviceSize sz,
                    uint32_t data, const char* tag) {
  if (building_) return;
  rec_->fill_buffer(d.buf, off, sz, data);
  ts_mark("fill", tag);
}

// ---- INK_GPU_TS: timestamp boundary after each recorded op ----

// op-class for the grouped totals. The nvfp4/mxfp8 slot matvecs are expert
// (MoE) weight GEMMs, so they group under moe_*, not dense matvec_*.
static uint8_t ts_class(const char* name) {
  if (!std::strcmp(name, "copy") || !std::strcmp(name, "fill")) return 5;
  if (std::strstr(name, "nvfp4") || std::strstr(name, "mxfp8") ||
      !std::strncmp(name, "moe", 3) || !std::strcmp(name, "router"))
    return 3;
  if (!std::strncmp(name, "matvec", 6)) return 0;
  if (!std::strncmp(name, "conv4", 5)) return 1;
  if (!std::strncmp(name, "attn", 4)) return 2;
  return 4;
}

void GpuGraph::ts_mark(const char* name, const char* tag) {
  if (!ts_en_) return;
  const uint32_t slot = ts_next_++;
  if (slot >= ts_cap_) return;  // pool exhausted: count but don't write
  rec_->write_timestamp(ts_pool_, slot);
  if (ts_segs_.size() < slot) {
    // First instrumented emit: label this boundary once. Duplicate
    // role names inside one scope get a ".N" suffix so every instance
    // stays its own row in the report.
    std::string base = cur_scope_;
    base += '/';
    base += tag ? tag : name;
    int& seen = ts_dup_[base];
    std::string l = seen ? base + "." + std::to_string(seen) : base;
    seen++;
    ts_segs_.push_back({std::move(l), ts_class(name), cur_phase_, 0.0});
  }
}

void GpuGraph::ts_collect() {
  const uint32_t n = std::min(ts_next_, ts_cap_);
  if (n < 2) return;
  VkResult r = vkGetQueryPoolResults(ctx_->device, ts_pool_, 0, n,
                                     size_t(n) * sizeof(uint64_t),
                                     ts_raw_.data(), sizeof(uint64_t),
                                     VK_QUERY_RESULT_64_BIT);
  if (r != VK_SUCCESS) {
    std::fprintf(stderr, "gpu-ts: vkGetQueryPoolResults: %s\n",
                 vk::result_name(r));
    return;
  }
  double busy = 0;
  const uint32_t m = std::min<uint32_t>(n - 1, uint32_t(ts_segs_.size()));
  for (uint32_t i = 0; i < m; i++) {
    const double ns = double(ts_raw_[i + 1] - ts_raw_[i]) * ts_period_;
    ts_segs_[i].ms += ns;
    busy += ns;
  }
  ts_gpu_ms_ += busy;
  ts_tokens_++;
  std::fprintf(stderr, "gpu-ts: token %llu gpu-busy %.2fms (%u segs)\n",
               (unsigned long long)ts_tokens_, busy / 1e6, m);
}

void GpuGraph::ts_report() const {
  if (!ts_en_ || ts_tokens_ == 0 || ts_segs_.empty()) return;
  const double tok = double(ts_tokens_);
  static const char* kCls[] = {"matvec_*", "conv4_*", "attn_*",
                               "moe_*",    "elementwise", "copies"};
  static const char* kPh[] = {"prelude", "pass0", "pass1",
                              "pass2",   "coda",  "final"};
  double cls_ms[6] = {}, ph_ms[6] = {};
  for (const TsSeg& s : ts_segs_) {
    cls_ms[s.cls] += s.ms;
    ph_ms[s.phase] += s.ms;
  }
  const double busy_tok = ts_gpu_ms_ / 1e6 / tok;
  std::fprintf(stderr,
               "gpu-ts: ===== aggregate over %llu token(s) =====\n",
               (unsigned long long)ts_tokens_);
  std::fprintf(stderr,
               "gpu-ts: gpu-busy %.2f ms/token  (host record %.2f ms, "
               "wait %.2f ms per token)\n",
               busy_tok, stats.record_ms / tok, stats.wait_ms / tok);
  const double ndisp = double(stats.dispatches) / tok;
  std::fprintf(stderr,
               "gpu-ts: overhead-est: dispatches=%.0f, GPU-total=%.2fms, "
               "per-dispatch=%.1fus\n",
               ndisp, busy_tok, busy_tok * 1e3 / std::max(1.0, ndisp));
  std::fprintf(stderr, "gpu-ts: -- by op-class (ms/token) --\n");
  for (int k = 0; k < 6; k++)
    std::fprintf(stderr, "gpu-ts:   %-12s %8.2f ms  %5.1f%%\n", kCls[k],
                 cls_ms[k] / 1e6 / tok, 100.0 * cls_ms[k] / ts_gpu_ms_);
  std::fprintf(stderr, "gpu-ts: -- by phase (ms/token) --\n");
  for (int k = 0; k < 6; k++)
    std::fprintf(stderr, "gpu-ts:   %-12s %8.2f ms  %5.1f%%\n", kPh[k],
                 ph_ms[k] / 1e6 / tok, 100.0 * ph_ms[k] / ts_gpu_ms_);
  std::vector<uint32_t> ord(ts_segs_.size());
  for (uint32_t i = 0; i < ord.size(); i++) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) {
    return ts_segs_[a].ms > ts_segs_[b].ms;
  });
  std::fprintf(stderr, "gpu-ts: -- top-25 stages (ms/token) --\n");
  const uint32_t top = std::min<uint32_t>(25, uint32_t(ord.size()));
  for (uint32_t i = 0; i < top; i++) {
    const TsSeg& s = ts_segs_[ord[i]];
    std::fprintf(stderr, "gpu-ts:   %2u %-24s %-11s %-7s %8.3f ms  %4.1f%%\n",
                 i + 1, s.label.c_str(), kCls[s.cls], kPh[s.phase],
                 s.ms / 1e6 / tok, 100.0 * s.ms / ts_gpu_ms_);
  }
}

void GpuGraph::sync() {
  if (building_) return;
  rec_->barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
  stats.barriers++;
}

// ------------------------------------------------------------------ visit

void GpuGraph::emit_visit(int i, int p, int64_t pos) {
  const ModelConfig& c = *cfg_;
  const int H = H_;
  const int64_t Hb = int64_t(H) * 4;
  const int vidx = c.visit_index(i, p);
  const int slot = c.cache_slot(i, p);
  const Lw& L = lw_[i];
  const float eps = float(c.rms_norm_eps);
  const uint32_t rank = uint32_t(c.loop.gate_rank);
  // INK_GPU_TS scope: v0-5 prelude, v6-17/18-29/30-41 pass0-2, v42-47 coda.
  cur_scope_ = "v" + std::to_string(vidx);
  cur_phase_ = vidx < 6 ? 0 : vidx < 18 ? 1 : vidx < 30 ? 2 : vidx < 42 ? 3 : 4;

  // visit-input taps (vidx 5/23/41) + per-visit dump row
  if (vidx == 5 || vidx == 23 || vidx == 41) {
    const int j = vidx == 5 ? 0 : vidx == 23 ? 1 : 2;
    cp(partial_, 0, readback_, (V_ + int64_t(j) * H) * 4, Hb, "tap_in");
  }
  cp(partial_, 0, readback_, (rb_vis_ + int64_t(vidx * 2) * H) * 4, Hb,
     "dump_in");
  const bool dbg = dbg_en_ && vidx == 5;
  auto dbg_cp = [&](const vk::Buffer& s, int row) {
    if (dbg && !building_)
      cp(s, 0, readback_, (rb_dbg_ + int64_t(row) * H) * 4, s.size, "dbg");
  };

  // messages from the previous pass (loop layers only)
  if (p > 0) {
    sh::PcAttnRes pc{2u, uint32_t(H), 4u};
    op(p_ares_,
       {BW(bank_), BA(msgq_, int64_t(i - c.loop.core_start) * 4 * H, 4 * H),
        BW(msgs_)},
       &pc, sizeof(pc), 1, 1, "ares_msg");
    sync();
  }

  // attn_res attention read: sources = block[0..nb0) ++ x_in (partial).
  // x_in always lands in the next block row — it is BOTH the last source
  // AND the pending boundary push (row nb0 is committed at i%6==0).
  const int nb0 = nblock_;
  cp(partial_, 0, block_, int64_t(nb0) * Hb, Hb, "blk_app");
  if (i > 0) {
    sync();
    sh::PcAttnRes pc{uint32_t(nb0 + 1), uint32_t(H), L.full ? 3u : 1u};
    op(p_ares_,
       {BW(block_), BA(attnq_, int64_t(i - 1) * 3 * H, int64_t(pc.Q) * H),
        BW(reads_)},
       &pc, sizeof(pc), 1, 1, "ares_rd");
  } else {
    cp(partial_, 0, reads_, 0, Hb, "rd_copy");
  }
  sync();

  // stream inputs: optional message gating, then per-stream norms.
  // Sliding layers share ONE query row (xq=xk=xv): p=0 produces only
  // inbuf row 0 and all matvecs bind X at offset 0.
  const bool uni = !L.full && p == 0;
  // element offsets into inbuf_ rows (BA() multiplies by 4 for bytes)
  const int64_t xo[3] = {0, uni ? 0 : int64_t(H), uni ? 0 : int64_t(2 * H)};
  if (p > 0) {
    for (int st = 0; st < 3; st++) {
      sh::PcMsgGate pc{uint32_t(H), rank};
      op(p_msg_,
         {BW(*L.ctrl[st][0]), BW(*L.ctrl[st][1]), BW(*L.ctrl[st][2]),
          BA(reads_, (L.full ? st : 0) * H, H), BA(msgs_, int64_t(st) * H, H),
          BA(mgated_, int64_t(st) * H, H)},
         &pc, sizeof(pc), 1, 1, "msg_gate");
    }
    sync();
    for (int st = 0; st < 3; st++) {
      sh::PcRmsnorm pc{uint32_t(H), eps, sh::F_USE_GAMMA};
      if (L.full)
        op(p_rms_,
           {BA(mgated_, int64_t(st) * H, H), BW(*L.norms[st]),
            BA(inbuf_, int64_t(st) * H, H)},
           &pc, sizeof(pc), 1, 1, "norm_in");
      else if (st == 0)
        op(p_rms_, {BW(mgated_), BW(*L.norms[0]), BW(inbuf_)}, &pc, sizeof(pc),
           3, 1, "norm_in");
    }
  } else {
    for (int st = 0; st < 3; st++) {
      sh::PcRmsnorm pc{uint32_t(H), eps, sh::F_USE_GAMMA};
      if (L.full)
        op(p_rms_,
           {BA(reads_, int64_t(st) * H, H), BW(*L.norms[st]),
            BA(inbuf_, int64_t(st) * H, H)},
           &pc, sizeof(pc), 1, 1, "norm_in");
      else if (st == 0)
        op(p_rms_, {BW(reads_), BW(*L.norms[0]), BW(inbuf_)}, &pc, sizeof(pc),
           1, 1, "norm_in");
    }
  }
  sync();

  // ---------------- attention (ops::attn_step mirror) ----------------
  mv(*L.wq.p, BW(*L.wq.b), BA(inbuf_, xo[0], H), BW(q_), 3072, H, "wq");
  mv(*L.wk.p, BW(*L.wk.b), BA(inbuf_, xo[1], H), BW(k_), 1024, H, "wk");
  mv(*L.wv.p, BW(*L.wv.b), BA(inbuf_, xo[2], H), BW(v_), 1024, H, "wv");
  mv(*L.wr.p, BW(*L.wr.b), BA(inbuf_, xo[0], H), BW(r_), 384, H, "wr");
  sync();

  const int Ckv = 1024;  // hidden_size/... kv width (HK*DH)
  dbg_cp(k_, 14);        // pre-conv k (visit5)
  if (L.full) {
    mv(*L.kconv_wp.p, BW(*L.kconv_wp.b), BA(inbuf_, xo[1], H), BW(tk_), 32, H,
       "kconp");
    mv(*L.vconv_wp.p, BW(*L.vconv_wp.b), BA(inbuf_, xo[2], H), BW(tv_), 32, H,
       "vconp");
    sync();
    sh::PcConv4Dyn pk{uint32_t(Ckv), 128u, uint32_t(conv_filled_[slot][0]),
                      sh::F_UPDATE};
    op(p_cvd_, {BW(k_), BW(*L.kconv_sw), BW(tk_), histb(slot, 0, Ckv), BW(k_)},
       &pk, sizeof(pk), wg64(Ckv), 1, "kconv");
    sh::PcConv4Dyn pv{uint32_t(Ckv), 128u, uint32_t(conv_filled_[slot][1]),
                      sh::F_UPDATE};
    op(p_cvd_, {BW(v_), BW(*L.vconv_sw), BW(tv_), histb(slot, 1, Ckv), BW(v_)},
       &pv, sizeof(pv), wg64(Ckv), 1, "vconv");
  } else {
    sh::PcConv4Static pk{uint32_t(Ckv), uint32_t(conv_filled_[slot][0]),
                         sh::F_UPDATE};
    op(p_cvs_, {BW(k_), BW(*L.kconv_w), histb(slot, 0, Ckv), BW(k_)}, &pk,
       sizeof(pk), wg64(Ckv), 1, "kconv");
    sh::PcConv4Static pv{uint32_t(Ckv), uint32_t(conv_filled_[slot][1]),
                         sh::F_UPDATE};
    op(p_cvs_, {BW(v_), BW(*L.vconv_w), histb(slot, 1, Ckv), BW(v_)}, &pv,
       sizeof(pv), wg64(Ckv), 1, "vconv");
  }
  sync();

  {
    PcClip cq{128u, float(c.qk_l2_clip_threshold), sh::F_USE_SCALE};
    op(p_l2_, {BW(q_), BW(*L.qclip)}, &cq, sizeof(cq), 24, 1, "clip_q");
    PcClip ck{128u, float(c.qk_l2_clip_threshold), 0u};
    op(p_l2_, {BW(k_), BW(dummy4_)}, &ck, sizeof(ck), 8, 1, "clip_k");
    PcClip cr{16u, float(c.rel_l2_clip_threshold), sh::F_USE_SCALE};
    op(p_l2_, {BW(r_), BW(*L.rclip)}, &cr, sizeof(cr), 24, 1, "clip_r");
    sh::PcRmsnorm cv{128u, eps, 0u};  // per-head unweighted v norm
    op(p_rms_, {BW(v_), BW(dummy4_), BW(v_)}, &cv, sizeof(cv), 8, 1, "norm_v");
  }
  sync();

  if (L.full) {
    const double tau =
        1.0 + c.log_scaling_alpha *
                  std::log(std::max(double(pos + 1) / c.log_scaling_n_floor, 1.0));
    PcTau pq{3072u, float(tau)};
    op(p_log_, {BW(q_)}, &pq, sizeof(pq), wg64(3072), 1, "log_q");
    PcTau pr{384u, float(tau)};
    op(p_log_, {BW(r_)}, &pr, sizeof(pr), wg64(384), 1, "log_r");
    sync();
  }

  const int64_t len = kv_len_[slot] + 1;
  const int64_t j0 =
      L.full ? 0 : std::max<int64_t>(0, len - c.sliding_window_size);
  const uint32_t n = uint32_t(len - j0), pstr = n + 1;
  cp(k_, 0, kvk_, kvb(kvk_, slot).off + VkDeviceSize(kv_len_[slot]) * 4096,
     4096, "kv_k");
  cp(v_, 0, kvv_, kvb(kvv_, slot).off + VkDeviceSize(kv_len_[slot]) * 4096,
     4096, "kv_v");
  {
    sh::PcRelBias pc{16u, 1024u, 24u * 1024u};
    op(p_rel_, {BW(r_), BW(*L.relproj), BW(rel_)}, &pc, sizeof(pc),
       wg64(24 * 1024), 1, "relbias");
  }
  mv(*L.gate.p, BW(*L.gate.b), BA(inbuf_, xo[0], H), BW(grate_), 3072, H,
     "gate");
  mv(*L.sink.p, BW(*L.sink.b), BA(inbuf_, xo[0], H), BW(sink_), 24, H, "sink");
  sync();

  {
    sh::PcAttnScores pc{24u, 8u, 128u, uint32_t(j0), n, uint32_t(pos), 1024u,
                        pstr};
    op(p_sco_, {BW(q_), kvb(kvk_, slot), BW(rel_), BW(probs_)}, &pc,
       sizeof(pc), 24, (n + 63) / 64, "scores");
  }
  sync();
  {
    sh::PcAttnSoftmax pc{n, pstr};
    op(p_smx_, {BW(probs_), BW(sink_)}, &pc, sizeof(pc), 24, 1, "softmax");
  }
  sync();
  {
    sh::PcAttnAccum pc{24u, 8u, 128u, uint32_t(j0), n, pstr};
    op(p_acc_, {BW(probs_), kvb(kvv_, slot), BW(o_)}, &pc, sizeof(pc), 24, 1,
       "accum");
  }
  sync();
  {
    PcN pc{3072u};
    op(p_sig_, {BW(o_), BW(grate_)}, &pc, sizeof(pc), wg64(3072), 1, "ogate");
  }
  sync();
  if (dbg && !building_) {
    cp(tk_, 0, readback_, (rb_dbg_ + int64_t(6) * H) * 4, 32 * 4);
    cp(tv_, 0, readback_, (rb_dbg_ + int64_t(6) * H) * 4 + 128, 32 * 4);
  }
  dbg_cp(q_, 0);
  dbg_cp(k_, 2);
  dbg_cp(v_, 3);
  dbg_cp(o_, 4);
  mv(*L.wo.p, BW(*L.wo.b), BW(o_), BW(attn_out_), H, 3072, "wo");
  sync();
  dbg_cp(attn_out_, 7);
  {
    sh::PcConv4Static pc{uint32_t(H), uint32_t(conv_filled_[slot][2]),
                         sh::F_UPDATE};
    op(p_cvs_, {BW(attn_out_), BW(*L.attn_sconv), histb(slot, 2, H), BW(a_out_)},
       &pc, sizeof(pc), wg64(H), 1, "attn_conv");
  }
  sync();

  // block boundary push / residual
  const vk::Buffer* resid;
  if (i % c.n_layers_per_attn_res_block == 0) {
    nblock_++;  // x_in was already written into row nb0 by the attn append
    resid = &a_out_;
  } else {
    PcN pc{uint32_t(H)};
    op(p_add_, {BW(partial_), BW(a_out_), BW(residual_)}, &pc, sizeof(pc),
       wg64(H), 1, "resid");
    sync();
    resid = &residual_;
  }
  cp(*resid, 0, block_, int64_t(nblock_) * Hb, Hb, "blk_push");
  sync();
  dbg_cp(*resid, 8);

  // mlp attn_res read: block[0..nblock) ++ residual
  {
    sh::PcAttnRes pc{uint32_t(nblock_ + 1), uint32_t(H), 1u};
    op(p_ares_, {BW(block_), BA(mlpq_, int64_t(i) * H, H), BW(m_in_)}, &pc,
       sizeof(pc), 1, 1, "ares_mlp");
  }
  sync();
  dbg_cp(m_in_, 9);

  if (p > 0) {
    sh::PcMsgGate pcg{uint32_t(H), rank};
    op(p_msg_,
       {BW(*L.ctrl[3][0]), BW(*L.ctrl[3][1]), BW(*L.ctrl[3][2]), BW(m_in_),
        BA(msgs_, 3 * H, H), BA(mgated_, 3 * H, H)},
       &pcg, sizeof(pcg), 1, 1, "msg_mlp");
    sync();
    sh::PcRmsnorm pc{uint32_t(H), eps, sh::F_USE_GAMMA};
    op(p_rms_, {BA(mgated_, 3 * H, H), BW(*L.mlp_norm), BW(mn_)}, &pc,
       sizeof(pc), 1, 1, "norm_mlp");
  } else {
    sh::PcRmsnorm pc{uint32_t(H), eps, sh::F_USE_GAMMA};
    op(p_rms_, {BW(m_in_), BW(*L.mlp_norm), BW(mn_)}, &pc, sizeof(pc), 1, 1,
       "norm_mlp");
  }
  sync();
  dbg_cp(mn_, 10);

  {
    sh::PcRouter pc{128u, 8u, uint32_t(H), float(c.route_scale)};
    op(p_rtr_, {BW(*L.router_w), BW(*L.router_b), BW(mn_), BW(ridx_), BW(rw_)},
       &pc, sizeof(pc), 1, 1, "router");
  }
  sync();
  if (dbg && !building_) {
    cp(ridx_, 0, readback_, (rb_dbg_ + int64_t(13) * H) * 4, 32);
    cp(rw_, 0, readback_, (rb_dbg_ + int64_t(13) * H) * 4 + 32, 32);
  }

  // MoE: per-slot gate/up (16 unsynced GEMVs), then 8 unsynced w2+swiglu
  // writing per-slot rows (zeroed up front; += semantics inside the shader),
  // then one moe_reduce summing the rows into moe_out_. The 8 expert weight
  // streams overlap on the link instead of serializing behind per-slot
  // barriers — w2 bytes are ~1/3 of the per-token PCIe traffic.
  fill(moe_rows_, 0, 8 * Hb, 0u, "moe_zero");
  const Pipe& p13 = L.mxfp8 ? p_mx8m_ : p_nv4m_;
  const int64_t I = c.intermediate_size;
  // All 8 expert slots in ONE dispatch per weight (slot = gl_WorkGroupID.y):
  // per-dispatch fixed cost, not bandwidth, dominated the per-slot version.
  if (L.mxfp8) {
    sh::PcMxfp8 pg{0u, uint32_t(L.w13R), uint32_t(H), 0u, 2u, 1.f, 0u};
    op(p13, {BW(*L.w13), BW(*L.w13s), BW(mn_), BW(gate_), BW(ridx_)}, &pg,
       sizeof(pg), uint32_t(I), 8, "w13");
    sh::PcMxfp8 pu{0u, uint32_t(L.w13R), uint32_t(H), 1u, 2u, 1.f, 0u};
    op(p13, {BW(*L.w13), BW(*L.w13s), BW(mn_), BW(up_), BW(ridx_)}, &pu,
       sizeof(pu), uint32_t(I), 8, "w13");
  } else {
    sh::PcNvfp4 pg{0u, uint32_t(L.w13R), uint32_t(H), 0u, 2u, 1.f, 0u};
    op(p13, {BW(*L.w13), BW(*L.w13s), BW(*L.w13a), BW(*L.w13o), BW(mn_),
             BW(gate_), BW(ridx_)},
       &pg, sizeof(pg), uint32_t(I), 8, "w13");
    sh::PcNvfp4 pu{0u, uint32_t(L.w13R), uint32_t(H), 1u, 2u, 1.f, 0u};
    op(p13, {BW(*L.w13), BW(*L.w13s), BW(*L.w13a), BW(*L.w13o), BW(mn_),
             BW(up_), BW(ridx_)},
       &pu, sizeof(pu), uint32_t(I), 8, "w13");
  }
  sync();
  {
    sh::PcMoeAccum pc{0u, uint32_t(L.w2R), uint32_t(I), L.mxfp8 ? 1u : 0u,
                      float(c.mlp_gate_softcap), float(c.mlp_up_softcap), 0.f};
    op(p_moem_,
       {BW(gate_), BW(up_), BW(*L.w2), BW(*L.w2s), BW(*L.w2a), BW(*L.w2o),
        BW(moe_rows_), BW(ridx_), BW(rw_)},
       &pc, sizeof(pc), uint32_t(H), 8, "w2");
  }
  sync();
  {
    sh::PcMoeReduce pc{uint32_t(H), 8u};
    op(p_mred_, {BW(moe_rows_), BW(moe_out_)}, &pc, sizeof(pc),
       wg64(H), 1, "moe_reduce");
  }
  sync();
  dbg_cp(moe_out_, 11);
  {
    sh::PcConv4Static pc{uint32_t(H), uint32_t(conv_filled_[slot][3]),
                         sh::F_UPDATE};
    op(p_cvs_, {BW(moe_out_), BW(*L.mlp_sconv), histb(slot, 3, H), BW(mo_)},
       &pc, sizeof(pc), wg64(H), 1, "mlp_conv");
  }
  sync();
  dbg_cp(mo_, 12);
  {
    PcN pc{uint32_t(H)};
    op(p_add_, {BW(*resid), BW(mo_), BW(partial_)}, &pc, sizeof(pc), wg64(H),
       1, "out_add");
  }
  sync();

  cp(partial_, 0, readback_, (rb_vis_ + int64_t(vidx * 2 + 1) * H) * 4, Hb,
     "dump_out");

  kv_len_[slot]++;
  for (int st = 0; st < 4; st++)
    conv_filled_[slot][st] = std::min(3, conv_filled_[slot][st] + 1);
}

void GpuGraph::emit_token(int token, int64_t pos) {
  const ModelConfig& c = *cfg_;
  const int H = H_;
  const int64_t Hb = int64_t(H) * 4;
  nblock_ = 0;

  cur_scope_ = "pre";  // INK_GPU_TS: non-visit sections get an explicit scope
  cur_phase_ = 0;
  {
    PcEmb pc{uint32_t(token), uint32_t(H), float(c.rms_norm_eps), emb_flags_};
    op(p_emb_, {BW(embedw_), BW(embn_), BW(partial_)}, &pc, sizeof(pc), 1, 1,
       "embed");
  }
  sync();

  for (int i = 0; i < c.loop.core_start; i++) emit_visit(i, 0, pos);

  // prelude snapshot: pre_partial = partial; pre_block = block
  cur_scope_ = "snap";
  cp(partial_, 0, pre_partial_, 0, Hb, "snap_p");
  pre_nblock_ = nblock_;
  for (int r = 0; r < nblock_; r++)
    cp(block_, int64_t(r) * Hb, pre_block_, int64_t(r) * Hb, Hb, "snap_b");
  sync();

  for (int p = 0; p < c.loop.num_passes; p++) {
    if (p > 0) {
      // bank0 = block.back() (last loop-boundary block), bank1 = prev partial
      cur_scope_ = "bank";
      cur_phase_ = uint8_t(1 + p);  // restore work belongs to this pass
      cp(block_, int64_t(nblock_ - 1) * Hb, bank_, 0, Hb, "bank0");
      cp(partial_, 0, bank_, Hb, Hb, "bank1");
      cp(pre_partial_, 0, partial_, 0, Hb, "rst_p");
      for (int r = 0; r < pre_nblock_; r++)
        cp(pre_block_, int64_t(r) * Hb, block_, int64_t(r) * Hb, Hb, "rst_b");
      nblock_ = pre_nblock_;
      sync();
    }
    for (int i = c.loop.core_start; i < c.loop.core_end; i++)
      emit_visit(i, p, pos);
  }
  for (int i = c.loop.core_end; i < c.num_hidden_layers; i++)
    emit_visit(i, 0, pos);

  // final learned attn_res read + rmsnorm + unembed + softcap
  cur_scope_ = "fin";
  cur_phase_ = 5;
  cp(partial_, 0, block_, int64_t(nblock_) * Hb, Hb, "blk_fin");
  sync();
  {
    sh::PcAttnRes pc{uint32_t(nblock_ + 1), uint32_t(H), 1u};
    op(p_ares_, {BW(block_), BW(outq_), BW(h_)}, &pc, sizeof(pc), 1, 1,
       "ares_fin");
  }
  sync();
  {
    sh::PcRmsnorm pc{uint32_t(H), float(c.rms_norm_eps), sh::F_USE_GAMMA};
    op(p_rms_, {BW(h_), BW(normw_), BW(hfin_)}, &pc, sizeof(pc), 1, 1,
       "norm_fin");
  }
  sync();
  cp(hfin_, 0, readback_, (V_ + 3 * int64_t(H)) * 4, Hb, "tap_fin");
  mv(*unemb_mv_.p, BW(*unemb_mv_.b), BW(hfin_), BW(logits_), V_, H, "unemb");
  sync();
  {
    PcCap pc{uint32_t(V_), float(c.final_logit_softcapping)};
    op(p_cap_, {BW(logits_)}, &pc, sizeof(pc), wg64(V_), 1, "softcap");
  }
  sync();
  cp(logits_, 0, readback_, 0, VkDeviceSize(V_) * 4, "logits");
}

// -------------------------------------------------------------------- api

void GpuGraph::reset() {
  std::fill(kv_len_, kv_len_ + 48, 0);
  std::memset(conv_filled_, 0, sizeof(conv_filled_));
  nblock_ = pre_nblock_ = 0;
}

const float* GpuGraph::tap(int i) const { return rb_ + V_ + int64_t(i) * H_; }
const float* GpuGraph::visit_dump(int vidx, int phase) const {
  return rb_ + rb_vis_ + int64_t(vidx * 2 + phase) * H_;
}
const float* GpuGraph::dbg(int row) const { return rb_ + rb_dbg_ + int64_t(row) * H_; }

const std::vector<float>& GpuGraph::forward_token(int token, int64_t pos) {
  vk::Recorder rec;
  if (!rec.begin(cpool_)) {
    err_ = "recorder begin failed: " + rec.error();
    return logits_host_;
  }
  rec_ = &rec;
  site_cursor_ = 0;
  building_ = false;
  const auto t0 = clk::steady_clock::now();
  if (ts_en_) {
    // slot 0 = cb start; slots 1.. = one mark after each recorded op.
    rec_->reset_query_pool(ts_pool_, 0, ts_cap_);
    rec_->write_timestamp(ts_pool_, 0);
    ts_next_ = 1;
  }
  emit_token(token, pos);
  stats.record_ms +=
      clk::duration<double, std::milli>(clk::steady_clock::now() - t0).count();
  const auto w0 = clk::steady_clock::now();
  if (!rec.submit_and_wait(ctx_->compute_queue)) {
    err_ = "token submit failed: " + rec.error();
    rec_ = nullptr;
    rec.shutdown();
    return logits_host_;
  }
  stats.wait_ms +=
      clk::duration<double, std::milli>(clk::steady_clock::now() - w0).count();
  stats.submits++;
  if (ts_en_) ts_collect();
  rec_ = nullptr;
  rec.shutdown();
  alloc_->invalidate(readback_);
  std::memcpy(logits_host_.data(), rb_, size_t(V_) * 4);
  return logits_host_;
}

// -------------------------------------------------------------------- init

bool GpuGraph::init(const ModelConfig& c, const WeightStore& w,
                    ink::vk::Context& ctx, ink::vk::Allocator& alloc,
                    const std::string& shader_dir, int64_t ctx_cap,
                    std::string* err) {
  cfg_ = &c;
  ctx_ = &ctx;
  alloc_ = &alloc;
  shader_dir_ = shader_dir;
  cap_ = std::max<int64_t>(ctx_cap, 1);
  H_ = c.hidden_size;
  V_ = c.unpadded_vocab_size;
  if (c.num_hidden_layers != 24 || c.loop.core_start != 6 ||
      c.loop.core_end != 18 || c.loop.num_passes != 3) {
    *err = "gpu backend: only the 24-layer / 3-pass looped layout is wired";
    return false;
  }

  if (!cpool_.init(ctx, ctx.compute_family) ||
      !tpool_.init(ctx, ctx.copy_family())) {
    *err = "gpu backend: command pool init failed";
    return false;
  }
  auto dev = [&](int64_t bytes) {
    vk::Buffer b = alloc.alloc_device_local(VkDeviceSize(bytes));
    if (!b.buf && err_.empty())
      err_ = "gpu backend: device alloc failed: " + alloc.last_error;
    return b;
  };
  const int64_t H = H_;

  staging_ = alloc.alloc_host(64ull << 20);
  dummy4_ = dev(16);
  partial_ = dev(H * 4);
  block_ = dev(5 * H * 4);
  pre_partial_ = dev(H * 4);
  pre_block_ = dev(4 * H * 4);
  bank_ = dev(2 * H * 4);
  reads_ = dev(3 * H * 4);
  msgs_ = dev(4 * H * 4);
  mgated_ = dev(4 * H * 4);
  inbuf_ = dev(3 * H * 4);
  q_ = dev(3072 * 4);
  k_ = dev(1024 * 4);
  v_ = dev(1024 * 4);
  r_ = dev(384 * 4);
  tk_ = dev(32 * 4);
  tv_ = dev(32 * 4);
  rel_ = dev(24 * 1024 * 4);
  probs_ = dev(24 * (cap_ + 8) * 4);
  grate_ = dev(3072 * 4);
  sink_ = dev(24 * 4);
  o_ = dev(3072 * 4);
  attn_out_ = dev(H * 4);
  a_out_ = dev(H * 4);
  residual_ = dev(H * 4);
  m_in_ = dev(H * 4);
  mn_ = dev(H * 4);
  moe_out_ = dev(H * 4);
  mo_ = dev(H * 4);
  ridx_ = dev(32);
  rw_ = dev(32);
  gate_ = dev(8 * 1152 * 4);
  up_ = dev(8 * 1152 * 4);
  moe_rows_ = dev(8 * H * 4);
  kvk_ = dev(48 * cap_ * 1024 * 4);
  kvv_ = dev(48 * cap_ * 1024 * 4);
  hist_ = dev(48 * 4 * 3 * H * 4);
  h_ = dev(H * 4);
  hfin_ = dev(H * 4);
  logits_ = dev(V_ * 4);
  rb_vis_ = V_ + 4 * H;
  rb_dbg_ = rb_vis_ + 96 * H;
  readback_ = alloc.alloc_host((rb_dbg_ + 16 * H) * 4);
  if (!readback_.buf) {
    *err = "gpu backend: readback alloc failed: " + alloc.last_error;
    return false;
  }
  rb_ = (float*)alloc.mapped(readback_);
  if (!err_.empty()) {
    *err = err_;
    return false;
  }

  // ------------------------------- pipelines
  const bool okp =
      load_pipe("rmsnorm", 3, sizeof(sh::PcRmsnorm), p_rms_) &&
      load_pipe("matvec_f32", 3, sizeof(sh::PcMatvec), p_mvf_) &&
      load_pipe("matvec_bf16", 3, sizeof(sh::PcMatvec), p_mvb_) &&
      load_pipe("matvec_f8e4m3", 3, sizeof(sh::PcMatvec), p_mv8_) &&
      load_pipe("conv4_static", 4, sizeof(sh::PcConv4Static), p_cvs_) &&
      load_pipe("conv4_dynamic", 5, sizeof(sh::PcConv4Dyn), p_cvd_) &&
      load_pipe("rel_bias", 3, sizeof(sh::PcRelBias), p_rel_) &&
      load_pipe("attn_scores", 4, sizeof(sh::PcAttnScores), p_sco_) &&
      load_pipe("attn_softmax", 2, sizeof(sh::PcAttnSoftmax), p_smx_) &&
      load_pipe("attn_accum", 3, sizeof(sh::PcAttnAccum), p_acc_) &&
      load_pipe("attn_res", 3, sizeof(sh::PcAttnRes), p_ares_) &&
      load_pipe("msg_gate", 6, sizeof(sh::PcMsgGate), p_msg_) &&
      load_pipe("router", 5, sizeof(sh::PcRouter), p_rtr_) &&
      load_pipe("moe_accum_mslot", 9, sizeof(sh::PcMoeAccum), p_moem_) &&
      load_pipe("matvec_nvfp4_mslot", 7, sizeof(sh::PcNvfp4), p_nv4m_) &&
      load_pipe("matvec_mxfp8_mslot", 5, sizeof(sh::PcMxfp8), p_mx8m_) &&
      load_pipe("moe_accum_slot", 9, sizeof(sh::PcMoeAccum), p_moe_) &&
      load_pipe("moe_reduce", 2, sizeof(sh::PcMoeReduce), p_mred_) &&
      load_pipe("sigmoid_mul", 2, sizeof(PcN), p_sig_) &&
      load_pipe("softcap", 1, sizeof(PcCap), p_cap_) &&
      load_pipe("vadd", 3, sizeof(PcN), p_add_) &&
      load_pipe("l2_clip", 2, sizeof(PcClip), p_l2_) &&
      load_pipe("log_scale", 1, sizeof(PcTau), p_log_) &&
      load_pipe("embed_norm", 3, sizeof(PcEmb), p_emb_) &&
      load_pipe("matvec_nvfp4_slot", 7, sizeof(sh::PcNvfp4), p_nv4s_) &&
      load_pipe("matvec_mxfp8_slot", 5, sizeof(sh::PcMxfp8), p_mx8s_);
  if (!okp) {
    if (err->empty()) *err = "gpu backend: pipeline load failed: " + err_;
    return false;
  }

  // -------------------------------- weights
  const int nl = c.num_hidden_layers;
  std::vector<std::string> missing;
  auto get = [&](const std::string& n) -> const TensorView* {
    const TensorView* t = w.find(n);
    if (!t || !t->data) missing.push_back(n);
    return t;
  };

  // globals — names mirror Graph::init exactly.
  {
    const TensorView* t = get("model.llm.embed.weight");
    if (t) {
      embedw_ = upload_dev(t->data, t->nbytes());
      emb_flags_ = (t->dtype == DType::BF16) ? sh::F_SRC_BF16 : 0u;
    }
    if (const TensorView* g = get("model.llm.embed_norm.weight")) {
      std::vector<float> f(size_t(g->numel()));
      for (int64_t i = 0; i < g->numel(); i++) f[size_t(i)] = g->f32_linear(i);
      embn_ = upload_dev(f.data(), H * 4);
    }
    if (const TensorView* g = get("model.llm.norm.weight")) {
      std::vector<float> f(size_t(g->numel()));
      for (int64_t i = 0; i < g->numel(); i++) f[size_t(i)] = g->f32_linear(i);
      normw_ = upload_dev(f.data(), H * 4);
    }
    if (const TensorView* g = get("model.llm.unembed.weight")) {
      unembw_ = upload_dev(g->data, g->nbytes());
      unemb_mv_.b = &unembw_;
      unemb_mv_.p = mvpipe(g->dtype);
      if (!unemb_mv_.p) missing.push_back("unembed dtype");
    }
    if (const TensorView* g = get("model.llm.attn_res.output_query")) {
      std::vector<float> f((size_t)H);
      for (int64_t i = 0; i < H; i++) f[size_t(i)] = g->f32(i);
      outq_ = upload_dev(f.data(), H * 4);
    }
  }

  // query tables packed at fixed strides: attn_q[i] at i*3H (Q<=3 rows
  // per tensor), mlp_q[i] at i*H — GLOBAL tables, not per-layer names.
  {
    std::vector<float> aq(size_t(23) * 3 * H, 0.f), mq(size_t(nl) * H, 0.f);
    for (int i = 0; i < 23; i++) {
      const TensorView* t =
          get("model.llm.attn_res.attn_queries." + std::to_string(i));
      if (!t) continue;
      const int rows = t->nd == 2 ? int(t->shape[0]) : 1;
      for (int r = 0; r < std::min(3, rows); r++)
        for (int64_t e = 0; e < H; e++)
          aq[size_t(i * 3 + r) * H + e] = t->f32(r, e);
    }
    for (int i = 0; i < nl; i++) {
      const TensorView* t =
          get("model.llm.attn_res.mlp_queries." + std::to_string(i));
      if (!t) continue;
      for (int64_t e = 0; e < H; e++) mq[size_t(i) * H + e] = t->f32(e);
    }
    attnq_ = upload_dev(aq.data(), VkDeviceSize(aq.size()) * 4);
    mlpq_ = upload_dev(mq.data(), VkDeviceSize(mq.size()) * 4);
  }
  // message queries: mq holds [q,mlp] row pairs, mkv holds [k,v] pairs;
  // merged stream order per loop layer is (q, k, v, mlp) — same merge as
  // Graph::init.
  {
    const int lc = c.loop.core_end - c.loop.core_start;
    const TensorView* mqt = get("model.llm.middle_cycle_cross_pass_queries");
    const TensorView* mkvt =
        get("model.llm.middle_cycle_cross_pass_kv_queries");
    std::vector<float> mbuf(size_t(lc) * 4 * H, 0.f);
    if (mqt && mkvt) {
      for (int li = 0; li < lc; li++)
        for (int64_t e = 0; e < H; e++) {
          mbuf[size_t(li * 4 + 0) * H + e] = mqt->f32(2 * li, e);
          mbuf[size_t(li * 4 + 1) * H + e] = mkvt->f32(2 * li, e);
          mbuf[size_t(li * 4 + 2) * H + e] = mkvt->f32(2 * li + 1, e);
          mbuf[size_t(li * 4 + 3) * H + e] = mqt->f32(2 * li + 1, e);
        }
    }
    msgq_ = upload_dev(mbuf.data(), VkDeviceSize(mbuf.size()) * 4);
  }

  for (int i = 0; i < nl; i++) {
    Lw& L = lw_[i];
    L.full = c.is_full_attn(i);
    L.mxfp8 = c.mxfp8_layers.count(i) != 0;
    const std::string p = lname(i, "attn.");
    auto W = [&](const std::string& n, WB& out) {
      const TensorView* t = get(n);
      if (t) {
        out.b = dev_w(*t);
        if (!out.b) {          // VRAM pressure: degrade to host mirror
          std::string e0 = err_;
          out.b = host_w(*t);
          if (out.b) err_ = e0;
        }
        out.p = mvpipe(t->dtype);
        if (!out.p) missing.push_back(n + " dtype");
      }
    };
    auto F = [&](const std::string& n, const vk::Buffer*& out) {
      const TensorView* t = get(n);
      if (t) out = dev_f32(*t);
    };
    auto HB = [&](const std::string& n, const vk::Buffer*& out) {
      const TensorView* t = get(n);
      if (t) out = host_w(*t);
    };
    W(p + "wq_du.weight", L.wq);
    W(p + "wk_dv.weight", L.wk);
    W(p + "wv_dv.weight", L.wv);
    W(p + "wo_ud.weight", L.wo);
    W(p + "gate_attn.weight", L.gate);
    W(p + "attn_sink_proj.weight", L.sink);
    W(p + "wr_du.weight", L.wr);
    F(p + "rel_logits_proj.proj", L.relproj);
    F(p + "q_l2_clip.scale", L.qclip);
    F(p + "rel_l2_clip.scale", L.rclip);
    if (L.full) {
      F(p + "k_sconv.static_w", L.kconv_sw);
      F(p + "v_sconv.static_w", L.vconv_sw);
      W(p + "k_sconv.weight_proj", L.kconv_wp);
      W(p + "v_sconv.weight_proj", L.vconv_wp);
      F(lname(i, "q_input_norm.weight"), L.norms[0]);
      F(lname(i, "k_input_norm.weight"), L.norms[1]);
      F(lname(i, "v_input_norm.weight"), L.norms[2]);
    } else {
      F(p + "k_sconv.weight", L.kconv_w);
      F(p + "v_sconv.weight", L.vconv_w);
      F(lname(i, "attn_norm.weight"), L.norms[0]);
      L.norms[1] = L.norms[2] = L.norms[0];
    }
    F(lname(i, "attn_sconv.weight"), L.attn_sconv);
    F(lname(i, "mlp_sconv.weight"), L.mlp_sconv);
    F(lname(i, "mlp_norm.weight"), L.mlp_norm);
    F(lname(i, "mlp.gate.weight"), L.router_w);
    F(lname(i, "mlp.gate.bias"), L.router_b);
    if (c.is_loop(i)) {
      static const char* kCtrl[4] = {"attn", "k", "v", "mlp"};
      static const char* kMat[3] = {"down_weight", "forget_weight",
                                    "gain_weight"};
      for (int s = 0; s < 4; s++)
        for (int mt = 0; mt < 3; mt++) {
          const TensorView* t =
              get(lname(i, "controllers.") + kCtrl[s] + "." + kMat[mt]);
          if (!t) continue;
          if (mt == 0)
            L.ctrl[s][0] = dev_w(*t);  // bf16-packed for msg_gate DOWN
          else if (mt == 1)
            L.ctrl[s][1] = dev_f32(*t);
          else
            L.ctrl[s][2] = dev_f32(*t);
        }
    }
    // expert tensors: device-resident while expert_vram_ budget lasts
    // (nvfp4 layers first in visit order — best bytes-saved per VRAM byte);
    // host-visible mirrors after that (amax/osd exist only on nvfp4)
    auto EX = [&](const std::string& n, const vk::Buffer*& out) {
      const TensorView* t = w.find(n);
      if (!t || !t->data) {
        missing.push_back(n);
        return;
      }
      out = nullptr;
      // dev-resident while the device heap has real headroom (VK_EXT mem
      // budget); reserve ~768MiB for remaining mandatory allocs + driver.
      VkDeviceSize hb = 0, hu = 0;
      const bool room =
          alloc_->heap_budget(1, hb, hu) &&
          hb > hu + VkDeviceSize(t->nbytes()) + (VkDeviceSize(768) << 20);
      if (room) {
        std::string e0 = err_;
        out = dev_w(*t);
        if (!out) err_ = e0;  // VRAM full: drop dev_w's error, mirror on host
      }
      if (!out) out = host_w(*t);
      if (!out) missing.push_back(n);
    };
    EX(lname(i, "mlp.experts.w13_weight"), L.w13);
    EX(lname(i, "mlp.experts.w13_weight_scale"), L.w13s);
    EX(lname(i, "mlp.experts.w2_weight"), L.w2);
    EX(lname(i, "mlp.experts.w2_weight_scale"), L.w2s);
    L.w13a = L.w13o = L.w2a = L.w2o = &dummy4_;
    if (!L.mxfp8) {
      HB(lname(i, "mlp.experts.w13_weight_amax"), L.w13a);
      HB(lname(i, "mlp.experts.w13_weight_outer_scale_denominator"), L.w13o);
      HB(lname(i, "mlp.experts.w2_weight_amax"), L.w2a);
      HB(lname(i, "mlp.experts.w2_weight_outer_scale_denominator"), L.w2o);
    }
    const TensorView* t13 = w.find(lname(i, "mlp.experts.w13_weight"));
    const TensorView* t2 = w.find(lname(i, "mlp.experts.w2_weight"));
    if (t13 && t13->nd == 3) L.w13R = t13->shape[1];
    if (t2 && t2->nd == 3) L.w2R = t2->shape[1];
  }
  if (!missing.empty()) {
    *err = "gpu backend: " + std::to_string(missing.size()) +
           " tensors missing/failed, first: " + missing.front() +
           (err_.empty() ? "" : " (" + err_ + ")");
    return false;
  }
  if (!err_.empty()) {
    *err = err_;
    return false;
  }

  // ------------------------- build pass: create the fixed descset stream
  dbg_en_ = std::getenv("INK_DBG") != nullptr;

  // INK_GPU_TS=1: timestamp pool covering every op boundary in the token cb.
  // Everything downstream checks ts_en_; without the env var not a single
  // command reaches the recorded buffer.
  if (const char* e = std::getenv("INK_GPU_TS");
      e && e[0] && std::strcmp(e, "0") != 0) {
    uint32_t valid_bits = 0;
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &nq, qf.data());
    if (ctx.compute_family < nq)
      valid_bits = qf[ctx.compute_family].timestampValidBits;
    if (valid_bits == 0 || ctx.props.limits.timestampPeriod <= 0.f) {
      std::fprintf(stderr,
                   "gpu-ts: timestamps unsupported on the compute queue; "
                   "disabled\n");
    } else {
      VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
      qi.queryCount = 8192;  // ~5.2k dispatches + copies + fills + cb start
      if (vkCreateQueryPool(ctx.device, &qi, nullptr, &ts_pool_) ==
          VK_SUCCESS) {
        ts_en_ = true;
        ts_cap_ = 8192;
        ts_period_ = ctx.props.limits.timestampPeriod;
        ts_raw_.assign(ts_cap_, 0);
        std::fprintf(stderr,
                     "gpu-ts: on (timestampPeriod %.3f ns, %u slots, %u "
                     "valid bits)\n",
                     ts_period_, ts_cap_, valid_bits);
      } else {
        std::fprintf(stderr, "gpu-ts: vkCreateQueryPool failed; disabled\n");
      }
    }
  }

  logits_host_.assign(size_t(V_), 0.f);
  building_ = true;
  emit_token(0, 0);
  building_ = false;
  reset();
  if (!err_.empty()) {
    *err = "gpu backend: descriptor build failed: " + err_;
    return false;
  }
  ok_ = true;
  return true;
}

void GpuGraph::shutdown() {
  if (!ctx_ || !ctx_->device) return;
  vkDeviceWaitIdle(ctx_->device);
  if (ts_en_) {
    ts_report();
    vkDestroyQueryPool(ctx_->device, ts_pool_, nullptr);
    ts_pool_ = VK_NULL_HANDLE;
    ts_en_ = false;
  }
  for (auto& kv : devw_) alloc_->free(kv.second);
  for (auto& kv : hostw_) alloc_->free(kv.second);
  for (auto& kv : f32w_) alloc_->free(kv.second);
  devw_.clear();
  hostw_.clear();
  f32w_.clear();
  for (auto& ds : sites_) ds.shutdown(*ctx_);
  sites_.clear();
  vk::Buffer* bufs[] = {&staging_, &dummy4_, &partial_, &block_,
                        &pre_partial_, &pre_block_, &bank_, &reads_, &msgs_,
                        &mgated_, &inbuf_, &q_, &k_, &v_, &r_, &tk_, &tv_,
                        &rel_, &probs_, &grate_, &sink_, &o_, &attn_out_,
                        &a_out_, &residual_, &m_in_, &mn_, &moe_out_, &mo_,
                        &ridx_, &rw_, &gate_, &up_, &kvk_, &kvv_, &hist_, &h_,
                        &hfin_, &logits_, &readback_, &embedw_, &embn_,
                        &normw_, &unembw_, &attnq_, &mlpq_, &msgq_, &outq_};
  for (auto* b : bufs)
    if (b->buf) alloc_->free(*b);
  Pipe* pipes[] = {&p_rms_, &p_mvf_, &p_mvb_, &p_mv8_, &p_cvs_, &p_cvd_,
                   &p_rel_, &p_sco_, &p_smx_, &p_acc_, &p_ares_, &p_msg_,
                   &p_rtr_, &p_moe_, &p_mred_, &p_moem_, &p_nv4m_, &p_mx8m_,
                   &p_sig_, &p_cap_, &p_add_, &p_l2_,
                   &p_log_, &p_emb_, &p_nv4s_, &p_mx8s_};
  for (auto* p : pipes) p->p.shutdown();
  cpool_.shutdown();
  tpool_.shutdown();
  ok_ = false;
  ctx_ = nullptr;
}

}  // namespace ink::gpu
