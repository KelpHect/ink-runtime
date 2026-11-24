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
// Visit number s in 0..47 -> (layer i, pass p) in emit order:
// prelude i<core_start (p=0), loop layers x num_passes, coda i>=core_end
// (p=0). visit_index(i,p) == s for every pair produced here.
void visit_ip(const ModelConfig& c, int s, int& i, int& p) {
  if (s < c.loop.core_start) {
    i = s;
    p = 0;
    return;
  }
  const int nc = c.loop.core_end - c.loop.core_start;
  const int r = s - c.loop.core_start;
  if (r < c.loop.num_passes * nc) {
    p = r / nc;
    i = c.loop.core_start + r % nc;
    return;
  }
  i = c.loop.core_end + (r - c.loop.num_passes * nc);
  p = 0;
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

// cpr: one vkCmdCopyBuffer carrying a whole region array — the driver
// emits a single SDMA packet set instead of one per slice.
void GpuGraph::cpr(const ink::vk::Buffer& s, const ink::vk::Buffer& d,
                   const VkBufferCopy* regs, uint32_t n, const char* tag) {
  if (building_) return;
  rec_->copy_buffer_regions(s.buf, d.buf, regs, n);
  stats.copies += n;
  ts_mark("copy", tag);
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
  if (!ts_en_ || ts_off_) return;
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

void GpuGraph::ts_gap() {
  if (!ts_en_) return;
  const uint32_t slot = ts_next_++;
  if (slot >= ts_cap_) return;
  rec_->write_timestamp(ts_pool_, slot);
  if (ts_segs_.size() < slot) {
    // Boundary at a new command buffer's start: the delta back to the
    // previous mark is the inter-submit gap (fence wait + host record).
    std::string l = cur_scope_;
    l += "/submit_gap";
    ts_segs_.push_back({std::move(l), 5, cur_phase_, 0.0});
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
  if (std::getenv("INK_GPU_TS_FULL")) {
    std::fprintf(stderr, "gpu-ts: -- all %zu stages (ms/token) --\n",
                 ts_segs_.size());
    for (uint32_t i = 0; i < ord.size(); i++) {
      const TsSeg& s = ts_segs_[ord[i]];
      std::fprintf(stderr, "gpu-ts:   %-24s %-11s %-7s %8.4f ms\n",
                   s.label.c_str(), kCls[s.cls], kPh[s.phase],
                   s.ms / 1e6 / tok);
    }
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

void GpuGraph::emit_visit_a(int i, int p, int64_t pos) {
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
    // pc.filled carries the STREAM index now; the count lives in this
    // slot's params row (prm_filled[stream]).
    sh::PcConv4Dyn pk{uint32_t(Ckv), 128u, 0u, sh::F_UPDATE};
    op(p_cvd_,
       {BW(k_), BW(*L.kconv_sw), BW(tk_), histb(slot, 0, Ckv), BW(k_),
        prm_b(slot)},
       &pk, sizeof(pk), wg64(Ckv), 1, "kconv");
    sh::PcConv4Dyn pv{uint32_t(Ckv), 128u, 1u, sh::F_UPDATE};
    op(p_cvd_,
       {BW(v_), BW(*L.vconv_sw), BW(tv_), histb(slot, 1, Ckv), BW(v_),
        prm_b(slot)},
       &pv, sizeof(pv), wg64(Ckv), 1, "vconv");
  } else {
    sh::PcConv4Static pk{uint32_t(Ckv), 0u, sh::F_UPDATE};
    op(p_cvs_, {BW(k_), BW(*L.kconv_w), histb(slot, 0, Ckv), BW(k_),
                prm_b(slot)},
       &pk, sizeof(pk), wg64(Ckv), 1, "kconv");
    sh::PcConv4Static pv{uint32_t(Ckv), 1u, sh::F_UPDATE};
    op(p_cvs_, {BW(v_), BW(*L.vconv_w), histb(slot, 1, Ckv), BW(v_),
                prm_b(slot)},
       &pv, sizeof(pv), wg64(Ckv), 1, "vconv");
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
    // pc.tau is dead under record-once decode: the shader reads prm_tau
    // from the params header row (write_params keeps it current).
    PcTau pq{3072u, float(tau)};
    op(p_log_, {BW(q_), prm_head()}, &pq, sizeof(pq), wg64(3072), 1, "log_q");
    PcTau pr{384u, float(tau)};
    op(p_log_, {BW(r_), prm_head()}, &pr, sizeof(pr), wg64(384), 1, "log_r");
    sync();
  }

  // len/j0/n still feed the (now-dead) PC fields; the shaders read the
  // slot's params row instead — and the kv-append is a dispatch, not a
  // recorded copy, because the bank dst offset varies per token.
  const int64_t len = kv_len_[slot] + 1;
  const int64_t j0 =
      L.full ? 0 : std::max<int64_t>(0, len - c.sliding_window_size);
  const uint32_t n = uint32_t(len - j0);
  {
    PcN kva{uint32_t(Ckv)};
    op(p_kva_, {BW(k_), BW(v_), kvb(kvk_, slot), kvb(kvv_, slot), prm_b(slot)},
       &kva, sizeof(kva), wg64(Ckv), 1, "kv_app");
  }
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
    // j0/n/pos are dead PC fields (slot params row supplies them); pstride
    // and the grid are fixed so the dispatch is identical every token.
    sh::PcAttnScores pc{24u, 8u, 128u, uint32_t(j0), n, uint32_t(pos), 1024u,
                        pstride_};
    op(p_sco_, {BW(q_), kvb(kvk_, slot), BW(rel_), BW(probs_), prm_b(slot)},
       &pc, sizeof(pc), 24, score_gy_, "scores");
  }
  sync();
  {
    sh::PcAttnSoftmax pc{n, pstride_};
    op(p_smx_, {BW(probs_), BW(sink_), prm_b(slot)}, &pc, sizeof(pc), 24, 1,
       "softmax");
  }
  sync();
  {
    sh::PcAttnAccum pc{24u, 8u, 128u, uint32_t(j0), n, pstride_};
    op(p_acc_, {BW(probs_), kvb(kvv_, slot), BW(o_), prm_b(slot)}, &pc,
       sizeof(pc), 24, 1, "accum");
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
    sh::PcConv4Static pc{uint32_t(H), 2u, sh::F_UPDATE};  // stream 2 = attn
    op(p_cvs_,
       {BW(attn_out_), BW(*L.attn_sconv), histb(slot, 2, H), BW(a_out_),
        prm_b(slot)},
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
  // host handoff for expert staging: the router's top-k ids are read back
  // after this submit's fence and program the next submit's slice copies.
  cp(ridx_, 0, readback_, VkDeviceSize(rb_ids_) * 4, 32, "ids");
  // Staged visits end their segment here: this fill is the LAST command,
  // so a host spin on the sentinel sees the ids one memory-write earlier
  // than the cb's fence would fire (fence + tail + wake ~0.2ms).
  if (!L.experts_dev && stage_ok_)
    fill(readback_, VkDeviceSize(rb_sen_ + vidx) * 4, 4,
         uint32_t(vidx + 1), "sen");
  if (dbg && !building_) {
    cp(ridx_, 0, readback_, (rb_dbg_ + int64_t(13) * H) * 4, 32);
    cp(rw_, 0, readback_, (rb_dbg_ + int64_t(13) * H) * 4 + 32, 32);
  }
}

// ---------------------------------------------------- visit tail (MoE->out)

// emit_visit_b: the part of a visit that needs the router result on the
// host before it can be recorded — staged expert-slice copies, then the
// expert GEMVs (mslot for dev-resident layers, *_stage over stg_*_ for
// host-resident ones), moe_reduce, mlp conv, residual add, dump cp and the
// host-side counters. In the record walk it is emitted at the head of the
// submit AFTER emit_visit_a(i,p)'s, so ridx_/rw_ still hold this visit's
// router output and stage_ids_ carries its freshly read top-k ids.
//
// part: 0 = whole tail; 1 = only the w13 phase (moe_zero + gate/up GEMVs,
// which need the w13 slices); 2 = the w2 phase onward (accum, reduce, mlp
// conv, out_add, dump, counters). The two-phase staging path records
// part 1 into its own segment so the w2 slice copies can stream on the
// SDMA queue WHILE the w13 GEMVs run — previously the copies hid under
// ~35ms of host record; record-once left them fully exposed.
void GpuGraph::emit_visit_b(int i, int p, int part) {
  const ModelConfig& c = *cfg_;
  const int H = H_;
  const int64_t Hb = int64_t(H) * 4;
  const int vidx = c.visit_index(i, p);
  const int slot = c.cache_slot(i, p);
  const Lw& L = lw_[i];
  const int64_t I = c.intermediate_size;
  cur_scope_ = "v" + std::to_string(vidx);
  cur_phase_ =
      vidx < 6 ? 0 : vidx < 18 ? 1 : vidx < 30 ? 2 : vidx < 42 ? 3 : 4;
  const bool dbg = dbg_en_ && vidx == 5;
  auto dbg_cp = [&](const vk::Buffer& s, int row) {
    if (dbg && !building_)
      cp(s, 0, readback_, (rb_dbg_ + int64_t(row) * H) * 4, s.size, "dbg");
  };
  // residual source for out_add — same pick as emit_visit_a's block push.
  const vk::Buffer* resid =
      (i % c.n_layers_per_attn_res_block == 0) ? &a_out_ : &residual_;

  // Host-resident experts with staging live: forward_token already ran
  // this visit's slice copies on the transfer queue (SDMA, overlapped
  // with this compute cb's record); the stage shaders then read stg_*_
  // instead of streaming weights over PCIe per GEMV. Without staging
  // (stage_ok_ false — VRAM full) the host mirrors stay bound and the
  // original mslot host-read path runs.
  const bool stage = !L.experts_dev && stage_ok_;
  // Warm cache: a staged visit with a private w13 set keeps last token's
  // slices resident; the rest share the single stg_w13w_/stg_w13s_ pair
  // (always-miss, today's behavior).
  const int bnd = stage ? stg_bmap_[vidx] : -1;
  const bool own = bnd >= 0 && wc_own_[bnd];

  // MoE: per-slot gate/up (16 unsynced GEMVs), then 8 unsynced w2+swiglu
  // writing per-slot rows (zeroed up front; += semantics inside the shader),
  // then one moe_reduce summing the rows into moe_out_. The 8 expert weight
  // streams overlap on the link instead of serializing behind per-slot
  // barriers — w2 bytes are ~1/3 of the per-token PCIe traffic.
  if (part != 2) {
    fill(moe_rows_, 0, 8 * Hb, 0u, "moe_zero");
    const Pipe& p13 =
        L.mxfp8 ? (stage ? p_mx8st_ : p_mx8m_) : (stage ? p_nv4st_ : p_nv4m_);
    const Bnd w13b =
        stage ? BW(own ? w13w_own_[bnd] : stg_w13w_) : BW(*L.w13);
    const Bnd w13sb =
        stage ? BW(own ? w13s_own_[bnd] : stg_w13s_) : BW(*L.w13s);
    // All 8 expert slots in ONE dispatch per weight (slot =
    // gl_WorkGroupID.y): per-dispatch fixed cost, not bandwidth, dominated
    // the per-slot version.
    if (L.mxfp8) {
      sh::PcMxfp8 pg{0u, uint32_t(L.w13R), uint32_t(H), 0u, 2u, 1.f, 0u};
      op(p13, {w13b, w13sb, BW(mn_), BW(gate_), BW(ridx_)}, &pg, sizeof(pg),
         uint32_t(I), 8, "w13");
      sh::PcMxfp8 pu{0u, uint32_t(L.w13R), uint32_t(H), 1u, 2u, 1.f, 0u};
      op(p13, {w13b, w13sb, BW(mn_), BW(up_), BW(ridx_)}, &pu, sizeof(pu),
         uint32_t(I), 8, "w13");
    } else {
      sh::PcNvfp4 pg{0u, uint32_t(L.w13R), uint32_t(H), 0u, 2u, 1.f, 0u};
      op(p13, {w13b, w13sb, BW(*L.w13a), BW(*L.w13o), BW(mn_), BW(gate_),
               BW(ridx_)},
         &pg, sizeof(pg), uint32_t(I), 8, "w13");
      sh::PcNvfp4 pu{0u, uint32_t(L.w13R), uint32_t(H), 1u, 2u, 1.f, 0u};
      op(p13, {w13b, w13sb, BW(*L.w13a), BW(*L.w13o), BW(mn_), BW(up_),
               BW(ridx_)},
         &pu, sizeof(pu), uint32_t(I), 8, "w13");
    }
    sync();
  }
  if (part == 1) return;  // w2 phase waits its own transfer submit
  {
    const Bnd w2b = stage ? BW(stg_w2w_) : BW(*L.w2);
    const Bnd w2sb = stage ? BW(stg_w2s_) : BW(*L.w2s);
    const Pipe& pac = stage ? p_moest_ : p_moem_;
    sh::PcMoeAccum pc{0u, uint32_t(L.w2R), uint32_t(I), L.mxfp8 ? 1u : 0u,
                      float(c.mlp_gate_softcap), float(c.mlp_up_softcap), 0.f};
    op(pac,
       {BW(gate_), BW(up_), w2b, w2sb, BW(*L.w2a), BW(*L.w2o),
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
    sh::PcConv4Static pc{uint32_t(H), 3u, sh::F_UPDATE};  // stream 3 = mlp
    op(p_cvs_,
       {BW(moe_out_), BW(*L.mlp_sconv), histb(slot, 3, H), BW(mo_),
        prm_b(slot)},
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

// ------------------------------------------------------- token sub-sections

void GpuGraph::emit_head(int token, int64_t pos) {
  const ModelConfig& c = *cfg_;
  const int H = H_;
  nblock_ = 0;
  (void)pos;
  cur_scope_ = "pre";  // INK_GPU_TS: non-visit sections get an explicit scope
  cur_phase_ = 0;
  {
    // pc.token is dead under record-once decode: the shader reads
    // prm_token out of the params header row (rewritten per token).
    PcEmb pc{uint32_t(token), uint32_t(H), float(c.rms_norm_eps), emb_flags_};
    op(p_emb_, {BW(embedw_), BW(embn_), BW(partial_), prm_head()}, &pc,
       sizeof(pc), 1, 1, "embed");
  }
  sync();
}

void GpuGraph::emit_snap() {
  const int64_t Hb = int64_t(H_) * 4;
  // prelude snapshot: pre_partial = partial; pre_block = block
  cur_scope_ = "snap";
  cp(partial_, 0, pre_partial_, 0, Hb, "snap_p");
  pre_nblock_ = nblock_;
  for (int r = 0; r < nblock_; r++)
    cp(block_, int64_t(r) * Hb, pre_block_, int64_t(r) * Hb, Hb, "snap_b");
  sync();
}

void GpuGraph::emit_bank(int p) {
  const int64_t Hb = int64_t(H_) * 4;
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

void GpuGraph::emit_fin() {
  const ModelConfig& c = *cfg_;
  const int H = H_;
  const int64_t Hb = int64_t(H) * 4;
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
  // slot 48 = the logits-ready sentinel (last command of the token cb)
  fill(readback_, VkDeviceSize(rb_sen_ + 48) * 4, 4, 49u, "sen");
}

// ------------------------------------------------------- expert-slice stage

// 32 vkCmdCopyBuffer slices (4 tensors x 8 top-k slots) for one
// host-resident layer: expert ids[s] -> slot s of each stg_*_ buffer.
// The 8 slot regions of each tensor go in ONE vkCmdCopyBuffer call —
// fewer driver packets for the same bytes. Recorded into a transfer-queue
// cb submitted right after the visit's compute cb fences — stage_ids_ was
// just written by that visit's router-id readback. Slice/offset math is
// in BYTES (vkCmdCopyBuffer), unlike BA() which takes float elements.
// part: 0 = all four tensors (single-phase), 1 = w13 pair, 2 = w2 pair.
// bnd selects the visit's private w13 buffer set when it has one — the
// buffer then holds last token's slices, and each slot resolves three
// ways:
//   - same slot, same expert: no copy at all (pf_hits)
//   - expert resident in another slot of the same buffer: a device-local
//     buf->buf copy (~10x cheaper than the PCIe pull) — emitted first, in
//     an order where every source slot still holds resident data
//     (pf_devs)
//   - cold: the host-mirror pull as before (pf_misses)
// Cross-slot reuse is what makes this pay: the router's top-8 MEMBERSHIP
// is stable token-to-token (~40%) but the ORDER shuffles, so pure
// positional hits are rare (~10%).
void GpuGraph::stage_copies(const Lw& L, int part, int bnd) {
  const int64_t E = cfg_->n_routed_experts;
  // The private w13 set only exists when this visit won VRAM — a cold
  // visit falls back to the single shared buffer (the emit side does the
  // same pick via `own`).
  const bool warm = bnd >= 0 && bnd < (int)wc_own_.size() && wc_own_[bnd];
  const struct Tn {
    const vk::Buffer* s;
    const vk::Buffer* d;
    int64_t sl;
    int want;
    const char* tag;
  } T[4] = {
      {L.w13, warm ? &w13w_own_[bnd] : &stg_w13w_, L.sl13w, 1, "stg13w"},
      {L.w13s, warm ? &w13s_own_[bnd] : &stg_w13s_, L.sl13s, 1, "stg13s"},
      {L.w2, &stg_w2w_, L.sl2w, 2, "stg2w"},
      {L.w2s, &stg_w2s_, L.sl2s, 2, "stg2s"},
  };
  // plan the w13 slots once — the same hit/dev/miss decision applies to
  // both the weight and the scale tensor (identical slot geometry).
  int devsrc[8];    // warm+want1: dev-copy source slot, -1 = host, -2 = hit
  char plan[8] = {};
  if (warm) {
    char donated[8] = {};
    for (int s = 0; s < 8; s++) {
      int64_t e = stage_ids_[s];
      if (e < 0 || e >= E) e = 0;
      stats.pf_slots++;
      if (e == int64_t(last_ids_[bnd * 8 + s])) {
        stats.pf_hits++;
        devsrc[s] = -2;
        continue;
      }
      int t = -1;
      for (int u = 0; u < 8; u++)
        if (u != s && !donated[u] &&
            last_ids_[bnd * 8 + u] == int32_t(e)) {
          t = u;
          break;
        }
      if (t >= 0) {
        donated[t] = 1;
        devsrc[s] = t;
        plan[s] = 1;
        stats.pf_devs++;
      } else {
        devsrc[s] = -1;
        stats.pf_misses++;
      }
    }
  }
  VkBufferCopy regs[8];
  for (const Tn& t : T) {
    if (t.sl <= 0 || (part != 0 && t.want != part)) continue;
    if (t.want == 1 && warm) {
      // dev copies first, threatened sources earliest: a copy may only read
      // a slot that no already-emitted copy overwrote. Whatever remains
      // cyclic (e.g. a 2-swap) downgrades to a host pull — rare and cheap.
      uint32_t nd = 0;
      char done[8] = {};
      for (int it = 0; it < 8; it++) {
        int pick = -1, fallback = -1;
        for (int s = 0; s < 8; s++) {
          if (plan[s] != 1 || done[s]) continue;
          const int ts = devsrc[s];
          if (done[ts]) continue;  // src slot already overwritten
          if (plan[ts] != 0 && !done[ts]) { pick = s; break; }
          fallback = s;
        }
        const int s = (pick >= 0) ? pick : fallback;
        if (s < 0) break;  // only dead-src devs left -> host-fallback below
        regs[nd++] = {VkDeviceSize(devsrc[s]) * VkDeviceSize(t.sl),
                      VkDeviceSize(s) * VkDeviceSize(t.sl),
                      VkDeviceSize(t.sl)};
        done[s] = 1;
      }
      if (nd) cpr(*t.d, *t.d, regs, nd, t.tag);
      uint32_t n = 0;
      for (int s = 0; s < 8; s++) {
        if (devsrc[s] == -2) continue;            // positional hit
        if (plan[s] == 1 && done[s]) continue;    // dev-copied
        int64_t e = stage_ids_[s];
        if (e < 0 || e >= E) e = 0;
        regs[n++] = {VkDeviceSize(e) * VkDeviceSize(t.sl),
                     VkDeviceSize(s) * VkDeviceSize(t.sl),
                     VkDeviceSize(t.sl)};
      }
      if (n) cpr(*t.s, *t.d, regs, n, t.tag);
      continue;
    }
    for (int s = 0; s < 8; s++) {
      int64_t e = stage_ids_[s];
      if (e < 0 || e >= E) e = 0;  // clamp: a bad id must never index OOB
      regs[s] = {VkDeviceSize(e) * VkDeviceSize(t.sl),
                 VkDeviceSize(s) * VkDeviceSize(t.sl),
                 VkDeviceSize(t.sl)};
    }
    cpr(*t.s, *t.d, regs, 8, t.tag);
  }
}


// ------------------------------------------------------------------- token
// emit_token: the BUILD walker — visits in order, each as a() then b(),
// with the mid-token sections between them. The record walk in
// forward_token produces the identical flat op() sequence (submits split
// only between a staged visit's a(v) and b(v); copies add no sites).
void GpuGraph::emit_token(int token, int64_t pos) {
  const ModelConfig& c = *cfg_;
  emit_head(token, pos);
  for (int s = 0; s < 48; s++) {
    if (s == 6)
      emit_snap();
    else if (s == 18)
      emit_bank(1);
    else if (s == 30)
      emit_bank(2);
    int i, p;
    visit_ip(c, s, i, p);
    emit_visit_a(i, p, pos);
    emit_visit_b(i, p);
  }
  emit_fin();
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

// ------------------------------------------------- record-once segments
// record_segments: the RECORD walker — same flat op() sequence as the
// build walk, but each compute command buffer is a persistent Recorder in
// segs_ that forward_token resubmits every token. A segment boundary sits
// between a staged visit's a() (which ends with the 32B router-id readback
// copy) and its b(): staged_layer_[k] names the visit whose ids the fence
// after segs_[k] makes host-visible. The ~32 slice copies per boundary are
// the ONLY commands still recorded per token (they carry the freshly read
// expert ids in their source offsets); everything else is baked.
bool GpuGraph::record_segments() {
  building_ = false;
  site_cursor_ = 0;
  const ModelConfig& c = *cfg_;
  auto open_seg = [&]() -> ink::vk::Recorder* {
    auto r = std::make_unique<ink::vk::Recorder>();
    if (!r->begin(cpool_)) {
      err_ = "segment recorder begin failed: " + r->error();
      return nullptr;
    }
    rec_ = r.get();
    segs_.push_back(std::move(r));
    return rec_;
  };
  if (!open_seg()) return false;
  if (ts_en_) {
    // slot 0 = cb0 start; slots 1.. = one mark after each recorded op.
    rec_->reset_query_pool(ts_pool_, 0, ts_cap_);
    rec_->write_timestamp(ts_pool_, 0);
    ts_next_ = 1;
  }
  emit_head(0, 0);
  for (int s = 0; s < 48; s++) {
    if (s == 6)
      emit_snap();
    else if (s == 18)
      emit_bank(1);
    else if (s == 30)
      emit_bank(2);
    int i, p;
    visit_ip(c, s, i, p);
    emit_visit_a(i, p, 0);
    if (!lw_[i].experts_dev && stage_ok_) {
      // Staged visit: close the segment right after a(v)'s router-id copy;
      // the next segment opens with b(v) and waits (device-side, via the
      // timeline semaphore) on this visit's slice copies.
      if (!rec_->end()) {
        err_ = "segment end failed";
        return false;
      }
      rec_ = nullptr;
      if (!open_seg()) return false;
      ts_gap();  // cb-start mark; segment = the inter-submit gap
      if (split_stg_) {
        // Two-phase staging: b(v) splits at moe_accum — the w13 phase is
        // its own (unfenced) segment gated on the w13 transfer submit;
        // the w2 copies stream on the SDMA queue while it runs.
        emit_visit_b(i, p, 1);
        if (!rec_->end()) {
          err_ = "w13 segment end failed";
          return false;
        }
        rec_ = nullptr;
        if (!open_seg()) return false;
        ts_gap();
        emit_visit_b(i, p, 2);
      } else {
        emit_visit_b(i, p, 0);
      }
    } else {
      emit_visit_b(i, p, 0);
    }
  }
  emit_fin();
  if (!rec_->end()) {
    err_ = "final segment end failed";
    return false;
  }
  rec_ = nullptr;
  const size_t nst = staged_layer_.size();
  trec13_.resize(nst);
  trec2_.resize(nst);
  for (auto& t : trec13_) t = std::make_unique<ink::vk::Recorder>();
  for (auto& t : trec2_) t = std::make_unique<ink::vk::Recorder>();
  // per-token counter/stat deltas: the emit helpers already counted the
  // recorded ops once; replay adds them to stats per forward_token.
  rec_disp_ = stats.dispatches;
  rec_cp_ = stats.copies;
  rec_bar_ = stats.barriers;
  return err_.empty();
}

// write_params: the ONLY per-token data the baked command buffers need —
// one memcpy of the 49 params rows, then the host-side counters advance
// exactly as emit_visit_b's bookkeeping did per visit.
void GpuGraph::write_params(int token, int64_t pos) {
  const ModelConfig& c = *cfg_;
  // Sentinels back to zero: the recorded fills re-arm them as each
  // boundary's last command executes (cleared while idle — tokens are
  // serial, so the previous cb is done before this runs).
  std::memset(rb_ + rb_sen_, 0, 64 * 4);
  const double tau =
      1.0 + c.log_scaling_alpha *
                std::log(
                    std::max(double(pos + 1) / c.log_scaling_n_floor, 1.0));
  sh::GpuParamsRow& hdr = *prm_row(48);
  hdr.token = uint32_t(token);
  hdr.pos = uint32_t(pos);
  hdr.tau = float(tau);
  for (int s = 0; s < 48; s++) {
    sh::GpuParamsRow& r = *prm_row(s);
    const int64_t len = kv_len_[s] + 1;
    const int64_t j0 = slot_full_[s]
                           ? 0
                           : std::max<int64_t>(
                                 0, len - c.sliding_window_size);
    r.kv_len = uint32_t(kv_len_[s]);
    r.j0 = uint32_t(j0);
    r.n = uint32_t(len - j0);
    r.pos = uint32_t(pos);
    for (int st = 0; st < 4; st++)
      r.filled[st] = uint32_t(conv_filled_[s][st]);
  }
  alloc_->flush(params_);  // no-op on coherent memory
  for (int s = 0; s < 48; s++) {
    kv_len_[s]++;
    for (int st = 0; st < 4; st++)
      conv_filled_[s][st] = std::min(3, conv_filled_[s][st] + 1);
  }
}

const std::vector<float>& GpuGraph::forward_token(int token, int64_t pos) {
  const auto t0 = clk::steady_clock::now();
  const double wait0 = stats.wait_ms;
  bool fail = false;
  write_params(token, pos);
  stats.dispatches += rec_disp_;
  stats.copies += rec_cp_;
  stats.barriers += rec_bar_;

  // Replay: each fenced segment ends with staged visit k's router-id
  // readback; the fence makes it host-visible, the host programs the
  // slice copies on the transfer queue, and the next compute segment
  // waits on the transfer signal on device (timeline semaphore — no
  // host-side copy wait). Dev-resident visits never split a segment.
  //
  // split_stg_: segs_ layout is [S0, A0, B0, A1, B1, ...] — the fenced
  // segs sit at even indices (segs_[2k] ends with boundary k's ids), the
  // odd segs_[2k+1] hold boundary k's w13 GEMV phase gated on the w13
  // transfer; the w2 transfer streams while that w13 segment runs.
  const size_t nst = staged_layer_.size();
  const auto seg_i = [&](size_t k) { return split_stg_ ? 2 * k : k; };
  auto do_trec = [&](size_t k, int part, uint64_t* out_v) -> bool {
    ink::vk::Recorder* tr =
        (part == 1 ? trec13_ : trec2_)[k].get();
    if (!tr->begin(tpool_)) {
      err_ = "transfer recorder begin failed: " + tr->error();
      return false;
    }
    rec_ = tr;
    ts_off_ = true;  // no timestamps on the transfer queue
    stage_copies(lw_[staged_layer_[k]], part, int(k));
    ts_off_ = false;
    rec_ = nullptr;
    if (!tr->end()) {
      err_ = "transfer recorder end failed";
      return false;
    }
    if (tl_en_) {
      const uint64_t v = ++tl_val_;
      if (!ctx_->submit_timeline(ctx_->copy_queue(), tr->cb(), nullptr,
                                 nullptr, 0, &tl_sem_, &v, 1,
                                 VK_NULL_HANDLE)) {
        err_ = "stage copy submit failed: " + ctx_->last_error;
        return false;
      }
      if (out_v) *out_v = v;
    } else if (!tr->submit(ctx_->copy_queue())) {
      err_ = "stage copy submit failed: " + tr->error();
      return false;
    }
    stats.submits++;
    return true;
  };
  uint64_t tl_a = 0, tl_b = 0;
  for (size_t k = 0; k <= nst && !fail; k++) {
    if (!tl_en_ && k > 0) {
      // No timeline semaphores: keep the old host-side ordering — wait the
      // staged copies before the dependent compute submit.
      if (!trec13_[k - 1]->wait()) {
        err_ = "stage copy wait failed";
        fail = true;
        break;
      }
    }
    const auto w0 = clk::steady_clock::now();
    const uint64_t wv = tl_en_ && k > 0 ? (split_stg_ ? tl_b : tl_val_) : 0;
    const VkSemaphore* ws = (tl_en_ && k > 0) ? &tl_sem_ : nullptr;
    const VkFence f = seg_fence_;
    if (!ctx_->submit_timeline(ctx_->compute_queue, segs_[seg_i(k)]->cb(),
                               ws, &wv, ws ? 1u : 0u, nullptr, nullptr, 0,
                               f)) {
      err_ = "token submit failed: " + ctx_->last_error;
      fail = true;
      break;
    }
    stats.submits++;
    {
      // The sentinel fill is the cb's last command: a host spin on the
      // mapped dword unblocks a wake-latency earlier than any fence would
      // (and is the ONLY readiness check for the boundary segments).
      const int sen_idx = (k == nst) ? 48 : staged_vidx_[k];
      volatile uint32_t* sen = (volatile uint32_t*)(rb_ + rb_sen_);
      const uint32_t want = uint32_t(sen_idx + 1);
      const auto dl =
          clk::steady_clock::now() + std::chrono::milliseconds(250);
      while (sen[sen_idx] != want)
        if (clk::steady_clock::now() > dl) break;
    }
    if (!ctx_->wait_fence(seg_fence_)) {
      err_ = "token wait failed: " + ctx_->last_error;
      fail = true;
      break;
    }
    vkResetFences(ctx_->device, 1, &seg_fence_);
    stats.wait_ms +=
        clk::duration<double, std::milli>(clk::steady_clock::now() - w0)
            .count();
    if (k == nst) break;
    // The staged visit's ids are final: program its slice copies.
    alloc_->invalidate(readback_, VkDeviceSize(rb_ids_) * 4, 32);
    std::memcpy(stage_ids_, rb_ + rb_ids_, 32);
    if (split_stg_) {
      // w13 copies -> tl_a, then the w13-phase segment (unfenced — the
      // compute queue runs it while the w2 copies stream behind it).
      if (!do_trec(k, 1, &tl_a)) { fail = true; break; }
      if (!ctx_->submit_timeline(ctx_->compute_queue,
                                 segs_[2 * k + 1]->cb(), &tl_sem_, &tl_a, 1,
                                 nullptr, nullptr, 0, VK_NULL_HANDLE)) {
        err_ = "w13 phase submit failed: " + ctx_->last_error;
        fail = true;
        break;
      }
      stats.submits++;
      // w2 copies -> tl_b; segs_[2k+2] (submitted next iteration) waits it.
      if (!do_trec(k, 2, &tl_b)) { fail = true; break; }
    } else {
      if (!do_trec(k, 0, nullptr)) { fail = true; break; }
    }
    // Boundary k's slots now hold the fresh ids: commit them as the warm
    // state for next token's hit test (AFTER the miss/dev list was
    // recorded — the compare reads last_ids_). The private buffer keeps
    // them resident until this visit's next boundary, no prefetch needed.
    if (!last_ids_.empty())
      std::memcpy(&last_ids_[k * 8], stage_ids_, 32);
  }
  if (!fail) {
    // The final compute fence implies every transfer signal was consumed,
    // so this token's copy cbs are finished — free them before next begin.
    for (auto& tr : trec13_)
      if (tr->cb()) tr->shutdown();
    for (auto& tr : trec2_)
      if (tr->cb()) tr->shutdown();
  }
  stats.record_ms +=
      clk::duration<double, std::milli>(clk::steady_clock::now() - t0)
          .count() -
      (stats.wait_ms - wait0);
  if (fail) return logits_host_;
  if (ts_en_) ts_collect();
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
  rb_ids_ = rb_dbg_ + 16 * H;  // 8 int32 router top-k ids (expert staging)
  rb_sen_ = rb_ids_ + 8;       // 64 sentinel dwords for the ids-ready spins
  readback_ = alloc.alloc_host((rb_sen_ + 64) * 4);
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
      load_pipe("conv4_static", 5, sizeof(sh::PcConv4Static), p_cvs_) &&
      load_pipe("conv4_dynamic", 6, sizeof(sh::PcConv4Dyn), p_cvd_) &&
      load_pipe("rel_bias", 3, sizeof(sh::PcRelBias), p_rel_) &&
      load_pipe("attn_scores", 5, sizeof(sh::PcAttnScores), p_sco_) &&
      load_pipe("attn_softmax", 3, sizeof(sh::PcAttnSoftmax), p_smx_) &&
      load_pipe("attn_accum", 4, sizeof(sh::PcAttnAccum), p_acc_) &&
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
      load_pipe("log_scale", 2, sizeof(PcTau), p_log_) &&
      load_pipe("embed_norm", 4, sizeof(PcEmb), p_emb_) &&
      load_pipe("kv_append", 5, sizeof(PcN), p_kva_) &&
      load_pipe("matvec_nvfp4_slot", 7, sizeof(sh::PcNvfp4), p_nv4s_) &&
      load_pipe("matvec_mxfp8_slot", 5, sizeof(sh::PcMxfp8), p_mx8s_) &&
      load_pipe("matvec_nvfp4_stage", 7, sizeof(sh::PcNvfp4), p_nv4st_) &&
      load_pipe("matvec_mxfp8_stage", 5, sizeof(sh::PcMxfp8), p_mx8st_) &&
      load_pipe("moe_accum_stage", 9, sizeof(sh::PcMoeAccum), p_moest_);
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
    const TensorView* t13s =
        w.find(lname(i, "mlp.experts.w13_weight_scale"));
    const TensorView* t2 = w.find(lname(i, "mlp.experts.w2_weight"));
    const TensorView* t2s = w.find(lname(i, "mlp.experts.w2_weight_scale"));
    if (t13 && t13->nd == 3) L.w13R = t13->shape[1];
    if (t2 && t2->nd == 3) L.w2R = t2->shape[1];
    // per-expert slice bytes (contiguous [E,R,packedK] => nbytes/E) and the
    // residency verdict: all four big expert tensors device-local -> mslot
    // reads them directly; anything host-mirrored -> staged via stg_*_.
    auto slice_bytes = [](const TensorView* t) -> int64_t {
      return (t && t->nd == 3 && t->shape[0] > 0)
                 ? t->nbytes() / t->shape[0]
                 : int64_t(0);
    };
    L.sl13w = slice_bytes(t13);
    L.sl13s = slice_bytes(t13s);
    L.sl2w = slice_bytes(t2);
    L.sl2s = slice_bytes(t2s);
    L.experts_dev = t13 && t13s && t2 && t2s && devw_.count(t13->data) &&
                    devw_.count(t13s->data) && devw_.count(t2->data) &&
                    devw_.count(t2s->data);
  }
  // dev-local expert-slice staging (only when some layer is host-resident):
  // 8 top-k slots x the largest per-expert slice of each tensor.
  {
    int64_t m13w = 0, m13s = 0, m2w = 0, m2s = 0;
    for (int i = 0; i < nl; i++) {
      const Lw& L = lw_[i];
      if (L.experts_dev) continue;
      m13w = std::max(m13w, L.sl13w);
      m13s = std::max(m13s, L.sl13s);
      m2w = std::max(m2w, L.sl2w);
      m2s = std::max(m2s, L.sl2s);
    }
    if (m13w > 0) {
      const std::string e0 = err_;  // staging is optional — restore on fail
      stg_w13w_ = dev(8 * m13w);
      stg_w13s_ = dev(8 * m13s);
      stg_w2w_ = dev(8 * m2w);
      stg_w2s_ = dev(8 * m2s);
      stage_ok_ = stg_w13w_.buf && stg_w13s_.buf && stg_w2w_.buf &&
                  stg_w2s_.buf;
      if (!stage_ok_) {
        err_ = e0;
        // VRAM is full: keep the host-read mslot path for these layers.
        std::fprintf(stderr,
                     "gpu backend: expert staging alloc failed (%.0fMB; "
                     "host-resident path kept): %s\n",
                     8.0 * (m13w + m13s + m2w + m2s) / (1 << 20),
                     alloc.last_error.c_str());
        for (vk::Buffer* b : {&stg_w13w_, &stg_w13s_, &stg_w2w_, &stg_w2s_})
          if (b->buf) alloc.free(*b);
      }
      // Staged-boundary tables, once, here: emit_visit_b's buffer pick
      // (build walk) and forward_token's per-boundary bookkeeping both read
      // them, so they must exist before emit_token below — record_segments
      // only consults staged_layer_/staged_vidx_ for the boundary splits.
      for (int s = 0; s < 48; s++) stg_bmap_[s] = -1;
      for (int s = 0; s < 48; s++) {
        int i, p;
        visit_ip(c, s, i, p);
        if (!lw_[i].experts_dev && stage_ok_) {
          stg_bmap_[c.visit_index(i, p)] = int(staged_layer_.size());
          staged_layer_.push_back(i);
          staged_vidx_.push_back(c.visit_index(i, p));
        }
      }
      const size_t nst0 = staged_layer_.size();
      last_ids_.assign(nst0 * 8, int32_t(-1));
      // Per-visit warm sets, greedily: a visit with a private 8-slot
      // w13w/w13s pair keeps last token's slices resident between tokens —
      // no prefetch traffic, ~40% of slots then resolve without a PCIe
      // pull. Failures just leave that visit on the shared buffer.
      w13w_own_.resize(nst0);
      w13s_own_.resize(nst0);
      wc_own_.assign(nst0, 0);
      if (std::getenv("INK_NOWARM") == nullptr) {
        int nw = 0;
        for (size_t b = 0; b < nst0; b++) {
          const Lw& L = lw_[staged_layer_[b]];
          const std::string e_warm = err_;
          w13w_own_[b] = dev(8 * L.sl13w);
          w13s_own_[b] = dev(8 * L.sl13s);
          wc_own_[b] = w13w_own_[b].buf && w13s_own_[b].buf;
          if (!wc_own_[b]) {
            err_ = e_warm;
            if (w13w_own_[b].buf) alloc.free(w13w_own_[b]);
            if (w13s_own_[b].buf) alloc.free(w13s_own_[b]);
          } else nw++;
        }
        std::fprintf(stderr, "gpu backend: warm w13 caches for %d/%d "
                     "staged visits\n", nw, int(nst0));
      }
    }
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

  // --------------------- record-once replay state ---------------------
  // params_: host-coherent GpuParamsRow[49] bound as the last binding of
  // every shader that needs a token-varying scalar (see bindings.h). Try
  // the ReBAR device-local host-visible heap first, plain host staging
  // memory as fallback.
  prm_stride_ =
      std::max<VkDeviceSize>(64, ctx.props.limits
                                     .minStorageBufferOffsetAlignment);
  params_ = alloc.alloc_bar_visible(
      VkDeviceSize(sh::GPU_PARAMS_ROWS) * prm_stride_);
  if (!params_.buf)
    params_ =
        alloc.alloc_host(VkDeviceSize(sh::GPU_PARAMS_ROWS) * prm_stride_);
  if (!params_.buf || !(prm_ = alloc.mapped(params_))) {
    *err = "gpu backend: params alloc failed: " + alloc.last_error;
    return false;
  }
  std::memset(prm_, 0, VkDeviceSize(sh::GPU_PARAMS_ROWS) * prm_stride_);
  // cache slot -> layer kind (attn window + slot->full for j0/n)
  for (int s = 0; s < 48; s++) {
    int i, p;
    visit_ip(c, s, i, p);
    slot_full_[c.cache_slot(i, p)] = c.is_full_attn(i) ? 1 : 0;
  }
  pstride_ = uint32_t(cap_ + 8);   // fixed probs row stride
  score_gy_ = wg64(cap_);          // attn_scores grid covers any n <= cap
  seg_fence_ = ctx_->create_fence();
  tl_en_ = ctx_->feats.timeline_semaphore;
  if (tl_en_) tl_sem_ = ctx_->create_timeline_semaphore(0);
  // Two-phase staging only pays off when the slice copies can truly run
  // while compute executes: dedicated SDMA queue + device-side ordering.
  split_stg_ = tl_en_ && ctx_->has_dedicated_transfer();
  if (!seg_fence_ || (tl_en_ && !tl_sem_)) {
    *err = "gpu backend: fence/semaphore create failed: " +
           ctx_->last_error;
    return false;
  }

  logits_host_.assign(size_t(V_), 0.f);
  building_ = true;
  emit_token(0, 0);
  building_ = false;
  if (!err_.empty()) {
    *err = "gpu backend: descriptor build failed: " + err_;
    return false;
  }
  // Record walk: persistent per-segment cbs resubmitted every token.
  if (!record_segments()) {
    *err = "gpu backend: segment record failed: " + err_;
    return false;
  }
  reset();
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
  for (auto& s : segs_) s->shutdown();
  segs_.clear();
  for (auto& t : trec13_) t->shutdown();
  for (auto& t : trec2_) t->shutdown();
  trec13_.clear();
  trec2_.clear();
  last_ids_.clear();
  staged_layer_.clear();
  staged_vidx_.clear();
  if (seg_fence_) ctx_->destroy_fence(seg_fence_);
  seg_fence_ = VK_NULL_HANDLE;
  if (tl_sem_) ctx_->destroy_semaphore(tl_sem_);
  tl_sem_ = VK_NULL_HANDLE;
  vk::Buffer* bufs[] = {&staging_, &dummy4_, &partial_, &block_,
                        &pre_partial_, &pre_block_, &bank_, &reads_, &msgs_,
                        &mgated_, &inbuf_, &q_, &k_, &v_, &r_, &tk_, &tv_,
                        &rel_, &probs_, &grate_, &sink_, &o_, &attn_out_,
                        &a_out_, &residual_, &m_in_, &mn_, &moe_out_, &mo_,
                        &ridx_, &rw_, &gate_, &up_, &kvk_, &kvv_, &hist_, &h_,
                        &hfin_, &logits_, &readback_, &params_, &embedw_,
                        &embn_,
                        &normw_, &unembw_, &attnq_, &mlpq_, &msgq_, &outq_,
                        &stg_w13w_, &stg_w13s_, &stg_w2w_, &stg_w2s_};
  for (auto* b : bufs)
    if (b->buf) alloc_->free(*b);
  for (auto& b : w13w_own_)
    if (b.buf) alloc_->free(b);
  for (auto& b : w13s_own_)
    if (b.buf) alloc_->free(b);
  w13w_own_.clear();
  w13s_own_.clear();
  wc_own_.clear();
  Pipe* pipes[] = {&p_rms_, &p_mvf_, &p_mvb_, &p_mv8_, &p_cvs_, &p_cvd_,
                   &p_rel_, &p_sco_, &p_smx_, &p_acc_, &p_ares_, &p_msg_,
                   &p_rtr_, &p_moe_, &p_mred_, &p_moem_, &p_nv4m_, &p_mx8m_,
                   &p_sig_, &p_cap_, &p_add_, &p_l2_,
                   &p_log_, &p_emb_, &p_nv4s_, &p_mx8s_,
                   &p_nv4st_, &p_mx8st_, &p_moest_, &p_kva_};
  for (auto* p : pipes) p->p.shutdown();
  cpool_.shutdown();
  tpool_.shutdown();
  ok_ = false;
  ctx_ = nullptr;
}

}  // namespace ink::gpu
