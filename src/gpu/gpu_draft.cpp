// gpu_draft.cpp — GPU MTP/DFlare speculative draft head (research/07 §9).
//
// One block = 8 positions (anchor at its own pos + 7 mask slots at
// pos+1..pos+7 — DFlash fill-in layout). Per propose():
//   1. fused ctx per consumer: softmax(fuse_weight[c]) . taps -> dflare_norm
//   2. per layer: ctx-KV append (kv_proj_ctx -> k_norm_ctx -> rope at the
//      anchor's own position) and the block layer (prenorm GQA attention
//      over [ctx + block] KV with q/k norm + rope, prenorm SwiGLU MLP,
//      bf16 residual adds)
//   3. norm + shared unembed -> base logits [8, V]
//   4. markov head: 7 sequential GRU steps chained on device (argmax writes
//      the next prev id into host-visible did_ cells). Step i consumes
//      slot i+1's hidden/base — each mask slot predicts the token at its
//      OWN position, so did[0..6] cover pos+1..pos+7. The fold is
//      dvec = hidden_out @ latent + hidden_out.bias [2304], then logits[v] =
//      softcap30(bf16(bf16(base) + unembed[v] . dvec)) — no [V,rank] table.
//   5. spare 8th id = argmax over the slot-7 base logits (unscored).
//
// Correctness-first like GpuGraph: a full barrier after every op, weights
// device-local, per-propose command buffer (push constants carry pos/n_ctx).
// The CPU reference (ops.cpp CpuDraft) arbitrates.

#include "gpu/gpu_draft.h"

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
constexpr uint32_t GRP = 64;
inline uint32_t wg64(int64_t n) { return uint32_t((n + GRP - 1) / GRP); }
struct PcN {
  uint32_t n;
};
struct PcPart {
  uint32_t n, chunk;
};
struct PcFin {
  uint32_t p;
};
constexpr int64_t kCtxMax = 4224;  // draft_attn shared cap minus 8 block keys

// fp32 decode of a (bf16) tensor row/whole tensor into a host vector.
void dec_f32(const TensorView& t, float* out, int64_t n) {
  for (int64_t i = 0; i < n; i++) out[i] = t.f32_linear(i);
}
}  // namespace

static GpuDrafter::Bnd BW(const ink::vk::Buffer& b) { return {&b, 0, 0}; }
static GpuDrafter::Bnd BA(const ink::vk::Buffer& b, int64_t fe, int64_t n) {
  return {&b, VkDeviceSize(fe) * 4, VkDeviceSize(n) * 4};
}

bool GpuDrafter::load_pipe(const char* name, uint32_t bindings,
                           uint32_t pc_bytes, Pipe& out) {
  vk::PipelineDesc d;
  d.spv_path = shader_dir_ + "/" + name + ".spv";
  d.storage_bindings = bindings;
  d.push_constant_bytes = pc_bytes;
  out.name = name;
  return vk::load_compute_pipeline(*ctx_, d, out.p, err_);
}

ink::vk::Buffer GpuDrafter::upload_dev(const void* src, VkDeviceSize bytes) {
  vk::Buffer dst = alloc_->alloc_device_local(bytes);
  if (!dst.buf) {
    // Host-resident fallback (GpuGraph policy): the weights mirror stays
    // host-visible and the kernels stream it over PCIe. Try the ReBAR
    // DEVICE_LOCAL|HOST_VISIBLE heap first, then plain host memory.
    dst = alloc_->alloc_bar_visible(bytes);
    if (!dst.buf) dst = alloc_->alloc_host(bytes);
    if (!dst.buf) {
      std::fprintf(stderr,
                   "gpu draft: upload %zuMB fallback failed: %s\n",
                   size_t(bytes >> 20), alloc_->last_error.c_str());
      if (err_.empty())
        err_ = "upload alloc failed: " + alloc_->last_error;
      return {};
    }
    void* mp = alloc_->mapped(dst);
    if (!mp) {
      alloc_->free(dst);
      err_ = "upload fallback buffer not mapped";
      return {};
    }
    std::memcpy(mp, src, size_t(bytes));
    alloc_->flush(dst);
    stats.uploads++;
    stats.upload_bytes += bytes;
    return dst;
  }
  const VkDeviceSize CH = 16ull << 20;  // smaller chunks; BAR heap is tight
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

const ink::vk::Buffer* GpuDrafter::dev_w(const TensorView& t) {
  if (!t.data) return nullptr;
  auto* b = new ink::vk::Buffer;
  *b = upload_dev(t.data, t.nbytes());
  if (!b->buf) {
    delete b;
    return nullptr;
  }
  return b;
}

const ink::vk::Buffer* GpuDrafter::dev_f32(const TensorView& t) {
  if (!t.data) return nullptr;
  const int64_t n = t.numel();
  std::vector<float> tmp = std::vector<float>(size_t(n));
  for (int64_t i = 0; i < n; i++) tmp[size_t(i)] = t.f32_linear(i);
  auto* b = new ink::vk::Buffer;
  *b = upload_dev(tmp.data(), VkDeviceSize(n) * 4);
  if (!b->buf) {
    delete b;
    return nullptr;
  }
  return b;
}

// ------------------------------------------------------------ emit helpers

const ink::vk::DescSet& GpuDrafter::dset(
    const Pipe& p, std::initializer_list<Bnd> bs) {
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

void GpuDrafter::op(const Pipe& p, std::initializer_list<Bnd> bs,
                    const void* pc, uint32_t pcb, uint32_t gx, uint32_t gy,
                    uint32_t gz) {
  const vk::DescSet& ds = dset(p, bs);
  if (building_) return;
  rec_->bind_pipeline(p.p.pipe);
  rec_->bind_descset(p.p.layout, ds.set);
  if (pcb) rec_->push_constants(p.p.layout, pc, pcb);
  rec_->dispatch(gx, gy, gz);
  stats.dispatches++;
}

void GpuDrafter::mv(Bnd w, Bnd x, Bnd o, int64_t M, int64_t K) {
  const vk::DescSet& ds = dset(p_mvb_, {w, x, o});
  if (building_) return;
  rec_->bind_pipeline(p_mvb_.p.pipe);
  rec_->bind_descset(p_mvb_.p.layout, ds.set);
  for (int64_t m0 = 0; m0 < M; m0 += 65535) {
    sh::PcMatvec pc{uint32_t(K), uint32_t(m0)};
    rec_->push_constants(p_mvb_.p.layout, &pc, sizeof(pc));
    rec_->dispatch(uint32_t(std::min<int64_t>(M - m0, 65535)));
    stats.dispatches++;
  }
}

void GpuDrafter::mv_seq(Bnd w, Bnd x, Bnd o, int64_t M, int64_t K, int64_t xs,
                        int64_t Mstride, int t) {
  const vk::DescSet& ds = dset(p_mvs_, {w, x, o});
  if (building_) return;
  rec_->bind_pipeline(p_mvs_.p.pipe);
  rec_->bind_descset(p_mvs_.p.layout, ds.set);
  for (int64_t m0 = 0; m0 < M; m0 += 65535) {
    sh::PcMatvecSeq pc{uint32_t(K), uint32_t(m0), uint32_t(xs),
                       uint32_t(Mstride)};
    rec_->push_constants(p_mvs_.p.layout, &pc, sizeof(pc));
    rec_->dispatch(uint32_t(std::min<int64_t>(M - m0, 65535)), uint32_t(t));
    stats.dispatches++;
  }
}

void GpuDrafter::cp(const ink::vk::Buffer& s, VkDeviceSize so,
                    const ink::vk::Buffer& d, VkDeviceSize dof,
                    VkDeviceSize sz) {
  if (building_) return;
  rec_->copy_buffer(s.buf, d.buf, sz, so, dof);
  stats.copies++;
}

void GpuDrafter::fill(const ink::vk::Buffer& d, VkDeviceSize off,
                      VkDeviceSize sz, uint32_t data) {
  if (building_) return;
  rec_->fill_buffer(d.buf, off, sz, data);
}

void GpuDrafter::sync() {
  if (building_) return;
  rec_->barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
  stats.barriers++;
}

// ------------------------------------------------------------- emit block

void GpuDrafter::emit_block(int64_t pos, int64_t n_ctx, bool appending) {
  const int64_t H = H_;
  const float eps = float(cfg_->mtp.rms_norm_eps);
  const uint32_t nctx = uint32_t(appending ? n_ctx + 1 : n_ctx);
  const int64_t dst_row = appending ? n_ctx : 0;
  const VkDeviceSize cp_rows = appending ? 4096 : 0;

  // staged inputs -> device
  cp(staging_, 0, x_, 0, VkDeviceSize(8 * H) * 4);
  cp(staging_, VkDeviceSize(8 * H) * 4, taps_, 0, VkDeviceSize(4 * H) * 4);
  if (dbg_en_) {  // staged block input
    cp(x_, 0, dbg_, VkDeviceSize(36 * H + 24576) * 4, VkDeviceSize(8 * H) * 4);
  }
  sync();  // staging writes visible before the mix reads taps_
  {
    sh::PcDraftMix pc{uint32_t(H), eps};
    op(p_mix_, {BW(taps_), BW(alpha_), BW(dgam_), BW(ctxb_)}, &pc, sizeof(pc),
       4);
  }
  sync();

  for (int i = 0; i < 3; i++) {
    // ---- ctx-KV append for this layer (anchor's fused features) ----
    mv(BW(wkc_[i]), BA(ctxb_, i * H, H), BW(kscr_), 1024, H);
    mv(BW(wvc_[i]), BA(ctxb_, i * H, H), BW(vscr_), 1024, H);
    sync();
    {
      sh::PcRmsnorm pc{128u, eps, sh::F_USE_GAMMA};
      op(p_rms_, {BW(kscr_), BA(knc_, i * 128, 128), BW(kscr_)}, &pc,
         sizeof(pc), 8);
    }
    sync();  // norm writes visible before rope reads
    {
      sh::PcDraftRope pc{8u, uint32_t(pos), float(cfg_->mtp.rope_theta)};
      op(p_rope_, {BW(kscr_)}, &pc, sizeof(pc), 8);
    }
    sync();
    if (dbg_en_ && i == 0)  // ctx key row for this anchor (norm+rope done)
      cp(kscr_, 0, dbg_, VkDeviceSize(52 * H + 57344) * 4, 4096);
    if (cp_rows) {  // vkCmdCopyBuffer regions must be non-empty
      cp(kscr_, 0, ctxk_, VkDeviceSize(i * ctxcap_ + dst_row) * 4096,
         cp_rows);
      cp(vscr_, 0, ctxv_, VkDeviceSize(i * ctxcap_ + dst_row) * 4096,
         cp_rows);
    }

    // ---- block layer i ----
    {
      sh::PcRmsnormSeq pc{uint32_t(H), eps, sh::F_USE_GAMMA, uint32_t(H),
                          uint32_t(H)};
      op(p_rmss_, {BW(x_), BA(inln_, i * H, H), BW(xn_)}, &pc, sizeof(pc), 8);
    }
    sync();  // prenorm writes visible before the qkv matvecs read xn_
    mv_seq(BW(wq_[i]), BW(xn_), BW(q8_), 3072, H, H, 3072, 8);
    mv_seq(BW(wk_[i]), BW(xn_), BW(kb8_), 1024, H, H, 1024, 8);
    mv_seq(BW(wv_[i]), BW(xn_), BW(vb8_), 1024, H, H, 1024, 8);
    sync();
    if (dbg_en_ && i == 0)
      cp(xn_, 0, dbg_, VkDeviceSize(44 * H + 24576) * 4,
         VkDeviceSize(8 * H) * 4);  // layer-0 prenorm (post-barrier)
    {
      sh::PcRmsnormSeq pc{128u, eps, sh::F_USE_GAMMA, 128u, 128u};
      op(p_rmss_, {BW(q8_), BA(qn_, i * 128, 128), BW(q8_)}, &pc, sizeof(pc),
         192);
      op(p_rmss_, {BW(kb8_), BA(kn_, i * 128, 128), BW(kb8_)}, &pc,
         sizeof(pc), 64);
    }
    sync();  // q/k norm writes visible before rope reads
    {
      sh::PcDraftRope pq{24u, uint32_t(pos), float(cfg_->mtp.rope_theta)};
      op(p_rope_, {BW(q8_)}, &pq, sizeof(pq), 192);
      sh::PcDraftRope pk{8u, uint32_t(pos), float(cfg_->mtp.rope_theta)};
      op(p_rope_, {BW(kb8_)}, &pk, sizeof(pk), 64);
    }
    sync();
    if (dbg_en_ && i == 0) {  // q0/k0 post norm+rope
      cp(q8_, 0, dbg_, VkDeviceSize(52 * H + 24576) * 4,
         VkDeviceSize(8 * 3072) * 4);
      cp(kb8_, 0, dbg_, VkDeviceSize(52 * H + 49152) * 4,
         VkDeviceSize(8 * 1024) * 4);
    }
    {
      const auto& ll = cfg_->mtp.local_layer_ids;
      const bool local =
          std::find(ll.begin(), ll.end(), i) != ll.end();
      sh::PcDraftAttn pc{nctx, uint32_t(pos),
                       local ? uint32_t(cfg_->mtp.sliding_window_size) : 0u,
                       float(1.0 / std::sqrt(128.0))};
      op(p_attn_,
         {BW(q8_), BA(ctxk_, i * ctxcap_ * 1024, ctxcap_ * 1024),
          BA(ctxv_, i * ctxcap_ * 1024, ctxcap_ * 1024), BW(kb8_), BW(vb8_),
          BW(ctxpos_), BW(att8_)},
         &pc, sizeof(pc), 24, 8);
    }
    sync();
    if (dbg_en_ && i == 0)
      cp(att8_, 0, dbg_, VkDeviceSize(36 * H) * 4,
         VkDeviceSize(8 * 3072) * 4);  // layer-0 attn output
    mv_seq(BW(wo_[i]), BW(att8_), BW(ao8_), H, 3072, 3072, H, 8);
    sync();
    {
      PcN pc{uint32_t(8 * H)};
      op(p_resid_, {BW(x_), BW(ao8_), BW(x_)}, &pc, sizeof(pc), wg64(8 * H));
    }
    sync();  // resid writes visible before postnorm reads x_
    {
      sh::PcRmsnormSeq pc{uint32_t(H), eps, sh::F_USE_GAMMA, uint32_t(H),
                          uint32_t(H)};
      op(p_rmss_, {BW(x_), BA(pln_, i * H, H), BW(xn_)}, &pc, sizeof(pc), 8);
    }
    sync();  // postnorm writes visible before gate/up matvecs read xn_
    mv_seq(BW(wg_[i]), BW(xn_), BW(g8_), 9216, H, H, 9216, 8);
    mv_seq(BW(wu_[i]), BW(xn_), BW(u8_), 9216, H, H, 9216, 8);
    sync();
    {
      PcN pc{uint32_t(8 * 9216)};
      op(p_swig_, {BW(g8_), BW(u8_), BW(a8_)}, &pc, sizeof(pc),
         wg64(8 * 9216));
    }
    sync();  // swiglu writes visible before down-proj reads a8_
    mv_seq(BW(wd_[i]), BW(a8_), BW(mo8_), H, 9216, 9216, H, 8);
    sync();
    {
      PcN pc{uint32_t(8 * H)};
      op(p_resid_, {BW(x_), BW(mo8_), BW(x_)}, &pc, sizeof(pc), wg64(8 * H));
    }
    sync();
    if (dbg_en_)
      cp(x_, 0, dbg_, VkDeviceSize(12 + i * 8) * H * 4,
         VkDeviceSize(8 * H) * 4);  // residual after layer i
  }

  // ---- final norm + base logits ----
  {
    sh::PcRmsnormSeq pc{uint32_t(H), eps, sh::F_USE_GAMMA, uint32_t(H),
                        uint32_t(H)};
    op(p_rmss_, {BW(x_), BW(normw_), BW(out8_)}, &pc, sizeof(pc), 8);
  }
  sync();  // out8 writes visible before the unembed matvec reads it
  mv_seq(BW(unemb_), BW(out8_), BW(base_), V_, H, H, VSTR_, 8);
  sync();

  // ---- markov head: 7 sequential steps, chained on device ----
  fill(state_, 0, 256 * 4, 0);
  sync();  // fill visible before pack reads state_ (iter 0)
  for (int i = 0; i < 7; i++) {
    {
      sh::PcDraftPack pc{uint32_t(i)};
      op(p_pack_, {BW(state_), BW(w1_), BW(out8_), BW(ctxb_), BW(did_),
                   BW(in_)},
         &pc, sizeof(pc), 80);
    }
    sync();
    mv(BW(rnn_), BW(in_), BW(proj_), 768, 5120);
    sync();
    {
      PcN pc{0u};  // fixed geometry
      op(p_gru_, {BW(proj_), BW(rnn_b_), BW(state_), BW(lat_)}, &pc, 0, 4);
    }
    sync();
    // dvec = hidden_out @ lat + hidden_out.bias  (fold applied per v below)
    mv(BW(hout_), BW(lat_), BW(dvec_), H, 256);
    sync();
    {
      PcN pc{uint32_t(H)};
      op(p_vadd_, {BW(dvec_), BW(houtb_), BW(dvec_)}, &pc, sizeof(pc),
         wg64(H));
    }
    sync();
    {
      sh::PcDraftLogits pc{uint32_t(i)};
      op(p_logits_, {BW(base_), BW(unemb_), BW(dvec_), BW(logits_)}, &pc,
         sizeof(pc), wg64(V_));
    }
    sync();
    {
      PcPart pc{uint32_t(V_), 2048u};
      op(p_argp_, {BW(logits_), BW(pv_), BW(pi_)}, &pc, sizeof(pc), 98);
    }
    sync();  // partials visible before argmax_final reads them
    {
      PcFin pc{98u};
      op(p_argf_, {BW(pv_), BW(pi_), {&did_, VkDeviceSize(i + 1) * 64, 64}},
         &pc, sizeof(pc), 1);
    }
    sync();
  }
  // implicit 8th id: plain argmax over the slot-7 base logits.
  {
    PcPart pc{uint32_t(V_), 2048u};
    op(p_argp_, {BA(base_, 7 * VSTR_, V_), BW(pv_), BW(pi_)}, &pc, sizeof(pc),
       98);
  }
  sync();
  {
    PcFin pc{98u};
    op(p_argf_, {BW(pv_), BW(pi_), {&did_, VkDeviceSize(8) * 64, 64}}, &pc,
       sizeof(pc), 1);
  }
  if (dbg_en_) {
    sync();
    cp(ctxb_, 0, dbg_, 0, VkDeviceSize(4 * H) * 4);
    cp(out8_, 0, dbg_, VkDeviceSize(4 * H) * 4, VkDeviceSize(8 * H) * 4);
  }
}

// -------------------------------------------------------------------- init

bool GpuDrafter::init(const ModelConfig& c, const SafetensorsFile& mtp,
                      const TensorView& embed, const TensorView& embn,
                      const TensorView& unemb, ink::vk::Context& ctx,
                      ink::vk::Allocator& alloc, const std::string& shader_dir,
                      int64_t ctx_cap, std::string* err,
                      const ink::vk::Buffer* shared_unemb) {
  cfg_ = &c;
  ctx_ = &ctx;
  alloc_ = &alloc;
  shader_dir_ = shader_dir;
  H_ = c.mtp.hidden_size;
  V_ = c.unpadded_vocab_size;
  VSTR_ = (V_ + 15) & ~int64_t(15);  // 200058 -> 200064
  ctxcap_ = std::min<int64_t>(ctx_cap, kCtxMax);
  embed_t_ = &embed;
  dbg_en_ = std::getenv("INK_DRAFT_DEBUG") != nullptr;

  if (c.mtp.num_hidden_layers != 3 || c.mtp.markov_rank != 256 ||
      c.mtp.block_size != 8) {
    *err = "gpu draft: expects 3 layers / rank 256 / block 8";
    return false;
  }
  if (!cpool_.init(ctx, ctx.compute_family) ||
      !tpool_.init(ctx, ctx.copy_family())) {
    *err = "gpu draft: command pool init failed";
    return false;
  }
  auto dev = [&](int64_t bytes) {
    vk::Buffer b = alloc.alloc_device_local(VkDeviceSize(bytes));
    if (!b.buf) b = alloc.alloc_bar_visible(VkDeviceSize(bytes));
    if (!b.buf) b = alloc.alloc_host(VkDeviceSize(bytes));
    if (!b.buf && err_.empty())
      err_ = "gpu draft: buffer alloc failed: " + alloc.last_error;
    return b;
  };
  auto host = [&](int64_t bytes) {
    vk::Buffer b = alloc.alloc_host(VkDeviceSize(bytes));
    if (!b.buf && err_.empty())
      err_ = "gpu draft: host alloc failed: " + alloc.last_error;
    return b;
  };
  const int64_t H = H_;

  staging_ = alloc.alloc_host(1ull << 20);  // grows on demand (uploads)
  dummy4_ = dev(16);
  x_ = dev(8 * H * 4);
  xn_ = dev(8 * H * 4);
  q8_ = dev(8 * 3072 * 4);
  kb8_ = dev(8 * 1024 * 4);
  vb8_ = dev(8 * 1024 * 4);
  att8_ = dev(8 * 3072 * 4);
  ao8_ = dev(8 * H * 4);
  g8_ = dev(8 * 9216 * 4);
  u8_ = dev(8 * 9216 * 4);
  a8_ = dev(8 * 9216 * 4);
  mo8_ = dev(8 * H * 4);
  out8_ = dev(8 * H * 4);
  ctxb_ = dev(4 * H * 4);
  taps_ = dev(4 * H * 4);
  kscr_ = dev(1024 * 4);
  vscr_ = dev(1024 * 4);
  ctxk_ = dev(3 * ctxcap_ * 1024 * 4);
  ctxv_ = dev(3 * ctxcap_ * 1024 * 4);
  base_ = dev(8 * VSTR_ * 4);
  in_ = dev(5120 * 4);
  proj_ = dev(768 * 4);
  state_ = dev(256 * 4);
  lat_ = dev(256 * 4);
  dvec_ = dev(H * 4);
  logits_ = dev(V_ * 4);
  pv_ = dev(1024 * 4);
  pi_ = dev(1024 * 4);
  did_ = host(9 * 64);
  ctxpos_ = host(ctxcap_ * 4);
  dbg_ = host((52 * H + 61440) * 4);  // ctx+out8+x[3]+att0+xin+xn0+q0+k0+ctxk0
  if (!err_.empty() || !staging_.buf || !dummy4_.buf || !ctxk_.buf ||
      !dvec_.buf || !base_.buf || !did_.buf || !ctxpos_.buf || !dbg_.buf) {
    if (err_.empty()) *err = "gpu draft: buffer alloc failed";
    if (!err_.empty()) *err = err_;
    return false;
  }
  did_map_ = (uint32_t*)alloc.mapped(did_);
  cp_map_ = (uint32_t*)alloc.mapped(ctxpos_);
  dbg_map_ = (float*)alloc.mapped(dbg_);

  const bool okp =
      load_pipe("matvec_bf16", 3, sizeof(sh::PcMatvec), p_mvb_) &&
      load_pipe("matvec_bf16_seq", 3, sizeof(sh::PcMatvecSeq), p_mvs_) &&
      load_pipe("rmsnorm", 3, sizeof(sh::PcRmsnorm), p_rms_) &&
      load_pipe("rmsnorm_seq", 3, sizeof(sh::PcRmsnormSeq), p_rmss_) &&
      load_pipe("argmax_part", 3, sizeof(PcPart), p_argp_) &&
      load_pipe("argmax_final", 3, sizeof(PcFin), p_argf_) &&
      load_pipe("vadd", 3, sizeof(PcN), p_vadd_) &&
      load_pipe("draft_mix", 4, sizeof(sh::PcDraftMix), p_mix_) &&
      load_pipe("draft_rope", 1, sizeof(sh::PcDraftRope), p_rope_) &&
      load_pipe("draft_attn", 7, sizeof(sh::PcDraftAttn), p_attn_) &&
      load_pipe("draft_swiglu", 3, sizeof(PcN), p_swig_) &&
      load_pipe("draft_resid", 3, sizeof(PcN), p_resid_) &&
      load_pipe("draft_pack", 6, sizeof(sh::PcDraftPack), p_pack_) &&
      load_pipe("draft_gru", 4, 0, p_gru_) &&
      load_pipe("draft_logits", 4, sizeof(sh::PcDraftLogits), p_logits_);
  if (!okp) {
    if (err->empty()) *err = "gpu draft: pipeline load failed: " + err_;
    return false;
  }

  // ------------------------------- weights -------------------------------
  std::vector<std::string> missing;
  auto get = [&](const char* n) -> const TensorView* {
    const TensorView* t = mtp.find(n);
    if (!t || !t->data) missing.push_back(n);
    return t;
  };
  auto W = [&](const char* n, ink::vk::Buffer& out) {
    if (const TensorView* t = get(n)) {
      const ink::vk::Buffer* b = dev_w(*t);
      if (!b) {
        missing.push_back(std::string(n) + " upload");
        return;
      }
      out = std::move(*b);
      delete b;
    }
  };
  auto F = [&](const char* n, ink::vk::Buffer& out) {
    if (const TensorView* t = get(n)) {
      const ink::vk::Buffer* b = dev_f32(*t);
      if (!b) {
        missing.push_back(std::string(n) + " upload");
        return;
      }
      out = std::move(*b);
      delete b;
    }
  };

  static const char* kL[3] = {"layers.0.", "layers.1.", "layers.2."};
  // packed gamma uploads: rows 0..2 = draft layers (dgam_ row 3 = markov's
  // dflare_norm), all decoded to f32.
  auto pack = [&](const char* suffix, ink::vk::Buffer& dst, int rows, int n,
                  bool with_markov) {
    std::vector<float> v(size_t(rows) * n, 0.f);
    bool any = false;
    for (int i = 0; i < 3; i++) {
      std::string nm = kL[i];
      nm += suffix;
      if (const TensorView* t = mtp.find(nm.c_str()))
        dec_f32(*t, v.data() + i * n, n), any = true;
      else
        missing.push_back(nm);
    }
    if (with_markov) {
      if (const TensorView* t = mtp.find("markov_head.dflare_norm.weight"))
        dec_f32(*t, v.data() + 3 * n, n), any = true;
      else
        missing.push_back("markov_head.dflare_norm.weight");
    }
    if (any) {
      vk::Buffer up = upload_dev(v.data(), VkDeviceSize(v.size()) * 4);
      if (!up.buf)
        missing.push_back(std::string(suffix) + " upload");
      else
        dst = std::move(up);
    }
  };
  pack("input_layernorm.weight", inln_, 3, int(H), false);
  pack("post_attention_layernorm.weight", pln_, 3, int(H), false);
  pack("dflare_norm.weight", dgam_, 4, int(H), true);
  pack("self_attn.q_norm.weight", qn_, 3, 128, false);
  pack("self_attn.k_norm.weight", kn_, 3, 128, false);
  pack("self_attn.k_norm_ctx.weight", knc_, 3, 128, false);
  if (const TensorView* t = get("norm.weight")) {
    std::vector<float> v = std::vector<float>(size_t(H));
    dec_f32(*t, v.data(), H);
    normw_ = upload_dev(v.data(), VkDeviceSize(H) * 4);
  }
  // fuse_weight [4,4] -> alpha[c][j] = softmax over j (fp32, host)
  if (const TensorView* t = get("fuse_weight")) {
    float al[16];
    for (int r = 0; r < 4; r++) {
      float m = -1e30f;
      for (int j = 0; j < 4; j++) m = std::max(m, t->f32(r, j));
      float s = 0.f;
      for (int j = 0; j < 4; j++)
        s += (al[r * 4 + j] = std::exp(t->f32(r, j) - m));
      for (int j = 0; j < 4; j++) al[r * 4 + j] /= s;
    }
    alpha_ = upload_dev(al, 64);
  }
  // host copies: embed_norm gamma + mask_embed (block input is built on host)
  if (embn.data) {
    emb_g_.assign(size_t(H), 0.f);
    for (int64_t i = 0; i < H; i++) emb_g_[size_t(i)] = embn.f32_linear(i);
  }
  if (const TensorView* t = get("mask_embed")) {
    mask_.assign(size_t(H), 0.f);
    dec_f32(*t, mask_.data(), H);
  }
  for (int i = 0; i < 3; i++) {
    std::string p = kL[i];
    W((p + "self_attn.q_proj.weight").c_str(), wq_[i]);
    W((p + "self_attn.k_proj.weight").c_str(), wk_[i]);
    W((p + "self_attn.v_proj.weight").c_str(), wv_[i]);
    W((p + "self_attn.o_proj.weight").c_str(), wo_[i]);
    W((p + "self_attn.k_proj_ctx.weight").c_str(), wkc_[i]);
    W((p + "self_attn.v_proj_ctx.weight").c_str(), wvc_[i]);
    W((p + "mlp.gate_proj.weight").c_str(), wg_[i]);
    W((p + "mlp.up_proj.weight").c_str(), wu_[i]);
    W((p + "mlp.down_proj.weight").c_str(), wd_[i]);
  }
  W("markov_head.markov_w1.weight", w1_);
  W("markov_head.rnn_proj.weight", rnn_);
  F("markov_head.rnn_proj.bias", rnn_b_);

  // unembed: share the graph's raw-bf16 buffer when given (saves ~883MB
  // on a full VRAM box); else upload a private copy.
  if (shared_unemb && shared_unemb->buf) {
    unemb_ = *shared_unemb;
    unemb_shared_ = true;
  } else {
    const TensorView& t = unemb;
    if (!t.data)
      missing.push_back("unembed");
    else
      unemb_ = upload_dev(t.data, t.nbytes());
    if (!unemb_.buf) missing.push_back("unembed host mirror");
  }

  // markov fold: dvec = hidden_out @ lat + hidden_out.bias. hidden_out
  // stays raw bf16 [2304,256] (matvec_bf16 consumes it directly); the
  // bias is f32 and folded in by vadd each step.
  W("markov_head.hidden_out.weight", hout_);
  F("markov_head.hidden_out.bias", houtb_);

  if (!missing.empty()) {
    for (auto& m : missing)
      std::fprintf(stderr, "gpu draft: missing %s\n", m.c_str());
    *err = "gpu draft: " + std::to_string(missing.size()) +
           " tensors missing/failed, first: " + missing.front() +
           (err_.empty() ? "" : " (" + err_ + ")");
    return false;
  }
  if (!err_.empty()) {
    *err = err_;
    return false;
  }

  // ------------------------- build pass: fixed descset stream ------------
  building_ = true;
  emit_block(0, 0, true);
  building_ = false;
  if (!err_.empty()) {
    *err = "gpu draft: descriptor build failed: " + err_;
    return false;
  }
  reset();
  ok_ = true;
  return true;
}

// --------------------------------------------------------------------- api

void GpuDrafter::reset() { n_ctx_ = 0; }

std::vector<int> GpuDrafter::propose(const float* taps4, int anchor,
                                     int64_t pos) {
  std::vector<int> ids;
  if (!ok_ || !taps4) return ids;
  const int64_t H = H_;
  const float eps = float(cfg_->mtp.rms_norm_eps);

  // ---- host staging: block input rows + tap block + cells ----
  float* sp = (float*)alloc_->mapped(staging_);
  // row 0: embed_norm(embed[anchor]) computed host-side (one row, trivial).
  {
    std::vector<float> e = std::vector<float>(size_t(H));
    const TensorView& t = *embed_t_;
    const int64_t off = int64_t(anchor) * H;
    double ss = 0.0;
    for (int64_t d = 0; d < H; d++) {
      e[size_t(d)] = t.f32_linear(off + d);
      ss += double(e[size_t(d)]) * e[size_t(d)];
    }
    const float rstd = float(1.0 / std::sqrt(ss / H + eps));
    for (int64_t d = 0; d < H; d++)
      sp[d] = e[size_t(d)] * rstd * emb_g_[size_t(d)];
  }
  for (int j = 1; j < 8; j++)
    std::memcpy(sp + j * H, mask_.data(), size_t(H) * 4);
  std::memcpy(sp + 8 * H, taps4, size_t(4 * H) * 4);

  const bool appending = n_ctx_ < ctxcap_;
  did_map_[0] = uint32_t(anchor);
  if (appending) cp_map_[n_ctx_] = uint32_t(pos);
  alloc_->flush(staging_, 0, VkDeviceSize(12 * H) * 4);
  alloc_->flush(did_, 0, 64);
  alloc_->flush(ctxpos_);

  const auto t0 = clk::steady_clock::now();
  const double wait0 = stats.wait_ms;
  vk::Recorder r;
  if (!r.begin(cpool_)) {
    err_ = "draft recorder begin failed: " + r.error();
    return {};
  }
  rec_ = &r;
  site_cursor_ = 0;
  emit_block(pos, n_ctx_, appending);
  rec_ = nullptr;
  if (!r.submit_and_wait(ctx_->compute_queue)) {
    err_ = "draft submit failed: " + r.error();
    r.shutdown();
    return {};
  }
  stats.submits++;
  stats.wait_ms +=
      clk::duration<double, std::milli>(clk::steady_clock::now() - t0).count();
  stats.record_ms +=
      clk::duration<double, std::milli>(clk::steady_clock::now() - t0)
          .count() -
      (stats.wait_ms - wait0);
  r.shutdown();
  if (appending) n_ctx_++;

  alloc_->invalidate(did_);
  ids.assign(8, 0);
  for (int j = 0; j < 8; j++) ids[j] = int(did_map_[(j + 1) * 16]);
  if (dbg_en_) alloc_->invalidate(dbg_);
  return ids;
}

void GpuDrafter::shutdown() {
  if (!ctx_ || !ctx_->device) return;
  vkDeviceWaitIdle(ctx_->device);
  for (auto& ds : sites_) ds.shutdown(*ctx_);
  sites_.clear();
  vk::Buffer* bufs[] = {&staging_, &dummy4_, &x_,     &xn_,    &q8_,   &kb8_,
                        &vb8_,     &att8_,   &ao8_,   &g8_,    &u8_,   &a8_,
                        &mo8_,     &out8_,   &ctxb_,  &taps_,  &kscr_, &vscr_,
                        &ctxk_,    &ctxv_,   &base_,  &in_,    &proj_,
                        &state_,   &lat_,    &dvec_,  &logits_, &pv_,  &pi_,
                        &did_,     &ctxpos_, &dbg_,   &w1_,   &rnn_,
                        &rnn_b_,   &hout_,   &houtb_, &alpha_, &dgam_, &normw_,
                        &inln_,    &pln_,    &qn_,    &kn_,    &knc_};
  for (auto* b : bufs)
    if (b->buf) alloc_->free(*b);
  if (!unemb_shared_ && unemb_.buf) alloc_->free(unemb_);
  unemb_.buf = VK_NULL_HANDLE;
  for (int i = 0; i < 3; i++) {
    vk::Buffer* lw[] = {&wq_[i], &wk_[i], &wv_[i], &wo_[i], &wkc_[i],
                        &wvc_[i], &wg_[i], &wu_[i], &wd_[i]};
    for (auto* b : lw)
      if (b->buf) alloc_->free(*b);
  }
  Pipe* pipes[] = {&p_mvb_,  &p_mvs_, &p_rms_,  &p_rmss_, &p_argp_,
                   &p_argf_, &p_vadd_, &p_mix_,  &p_rope_, &p_attn_,
                   &p_swig_, &p_resid_, &p_pack_, &p_gru_, &p_logits_};
  for (auto* p : pipes) p->p.shutdown();
  cpool_.shutdown();
  tpool_.shutdown();
  ok_ = false;
  ctx_ = nullptr;
}

}  // namespace ink::gpu
