// vk_backend.cpp — see header. Correctness-first P1 implementation:
// one submit_and_wait per op; weights staged via SDMA copy; activations and
// outputs live in HOST_VISIBLE coherent buffers (x/out are small, the GPU
// reads/writes them over PCIe — saves a copy and keeps the code obvious).
#include "gpu/vk_backend.h"

#include "bindings.h"  // shaders/ push-constant contracts

#include <algorithm>
#include <cstring>

namespace ink::gpu {
namespace sh = ink::shaders;

namespace {

// TensorView may be a view; only contiguous layouts upload as one blob.
bool contiguous(const TensorView& t) {
  int64_t expect = 1;
  for (int i = t.nd - 1; i >= 0; i--) {
    if (t.stride[i] != expect) return false;
    expect *= t.shape[i];
  }
  return true;
}

}  // namespace

bool VkBackend::load_pipe(const char* name, uint32_t bindings,
                          uint32_t pc_bytes, Pipe& out, bool dual_set) {
  ink::vk::PipelineDesc d;
  d.spv_path = shader_dir_ + "/" + name + ".spv";
  d.storage_bindings = bindings;
  d.push_constant_bytes = pc_bytes;
  if (!ink::vk::load_compute_pipeline(ctx_, d, out.p, err_)) {
    err_ = std::string(name) + ": " + err_;
    return false;
  }
  if (!ink::vk::alloc_descset(ctx_, out.p.dsl, bindings, out.ds, err_) ||
      (dual_set &&
       !ink::vk::alloc_descset(ctx_, out.p.dsl, bindings, out.ds2, err_))) {
    err_ = std::string(name) + ": " + err_;
    return false;
  }
  return true;
}

bool VkBackend::init(const std::string& shader_dir, std::string& err,
                     VkDeviceSize cache_cap) {
  shader_dir_ = shader_dir;
  cache_cap_ = cache_cap;
  if (!ctx_.init()) {
    err = ctx_.last_error;
    return false;
  }
  if (!alloc_.init(ctx_)) {
    err = alloc_.last_error;
    return false;
  }
  if (!cpool_.init(ctx_, ctx_.compute_family)) {
    err = "compute cmdpool init failed";
    return false;
  }
  if (!tpool_.init(ctx_, ctx_.copy_family())) {
    err = "copy cmdpool init failed";
    return false;
  }
  dummy4_ = alloc_.alloc_device_local(4);
  staging_ = alloc_.alloc_host(64ull << 20);  // 64 MiB staging chunk
  if (!alloc_.ok(dummy4_) || !alloc_.ok(staging_)) {
    err = "staging alloc failed: " + alloc_.last_error;
    return false;
  }
  bool ok = load_pipe("matvec_f32", 3, sizeof(sh::PcMatvec), mv_f32_) &&
            load_pipe("matvec_bf16", 3, sizeof(sh::PcMatvec), mv_bf16_) &&
            load_pipe("matvec_f8e4m3", 3, sizeof(sh::PcMatvec), mv_f8_) &&
            load_pipe("matvec_nvfp4", 6, sizeof(sh::PcNvfp4), mv_nv4_, true) &&
            load_pipe("matvec_mxfp8", 4, sizeof(sh::PcMxfp8), mv_mx8_, true) &&
            load_pipe("moe_accum", 7, sizeof(sh::PcMoeAccum), moe_acc_);
  if (!ok) {
    err = err_;
    return false;
  }
  ok_ = true;
  return true;
}

void VkBackend::shutdown() {
  if (ctx_.device) vkDeviceWaitIdle(ctx_.device);
  for (auto& kv : cache_) alloc_.free(kv.second.b);
  cache_.clear();
  cache_bytes_ = 0;
  for (ink::vk::Buffer* b : {&staging_, &xbuf_, &obuf_, &gbuf_, &ubuf_, &eobuf_,
                             &dummy4_})
    alloc_.free(*b);
  for (Pipe* p : {&mv_f32_, &mv_bf16_, &mv_f8_, &mv_nv4_, &mv_mx8_, &moe_acc_}) {
    p->ds.shutdown(ctx_);
    p->ds2.shutdown(ctx_);
    p->p.shutdown();
  }
  tpool_.shutdown();
  cpool_.shutdown();
  alloc_.shutdown();
  ctx_.shutdown();
  ok_ = false;
}

bool VkBackend::grow_host(ink::vk::Buffer& b, VkDeviceSize bytes) {
  if (b.buf && b.size >= bytes) return true;
  alloc_.free(b);
  b = alloc_.alloc_host(std::max<VkDeviceSize>(bytes, 1 << 20));
  return alloc_.ok(b);
}

bool VkBackend::grow_dev(ink::vk::Buffer& b, VkDeviceSize bytes) {
  if (b.buf && b.size >= bytes) return true;
  alloc_.free(b);
  b = alloc_.alloc_device_local(std::max<VkDeviceSize>(bytes, 1 << 20));
  return alloc_.ok(b);
}

void VkBackend::evict_for(VkDeviceSize need) {
  while (cache_bytes_ + need > cache_cap_ && !cache_.empty()) {
    auto victim = cache_.begin();
    for (auto it = cache_.begin(); it != cache_.end(); ++it)
      if (it->second.tick < victim->second.tick) victim = it;
    cache_bytes_ -= victim->second.b.size;
    alloc_.free(victim->second.b);
    cache_.erase(victim);
  }
}

const ink::vk::Buffer* VkBackend::upload(const TensorView& t) {
  if (!t.data || !contiguous(t)) return nullptr;
  VkDeviceSize bytes = VkDeviceSize(t.nbytes());
  auto it = cache_.find(t.data);
  if (it != cache_.end()) {
    it->second.tick = ++tick_;
    return &it->second.b;
  }
  if (bytes > cache_cap_) return nullptr;  // will never fit; CPU fallback
  evict_for(bytes);
  ink::vk::Buffer dev = alloc_.alloc_device_local(bytes);
  if (!alloc_.ok(dev)) return nullptr;

  // Chunked staging upload: memcpy RAM->staging (full speed), then one
  // recorded copy per chunk submitted once on the SDMA queue.
  ink::vk::Recorder rec;
  if (!rec.begin(tpool_)) {
    alloc_.free(dev);
    return nullptr;
  }
  const std::byte* src = t.data;
  char* stg = static_cast<char*>(alloc_.mapped(staging_));
  VkDeviceSize chunk = staging_.size;
  for (VkDeviceSize off = 0; off < bytes; off += chunk) {
    VkDeviceSize n = std::min(chunk, bytes - off);
    std::memcpy(stg, src + off, size_t(n));
    rec.copy_buffer(staging_.buf, dev.buf, n, 0, off);
  }
  if (!rec.end_and_submit(ctx_.copy_queue())) {
    rec.shutdown();
    alloc_.free(dev);
    return nullptr;
  }
  rec.shutdown();
  Cached c{dev, ++tick_};
  auto [ins, _] = cache_.emplace(t.data, c);
  (void)_;
  cache_bytes_ += bytes;
  return &ins->second.b;
}

bool VkBackend::write_n(const ink::vk::DescSet& ds,
                        const ink::vk::Buffer* const* bufs, uint32_t n) {
  for (uint32_t i = 0; i < n; i++)
    if (!ink::vk::write_storage(ctx_, ds, i, bufs[i]->buf, bufs[i]->offset,
                                bufs[i]->size, err_))
      return false;
  return true;
}

bool VkBackend::matvec(const TensorView& W, const float* x, float* out,
                       int64_t M, int64_t K) {
  if (!ok_ || W.nd != 2 || W.shape[0] != M || W.shape[1] != K) return false;
  Pipe* pipe = nullptr;
  switch (W.dtype) {
    case DType::F32:
      pipe = &mv_f32_;
      break;
    case DType::BF16:
      if (K % 2) return false;
      pipe = &mv_bf16_;
      break;
    case DType::F8E4M3:
      if (K % 4) return false;
      pipe = &mv_f8_;
      break;
    default:
      return false;  // U8/I8 stay on CPU
  }
  const ink::vk::Buffer* w = upload(W);
  if (!w) return false;
  if (!grow_host(xbuf_, VkDeviceSize(K) * 4) ||
      !grow_host(obuf_, VkDeviceSize(M) * 4))
    return false;
  std::memcpy(alloc_.mapped(xbuf_), x, size_t(K) * 4);
  const ink::vk::Buffer* b3[3] = {w, &xbuf_, &obuf_};
  if (!write_n(pipe->ds, b3, 3)) return false;

  ink::vk::Recorder rec;
  if (!rec.begin(cpool_)) return false;
  rec.bind_pipeline(pipe->p.pipe);
  rec.bind_descset(pipe->p.layout, pipe->ds.set);
  // grid.x is capped at 65535; the shader adds pc.m0 to the workgroup id.
  for (int64_t m0 = 0; m0 < M; m0 += 65535) {
    sh::PcMatvec pc{uint32_t(K), uint32_t(m0)};
    rec.push_constants(pipe->p.layout, &pc, sizeof(pc));
    rec.dispatch(uint32_t(std::min<int64_t>(M - m0, 65535)));
  }
  if (!rec.end_and_submit(ctx_.compute_queue)) {
    rec.shutdown();
    return false;
  }
  rec.shutdown();
  std::memcpy(out, alloc_.mapped(obuf_), size_t(M) * 4);
  return true;
}

bool VkBackend::expert_ffn(const TensorView& w13, const TensorView& w13s,
                           const TensorView& w13amax, const TensorView& w13osd,
                           const TensorView& w2, const TensorView& w2s,
                           const TensorView& w2amax, const TensorView& w2osd,
                           bool mxfp8, int e, const float* x, int64_t D,
                           int64_t I, float gcap, float ucap, float wscale,
                           float* out) {
  if (!ok_) return false;
  const uint32_t align = mxfp8 ? 128 : 64;
  if (D % align || I % align) return false;

  const ink::vk::Buffer* bufs[8];
  const TensorView* ts[8] = {&w13, &w13s, &w13amax, &w13osd,
                             &w2,  &w2s,  &w2amax,  &w2osd};
  for (int i = 0; i < 8; i++) {
    bufs[i] = ts[i]->data ? upload(*ts[i]) : &dummy4_;
    if (!bufs[i]) return false;
  }
  if (!grow_host(xbuf_, VkDeviceSize(D) * 4) ||
      !grow_dev(gbuf_, VkDeviceSize(I) * 4) ||
      !grow_dev(ubuf_, VkDeviceSize(I) * 4) ||
      !grow_host(eobuf_, VkDeviceSize(D) * 4))
    return false;
  std::memcpy(alloc_.mapped(xbuf_), x, size_t(D) * 4);
  std::memcpy(alloc_.mapped(eobuf_), out, size_t(D) * 4);

  // Descriptor contents are sampled at execution time, so the two w13
  // dispatches need separate sets (gate set: OUT=gbuf; up set: OUT=ubuf).
  Pipe& p13 = mxfp8 ? mv_mx8_ : mv_nv4_;
  const ink::vk::Buffer* bg[6] = {bufs[0], bufs[1], bufs[2],
                                  bufs[3], &xbuf_,  &gbuf_};
  const ink::vk::Buffer* bu[6] = {bufs[0], bufs[1], bufs[2],
                                  bufs[3], &xbuf_,  &ubuf_};
  const ink::vk::Buffer* bd[7] = {&gbuf_, &ubuf_, bufs[4], bufs[5],
                                  bufs[6], bufs[7], &eobuf_};
  if (mxfp8) {
    // mx8 layout: W,S,X,OUT (4 bindings)
    const ink::vk::Buffer* mg[4] = {bufs[0], bufs[1], &xbuf_, &gbuf_};
    const ink::vk::Buffer* mu[4] = {bufs[0], bufs[1], &xbuf_, &ubuf_};
    if (!write_n(p13.ds, mg, 4) || !write_n(p13.ds2, mu, 4) ||
        !write_n(moe_acc_.ds, bd, 7))
      return false;
  } else if (!write_n(p13.ds, bg, 6) || !write_n(p13.ds2, bu, 6) ||
             !write_n(moe_acc_.ds, bd, 7)) {
    return false;
  }

  ink::vk::Recorder rec;
  if (!rec.begin(cpool_)) return false;
  for (uint32_t row0 = 0; row0 < 2; row0++) {
    const ink::vk::DescSet& ds = row0 ? p13.ds2 : p13.ds;
    rec.bind_pipeline(p13.p.pipe);
    rec.bind_descset(p13.p.layout, ds.set);
    if (mxfp8) {
      sh::PcMxfp8 pm{uint32_t(e), uint32_t(w13.shape[1]), uint32_t(D),
                     row0,        2,                      1.0f,
                     0};
      rec.push_constants(p13.p.layout, &pm, sizeof(pm));
    } else {
      sh::PcNvfp4 pc{uint32_t(e), uint32_t(w13.shape[1]), uint32_t(D),
                     row0,        2,                      1.0f,
                     0};
      rec.push_constants(p13.p.layout, &pc, sizeof(pc));
    }
    rec.dispatch(uint32_t(I));
    rec.compute_barrier();
  }
  rec.bind_pipeline(moe_acc_.p.pipe);
  rec.bind_descset(moe_acc_.p.layout, moe_acc_.ds.set);
  {
    sh::PcMoeAccum pc{uint32_t(e), uint32_t(w2.shape[1]), uint32_t(I),
                      mxfp8 ? 1u : 0u, gcap, ucap, wscale};
    rec.push_constants(moe_acc_.p.layout, &pc, sizeof(pc));
    rec.dispatch(uint32_t(D));
  }
  if (!rec.end_and_submit(ctx_.compute_queue)) {
    rec.shutdown();
    return false;
  }
  rec.shutdown();
  std::memcpy(out, alloc_.mapped(eobuf_), size_t(D) * 4);
  return true;
}

}  // namespace ink::gpu
