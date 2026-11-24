// vk_backend.h — Vulkan compute backend for ops (P1 vertical slice).
// Correctness-first: per-op submit_and_wait, weights cached on device,
// activations in host-visible coherent buffers. The golden gate arbitrates.
#pragma once

#include "model/backend.h"
#include "vk/cmd.h"
#include "vk/mem.h"
#include "vk/pipe.h"
#include "vk/vk.h"

#include <string>
#include <unordered_map>

namespace ink::gpu {

class VkBackend : public ops::Backend {
 public:
  // shader_dir: directory of <name>.spv files (build/shaders).
  // cache_cap: device-memory budget for the weight cache.
  bool init(const std::string& shader_dir, std::string& err,
            VkDeviceSize cache_cap = 12ull << 30);
  void shutdown();
  ~VkBackend() { shutdown(); }

  bool matvec(const TensorView& W, const float* x, float* out, int64_t M,
              int64_t K) override;

  bool expert_ffn(const TensorView& w13, const TensorView& w13s,
                  const TensorView& w13amax, const TensorView& w13osd,
                  const TensorView& w2, const TensorView& w2s,
                  const TensorView& w2amax, const TensorView& w2osd,
                  bool mxfp8, int e, const float* x, int64_t D, int64_t I,
                  float gcap, float ucap, float wscale, float* out) override;

 private:
  struct Pipe {
    ink::vk::ComputePipeline p;
    ink::vk::DescSet ds;   // expert pipes: OUT=gbuf (gate/down variant)
    ink::vk::DescSet ds2;  // expert pipes: OUT=ubuf (up variant)
  };

  ink::vk::Context ctx_;
  ink::vk::Allocator alloc_;
  ink::vk::CmdPool cpool_;  // compute family
  ink::vk::CmdPool tpool_;  // copy family (SDMA when present)
  std::string shader_dir_;

  Pipe mv_f32_, mv_bf16_, mv_f8_, mv_nv4_, mv_mx8_, moe_acc_;

  // Weight cache: TensorView.data -> device-local buffer. LRU eviction
  // against cache_cap_. host_cache_: expert tensors mirrored into
  // HOST_VISIBLE memory once — the GPU streams only the rows it needs
  // over PCIe (8/128 experts per layer), no VRAM residency, no churn.
  struct Cached {
    ink::vk::Buffer b;
    uint64_t tick = 0;
  };
  std::unordered_map<const void*, Cached> cache_;
  std::unordered_map<const void*, ink::vk::Buffer> host_cache_;
  uint64_t tick_ = 0;
  VkDeviceSize cache_bytes_ = 0, cache_cap_ = 0;

  ink::vk::Buffer staging_;  // host->device upload staging (chunked reuse)
  ink::vk::Buffer xbuf_;     // host-coherent activation in  (f32)
  ink::vk::Buffer obuf_;     // host-coherent matvec out      (f32)
  ink::vk::Buffer gbuf_, ubuf_, eobuf_;  // expert scratch
  ink::vk::Buffer dummy4_;   // 1-float dummy for unbound amax/osd on mxfp8

  bool ok_ = false;
  std::string err_;

 public:
  // P2 instrumentation: cumulative counters, read for reporting.
  struct Stats {
    uint64_t submits = 0, uploads = 0, upload_bytes = 0;
    double wait_ms = 0, upload_ms = 0;
    uint64_t n_matvec = 0, n_ffn = 0;
    // reject reasons (why a call fell back to CPU)
    uint64_t rej_shape = 0, rej_dtype = 0, rej_upload = 0, rej_alloc = 0,
             rej_submit = 0;
    uint64_t upf_noncontig = 0, upf_toobig = 0, upf_alloc = 0, upf_rec = 0;
    uint64_t upf_begin = 0, upf_submit = 0;
    std::string first_err;
  } stats;

  // Upload t.data (contiguous only) to device memory; cached by pointer.
  // Returns nullptr when the tensor can't be served (non-contiguous, OOM).
  const ink::vk::Buffer* upload(const TensorView& t);
  // Mirror t.data into a persistent HOST_VISIBLE buffer (one memcpy, kept
  // for the run). GPU reads it over PCIe on demand.
  const ink::vk::Buffer* host_buf(const TensorView& t);
  void evict_for(VkDeviceSize need);

  bool grow_host(ink::vk::Buffer& b, VkDeviceSize bytes);
  bool grow_dev(ink::vk::Buffer& b, VkDeviceSize bytes);

  bool load_pipe(const char* name, uint32_t bindings, uint32_t pc_bytes,
                 Pipe& out, bool dual_set = false);
  bool write_n(const ink::vk::DescSet& ds, const ink::vk::Buffer* const* bufs,
               uint32_t n);

  bool fail(const char* m) {
    err_ = m;
    return false;
  }
};

}  // namespace ink::gpu
