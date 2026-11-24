// Vulkan plumbing smoke test: context, queues, memory types, feature bits,
// host<->device roundtrip over the dedicated transfer (SDMA) queue, and one
// trivial compute dispatch when a .spv is available.
//
// Usage: test_vk_smoke [path/to/kernel.spv]
//   Default spv search: INK_SHADER_DIR env/define dir (build/shaders), then
//   ./shaders, ./build/shaders relative to cwd.

#include "vk/vk.h"
#include "vk/mem.h"
#include "vk/cmd.h"
#include "vk/pipe.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace ink;

static int fails = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);    \
      fails++;                                                        \
    }                                                                 \
  } while (0)

static const char* qflag_str(VkQueueFlags f, char* buf) {
  buf[0] = 0;
  if (f & VK_QUEUE_GRAPHICS_BIT) std::strcat(buf, "GFX|");
  if (f & VK_QUEUE_COMPUTE_BIT) std::strcat(buf, "COMP|");
  if (f & VK_QUEUE_TRANSFER_BIT) std::strcat(buf, "XFER|");
  if (f & VK_QUEUE_SPARSE_BINDING_BIT) std::strcat(buf, "SPARSE|");
  if (f & VK_QUEUE_PROTECTED_BIT) std::strcat(buf, "PROT|");
  size_t n = std::strlen(buf);
  if (n) buf[n - 1] = 0;
  return buf;
}

static void report_context(vk::Context& c) {
  std::printf("== device ==\n");
  std::printf("  name: %s\n", c.props.deviceName);
  std::printf("  api: %u.%u.%u  driver: %u.%u.%u  vendor: 0x%04x\n",
              VK_API_VERSION_MAJOR(c.props.apiVersion),
              VK_API_VERSION_MINOR(c.props.apiVersion),
              VK_API_VERSION_PATCH(c.props.apiVersion),
              VK_API_VERSION_MAJOR(c.props.driverVersion),
              VK_API_VERSION_MINOR(c.props.driverVersion),
              VK_API_VERSION_PATCH(c.props.driverVersion), c.props.vendorID);
  std::printf("  compute family %u queue %p\n", c.compute_family,
              (void*)c.compute_queue);
  if (c.has_dedicated_transfer())
    std::printf("  transfer family %u queue %p (dedicated)\n",
                c.transfer_family, (void*)c.transfer_queue);
  else
    std::printf("  no dedicated transfer family; copies use compute queue\n");

  std::printf("== queue families ==\n");
  uint32_t nq = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nq, nullptr);
  std::vector<VkQueueFamilyProperties> qf(nq);
  vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nq, qf.data());
  for (uint32_t i = 0; i < nq; i++) {
    char fb[96];
    std::printf("  family %u: count=%u flags=0x%x %s%s\n", i,
                qf[i].queueCount, qf[i].queueFlags,
                qflag_str(qf[i].queueFlags, fb),
                i == c.compute_family      ? "  <- compute"
                : i == c.transfer_family   ? "  <- transfer"
                                           : "");
  }

  std::printf("== memory types ==\n");
  const auto& mp = c.memprops;
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    const auto& t = mp.memoryTypes[i];
    const auto& h = mp.memoryHeaps[t.heapIndex];
    std::printf(
        "  type %2u: heap %u (%llu MiB%s) flags=0x%02x %s%s%s%s%s\n", i,
        t.heapIndex, (unsigned long long)(h.size >> 20),
        (h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " DEV" : "",
        t.propertyFlags,
        (t.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? " LOCAL" : "",
        (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? " VIS" : "",
        (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? " COH" : "",
        (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? " CACHED" : "",
        (t.propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) ? " LAZY"
                                                                  : "");
  }
  if (c.memory_budget_ext) {
    std::printf("== memory budget (VK_EXT_memory_budget) ==\n");
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
      std::printf("  heap %u: budget %llu MiB  usage %llu MiB\n", i,
                  (unsigned long long)(c.budget.heapBudget[i] >> 20),
                  (unsigned long long)(c.budget.heapUsage[i] >> 20));
  }

  std::printf("== features ==\n");
  std::printf("  timeline_semaphore   : %s\n",
              c.feats.timeline_semaphore ? "on" : "off");
  std::printf("  buffer_device_address: %s\n",
              c.feats.buffer_device_address ? "on" : "off");
  std::printf("  cooperative_matrix   : %s\n",
              c.feats.cooperative_matrix ? "on" : "off");
  std::printf("  shader_float8        : %s\n",
              c.feats.shader_float8 ? "on" : "off");
  std::printf("  shader_bfloat16      : %s\n",
              c.feats.shader_bfloat16 ? "on" : "off");

  CHECK(c.feats.timeline_semaphore);
  CHECK(c.compute_queue != VK_NULL_HANDLE);
}

// 4 MiB roundtrip: host A -> device -> host B through the dedicated
// transfer queue.
static void test_copy_roundtrip(vk::Context& c, vk::Allocator& a) {
  std::printf("== copy roundtrip (transfer queue) ==\n");
  const VkDeviceSize n = 4ull << 20;

  vk::Buffer src = a.alloc_host(n);
  vk::Buffer dst = a.alloc_host(n);
  vk::Buffer dev = a.alloc_device_local(n);
  CHECK(src.buf && src.ptr);
  CHECK(dst.buf && dst.ptr);
  CHECK(dev.buf);
  if (!src.buf || !dst.buf || !dev.buf) {
    std::printf("  alloc failed: %s\n", a.last_error.c_str());
    return;
  }

  for (VkDeviceSize i = 0; i < n; i++)
    ((uint8_t*)src.ptr)[i] = (uint8_t)(i * 2654435761u >> 13);
  std::memset(dst.ptr, 0, n);
  a.flush(src);  // no-op on coherent, keeps the API honest

  vk::CmdPool tp;
  CHECK(tp.init(c, c.copy_family()));
  {
    vk::Recorder r;
    CHECK(r.begin(tp));
    r.copy_buffer(src.buf, dev.buf, n);
    CHECK(r.submit_and_wait(c.copy_queue()));
    r.shutdown();
  }
  {
    vk::Recorder r;
    CHECK(r.begin(tp));
    r.copy_buffer(dev.buf, dst.buf, n);
    CHECK(r.submit_and_wait(c.copy_queue()));
    r.shutdown();
  }
  tp.shutdown();
  a.invalidate(dst);

  CHECK(std::memcmp(src.ptr, dst.ptr, n) == 0);
  if (std::memcmp(src.ptr, dst.ptr, n) == 0)
    std::printf("  %llu bytes roundtrip OK\n", (unsigned long long)n);
  else {
    for (VkDeviceSize i = 0; i < n; i++)
      if (((uint8_t*)src.ptr)[i] != ((uint8_t*)dst.ptr)[i]) {
        std::printf("  first mismatch at %llu: 0x%02x vs 0x%02x\n",
                    (unsigned long long)i, ((uint8_t*)src.ptr)[i],
                    ((uint8_t*)dst.ptr)[i]);
        break;
      }
  }

  a.free(src);
  a.free(dst);
  a.free(dev);
}

// The dispatch check needs a kernel matching the test's interface (1 storage
// buffer, 8-byte push constants): that's shaders/test_fill.comp. Other agents'
// kernels have their own contracts, so only test_fill.spv (or an explicit
// argv[1]) is a valid pick.
static std::string find_spv(int argc, char** argv) {
  if (argc > 1) return argv[1];
  std::vector<std::string> dirs;
#ifdef INK_SHADER_DIR
  dirs.push_back(INK_SHADER_DIR);
#endif
  if (const char* e = std::getenv("INK_SHADER_DIR")) dirs.push_back(e);
  dirs.push_back("shaders");
  dirs.push_back("build/shaders");
  dirs.push_back("../build/shaders");
  for (const auto& d : dirs) {
    std::error_code ec;
    std::filesystem::path p = std::filesystem::path(d) / "test_fill.spv";
    if (std::filesystem::is_regular_file(p, ec)) return p.string();
  }
  return "";
}

// One trivial dispatch: fill kernel writes a pattern, we copy it back.
static void test_dispatch(vk::Context& c, vk::Allocator& a,
                          const std::string& spv_path) {
  std::printf("== dispatch: %s ==\n", spv_path.c_str());

  vk::ComputePipeline p;
  std::string err;
  vk::PipelineDesc pd;
  pd.spv_path = spv_path;
  pd.storage_bindings = 1;
  pd.push_constant_bytes = 8;  // {n, value}
  CHECK(load_compute_pipeline(c, pd, p, err));
  if (!p) {
    std::printf("  pipeline failed: %s\n", err.c_str());
    return;
  }

  const uint32_t words = 1u << 20;  // 4 MiB of u32
  vk::Buffer dev = a.alloc_device_local(words * 4);
  vk::Buffer back = a.alloc_host(words * 4);
  CHECK(dev.buf && back.buf && back.ptr);

  vk::DescSet ds;
  CHECK(alloc_descset(c, p.dsl, 1, ds, err));
  CHECK(write_storage(c, ds, 0, dev.buf, 0, words * 4, err));

  std::memset(back.ptr, 0, words * 4);

  vk::CmdPool cp;
  CHECK(cp.init(c, c.compute_family));
  {
    vk::Recorder r;
    CHECK(r.begin(cp));
    r.bind_pipeline(p.pipe);
    r.bind_descset(p.layout, ds.set);
    struct PC {
      uint32_t n, value;
    } pc{words, 0x2a2a2a2au};
    r.push_constants(p.layout, &pc, sizeof(pc));
    r.dispatch((words + 255) / 256);
    r.compute_barrier();
    r.copy_buffer(dev.buf, back.buf, words * 4);
    CHECK(r.submit_and_wait(c.compute_queue));
    r.shutdown();
  }
  cp.shutdown();
  a.invalidate(back);

  const uint32_t* w = (const uint32_t*)back.ptr;
  uint32_t bad = 0;
  for (uint32_t i = 0; i < words; i++)
    if (w[i] != 0x2a2a2a2au) {
      bad = w[i];
      std::printf("  first bad word at %u: 0x%08x\n", i, bad);
      break;
    }
  CHECK(bad == 0);
  if (!bad) std::printf("  %u words filled with 0x2a2a2a2a OK\n", words);

  ds.shutdown(c);
  p.shutdown();
  a.free(dev);
  a.free(back);
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // crash-safe reporting
  vk::Context c;
  if (!c.init()) {
    std::printf("context init failed: %s\n", c.last_error.c_str());
    return 1;
  }
  report_context(c);

  vk::Allocator alloc;
  if (!alloc.init(c)) {
    std::printf("allocator init failed\n");
    return 1;
  }

  // BAR-visible heap should exist on this ReBAR box; report, don't require.
  {
    vk::Buffer bar = alloc.alloc_bar_visible(1 << 20);
    if (bar.buf && bar.ptr)
      std::printf("BAR-visible alloc OK: type %u ptr %p\n", bar.memtype,
                  bar.ptr);
    else
      std::printf("BAR-visible alloc unavailable (non-ReBAR box?)\n");
    if (bar.buf) alloc.free(bar);
  }

  test_copy_roundtrip(c, alloc);

  std::string spv = find_spv(argc, argv);
  if (spv.empty())
    std::printf("== dispatch: no .spv found, skipping (graceful) ==\n");
  else
    test_dispatch(c, alloc, spv);

  alloc.shutdown();
  c.shutdown();

  std::printf(fails ? "vk_smoke: %d FAILURES\n" : "vk_smoke: PASS\n", fails);
  return fails ? 1 : 0;
}
