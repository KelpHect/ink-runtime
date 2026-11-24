// vk.h — Vulkan context: instance, device, queues, feature bits, submit/sync.
// Correctness-first plumbing for the RX 9070 XT compute backend.
#pragma once

#include <volk.h>

#include <cstdint>
#include <string>
#include <vector>

namespace ink::vk {

// Which optional device features were found AND enabled at device create.
struct FeatureSet {
  bool timeline_semaphore = false;
  bool buffer_device_address = false;
  bool cooperative_matrix = false;   // VK_KHR_cooperative_matrix
  bool shader_float8 = false;        // VK_EXT_shader_float8
  bool shader_bfloat16 = false;      // VK_KHR_shader_bfloat16
};

struct Context {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;

  // Queue family picks. transfer_family is ~0u when no dedicated TRANSFER-only
  // family exists; on the target box this is family 2 (flags=0xc, SDMA).
  uint32_t compute_family = ~0u;
  uint32_t transfer_family = ~0u;
  VkQueue compute_queue = VK_NULL_HANDLE;
  VkQueue transfer_queue = VK_NULL_HANDLE;  // null when transfer_family == ~0u

  FeatureSet feats;
  bool memory_budget_ext = false;

  VkPhysicalDeviceProperties props{};
  VkPhysicalDeviceMemoryProperties memprops{};
  // Zeroed unless VK_EXT_memory_budget is on. Snapshot from init; call
  // refresh_budget() for live numbers.
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};

  std::string last_error;

  bool init();      // false + last_error on failure
  void shutdown();  // safe on partial init
  explicit operator bool() const { return device != VK_NULL_HANDLE; }

  bool has_dedicated_transfer() const { return transfer_family != ~0u; }
  // Queue to use for buffer copies: SDMA when present, else compute.
  VkQueue copy_queue() const {
    return transfer_queue ? transfer_queue : compute_queue;
  }
  uint32_t copy_family() const {
    return has_dedicated_transfer() ? transfer_family : compute_family;
  }

  void refresh_budget();

  // One-shot submit + fence wait. Binary semaphores only; use the timeline
  // helpers below for value signalling.
  bool submit_and_wait(VkQueue q, VkCommandBuffer cb,
                       uint64_t timeout_ns = UINT64_MAX);

  // --- fence / timeline-semaphore utilities ---
  VkFence create_fence(bool signaled = false);
  bool wait_fence(VkFence f, uint64_t timeout_ns = UINT64_MAX);
  void destroy_fence(VkFence f);

  VkSemaphore create_timeline_semaphore(uint64_t initial = 0);
  bool timeline_wait(VkSemaphore sem, uint64_t value,
                     uint64_t timeout_ns = UINT64_MAX);
  bool timeline_signal(VkSemaphore sem, uint64_t value);  // host-side signal
  uint64_t timeline_value(VkSemaphore sem);
  void destroy_semaphore(VkSemaphore s);

  // Queue submit with timeline wait/signal values (entries may be null/0).
  bool submit_timeline(VkQueue q, VkCommandBuffer cb,
                       const VkSemaphore* wait_sems, const uint64_t* wait_vals,
                       uint32_t nwait, const VkSemaphore* sig_sems,
                       const uint64_t* sig_vals, uint32_t nsig,
                       VkFence fence = VK_NULL_HANDLE);

  bool fail(const char* msg);  // sets last_error, returns false
  bool fail(VkResult r, const char* what);
};

const char* result_name(VkResult r);

}  // namespace ink::vk
