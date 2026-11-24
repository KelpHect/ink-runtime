// vk.cpp — context bring-up for the Vulkan compute backend.
#include "vk/vk.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ink::vk {

namespace {

bool failf(std::string& out, const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  out = buf;
  return false;
}

bool has_ext(const std::vector<VkExtensionProperties>& exts, const char* name) {
  for (const auto& e : exts)
    if (std::strcmp(e.extensionName, name) == 0) return true;
  return false;
}

}  // namespace

bool Context::fail(const char* msg) {
  last_error = msg;
  return false;
}

bool Context::fail(VkResult r, const char* what) {
  return failf(last_error, "%s: %s (%d)", what, result_name(r), (int)r);
}

const char* result_name(VkResult r) {
  switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
    default: return "VkResult";
  }
}

bool Context::init() {
  if (VkResult r = volkInitialize(); r != VK_SUCCESS)
    return fail(r, "volkInitialize (is vulkan-1.dll on PATH?)");

  // ---- instance ----
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "ink";
  app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  app.pEngineName = "ink";
  app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  app.apiVersion = VK_API_VERSION_1_3;

  std::vector<const char*> layers;
#ifndef NDEBUG
  uint32_t nlay = 0;
  vkEnumerateInstanceLayerProperties(&nlay, nullptr);
  if (nlay) {
    std::vector<VkLayerProperties> lp(nlay);
    vkEnumerateInstanceLayerProperties(&nlay, lp.data());
    for (const auto& l : lp)
      if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0)
        layers.push_back("VK_LAYER_KHRONOS_validation");
  }
#endif

  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  ici.enabledLayerCount = (uint32_t)layers.size();
  ici.ppEnabledLayerNames = layers.data();

  if (VkResult r = vkCreateInstance(&ici, nullptr, &instance); r != VK_SUCCESS)
    return fail(r, "vkCreateInstance");
  volkLoadInstance(instance);

  // ---- physical device: prefer discrete AMD (0x1002), then any discrete ----
  uint32_t ndev = 0;
  vkEnumeratePhysicalDevices(instance, &ndev, nullptr);
  if (ndev == 0) return fail("no Vulkan physical devices");
  std::vector<VkPhysicalDevice> devs(ndev);
  vkEnumeratePhysicalDevices(instance, &ndev, devs.data());

  int best = -1, best_score = -1;
  for (uint32_t i = 0; i < ndev; i++) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(devs[i], &p);
    int score = 0;
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 100;
    if (p.vendorID == 0x1002) score += 200;  // AMD
    if (score > best_score) { best_score = score; best = (int)i; }
  }
  phys = devs[best];
  vkGetPhysicalDeviceProperties(phys, &props);
  vkGetPhysicalDeviceMemoryProperties(phys, &memprops);
  if (props.apiVersion < VK_API_VERSION_1_2)
    return fail("device API < 1.2 (need timeline semaphores / BDA core)");

  // ---- queue families ----
  uint32_t nq = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
  std::vector<VkQueueFamilyProperties> qf(nq);
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());

  // Compute: prefer a COMPUTE family without GRAPHICS (dedicated ACE), else
  // the first COMPUTE-capable family.
  for (uint32_t i = 0; i < nq; i++) {
    const auto f = qf[i].queueFlags;
    if ((f & VK_QUEUE_COMPUTE_BIT) && !(f & VK_QUEUE_GRAPHICS_BIT) &&
        qf[i].queueCount > 0) {
      compute_family = i;
      break;
    }
  }
  if (compute_family == ~0u)
    for (uint32_t i = 0; i < nq; i++)
      if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].queueCount > 0) {
        compute_family = i;
        break;
      }
  if (compute_family == ~0u) return fail("no compute-capable queue family");

  // Dedicated TRANSFER-only family (SDMA): TRANSFER set, GRAPHICS+COMPUTE clear.
  for (uint32_t i = 0; i < nq; i++) {
    const auto f = qf[i].queueFlags;
    if ((f & VK_QUEUE_TRANSFER_BIT) && !(f & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) &&
        qf[i].queueCount > 0) {
      transfer_family = i;
      break;
    }
  }

  // ---- device extensions + feature chain ----
  uint32_t next = 0;
  vkEnumerateDeviceExtensionProperties(phys, nullptr, &next, nullptr);
  std::vector<VkExtensionProperties> exts(next);
  vkEnumerateDeviceExtensionProperties(phys, nullptr, &next, exts.data());

  const bool ext_coopmat = has_ext(exts, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  const bool ext_f8 = has_ext(exts, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
  const bool ext_bf16 = has_ext(exts, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
  const bool ext_budget = has_ext(exts, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

  // Query: chain every feature struct we might enable; unsupported stay 0.
  VkPhysicalDeviceTimelineSemaphoreFeatures ts{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
  VkPhysicalDeviceBufferDeviceAddressFeatures bda{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceShaderFloat8FeaturesEXT f8{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
  VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR};

  ts.pNext = &bda;
  void** link = &bda.pNext;
  // Only chain extension feature structs when the extension is advertised.
  if (ext_coopmat) { *link = &cm; link = &cm.pNext; }
  if (ext_f8) { *link = &f8; link = &f8.pNext; }
  if (ext_bf16) { *link = &bf16; link = &bf16.pNext; }
  *link = nullptr;

  VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  query.pNext = &ts;
  vkGetPhysicalDeviceFeatures2(phys, &query);

  feats.timeline_semaphore = ts.timelineSemaphore == VK_TRUE;
  feats.buffer_device_address = bda.bufferDeviceAddress == VK_TRUE;
  feats.cooperative_matrix = ext_coopmat && cm.cooperativeMatrix == VK_TRUE;
  feats.shader_float8 = ext_f8 && f8.shaderFloat8 == VK_TRUE;
  feats.shader_bfloat16 = ext_bf16 && bf16.shaderBFloat16Type == VK_TRUE;

  // Enable: same chain, bits set only for features we turn on.
  VkPhysicalDeviceTimelineSemaphoreFeatures ts_on{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
  VkPhysicalDeviceBufferDeviceAddressFeatures bda_on{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm_on{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceShaderFloat8FeaturesEXT f8_on{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
  VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16_on{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR};

  ts_on.timelineSemaphore = feats.timeline_semaphore ? VK_TRUE : VK_FALSE;
  bda_on.bufferDeviceAddress = feats.buffer_device_address ? VK_TRUE : VK_FALSE;
  cm_on.cooperativeMatrix = feats.cooperative_matrix ? VK_TRUE : VK_FALSE;
  // Robust buffer access keeps coopmat reads in-bounds; enable if reported.
  cm_on.cooperativeMatrixRobustBufferAccess =
      feats.cooperative_matrix && cm.cooperativeMatrixRobustBufferAccess ? VK_TRUE : VK_FALSE;
  f8_on.shaderFloat8 = feats.shader_float8 ? VK_TRUE : VK_FALSE;
  f8_on.shaderFloat8CooperativeMatrix =
      feats.shader_float8 && f8.shaderFloat8CooperativeMatrix ? VK_TRUE : VK_FALSE;
  bf16_on.shaderBFloat16Type = feats.shader_bfloat16 ? VK_TRUE : VK_FALSE;
  bf16_on.shaderBFloat16DotProduct =
      feats.shader_bfloat16 && bf16.shaderBFloat16DotProduct ? VK_TRUE : VK_FALSE;
  bf16_on.shaderBFloat16CooperativeMatrix =
      feats.shader_bfloat16 && bf16.shaderBFloat16CooperativeMatrix ? VK_TRUE : VK_FALSE;

  ts_on.pNext = &bda_on;
  link = &bda_on.pNext;
  if (feats.cooperative_matrix) { *link = &cm_on; link = &cm_on.pNext; }
  if (feats.shader_float8) { *link = &f8_on; link = &f8_on.pNext; }
  if (feats.shader_bfloat16) { *link = &bf16_on; link = &bf16_on.pNext; }
  *link = nullptr;

  std::vector<const char*> dev_exts;
  if (feats.cooperative_matrix) dev_exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  if (feats.shader_float8) dev_exts.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
  if (feats.shader_bfloat16) dev_exts.push_back(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
  if (ext_budget) dev_exts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

  // ---- logical device: queues on compute family + dedicated transfer ----
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci[2]{};
  uint32_t nqci = 0;
  qci[nqci].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci[nqci].queueFamilyIndex = compute_family;
  qci[nqci].queueCount = 1;
  qci[nqci].pQueuePriorities = &prio;
  nqci++;
  if (has_dedicated_transfer() && transfer_family != compute_family) {
    qci[nqci].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci[nqci].queueFamilyIndex = transfer_family;
    qci[nqci].queueCount = 1;
    qci[nqci].pQueuePriorities = &prio;
    nqci++;
  }

  VkPhysicalDeviceFeatures2 enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  enable.pNext = &ts_on;

  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &enable;
  dci.queueCreateInfoCount = nqci;
  dci.pQueueCreateInfos = qci;
  dci.enabledExtensionCount = (uint32_t)dev_exts.size();
  dci.ppEnabledExtensionNames = dev_exts.data();

  if (VkResult r = vkCreateDevice(phys, &dci, nullptr, &device); r != VK_SUCCESS)
    return fail(r, "vkCreateDevice");
  volkLoadDevice(device);

  vkGetDeviceQueue(device, compute_family, 0, &compute_queue);
  if (has_dedicated_transfer())
    vkGetDeviceQueue(device, transfer_family, 0, &transfer_queue);

  memory_budget_ext = ext_budget;
  if (memory_budget_ext) refresh_budget();
  return true;
}

void Context::refresh_budget() {
  if (!memory_budget_ext) return;
  VkPhysicalDeviceMemoryBudgetPropertiesEXT b{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 mp{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
  mp.pNext = &b;
  vkGetPhysicalDeviceMemoryProperties2(phys, &mp);
  budget = b;
}

void Context::shutdown() {
  if (device) {
    vkDeviceWaitIdle(device);
    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;
  }
  if (instance) {
    vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
  }
  phys = VK_NULL_HANDLE;
  compute_queue = transfer_queue = VK_NULL_HANDLE;
  compute_family = transfer_family = ~0u;
}

bool Context::submit_and_wait(VkQueue q, VkCommandBuffer cb, uint64_t timeout_ns) {
  VkFence f = create_fence();
  if (!f) return false;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  VkResult r = vkQueueSubmit(q, 1, &si, f);
  if (r != VK_SUCCESS) {
    destroy_fence(f);
    return fail(r, "vkQueueSubmit");
  }
  bool ok = wait_fence(f, timeout_ns);
  destroy_fence(f);
  return ok;
}

VkFence Context::create_fence(bool signaled) {
  VkFenceCreateInfo ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (signaled) ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VkFence f = VK_NULL_HANDLE;
  if (vkCreateFence(device, &ci, nullptr, &f) != VK_SUCCESS) {
    fail("vkCreateFence");
    return VK_NULL_HANDLE;
  }
  return f;
}

bool Context::wait_fence(VkFence f, uint64_t timeout_ns) {
  VkResult r = vkWaitForFences(device, 1, &f, VK_TRUE, timeout_ns);
  if (r != VK_SUCCESS) return fail(r, "vkWaitForFences");
  return true;
}

void Context::destroy_fence(VkFence f) {
  if (f) vkDestroyFence(device, f, nullptr);
}

VkSemaphore Context::create_timeline_semaphore(uint64_t initial) {
  VkSemaphoreTypeCreateInfo tci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  tci.initialValue = initial;
  VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  ci.pNext = &tci;
  VkSemaphore s = VK_NULL_HANDLE;
  if (vkCreateSemaphore(device, &ci, nullptr, &s) != VK_SUCCESS) {
    fail("vkCreateSemaphore (timeline)");
    return VK_NULL_HANDLE;
  }
  return s;
}

bool Context::timeline_wait(VkSemaphore sem, uint64_t value, uint64_t timeout_ns) {
  VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wi.semaphoreCount = 1;
  wi.pSemaphores = &sem;
  wi.pValues = &value;
  VkResult r = vkWaitSemaphores(device, &wi, timeout_ns);
  if (r != VK_SUCCESS) return fail(r, "vkWaitSemaphores");
  return true;
}

bool Context::timeline_signal(VkSemaphore sem, uint64_t value) {
  VkSemaphoreSignalInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
  si.semaphore = sem;
  si.value = value;
  VkResult r = vkSignalSemaphore(device, &si);
  if (r != VK_SUCCESS) return fail(r, "vkSignalSemaphore");
  return true;
}

uint64_t Context::timeline_value(VkSemaphore sem) {
  uint64_t v = 0;
  vkGetSemaphoreCounterValue(device, sem, &v);
  return v;
}

void Context::destroy_semaphore(VkSemaphore s) {
  if (s) vkDestroySemaphore(device, s, nullptr);
}

bool Context::submit_timeline(VkQueue q, VkCommandBuffer cb,
                              const VkSemaphore* wait_sems,
                              const uint64_t* wait_vals, uint32_t nwait,
                              const VkSemaphore* sig_sems,
                              const uint64_t* sig_vals, uint32_t nsig,
                              VkFence fence) {
  VkTimelineSemaphoreSubmitInfo ti{
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  ti.waitSemaphoreValueCount = nwait;
  ti.pWaitSemaphoreValues = wait_vals;
  ti.signalSemaphoreValueCount = nsig;
  ti.pSignalSemaphoreValues = sig_vals;

  std::vector<VkPipelineStageFlags> stages(nwait, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.pNext = &ti;
  si.waitSemaphoreCount = nwait;
  si.pWaitSemaphores = wait_sems;
  si.pWaitDstStageMask = stages.data();
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  si.signalSemaphoreCount = nsig;
  si.pSignalSemaphores = sig_sems;

  VkResult r = vkQueueSubmit(q, 1, &si, fence);
  if (r != VK_SUCCESS) return fail(r, "vkQueueSubmit (timeline)");
  return true;
}

}  // namespace ink::vk
