// pipe.h — .spv loading, compute pipeline + descriptor set plumbing.
// Boring on purpose: every descriptor is a STORAGE_BUFFER, set 0, N bindings.
#pragma once

#include "vk/vk.h"

#include <string>
#include <vector>

namespace ink::vk {

struct ComputePipeline {
  VkShaderModule mod = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipe = VK_NULL_HANDLE;
  uint32_t push_bytes = 0;
  Context* ctx = nullptr;

  void shutdown();
  explicit operator bool() const { return pipe != VK_NULL_HANDLE; }
};

struct PipelineDesc {
  std::string spv_path;         // runtime path to a compiled .spv
  uint32_t storage_bindings = 1;  // N STORAGE_BUFFER bindings in set 0
  uint32_t push_constant_bytes = 0;  // one push-constant range, compute stage
  const char* entry = "main";
};

// Reads the .spv file, builds descriptor-set layout + pipeline layout +
// compute pipeline. err gets a human-readable failure reason.
bool load_compute_pipeline(Context& ctx, const PipelineDesc& desc,
                           ComputePipeline& out, std::string& err);

// Descriptor set: its own small pool, exactly one set of dsl.
struct DescSet {
  VkDescriptorPool pool = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;
  void shutdown(Context& ctx);
  explicit operator bool() const { return set != VK_NULL_HANDLE; }
};

bool alloc_descset(Context& ctx, VkDescriptorSetLayout dsl,
                   uint32_t n_storage_bindings, DescSet& out,
                   std::string& err);

// Point a binding at (a slice of) a buffer.
bool write_storage(Context& ctx, const DescSet& ds, uint32_t binding,
                   VkBuffer buf, VkDeviceSize offset, VkDeviceSize size,
                   std::string& err);

// Read a whole .spv file into a word-aligned vector. Separate so callers can
// sanity-check headers before making a pipeline.
bool read_spirv(const std::string& path, std::vector<uint32_t>& words,
                std::string& err);

}  // namespace ink::vk
