// pipe.cpp — .spv loading, compute pipeline + descriptor set plumbing.
#include "vk/pipe.h"

#include <cstdio>
#include <vector>

namespace ink::vk {

bool read_spirv(const std::string& path, std::vector<uint32_t>& words,
                std::string& err) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    err = "read_spirv: cannot open " + path;
    return false;
  }
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n <= 0 || (n & 3) != 0) {
    std::fclose(f);
    err = "read_spirv: bad size in " + path;
    return false;
  }
  words.resize((size_t)n / 4);
  size_t got = std::fread(words.data(), 4, words.size(), f);
  std::fclose(f);
  if (got != words.size()) {
    err = "read_spirv: short read on " + path;
    return false;
  }
  if (words[0] != 0x07230203u) {
    err = "read_spirv: bad magic in " + path;
    return false;
  }
  return true;
}

void ComputePipeline::shutdown() {
  if (!ctx) return;
  VkDevice d = ctx->device;
  if (pipe) vkDestroyPipeline(d, pipe, nullptr);
  if (layout) vkDestroyPipelineLayout(d, layout, nullptr);
  if (dsl) vkDestroyDescriptorSetLayout(d, dsl, nullptr);
  if (mod) vkDestroyShaderModule(d, mod, nullptr);
  pipe = VK_NULL_HANDLE;
  layout = VK_NULL_HANDLE;
  dsl = VK_NULL_HANDLE;
  mod = VK_NULL_HANDLE;
  ctx = nullptr;
}

bool load_compute_pipeline(Context& ctx, const PipelineDesc& desc,
                           ComputePipeline& out, std::string& err) {
  std::vector<uint32_t> spv;
  if (!read_spirv(desc.spv_path, spv, err)) return false;

  if (desc.push_constant_bytes > ctx.props.limits.maxPushConstantsSize) {
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "push_constant_bytes %u > maxPushConstantsSize %u",
                  desc.push_constant_bytes, ctx.props.limits.maxPushConstantsSize);
    err = buf;
    return false;
  }

  VkShaderModuleCreateInfo mci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  mci.codeSize = spv.size() * 4;
  mci.pCode = spv.data();
  if (VkResult r = vkCreateShaderModule(ctx.device, &mci, nullptr, &out.mod);
      r != VK_SUCCESS) {
    err = "vkCreateShaderModule failed";
    return false;
  }
  out.ctx = &ctx;  // from here on, shutdown() cleans up on failure paths

  // Set 0: N bindings, all STORAGE_BUFFER, compute stage.
  std::vector<VkDescriptorSetLayoutBinding> binds(desc.storage_bindings);
  for (uint32_t i = 0; i < desc.storage_bindings; i++) {
    binds[i].binding = i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dli{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dli.bindingCount = (uint32_t)binds.size();
  dli.pBindings = binds.data();
  if (vkCreateDescriptorSetLayout(ctx.device, &dli, nullptr, &out.dsl) !=
      VK_SUCCESS) {
    err = "vkCreateDescriptorSetLayout failed";
    out.shutdown();
    return false;
  }

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = desc.push_constant_bytes;

  VkPipelineLayoutCreateInfo pli{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount = 1;
  pli.pSetLayouts = &out.dsl;
  pli.pushConstantRangeCount = desc.push_constant_bytes ? 1u : 0u;
  pli.pPushConstantRanges = desc.push_constant_bytes ? &pcr : nullptr;
  if (vkCreatePipelineLayout(ctx.device, &pli, nullptr, &out.layout) !=
      VK_SUCCESS) {
    err = "vkCreatePipelineLayout failed";
    out.shutdown();
    return false;
  }

  VkPipelineShaderStageCreateInfo st{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  st.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  st.module = out.mod;
  st.pName = desc.entry ? desc.entry : "main";

  VkComputePipelineCreateInfo cpi{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage = st;
  cpi.layout = out.layout;
  if (VkResult r = vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpi,
                                            nullptr, &out.pipe);
      r != VK_SUCCESS) {
    err = "vkCreateComputePipelines failed";
    out.shutdown();
    return false;
  }

  out.push_bytes = desc.push_constant_bytes;
  return true;
}

void DescSet::shutdown(Context& ctx) {
  if (pool) vkDestroyDescriptorPool(ctx.device, pool, nullptr);
  pool = VK_NULL_HANDLE;
  set = VK_NULL_HANDLE;
}

bool alloc_descset(Context& ctx, VkDescriptorSetLayout dsl,
                   uint32_t n_storage_bindings, DescSet& out,
                   std::string& err) {
  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = n_storage_bindings;

  VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pci.maxSets = 1;
  pci.poolSizeCount = 1;
  pci.pPoolSizes = &ps;
  if (vkCreateDescriptorPool(ctx.device, &pci, nullptr, &out.pool) !=
      VK_SUCCESS) {
    err = "vkCreateDescriptorPool failed";
    return false;
  }
  VkDescriptorSetAllocateInfo ai{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = out.pool;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &dsl;
  if (vkAllocateDescriptorSets(ctx.device, &ai, &out.set) != VK_SUCCESS) {
    err = "vkAllocateDescriptorSets failed";
    out.shutdown(ctx);
    return false;
  }
  return true;
}

bool write_storage(Context& ctx, const DescSet& ds, uint32_t binding,
                   VkBuffer buf, VkDeviceSize offset, VkDeviceSize size,
                   std::string& err) {
  if (!ds.set) {
    err = "write_storage on empty DescSet";
    return false;
  }
  VkDescriptorBufferInfo bi{buf, offset, size};
  VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  w.dstSet = ds.set;
  w.dstBinding = binding;
  w.dstArrayElement = 0;
  w.descriptorCount = 1;
  w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  w.pBufferInfo = &bi;
  vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
  return true;
}

}  // namespace ink::vk
