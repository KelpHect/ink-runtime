// cmd.cpp — command pools, one-shot submit, Recorder.
#include "vk/cmd.h"

namespace ink::vk {

bool CmdPool::init(Context& c, uint32_t queue_family,
                   VkCommandPoolCreateFlags flags) {
  ctx = &c;
  family = queue_family;
  VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  ci.flags = flags;
  ci.queueFamilyIndex = queue_family;
  VkResult r = vkCreateCommandPool(c.device, &ci, nullptr, &pool);
  if (r != VK_SUCCESS) return c.fail(r, "vkCreateCommandPool");
  return true;
}

void CmdPool::shutdown() {
  if (ctx && pool) vkDestroyCommandPool(ctx->device, pool, nullptr);
  pool = VK_NULL_HANDLE;
}

VkCommandBuffer CmdPool::alloc(VkCommandBufferLevel level) {
  VkCommandBufferAllocateInfo ai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool;
  ai.level = level;
  ai.commandBufferCount = 1;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(ctx->device, &ai, &cb) != VK_SUCCESS) {
    ctx->fail("vkAllocateCommandBuffers");
    return VK_NULL_HANDLE;
  }
  return cb;
}

void CmdPool::free_cmd(VkCommandBuffer cb) {
  if (cb) vkFreeCommandBuffers(ctx->device, pool, 1, &cb);
}

void CmdPool::reset() {
  vkResetCommandPool(ctx->device, pool, 0);
}

bool run_oneshot(Context& ctx, VkQueue q, uint32_t family,
                 bool (*record)(VkCommandBuffer, void*), void* arg) {
  CmdPool pool;
  if (!pool.init(ctx, family, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT)) {
    ctx.last_error = "run_oneshot: pool init failed";
    return false;
  }
  VkCommandBuffer cb = pool.alloc();
  if (!cb) {
    pool.shutdown();
    return false;
  }
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cb, &bi);
  bool ok = record(cb, arg);
  vkEndCommandBuffer(cb);
  if (ok) ok = ctx.submit_and_wait(q, cb);
  pool.shutdown();
  return ok;
}

// ---- Recorder ----

bool Recorder::begin(CmdPool& pool) {
  ctx_ = pool.ctx;
  pool_ = &pool;
  owns_pool_ = false;
  cb_ = pool.alloc();
  if (!cb_) {
    err_ = pool.ctx->last_error;
    return false;
  }
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  if (vkBeginCommandBuffer(cb_, &bi) != VK_SUCCESS)
    return fail("vkBeginCommandBuffer");
  recording_ = true;
  ended_ = false;
  return true;
}

bool Recorder::begin(Context& ctx, uint32_t family) {
  ctx_ = &ctx;
  if (!own_pool_.init(ctx, family)) {
    err_ = ctx.last_error;
    return false;
  }
  owns_pool_ = true;
  pool_ = &own_pool_;
  cb_ = pool_->alloc();
  if (!cb_) {
    err_ = ctx.last_error;
    return false;
  }
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  if (vkBeginCommandBuffer(cb_, &bi) != VK_SUCCESS)
    return fail("vkBeginCommandBuffer");
  recording_ = true;
  ended_ = false;
  return true;
}

void Recorder::bind_pipeline(VkPipeline p) {
  vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, p);
}

void Recorder::bind_descset(VkPipelineLayout layout, VkDescriptorSet set,
                            uint32_t first_set) {
  vkCmdBindDescriptorSets(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                          first_set, 1, &set, 0, nullptr);
}

void Recorder::push_constants(VkPipelineLayout layout, const void* data,
                              uint32_t bytes, uint32_t offset,
                              VkShaderStageFlags stages) {
  vkCmdPushConstants(cb_, layout, stages, offset, bytes, data);
}

void Recorder::dispatch(uint32_t x, uint32_t y, uint32_t z) {
  vkCmdDispatch(cb_, x, y, z);
}

void Recorder::copy_buffer(VkBuffer src, VkBuffer dst, VkDeviceSize size,
                           VkDeviceSize src_off, VkDeviceSize dst_off) {
  VkBufferCopy r{src_off, dst_off, size};
  vkCmdCopyBuffer(cb_, src, dst, 1, &r);
}

void Recorder::fill_buffer(VkBuffer dst, VkDeviceSize off, VkDeviceSize size,
                           uint32_t data) {
  vkCmdFillBuffer(cb_, dst, off, size, data);
}

void Recorder::barrier(VkPipelineStageFlags src_stage,
                       VkPipelineStageFlags dst_stage,
                       VkAccessFlags src_access, VkAccessFlags dst_access) {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = src_access;
  mb.dstAccessMask = dst_access;
  vkCmdPipelineBarrier(cb_, src_stage, dst_stage, 0, 1, &mb, 0, nullptr, 0,
                       nullptr);
}

void Recorder::compute_barrier() {
  barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
              VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT,
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);
}

void Recorder::write_timestamp(VkQueryPool pool, uint32_t slot,
                               VkPipelineStageFlags stage) {
  vkCmdWriteTimestamp(cb_, static_cast<VkPipelineStageFlagBits>(stage), pool,
                      slot);
}

void Recorder::reset_query_pool(VkQueryPool pool, uint32_t first,
                                uint32_t count) {
  vkCmdResetQueryPool(cb_, pool, first, count);
}

bool Recorder::end() {
  if (!recording_) return fail("Recorder::end without begin");
  if (ended_) return true;
  if (vkEndCommandBuffer(cb_) != VK_SUCCESS)
    return fail("vkEndCommandBuffer");
  recording_ = false;
  ended_ = true;
  return true;
}

bool Recorder::submit(VkQueue q) {
  if (!ended_ && !end()) return false;
  if (!fence_) fence_ = ctx_->create_fence();
  if (!fence_) {
    err_ = ctx_->last_error;
    return false;
  }
  vkResetFences(ctx_->device, 1, &fence_);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb_;
  if (vkQueueSubmit(q, 1, &si, fence_) != VK_SUCCESS)
    return fail("vkQueueSubmit");
  return true;
}

bool Recorder::submit_and_wait(VkQueue q) {
  if (!ended_ && !end()) return false;
  VkFence f = ctx_->create_fence();
  if (!f) {
    err_ = ctx_->last_error;
    return false;
  }
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb_;
  VkResult r = vkQueueSubmit(q, 1, &si, f);
  if (r != VK_SUCCESS) {
    ctx_->destroy_fence(f);
    err_ = std::string("vkQueueSubmit ") + result_name(r);
    return false;
  }
  bool ok = ctx_->wait_fence(f);
  if (!ok) err_ = ctx_->last_error;
  ctx_->destroy_fence(f);
  return ok;
}

bool Recorder::wait(uint64_t timeout_ns) {
  if (!fence_) return fail("Recorder::wait before submit");
  bool ok = ctx_->wait_fence(fence_, timeout_ns);
  if (!ok) err_ = ctx_->last_error;
  return ok;
}

void Recorder::shutdown() {
  if (ctx_) {
    if (fence_) ctx_->destroy_fence(fence_);
    fence_ = VK_NULL_HANDLE;
    if (cb_ && pool_) pool_->free_cmd(cb_);
    cb_ = VK_NULL_HANDLE;
    if (owns_pool_) own_pool_.shutdown();
  }
  owns_pool_ = false;
  pool_ = nullptr;
  ctx_ = nullptr;
  recording_ = ended_ = false;
}

}  // namespace ink::vk
