// cmd.h — command pools, one-shot submit, and a Recorder that accumulates
// dispatch/copy ops into a prerecorded command buffer and submits once.
#pragma once

#include "vk/vk.h"

namespace ink::vk {

struct CmdPool {
  VkCommandPool pool = VK_NULL_HANDLE;
  uint32_t family = ~0u;
  Context* ctx = nullptr;

  bool init(Context& c, uint32_t queue_family,
            VkCommandPoolCreateFlags flags =
                VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
  void shutdown();
  explicit operator bool() const { return pool != VK_NULL_HANDLE; }

  VkCommandBuffer alloc(
      VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);
  void free_cmd(VkCommandBuffer cb);
  void reset();  // vkResetCommandPool — frees all recordings' contents
};

// One-shot helper: record cmds on a pool of the right family, submit to the
// queue, wait. Convenience over the Recorder for single-shot work.
bool run_oneshot(Context& ctx, VkQueue q, uint32_t family,
                 bool (*record)(VkCommandBuffer, void*), void* arg);

// Accumulates bind/dispatch/copy/barrier ops into one command buffer, then
// submits once to a chosen queue. Direction: prerecorded buffers — the same
// recording can be re-submitted via submit() while the pool stays alive.
class Recorder {
 public:
  bool begin(CmdPool& pool);  // allocs + begins the command buffer
  bool begin(Context& ctx,
             uint32_t family);  // convenience: makes its own pool

  void bind_pipeline(VkPipeline p);
  void bind_descset(VkPipelineLayout layout, VkDescriptorSet set,
                    uint32_t first_set = 0);
  void push_constants(VkPipelineLayout layout, const void* data,
                      uint32_t bytes, uint32_t offset = 0,
                      VkShaderStageFlags stages =
                          VK_SHADER_STAGE_COMPUTE_BIT);
  void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);
  // SDMA path: vkCmdCopyBuffer on whatever queue this is submitted to.
  void copy_buffer(VkBuffer src, VkBuffer dst, VkDeviceSize size,
                   VkDeviceSize src_off = 0, VkDeviceSize dst_off = 0);
  // vkCmdFillBuffer (TRANSFER write) — e.g. zeroing an accumulator mid-cb.
  void fill_buffer(VkBuffer dst, VkDeviceSize off, VkDeviceSize size,
                   uint32_t data = 0);
  void barrier(VkPipelineStageFlags src_stage,
               VkPipelineStageFlags dst_stage, VkAccessFlags src_access,
               VkAccessFlags dst_access);
  void compute_barrier();  // SHADER_WRITE -> SHADER_READ|TRANSFER_READ

  // Timestamp instrumentation (INK_GPU_TS): record a timestamp write and a
  // query-pool reset into the command buffer. The pool is caller-owned.
  void write_timestamp(VkQueryPool pool, uint32_t slot,
                       VkPipelineStageFlags stage =
                           VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
  void reset_query_pool(VkQueryPool pool, uint32_t first, uint32_t count);

  bool end();
  bool submit(VkQueue q);                    // submit recorded buffer, no wait
  bool submit_and_wait(VkQueue q);           // end() if needed + fence wait
  bool end_and_submit(VkQueue q) { return submit_and_wait(q); }
  bool wait(uint64_t timeout_ns = UINT64_MAX);  // wait on the last submit fence

  VkCommandBuffer cb() const { return cb_; }
  bool recording() const { return recording_; }
  const std::string& error() const { return err_; }
  void shutdown();  // frees cb + owned pool (from the 2-arg begin)

 private:
  Context* ctx_ = nullptr;
  CmdPool* pool_ = nullptr;
  CmdPool own_pool_;          // used when begin(ctx, family) overload is taken
  bool owns_pool_ = false;
  VkCommandBuffer cb_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  bool recording_ = false;
  bool ended_ = false;
  std::string err_;
  bool fail(const char* m) { err_ = m; return false; }
};

}  // namespace ink::vk
