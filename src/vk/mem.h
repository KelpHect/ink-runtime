// mem.h — slab/arena allocator over plain vkAllocateMemory (no VMA).
// One VkDeviceMemory slab per ~64 MiB of each memory type; buffers are bump /
// first-fit suballocations. All buffer usages get TRANSFER_SRC|DST +
// STORAGE_BUFFER so staging and SSBO paths share one type.
#pragma once

#include "vk/vk.h"

#include <vector>

namespace ink::vk {

struct Buffer {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;  // owning slab (not 1:1 with buf)
  VkDeviceSize size = 0;                // requested bytes
  VkDeviceSize offset = 0;              // suballocation offset inside mem
  uint32_t memtype = ~0u;
  void* ptr = nullptr;                  // mapped base + offset, else nullptr
  uint32_t slab = ~0u;                  // Allocator-internal handle
};

class Allocator {
 public:
  // slab_size: arena granularity; bigger requests get a dedicated slab.
  bool init(Context& ctx, VkDeviceSize slab_size = 64ull << 20);
  void shutdown();  // frees every slab + buffer
  explicit operator bool() const { return ctx_ != nullptr; }

  // DEVICE_LOCAL weight/activation memory (not necessarily host-visible).
  Buffer alloc_device_local(VkDeviceSize size, VkBufferUsageFlags extra = 0);
  // Staging: HOST_VISIBLE; coherent preferred + required when coherent=true
  // and no non-coherent fallback exists.
  Buffer alloc_host(VkDeviceSize size, bool coherent = true,
                    VkBufferUsageFlags extra = 0);
  // DEVICE_LOCAL|HOST_VISIBLE — the ReBAR BAR heap on this box.
  Buffer alloc_bar_visible(VkDeviceSize size, VkBufferUsageFlags extra = 0);

  void free(Buffer& b);
  bool ok(const Buffer& b) const { return b.buf != VK_NULL_HANDLE; }

  // Mapped access (null when not host-visible). flush/invalidate are no-ops
  // on HOST_COHERENT types but keep the API for non-coherent paths.
  void* mapped(const Buffer& b) const { return b.ptr; }
  bool flush(const Buffer& b, VkDeviceSize off = 0,
             VkDeviceSize size = VK_WHOLE_SIZE);
  bool invalidate(const Buffer& b, VkDeviceSize off = 0,
                  VkDeviceSize size = VK_WHOLE_SIZE);

  // VK_EXT_memory_budget (returns false when the ext is off).
  bool heap_budget(uint32_t heap, VkDeviceSize& budget, VkDeviceSize& usage);
  VkDeviceSize slab_bytes() const;  // total vkAllocateMemory footprint

  std::string last_error;

 private:
  struct Block {
    VkDeviceSize off, size;
    bool free;
  };
  struct Slab {
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    uint32_t memtype = ~0u;
    void* base = nullptr;  // whole-slab map when HOST_VISIBLE
    std::vector<Block> blocks;
  };

  Context* ctx_ = nullptr;
  VkDeviceSize slab_size_ = 0;
  std::vector<Slab> slabs_;

  uint32_t find_type(uint32_t type_bits, VkMemoryPropertyFlags required,
                     VkMemoryPropertyFlags preferred, bool& got_preferred);
  Buffer alloc(VkDeviceSize size, VkMemoryPropertyFlags required,
               VkMemoryPropertyFlags preferred, VkBufferUsageFlags usage);
  bool fail(const char* msg) { last_error = msg; return false; }
};

}  // namespace ink::vk
