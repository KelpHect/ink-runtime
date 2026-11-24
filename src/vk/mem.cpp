// mem.cpp — slab/arena allocator over vkAllocateMemory.
#include "vk/mem.h"

#include <cstring>

namespace ink::vk {

static VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) {
  return (v + a - 1) & ~(a - 1);
}

bool Allocator::init(Context& ctx, VkDeviceSize slab_size) {
  ctx_ = &ctx;
  slab_size_ = slab_size;
  return true;
}

void Allocator::shutdown() {
  if (!ctx_) return;
  VkDevice dev = ctx_->device;
  for (auto& s : slabs_) {
    if (s.base) vkUnmapMemory(dev, s.mem);
    if (s.mem) vkFreeMemory(dev, s.mem, nullptr);
    s.mem = VK_NULL_HANDLE;
  }
  slabs_.clear();
  ctx_ = nullptr;
}

uint32_t Allocator::find_type(uint32_t type_bits, VkMemoryPropertyFlags required,
                              VkMemoryPropertyFlags preferred,
                              bool& got_preferred) {
  got_preferred = false;
  const auto& mp = ctx_->memprops;
  uint32_t fallback = ~0u;
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if (!(type_bits & (1u << i))) continue;
    const auto f = mp.memoryTypes[i].propertyFlags;
    if ((f & required) != required) continue;
    if ((f & preferred) == preferred) {
      got_preferred = true;
      return i;
    }
    if (fallback == ~0u) fallback = i;
  }
  return fallback;
}

Buffer Allocator::alloc(VkDeviceSize size, VkMemoryPropertyFlags required,
                        VkMemoryPropertyFlags preferred,
                        VkBufferUsageFlags usage) {
  Buffer out;
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT |
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (ctx_->feats.buffer_device_address)
    bci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (VkResult r = vkCreateBuffer(ctx_->device, &bci, nullptr, &out.buf);
      r != VK_SUCCESS) {
    ctx_->fail(r, "vkCreateBuffer");
    last_error = ctx_->last_error;
    return out;
  }

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(ctx_->device, out.buf, &req);

  bool got_pref = false;
  uint32_t mt = find_type(req.memoryTypeBits, required, preferred, got_pref);
  if (mt == ~0u) {
    vkDestroyBuffer(ctx_->device, out.buf, nullptr);
    out.buf = VK_NULL_HANDLE;
    fail("no memory type satisfies required flags");
    return out;
  }

  // Suballocation alignment: memory reqs + storage-buffer offset alignment.
  VkDeviceSize a = req.alignment;
  const VkDeviceSize sbo = ctx_->props.limits.minStorageBufferOffsetAlignment;
  if (sbo > a) a = sbo;
  if (a < 16) a = 16;
  if ((a & (a - 1)) != 0) a = align_up(a, 16);  // paranoia: alignment is pow2

  // First-fit across existing slabs of this memory type.
  for (uint32_t si = 0; si < slabs_.size(); si++) {
    Slab& s = slabs_[si];
    if (s.memtype != mt) continue;
    for (size_t bi = 0; bi < s.blocks.size(); bi++) {
      if (!s.blocks[bi].free) continue;
      VkDeviceSize off = align_up(s.blocks[bi].off, a);
      VkDeviceSize pad = off - s.blocks[bi].off;
      if (pad + size > s.blocks[bi].size) continue;
      // Split the free run into [pad lead-in][this][tail].
      const VkDeviceSize tail = s.blocks[bi].size - pad - size;
      const VkDeviceSize lead_off = s.blocks[bi].off;
      const VkDeviceSize tail_off = off + size;
      std::vector<Block> nb;
      nb.reserve(s.blocks.size() + 2);
      for (size_t j = 0; j < s.blocks.size(); j++) {
        if (j == bi) {
          if (pad) nb.push_back(Block{lead_off, pad, true});
          nb.push_back(Block{off, size, false});
          if (tail) nb.push_back(Block{tail_off, tail, true});
        } else {
          nb.push_back(s.blocks[j]);
        }
      }
      s.blocks = std::move(nb);
      if (VkResult r =
              vkBindBufferMemory(ctx_->device, out.buf, s.mem, off);
          r != VK_SUCCESS) {
        // Roll back: mark the carved block free and coalesce with neighbours.
        for (size_t k = 0; k < s.blocks.size(); k++) {
          if (s.blocks[k].off == off && s.blocks[k].size == size) {
            s.blocks[k].free = true;
            while (k + 1 < s.blocks.size() && s.blocks[k + 1].free) {
              s.blocks[k].size += s.blocks[k + 1].size;
              s.blocks.erase(s.blocks.begin() + (long long)k + 1);
            }
            while (k > 0 && s.blocks[k - 1].free) {
              s.blocks[k - 1].size += s.blocks[k].size;
              s.blocks.erase(s.blocks.begin() + (long long)k);
              k--;
            }
            break;
          }
        }
        vkDestroyBuffer(ctx_->device, out.buf, nullptr);
        out.buf = VK_NULL_HANDLE;
        ctx_->fail(r, "vkBindBufferMemory");
        last_error = ctx_->last_error;
        return out;
      }
      out.mem = s.mem;
      out.size = size;
      out.offset = off;
      out.memtype = mt;
      out.slab = si;
      out.ptr = s.base ? (void*)((char*)s.base + off) : nullptr;
      return out;
    }
  }

  // New slab for this memory type.
  Slab s;
  s.memtype = mt;
  s.size = slab_size_ > req.size + a ? slab_size_ : align_up(req.size + a, a);

  VkMemoryAllocateFlagsInfo afi{
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  afi.flags = ctx_->feats.buffer_device_address
                  ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
                  : 0;
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  if (afi.flags) mai.pNext = &afi;
  mai.allocationSize = s.size;
  mai.memoryTypeIndex = mt;
  if (VkResult r = vkAllocateMemory(ctx_->device, &mai, nullptr, &s.mem);
      r != VK_SUCCESS) {
    vkDestroyBuffer(ctx_->device, out.buf, nullptr);
    out.buf = VK_NULL_HANDLE;
    ctx_->fail(r, "vkAllocateMemory");
    last_error = ctx_->last_error;
    return out;
  }
  const auto tf = ctx_->memprops.memoryTypes[mt].propertyFlags;
  if (tf & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    if (vkMapMemory(ctx_->device, s.mem, 0, VK_WHOLE_SIZE, 0, &s.base) !=
        VK_SUCCESS)
      s.base = nullptr;
  }

  VkDeviceSize off = 0;  // fresh slab: alignment already satisfied
  s.blocks.push_back(Block{off, size, false});
  if (size < s.size)
    s.blocks.push_back(Block{size, s.size - size, true});

  uint32_t si = (uint32_t)slabs_.size();
  slabs_.push_back(s);
  if (VkResult r = vkBindBufferMemory(ctx_->device, out.buf,
                                      slabs_.back().mem, off);
      r != VK_SUCCESS) {
    if (slabs_.back().base) vkUnmapMemory(ctx_->device, slabs_.back().mem);
    vkFreeMemory(ctx_->device, slabs_.back().mem, nullptr);
    slabs_.pop_back();
    vkDestroyBuffer(ctx_->device, out.buf, nullptr);
    out.buf = VK_NULL_HANDLE;
    ctx_->fail(r, "vkBindBufferMemory");
    last_error = ctx_->last_error;
    return out;
  }
  out.mem = slabs_.back().mem;
  out.size = size;
  out.offset = off;
  out.memtype = mt;
  out.slab = si;
  out.ptr = slabs_.back().base
                ? (void*)((char*)slabs_.back().base + off)
                : nullptr;
  return out;
}

Buffer Allocator::alloc_device_local(VkDeviceSize size, VkBufferUsageFlags extra) {
  return alloc(size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
               extra);
}

Buffer Allocator::alloc_host(VkDeviceSize size, bool coherent,
                             VkBufferUsageFlags extra) {
  Buffer b = alloc(size, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                   coherent ? (VkMemoryPropertyFlags)VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                            : (VkMemoryPropertyFlags)VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                   extra);
  if (!b.buf) return b;
  const auto tf = ctx_->memprops.memoryTypes[b.memtype].propertyFlags;
  if (coherent && !(tf & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    // Caller asked for coherent; keep the buffer but flag in last_error only
    // if we truly failed — non-coherent still works via flush/invalidate.
  }
  return b;
}

Buffer Allocator::alloc_bar_visible(VkDeviceSize size, VkBufferUsageFlags extra) {
  return alloc(size,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, extra);
}

void Allocator::free(Buffer& b) {
  if (!b.buf) return;
  vkDestroyBuffer(ctx_->device, b.buf, nullptr);
  b.buf = VK_NULL_HANDLE;
  if (b.slab >= slabs_.size()) return;
  Slab& s = slabs_[b.slab];
  for (size_t i = 0; i < s.blocks.size(); i++) {
    if (s.blocks[i].off == b.offset && s.blocks[i].size == b.size) {
      s.blocks[i].free = true;
      // Coalesce with free neighbours (blocks stay offset-sorted).
      while (i + 1 < s.blocks.size() && s.blocks[i + 1].free) {
        s.blocks[i].size += s.blocks[i + 1].size;
        s.blocks.erase(s.blocks.begin() + (long long)i + 1);
      }
      while (i > 0 && s.blocks[i - 1].free) {
        s.blocks[i - 1].size += s.blocks[i].size;
        s.blocks.erase(s.blocks.begin() + (long long)i);
        i--;
      }
      break;
    }
  }
  b.mem = VK_NULL_HANDLE;
  b.ptr = nullptr;
}

bool Allocator::flush(const Buffer& b, VkDeviceSize off, VkDeviceSize size) {
  if (!b.mem) return fail("flush on null buffer");
  const auto tf = ctx_->memprops.memoryTypes[b.memtype].propertyFlags;
  if (tf & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) return true;
  const VkDeviceSize atom = ctx_->props.limits.nonCoherentAtomSize;
  VkDeviceSize lo = (b.offset + off) & ~(atom - 1);
  VkDeviceSize hi = align_up(b.offset + off +
                                 (size == VK_WHOLE_SIZE ? b.size - off : size),
                             atom);
  VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
  r.memory = b.mem;
  r.offset = lo;
  r.size = hi - lo;
  if (VkResult res = vkFlushMappedMemoryRanges(ctx_->device, 1, &r);
      res != VK_SUCCESS) {
    ctx_->fail(res, "vkFlushMappedMemoryRanges");
    last_error = ctx_->last_error;
    return false;
  }
  return true;
}

bool Allocator::invalidate(const Buffer& b, VkDeviceSize off, VkDeviceSize size) {
  if (!b.mem) return fail("invalidate on null buffer");
  const auto tf = ctx_->memprops.memoryTypes[b.memtype].propertyFlags;
  if (tf & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) return true;
  const VkDeviceSize atom = ctx_->props.limits.nonCoherentAtomSize;
  VkDeviceSize lo = (b.offset + off) & ~(atom - 1);
  VkDeviceSize hi = align_up(b.offset + off +
                                 (size == VK_WHOLE_SIZE ? b.size - off : size),
                             atom);
  VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
  r.memory = b.mem;
  r.offset = lo;
  r.size = hi - lo;
  if (VkResult res = vkInvalidateMappedMemoryRanges(ctx_->device, 1, &r);
      res != VK_SUCCESS) {
    ctx_->fail(res, "vkInvalidateMappedMemoryRanges");
    last_error = ctx_->last_error;
    return false;
  }
  return true;
}

bool Allocator::heap_budget(uint32_t heap, VkDeviceSize& budget,
                            VkDeviceSize& usage) {
  if (!ctx_->memory_budget_ext || heap >= VK_MAX_MEMORY_HEAPS) return false;
  ctx_->refresh_budget();
  budget = ctx_->budget.heapBudget[heap];
  usage = ctx_->budget.heapUsage[heap];
  return true;
}

VkDeviceSize Allocator::slab_bytes() const {
  VkDeviceSize t = 0;
  for (const auto& s : slabs_) t += s.size;
  return t;
}

}  // namespace ink::vk
