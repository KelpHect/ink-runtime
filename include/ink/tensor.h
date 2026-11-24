#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

namespace ink {

// Storage dtypes. U8_NVFP4 is a runtime tag for a U8 buffer holding 2 E2M1
// values per byte (low nibble = even element); it never appears on disk.
enum class DType : uint8_t { F32, BF16, F8E4M3, U8_NVFP4, U8, I8 };

inline const char* dtype_name(DType t) {
  switch (t) {
    case DType::F32: return "F32";
    case DType::BF16: return "BF16";
    case DType::F8E4M3: return "F8_E4M3";
    case DType::U8_NVFP4: return "U8_NVFP4";
    case DType::U8: return "U8";
    case DType::I8: return "I8";
  }
  return "?";
}

// Bytes per storage unit (for U8_NVFP4 one unit = one byte = 2 values).
inline int dtype_size(DType t) {
  switch (t) {
    case DType::F32: return 4;
    case DType::BF16: return 2;
    default: return 1;
  }
}

// Logical values per storage unit (U8_NVFP4 packs 2 E2M1 per byte).
inline int dtype_elems_per_unit(DType t) {
  return t == DType::U8_NVFP4 ? 2 : 1;
}

// bf16 <-> f32 (round-to-nearest-even on the way down).
inline float bf16_to_f32(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}
// Round-trip through bf16 (precision-discipline helper, see research/07 §12).
inline float bf16_round(float f) { return bf16_to_f32(f32_to_bf16(f)); }

// F8E4M3 (fn variant: no inf, 0x7F/0xFF = NaN).
inline float f8e4m3_to_f32(uint8_t v) {
  int s = v >> 7, e = (v >> 3) & 0xF, m = v & 7;
  float x;
  if (e == 0) x = std::ldexp(float(m) / 8.f, -6);
  else x = std::ldexp(1.f + float(m) / 8.f, e - 7);
  return s ? -x : x;
}

// E2M1 nibble decode (NVFP4 payload).
inline float e2m1_to_f32(uint8_t nib) {
  static constexpr float t[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                  -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  return t[nib & 15];
}

// UE8M0 power-of-two scale (MXFP8). 0xFF is NaN in the spec; unused here.
inline float ue8m0_to_f32(uint8_t v) { return std::ldexp(1.f, int(v) - 127); }

// Non-owning dense row-major tensor view over a storage-unit buffer.
// shape/stride are in storage units (for U8_NVFP4 the last dim counts bytes).
struct TensorView {
  DType dtype = DType::F32;
  const std::byte* data = nullptr;
  std::array<int64_t, 4> shape{1, 1, 1, 1};
  std::array<int64_t, 4> stride{1, 1, 1, 1};  // in storage units
  int nd = 0;

  int64_t numel() const {
    int64_t n = 1;
    for (int i = 0; i < nd; i++) n *= shape[i];
    return n;
  }
  int64_t nbytes() const { return numel() * dtype_size(dtype); }

  static TensorView make(DType t, const void* p, std::vector<int64_t> sh) {
    TensorView v;
    v.dtype = t;
    v.data = static_cast<const std::byte*>(p);
    v.nd = int(sh.size());
    for (int i = 0; i < 4; i++) v.shape[i] = i < v.nd ? sh[i] : 1;
    for (int i = v.nd - 1; i >= 0; i--)
      v.stride[i] = (i == v.nd - 1) ? 1 : v.stride[i + 1] * v.shape[i + 1];
    return v;
  }

  const std::byte* ptr(int64_t i = 0, int64_t j = 0, int64_t k = 0) const {
    int64_t off = i * stride[0] + j * stride[1] + k * stride[2];
    return data + off * dtype_size(dtype);
  }

  // Flat-index decode (contiguous storage order).
  float f32_linear(int64_t idx) const {
    const std::byte* p = data + idx * dtype_size(dtype);
    switch (dtype) {
      case DType::F32: { float f; std::memcpy(&f, p, 4); return f; }
      case DType::BF16: { uint16_t u; std::memcpy(&u, p, 2); return bf16_to_f32(u); }
      case DType::F8E4M3: return f8e4m3_to_f32(uint8_t(*p));
      case DType::U8_NVFP4:
      case DType::U8: return float(uint8_t(*p));
      case DType::I8: return float(int8_t(*p));
    }
    return 0.f;
  }

  // Element decode for the plain dtypes (U8_NVFP4: use nvfp4_elem).
  float f32(int64_t i, int64_t j = 0, int64_t k = 0) const {
    const std::byte* p = ptr(i, j, k);
    switch (dtype) {
      case DType::F32: { float f; std::memcpy(&f, p, 4); return f; }
      case DType::BF16: { uint16_t u; std::memcpy(&u, p, 2); return bf16_to_f32(u); }
      case DType::F8E4M3: return f8e4m3_to_f32(uint8_t(*p));
      case DType::U8_NVFP4:
      case DType::U8: return float(uint8_t(*p));
      case DType::I8: return float(int8_t(*p));
    }
    return 0.f;
  }

  // j indexes unpacked values; 2 per byte, low nibble = even j.
  static float nvfp4_elem(const std::byte* packed_row, int64_t j) {
    uint8_t b = uint8_t(packed_row[j >> 1]);
    return e2m1_to_f32((j & 1) ? (b >> 4) : (b & 15));
  }
};

}  // namespace ink
