#include "safetensors.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fstream>
#endif

namespace ink {

static bool dtype_from_name(const std::string& s, DType* t) {
  if (s == "F32") *t = DType::F32;
  else if (s == "BF16") *t = DType::BF16;
  else if (s == "F8_E4M3") *t = DType::F8E4M3;
  else if (s == "U8") *t = DType::U8;
  else if (s == "I8") *t = DType::I8;
  else return false;
  return true;
}

bool SafetensorsFile::open(const std::filesystem::path& p, std::string* err) {
  close();
  path_ = p;
  const std::byte* view = nullptr;
  uint64_t file_size = 0;
#ifdef _WIN32
  file_ = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file_ == INVALID_HANDLE_VALUE) {
    *err = "CreateFile failed: " + p.string();
    file_ = nullptr;
    return false;
  }
  LARGE_INTEGER sz;
  if (!GetFileSizeEx(file_, &sz)) {
    *err = "GetFileSizeEx failed";
    return false;
  }
  file_size = uint64_t(sz.QuadPart);
  mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mapping_) {
    *err = "CreateFileMapping failed";
    return false;
  }
  view_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
  if (!view_) {
    *err = "MapViewOfFile failed";
    return false;
  }
  view = static_cast<const std::byte*>(view_);
#else
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) {
    *err = "open failed: " + p.string();
    return false;
  }
  file_size = uint64_t(f.tellg());
  buf_.resize(size_t(file_size));
  f.seekg(0);
  if (!f.read(reinterpret_cast<char*>(buf_.data()), std::streamsize(file_size))) {
    *err = "read failed";
    return false;
  }
  view = buf_.data();
#endif

  if (file_size < 8) {
    *err = "too small for header";
    return false;
  }
  uint64_t hlen;
  std::memcpy(&hlen, view, 8);  // LE u64; hosts here are little-endian
  if (hlen > file_size - 8 || hlen > 256ull << 20) {
    *err = "bad header length";
    return false;
  }
  nlohmann::json hdr;
  try {
    hdr = nlohmann::json::parse(reinterpret_cast<const char*>(view + 8),
                                reinterpret_cast<const char*>(view + 8 + hlen));
  } catch (const std::exception& e) {
    *err = std::string("header JSON: ") + e.what();
    return false;
  }
  blob_ = view + 8 + hlen;
  const uint64_t blob_size = file_size - 8 - hlen;

  for (auto& [name, e] : hdr.items()) {
    if (name == "__metadata__") continue;
    try {
      DType dt;
      if (!dtype_from_name(e.at("dtype").get<std::string>(), &dt)) {
        *err = "unsupported dtype in " + name;
        return false;
      }
      std::vector<int64_t> shape;
      for (auto& d : e.at("shape")) shape.push_back(d.get<int64_t>());
      auto offs = e.at("data_offsets").get<std::pair<uint64_t, uint64_t>>();
      if (offs.second < offs.first || offs.second > blob_size) {
        *err = "offsets out of range in " + name;
        return false;
      }
      TensorView tv = TensorView::make(dt, blob_ + offs.first, std::move(shape));
      if (offs.second - offs.first != uint64_t(tv.nbytes())) {
        *err = "size mismatch in " + name;
        return false;
      }
      tensors_.emplace(name, tv);
    } catch (const std::exception& ex) {
      *err = "bad entry " + name + ": " + ex.what();
      return false;
    }
  }
  return true;
}

void SafetensorsFile::close() {
  tensors_.clear();
  blob_ = nullptr;
#ifdef _WIN32
  if (view_) { UnmapViewOfFile(view_); view_ = nullptr; }
  if (mapping_) { CloseHandle(mapping_); mapping_ = nullptr; }
  if (file_) { CloseHandle(file_); file_ = nullptr; }
#else
  buf_.clear();
#endif
}

const TensorView& SafetensorsFile::at(const std::string& name) const {
  const TensorView* t = find(name);
  if (!t) throw std::runtime_error("tensor not found: " + name);
  return *t;
}

std::unordered_map<std::string, DTypeStat> SafetensorsFile::census() const {
  std::unordered_map<std::string, DTypeStat> m;
  for (auto& [n, t] : tensors_) {
    auto& s = m[dtype_name(t.dtype)];
    s.tensors++;
    s.bytes += t.nbytes();
  }
  return m;
}

bool WeightStore::open_dir(const std::filesystem::path& dir,
                           const std::string& skip, std::string* err) {
  std::vector<std::filesystem::path> files;
  for (auto& de : std::filesystem::directory_iterator(dir)) {
    if (!de.is_regular_file()) continue;
    if (de.path().extension() != ".safetensors") continue;
    if (de.path().filename().string() == skip) continue;
    files.push_back(de.path());
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    *err = "no .safetensors in " + dir.string();
    return false;
  }
  for (auto& p : files) {
    auto f = std::make_unique<SafetensorsFile>();
    if (!f->open(p, err)) return false;
    for (auto& [n, t] : f->tensors()) index_[n] = &t;
    files_.push_back(std::move(f));
  }
  return true;
}

const TensorView* WeightStore::find(const std::string& name) const {
  auto it = index_.find(name);
  return it == index_.end() ? nullptr : it->second;
}

std::unordered_map<std::string, DTypeStat> WeightStore::census() const {
  std::unordered_map<std::string, DTypeStat> m;
  for (auto& f : files_)
    for (auto& [n, t] : f->tensors()) {
      auto& s = m[dtype_name(t.dtype)];
      s.tensors++;
      s.bytes += t.nbytes();
    }
  return m;
}

int64_t WeightStore::tensor_count() const { return int64_t(index_.size()); }

}  // namespace ink
