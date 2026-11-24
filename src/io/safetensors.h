#pragma once

#include "ink/tensor.h"

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ink {

struct DTypeStat {
  int64_t tensors = 0;
  int64_t bytes = 0;
};

// Memory-mapped safetensors file: `u64 LE header_len` + JSON dict
// {name: {dtype, shape, data_offsets}} + raw blob. Views point into the map.
class SafetensorsFile {
 public:
  ~SafetensorsFile() { close(); }
  SafetensorsFile() = default;
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;

  bool open(const std::filesystem::path& p, std::string* err);
  void close();

  const TensorView* find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
  }
  const TensorView& at(const std::string& name) const;  // throws
  const std::unordered_map<std::string, TensorView>& tensors() const { return tensors_; }
  std::unordered_map<std::string, DTypeStat> census() const;
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
  const std::byte* blob_ = nullptr;  // start of tensor data (after header)
  std::unordered_map<std::string, TensorView> tensors_;
#ifdef _WIN32
  void* file_ = nullptr;
  void* mapping_ = nullptr;
  const void* view_ = nullptr;
#else
  std::vector<std::byte> buf_;
#endif
};

// Opens every `*.safetensors` in `dir` except `skip` (e.g. mtp.safetensors).
class WeightStore {
 public:
  bool open_dir(const std::filesystem::path& dir, const std::string& skip,
                std::string* err);
  const TensorView* find(const std::string& name) const;
  std::unordered_map<std::string, DTypeStat> census() const;
  int64_t tensor_count() const;
  const std::vector<std::unique_ptr<SafetensorsFile>>& files() const { return files_; }

 private:
  std::vector<std::unique_ptr<SafetensorsFile>> files_;
  std::unordered_map<std::string, const TensorView*> index_;
};

}  // namespace ink
