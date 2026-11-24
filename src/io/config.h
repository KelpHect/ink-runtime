#pragma once

#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace ink {

struct LoopConfig {
  int core_start = 6;
  int core_end = 18;     // exclusive
  int num_passes = 3;
  int gate_rank = 64;
  bool is_varloop = false;  // must be false for this checkpoint (07 §12.10)
};

struct MTPConfig {
  int hidden_size = 2304;
  int num_hidden_layers = 3;
  int num_attention_heads = 24;
  int num_key_value_heads = 8;
  int head_dim = 128;
  int intermediate_size = 9216;
  int64_t max_position_embeddings = 1048576;
  double rope_theta = 1e6;
  double rms_norm_eps = 1e-6;
  std::vector<int> local_layer_ids;
  int sliding_window_size = 1024;
  int block_size = 8;
  int markov_rank = 256;
  std::string markov_head_type;
  std::vector<int> target_layer_ids;  // virtual visit ids {5,23,41,48}
};

struct VisionConfig {
  int hidden_size = 1024;
  int num_hidden_layers = 23;
  int num_attention_heads = 16;
  double mlp_ratio = 4.0;
  int native_patch_size = 20;
  int temporal_patch_size = 2;
  int adapter_merge_size = 2;
  int posemb_grid_size = 32;
  int projection_hidden_dim = 4096;
  int text_hidden_size = 2304;
  double layer_norm_eps = 1e-5;
};

struct ModelConfig {
  // top level
  int eos_token_id = 200006;
  std::string model_type;

  // text_config
  int64_t model_max_length = 1048576;
  int hidden_size = 2304;
  int num_hidden_layers = 24;
  int vocab_size = 201024;
  int unpadded_vocab_size = 200058;
  int num_attention_heads = 24;
  int num_key_value_heads = 8;
  int head_dim = 128;
  int d_rel = 16;
  int rel_extent = 1024;
  double log_scaling_n_floor = 8192;
  double log_scaling_alpha = 0.1;
  double rms_norm_eps = 1e-6;
  bool use_value_norm_no_gamma = true;
  double qk_l2_clip_threshold = 22.627416997969522;
  double rel_l2_clip_threshold = 8.0;
  std::vector<int> local_layer_ids;
  int sconv_kernel_size = 4;
  int swa_head_dim = 128;
  int swa_num_attention_heads = 24;
  int swa_num_key_value_heads = 8;
  int sliding_window_size = 1024;
  int n_layers_per_attn_res_block = 6;
  bool attn_res_normalize_values = true;
  int n_routed_experts = 128;
  int num_experts_per_tok = 8;
  int n_shared_experts = 0;
  int intermediate_size = 1152;
  double mlp_gate_softcap = 4.0;
  double mlp_up_softcap = 25.0;
  double route_scale = 8.0;
  bool use_gate_bias = true;
  double final_logit_softcapping = 30.0;
  LoopConfig loop;

  MTPConfig mtp;
  VisionConfig vision;

  // From hf_quant_config.json: which expert-store format each layer uses.
  std::set<int> nvfp4_layers;  // weight_block_shape [1,16], amax_denominator
  std::set<int> mxfp8_layers;  // weight_block_shape [32,32] (per-row 32-group scales)
  int nvfp4_scale_block = 16;
  int mxfp8_scale_block = 32;

  int total_visits() const {
    return num_hidden_layers + (loop.num_passes - 1) * (loop.core_end - loop.core_start);
  }
  bool is_full_attn(int layer) const {
    return (layer + 1) % n_layers_per_attn_res_block == 0;
  }
  bool is_loop(int layer) const {
    return layer >= loop.core_start && layer < loop.core_end;
  }
  // Virtual KV/conv cache slot for physical layer i on pass p (07 §3):
  // pass 0 keeps physical indices, loop passes land at 24+(p-1)*12+(i-6),
  // coda keeps 18..23. 48 distinct slots for this checkpoint.
  int cache_slot(int layer, int pass) const {
    if (pass == 0) return layer;
    return num_hidden_layers + (pass - 1) * (loop.core_end - loop.core_start) +
           (layer - loop.core_start);
  }
  // Visit index in the 48-application order: prelude 0..5, loop
  // 6+p*12+(i-6), coda i+24 → 42..47 (for MTP taps {5,23,41,48}).
  int visit_index(int layer, int pass) const {
    if (layer < loop.core_start) return layer;
    if (layer < loop.core_end)
      return loop.core_start + pass * (loop.core_end - loop.core_start) +
             (layer - loop.core_start);
    return layer + (loop.num_passes - 1) * (loop.core_end - loop.core_start);
  }

  bool load(const std::filesystem::path& model_dir, std::string* err);
};

}  // namespace ink
