#include "config.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <regex>
#include <sstream>

namespace ink {

template <typename T>
static void get_to(const nlohmann::json& j, const char* k, T& v) {
  auto it = j.find(k);
  if (it != j.end() && !it->is_null()) it->get_to(v);
}

static bool load_json(const std::filesystem::path& p, nlohmann::json* j,
                      std::string* err) {
  std::ifstream f(p);
  if (!f) {
    *err = "cannot open " + p.string();
    return false;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  try {
    *j = nlohmann::json::parse(ss.str());
  } catch (const std::exception& e) {
    *err = p.string() + ": " + e.what();
    return false;
  }
  return true;
}

bool ModelConfig::load(const std::filesystem::path& dir, std::string* err) {
  nlohmann::json cj;
  if (!load_json(dir / "config.json", &cj, err)) return false;

  get_to(cj, "eos_token_id", eos_token_id);
  get_to(cj, "model_type", model_type);

  const nlohmann::json& t = cj.contains("text_config") ? cj["text_config"] : cj;
  get_to(t, "model_max_length", model_max_length);
  get_to(t, "hidden_size", hidden_size);
  get_to(t, "num_hidden_layers", num_hidden_layers);
  get_to(t, "vocab_size", vocab_size);
  get_to(t, "unpadded_vocab_size", unpadded_vocab_size);
  get_to(t, "num_attention_heads", num_attention_heads);
  get_to(t, "num_key_value_heads", num_key_value_heads);
  get_to(t, "head_dim", head_dim);
  get_to(t, "d_rel", d_rel);
  get_to(t, "rel_extent", rel_extent);
  get_to(t, "log_scaling_n_floor", log_scaling_n_floor);
  get_to(t, "log_scaling_alpha", log_scaling_alpha);
  get_to(t, "rms_norm_eps", rms_norm_eps);
  get_to(t, "use_value_norm_no_gamma", use_value_norm_no_gamma);
  get_to(t, "qk_l2_clip_threshold", qk_l2_clip_threshold);
  get_to(t, "rel_l2_clip_threshold", rel_l2_clip_threshold);
  get_to(t, "local_layer_ids", local_layer_ids);
  get_to(t, "sconv_kernel_size", sconv_kernel_size);
  get_to(t, "swa_head_dim", swa_head_dim);
  get_to(t, "swa_num_attention_heads", swa_num_attention_heads);
  get_to(t, "swa_num_key_value_heads", swa_num_key_value_heads);
  get_to(t, "sliding_window_size", sliding_window_size);
  get_to(t, "n_layers_per_attn_res_block", n_layers_per_attn_res_block);
  get_to(t, "attn_res_normalize_values", attn_res_normalize_values);
  get_to(t, "n_routed_experts", n_routed_experts);
  get_to(t, "num_experts_per_tok", num_experts_per_tok);
  get_to(t, "n_shared_experts", n_shared_experts);
  get_to(t, "intermediate_size", intermediate_size);
  get_to(t, "mlp_gate_softcap", mlp_gate_softcap);
  get_to(t, "mlp_up_softcap", mlp_up_softcap);
  get_to(t, "route_scale", route_scale);
  get_to(t, "use_gate_bias", use_gate_bias);
  get_to(t, "final_logit_softcapping", final_logit_softcapping);

  if (t.contains("loop_config")) {
    const nlohmann::json& l = t["loop_config"];
    get_to(l, "core_start", loop.core_start);
    get_to(l, "core_end", loop.core_end);
    get_to(l, "num_passes", loop.num_passes);
    get_to(l, "gate_rank", loop.gate_rank);
    get_to(l, "is_varloop", loop.is_varloop);
  }
  if (loop.is_varloop) {
    *err = "is_varloop checkpoints (loop_share_cache / loop_num_passes "
           "overrides) are not supported by this runtime";
    return false;
  }

  if (cj.contains("mtp_config")) {
    const nlohmann::json& m = cj["mtp_config"];
    get_to(m, "hidden_size", mtp.hidden_size);
    get_to(m, "num_hidden_layers", mtp.num_hidden_layers);
    get_to(m, "num_attention_heads", mtp.num_attention_heads);
    get_to(m, "num_key_value_heads", mtp.num_key_value_heads);
    get_to(m, "head_dim", mtp.head_dim);
    get_to(m, "intermediate_size", mtp.intermediate_size);
    get_to(m, "max_position_embeddings", mtp.max_position_embeddings);
    get_to(m, "rope_theta", mtp.rope_theta);
    get_to(m, "rms_norm_eps", mtp.rms_norm_eps);
    get_to(m, "local_layer_ids", mtp.local_layer_ids);
    get_to(m, "sliding_window_size", mtp.sliding_window_size);
    get_to(m, "block_size", mtp.block_size);
    get_to(m, "markov_rank", mtp.markov_rank);
    get_to(m, "markov_head_type", mtp.markov_head_type);
    get_to(m, "target_layer_ids", mtp.target_layer_ids);
  }

  if (cj.contains("vision_config")) {
    const nlohmann::json& v = cj["vision_config"];
    get_to(v, "hidden_size", vision.hidden_size);
    get_to(v, "num_hidden_layers", vision.num_hidden_layers);
    get_to(v, "num_attention_heads", vision.num_attention_heads);
    get_to(v, "mlp_ratio", vision.mlp_ratio);
    get_to(v, "native_patch_size", vision.native_patch_size);
    get_to(v, "temporal_patch_size", vision.temporal_patch_size);
    get_to(v, "adapter_merge_size", vision.adapter_merge_size);
    get_to(v, "posemb_grid_size", vision.posemb_grid_size);
    get_to(v, "projection_hidden_dim", vision.projection_hidden_dim);
    get_to(v, "text_hidden_size", vision.text_hidden_size);
    get_to(v, "layer_norm_eps", vision.layer_norm_eps);
  }

  // hf_quant_config.json: classify expert-store formats per layer.
  nlohmann::json qj;
  if (load_json(dir / "hf_quant_config.json", &qj, err)) {
    static const std::regex exre(R"(layers\.(\d+)\.mlp\.experts\.w(13|2)_weight$)");
    for (const auto& g : qj.at("model").at("parameter_groups")) {
      std::string cd = g.value("compute_dtype", "");
      bool nv = cd == "nvfp4", mx = cd == "mxfp8";
      if (!nv && !mx) continue;
      std::smatch m;
      for (const auto& tn : g.at("tensors")) {
        std::string name = tn.get<std::string>();
        if (std::regex_search(name, m, exre))
          (nv ? nvfp4_layers : mxfp8_layers).insert(std::stoi(m[1]));
      }
      if (nv && g.contains("required_activation_quantization"))
        get_to(g["required_activation_quantization"], "block_size",
               nvfp4_scale_block);
      if (mx && g.contains("required_activation_quantization"))
        get_to(g["required_activation_quantization"], "block_size",
               mxfp8_scale_block);
    }
  } else {
    // Not fatal: fall back to the documented 0..19 NVFP4 / 20..23 MXFP8 split.
    nvfp4_layers.clear();
    mxfp8_layers.clear();
    for (int i = 0; i < num_hidden_layers; i++)
      (i < 20 ? nvfp4_layers : mxfp8_layers).insert(i);
    err->clear();
  }
  return true;
}

}  // namespace ink
