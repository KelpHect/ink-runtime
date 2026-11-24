#include "io/config.h"
#include "io/safetensors.h"
#include "io/tokenizer.h"
#include "model/backend.h"
#include "model/graph.h"
#ifdef INK_WITH_VULKAN
#include "gpu/vk_backend.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ink;

// Write a flat f32 array for the golden-gate comparator (.bin, no header).
static bool dump_f32(const std::string& dir, const char* name, int step,
                     const float* p, size_t n) {
  char path[512];
  std::snprintf(path, sizeof(path), "%s\\%s_step%d.bin", dir.c_str(), name,
                step);
  FILE* f = std::fopen(path, "wb");
  if (!f) return false;
  size_t w = std::fwrite(p, sizeof(float), n, f);
  std::fclose(f);
  return w == n;
}

// Golden filenames mirror tools/dump_golden.py; compare last position only.
static void dump_step(const std::string& dir, const RunState& s, int step) {
  static const char* tapn[3] = {"hidden_visit5", "hidden_visit23",
                                "hidden_visit41"};
  for (int i = 0; i < 3; i++)
    if (!s.taps[i].empty())
      dump_f32(dir, tapn[i], step, s.taps[i].data(), s.taps[i].size());
  if (!s.h_fin.empty())
    dump_f32(dir, "hidden_final", step, s.h_fin.data(), s.h_fin.size());
  dump_f32(dir, "logits", step, s.logits.data(), s.logits.size());
}

static void print_census(const WeightStore& store) {
  // expectations from research/00 (verified census 2026-11-24)
  struct {
    const char* name;
    int64_t tensors;
    double gb;
  } want[] = {
      {"U8", 48, 10.32}, {"F8_E4M3", 48, 5.35}, {"BF16", 994, 3.85},
      {"F32", 154, 0.01},
  };
  auto c = store.census();
  std::printf("census: %lld tensors (expect 1244)\n",
              (long long)store.tensor_count());
  for (auto& w : want) {
    auto it = c.find(w.name);
    int64_t n = it == c.end() ? 0 : it->second.tensors;
    double gb = it == c.end() ? 0.0 : it->second.bytes / 1e9;
    bool ok = n == w.tensors && std::abs(gb - w.gb) < 0.2;
    std::printf("  %-8s %4lld tensors %6.2f GB   (expect %4lld ~%5.2f) %s\n",
                w.name, (long long)n, gb, (long long)w.tensors, w.gb,
                ok ? "OK" : "MISMATCH");
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: ink <model_dir> [--prompt \"...\"] [--tokens N] "
                 "[--ctx N] [--token-ids a,b,c] [--dump dir] "
                 "[--backend cpu|vk] [--shader-dir dir]\n");
    return 1;
  }
  std::string dir = argv[1];
  std::string prompt;
  std::string dump_dir;
  std::string backend = "cpu";
  std::string shader_dir =
#ifdef INK_SHADER_DIR
      INK_SHADER_DIR;
#else
      "shaders";
#endif
  int gen_tokens = 8;
  int ctx = 4096;
  std::vector<int> ids;
  for (int i = 2; i < argc; i++) {
    auto need = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", flag);
        exit(1);
      }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--prompt")) prompt = need("--prompt");
    else if (!std::strcmp(argv[i], "--dump")) dump_dir = need("--dump");
    else if (!std::strcmp(argv[i], "--backend")) backend = need("--backend");
    else if (!std::strcmp(argv[i], "--shader-dir"))
      shader_dir = need("--shader-dir");
    else if (!std::strcmp(argv[i], "--tokens")) gen_tokens = atoi(need("--tokens"));
    else if (!std::strcmp(argv[i], "--ctx")) ctx = atoi(need("--ctx"));
    else if (!std::strcmp(argv[i], "--token-ids")) {
      std::string s = need("--token-ids");
      size_t p = 0;
      while (p <= s.size()) {
        size_t c = s.find(',', p);
        ids.push_back(atoi(s.substr(p, c == std::string::npos
                                        ? c
                                        : c - p)
                               .c_str()));
        if (c == std::string::npos) break;
        p = c + 1;
      }
    }
  }
  (void)ctx;

  std::string err;
  ModelConfig cfg;
  if (!cfg.load(dir, &err)) {
    std::fprintf(stderr, "config: %s\n", err.c_str());
    return 1;
  }
  std::printf(
      "config: hidden=%d layers=%d visits=%d loop={%d..%d x%d r%d} "
      "experts=%d top=%d softcap=%.1f/%.1f route_scale=%.1f rel=%d/%d "
      "clip=%.2f/%.1f swa=%d sconv=%d vocab=%d/%d logit_softcap=%.0f eos=%d "
      "mtp_layers=%d taps={",
      cfg.hidden_size, cfg.num_hidden_layers, cfg.total_visits(),
      cfg.loop.core_start, cfg.loop.core_end, cfg.loop.num_passes,
      cfg.loop.gate_rank, cfg.n_routed_experts, cfg.num_experts_per_tok,
      cfg.mlp_gate_softcap, cfg.mlp_up_softcap, cfg.route_scale, cfg.d_rel,
      cfg.rel_extent, cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold,
      cfg.sliding_window_size, cfg.sconv_kernel_size, cfg.vocab_size,
      cfg.unpadded_vocab_size, cfg.final_logit_softcapping, cfg.eos_token_id,
      cfg.mtp.num_hidden_layers);
  for (size_t i = 0; i < cfg.mtp.target_layer_ids.size(); i++)
    std::printf("%s%d", i ? "," : "", cfg.mtp.target_layer_ids[i]);
  std::printf("} nvfp4_layers=%zu mxfp8_layers=%zu\n",
              cfg.nvfp4_layers.size(), cfg.mxfp8_layers.size());

  WeightStore store;
  if (!store.open_dir(dir, "mtp.safetensors", &err)) {
    std::fprintf(stderr, "weights: %s\n", err.c_str());
    return 1;
  }
  print_census(store);

  Graph g;
  if (!g.init(cfg, store, &err)) {
    std::fprintf(stderr, "graph init: %s\n", err.c_str());
    return 1;
  }
  std::printf("graph: all weights resolved\n");

#ifdef INK_WITH_VULKAN
  ink::gpu::VkBackend vk_backend;
  if (backend == "vk") {
    if (!vk_backend.init(shader_dir, err)) {
      std::fprintf(stderr, "vk backend: %s\n", err.c_str());
      return 1;
    }
    ops::backend_slot() = &vk_backend;
    std::printf("backend: vk (shader-dir %s)\n", shader_dir.c_str());
  } else
#endif
  {
    if (backend != "cpu")
      std::fprintf(stderr, "backend '%s' unknown, using cpu\n",
                   backend.c_str());
    std::printf("backend: cpu\n");
  }

  int cur_step = -1;  // visit-hook tag; -1 = don't write per-visit files
  if (!dump_dir.empty()) {
    g.visit_hook = [&](int vidx, int phase, const std::vector<float>& h) {
      if (cur_step < 0) return;
      char name[64];
      std::snprintf(name, sizeof(name), "visit%02d_%s", vidx,
                    phase ? "out" : "in");
      dump_f32(dump_dir, name, cur_step, h.data(), h.size());
    };
  }

  Tokenizer tok;
  bool tok_loaded = tok.load(dir, &err);
  if (!prompt.empty()) {
    if (!tok_loaded) {
      std::fprintf(stderr, "tokenizer: %s\n", err.c_str());
      return 1;
    }
    ids = tok.encode_chat_user(prompt);
    std::printf("prompt: %zu tokens\n", ids.size());
  } else if (ids.empty()) {
    ids = {200028};  // <|begin_of_text|>
  }

  int64_t pos = 0;
  size_t ip = 0;
  for (int t : ids) {
    if (++ip == ids.size()) cur_step = 0;  // tag last prompt token as step0
    g.forward_token(t, pos++);
    int am = ops::argmax(g.state().logits.data(),
                         int(g.state().logits.size()));
    std::printf("pos %lld prompt tok %d -> argmax %d\n", (long long)(pos - 1),
                t, am);
    if (!dump_dir.empty() && ip == ids.size())
      dump_step(dump_dir, g.state(), 0);  // step0 = last prompt position
  }
  int cur = ops::argmax(g.state().logits.data(),
                        int(g.state().logits.size()));
  std::vector<int> gen_ids;
  for (int t = 0; t < gen_tokens; t++) {
    cur_step = t + 1;
    const std::vector<float>& logits = g.forward_token(cur, pos++);
    int am = ops::argmax(logits.data(), int(logits.size()));
    float lp = logits[am];
    gen_ids.push_back(am);
    std::string txt = tok_loaded ? tok.decode({am}) : "";
    std::printf("gen %d: token %d (logit %.3f) %s\n", t, am, lp,
                txt.c_str());
    if (!dump_dir.empty()) dump_step(dump_dir, g.state(), t + 1);
    if (am == cfg.eos_token_id) break;
    cur = am;
  }
  if (tok_loaded && !gen_ids.empty())
    std::printf("text: %s\n", tok.decode(gen_ids).c_str());
  return 0;
}
