#include "io/config.h"
#include "io/safetensors.h"
#include "model/graph.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ink;

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
                 "[--ctx N] [--token-ids a,b,c]\n");
    return 1;
  }
  std::string dir = argv[1];
  std::string prompt;
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

  if (!prompt.empty())
    std::printf("note: BPE tokenizer not wired in this milestone; ignoring "
                "--prompt, use --token-ids a,b,c\n");
  if (ids.empty()) ids = {200028};  // <|begin_of_text|>

  int64_t pos = 0;
  for (int t : ids) {
    g.forward_token(t, pos++);
    int am = ops::argmax(g.state().logits.data(),
                         int(g.state().logits.size()));
    std::printf("pos %lld prompt tok %d -> argmax %d\n", (long long)(pos - 1),
                t, am);
  }
  int cur = ops::argmax(g.state().logits.data(),
                        int(g.state().logits.size()));
  for (int t = 0; t < gen_tokens; t++) {
    const std::vector<float>& logits = g.forward_token(cur, pos++);
    int am = ops::argmax(logits.data(), int(logits.size()));
    float lp = logits[am];
    std::printf("gen %d: token %d (logit %.3f)\n", t, am, lp);
    if (am == cfg.eos_token_id) break;
    cur = am;
  }
  return 0;
}
