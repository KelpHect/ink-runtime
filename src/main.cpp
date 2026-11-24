#include "io/config.h"
#include "io/safetensors.h"
#include "io/tokenizer.h"
#include "model/backend.h"
#include "model/graph.h"
#ifdef INK_WITH_VULKAN
#include "gpu/gpu_draft.h"
#include "gpu/gpu_graph.h"
#include "gpu/gpu_prefill.h"
#include "gpu/vk_backend.h"
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

using namespace ink;

// Write a flat f32 array for the golden-gate comparator (.bin, no header).
static bool dump_f32(const std::string& dir, const char* name, int step,
                     const float* p, size_t n) {
  char path[512];
  static bool dir_made = false;
  if (!dir_made) {
    dir_made = true;
#ifdef _WIN32
    CreateDirectoryA(dir.c_str(), nullptr);  // fine if it already exists
#else
    mkdir(dir.c_str(), 0755);
#endif
  }
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
#ifdef _WIN32
  // Serialize heavy GPU runs across concurrent ink.exe processes (parallel
  // agent/test invocations would otherwise contend for ~13GB VRAM + ~20GB
  // mapped weights and starve the desktop). INK_NOLOCK=1 opts out.
  static HANDLE run_lock = nullptr;
  if (!std::getenv("INK_NOLOCK")) {
    run_lock = CreateMutexW(nullptr, FALSE, L"Local\\InkRunLock");
    if (run_lock && WaitForSingleObject(run_lock, 0) == WAIT_TIMEOUT) {
      std::fprintf(stderr, "ink: another instance holds the run lock; waiting...\n");
      WaitForSingleObject(run_lock, INFINITE);
    }
  }
  // Yield to the desktop under load; INK_NORMAL_PRI=1 restores normal.
  if (!std::getenv("INK_NORMAL_PRI"))
    SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
#endif
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: ink <model_dir> [--prompt \"...\"] [--tokens N] "
                 "[--ctx N] [--token-ids a,b,c] [--dump dir] "
                 "[--backend cpu|vk|gpu] [--shader-dir dir] "
                 "[--prefill-chunk N]\n");
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
  int prefill_chunk = 0;  // >0: gpu batched-chunk prefill rows per chunk
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
    else if (!std::strcmp(argv[i], "--prefill-chunk"))
      prefill_chunk = atoi(need("--prefill-chunk"));
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
  ink::vk::Context vkctx;
  ink::vk::Allocator vkalloc;
  ink::gpu::VkBackend vk_backend;
  ink::gpu::GpuGraph gpu_graph;
  ink::gpu::GpuPrefill gpu_prefill;
  ink::gpu::GpuDrafter gpu_drafter;
  ink::SafetensorsFile mtp_file;  // stays mapped for the drafter's lifetime
  ink::ops::CpuDraft cpu_drafter;
  char draft_mode = 0;            // INK_DRAFT: 'g' gpu-only, 'c' +cpu ref
  const bool gpu_mode = backend == "gpu";
  if (backend == "vk") {
    if (!vk_backend.init(shader_dir, err)) {
      std::fprintf(stderr, "vk backend: %s\n", err.c_str());
      return 1;
    }
    ops::backend_slot() = &vk_backend;
    std::printf("backend: vk (shader-dir %s)\n", shader_dir.c_str());
  } else if (gpu_mode) {
    if (!vkctx.init()) {
      std::fprintf(stderr, "vk context: %s\n", vkctx.last_error.c_str());
      return 1;
    }
    if (!vkalloc.init(vkctx)) {
      std::fprintf(stderr, "vk allocator: %s\n", vkalloc.last_error.c_str());
      return 1;
    }
    if (!gpu_graph.init(cfg, store, vkctx, vkalloc, shader_dir, ctx, &err)) {
      std::fprintf(stderr, "gpu backend: %s\n", err.c_str());
      return 1;
    }
    std::printf("backend: gpu (shader-dir %s, ctx %d)\n", shader_dir.c_str(),
                ctx);
    if (prefill_chunk > 0) {
      if (!gpu_prefill.init(gpu_graph, prefill_chunk, &err)) {
        std::fprintf(stderr, "gpu prefill: %s\n", err.c_str());
        return 1;
      }
      std::printf("prefill: chunked gpu (T=%d)\n", gpu_prefill.chunk());
    }
    // INK_DRAFT: env-gated MTP/DFlare draft head (proposal instrument only;
    // NOT wired into the decode loop). 'c' additionally runs the CPU
    // reference on the first anchor for a GPU-vs-CPU consistency check.
    if (const char* de = std::getenv("INK_DRAFT")) {
      draft_mode = (*de == 'c' || *de == 'C') ? 'c' : 'g';
      if (!mtp_file.open(std::filesystem::path(dir) / "mtp.safetensors",
                         &err)) {
        std::fprintf(stderr, "draft weights: %s\n", err.c_str());
        return 1;
      }
      const TensorView* tE = store.find("model.llm.embed.weight");
      const TensorView* tN = store.find("model.llm.embed_norm.weight");
      const TensorView* tU = store.find("model.llm.unembed.weight");
      if (!tE || !tN || !tU) {
        std::fprintf(stderr, "draft: embed/embed_norm/unembed missing\n");
        return 1;
      }
      if (!gpu_drafter.init(cfg, mtp_file, *tE, *tN, *tU, vkctx, vkalloc,
                            shader_dir, ctx, &err,
                            &gpu_graph.unembed_buffer())) {
        std::fprintf(stderr, "gpu draft: %s\n", err.c_str());
        return 1;
      }
      if (draft_mode == 'c' &&
          !cpu_drafter.init(cfg, mtp_file, *tE, *tN, *tU, &err)) {
        std::fprintf(stderr, "cpu draft: %s\n", err.c_str());
        return 1;
      }
      std::printf("draft head: gpu%s (ctx %d)\n",
                  draft_mode == 'c' ? " + cpu ref" : "", ctx);
    }
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

  // Backend-agnostic forward + dump plumbing. GpuGraph keeps state on the
  // device and exposes taps/visit dumps through its readback after the
  // single per-token submit; Graph exposes them via visit_hook / state().
  auto forward = [&](int tok, int64_t p) -> const std::vector<float>& {
#ifdef INK_WITH_VULKAN
    if (gpu_mode) return gpu_graph.forward_token(tok, p);
#endif
    return g.forward_token(tok, p);
  };
  auto last_logits = [&]() -> const std::vector<float>& {
#ifdef INK_WITH_VULKAN
    if (gpu_mode) return gpu_graph.logits();
#endif
    return g.state().logits;
  };
  auto dump_all = [&](int step) {
#ifdef INK_WITH_VULKAN
    if (gpu_mode) {
      // per-visit in/out for tools/compare_visits.py (visit%02d_*_stepN.bin)
      for (int v = 0; v < 48; v++) {
        char name[64];
        for (int ph = 0; ph < 2; ph++) {
          std::snprintf(name, sizeof(name), "visit%02d_%s", v,
                        ph ? "out" : "in");
          dump_f32(dump_dir, name, step, gpu_graph.visit_dump(v, ph),
                   size_t(cfg.hidden_size));
        }
      }
      static const char* tapn[3] = {"hidden_visit5", "hidden_visit23",
                                    "hidden_visit41"};
      for (int i = 0; i < 3; i++)
        dump_f32(dump_dir, tapn[i], step, gpu_graph.tap(i),
                 size_t(cfg.hidden_size));
      dump_f32(dump_dir, "hidden_final", step, gpu_graph.tap(3),
               size_t(cfg.hidden_size));
      dump_f32(dump_dir, "logits", step, gpu_graph.logits().data(),
               gpu_graph.logits().size());
      if (std::getenv("INK_DBG") && step == 0) {
        for (int r = 0; r < 15; r++) {
          char name[16];
          std::snprintf(name, sizeof(name), "dbg%d", r);
          // rows 0 and 4 hold 3072-wide vectors (q, gated o); rest are H-wide
          size_t n = (r == 0 || r == 4) ? 3072 : size_t(cfg.hidden_size);
          dump_f32("testdata/ink_dbg_gpu", name, 0, gpu_graph.dbg(r), n);
        }
        const float* r8 = gpu_graph.dbg(13);
        const int32_t* ii = reinterpret_cast<const int32_t*>(r8);
        const float* ww = r8 + 8;
        std::fprintf(stderr, "router idx:");
        for (int e = 0; e < 8; e++) std::fprintf(stderr, " %d", ii[e]);
        std::fprintf(stderr, " w:");
        for (int e = 0; e < 8; e++) std::fprintf(stderr, " %.4f", ww[e]);
        std::fprintf(stderr, "\n");
      }
      return;
    }
#endif
    dump_step(dump_dir, g.state(), step);
  };

  int64_t pos = 0;
  size_t ip = 0;
  bool prefilled = false;
#ifdef INK_WITH_VULKAN
  if (gpu_mode && prefill_chunk > 0 && ids.size() > 1) {
    // Batched-chunk prompt prefill: all P positions through all 48 visits
    // in T-row chunks inside a handful of submissions. Leaves kv/conv/readback
    // state identical to P sequential forward_token calls.
    const auto pw0 = std::chrono::steady_clock::now();
    gpu_graph.reset();
    if (!gpu_prefill.prefill(ids.data(), int64_t(ids.size()))) {
      std::fprintf(stderr,
                   "prefill failed (%s); falling back to per-token\n",
                   gpu_prefill.error().c_str());
      gpu_graph.reset();
    } else {
      prefilled = true;
      pos = int64_t(ids.size());
      cur_step = 0;
      const std::vector<float>& lg = gpu_graph.logits();
      int am = ops::argmax(lg.data(), int(lg.size()));
      std::printf("pos %lld prefill -> argmax %d\n", (long long)(pos - 1),
                  am);
      if (!dump_dir.empty()) dump_all(0);
      const double pw = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - pw0)
                            .count();
      std::printf("prefill wall: %.2fs for %zu tokens (%.1f tok/s)\n", pw,
                  ids.size(), ids.size() / std::max(pw, 1e-9));
    }
  }
#endif
  for (int t : ids) {
    if (prefilled) break;
    if (++ip == ids.size()) cur_step = 0;  // tag last prompt token as step0
    forward(t, pos++);
    const std::vector<float>& lg = last_logits();
    int am = ops::argmax(lg.data(), int(lg.size()));
    std::printf("pos %lld prompt tok %d -> argmax %d\n", (long long)(pos - 1),
                t, am);
    if (!dump_dir.empty() && ip == ids.size()) dump_all(0);
  }
  int cur = ops::argmax(last_logits().data(), int(last_logits().size()));
  auto wall0 = std::chrono::steady_clock::now();
  std::vector<int> gen_ids;
#ifdef INK_WITH_VULKAN
  // INK_DRAFT accept-rate bookkeeping: each propose() opens a window of 7
  // predicted positions (anchor+1 .. anchor+7, DFlash fill-in); every
  // generated token scores depth = q - anchor. ids[7] is an unscored spare.
  struct DraftWin {
    int64_t anchor;
    int ids[8];
    bool broken = false;  // first miss seen -> prefix stops growing
    int prefix = 0;
  };
  std::deque<DraftWin> dra_wins;
  int64_t dra_hits[9] = {}, dra_tot[9] = {}, dra_prefix_sum = 0,
          dra_nwin = 0;
  bool dra_cpu_checked = false;
#endif
  for (int t = 0; t < gen_tokens; t++) {
    cur_step = t + 1;
    const int64_t anchor_pos = pos;  // `cur` sits at this position
    const int anchor_tok = cur;
    const std::vector<float>& logits = forward(cur, pos++);
    int am = ops::argmax(logits.data(), int(logits.size()));
    float lp = logits[am];
    gen_ids.push_back(am);
    std::string txt = tok_loaded ? tok.decode({am}) : "";
    std::printf("gen %d: token %d (logit %.3f) %s\n", t, am, lp,
                txt.c_str());
    if (!dump_dir.empty()) dump_all(t + 1);
#ifdef INK_WITH_VULKAN
    if (gpu_drafter.ok()) {
      const int64_t q = pos;  // `am` landed at position q = anchor+1
      // score open windows against this actual token
      for (DraftWin& win : dra_wins) {
        const int depth = int(q - win.anchor);
        if (depth < 1 || depth > 7) continue;
        dra_tot[depth]++;
        if (win.ids[depth - 1] == am) {
          dra_hits[depth]++;
          if (!win.broken) win.prefix++;
        } else {
          win.broken = true;
        }
      }
      while (!dra_wins.empty() &&
             q - dra_wins.front().anchor >= 8) {
        dra_prefix_sum += dra_wins.front().prefix;
        dra_wins.pop_front();
      }
      // propose from the anchor just forwarded (taps are live now)
      std::vector<float> taps4(size_t(4) * cfg.hidden_size);
      for (int i = 0; i < 4; i++)
        std::memcpy(taps4.data() + size_t(i) * cfg.hidden_size,
                    gpu_graph.tap(i), size_t(cfg.hidden_size) * 4);
      if (std::getenv("INK_DRAFT_DEBUG") && dra_nwin < 6) {
        // per-window tap dumps for the numpy semantic harness
        char tp[64];
        std::snprintf(tp, sizeof(tp), "taps_w%lld_t%d",
                      (long long)anchor_pos, anchor_tok);
        dump_f32(dump_dir.empty() ? "." : dump_dir, tp, 0, taps4.data(),
                 taps4.size());
      }
      const std::vector<int> did =
          gpu_drafter.propose(taps4.data(), anchor_tok, anchor_pos);
      if (did.size() == 8) {
        DraftWin w;
        w.anchor = anchor_pos;
        for (int i = 0; i < 8; i++) w.ids[i] = did[i];
        dra_wins.push_back(w);
        dra_nwin++;
        // first proposal targets position q = anchor+1 (am, known
        // already): score it now as a head-fidelity check
        dra_tot[1]++;
        if (did[0] == am) {
          dra_hits[1]++;
          dra_wins.back().prefix = 1;
        } else {
          dra_wins.back().broken = true;
        }
        std::printf("draft %lld: [", (long long)anchor_pos);
        for (int i = 0; i < 8; i++)
          std::printf("%s%d", i ? "," : "", did[i]);
        std::printf("] (p1 %s)\n", did[0] == am ? "HIT" : "miss");
      }
      // GPU-vs-CPU consistency on the first anchor (INK_DRAFT=c)
      if (draft_mode == 'c' && !dra_cpu_checked && did.size() == 8) {
        dra_cpu_checked = true;
        const std::vector<int> cid =
            cpu_drafter.propose(taps4.data(), anchor_tok, anchor_pos);
        int same = 0;
        for (int i = 0; i < 8 && i < int(cid.size()); i++)
          same += cid[i] == did[i];
        std::printf("draft cpu-ref: [");
        for (size_t i = 0; i < cid.size(); i++)
          std::printf("%s%d", i ? "," : "", cid[i]);
        std::printf("] ids-match %d/8\n", same);
        if (gpu_drafter.dbg_ctx() && std::getenv("INK_DRAFT_DEBUG")) {
          const float* gc = gpu_drafter.dbg_ctx();
          const float* go = gpu_drafter.dbg_out();
          std::printf("draft dbg: ctx/out dumped (%d floats each)\n",
                      4 * cfg.hidden_size);
          dump_f32(dump_dir.empty() ? "." : dump_dir, "draft_ctx", 0, gc,
                   size_t(4) * cfg.hidden_size);
          dump_f32(dump_dir.empty() ? "." : dump_dir, "draft_out", 0, go,
                   size_t(8) * cfg.hidden_size);
          // taps sanity + GPU-vs-CPU max-abs on ctx / out8
          const int H = cfg.hidden_size;
          for (int i = 0; i < 4; i++) {
            double ss = 0;
            for (int d = 0; d < H; d++) {
              const float v = taps4[size_t(i) * H + d];
              ss += double(v) * v;
            }
            std::printf("draft dbg: tap%d rms=%.4f\n", i,
                        std::sqrt(ss / H));
          }
          auto maxdiff = [&](const float* a, const std::vector<float>& b,
                             const char* name) {
            if (b.empty()) return;
            double md = 0, ma = 0;
            for (size_t i = 0; i < b.size(); i++) {
              md = std::max(md, double(std::fabs(a[i] - b[i])));
              ma = std::max(ma, double(std::fabs(b[i])));
            }
            std::printf("draft dbg: %s maxdiff=%.4g (cpu maxabs=%.4g)\n",
                        name, md, ma);
          };
          maxdiff(gc, cpu_drafter.dbg_ctx, "ctx");
          maxdiff(go, cpu_drafter.dbg_out8, "out8");
          for (int li = 0; li < 3 && !cpu_drafter.dbg_xall.empty(); li++) {
            const std::vector<float> cx(
                cpu_drafter.dbg_xall.begin() + li * 8 * H,
                cpu_drafter.dbg_xall.begin() + (li + 1) * 8 * H);
            char nm[16];
            std::snprintf(nm, sizeof(nm), "x_L%d", li);
            maxdiff(gpu_drafter.dbg_x(li), cx, nm);
          }
          if (!cpu_drafter.dbg_att0.empty())
            maxdiff(gpu_drafter.dbg_att0(), cpu_drafter.dbg_att0, "att0");
          if (!cpu_drafter.dbg_xin.empty())
            maxdiff(gpu_drafter.dbg_xin(), cpu_drafter.dbg_xin, "xin");
          if (!cpu_drafter.dbg_xn0.empty())
            maxdiff(gpu_drafter.dbg_xn0(), cpu_drafter.dbg_xn0, "xn0");
          if (!cpu_drafter.dbg_q0.empty())
            maxdiff(gpu_drafter.dbg_q0(), cpu_drafter.dbg_q0, "q0");
          if (!cpu_drafter.dbg_k0.empty())
            maxdiff(gpu_drafter.dbg_k0(), cpu_drafter.dbg_k0, "k0");
          if (!cpu_drafter.dbg_ctxk0.empty())
            maxdiff(gpu_drafter.dbg_ctxk0(), cpu_drafter.dbg_ctxk0, "ctxk0");
          // raw dumps for offline bisect
          const char* dd = dump_dir.empty() ? "." : dump_dir.c_str();
          auto wr = [&](const char* nm, const float* p, size_t n) {
            std::string f = std::string(dd) + "/" + nm + ".bin";
            if (FILE* fp = std::fopen(f.c_str(), "wb")) {
              std::fwrite(p, 4, n, fp);
              std::fclose(fp);
            }
          };
          wr("g_xn0", gpu_drafter.dbg_xn0(), size_t(8) * H);
          wr("g_xin", gpu_drafter.dbg_xin(), size_t(8) * H);
          wr("g_ctxk0", gpu_drafter.dbg_ctxk0(), 1024);
          wr("c_xn0", cpu_drafter.dbg_xn0.data(), cpu_drafter.dbg_xn0.size());
          wr("c_xin", cpu_drafter.dbg_xin.data(), cpu_drafter.dbg_xin.size());
          wr("c_ctxk0", cpu_drafter.dbg_ctxk0.data(), 1024);
          wr("c_ctx", cpu_drafter.dbg_ctx.data(), cpu_drafter.dbg_ctx.size());
          wr("c_xall", cpu_drafter.dbg_xall.data(), cpu_drafter.dbg_xall.size());
          wr("taps", taps4.data(), taps4.size());
          wr("c_base0", cpu_drafter.dbg_base0.data(),
             cpu_drafter.dbg_base0.size());
          if (!cpu_drafter.dbg_out8.empty()) {
            // per-row (slot) divergence to localize the failing stage
            for (int j = 0; j < 8; j++) {
              double md = 0;
              for (int d = 0; d < H; d++)
                md = std::max(
                    md, double(std::fabs(
                                go[j * H + d] -
                                cpu_drafter.dbg_out8[size_t(j) * H + d])));
              std::printf("draft dbg: out8 slot%d maxdiff=%.4g\n", j, md);
            }
            dump_f32(dump_dir.empty() ? "." : dump_dir, "cpu_out8", 0,
                     cpu_drafter.dbg_out8.data(),
                     cpu_drafter.dbg_out8.size());
          }
        }
      }
    }
#endif
    if (am == cfg.eos_token_id) break;
    cur = am;
  }
  if (tok_loaded && !gen_ids.empty())
    std::printf("text: %s\n", tok.decode(gen_ids).c_str());
  double wall = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - wall0)
                    .count();
  std::printf("wall: %.2fs for %lld tokens (%.2f s/tok)\n", wall,
              (long long)(pos - (int64_t)ids.size()),
              wall / std::max<int64_t>(1, pos - (int64_t)ids.size()));
#ifdef INK_WITH_VULKAN
  if (gpu_mode) {
    auto& st = gpu_graph.stats;
    std::printf(
        "gpu: submits=%llu dispatches=%llu copies=%llu barriers=%llu "
        "record=%.0fms wait=%.0fms uploads=%llu (%.2fGB) "
        "warm{slots=%llu hit=%llu dev=%llu miss=%llu} err: %s\n",
        (unsigned long long)st.submits, (unsigned long long)st.dispatches,
        (unsigned long long)st.copies, (unsigned long long)st.barriers,
        st.record_ms, st.wait_ms, (unsigned long long)st.uploads,
        st.upload_bytes / 1e9, (unsigned long long)st.pf_slots,
        (unsigned long long)st.pf_hits, (unsigned long long)st.pf_devs,
        (unsigned long long)st.pf_misses,
        gpu_graph.error().c_str());
    if (gpu_drafter.ok()) {
      for (const DraftWin& w : dra_wins) dra_prefix_sum += w.prefix;
      std::printf(
          "draft: windows=%lld prefix-accept=%.2f/8 | depth hits:",
          (unsigned long long)dra_nwin,
          dra_nwin ? double(dra_prefix_sum) / double(dra_nwin) : 0.0);
      for (int d = 1; d <= 8; d++)
        std::printf(" %lld/%lld", (long long)dra_hits[d],
                    (long long)dra_tot[d]);
      auto& ds = gpu_drafter.stats;
      std::printf(
          "\ndraft gpu: submits=%llu dispatches=%llu barriers=%llu "
          "wait=%.0fms uploads=%llu (%.2fGB)\n",
          (unsigned long long)ds.submits, (unsigned long long)ds.dispatches,
          (unsigned long long)ds.barriers, ds.wait_ms,
          (unsigned long long)ds.uploads, ds.upload_bytes / 1e9);
      gpu_drafter.shutdown();  // before allocator/context teardown
    }
    gpu_graph.shutdown();  // before allocator/context teardown
    vkalloc.shutdown();
    vkctx.shutdown();
  } else if (backend == "vk") {
    auto& st = vk_backend.stats;
    std::printf(
        "vk: submits=%llu wait=%.0fms uploads=%llu (%.2fGB %.0fms) "
        "matvec=%llu ffn=%llu rej{sh%llu dt%llu up%llu al%llu su%llu}\n",
        (unsigned long long)st.submits, st.wait_ms,
        (unsigned long long)st.uploads, st.upload_bytes / 1e9, st.upload_ms,
        (unsigned long long)st.n_matvec, (unsigned long long)st.n_ffn,
        (unsigned long long)st.rej_shape, (unsigned long long)st.rej_dtype,
        (unsigned long long)st.rej_upload, (unsigned long long)st.rej_alloc,
        (unsigned long long)st.rej_submit);
    std::printf("  upf{noncontig %llu toobig %llu alloc %llu rec %llu "
                "(begin %llu submit %llu)} err: %s\n",
                (unsigned long long)st.upf_noncontig,
                (unsigned long long)st.upf_toobig,
                (unsigned long long)st.upf_alloc,
                (unsigned long long)st.upf_rec,
                (unsigned long long)st.upf_begin,
                (unsigned long long)st.upf_submit,
                st.first_err.c_str());
  }
#endif
  return 0;
}
