# A05 — Survey: NVMe + RAM-tiered inference runtimes (weights & KV offload)

Scope: every runtime/technique that runs LLMs larger than VRAM by tiering weights and/or KV
across VRAM → system RAM → NVMe, with measured numbers. Focus on what is stealable for a
custom C++ runtime for **Inkling 2 Mini** (26.4B MoE, 128 experts top-8, ~5.4B active,
~19.5 GB NVFP4+MXFP8 single-file safetensors, 1M ctx) on **RX 9070 XT 16 GB + 48 GB DDR5 +
PCIe4 NVMe, Windows**.

**Key model-specific numbers (computed from `Inkling-2-Mini/config.json` + file size):**
- Per-expert size ≈ 3 × 2304 × 1152 ≈ 7.96 M params ≈ **~4.4 MB at NVFP4** (~0.55 B/param incl. scales).
- Total experts = 128 × 24 layers = **3072 experts ≈ 13.5 GB**.
- Expert visits per decoded token = 8/layer × 24 layers = 192 (≈0.85 GB). **Important:**
  `loop_config` re-runs layers 6–18 three times (≈50 MoE-layer executions/token → up to ~400
  expert visits ≈ 1.76 GB/token worst case), but cross-pass router correlation makes the
  *unique* working set much smaller — a cache, not bandwidth, problem.
- 20 of 24 layers are sliding-window attention (window=1024); only ~4 layers are global →
  global KV at 1M ctx ≈ 16 GB bf16 (vs ~98 GB if all layers were global). KV is a RAM/SSD
  problem at long ctx, but weights are not: **the whole 19.5 GB file fits in 48 GB RAM.**

---

## 1. The tiered-latency mental model

Every system below is governed by one formula (corroborated by gpuwerk, aifoss, Strata's
paper, and llama.cpp practitioners):

```
decode tok/s ceiling ≈ bandwidth_of_slowest_tier_touched / bytes_read_per_token
```

| Tier | Bandwidth (this rig) | Latency character | Implication for 0.85–1.76 GB/token |
|---|---|---|---|
| VRAM (9070 XT GDDR6) | ~640 GB/s | — | not the bottleneck if expert resident |
| PCIe4 x16 H2D | ~25–28 GB/s | ~0.16–0.18 ms per 4.4 MB expert | ~15–33 tok/s if streaming every expert from RAM to GPU |
| DDR5-6000 dual-channel | ~77–96 GB/s | ~0.05 ms per expert read | ~45–110 tok/s if experts are **computed on CPU** in place |
| PCIe4 NVMe | ~5–7 GB/s seq; ~0.5–1.5 GB/s small random | ~0.6–0.9 ms per expert read | ~3–8 tok/s streaming every expert from SSD |

Three structural tricks recur everywhere:
1. **Don't move weights to the compute — move compute to the weights** (Fiddler, Strata,
   PowerInfer, ktransformers, llama.cpp `--cpu-moe`): computing a 4.4 MB expert as a CPU GEMV
   touches ~46 µs of DRAM bandwidth vs ~180 µs of PCIe to copy it. CPU compute wins for
   batch-1 decode whenever the expert lives in RAM.
2. **Cache experts in VRAM** (every modern engine): with ~10 GB free after dense+KV, ~2300 of
   3072 Inkling experts (~75%!) fit in VRAM. Hit-rate is the entire game; traffic/token drops
   from 0.85 GB to ~0.2 GB at realistic hit rates.
3. **Predict the router before it runs** (Pre-gated MoE, Strata, MoE-Infinity, PreScope):
   the next layer's experts are ~70–90% predictable from the current hidden state; prefetch
   hides the miss latency behind current-layer compute.

---

## 2. llama.cpp family

### 2.1 Mechanism
- GGUF is `mmap`'d; tensors are placed per-device by name. `-ngl N` offloads the first N
  layers' *whole* tensors; the newer MoE path is **per-tensor**: `-ot "exps=CPU"`,
  `--cpu-moe` (all routed experts in RAM, computed by CPU kernels), `--n-cpu-moe N` (first N
  MoE layers' experts on CPU), and since ~b74xx **`--moe-cache-mib N`** = a GPU-resident
  expert cache in front of the CPU experts, plus `--moe-cache-cpu-share F` (fraction of
  misses computed on CPU instead of uploaded). Auto `--fit` picks a split if you don't.
  Flags: https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md ,
  https://github.com/ggml-org/llama.cpp/pull/15077 ,
  guide: https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide
- Important flag: **`--no-op-offload`** — without it, the scheduler can "offload" host-weight
  MoE ops *to* the GPU and stream weights over PCIe each step; one measured report saw PP
  collapse from ~91–115 to ~11 tok/s on a 9.2k prompt.
  https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/68
- **`--no-mmap` / direct I/O**: mmap is great for warm reloads and for *demand-paging experts
  that don't fit in RAM*; but first-load is slow and Windows page-fault storms are real
  (one report: ~15 min first load after reboot; llama.cpp issue #705). Direct-I/O async read
  (`-dio`, implied by `--no-mmap` on Linux) loads gpt-oss-120B-MXFP4 in a consistent ~10.5 s
  vs ~110 s cold mmap (DGX Spark). On Windows, `--no-mmap` loaded ~50% faster in PR #7896
  testing (mmap ~6 GB/s vs no-mmap ~3.5 GB/s raw, but better copy interleave wins overall).
  https://github.com/ggml-org/llama.cpp/pull/18012 ,
  https://github.com/ggerganov/llama.cpp/pull/7420 ,
  https://github.com/ggml-org/llama.cpp/issues/705
- With CPU tensor overrides + mmap, llama.cpp itself warns to use `--load-mode none`; the OS
  page cache becomes your expert cache (works until RAM < file and it thrashes).
  https://modelvram.com/llm-vram-calculator/moe-offload-guide/

### 2.2 Measured numbers (consumer GPUs + CPU-resident experts)
- **gpt-oss-120B** (36 layers, 128 ex, top-4, ~1.9 GB expert reads/token): ~22 tok/s on RTX
  4090 (`--n-cpu-moe 28`); ~30 tok/s on RTX 5090 (`--n-cpu-moe 21`). RAM bandwidth, not GPU,
  sets the ceiling; measured ≈ half of `RAM_BW / expert_GB`.
  https://gpuwerk.com/blog/moe-cpu-offload-llama-cpp/ ,
  https://github.com/ggml-org/llama.cpp/discussions/15396
- **Qwen3-235B-A22B** on RTX 3090 + 128 GB DDR4-2933 + TR 2950x (ik): ~7.4 tok/s TG, ~24 t/s
  PP with `-ot` splits; on a 3090Ti + 9950X + 96 GB DDR5-6400: ~10 tok/s TG, ~140 tok/s PP.
  4–10 tok/s reported on RTX 3080 + 128 GB.
  https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF ,
  https://skeskin.com/en/publications/qwen3-235b-a22b-instruct-2507-gguf-and-llamacpp
- **Qwen3.6-35B-A3B** on RTX 4070 (12 GB) + i7-14700F + 32 GB: `fit` ~51–57 tok/s TG; `CPU
  MoE` placement ~45 tok/s — offload cost is modest when active params are small.
  https://qiita.com/Marron-chan/items/05be57d7aeee1a2a31f5
- `--moe-cache-mib` caveat (HF blog): fragments the graph (2 splits/token → ~50), loses ~10%
  even when all weights fit. https://huggingface.co/blog/moe-expert-streaming

**Steal:** per-tensor regex placement (`exps=CPU` pattern over tensor names), GPU expert
cache with CPU-compute-on-miss hybrid (`--moe-cache-cpu-share`), `--no-op-offload` as a
"sanity" switch, and keep mmap **optional** — for a 19.5 GB file on a 48 GB box, pin dense
tensors, mmap experts, and let the page cache be the RAM tier.

## 3. ik_llama.cpp (ikawrakow fork) — the MoE-offload specialist

Mechanism beyond upstream: fused CPU MoE kernels (gather expert weights more efficiently —
measured 13–23 GB/s RAM read vs 11.5–19.8 mainline), `-fmoe`, `-rtr` (run-time repack for CPU
throughput), `-amb` (attention max batch), `-mla`, `--defer-ple`, `--defer-experts`,
`--prefetch-experts`, its own GPU `--moe-cache-mib` LRU expert cache.
DeepWiki flags: https://deepwiki.com/ikawrakow/ik_llama.cpp/3.1-command-line-tools

- **`--prefetch-experts` (PR #2101)**: pool of `pread()` workers warms the page cache for
  mmap'd experts that don't fit in RAM. Two modes: *selective* (decode — enqueue the expert
  slices as soon as routing ids are host-visible; matmul entry self-enqueues too) and
  *lookahead* (prefill — stream next N splits' experts wholesale; `MADV_COLD` after use).
  Tuning: `GGML_MOE_PREFETCH_THREADS` (default min(8,hw)), `GGML_MOE_PREFETCH_AHEAD`
  (default **3 splits**). ~**2× PP** on models ~1.2–2× RAM. This is the closest existing
  implementation of "mmap the model, prefetch routed experts, cap the working set".
  https://github.com/ikawrakow/ik_llama.cpp/pull/2101
- **`--defer-experts`/`--defer-ple` + `--moe-cache-mib`**: GPU-resident expert LRU. ikawrakow
  reports **45 tok/s** on a 12 GB VRAM + 64 GB RAM box with a 10 GiB GPU MoE cache on an
  IQ4_XS model. https://github.com/ikawrakow/ik_llama.cpp/issues/2665
- **Streaming from Optane/SSD (CPU-only, mmap)**: Kimi-K2.7-class 1 T MoE, ~378 GiB IQ3_S
  mmap'd: **3.14 tok/s TG**; dropping `expert_used_count` 8→6 gives +14% (3.58 t/s) at 89%
  top-1 agreement — direct evidence that *fewer experts/token is a throughput knob*.
  https://github.com/ikawrakow/ik_llama.cpp/discussions/2106
- **110 tok/s on 12 GB VRAM + 48 GB DDR5-6000**: Qwen3.6-35B-A3B IQ4_XS with ik's fused CPU
  MoE + MTP speculative decoding — essentially our exact hardware envelope (small model,
  but shows the CPU-offload + MTP multiplier).
  https://bayesiansapien.github.io/cere-bro/inference-efficiency/2026-05-21-ik-llamacpp-mtp-cpu-offload-qwen36/
- CPU-only uplift on MoE: +74% vs mainline decode on Qwen A3B-class (22.7 vs 13.1 t/s).
  https://github.com/jimliddle/ik-llama-cpu-moe-benchmarks
- gpt-oss-120B on RTX 4080 + Threadripper: 24.9→22.7 t/s TG depending on whether expert
  *biases* also stay on GPU — tensor-granularity detail that matters.
  https://github.com/ikawrakow/ik_llama.cpp/pull/829

**Steal:** (a) `pread`-worker page-cache warming triggered by router outputs — implement on
Windows with `ReadFileScatter`/IOCP or `PrefetchVirtualMemory` over the mapped expert
region; (b) prefetch-ahead depth ~3 graph splits; (c) `MADV_COLD`-equivalent
(`VirtualUnlock`/reclaim hints) for one-shot prefill streams; (d) fused MoE GEMV over
gathered expert rows is worth ~1.5–2× over naive per-expert calls.

## 4. Strata — the closest existing design (already supports Inkling 2 Mini)

A single-model-family C++/CUDA+HIP engine (Niko Veit). Design doc:
https://raw.githubusercontent.com/Niko1221/Strata/main/docs/HOW_IT_WORKS.md ,
paper: https://github.com/Niko1221/Strata/blob/master/docs/paper/Strata-Paper.pdf ,
releases: https://github.com/Niko1221/Strata/releases

Mechanism (4-tier, exactly our problem shape):
- GPU keeps the dense trunk (attention/mixers, routers, shared experts, output head, MTP
  draft, hot KV) + a **VRAM expert cache** that fills remaining space (~700 experts/GB).
- RAM pins *all* experts; a persistent SIMD CPU worker pool computes cache-miss experts
  **in place** concurrently with GPU work; "miss-sharing" lets the CPU take misses instead
  of uploading when PCIe is saturated.
- SSD holds overflow: Engram/n-gram tables and streamed experts (row-granularity reads via
  OS page cache). Modes: full 4-tier, GPU+SSD-only (`--ssd-experts --ram-experts-gb 0`),
  CPU+SSD-only. **Release 0.2.20 lists Inkling 2 Mini as a supported family** alongside
  Qwen3.8-Flash-Next and HY-4 Flash — we can literally run/benchmark it as a baseline.
- Numbers (Qwen3.8-Flash-Next, 125B/6B-active, 48 layers × 512 ex): RTX 5070 12 GB + Ryzen
  7600 + 64 GB DDR5 → **95 tok/s at Q2_0, 45–95 t/s across quants, 285–571 t/s prefill**;
  expert traffic only ~0.66 GB/token (2% of expert bytes) with ~4,500 VRAM-cached experts.
  On M5 32 GB Mac with SSD expert streaming: ~13.5–14.8 tok/s decode on the 180B.
- Scheduling detail worth copying: pinned staging buffers + separate copy stream +
  event-controlled ring so H2D overlaps compute; **next-layer experts predicted by running
  that layer's router on the current hidden state** (the HF streaming project confirms this
  trick: routers run early, ~1 guessed expert/layer prefetched while SSD is idle).

## 5. KTransformers — CPU-expert-compute + AMX at datacenter scale

https://github.com/kvcache-ai/ktransformers , paper SOSP'25:
https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf

- Mechanism: MoE FFN on CPU (AMX-optimized int4/int8 kernels), attention+KV on GPU;
  asynchronous CPU-GPU task scheduling (single CUDA graph per token, expert deferral raises
  CPU utilization ~75%→~100%); 3-layer KV hierarchy (GPU-CPU-disk); optional
  `--kt-mmap-experts-dir` keeps CPU experts in a file instead of anonymous RAM.
- Numbers: DeepSeek-V3 Q4_K_M on dual Xeon Gold 6454S + 4090D 24 GB: **13.7 decode / 97.3
  prefill t/s** (vs llama.cpp 4.5/10.3 → 3.03×/9.44×). V0.3 prefill up to 286 t/s (BF16,
  multi-GPU). mmap'd experts cost ~nothing: 13.1→13.0 t/s steady decode on CMP-170HX + 9900X.
  https://github.com/kvcache-ai/ktransformers/blob/main/doc%2Fen%2FDeepseekR1_V3_tutorial.md ,
  https://github.com/kvcache-ai/ktransformers/pull/2190
- Consumer reality check (Thaki Cloud): Qwen3-30B-A3B on RTX 4090 + 7950X: full-GPU 261.5
  t/s, CPU-expert-offload **12.0 t/s**, CPU-only 7.4 t/s — offload ≈1.6× CPU-only but 22×
  slower than resident. The "28×" marketing is a *prefill* multiplier vs llama.cpp CPU.
  https://thakicloud.com/tech-blog/en/llmops/ktransformers-moe-offload-28x-validation/
- Profiling insight (from the paper): Fiddler/llama.cpp-style decode triggers ~7,000 kernel
  launches/token × ~16 µs each = 73% of GPU time is launch overhead → **they built a
  CUDA-graph-per-token executor; we must do the equivalent (batched command submission,
  persistent kernels) or the graph launches eat the budget.**

## 6. PowerInfer & PowerInfer-2 — activation-sparsity hot/cold split

- PowerInfer-1 (SOSP'24, https://github.com/Tiiny-AI/PowerInfer ): offline profiler
  identifies power-law "hot" neurons (26% of neurons → 80% of compute) → hot neurons
  preloaded in VRAM, cold computed on CPU; tiny per-layer predictors pick activated neurons;
  neuron-aware sparse ops on both sides. **13.2 avg / 29.1 peak t/s across OPT-7B–175B on a
  single RTX 4090, up to 11.7× llama.cpp; OPT-30B reaches 82% of an A100.** Caveat: relies on
  ReLU-style natural sparsity (OPT/Falcon-ReLU/Bamboo) — for a router-gated MoE, the "hot
  expert" analogue is the expert cache, and their per-layer predictor idea maps to our
  router-prediction prefetch.
- PowerInfer-2 (https://powerinfer.ai/v2/ , https://arxiv.org/html/2406.06282 ): same idea
  on phones with **UFS flash as the weight tier**. Neuron clusters instead of neurons;
  I/O-aware pipeline with differentiated strategies per weight type; splits FFN rows between
  NPU (dense hot) and CPU (sparse cold). TurboSparse-Mixtral-47B on a phone: **11.68 t/s,
  21.2× llama.cpp / 3.1× LLM-in-a-Flash-proxy**, with only 50% of FFN weights even in DRAM —
  the rest streams from flash. Their measured breakdown: flash reads/token were the critical
  path; cutting FFN-resident fraction to ~18% only cost ~2.5× speed.
  https://hackernoon.com/performance-evaluation-of-powerinfer2-offloading-prefill-and-inmemory-efficiency

## 7. Fiddler + HOBBIT — activation-copy vs weight-copy

- Fiddler (UW, https://github.com/efeslab/fiddler , paper https://arxiv.org/abs/2402.07033 ):
  per-expert decision — if expert is on GPU, run it there; if not, **copy the activation
  (≈4 KB×batch) to CPU, run the expert on CPU, copy the output back** — because for small
  batches an activation is ~10,000× smaller than the expert (Mixtral: 4096 floats vs
  3×4096×14336 params). Runs **unquantized Mixtral-8x7B (>90 GB) at >3 t/s on one 24 GB
  GPU**; 8.2–10.1× vs naive offload, 1.26× vs best baseline single-batch. This is the
  foundational argument for CPU-side expert compute on PCIe-limited rigs.
- HOBBIT (https://arxiv.org/abs/2411.01433 ): mixed-precision variant — hot experts high
  precision on GPU, cold experts low precision on CPU; 0.99–1.46× Fiddler decode, 42–85%
  lower prefill latency. Adds: expert *precision* is a per-expert cache-tier decision too
  (we have NVFP4+MXFP8 mixed formats natively — same idea).

## 8. MoE-Infinity — trace-based expert cache + SSD tier

https://arxiv.org/abs/2401.14361 , https://github.com/EfficientMoE/MoE-Infinity
- Sparsity-aware expert cache: traces per-request expert activation (temporal locality is
  strong: one expert reused 15× within a sequence), picks the representative trace to drive
  replacement+prefetch; experts tier GPU→RAM→SSD. **3.1–16.7× lower per-token latency vs
  vLLM/Ollama/DeepSpeed/BrainStorm** on DeepSeek/Mixtral tasks; ~40 s/token naive offload →
  interactive. Their evidence: frequency-based (BrainStorm/DeepUM-style) policies
  under-perform because reuse is *per-sequence*, so they trace request-level patterns.

## 9. Gate-prediction family: Pre-gated MoE, AdapMoE, EdgeMoE, SpecMoEOff, OD-MoE, "speculating experts"

- **Pre-gated MoE** (ISCA'24, https://arxiv.org/abs/2308.12066 ): modifies the model — a
  trained *pre-gate* selects next-block experts at the current block, so prefetch has a full
  layer of slack and the cache needs only top-κ slots (not all experts). A100 80 GB: within
  **23% of oracular all-GPU** latency, 4.2× lower peak GPU memory; 1.7× (up to 1.9×) over
  on-demand offload, 42–125× over prefetch-all. Proof that one-layer-ahead prediction is
  good enough *if it's accurate*; we can't retrain the model, but…
- **"Speculating Experts"** (https://arxiv.org/abs/2603.19289 , code
  github.com/axonn-ai/yalis `offload_prefetch`): training-free version — predicts future
  experts from current internal representations; even *executing speculated experts* often
  preserves downstream accuracy; **up to 14% TPOT reduction** vs on-demand CPU loading.
- **AdapMoE** (https://arxiv.org/abs/2408.10284 ): sensitivity-based adaptive gating —
  unimportant layers may activate *fewer* experts without accuracy loss; + expert prediction,
  prefetching, DP-based cache sizing. **−25% activated experts, 1.35× vs
  Mixtral-Offloading, no accuracy degradation** on 4090/A6000. Cheap knob for us:
  top-8→top-6 on insensitive layers = 25% less I/O (matches ik's `expert_used_count` result).
- **EdgeMoE** (MobiCom'23, https://arxiv.org/abs/2308.14352 ): first storage-streamed
  expert system for phones; per-expert bitwidth tuning + expert buffer in a 1.5–3 GB memory
  budget; 2.1–38.6× memory savings vs keeping model resident. Historical interest: proved
  expert-granularity on-demand loading works.
- **SpecMoEOff** (https://arxiv.org/abs/2508.21706 ): overlaps expert offload latency with
  speculative-decoding verify steps — the draft/verify structure gives free I/O slack;
  up to **2.5×**. Directly relevant: Inkling has MTP heads (target_layer_ids 5,23,41,48) —
  speculation masks prefetch latency.
- **OD-MoE** (https://arxiv.org/abs/2512.03927 ): cacheless fully on-demand loading with an
  "emulative predictor" forecasting activations several layers ahead — argues a good
  predictor beats a big cache when RAM is truly scarce.

## 10. Pipeline/batch family: MoE-Lightning, Klotski, PreScope, PDScope, LayerScope

- **MoE-Lightning** (https://arxiv.org/abs/2411.11217 ): CPU-GPU-I/O 3-stage pipeline
  (CGOPipe) with *paged weights* + a Hierarchical Roofline Model to pick the schedule; CPU
  attention for deep-enough batches launched two micro-batches ahead. **10.3× vs FlexGen on
  Mixtral-8x7B, single T4 16 GB**; reaches GPU-memory-bound throughput with 2–3× less CPU
  RAM. Batch-oriented but the paged-weight layout + model-driven scheduling is stealable.
- **Klotski** (https://arxiv.org/abs/2502.06888 ): expert-aware multi-batch pipeline —
  extends compute time to cover I/O by orchestrating batch order; correlation-aware expert
  prefetcher; up to **85× throughput** vs SOTA in its regime. Caveat from PreScope: bigger
  batches activate more experts → more I/O.
- **PreScope** (https://arxiv.org/abs/2509.23638 ): LLaPor — a small learnable predictor
  exploiting that expert-activation predictability differs by layer group (near-input /
  middle / near-output); prefetch-aware cross-layer scheduler balances prefetch vs
  on-demand PCIe bandwidth; async I/O decoupled from compute. **+141% over Klotski, up to
  74.6% lower latency.** Key data point for our prefetcher: prediction accuracy is
  layer-position-dependent — budget prefetch depth per-layer-group, not uniformly.
- **PDScope** (https://arxiv.org/abs/2610.39216 ): prefill benefits from GPU-heavy
  placement, decode from CPU-side compute → *different tiering per phase*; 2.5–10.3×
  prefill, up to 6.1× decode. Steal: our prefill should stream experts to GPU (all experts
  get touched anyway at batch≥active), decode should CPU-compute misses.

## 11. SSD-native expert streaming: SSD-LLaMA, FlashMoE, HF Transformers `experts_offload`, "pooled"

- **SSD-LLaMA** (https://arxiv.org/abs/2609.18110 , Sep 2026): SSD-native MoE engine —
  (1) offline **expert-pack** layout: gate+up+down of each (layer,expert) contiguous and
  aligned → one contiguous read per expert instead of scattered tensor reads; (2) lossless
  compression of expert blocks; (3) three-tier SSD→RAM→VRAM delivery with async overlap of
  SSD read, H2D, and compute; (4) executes *every* selected expert (no pruning). **2.10–
  15.58× decode, 1.52–4.19× prefill vs baselines; >1 t/s on a trillion-param model with a
  single RTX 5090 + ≤32 GB RAM.** The layout criticism applies to us: our safetensors stores
  gate/up/down as separate tensors per layer — a repack to per-expert blocks is required for
  SSD efficiency.
- **FlashMoE** (KAIST, https://arxiv.org/abs/2601.17063 ): SSD-offloaded experts on a
  desktop (RTX 5070 Ti 16 GB + PCIe5 NVMe 7.4 GB/s + only 16 GB RAM); ML approximation of
  Belady's optimal replacement (mixes recency+frequency) → **+51% cache hit rate vs LRU/LFU,
  up to 2.6× end-to-end**. Also separates non-expert files from expert files so the model
  *starts* fast (dense first, experts on demand). Hit-rate evidence: LRU fails because
  experts have routing-correlated reuse — a token's expert often won't be reused next token
  but will recur later.
- **HF Transformers `experts_offload` (PR #50717 experiment)**
  https://huggingface.co/blog/moe-expert-streaming : experts→"disk", a **slot pool** on
  device (one preallocated tensor + per-layer expert-id→slot table, so `grouped_mm` runs
  unmodified — elegant, no custom kernel), LRU per layer, 8 I/O threads doing one
  contiguous read/expert, next-2-layers' routers run early on current hidden state to
  prefetch; experts re-quantized smaller on disk (3-bit). 180B model on 32 GB M5 Mac:
  ~14 t/s (within 6% of Strata). Engine comparison on A10G 24 GB, Qwen3.6-35B-A3B decode:
  **Strata Q8_0 82–88; llama.cpp `--fit` 63–64 (Q8_0); vLLM FP8 cpu-offload-experts 20.1;
  vLLM all-weight offload 9.7; Transformers+Accelerate 0.16.** Best public apples-to-apples
  of the design space.
- **"pooled" (community engine, roadmap doc)**
  https://github.com/Nehanth/pooled/blob/main/roadmap/32-ssd-expert-offload.md : same 3-tier
  plan on consumer PC — measured: parked RAM expert→VRAM at **51 GB/s (0.121 ms/5.5 MB
  expert)**, `writeBuffer` only 3.8–10 GB/s; per-layer LRU in VRAM; disk tier = GGUF
  offsets recorded per expert, worker threads read ~1 MiB chunks into mapped slots, fall
  back to queue.writeBuffer. Confirms: a single expert read at ~4–6 MB is *large enough* for
  near-sequential NVMe throughput; the kill factor is count of reads, not read size.

## 12. Whole-model layer streaming (the naive baseline): AirLLM, ZeRO-Inference, FlexGen

- **AirLLM** (https://github.com/lyogavin/airllm ): keeps exactly one layer on GPU; model
  read from disk **once per token**. 70B on 4 GB works but ~20 s/token floor at FP16 on Gen4
  NVMe; community measurements ~0.1–0.7 t/s quantized. Qwen3-235B needs ~3 GB VRAM, 405B ~8
  GB. It is the cautionary lower bound — layer-granularity streaming is dead for decode;
  expert-granularity + caching is the fix.
  https://umesh-malik.com/blog/run-70b-llm-on-4gb-gpu-airllm ,
  https://aifoss.dev/blog/airllm-70b-4gb-setup-guide-2026/
- **DeepSpeed ZeRO-Inference** (https://www.deepspeed.ai/2022/09/09/zero-inference.html ,
  GDS results https://github.com/deepspeedai/DeepSpeedExamples/tree/master/deepnvme ):
  streams layers from pinned CPU or NVMe (GPUDirect optional). H100 + NVMe: Mixtral-8x22B
  **9.15 t/s @ bsz200**, Llama3-70B 8.6 @ bsz96, BLOOM-176B 0.29 @ bsz8 — GDS vs bounce
  buffer barely differs (9.15→9.23); 4×Gen5 NVMe scales 7→26 t/s (batch). Conclusion: disk
  streaming only pays at large batch; batch-1 must cache, not stream.
- **FlexGen** (ICML'23, https://github.com/FMInference/FlexLLMGen ): LP-solved GPU/CPU/disk
  tensor placement, zig-zag block schedule, 4-bit weights+KV. OPT-175B on a T4 16 GB +
  208 GB RAM + SSD: **1.12 t/s at effective batch 144** — throughput-optimized, useless for
  interactive decode but the right frame: at large batch, I/O amortizes (a fetched expert
  serves hundreds of tokens).
- **LLM-in-a-Flash** (Apple, https://arxiv.org/abs/2312.11514 ): dense-model flash
  streaming exploiting ReLU sparsity — **windowing** (reuse last token's active neuron set,
  predict via low-rank projector) + **row-column bundling** (co-locate up-proj rows with
  down-proj cols → ~32–35 KiB contiguous reads). 4–5× CPU / 20× GPU vs naive on 2×-DRAM
  models. For us: the *bundling* principle = SSD-LLaMA's expert-pack; windowing = expert
  reuse across loop passes.
- **InfiniGen** (OSDI'24, https://arxiv.org/abs/2404.07143 , repo
  snu-comparch/InfiniGen): KV-side — CPU-resident KV pool, skewed Q/K matrices let layer
  *i−1* speculate which few KV entries layer *i* needs → prefetch only important tokens'
  KV. **Up to 3× vs prior KV offload, better accuracy**; the layer-ahead speculative
  prefetch is the KV analogue of router-lookahead expert prefetch.
- **Flash-LLM / SpInfer** (https://arxiv.org/abs/2309.10285 ,
  https://dl.acm.org/doi/10.1145/3689031.3717481 ): unstructured-sparsity GPU kernels —
  "load-as-sparse compute-as-dense" (Flash-LLM, up to 3.8× vs DeepSpeed on OPT) and
  bitmap-encoded SpMM (SpInfer, up to 2.1× Flash-LLM on 4090). Relevant only if we exploit
  activation sparsity inside experts; for bandwidth-bound GEMV, dense 4-bit wins anyway.

## 13. vLLM ecosystem — weight offload + the KV tiering stack

- **`--cpu-offload-gb N`** (+ new `--cpu-offload-params experts`): N GB of weights in pinned
  host RAM, read in place over PCIe every use — no caching, no disk tier; offloading *only
  experts* doubled decode (20.1 vs 9.7 t/s on the HF A10G test). Designed for big batches
  where most experts fire anyway. https://huggingface.co/blog/moe-expert-streaming
- **KV swap / native KV offloading connector**: `--kv_offloading_backend native
  --kv_offloading_size GB` — async GPU↔CPU KV block transfers; TTFT on hits **2–22× better**
  than recompute (H100, Llama-3.1-8B); offload is off critical path on miss.
  https://vllm.ai/blog/2026-01-08-kv-offloading-connector
- **LMCache** (https://arxiv.org/html/2510.09665 ): KV cache layer CPU/disk/remote; on a
  vLLM+A100 eval it raised sustainable concurrency 75→100 (multi-proc); one honest eval
  found *disk backend added no measurable benefit* for chat workloads — disk KV pays only
  for very long prefixes. https://hps.vi4io.org/_media/research/theses/dennis_treder...pdf
- **Mooncake Store** (Moonshot/Kimi, https://kvcache-ai.github.io/Mooncake/ ,
  https://dl.acm.org/doi/full/10.1145/3773772 ): disaggregated KVCache pool over cluster
  DRAM+SSD+RDMA, block-addressed, Conductor scheduler; +59–498% effective request capacity,
  +115% requests on A800 clusters in production (100 B tokens/day). Architectural reference
  for KV-tiering as a *service*, not our single-box shape.
- **py-kvcache** (https://arxiv.org/html/2609.11744 ): careful study of KV-from-NVMe in
  vLLM — async direct I/O, bounded staging, *scheduler-aware preloading* (start disk reads
  while request still queued). At 80 k tokens, 2.0× faster hits than LMCache (preload alone
  1.34×); on H100 most prefixes fall below the recompute break-even → **external KV is an
  admission decision, not a default**. The break-even math transfers to expert streaming:
  if recompute-on-CPU < fetch-from-SSD, don't fetch.

## 14. Cross-cutting findings

### 14.1 Read amplification & layout
- Expert size vs page size: our ~4.4 MB expert ≈ ~1100 4 KB pages — each expert read is
  already a large sequential read, so *read-size* amplification is NOT the problem at MoE
  granularity. The real amplification is **scattered layout**: safetensors/GGUF stores
  gate/up/down of expert *e* as slices inside three per-layer tensors → 3+ disjoint reads,
  and naive mmap demand-paging fault-in granularity + ordering adds overhead. Fix =
  offline repack to **expert-pack** (SSD-LLaMA) or per-expert contiguous blocks (HF
  streaming: "one contiguous read per expert", 8 I/O threads).
- Page-fault economics (RMAI paper, https://euromlsys.eu/pdf/euromlsys25-28.pdf ): a 4 GB
  region on 4 KB pages = ~1 M faults × 6–12 µs = 6–12 s worst case; use huge pages/THP or
  pread-workers (ik's approach) rather than trusting the fault path. On Windows, first-load
  mmap page faults are demonstrably worse (llama.cpp #705, #7896) — for us: **prefetch via
  async `ReadFile` into an owned RAM pool, or `PrefetchVirtualMemory` on the mapping, not
  demand faults.**
- Chunk size sweet spot: LLM-in-a-Flash pushed flash reads to 32–35 KiB for bundling; pooled
  uses ~1 MiB chunks; HF uses whole-expert single reads. At 4.4 MB/expert and ~8 experts/
  layer, ~35 MB/layer-token: reads are big; use 1–4 MiB iovecs, ≥4–8 concurrent I/O threads
  or io_uring/IOCP queue depth ≥8.

### 14.2 Prefetch depth evidence
- ik `--prefetch-experts`: default lookahead **3** graph splits (PP); decode mode issues
  readahead as soon as router ids are visible — depth-0 prediction, pure latency hiding.
- Strata / HF streaming: **1–2 layers ahead** using the *next layer's router on the current
  hidden state*; HF found "one guessed expert per layer while SSD idle" beat bigger
  speculative reads; LRU≈best simple policy on their workload but FlashMoE's ML-Belady beats
  LRU by up to 51% on hit rate.
- Pre-gated MoE: **full layer ahead** requires a trained pre-gate; PreScope: per-layer-group
  predictors (predictability differs near-input vs middle vs near-output); OD-MoE claims
  multi-layer-ahead emulation is feasible.
- Consensus: **prefetch 1 layer ahead is the sweet spot for decode** (compute per layer
  ~1–3 ms ≈ time to fetch 1–2 missed experts at 5–7 GB/s); deeper lookahead only for prefill
  (whole-layer streaming) — and prefill should stream *all* experts anyway.

### 14.3 Cache policy evidence
- LRU is the default in llama.cpp `--moe-cache-mib`, Strata, HF streaming, pooled ("no
  cache policy beat LRU by much" — HF). MoE-Infinity's trace-based and FlashMoE's ML-Belady
  beat LRU on hit rate (up to +51%) but cost machinery; AdapMoE adds DP sizing across
  layers (unbalanced cache demand per layer — early vs late layers have different reuse).
- For Inkling's *loop*: passes re-run layers 6–18 → cross-pass reuse is the dominant
  locality source; cache should weight *within-token* recency (the same expert is likely
  re-picked on the next pass) → a small **per-layer MRU pin set** plus global LRU is cheap
  and probably optimal.

### 14.4 CPU-compute vs copy-to-GPU per miss (the Fiddler rule)
- Activation copy ≈ hidden×4 B ≈ 9 KB vs expert copy ≈ 4.4 MB: CPU-compute wins until
  CPU time-per-expert > PCIe copy time ≈ 0.17 ms. A 4.4 MB expert GEMV needs ≥4.4 MB of DRAM
  reads ≈ 46–57 µs at 77–96 GB/s — **~3× faster than the copy even before GPU-side launch
  overhead**. Verdict for decode: keep misses on CPU (Strata/Fiddler/llama.cpp all agree);
  GPU expert cache exists to reduce *count* of CPU computes, not to enable copies.
- Exception: batched prefill — every expert fires → stream experts to GPU (MoE-Lightning,
  ik lookahead mode, PDScope's phase-dependent placement).

### 14.5 Speculative decoding synergy
- MTP (Inkling has 4 draft heads) both (a) multiplies tokens per pass (ik: 110 t/s case) and
  (b) creates I/O slack — draft/verify can run while expert reads complete (SpecMoEOff,
  2.5×). And MTP verify batches make each fetched expert serve ~2–5 tokens → effective
  I/O/token drops proportionally.

## 15. What to steal — concrete design for the Inkling runtime

1. **Tier map**: VRAM = dense trunk + KV + expert slot pool (~10–11 GB → ~2,300–2,500
   experts, i.e., ~78% of all experts resident); RAM = owned expert pool (all 13.5 GB fit —
   pin if possible, else mmap'd pages); NVMe = expert-pack file for cold misses, fast
   startup, and RAM-pressure mode + long-context KV overflow.
2. **Slot-pool expert cache** (HF/Strata pattern): one preallocated device buffer +
   per-layer `expert_id → slot` table; grouped-GEMM reads the pool in place — zero kernel
   changes, no fragmentation.
3. **Per-miss CPU compute** (Fiddler/Strata/llama.cpp `--cpu-moe`): SIMD NVFP4→dot kernels
   over RAM-resident expert bytes; ~46 µs/expert floor on 96 GB/s DDR5; 8-miss layer ≈
   0.4 ms CPU while GPU does cached experts concurrently.
4. **Router-lookahead prefetch**: run layer *i+1*'s router on the current hidden state
   during layer *i*'s expert compute (it's a 2304×128 matmul — free); prefetch predicted
   misses from the SSD pack; depth 1 layer, possibly 2 where the loop re-enters a pass.
   Optionally a PreScope-style tiny learned predictor per layer-group later.
5. **Expert-pack file format**: offline repack gate+up+down per (layer,expert) into
   contiguous aligned blocks (+ optional further quant); keep the original safetensors
   untouched; read via IOCP async `ReadFile` into pinned staging → either compute on CPU or
   DMA to a free slot (decide per miss by a cost model: CPU-compute if PCIe busy/depth low).
6. **Page-fault hygiene**: never rely on demand-paging for expert reads — explicit async
   reads into an owned pool; mmap only as cold-start convenience; use large pages for the
   RAM expert pool (2 MiB pages → ~6,600 pages for 13.5 GB).
7. **Phase-dependent policy** (PDScope/Klotski): prefill = stream all experts (big batches,
   amortize); decode = cache + CPU-miss. ik's measured ~2× PP gain from lookahead-3 whole-
   split streaming validates this split.
8. **Loop-aware reuse**: pin the previous pass's expert set per looped layer (tiny MRU);
   expected to cut unique-expert traffic/token well below the 1.76 GB worst case.
9. **Adaptive expert count** (AdapMoE/ik `expert_used_count`): top-6 vs top-8 on insensitive
   layers ≈ +10–14% speed at ~89–92% top-1 agreement — keep as a user-facing quality/speed
   dial.
10. **KV tiering at 1M ctx**: sliding-window layers need only ~8 MB KV total; the ~4 global
    layers × 1M ctx ≈ 16 GB → keep global-KV in RAM with InfiniGen-style selective
    speculative fetch to GPU, or fp8/4-bit KV (≈8/4 GB). Disk KV only for session
    persistence — py-kvcache's lesson: on a fast GPU, recompute usually beats disk hits.
11. **Launch overhead discipline** (KTransformers finding): ~7k kernel launches/token kills
    (73% GPU time) → persistent kernels / graph capture per token; batch all expert-slot
    waits on GPU events, not host syncs (Strata's event-ring).
12. **Benchmark harness**: reproduce the HF A10G table on our box (Strata vs llama.cpp
    `--cpu-moe`+`--moe-cache-mib` vs ours) — Strata already runs Inkling 2 Mini, giving an
    immediate baseline and reference implementation to study.

### Rough expected envelope (batch-1 decode, this rig)
- All-experts-VRAM-impossible (13.5 GB > free VRAM); best case ~78% VRAM hit → remaining
  ~0.19 GB/token on CPU ≈ 2–2.5 ms + dense+attn ≈ **~40–80 t/s decode potential** (Strata on
  a 125B/6B-active model with ~0.66 GB/token gets 45–95 t/s on weaker hardware; our
  per-token traffic is comparable after loop-reuse, and our experts are smaller).
- All-RAM CPU compute (no GPU cache): ~0.85–1.76 GB/token ÷ ~85 GB/s ≈ 10–20 ms →
  **~50–100 t/s theoretical, realistically 30–60 t/s** (matching llama.cpp/ik MoE-offload
  measurements on DDR5 rigs).
- SSD-stream-everything (no RAM tier): ~1.5–3.5 t/s — only viable as cold-miss path.
- These brackets match every measured system above; the design variable that moves us
  between the bottom and top of each bracket is *expert-cache hit rate + prefetch hit rate*,
  not raw bandwidth.

## 16. Source list (30+)

llama.cpp: PR #15077 (n-cpu-moe), PR #18012 (async DIO), PR #7896 (Windows no-mmap),
issue #705 (Windows fault storms), tools/cli README (moe-cache-mib);
guides: huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide ,
llmhardware.io/guides/moe-hybrid-gpu-cpu-offloading-guide ,
modelvram.com/llm-vram-calculator/moe-offload-guide , gpuwerk.com/blog/moe-cpu-offload-llama-cpp/
ik_llama.cpp: PR #2101 (prefetch-experts), PR #829 (gpt-oss biases), discussion #2106
(Optane-streamed 1T), issue #2665 (moe-cache 45 t/s), deepwiki flags page;
benchmarks: github.com/jimliddle/ik-llama-cpu-moe-benchmarks ,
huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF (+disc/3), qiita RTX4070 hybrid,
bayesiansapien cere-bro 110 t/s post
Strata: github.com/Niko1221/Strata (HOW_IT_WORKS.md, Strata-Paper.pdf, release 0.2.20)
KTransformers: SOSP'25 paper (madsys PDF), DeepseekR1_V3_tutorial.md, PR #2190 (mmap
experts), thakicloud validation, deepwiki 3.2
PowerInfer: SOSP'24 abstract + ipads slides + github; PowerInfer-2: arxiv 2406.06282,
powerinfer.ai/v2, hackernoon eval, blog.aifetchly numbers
Fiddler: github.com/efeslab/fiddler, arxiv 2402.07033, syfi page; HOBBIT arxiv 2411.01433
MoE-Infinity: arxiv 2401.14361, github EfficientMoE/MoE-Infinity
Pre-gated MoE: arxiv 2308.12066 (ISCA'24), github ranggihwang/Pregated_MoE
AdapMoE: arxiv 2408.10284; EdgeMoE: arxiv 2308.14352
MoE-Lightning: arxiv 2411.11217; Klotski: arxiv 2502.06888; PreScope: arxiv 2509.23638;
PDScope: arxiv 2610.39216; SpecMoEOff: arxiv 2508.21706; OD-MoE: arxiv 2512.03927;
Speculating-Experts: arxiv 2603.19289 (yalis offload_prefetch)
SSD-LLaMA: arxiv 2609.18110; FlashMoE: arxiv 2601.17063; pooled roadmap:
github.com/Nehanth/pooled roadmap/32-ssd-expert-offload.md
HF streaming: huggingface.co/blog/moe-expert-streaming (PR #50717)
AirLLM: github.com/lyogavin/airllm, HF blog 2023, umesh-malik + aifoss evals
ZeRO-Inference: deepspeed.ai blog 2022-09-09, DeepSpeedExamples deepnvme README
FlexGen: PMLR sheng23a, github FMInference/FlexLLMGen
LLM-in-a-Flash: arxiv 2312.11514, Apple ML page, montanaresearch explainer
InfiniGen: OSDI'24 (usenix PDF), github snu-comparch/InfiniGen
Flash-LLM arxiv 2309.10285; SpInfer EuroSys'25; InferFast ICS'26
vLLM: KV-offload-connector blog (vllm.ai 2026-01-08); LMCache arxiv 2510.09665 + HPS thesis;
Mooncake: ACM TOS paper + docs site; py-kvcache arxiv 2609.11744;
GKE tiered-KV blog (cloud.google.com)
RMAI (page-fault costs): euromlsys.eu/pdf/euromlsys25-28.pdf
