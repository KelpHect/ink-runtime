# A09 — KV-cache architecture for 1M-token context on RX 9070 XT (16 GB VRAM + 48 GB RAM + NVMe)

Scope: Inkling 2 Mini, custom C/C++ runtime, single-user, maximize context. Hardware facts used
throughout: RX 9070 XT = 64 RDNA4 CUs, 16 GB GDDR6 @ 640 GB/s, PCIe 5.0 x16 (~63 GB/s peak,
~50 GB/s realistic), 2nd-gen AI accelerators with native FP8 (E4M3/E5M2) WMMA at 2× the FP16 rate
and INT8 at the same rate (AMD product brief:
https://www.amd.com/content/dam/amd/en/documents/corporate/events/radeon-9000-series-qrg-competitive.pdf,
https://www.techpowerup.com/review/amd-radeon-rx-9070-series-technical-deep-dive/3.html). Dense
FP16-matrix ≈ 190–200 TFLOPS, FP8/INT8 ≈ 390 TFLOPS, INT4 ≈ 780 TOPS.

---

## 0. TL;DR

1. **Correction to the brief's headline number.** The middle 12 layers (6–17, `core_end` is
   exclusive — README says "48 layer applications") run **three passes** under `loop_config`.
   Global attention therefore fires at **8 layer-passes**, not 4: unrolled positions
   {5, 11, 17, 23, 29, 35, 41, 47}. Unless the loop shares one KV slot per token (an unusual
   "MELT-style" design — see §9), the global KV is **32 KiB/token bf16**, i.e. **32 GiB at 1M**,
   not 16 GiB. The MTP `target_layer_ids = [5, 23, 41, 48]` land exactly on three of these
   unrolled global positions, which supports the per-pass interpretation. **Verify against the
   SGLang/HF reference implementation before locking the allocator.**
2. Per-pass KV also doubles the SWA budget: 40 local layer-passes × 1024-token window × 4 KiB
   ≈ **160 MiB** (constant), still negligible.
3. **Recommended default: FP8-E4M3 storage for global-layer KV, bf16 compute.** Halves bytes and
   PCIe traffic with ~0 measured quality loss (vLLM, KIVI, SqueezeBits evidence below). Keep the
   160 MiB of SWA KV in bf16 — quantizing it buys nothing. Do **not** run the attention
   accumulation itself in FP8 on long context (vLLM's 128K needle regression, §2).
4. **The global layers already are the "retrieval" layers.** SWA+4-global is a layer-level
   DuoAttention. The highest-value remaining lever is DuoAttention-style *head* eviction inside
   those 8 global passes (<5% of heads are retrieval heads in every studied model), which stacks
   multiplicatively with quantization.
5. **At ≥256K the design must be tiered**: GPU holds a hot working set (sinks + recent window +
   query-selected pages), 48 GB RAM holds the fp8 cold KV, NVMe holds dormant prefix blocks for
   agentic reuse. Permanent-eviction methods (H2O/TOVA/SnapKV) are the *wrong* tool — query-aware
   selection (Quest/ShadowKV style) preserves needles; eviction destroys them.
6. **Prefill is the real 1M problem**, not decode: dense global attention costs ~65 PFLOP
   across a 1M-token prefill → ~7–11 min on this card even before dequant; naive RAM-tiered
   dense
   prefill moves ~1 TiB over PCIe → hours. Options: dense prefill only while fp8 KV fits in
   VRAM (≈128–256K), MInference-style head-pattern sparse prefill (~10× attention reduction), or
   int2/int4 GPU-resident KV as an "extended" mode.

---

## 1. KV-state inventory (from config.json + safetensors index)

Per token, per attention layer-pass:

| Component | Size (bf16) | Notes |
|---|---|---|
| K (8 KV heads × 128) | 2,048 B | post-RoPE-or-rel, **post k_sconv** (causal conv, kernel 4) |
| V (8 KV heads × 128) | 2,048 B | post v_sconv; `use_value_norm_no_gamma` → V is normalized |
| rel path (`wr_du`, d_rel=16) | ~256–512 B | `rel_logits_proj`+`rel_l2_clip`; extent 1024 — verify whether rel keys are cached |
| k_sconv / v_sconv rolling state | ~18 KiB/layer **total** | last 3 pre-conv vectors × 2 × 2304/1024-dim — tiny, keep bf16, needed at chunk boundaries |
| attn_sconv / mlp_sconv state | ~18 KiB/layer | kernel-4 rolling state, constant size |
| learned `attn_sink_proj` | weight only | sink is a learned projection, not a token — good news for eviction |
| MTP draft head (3 layers, 1 global) | +4 KiB/token | see §9 |
| `controllers` (down/forget/gain) | weight only | per-pass modulation of q/k/v/mlp streams on layers ≥10 — this is *why* each pass needs its own KV |

Derived totals (per-pass KV assumed):

- **Global KV: 8 passes × 4 KiB = 32 KiB/token bf16** (16 KiB fp8/int8, 8 KiB int4).
- If reference impl shares one cache across loop passes: 4 × 4 KiB = 16 KiB/token bf16. Halve
  every table entry below.
- **SWA KV: 40 passes × 1024 × 4 KiB ≈ 160 MiB fixed** (window caps it).
- sconv + MTP-side states: ≤ a few MiB fixed + 4 KiB/token for the MTP global layer.

---

## 2. KV quantization: what the evidence says

### 8-bit is free; sub-8-bit needs care

- **vLLM FP8 KV (E4M3), April 2026 stress test**
  (https://vllm.ai/blog/2026-04-22-fp8-kvcache): near-baseline accuracy on validated paths —
  *with two traps that matter to us*. (a) On Hopper FA3, FP8 attention **accumulation** lost
  precision at long context: 128K needle-in-haystack fell **91% → 13%**, traced to imprecise
  FP32 accumulation over large contraction dims (same issue DeepSeek-V3 hit in training); fixed
  by two-level accumulation. (b) On hybrid models with small sliding-window layers, FP8 bought
  ~nothing (SWA layers dominate ITL time) — "skipping those layers is often better." Both map
  directly onto us: **store FP8, dequantize to bf16 in the kernel, accumulate in fp32/bf16**;
  and leave SWA KV in bf16.
- FP8 KV is officially supported on **ROCm** (`fp8_e4m3`; `fp8_e5m2` CUDA-only):
  https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/. For a hand-rolled
  RDNA4 kernel, E4M3 is the right format (better mantissa; the K/V range is small thanks to
  value-norm + l2-clip, §2.3).
- Community validation at native 256K context (Qwen3.8-27B, FP16 vs FP8 KV): 80/120 vs 82/120 —
  noise-level difference, identical decode speed:
  https://thetechbasement.no/journal/opencode-qwen38-vllm/
- SqueezeBits' framework comparison: 8-bit KV is accuracy-neutral; 4-bit already hurts,
  2-bit hurts a lot: https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079

### Sub-8-bit: KIVI / KVQuant / GEAR / ZipCache

- **KIVI** (ICML'24, https://arxiv.org/abs/2402.02750): the key result for layout decisions —
  **Keys quantize per-channel, Values per-token.** 2-bit K per-channel + V per-token + a small
  fp16 residual buffer of the last ~32–128 tokens is near-lossless; 2-bit per-token-everything
  collapses (GSM8K 12.2 → 20.8 only with the residual). Per-channel K conflicts with
  token-append streaming — solved by the residual buffer + group quantization of drained
  tokens. If we ever go ≤4-bit, copy this scheme.
- **KVQuant** (NeurIPS'24, https://arxiv.org/abs/2401.18079): pushes to ~3.3-bit via per-channel
  keys, **pre-RoPE key quantization** (quantize before rotary, apply RoPE to dequantized keys —
  RoPE couples channel pairs, quantizing post-RoPE is worse), non-uniform codebooks, and
  Dense-and-Sparse decomposition (~1% outliers in fp16). Demonstrates perplexity parity toward
  10M context on LLaMA. Caveat: kernel complexity is high; treat as a research direction, not
  v1.
- **GEAR** (ICML'24 ENLSP, https://proceedings.mlr.press/v262/kang24a.html): quantized backbone
  + low-rank residual + sparse outlier matrix; near-lossless at 2-bit, +24% over baselines,
  2.39× peak-memory cut. Same lesson: at ≤4-bit you need an error-residual path.
- **ZipCache** (NeurIPS'24, https://arxiv.org/abs/2405.14256): channel-separable tokenwise
  quantization (normalize channels once, then per-token scales — gets most of per-channel
  quality without group-parameter bloat) + salient-token fp16 retention. A pragmatic middle
  ground if per-channel K storage is awkward in our paged layout.

### Per-channel vs per-token — decision for our engine

- Per-token scales fit paged layouts naturally (one scale+zero per token per head per page row);
  per-channel K wants scales along the sequence dimension — awkward but compatible with a
  page-group granularity: store per-channel min/max **per 64-token page**, which is exactly the
  metadata Quest-style selection wants anyway (§5). So: **8-bit → per-token (or
  channel-separable). ≤4-bit → per-channel-per-page K / per-token V + fp16 residual window.**
- This model is unusually quantization-friendly: `use_value_norm_no_gamma` bounds V magnitude;
  `q_l2_clip`/`rel_l2_clip` bound logits (clip ≈ 22.6/8.0) so QK errors can't explode through
  softmax; k/v_sconv smooths per-token outliers. Expect FP8 KV to be effectively free and even
  int4-with-residual to be plausible — but **validate on a ≥128K needle task per the vLLM
  finding**; nothing substitutes for measuring it on this model.

---

## 3. Paged layout for a hand-rolled engine

- Baseline: **PagedAttention** (vLLM paper, https://arxiv.org/abs/2309.06180): fixed-size
  physical blocks + per-sequence block tables; <4% waste, no external fragmentation, enables
  copy-on-write sharing. We need a **hybrid** variant: two block pools —
  - *Global pool*: pages allocated for every token (all 8 passes).
  - *SWA pool*: ring buffer of ⌈1024/page⌉ pages per pass; tokens older than the window are
    freed immediately — exactly vLLM's hybrid KV-cache manager policy
    (https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/): "full layers reserve all
    tokens; sliding-window layers reserve only the last `sliding_window_size`."
  - vLLM PR #29143 (https://github.com/vllm-project/vllm/pull/29143) solves our exact
    mixed-dtype case (bf16 SWA + fp8 global) by giving each layer *group* a different block_size
    so physical bytes/page match, and hashing prefix blocks at the LCM granularity — steal this.
- **Page size**: 16 tokens (vLLM default) is tuned for decode; for PCIe/NVMe tiering choose
  **64 tokens/page**: global page = 8 passes × 64 × 4 KiB = 2 MiB bf16 / 1 MiB fp8 — efficient
  PCIe bursts, 16,384 pages at 1M. Per-page K min/max metadata for selection: 8 heads × 128 ch
  × 2 (min,max) × 4 B = 8 KiB per page per pass → ~128 MiB per pass at 1M → **~1 GiB total fp32,
  ~0.5 GiB fp16** — keep metadata resident on GPU.
- Allocation is trivially single-user: one (few) sequences, so skip vLLM's multi-tenant
  machinery; a flat block table + freelist + lock-free DMA queue is enough.

---

## 4. Prefix / radix caching for agentic reuse

Single-user agentic loops re-send huge shared prefixes (system prompt + tool history + repo
context). Research consensus:

- **RadixAttention** (SGLang, https://arxiv.org/abs/2312.07104): keep KV of finished requests in
  a radix tree keyed by token ids; longest-prefix match → instant reuse; LRU evict; ~free when
  no hit. For us: tree over **page-aligned blocks** (hash block contents like vLLM's
  `BlockHash`), which gives reuse even after tree eviction via a hash→tier lookup.
- **LMCache-style tiered reuse** (https://docs.lmcache.ai/developer_guide/architecture.html,
  https://llm-d.ai/docs/architecture/kv-management/kv-offloader): GPU → pinned CPU RAM → NVMe;
  hits in RAM/NVMe are "up to 16× faster than recomputing long prompts" and, critically for us,
  **turn a 10-minute 1M prefill into a 30–60 s load** (16 GiB fp8 KV from RAM at ~40 GB/s
  ≈ 25 s; from Gen4/5 NVMe ≈ 2–8 s of I/O, overlapped layerwise).
- Design notes for this model:
  - SWA layers need only the last 1024 tokens of a reused prefix — a prefix hit can skip
    re-materializing SWA KV entirely (vLLM hybrid manager formalizes this; huge saving: only the
    8 global passes must be restored).
  - k/v_sconv state means a *prefix boundary* must store the 3-token conv state (or re-run conv
    on the boundary window) — store it next to the page metadata, it's bytes.
  - Cache keyed blocks by (token-hash, model-version, dtype); quantize-on-write (fp8) so cold
    blocks are half the size.

---

## 5. Sparse attention / KV eviction — honest numbers

Two families. **Permanent eviction** (decide once, drop forever) and **query-aware selection**
(keep everything, fetch what each query needs). The evidence is unflattering to the first for
retrieval workloads:

- **H2O** (accumulated-attention heavy hitters) collapses at tight budgets: in the
  "Who Should Score the Cache?" study (https://arxiv.org/abs/2611.02230) H2O scored **8–22%**
  where no-eviction scored 76–93%; SnapKV 28–58%; PyramidKV's depth schedule *hurt*
  (−8 to −29 pp vs SnapKV at equal budget). Quest's analysis (https://arxiv.org/abs/2406.10774)
  explains why: token criticality is *query-dependent* — a token irrelevant to all past queries
  becomes the needle for the next one, and eviction can't recover it.
- **TOVA** (https://arxiv.org/abs/2402.06262): simplest possible policy (drop lowest current
  attention) — decent perplexity, but it's still permanent loss; numbers on zero-shot long-ctx
  tasks are mediocre.
- **SnapKV** (https://arxiv.org/abs/2404.14469): compress once at prefill using the observation
  window's attention; keeps 1–4K tokens with mild LongBench degradation. Fine for "ask once
  about a doc," wrong for an agentic session that re-queries the same context with different
  questions — the compression is query-specific at heart.
- **PyramidKV** (https://arxiv.org/abs/2406.02069): smaller budgets for higher layers; the
  newer head-to-head above shows the pyramid schedule is not robust — skip.
- **Locret** (https://arxiv.org/abs/2410.01805): the best eviction result — trained per-head
  "retaining heads," **chunked-prefill compatible**, claims 20× KV compression with <10% quality
  loss and 128K on a 4090. Costs a small training run + trust in a learned evictor; it is a
  *training-time* method, not a deploy-time switch.
- **Quest** (https://arxiv.org/abs/2406.10774): per-page per-channel **min/max key metadata**;
  at each step, bound each page's max possible logit against the current query, attend only to
  top-k pages + recent window. 7× attention speedup, ~lossless needles at 100K+; metadata is the
  same min/max we'd store for per-channel quant — one structure serves both.
- **TokenSelect** (https://arxiv.org/abs/2411.02886): query-aware token-level (not page-level)
  selection, ~24× attention speedup claims — finer granularity, more complex gather.
- **MInference** (https://aka.ms/MInference, NeurIPS'24): classifies each head offline into
  A-shape / vertical-slash / block-sparse; ~10× prefill speedup at 1M on one A100 (SGLang reports
  up to 15× at 1M: https://github.com/microsoft/MInference/) with NIAH intact. This is the only
  demonstrated path to *dense-quality* 1M prefill on one GPU — see §7.
- **Retrieval heads / DuoAttention** (https://arxiv.org/abs/2404.15574,
  https://arxiv.org/abs/2410.05263 → ICLR'25 PDF
  https://proceedings.iclr.cc/paper_files/paper/2025/file/5c1ddd2e59df46fd2aa85c833b1b36ed-Paper-Conference.pdf):
  <5% of heads do cross-context retrieval; DuoAttention keeps full KV only on them, gives
  streaming heads a sink+recent cache → 1.67× memory (GQA) — and with quantization pushed
  Llama-3-8B to 3.3M ctx on one A100.

### What this means for Inkling 2 Mini

The architecture is a **layer-level DuoAttention baked into training**: 40 of 48 layer-passes
attend only 1024 back; 8 global passes carry the long-range load. So "does SWA+few-global
already give most of the win?" — **yes, ~84% of the KV that a dense model would keep is already
structurally gone** (48 passes × ctx vs 8 × ctx + 40 × 1024). The remaining questions:

1. *Can we evict within the 8 global passes' 8 KV heads?* Almost certainly partially —
   retrieval-head sparsity (<5%) holds across all studied GQA models. A DuoAttention-style
   calibration (synthetic retrieval probe, per-head gate) could plausibly cut global KV
   **2–4×** (keep ~2–4 of 8 KV heads full). Because heads share KV in GQA (24 q-heads → 8 kv),
   granularity is the KV head. This stacks with fp8 → **4–8 GiB at 1M** — i.e. could keep the
   *entire* 1M KV on-GPU. Top-priority experiment, needs a calibration harness; risk: this
   model's heads may be less redundant since the network already offloaded locality to SWA.
2. *Permanent token eviction inside global layers?* Don't — the evidence above says it breaks
   exactly the retrieval behavior the global layers exist for. If VRAM pressure demands it,
   prefer Locret-style trained retaining heads over H2O/SnapKV heuristics.
3. *Query-aware selection?* Yes — this is the mechanism that makes the tiered KV work (§6):
   select ~8–32K tokens of top pages per step per pass; it's *approximate per step* but
   *lossless across steps* (nothing is ever deleted, different queries fetch different pages).

---

## 6. KV offload / tiering — GPU ↔ RAM ↔ NVMe

Bandwidth math (PCIe 5.0 x16, ~50 GB/s achievable):

- **Full-KV decode step at 1M, fp8 (8-pass)**: 16 GiB → 340 ms → **~3 tok/s**. bf16: ~1.5 tok/s.
  Unacceptable → selection is mandatory, not optional.
- **Selected decode step**: resident recent window (GPU) + fetched top-k pages: e.g. 8 passes ×
  8K tokens × 2 KiB = 128 MiB → ~3 ms/step → KV I/O disappears under the ~5–10 ms weight-read
  floor (5.4B active × ~0.6 B/param fp4 ≈ 3.2 GB @ 640 GB/s). **This is the whole design.**

Systems to copy:

- **InfiniGen** (OSDI'24, https://www.usenix.org/conference/osdi24/presentation/lee): offloaded
  KV + *speculative prefetch* — rehearse next layer's query cheaply to prefetch only essential
  KV; up to 3× over naive offload. Our loop structure is even better: while pass *p* computes,
  prefetch pass *p+1*'s selected pages — deep pipelining across 8 global reads/step.
- **ShadowKV** (https://arxiv.org/abs/2410.21465, ICML'25): low-rank-compressed keys stay on
  GPU, values offload to CPU; per-chunk landmark keys + cosine similarity select what to pull;
  outliers stay resident. 6× batch, 3× throughput on A100 at 1M-class contexts. Its landmark
  idea doubles as our prefill selector (§7).
- **LMCache** (§4) for the RAM/NVMe tiers; **PulseInfer** (https://arxiv.org/abs/2609.34555) for
  why to coalesce sparse recalls into big gathered I/Os (fragmented page recalls kill PCIe) —
  gather selected pages into one staging buffer per pass.
- llm-d KV-offload docs (https://llm-d.ai/docs/architecture/kv-management/kv-offloader):
  practical tiering semantics — writes async, reads on critical path, overlap per-layer.

Tier plan at 1M (8-pass, fp8 = 16 GiB global + ~1 GiB page metadata + 160 MiB SWA):

| Tier | Contents | Size |
|---|---|---|
| VRAM (~4–6 GiB KV budget; weights take the rest) | sinks+first page, last 32–64K tokens (~0.5–1 GiB), per-step top-k staging (~0.25 GiB), page min/max metadata (~0.5–1 GiB), SWA KV (0.16 GiB) | ~3–4 GiB |
| RAM | full fp8 global KV + metadata | ~17 GiB |
| NVMe | dormant sessions, hash-keyed prefix blocks for reuse (§4) | unbounded |

48 GB RAM holds 16 GiB KV + resident weights headroom fine; **bf16 1M (32 GiB) + RAM weights
would blow the RAM budget** — fp8 KV is what makes 1M workable at all in 48 GB.

---

## 7. Chunked prefill + TTFT estimates

### Chunk size

Sarathi-Serve (OSDI'24, https://arxiv.org/abs/2308.16369): prefill throughput saturates at
512–2048-token chunks on datacenter GPUs (smaller for big hidden sizes); stall-free mixing of
chunks with decodes is the point. We're single-user and our hidden size is small (2304), so
saturation needs **bigger** chunks: recommend **8K–32K-token chunks** (default 8192). Reasons
beyond saturation: bounds transient attention memory, lets each chunk's KV stream to RAM as it's
produced, gives natural points for Locret/MInference-style selection, and keeps MTP-draftable
decode interleaved on long ingests.

### FLOP model (per-pass KV, dense attention)

- Weights: 2 × 5.4 G-active ≈ **10.8 GFLOP/token**.
- Global attention during prefill: 8 passes × ~12,300 FLOP/token-of-KV-depth → at average depth
  S/2: **≈ 49K × S FLOP/token**.
- SWA attention: 40 passes × 12,300 × 1024 ≈ 0.5 GFLOP/token (flat).

| Prompt | Attn GFLOP/tok | Total PFLOP | @150 TF eff | @100 TF eff |
|---|---|---|---|---|
| 32K | 1.6 | 0.42 | ~3 s | ~4 s |
| 128K | 6.3 | 2.3 | ~15 s | ~23 s |
| 512K | 25.2 | 19.1 | ~2.1 min | ~3.2 min |
| 1M | 51.5 (avg depth 500K) | ~65 | ~7 min | ~11 min |

(Effective rate = hand-rolled FP16 WMMA at ~50–75% of ~195 TFLOPS dense peak. Sanity check:
MInference measured ~30 min prefill for an 8B dense model at 1M on A100-312TF
(https://aka.ms/MInference); our attention FLOPs are ~5× lower than a dense 48-layer model, so
~7–11 min dense is consistent.)

**But the memory wall breaks dense prefill first**: fp8 global KV exceeds the ~4–6 GiB GPU
budget at ~256–384K. Beyond that, dense prefill must stream KV from RAM per chunk — ~1 TiB of
PCIe traffic over a full 1M prefill → **hours**. So:

- **≤128–256K**: dense prefill, everything on GPU. Fine.
- **>256K**: needs *sparse* prefill — per 8K-token chunk, score landmarks/page-metadata, fetch
  top pages, exact-attend over ~5–10% of KV. MInference shows this stays needle-lossless at 1M;
  ShadowKV/InfLLM use the same trick. Engineering cost is real — schedule it as the "extended
  context" feature, not v1.
- Mitigation that ships free: **prefix reuse** (§4) — in practice 1M prompts are assembled from
  cached pieces; with radix+NVMe tiers, TTFT for a re-queried 1M context is seconds.

### Decode speed at 1M (fp8, selected pages)

Weight-read floor ~5 ms + KV page fetch ~3 ms + local attention → **~15–40 tok/s** plausible
end-to-end (weights residency is the other team's problem; if weights stream from RAM it's
slower). Compare: any design that touches all 16 GiB per step is ≤3 tok/s.

---

## 8. Concrete memory plan

Global-layer KV bytes/token = 8 passes × 2 (K,V) × 8 heads × 128 × dtype-bytes.
(If the loop shares KV — verify! — halve the *global* column.)

### Global KV totals

| Context | bf16 (32 KiB/t) | FP8/INT8 (16 KiB/t) | INT4 (8 KiB/t) | INT2+residual (~5 KiB/t) |
|---|---|---|---|---|
| 32K | 1.0 GiB | 0.5 GiB | 0.25 GiB | 0.16 GiB |
| 128K | 4.0 GiB | 2.0 GiB | 1.0 GiB | 0.6 GiB |
| 512K | 16 GiB | 8.0 GiB | 4.0 GiB | 2.5 GiB |
| 1M | 32 GiB | 16 GiB | 8.0 GiB | 5.0 GiB |

Plus fixed: SWA 160 MiB + sconv/MTP state ~10 MiB + MTP global layer 4 KiB/tok (bf16; can be
fp8 → 2 KiB) + selection metadata ~0.5–1 GiB at 1M.

### Where it lives at 1M

| Scheme | VRAM | RAM | NVMe | Decode feel |
|---|---|---|---|---|
| bf16 dense (impossible) | — | 32 GiB | — | ~1.5 tok/s if streamed; can't anyway |
| **fp8 + tiered + Quest-select (RECOMMENDED)** | ~4 GiB hot | ~17 GiB | prefix store | ~15–40 tok/s |
| fp8 + DuoAttention-head-evict 2–4× | 4–8 GiB (all fits!) | — | prefix store | ~50+ tok/s, best if calibration works |
| int4 + tiered | ~3 GiB hot | ~9 GiB | prefix store | like fp8, slightly worse quality risk |
| int2 KIVI-style | ~3 GiB hot | ~6 GiB | — | more VRAM headroom, quant risk highest |

**Recommended defaults**
- Storage: **fp8_e4m3 global KV** (per-token scales initially; per-channel-per-page if ≤4-bit
  ever ships), bf16 compute path, bf16 SWA, 64-token pages, 8K-token chunks.
- Context tiers: **0–256K** = all-GPU dense. **256K–1M** = tiered + page selection.
- Budgets: sinks+first pages pinned, recent 32K pinned, top-k ≈ 8K tokens/pass fetched/step.
- Persist all cold KV pages hash-keyed to RAM→NVMe for agentic prefix reuse.

---

## 9. Model-specific notes & open questions

- **log_scaling** (`alpha=0.1`, `n_floor=8192`): logits scaled by a slowly growing function of
  position beyond 8K (logn-style extrapolation — the model is *designed* to stretch to 1M from
  shorter training). Implications: attention distributions are deliberately sharpened/flattened
  by position — don't apply ad-hoc softmax temperature in kernels; confirm exact formula in the
  reference impl. Quant interaction: sharpened distributions are *more* sensitive to K noise —
  another reason to keep K at fp8+ per-channel rather than int4 per-token.
- **sconv layers**: every layer carries attn_sconv + mlp_sconv (kernel 4) and global layers
  convolve K and V themselves (k_sconv/v_sconv). Consequences: (a) store **post-conv** K/V in
  the cache; (b) conv rolling state (last 3 inputs) must be carried across chunk boundaries and
  stored with each prefix-cache checkpoint — bytes, but a correctness bug if forgotten; (c)
  conv on K means K of token t depends on tokens t-3..t — quantize post-conv.
- **MTP head**: 3 layers {local, global, local} + markov_rank 256 rnn hidden. Its global layer
  wants its own KV (+4 KiB/tok bf16). At 1M that's 4 GiB — make MTP's global KV fp8, share the
  same tiering, or degrade MTP gracefully beyond a depth threshold (draft acceptance may drop
  anyway at extreme context).
- **Learned `attn_sink_proj`**: sinks are parameter-generated, not token positions — still keep
  page 0 pinned (cheap insurance; StreamingLLM https://arxiv.org/abs/2309.17453 shows initial
  tokens carry real mass even with learned sinks).
- **Controllers / loop**: per-pass down/forget/gain projections on q/k/v streams — this is what
  forces per-pass KV; also means a pass-p cache slot is written exactly once per token, so the
  block table can be indexed by (unrolled position, token, pass) with zero ambiguity.
- **Open questions for the parent**: (1) confirm per-pass vs shared loop KV (check SGLang/HF
  ref impl `past_key_values` shape — [batch, 48?, heads, seq, dim]); (2) does the rel path cache
  keys (d_rel=16 → ~12% overhead if so); (3) does the reference store sink tokens or regenerate
  them; (4) measure fp8-vs-bf16 on a ≥128K needle + RULER-style suite before committing —
  vLLM's 91%→13% accumulation trap is exactly the kind of bug a hand-rolled kernel repeats.

---

## Sources

1. vLLM FP8 KV-cache & attention quantization stress test — https://vllm.ai/blog/2026-04-22-fp8-kvcache
2. vLLM quantized KV cache docs (fp8_e4m3 ROCm) — https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/
3. Qwen3.8 FP8-vs-FP16 KV at 256K — https://thetechbasement.no/journal/opencode-qwen38-vllm/
4. SqueezeBits vLLM-vs-TRT-LLM KV quantization — https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079
5. KIVI — https://arxiv.org/abs/2402.02750
6. KVQuant — https://arxiv.org/abs/2401.18079
7. GEAR — https://proceedings.mlr.press/v262/kang24a.html
8. ZipCache — https://arxiv.org/abs/2405.14256
9. PagedAttention/vLLM — https://arxiv.org/abs/2309.06180
10. vLLM hybrid KV cache manager — https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/ + PRs https://github.com/vllm-project/vllm/pull/29143 , https://github.com/vllm-project/vllm/pull/11938
11. RadixAttention/SGLang — https://arxiv.org/abs/2312.07104
12. LMCache — https://docs.lmcache.ai/developer_guide/architecture.html ; llm-d KV offloading — https://llm-d.ai/docs/architecture/kv-management/kv-offloader
13. H2O critique / scoring study — https://arxiv.org/abs/2611.02230 ; SnapKV — https://arxiv.org/abs/2404.14469 ; PyramidKV — https://arxiv.org/abs/2406.02069 ; TOVA — https://arxiv.org/abs/2402.06262 ; Locret — https://arxiv.org/abs/2410.01805
14. Quest — https://arxiv.org/abs/2406.10774 ; TokenSelect — https://arxiv.org/abs/2411.02886
15. DuoAttention — https://proceedings.iclr.cc/paper_files/paper/2025/file/5c1ddd2e59df46fd2aa85c833b1b36ed-Paper-Conference.pdf ; Retrieval heads — https://arxiv.org/abs/2404.15574
16. InfiniGen — https://www.usenix.org/conference/osdi24/presentation/lee ; ShadowKV — https://arxiv.org/abs/2410.21465 ; PulseInfer — https://arxiv.org/abs/2609.34555
17. MInference — https://aka.ms/MInference + https://github.com/microsoft/MInference/
18. Sarathi / chunked prefill — https://arxiv.org/abs/2308.16369 + https://www.usenix.org/system/files/osdi24-agrawal.pdf
19. StreamingLLM attention sinks — https://arxiv.org/abs/2309.17453
20. RX 9070 XT specs — https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070.html + https://www.techpowerup.com/review/amd-radeon-rx-9070-series-technical-deep-dive/3.html
21. Looped-model KV designs (per-pass vs shared) — https://arxiv.org/html/2608.09444 (depth-indexed KV) ; https://arxiv.org/html/2605.07721v2 (MELT shared KV)
