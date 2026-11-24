# THEORY — ceilings and budgets for ink-runtime

RX 9070 XT (gfx1201, 16GB, ~640 GB/s GDDR6 spec) + i7-13700KF (2-ch DDR5) + PCIe-4 NVMe, Windows 11.
Model: Inkling-2-Mini — 26.4B params, ~3.4B active/token, 128 experts top-8, looped core (24 weight layers = 48 layer-apps; global attn at 8 unrolled positions), NVFP4+MXFP8 checkpoint 19.5GB, 1M ctx, DFlare-style block draft head.

Sources: `research/01` (hardware), `03` (ROCm), `04` (APIs), `05`/`11` (tiered runtimes, MoE streaming), `09` (KV), `10`/`12` (NVMe/bandwidth measured).

## 1. Bandwidth hierarchy (measured, not spec)

| Tier | Planning number | Implication |
|---|---|---|
| VRAM GEMV | ~500 GB/s (455–530 in llama.cpp; 592 cl-mem) | resident weights = ~1µs per MB/token |
| PCIe 5 x16 pinned H2D | ~45–55 GB/s (board-dependent 38–57 — **verify LnkSta**) | 4.5MB expert ≈ 90µs |
| CPU GEMV over **DDR4-2666 (measured: mismatched 2×8+2×16GB kits)** | ~35–42 GB/s theoretical → ~25–35 effective | 4.5MB expert ≈ 130–180µs — **slower than PCIe**; prefer H2D streaming |
| NVMe expert reads (QD8+, ≥1MB) | ~4–6.5 GB/s | 4.5MB expert ≈ 0.7–1.1ms |
| NVMe 4K QD1 / mmap faults | ~0.1–1.2 GB/s | forbidden on hot path |

Decode ceiling formula: `tok/s ≈ bandwidth_of_slowest_tier_touched / bytes_read_per_token`.

## 2. Per-token weight traffic

- Experts/token: 48 MoE execs × 8 = ≤384 activations; ~4.48MB fp4-expert / ~8.2MB fp8-expert → **1.84 GB/token fully cold**.
- With ~9–11GB expert pool in VRAM (~2,000–2,400 of 3,072 slots = 65–80% coverage) + within-token loop reuse: realistic touched bytes ~0.2–0.5 GB/token.
- Dense trunk (attn, embed, routers, norms, vision-on-demand, MTP head 636MB): ~3.8GB bf16 → resident.
- **Ceilings**: all-SSD ~3–8 t/s; all-RAM-CPU ~30–60 t/s; RAM-streamed GPU ~25–60 t/s; good-cache+prefetch ~40–80 t/s; theoretical VRAM-resident ~180 t/s (unreachable — 15.7GB experts > free VRAM).
- MTP verify amortizes each fetched expert over ~9 positions (~2–4× effective) — target **80–140 t/s decode** is plausible per existing data points (Unsloth llama.cpp fork already does 84.9 t/s on this card/model).

## 3. KV budget (verified correction: 8 global positions, not 4)

| Ctx | SWA (40 passes × 1024 × 4KiB) | Global bf16 (8 × 4KiB/tok) | Global fp8 | int4 |
|---|---|---|---|---|
| 128K | 160 MiB | 4 GiB | 2 GiB | 1 GiB |
| 512K | 160 MiB | 16 GiB | 8 GiB | 4 GiB |
| 1M | 160 MiB | **32 GiB** | **16 GiB** | 8 GiB |

- fp8-E4M3 storage + bf16 compute is the default (vLLM 128K NIAH regression was fp8 *accumulation* — store fp8, accumulate fp32/bf16).
- Tiered design at >256K: GPU hot = sinks + last ~32–64K + query-selected pages (Quest min/max metadata); RAM = full fp8 KV; NVMe = dormant prefix blocks (radix-hash reuse → a re-queried 1M ctx loads in seconds, not minutes).
- Dense prefill: 0–256K fine; 1M ≈ 65 PFLOP ≈ **7–11 min** and ~1TiB PCIe KV spill → needs MInference-style sparse prefill or prefix reuse (the realistic path for 1M is *assembled* context, not fresh 1M prompts).
- Open verify item: does the loop share KV across passes or keep per-pass caches (halve global column if shared).

## 4. Hardware tuning budget ("the 3090-unlock analog")

No math-rate fuse on AMD. Gains are platform/software:

| Knob | Effect | Cost |
|---|---|---|
| **PCIe Gen5 x16 — VERIFIED on this box** (PnP: CurrentLinkSpeed=5, Width=16) | max H2D ~50–55 GB/s available | done |
| **ReBAR — VERIFIED ON** (heap1 = 15.92GiB DEVICE_LOCAL+HOST_VISIBLE+COHERENT) | zero-copy CPU→VRAM staging works | done |
| **Dedicated TRANSFER queue — VERIFIED** (family2, SDMA) + 8 compute queues | async H2D/compute overlap on Win prop driver | done |
| `VK_EXT_external_memory_host` present | RAM-resident experts readable by shaders over PCIe | done |
| Undervolt −100…−120 mV | +8–9% when power-limited (junction ~104°C is the wall) | stability test required |
| Power limit +10% (MCT → 340W) | stacks with UV | thermal |
| Memory OC | **negative** — GDDR6 link retry makes it *slower* (−6.6% measured); leave stock | — |
| PCIe ASPM off | +10.8% decode (dense), ~0 MoE | free |
| perf-level `auto` (not peak/high) | peak = −23% on mem-bound; high = −4.3% | free |
| RAS/ECC off if enabled | ~+9% BW on R9700 sibling; verify on consumer | free if present |
| Keep GPU saturated (fused kernels, prerecorded cmd bufs) | avoids mclk 96↔1258 bounce | design |
| WMMA coverage of every GEMM/attention | up to 4× vs scalar pipe (195→780 TF sparse FP8); the real "unlock" | kernel work |
| FP8 WMMA (HIP path) | 2× FP16 rate for MXFP8 layers | backend work |

## 5. API decision

**Vulkan compute primary** (Windows): zero install friction (Adrenalin = the runtime), coopmat fp16/bf16/int8 confirmed on this driver, decode ≈ parity with HIP, portable to Linux/RADV where it wins MoE decode. **HIP SDK 7.2 secondary**: the only path to native FP8 WMMA (+2× on MXFP8 layers) and real profiling; also the Linux dev target. D3D12/DML out (no stable WMMA until SM6.10); DirectStorage unnecessary (replicate its ring in ~200 lines, and it's D3D12-bound). Kernels behind a vtable; Slang as shader language (one source → SPIR-V now, DXIL later).

## 6. Kernel plan

- **NVFP4**: no FP4 instruction on gfx1201 → dequant-fused GEMV (wave64, `v_dot4`/`v_bfe` unpack, E4M3-block×fp32-global scale in epilogue) for decode; dequant→bf16 WMMA (wave32) for prefill. Alternative: INT4-WMMA reinterpretation (779 TOPS) — benchmark both.
- **MXFP8**: native E4M3 → FP8 WMMA (HIP) or FP8→BF16 convert (Vulkan until FP8-coopmat enumerates).
- **Attention**: custom kernel mandatory (SWA 1024 + QK-L2 clip + softcap 30 + rel-pos d_rel=16 + learned sink). Base: ck_tile FMHA (SWA + softcap already templated, HIP); Vulkan port of same structure. wave64.
- **MoE exec**: dequant-fused per-expert GEMV at decode (M=1–9 — bandwidth-bound, skip grouped-GEMM padding); grouped GEMM (BLOCK_M) for prefill; `moe_sorting` (aiter) as reference.
- **Dispatch overhead**: ~7k launches/token kills naive backends (73% GPU time in kTransformers profile) → pre-recorded command buffers per shape (decode/draft-verify/prefill-bucket), timeline semaphores, dedicated transfer queue for H2D.

**RAM caveat (measured on this box)**: 48GB = 2×8GB DDR4-2666 + 2×16GB DDR4-3200 mismatched → all run at 2666 MT/s in flex-asymmetric mode → ~40 GB/s tier bandwidth, ~half the assumed DDR5 budget. Consequences: (a) PCIe-5 streaming (~50 GB/s) beats CPU-side expert compute (~25–35 GB/s delivered) — invert the Fiddler default: prefer H2D copies for misses, keep CPU-exec only for truly-late fp8-tail misses or when PCIe is saturated; (b) an XMP bump to 3200 is capped by the slow sticks — treat ~40 GB/s as the planning number; (c) the RAM tier still holds the full 15.7GB expert mirror + fp8 KV — capacity is unaffected, only its bandwidth.

## 7. The expert-tiering engine (the differentiator, ~1k LOC)

- Slot-pool VRAM cache (HF/Strata pattern): one preallocated buffer + per-layer `expert_id→slot`; grouped exec reads in place.
- RAM = full expert mirror (15.7GB fits); NVMe = repacked **expert-pack file** (contiguous per-expert blocks — SSD-LLaMA layout) for cold misses/fast start.
- Prefetch: run next layer's router on current hidden state during current expert exec (free [2304×128] GEMV); depth 1; **cross-pass prediction inside the loop** (novel — same expert pool refires pass 2/3: pin loop-pass working set).
- Early preemption at gate time; chunked per-matrix fetch; CPU-exec misses discovered too late (~break-even 4.5MB, win on 8.2MB fp8 tail).
- Eviction: composite (staleness, predicted-reuse, request-freq, fp8-size-penalty); never evict loop-pinned or ≤1-layer-ahead-needed.
- Prefill streams everything (dense activation) and harvests per-request histogram → seeds decode cache.
- k1/k2 renorm knob: top-8→top-4 normalize-by-top-16 ≈ −0.35pts (Qwen3.6 measured) = ~2× expert-bandwidth lever — verify router renorm semantics first.

## 8. MTP/draft integration

Head = DFlare block drafter (3 layers, taps hidden at apps {5,23,41,48} = real L4/L10p2/L16p3/L23) + rank-256 GRU markov chain; 8-token block, linear verify at 9 positions. Runner must expose mid-loop hidden taps. Draft lives in VRAM permanently (636MB). Verify caps expert union ~12–16/layer (MoE-Spec); log per-verify union → adaptive block length. N-gram proposer as warmup + fallback. Expected +2–3× decode code/agentic, ~1.5–2× chat.

## 9. Numbers to beat (baselines on this card/model)

- **84.9 t/s chat @128K** — patched Unsloth llama.cpp ROCm fork (IQ4_XS + q4_0 KV + DSpark + expert-offload, 15.2GB VRAM peak); 1459 t/s pp54K.
- 68.6 t/s — SmallTide engine, Windows, adaptive expert streaming (lossy ~96% top-1).
- Strata 0.2.20+ — runnable baseline for our design family (45–95 t/s on 125B/6B-active MoE, weaker HW).
