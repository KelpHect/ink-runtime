# RX 9070 XT / RDNA4 / gfx1201 — Hardware Reference for LLM Inference

Research report A01 for the ink-runtime project (Inkling-2-Mini, custom C/C++ inference runtime).
Target card: AMD Radeon RX 9070 XT 16 GB (Navi 48, RDNA4, `gfx1201`), on Windows 11 + i7-13700KF.
Compiled 2026-11-24 from ~20 primary sources; every non-obvious claim is cited inline.

---

## 1. Exact silicon specifications

### 1.1 Top-level

| Property | RX 9070 XT | Source |
|---|---|---|
| Die | Navi 48 XTX, monolithic, TSMC N4P (4 nm), 356.5 mm², 53.9 B transistors (~151 M tr/mm²) | [TPU deep dive](https://www.techpowerup.com/review/amd-radeon-rx-9070-series-technical-deep-dive/3.html), [C&C](https://chipsandcheese.com/p/amds-rdna4-architecture-video) |
| LLVM target | `gfx1201` (GFXIP 12.0); Navi 44 (9060 XT) is `gfx1200` | [ROCm GPU specs](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html) |
| Shader engines | 4 SEs × 8 dual-CU "WGPs" = 32 WGP = 64 CU | [C&C die analysis](https://chipsandcheese.com/p/amds-rdna4-architecture-video) |
| Stream processors | 4096 (64 SP/CU = 2 SIMD32) | [AMD product page](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html) |
| Ray accelerators | 64 (3rd gen, 8 box / 2 triangle tests/clk, BVH8) | [AMD](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html), [C&C RT](https://chipsandcheese.com/p/rdna-4s-raytracing-improvements) |
| AI accelerators | 128 (2nd gen WMMA units; 2 per CU) | [AMD QRG](https://www.amd.com/content/dam/amd/en/documents/corporate/events/radeon-9000-series-qrg-competitive.pdf) |
| TMUs / ROPs | 256 / 128 | [TPU](https://www.techpowerup.com/review/amd-radeon-rx-9070-series-technical-deep-dive/3.html) |
| Game / Boost clock | 2400 MHz / "up to" 2970 MHz | [AMD newsroom](https://newsroom.amd.com/news/amd-unveils-next-generation-amd-rdna-4-architectu.html) |
| TBP | 304 W reference; AIB cards up to ~340 W stock, +10% power-limit slider (max ~374 W on some AIBs); 2×8-pin | [AMD](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html), [ComputerBase OC/UV](https://www.computerbase.de/artikel/grafikkarten/radeon-rx-9070-xt-oc-uv-test.91714/) |
| Host interface | **PCIe 5.0 x16** (~64 GB/s/direction theoretical) | [TPU deep dive](https://www.techpowerup.com/review/amd-radeon-rx-9070-series-technical-deep-dive/3.html) |
| VRAM | 16 GB GDDR6, 256-bit, 20 Gbps → 640–644.6 GB/s | [AMD](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html), [TPU GPU DB](https://www.techpowerup.com/gpu-specs/radeon-rx-9070-xt.c4229) |
| Infinity Cache | 64 MB "3rd gen" — 16 Coherent-Station blocks × 4 MB, on-die | [C&C Hot Chips](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot) |
| Media | 2× VCN (dual media engine); Radiance 2 display engine; DP 2.1a UHBR13.5, HDMI 2.1b | [TPU deep dive](https://www.techpowerup.com/review/amd-radeon-rx-9070-series-technical-deep-dive/3.html), [C&C HC](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot) |

### 1.2 Per-CU / per-WGP microarchitecture

Per the [ROCm GPU specs table](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html), [TPU deep dive](https://www.techpowerup.com/review/amd-radeon-rx-9070-series-technical-deep-dive/3.html) and the [zolotukhin RDNA4 inference reference](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/):

- **CU internals**: 2 SIMD32 vector units + 1 scalar unit; "32 FMA ALUs and 32 FMA+INT ALUs" (the second SIMD carries the INT-capable pipes); 8 transcendental (SFU) units — transcendental ops (`V_EXP_F32`, `V_RCP_F32`, `V_RSQ_F32`, `V_SQRT_F32`, `V_LOG_F32`) issue at **¼ rate**, ⅛ on wave64 ([zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)).
- **Wave sizes**: both **wave32 and wave64** supported (LLVM target feature); wave32 is the hardware default — "RDNA4 GPUs run all shader programs in wave32 mode by default" ([ROCm WMMA docs](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html)). Wave64 instructions take 2 cycles per issue on the 32-lane SIMDs ([zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)).
- **Occupancy**: max 16 waves per SIMD (32 per CU) regardless of wave size; VGPR budget gates it: ≤96 VGPR/thread → 16 waves, 97–128 → 12, 129–192 → 8, 193–256 → 6 ([zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)).
- **Register file**: 192 KB VGPR per SIMD → 384 KB/CU → **768 KB per WGP** (ROCm table lists 768 KiB); up to 256 VGPRs per thread; SGPR file 32 KiB/WGP (8 KB scalar GPR per scheduler block per [TPU](https://www.techpowerup.com/review/amd-radeon-rx-9070-series-technical-deep-dive/3.html)). **Dynamic register allocation** is new in RDNA4: VGPRs allocated on demand in blocks of 16 or 32 (up to 8 blocks/thread), replacing RDNA3's static allocation ([C&C](https://chipsandcheese.com/p/amds-rdna4-architecture-video), [zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)).
- **LDS**: 128 KB per WGP (64 KB usable per CU); **64 banks × 4 B** = 256 B/clk per WGP, each SIMD-pair shares a 64 B bidirectional port, a single wave can pull ≤64 B/clk; ~2-cycle latency, broadcasts free, bank conflicts serialize ([ROCm programming guide](https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/develop/conceptual/introduction/hw_impl.html), [zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)).
- **L0 caches** (per CU): 32 KB vector (replaces RDNA3's mid-level L1 — the old L1 is now just a read/write-coalescing buffer), 16 KB scalar, 32 KB instruction; ~30-cycle hit latency, 64 B lines ([ROCm specs](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html), [C&C](https://chipsandcheese.com/p/amds-rdna4-architecture-video)).
- **L2**: 8 MB total, 2 MB per shader engine (2× capacity and 2× banks-per-slice vs RDNA3), 128 B lines, ~100-cycle hit ([C&C](https://chipsandcheese.com/p/amds-rdna4-architecture-video), [zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)).
- **Infinity Cache / MALL**: 64 MB as 16 CS blocks; Infinity Fabric runs DVFS **1.5–2.5 GHz** at 1024 B/clk → **~2.5 TB/s theoretical MALL bandwidth**; ~150-cycle / >50 ns-over-L2 hit latency; acts as victim/memory-side cache between L2 and DRAM ([C&C Hot Chips](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot), [zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)). Compression applies to data stored in L2 and Infinity Cache on RDNA4 ([C&C HC](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot)).
- **GDDR6 latency**: ~300+ cycles full miss ([zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)).
- **SDMA**: 2 SDMA engines, 6 user queues/engine (from a real `gfx1201` KFD topology dump mirrored in [tinygrad's mockgpu](https://github.com/tinygrad/tinygrad/blob/9ea7de02/test/mockgpu/amd/amdgpu.py)); also `num_gws 64`, `num_cp_queues 8`, `max_slots_scratch_cu 32` in the same dump. Copies are used for paging/page-table updates and user-space DMA ([kernel driver docs](https://github.com/torvalds/linux/blob/master/Documentation/gpu/amdgpu/driver-core.rst)). Not for bulk weight streaming at line rate — use shader reads or PCIe peer paths for that.

### 1.3 Compute rates (dense / sparse)

AMD's official peak table (at 2970 MHz boost) vs. derived per-clock rates:

| Path | AMD official | Derived | Source |
|---|---|---|---|
| FP32 vector | 48.7 TFLOPS | 128 FMA/clk/CU | [AMD](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html) |
| FP16 vector | 48.7 TFLOPS (same as FP32!) | — | [AMD](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html) |
| FP16 vector (TPU view) | 97.3 TFLOPS "2:1" | — | [TPU GPU DB](https://www.techpowerup.com/gpu-specs/radeon-rx-9070-xt.c4229) — conflicts with AMD; AMD's own spec says parity with FP32, and per-instruction tables show `V_FMA_F16` at 1/clk = F32 rate ([zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)). The 2× FP16 VALU path is `V_DOT2_F32_F16` (64 FMA/clk) — **use dot2/dot4/dot8 in dequant kernels, not packed FMA**. |
| FP16/BF16 matrix (WMMA) | **195 TFLOPS dense / 389 sparse** | 1024 FLOP/clk/CU | [AMD](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html) |
| FP8 E4M3/E5M2 matrix | **389 TFLOPS dense / 779 sparse** | 2048/clk/CU | [AMD](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html) |
| INT8 matrix | 389 TOPS dense / 779 sparse | 2048 ops/clk/CU | same |
| INT4 matrix | **779 TOPS dense / 1557 sparse** | 4096 ops/clk/CU | same — this is the "~1600 AI TOPS" marketing number ([TPU](https://www.techpowerup.com/review/sapphire-radeon-rx-9070-xt-nitro/)) |
| FP64 | 1.52 TFLOPS (1:32) | — | [TPU GPU DB](https://www.techpowerup.com/gpu-specs/radeon-rx-9070-xt.c4229) |
| Scalar FP32 | ~1 FMA/clk/CU | new scalar FP (`S_FMAAK_F32`, `S_FMAMK_F32`, `S_CVT_*`) | [C&C HC](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot) — 4-cycle latency vs 5 for vector |

**WMMA instruction shapes** ([ROCm builtins](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html), [zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)):

- All WMMA = **wave32 only** on RDNA4, 16×16 output tile per wave; FP16/BF16 K=16, FP8/BF8 K=32, INT8 K=16, **INT4 K=32** (deeper-K variant new in RDNA4); sparse `SWMMAC` variants use 2:4 (a.k.a. 4:2) structured sparsity to double throughput and halve A-operand bandwidth.
- New vs RDNA3: FP8 (E4M3) + BF8 (E5M2) inputs including **mixed fp8×bf8**; K-split fragment layout (no more inter-lane replication — lanes 0–15 hold K{0–3,8–11}, lanes 16–31 hold K{4–7,12–15}); accumulator = contiguous 8-row blocks (lanes 0–15 = rows 0–7); **hardware matrix-load-with-transpose builtins** (`rdna4-wmma-transpose-builtins`); `_gfx12` builtin suffix.
- FP8 conversions exist in ISA (`V_CVT` family with OPSEL controlling packing of fp8/bf8 pairs — [RDNA4 ISA guide §7.6.2](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content)).
- **No FP4/E2M1 (NVFP4, MXFP4) hardware path.** Explicitly: "INT4 support does not imply native FP4 or arbitrary mixed-format support" ([AMD-AGI cheatsheet](https://github.com/AMD-AGI/AgentKernelArena/blob/main/src/prompts/cheatsheet/RDNA4_architecture.md)). For Inkling's NVFP4 weights the realistic options are: (a) dequantize to FP16/BF16/FP8 and feed the 195/389 TF paths; (b) treat the 4-bit codes as integers through the 779-TOPS INT4 WMMA and fold the E2M1→scale expansion into the epilogue; or (c) scalar `V_DOT4/V_DOT8_I32_IU4` + `V_BFE_U32` unpack for decode-time DMMV where bandwidth, not MACs, dominates.
- INT dot-product VALU path (non-WMMA): `V_DOT4_I32_IU8` 1/clk = 128 ops/clk/SIMD; `V_DOT4/DOT8_I32_IU4` exists for the 4-bit path ([zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)).

---

## 2. RDNA4 ISA/microarchitecture changes relevant to inference

From the [RDNA4 ISA Reference](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content), [C&C launch analysis](https://chipsandcheese.com/p/amds-rdna4-architecture-video), [C&C OoO-memory investigation](https://chipsandcheese.com/p/rdna-4s-out-of-order-memory-accesses), [C&C Hot Chips](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot) and [zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/):

- **Out-of-order memory across waves.** RDNA≤3 had a *shared, strictly-ordered* memory return queue per WGP: one wave's DRAM misses blocked completion delivery to *other* waves (`s_waitcnt vmcnt` waited on foreign requests — verified experimentally on RDNA3 vs fixed on RDNA4). RDNA4 adds per-type out-of-order queues; a latency-sensitive wave (e.g., KV-cache/attention) is no longer serialized behind a streaming wave (weights). Directly relevant to interleaving prefetch + compute in our runtime.
- **Split wait counters.** `vmcnt` → `loadcnt` (global), `samplecnt` (texture), `bvhcnt` (RT); `lgkmcnt` → `kmcnt` (scalar loads) + `dscnt` (LDS/GDS). A wave can wait only on the class it needs, so the compiler can hoist independent LDS work ahead of global-load waits.
- **Split barriers.** `S_BARRIER_SIGNAL` / `S_BARRIER_WAIT` replace `S_BARRIER`: signal after producing shared data, do independent work, wait only when consuming. (Note: "Named Barriers" on the early AMD slide are *not* in shipping RDNA4 per [C&C](https://chipsandcheese.com/p/amds-rdna4-architecture-video).)
- **Scalar FP ALU ops** (inherited from RDNA3.5, new to dGPU): `S_FMAAK_F32`, `S_FMAMK_F32`, `S_CVT_F32_*` — good for uniform dequant scales, RoPE constants, 4-cycle latency.
- **Register block spill/fill**: save/restore up to 32 VGPRs in one op — cheap context switches for persistent-kernel designs.
- **No L1 data cache**: the mid-level L1 is gone; the 32 KB L0 vector cache + coalescing buffer goes straight to a fatter 2 MB/SE L2. Streaming inference kernels see less benefit from L1 anyway; the change mainly shifts where gather/scatter hurts.
- **Transparent compression** extends through L2 *and* Infinity Cache ([C&C HC](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot)) — one reason 64 MB IC + 256-bit bus punches above its weight; benefit is data-dependent (mostly graphics surfaces, don't count on it for weights).
- **No TMA/`cp.async` equivalent.** Loads are still load-to-register; the "async" story is the OoO queues + split counters, not a bulk-copy engine. Async-style staging must be built from ordinary global loads + `s_waitcnt` discipline.
- **VOPD dual-issue** exists (carried from RDNA3) but "dual-issuing capability hasn't seen improvement" ([C&C](https://chipsandcheese.com/p/amds-rdna4-architecture-video)).
- **WMMA operand transpose in hardware** (`V_WMMA` transpose-load builtins) — B-operand can be loaded transposed without a shader-side shuffle ([ROCm transpose builtins page](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html)).

---

## 3. Memory subsystem: spec vs. reality

- **Spec**: 20 Gbps GDDR6 × 256-bit = 640 GB/s (AMD), 644.6 GB/s by TPU's math (the top memory DPM step is 1258 MHz → 20.13 Gbps effective, see §5.3). Sources: [AMD](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html), [TPU](https://www.techpowerup.com/gpu-specs/radeon-rx-9070-xt.c4229), [r9700 findings](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/main/docs/r9700-oc-uv-findings.md).
- **Measured effective bandwidth in LLM decode**: ~490–505 GB/s implied from llama.cpp tok/s on 8B/14B models = **77–79% of peak** ([vachsark.com](https://blog.vachsark.com/blog/vulkan-beats-rocm/)). ZINC's per-op profiling on gfx1201 shows the spread: big GEMV rows hit **83–93%** of stream bandwidth (vocab head 93.2%, m=8192 attention 83.6%), MoE expert matvecs ~60%, small matmuls as low as 2.7% — **per-token aggregate decode lands ~28–31% of stream bandwidth because non-matmul ops and small matrices eat the time** ([zolotukhin tuning](https://zolotukhin.ai/zinc/docs/rdna4-tuning/)). On large DMMV, 67–93% utilization is typical across RDNA4 boards ([zolotukhin reference](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)).
- **Infinity Cache on streaming/decode**: victim-cache behavior — decode's weight stream (tens of GB/s of unique data) blows through it; it mainly shields KV-cache/activations and compressible surfaces. A 64 MB IC covers the KV of a few 4K-context sessions ([zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)). The IC adds >50 ns over an L2 hit ([C&C HC](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot)).
- **Memory clock & fabric coupling**: the memory-side Infinity Fabric DVFSes 1.5–2.5 GHz independent of shader clocks ([C&C HC](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot)). On RDNA3 C&C documented that the fabric downclocks under low outstanding-request counts, inflating measured IC/VRAM latency unless the card is in a high power state ([C&C "Latency Testing is Hard"](https://chipsandcheese.com/p/latency-testing-is-hard-rdna-3-power-saving)) — same power-management model carries over; expect fabric/mclk demotion between micro-batches. Confirmed live on gfx1201: "the SMU bounces MCLK through every level (96/456/772/875/1124/1258) during inference, including dropping to 96 MHz between micro-batches" ([r9700 findings](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/main/docs/r9700-oc-uv-findings.md)). RDNA4 also adds an *intermediate* GDDR6 power state for display duty ([C&C HC](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot)).
- **GDDR6 link error handling**: beyond the stable OC point, performance *regresses without artifacts* — the link does error detection + retry; this is the mechanism behind the der8auer/TPU "memory OC makes it slower" observation (§5).

---

## 4. PCIe & host path

- PCIe 5.0 x16 ≈ 63 GB/s/direction theoretical ([TPU](https://www.techpowerup.com/review/amd-radeon-rx-9070-series-technical-deep-dive/3.html)); ~57–60 GB/s realistic. On i7-13700KF the card gets CPU-attached Gen5 x16.
- **Measured importance for MoE offload**: a llamaperf report on Qwen3.8-125B-Flash-Next with experts in host RAM saw ~3700 t/s prefill on PCIe5 x16 ("nearly saturated") vs ~2000 t/s when the link fell back to Gen4; another found decode degrades 18–19%/100k ctx ([llamaperf RX 9070 XT](https://llamaperf.com/gpu/rx-9070-xt-16gb)). **For Inkling (19.5 GB > 16 GB VRAM), keeping the link at Gen5 x16 and saturating it is a first-class requirement.**
- PCIe ASPM power saving measurably hurts decode: `pcie_aspm.policy=performance` gave **+10.8% decode tok/s on a dense 27B** model (~0% on MoE, which hides latency better) on gfx1201 ([llama.cpp discussion #21043 via r9700 findings](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/main/docs/r9700-oc-uv-findings.md)). On Windows the equivalent is disabling PCIe link-state power management (power plan → "Link State Power Management: Off") and ASPM in BIOS.
- Resizable BAR: enable it. For a Vulkan MoE-offload engine, ReBAR off cut throughput from 0.25→0.11 tok/s on a GLM-5.2 run ([llamaperf](https://llamaperf.com/gpu/rx-9070-xt-16gb)); the driver auto-enables SAM on supported systems ([clip.dor.gg guide](https://clip.dor.gg/en/blog/resizable-bar-sam-setup)). Whole-VRAM BAR mapping also matters for mmap-style weight streaming.

---

## 5. "Locked" and underutilized performance — the 9070 XT unlock file

This is the section analogous to the RTX 3090 memory-junction story. The short version: **the card ships voltage/power-limited below its real boost envelope, memory overclocking is counterproductive because firmware punishes it, and on Linux several silent bandwidth/power heuristics shave 5–25%.**

### 5.1 Undervolt = free clocks (the big one)

The V/F curve is set conservatively; the card cannot hold advertised boost at stock in sustained load:

- Der8auer (PowerColor Red Devil): stock 2.90 GHz → **3.36 GHz at −170 mV + 110% power limit**, +10% Cyberpunk FPS, beating an RTX 5080 ([Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/undervolted-rx-9070-xt-beats-rtx-5080-rx-9070-and-9070-xt-models-with-heavy-coolers-have-massive-oc-headroom)). −200 mV gave +300 MHz but unstable ([overclocking.com](https://overclocking.com/rx-9070-xt-loverclocking-est-un-peu-petax/)).
- TPU's OC methodology, validated across many AIB cards: "**you MUST lower the GPU voltage, or you won't see meaningful performance gains**"; Max Frequency Offset +400 MHz headroom; typical results ~2.8–3.2 GHz sustained ([TPU ASUS 9070 OC](https://www.techpowerup.com/review/asus-radeon-rx-9070-tuf-oc/43.html)).
- Quantified on gfx1201 *for inference* (R9700, same silicon): **−120 mV → +8.9% pp32768 prefill, sclk 3066→3357 MHz at the same ~295 W**; diminishing after −120 mV (−140 = +1.2%, −150 = +0.5% more); −150 mV + 330 W crashed with a gfxhub page fault under deep-prompt load — "short benchmarks do not prove stability" ([r9700 findings](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/main/docs/r9700-oc-uv-findings.md)). Caveat: undervolt only pays when the card is *power-limited*; when it was thermally limited the same offset did nothing, and on one driver/workload combo *any* offset cost ~4.5% (SMU treated it as a stability hint) — re-validate on your exact stack.
- Mechanism: AMD's voltage offset slides the whole V/F curve; cards "run at their limits" stock, so less voltage ⇒ less power ⇒ more boost inside the same envelope ([overclocking.com](https://overclocking.com/rx-9070-xt-loverclocking-est-un-peu-petax/), [ComputerBase](https://www.computerbase.de/artikel/grafikkarten/radeon-rx-9070-xt-oc-uv-test.91714/)).
- **Windows tooling**: Adrenalin exposes negative Voltage Offset, Max Frequency Offset, memory clock, ±power limit (no positive voltage offset). MoreClockTool (MCT ≥1.1.11v2/2.x) supports RDNA4 — session PPT cap up to **340 W** on 9070 XT, full fan-curve/ZeroRPM control, and SoftTuningTable profiles; MCT ships three ready-made 9070 XT profiles ([Igor's Lab MCT](https://www.igorslab.de/en/moreclocktool-new-ppt-limit-radeon-graphics-cards/), [Igor's Lab downloads](https://www.igorslab.de/en/download-area-new-version-of-morepowertool-mpt-and-final-release-of-redbioseditor-rbe/)).
- **What does NOT exist on RDNA4**: MorePowerTool/SoftPowerPlayTables — AMD locked SPPT writes starting with RDNA3; the MPT author confirmed no RDNA3/4 support ever. The STT on RDNA4 is "guarded" (safer but narrower). Actual power/current-limit removal requires the Linux route: R.B.R.T. **RebelsTool + Adrenalift** ([Igor's Lab RBR team page](https://www.igorslab.de/en/red-bios-rebellion-team/4/), [cryptolexicon MCT guide](https://cryptolexicon.org/en/blog/moreclocktool-mct-amd-gpu-tuning-for-mining/)).

### 5.2 Memory overclocking: the trap

- Stock = 2500 MHz marketed (20 Gbps; DPM top shows 1258 MHz = 20.13 Gbps). Practical silicon headroom ~2700–2800 MHz (21.6–22.4 Gbps → ~704–717 GB/s); one card locked up at 23 Gbps / froze at 24 ([FPSReview XFX](https://www.thefpsreview.com/2025/03/05/xfx-quicksilver-amd-radeon-rx-9070-xt-magnetic-air-edition-video-card-launch-review/3/)).
- **But raising it makes things slower**: "overclocking the memory clock actually results in slower performance compared to stock" — der8auer ([Tom's HW](https://www.tomshardware.com/pc-components/gpus/undervolted-rx-9070-xt-beats-rtx-5080-rx-9070-and-9070-xt-models-with-heavy-coolers-have-massive-oc-headroom)). TPU explains the mechanism: past the error-free point the link's error correction kicks in — "performance will be reduced due to the error correction kicking in," no artifacts ([TPU](https://www.techpowerup.com/review/asus-radeon-rx-9070-tuf-oc/43.html)).
- **Worse on Linux/amdgpu**: writing any `OD_MCLK` above the firmware DPM top (1258 MHz) is silently rejected *and pins mclk ~30% below stock* (875–1124 MHz, never 1258) → measured **−6.6% decode**; it's an any-above-top trigger, not proportional ([r9700 findings](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/main/docs/r9700-oc-uv-findings.md)). Verdict there: "no upside, a measurable downside. Leave OD_MCLK alone."
- **Memory vendor lottery**: later 9070 XTs ship Samsung GDDR6 instead of SK Hynix — Samsung runs looser timings, ~**2.6% slower** in 3DMark Speed Way across a 22-card survey; Samsung clocks *higher* but Hynix performs better at equal clocks ([PC Gamer/51972](https://www.pcgamer.com/hardware/graphics-cards/some-rx-9070-xts-are-reportedly-slower-than-others-thanks-to-samsung-gddr6-memory-chips/), [HWCooling](https://www.hwcooling.net/en/overclocking-rx-9070-xt-memory-samsung-or-hynix-which-is-better/), [overclocking.com](https://en.overclocking.com/rx-9070-xt-with-samsung-vram-are-slower/)). Identify your chips via GPU-Z teardown data; for a bandwidth-bound runtime this is a real (if small) term.

### 5.3 Clock/power-management quirks that silently cost throughput

- **SMU14 doesn't boost GFXCLK for pure compute**: ROCm issue — gfx1201 stuck at **41 MHz sclk** during llama.cpp inference (≈3× slowdown, ~50 W draw); only `power_dpm_force_performance_level=profile_peak` held 2400 MHz, at the cost of clock gating ([ROCm/ROCm#6289](https://github.com/ROCm/ROCm/issues/6289)). Related SMU-interface mismatch on kernel 6.17 pins sustained clocks ~2200 MHz on R9700 ([zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)).
- **But don't blindly force peak**: `RADV_PROFILE_PSTATE=peak` / `profile_peak` causes a **−23% regression on memory-bound RDNA4 work** (power burned on core clocks throttles the memory path), and `high` locks a fixed nominal DPM state that costs ~4.3% decode vs `auto` ([zolotukhin tuning](https://zolotukhin.ai/zinc/docs/rdna4-tuning/), [r9700 findings](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/main/docs/r9700-oc-uv-findings.md)). On a healthy system **`auto` beats `high`/`peak` for decode** — it still reaches 3.0+ GHz while letting the SMU keep the fabric fed. On Windows: don't enable Adrenalin's "force max clocks"-type hacks for decode; leave default DPM, do the undervolt.
- **mclk bounces 96↔1258 MHz between micro-batches** under inference load ([r9700 findings](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/main/docs/r9700-oc-uv-findings.md)); keeping the GPU saturated (large fused kernels, pre-recorded command buffers) is itself a clock-stability optimization.
- **GECC/RAS**: RDNA4 enables GPU ECC by default *on the R9700 workstation card* — costs ~10% memory bandwidth; `amdgpu.ras_enable=0` gave **+9% (101→110 tok/s)** on Qwen3.6-35B-A3B, ~+1 tok/s in the toolkit's tests ([zolotukhin tuning](https://zolotukhin.ai/zinc/docs/rdna4-tuning/), [r9700 findings](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/main/docs/r9700-oc-uv-findings.md)). Whether consumer 9070 XT ships RAS enabled is unverified — check `rocm-smi --showrasinfo` equivalent / Windows equivalent; if on, disabling is free bandwidth.
- **Runtime PM / resume races**: `amdgpu.runpm=0` (or a udev rule pinning `power/control=on`) fixes silent OD-upload failures and `rocminfo` non-detection ([r9700 findings](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/main/docs/r9700-oc-uv-findings.md)). Windows analogue: keep the card out of D3 — PCIe link-state PM off, monitor/driver path stays hot.
- **Toolchain cliffs (Linux/Vulkan)**: shaderc v2026.x emits `NonWritable` decorations RADV/ACO mishandles → **5× slowdown** (110→19–25 tok/s) vs distro shaderc 2023.8; Mesa 25.2.8 has a ~14% RADV regression on the Q4_K DMMV path vs 25.0.7 ([zolotukhin tuning](https://zolotukhin.ai/zinc/docs/rdna4-tuning/)). If we ever go through Vulkan/SPIR-V, pin the toolchain.
- **Thermal reality**: under sustained compute the card hits ~104 °C junction *before* the 300 W power limit; 330 W cap adds <2% sustained ([r9700 findings](https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/main/docs/r9700-oc-uv-findings.md)). Junction, not board power, is the wall — cooler quality and undervolt are the levers.
- Efficiency note: the RX 9070 (56 CU, 220 W) at 154 W delivers ~70% of the 9070 XT's perf at 44% of the power — the XT is pushed well past the efficiency knee ([C&C](https://chipsandcheese.com/p/amds-rdna4-architecture-video)); a 230–260 W capped profile loses only a few % of throughput (one llamaperf report: 231 W cap = 3.6% prefill penalty, zero decode loss — [llamaperf](https://llamaperf.com/gpu/rx-9070-xt-16gb)).

### 5.4 Wave64 vs wave32 selection (software-side unlock)

- Decode DMMV (GEMV): **wave64 measured faster** — the driver's default is optimal; wave32 showed no gain ([zolotukhin tuning](https://zolotukhin.ai/zinc/docs/rdna4-tuning/) — "Wave32 for DMMV: No improvement; driver's default wave64 is optimal"; same in the [reference doc](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)).
- WMMA kernels: **wave32 required** (all `__builtin_amdgcn_wmma_*_w32_gfx12` need `wavefrontsize32`).
- Per-pipeline wave size on Vulkan: `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`; on HIP: `__launch_bounds__`/`-mwavefrontsize64` per TU. Expect to ship *both* modes in the same process — wave64 DMMV/dequant kernels for decode, wave32 WMMA kernels for prefill/batch.
- Rows-per-workgroup tuning matters: `rm_kq>2` caused −75% on wave64 DMMV ([zolotukhin tuning](https://zolotukhin.ai/zinc/docs/rdna4-tuning/)).

---

## 6. Measured LLM performance on this card

### 6.1 Backend landscape (as of Nov 2026)

| Backend | Status on gfx1201 |
|---|---|
| llama.cpp HIP/ROCm | Native gfx1201 in official builds/containers; no `HSA_OVERRIDE_GFX_VERSION` needed ([lemmy report via llamaperf](https://llamaperf.com/gpu/rx-9070-xt-16gb)). HIP SDK for Windows officially supports RX 9070 XT ([ROCm Windows reqs](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html)). |
| llama.cpp Vulkan | Works on RADV and AMDVLK/Windows driver; historically the faster decode path on RDNA4, though recent ROCm wins K-quants with flash attention on — it's workload-dependent ([kr4ckhe4d bench](https://github.com/kr4ckhe4d/local-llm-benchmarks), [Phoronix](https://www.phoronix.com/review/llama-cpp-windows-linux/5)). **Known bug**: a 2026-10 report shows Vulkan decode collapsing 4.7–6.7× vs HIP on models with hidden ≥4096 (Qwen3-8B 21 vs 98.8 t/s) — verify current status ([llamaperf](https://llamaperf.com/gpu/rx-9070-xt-16gb)). |
| vLLM | Runs, but **FP8 models silently fell back to FP32 dequant** on gfx1201, bypassing all 128 AI accelerators → 48 t/s vs llama.cpp-Vulkan's 62 t/s on Qwen3.5-9B-Q6_K (vllm-project/vllm#28649; [digtvbg](https://digtvbg.com/blog/llama-server-vulkan-rdna4-vllm-rocm-benchmark/)). Recheck before relying on FP8 paths. |
| LM Studio | 9070 XT supported since llama.cpp runtime 1.52.3; ships both Vulkan and ROCm backends ([LM Studio tracker #574](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/574)). |

### 6.2 Reported numbers (single RX 9070 XT 16 GB, no spec decode unless noted)

- **Inkling-2-Mini itself**: **84.9 tok/s chat, 1459 tok/s on a 54k-token prompt at 128k ctx** — patched Unsloth llama.cpp fork (ROCm), GSQ-RCO IQ4_XS weights, q4_0 KV, DSpark spec-dec + n-gram drafting, experts partly in system RAM, peak VRAM 15.2 GB; vs upstream llama.cpp ~3× chat / ~10× prompt ([llamaperf](https://llamaperf.com/gpu/rx-9070-xt-16gb)). Also 68.6 tok/s via SmallTide engine on Windows 11 with adaptive expert streaming (lossy, ~96% top-1 agreement) ([llamaperf](https://llamaperf.com/gpu/rx-9070-xt-16gb)).
- Qwen3-8B Q4_K_M: **98.8 t/s decode / 4245 t/s pp512** (Windows HIP, llama.cpp b10290) ([llamaperf](https://llamaperf.com/gpu/rx-9070-xt-16gb)); ~100 t/s tg on Vulkan per [vachsark](https://blog.vachsark.com/blog/vulkan-beats-rocm/) (implied 490–505 GB/s).
- Qwen3-14B Q4_K_M: 54.3 t/s HIP ([llamaperf](https://llamaperf.com/gpu/rx-9070-xt-16gb)).
- gpt-oss-20B MXFP4: **130–133 t/s Vulkan vs 109–112 t/s ROCm** on Windows LM Studio ([issue #574](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/574)); 87 t/s on the smaller 9060 XT ([ComputerBase forum](https://www.computerbase.de/forum/threads/rx-9060-xt-16-gb-lm-studio-unter-windows-5-modelle-gemessen-vulkan-schlaegt-rocm-deutlich.2281936/)).
- Qwen3.6-27B IQ3_M + MTP draft: **46.06 t/s gen / 549 t/s prompt, 62.7% draft acceptance**, 14.46 GB VRAM — vs 25.9 t/s without MTP ([llamaperf/lemmy](https://llamaperf.com/gpu/rx-9070-xt-16gb)).
- Qwen3.6-35B-A3B Q4_K_M (`-ncmoe 40 -ub 512`): 706 t/s pp on ROCm ([kr4ckhe4d](https://github.com/kr4ckhe4d/local-llm-benchmarks)); ZINC engine reports 117 t/s decode at 31% BW utilization on the 32 GB sibling ([zolotukhin tuning](https://zolotukhin.ai/zinc/docs/rdna4-tuning/)).
- MoE with expert streaming: Qwen3.8-125B-A6B ~40–88 t/s depending on engine/quant with experts in RAM+NVMe ([llamaperf](https://llamaperf.com/gpu/rx-9070-xt-16gb)).
- **What limits it**: decode is bandwidth-bound — per-token traffic ≈ active-parameter bytes. Ceiling ≈ 640 GB/s ÷ bytes/token; at ~3.5 GB/token (5.4B×~5bpw + KV) that's ~180 t/s theoretical, ~140–160 realistic at 77–93% efficiency. vLLM-level batching doesn't change per-token BW; only smaller effective weight traffic (better quant, resident-hot experts, MTP) does.

### 6.3 Implications for ink-runtime (26.4B MoE, ~19.5 GB weights, 1M ctx)

1. **19.5 GB > 16 GB** → expert/weight streaming over PCIe5 x16 is mandatory; the ~60 GB/s link and Gen5 must be preserved (no bifurcation; ASPM off; ReBAR on). Hot experts in VRAM, cold streamed; llamaperf data shows this is proven at ~3700 t/s prefill scale.
2. **MXFP8 maps natively** to FP8 WMMA (16×16×32, 389 TF dense) — E4M3 payloads go straight to matrix cores; UE8M0 block scales handled in epilogue. **NVFP4 has no native path** — dequant to FP8/BF16 or drive the INT4 WMMA (779 TOPS) with E2M1→int reinterpretation + scales.
3. Decode kernel shape: wave64 DMMV with `V_DOT*` + `V_BFE` unpack; keep LDS ≤16 KB/WG for ≥4 resident WGs/CU; VGPR ≤96 for full occupancy; fuse norm+mul+activation (each fused pass saved ~500 µs/token in ZINC's data); pre-record the decode graph — replaying 1500 dispatches costs ~54 µs total vs ~33 µs per submit ([zolotukhin](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)).
4. Sliding-window (1024) + conv attention on most layers ⇒ KV working set is small; the 64 MB IC + 8 MB L2 will hold a lot of it — prioritize keeping KV hot in L2/IC over exotic attention kernels; only 4 global layers have big-KV costs.
5. MTP draft head: llama.cpp's `draft-mtp` measured ~1.8× (25.9→46 t/s) with 62.7% acceptance — the single biggest free multiplier; the verify pass is a small-batch GEMM that can use WMMA.
6. Tune the card: **undervolt ~−100 to −120 mV (+8–9% when power-limited), +10% power limit if junction allows, leave memory clock at stock, perf level auto, PCIe ASPM off, ReBAR on, verify RAS/ECC state, keep card awake.** Expect ~+10–15% net over naive defaults on sustained inference.

---

## 7. Source list

1. AMD RX 9070 XT product page — https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html
2. AMD RDNA4 launch newsroom — https://newsroom.amd.com/news/amd-unveils-next-generation-amd-rdna-4-architectu.html
3. AMD RX 9000 QRG PDF — https://www.amd.com/content/dam/amd/en/documents/corporate/events/radeon-9000-series-qrg-competitive.pdf
4. ROCm GPU specifications — https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
5. AMD RDNA4 ISA Reference Guide — https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content
6. ROCm optimization guide: RDNA4 dense WMMA builtins — https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html (+ sparse + transpose siblings)
7. ROCm programming guide hw_impl (LDS org) — https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/develop/conceptual/introduction/hw_impl.html
8. zolotukhin.ai AMD RDNA3/RDNA4 GPU Reference for Inference — https://zolotukhin.ai/zinc/docs/amd-gpu-reference/
9. zolotukhin.ai RDNA4 Tuning Guide — https://zolotukhin.ai/zinc/docs/rdna4-tuning/
10. Chips & Cheese: AMD's RDNA4 Architecture — https://chipsandcheese.com/p/amds-rdna4-architecture-video
11. Chips & Cheese: RDNA4 at Hot Chips 2025 — https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot
12. Chips & Cheese: RDNA4 Out-of-Order Memory — https://chipsandcheese.com/p/rdna-4s-out-of-order-memory-accesses
13. Chips & Cheese: RDNA3 power-saving latency (mechanism carries to RDNA4) — https://chipsandcheese.com/p/latency-testing-is-hard-rdna-3-power-saving
14. TechPowerUp: RX 9070 series technical deep dive — https://www.techpowerup.com/review/amd-radeon-rx-9070-series-technical-deep-dive/3.html ; OC methodology — https://www.techpowerup.com/review/asus-radeon-rx-9070-tuf-oc/43.html ; GPU DB — https://www.techpowerup.com/gpu-specs/radeon-rx-9070-xt.c4229
15. Tom's Hardware: undervolted 9070 XT vs RTX 5080 — https://www.tomshardware.com/pc-components/gpus/undervolted-rx-9070-xt-beats-rtx-5080-rx-9070-and-9070-xt-models-with-heavy-coolers-have-massive-oc-headroom ; overclocking.com der8auer — https://overclocking.com/rx-9070-xt-loverclocking-est-un-peu-petax/
16. Igor's Lab: MoreClockTool for RDNA4 (PPT 340 W, STT guarded, MPT dead, RebelsTool+Adrenalift) — https://www.igorslab.de/en/moreclocktool-new-ppt-limit-radeon-graphics-cards/ , https://www.igorslab.de/en/red-bios-rebellion-team/4/ , https://www.igorslab.de/en/download-area-new-version-of-morepowertool-mpt-and-final-release-of-redbioseditor-rbe/
17. daimonionnn amd-r9700-vllm-and-tuning-toolkit: r9700-oc-uv-findings.md — https://github.com/daimonionnn/amd-r9700-vllm-and-tuning-toolkit/blob/main/docs/r9700-oc-uv-findings.md (and llama.cpp discussion #21043 it summarizes: https://github.com/ggml-org/llama.cpp/discussions/21043)
18. GDDR6 vendor lottery: PCGamer — https://www.pcgamer.com/hardware/graphics-cards/some-rx-9070-xts-are-reportedly-slower-than-others-thanks-to-samsung-gddr6-memory-chips/ ; HWCooling — https://www.hwcooling.net/en/overclocking-rx-9070-xt-memory-samsung-or-hynix-which-is-better/
19. LLM numbers: llamaperf 9070 XT — https://llamaperf.com/gpu/rx-9070-xt-16gb ; digtvbg vLLM-vs-Vulkan — https://digtvbg.com/blog/llama-server-vulkan-rdna4-vllm-rocm-benchmark/ ; vachsark Vulkan-beats-ROCm — https://blog.vachsark.com/blog/vulkan-beats-rocm/ ; kr4ckhe4d local-llm-benchmarks — https://github.com/kr4ckhe4d/local-llm-benchmarks ; LM Studio #574 — https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/574 ; Phoronix — https://www.phoronix.com/review/llama-cpp-windows-linux/5 ; openbenchmarking — https://openbenchmarking.org/result/2509078-NE-ROCMVSVUL92
20. ROCm Windows support — https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html ; ROCm issue #6289 (41 MHz sclk bug) — https://github.com/ROCm/ROCm/issues/6289 ; tinygrad gfx1201 KFD dump — https://github.com/tinygrad/tinygrad/blob/9ea7de02/test/mockgpu/amd/amdgpu.py ; AMD-AGI RDNA4 cheatsheet — https://github.com/AMD-AGI/AgentKernelArena/blob/main/src/prompts/cheatsheet/RDNA4_architecture.md ; FPSReview memory OC — https://www.thefpsreview.com/2025/03/05/xfx-quicksilver-amd-radeon-rx-9070-xt-magnetic-air-edition-video-card-launch-review/3/ ; ROCm #5750 / CachyOS #888 (compute-clock bugs) — https://github.com/ROCm/ROCm/issues/5750 , https://github.com/CachyOS/linux-cachyos/issues/888

*Unreachable during research: techspot.com news article on der8auer (403) — the same findings were captured via Tom's Hardware and overclocking.com.*
