# RUNTIME_PLAN — ink-runtime (C/C++ inference engine for Inkling-2-Mini)

Status: research complete — all 14 reports in `research/` (see `research/README.md`).
Ceilings/budgets: `docs/THEORY.md`. Model facts: `research/00-model-facts.md`.

## Decision summary

1. **Own thin C++ runtime**, not a llama.cpp fork. Fixed-shape static graph (no dynamic DAG): the model is one file's worth of forward pass (`inkling2_loop`), ~4 recorded shapes (prefill-buckets, decode, draft, verify). Structure cribbed from ggml: `buft` (placement policy) / `buffer` (slab) / `backend` (queue) / `sched` (split + async copies + timeline events). Reference impls to study: `vendor/Strata` (already runs this model with our exact 4-tier design), GGUFModel GPTQModel Triton kernels, vLLM `ROCM_RDNA_ATTN`.
2. **Vulkan compute primary** on Windows (coopmat fp16/bf16/int8 confirmed on this Adrenalin build; Slang shaders; wave64 GEMV / wave32 WMMA). **HIP backend #2** for native FP8 WMMA (MXFP8 layers) + profiling on Linux. Backend vtable from day 1.
3. **3-tier expert store**: VRAM slot-pool (~9–11GB, ~75% of experts) / RAM full mirror (15.7GB) / NVMe expert-pack file (cold + fast-start). Decode = router-lookahead prefetch + CPU-exec-on-late-miss; prefill = stream all.
4. **KV**: hybrid pools (SWA ring 160MiB + paged global), fp8-E4M3 storage / bf16 compute, 64-token pages with min/max metadata → Quest-style selection past ~256K; radix-hash NVMe prefix store.
5. **MTP built in**: DFlare 8-token block draft + linear verify; mid-loop hidden taps at apps {5,23,41,48}.

## Toolchain gaps to close first

- No C/C++ toolchain in PATH: install **VS Build Tools (cl.exe) + CMake + Vulkan SDK**; HIP SDK 7.2 later for backend #2. Keep all dev artifacts in `C:\dev-overflow\` per machine policy.
- Runtime probes before any ceiling claim: `LnkSta` width/speed, ReBAR state, `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` (FP8 enum?), transfer-queue families, `VK_EXT_external_memory_host`, RAS/ECC, memory vendor (Hynix vs Samsung).

## Phases

### P0 — ground truth (days) — **DONE 2026-11-24**
- C++ toolchain (MSVC 2022 + CMake 4.4.4 + Vulkan SDK 1.4.363 — `docs/toolchain-status.md`); safetensors mmap parser + config schema in `src/io/`; fp32 CPU reference (`src/model/`) runs the full 48-visit pipeline.
- Hardware probes done (`tools/vkprobe.c`, `vkbench.c`): FP8-E4M3 coopmat on Adrenalin verified, 18× 16×16×16 configs; SDMA copy = 32.5 GB/s (DDR4-bound); ReBAR + Gen5 x16 confirmed.
- **Golden gate GREEN**: `tools/dump_golden.py` (HF ref, custom offload hooks — accelerate's own offload segfaults), `tools/compare_golden.py`, `tools/dump_visits.py` + `compare_visits.py` for per-visit bisection. All 48 visits cos ≥ 0.998, all 8 argmax match (bf16-vs-fp32 floor). Resolved bit-level: dynamic-conv taps lag=backward-distance (correct), **static conv1d taps were REVERSED — fixed** (w[..,3]=current), w13 interleave = pairs on disk (correct), sink-column + router renorm + message bank (2-src: last loop-boundary block + prev final) all verified correct.

### P1 — vertical slice decode — **DONE 2026-11-24**
- `src/vk/` plumbing (Context: compute fam1 + SDMA fam2 picks, coopmat/f8/bf16 feature bits, timeline sems; slab Allocator; CmdPool/Recorder; pipeline+descset) — `test_vk_smoke` PASS on device.
- `shaders/`: 24 correctness-first GLSL-460 kernels + `bindings.h` contract (all glslc-clean).
- `src/io/tokenizer.*` byte-level BPE — `encode_chat_user` reproduces the golden 25 ids exactly.
- `src/gpu/vk_backend.cpp`: `Backend` seam (`matvec` + batched `expert_ffn` = 3 dispatches/expert); `--backend vk`.
- **Two hard bugs found**: (1) descriptor ranges are *buffer-relative* — binding at the slab offset shifted every descriptor past the buffer end → OOB → real Adrenalin TDR/DEVICE_LOST (only offset-0 buffers worked, so early ops passed then died); (2) chunked staging recorded all copies then reused the buffer — smeared the last chunk everywhere (whole-tensor staging now).
- **Architecture pivot**: expert tensors live in persistent HOST_VISIBLE RAM buffers — GPU streams only the 8/128 selected rows over PCIe (~34MB/layer vs 340MB upload); host mirror ~17GB one-time, zero churn. This IS the planned tiered design — RAM is the expert tier, no copy needed.
- Gate: **PASS on GPU path** — all 8 argmax match HF, worst cos 0.9857 (bf16 floor). Perf: **0.62 s/tok** @ ~940 submits/token (vs ~20 s/tok CPU-fallback).

### P2 — speed: bandwidth-bound decode path (weeks 3–6)
- Wave64 dequant-fused NVFP4 GEMV (v_dot4 unpack, E4M3×fp32 scale epilogue); WMMA prefill GEMMs (wave32); fused norm+act; paged KV fp8 storage path.
- **Device-resident forward** (P2a, in flight): all activations/caches on device, one recorded submission per token — kills the ~940 submits×0.3ms + ~180ms descriptor-write overhead that dominates the 0.62 s/tok.
- Expert streaming already via host-visible buffers; remaining: router-lookahead prefetch (L+1 gate on h_L) if PCIe reads become the bound, loop-pass pinning, early-preemption.
- Prerecorded command buffers per shape; logits→sample small D2H only.
- Bench harness: tok/s, per-layer timing, hit rates; A/B vs llama.cpp fork baseline (84.9 t/s @128K is the bar).

### P2 — speed: bandwidth-bound decode path (weeks 3–6)
- Wave64 dequant-fused NVFP4 GEMV (v_dot4 unpack, E4M3×fp32 scale epilogue); WMMA prefill GEMMs (wave32); fused norm+act; paged KV fp8 storage path.
- **Expert slot-pool cache + router-lookahead prefetch** (L+1 gate on h_L) + pinned-ring H2D on transfer queue; loop-pass pinning; early-preemption.
- Prerecorded command buffers per shape; one-deep CPU/GPU overlap pipeline (SGLang pattern); logits→sample small D2H only.
- Bench harness: tok/s, per-layer timing, hit rates; A/B vs llama.cpp fork baseline (84.9 t/s @128K is the bar).

### P3 — context + MTP (weeks 6–9)
- Tiered KV (GPU hot + RAM fp8 + NVMe radix-prefix); Quest page selection >256K; sparse-prefill path for >256K fresh prompts (MInference-style landmarks).
- MTP: draft head resident, hidden-tap ring, 8-token block draft, linear verify @9 pos, acceptance metrics; expert-union cap; n-gram proposer as scaffold.
- Quality dials: fp8-KV needle eval ≥128K; k1/k2 expert-count eval.

### P4 — second backend + distribution (weeks 9+)
- HIP backend (native FP8 WMMA for MXFP8 layers, ck_tile FMHA port, rocprof); Linux dev build (TheRock wheels).
- Vision encoder path (self_attn.* tower — multimodal parity), chat template, minimal server.

## Risks / open questions (ranked)

1. **PCIe link negotiated below Gen5 x16** — halves the streaming ceiling; probe first.
2. **Vulkan FP8-coopmat not enumerated on Windows prop driver** → MXFP8 layers run fp8→bf16 convert (still fine) or HIP backend.
3. ~~Per-pass vs shared loop KV~~ **resolved**: per-pass, 48 cache sets (07 §3) — 1M-ctx global KV = 32GiB bf16 / 16GiB fp8.
4. **Windows Vulkan driver perf states** (transient 5–7× collapse documented) — runtime self-audit: effective-BW monitor + mitigation hints.
5. **Kernel launch overhead** — prerecorded shapes mandatory; validate dispatch cost <5% GPU time in P2.
6. **attn_res correctness** — the residual-stream replacement is the arch's core novelty (fp32 softmax over ≤5 block snapshots per sublayer read); it's cheap math but *every layer boundary is an off-by-one trap* (push at i%6==0 with the layer INPUT; attn_queries indexed i−1). Golden per-layer hidden dumps from HF are the gate.
7. **GGUF-vs-native-layout quality**: native NVFP4 kept end-to-end (no requant) — avoids the llama.cpp #32704 scale-plumbing gibberish class of bug; greedy-vs-HF test gates every change.

## File map

- `research/00` model facts · `01` hw deep-dive · `02` 3090-unlock+AMD analogs · `03` ROCm · `04` GPU APIs · `05` tiered runtimes · `06` NVFP4 formats · `08` MTP/draft · `09` KV · `10` NVMe I/O · `11` MoE streaming · `12` measured bandwidth · `13` ecosystem · `14` C++ arch patterns · `07` internals (pending)
- `docs/THEORY.md` ceilings · `docs/RUNTIME_PLAN.md` this file
- `vendor/Strata/` reference engine (Apache — check license before cribbing code)

## P2a/b done (2026-xx): device-resident forward

`--backend gpu` (src/gpu/gpu_graph.*): whole 48-visit forward is one static
descriptor stream + prerecorded cb; 1 submit/token (~0.3 s/tok wall on
Adrenalin, ~0.27s GPU-wait/token). Expert selection consumed on-device via
`*_slot` shader variants (router idx read in-shader). GATE PASS 8/8 argmax,
worst cos 0.9857.

Bugs found gating it: compare tools ignored argv (all early PASSes were
against stale dumps — comparator must be invoked with real paths); gpu xo[]
byte-vs-element offset => full-layer k/v/wproj OOB; block_[0] unseeded at
i==0. INK_DBG=1 now dumps per-stage intermediates for visit5 on both CPU and
GPU backends (16-row debug region appended to readback).
