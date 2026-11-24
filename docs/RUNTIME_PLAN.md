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

### P0 — ground truth (days)
- C++ toolchain; safetensors mmap parser (~200 LOC); config/schema dump. Model questions resolved by `research/07` (loop = block-sequential passes 6..17 ×3; **KV is per-pass** → 48 cache sets; router renormalizes sigmoid scores over top-8 then ×8 → k1/k2 knob valid; MTP = anchor+7-mask block draft with GRU markov chain, `rnn_proj` input = [state|markov_w1(prev)|hidden|f_rnn]=5120; no RoPE — content-conditioned rel-bias d_rel=16 extent 1024; attn_res = learned softmax reads over block-boundary snapshots replacing the residual stream). Remaining to verify in P0 only at the bit level: dynamic-conv tap semantics, sink-column trick (subtract-t variant), w13 gate/up interleave.
- Hardware probes above; undervolt profile (−100mV start) + ASPM off + ReBAR verified; microbench: per-instruction WMMA rates, H2D pinned bandwidth, DDR5, NVMe 2-thread QD8.
- Reference correctness harness: run HF `inkling2_loop` (eager) on a small prompt → dump per-layer hidden states + logits → our golden outputs. Bit-compare standard for every kernel change.

### P1 — vertical slice decode (weeks 1–3): one token out, correctness first
- Vulkan backend skeleton (volk, device probe, slab allocator, cmd-pool, timeline sems).
- Kernels v1 (correctness over speed): rmsnorm/L2-clip, RoPE-rel, sconv-4, dense GEMV bf16, NVFP4-dequant GEMV (naive), attention naive-but-correct (SWA + global + softcap + sink), router top-k + bias + softcap, embed/unembed, sampler.
- CPU-only weight residency OK for v1 (RAM mirror → per-layer H2D staging). Goal: **greedy-equivalence vs HF reference on a test prompt**.
- CLI: load, prompt, decode; `st_header.json` deleted (scratch file cleanup noted in research).

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
