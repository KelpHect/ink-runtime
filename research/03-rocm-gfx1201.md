# A03 — ROCm on RDNA4 / gfx1201 (RX 9070 XT) — state as of late 2026

Scope: everything relevant to a hand-rolled C++ LLM inference runtime for a 26.4B MoE
(NVFP4 weights + MXFP8, ~20 GB, 128 experts × 8 active, GQA head_dim=128,
sliding-window 1024, QK-L2 logit clipping, MTP draft head) on a single
Radeon RX 9070 XT (gfx1201, 16 GB) + 48 GB RAM + NVMe, Windows 11 host with
Linux dual-boot / WSL2 possible.

---

## TL;DR / verdict

**Use HIP/ROCm as the primary backend, develop and run on bare-metal Linux
(ROCm 10.x). Keep Vulkan as an optional secondary backend, not the main one.**

Why, in one paragraph: gfx1201 is now *officially* supported (not just
community-enabled) on Linux since ROCm 6.4.1 and is Release-Ready in TheRock on
**both** Linux and Windows. HIP is the only path that reaches the hardware's
FP8 tensor-core instructions (`v_wmma_f32_16x16x16_fp8`, E4M3/E5M2 OCP) through
hipBLASLt, CK, or raw WMMA intrinsics — and MXFP8 is the one quantization in
our model that maps to real silicon. **There is no FP4 WMMA on gfx1201:**
NVFP4 must be dequantized to FP16/BF16/FP8 in-kernel regardless of backend, so
FP4 buys memory savings, not compute. The attention stack we need (sliding
window + logit softcap/clip + paged decode) has no off-the-shelf kernel on
gfx1201 anyway — we will write one, and HIP gives us the best substrate for it
(ck_tile FMHA template already implements SWA + softcap, rocWMMA/`__builtin`
intrinsics are documented, rocprofv3 SPM now works on gfx1201). Vulkan is fine
for portability but has no exposed FP8 matrix path on AMD drivers today and has
shown pathological per-model slowdowns on this exact GPU in llama.cpp.

Windows is no longer a dealbreaker for HIP: TheRock ships release-ready Windows
artifacts for gfx1201 (hipBLASLt, rocBLAS, CK, MIOpen included) and the HIP SDK
for Windows 7.2.0 lists the 9070 XT as fully supported. But profiling,
debugger, and several libs are still Linux-only, and WSL2 (librocdxg), while
production-supported since Adrenalin 26.2.2, keeps adding small failure modes
(TDR hangs on ROCm 10.1/10.2 wheels). **Linux bare metal is the low-risk
development target; WSL2 is a workable fallback; native Windows HIP is viable
but the least mature option.**

---

## 1. ROCm version landscape and gfx1201 support status

Release history (https://rocm.docs.amd.com/en/latest/release/versions.html):

| Release | Date | Notes |
|---|---|---|
| 6.4.1 | May 2025 | **First official gfx1201 support on Linux** (RX 9070/9070 XT, R9700) |
| 6.4.2 / 6.4.3 | Jul–Aug 2025 | Bugfix |
| 7.0.x | Sep–Oct 2025 | New numbering era; gfx1201 still Ubuntu/RHEL-only |
| 7.1.x | Oct–Nov 2025 | gfx1201 official; ROCm runtime feature enabling DXG/WSL |
| 7.2.0–7.2.4 | Jan–May 2026 | WSL production support (ROCDXG); PyTorch-on-Windows gains ROCm 7.2.1 components |
| 7.14.0 / 7.14.1 | Jul / Sep 2026 | **Transition release: ROCm switches to TheRock build system** ("ROCm Core SDK", modular installs) |
| 10.0.0 | Aug 26 2026 | First 10.x |
| 10.1.0 | Oct 5 2026 | rocSPARSE RDNA4 tuning; adds AI PRO R9600 |
| **10.2.0** | **Nov 18 2026** | **Current release.** PyTorch 2.15, vLLM 0.32, SGLang 0.5.21 validated; hipBLASLt 1.5.0; SPM profiling on gfx1201 |

(Sources: https://rocm.docs.amd.com/en/latest/release/versions.html,
https://rocm.docs.amd.com/en/docs-10.2.0/about/release-notes.html,
https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.2/reference/system-requirements.html)

**Is gfx1201 "supported" or community-supported?** Officially supported.
The ROCm support matrix lists RX 9070 XT / 9070 / 9070 GRE / AI PRO R9700 /
R9600(D) under the ✅ "Supported" tier on Linux since 6.4.1, and the TheRock
support matrix marks gfx1201 **Build Passing ✅ / Sanity Tested ✅ / Release
Ready ✅ on both Linux and Windows** — the only RDNA arch with full marks
besides gfx1200 and the gfx115x APUs. Caveat: it is supported the way consumer
Radeons are supported — validated OS list is narrower than Instinct (Ubuntu
24.04, RHEL 9/10), and per-release fixes for RDNA4-specific issues visibly lag
CDNA. gfx1201 is genuinely a second-class citizen relative to gfx942/gfx950,
but a *supported* one, and the situation improved steadily through 2026.

Reality check on quality: late-2026 still sees gfx1201-specific bugs being
filed and fixed — e.g. GFXCLK stuck at 41 MHz under inference on ROCm 7.2.3
(3× perf loss until `profile_peak` workaround;
https://github.com/ROCm/ROCm/issues/6289), `amdsmi_set_clk_freq` falsely
returning INVAL on Navi48 (fixed in 10.2.0), and a WSL2 TDR on ROCm 10.x
wheels. The stack works and is fast, but expect to occasionally set an env var,
pin a version, or apply a workaround.

---

## 2. TheRock — what it is and what it ships for gfx1201

**What it is** (https://github.com/ROCm/TheRock): AMD's unified build/release
system for all of ROCm — a CMake super-project that builds rocm-systems,
rocm-libraries, compilers, and PyTorch from source, for Linux and Windows.
Since ROCm 7.14 (July 2026), *TheRock is no longer a side project: it is how
ROCm itself is built and released.* The "ROCm Core SDK" of 7.14/10.x is a
TheRock output. Repo self-description still says "under active development, not
yet stable for production," but the official ROCm releases are now produced by
it, so treat that warning as legal hedging.

**Distribution channels:**
- Multi-arch nightly + release tarballs at `nightly.repo.amd.com/rocm/core/tarball/`
  and `stable.repo.amd.com/rocm/core/tarball/` (see RELEASES.md:
  https://github.com/ROCm/TheRock/blob/main/RELEASES.md). Since the multi-arch
  refactor, there is **one index for all GPUs**; GPU-specific kernel packs are
  split out per-target (`gfx120X-all` family + per-LLVM-target device packs).
- **Python wheels**: `rocm-sdk-core`, `rocm-sdk-devel`,
  `rocm-sdk-device-gfx1201`, `rocm-sdk-libraries`, etc. — install ROCm with pip
  (users have built FlashAttention against `rocm-sdk-* 10.1.0a` wheels *on
  Windows 11*; https://github.com/Dao-AILab/flash-attention/issues/2800).
- **PyTorch wheels for gfx1201 on both Linux *and* Windows** — there is a
  dedicated "Multi-Arch Release Windows PyTorch Wheels" CI job (RELEASES.md).
- llama.cpp's official `win-hip-radeon` binaries now run on Windows by simply
  putting the TheRock runtime tarball (`rocm-runtime-gfx120X-7.14`) on PATH —
  evidence the Windows runtime artifacts are real and usable in production
  tooling (https://github.com/Gildartslol/9070xt-uncensored-bench).

**Windows subproject coverage for gfx1201**
(https://raw.githubusercontent.com/ROCm/TheRock/main/docs/development/windows_support.md):
rocRAND/hipRAND, rocPRIM, hipCUB, rocThrust, rocFFT/hipFFT, hipBLAS-common,
**hipBLASLt, rocBLAS, rocSPARSE, hipSPARSE, rocSOLVER, hipSOLVER, hipBLAS,
rocALUTION, rocWMMA, Composable Kernel, hipTensor, libhipcxx — all ✅, plus
MIOpen, hipdnn providers ✅.** Missing on Windows: RCCL (single-GPU runtime
doesn't care), rocprofiler-sdk/aqlprofile/rocprofiler-compute (no perf counters
— the real gap for kernel work), amdsmi/rocm_smi, ROCR-Runtime + rocminfo
(Windows uses AMD's closed-source thunk instead — i.e. you do *not* get the
open-source user-mode runtime on Windows), rocgdb, hipFile, hipSPARSELt,
rocRoller, mxDataGenerator. `clr` is 🟡 (needs prebuilt static libs).

**Maturity verdict:** on Linux, TheRock outputs *are* ROCm 10.x — fully usable.
On Windows, artifacts exist, are release-flagged, and real software consumes
them, but the Windows bringup is ~18 months old, the runtime is closed-source,
and the toolchain minus profilers/debugger makes it a "works, but you fly
half-blind" platform for hand-rolled kernel development.

---

## 3. Kernel libraries for gfx1201

### 3.1 What the silicon actually does (this constrains everything)

RDNA4 matrix hardware = **WMMA (wave32)** family, builtins
`__builtin_amdgcn_wmma_*_w32_gfx12` (ROCm handbook:
https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html).

Supported element types on gfx1200/gfx1201: **f16, bf16, fp8_e4m3 (with
fp8_e5m2 as B operand), int8, iu4 (int4), plus 2:4 sparse variants** and a
deeper-K INT4 form (16×16×32). **No FP4 (E2M1) instruction exists** — FP4
hardware GEMM is a gfx950 (CDNA4) / gfx1250 (CDNA5) feature. Multiple sources
confirm: https://zolotukhin.ai/blog/2026-05-09-the-fp4-wave-breaks-at-rdna4-and-fp8-wmma-already-does-what-local-qwen3-needs/
and the AMD ROCm optimization guide above. All RDNA4 shaders default to wave32.

Published peaks: RX 9070 XT ≈ 97 TFLOPS FP16-vector / ~389 TFLOPS FP16-matrix
dense (per the mblas-bench analysis in
https://github.com/ROCm/ROCm/issues/5861); the AI PRO R9700's lower clocks put
its dense FP16 matrix peak at 191 TFLOPS per AMD's own numbers. FP8 matrix ≈
2× FP16.

**Implication for our model:** MXFP8 E4M3 weights have a native 2×-matrix-rate
path; NVFP4 (E2M1) does not — every NVFP4 GEMM on gfx1201 is "dequantize to
f16/bf16 (or fp8) in-kernel, then WMMA." The win from NVFP4 is memory/bandwidth
only — which still matters a lot when ~20 GB of weights must stream over a
16 GB VRAM budget and PCIe. A community fused MXFP4→FP16-WMMA GEMM reaches
40.8 TFLOPS on the R9700 via an LDS LUT decoder
(https://github.com/JohnTDI-cpu/rdna4-wmma-guide), i.e. ~21% of the FP16
matrix peak — showing both that it's feasible and that FP4-on-RDNA4 is a
hand-rolled affair everywhere, HIP included. (That repo also documents the
non-obvious gfx12 WMMA *column*-distributed fragment layout — worth bookmarking
before writing any WMMA kernel.)

Note the **int4 (`iu4`) WMMA** alternative: on gfx1201 the only sub-8-bit
matrix instruction is integer. An int4-path quantization of the same weights
could in principle hit int-tensor rates; worth remembering if NVFP4 dequant
kernels underperform.

### 3.2 hipBLASLt — GEMM coverage on gfx1201

Library-level dtype table
(https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/data-type-support.html):
int8, FP8 E4M3, FP8 E5M2, FP16, BF16, FP32 fully supported as in+out; **FP4
(E2M1) input-only**. Hardware availability: FNUZ FP8 = gfx942 only; **OCP
E4M3/E5M2 = gfx950 + gfx12** (https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/what-is-hipBLASLt.html).
Scale-granularity modes: scalar, outer-vector (per row/col), and
**VEC32_UE8M0 block scaling — the MXFP8 granularity** — supported for FP8/FP4
operands subject to m/n/k alignment and transpose restrictions. **FP4 input
GEMM and the newest MX (FP6/FP4 mixed) kernels ship for gfx950/gfx1250, not
gfx1201** (ROCm 10.2.0 hipBLASLt 1.5.0 changelog:
https://rocm.docs.amd.com/en/docs-10.2.0/about/release-notes.html).

gfx1201 coverage in shipped catalogs exists since ~ROCm 6.4 (hipBLASLt PR
#1715, Mar 2025: tuned equality kernels for Llama-3.1-8B / Llama-3.2-3B /
Qwen2.5-7B FP16+BF16 vLLM shapes at batch 1, Q4_K_M llama.cpp shapes, SD3 —
https://github.com/ROCm/hipBLASLt/pull/1715). Coverage quality is **thin and
shape-dependent**, though improving:

- ROCm 7.1.1: 4k³ FP16 GEMM hits only ~71 TFLOPS (18% of peak), FP32 ~12 TFLOPS
  (issue https://github.com/ROCm/ROCm/issues/5861).
- Kernel selection is **nondeterministic across processes** on RDNA4 (same
  shape → different kernel → different bitwise results; GPTQModel documents it
  and forces `ROCBLAS_USE_HIPBLASLT=0`:
  https://github.com/ModelCloud/GPTQModel/blob/main/docs/rocm-rdna4.md). For
  our deterministic-inference goals, pin solutions via
  `HIPBLASLT_TUNING_FILE`/`HIPBLASLT_TUNING_OVERRIDE_FILE` (much improved in
  1.5.0) or use custom kernels for the hot GEMMs.
- 10.2.0 changelog: *"Improved fp16/bf16 NN GEMM on gfx1100 and **fp8, bf8 and
  int8 TN GEMM performance on gfx1201 with Origami libraries**"* — AMD is
  actively retuning gfx1201 via the Origami analytical selection framework.
  Community experience on 7.2-era: a hand-tuned solution table made a real
  model's prompt phase *slower* than heuristics
  (https://github.com/Niko1221/Strata/issues/979) — i.e., the heuristic path is
  already decent for tuned LLM shapes; generic shapes remain the weak spot.
- **GFA (Gemm-From-Anywhere)** in 1.5.0 is noteworthy: hipBLASLt can now load
  and dispatch *externally authored* GEMM kernels (CK, rocRoller, Wave, Triton,
  AITER, or ours) through its solution-selection path — a clean integration
  seam for hand-written gfx1201 kernels.
- w4a16 (GPTQ/AWQ-style int4-weight) GEMM API exists in 1.5.0 but the custom
  assembly kernels listed are gfx1151-only so far.
- rocBLAS forwards most GEMMs to hipBLASLt on this arch — the same tuning caveats
  apply transitively.

**Bottom line:** FP16/BF16/INT8/FP8-E4M3(+E5M2) dense GEMM, MXFP8-class block
scaling, grouped/batched — all callable on gfx1201 today from C++, with
improving-but-incomplete tuning. No FP4 compute. Budget for custom GEMM kernels
for NVFP4-dequant and possibly for MoE grouped GEMM.

### 3.3 Composable Kernel (CK)

Header-template library, WMMA gfx12 support landed through 2025–2026
(CHANGELOG: "Added WMMA (gfx12) support for FMHA"; "Improved FMHA forward
performance for head dimension 128 on gfx11 and gfx12";
https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md).

- **CK Tile FMHA** (the flash-attention kernel): supports gfx9/gfx11/gfx12 in
  the QR / QR-Async pipelines (TrLoad/V3 pipelines are gfx950+). Feature set
  per example README
  (https://github.com/ROCm/composable_kernel/blob/develop/example/ck_tile/01_fmha/README.md):
  head dims 32/64/128/256 (64/128 best-tuned), fp16/bf16, causal + **FA-style
  sliding-window masks `t:l,r` / `b:l,r`**, xformer-style windows, bias, ALiBi,
  varlen, paged-KV support in the varlen path, and **`logits_soft_cap`
  (softcap) as a template parameter** (`kHasLogitsSoftCap`, fmha_fwd.hpp).
  Backward exists for gfx12 but the deterministic variant is unstable (GPU
  hang) — the Dao-AILab flash-attention merge for RDNA explicitly enables
  nondeterministic-only bwd on gfx12
  (https://github.com/Dao-AILab/flash-attention/pull/2400). Irrelevant for
  inference-only, worth knowing.
- **CK Tile Engine GEMM**: ROCm 10.2 added bf16/fp32/**fp8/bf8** dtypes and
  rrr/crr/ccr layouts to the tile-engine batched GEMM + dispatcher — i.e. a
  usable blockscale-adjacent grouped/batched GEMM substrate on gfx12.
- **CK MoE/fused-MoE GEMM**: effectively CDNA-only in practice. aiter's CK MoE
  kernels (`fmoe_g1u1` etc.) find no gfx12 instances
  (https://github.com/ROCm/aiter/issues/4492); aiter's `gemm_a8w8` CK int8 and
  CK a8w8-blockscale paths emit no gfx12 kernels and get routed to Triton
  instead (data point in https://github.com/ROCm/aiter/issues/3294). For our
  MoE GEMMs: Triton/aiter-Triton works, CK source is adaptable, or hand-roll.

### 3.4 AITER (AMD's AI kernel library)

README support table (https://github.com/ROCm/aiter/blob/main/README.md):
gfx942 fully supported, gfx950 supported, **gfx1201 (R9700) = "Experimental"**
— "on RDNA, Triton and most FlyDSL kernels run, as do most HIP kernels; most
CK and ASM kernels are CDNA-only." Experimental ≠ slow: AMD recommends
installing it even on RDNA because it often beats framework defaults.

What works on gfx1201 today (community-verified list, Sep 2026, aiter issue
#3294 + release notes):
- Triton: `gemm_a8w8`, `gemm_a8w8_blockscale` (dispatch→Triton, 1.15–1.3× vs
  fp16 at M≥256, up to 3× at M=1 wide shapes), Triton MoE, `unified_attention`
  (works modulo a vLLM-version-dependent LDS overflow bug,
  https://github.com/vllm-project/vllm/issues/48723).
- HIP kernels: `rms_norm`, `silu_and_mul`, per-tensor/per-token/**per-group
  (1×128) FP8 quant** (that's the MXFP8-block quant shape), `rope_fwd`,
  `moe_sorting`, `fused_qk_norm` (fixed for gfx12 asm gaps, PR #3330),
  **chunk-gated-delta-rule** (hybrid-SSM kernel, ported to gfx12 WMMA —
  https://github.com/ROCm/aiter/pull/4523), gfx1201-specific **SageAttention**
  (INT8 QK / FP8 PV) landing via ROCm/aiter#6390.
- **FlyDSL `flash_attn_func` for gfx1201**: first native RDNA4 attention
  backend in aiter — BSHD bf16/f16, causal/non-causal, exact tail masking, plus
  an FP8-E4M3 attention path with a fused gfx1201 FP8 quant producer
  (https://github.com/ROCm/aiter/pull/4188). Feature caveats: no documented
  sliding-window or softcap support — it targets diffusion attention.

What doesn't: all hand-written gfx9 ASM kernels (fmha_v3 etc. — dispatch bugs
now fail loudly instead of silently, per PR #3330), CK-based fused MoE
(`fmoe_g1u1`: no gfx12 kernel, issue #4492, still open as of Nov 19 2026), CK
gemm_a8w8 int8 / CK a8w8-blockscale (no gfx12 instances), top-level CK
`flash_attn_func` (JIT-compiles 134 s then `invalid argument`).

Value for us: **reference implementations and a parts bin** — moe_sorting, the
FP8 quant kernels (incl. 1×128 group), rope, rmsnorm, gated-delta-rule, and the
FlyDSL gfx1201 attention are directly portable ideas/code; fused MoE is the
notable gap (Triton MoE in vLLM is the working alternative).

### 3.5 MIOpen / MIGraphX / rocWMMA

- MIOpen 3.6.2: officially supports gfx1201, Linux *and* Windows in ROCm 10.2 —
  irrelevant for LLM inference (conv-focused) but a marker of how far Windows
  coverage has come.
- **MIGraphX 2.19.0: gfx1200/gfx1201 officially listed**
  (https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/migraphx.html).
  ONNX graph engine — no paged-attention/MoE LLM path of interest; note only.
- rocWMMA: header-level WMMA fragment library, gfx12 supported on both OSes —
  a reasonable substrate for our custom attention/GEMM kernels, with the caveat
  that the RDNA4 fragment layout differs from naive expectations (see §3.1
  JohnTDI guide).

---

## 4. PyTorch / vLLM / SGLang on gfx1201

### PyTorch

- Officially validated: **PyTorch 2.15.0 with ROCm 10.2** (Linux) and PyTorch
  2.15.0 Windows wheels covering gfx1100–gfx1201 (ROCm 10.2.0 release-notes
  framework table). Earlier: torch 2.9.x on ROCm 7.2, torch 2.13+rocm10.x
  nightlies ship in `rocm/pytorch` and `vllm/vllm-openai-rocm` images.
- Windows: PyTorch wheels on Windows ship embedded ROCm components (7.2.1-era
  doc: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows%5Fcompatibility.html)
  and TheRock CI now produces Windows gfx1201 torch wheels.
- SDPA on gfx1201 → **AOTriton** (the default ROCm flash backend). gfx1201
  support landed in aotriton 0.9.2b behind
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (March 2025, pytorch#148433);
  since aotriton 0.14b (Sep 2026) the FlyDSL `flyc` kernel gives gfx950/gfx1201
  big gains (+16–100% fwd vs 0.13b) and feature parity with the Triton kernel;
  0.15b (Nov 2026) fixed a gfx1201 bf16-dQ rounding accuracy bug
  (https://github.com/ROCm/aotriton/releases,
  https://pytorch.org/blog/accelerating-pytorch-on-amd-gpus-with-flydsl/).
  Residual rough edges exist (e.g. head_dim 72 sticky `hipErrorInvalidValue`
  on torch 2.12+rocm10.0, https://github.com/ROCm/aotriton/issues/251).
  CK-SDPA is also available in PyTorch but **opt-in** via
  `USE_ROCM_CK_SDPA_GFX12` (pytorch#193585/#188114).
- HSA_OVERRIDE_GFX_VERSION=12.0.1: **obsolete for gfx1201.** It's a
  code-object spoof for loading a *different* arch's kernels; on ROCm ≥ 6.4.1
  gfx1201 is native, and AMD support explicitly told a user to drop it on 7.2.x
  (https://github.com/pewdiepie-archdaemon/odysseus/issues/4596). Harmless
  no-op today, cargo-culted by old guides; never worked on Windows
  (https://amdai.net/rocm-problems/). Only use case left: forcing e.g.
  gfx1100 objects on very old ROCm installs.

### vLLM

- Officially supported: vLLM 0.32 validated on gfx1201 (ROCm 10.2); device
  table has both 0x7550 (9070 XT) and 0x7551 (R9700) entries
  (https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py).
- **Attention backends:** default `ROCM_ATTN` is buggy on RDNA — wrong outputs
  and GPU faults for many head-size/KV-block combos (head≠128 or block≠16,
  plus MHA head 192/256; 635 wrong + 119 faults over 12k batches,
  https://github.com/vllm-project/vllm/issues/62697). `TRITON_ATTN` is
  correct on all tested shapes (and required for FP8 KV cache on gfx1201).
  New opt-in `ROCM_RDNA_ATTN` (PR #67445) ships gfx1201-tuned tables: **77% of
  decode roofline, 68% prefill; 1.8× TRITON_ATTN, 30× ROCM_ATTN decode** —
  the current best reference for a gfx1201 paged-attention implementation.
- AITER path: `ROCM_AITER_UNIFIED_ATTN` works but hit an LDS-overflow bug on
  gfx1201 in vLLM v0.25.0 (issue #48723); `VLLM_ROCM_USE_AITER=1` still has
  gaps on gfx12 (aiter #3294).
- FP8 works: vLLM enables the FP8 expert path on gfx12x and a tuned Triton FP8
  MoE config for R9700 exists for Qwen3-30B-A3B-class shapes (PR #36659).
- Community data point: a 4×R9700 deployment ran `Qwen3.6-27B-FP8` — close
  analog to our ~26B MoE workload (issue #48723 env block).

### SGLang

- ROCm 7.14 added *initial* Radeon support; officially, on Radeon you must
  disable AITER (`SGLANG_USE_AITER=0`) and use `--attention-backend triton`,
  and **some MoE models (GPT-OSS-20B, MiniMax-M2.7) don't run correctly** —
  actively being fixed upstream
  (https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/sglang.html).
  RDNA4-specific work is landing steadily (gfx1201 SageAttention, HIP gated
  delta-rule for Qwen3.6 hybrids). Net: usable, Triton-first, not turnkey for
  MoE — reinforces the "don't build on aiter/CK for gfx12" theme.

---

## 5. Windows native, WSL2, or Linux?

### Native Windows HIP
- **HIP SDK for Windows 7.2.0 lists RX 9070 XT (gfx1201): Runtime ✅, HIP SDK
  ✅, ROCm Debugger ✅**
  (https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html).
- TheRock Windows artifacts: gfx1201 Release-Ready; hipBLASLt/rocBLAS/CK/MIOpen
  build ✅; wheels `rocm-sdk-device-gfx1201` exist and are used in the wild.
- Gaps: closed-source HIP runtime/thunk; no RCCL; **no profiler stack
  (rocprofiler-sdk/compute unsupported)**; no rocRoller/hipFile/hipSPARSELt;
  `clr` needs prebuilt statics; FlashAttention CK **backward** on Windows
  gfx1201 measured 4.4–7.4× slower than SDPA
  (https://github.com/Dao-AILab/flash-attention/issues/2800 — fwd is fine).
  Component-support doc also still lists AI libs/frameworks as not available
  in the classic HIP SDK bundle (TheRock wheels bypass that:
  https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html).
- Verdict: **viable for running/shipping, poor for developing kernels** (no
  counters/debugger parity). Good tier-2 target.

### WSL2 (librocdxg / ROCDXG)
- **Production-supported since Adrenalin 26.2.2 + ROCm 7.2.1** — AMD's
  open-source user-mode DXG thunk; decoupled from ROCm/driver versions
  (https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html).
- Works on gfx1201: llama.cpp HIP backend built natively for gfx1201 runs
  stable on WSL2 (TinyLlama tg ≈ 280 t/s, 50-min soak, no zombie-state;
  https://github.com/ROCm/librocdxg/issues/74). ROCm 7.13+ auto-detects DXG
  (no more `HSA_ENABLE_DXG_DETECTION`).
- Caveats are real: **ROCm 10.1.0 wheels on WSL2 hung the first kernel with a
  Windows TDR** (fixed — rocm-systems#13291); ROCm 10.2.0 still lists a WSL2
  TDR known-issue for a gpt-oss FP4 vLLM workload, and an illegal-memory-access
  known-issue under WSL2 for APU-based workloads (10.2.0 release notes).
  User-mode thunk = no native KFD → some low-level features (certain profiling
  counters, VM tricks) are absent or emulated; historically CK-heavy workloads
  on WSL showed extra overhead, and the newer known-issues keep that
  reputation alive.
- Verdict: **acceptable dev environment, not the perf target.** For a project
  that lives in the kernel/ISA weeds, dual-boot Linux avoids an entire class of
  "is it WSL or is it me" debugging.

### Linux bare metal — the recommended dev target
amdgpu KFD driver + ROCm 10.x packages or TheRock wheels; full toolchain
(rocgdb, rocprofv3 incl. SPM on gfx1201, amdsmi), deterministic behavior, and
the platform every gfx1201 fix lands on first.

---

## 6. FlashAttention on RDNA4 — feature matrix

Our model needs: head_dim 128, GQA, **sliding window 1024**, **QK logit
softcap/L2-clip**, paged/varlen decode, bf16 — inference only.

| Path | gfx1201 status | SWA | softcap | notes |
|---|---|---|---|---|
| **CK Tile FMHA** (ck_tile headers) | ✅ WMMA pipeline, fwd+bwd (bwd nondeterministic only) | ✅ FA-style l,r masks | ✅ `logits_soft_cap` template param | Best fit: source-available, both features in one template |
| flash-attention CK backend (Dao-AILab + ROCm fork) | ✅ gfx11/gfx12 merged (PR #2400) | ✅ | ✅ | Python extension; still a useful reference for flag plumbing |
| aiter FlyDSL `flash_attn_func` | ✅ native gfx1201 | ✗/undocumented | ✗ | + FP8-E4M3 attention variant; diffusion-oriented |
| aiter gfx1201 SageAttention | 🚧 landing (aiter#6390) | n/a | n/a | INT8 QK + FP8 PV, head 128, non-causal — diffusion |
| aotriton (PyTorch SDPA) | ✅ flyc backend, fast | via SDPA attn_mask only | ✗ | not exposed as standalone lib with these flags; SDPA API lacks softcap |
| vLLM `ROCM_RDNA_ATTN` (Triton paged) | ✅ tuned tables for 9070 XT | backend-dependent | n/a | 77% decode roofline — **best public reference for paged gfx1201 attention** |
| vLLM `TRITON_ATTN` | ✅ correct all shapes | ✅ | n/a | slower than RDNA_ATTN but the correctness baseline |
| Triton FA (aiter `mha` / ROCm FA Triton backend) | ✅ runs | WIP→partial | unclear | supports varlen/GQA/ALiBi/paged/FP8(FA3 API) |
| FlashAttention-3-style | ✗ N/A | — | — | FA3 is Hopper-specific; nothing async-pipeline-class on gfx12 (FlyDSL is the closest modern effort) |

**Assessment:** no shipped kernel does all of SWA + softcap + (our custom)
QK-L2 clip + paged decode on gfx1201. That was always going to be a custom
kernel. Best starting points, in order: **(1) ck_tile FMHA template** — SWA and
softcap already parameterized; adding a QK clip is a small tile-pipeline edit;
**(2) hand-rolled WMMA FA** using gfx12 builtins/rocWMMA informed by the
RDNA4-WMMA lane-layout guide and aiter's FlyDSL gfx1201 kernel as reference;
**(3) port vLLM's ROCM_RDNA_ATTN Triton kernels** if we want a Triton path
(Triton works well on gfx1201 and compiles for it via the ROCm backend).

---

## 7. Recommendation and reasoning

**Primary: HIP/ROCm on Linux (bare-metal dual boot), ROCm 10.2.x via TheRock
wheels or distro packages.**

1. **Only HIP reaches the matrix hardware we need.** MXFP8-E4M3 block-scaled
   GEMM is a first-class hipBLASLt path on gfx12 (VEC32_UE8M0 scale mode), and
   FP8 WMMA gives ~2× the FP16 matrix rate. Vulkan on AMD exposes cooperative
   matrix for fp16/bf16 (and int dot products) but has no shipping FP8-matrix
   or FP4 path on RDNA4 today — llama.cpp's own device probe on this GPU shows
   `fp4: 0` and no fp8 coopmat flag
   (https://github.com/ggml-org/llama.cpp/issues/26663). For NVFP4, both
   backends need custom dequant-WMMA kernels anyway, but only HIP lets the
   dequant target FP8 tensor cores.
2. **MoE is the project's hottest path and the gfx12 MoE story is
   Triton/HIP-kernel-shaped, not Vulkan-shaped.** Working pieces to borrow:
   aiter `moe_sorting` + Triton/Gluon blockscale MoE GEMMs, vLLM's tuned R9700
   FP8 MoE config, CK Tile Engine batched GEMM (fp8/bf8 added in 10.2), and
   hipBLASLt grouped GEMM. A hand-rolled HIP grouped-GEMM (dequant NVFP4 →
   FP8/BF16 → WMMA) is well-trodden territory; there's no equivalent substrate
   in Vulkan-land except writing everything from scratch in GLSL/SPIR-V with
   weaker tooling.
3. **Custom attention is mandatory** (QK-L2 clip) and HIP offers the two best
   starting templates (ck_tile FMHA with SWA+softcap; FlyDSL gfx1201 kernel)
   plus real profilers (rocprofv3 SPM now supports gfx1201) — Vulkan kernel dev
   on Windows/RADV means RGP-only counters and a much thinner debug story.
4. **Observed Vulkan fragility on this exact card**: a 5–7× decode collapse on
   llama.cpp for hidden_size ≥ 4096 (issue #26663), and even favorable
   benchmarks show HIP winning prefill by ~25–31% while decode is a wash
   (retrivium.ai owner benchmarks). For a memory-bound decode the backends
   converge; for prefill and MoE GEMMs HIP is strictly ahead.
5. **Ecosystem gravity**: everything that will debug our model class on this
   GPU — vLLM, aiter, CK examples, llama.cpp HIP, TheRock CI — is HIP-first.

**Secondary (keep, don't lead): Vulkan**, as a portability/fallback path —
useful for a Windows-native build with zero ROCm runtime, for environments
where the ROCm stack is a problem, and as a correctness cross-check. It's a
real option precisely because our worst-case (custom kernels everywhere) costs
the same order of effort in either API, and ggml has shown coopmat RDNA3/4 can
be competitive on decode. But it should not gate the design: FP8 tensor cores,
the MoE parts bin, and the profiling toolchain are all HIP-side.

**Do not design around:** aiter CK/ASM kernels, CK fused MoE, FA3-style async
pipelines, FP4 compute (does not exist on gfx1201 in *any* backend), PyTorch
itself (we're C++; use headers/libs only).

**Practical stack sketch:**
- `amdclang++`/`hipcc`, `-mcpu=gfx1201` (or `--offload-arch=gfx1201`), wave32.
- hipBLASLt for dense + MXFP8-blockscale GEMM where shapes are well-tuned;
  pin solutions (tuning-override file) for determinism; keep a custom-WMMA
  fallback for shapes that fall off the tuned catalog (notably NVFP4-dequant
  GEMM and odd MoE groupings; consider GFA to register them).
- Custom attention kernel derived from ck_tile FMHA (SWA=1024, softcap, plus
  QK-L2 clip in the score pipeline), bf16 KV cache; optionally an FP8-KV
  variant using the E4M3 path.
- MoE: hand-rolled grouped GEMM (expert-sorted token batch via rocPRIM/own
  sort; aiter `moe_sorting` as reference), NVFP4→bf16 in-kernel dequant,
  WMMA; FP8 path where weights are MXFP8.
- Weight streaming: ~20 GB vs 16 GB VRAM → layer/MoE-expert offload over
  PCIe 5.0 from RAM (+NVMe cold), plus MTP draft head for speculative decode;
  HIP streams + pinned host memory; hipFile isn't on Windows anyway — keep
  the IO engine bespoke.
- Env hygiene: `AMDGPU_TARGETS=gfx1201` at build; **don't** set
  `HSA_OVERRIDE_GFX_VERSION` (no-op/negative value on gfx1201); consider
  `ROCBLAS_USE_HIPBLASLT=0`-style pinning only if nondeterminism bites.
- Dev loop: Linux ROCm 10.2.x, rocgdb + rocprofv3; CI sanity on WSL2
  (librocdxg) optional; Windows-native HIP build as a later distribution
  experiment, expecting profiler-less debugging.

**Residual risks to track:** gfx1201 tuning gaps in hipBLASLt for our exact
GEMM shapes; TensileLite nondeterminism; aiter/vLLM gfx12 drift breaking
borrowed code; WSL2 TDR class of bugs if we demo on Windows-side; and the
ever-present possibility that RDNA4 stays second-class vs gfx950 for new
features (e.g. all 10.2 FP4/FP6 goodies are gfx950/gfx1250-only).

---

## Source list (21 primary)

1. ROCm release history — https://rocm.docs.amd.com/en/latest/release/versions.html
2. ROCm 10.2.0 release notes — https://rocm.docs.amd.com/en/docs-10.2.0/about/release-notes.html
3. ROCm Linux system requirements (6.4.2, gfx1201 first official) — https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.2/reference/system-requirements.html
4. ROCm compatibility matrix — https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
5. TheRock SUPPORTED_GPUS.md — https://raw.githubusercontent.com/ROCm/TheRock/main/SUPPORTED_GPUS.md
6. TheRock RELEASES.md — https://github.com/ROCm/TheRock/blob/main/RELEASES.md
7. TheRock windows_support.md — https://raw.githubusercontent.com/ROCm/TheRock/main/docs/development/windows_support.md
8. HIP SDK for Windows system requirements — https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html
9. HIP SDK component support — https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html
10. WSL how-to (ROCDXG production) — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html
11. librocdxg issue #74 (gfx1201 WSL2 verified + benchmarks) — https://github.com/ROCm/librocdxg/issues/74
12. rocm-systems issue #13291 (WSL2 10.1.0 TDR) — https://github.com/ROCm/rocm-systems/issues/13291
13. hipBLASLt data-type support — https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/data-type-support.html
14. What is hipBLASLt (scale modes, gfx12 OCP) — https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/what-is-hipBLASLt.html
15. RDNA4 dense WMMA builtins — https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html
16. "FP4 wave breaks at RDNA4" (zolotukhin) — https://zolotukhin.ai/blog/2026-05-09-the-fp4-wave-breaks-at-rdna4-and-fp8-wmma-already-does-what-local-qwen3-needs/
17. RDNA4 WMMA guide + fused MXFP4 GEMM — https://github.com/JohnTDI-cpu/rdna4-wmma-guide
18. hipBLASLt PR #1715 (gfx1201 tuning) — https://github.com/ROCm/hipBLASLt/pull/1715
19. ROCm issue #5861 (low BLAS perf gfx1201) — https://github.com/ROCm/ROCm/issues/5861
20. GPTQModel RDNA4 notes (nondeterministic selection) — https://github.com/ModelCloud/GPTQModel/blob/main/docs/rocm-rdna4.md
21. ROCm issue #6289 (GFXCLK stuck) — https://github.com/ROCm/ROCm/issues/6289

Plus: AITER README/issues/PRs (README, #3294, #4492, #4188, #4523, #3330, v0.1.13/v0.1.15 notes), CK fmha example + CHANGELOG, Dao-AILab flash-attention #2400/#2800, pytorch #193585/#188114/#148433, aotriton releases + #251, vllm #62697/#67445/#48723/#36659 + rocm.py, SGLang ROCm doc, MIGraphX doc, llama.cpp #26663, amdai.net ROCm problems, retrivium.ai benchmarks, pytorch.org FlyDSL blog, Phoronix RX 9070 compute review, ROCm FP8-GEMM-on-RDNA4 blog, vrock.ch ComfyUI-on-9070XT guide.
