# Research note 04 — GPU compute APIs for LLM inference on RX 9070 XT (gfx1201) under Windows 11

Subagent A04. Scope: every viable GPU compute API for a hand-rolled C/C++ LLM inference
runtime targeting a 26.4B-parameter MoE model on an AMD RX 9070 XT (RDNA4 / gfx1201,
16 GB GDDR6, 640 GB/s), Windows 11 host, 48 GB RAM, NVMe SSD. Compared on measured
LLM inference performance and tensor-core (WMMA / "AI accelerator") access.

Hardware reference points used throughout (AMD product page):
64 CUs, 128 AI accelerators, boost 2.97 GHz, 64 MB Infinity Cache, 640 GB/s.
Matrix-engine peak rates (dense | 2:4-sparse): FP16/BF16 195 | 389 TFLOPS;
FP8 and INT8 389 | 779 TFLOPS/TOPS; INT4 779 | 1557 TOPS.
(https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html)

A critical fact for all API comparisons below: **RDNA4's WMMA instructions are
wave32-only**. All `V_WMMA_*` ops operate on a 16×16×16 tile distributed across a
single 32-lane wavefront; the ROCm handbook notes "RDNA4 GPUs run all shader
programs in wave32 mode by default" and every WMMA builtin carries a `_w32` suffix
requiring `wavefrontsize32`. The ISA still supports both wave32 and wave64 programs
(wave64 issues each VALU/VMEM instruction once per 32-lane half = 2 cycles), so
non-WMMA kernels — e.g. decode GEMV — may still prefer wave64. Per-type
FLOPs/clock/CU: FP16 1024, BF16 1024 (2× RDNA3), INT8 2048 (4× RDNA3), plus new
FP8 E4M3/E5M2 inputs and a deeper-K INT4 variant; RDNA4 fragment layouts differ
from RDNA3 (`_gfx12` builtins), so RDNA3 WMMA code is not binary/source-compatible.
(https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html,
https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content (RDNA4 ISA ref §2.1),
https://gpuopen.com/learn/using_matrix_core_amd_rdna4/,
https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)

---

## 1. Vulkan compute

### 1.1 Tensor-core access: VK_KHR_cooperative_matrix on RDNA4

`VK_KHR_cooperative_matrix` is the only portable Vulkan path to the WMMA units.
GLSL exposes it via `GL_KHR_cooperative_matrix` (glslang support merged 2023,
https://github.com/KhronosGroup/glslang/pull/3248); Slang exposes it as `CoopMat`.
Valid M×N×K / dtype / scope combinations are driver-enumerated through
`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`.

Measured enumeration on gfx1201 (vulkanforge probe, Mesa RADV): **20 coopmat
configurations** —

* entries 0–3: FP8 (E4M3, E5M2) × FP8 → FP32
* entries 4–15: U8/I8 and mixed U8×I8 → I32, 16×16×16, saturating + non-saturating
* entries 16–19: FP16 and BF16 → FP16/FP32, 16×16×16, scope = Subgroup

(https://github.com/maeddesg/vulkanforge/commit/fa024c13ed6a8c532ff0cad80652b4d5fd9a93b6)

An INT8 coopmat smoke test compiled via shaderc and ran correctly on the RX 9070 XT
(C = 16 for 16×16×16 I8×I8→I32 of all-ones). Note this probe ran under RADV on
Linux; see §1.2 for the Windows proprietary driver's deltas.

dtype coverage summary:

| dtype | HW WMMA (gfx1201) | KHR coopmat path |
|---|---|---|
| FP16→F32/F16 | yes, 16×16×16 | yes (enumerated) |
| BF16→F32/BF16 | yes | yes (needs `VK_KHR_shader_bfloat16` — advertised on the Windows driver) |
| INT8/U8→I32 | yes | yes on RADV (enumerated); expected on Windows prop driver, **verify** |
| FP8 E4M3/E5M2→F32 | yes (new in RDNA4) | yes on RADV; **ambiguous on Windows prop driver — see below** |
| INT4 | yes (16×16×32) | **no** — no int4 component type in the spec |
| FP4 | no | no (`fp4: 0` in llama.cpp's probe) |

### 1.2 Windows proprietary driver vs RADV: the FP8 gap

The AMD Adrenalin Vulkan driver on Windows is LLPC-based — the same compiler
codebase as the discontinued AMDVLK, which AMD formally killed on 15 Sep 2025 in
favour of RADV ("the code base is still used for Windows", per an AMD engineer;
https://github.com/GPUOpen-Drivers/AMDVLK/discussions/416,
https://www.phoronix.com/news/AMDVLK-Discontinued). Adrenalin 26.10.1 exposes
Vulkan 1.4.359 and 224 device extensions on the 9070 XT
(https://www.geeks3d.com/20261023/amd-radeon-adrenalin-26-10-x-graphics-driver/).
Relevant extension history on Windows:

* `VK_KHR_cooperative_matrix`, `VK_KHR_shader_bfloat16`, `VK_KHR_shader_integer_dot_product`,
  `VK_EXT_subgroup_size_control` — present (llama.cpp probe on Windows:
  `fp16: 1 | bf16: 1 | fp4: 0 | warp size: 64 | int dot: 1 | matrix cores: KHR_coopmat`,
  https://github.com/ggml-org/llama.cpp/issues/26663)
* `VK_EXT_shader_float8` — added in Adrenalin 25.10.2 (Oct 2025). This ratified
  extension adds `VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT`/`E5M2` for cooperative matrix
  (https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_shader_float8.html,
  https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-VULKAN.html)
* `VK_EXT_cooperative_matrix_maintenance1` — added Oct 2026 (driver 26.10.1)
* `VK_EXT_shader_long_vector`, `VK_KHR_maintenance9/10/11`,
  `VK_KHR_copy_memory_indirect`, `VK_AMDX_shader_enqueue` — all exposed

FP8 caveat: despite `VK_EXT_shader_float8` being advertised, an AMD engineer told
the VulkanForge team in May 2026 that "**FP8 WMMA isn't exposed in Vulkan right
now**" — VulkanForge consequently does FP8→BF16 conversion and runs
`V_WMMA_F32_16X16X16_BF16` on Windows, while their RADV probe enumerates the FP8
configs (https://devcommunity.amd.com/t/rdna4-vulkan-compute-fp8-wmma-questions-from-an-llm-inference-engine-dev/503).
Interpretation: the Windows prop driver advertises the float8 extension but did
not (as of mid-2026) enumerate FP8 coopmat configs; **enumerate
`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` at init and treat FP8-coopmat
as a probe-time feature, not an assumption**. Front-end support is no longer the
blocker: glslang 15.4.0 (Jun 2025) added `GL_EXT_float8_e5m2_e4m3` and 16.3.0
(May 2026) added "bfloat16 and float8 tensors", so `coopmat` with float8 is
expressible in GLSL today (https://github.com/KhronosGroup/glslang/blob/main/CHANGES.md).

Upstream ggml is adding exactly this path: PR ggml-org/ggml#1528 adds an
`GGML_TYPE_E4M3` portable dequant-to-F16 type plus an opt-in
`-DGGML_VULKAN_FP8_NATIVE=ON` path that runs RDNA4's native FP8 WMMA — measured
~2× the matmul throughput of FP16 on a Radeon AI PRO R9700 and 1.18× faster than
Q4_K at whole-model level (https://github.com/ggml-org/ggml/pull/1528).
INT4 and INT8×4-bit styles have no coopmat route; the standard trick remains
dequant-to-F16 (or the INT8 coopmat path with int8 activations, which
VulkanForge's analysis calls "the only remaining candidate that could combine
mul_mmq's bandwidth advantage with WMMA throughput").

`VK_NV_cooperative_matrix2` (reductions, conversions, per-element ops, tensor
addressing, block loads, workgroup scope) is **NVIDIA-only and not ratified** —
do not design around it on AMD (https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_cooperative_matrix2.html).
Slang nevertheless exposes coopmat2-style operations (`Reduce`, `Transpose`,
`MapElement`, block loads) and lowers them where hardware support exists —
useful sugar for softmax/fusion kernels (https://github.com/shader-slang/slang/pull/7027).

### 1.3 Measured llama.cpp Vulkan vs HIP on RX 9070 XT

All numbers below are llama.cpp unless noted; treat each as build-specific.

**Windows 11, single/dual 9070 XT:**

* Qwen3.8-27B UD-Q4_K_M, 2× 9070 XT, FlashAttention, f8e4m3 KV (llama.cpp-rdna-lab
  builds, Sep 2026): HIP prefill 2061 / 1879 / 1556 tok/s vs Vulkan
  1569 / 1471 / 1244 tok/s at ~8.4k / 33k / 64k input tokens — **HIP +25–31% on
  prefill**; generation 24.7 / 25.7 / 24.9 (HIP) vs 25.4 / 23.6 / 22.2 (Vulkan) —
  **≈ parity, Vulkan marginally ahead at short context**
  (https://retrivium.ai/journal/rx-9070-xt-hip-vulkan-owner-benchmark-conditions/)
* Qwen3.8-27B, Windows, single card: ROCm pp mean 898.6 vs Vulkan 635.7 (ROCm
  +41%; pp512 peak 1138.6 vs 751.3 = +52%); tg mean 29.1 vs 33.7 (**Vulkan +16%**)
  (https://www.yeyulingfeng.com/a/1088527.html)
* Qwen3-27B IQ3_XXS, Windows, single card: HIP 1190 pp / 30.2 tg vs Vulkan
  740–814 pp / 17.2 tg — HIP ~1.8× tg (https://cangshui.net/5528.html).
  **Conflicting data point**: a different user on the same quant class measured
  Vulkan 788 pp / 38.9 tg vs HIP 371 pp / 31.8 tg — Vulkan ahead on both. The
  spread is build/config dependent, not a hardware constant (retrivium link above).
* Issue ggml-org/llama.cpp#26663 reported 4.7–6.7× Vulkan slowdown for
  hidden_size ≥ 4096 on gfx1201 (~70–100 GB/s effective vs ~460 GB/s). Follow-up
  diagnostics showed the state was **transient** (cleared by reboot, unreproducible
  by others); a healthy run on the same card gave Qwen3-8B Q4_K_M pp512 2635 /
  tg128 109.9 ≈ **514 GB/s effective — Vulkan can hit near-peak bandwidth on
  gfx1201**. Closed as not-planned/stale
  (https://github.com/ggml-org/llama.cpp/issues/26663). Takeaway: Windows prop
  driver perf is *state-sensitive*; validate with your own build.

**Linux RADV (same silicon — indicates what the API can do when the driver is better):**

* Qwen3-30B-A3B Q4_K_M on R9700 (gfx1201, Ubuntu, RADV): Vulkan ~183 t/s vs ROCm
  ~150 t/s decode — **Vulkan +22%**
* gpt-oss-20B MXFP4 on R9700: Vulkan 170.8 vs ROCm 142.6 — **+19%**
* Qwen3.5-35B-A3B on R9700: Vulkan 133 t/s (AMDVLK) / 112 (RADV) vs ROCm 68.8 —
  **+63–93% for MoE decode**
  (all three: https://www.yeyulingfeng.com/a/1088527.html)
* Llama-2-7B Q4_0 on 9070 XT (Linux): Vulkan 137 tg vs ROCm 111 (+23%); prefill
  5202 (Vulkan) (https://hearthscale.com/blog/local-ai-amd-gpu)

Pattern across all data: **decode (bandwidth-bound) is roughly parity ±25%** —
Vulkan frequently wins on MoE decode, especially under RADV; **prefill
(compute-bound) favours HIP by ~25–40%** because its GEMM path (rocBLAS/hipBLASLt
tuned kernels, direct WMMA builtins) is better optimized than ggml-vulkan's
coopmat GEMMs. A hand-rolled backend can close much of the prefill gap; the
decode gap is already a wash.

### 1.4 Structure, queues, and copies

ggml-vulkan is the reference design: a compute(+transfer)-only Vulkan library on
Vulkan-hpp targeting a Vulkan 1.2 baseline; each graph node → one compute
dispatch; handwritten GLSL shaders specialized on type/constant combos, embedded
as SPIR-V; SSBO bindings + push constants; arch detected via
`VK_AMD_shader_core_properties`/`VK_KHR_shader_integer_dot_product`/
`VK_EXT_subgroup_size_control`/`VK_EXT_shader_float8`
(Jeff Bolz, Vulkanised 2026: https://vulkan.org/user/pages/09.events/vulkanised-2026/0930-Jeff-Bolz-NVIDIA.pptx.pdf;
arch detection: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp).
The earlier Nomic/GPT4All Kompute-based backend (ggml-kompute,
https://github.com/ggml-org/llama.cpp/pull/4456) was retired in favour of this
design — Kompute (Manager/Sequence/Operation/Tensor/Algorithm over raw Vulkan)
is a fine prototyping wrapper but is unnecessary for a custom runtime
(https://kompute.cc/overview/reference.html).

Async copies: Vulkan gives `vkCmdCopyBuffer`/`vkCmdCopyBufferToImage` on any
queue; AMD hardware has SDMA engines that drivers expose as dedicated
`VK_QUEUE_TRANSFER_BIT` families on Linux (RADV) — on the Windows prop driver the
exposed family set is narrower and community engines flag "no separate transfer
queue" for AMD/Windows (STRATA_VK_NO_TQ,
http://rustingstudio.com/strata/docs/amd/). Fallback: a second queue in the
compute family still overlaps copies with compute reasonably well; also note
`VK_KHR_copy_memory_indirect` (indirect batched copies) and
`VK_EXT_host_image_copy` are exposed. With ReBAR enabled, DEVICE_LOCAL+HOST_VISIBLE
memory gives CPU-direct writes into all 16 GB VRAM — the simplest high-throughput
weight/KV load path. `VK_EXT_external_memory_host` (mapping pinned host memory so
shaders can read RAM-resident expert weights over PCIe directly — attractive for
a 26.4B MoE on 16 GB) should be probed at init.

Driver-friction notes: pipeline barriers between every dispatch (ggml-vulkan's
model) are a known tax — a custom runtime should fuse aggressively and batch
dispatches; `VK_AMDX_shader_enqueue` (device-side enqueue) is exposed if we ever
want GPU-generated work. There is one documented hazard: the Windows prop driver
can enter degraded perf states (issue #26663) — monitor effective bandwidth at
runtime.

## 2. DirectX 12 compute + DirectML + ONNX Runtime

**Stable D3D12 has no WMMA access.** WaveMMA (HLSL "Wave Matrix" intrinsics) has
existed only as a Shader Model 6.8 *preview* feature since Agility SDK
1.711.3-preview (Jun 2023), and AMD's preview driver for it covered **RX 7000
only** — 16×16 matrices, FP32 only as accumulator
(https://devblogs.microsoft.com/directx/agility-sdk-1-711/,
https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-MS-AGILITY-SDK-2023-6-711.html).
Retail SM 6.9 (Agility SDK 1.619, Feb 2026) shipped long-vectors, SER, OMM —
**but not WaveMMA** (https://devblogs.microsoft.com/directx/shader-model-6-9-retail-and-more/).
The successor is the SM 6.10 "Linear Algebra Matrix" (`linalg::Matrix` API, HLSL
proposal 0035) plus "Cooperative Vector", currently in *preview*: AMD's
AgilitySDK Developer Preview driver 26.10.07.02 lists them for **RX 9000 series**
(https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-MS-AGILITY-SDK-26-10-07-02.html).
Using them requires the preview runtime + developer-mode driver — unacceptable
driver friction for v1, but a real future path once SM 6.10 goes retail (Slang
already lowers `CoopMat` to DXIL cs_6_10, partially:
https://github.com/shader-slang/slang/commit/e55e901c05f390d774ce6c3e9183bd246a0285d6).

**DirectML** is a closed operator-graph API over D3D12; AMD's driver supplies
"ML metacommands" that internally use WMMA — that's how the optimized Llama2 ONNX
path worked (https://www.amd.com/en/blogs/2023/-how-to-running-optimized-llama2-with-microsoft-d.html).
You cannot inject custom WGSL/HLSL WMMA kernels into DML; you can interleave your
own D3D12 dispatches around DML operators, but without WaveMMA those are
SIMD-FMA-only kernels. ONNX Runtime's DML EP is turnkey but op-coverage-limited
(e.g. no ComplexFloat/FFT) and Microsoft's direction is now the plugin GPU-EP —
which on AMD is moving to a **ROCm 7.1-based runtime on Windows** anyway
(https://www.amd.com/en/blogs/2026/advancing-windows-ml-acceleration-with-amd-at-microsoft-build-2026.html).
Verdict: fine for turnkey ONNX inference, wrong tool for a hand-rolled
dequant+WMMA MoE engine.

D3D12-only niceties worth noting if the API were chosen anyway: GPU upload heaps
+ ReBAR for direct CPU→VRAM writes, work graphs (SM 6.8+ preview on Radeon),
enhanced barriers. None compensate for missing stable matrix instructions.

## 3. DirectStorage

Current version: **1.3** (NuGet `Microsoft.Direct3D.DirectStorage` 1.3.0,
Jul 2025) — adds `EnqueueRequests` (batched requests synced to a D3D12 fence) and
`DSTORAGE_DESTINATION_MULTIPLE_SUBRESOURCES_RANGE`
(https://devblogs.microsoft.com/directx/directstorage-1-3-is-now-available/,
https://www.nuget.org/packages/Microsoft.Direct3D.DirectStorage/1.3.0).

What it does NOT do on PC: **there is no NVMe→VRAM peer DMA** (that's Xbox's
decompression silicon / Linux GDS). The pipeline is file → **system-memory
staging buffer** (default 32 MB, resizable via `SetStagingBufferSize`; 128 MB
recommended for GPU decompression) → optional GPU decompression → VRAM via the
GPU copy engine (https://github.com/microsoft/DirectStorage/blob/main/Docs/DeveloperGuidance.md,
https://sherief.fyi/post/directstorage-poor-pc-port/,
https://developer.nvidia.com/blog/accelerating-load-times-for-directx-games-and-apps-with-gdeflate-for-directstorage/).

GPU decompression (GDeflate, since DS 1.1) runs on **any DX12 + SM6.0 GPU
including AMD**; vendor "metacommands" provide optimized paths and AMD ships them
(https://devblogs.microsoft.com/directx/directstorage-1-1-now-available/,
https://gpuopen.com/amd-support-for-microsoft-directstorage-1-1). Decompression
lands on compute or copy queue (`DSTORAGE_COMPRESSION_SUPPORT_USES_COMPUTE_QUEUE`
etc., https://learn.microsoft.com/en-us/windows/win32/dstorage/dstorage/ne-dstorage-dstorage_compression_support).
No ReBAR requirement; Windows 11 preferred (lower-overhead IO path).

Relevance for us: **DirectStorage is welded to D3D12 resources** — you cannot
write into a `VkBuffer` with it. For a 26.4B MoE, weight files are raw quant
blobs (not GDeflate), so the win over hand-rolled overlapped `ReadFile`/IOCP →
pinned ring → GPU copy is modest anyway; GDeflate on weights would trade a
compression pass for maybe 1.3–1.8× effective disk bandwidth. If v1 is Vulkan,
re-implement the same pattern (staging ring + async copies); if a D3D12
companion device is ever added, DS 1.3's fence-synced batching is genuinely nice
for KV paging. Also see IBM's fastsafetensors which does DirectStorage-based
model loading on Windows for reference
(https://github.com/foundation-model-stack/fastsafetensors/blob/main/fastsafetensors/cpp/dstorage.h).

## 4. HIP on Windows (HIP SDK)

**Support status: solved.** HIP SDK 7.2 for Windows 11 officially lists
RX 9070 XT (gfx1201) for Runtime + HIP SDK + ROCm Debugger
(https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html).
This is recent — mid-2025 the last SDK (6.2.4) had no RDNA4 and the workaround
was TheRock native-Windows ROCm builds (https://github.com/ROCm/ROCm/issues/4934);
PyTorch-on-Windows ROCm 7.2.1 wheels now exist with FP8 "supported only on RDNA4"
(https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows%5Fcompatibility.html).
Windows releases trail Linux and skip versions (5.6, 6.0, 6.3 skipped; 7.2.0 joint)
(https://rocm.docs.amd.com/projects/install-on-windows/en/latest/about/release-versioning.html).

Compiling kernels: `hipcc`/`amdclang++` with `--offload-arch=gfx1201`; WMMA via
`__builtin_amdgcn_wmma_*_w32_gfx12` intrinsics or the header-only **rocWMMA**
library, which supports gfx1200/gfx1201 including F8/BF8
(https://rocm.docs.amd.com/projects/rocWMMA/en/latest/install/source-build.html).
**This is currently the only Windows path that reaches native FP8 and INT4 WMMA.**
hipBLASLt exists on Windows since 6.4.2 for tuned GEMMs.

Windows vs Linux deltas (https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html):
closed-source HIP runtime; **no RCCL/comms, no MIOpen/MIGraphX**; hipInfo instead
of ROCm SMI; Radeon GPU Profiler instead of ROCProfiler; VS plugin; CMake HIP
language unsupported. Memory/IPC limits: "not all HIP runtime API functions are
yet supported on Windows"; `hipMemPrefetchAsync` "under development on Microsoft
Windows"; `hipMallocManaged` degrades to `hipMallocHost` without HMM
(https://rocmdocs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management/unified_memory_reference.html).
**No GPU P2P on Windows**: measured `hipDeviceCanAccessPeer=false` on 2× R9700 —
every cross-GPU copy host-staged; tensor-parallel LLM inference ran **2.15×
slower than a single card** while the same hardware gains from TP on Linux
(https://github.com/ROCm/rocm-systems/issues/13715). Pinned host memory
(`hipHostMalloc`) + `hipMemcpyAsync` on SDMA engines works fine for async loads;
there is no GPUDirect Storage on Windows. Deployed-binary friction is real:
llama.cpp's Windows HIP build silently falls back if a ROCm DLL dependency is
missing (https://www.yeyulingfeng.com/a/1088527.html).

## 5. Toolchain and shader-language choices

* **Vulkan SDK on Windows** (LunarG): headers, `glslangValidator`, `glslc`
  (shaderc), SPIRV-Tools, validation layers — everything needed; `vulkaninfo` for
  feature probing (https://vulkan.lunarg.com/doc/view/latest/windows/vulkaninfo.html).
* **GLSL → SPIR-V (glslang)**: reference path for coopmat
  (`GL_KHR_cooperative_matrix` + `GL_EXT_shader_explicit_arithmetic_types*` +
  `GL_KHR_memory_scope_semantics`; float8 types via `GL_EXT_float8_e5m2_e4m3`
  since glslang 15.4.0). Proven by ggml-vulkan and VulkanForge.
* **Slang**: `CoopMat` for SPIR-V merged Apr 2025 (PR #6565), coopmat2-style
  ops May 2025, and partial SM 6.10 DXIL `linalg` lowering Apr 2026 — the only
  language that gives one source for Vulkan-coopmat now and DXIL-linear-algebra
  later, plus better metaprogramming than GLSL. Strongly recommended over raw
  GLSL for a new codebase (https://github.com/shader-slang/slang/pull/6565).
* **HLSL → SPIR-V via DXC (`-spirv`)**: fine for ordinary compute, but the HLSL
  Vulkan-coopmat spec (proposal 0021) is not implemented — can't emit coopmat
  this way. Use DXC only if shaders must double-compile for D3D12.
* **SPIR-V by hand / spirv-as**: fallback for FP8 coopmat if the driver
  enumerates it but the front end balks.
* **wave32 vs wave64**: WMMA is wave32-only in hardware. A subgroup-64 coopmat
  shader gets split across the two wave32 halves — VulkanForge runs
  `requiredSubgroupSize=64` this way and reached ~14 t/s decode / ~338 t/s
  prefill (pp512) on a 14B FP8→BF16 model at ~15% FFN-GEMM efficiency; AMD's
  guidance was "test both, WMMA layouts are often built for wave64" (devcommunity
  link above). For non-WMMA memory-bound kernels (decode GEMV), wave64 is
  generally better on RDNA4 (2-lane-half issue still coalesces well; fewer waves
  = less scheduling overhead; a measured llama.cpp experiment forcing subgroup 32
  on mul_mat_vec was ~14% *worse*; zolotukhin's reference recommends wave64 for
  DMMV). In HIP use `-mwavefrontsize64`/function attribute; in Vulkan use
  `VK_EXT_subgroup_size_control` + `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`.
  Practical rule: **wave64 for GEMV/attention decode paths, wave32 (or both, measured)
  for WMMA GEMM**.
* Subgroup ops: core Vulkan 1.1 ballot/shuffle/reduce plus
  `VK_EXT_shader_subgroup_extended_types` (8/16-bit types) — all available; quad
  ops also present on AMD.

## 6. Comparison and recommendation

| API | WMMA access on gfx1201/Win11 | Measured LLM perf vs best | Async copies | Friction |
|---|---|---|---|---|
| Vulkan compute | coopmat F16/BF16/INT8; FP8 ext present, config enumeration ambiguous (RADV: yes) | decode ≈ HIP ±25% (often faster on MoE); prefill −25–40% | transfer/compute queues, ReBAR host-visible VRAM, indirect copies | none — ships with Adrenalin |
| HIP SDK 7.2 | **all dtypes incl. FP8/INT4 native** via builtins/rocWMMA | best prefill; decode ≈ parity with Vulkan on Windows | hipMemcpyAsync + SDMA on pinned mem; **no P2P, no managed mem** | SDK+runtime DLLs; closed subset of ROCm |
| D3D12 (+DML/ORT) | none in retail (SM6.10 linalg preview needs dev driver) | DML turnkey only; metacommands opaque | copy queues, upload heaps, DirectStorage (DS is D3D12-only) | high for custom kernels |
| DirectStorage 1.3 | n/a (IO API) | n/a | staged NVMe→VRAM via sysmem, GDeflate on any DX12 GPU | tied to D3D12 resources |
| Linux ROCm/RADV | full — rocWMMA all types; RADV exposes FP8 coopmat | prefill best; RADV MoE decode best measured | P2P yes, GDS no (Radeon) | dual-boot |

**Recommendation — primary: Vulkan compute on Windows.**

1. Zero deployment friction: the Adrenalin driver *is* the Vulkan runtime; no SDK
   or runtime install for end users, no DLL dependency chains (the #1 failure
   mode of Windows HIP builds).
2. Tensor access is adequate: enumerated F16/BF16/INT8 coopmat maps onto
   195/195/389 dense TFLOPS; decode is bandwidth-bound anyway (a 26.4B MoE at
   ~Q4 active-param reads sits under ~60 GB/s/token-equivalent — far below the
   640 GB/s roof), and measured decode parity between Vulkan and HIP on this
   exact card confirms the API is not the bottleneck there.
3. Portability: the same SPIR-V runs under RADV on Linux tomorrow — where Vulkan
   is *faster* than ROCm for MoE decode (+19–93% measured).
4. Structure: per-op fused shaders, Slang source, subgroup-64 for GEMV /
   coopmat-GEMM at both subgroups benchmarked, ReBAR staging ring for loads,
   runtime probing (`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`,
   `VkPhysicalDeviceShaderCorePropertiesAMD`) for every dtype claim.

**Fallback/secondary: HIP.** Build the GEMM/dequant layer behind an interface
and prototype the HIP path early — it is the only way to get *native* FP8
(2× FP16 matmul rate) and INT4 WMMA on Windows, and it wins prefill by ~25–40%
in every measured comparison. If the model ends up prefill-heavy or needs
native low-bit matrix ops, HIP is the better engine. HIP-on-Windows is now
official for gfx1201, so the dual-boot question reduces to: is Linux ROCm/RADV
worth it? **For v1, no** — the quantified gap on Windows is ≈ parity for decode
and ~25–40% on prefill, i.e. not a 2× gap that justifies a second OS. Revisit
for multi-GPU (P2P is Linux-only; measured TP regression on Windows is severe:
2.15× slower than one card) or if RADV's FP8-coopmat and MoE-decode edge becomes
decisive.

**Explicitly not recommended**: D3D12/DirectML/ONNX Runtime for the custom
kernel path (no stable WMMA; revisit when SM 6.10 linear-algebra ships retail);
DirectStorage as a reason to adopt D3D12 (its value — staged async NVMe IO —
can be replicated in ~200 lines over IOCP + Vulkan copies); Kompute or any
wrapper (write raw Vulkan).

### Known unknowns to verify at init (week-1 probes)

* FP8 coopmat enumeration on the *current* Adrenalin driver via
  `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` (RADV has it; Windows prop
  driver unclear as of mid-2026 — `VK_EXT_shader_float8` present is necessary
  but not sufficient).
* Whether a dedicated transfer queue family is exposed on the Windows driver;
  otherwise measure overlap via second compute queue.
* `VK_EXT_external_memory_host` availability for zero-copy host-mapped experts.
* Coopmat throughput wave32-vs-wave64 and F16-vs-BF16-vs-INT8 on the prop driver
  (VulkanForge sees ~15% of peak on FFN GEMM — there is headroom; ggml's fp8 PR
  shows ~2× FP16 rate when native FP8 is available).
