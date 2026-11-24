# The RTX 3090 ">2x unlock" story — what actually happened, and what transfers to an RX 9070 XT

Research note A02 · November 2026 · Inkling 2 Mini runtime project

---

## 1. TL;DR

The story the user means is almost certainly **Zagitta/gpu-unlock** (GitHub, ~21k stars, Hacker News front page late October 2026): a small Linux kernel module that removes an **instruction issue-rate cap** NVIDIA ships on every GeForce RTX 20/30/40 card. On the RTX 3090 the cap halves the throughput of tensor-core math that accumulates into FP32 registers — i.e. exactly the `FP16→FP32`, `BF16→FP32` and `TF32→FP32` MMA shapes used by unquantized LLM inference and training. Two privileged writes (`SS0 = 0x88888888`, `SS1 = 0x8`), applied before the NVIDIA driver binds the card, restore the full rate that the same silicon runs at when sold as an RTX A-series workstation or datacenter card. Measured: ~1.9–2.0× on the affected GEMM paths (e.g. RTX 4090 f16/f32: 165.6 → 330.8 TFLOPS; RTX 3090 BF16: 75 → 125 TFLOPS at stock power, ~82 → 155 at equal wattage on some boards), bit-identical outputs, **~+25–55% real-world LLM prefill, ~0% decode gain** (decode is memory-bandwidth-bound).

It is **not** the P-state / `nvidia-smi -lgc` clock-locking story (real, but worth ~10–30% in pathological cases), not ReBAR, not GSP firmware, not a memory-strap mod, and not ECC. It is a hardware segmentation fuse — a deliberate half-rate selector — bypassed via a stack-overflow ROP exploit in NVIDIA's signed SEC2 "booter" firmware, inherited from the earlier CMP mining-card unlock scene where the same mechanism produced up to **16×** speedups.

For our target (RX 9070 XT, gfx1201, Windows 11): **there is no known AMD equivalent fuse** — Radeon consumer silicon already runs FP32-accumulate WMMA at full rate (in fact FP16→FP16 is the *half*-rate path on gfx1201, the inverse of NVIDIA). The honest transferable lesson is different: on AMD, performance "locked behind" is **software coverage** (whether your kernels actually emit `v_wmma` instructions — a real 4× gap vs scalar FMA, and a demonstrated 2× regression in llama.cpp when the WMMA path was dropped), **format coverage** (FP8 WMMA is native but unreachable through Vulkan cooperative-matrix today), and **platform knobs** (power limit, memory OC, clock pinning, ReBAR, PCIe link width). Details and a checklist below.

---

## 2. Which story it is — and the lineage

### 2.1 The discovery

`github.com/Zagitta/gpu-unlock` — tagline "Your GeForce runs AI at half speed. This fixes it for 20/30/40 series cards. Up to 2× FP16 / BF16 / TF32 tensor throughput." HN thread (`news.ycombinator.com/item?id=50219700`, 353 points) hit front page ~late October 2026; Tom's Hardware, VideoCardz and Notebookcheck covered it within days. The author states plainly: *"the unlock just flips the 2 bits that halves how often the core can issue the specific instructions Nvidia locks to full rate on the professional and datacenter cards."*

Companion modules: `JustAResearcher/gpu-unlock-ada` (RTX 40 series — where the clean 2.0× measurements come from), `nlzy/turing-unlock` (RTX 20 series, different register addresses), and a merged documentation site `gpuunlock.com`. A Windows path exists as `impacifist/Windows-GPU-UNLOCK`, a UEFI application that runs pre-boot and chain-loads Windows — but it is **source-only, per-motherboard-profile, headless-focused, and validated on a handful of exact systems** (e.g. "ASUS TUF B650M BIOS 3057 + RTX 3090 FE"). Zagitta explicitly warns about random Windows installers.

### 2.2 Where it came from — the CMP mining cards

The exploit predates consumer GeForce unlocks. NVIDIA's CMP mining cards (especially the 90HX — a physically complete GA102 die, same silicon as the RTX 3080/3090, sold compute-crippled) had **all nine** SM issue-rate fields throttled by eFuse-set defaults: tensor cores ~16× down, FP32 FMA ~8× down, DP4A ~16× down, plus scheduler restrictions. The `arabel1a/ml-on-cmp` repo documented per-instruction issue rates (every `mma.sync` shape pinned to 285 ns/op on the 90HX). Community unlocks (`WildFlash1st/cmp90hx-unlock-for-windows`, `xrip/cmp50hx-unlock`, the "CMP40HX-Unlock" UEFI app) restored full throughput — llama-bench prefill 224 → 1824 t/s (~8×) on one verified setup, ~230 → ~3705 t/s on a 7B Q4_0 in another. Zagitta credits the CMP unlocks as the direct inspiration ("my repo is all based on the CMP90/50 unlocks"); the GA102 six-write gadget chain is credited in source to **Petri Krohn**.

So when the user says "more than double," they may be conflating the mainstream 3090 story (2× on the fused path) with the CMP scene (8–16× end-to-end, because *everything* was throttled). Same mechanism, different fuse defaults.

### 2.3 The exact mechanism

Two MMIO registers govern per-opcode issue rates on GA102-class chips:

| Register | Address (Ampere/Ada) | Address (Turing) | Contents |
|---|---|---|---|
| `FEAT_OVR_SM_SPD` ("SS0") | `0x0082381c` | `0x00409664` | eight 4-bit fields: DP, FFMA, FMLA16, **FMLA32**, IMLA0–3 |
| `FEAT_OVR_SM_SPD_1` ("SS1") | `0x00823820` | `0x0040966c` | one 4-bit field (IMLA4) |

Each field: `bit3` = override enable, `bits2:0` = speed selector (`00` full, `01` ½, `10` ¼, `11` ⅛). GeForce cards boot with `SS0 = 0x00000001` — the FMLA32 field set to half rate. The fix is `SS0 = 0x88888888`, `SS1 = 0x8` (override + full rate on all fields).

The registers are gated by a **Privilege Level Mask (PLM)**: host-driver writes are silently ignored; only "heavy-secure" mode code can open them. The exploit path:

1. The driver uploads signed "booter" firmware to the GPU's SEC2 security processor to authenticate/decrypt the GSP firmware. That booter has a **stack buffer overflow**.
2. Researchers decrypted a *debug* booter blob NVIDIA accidentally published, mined ROP gadgets, and built the **"V67" payload** — a gadget chain that executes in heavy-secure mode and performs `MEMORY[addr] = value` writes.
3. V67 opens WPR/PLM masks → host writes SS0/SS1 → **PCIe Function Level Reset** clears the latched protection registers *while the speed selectors survive* → the unmodified stock driver then probes the card and finds `FMLA32 = full`.
4. Volatility: the selectors are **not** e-fuses and not VBIOS; they're boot-time-latched registers. They revert on cold power-off / full re-POST and must be re-applied before every driver bind (hence initramfs loading and a suspend/resume re-unlock path; laptops that power-gate the dGPU silently lose it).

What's capped vs not (measured on RTX 4090, gpuunlock.com):

| Path | Stock | Unlocked | Ratio |
|---|---|---|---|
| f16×f16→f16 accum | 325.7 TF | 331.6 TF | 1.02× (never capped) |
| **f16→f32** | 165.6 TF | 330.8 TF | **1.998×** |
| **bf16→f32** | 165.5 TF | 330.3 TF | **1.996×** |
| **tf32→f32** | (half rate) | doubled | ~2× |
| int8→int32 | 661.6 TOP | 659.5 TOP | 1.00× (never capped) |
| int4, dp4a, fp8 e4m3, fp32 FFMA | — | — | ~1.00× (never capped) |

Note the asymmetry: on GeForce, **FP32-accumulate is the taxed path** — the precision-preserving accumulate every serious inference kernel uses. FP16-accumulate ran full rate all along. That is textbook product segmentation (GeForce vs RTX A6000, same GA102).

### 2.4 Real-world effect — why "2×" becomes "+15–55%"

- Prefill/prompt processing in FP16/BF16 and AWQ/GPTQ inference: **+25–55%** (vLLM, HF). TTFT −25–30%.
- FlashAttention/SDPA kernels: +50–60% (they're dense f16→f32 MMA).
- Fine-tuning in BF16: +25–50%. Image gen (ComfyUI, BF16 DiT): +30–55%.
- **llama.cpp with quantized GGUF: 0–15%**, growing with context — because the weight GEMMs run on INT8 MMQ kernels (never capped); only attention benefits.
- **Decode/token generation: ~0%** — bandwidth-bound; unlocking math rate does nothing when the card waits on GDDR6X.
- Power interplay: doubling issue rate doubles dynamic power in the tensor units, so at a fixed wattage the card trades frequency for rate. Zagitta: "it shifts the card down on the frequency curve to keep the wattage in check" — gains grow with power-limit headroom (3090 Ti at 450 W: +80% BF16).
- Outputs verified **bit-identical** (one reporter: 1996-position logprob dump byte-for-byte equal). It's a rate change, not a numerical change.
- Verification: the driver-visible state reports `FMLA32 = 1/2 → full` (CMP check scripts dump all nine fields), plus microbenchmark A/B (cuBLAS 8192³ GEMM medians).

### 2.5 Ruling out the other candidate stories

| Candidate | Verdict |
|---|---|
| (b) P-state / `nvidia-smi -lgc`, `-lmc` | Real but separate. GeForces can downclock mem/core between bursty kernels (the "prefer maximum performance" fix); NVIDIA's own NIM-on-WSL2 docs prescribe clock locking for benchmarking. Typical effect: tens of %, occasionally large on idle-clock-ramp pathologies — not the viral 2× story. |
| (c) GSP vs non-GSP firmware | Real tuning axis on Linux (`NVreg_EnableGpuFirmware`), small single-digit effects; unrelated. |
| (d) **Instruction-rate unlock** | ✅ **This is it** (FMLA32 speed-select). |
| (e) ReBAR / BAR1 | Affects host↔device transfer path, not math rate; ~0 for resident-weight decode. |
| (f) WMMA dual-issue | No such NVIDIA discovery; closest real thing is on the AMD side (below). |
| (g) GDDR6X strap/OC unlocks | No strap tooling exists on 30-series; thermal-pad replacements fixed mining-era memory throttling (2021 story, different). |
| (h) CUDA MPS / SM partitioning | Unrelated; GeForce has no MIG. |
| (i) ECC disable / async copy | GeForce has no ECC to disable; not a throughput unlock. |
| (bonus) LHR mining-hashrate unlock (2022) | Real historical 2×-for-ETH story; possible source of user confusion, but it's old and mining-specific. |

---

## 3. Mapping to the RX 9070 XT (gfx1201) — what transfers, what doesn't

### 3.1 The direct analog: does AMD fuse consumer math rates?

**No evidence it does — and the published evidence points the other way.** On Navi 48 / gfx1201, dense WMMA rates are 1024 FLOP/clk/CU for FP16→FP32 and BF16→FP32 — double RDNA3's 512 — and these ship *on the consumer card*. Third-party spec aggregation even shows the relationship **inverted vs NVIDIA**: on the 9070 XT, `FP16→FP16` matrix = 97.3 TFLOPS while `FP16→FP32` = 194.6 TFLOPS — i.e., the wide-accumulate path is the *full*-rate one. `FP8→FP32` similarly = 389 TFLOPS dense vs `FP8→FP16` 194.6. AMD segments Radeon vs Radeon PRO/Instinct by VRAM size, ECC, certifications, FP64 (fused in hardware at 1:64, not unlockable) and price — not by an issue-rate selector on consumer MMA. The Radeon AI PRO R9700 shares the same Navi 48 die and the same per-clock WMMA rates.

Structurally, AMD *does* have "locked" things — SMU feature masks (`ppfeaturemask`), PowerPlay table limits, and historically fused-off CUs (the old Vega56→64 / RX 460 shader-count BIOS unlocks). But none is a 2× math-rate fuse, and nothing in the SS0/SS1 exploit class (SEC2 booter ROP → PLM → selector registers) has an AMD counterpart anyone has published. AMD's boot chain (PSP → SMU signed firmware) is a different design; don't expect a "flip two bits" moment on Radeon.

**Bottom line: the *specific* mechanism is NVIDIA-only. The *transferable* parts are the categories of lock it exposed:**

### 3.2 What actually transfers — ranked by expected gain for an Inkling-class MoE runtime

**(A) "The card does 4× the math your kernel asks for" — WMMA coverage is the real gfx1201 unlock.** On the 9070 XT, scalar FP32 vector FMA peaks at 48.7 TFLOPS; `v_wmma_f32_16x16x16_*` (wave32, gfx12-suffixed builtins: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`, `_bf16`, `_fp8_fp8`, `_bf8_bf8`, int8/int4 variants) peaks at **194.6 TFLOPS dense FP16/BF16→FP32 — a hard 4×** — and 389 TFLOPS for FP8/INT8. Any matmul written against the vector pipe is leaving 4× behind, no fuse required. This is not hypothetical: llama.cpp issue #26220 measured **up to 2× slower prompt processing at long context** on gfx1201 when the rocWMMA FlashAttention path was replaced by a native kernel that didn't use tensor cores properly (777 → 395 t/s at 127k ctx on an R9700); PR #26419 recovered most of it by gating on head-dim and bypassing LDS for K/V. **Design rule: every GEMM and every attention inner loop in our runtime must go through WMMA — verify at the ISA level (`v_wmma` in SASS/disassembly), not by trusting the library.**

Caveats: RDNA4 WMMA requires **wave32** mode and uses a *different fragment/VGPR layout than RDNA3* — no binary compatibility; compile with `--offload-arch=gfx1201` (not gfx1200-generic) and use the `_gfx12` builtins. WMMA is also per-`16×16×16`-tile; K-dim and LDS bank-conflict handling dominate efficiency (the VulkanForge engine got +113% prefill just from BN=16→32 tiling).

**(B) Format coverage = the second "locked" multiplier.** gfx1201 natively does FP8 E4M3/E5M2 WMMA at 389 TFLOPS dense and INT4 at 779 TOPS — but *software* gates it: Vulkan `VK_KHR_cooperative_matrix` today exposes no FP8 component types (VulkanForge had to upconvert FP8→BF16, halving the available rate), and rocWMMA/hipBLASLt FP8 paths on Windows HIP SDK are still maturing. For our NVFP4+MXFP8 checkpoint specifically: **there is no FP4 matrix instruction on RDNA4** — NVFP4 (e2m1 + fp8 scale) must be dequantized (to bf16 for WMMA, or via the INT8/DP4A-style paths), while MXFP8 E4M3 activations/weights can feed native FP8 WMMA **only through HIP/CK**, not through Vulkan coopmat today. Choosing the wrong API silently costs 2× on FP8-eligible layers — the same shape of loss as the NVIDIA fuse, different cause. Also note the **4:2 structured-sparsity** variant doubles all WMMA rates again (389→779) but requires restructuring weights; incompatible with our dense quantized checkpoint, listed for completeness.

**(C) Power/clock knobs — the small-but-real 5–15% tier.** Analog of NVIDIA's clock-locking/power-headroom effects:

- **Adrenalin (Windows, supported):** power limit slider +10% (304 → ~334 W; some AIB cards ship 330 W → 360 W), negative voltage offset −50…−150 mV typical-stable, max-frequency offset, memory clock 2500 → ~2750–2832 MHz (20 → ~22–22.7 Gbps ≈ **+10–13% bandwidth** — directly helps decode). Community data shows undervolted 9070 XTs holding ~3.0–3.4 GHz at 210–300 W; memory OC hits a practical wall (GDDR6 error-detection/replay + ~90 °C mem temps).
- **MoreClockTool (MCT):** session-only PPT limit raise — measured up to **340 W on the 9070 XT** — without flashing.
- **MorePowerTool / SoftPowerPlayTables: dead end on RDNA3/RDNA4.** AMD locked SPPT registry injection behind driver validation in the Adrenalin-2020 era; MPT covers only up to RDNA2. RDNA4 alternatives: R.B.R.T. **RebelsTool + Adrenalift** (Linux side / bootable USB) and `sibradzic/upp` which parses the RDNA4 PP-table v23.0 extracted from VBIOS (the sysfs pp_table read is truncated to 4095 B by a kernel bug — extract from the ROM instead).
- **Linux equivalents of `-lgc`:** `amdgpu.ppfeaturemask=0xfffd7fff` (or `0xffffffff`) enables OverDrive → `pp_od_clk_voltage`; `echo high > power_dpm_force_performance_level` or `profile_peak` pins clocks — worth doing for benchmarks and for bursty MoE expert execution where DPM ramp latency is real.

**(D) The memory/transfer plane — where decode actually lives.** Decode on a 26.4 B MoE with ~3B active is bandwidth-dominated: 640 GB/s on the 9070 XT is the ceiling and everything that touches the bus matters:

- **ReBAR / SAM: mandatory, not optional, for an offload design.** Without it the CPU sees a 256 MB aperture; host-visible-VRAM allocations (which llama.cpp's Vulkan backend requests by default — `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM` exists precisely as the workaround) degrade badly; AMD's own R9700 guide notes P2P is *deactivated* without Re-Size BAR. Enable Above-4G-Decoding + Re-Size BAR in BIOS and verify (Windows: Device Manager shows large memory range / `Vulkan` reports `rebar` memory type; Linux: `dmesg | grep BAR` / `lspci -vv` Resizable BAR line).
- **PCIe link audit:** the card is Gen5 x16; verify actual trained width/speed (AMD `lspci` LnkSta / HWiNFO bus interface) — a card in a chipset-attached x4 or Gen3 slot halves/quarters offload staging bandwidth.
- **NVMe lane plan:** with ~16 GB of model resident and the rest streaming from the 48 GB RAM/NVMe pool, keep the system NVMe on dedicated CPU lanes (AM5: the CPU-attached M.2) so expert streaming doesn't contend with GPU BAR traffic on the same root-complex path.
- **Memory timing straps:** unlike Polaris/Vega, there is no practical GDDR6 "fast timings" unlock on RDNA4 — the PP table's timing section isn't user-editable through any current tool; the effective memory lever is frequency OC (above).

**(E) Software-path selection on Windows 11 — the hidden Vulkan/HIP analog of the story.** Options for emitting matrix ops on gfx1201, ranked: (1) **HIP** (ROCm/HIP SDK ≥6.4 w/ gfx1201, `amdclang++ --offload-arch=gfx1201`, rocWMMA/hipBLASLt/CK) — the only route to native FP8 WMMA and the fullest control; (2) **Vulkan `VK_KHR_cooperative_matrix`** — cross-vendor, works today for FP16/BF16/INT8 coopmat, no FP8 types yet, Wave64-vs-wave32 subtleties (one engine uses `requiredSubgroupSize=64` on the same silicon; on HIP, WMMA builtins require wave32 — validate per-path); (3) **D3D12 Wave Matrix** — previewed in SM 6.8, *deprecated in favour of a unified `linalg` matrix type targeted at SM 6.10* — not yet a stable target; (4) wgpu/Dawn coopmat — inherits Vulkan's limits. Also verify `HSA_OVERRIDE_GFX_VERSION` isn't masking a missing native-target issue if borrowing gfx110x binaries — RDNA4 WMMA fragments differ so wrong-target binaries can silently fall back.

**(F) What does NOT transfer (be honest):** no SS0/SS1-style rate fuse is known on AMD; no equivalent of the SEC2/V67 exploit is published for PSP/SMU; no "2× hidden tensor rate" to flip. If a claim of an "RDNA4 pro-rate unlock" surfaces, treat it like the CMP scene did — demand per-instruction issue-rate measurements (ns/op microbenchmarks à la arabel1a) and bit-identical output checks, not vibes.

### 3.3 One more genuinely useful idea the story hands us

The unlock scene's *methodology* transfers even where its mechanism doesn't:

- **Measure per-path issue rates directly.** They didn't trust spec sheets — they timed `mma.sync`/`FMA`/`DP4A` shapes and found every throttled field. Do the same on gfx1201: microbenchmark `v_wmma_f32_16x16x16_{f16,bf16,fp8_fp8}`, `v_dot4`, packed-fp16 FMA, scalar FMA — build a per-instruction rate table for Navi 48 before trusting any "195 TFLOPS" marketing number (AMD's own GPUOpen table gives FLOP/clk/CU = 1024/2048/4096 for 16-bit/8-bit/4-bit dense; verify achievable ~85–90% at boost).
- **Amdahl discipline.** 2× on the fused path ⇒ +25–55% prefill, 0% decode. For our MoE: estimate the *fraction of wall-clock each layer type occupies* (expert GEMMs vs attention vs quant/dequant vs host transfers) before optimizing; decode gains will come from bandwidth/latency work, not math rate.
- **Bit-identical output as the regression test.** The community verified the unlock changed speed only. Same standard for our kernel work: fixed-seed logprob dumps before/after any WMMA/sparsity/dequant change.
- **The "check the fuse" pattern generalizes:** AMD's analog diagnostic surface is `ppfeaturemask` (what SMU features are enabled), `power_dpm_force_performance_level`, VBIOS PP-table contents (`upp`), PCIe link state, and BAR size — audit all of them at runtime init, like gpu-unlock's `check.sh` dumps `FMLA32` state.

---

## 4. Checklist for the Inkling runtime on RX 9070 XT / Windows 11

1. BIOS: Above-4G Decoding + Re-Size BAR **on**; verify full 16 GB BAR mapped.
2. Verify PCIe link trains Gen4/Gen5 x16 (physical slot + lanes, not chipset x4).
3. Adrenalin: PL +10%, UV ≈ −100 mV (validate), memory OC toward ~2750 MHz if temps allow — decode scales with this almost linearly.
4. Optional: MCT session PPT → 340 W for sustained prefill; Linux side: `ppfeaturemask` + `power_dpm_force_performance_level=high`.
5. Kernels: all GEMM/attention through gfx1201 WMMA builtins (wave32); disassemble to confirm `v_wmma`; target ≥85% of the 194.6 TF FP16-32 rate on large tiles.
6. FP8 (MXFP8) path via HIP/CK; NVFP4 dequant to bf16 at WMMA edge, or INT8 emulation — benchmark both.
7. Attention: use a WMMA FA kernel with LDS-bypass for deep context (llama.cpp #26220/#26419 as the cautionary tale — the "2×" regression was exactly this).
8. Host↔device: pinned host-visible VRAM staging over ReBAR for expert streaming; dedicated CPU-lane NVMe; overlap via SDMA/async queues.
9. Runtime self-audit at init: BAR size, link width/speed, perf level, mem clock, power limit — print them like `FMLA32` state.
10. Re-audit numerics (bit-compare vs reference) after every rate-path change.

---

## 5. Sources

Primary:
- https://github.com/Zagitta/gpu-unlock — module README, supported cards, per-card BF16 table, FAQ (Windows/Blackwell/why)
- https://github.com/Zagitta/gpu-unlock/blob/main/docs/How%20Ampere%20was%20unlocked.md — SEC2 booter ROP, V67 payload, SS0/SS1 register table and field encoding
- https://github.com/zagitta/gpu-unlock/blob/HEAD/kernel/gpu_unlock/recipes.c — per-family recipes; Petri Krohn GA102 six-write chain comment; Turing aperture addresses
- https://gpuunlock.com/llms.txt + https://gpu-unlock.com/en/faq + /en/windows — measured rows (4090 1.998×, 4070 Ti S 1.99×, 2080 Ti 1.36–1.64×), never-capped paths, load-order requirement
- https://news.ycombinator.com/item?id=50219700 — HN discussion; Zagitta on mechanism/power trade-off; modeless on FP8 accumulate caveat
- https://www.tomshardware.com/pc-components/gpus/nvidia-rtx-20-30-and-40-series-gpus-hacked-to-remove-a-half-speed-ai-throttle-firmware-exploit-doubles-rtx-4090s-peak-tensor-rate-lifts-rtx-3090-ti-matrix-math-up-to-80-percent — press summary + per-generation table
- https://www.notebookcheck.net/Nvidia-RTX-3090-performance-nearly-doubles-with-free-GPU-unlock-but-there-s-a-catch.1432805.0.html — 77.5→129.6 TF BF16 report; ComfyUI +47–59%
- https://doug.sh/posts/gpu-unlock-rtx-3090/ — 2×3090 A/B: +13–20% vLLM prefill, no decode gain
- https://github.com/Zagitta/gpu-unlock/issues/158 — EVGA 3090 pair: FP16/BF16 +72–83%, llama.cpp +3–17% prefill scaling with ctx
- https://github.com/Zagitta/gpu-unlock/issues/75 — 4×3090 power-cap analysis: decode flat (~936 GB/s bound), +9% prefill
- https://xitter.cf/superalesha/status/2113332444283691105 — 4×3090: BF16 82→155 TF same 300 W; MoE prefill +10–12%, byte-identical logprobs
- https://www.thesiliconreport.com/articles/architectural-teardown-zagitta-gpu-unlock-doubles-rtx-3090-tensor-core-issue-rates — FMLA32=1/2 driver-visible state narrative
- https://gpuunlock.org/how-it-works/ — plain-language mechanism recap
- https://note.com/genelab_999/n/na4defa988b67 — independent verification write-up (JP)

CMP lineage / deeper mechanism:
- https://github.com/WildFlash1st/cmp90hx-unlock-for-windows — EFI unlock flow (WPR2→V67→PLM→SS0/SS1→FLR); 230→3705 t/s
- https://github.com/WildFlash1st/cmp90hx-unlock — kernel-side variant; 224→1824 t/s; check.sh field dump format
- https://github.com/arabel1a/ml-on-cmp — per-instruction throttling measurements (mma 285 ns/op, FMA 8×, DP4A 16×, DP2A escape)
- https://github.com/xrip/cmp50hx-unlock — V67 UEFI port; A/B vs stock driver
- https://cmp.wiki/wiki/Other_CMP_cards_and_other_NVIDIA_GPUs — genealogy: Zagitta on CMP-derived origin
- https://github.com/impacifist/Windows-GPU-UNLOCK (+ docs/COMPATIBILITY.en.md, INTERNAL-BOOT.en.md, evidence/headless-handoff) — Windows UEFI port: verified 3090 FE build, FP16 125 / BF16 129 TF on Windows, per-board profiles only
- https://github.com/MindAllik/GPU-Tensor-Unlock-Universal — derivative "universal" USB tool (unverified; use caution)

P-state/clock-lock (ruled-out story):
- https://docs.nvidia.com/nim/wsl2/latest/benchmarking.html — NVIDIA's own `-lgc`/`-lmc` benchmarking guidance
- https://docs.nvidia.com/deploy/nvidia-smi/ — `-lgc` semantics (Volta+)
- https://github.com/ikawrakow/ik_llama.cpp/discussions/1415 — 3090 OC/TDP interplay in real llama inference
- https://stackoverflow.com/questions/30663524 — P0/P2 explanation (compute P-state ≠ power intent)

AMD side:
- https://gpuopen.com/learn/using_matrix_core_amd_rdna4/ — RDNA4 vs RDNA3/RDNA2 WMMA FLOP/clk/CU (1024 dense fp16/bf16), new fragment layout
- https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/ — RDNA4 2×16-bit / 4×8-bit WMMA per CU, FP8/BF8, 4:2 sparsity, transpose loads
- https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html — `_gfx12` builtins, wave32 requirement
- https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp16-gemm/README.html — stepwise FP16 WMMA GEMM vs hipBLASLt on 9070 XT
- https://gpucompat.com/gpus/amd-radeon-rx-9070-xt/ + https://hmc-tech.com/gpus/amd-radeon-rx-9070-vs-amd-radeon-rx-9070-xt — spec cross-check incl. the f16-accum half-rate inversion
- https://github.com/ggml-org/llama.cpp/issues/26220 + PR https://github.com/ggml-org/llama.cpp/pull/26419 + PRs #17156, #25940 — the real 2×-class gfx1201 "unlock": WMMA-path coverage
- https://www.igorslab.de/en/moreclocktool-new-ppt-limit-radeon-graphics-cards/ — MCT session PPT (340 W on 9070 XT)
- https://www.igorslab.de/en/red-bios-rebellion-team/4/ + https://cryptolexicon.org/en/blog/moreclocktool-mct-amd-gpu-tuning-for-mining/ — MPT/MCT/RBE support matrix: MPT dead on RDNA3/4; RebelsTool/Adrenalift path
- https://github.com/sibradzic/upp/issues/54 — RDNA4 PP-table v23.0 parsing via VBIOS extract
- https://docs.kernel.org/gpu/amdgpu/thermal.html + kernel.org amdgpu module parameters + https://rocmdocs.amd.com/projects/amdsmi/en/latest/conceptual/perf-determinism.html — `power_dpm_force_performance_level`, `ppfeaturemask`, perf determinism
- https://quasarzone.com/bbs/qf_vga/views/6494954 + https://hardforum.com/threads/anyone-overclocking-their-9070-xt.2040235/ + https://whysogeek.com/undervolt-overclock-rx-9070-xt-adrenalin-2026/ — community UV/OC data (−50…−150 mV, mem 2750–2832 MHz, +10% PL)
- https://blogs.damiendye.uk/en/proxmox/pcie-resizable-bar/ + https://bitworks.io/amd-inference/system-setup/ + AMD R9700 getting-started PDF + https://rocmdocs.amd.com/en/latest/reference/system-optimization/bar-access-limits.html — ReBAR mechanics and inference relevance
- https://devcommunity.amd.com/t/rdna4-vulkan-compute-fp8-wmma-questions-from-an-llm-inference-engine-dev/503 — Vulkan coopmat on gfx1201: works for f16/bf16, no FP8 component types yet
- https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_x_WaveMatrix.html + SM 6.9 retail blog + hlsl-specs proposal 0035 — D3D12 matrix path status (SM 6.10 `linalg` pending)
