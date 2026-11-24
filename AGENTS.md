# ink-runtime — agent notes

C/C++20 inference runtime for Inkling-2-Mini (arch `inkling2_loop`), targeting
RX 9070 XT (16 GB) + 48 GB RAM + NVMe on Windows. Read `docs/RUNTIME_PLAN.md`
(phases) and `docs/THEORY.md` (verified ceilings) before writing code. Model
internals spec: `research/07-inkling2-loop-internals.md` — follow it op-for-op.

## Verified hardware facts (this box)

- PCIe **Gen5 x16** at max link; **ReBAR on** (all 15.92 GiB VRAM host-visible);
  dedicated TRANSFER queue + 8 compute queues on the AMD prop driver.
- `VK_KHR_cooperative_matrix` rev2 + fp8/bf16 features present; per-dtype
  config list = run `vkprobe.exe` once built.
- RAM is **DDR4-2666** (mismatched kits) ~40 GB/s — PCIe streaming beats
  CPU-side expert compute; don't assume DDR5 bandwidth.
- Vulkan API 1.4.359, driver 2.0.416. `VK_EXT_external_memory_host` present.

## Build / test

Toolchain: MSVC (VS Build Tools) or clang + CMake + Vulkan SDK installed via
winget (status: `docs/toolchain-status.md`).

```sh
cmake -B build -S . -DINK_WITH_VULKAN=ON   # when toolchain is up
cmake --build build --config Release
build\Release\ink.exe <model_dir> --prompt "..." --tokens 8
```

Golden correctness gate: `testdata/golden/` (from `tools/dump_golden.py`) —
bit-compare our logits/hidden taps vs HF reference for every kernel change.
Expert pack: `tools/make_expertpack.py` → `data/expertpack.v1.bin`.

## Conventions

- Fixed-shape static graph, no dynamic DAG (see `docs/RUNTIME_PLAN.md` §design).
- fp32 where research/07 §12 says fp32 — precision bugs are the #1 risk class
  (llama.cpp #32704 shipped gibberish from a scale-plumbing error).
- Weights/venv/vendor/data dirs are git-ignored; never commit binaries.
- `vendor/Strata/` is Apache-licensed reference code — study, don't copy wholesale.
- Git: commit only files you changed; identity is the repo's configured user.

## Vulkan backend gotchas (learned the hard way)
- `ink::vk::Buffer` is a dedicated VkBuffer already bound at its slab offset —
  descriptor ranges are BUFFER-RELATIVE: write_storage(buf, 0, size). Passing
  the slab offset shifts the range past the buffer end -> OOB shader reads ->
  page fault -> Adrenalin TDR / VK_ERROR_DEVICE_LOST. (2026-11-24 incident)
- Descriptor contents are sampled at EXECUTE time: dispatches in the same
  command buffer needing different bindings must use separate DescSets.
- Every recorded copy/dispatch sees buffer contents at execute time: never
  record a copy then reuse the staging region inside the same cb.
- Expert tensors live in HOST_VISIBLE RAM buffers (host_cache_/host_buf) —
  GPU streams only selected rows over PCIe; do NOT upload experts to VRAM
  (vk backend rule). In `gpu_graph` the opposite is preferred: experts go
  device-resident while VK_EXT_memory_budget reports headroom (host fallback
  per-tensor) because...
- MEASURED (RX 9070 XT, Adrenalin): compute-shader reads of HOST_VISIBLE
  sysmem cap ~8GB/s aggregate; SDMA vkCmdCopyBuffer RAM->VRAM ~32.5GB/s;
  host-visible pool ~11GB effective, device heap 15.9GB minus driver reserve.
- `*_mslot.comp` shaders: slot = gl_WorkGroupID.y — merge all top-k expert
  dispatches into one (per-dispatch cost dominated); moe_reduce folds the
  per-slot rows.
- `ink --backend vk|gpu --dump dir` + `python tools/compare_golden.py gold ink_dir`
  is the correctness gate (argmax must match 8/8; hidden cos >= 0.98).
  WATCH OUT: the comparators ignored argv once — always pass real paths and
  sanity-check one file argmax manually when a PASS looks suspicious.
