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
