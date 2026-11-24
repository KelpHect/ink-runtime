# Research index — ink-runtime

| File | Topic | Status |
|---|---|---|
| 00-model-facts.md | Verified checkpoint architecture (config + safetensors census) | done |
| 01-rx9070xt-hardware.md | gfx1201 specs, WMMA, ISA changes, tuning/unlock knobs, measured LLM perf | done |
| 02-rtx3090-unlock-and-amd-analogs.md | Zagitta/gpu-unlock story decoded; what transfers to AMD | done |
| 03-rocm-gfx1201.md | ROCm/TheRock state, hipBLASLt/CK/aiter coverage, Win/WSL/Linux verdict | done |
| 04-windows-amd-gpu-apis.md | Vulkan/D3D12/DML/HIP comparison + recommendation | done |
| 05-nvme-tiered-runtimes.md | llama.cpp/ik/Strata/kTransformers/etc — tiered-offload survey | done |
| 06-nvfp4-mxfp8-formats.md | Format layouts, kernel strategy (PARTIAL — Marlin internals + GGUF quality open) | partial |
| 07-inkling2-loop-internals.md | Full op-level internals (attn_res blocks, loop mechanism, per-pass KV confirmed, no-RoPE rel-bias, sconv, MoE, MTP wiring, tokenizer) | done |
| 08-mtp-spec-decode.md | Draft head decoded (DFlare + GRU markov), verify design, acceptance data | done |
| 09-kv-long-context.md | KV budgets (8 global positions!), fp8 plan, tiering, sparse prefill | done |
| 10-nvme-windows-io.md | Windows NVMe paths: IOCP threads, DirectStorage, mmap limits, pinned H2D | done |
| 11-moe-expert-streaming.md | Prediction/caching lit, loop-reuse exploitation, k1/k2 renorm knob | done |
| 12-measured-bandwidth.md | All measured ceilings consolidated (VRAM/PCIe/RAM/NVMe) | done |
| 13-inkling2-ecosystem.md | HF/vLLM/SGLang/llama.cpp/GGUF support state + prior art | done |
| 14-cpp-runtime-architecture.md | Structural patterns to crib (ggml, fastllm, vLLM, SGLang) | done |

Synthesis: `../docs/THEORY.md` (ceilings) + `../docs/RUNTIME_PLAN.md` (phases).
Reference impl vendored: `../vendor/Strata` (Apache-2.0? — verify license before cribbing).
Scratch file `st_header.json` in repo root — safe to delete (msg noted).
