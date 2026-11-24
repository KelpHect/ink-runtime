# ink-runtime

A C/C++ inference runtime for **[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** — Thinking Machines Lab's 26.4B-parameter multimodal MoE (`inkling2_loop`: 128 experts, top-8, looped 48-application transformer, 1M-token context, NVFP4+MXFP8 checkpoint).

Built for a single consumer machine: **RX 9070 XT (16 GB) + 48 GB RAM + NVMe**, targeting maximum decode speed and maximum context on commodity hardware.

## Design

- **Vulkan compute** primary backend (cooperative-matrix WMMA on RDNA4 — fp16/bf16/int8; fp8 where the driver enumerates it) with a HIP backend for native FP8 tensor-core paths.
- **Three-tier expert store**: VRAM slot-pool cache (~75% of all experts resident) → RAM mirror of the full expert set → repacked per-expert NVMe "expert pack" for cold misses and fast start.
- **Router-lookahead prefetch**: the next layer's router runs speculatively on the current hidden state; loop-pass reuse is exploited explicitly (the 12 core layers re-run 3× per token).
- **Tiered KV cache**: fp8-E4M3 storage / bf16 compute; sliding-window layers bounded; global layers paged with query-aware page selection for contexts beyond ~256K; radix-hash NVMe prefix store for agentic reuse.
- **Built-in speculative decoding**: the shipped DFlare block draft head (8-token block + rank-256 markov chain) verified linearly against the target — each expert fetch amortized over up to 9 positions.

## Status

Early development. `research/` contains the full investigation corpus (~15 reports: gfx1201 hardware, ROCm/Vulkan API landscape, tiered-offload literature, model internals, measured bandwidth ceilings). `docs/` has the consolidated ceilings (`THEORY.md`) and the implementation plan (`RUNTIME_PLAN.md`).

Model weights are **not** included — download `thinkingmachines/Inkling-2-Mini` from HuggingFace.

## License

Apache-2.0 — see [LICENSE](LICENSE).
