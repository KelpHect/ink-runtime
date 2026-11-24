# 12 — Measured bandwidth & throughput ceilings (RX 9070 XT + i7-13700KF + FireCuda 530 + Windows)

Condensed persistence of subagent A12's report (read-only agent; full inline text in conversation log). All numbers measured unless marked spec/estimate.

## GPU-side ceilings

| Path | Spec | Measured |
|---|---|---|
| GDDR6 raw | 640–645 GB/s | 592 GB/s cl-mem read; ~455–530 GB/s effective in llama.cpp GEMV decode (~70–85%) |
| llama.cpp Vulkan tg128 | — | Llama-7B Q4_0 136.6; Qwen3-8B 121.4; Qwen3-30B-A3B **Q2_K MoE 220.4**; GLM-4.7-Flash Q4_K (16.9GiB, spills) 45.9 |
| llama.cpp HIP tg128 | — | Qwen3-8B Q4_K_M 98.8; Qwen3-14B 54.3; gpt-oss-20b Q8_0 117.2 |
| LM Studio gpt-oss-20b MXFP4 | — | Vulkan 130.8–132.8 vs ROCm 109.3–111.6 |

Vulkan usually wins decode on gfx1201 (14/17 models), HIP wins large-MoE prefill. Both had pathological build-specific regressions — benchmark the exact build.

## PCIe / host path

| Link | Spec/dir | Measured H2D pinned |
|---|---|---|
| PCIe 5.0 x16 | 63 GB/s | **54.27 GB/s (9070 XT, 3DMark)**; only 37.7 on another board — motherboard/retimer dependent, verify `LnkSta` |
| Gen5 x8 = Gen4 x16 | 31.5 | 23.1–28.4 GB/s |
| Gen4 x8 | 15.75 | 14.4 GB/s (Strata probe) |
| Pageable staging | — | ~60–70% of pinned |
| Small transfers | — | <8 GB/s below ~64–256KB; latency floor ~5–7µs |
| CPU→VRAM via ReBAR WC | — | ~10 GB/s write, ~0.1 GB/s read (never read VRAM from CPU) |

ReBAR-off cliff on RDNA3: 4.3× prefill / 2× decode loss. **Verify LnkSta + ReBAR first** — some boards wire the top slot Gen4.

## RAM

- i7-13700KF + DDR5-5600 → 89.6 GB/s spec; measured 78–105 GB/s (AIDA64), copy ~78–97.
- CPU-computed experts: llama.cpp-class GEMV delivers only **~40–65 GB/s** effective weight traffic; ik fused MoE ~13–23 GB/s per-kernel.
- Strix Halo analog: ~215 GB/s unified → ~70 t/s at ~3.3B active.

## NVMe (FireCuda 530, Phison E18, PCIe4 x4)

- Seq spec 7.3 GB/s → measured 6.1–7.46 (needs **2 threads × QD8–16 × 1–4MB** on Windows; single thread caps ~4.5–5 GB/s).
- 4K QD1 random: **83–111 MB/s** — latency-bound, useless for hot path.
- 32K random QD20: ~4.1 GB/s; expert-sized (1–4MB) reads QD8–32 ≈ 4–6.5 GB/s.
- DirectStorage: ~5 GB/s plain disk→VRAM; 14.6–17.9 GB/s with GDeflate decompression (D3D12-only).
- mmap demand-paging: 1.2 GB/s ST / ~7 GB/s MT — cold tier only.

## Composite offload datapoints

- Qwen3-30B-A3B experts-in-RAM (RTX 5070, DDR5-6400): 46–55 t/s.
- Ornith-35B-A3B on PCIe 3.0 x8 (7.9 GB/s!) Strata expert-streaming: ~112 t/s — expert traffic is small enough that 8 GB/s suffices at ~3B active.
- Qwen3.6-35B-A3B, 12GB VRAM + 48GB DDR5, ik fused CPU-MoE + MTP: **110 t/s**.
- DeepSeek-V4-Flash 80.8GiB experts NVMe→2×5060Ti: ~5.8 t/s.
- 9070 XT adding 2nd GPU to remove CPU offload: +38–183%.

## Planning ceilings for Inkling 2 Mini

| Tier | Planning bandwidth | Decoded implication (per-token bytes) |
|---|---|---|
| VRAM resident | ~500 GB/s | ~0.5–1 µs per MB |
| RAM→VRAM Gen5 x16 pinned | ~45–55 GB/s | 4.5MB expert ≈ 90µs |
| CPU-exec on RAM expert | ~40–65 GB/s | 4.5MB expert GEMV ≈ 70–110µs (≈break-even vs PCIe) |
| NVMe→RAM | ~4–6.5 GB/s batched | 4.5MB expert ≈ 0.7–1.1ms |
| NVMe 4K QD1 | ~0.1 GB/s | never on hot path |

Net: miss-bound decode floor ≈ `min(PCIe_bw, CPU_bw)/bytes_per_token`. With loop-reuse + ~75% VRAM cache + router-lookahead prefetch, realistic decode ~40–80 t/s; uncached SSD ~3–8 t/s. MTP verify amortizes each fetched expert ~2–4×.
