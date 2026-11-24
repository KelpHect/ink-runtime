# 06 — NVFP4 / MXFP8 formats and kernel strategy

> Status: PARTIAL — distilled from research agent A06's verified interim findings (it was paused mid-task) plus cross-checks against the local checkpoint. Follow-up needed on: Marlin FP4 kernel internals detail, Unsloth/GGUF requant quality numbers, EXL3/Q4_K comparison. See `13-inkling2-ecosystem.md` for GPTQModel Triton kernel details (the most relevant prior art).

## Verified against the local checkpoint (direct header/config inspection)

- `config.json`: `quantization_config = {quant_method: "modelopt", quant_algo: "NVFP4"}`; `hf_quant_config.json` uses `quant_method: "inkling2"` with explicit fp32/bf16 parameter groups (router biases, norms, attn_res queries stay full precision).
- Expert tensor naming is TRT-LLM-style: `w13_weight` / `w2_weight` (packed), `w13_weight_scale` / `w2_weight_scale` (per-block scales), `w13_weight_amax` / `w2_weight_amax` (F32), `w13_weight_outer_scale_denominator` (F32 global scale as a **denominator** — differs from canonical `weight_scale_2`).
- 1244 tensors: 24 layers × (w13 + w2) expert blocks; attention/norm/router in bf16/fp32.
- **Per-layer split discovered locally**: layers 0–19 experts = packed FP4 (`U8 [128,2304,1152]` w13, `U8 [128,2304,576]` w2 → logical N doubled) + `F8_E4M3` scales (`[128,2304,144]`/`[128,2304,72]`); layers 20–23 experts = MXFP8 (`F8_E4M3 [128,2304,2304]`/`[128,2304,1152]`); only layers 0–19 have `amax`/`outer_scale_denominator`.

## Format facts (verified research)

- **NVFP4**: E2M1 elements, **E4M3 scale per 16-element block**, FP32 global scale; ~4.5 effective bits/weight.
- **ModelOpt export layout**: `weight` (uint8, 2 nibbles/byte), `weight_scale` (E4M3, [out, in/16]), `weight_scale_2` (FP32), `input_scale` (FP32). This checkpoint deviates: scale layout is `[K, N/16]`-style per the shapes above and uses `outer_scale_denominator` naming — the layout must be taken from `hf_quant_config.json` + GPTQModel's docs, not assumed.
- **MXFP8**: E8M0 scale per 32-element block.

## Hardware support matrix (verified)

- RDNA4 / gfx1201 WMMA dtypes: FP16, BF16, FP8(E4M3), BF8(E5M2), INT8, INT4 — **no FP4/FP6 path**. FP4 must be dequantized (in-kernel or up-front).
- This GPU's Vulkan driver exposes `shaderFloat8CooperativeMatrix` + `shaderBFloat16CooperativeMatrix` (confirmed via vulkaninfo) → FP8/BF16 tensor-core GEMMs are reachable from Vulkan compute without ROCm.
- vLLM runs NVFP4 on non-native-FP4 GPUs via **Marlin W4A16** fallback (works back to Ampere).
- llama.cpp has `GGML_TYPE_NVFP4` with a Vulkan dequant path; its RDNA4 int8-coopmat NVFP4 path was disabled as slower than fp16 matmul (per A06).

## Strategy implications for this runtime

1. **Keep the on-disk formats** (NVFP4 0–19, MXFP8 20–23): they are already the most compact correct form; streaming experts in their native packed layout minimizes PCIe/NVMe bytes — the dominant constraint.
2. **Dequantize inside the expert GEMM** (fused): unpack E2M1 nibbles → apply E4M3 block scale → FP32 global scale → feed BF16/FP8 compute. The MXFP8 layers (20–23) can GEMM in native FP8-WMMA on RDNA4/Vulkan-coopmat.
3. Per-expert streamed bytes: FP4 layers ~4.5 MB, FP8 layers ~8 MB; group-of-8 per token ≈ 36MB (fp4 layers) / 64MB (fp8 layers).
4. Alternative disk formats (re-quantize offline once, store converted): GGUF Q4_K_M / IQ4_NL, EXL3 ~4bpw — worth evaluating only if in-kernel FP4 dequant proves slow or numerically poor; A13 found Unsloth "UD" GGUFs keep native nvfp4 blocks, mainline GGUF upcasts.
5. Quality caution: llama.cpp bug #32704 — running loop-pass experts without per-expert scales produced gibberish (PPL 130 vs 7.65); scale plumbing is correctness-critical, test greedy-equivalence early.
