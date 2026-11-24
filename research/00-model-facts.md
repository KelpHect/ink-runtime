# 00 — Grounded facts: Inkling 2 Mini checkpoint + this machine

Everything in this file was verified locally on 2026-11-24 (config files, safetensors header census, `vulkaninfo`, WMI). No web claims here — each agent report `NN-*.md` holds the literature.

## Model card summary (huggingface.co/thinkingmachines/Inkling-2-Mini)

- Provider: Thinking Machines Lab. Released **2026-11-20**. License **Apache-2.0**.
- Multimodal (text + image in, text out). 26.4B total / 3.4B active → **5.4B effective active** (looped).
- Checkpoint: single `model-00001-of-00001.safetensors`, **19,525,870,752 B (≈19.5 GB)**, mixed NVFP4 + MXFP8 (modelopt), plus `mtp.safetensors` (635,801,416 B — built-in draft head) + `bf16`/`f32` small tensors.
- Supported stacks claimed: HF Transformers, SGLang, vLLM, TokenSpeed, NVIDIA Dynamo, Unsloth (GGUF), llama.cpp (GGUF), Llama App; Ollama/Exo on Apple. Post-training: Miles, SkyRL.
- Context: **1,048,576 tokens** (`model_max_length`).
- Eval claim highlights (their numbers): SWEBench Verified 79.2%, BrowseComp 79%, Terminal-Bench v2.1 71.6%.

## config.json (text_config)

| field | value |
|---|---|
| model_type | `inkling2_loop` (`Inkling2LoopForConditionalGeneration`) |
| hidden_size | 2304 |
| num_hidden_layers | 24 |
| vocab_size | 201024 (unpadded 200058) |
| attention heads / KV heads / head_dim | 24 / 8 / 128 (GQA, ratio 3) |
| `d_rel` / `rel_extent` | 16 / 1024 — separate relative-position feature pathway |
| log_scaling | `n_floor 8192`, `alpha 0.1` — long-context attention scaling |
| `use_value_norm_no_gamma` | true — value-residual norm w/o learned γ |
| `qk_l2_clip_threshold` / `rel_l2_clip_threshold` | 22.6274 / 8.0 — L2 clipping |
| global attention layers | **{5, 11, 17, 23}** (every 6th) — all others SWA-1024 |
| `sliding_window_size` / `swa_*` | 1024, head_dim 128, 24/8 heads |
| `sconv_kernel_size` | 4 — depthwise short conv |
| `n_layers_per_attn_res_block` | 6; `attn_res_normalize_values` true |
| MoE | 128 routed experts, top-8, **0 shared**, intermediate 1152, `mlp_gate_softcap` 4.0, `mlp_up_softcap` 25.0, `route_scale` 8.0, `use_gate_bias` true |
| `loop_config` | core layers 6–17, **num_passes 3**, `gate_rank` 64, `is_varloop` false |
| `final_logit_softcapping` | 30.0 |
| `rms_norm_eps` | 1e-6 |

## mtp_config (draft head)

- 3 layers, hidden 2304, 24/8 heads, head_dim 128, intermediate 9216.
- `block_size` 8, `markov_rank` 256, `markov_head_type` `rnn_hidden`.
- `target_layer_ids` {5, 23, 41, 48} — reads hidden states at those *layer applications* (24 layers × loop passes → 48 applications).
- local_layer_ids {0, 2} + sliding_window 1024; rope_theta 1e6.

## Vision config

- `inkling2_vision`: 23-layer ViT, hidden 1024, 16 heads, mlp_ratio 4, `native_patch_size` 20, `temporal_patch_size` 2 (video-aware), `posemb_grid_size` 32, adapter `merge_size` 2, `projection_hidden_dim` 4096 → text hidden 2304. `proj_in` 33MB + `proj_out` 20MB + backbone 0.58GB.

## Safetensors census (header parsed 2026-11-24)

1244 tensors (index reports 1245 w/ metadata). dtype totals:

| dtype | tensors | bytes | role |
|---|---|---|---|
| U8 | 48 | **10.32 GB** | packed NVFP4 (2×E2M1/byte): `w13_weight`,`w2_weight` for layers **0–19** |
| F8_E4M3 | 48 | **5.35 GB** | layers **20–23** expert weights (MXFP8, [128,2304,2304]/[128,2304,1152]) + all 24 layers' `w13/w2_weight_scale` |
| BF16 | 994 | **3.85 GB** | all attention/dense weights, embed+unembed (926MB ea), vision |
| F32 | 154 | ~0.01 GB | `gate.bias`, `*_amax`, `*_outer_scale_denominator`, `attn_res` queries, `middle_cycle_cross_pass*` |

Per-expert bytes: FP4 layers ≈ **4.5 MB/expert** (w13 2.65MB + w2 1.33MB + scales ~0.5MB); FP8 layers ≈ ~8 MB/expert + scales.
Per-token expert traffic (uncached): 8×4.5MB×20 + 8×8MB×4 ≈ **980 MB/token**; active non-expert weights ≈ 2.7GB (attn 63MB/layer + embeds 1.85GB etc).

### Per-layer tensor map (`model.llm.layers.N.*`)

- `attn.wq_du` [3072,2304] (24×128), `wk_dv`/`wv_dv` [1024,2304] (8×128), `wo_ud` [2304,3072], `wr_du` [**384**,2304] = 24 heads × `d_rel`16, `rel_logits_proj.proj` [16,1024]
- `attn.gate_attn` [3072,2304] (sigmoid gate on attn out), `attn.attn_sink_proj` [24,2304]
- `attn.q_l2_clip.scale` [24,128], `attn.rel_l2_clip.scale` [24,16]
- `attn_sconv` [2304,1,4] (input depthwise), `mlp_sconv` [2304,1,4]
- Local layers: `attn.k_sconv.weight`/`v_sconv.weight` [1024,1,4] static depthwise
- Global layers (5,11,17,23): `k/v_sconv.static_w` [4,1024] **+ `weight_proj` [32,2304] — input-dependent conv kernels**; `q/k/v_input_norm` [2304] instead of `attn_norm`
- `mlp.gate.weight` [128,2304] + `gate.bias` [128]; `mlp_norm` [2304]
- Loop layers 6–17 add `controllers.{attn,k,mlp,v}.{down_weight[64,4608], forget_weight[2304,64], gain_weight[2304,64]}` (rank-64 forget/gain gating, input = 2×hidden concat)
- Top level: `attn_res.attn_queries.{0..22}` (4 of them [3,2304] — loop passes), `attn_res.mlp_queries.{0..23}`, `attn_res.output_query` [2304], `middle_cycle_cross_pass_queries` + `_kv_queries` [24,2304] — cross-pass residual/query mechanism (attn-res-block size 6)
- `embed`/`unembed` [201024,2304] bf16 each 926MB (untied, padded vocab)

## Machine inventory (verified)

| item | value |
|---|---|
| GPU | AMD Radeon RX 9070 XT, devID 0x7550 (gfx1201/Navi48), driver 32.0.32015.7029, Adrenalin Vulkan 1.4.359 / drv 2.0.416 |
| CPU | i7-13700KF (8P+8E) |
| RAM | 48 GB DDR5 |
| Disks | FireCuda 530 1TB (PCIe4, ~7GB/s spec) + PNY CS2230 1TB (PCIe4, DRAM-less, slower) + 2TB HDD; D: has 469GB free |
| Toolchain | Python 3.15, Rust/cargo, git, vulkaninfo; **no MSVC/clang/cmake/ninja on PATH** |

## Vulkan capabilities on this GPU (vulkaninfo, Adrenalin)

- `VK_KHR_cooperative_matrix` rev2 **present**; `shaderBFloat16CooperativeMatrix` = true, `shaderFloat8CooperativeMatrix` = true → **BF16 + FP8 tensor-core GEMMs from Vulkan**, no ROCm needed.
- Packed 4×8-bit integer dot product accelerated → INT4 dot path exists.
- Subgroup 32–64 (wave32/wave64 selectable), `computeFullSubgroups`, shuffle/ballot/rotate/clustered all present.
- `VK_EXT_memory_budget`, `VK_EXT_shader_atomic_float2`, `VK_KHR_16bit_storage`, `VK_EXT_subgroup_size_control`, `VK_EXT_cooperative_matrix_maintenance1`.
- Queues: 8×(gfx+compute+transfer), 8×(compute+transfer), 1×transfer-only (SDMA) + video queues → dedicated async copy queue available for expert streaming.
- Memory types: device-local heap (16GB VRAM) + host-visible heap (ReBAR — verify BAR size on this board; WMI AdapterRAM field unreliable).

## Immediate implications

1. Decode is expert-streaming bound: ~980MB/token uncached → PCIe-Gen5 pinned H2D ~50GB/s ⇒ ~20 tok/s ceiling w/o caching; each DRAM-resident byte moved at DDR5 ~60-90GB/s; NVMe ~6.5GB/s ⇒ ~6 tok/s floor if all experts came from disk (they won't — 48GB RAM holds all 19.5GB weights).
2. 16GB VRAM can't hold the model → VRAM = activations + KV + **hot-expert cache** + attention/dense weights (~4GB) + MTP head (0.6GB).
3. Loop passes (layers 6–17 ×3) + MTP verify batches multiply expert reuse → caching window larger than naive top-8/128.
4. Only 4 layers hold global KV → 1M-ctx KV ≈ 16GB bf16 / ~8GB fp8 / ~4GB int8 — KV quant decides feasibility; SWA layers ~80MB total.
5. Global-layer convs are input-dependent (`weight_proj`) — custom kernel needed; static convs elsewhere.
