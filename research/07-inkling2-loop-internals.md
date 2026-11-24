# Inkling 2 Mini (`inkling2_loop`) — exact forward-pass internals

Research notes for a C/C++ runtime implementation. Everything below is derived from the
actual reference code (not guesses), cross-checked across four implementations:

| Source | Files |
|---|---|
| HF Transformers (PR #51050, merged 2026-11-21) | `src/transformers/models/inkling2_loop/modeling_inkling2_loop.py` (1840 lines), `configuration_inkling2_loop.py`, `src/transformers/models/inkling2/modeling_inkling2.py` (shared components + vision tower), `src/transformers/conversion_mapping.py`, `src/transformers/core_model_loading.py` — <https://github.com/huggingface/transformers/tree/main/src/transformers/models/inkling2_loop> |
| vLLM (PR #69028, merged 2026-11-21) | `vllm/models/inkling2/nvidia/{model,loop,attention,sconv,moe,dspark,vision}.py`, `vllm/models/inkling2/nvidia/ops/{attn_res,conv_mix,sconv,loop,dspark,attention}.py`, `vllm/models/inkling2/{configs,speculator,speculator_config}.py` — <https://github.com/vllm-project/vllm/tree/main/vllm/models/inkling2> |
| llama.cpp (main branch) | `src/models/inkling2.cpp` (base layer graph, ~460 lines), `src/models/inkling2-loop.cpp` (loop wrapper) — <https://github.com/ggml-org/llama.cpp/tree/main/src/models> |
| SGLang | `python/sglang/srt/models/{inkling2,inkling2_mtp,inkling2_vision}.py` — <https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models> |
| Checkpoint | `D:\Projects\ink-runtime\Inkling-2-Mini\` (`config.json`, `model-00001-of-00001.safetensors`, `mtp.safetensors`, `hf_quant_config.json`, `tokenizer.json`, `tiktoken/tokenizer.model`, `chat_template.jinja`) |

HF `modeling_inkling2_loop.py` is the semantic ground truth (eager, unfused). vLLM
fuses the same math into Triton/CUDA kernels; llama.cpp is the cleanest C++-shaped
reference and matches op-for-op. All three agree on every formula below.

---

## 1. Model shape and the virtual 48-layer stack

Physical layers = 24 (`hidden 2304`). `loop_config = {core_start 6, core_end 18,
num_passes 3, gate_rank 64, is_varloop false}` ⇒ the core layers **6..17 (12 layers)**
run **3 passes with shared weights**. Virtual visit order:

```
visits = [(i, pass0) for i in 0..5]            # prelude, 6 visits
       + [(i, p) for p in 0..2 for i in 6..17] # 3 passes, 36 visits
       + [(i, pass0) for i in 18..23]          # coda, 6 visits        => 48 visits
```

`total_num_hidden_layers = 24 + (3-1)*12 = 48`. Layer types alternate inside each
6-layer block: **layers 5, 11, 17, 23 are "hybrid" (full/global attention)**;
all others are "hybrid_sliding" (window 1024). So each 6-layer block is
`[5×SWA + 1×full]`, with the full layer LAST (`layer_types[i] = "hybrid" if (i+1)%6==0`).

There is **no RoPE anywhere in the LLM** (llama.cpp: `rope_pattern.fill(false)` —
"the relative-position bias carries the positions"). Positions enter only through the
content-conditioned relative bias, the causal masks, the convs, and the log-scaling.

### Checkpoint → module name map (HF `conversion_mapping.py` + `core_model_loading.py`)

- `model.llm.` → trunk. `embed.weight`→embed, `embed_norm.weight`→embedding RMSNorm,
  `norm.weight`→final RMSNorm, `unembed.weight`→lm_head (see vocab padding below).
- Per layer `model.llm.layers.N.`:
  - `attn.wq_du.weight` [3072,2304] → q proj; `attn.wk_dv.weight`/`attn.wv_dv.weight`
    [1024,2304] → k/v proj (8 kv heads ×128); `attn.wo_ud.weight` [2304,3072] → o proj;
    `attn.gate_attn.weight` [3072,2304] → output gate proj; `attn.attn_sink_proj.weight`
    [24,2304] → per-head sink logits; `attn.wr_du.weight` [384,2304] → rel-state proj
    (24 heads×d_rel 16); `attn.rel_logits_proj.proj` [16,1024] → learned bias-vs-distance
    bank; `attn.q_l2_clip.scale` [24,128] → per-(head,dim) q scale;
    `attn.rel_l2_clip.scale` [24,16] → rel-state scale.
  - `attn.k_sconv`/`attn.v_sconv`: sliding layers have `.weight` [1024,1,4] (static
    depthwise conv); full layers have `.static_w` [4,1024] + `.weight_proj` [32,2304]
    (dynamic taps). `attn_sconv.weight`/`mlp_sconv.weight` [2304,1,4] → post-attn /
    post-mlp hidden-stream convs (conv_idx 0=K, 1=V, 2=attn-out, 3=mlp-out).
  - `attn_norm.weight` → input_layernorm (sliding layers only);
    `q_input_norm`/`k_input_norm`/`v_input_norm.weight` → per-stream norms
    (full layers only); `mlp_norm.weight` → post_attention_layernorm (all layers).
  - `mlp.gate.weight` [128,2304] fp32 → router weight; `mlp.gate.bias` [128] fp32 →
    `e_score_correction_bias` (selection only).
  - `mlp.experts.w13_weight` U8 [128,2304,1152] + `_scale` F8E4M3 [128,2304,144] +
    `_amax` F32 [128] + `_outer_scale_denominator` F32 [1] → NVFP4 `gate_up_proj`;
    `w2_*` [128,2304,576]+[128,2304,72] → `down_proj`. **w13 rows are interleaved
    (gate,up) pairs along dim 1** — HF applies `Interleave(dim=1)` after dequant
    (reshape [1152,2]→transpose→[gate|up] contiguous).
  - `controllers.{attn,k,v,mlp}.{down_weight [64,4608], forget_weight [2304,64],
    gain_weight [2304,64]}` → the 4 message gates on loop layers 6..17 only
    (`attn` stream = the q stream).
- `attn_res.attn_queries.{0..22}` → `layers.{i+1}.attention_input.query`
  (**offset +1**; layer 0 has none). Sliding-layer entries are [2304];
  full-layer entries (indices 4,10,16,22 ↔ layers 5,11,17,23) are [3,2304] = the
  separate q/k/v read queries. fp32.
- `attn_res.mlp_queries.{0..23}` → per-layer MLP read queries [2304]. fp32.
- `attn_res.output_query` → `final_input.query` (the trunk's last read). fp32.
- `middle_cycle_cross_pass_queries` [24,2304] + `middle_cycle_cross_pass_kv_queries`
  [24,2304] → merged into `message_input.query` [12,4,2304]: first tensor holds each
  loop layer's **[q, mlp] query pair**, second holds **[k, v] pair**; merged order is
  (q, k, v, mlp) per loop layer (`Inkling2LoopMergeMessageQueries`). fp32.

---

## 2. Attention-residual blocks (attn_res) — the residual-stream replacement

This is the central architectural novelty: **there is no running residual add across
the whole model**. Instead, hidden state flows through "blocks" of ≤6 layers, and
each sublayer *reads* its input by a tiny softmax attention over stored
block-boundary snapshots.

State during forward: `block_states` — a stack (list) of `[T,2304]` tensors, plus a
current "partial" (running residual within the block).

**Push rule:** layer `i` is a *block boundary* iff `i % 6 == 0` (layers 0,6,12,18).
At a boundary layer, its **input** `x_in` is appended to `block_states`, and the
residual accumulator **restarts** (the layer's output is `attn_out + mlp_out`,
NOT `x_in + ...`). At non-boundary layers, the residual is a normal
`x_in + attn_processed` running sum.

**Read op (`Inkling2LoopBlockAttentionResidual` / vLLM `attn_res` kernel /
llama.cpp `mix`):**

```
# block_states: list of N tensors [T,2304]; partial: [T,2304]; query: [Q,2304] fp32
def attn_res(block_states, partial, query):            # per token t
    S   = stack(block_states + [partial])              # (N+1, 2304)
    rms = sqrt(mean(S^2, dim=-1) + 1e-6)               # per source row, eps=1e-6
    xn  = S / rms                                      # normalize_values=True
    logits_n = (q . xn_n) * 2304**-0.5                 # per query q, per source n
    p   = softmax(logits)                              # over the N+1 sources, fp32
    out = sum_n p_n * xn_n                             # (Q,2304) fp32
```

Equivalent vLLM kernel form (identical result): `logit_n = (q_pre·x_n)·rstd_n` with
`q_pre = q·D^-0.5` pre-scaled at load; `out = Σ softmax_n·rstd_n·x_n / Σ softmax_n`.
Queries are fp32 parameters (attn_res group stays fp32 per `hf_quant_config.json`).
The attn_res read is then fed through the next sublayer's RMSNorm (vLLM fuses this
in-kernel; HF applies `input_layernorm`/`post_attention_layernorm` separately).

**Usage sites (pseudocode of one decoder layer visit, HF `*DecoderLayer.forward`):**

```
x_in = hidden                                    # "partial" carried between layers
# --- attention input read ---
if layer > 0:
    qkv_reads = attn_res(block_states, partial=x_in, query=attn_queries[i-1])
    # sliding layer: Q=1 query -> one read shared by q,k,v
    # full layer:    Q=3 queries -> separate q_in,k_in,v_in
else:
    q_in = k_in = v_in = x_in                    # layer 0: no read
q_in = attn_norm(q_in); k_in = attn_norm(k_in); v_in = attn_norm(v_in)
# sliding: single attn_norm on the shared input; full: q/k/v_input_norm each
attn_out = attention(q_in, k_in, v_in)                      # §4
attn_out = post_attn_sconv(attn_out)                        # (conv+x)/sqrt(5), §5
residual = attn_out
if i % 6 == 0:  block_states.append(x_in)                   # push INPUT, restart
else:           residual = hidden = x_in + attn_out
# --- mlp input read ---
mlp_in = attn_res(block_states, partial=residual, query=mlp_queries[i])[0]
mlp_in = mlp_norm(mlp_in)
mlp_out = post_mlp_sconv(moe(mlp_in))                       # (conv+x)/sqrt(5)
return residual + mlp_out                                   # new partial
```

Wait — one subtlety in HF: at a boundary layer, `hidden` after the attn step is
`attn_out` only (residual restarted), so the mlp read's "partial" key is `attn_out`;
at non-boundary it's `x_in + attn_out`. The vLLM kernel encodes the same: last
source = `partial + addend` (addend = attn conv out), or just the conv-out at a
boundary where the old partial was already pushed into `stack`.

Final read: `hidden = attn_res(block_states, partial=hidden, query=output_query)[0]`
then `final_norm`. At that point `block_states` has ≤5 entries
([emb, L6-in, L12-in, L18-in] of the last pass + partial = 5 keys max).

**Pipelined view (how vLLM/llama.cpp structure it — equivalent):** the read for
sublayer *k+1* is fused into the conv of sublayer *k* (`conv_mix`/`convolve`+`mix`):
each conv writes the new partial into `stack[num_blocks]` and immediately produces
the *next* read through the *next* norm. C++ implementation can keep the simple
two-pass form.

---

## 3. The loop mechanism (layers 6–17 ×3)

Pass 0 runs the core exactly like normal layers (no messages). Before each pass
p ≥ 1:

```
previous_block_states = block_states[num_pre_loop_blocks + 1 :]   # == [s12] here
previous_hidden       = hidden          # output of layer 17 of pass p-1
hidden, block_states  = pre_loop_hidden, pre_loop_block_states    # RESTART
```

`block_states` after the prelude = `[emb]` only (layer 0 is the only boundary in
0..5) ⇒ `num_pre_loop_blocks = 1`, so `previous_block_states = block_states[2:]` =
**exactly one state**: the snapshot pushed at layer 12's input during the previous
pass (= output of layer 11 = end of loop-block-1). Plus `previous_hidden` (= end of
loop-block-2) as the "partial". llama.cpp: "the bank of the previous pass: its
blocks after the first core block, and its output" — the layer-6 snapshot
(= prelude output = the restart point) is deliberately skipped.

**Message extraction (`message_input`, `Inkling2LoopMessageAttentionResidual`):**
the SAME attn_res read, but with per-loop-layer queries `[4,2304]`:

```
keys  = [s12_prevpass, final_prevpass]        # 2 sources, fp32, rms-normalized
msgs  = attn_res(keys, queries=message_queries[li])   # (4,2304): q,k,v,mlp messages
```

computed **before each core layer of pass ≥1**, so the bank is the previous pass's
pair of block-end snapshots while `partial` = previous pass's final output.

**Message blending (`Inkling2LoopMessageGate` / `controllers`, rank 64):**
each loop layer has 4 controllers (`attn`=q, `k`, `v`, `mlp`):

```
# dest = current sublayer input (the attn_res read, before its RMSNorm), fp32
# msg  = the message for this stream, fp32
h    = down_proj( bf16( concat[ rms_norm(dest), rms_norm(msg) ] ) )   # 4608 -> 64, bf16
forget = tanh( forget_weight @ h )        # [2304,64]@64 -> 2304, fp32
gain   = tanh( gain_weight   @ h )        # fp32
out    = (1 - forget) * dest + gain * msg # fp32
out    = sublayer_norm(out)               # then the usual input norm
```

`rms_norm` here is **unweighted** (`Inkling2LoopUnweightedRMSNorm`, eps 1e-6).
Down-proj runs in bf16, gates in fp32, and `forget`/`gain` weights are fp32 in the
checkpoint. Applied per stream:

- sliding loop layer (6–10,12–16): single attn_res read → expand to 3 dests →
  gate with attn/k/v messages → **single shared `attn_norm`** on each.
- full loop layer (11,17): 3 attn_res reads (q,k,v) → gate each with its message →
  `q/k/v_input_norm` each.
- mlp stream: `mlp_in` (post mlp-query read) → gate with mlp message → `mlp_norm`.

**Caches:** every loop layer keeps a **separate KV cache and 4 conv-state streams
per pass** (48 virtual cache layers total). HF: `cache_offset = 24 + (p-1)*12 - 6`
(physical layer 6 → virtual cache 24/36 on passes 1/2). vLLM gives each
`InklingConvState` a `.pass{p}` name. llama.cpp maps virtual layer `il ≥ 24` to
physical `6 + (il-24) % 12` for weights while keeping per-slot caches. So pass p's
attention only sees pass p's own KV history — each pass is a fresh causal context
re-reading the same tokens through different (message-conditioned) projections.
The `is_varloop` checkpoint can also set `loop_share_cache` (share KV across
passes; conv states stay per-pass) and `loop_num_passes ≠ 3`; not applicable here.

Also worth noting: the layer-6 attn_res read (`attn_queries[5]` over
`[emb, prelude_partial]`) is identical across passes because both sources are —
vLLM caches it as `prelude_dests`. The messages are the only per-pass difference
to layer inputs.

---

## 4. Attention (all layers: 24 Q heads, 8 KV heads, head_dim 128)

Per-token, per-layer, with `x_q/x_k/x_v` the three normed sublayer inputs (equal on
sliding layers in pass 0):

```
q   = wq_du @ x_q                       # [24*128]  -> [24,128] per head
k   = wk_dv @ x_k                       # [8*128]   -> [8,128]
v   = wv_dv @ x_v                       # [8*128]   -> [8,128]
q   = scaled_l2_clip(q)                 # q *= q_l2_clip.scale (per head,dim),
                                        # then ||q_h|| capped at 22.627417
k'  = k_sconv(k);  v' = v_sconv(v)      # §5 (k/v convs on the 1024-wide kv stream)
k'  = l2_clip(k')                       # plain cap at 22.627417, NO learned scale
v'  = rms_norm_no_weight(v')            # per-head RMSNorm, eps 1e-6, no gamma
                                        # ("use_value_norm_no_gamma")
r   = wr_du @ x_q                       # [24*16] -> [24,16] "relative states"
r   = scaled_l2_clip(r)                 # r *= rel_l2_clip.scale [24,16],
                                        # cap ||r_h|| at 8.0
# full-attention layers ONLY (5,11,17,23):
q *= tau(pos);  r *= tau(pos)           # log-scaling, see below
```

`l2_clip`: `x *= min(1, threshold / (||x|| + 1e-6))` over each head's 128 (or 16)
dims, fp32. `scaled_l2_clip` multiplies by the learned `[heads,dim]` scale first
(fp32 math, all of it).

**Log-scaling** (`log_scaling`, full layers only):
`tau(p) = 1 + 0.1 * log( max((p+1)/8192, 1) )` — multiplies every query head vector
and every relative-state vector at full layers (context-length extension; ~×1.9 at
1M positions). Sliding layers never apply it.

**Relative position bias** (replaces RoPE — content-conditioned!):
`proj` = learned `[d_rel=16, rel_extent]` table (extent = 1024 for both sliding and
full; sliding uses `sliding_window_size`):

```
rel_logits = (r @ proj)                  # [B,T,24,extent] -> [B,24,T,extent]
d_ij       = pos_i - pos_j               # backward distance to key j
bias[h,i,j] = rel_logits[h,i, clamp(d_ij,0,extent-1)]
bias = 0 where d_ij < 0 or d_ij >= extent
```

i.e. each query token mixes a 16-dim learned basis of "bias vs distance" profiles
by content. Gather, then mask.

**Attention sink:** `t[h,i] = attn_sink_proj @ x_q` — one extra logit per head/query
with **zero value**: softmax is computed over `[logits ; t]` and the sink column is
dropped after normalization (`exp(s-t)` trick: llama.cpp instead subtracts `t` from
all logits and appends a constant-0 sink — algebraically identical).

**Core attention (eager semantics):**

```
S = q @ k'^T * 128^-0.5 + bias + causal_or_sliding_mask   # fp32 softmax
P = softmax([S , sink_column]) [...,-1]                    # drop sink col
o = P @ v'                                              # [24,128]
gate = sigmoid( gate_attn @ x_q )                       # [3072], fp32 sigmoid
o  = flatten(o) * gate
out = wo_ud @ o                                         # [2304]
```

- scaling = `head_dim^-0.5` (1/√128). Softmax always fp32.
- Sliding layers: causal mask + window (`key_pos > query_pos - 1024`), GQA 24/8.
- Full layers: causal only (rel bias provides local→global falloff to distance 1024).
- Everything downstream of `x_q` uses the **q-stream input only** (q proj, rel
  states, sink, output gate). K path uses `x_k`, V path uses `x_v`; on full layers
  `x_k`,`x_v` also feed the dynamic conv tap projections.

**Draft-free intuition:** 20 sliding-window layers do local work with static convs;
the 4 full layers (5,11,17,23) get log-scaled queries, dynamic (input-conditioned)
k/v convs, and the wide rel_extent.

---

## 5. Short convolutions (sconv, kernel 4, 4 streams per layer)

All convs are **causal depthwise conv1d, kernel_size 4, no bias, fp32 state/compute**,
with a conv-state cache of the last 3 input tokens per channel (decode path:
`causal_conv1d_update` shifts state + one-step conv). Per-layer channel layout:
`[K(1024), V(1024), attn-out(2304), mlp-out(2304)]` packed head-major
(vLLM `InklingConvState` `stream_ranges`; llama.cpp offsets
`0, n_kv, 2*n_kv, 2*n_kv + n_embd`).

**Static conv** (k/v on sliding layers; attn/mlp-out on all layers):

```
y = depthwise_conv1d(x, w)      # w [C,1,4], tap 0 = current token, causal padding
out = (y + x) / sqrt(4 + 1)     # = /sqrt(5)
```

**Dynamic conv** (k/v on full layers 5,11,17,23 only):

```
taps_dyn = weight_proj(x_in)                # [T, 8heads*4] from the sublayer INPUT
taps     = static_w.T.view(8,128,4) + taps_dyn.view(T,8,1,4)   # per head,per token
y[t,c]   = sum_lag taps[t,h(c),lag] * x[t-lag, c]              # lag 0 = current
out      = (y + x) / sqrt(2*4 + 1)          # = /3
```

(`static_w` [4,1024] is per-channel static taps; `weight_proj` [32,2304] produces
4 dynamic taps per kv head per token — "input-dependent taps" like a tiny
Hyena/GDN conv.) `conv_mask` = the 2D padding mask (`create_recurrent_attention_mask`);
zero-pad/ignore it in a padding-free runtime. `seq_idx` handles packed sequences —
n/a for single-sequence inference.

So per layer visit: `k,v` get conv idx 0,1; the *attention output* gets conv 2
(`(conv(o)+o)/√5` on the 2304 stream); the *MoE output* gets conv 3. The conv
output IS the residual update (the "+x" inside is the sublayer's own output,
not the block residual — the block residual/partial add happens around it, §2).

---

## 6. MoE (all 24 layers): 128 experts, top-8, softcapped SwiGLU

Router (`Inkling2LoopTopKRouter`, all fp32):

```
logits  = gate.weight @ x                # [128] fp32 (gate.weight kept fp32)
scores  = sigmoid(logits)                # scoring_func = "sigmoid"
idx     = topk(scores + gate.bias, 8)    # bias ONLY affects selection
w       = scores[idx]; w /= (sum(w) + 1e-20)
w      *= 8.0                            # route_scale
```

No renormalization against non-selected experts; no shared expert (`n_shared_experts=0`).

Expert (SwiGLU with softcaps, fp32 activation):

```
gate_up = w13_e @ x                      # [2*1152], gate|up halves (de-interleaved)
a = softcap4(gate) * sigmoid(gate) * softcap25(up)
# softcap_c(z) = c * tanh(z/c);  NB: sigmoid sees the RAW gate, cap only bounds
# the linear factor and up
out = w2_e @ a                           # [2304]
mlp(x) = sum_{e in idx} w_e * out_e      # fp32 accumulate
```

`intermediate_size=1152`, so each expert is tiny; 3.4B active/token.

**Quantization** (`hf_quant_config.json`): experts in layers 0–19 are **NVFP4**
(`w13/w2`: 2×E2M1 per byte — low nibble = even element — one F8E4M3 scale per
**16** values along the last dim, per-expert factor `amax/outer_scale_denominator`,
`weight_block_shape [1,16]`); layers 20–23 are **MXFP8** (E4M3 values, power-of-2
scale `2^(s-127)` per 32 values, `[32,32]` blocks). Dequant formulas are in
`Inkling2LoopDequantizeExperts` (core_model_loading.py:772). Everything else bf16,
except: attn_res queries, `mlp.gate.bias`, and message queries are fp32; unembed
runs bf16 in / fp32 out. For a C++ port: store NVFP4 packed nibbles + fp8 scales +
amax, dequantize on the fly in the expert GEMV (or pre-expand to bf16 at load —
19.5 GB checkpoint → ~26B params bf16 won't fit 32 GB alongside everything else, so
on-the-fly dequant is recommended; MXFP8 similarly).

---

## 7. Embedding, final norm, LM head

```
h0 = embed_norm(embed[token])            # RMSNorm(eps 1e-6) WITH learned weight
                                         # applied to token embeddings only;
                                         # image features are scattered in raw
... 48 layer visits (§1–§6) ...
h  = attn_res(block_states, h_last, output_query)   # final learned read
h  = rms_norm(h)                         # model.llm.norm
logits = unembed[0:200058] @ h           # bf16 in / fp32 out
logits = 30 * tanh(logits / 30)          # final_logit_softcapping
```

`vocab_size 201024` vs `unpadded_vocab_size 200058`: `unembed` is stored padded;
**only the first 200058 rows produce logits** (`Inkling2UnpadVocab` drops the tail;
llama.cpp slices `[0:200058]` after the softcap). Embedding table is full 201024.
Embed/unembed are NOT tied.

**RMSNorm everywhere**: `x / sqrt(mean(x²) + 1e-6) * weight` in fp32, eps 1e-6
(vision uses LayerNorm eps 1e-5 instead). Unweighted variant (no gamma) used for:
attn_res value normalization, message-gate input normalization, and **V head
normalization** (`use_value_norm_no_gamma`).

---

## 8. Full forward pseudocode (putting it together)

```
h = normed_embed(tokens)                      # §7
block = []                                    # attn_res stack
partial = h
pre_blocks = pre_partial = None
def visit(i, pass_idx):
    nonlocal partial, block
    # messages for loop layers on pass>0
    msgs = None
    if pass_idx > 0:
        msgs = attn_res([bank0], bank1, message_queries[i-6])   # (4,2304)
    x_in = partial
    if i > 0:
        reads = attn_res(block, x_in, attn_queries[i-1])        # 1 or 3
    else:
        reads = [x_in]
    q,k,v = norm_streams(reads, msgs and msgs[0:3])             # §3 gate + norms
    a  = post_attn_conv( attention(q,k,v) )                     # §4,§5
    residual = a
    if i % 6 == 0:
        block.append(x_in)
    else:
        residual = partial = x_in + a
    m_in = attn_res(block, residual, mlp_queries[i])[0]
    if msgs is not None:
        m_in = rmsnorm_after_gate(m_in, msgs[3])
    else:
        m_in = mlp_norm(m_in)
    mo = post_mlp_conv( moe(m_in) )                             # §5,§6
    partial = residual + mo

for i in 0..5:  visit(i, 0)
pre_hidden, pre_blocks_state = partial, copy(block)   # = [emb]
for p in 0..2:
    if p > 0:
        bank0, bank1 = block[-1], partial             # s12-snapshot, pass-final
        partial, block = pre_hidden, copy(pre_blocks_state)  # reset to [emb]
    for i in 6..17: visit(i, p)
for i in 18..23: visit(i, 0)
h = rms_norm( attn_res(block, partial, output_query)[0] )
logits = 30*tanh(unembed[:200058] @ h / 30)
```

(bank = `block_states[num_pre+1:]` + prev partial; for this config exactly
`[s12_of_prev_pass, prev_final]` — §3. `visit` numbers = virtual visits 0..47.)

**Decode-time state:** per pass p ∈ {0,1,2} and per core layer: KV cache + 4 conv
states (each 3 tokens × its channels). Prelude/coda layers have one cache set.
Conv states for k/v streams live on the 1024-wide kv stream; attn/mlp convs on 2304.
SWA KV cache can be truncated to the 1024 window; the conv state is always 3 tokens.

---

## 9. MTP / DSpark speculator head (`mtp.safetensors`, 635 MB)

Config: 3 layers, hidden 2304, 24/8 heads ×128, intermediate 9216, SwiGLU,
`rope_theta 1e6` (the draft DOES use standard RoPE, neox half-split, on per-head
RMS-normed q/k — `q_norm`/`k_norm`/`k_norm_ctx` [128]), local layers {0,2} (window
1024+8 keys), `block_size 8`, `markov_rank 256`, `markov_head_type rnn_hidden`,
`target_layer_ids {5,23,41,48}`.

**Target taps:** `target_layer_ids` are *virtual visit indices* whose INPUT (the
attn-res partial) is captured — visit 5 (input of layer 5), 23 (input of layer 11
on pass 1), 41 (input of layer 17 on pass 2); id 48 = the final normed output.
Each captured feature is `[T,2304]`; together `[T,4,2304]` raw features are stored
("DFlare" taps).

**Draft backbone (per generated block of 8 positions = 1 anchor + 7 mask slots):**

```
x[0]   = embed_norm(embed[anchor_id])       # shared embed+norm w/ target
x[1:8] = mask_embed                          # learned [2304]
ctx    = fuse_alpha[i] . features            # per-layer i: softmax(fuse_weight[i])
                                           # weighted mix of the 4 taps, fp32
ctx    = dflare_norm_i(ctx)                  # per-layer RMSNorm
ctx_kv = kv_proj_ctx(ctx) -> k_norm_ctx(k) -> rope  -> draft KV cache
         # context K/V derived from TARGET features, precomputed for the prefix
block qkv = qkv_proj(x); q_norm,k_norm; rope at the block's real positions
attn  = block-attention over [context + block] KV — NON-causal inside the block;
        local layers see last 1024+8 keys; standard 128^-0.5 scaling, GQA 24/8
layer = prenorm attn + prenorm MLP(down(up*silu(gate)), residual bf16 adds
out   = rms_norm(last residual)
base_logits = unembed @ out                # bf16
```

`fuse_weight` checkpoint [4,4] → `fuse_alpha = softmax(fuse_weight)`; rows 0–2 for
draft layers, row 3 (index −1) for the markov head's `f_rnn` (computed from the
last committed token's features through `dflare_norm`).

**Sequential Markov head (`rnn_hidden`, rank 256), per draft step i:**

```
# once per block: context_i = rnn_bias + [h_i, f_rnn] @ rnn_proj[:, 512:].T   (3*256, fp32)
raw_i   = context_i + [state, markov_w1[prev_token]] @ rnn_proj[:, :512].T   # fp32
g,c,o   = split(raw_i, 3)                                                  # 256 each, bf16-rounded
gate    = sigmoid(g);  cand = tanh(c);  latent = tanh(o)                   # bf16
state   = gate*state + (1-gate)*cand                                       # GRU update
bias    = fold_w @ latent + fold_b         # fold_w = unembed @ hidden_out.T (vocab×256)
logits  = 30*tanh(bf16(base_logits_i + bias)/30)                           # softcap 30
sample token_i;  prev = token_i                                            # 7 sequential steps
```

`rnn_proj` input order = `[state(256), markov_w1(prev)(256), hidden(2304),
f_rnn(2304)]` = 5120 → 768. `markov_w1` is a vocab×256 embedding (low-rank
prev-token table), `hidden_out` 256→2304 folded through the shared lm_head into a
vocab×256 "Markov bias" table. RNN state resets to zeros per proposal; `f_rnn`
per request comes from the anchor's mixed features. num_speculative_tokens = 7
(block−1); anchor is the bonus token (`sample_from_anchor=False`, DFlash layout:
each position's logits predict the token AT that position +1 via standard next-token).

`mtp.safetensors` keys: `layers.{0..2}.{input,post_attention}_layernorm`,
`self_attn.{q,k,v,o}_proj`, `self_attn.{k_norm,k_proj_ctx,k_norm_ctx,v_proj_ctx,q_norm}`,
`mlp.{gate,up,down}_proj`, `dflare_norm`, `markov_head.{markov_w1,rnn_proj,
hidden_out,dflare_norm}`, `fuse_weight`, `mask_embed`, `norm`. All bf16.

---

## 10. Vision tower (shares `inkling2` Small's `Inkling2VisionModel`)

Config: 23-layer ViT, hidden 1024, 16 heads (head_dim 64), `patch_size 20`,
`temporal_patch_size 2`, `spatial_merge/adapter_merge_size 2`, `posemb_grid 32`,
`projection_hidden_dim 4096`, LN eps 1e-5, `hidden_act gelu`, `layer_scale_init 0.1`,
axial 2D RoPE with default theta 10000.

```
# image processor (HF image_processing_inkling2.py): pad to multiple of 40×40
# (fill -1 pre-normalization), normalize (x/255 - .5)/.5 -> [-1,1],
# cut into (rows*cols) tokens of 40×40×3 in raster order; grid_thw=[1,rows,cols]
# (one LM token per 2×2 native patches = 40×40 px; video T handled upstream)
patches = pixel_values.reshape(-1, merge=2, patch=20, merge=2, patch=20, C=3)
                       .transpose(2,3)                       # (m,m,p,p,C)
emb   = patch_embedding(patch.flatten())                     # Linear 1200->1020? 
        # patch_embedding.weight [1024, 1200]: Linear(3*20*20=1200 -> 1024), no bias
emb  += bilinear_interp(position_embedding[32x32 grid], to (2r,2c) patch grid,
                        align_corners=False)                 # 4 taps, border-pad
# prepend per image a CLS token = class_embedding + cls_position_embedding
h = ln_pre(h)                                               # LN eps 1e-5
for 23 layers:
    # per-image (non-causal, cu_seqlens = patches+1 CLS per image)
    h += lambda_1 * attn(LN(h))     # qkv w/ bias, axial 2D rope:
    #   head_dim 64 -> inv_freq over dim/4=16 per axis; concat [freq_w,freq_h]
    #   repeated per interleaved pair; rotate_pairwise (x1,x2)->(-x2,x1)
    #   positions = (row,col) 1-indexed per patch, CLS at (0,0)
    h += lambda_2 * mlp(LN(h))      # c_fc 1024->4096 gelu, c_proj 4096->1024, bias
tokens = merger(h[patch_idx].reshape(-1, 4*1024))             # drop CLS rows
       = proj_out(act(proj_in(tokens)))                       # 4096->4096 gelu
                                                            # ->2304, no bias
# -> [num_image_tokens,2304] scattered into <|unused_200054|> (id 200054)
# placeholder positions of the input embedding sequence (replacing embed_norm'd
# token embeddings — image embeddings are NOT re-normed)
```

`image_grid_thw` counts in **language-model tokens** (each = 2×2×20×20 = 40×40 px).
`position_ids` inside the ViT are 1-indexed patch coords (block-major merge order),
CLS = (0,0).

---

## 11. Tokenizer & chat template

- `tiktoken/tokenizer.model`: **o200k-style tiktoken BPE** — 200058 lines =
  199998 mergeable tokens (ranks 0–199997; single bytes 0–255, GPT-4o merge order —
  the rank tail " DELL/UB/NAN/SCHO" matches o200k_base) + **60 special tokens at
  199998–200057** stored as base64 `<|...|>` strings. `tokenizer.json` wraps the
  same BPE with the **o200k pre-tokenizer regex**
  (`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|...|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`)
  + ByteLevel (no prefix space on encode, `use_regex` decode). So: reuse an o200k
  merge table, keep the custom special map.
- Specials: `<|endoftext|>` 199999 (pad), `<|message_user|>` 200000,
  `<|message_model|>` 200001, `<|message_system|>` 200002, `<|message_tool|>` 200003,
  `<|content_text|>` 200004, `<|content_image|>` 200005,
  `<|content_model_end_sampling|>` 200006 (**EOS**), `<|content_thinking|>` 200008,
  `<|end_message|>` 200010, `<|content_tool_error|>` 200022, `<|content_xml|>` 200024,
  `<|begin_of_text|>` 200028, `<|content_invoke_tool_json|>` 200049,
  `<|unused_200054|>` = image placeholder 200054,
  `<|content_invoke_tool_text|>` 200057; everything else `<|unused_N|>`.
- Chat wire format (`chat_template.jinja`): every message =
  `<|message_{role}|>[name]<|content_{type}|>…<|end_message|>`; assistant messages
  additionally end with `<|content_model_end_sampling|>` (the actual stop token);
  assistant thinking goes in `<|content_thinking|>…<|end_message|>` BEFORE content;
  tools: a system message `tool_declare<|content_xml|>[{json specs}]<|end_message|>`;
  calls emit `<|message_model|>{fn}<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>`;
  a thinking-effort system message (`Thinking effort level: {0..0.99}`) is injected
  before the first non-system message. Generation prompt = the rendered prefix
  ending at `<|message_model|>` (the `{% generation %}` block wraps assistant turns).

---

## 12. Implementation notes / gotchas for the C++ runtime

1. **Precision discipline matters.** fp32: attn_res queries+mixes, message gates
   (down-proj bf16, tanh gates fp32), all RMSNorm accumulators, convs (weights &
   state), router (everything incl. sigmoid+softmax norm), SwiGLU softcap math,
   softmax, sink/gate sigmoid, unembed output, L2 clips, rel-bias gather. bf16
   elsewhere; markov head ops round intermediates to bf16 at each step (see
   `markov_gate` kernel).
2. The two conv-state kinds: static `(conv(x)+x)/√5` and dynamic `(conv(x)+x)/3`
   — denominators differ; dynamic taps come from the *sublayer input*, not the
   projected k/v.
3. `attn_queries` indexing is `layers[i-1]` (layer 0 has none) — easy off-by-one.
4. `block_states` stack: push happens at layers {0,6,12,18} with the **layer
   input**; at boundaries the residual accumulator restarts (output = attn+mlp
   only). Max live stack = 5 entries at the final read.
5. Loop restart restores BOTH `partial` (prelude output) and `block_states`
   (`[emb]`); the message bank is `block_states[2:]` of the finished pass +
   previous final hidden → exactly 2 keys for this config.
6. Passes share weights but NOT caches: 48 KV slots + 48 conv-state sets for the
   3-pass model (llama.cpp: `layer_phys` maps slot→physical layer).
7. The attention sink is an extra softmax column with **zero V** — implement as
   `softmax([S,t])` then drop, or subtract `t` from logits and append 0 (llama
   trick, exact).
8. `lm_head` outputs 200058 logits (first rows of the 201024 unembed); embed uses
   all 201024 rows. EOS = `<|content_model_end_sampling|>` = 200006.
9. The MoE gate bias is added ONLY for top-k selection, not to the kept weights;
   kept sigmoid scores are renormalized to sum 1 then ×8.
10. `is_varloop`/varloop checkpoint only: `loop_num_passes` and `loop_share_cache`
    overrides — reject for this fixed checkpoint.
11. Sliding layers use `rel_extent = sliding_window_size = 1024` (same value as
    full layers' rel_extent — no behavioral difference, but they index different
    per-layer `proj` tables).
12. Vision: patches arrive pre-cut by the processor (`pixel_values` rows =
    40×40×3 LM tokens), CLS per image, LN + LayerScale 0.1, axial rope θ=10000,
    merger `gelu(4096→4096)→2304`. `temporal_patch_size=2` matters only for video
    (T frames); images use T=1.

---

## 13. Quick tensor reference (checkpoint names, all `model.llm.`)

| Tensor | Shape | Role |
|---|---|---|
| `embed.weight` | [201024,2304] bf16 | token embedding |
| `embed_norm.weight` | [2304] | embedding RMSNorm |
| `layers.N.attn.{wq_du,wk_dv,wv_dv,wo_ud}.weight` | [3072\|1024,2304] | q/k/v/o projections |
| `layers.N.attn.gate_attn.weight` | [3072,2304] | output gate |
| `layers.N.attn.attn_sink_proj.weight` | [24,2304] | sink logits |
| `layers.N.attn.wr_du.weight` | [384,2304] | rel states (24×16) |
| `layers.N.attn.rel_logits_proj.proj` | [16,1024] | bias-vs-distance bank |
| `layers.N.attn.{q,rel}_l2_clip.scale` | [24,128]/[24,16] | learned clip scales |
| `layers.N.attn.{k,v}_sconv.weight` | [1024,1,4] | static k/v conv (sliding) |
| `layers.N.attn.{k,v}_sconv.{static_w,weight_proj}` | [4,1024]/[32,2304] | dynamic conv (full only) |
| `layers.N.{attn,mlp}_sconv.weight` | [2304,1,4] | hidden-stream convs |
| `layers.N.{attn,mlp}_norm` or `{q,k,v}_input_norm` | [2304] | sublayer norms |
| `layers.N.controllers.{attn,k,v,mlp}.{down,forget,gain}_weight` | [64,4608]/[2304,64] | loop gates (layers 6–17) |
| `layers.N.mlp.gate.{weight,bias}` | [128,2304] f32/[128] f32 | router |
| `layers.N.mlp.experts.{w13,w2}_weight(+_scale,_amax,_outer_scale_denominator)` | §6 | NVFP4/MXFP8 experts |
| `attn_res.attn_queries.{0..22}` | [2304] or [3,2304] | layer i+1 attn reads (f32) |
| `attn_res.mlp_queries.{0..23}` | [2304] | layer i mlp reads (f32) |
| `attn_res.output_query` | [2304] | final read (f32) |
| `middle_cycle_cross_pass_{,kv_}queries` | [24,2304] each | loop message queries (f32) |
| `norm.weight`, `unembed.weight` | [2304], [201024,2304] | final norm + padded head |
