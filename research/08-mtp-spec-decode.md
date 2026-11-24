# 08 — Speculative Decoding & the Inkling-2-Mini "MTP" Draft Head

Scope: how the shipped `mtp.safetensors` (~636 MB, 54 tensors) works, how DeepSeek-style MTP
heads are wired in vLLM/SGLang, what `markov_head_type: "rnn_hidden"`, `markov_rank: 256`,
`block_size: 8`, `target_layer_ids`, `fuse_weight`, `dflare_norm`, and `mask_embed` mean,
measured acceptance/speedup data, interplay with tiered MoE memory, and an implementation
sketch for our C/C++ runtime (RX 9070 XT + 48 GB RAM + NVMe).

**Bottom line up front.** Despite the name `mtp_config`, this head is *not* a
DeepSeek-style chained NextN module. The tensor inventory and config fields match the
**DFlash/DFlare block-drafting family**: a small dense transformer that predicts a whole
block of `block_size = 8` tokens in one pass, conditioned on hidden states tapped from
multiple target layers (`target_layer_ids`), with per-consumer learnable fusion
(`fuse_weight` + `dflare_norm` = DFlare), mask-token slots (`mask_embed`), and a shared
target `unembed` for logits (no lm_head ships in the file). On top of that, the
`markov_head` adds a **ReDrafter-style rank-256 GRU** that injects explicit
token-to-token autoregressive dependence into the parallel block — the fix that the
AngelSpec/"DFly" and DFlare follow-up papers apply to plain block diffusion. Verification
is the standard *linear* (chain) scheme: accept the longest matching prefix, take the
bonus token. Expect ~2–3× on code-heavy/chat decoding on our hardware if acceptance
behaves like other MTP-family heads (~0.85–0.95 first-position, mean accept length ~3–5).

---

## 1. Speculative decoding in one paragraph

Speculative decoding (Leviathan et al., https://arxiv.org/abs/2211.17192; Chen et al.,
https://arxiv.org/abs/2302.01318) lets a cheap "drafter" propose γ tokens and the target
model verify all of them in one forward pass. If per-token acceptance is α, the expected
tokens committed per cycle is `(1 − α^(γ+1)) / (1 − α)` — an upper bound, since acceptance
decays with draft depth (worked table: https://9io.ai/blog/speculative-decoding-production,
same formula in https://www.issoh.co.jp/tech/details/12353/). Greedy verification accepts
the longest prefix where `argmax p_target == draft_token`, then emits one "bonus" token
from the target logits at the first divergence; stochastic verification uses rejection
sampling (`accept w.p. min(1, p_t/p_d)`, resample from the normalized residual
`(p_t − p_d)+`), which preserves the target distribution exactly. Spec decoding is
*lossless* — the same output distribution, just fewer target passes — but a speculative
step is never cheaper than a plain step, so speedup = (tokens per step) / (step cost ratio).

## 2. How DeepSeek-style MTP runs in vLLM and SGLang

### 2.1 The DeepSeek-V3 MTP module (the reference design)

DeepSeek-V3 trains D sequential MTP modules; inference ships one ("NextN"). Per the tech
report (https://arxiv.org/html/2412.19437, §2.2) and weight docs
(https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md), each module
contains: `enorm`/`hnorm` (RMSNorms), `eh_proj` (d×2d projection merging
`RMSNorm(h_prev); RMSNorm(Emb(t_{i+1}))`), one full transformer block, and — crucially —
**shared embedding and shared lm_head with the target**. The module sees the target's
*final* hidden state plus the *next* token embedding, i.e. it conditions on one layer
(the last) and on ground-truth/drafted token identity. Official numbers: second-token
acceptance **85–90 %**, ~**1.8× TPS**.

vLLM's implementation (`vllm/model_executor/models/deepseek_mtp.py`,
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_mtp.py)
reproduces this: `eh_proj(cat[enorm(embeds), hnorm(prev_hidden)])` → decoder layer →
shared_head. The code masks `inputs_embeds` at position 0 and, for `num_speculative_tokens
> 1`, **re-runs the same single module autoregressively** ("Recycle the post-final-norm
hidden into the next draft step"). In vLLM V1 the draft is driven by
`vllm/v1/spec_decode/eagle.py` (`EagleProposer.propose()` takes `target_token_ids`,
`target_positions`, `target_hidden_states`, `next_token_ids`), and acceptance is a Triton
kernel in `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` +
`vllm/v1/spec_decode/utils.py` (computes `num_rejected_tokens`, picks
`token_indices_to_sample`, emits the bonus token).

### 2.2 Enabling it

vLLM (https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/):

```python
llm = LLM(model="XiaomiMiMo/MiMo-7B-Base",
          speculative_config={"method": "mtp", "num_speculative_tokens": 1})
```

- `num_speculative_tokens` = proposals per step; vLLM advises starting at 1 and raising
  only after per-position acceptance is verified (https://www.issoh.co.jp/tech/details/12353/).
- No separate draft model path is needed — weights live inside the checkpoint.
- `draft_lm_head_quantization: "fp8"/"nvfp4"` gives the drafter its own quantized copy of
  the shared lm_head to cut draft logits bandwidth (relevant for our 201k vocab).
- Verify pass shape: the target runs **one forward over `num_spec+1` positions per
  request** (anchor + γ drafts) with ordinary causal masking; logits are computed at all
  positions; the rejection kernel does accept/resample on GPU.

SGLang (https://docs.sglang.io/docs/advanced_features/speculative_decoding):
`--speculative-algorithm NEXTN` is an alias of the EAGLE worker — MTP heads are drafted
through the EAGLE pipeline with `--speculative-num-steps` (AR depth),
`--speculative-eagle-topk` (tree branching), `--speculative-num-draft-tokens` (verify
width). For DeepSeek-V3 their launch line used steps 2 / topk 4 / draft 4 and measured
**38.2 → 67.9 tok/s (1.78×)**, 77.7 t/s with torch.compile, batch 1
(https://github.com/sgl-project/sglang/pull/3582). When the MTP weights are inside the
checkpoint, `--speculative-draft-model-path` can point at the model dir itself; the loader
picks out the `mtp.*` keys (real-world walkthrough, incl. tuning logs:
https://lcz.me/topic/1629). MiMo-7B on SGLang uses steps 1 / topk 1 / draft 2.

Other MTP-enabled deployments: MiMo-7B (1 MTP layer tuned in pretrain+SFT, ~90 %
acceptance; extra pretrained `mtp_layers.{1,2}` shipped separately —
https://github.com/XiaomiMiMo/MiMo/, https://huggingface.co/XiaomiMiMo/MiMo-7B-MTPs; vLLM
fix PR https://github.com/vllm-project/vllm/pull/25136 shows mean accept len 2.06 with
position-0 acceptance 0.88 but position-1 only 0.18 — an untuned deeper step collapses).
Qwen3-Next recommends `num_speculative_tokens: 2`; GLM-4.5/4.6, DeepSeek-V3.1/V3.2, and
RedHatAI Qwen3.8-27B builds ship `model_mtp.safetensors` (lcz.me: code-gen decode 41.5 →
89.3 t/s, +115 %, acceptance 0.94, accept len 3.76; creative Chinese only +44 %, α 0.51).

### 2.3 The Inkling family's own precedent

The 1T sibling ships **8 chained single-layer MTP heads** enabling "up to 9 tokens per
forward step" (https://recipes.vllm.ai/thinkingmachines/Inkling), served as
`"method": "mtp", "num_speculative_tokens": 8`
(https://vllm.ai/blog/2026-07-15-inkling — measured **380 vs 140 tok/s/user, ~2.7×**, on
4× GB200). NVIDIA's Dynamo recipe runs "MTP, 8 draft tokens" and rejects `min_p`/
`logit_bias` requests under spec decode (https://docs.nvidia.com/dynamo/latest/recipes/inkling.md).
Modal deployed a **custom DFlash speculator** for the same model and saw ~67 % faster
interactivity than the built-in MTP path at matched throughput
(https://modal.com/blog/inkling-by-thinking-machines-labs-now-available-on-modal).
So the vendor treats "MTP" as a brand covering both chained-NextN and block-draft heads;
the Mini's head is the block-draft flavor.

## 3. How this head differs from EAGLE / EAGLE-3 and DeepSeek-MTP

| | DeepSeek NextN | EAGLE-1/2 | EAGLE-3 | **Inkling-2-Mini head** |
|---|---|---|---|---|
| Draft params | 1 TRM block + norms/proj | 1 TRM layer + fc | 1 TRM layer + fc (multi-layer tap) | 3 TRM layers + fusion + GRU |
| Target features | final hidden only | final hidden | low/mid/high layers → fc | **4 tapped layers → per-layer softmax mix** |
| Token input | next-token embedding (shared) | shifted token emb | shifted token emb (reduced "hot" vocab + d2t/t2d map) | `mask_embed` block slots + GRU token emb |
| lm_head | shared w/ target | shared w/ target | own (often hot-vocab) | **shared `unembed`** + rank-256 factored `markov_w1` |
| Proposals | chain (re-run k×) | tree via repeated top-k | tree | **linear block of 8** |
| Extra state | none | none | none | **GRU hidden (256-d) per draft position** |

EAGLE (https://arxiv.org/abs/2401.15077) drafts in feature space: it predicts the *next
feature* and the next token, reusing the target's lm_head and final hidden state; EAGLE-2
adds confidence-aware dynamic trees (https://arxiv.org/abs/2406.16858). EAGLE-3
(https://arxiv.org/abs/2503.01840) removes feature regression, fuses low/mid/high target
hidden states through an `fc`, trains with "training-time test" (multi-step unroll), and
commonly ships a reduced hot-vocabulary draft head with `d2t`/`t2d` maps (SGLang scatters
its proposal distribution onto the full vocab for block verification; FR-Spec,
https://arxiv.org/abs/2502.14824, motivates the same trick to cut lm_head cost). Reported
EAGLE-3 numbers: mean accept length τ ≈ 5.3–7.5 and 3.7–6.5× speedups on dense targets —
but those are tree-drafted; per-chain acceptance is lower.

The key structural difference for us: DeepSeek-MTP and EAGLE-1/2 read **only the last
hidden state**; EAGLE-3 reads 3 layers through one shared projection; our head reads
**4 layers and gives each consumer its own fusion** (DFlare), and it proposes a **block,
not a chain/tree** — the 8 positions are predicted in parallel rather than re-running the
module 8×.

## 4. Decoding this checkpoint: it's DFlare + a Markov GRU

Weight inventory (all BF16; ≈315 M params ≈ 630 MB):

- `layers.{0,1,2}`: full-attention draft transformer layers — `q_proj [3072,2304]`
  (24 heads × 128), `o_proj`, `k_proj/v_proj [1024,2304]` (8 KV heads), **plus a second
  pair `k_proj_ctx`/`v_proj_ctx` and `k_norm_ctx`**, `q_norm/k_norm`, SwiGLU MLP
  (9216), `input_`/`post_attention_layernorm`, and `dflare_norm [2304]` each.
- `fuse_weight [4,4]`, `mask_embed [2304]`, final `norm`.
- `markov_head`: `dflare_norm [2304]`, `markov_w1 [201024,256]`, `rnn_proj [768,5120]`
  (+bias), `hidden_out [2304,256]` (+bias).

### 4.1 DFlash/DFlare lineage

DFlash (arXiv 2602.06036, https://arxiv.org/abs/2602.06036; vLLM speculators doc
https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash.md;
NeMo recipe https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dflash-speculative-decoding):
a small draft model predicts an entire `block_size` block **in one non-causal pass**,
conditioned on a context feature built from hidden states of several target layers
(`target_layer_ids`), via mask-token embeddings; the draft **shares and freezes the
target's embed_tokens and lm_head**; verify is linear longest-prefix. Reported: >6×
lossless, up to 2.5× over EAGLE-3 (their numbers; block sizes 8–16 in public checkpoints).

DFlare (EMNLP 2026, https://aclanthology.org/2026.emnlp-main.1503/; arXiv 2606.02091;
analysis https://www.iplexlaw.co.kr/en/blog/dflare-block-diffusion-drafting) widens the
conditioning bottleneck: instead of one shared `W_c` projection, **each draft layer i
learns a softmax mixture over the tapped target hidden states**, `f_t^(i) =
RMSNorm(Σ_j α_j^(i) h_t^(j))` — a per-layer fusion weight plus a norm. It also applies
**separate K/V projections for the two attention sources** (target-context features vs
draft block features). Reported: 5.52×/5.46×/3.91× end-to-end on Qwen3-4B/8B/GPT-OSS-20B,
~5–11 % over DFlash. Public checkpoint `AngelSlim/Qwen3-4b-dflare`
(https://huggingface.co/AngelSlim/Qwen3-4b-dflare) confirms the config vocabulary we see
(`block_size`, `target_layer_ids`, `rope_theta`, tied embeddings).

Mapping to our tensors:

- **`dflare_norm`** — the per-consumer RMSNorm after the DFlare mixture. Four consumers
  carry it: draft layers 0/1/2 **and** `markov_head` — and `fuse_weight` is `[4,4]`, i.e.
  W_fuse ∈ R^{(D+1)×T}: softmax over the T=4 target-layer features for each of the 3
  draft layers + the markov head. Each consumer gets its own fused view of
  `h^{(5)}, h^{(23)}, h^{(41)}, h^{(48)}`.
- **`k_proj_ctx`/`v_proj_ctx`** — DFlare's source-specific projections: queries come from
  draft features; K/V are projected separately for the *context* sequence (fused target
  features over the prefix) and the draft's own tokens. Context is delivered through the
  draft attention KV exactly as DFlash describes.
- **`mask_embed`** — the learned block-slot embedding: positions in the 8-token block
  whose tokens are unknown during drafting (DFlash noise/mask slots).
- **`block_size: 8`** — the parallel proposal width = `num_speculative_tokens` (matching
  the family's 8-token MTP deployments and SGLang's DFLASH `--speculative-num-draft-tokens`
  semantics, which are "inferred from draft config, otherwise 16").
- **`local_layer_ids {0,2}` + `sliding_window_size 1024`** — draft layers 0 and 2 use
  1024-token sliding-window attention (bounding ctx-KV growth), layer 1 is global —
  mirroring the 5:1 hybrid pattern of the base model.
- No `embed_tokens`/`lm_head`/`eh_proj` in the file → token embeddings and logits come
  from the target's `model.llm.embed`/`model.llm.unembed` ([201024,2304] each, ~926 MB
  bf16 apiece in the main safetensors), consistent with DFlash's "shares and freezes the
  target's embed and lm_head".

### 4.2 `markov_head_type: "rnn_hidden"` + `markov_rank: 256`

A pure block drafter predicts the 8 positions conditionally independently given the fused
context — it loses the *token chain*. The published fix is "a lightweight autoregressive
head [that] introduces predecessor information after the parallel backbone"
(AngelSpec/DFly, https://arxiv.org/html/2607.25852), i.e. graft an RNN onto the block
draft. That is exactly what `markov_head` is — and it is also a direct descendant of
**ReDrafter** (Apple, https://arxiv.org/abs/2403.09919): "a recurrent neural network as
the draft model conditioning on LLM's hidden states," updating its state with the
embedding of each new token.

Shape arithmetic pins down the wiring:

- `rnn_proj.weight [768, 5120]` + bias: **768 = 3 × 256 — the three gates of a GRU**
  (reset/update/new) over a rank-256 state.
- Input width 5120 = 2304 + 2304 + 256 + 256: almost certainly
  `[draft_hidden ; fused_target_ctx ; token_emb_256 ; rnn_prev_256]` — i.e. the GRU sees
  the draft-transformer state, its own DFlare-fused target view, the previous token's
  factored embedding, and the carried Markov state.
- `markov_w1 [201024, 256]` is the rank-256 **factored token table** — token → 256-d
  embedding feeding the GRU, and plausibly tied as a low-rank logit bottleneck
  (`h_256 @ markov_w1^T` → cheap full-vocab logits/auxiliary distribution, à la
  FR-Spec's vocabulary reduction but rank-factored rather than hot-listed).
- `hidden_out [2304, 256]` (+bias) lifts the 256-d state back to model width so the Markov
  stream can be added into the draft hidden (or read by the unembed path).
- `"rnn_hidden"` thus reads as "the Markov head carries its state as an RNN hidden
  vector" — vs plausible alternatives (stateless bigram head, or logits-only head). The
  state is 256 values per position: trivially cheap, but it must be checkpointed per block
  position so verify-time rollback can restore the state at the last accepted token.

So the draft computes, per step: fused context features per consumer → 3-layer draft over
the 8 masked block positions (+ ctx K/V) → per-position GRU update chain feeding
`hidden_out` → logits through shared `norm` + target `unembed` (optionally modulated by
the markov factored head). The Markov chain makes `p(d_i | d_1..d_{i-1}, ctx)` a real
conditional — the property DFlash lacks and DFlare/DFly papers add post-hoc.

### 4.3 `target_layer_ids {5,23,41,48}` under the looped core

The base model runs 24 weight layers as **48 layer applications** (`loop_config`: core
layers 6–17 × 3 passes; apps 1–6 pre-core, 7–18/19–30/31–42 passes, 43–48 post-core,
1-based). Read as 1-based application indices, the taps land at: app 5 = real layer 4
(late pre-core), app 23 = pass 2, real layer 10, app 41 = pass 3, real layer 16, app 48 =
real layer 23 (final). That is a textbook low/mid/high spread *across the looped depth* —
EAGLE-3-style multi-level features — and implies the runner must expose hidden states at
four points inside the per-token forward, including mid-loop. (If the ids were 0-based the
last tap would be out of range — another reason to read them as 1-based.)

## 5. Acceptance rates and speedups — measured numbers

| Setup | Acceptance / τ | Speedup | Source |
|---|---|---|---|
| DeepSeek-V3 NextN, 2nd token | 85–90 % | 1.8× TPS | https://arxiv.org/html/2412.19437 |
| DeepSeek-V3 NEXTN on SGLang, bs1, steps2/topk4/draft4 | — | 38.2→67.9 t/s (1.78×); 77.7 w/ compile | https://github.com/sgl-project/sglang/pull/3582 |
| TML Inkling-1T MTP×8 on vLLM, 4×GB200 | — | 140→380 tok/s/user (~2.7×) | https://vllm.ai/blog/2026-07-15-inkling |
| TML Inkling + custom DFlash speculator, 8×B200 | — | +67 % interactivity vs built-in MTP | https://modal.com/blog/inkling-by-thinking-machines-labs-now-available-on-modal |
| MiMo-7B 1-layer MTP | ~90 % | — | https://github.com/XiaomiMiMo/MiMo/ |
| MiMo-7B vLLM mtp, num_spec 2 | τ 2.06 (α₀ .88, α₁ .18) | — | https://github.com/vllm-project/vllm/pull/25136 |
| Qwen3.8-27B-INT4 MTP, SGLang NEXTN, steps3/draft4 | α .94, τ 3.76 (code); α .51, τ 2.56 (creative zh) | +115 % code; +44 % creative | https://lcz.me/topic/1629 |
| Same model, NGRAM fallback | α .06–.23 | — | https://lcz.me/topic/1629 |
| Meta 7B 4-head MTP (Gloeckle) | ~2.5/3 code | ~3.0× code, 2.7× text | https://arxiv.org/abs/2404.19737 via https://www.issoh.co.jp/tech/details/12353/ |
| EAGLE-3 (Vicuna-13B / L31-8B, T=0) | τ 6.62 / 6.23 | 5.5× / 4.4× | https://arxiv.org/abs/2503.01840 |
| DFlash | — | >6×, up to 2.5× over EAGLE-3 | https://arxiv.org/abs/2602.06036 |
| DFlare (Qwen3-4B/8B, GPT-OSS-20B) | — | 5.52× / 5.46× / 3.91× | https://aclanthology.org/2026.emnlp-main.1503/ |
| ReDrafter (Vicuna, H100 / Apple-silicon MLX) | — | 2.8× / 2.3× | https://arxiv.org/abs/2403.09919 |
| ngram prompt-lookup (CNN/DM, 4×H100, 70B) | — | up to 2.8× | https://vllm.ai/blog/2024-10-17-spec-decode |
| SuffixDecoding (agentic, SWE-Bench) | — | up to 5.3×, ~2.8× over EAGLE | https://arxiv.org/abs/2411.04975 |

Takeaways for code+chat on our target: trained MTP-family heads hold ~0.85–0.95 first-token
acceptance on predictable text (code, reasoning, templated/agentic output) and ~0.5–0.6 on
creative/multilingual text; acceptance drops with depth and with temperature (T5 study:
0.75→0.62 going T0→T1). With an 8-token block and a per-position Markov chain, expect τ ≈
3.5–5 on code/chat at T≤0.7 and ~2–3 on open-ended text. Because the verify cost on our
hardware is dominated by expert-weight reads (§6), a 9-token verify is far cheaper than
9 sequential decodes, so realized speedup should track τ fairly closely: **budget 2–3×
decode speedup for code/agentic workloads, ~1.5–2× for chat** — in line with the big
sibling's measured 2.7× with an 8-token MTP.

## 6. Interaction with tiered memory + MoE (the big win)

Inkling-2-Mini is 26.4 B params in NVFP4/MXFP8 (~20 GB) over **48 GB of VRAM-less budget**
— RX 9070 XT has 16 GB. The dense/attention side (~3.4 B active, plus embed/unembed ~1.9 GB
bf16) fits; the 128-expert MoE body does not, so experts live in RAM/NVMe and stream H2D.
In that regime decode is not bound by FLOPs but by *expert fetches per step*. Three 2026
papers quantify why speculative decoding is disproportionately good here:

- **"Why Speculative Decoding Works Better Than Predicted on Sparse MoE Models"**
  (https://arxiv.org/abs/2610.38981): consecutive tokens share far more routed experts
  than independence predicts — measured step-1 overlap ~35 % (vs ~10.5 % independence
  baseline, 3.3×). So a 9-token verify activates much *less* than 9×8=72 distinct experts
  per layer; each fetched expert serves multiple verified positions.
- **SpecFit** (https://arxiv.org/abs/2610.15417): for offloaded MoE, H2D expert transfer is
  the critical path; a bigger draft tree transfers more experts — sometimes making SD
  *slower*. They choose the verified subtree by predicted throughput; 1.97×/1.72× over
  EAGLE-3 on offloaded Qwen3-30B-A3B.
- **EcoSpec** (https://arxiv.org/abs/2607.12696): "expert scattering" — high-probability
  draft tokens can route to disjoint experts; cost-aware draft selection that prefers
  paths reusing the current expert set gives up to 1.62× on DSv3.1/Qwen3-235B/GPT-OSS-120B.

Concretely for us: a verify pass over 9 positions reads *the same expert population* a
single decode would (modulo modest union growth) — so each NVMe/RAM fetch is amortized
across up to 9 committed tokens instead of 1. In a tiered system this is the difference
between spec decode being a wash and being a 2–3× win. Two design corollaries:

1. **Expert-union accounting must feed the loop.** Log per-verify, per-layer the union of
   routed experts; if union > cache budget, cap effective block length (early-exit the
   chain at k<8) — the poor man's version of SpecFit/EcoSpec. Our chain verify makes this
   trivial: the union is computable before fetching anything since routing is per-position.
2. **The draft lives on a separate tier.** 636 MB bf16 of dense weights stays resident in
   VRAM permanently — it never competes with the expert cache. Its KV is ~KB-class
   (3 layers × 8 KV heads × 128 × ≤1024-window × 2 B × 2 = ~12 MB/seq on the local layers;
   global layer 1's ctx-KV over the fused-feature sequence is ~4 KB/token — worth capping
   at a few 10s of thousands of positions or recomputing). Draft state is separate from
   the target KV and survives expert tiering decisions untouched. vLLM's
   `draft_lm_head_quantization` hint applies too: draft logits over 201k vocab × 2304 is
   the draft's biggest matmul; a quant copy or the rank-256 `markov_w1` path can prefilter.

## 7. Implementation sketch for the runtime

**Per-token state the runner must expose.** During every target forward (prefill and
verify), capture post-layer hidden states at applications {5,23,41,48} (real L4, L10@pass2,
L16@pass3, L23) into a small ring buffer — 4 × 2304 × 2 B = 18 KB per position; during
prefill this covers the prompt (context features for the draft's ctx-KV), during verify
only the ~9 new positions.

**Draft step (GPU, ~sub-ms).**
1. For each consumer c ∈ {layers 0,1,2, markov}: `ctx_c[t] = dflare_norm_c(Σ_j
   softmax(fuse_weight[c])_j · h^{(j)}[t])` — a 4×4 softmax-weighted sum over tapped
   features. Do it once per new position; then project ctx K/V via `k_proj_ctx`/
   `v_proj_ctx` (append to ctx-KV cache).
2. Build the 8-slot block: `mask_embed` (+ position ids, RoPE θ=1e6) for slots 1..8 — or,
   if the head was trained with the committed anchor embedding in slot 0, embed via
   `model.llm.embed`. Run 3 layers: q from block features; attention over [ctx-KV ‖
   block-KV], local window 1024 on layers 0/2, full on layer 1; SwiGLU MLP.
3. Markov chain: walk the block left→right; per slot
   `g = σ/split(rnn_proj [h_draft ; ctx_markov ; markov_w1[tok_prev] ; rnn_prev])`
   (768 = 3 gates), `rnn = GRU-update`; `h_draft += hidden_out(rnn)`. Persist `rnn` per
   slot (8 × 256 × 2 B = 4 KB) for rollback. Slot-0 `tok_prev` is the last committed token.
4. Logits: `norm(h_draft) @ unembed^T` over 8 positions (8 × 2304 × 201024 ≈ 7.4 GFLOP —
   the draft's dominant cost; a fused GEMV/fp8 copy of unembed is the obvious optimization,
   or evaluate `markov_w1`-path logits and only unembed top-m positions).

**Verify pass.** Standard linear verify: positions [anchor, d1..d8] through the 48-app
target with the normal causal mask and the spec-decode slot mapping (rejected slots' KV
are simply never committed — our paged KV already supports this). Collect logits at all 9
positions plus the four hidden-state taps for the draft's next context. MoE routing runs
per position as usual; *record the expert union*.

**Acceptance on GPU.** Greedy: `argmax` compare kernel, first-divergence scan, emit
accepted prefix + bonus token (target argmax at divergence). Stochastic: rejection
sampling — accept `d_i` w.p. `min(1, p_t(d_i)/p_d(d_i))`; on first rejection sample from
renormalized `(p_t − p_d)+`. This needs `p_d` in target vocab — fine since we share
unembed (SGLang's block-verification path documents the reduced-vocab scatter alternative;
vLLM `rejection_sample_method: standard|synthetic|block` and SGLang's
`--speculative-use-block-verification` (arXiv 2403.10444) offer lossless refinements:
block verification can accept a later prefix even when an earlier one fails).
Rollback = pick `rnn_state[k]` at last accepted slot, truncate draft block-KV to k,
advance ctx-KV by the committed positions only.

**Loop order.** Draft(≤1 ms) → verify fwd (dominant) → accept kernel → commit tokens →
next. No CPU round-trips between stages; the whole cycle should be one command stream.

**Metrics to log** (mirroring vLLM `spec_decode/metrics`): drafts, proposed/accepted
tokens, **mean acceptance length τ** (=1+accepted/drafts), **per-position α_i**,
tokens/s, draft ms vs verify ms, expert-union per layer per verify, ctx-KV occupancy, and
ngram-fallback hit rate. Tune `effective_block ≤ 8` and acceptance thresholds from these.

**Batching.** vLLM pads drafter batches (`disable_padded_drafter_batch` exists for backends
that allow ragged spec input) — for v1 of our runtime, fixed 8-wide blocks per sequence is
fine; multi-seq batching multiplies draft width linearly.

## 8. Fallbacks and adjacent drafters

- **N-gram / prompt-lookup decoding**: hash (n-gram → next tokens) over prompt + generated
  text; propose the longest-match continuation. Zero model cost; up to 2.8× on
  summarization (vLLM blog above; origin https://github.com/apoorvumang/prompt-lookup-decoding,
  vLLM issue https://github.com/vllm-project/vllm/issues/1802; config keys
  `prompt_lookup_min/max`). Acceptance is workload-limited (~0.06–0.23 on general chat per
  lcz.me) but excellent on repetitive agentic/code-diff traffic. ~100 lines in our
  runtime; worth building first as the spec-decode test harness proposer, then kept as a
  complementary proposer (ngram when a long exact match exists, else MTP block — both feed
  the same linear verify).
- **SuffixDecoding / SAM-Decoding**: suffix trees/automata over global+per-request history
  (https://arxiv.org/abs/2411.04975, https://arxiv.org/abs/2411.10666) — up to 5.3× on
  agentic loops, 18 %+ over other retrieval methods; the natural upgrade of prompt lookup
  for our agentic target market.
- **ReDrafter** (Apple; https://arxiv.org/abs/2403.09919; TRT-LLM integration
  https://developer.nvidia.com/blog/nvidia-tensorrt-llm-now-supports-recurrent-drafting-for-optimizing-llm-inference/):
  the closest relative of our `markov_head` — an RNN conditioned on target hidden +
  token embeddings, drafting a beam then flattening it into a tree for one verify
  ("explicit draft tokens" — acceptance logic inside the engine). Differences: ReDrafter
  *is* the whole drafter (no transformer block, no target-context KV); ours uses the GRU
  only to thread AR dependence through an 8-wide block draft, and we verify a chain, not a
  beam tree — simpler, and the verify cost is fixed at 9 positions. If block drafting
  under-delivers, a ReDrafter-style beam over our GRU head alone is a cheap plan B (all
  the machinery — per-slot states, factored token table — is already there).
- **Chained-NextN fallback**: if the block head misbehaves, re-running slot-0 predictions
  autoregressively (DeepSeek-style, recycle hidden → next step) gives a graceful
  chain mode at higher draft latency; the sibling 1T model ships exactly that.

## 9. Open questions / risks

1. **Exact markov wiring** — the 5120 input split (2304+2304+256+256) and whether
   `markov_w1` doubles as a logit table are inferred from shapes; verify against the
   modeling code / HF `inkling2_loop` implementation before freezing kernels.
2. **Slot-0 semantics** — whether block slot 0 carries the anchor token embedding (bonus
   token) or is pure mask (DFlash's `sample_from_anchor` duality); affects whether we
   emit 8 or 9 spec tokens' worth of logits.
3. **Verify semantics under sampling** — whether the head was trained to publish a
   proposal distribution (needed for strict rejection sampling) or is greedy-only;
   `min_p`/`logit_bias` interactions (Dynamo recipe rejects them under MTP).
4. **Ctx-KV growth** — global layer 1's context KV is ~4 KB/token; at 1 M ctx ≈ 4 GB.
   Cap or window it; sliding layers are already bounded.
5. **Quantization** — head ships bf16; target unembed/attention are bf16 but experts are
   NVFP4. Draft needs bf16 activation path even if the verify path is fp4-heavy.
6. **Acceptance on real traffic** — measure τ per workload; if τ < ~2.5 the verify
   overhead may not pay at high batch (spec decode is a latency tool, not a throughput
   tool — see vLLM blog's high-QPS caveat).

## Sources

- vLLM spec-decode hub/MTP page/schema — https://docs.vllm.ai/en/stable/features/speculative_decoding/ , /mtp/
- vLLM code — https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_mtp.py ; …/vllm/v1/spec_decode/eagle.py ; …/v1/worker/gpu/spec_decode/rejection_sampler.py ; …/v1/spec_decode/utils.py
- vLLM spec-decode perf blog — https://vllm.ai/blog/2024-10-17-spec-decode
- vLLM Inkling day-0 — https://vllm.ai/blog/2026-07-15-inkling ; https://recipes.vllm.ai/thinkingmachines/Inkling
- SGLang spec-decoding docs (incl. NEXTN/DFLASH/UNO/block-verification) — https://docs.sglang.io/docs/advanced_features/speculative_decoding
- SGLang NEXTN PR — https://github.com/sgl-project/sglang/pull/3582
- AMD ROCm MTP/SGLang guide — https://rocm.blogs.amd.com/software-tools-optimization/mtp/README.html
- DeepSeek-V3 report + weights doc — https://arxiv.org/abs/2412.19437 ; https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md
- Meta/Gloeckle MTP — https://arxiv.org/abs/2404.19737
- EAGLE / EAGLE-2 / EAGLE-3 — https://arxiv.org/abs/2401.15077 ; https://arxiv.org/abs/2406.16858 ; https://arxiv.org/abs/2503.01840
- Spec-Bench — https://github.com/hemingkx/Spec-Bench
- DFlash — https://arxiv.org/abs/2602.06036 ; https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash.md ; https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dflash-speculative-decoding
- DFlare — https://aclanthology.org/2026.emnlp-main.1503/ ; https://www.iplexlaw.co.kr/en/blog/dflare-block-diffusion-drafting ; https://huggingface.co/AngelSlim/Qwen3-4b-dflare
- AngelSpec/DFly, DynaDFlash — https://arxiv.org/html/2607.25852 ; https://arxiv.org/abs/2611.19085
- ReDrafter — https://arxiv.org/abs/2403.09919 ; https://developer.nvidia.com/blog/nvidia-tensorrt-llm-now-supports-recurrent-drafting-for-optimizing-llm-inference/
- MiMo — https://github.com/XiaomiMiMo/MiMo/ ; https://huggingface.co/XiaomiMiMo/MiMo-7B-MTPs ; https://github.com/vllm-project/vllm/pull/25136
- MoE+SD — https://arxiv.org/abs/2610.38981 ; https://arxiv.org/abs/2610.15417 ; https://arxiv.org/abs/2607.12696
- Prompt lookup / suffix / SAM — https://github.com/apoorvumang/prompt-lookup-decoding ; https://arxiv.org/abs/2411.04975 ; https://arxiv.org/abs/2411.10666 ; https://www.snowflake.com/en/blog/engineering/suffixdecoding-arctic-inference-vllm/
- FR-Spec — https://arxiv.org/abs/2502.14824
- Real-world MTP tuning (Qwen3.8-27B, SGLang) — https://lcz.me/topic/1629
- Metrics/acceptance discussion — https://www.issoh.co.jp/tech/details/12353/ ; https://9io.ai/blog/speculative-decoding-production
- HF blog (Inkling family, MTP drafters) — https://huggingface.co/blog/thinkingmachines-inkling
- Modal Inkling + custom DFlash — https://modal.com/blog/inkling-by-thinking-machines-labs-now-available-on-modal
- NVIDIA Dynamo Inkling recipe — https://docs.nvidia.com/dynamo/latest/recipes/inkling.md
- Leviathan/Chen speculative sampling — https://arxiv.org/abs/2211.17192 ; https://arxiv.org/abs/2302.01318
- Medusa — https://arxiv.org/abs/2401.10774
- Block verification — https://arxiv.org/abs/2403.10444
