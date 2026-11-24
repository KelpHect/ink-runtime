# MTP speculative decode — implementation plan

Target: ~65–85 t/s decode (baseline to beat: Unsloth llama.cpp ROCm fork,
84.9 t/s @128K on this exact card/model — see research/00-model-facts.md §
"numbers to beat"). Mechanism: the model ships its own drafter,
`Inkling-2-Mini/mtp.safetensors` (635.8MB, all bf16). Full math spec:
research/07 §9. Theory notes: research/08, docs/THEORY.md §8.

## Architecture summary (from research/07 §9)

- **Taps**: visit-input hiddens at vidx {5,23,41} + final normed output.
  GpuGraph already captures all 4 (`tap(i)` accessor, readback_ layout
  `[logits V][taps 4H][...]`) — no target-graph changes needed for reads.
- **Draft backbone**: 8-position non-causal block = anchor token embed +
  7 `mask_embed` slots; 3 layers of {prenorm attn (24/8 GQA, head_dim 128,
  RoPE theta 1e6, q_norm/k_norm, local window 1024+8 for layers 0,2) +
  SwiGLU MLP(9216)}. Context KV comes from `kv_proj_ctx`/`k_norm_ctx` of
  the fused tap features (`ctx = softmax(fuse_weight[i])·taps` →
  `dflare_norm_i`) — a prefix computed once per anchor.
- **Markov head** (7 sequential steps): GRU rank-256 —
  `rnn_proj[768,5120]` input `[state, markov_w1[prev], hidden, f_rnn]`;
  `state = g*state + (1-g)*tanh(c)`; logits = softcap30(base_logits_i +
  fold_w·latent) where fold_w = unembed·hidden_out.T precomputed once
  ([201024,256], 206MB f32). num_spec = 7, anchor = bonus token.
- All bf16 with listed fp32 islands (research/07 §12.1); markov head
  rounds intermediates to bf16 per step.

## Implementation plan

### Phase 1 — GpuDrafter (new files, no conflicts)
`src/gpu/gpu_draft.{h,cpp}` + `shaders/draft_*.comp` as needed:
- Weights: upload all 636MB once via upload_dev (VRAM-resident).
- Reuse: `matvec_bf16` for all projections (M=8 GEMV-block), `rmsnorm`,
  `embed_norm` (shares target embedw_/embn_), `sigmoid_mul`-class
  elementwise, shared `unemb_mv_` pipeline for base_logits (8-row M).
- New small kernels likely needed: fuse_alpha tap-mix (4×2304 weighted
  sum), the GRU cell (1 dispatch per step — sequential 7 steps, hidden
  256...768), non-causal block attention over [ctx_kv + block] (8 queries
  — a dedicated small shader or reuse attn_scores/softmax/accum with
  window params).
- Output: 8 drafted token ids + anchor's committed logits.
- Perf target: ≤5ms/block (weights are tiny; everything VRAM-resident).

### Phase 2 — Verify forward (reuse gpu_prefill machinery)
- Verify = a T=8 (or 9 with anchor) chunk of `GpuPrefill` over the
  drafted ids at the current position: produces per-position logits +
  correct kv/conv state updates and router ids per row.
- Acceptance (greedy): walk positions, accept draft[i] iff
  argmax(logits_i) == draft[i]; on first mismatch emit that argmax as the
  real token (bonus) and stop. Also stop at EOS.
- State commit/rollback:
  - kv banks are append-only → truncating `kv_len`/slot counters to the
    accepted count discards rejected positions (cheap).
  - conv state is a 3-token window → snapshot all T positions' conv
    states during verify (tiny buffers) or recompute the accepted
    position's state; pick whichever is simplest given gpu_prefill's
    layout.
  - router/ridx per-position → verify stages the UNION of per-row ids
    per layer (cap ~12–16); stage-copy machinery extends naturally.

### Phase 3 — Integration + tuning
- Decode loop in main.cpp: draft-block → verify → commit accepted →
  repeat; fall back to single-token on EOS/edge cases.
- Adaptive block length from observed acceptance (log union size).
- Expected: verify cycle ~70–85ms → ~5–7 tokens → 12–15ms/token.
- Coopmat/fp8 GEMM for the M=8/9 shapes only if profiling shows the
  verify GEMMs off bandwidth-bound.

## Dependencies / sequencing
- Phase 1 (GpuDrafter) is fully independent — new files, shared shaders
  only. Can start now.
- Phase 2 needs gpu_prefill's T-row machinery (landed, committed 0324c93)
  + whatever record-once form agent 1e6f000f lands (verify reuses the
  same persistent-cb machinery ideally).
- Warm-cache/prefetch (docs/STAGED_CACHE.md) helps verify too (union
  staging) — orthogonal.
