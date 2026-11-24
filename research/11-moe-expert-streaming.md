# 11 — MoE Expert Streaming: Offloading, Prediction, and Caching for Inkling 2 Mini

Research subagent A11 — literature survey on expert offloading/prediction for MoE inference,
applied to our model: 128 routed experts/layer, top-8, no shared experts, tiny experts
(hidden 2304, intermediate 1152), 24 layers with layers 6–17 looped 3× (shared weights),
gate softcap 4.0 / route_scale 8.0 / learned gate bias, MTP head for spec-decode.
Hardware: 16 GB VRAM GPU, 48 GB DDR5 RAM, NVMe, PCIe 5 x16.

---

## 0. Executive summary — what the literature says for *our* shape

1. **Decode is a weight-streaming problem, not a compute problem.** Per token we execute
   48 MoE blocks (6 + 12×3 + 6) × top-8 = up to 384 expert activations. Every expert is
   ~4.5 MB (layers 0–19, NVFP4 packed `w13`/`w2` + E4M3 block scales) or ~8.2 MB (layers
   20–23, FP8 E4M3 — measured from the checkpoint index). A fully-cold decode moves
   ~1.84 GB/token over PCIe; at ~50 GB/s effective that is ~37 ms → ~27 tok/s absolute
   ceiling with perfect overlap and zero reuse. Every hit in a ~9–11 GB expert cache
   (~2,000–2,400 expert slots) directly buys latency.
2. **Expert routing is predictable — but only modestly so for balanced modern routers.**
   Three independent measurement families agree: residual-stream gate prediction
   (apply next layer's router to this layer's input) reaches ~80–90% per-expert coverage;
   learned MLP predictors reach ~85% (ProMoE); correlation-table/history predictors reach
   ~85% (ST-MoE). Older encoder-decoder MoEs (Switch, NLLB) show power-law activation and
   91–99% predictability; modern balanced decoder-only MoEs (Mixtral, DeepSeek, Qwen —
   like ours) are deliberately de-skewed, so static popularity pinning buys only 3–5 pp
   of hit rate over random (Fiddler). **Prediction, not popularity, is the lever.**
3. **Our loop is a structural gift none of the papers have.** Layers 6–17 reuse the *same*
   128-expert pool three times per token. That means (a) guaranteed reuse distance of ~11
   layer-steps for a per-layer cache, (b) a cross-*pass* correlation channel identical in
   spirit to the cross-*layer* correlation ST-MoE/Patterns papers exploit, and (c) a
   prefetch window: while pass *p* of the block computes, we have ~11 layer-steps of
   compute to stage experts for pass *p+1*. Nobody has published this; we should exploit it.
4. **Spec-decode both helps and hurts.** Draft/verify raises unique experts per layer-step
   from ~8 to ~16–24 (of 128), but token-adjacent routing correlation means the union is
   far smaller than 8×k. SpecMoEOff shows SD *enlarges* per-expert workload M from 1 to
   ~2–4, making each fetched expert more amortized — a 2.5× decode win over offload baselines.
5. **CPU-execute-on-miss (Fiddler/HybriMoE/kTransformers) is marginal at our expert size.**
   A 4.5 MB expert GEMV on CPU ≈ 75 µs at DDR5 ~60 GB/s vs ~90 µs to transfer it — nearly
   break-even *before* contending with prefetch traffic. Big-expert models (Mixtral: ~350 MB
   bf16 expert) make CPU execution obviously right; ours makes it a scheduling tool, not a
   pillar. It still wins for the FP8 tail layers (8.2 MB → ~165 µs transfer) and for misses
   discovered too late to prefetch.
6. **Kernels: our GEMMs are bandwidth-bound at every batch size that matters.**
   M=1–24 tokens × N∈{1152, 2304} × K∈{1152, 2304} gives arithmetic intensity ~1–4 FLOP/B.
   `moe_align_block_size`-style grouped GEMM (vLLM/SGLang) pads M to BLOCK_M — wasted
   tensor-core work, but irrelevant when the kernel is DRAM-bound. For decode we want
   dequant-fused GEMV/mmvq-style kernels (Marlin/AWQ lineage or gfx12 WMMA/V-dot RDNA4
   kernels), not tiled GEMM. Only prefill (M in the hundreds) justifies BLOCK_M≥64 tiles.
7. **Dropping top-8→top-4 is nearly free — IF we fix the renormalization.** Naive k=4
   costs −4.65 MMLU pts (Qwen3.6-35B) and −8 GSM8K pts (measured independently). With
   the k2-decoupled reference set (activate top-4, normalize by top-16 router mass) the
   same halving costs **−0.35 pts, statistically indistinguishable** — a one-line change
   that halves both expert bandwidth and compute. Whether this transfers depends on our
   router's normalization semantics (softcap+bias — needs a quick check + eval).
8. **Recommended design:** three-tier expert store (VRAM pool ~2,000–2,400 slots / RAM
   full copy / NVMe-mmap tail), per-layer working-set sizing, gate-decision prefetch
   L→L+1 plus cross-pass prediction inside the loop, early-preemption at router time,
   chunked (per-matrix) fetch granularity, and score-aware (MRS-style) + loop-pass-pinned
   eviction. Expected effect: keep the ~35 ms/token streaming floor hidden; realistic
   decode lands wherever attention+overheads set it, not expert I/O.

---

## 1. The problem, sized for our model

Measured from `model.safetensors.index.json` (checkpoint total 19.5 GB):

| Item | Value |
|---|---|
| Expert layout | `layers.{L}.mlp.experts.w13_weight` [128, 2304, 1152] U8 packed-FP4 + `w13_weight_scale` [128, 2304, 144] F8_E4M3 + per-expert FP32 `amax`; `w2_weight` [128, 2304, 576] U8 + scale [128, 2304, 72] |
| Bytes per expert (layers 0–19, NVFP4) | 2.65 MB (w13) + 0.32 MB (w13 scales) + 1.33 MB (w2) + 0.17 MB (w2 scales) ≈ **4.48 MB** |
| Bytes per expert (layers 20–23, FP8) | ~**8.21 MB** (w13 [128,2304,2304] E4M3 + w2 [128,2304,1152] E4M3 + scales) |
| Total expert bytes | **15.67 GB** (80% of checkpoint) |
| Experts per layer | 128 (router `mlp.gate.weight` BF16 [128,2304] + `mlp.gate.bias` [128]) |
| MoE executions per decode token | 48 (layers 0–5, 6–17 ×3 passes, 18–23) |
| Max expert-activations per token | 48 × 8 = 384 |
| All-miss streaming per token | 44 fp4-exec × 8 × 4.48 MB + 4 fp8-exec × 8 × 8.21 MB ≈ **1.84 GB** |
| PCIe 5 x16 effective | ~50–55 GB/s (63 GB/s line rate, 128b/130b) |
| DDR5 (dual-channel) | ~60–90 GB/s |
| Transfer time, fp4 expert | ~85–90 µs; fp8 expert ~150–165 µs |
| CPU GEMV, fp4 expert (batch-1) | ~50–75 µs at 60–90 GB/s — same order as PCIe |
| Loop-block params per pass | layers 6–17 share weights; `controllers.{attn,k,mlp,v}` rank-64 gain/forget adapters modulate each pass |

Non-expert weights ≈ 3.8 GB (attention, embeddings, vision tower, controllers — stay
resident in VRAM). KV cache is bounded by the 1024 sliding window on most layers (every
6th layer, ids 5/11/17/23, is "local"/global — see `local_layer_ids`), so several GB of
KV headroom exists; budget ~2–4 GB → **expert pool ≈ 9–11 GB ≈ 2,000–2,400 fp4 slots
(vs 2,560 fp4 + 512 fp8 experts total)**.

That is an unusually high cache fraction (~65–80% of the fp4 pool). Most papers assume
<50%. It changes the calculus: our miss stream is the minority path, and **prefetch
quality + eviction policy decide whether we sit near the 27 tok/s streaming floor or
near the compute ceiling**.

Loop interpretation (must verify in code, but standard for "varloop"/looped-block
architectures): passes run block-sequentially — layer 6p1, 7p1, …, 17p1, 6p2, … —
so an expert's intra-token reuse distance is ~11 layer-steps. If it instead runs
layer-sequentially (6,6,6,7,7,7,…) reuse is trivially adjacent. Either way the expert
*pool* is identical across passes; only the rank-64 controllers differ the computation.

---

## 2. Expert activation predictability — the measured record

### 2.1 Residual-stream gate prediction ("speculative expert loading")

The cheapest predictor uses the model itself: transformer layers are residual, so the
hidden state changes slowly across layers — apply *layer L+1's router* to *layer L's
input* and prefetch its top-k guesses. Proposed in the Mixtral-Offloading work
(Eliseev & Mazur, arXiv:2312.17238): guessing with top-1 candidate hits ~80% of the
time, top-3 candidates ~90%. An independent implementation (arXiv:2511.05814) measured
**84.6% precision/recall** on Mixtral. DALI (arXiv:2602.03495) uses the same
inter-layer residual signal ("Residual-Based Prefetching") for high-workload expert
prediction, and reports 1.32–3.97× over llama.cpp/kTransformers/MoE-Lightning/HybriMoE.

Cost for us: one extra router GEMV per layer — a [2304→128] matvec + softcap + bias +
top-k, ~µs. We can run it speculatively on the *previous* hidden state as soon as layer
L's attention output exists, before L's MoE even finishes — giving ~1 full layer-step
(~attention + dense compute) of lead time ≈ hundreds of µs ≈ 3–10 experts of prefetch
budget per step. Also applicable *inside* the loop: predict pass p+1's gate from pass
p's output — residual approximation is even closer there (same weights, small
controller modulation).

### 2.2 Learned predictors (best measured accuracy)

- **ProMoE** (arXiv:2410.22134, SOSP'25): small MLP per layer takes that layer's input
  hidden state → predicts expert selection. **84.7% average per-expert accuracy** on
  DeepSeek-MoE and Qwen-MoE; ~5 pp drop under "stride" (predict 2 layers ahead).
  Compares against token-ID predictor (58.3%) and skip-based predictor (66.9% on
  Qwen2-MoE). Runs predictors on CPU (~200 µs) in parallel with GPU compute; "stride
  prefetching" pipelines prediction with fetching so bandwidth is never idle waiting
  on the predictor. Speedups: 1.78×/1.34× prefill/decode vs hand-tuned caches,
  2.20×/2.07× vs framework offloading. Also: **GoodPred metric = Accuracy × FetchRate**
  — a predictor that is right but late is useless; measure both.
- **SiDA-MoE** (arXiv:2310.18859, MLSys'24): 2-layer LSTM + sparsemax-attention hash
  function predicts *all layers' experts at once from input embeddings*, before the
  model runs. 91.7–99% top-3 hit rate on Switch-base-8/128 (encoder-decoder; easier
  routing), preserving 92.6–99% of task accuracy. The "predict everything upfront"
  shape is what our loop needs conceptually (predict all 3 passes), but accuracy will
  be lower on a balanced router — ProMoE's ~85% on modern MoEs is the honest prior.
- **ExpertFlow** (arXiv:2410.17954): a small transformer predicts *all* layers' expert
  usage in one forward pass at iteration start, plus a token scheduler that groups
  tokens with similar predicted routes (raises per-expert M → better GEMM/bandwidth
  amortization) and a cache that self-corrects mispredictions. −93.7% GPU memory, up
  to 10× throughput vs offloading baselines.
- **DuoServe-MoE** (arXiv:2509.07379): phase-specialized — prefill activates experts
  almost densely (prefetch is trivially "load everything, overlapped"), decode uses a
  small layer-level MLP predictor ("ExpertMLP") trained on activation traces. 5.34×
  TTFT, 7.55× E2E improvements. Reinforces that **prefill and decode want different
  policies** — our prefill can prefetch the entire model through in ~0.4 s of PCIe
  time while compute-bound anyway.

### 2.3 Correlation tables — the no-training alternative

- **ST-MoE** (arXiv:2606.15453): builds P(e_{L+1,j} | e_{L,i}) conditional-probability
  tables + a token-history table from observed activations; **85% expert prediction
  accuracy**, 2.5×/2.2×/1.5× over GPU-only/Adap-Gating/Pre-gated-MoE, no model change,
  no training. Chi-squared tests confirm cross-layer and cross-token routing is not
  independent (p<0.01 on DeepSeek/Qwen/Mixtral).
- **Inter-layer expert affinity** (arXiv:2401.08383): formalizes P(e_{L+1}|e_L)
  affinity and prefetches the top-affiliated expert — the simplest version of the
  same idea.
- **Patterns behind Chaos** (arXiv:2510.05497, ISCA'26; 24k requests, >2000 GPU-hours,
  traces public on HF `core12345/MoE_expert_selection_trace`): the best measured
  numbers on the correlation structure itself:
  - **Cross-layer**: top-20% of next-layer candidates carry **50% / 65% / 77% / 56%**
    of conditional probability mass (DeepSeek-V3 / Qwen3-235B / Llama-4 / Kimi-K2).
    Qwen3 — a 128-expert top-8 model like ours — is strongly correlated.
  - **Cross-token (same layer, t→t+1)**: top-20% carry **47% / 62% / 80% / 53%**;
    a bright diagonal = the *same* expert tends to re-fire for the next token,
    predominantly in **higher layers** — i.e., keep last-token experts cached.
  - **Prefill→decode**: top-5 prefill experts cover ~60% of top-5 decode experts;
    ~75% (top-10) and ~90% (top-20). Spearman correlation between prefill/decode
    heatmaps is strong in most layers. → **Seed the decode cache with the request's
    own prefill histogram; don't start decode cold.**
  - **Pair co-activation**: expert pairs co-fire at 20–40× the chance rate; the top
    10% of pairs account for 60–80% of co-activations (DeepSeek/Qwen3).
  - **Imbalance**: hottest experts fire >16× average in Llama-4 layer 7; patterns
    are task- and language-dependent (different MMLU subjects and C-MMLU pick
    different hot sets → frequency pinning must be domain-aware or request-scoped).

### 2.4 Architectural pre-gating (changes the model, cited for completeness)

- **Pre-gated MoE** (arXiv:2308.12066, ISCA'24): train the gate at block N to predict
  block N+1's experts → prefetch is always one block ahead, exactly hidden. 1.7× vs
  on-demand, 81% of GPU-only throughput at 23% of GPU-only peak memory; accuracy
  preserved after fine-tuning (Switch top-1-of-128). We can't retrain, but it proves
  the mechanism: **one block of lookahead is enough to fully hide H2D latency** —
  consistent with our own timing math.
- **Read-ME** (arXiv:2410.19123): "router-decoupled" MoE — same pre-gating family,
  pipelines exact-expert loads; confirms the architectural direction.

### 2.5 What balance costs predictability — the honest ceiling

Fiddler's appendix C is the most honest measurement here: Mixtral-8x7B expert
popularity is nearly flat (normalized mean 0.71, σ=0.08, min 0.22, only 15/256 <0.6).
Static popularity pinning therefore gives only **25.2% vs 21.9% random** at 22%
capacity, **53.0% vs 48.8%** at 49% capacity — ~3–5 pp. ProMoE's Figure 5 shows the
same: Switch/NLLB access is power-law, modern decoder-only MoEs are near-uniform —
**by design** (auxiliary balance losses, device-limited routing, bias-corrected
aux-free balancing à la DeepSeek-V3). Our model's `use_gate_bias` is almost certainly
the aux-loss-free balancing trick → expect flat-ish global histograms, and expect
static pinning to do little. The exploitable skew is **conditional and request-scoped**:
within one request (MoE-Infinity: one expert reused 15+× in Mixtral's last layer for a
single sequence — skew vanishes across >1000 requests), within one domain, within the
cross-layer/cross-token correlations above.

Coverage numbers people actually report: no paper cleanly reports "x% of experts =
90% of traffic" for Qwen3/DeepSeek — the honest statement is that global coverage is
near-linear (flat), while **request-level** coverage is sharply sublinear (small
working set serves most of a single decode). BAA's profiling of Qwen3-235B (128
experts, like us) classes ~90% of expert-instances as "standard" tier, only ~6.6%
prunable at <0.05% frequency — balanced, with a modest tail; the 512-expert Qwen3.5
has a much fatter tail (18% prunable). Conclusion: **more experts = flatter but more
predictable-conditional; our 128 is in the middle of this trade.**

---

## 3. The loop block — our unique 3× reuse window

No published work covers looped MoE blocks (weight-sharing across passes). Closest
analogies and how they transfer:

1. **Cross-layer prediction ≈ cross-pass prediction.** The 12 loop layers visited 3×
   give 36 routing events on 12 weight pools. A correlation table
   P(e_{L,p+1}|e_{L,p}) is the direct analog of ST-MoE's layer table — likely
   *stronger* signal since the router and expert pool are literally the same weights,
   and the rank-64 controller only modulates magnitude. Free measurement: log
   (layer, pass) → expert histograms on ~1k generations and check the diagonal
   (same-expert-refire-across-passes rate). If ST-MoE's 85% holds here, passes 2–3
   are ~85% predictable during pass 1 — i.e., **~11 layer-steps of prefetch runway
   per loop layer, 3–10× more than the single-step lookahead non-loop layers get.**
2. **Residual-gate works across the pass boundary too.** Predict next pass's gate
   from the current pass output — the hidden state is continuously updated by the
   rank-64 gain/forget controllers, so it drifts, but one pass ≈ one layer of drift.
3. **Pin-or-die for the loop pool.** While inside the block, an expert activated in
   pass 1 has its next opportunity ~11 layer-steps away (pass 2) and ~22 away
   (pass 3). Naive LRU over a shared pool will evict it under the intervening
   ~11 × 8–24 unique experts unless the cache is explicitly pass-aware: tag loop
   experts with the pass they were loaded for, and never evict a loop-pass expert
   during the same token's remaining passes (a ~60–100 expert ≈ 270–450 MB "loop
   working set" per token worst case — fits easily).
4. **Union-bound for the loop block.** Across 3 passes, a loop layer touches up to
   ~24 unique experts (or ~16–24 per pass under verify — see §4). At 4.5 MB that's
   ≤ ~110 MB/layer — prefetching the *union* of predicted experts for all 3 passes
   during pass 1 costs ~2.4 ms of PCIe at 50 GB/s worst case, hidden under 36
   layer-steps of compute. Even a union-of-3-prefetch (≈24 experts/layer) is
   affordable and converts nearly all loop traffic into hits.
5. **Prefill fills the loop cache for free.** Prefill activates experts near-densely
   anyway; harvest the loop-block histogram during prefill (MoE-Infinity's EAM
   trick) to seed per-layer hot sets for decode pass 1.

## 4. Spec-decode interaction (MTP head exists — `mtp.safetensors`)

- Verify with *g* draft tokens raises per-layer-step demand from top-8 to
  ≤ min(8·g, 128) unique experts. The prompt's estimate (~16–24 unique of 128 at
  g≈4–5) matches what cross-token correlation predicts: adjacent tokens share
  experts (Patterns Ob2 diagonal, strongest in higher layers — our FP8 tail layers
  20–23 are exactly those).
- **SpecMoEOff** (arXiv:2508.21706) — the most relevant result: speculative decoding
  under offloading *enlarges the workload per fetched expert* (more tokens per
  expert per fetch → better bandwidth amortization), giving 2.5× decode throughput
  over SOTA MoE offloading and 17.7× over DeepSpeed-ZeRO-Inference. Key phrase:
  offload systems are I/O-bound; SD makes each byte of transferred weight serve
  more tokens.
- **SP-MoE** (arXiv:2510.10302): draft model's routing decisions predict the verify
  pass's experts — structural correspondence between draft and target. Cutoff-layer
  policy bounds prefetch depth so transfers stay just-in-time; 1.07–3.5× TPOT.
- **MoE-SpeQ** (arXiv:2511.14102): same idea — on-device draft predicts expert
  sequence ahead of tokens; 2.34–3.3× on Phi-MoE (536.7→163.1 ms/token).
- **MoE-Spec** (arXiv:2602.16052): cap unique experts per layer during verify
  (drop the long tail) — 10–30% over EAGLE-3 at "comparable quality." Evidence that
  mild expert dropping during *verify* is tolerable — the dropped experts serve
  rarely-hit draft tokens that were going to be rejected anyway. For us: budget
  verify fetches at ~12–16 experts/layer-step rather than the full union.

## 5. Caching policy — what the evidence says to build

### 5.1 Candidate policies, ranked by evidence

| Policy | Evidence |
|---|---|
| **Static frequency pinning** | Fiddler: +3–5 pp over random on Mixtral only; works better on skewed models (Switch/NLLB) — weak for balanced routers. Still worth it for the prefill-harvested *per-request* hot set (Patterns Ob3). |
| **Global LRU** | Eliseev & Mazur's baseline — "significantly speeds up" decode; ProMoE Fig.5 shows LRU hit rate is poor on modern MoEs at <50% capacity because access is flat. At our ~75% capacity LRU is already decent — it becomes the floor to beat, not the design. |
| **Request-scoped trace cache (EAM/MoE-Infinity)** | Exploits within-request skew; 3.1–16.7× TPOT on A5000-24GB/PCIe4 vs vLLM/Ollama/DeepSpeed/BrainStorm. Directly applicable: keep a per-request activation matrix (cheap: 48×128 bitset) and bias eviction by "fired in this request" + "predicted to fire again." |
| **Score-aware caching (HybriMoE MRS)** | Uses *router scores*, not just binary activation: experts with high gate scores that missed top-k still have elevated reuse probability — accumulate top-p scores per iteration, evict lowest-priority. Free signal for us: router already computes all 128 scores + bias + softcap. 1.33×/1.70× over kTransformers. |
| **Learned-predictor cache (ProMoE)** | Prefetch list = predictor's top-k; cache = whatever the prefetcher already fetched; misses handled by "early preemption" (at gate time, enqueue real top-k at high priority — gate reveals all 8 at once, so fetch misses *during* the layer's own attention/expert compute rather than when each expert is read). |
| **Predicted-importance / GoodPred-ranked** | ProMoE's GoodPred = accuracy × fetched-in-time rate — the right objective: don't prefetch what won't arrive in time. |
| **Impact-driven prefetch (HybriMoE)** | When multiple future layers compete for PCIe bandwidth, prefetch the expert whose absence costs most (simulate marginal latency saved). For us: fp8 experts cost 1.83× to move → deprioritize vs loop-block experts (which can hit 2 more times). |

### 5.2 Recommended composite policy for Inkling

- **Pool accounting**: track experts as (layer, id) slots — 4.5 MB fp4, 8.2 MB fp8;
  per-layer budgets proportional to measured per-layer skew and reuse distance
  (loop layers get a "pass-persistence" flag).
- **Eviction priority** (composite score, cheapest first):
  `evict_score = w1·staleness + w2·(1 − predicted_reuse) + w3·(1 − request_freq) + w4·fp8_size_penalty`
  with a hard rule: never evict a loop-layer expert between its first and last pass
  of the current token, and never evict the exact experts needed ≤1 layer-step ahead.
- **Prefetch order**: (a) real top-8 of next layer if residual-gate already fired
  (highest — it's "verified"), (b) predicted top-8..16 of layer L+1 from
  residual-gate/correlation-table, (c) loop pass p+1 predicted union, (d) tail fp8
  experts if bandwidth idle.
- **Granularity**: fetch in per-matrix chunks (w13 = 2.98 MB, w2 = 1.50 MB fp4) —
  ProMoE's chunked prefetching: high-priority miss preempts after ≤1 chunk (~30–60 µs)
  rather than a whole expert; also lets us early-start an expert's w13 GEMM while w2
  still streams (fuse-dequant pipeline — see §6).
- **Early preemption** (ProMoE): the router output reveals *all* required experts at
  once at gate time — enqueue real misses immediately so they stream during that
  layer's attention, not when the grouped GEMM needs them.
- **Miss execution**: run truly-late misses on CPU (Fiddler choice) iff CPU GEMV
  finishes before the PCIe transfer would — threshold at expert bytes ≈ PCIe_time ↔
  DDR5_time ≈ for us ~4.5 MB fp4 is roughly break-even; for fp8 tail experts CPU
  wins (~75 vs ~165 µs) *if* DRAM isn't saturated by prefetch. With 48 GB RAM,
  keeping the full 15.7 GB expert set in RAM is free — the SSD path only matters
  for cold-start mmap or RAM-starved configs.

### 5.3 Capacity sizing for our budget

With ~9–11 GB pool: pin ~600 hottest cross-request experts (measured), reserve
~1,200 slots for the loop-block working set + per-request LRU, keep ~600–1,000
slots as prefetch landing space (you must be able to land a prediction without
evicting what you're about to need — ProMoE's lesson). Rule of thumb from the
literature: at ≥70% capacity, dumb caches look good; the differentiator is
*prefetch hit-timeliness*, not hit rate.

## 6. Grouped/batched GEMM at our expert sizes

Per-expert GEMM shapes (token-major): A[M,2304] × B[2304→2304] (w13 = fused gate+up)
then A[M,1152] × B[1152→2304] (w2). Decode M per expert = #tokens routed to it:
batch-1 → 1; verify g≈5 → up to ~5 (rarely, most get 0–2); prefill → hundreds.

- **Arithmetic intensity**: batch-1 decode AI = 2·M·N·K / (weight bytes) ≈
  2·1·2304·2304 / 2.98e6 ≈ **3.6 FLOP/B** — deep in the bandwidth-bound regime on
  any GPU. Compute efficiency of the tile is irrelevant; what matters is
  *dequant-fused streaming* — read packed fp4 + E4M3 scales, dequant in registers,
  dp4a/dot-accumulate, never materialize bf16 weights. The NVFP4 layout we have
  (16-elem E4M3 blocks + per-expert FP32 outer scale) is exactly what
  Marlin/AWQ/exllama-style kernels and DeepGEMM consume.
- **moe_align_block_size** (vLLM `fused_moe`, SGLang, now upstreamed CUDA kernel):
  sorts (token,expert) pairs into BLOCK_M-aligned segments so each GEMM tile maps
  to one expert (`expert_ids` per block). At decode M=1–8 per expert, BLOCK_M=16
  pads ~50–95% of tile rows — pure waste *on tensor cores*, fine *on bandwidth*.
  For our decode path: **skip the aligned-grouped-GEMM entirely; use a
  per-expert GEMV/scatter kernel** (like llama.cpp's `mul_mat_id` path) or masked
  grouped GEMM with BLOCK_M=8/16 only when verify batches exist.
- **DeepGEMM** (deepseek-ai/DeepGEMM): M-grouped contiguous + masked layouts,
  fixed N,K per group — exactly our case (all experts same shape). Its scheduler
  skips empty groups; SM90/SM100 only. For gfx12/RDNA4: llama.cpp's Vulkan
  `mul_mat_id` + MoE-aware tile selection (issue #29182/#29892 — a 12% prefill
  regression when tile size mismatched n_per_expert=32) shows the knob that
  matters: **tile height vs expected tokens-per-expert** (hot experts get
  >avg rows). hipfire's gfx12 grouped-WMMA MoE kernels and RDNA4
  `v_dot4`/`coopmat` paths show the pattern for our target GPU.
- **Is top-8-of-128 / i=1152 "too small"?** For decode: irrelevant — we're
  bandwidth-bound; the GEMM is ~4.5 µs of HBM time vs ~90 µs of PCIe. For prefill:
  per-layer expert load M̄ = T·8/128; at T=2048 → M̄=128 rows/expert — a clean
  BLOCK_M. N=2304/K=2304 tiles fine (18×18 blocks at 128²). **Verdict: our experts
  are small in bytes (good for streaming) but the GEMM shapes are normal —
  nothing below the tile granularity except M in decode, where we shouldn't use
  tensor cores anyway.**
- **Pipelined chunk-exec**: fetch w13 chunk → start gate/up GEMV while w2 streams;
  a ~90 µs transfer hides under ~10–20 µs of compute per chunk only if we also
  overlap *other* layers' fetches — the real pipeline is inter-layer (see §7).

## 7. Systems comparison — the pattern that wins

| System | Venue | Idea | Best-fit lesson for us |
|---|---|---|---|
| Mixtral-Offloading (2312.17238) | 2023 | LRU + residual-gate speculative load | baseline predictor: free, ~80–90% |
| Pre-gated MoE (2308.12066) | ISCA'24 | trained pre-gate, 1-block lookahead | 1 block of lead = full hiding |
| SiDA-MoE (2310.18859) | MLSys'24 | LSTM predicts all layers from embeddings | "predict the whole itinerary" for the loop |
| MoE-Infinity (2401.14361) | ICML'25 | request-scoped EAM trace cache | per-request skew is the only durable skew |
| Fiddler (2402.07033) | ICLR'25 | popularity pin + CPU-exec misses + batch-aware GPU/CPU split | CPU is a scheduling tool, not a pillar |
| ProMoE (2410.22134) | SOSP'25 | learned predictor + stride prefetch + chunked + early-preempt | the full prefetcher playbook — copy it |
| MoE-Lightning (2411.11217) | ASPLOS'25 | CGOPipe + hierarchical roofline, paged weights | our scheduler should be roofline-driven |
| Klotski (2502.06888) | ASPLOS'25 | expert-aware multi-batch pipeline, 85× throughput | batching buys overlap; for offline mode |
| HybriMoE (2504.05897) | DAC'25 | dynamic intra-layer CPU/GPU split + MRS score cache + impact prefetch | score-aware eviction; load-balance CPU/GPU |
| ExpertFlow (2410.17954) | 2024 | all-layer transformer predictor + token grouping | batch tokens by predicted route |
| SpecMoEOff (2508.21706) | 2025 | SD enlarges per-fetch workload | our MTP is free bandwidth amortization |
| SP-MoE (2510.10302) | 2025 | draft-predicted verify experts + cutoff depth | draft-gate predicts verify experts |
| MoE-SpeQ (2511.14102) | 2025 | same, quantized | 2.34–3.3× |
| MoE-Spec (2602.16052) | 2026 | expert budget during verify | cap verify fetches ~12–16 |
| DuoServe-MoE (2509.07379) | 2025 | phase-specialized prefill/decode | prefetch all in prefill, predict in decode |
| ST-MoE (2606.15453) | 2026 | correlation+history tables, no training | 85% accuracy, zero infra — build first |
| DALI (2602.03495) / Apex (2610.23006) | 2026 | residual prefetch + adaptive CPU/GPU assignment of misses | assign each miss to cheaper of transfer-vs-CPU |
| kTransformers (SOSP'25) | 2025 | AMX CPU kernels, expert deferral → ~100% CPU util | CPU kernel quality decides if CPU-exec is worth it |
| llama.cpp `--cpu-moe`/`-ot` | practice | experts-in-RAM convention | the de-facto baseline we must beat |

**Consensus skeleton** (nearly every post-2024 system): router reveals all k experts
at once → prefetch predicted set during previous layer → on gate, re-prioritize real
misses (early preemption) → chunked transfers let misses preempt fast → CPU executes
what can't arrive in time → per-request trace + score-aware eviction decides what's
resident. Our additions: loop-pass persistence, fp8-tail deprioritization,
verify-aware budget caps, and residual-gate-as-predictor since our gate is one
[128×2304] matvec (~free) with softcap — run it speculatively every layer.

## 8. Quality: does dropping or approximating experts break the model?

Measured evidence (most useful papers in the whole survey for a "run fewer experts"
knob):

- **Training-Free Halving of Activated Experts** (arXiv:2609.04575): the
  renormalization denominator, not expert capacity, is what breaks. Fine-grained MoEs
  renormalize router probs over top-k; Qwen3.6-35B's top-8-of-256 mass is only 0.182
  → renorm amplifies ~5.5×. Activate top-k1 but normalize by top-k2 mass:
  - Qwen3.6-35B-A3B, 8→4 experts: **−4.65 MMLU pts naive, −0.35 pts at k2=16**
    (p=0.66, indistinguishable); WikiText +4.08% ppl naive, ~+0% with proper k2.
  - Qwen3.5-397B-A17B, 10→5: −2.10 naive, **−0.55 at k2=10**.
  - Removing renorm entirely (k2=E): catastrophic (−27.5 MMLU, +356% ppl).
  - *Which* experts matters ≫ weights: uniform weights on the right experts +49.7%
    ppl; right-weighted wrong experts (random selection) +1667%.
- **Independent measurement** (hackbig.com, Qwen3.6-35B-A3B): k=8→6: +3.3 GSM8K pts
  (noise), k=4: −8.0 pts, k=2: −49.3 pts — **with naive renorm**, consistent with the
  above. Decode speedup only 1.07–1.15× because routed experts are <half the bytes in
  that model; for us experts are 80% of bytes → the same drop buys ~1.5–1.9× on the
  streaming-bound portion.
- **DERN** (arXiv:2509.10377, EMNLP'25): permanent pruning — drop experts + neuron-
  level recombination of survivors, retraining-free; >5% avg drop at 50% sparsity on
  Mixtral/Qwen/DeepSeek; an 8→6-expert variant matched accuracy at −25% memory,
  +18% throughput. Pruning whole experts ≠ our use (we keep all weights, just fetch
  lazily), but it bounds the redundancy budget.
- **"Don't Skip the Baseline"** (arXiv:2610.28369): dynamic expert-skipping schemes
  rarely beat matched-budget *static* k reduction — a warning against cleverness.
- **MoE-Spec** (§4): dropping the long tail during verify: 10–30% throughput, no
  reported quality regression — the cheapest place to drop is inside spec-decode.

**Applied to Inkling**: our gate has `softcap=4.0` (tanh cap on logits), learned bias
(aux-free balancing), `route_scale=8.0`, and almost certainly renormalizes over
top-8 (GPT-OSS-style softmax→top-k→renorm lineage). If so, the k2 trick applies
directly: **top-8→top-6 with k2≈8–12 should be nearly free; top-8→top-4 with k2≈12–16
is the ~2× bandwidth lever at ~0.3–0.6 pts cost**. Must verify (a) the renorm exists
in our router, (b) eval MMLU/GSM8K at (k1,k2) ∈ {(8,8),(6,10),(6,12),(4,12),(4,16)}.
This is the single cheapest decode-speedup knob in the whole system — it multiplies
every other optimization (fewer fetches, fewer misses, fewer GEMV rows).

## 9. Synthesis — the streaming engine we should build

1. **Layout**: experts stay packed NVFP4 w13+w2 (+scales+amax) in one 4.5 MB blob;
   fp8 tail experts as second pool. RAM mirror = mmap'd checkpoint (cold-start
   = page cache, no parse).
2. **Per-decode-step schedule**:
   - Layer L: launch attention; **speculatively run gate_{L+1} on h_L input**
     (and inside the loop, gate for pass p+1 on pass p output);
   - enqueue predicted experts of L+1 (+loop pass unions) at low priority;
   - when gate_L fires → promote real misses to high priority (early preemption),
     chunked per-matrix;
   - expert exec = dequant-fused GEMV on arrival; truly-late → CPU kernel
     (fp8 experts preferentially; fp4 if PCIe busy).
3. **Cache**: per-layer budgets; composite eviction (staleness, predicted reuse,
   request-freq, size penalty); **loop-pass pinning**; prefill-histogram seeding;
   prefetch-landing headroom.
4. **Verify path**: draft-gate → predicted verify experts (SP-MoE), budget-capped
   at ~16/layer; SD batching amortizes each fetched expert ~2–4× (SpecMoEOff).
5. **Prefill**: stream everything through once (dense activation); harvest EAM.
6. **Measure before building**: log (layer, pass, request-domain) → expert-id for
   ~1k generations; compute (a) residual-gate accuracy per layer/pass, (b) cross-
   pass correlation diagonal, (c) request-level coverage curve, (d) LRU-75% hit
   baseline. These four numbers decide how fancy the predictor must be — the
   literature predicts ~80–90% for (a) and (b), flat-but-request-skewed for (c).

Open questions to verify in-repo: (i) exact pass-order of the loop (block- vs
layer-sequential); (ii) does the loop gate include the rank-64 controller output —
i.e., is routing per-pass different by a lot or a little; (iii) router
renormalization semantics for the k2 trick; (iv) real PCIe/DDR5 bandwidth on the
target box (run a pinned-buffer H2D bench — papers quote 24 GB/s for PCIe4-real,
~50–55 GB/s PCIe5).

---

## 10. Sources

1. Fiddler — https://arxiv.org/abs/2402.07033 (ICLR'25; code: github.com/efeslab/fiddler)
2. MoE-Infinity — https://arxiv.org/abs/2401.14361 (ICML'25; github.com/EfficientMoE/MoE-Infinity)
3. Pre-gated MoE — https://arxiv.org/abs/2308.12066 (ISCA'24)
4. ProMoE — https://arxiv.org/abs/2410.22134 (SOSP'25; github.com/promoe-opensource/promoe)
5. Mixtral-Offloading — https://arxiv.org/abs/2312.17238
6. SiDA-MoE — https://arxiv.org/abs/2310.18859 (MLSys'24)
7. Klotski — https://arxiv.org/abs/2502.06888 (ASPLOS'25)
8. HybriMoE — https://arxiv.org/abs/2504.05897 (DAC'25; github.com/PKU-SEC-Lab/HybriMoE)
9. MoE-Lightning — https://arxiv.org/abs/2411.11217 (ASPLOS'25)
10. ExpertFlow — https://arxiv.org/abs/2410.17954 (impl: github.com/expertflow-dac/expertflow; cf. 2510.26730)
11. SpecMoEOff — https://arxiv.org/abs/2508.21706
12. SP-MoE — https://arxiv.org/abs/2510.10302
13. MoE-SpeQ — https://arxiv.org/abs/2511.14102
14. MoE-Spec — https://arxiv.org/abs/2602.16052
15. DuoServe-MoE — https://arxiv.org/abs/2509.07379
16. ST-MoE — https://arxiv.org/abs/2606.15453
17. Patterns behind Chaos — https://arxiv.org/abs/2510.05497 (ISCA'26; traces: huggingface.co/datasets/core12345/MoE_expert_selection_trace)
18. Scaling Multi-Node MoE via Expert Activation Patterns — https://arxiv.org/abs/2604.23150
19. Inter-Layer Expert Affinity — https://arxiv.org/abs/2401.08383
20. Read-ME (router-decoupled MoE) — https://arxiv.org/abs/2410.19123
21. DERN (expert pruning) — https://arxiv.org/abs/2509.10377 (EMNLP'25)
22. Training-Free Halving of Activated Experts — https://arxiv.org/abs/2609.04575; companion blog https://raincchio.github.io/blog/moe/
23. Don't Skip the Baseline (expert skipping) — https://arxiv.org/abs/2610.28369
24. Hackbig: cutting Qwen3.6 experts measured — https://hackbig.com/blog/qwen36-moe-fewer-experts
25. BAA: Profiling Expert Activation Patterns in 512-Expert MoE Models — https://baa.ai/research/01-profiling-expert-activation-patterns.html
26. KTransformers — https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf (SOSP'25)
27. DeepGEMM — https://github.com/deepseek-ai/DeepGEMM
28. vLLM moe_align_block_size / fused_moe — https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/fused_moe/moe_align_block_size/ ; https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py ; Triton grouped-GEMM explainer https://vidyal.ai/blog/moe-kernels/
29. llama.cpp `--cpu-moe` practice — https://dreaming.press/posts/run-671b-moe-single-gpu-ktransformers-vs-llama-cpp.html ; RDNA4 MoE tile issue https://github.com/ggml-org/llama.cpp/issues/29892 ; gfx12 grouped-WMMA MoE https://github.com/warpfront/hipfire
30. NVFP4 format — https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html ; https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
31. DALI — https://arxiv.org/abs/2602.03495 ; Apex — https://arxiv.org/abs/2610.23006
32. EAQuant (power-law claim, quant side) — https://arxiv.org/abs/2506.13329
33. Speculative expert loading re-impl — https://arxiv.org/abs/2511.05814
