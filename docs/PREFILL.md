# PREFILL — batched-chunk prompt processing for `--backend gpu`

Status: **implemented and gate-verified.** `--prefill-chunk 8` passes
`tools/compare_golden.py` (worst hidden cos 0.9896, worst logits cos
0.99955, all 8 argmax match, expected output text). Design +
implementation notes for `src/gpu/gpu_prefill.{h,cpp}` and the
`shaders/*_seq.comp` kernels. The CPU reference is
`src/model/graph.cpp`; the single-token GPU mirror is `src/gpu/gpu_graph.cpp`.

## 1. Problem

`--backend gpu` replays one prerecorded command buffer per token
(`GpuGraph::forward_token`): ~1000 dispatches + ~200 copies + one
`submit_and_wait` per position. A P-token prompt therefore costs P sequential
submits (~0.3 s each on this box ⇒ ~7.5 s for P = 25), and every dense weight
GEMV re-reads its matrix once per prompt token.

## 2. Key observation (why chunking is exact)

Inside `Graph::forward_token(token, pos)` every per-position value depends
only on:

- `embed[token]` (row's own token id),
- the *persistent* per-(layer,pass) state written by earlier positions:
  `cache.kv` rows (absolute position indexing) and `cache.conv[4]` last-3
  input history per stream,
- per-token state rebuilt inside the token (`block`, `partial`, `bank`,
  `pre_block`, `pre_partial`, `msgs`) — **all reset/restored inside the
  token**, never carried across positions.

So for a chunk of T tokens at positions `pos0 .. pos0+t-1`:

- rows are independent given the boundary state (kv rows `pos0+t-1` written
  by earlier rows *of the same chunk* are only consumed by attention, which
  masks them — a row-`t` query attends `j <= pos0+t`),
- the ONLY cross-row inputs inside a chunk are:
  - **KV history**: row `t` attends `kv[j], j <= pos0+t` — handled by
    appending all T k/v rows first and masking scores causally,
  - **conv taps**: row `t` needs raw inputs `x[t-1..t-3]` — for `t-lag < 0`
    these are the last inputs of the previous chunk (or zeros at pos 0).
    Handled by a persistent 3-row history buffer `h3` per conv stream.

Everything else is embarrassingly row-parallel ⇒ take the existing
single-token op sequence and widen every activation buffer to T rows.

## 3. Layout: T-row mirrors of the single-token state

`H = 2304`, `T <= Tmax` (16). All f32, device-local unless noted:

| GpuGraph (1 row)      | GpuPrefill (T rows)                | layout |
|---|---|---|
| `partial_`  | `partial_seq_` | `[T][H]` |
| `block_`    | `block_seq_`   | `[T][5][H]` (4 pushes + append row) |
| `pre_block_`| `pre_block_seq_`| `[T][4][H]` |
| `pre_partial_` | `pre_partial_seq_` | `[T][H]` |
| `bank_`     | `bank_seq_`    | `[T][2][H]` {block.back, prev partial} |
| `reads_`    | `reads_seq_`   | `[T][3][H]` attn q/k/v reads |
| `msgs_`     | `msgs_seq_`    | `[T][4][H]` pass>0 messages (q,k,v,mlp) |
| `mgated_`   | `mgated_seq_`  | `[T][3][H]` stream-gated inputs |
| —           | `mlpegated_seq_` | `[T][H]` gated mlp input |
| `inbuf_`    | `inbuf_seq_`   | `[T][3][H]` normed streams (xq,xk,xv) |
| `q_,gate_,o_` | `q_seq_,grate_seq_,o_seq_` | `[T][3072]` |
| `k_,v_` raw | `kraw_seq_,vraw_seq_` | `[T][1024]` pre-conv |
| —           | `kout_seq_,vout_seq_` | `[T][1024]` post-conv (clip in place) |
| `r_`        | `r_seq_`       | `[T][384]` |
| `tk_,tv_`   | `tk_seq_,tv_seq_` | `[T][32]` dynamic conv taps (full) |
| `rel_`      | `rel_seq_`     | `[T][24][1024]` |
| `probs_`    | `probs_seq_`   | `[T][24][pstride]`, `pstride = N+1` |
| `sink_`     | `sink_seq_`  | `[T][24]` |
| `attn_out_,a_out_,residual_` | `*_seq_` | `[T][H]` |
| `m_in_,mn_,moe_out_,mo_` | `*_seq_` | `[T][H]` |
| `ridx_,rw_` | `ridx_seq_,rw_seq_` | `[T][8]` i32 / f32 |
| `gate_,up_` | `gate_seq_,up_seq_` | `[T][8][1152]` slot-major |

Persistent cross-chunk state (shared with / exported to GpuGraph):

- `kvk_/kvv_` — **the GpuGraph buffers**, rows `pos0..pos0+t-1` appended per
  chunk at slot slices `kvb(slot)`.
- `h3_seq_` `[48*4][3*H]` — NEW row-major `[3][C]` last-3 *raw* inputs per
  (slot,stream). `h3[s][0]=pos0-3 … h3[s][2]=pos0-1`; zero = cold start.
- `hist_` — GpuGraph's channel-major `[C*3]` conv state; written
  (transposed) by the conv state-dispatch so decode can continue with the
  existing `conv4_*` single-token shaders.
- `toks_tau_` — HOST-visible `[Pmax]i32 tokens ++ [Pmax]f32 taus`; bound
  directly (execute-time contents); `tau[p]` computed host-side in double,
  identical to `attn_step`'s log-scaling formula.

## 4. Shader plan — reuse vs `*_seq`

Elementwise kernels over contiguous `[T*n]` rows batch trivially (n scaled):
**vadd, sigmoid_mul, softcap** — reused as-is.

`rmsnorm` is *already* row-batched (`grid.x = n_rows`, `x[row*n]`); reused
for contiguous-row cases (v-head norms `T*8 x 128`, contiguous
`mgated→inbuf`). Strided rows use `rmsnorm_seq` (`xstride/ostride` in PC).

New `*_seq.comp` kernels (all local_size 64 unless noted):

| shader | grid | PC (new, appended to bindings.h) | notes |
|---|---|---|---|
| `embed_norm_seq` | `(T)` | PcEmbedSeq{base,n,eps,flags} | `token=toks[base+row]` |
| `rmsnorm_seq` | `(rows)` | PcRmsnormSeq{n,eps,flags,xs,os} | strided rows |
| `matvec_{f32,bf16,f8e4m3}_seq` | `(M,T)` | PcMatvecSeq{K,m0,xstride,M} | `o[t*M+m]=W[m]·x[t*xstride]` |
| `matvec_nvfp4_seq`/`matvec_mxfp8_seq` | `(I,T)` | +`{ktop,M}` | `e=idx[t*ktop+slot]`, out row `(t*ktop+slot)*M+m` |
| `moe_accum_seq` | `(D,T)` | PcMoeAccumSeq{…,ktop,gstride} | `gate` bound at `+s*I`, row base `t*gstride` |
| `conv4_static_seq` | `(ceil(C/64),T)` + `(ceil(C/64),1)` | PcConv4StaticSeq{C,T,flags} | h3 taps + state shift/hist export |
| `conv4_dynamic_seq` | same | PcConv4DynSeq{C,hd,T,flags} | + `dyn[t*32+h*4+lag]` |
| `attn_res_seq` | `(T)` | PcAttnResSeq{N,D,Q,ss,os} | src row `t*ss`, out `t*os` |
| `msg_gate_seq` | `(T)` | PcMsgGateSeq{D,rank,ds,ms,os} | |
| `router_seq` | `(T)` (ls=128) | PcRouterSeq{E,k,K,scale} | `idx[t*k+s]`, `w[t*k+s]` |
| `attn_scores_seq` | `(24,ceil(N/64),T)` | PcAttnScoresSeq{hq,hk,dh,n,pos0,ext,pstride,swa,flags} | causal+SWA mask, `d=pos0+t-j` |
| `attn_softmax_seq` | `(24,T)` | PcAttnSoftmaxSeq{n,pstride,hq} | row `(t*hq+h)*pstride` |
| `attn_accum_seq` | `(24,T)` | PcAttnAccumSeq{hq,hk,dh,n,pstride} | masked probs are 0 |
| `rel_bias_seq` | `(ceil(24*1024/64),T)` | PcRelBiasSeq{dr,ext,n} | `rel[t*HQ*ext+g]` |
| `l2_clip_seq` | `(heads,T)` | PcL2ClipSeq{dim,thr,flags,hstride} | `v[t*hstride+h*dim+d]` |
| `log_scale_seq` | `(ceil(T*n/64))` | PcLogScaleSeq{n} | `v[i]*=tau[i/n]` from bound buf |

## 5. Per-visit op sequence (row-batched mirror of `emit_visit`)

`emit_visit_seq(i, p, pos0, t)` — identical order to `emit_visit`; every
`op` below is a dispatch, `cp` a recorded copy, `sync` a full barrier:

1. `p>0`: `attn_res_seq` SRC=bank (N=2,ss=2H,Q=4,q=msgq+li*4H) → msgs; sync.
2. append `x_in`: t× `cp partial[r]→block[r][nb0]` (row-major scatter).
3. `i>0`: `attn_res_seq` SRC=block (N=nb0+1,ss=5H,Q=1 or 3,q=attnq+(i-1)*3H)
   → reads; else t× `cp partial[r]→reads[r*3H]`; sync.
4. stream inputs (`uni = !full && p==0` ⇒ xo = {0,0,0} else {0,H,2H}):
   - `p>0`: 3× `msg_gate_seq` (dst=reads+(full?st:0)H ds=3H, msg=msgs+st*H
     ms=4H, out=mgated+st*H os=3H); sync; norm to inbuf:
     full: 3× `rmsnorm_seq` (x=mgated+st*H xs=3H→inbuf+st*H os=3H)
     sliding: 1× `rmsnorm_seq` contiguous T*3 rows mgated→inbuf
   - `p==0`: full: 3× `rmsnorm_seq` reads+st*H→inbuf+st*H;
     sliding: 1× reads+0→inbuf+0.
   sync.
5. projections: `matvec_seq` wq→q (M=3072), wk→kraw (1024), wv→vraw (1024),
   wr→r (384); X=inbuf+xo[st], xstride=3H. sync.
6. convs on k/v (state in `h3[slot*4+{0,1}]`):
   full: wproj matvecs→tk/tv `[T][32]`; 2× `conv4_dynamic_seq`(compute) +
   state dispatch each.
   sliding: 2× `conv4_static_seq`(compute) + state dispatch each.
   out→kout/vout. sync.
7. clips: `l2_clip_seq` q (24×128 scaled), k (8×128), r (24×16 scaled);
   `rmsnorm_seq` v per-head (grid T*8, n=128, unweighted, in-place). sync.
8. `full`: `log_scale_seq` q (n=3072) + r (n=384) with host `tau[]`. sync.
9. KV append: `cp kout→kvk[slot][pos0..pos0+t)` and `cp vout→kvv[…]`
   (single contiguous `t*4096` copy each — absolute row indexing).
10. `rel_bias_seq` r→rel[T*24*1024]; `matvec_seq` gate→grate(3072),
    sink→sink(24); X=inbuf+0. sync.
11. `attn_scores_seq` grid(24,⌈N/64⌉,t), N=pos0+t, pstride=N+1:
    `valid j ⇔ j0_t<=j<=pos0+t`, `j0_t = full?0:max(0,pos0+t+1-1024)`;
    invalid ⇒ `-1e30`; else `q·k/√128 + rel[t][h][pos0+t-j]` (0<=d<ext).
    sync.
12. `attn_softmax_seq` grid(24,t): row (t*24+h), n=N, sink=sink[t*24+h] at
    index N. sync. `attn_accum_seq` grid(24,t) over jj<N → o[t*3072+..].
    sync. `sigmoid_mul` n=T*3072 (o*=σ(grate)). sync.
13. `matvec_seq` wo o→attn_out (M=H,K=3072); sync;
    `conv4_static_seq` C=H stream2 → a_out; sync.
14. boundary/residual: `i%6==0` ⇒ `nblock++`, resid=a_out; else
    `vadd` residual=partial+a_out (n=T*H), resid=residual; sync;
    t× `cp resid[r]→block[r][nblock]`; sync.
15. mlp read: `attn_res_seq` SRC=block (N=nblock+1,ss=5H,Q=1,q=mlpq+i*H)
    → m_in; sync.
16. `p>0`: `msg_gate_seq` dst=m_in ds=H msg=msgs+3H out=mlpegated; sync;
    `rmsnorm_seq` X=(mlpegated|m_in)→mn. sync.
17. `router_seq` grid(t): idx/w `[t*8]`; sync. `fill` moe_out zeros.
18. `for s<8`: 2× `matvec_{nvfp4,mxfp8}_seq` w13 gate/up →
    `gate_seq[(r*8+s)*I]`, `up_seq[…]`; X=mn xs=H; `e=idx[r*8+s]`. sync.
    `for s<8`: `moe_accum_seq` grid(D,t) gate+up bound at +s*I,
    `o[r*D+d] += wt[r*8+s]*W2[idx[r*8+s],d,:]·a`; sync between s
    (serial accumulate, same as decode).
19. `conv4_static_seq` moe_out stream3 → mo; sync;
    `vadd` partial=resid+mo (n=T*H); sync.

`nblock` bookkeeping is host-side, shared by all rows (identical depth).

## 6. Chunk driver (`GpuPrefill::prefill`)

- `for c in chunks` (t = T except tail): same emission as `emit_token`:
  `embed_seq` → visits 0..5 → snapshot (partial→pre_partial;
  block[r]→pre_block[r] per row) → `for p in 0..2` (p>0: per-row bank fill
  {block[nblock-1], partial} + restore pre_*) visits 6..17 → coda 18..23.
- All chunks of a prompt are recorded in **one** command buffer (state is
  device-side; token ids/taus for the whole prompt sit in the host staging
  bound at execute time) → **1 submit for P ≤ 1024**, else one submit per
  1024-token super-block.
- **Last chunk only** (`last` flag): per-visit dump copies of row `t-1` to
  the GpuGraph `readback_` at the same offsets (`tap`, `visit_dump`, dbg
  rows), then the single-row tail on that row: `cp partial[last]→block[last][nblock]`;
  `attn_res_seq`(grid 1)→`h_`; `rmsnorm`→`hfin_`; `matvec_bf16` unembed→
  `logits_`; `softcap`; `cp` hfin/logits→readback; `memcpy` logits→
  `logits_host_`. After this, `gpu_graph.tap()/visit_dump()/logits()` are
  gate-compatible with a sequential run.
- After the submit: `g->kv_len_[*]=P`, `conv_filled_[*][*]=min(3,P)`,
  `nblock_=0` — decode continues byte-identically to P sequential forwards.

## 7. Descriptor/site strategy

Mirrors `GpuGraph`'s dual-mode trick with an independent `sites_` stream:
one build pass (`building_=true`) emits a full `t=T,last=true` chunk to
materialize the fixed desc-set sequence (all bindings are position- and
`t`-invariant — variable quantities ride push constants or grid dims);
each chunk replays `sites_[site_cursor_++]`. Non-last chunks consume a
prefix of the stream; the last chunk consumes it plus the appended tail
sites. No descriptor writes at run time.

## 8. Submit/perf — measured (P = 25, T = 8, ctx 128)

| | sequential | chunked |
|---|---|---|
| submits | 400 (16/token) | **4** (one per chunk) |
| dispatch+copy records | ~25×1200 | ~4×1200 (grids carry T rows) |
| dense-weight reads | 25× ~60MB/visit-set | ~4× (shared across rows) |
| expert host-RAM reads | 200 (8×25) | 200 — unchanged (per-row ids) |
| prompt time (measured) | ~2.5 s GPU (record 1120ms+wait 1411ms) | 1.4–2.3 s wall |

Attention scores do `T*N` work vs sequential `sum_p p` — for small prompts
this is a mild overcompute on the tiny score kernel, irrelevant vs the
submit/dispatch savings. Note the timing range above was measured under
concurrent GPU contention; the clean-run figure is ~1.4 s. The prompt-phase
speedup grows with P (submit overhead amortizes; the per-visit sync barriers
inside a chunk are cheap relative to queue round-trips).

## 9. Correctness risks (checked against graph.cpp)

- Conv history: `h3` zeros reproduce `filled` cold-start gating exactly;
  for `t_c<3` the shift reads old `h3` tail — the state dispatch does
  read-then-write per channel, single dispatch ⇒ race-free.
- Attention mask: `-1e30` masked entries are exact zeros post-softmax;
  every row has ≥1 valid key (j=pos0+t ≥ 0) ⇒ no NaN.
- `tau`: host-double→f32 buffer ⇒ identical to sequential (which computes
  in double on host).
- `log_scale`/`qk-l2 clip`/`v-norm` ordering preserved op-for-op.
- Expert `amax[e]/osd` factor and slot indexing identical.
- Non-boundary residual overwrite of `block[nb0]` preserved (x_in append
  row is transient by design, see gpu_graph.cpp comment).

## 10. Bug found during bring-up (fixed)

The first gate run produced corruption localized to layers 20–23 (the four
MXFP8 expert layers): `gate_seq_`/`up_seq_` held ~1/8-norm garbage
(cos ≈ 0.13) while the `mn` input, router ids and weights were verified
correct bit-for-bit. A sentinel-fill test proved the w13 GEMV dispatched and
wrote to the right addresses; the wrong values were the kernel's own output.

Root cause: `shaders/matvec_mxfp8_seq.comp` wrote the per-thread partial
`acc` instead of the reduced `red[0]` in the final `if (tid == 0)` store —
thread 0's `acc` covers exactly K/64 = 36 of 2304 terms, giving norm ≈
truth/8 (observed 3.15 vs 25.25) and near-orthogonal direction. One-word
fix (`acc` → `red[0]`, matching the sequential `matvec_mxfp8.comp` variants);
no other `*_seq` kernel had the error (verified by grep).
