# Staged-visit latency: measured data and the warm-cache plan

Measured on RX 9070 XT (16GB), `--backend gpu`, 2025-*. All numbers from
INK_GPU_TS=1 and INK_IDS router dumps (`--backend vk`, 25 prompt + 8 gen).

## Where the time goes (per token)

- gpu-busy ~120.6ms/token; host record ~65ms (agent 1e6f000f is removing this
  via record-once params buffer); wait ~56ms.
- The top-25 INK_GPU_TS stages are ALL `*/submit_gap copies`: ~21 staged
  (host-resident expert) visits × ~2.7–5.1ms ≈ **~63ms/token of serial
  staging round-trip**. mxfp8 coda visits v44–47 ~5ms each; nvfp4 loop
  visits ~2.8ms each. All other stages <0.3ms.
- Per staged visit the gap = fence(visit_a readback) + host ids read +
  record ~32 copies + submit transfer cb + wait for copies. Actual
  vkCmdCopyBuffer time is only ~0.8ms of the ~3ms — the rest is host
  round-trip + submit latency.

## Router id overlap (INK_IDS dump, 8 gen tokens)

- Same-visit prev-token top-8 set overlap: **43.4%** (~3.5/8 experts reused).
- Same-layer cross-pass within a token: 5.6% — passes route independently,
  so caches must be per-VISIT (48 visits), not per-layer.
- Union of ids per visit over 8 tokens: ~29/128 — modest pool per visit.
- (An earlier estimate of 13% was wrong; measured directly here.)

## Design: warm per-visit slot caches + transfer-queue prefetch

1. Each staged visit gets a persistent 8-slot VRAM cache holding
   {w13,w13s,w2,w2s} slices for the LAST token's routed experts
   (~25MB/visit × ~21 visits ≈ 525MB — budget against resident-expert space).
2. At token start (or better: during the previous token's tail), the host
   issues prefetch copies for each staged visit's previous-token ids onto
   the transfer queue — the copy engine is otherwise idle during compute.
   ~43% land as hits for the real ids.
3. On the visit_a fence: read real ids; copy ONLY misses (~57% ≈ 14MB ≈
   0.45ms); hits are already in the right slot iff slot assignment is kept
   positional — i.e. prefetch writes slice for last_ids[k] into slot k, and
   misses re-fill slot k with the new id. (Reorder so hit slots keep their
   expert; or add a small id→slot remap the stage shaders read — they
   already index by slot, so simplest is: slot k always serves position k,
   prefetch guesses last_ids[k]→slot k, misses overwrite.)
4. Semaphore-sync the copies into visit_b (agent A's work) — no host wait.

Expected staged-visit cost: visit_a (~1.5ms) + fence ~0.3ms + miss copies
~0.5ms + visit_b ≈ gap ~1ms vs ~3ms now → saves ~2ms × 21 ≈ **~40ms/token**.

## Secondary levers (same area)

- Residency reprioritization: a loop layer (appears in 3 visits) saves
  ~8.4ms when resident vs a coda layer's single ~5ms visit. Weight the
  heap-budget residency choice by visit count, not layer order.
- Optional: replace vkWaitForFences for ids with mapped-memory polling on
  a device-written sentinel (saves ~0.2-0.4ms × 21).
- Rejected: device-side gather shader for all slices — host-MRd ~8GB/s
  makes a 25MB gather ~3.1ms, worse than ~0.45ms of warm-miss SDMA.
  Persistent-cache idea previously rejected at 13% overlap is now viable
  at 43.4%.
