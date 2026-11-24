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

## What shipped: per-visit persistent w13 buffers (no prefetch)

Measured iterations (RX 9070 XT, submits≈392 config ≈ 11–12 staged
visits; copies class = GPU time in `*/submit_gap copies` stages):

- **Prefetch killed by traffic, not hits.** Same-queue prefetch of last
  token's ids: copies 27.15ms, gpu-busy ~77ms. Second SDMA queue +
  3-buffer rotation + prefetch 2-ahead: gaps ~1ms but copies 31.63ms,
  gpu-busy 81.32ms — prefetch ADDS ~56% PCIe traffic which itself loads
  the link, and the in-order transfer queue overruns the ~2ms boundary
  window anyway.
- **43.4% is SET overlap, not positional.** Router membership is stable
  but slot ORDER shuffles: positional hit rate measured only ~10%
  (`hit=232 miss=2072` in the prefetch run). Positional prefetch is
  worthless on its own.
- **Fix: cross-slot dev copies, zero prefetch.** Each staged visit keeps
  a private 8-slot w13w/w13s pair (~15MB/visit) that retains last token's
  slices. On the ids readback, each slot resolves: same slot+expert → no
  copy (hit); expert resident in another slot of that visit's buffer →
  device-local `buf→buf` copy (~25µs vs ~100µs PCIe, dev); cold → host
  mirror pull (miss). Dev copies are emitted in threatened-src-first
  order so every source still holds resident data. No extra PCIe traffic
  at all — the 40% coverage is free.
- **Result** (16 tokens): `warm{slots=3072 hit=288 dev=954 miss=1830}` =
  40.4% non-PCIe coverage — the full set-overlap. copies **16.80
  ms/token** (baseline 24.04, -30%), gpu-busy **66.68 ms/token** (~69–73
  baseline), worst submit_gap for copies 1.78ms. Gate: worst_cos=0.989611.
- `INK_NOWARM=1` disables (per-visit allocs skipped, visits fall back to
  the shared staging buffer). Per-visit sets are allocated greedily in
  init; a visit that doesn't fit stays on the shared buffer.
- w2/w2s are NOT cached: they stage in pass2 of the same boundary, ~5ms
  after the ids are read — plenty of in-window time for plain copies, and
  doubling the cache size for them isn't worth it.

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
