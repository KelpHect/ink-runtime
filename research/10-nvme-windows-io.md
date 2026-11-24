# A10 — Fastest legal I/O paths: NVMe SSD → RAM → VRAM on Windows 11

**Scope.** Real, measured numbers for getting ~19.5 GB of weights/experts off a
FireCuda 530-class PCIe 4.0 NVMe drive, through DDR5 system memory, into the
16 GB of VRAM on an RX 9070 XT (PCIe 5.0 x16), while compute keeps running.
All rates are measured, not spec-sheet, unless marked "spec".

---

## 0. TL;DR — the answer

1. **The disk is not the problem; the Windows I/O path is.** One thread doing
   unbuffered `ReadFile` tops out around **4.5–5 GB/s** on a Gen5 drive — a
   Gen4 drive saturating at ~7 GB/s needs **two issuing threads with QD ≥ 8**
   each, or DirectStorage/IoRing which batch submissions in the kernel.
2. **Best raw path, any API:** `CreateFile` + `FILE_FLAG_NO_BUFFERING |
   FILE_FLAG_OVERLAPPED` + IOCP, 1–4 MB requests, ~32–64 MB in flight, into a
   ring of GPU-visible staging buffers. Expect **6.5–7 GB/s** from the
   FireCuda 530 (its real ceiling), zero CPU copies, <5 % CPU.
3. **Best "batteries included" path:** DirectStorage 1.3 with
   `DSTORAGE_DESTINATION_BUFFER` writes file→VRAM with **zero CPU copies**
   (NVMe→upload-heap staging→GPU copy engine). Works on AMD via the generic
   DirectCompute path; GDeflate GPU decompression also works on any SM 6.0
   DX12 GPU. It requires a D3D12 device — no Vulkan equivalent exists on
   Windows — and it buys you BypassIO + IoRing underneath for free.
   Demonstrated ~5 GB/s end-to-end on a midrange GPU.
4. **RAM→VRAM:** use the GPU's **copy/DMA engine** (D3D12 copy queue or a
   dedicated Vulkan transfer queue; on AMD these run on SDMA engines),
   from **pinned/GPU-visible staging**. Measured pinned H2D: **~25 GB/s on
   PCIe 4.0 x16, ~50–57 GB/s on PCIe 5.0 x16**. CPU `memcpy` into
   ReBAR-mapped VRAM is ~2–5× slower (~10 GB/s write-combined, and
   **~0.1 GB/s if you ever read it back** — never read VRAM from the CPU).
5. **Memory-mapped files are a cache, not a pipeline.** ~1.2 GB/s
   single-threaded on cold data (demand page faults serialize at disk
   latency), ~7 GB/s multithreaded. Fine as a free OS-managed cold tier
   behind `PrefetchVirtualMemory`; useless for on-demand expert loads.
6. **DDR5 dual-channel is not a bottleneck.** ~85–95 GB/s read / ~70–85 GB/s
   write on i7-13700K-class; the whole pipeline burns ~15–25 GB/s of memory
   bandwidth.

---

## 1. Platform baseline

| Component | Relevant numbers |
|---|---|
| FireCuda 530 (2 TB shown) | Seq read **7,300 MB/s spec** (128 KiB, Phison E18 + Micron 176L TLC); 4K random **1,000,000 IOPS spec @ QD32×T8**; measured **7,461 MB/s** CrystalDiskMark (TweakTown, AMD X570); sustained real-world ~6.5–7.2 GB/s. [1][2][3] |
| RX 9070 XT | PCIe **5.0 x16**, 16 GB GDDR6 @ 640 GB/s, AMD lists Smart Access Memory (ReBAR) support. [4] |
| i7-13700KF + 2×DDR5 | Dual channel. Spec 89.6 GB/s @ DDR5-5600 (Intel ARK); DDR5-6000 kit → 96 GB/s theoretical, **94.5 GB/s measured** AIDA64 read (13700K, tuned kit). [5][6] |
| PCIe 5.0 x16 | 63 GB/s/direction theoretical; **~53–57 GB/s measured** pinned H2D (see §6). [7][8] |

The interesting constraint is software, not hardware: the drive wants a deep
pipeline (~16–64 MB in flight), and Windows' classic synchronous path
won't keep it fed.

---

## 2. Path A — unbuffered overlapped `ReadFile` + IOCP

This is the workhorse and the API-agnostic choice (works identically whether
the GPU side is Vulkan, D3D12, HIP or OpenCL).

### 2.1 The mechanics that matter

- Open with **`FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`**
  (optionally `| FILE_FLAG_SEQUENTIAL_SCAN` for bulk loads). No-buffering
  skips the cache manager copy — data DMAs straight from the drive into your
  buffer. It also means the file's contents never pollute the standby list,
  which is exactly right for 19.5 GB you don't want cached anyway. [9][10]
- **Alignment**: file offset, buffer address, and transfer length must all be
  multiples of the volume's sector size — **4096 B** on the FireCuda 530.
  Keep expert offsets 4 KiB-aligned (pad the file format if needed).
- **Request size**: the sweet spot on Windows is **1–4 MB per request**.
  Smaller wastes per-IRP overhead; larger gives the storage stack fewer
  parallel units. m-schuetz's benchmark hit ~11 GB/s on a 14.8 GB/s Gen5
  drive using 1 MB chunks across threads. [11]
- **Queue depth**: latency ≈ in-flight-bytes ÷ bandwidth. For ~7 GB/s with
  ~1 ms worst-case completion latency you want **≥ 8–16 × 1 MB or 4–8 × 4 MB
  in flight at all times**, i.e. a self-rearming ring of 16–32 requests.
  PassMark uses QD20×128 KB and saturates sequential; the classic
  StackOverflow experiment saturated a 3.5 GB/s 970 EVO with a trivial
  4-deep ring of 1 MB requests. [12][13]
- **Completions**: associate the handle with an IOCP
  (`CreateIoCompletionPort`, `NumberOfConcurrentThreads` = 0 → one thread per
  logical core; 2 threads is plenty) and re-issue on completion. [9]
- **Scatter-gather**: `ReadFileScatter` takes an array of
  `FILE_SEGMENT_ELEMENT`, one **per system page (4 KB)**, each page-aligned —
  handy for reassembling an expert from discontiguous pinned pages, but the
  4 KB granularity makes it niche; a single contiguous buffer is usually
  simpler. Requires `NO_BUFFERING|OVERLAPPED`. [14][15]

### 2.2 Measured reality on Windows

The most useful published measurement is **m-schuetz/file_io_benchmark**
(Ryzen 7950X + Samsung 9100 PRO 14.8 GB/s, cold 1 MB chunks, ~1 GB/test): [11]

| Method | Threads | Throughput |
|---|---|---|
| Unbuffered overlapped | all cores | **11.3–11.4 GB/s** |
| Unbuffered overlapped | 1 | **4.7–4.9 GB/s** |
| `fread` | 1 / all | 4.4–6.3 / 4.2–5.0 GB/s |
| `MapViewOfFile` | 1 / all | 1.2 / 6.8–7.3 GB/s |

A .NET-focused investigation (PCIe 4.0 DRAM-less drive, Win10) found the same
shape: **~2.7 GB/s per issuing stream** regardless of buffer size, and needed
multiple streams/processes to approach the 5.2 GB/s drive ceiling — evidence
of a per-queue/per-thread ceiling in the Windows storage path, not the
drive. [16] A careful single-stream test of a 19.9 GB GGUF on a KC3000
measured **2.49 GB/s at QD1** — again, depth, not drive, was the limit. [17]

**Takeaway for the FireCuda 530 (~7 GB/s):** one issuing thread gets you
~4.5–5 GB/s; **two threads × QD 8–16 × 1–2 MB** or one IoRing/DS submission
engine saturates it.

### 2.3 Windows IoRing — io_uring's cousin

Windows 11 (build 22000+) has `IoRing`, modeled on io_uring: a shared-memory
SQ/CQ pair, up to **0x10000 entries per submission**, registered file handles
and registered buffers (pay validation/pinning cost once), ops: `READ`,
`WRITE`, `FLUSH`, `CANCEL`, `READ_SCATTER`, `WRITE_GATHER`. [18][19]
For comparison, Linux io_uring reaches ~1.7 M 4K IOPS/core with polling —
Windows has no polled completion, but IoRing removes most syscall overhead
and is what DirectStorage uses internally. [20][21]

### 2.4 BypassIO

Windows 11's optimized kernel path for NVMe reads: skips filter/volume
layers. **NVMe + NTFS + non-cached reads only**, per-handle via
`FSCTL_MANAGE_BYPASS_IO`. Vetoed by BitLocker, NTFS compression/encryption,
sparse files, volume snapshots, or any non-supporting minifilter (AV!). Check
with `fsutil bypassIo state`. DirectStorage enables it automatically; you can
also request it yourself on your `NO_BUFFERING` handle. [22][23][24]

---

## 3. Path B — DirectStorage 1.1 / 1.2 / 1.3

### 3.1 What it actually does

`IDStorageQueue::EnqueueRequest` takes `DSTORAGE_SOURCE_FILE` →
`DSTORAGE_DESTINATION_*` requests; `Submit()` sends them down a pipeline the
runtime manages: [25][26]

- **Uncompressed → GPU buffer** (`DSTORAGE_DESTINATION_BUFFER`):
  NVMe →(BypassIO/IoRing, kernel DMA)→ **upload-heap staging** → GPU
  **copy-engine** `CopyBufferRegion` → your `ID3D12Resource`. Zero CPU
  copies; the CPU only enqueues. [27]
- **Uncompressed → memory**: NVMe → sysmem staging → memcpy to your pointer.
- **GDeflate-compressed → GPU**: NVMe → upload heap → GPU copy to VRAM input
  staging → **GPU compute decompression** → VRAM output staging → copy to
  destination. Decompresses via vendor "metacommand" if the driver provides
  one, else a built-in DirectCompute shader — **works on any DX12 SM 6.0
  GPU**, including all RDNA. [28]
- GDeflate = DEFLATE variant, **64 KiB tiles**, tile-level random access —
  well suited to per-expert compressed streams. [28]
- Staging buffer: default 32 MB; **`SetStagingBufferSize` up to your request
  size — any request's uncompressed data must fit in staging or it fails**.
  NVIDIA recommends ≥ 128 MB when doing GPU decompression. [29][30]
- DS spins up its own **compute queue + two copy queues**; completions can
  signal `ID3D12Fence`s. DS 1.3 adds `EnqueueRequests` (batched requests +
  fence sync, e.g. "wait for fence before touching source data" — useful to
  interleave with a rendering/compute timeline) and a multi-subresource-range
  destination. [31][32]

### 3.2 Real numbers

- Intel's Expanse streaming demo (Arc A770 + Samsung 980 Pro, i9-12900K,
  Win11, 128 MB staging, ~408k requests): **4.97 GB/s from disk, 5.22 GB/s
  uncompressed→GPU** end-to-end. [33]
- NVIDIA: uncompressed streaming was PCIe-Gen3-bound at ~3 GB/s on their test
  system; GPU-side GDeflate multiplies effective throughput by compression
  ratio while the CPU stays idle. [30]
- AMD: officially supports DS 1.1 GPU decompression on all DX12+SM6 GPUs;
  the optimized **metacommand driver was shared with ISVs under NDA** — on
  public Adrenalin drivers expect the generic compute-shader path (still
  GPU-side, still fast; don't count on vendor-tuned decompression). [34][35]

### 3.3 Verdict for this runtime

- If the runtime's GPU API is **D3D12**, DS is strictly good: `DESTINATION_BUFFER`
  into weight buffers = the §2 pipeline with BypassIO+IoRing included, plus
  optional GDeflate for cold experts (compression ratio ≈ free SSD
  bandwidth × GPU compute you weren't using).
- If the runtime is **Vulkan**, DS doesn't apply directly (it's a D3D12
  API). Options: use D3D12 solely as the storage→VRAM path and interop
  buffers via `VK_KHR_external_memory_win32`/`D3D12_HEAP_FLAG_SHARED`, or —
  simpler — replicate the same pipeline with §2 IOCP + a Vulkan staging
  buffer + copy on the transfer queue (§6). The manual version loses
  BypassIO's CPU savings unless you enable the FSCTL yourself.
- GDeflate on experts: plausible win (weights compress ~1.3–2×) but adds 3
  GPU-side passes (copy-in, decompress, copy-out) — measure against plain
  uncompressed streaming; only worthwhile if SSD-bound after caching.

---

## 4. Path C — memory-mapped files

`CreateFileMapping`/`MapViewOfFile` wins on convenience, loses on control:

- **Measured** (same benchmark as §2.2): **1.2 GB/s single-threaded,
  6.8–7.3 GB/s multithreaded** on cold pages. [11]
- Every first touch of a cold page is a demand fault → an I/O at
  ~50–80 µs NVMe latency; faulting is serialized per thread and the
  readahead granularity is far smaller than an explicit 1–4 MB IRP. This is
  why single-thread mmap sits ~6× below the drive. [36][37]
- `PrefetchVirtualMemory` (Win8+) is Windows' `madvise(WILLNEED)`: advisory,
  issues **large concurrent I/Os** for paged-out ranges — genuinely useful to
  warm a mapped cold tier a layer ahead, but still at the memory manager's
  mercy (it may ignore you). [38]
- Mapped pages are evictable under memory pressure — you cannot pin a mapped
  view as a guaranteed-resident expert cache (VirtualLock on it just keeps it
  in the working set). For deterministic "cold experts in RAM" use your own
  allocation, not a mapping.

**Role in the design:** optional zero-effort third tier (map the file, let
the OS hold ~whatever fits, touch-ahead with `PrefetchVirtualMemory`), never
the latency-critical path.

---

## 5. "Pinned" memory on Windows — what each API really does

| API | Effect | DMA/GPU-visible? |
|---|---|---|
| `VirtualLock` | Locks pages into the process **working set** (no pagefile-out until unlock). Limit = min working set (raise via `SetProcessWorkingSetSize`); needs care, can starve the system. | No — it does **not** pin pages for DMA or map them to the GPU. [39] |
| `VirtualAlloc(MEM_LARGE_PAGES)` | 2 MB pages, **always non-pageable, physically contiguous**; needs `SeLockMemoryPrivilege`; allocate once at startup (fragmentation). | Reduces TLB pressure; still just RAM until a driver maps it. [40] |
| AWE `AllocateUserPhysicalPages` | Non-pageable PFNs you remap into a window (SQL Server's trick). | Same caveat — physical residency ≠ device mapping. [41] |
| I/O in flight | The I/O manager pins your buffer for the transfer's duration automatically (`MmProbeAndLockPages`) — overlapped `ReadFile` buffers are effectively pinned while busy. | Yes, transiently — this is why plain `VirtualAlloc` staging buffers work. |
| `cudaHostRegister` / `hipHostRegister` (HIP) | Page-locks existing malloc'd memory **and maps it into GPU address space** → full-speed async H2D. Register once at startup; measured H2D afterward ≈ `cudaHostAlloc` memory (within ~3 %). Pageable sources get staged through driver bounce buffers at ~half rate. [42][43] | Yes — this is the "RegisterMemory" path. |
| D3D12 upload/custom heap | `HEAP_TYPE_UPLOAD` (or CUSTOM: `CPU_PAGE_PROPERTY_WRITE_BACK` + `MEMORY_POOL_L0`) = GPU-visible **system memory**; GPU reads it over PCIe at link rate (~32 GB/s Gen4-class), persistent. | Yes — GPU-mapped from allocation. [44][45] |
| Vulkan staging | `HOST_VISIBLE|HOST_COHERENT` sysmem buffer → persistent GPU mapping; `vkCmdCopyBuffer` on transfer queue DMAs it to `DEVICE_LOCAL`. | Yes. |

Pinning budget for our design: a **staging ring of ~128–256 MB** (registered
once, HIP-pinned or upload-heap — the two GPU-API flavors of the same thing)
plus however much of the 48 GB you dedicate to the resident expert cache.
48 GB total → comfortably ~24 GB pinnable + OS + working set; just don't
`VirtualLock` the world — pin only what the GPU touches, via the GPU API's
own pinning, not `VirtualLock`.

---

## 6. RAM → VRAM: BAR writes vs copy-engine DMA

### 6.1 The two ways data reaches VRAM

**(a) CPU writes into the BAR window.** Without ReBAR the CPU-visible window
is 256 MB; with **ReBAR/SAM on the 9070 XT the whole 16 GB is CPU-mappable**
(works on Intel Z690/Z790-class boards too — it's a standard PCIe feature;
enable Above-4G-Decoding + Re-Size BAR, verify in Adrenalin). The mapping is
uncacheable/**write-combined**: [46][47]

| Access | Measured (RTX 5000 Ada, PCIe 4.0) |
|---|---|
| CPU write via WC mapping | **~10 GB/s** |
| CPU write via UC mapping | ~44 MB/s |
| CPU **read** from VRAM | **~100–110 MB/s** |

So: CPU memcpy into mapped VRAM ≈ 10 GB/s ceiling (platform-dependent, often
8–20 on Gen5); reading VRAM from CPU is catastrophic — design so the CPU
never reads GPU memory (expert KV/metrics live in RAM). [46][47]

**(b) GPU copy engine (SDMA on AMD).** Dedicated DMA hardware, independent of
compute units — copies run **concurrently with kernels**. AMD GPUs expose
≥2 DMA engines (`asyncEngineCount` > 0 → async copies possible); page-locked
host memory required for full speed. [48][49]

Measured pinned-memory copy-engine rates:

| Link | H2D pinned | D2H pinned | Source |
|---|---|---|---|
| PCIe 4.0 x16 (A100) | **24.8 GB/s** | 25.9 GB/s | nvbandwidth [50] |
| PCIe 5.0 x16 (RTX 5080) | **53.1 GB/s** | 39.3 GB/s | CUDA bandwidthTest [51] |
| PCIe 5.0 x16 (RTX PRO 5000) | **56.9 GB/s** @16 MB+ | 56.5 GB/s | microarch bench [7] |
| PCIe 4.0 x16 (RTX 3060) pageable-staged | 14.7 GB/s | — | CUDA handbook [52] |
| PCIe 5.0 x16 pageable (large) | ~35–37 GB/s | ~37 GB/s | [7] |

Rules of thumb: **pinned ≈ 1.5–1.8× pageable**; transfers need ≥4–16 MB to
reach plateau (1 MB ≈ 40 GB/s on Gen5; latency floor ~5–7 µs/transfer).
Bidirectional Gen4 showed H2D droop (25→11 GB/s) under full-duplex load —
keep H2D and D2H from contending, or accept the asymmetry. [7][50]

### 6.2 Queue plumbing

- **D3D12:** `D3D12_COMMAND_LIST_TYPE_COPY` queues run on SDMA;
  `CopyBufferRegion` staging→VRAM overlaps compute. This is what DS uses
  internally. [27][48]
- **Vulkan:** look for a queue family with `VK_QUEUE_TRANSFER_BIT` and no
  graphics/compute bits — on AMD hardware that's the SDMA engine. AMD's
  Windows driver (AMDVLK lineage) exposes it; RADV only added an equivalent
  dedicated transfer queue in Mesa 26 (Linux-side context). Verify via
  `vkGetPhysicalDeviceQueueFamilyProperties`; fall back to a second compute
  queue if absent (still overlaps). [53][54]
- **HIP/ROCm:** `hipMemcpyAsync` on a non-blocking stream + pinned host
  memory → SDMA path; `asyncEngineCount` confirms capability. [49]

### 6.3 A sneaky third option: zero-copy sysmem experts

A D3D12 CUSTOM heap with `MEMORY_POOL_L0 + WRITE_BACK` (or Vulkan
`HOST_VISIBLE` device-accessible memory) lets the GPU **read expert weights
directly out of system RAM** over PCIe — no copy at all. On Gen5 x16 that's
~50 GB/s worst-case read rate, ~8× slower than GDDR6 (640 GB/s) but ~7×
faster than pulling from the SSD. Viable for "warm" experts that are
compute-bound anyway; not for hot ones. [44][45]

---

## 7. Bandwidth budget check

Pipeline for the steady-state expert stream (per byte delivered to VRAM):

| Stage | Cost on i7-13700KF DDR5-6000 (~95 GB/s read, ~80 write) |
|---|---|
| NVMe → staging buffer | 1× RAM write ≈ 7 GB/s |
| staging → VRAM (SDMA reads sysmem over PCIe) | 1× RAM read ≈ 7 GB/s |
| **Total memory traffic** | **~14 GB/s ≈ 15 % of DDR5** — trivial. Even a memcpy-based path (read+write) stays <40 %. |
| PCIe | H2D 7 GB/s uses ~12 % of Gen5 x16 one direction — no contention with compute's PCIe usage. |

Bottleneck order (slowest first): **SSD 7 → SDMA-to-VRAM 50+ → DDR5 80+ →
GDDR6 640**. The only real failure modes are shallow queues (§2.2) and
pageable staging halving the H2D leg (§6.1).

Latency math for prefetch distance (FireCuda 530 at ~6.8 GB/s, Gen5 H2D at
~50 GB/s): a **2 MB expert** ≈ 0.29 ms SSD + 0.04 ms DMA ⇒ schedule ~1 layer
ahead; a **32 MB layer-batch** ≈ 4.7 + 0.64 ms ⇒ start ~2 layers ahead at
typical per-layer compute times (10–40 ms, it's comfortable).

---

## 8. Recommended I/O architecture

**File layout** — one/few big weights files on a dedicated NTFS volume:
uncompressed, unencrypted, non-sparse, no BitLocker, every expert blob
4 KiB-aligned; verify `fsutil bypassIo state` if going the DS route.

**Tiers** — hot weights + KV in VRAM (≤~14 GB); warm expert cache in RAM
(~16–24 GB, GPU-registered/pinned once at startup); cold remainder on SSD.

**Loader (Path A, Vulkan/HIP-agnostic):**
- `NO_BUFFERING|OVERLAPPED` handle(s) + IOCP; **2 issuing threads, QD 16–32
  total, 1–4 MB per request**, ≈ 32–64 MB in flight → ~6.5–7 GB/s.
- Requests land in a **ring of ~8 × 32 MB staging buffers** that are
  HIP-pinned (or Vulkan `HOST_VISIBLE` / D3D12 upload-heap) — registered
  once, reused forever.
- On completion: `hipMemcpyAsync`/`vkCmdCopyBuffer`/`CopyBufferRegion` on the
  **copy/transfer queue** (SDMA), chunk to VRAM, fence/event → slot free.
- Optional `FSCTL_MANAGE_BYPASS_IO` on the file handle for lower CPU.

**Loader (Path B, D3D12 available):** one `IDStorageQueue`,
`SetStagingBufferSize(128–256 MB)`, `DSTORAGE_DESTINATION_BUFFER` requests of
32–128 MB each, `EnqueueRequests` + `ID3D12Fence` (DS 1.3) for layer-granular
sync — BypassIO + IoRing + copy-queue plumbing all free. Consider GDeflate
for the cold tier only if measurements show SSD-bound streaming after the RAM
cache warms.

**Expert-miss path** (rare, on-demand): same IOCP path; coalesce each expert
to ≥256 KB–1 MB per request (4K QD1 is ~60–80 MB/s — latency-dominated;
batch by layer, not by expert row).

**CPU-mapped VRAM** (ReBAR on): reserve for small, frequent, write-only data
(dispatch tables, routing indices). ~10 GB/s ceiling; **never read it**.

**What to avoid:** pageable staging buffers (halves H2D), single-thread
reads (~5 GB/s ceiling), mapped files for the hot path, CPU memcpy to VRAM
for bulk weights, `VirtualLock` as a pinning strategy for DMA (wrong tool).

**Expected end state:** sustained ~6.5–7 GB/s SSD→VRAM concurrent with
compute at <5 % total CPU, bounded in practice by the drive — which is the
correct bottleneck to have.

---

## Sources

1. Seagate FireCuda 530 datasheet — https://www.seagate.com/content/dam/seagate/migrated-assets/www-content/datasheets/pdfs/firecuda-530-ssd-DS2059-3-2112US-en_US.pdf
2. TweakTown FireCuda 530 review (7,461 MB/s measured) — https://www.tweaktown.com/reviews/9878/seagate-firecuda-530-1tb-ssd-tested-with-windows-11/index.html
3. StorageReview FireCuda 530 — https://www.storagereview.com/review/seagate-firecuda-530-ssd-review
4. AMD RX 9070 XT product page (PCIe 5.0 x16, SAM) — https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html
5. Intel ARK i7-13700K (89.6 GB/s DDR5-5600, 2ch) — https://www.intel.com/content/www/us/en/products/compare.html?productIds=230490%2C230500
6. hwcooling AIDA64 database (13700K 94.55 GB/s read) — https://www.hwcooling.net/wp-content/uploads/2024/09/amd-ryzen-9-9900x-g308.html
7. SM_120 microarch reference, PCIe Gen5 x16 measured 56.9 GB/s pinned — https://zartbot.github.io/micro_arch/nvidia/sm_120/10_system_level_multi_gpu.html
8. PCIe generation bandwidth table — https://topnotchnote.com/gpu/pcie
9. MS Learn: I/O Completion Ports — https://learn.microsoft.com/en-us/windows/win32/fileio/i-o-completion-ports
10. MS Learn: ReadFile / overlapped I/O — https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfile , https://learn.microsoft.com/en-us/windows/win32/fileio/synchronous-and-asynchronous-i-o
11. m-schuetz/file_io_benchmark (real Windows numbers) — https://github.com/m-schuetz/file_io_benchmark
12. PassMark disk read test methodology (QD20×128KB unbuffered async) — https://www.passmark.com/products/performancetest/v12/help/2-Benchmarks/5-Disk-Mark/29-Disk-Read/disk-read-help.php
13. StackOverflow: fastest file read, overlapped ring saturates 3.5 GB/s drive — https://stackoverflow.com/questions/10836609/fastest-technique-to-read-a-file-into-memory
14. MS Learn: ReadFileScatter — https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfilescatter
15. MS Learn: FILE_SEGMENT_ELEMENT — https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-file_segment_element
16. dotnet/runtime discussion #104563 (2.7 GB/s per-stream ceiling) — https://github.com/dotnet/runtime/discussions/104563
17. bestin-it: 19.9 GB GGUF read, KC3000 2.49 GB/s at QD1 — https://bestin-it.com/clone-windows-to-nvme-ssd/
18. MS Learn: Windows IoRing (BuildIoRingRegisterBuffers, IORING_OP_CODE) — https://learn.microsoft.com/en-us/windows/win32/api/ioringapi/nf-ioringapi-buildioringregisterbuffers , https://learn.microsoft.com/en-us/windows/win32/api/ntioring_x/ne-ntioring_x-ioring_op_code
19. Winsider: One Year to I/O Ring — https://windows-internals.com/one-year-to-i-o-ring-what-changed/
20. Axböe, "Efficient IO with io_uring" — https://kernel.dk/io_uring.pdf
21. AMD GDC 2023 slides: "DirectStorage uses BypassIO and IORing" + LZ4 ~7 GB/s/core note — https://gpuopen.com/download/GDC-2023-DirectStorage-optimizing-load-time-and-streaming.pdf
22. MS Learn: BypassIO in storage drivers — https://learn.microsoft.com/en-us/windows-hardware/drivers/storage/bypassio
23. MS Learn: BypassIO for filter drivers — https://learn.microsoft.com/en-us/windows-hardware/drivers/ifs/bypassio
24. MS Learn: IOCTL_STORAGE_MANAGE_BYPASS_IO — https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntddstor/ni-ntddstor-ioctl_storage_manage_bypass_io
25. MS Learn: DSTORAGE_DESTINATION — https://learn.microsoft.com/en-us/windows/win32/dstorage/dstorage/ns-dstorage-dstorage_destination
26. MS Learn: IDStorageQueue — https://learn.microsoft.com/en-us/windows/win32/dstorage/dstorage/nn-dstorage-idstoragequeue
27. microsoft/DirectStorage DeveloperGuidance.md + diagrams.mmd (staging flow, copy queues) — https://github.com/microsoft/DirectStorage/blob/main/Docs/DeveloperGuidance.md , https://github.com/microsoft/DirectStorage/blob/main/Docs/diagrams.mmd
28. DirectX blog: DirectStorage 1.1 (GDeflate, 64 KiB tiles, compute fallback) — https://devblogs.microsoft.com/directx/directstorage-1-1-now-available/
29. MS Learn: IDStorageFactory::SetStagingBufferSize — https://learn.microsoft.com/en-us/windows/win32/dstorage/dstorage/nf-dstorage-idstoragefactory-setstagingbuffersize
30. NVIDIA blog: GDeflate for DirectStorage (≥128 MB staging rec., ~3 GB/s Gen3 uncompressed) — https://developer.nvidia.com/blog/accelerating-load-times-for-directx-games-and-apps-with-gdeflate-for-directstorage/
31. DirectX blog: DirectStorage 1.3 (EnqueueRequests, fences) — https://devblogs.microsoft.com/directx/directstorage-1-3-is-now-available/
32. NuGet: Microsoft.Direct3D.DirectStorage changelog — https://www.nuget.org/packages/Microsoft.Direct3D.DirectStorage/1.3.0
33. Intel "Maximum DirectStorage" (Expanse: 4.97/5.22 GB/s, 128 MB staging) — https://game.intel.com/wp-content/uploads/2024/02/GameDev2023_DirectStorage_v1.pdf , https://www.intel.com/content/www/us/en/developer/articles/news/directstorage-on-intel-gpus.html
34. AMD GPUOpen: DirectStorage 1.1 support (metacommand NDA) — https://gpuopen.com/news/amd-support-for-microsoft-directstorage-1-1/
35. NuGet changelog 1.2.x (generic GPU decompression shader notes) — https://www.nuget.org/packages/Microsoft.Direct3D.DirectStorage/1.3.0
36. USENIX HotStorage'17: mmap page-fault overhead analysis — https://www.usenix.org/system/files/conference/hotstorage17/hotstorage17-paper-choi.pdf
37. GoHacks: mmap vs Read trade-offs (cold fault ≈ disk latency) — https://gohacks.dev/library/go-mmap
38. MS Learn: PrefetchVirtualMemory — https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory ; Old New Thing — https://devblogs.microsoft.com/oldnewthing/20160225-00/?p=93091
39. MS Learn: VirtualLock — https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtuallock
40. MS Learn: Large-Page Support — https://learn.microsoft.com/en-us/windows/win32/memory/large-page-support
41. MS Learn: AllocateUserPhysicalPages (AWE) — https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-allocateuserphysicalpages
42. CuPy issue #3450: cudaHostRegister on existing buffer — https://github.com/cupy/cupy/issues/3450
43. CUDA Handbook ch.5 (pageable staged via driver bounce buffers) — https://www.cudahandbook.com/book/ch5/host-memory ; pinned-vs-pageable 1 GB test (53.8 vs 27.2 GB/s) — https://jinseok-moon.github.io/p/pinned-memory/
44. MS Learn: D3D12_HEAP_TYPE / GetCustomHeapProperties — https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_heap_type , https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-getcustomheapproperties(uint_d3d12_heap_type)
45. therealmjp: GPU Memory Pools in D3D12 (GPU reads SysRAM ~16/32 GB/s Gen3/Gen4, BAR 256MB→ReBAR) — https://therealmjp.github.io/posts/gpu-memory-pool/
46. arXiv DMA streaming framework: WC BAR 10.1 GB/s, UC 44 MB/s, GPU→host read ~107 MB/s — https://arxiv.org/html/2603.10030
47. AnyPS5-RDNA issue #2: SAM on RX 9070 XT, WC writes fine/reads slow, HOST_VISIBLE not HOST_CACHED — https://github.com/DenizSAHIN570/AnyPS5-RDNA/issues/2
48. ROCm programming guide: DMA engines, ≥2 per GPU, concurrent transfers — https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/latest/conceptual/introduction/hw_impl.html
49. ROCm HIP async docs: asyncEngineCount, page-locked host memory — https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip%5Fruntime%5Fapi/asynchronous.html ; AMD ROCm blog: GPU-initiated SDMA — https://rocm.blogs.amd.com/software-tools-optimization/gpu-initiated-sdma/README.html
50. NVIDIA nvbandwidth issue #53: A100 PCIe4 x16 pinned H2D 24.8/D2H 25.9, bidirectional H2D droop — https://github.com/NVIDIA/nvbandwidth/issues/53
51. NVIDIA dev forums: RTX 5080 PCIe5 pinned H2D 53.1 GB/s; 4090 PCIe4 23.9 — https://forums.developer.nvidia.com/t/blackwell-integer/320578/56
52. CUDA Handbook memcpy benchmarks (RTX 3060 pinned 26.8 vs staged 14.7) — https://www.cudahandbook.com/benchmarks/memcpy/
53. Vulkan docs: dedicated transfer queues — https://docs.vulkan.org/tutorial/latest/Synchronization/Transfer_Queues_Streaming/02_non_blocking_uploads.html
54. Phoronix: Mesa 26 RADV dedicated transfer queue via SDMA (notes AMDVLK already had it) — https://www.phoronix.com/news/Mesa-26.0-RADV-Transfer-SDMA ; Mesa RADV docs — https://docs.mesa3d.org/drivers/radv.html
