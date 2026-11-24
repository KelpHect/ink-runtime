# A14 — C/C++ inference-runtime architecture patterns worth cribbing

Scope: structural/architecture patterns from existing small-to-medium C/C++ LLM
inference engines, plus what vLLM/SGLang teach us at the systems level. Target:
Windows-first C++ runtime for a 26.4B MoE (inkling2_loop: 24 layers, middle 12
looped 3×, 128 experts top-8, hybrid SWA/global attention, MTP draft head,
NVFP4+MXFP8 safetensors, tiktoken) on RX 9070 XT 16 GB + 48 GB RAM + NVMe.

Sources consulted (all read at source level unless noted):

1. llama.cpp `ggml-backend.h` — backend vtable surface
   https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h
2. llama.cpp `ggml-backend.cpp` — scheduler, splits, events, prefetch
   https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp
3. llama.cpp `ggml-vulkan.cpp` — Vulkan backend internals (queues, submits, RDNA4 detect)
   https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp
4. ik_llama.cpp README + PR #2644 — hybrid CPU/GPU MoE offload, expert LRU cache
   https://github.com/ikawrakow/ik_llama.cpp/blob/main/README.md
   https://github.com/ikawrakow/ik_llama.cpp/pull/2644
5. llama2.c `run.c` — minimal engine skeleton
   https://github.com/karpathy/llama2.c/blob/master/run.c
6. llm.c `train_gpt2.cu` — pinned staging, chunked disk→device streaming
   https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu
7. fastllm `include/fastllm.h`, `include/model.h`, `src/fastllm.cpp`, `src/models/`
   https://github.com/ztxz16/fastllm
8. ctransformers — per-arch C++ model classes under Python bindings (dead project)
   https://github.com/marella/ctransformers
9. PowerInfer (Tiiny-AI) + PowerInfer-2 paper — hot/cold neuron split, I/O-compute pipeline
   https://github.com/Tiiny-AI/PowerInfer
   https://arxiv.org/html/2406.06282 , https://powerinfer.ai/v2/
10. vLLM v1 `vllm/v1/core/sched/scheduler.py` — scheduler/KV-manager separation
    https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py
11. SGLang `python/sglang/srt/managers/scheduler.py` — overlap event loop, FutureMap
    https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py
12. MLC tokenizers-cpp — tokenizer.json in C++
    https://github.com/mlc-ai/tokenizers-cpp
13. safetensors spec (README = format spec)
    https://github.com/huggingface/safetensors/blob/main/README.md
14. C++ tiktoken ports: gh-markt/cpp-tiktoken, meta-pytorch/tokenizers, KolosalAI/tiktoken.cpp
    https://github.com/gh-markt/cpp-tiktoken
    https://github.com/meta-pytorch/tokenizers
    https://github.com/KolosalAI/tiktoken.cpp
15. AMD ROCm-on-Windows support matrix (HIP SDK 7.2, gfx1201 = RX 9070 XT supported)
    https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html

---

## 1. llama.cpp / ggml — the pattern that matters

Everyone copies ggml's layering. It has four separable ideas; you can adopt any
subset without adopting the codebase.

### 1.1 The three-level object model

From `ggml-backend.h`:

- **`ggml_backend_buffer_type` ("buft")** — a *placement policy*: where memory of
  a given class lives (CPU RAM, CUDA device, Vulkan device, "CUDA host-pinned").
  `ggml_backend_buft_alloc_buffer(buft, size)` allocates one contiguous slab;
  `buft->iface.is_host`, `get_alignment`, `get_max_size`, `get_alloc_size`.
- **`ggml_backend_buffer`** — one allocated slab that backs many tensors.
  `ggml_tallocr` (in `ggml-alloc.h`) is a trivial bump allocator that carves
  tensors out of a buffer with the buft's alignment. `buffer_usage` is tagged
  `WEIGHTS` or `COMPUTE` — the scheduler uses this to decide prefetch policy.
- **`ggml_backend`** — a stream of work: `graph_compute`, `tensor_set/get_async`,
  `cpy_tensor_async`, `synchronize`, `event_record`/`event_wait`.

Key point: **tensors are views; buffers are memory; backends are queues.** A
`ggml_tensor` is just shape + type + `data` pointer + `view_src`/`view_offs`.
Weights get `data` pointing into a buffer (possibly an mmapped GGUF); activation
tensors get re-pointed every step by the graph allocator.

### 1.2 `ggml_backend_sched` — the multi-device executor

`ggml-backend.cpp` (~L1187–1320) is the most directly relevant code to our
design. Mechanics:

- `ggml_backend_sched_split_graph` walks the op graph once and assigns each op a
  backend id (`node_backend_id`), based on (a) which buffer type its inputs live
  in, (b) `supports_op`, (c) user overrides. Contiguous same-backend runs become
  **splits** (`splits[]` with `i_start`/`i_end` + `graph` view + `inputs`).
- `ggml_backend_sched_compute_splits` (~L2225–2312): for each split, issue async
  copies of cross-backend inputs (`hv_tensor_requires_copy`), then
  `ggml_backend_graph_compute_async` on the split's backend, then
  `ggml_backend_event_record` into `events[backend_id][cur_copy]` — a 2-D event
  array so several concurrent "copies" (logical pipelines, e.g. per-request or
  draft vs. target) can share one scheduler without aliasing.
- Ordering between splits is implicit: the *next* split's backend `event_wait`s
  on the producing backend's event; no host synchronization unless a CPU read
  is actually needed.
- `ggml_backend_sched_prefetch_inputs` (~L2180–2223): while split *i* computes,
  it issues `tensor_get_async` D2H→host copies of split *i+1*'s inputs and records
  `prefetch_events[backend]`; comment: "the copy callback of this split runs
  later and can write other tensors (e.g. the MoE cache)" — i.e. upstream's
  dynamic expert cache hooks in exactly here.
- `parallel` flag at sched creation chooses `n_copies = GGML_SCHED_MAX_COPIES`
  (multi-graph interleaving) vs 1.

**What to crib:** the split/event/prefetch skeleton is ~300 lines of logic. Our
GPU↔host expert pipeline is exactly "splits + async copies + events," except our
split boundaries are fixed (attention block → expert block) rather than computed.

### 1.3 `ggml_gallocr` — compute-buffer arena

`ggml-alloc.c`: `ggml_gallocr_reserve` does a liveness pass over the cgraph and
assigns every intermediate tensor an offset in one big "compute buffer" per
buffer type (bump/arena allocation, peak reuse). `ggml_allocr` = same idea for a
single graph. **Crib it**: allocate activations out of one VRAM slab per step
shape; never `vkAllocateMemory` per tensor at runtime.

### 1.4 The Vulkan backend — queue model, not CUDA graphs

`ggml-vulkan.cpp` findings (verified in source):

- **No command-buffer replay.** `ggml_backend_vk_graph_compute` (~L15485) walks
  nodes, packs them into command buffers from a pool (`ggml_vk_get_or_create_cmd_buffer`
  marks `in_use`; freed and recycled), and **submits incrementally**: "Submit
  after enough work has accumulated, to overlap CPU cmdbuffer generation with
  GPU execution… submit every 200 GFLOP (scaled down for small models) and at
  least every 100 nodes" (~L15550). Weaker AMD GPUs get a lower flops cap to
  dodge TDR. So upstream's answer to "CUDA-graph-equivalent" is: don't replay —
  keep CPU encode fast and stream submissions ahead of execution.
- **Two-queue overlap.** Optional `GGML_VK_ASYNC_USE_TRANSFER_QUEUE`
  (~L5571): H2D copies go on a dedicated transfer queue in `transfer_ctx`;
  `transfer_semaphore` is a **timeline semaphore**; the compute queue waits via
  `VkTimelineSemaphoreSubmitInfo` (~L6537, L14686). Compute submissions use a
  plain `vk::Fence` per context; `ggml_vk_synchronize` drains both.
- **Host-visible + imported memory.** `ggml_backend_vk_cpy_tensor_async`
  (~L14580+): host→device copies look up `ggml_vk_host_get` — a registry mapping
  pinned host ranges to `vk_buffer`s created via
  `VK_EXT_external_memory_host` import (`host_buffer_import` usability checked
  per-device ~L4542, L6146–6166). If backend idle and dst is host-coherent and
  ≤128 KiB, it does a straight CPU `memcpy` into the mapped device buffer
  instead of a GPU copy — nice micro-pattern.
- **RDNA4 detection** (`get_device_architecture`, top of file): AMD path checks
  `VK_AMD_shader_core_properties` + `VK_KHR_shader_integer_dot_product` +
  `VK_EXT_subgroup_size_control`; minSubgroup=32/max=64 ⇒ RDNA;
  `VK_EXT_shader_float8` present ⇒ `AMD_RDNA4`. RDNA3 via
  `integerDotProduct4x8BitPackedMixedSignednessAccelerated`. Cooperative-matrix
  is probed on NVIDIA/Qualcomm (`VK_KHR_cooperative_matrix`, `coopmat1`/`coopmat2`
  pipeline variants for flash-attn); AMD gets `q_in_regs` special-casing instead.
- Pipelines are cached per (op, dtype, shape-class) in a `pipeline_cache`;
  specialization constants bake subgroup size/tiling.

**Implication for us:** Vulkan gives you command-buffer reuse for free-ish
(command pool + `vkCmdExecuteCommands` secondaries), but there is no
"vkGraphExec". The cheap CUDA-graph equivalent = **cache one recorded secondary
command buffer per decode shape** (batch-size bucket × draft-length) and only
re-record on shape change; parameterize tensors via push constants/descriptor
offsets, not baked addresses, so the same recording survives buffer rebinding
(e.g. ring-buffer slots rotating). Feasible; ggml just doesn't bother because
its per-node encode overhead is already hidden by incremental submits.

### 1.5 Expert offload in practice — `-ot`, `--cpu-moe`, MoE LRU cache

Both upstream and ik_llama.cpp route tensors by **name/regex → buffer type**:

- `--override-tensor 'blk\.\d+\.ffn_(up|down|gate)_exps\.weight=CPU'` keeps
  expert weights host-resident; the scheduler then runs the corresponding
  `MUL_MAT_ID` on the CPU backend or copies to GPU, per op support.
- `--cpu-moe`/`-cmoe` (all MoE experts to CPU) and `--n-cpu-moe N` (first N
  layers' experts to CPU) are sugar over the same mechanism.
- ik_llama.cpp `-rtr` repacks host tensors to row-interleaved quants for the
  IQK AVX2/AVX-512 kernels — README warns it *forces* CPU compute for those
  tensors (no CUDA row-interleaved kernels), often slower. Lesson: repacking
  host-side experts is a placement decision, not just a layout decision.
- **Dynamic expert cache** — upstream llama.cpp PRs #29887/#30112, ported to
  ik_llama.cpp as PR #2644 (`--moe-cache-mib N`, `--moe-cache-policy`,
  `--moe-cache-cpu-share`): a VRAM budget holds the most-used experts;
  hits run on GPU, misses on CPU; the sched "copy callback" (see §1.2) is the
  hook that installs hot experts into GPU buffers.
- ik README's honest numbers: ~+20% TG for one model, **−25% for another** vs
  static layer offload. Expert caching is workload-dependent; make the policy
  pluggable and keep a static "first-K-layers/last-K-experts" baseline.
- ik_llama.cpp also lands "MoE combine runs on the CPU with
  `-ot 'ffn_.*_exps=CPU'`" cutting CPU→GPU activation copies ~9× (PR #2901):
  **put the weighted-sum reduction next to the expert weights**, not next to
  the GPU — this matters for us because top-8 combine of 128 experts is itself
  a bandwidth cost.

## 2. Minimal engines — what each teaches

### 2.1 llama2.c — the skeleton

`run.c`: one `Transformer` = `Config` + `TransformerWeights` (all float* views
into one mmapped checkpoint, laid out by `memory_map_weights` ~L130) +
`RunState` (every activation incl. KV cache, calloc'd once). `forward()` is a
straight sequential loop: embed → per-layer {rmsnorm, qkv matmuls, rope,
attention, ffn} → logits. `matmul` is `#pragma omp parallel for`.

The enduring lesson: **the decode loop is ~300 lines; everything else is I/O,
memory placement, and kernels.** Keep our forward path readable at that level —
an explicit layer loop, not a generic DAG evaluator. A fixed-shape static graph
(the same ops every step) means we don't need ggml's dynamic node lists.

### 2.2 llm.c — staged streaming, one stream

`train_gpt2.cu`:

- `IO_BUF_SIZE = 32 MB` (~L84). `file_to_device(params_memory, file, bytes,
  IO_BUF_SIZE, main_stream)` (~L513): loop fread → pinned buffer →
  `cudaMemcpyAsync` → next chunk — the canonical disk→pinned-staging→VRAM pipe.
- `cudaMallocHost` only where a readback must not stall (`cpu_losses` ~L385).
- One `main_stream` for everything; concurrency comes from kernel overlap
  *within* a stream being impossible — they serialize. (Multi-GPU uses NCCL.)
- Single `cudaMalloc` arena per tensor class (`malloc_and_point_parameters`):
  same arena pattern as llama2.c, on device.

**Crib:** a fixed-size pinned staging ring + chunked async copies is the whole
trick. For us: 4–8 × 32–64 MB pinned slots circulating NVMe→RAM→VRAM.

### 2.3 fastllm — the closest prior art for tiered MoE

fastllm (C++, Chinese project, MIT) does *exactly* RAM+GPU(+disk) tiered MoE.
Structure:

- **`Data` is a fat tagged tensor** (`include/fastllm.h` ~L524–651): `cpuData`,
  `cudaData`, `deviceData`, `dataDevice` (`CPU=0, CUDA=1` — extended by a disk
  device `include/devices/disk/diskdevice.h`), `dataType`+quant params
  (group/perChannel/blockK×blockM incl. `NVFP4_BLOCK_16_E4M3_PACKED`),
  `isModelWeight`, `isKVCache`/`isPagedKVCache`+`PagedCacheManager`,
  `fileName`+`filePos`+`mapFile` (`FileMmap` shared_ptr) — and crucially
  **`isDiskWeight` + `diskWeightParts[]`** (fileName/fileOffset/bytes per part,
  ~L455–463, 606–610): a weight can exist only as a disk location descriptor and
  be materialized on demand.
- **Device maps** (`src/fastllm.cpp` ~L6099–6150): `SetDeviceMap`,
  `SetMoeDeviceMap`, `SetLayeredMoeDeviceMap` map `{device_name: fraction}`
  (devices: `cpu`, `cuda`, `disk`); `SelectDeviceFromMap` picks each weight's
  home by index — i.e. declarative tensor placement, same idea as llama.cpp's
  `-ot` but coarser and friendlier.
- **Two-tier expert cache**: `SetMoeCudaCacheBytes` /
  `moeCudaCacheDeviceBytes` (VRAM budget, ~L323–495) +
  `SetMoeCpuCacheBytes` (RAM budget) + `TrimDiskMoeCache`/`ReleaseDiskMoeCache`
  — experts flow disk→RAM→GPU through LRU caches at both tiers.
  `ToCudaTemporary`/`FreeCudaTemporary` (~L716) stage a host tensor on GPU
  inside a stream and drop it after — the transient-residency pattern.
- **Model files are per-architecture**: `src/models/deepseekv2.cpp` (107 KB),
  `deepseekv4.cpp` (701 KB!), `deepseekv41_*` (execution/hybrid/gpu/inc splits),
  `qwen4_exp_*`, `moe.cpp`… The architecture code is a hand-written op sequence
  per model, dispatched through an `Executor` to `devices/cpu|cuda|numas|disk`
  backends; CPU parallelism via `AliveThreadPool` (`devices/cpu/alivethreadpool.h`
  — a persistent spinning thread pool).
- `PagedCacheManager` (paged KV), `SetPageLen`, `SetCudaSlabMB`
  (CUDA allocation slabs), `FastllmEnv` env-var knobs incl. `cudaGraph`.

Weaknesses for us: CUDA-only GPU path (no Vulkan; ROCm exists but community
reports it's second-class), `Data` carries ~40 fields (fat but workable),
model sources are enormous hand-written files, comments/API docs largely in
Chinese. **Crib the placement model and tiered cache; don't crib the code org.**

### 2.4 ctransformers — cautionary

README confirms it's a thin Python wrapper over one `.cpp` per model family
(llama/gptj/mpt/…) with `gpu_layers` offload count — basically an older ggml
snapshot productized. Dead since ~2023. Lesson: **a zoo of per-model files rots
fast**; it died the moment upstream moved to the generic-graph architecture.
If we hand-write the inkling2 forward pass (we should), keep it *one* file with
the arch quirk table (loop count, SWA mask pattern, expert layout) as data.

### 2.5 PowerInfer / PowerInfer-2

- PowerInfer (Tiiny-AI, open): hot neurons preloaded on GPU, cold computed on
  CPU; per-layer tiny MLP **predictors** select active neurons before fetching —
  the "predict, then fetch only what you'll use" trick. That maps to MoE:
  the router is the predictor, so *router output → expert fetch* is the same
  dependency, except we know routing exactly rather than predicting it.
- PowerInfer-2 is **closed-source** (both SJTU-IPADS and Tiiny-AI repos 404;
  paper only — arXiv:2406.06282). The shipped successor is `smallthinker/` in
  the PowerInfer repo. Paper patterns worth stealing anyway: "neuron clusters"
  as the pipeline granule (for us: one expert's 3 matrices = one cluster),
  segmented neuron cache (= per-layer expert cache partitions), and a
  cluster-level I/O pipeline overlapping fetch with compute — i.e. prefetch
  expert *i+1*'s top-8 while expert *i*'s are still computing.

## 3. vLLM / SGLang — what a C++ port actually needs

These are server engines; we need their *invariants*, not their scale.

### 3.1 vLLM v1

`vllm/v1/core/sched/scheduler.py`:

- **Scheduler is a pure-CPU object** producing `SchedulerOutput` (which
  requests, how many tokens each, which KV blocks) each step; the
  `ModelRunner` consumes it on GPU. Hard separation — scheduler never touches
  tensors.
- Budgeting: `max_num_running_reqs`, `max_num_scheduled_tokens` per step —
  continuous batching = fill a token budget per iteration across requests.
- `KVCacheManager` + block pool: fixed-size KV blocks (default 16 tokens),
  per-request block tables, **prefix caching by hashing block contents**
  (shared prompts reuse blocks; COW semantics), preemption = free blocks and
  recompute later.
- Connectors (`KVConnectorFactory`, SCHEDULER vs WORKER roles) abstract KV
  offload/fetch to CPU/remote — the same seam we'd use for "KV in RAM" or
  SWA-layer spill.
- Spec decode is first-class: `num_sampled_tokens_per_step` and draft handling
  are wired into the scheduler loop, matching our MTP head.

For a single-user 26.4B engine: keep the **scheduler/runner split** and the
**block-pool KV manager**; skip preemption/connectors/DP.

### 3.2 SGLang — the CPU/GPU overlap loop

`scheduler.py` `event_loop_overlap` (~L2257): "overlaps the CPU processing and
GPU computation":

- Each iteration launches GPU forward via `forward_batch_generation`, then
  pushes `(batch, result)` onto a `result_queue` and processes *the previous*
  iteration's result — CPU bookkeeping for step *n* runs while GPU computes
  step *n*+1. That's a one-deep pipeline; it buys you a full step of CPU time
  (sampling, scheduling, HTTP) hidden behind GPU work.
- `FutureMap` (~L1845, `resolve_forward_inputs`, `future_map.publish`):
  values that are only known on GPU (sampled next-token ids, seq_lens) are
  represented as **futures/placeholders** and resolved by GPU-side copy without
  a host roundtrip — the next step's `input_ids` literally stays in device
  memory as an unresolved future until consumed. Also `resolve_seq_lens_cpu`
  when a CPU copy is genuinely needed (`AsyncD2HCopyWorker` preserves overlap).
- Massive component decomposition (`scheduler_components/*`: result processor,
  dp-attn, grammar, metrics, recv-skipper, idle-sleeper) — overkill for us;
  steal the *one-deep pipeline + future-token relay* only.
- `RadixCache` (`srt/mem_cache/radix_cache.py`): prefix cache as a radix tree
  over token ids (not vLLM's block hashing) — nicer for chat prefixes; either
  works.

## 4. Component-by-component: recommended libraries

| Component | Recommendation | Why / notes |
|---|---|---|
| **safetensors parse** | Own it, ~200 LOC. | Format = `u64 LE header_len` + JSON `{name:{dtype,shape,data_offsets}}` + packed blob, offsets relative to blob start, little-endian, C-order (spec: huggingface/safetensors README). mmap the file (`CreateFileMapping`/`MapViewOfFile`), parse header with nlohmann/json or simdjson, hand out `{name→(ptr,dtype,shape)}`. NVFP4/MXFP8 are just more dtype tags to add to the switch. Header ≤100 MB per spec. |
| **tokenizer (tokenizer.json)** | `mlc-ai/tokenizers-cpp`. | C++ wrapper over HF Rust `tokenizers` + sentencepiece; consumes `tokenizer.json` blob directly; ships in MLC LLM. Cost: needs a Rust toolchain in build (`cargo`+`cxx`). Battle-tested. |
| **tokenizer (tiktoken dir)** | `meta-pytorch/tokenizers` (pure C++, BSD) — or port llama.cpp's `llama-vocab` BPE. | meta-pytorch implements tiktoken BPE in C++ (adapted from sewenew/tokenizer) and is ExecuTorch-proven. gh-markt/cpp-tiktoken also fine (o200k/cl100k). llama.cpp's `src/llama-vocab.cpp` implements tiktoken-style BPE (regex pre-tokenizer + rank merge) — crib-worthy if we fork. Avoid the tiny hobby ports (bobqianic, KolosalAI) for prod. |
| **sampler** | Own it, ~300 LOC. | Everyone converges on: penalties (presence/frequency/repetition over last-N) → top-k → top-p → min-p → temp → categorical/argmax. llama.cpp `src/llama-sampling.cpp` is the reference chain + grammar hook; fastllm's `GenerationConfig` (fastllm.h ~L140) shows a compact config struct. Needs a seedable RNG (MT or PCG) and a vocab-sized scratch buffer. Logit readback is one small D2H per step — put it on the pinned ring. |
| **KV cache manager** | Own, paged. | fastllm `PagedCacheManager` + vLLM block pool agree: fixed `page_len` (64–128 tokens), `pageIndex[]` per sequence, free-page list. For hybrid SWA/global: *two pools* — SWA layers get a sliding window (bounded per-layer footprint, can be a ring per layer); global layers get paged blocks. MTP draft needs a fork/rollback (speculative pages refcounted, freed on reject — vLLM's spec handling is the model). |
| **Async I/O (NVMe)** | Dedicated I/O threads + `ReadFile` into pinned slots. | On Windows: 2–4 I/O threads issuing `ReadFile`/`ReadFileScatter` (unbuffered `FILE_FLAG_NO_BUFFERING`, sector-aligned) into the pinned ring; completion via events/`std::promise`. io_uring is Linux-only; I/O rings (`WriteFileEx`+APC or `GetQueuedCompletionStatus`) optional v2. mmap page-fault loading is simpler but stalls the consumer thread — acceptable for *first* load, wrong for per-token expert misses. |
| **GPU backend** | Vulkan: `volk` (loader) + `Vulkan-Hpp` headers + own suballocator (or VMA). | volk = meta-loader for entry points (https://github.com/zeux/volk). Vulkan-Hpp optional sugar. VMA (GPUOpen VulkanMemoryAllocator) is battle-tested but own slab/arena is ~400 LOC and gives ring-buffer semantics we need anyway. HIP SDK 7.2 on Windows officially supports RX 9070 XT/gfx1201 (AMD docs) — keep a backend seam so a HIP port is a plan-B, but Vulkan is the primary: single API for Windows+Linux, coopmat/WMMA on RDNA4, and ggml-vulkan proves perf. |
| **Threading** | `std::jthread` + a bounded MPMC work queue; optionally `enkiTS` for CPU-side kernels. | Thread roles are few and fixed (see §5) — a fiber/job system is over-engineering for 4 threads. enkiTS (https://github.com/dougbinks/enkiTS) only if CPU-expert GEMMs need fork-join parallelism; otherwise `std::jthread`+`atomic_flag` spin pools (fastllm's `AliveThreadPool` model) suffice. |
| **GPU↔CPU sync** | Timeline semaphores everywhere + one `VkFence` per "epoch". | ggml-vulkan's pattern: transfer queue signals `transfer_semaphore` (timeline, monotonic counter); compute queue `wait`s on value N before consuming slot N; CPU reads semaphore values (`vkGetSemaphoreCounterValue`) instead of fence-polling. Events (VkEvent) for coarse "logits ready" readback; `vkCmdWaitEvents` for device-side. A `ggml_backend_event`-like opaque wrapper keeps it backend-agnostic. |
| **JSON/config** | `nlohmann/json` or `simdjson` | header+config parsing. |
| **CLI/logging** | `fmt` + `spdlog` (or `argparse`). | |

## 5. Recommended decomposition + threading model

### 5.1 Modules (each ≈ one dir, one owner)

```
ink/
  core/      Tensor view (shape+dtype+ptr), Dtype enum (NVFP4, MXFP8(E4M3/E5M2),
             FP16/BF16/FP32), small ops registry. Static-shape: no dynamic DAG.
  io/        safetensors.cpp (mmap+header), weight_source.cpp
             (descriptor: {shard file, offset, bytes, tier}), staged_read.cpp
             (pinned-slot ring, thread pool, ReadFile loops).
  gpu/       backend.h vtable {alloc_slab, enqueue_op, copy_h2d/d2h, semaphore}
             vulkan/ (volk init, device detect a la get_device_architecture,
             pipeline cache, cmd-buffer pool, transfer queue, timeline sems)
             later: hip/ behind same vtable.
  model/     inkling2.cpp — THE forward pass: embed → 24 layer dispatch
             (12 base + 12 looped×3) → head; expert_dispatch.cpp
             (router → top-8 → placement decision → fetch schedule);
             mtp.cpp — draft head + verify/rollback.
  kv/        page_pool.cpp, swa_window.cpp, seq_table.cpp.
  sched/     request queue, token budget, one-deep overlap pipeline (§3.2),
             spec-decode bookkeeping.
  serve/     (optional) CLI/chat REPL + minimal HTTP later.
```

Fixed-shape execution: build the op list once per (batch,phase) shape —
*prefill* vs *decode* vs *draft-verify* — not per token. ~4 recorded shapes.

### 5.2 Threads (4 roles, fixed)

1. **GPU/driver thread** — owns compute queue. Receives "step plans" (op list +
   resident-expert bitmap), records/executes the decode step, signals
   `gpu_timeline`. Also owns the transfer-queue submits for expert uploads.
2. **I/O pool (2–3 threads)** — pull fetch descriptors (layer, expert ids for
   *next* step, predicted from router outputs as soon as they're computed —
   i.e. mid-step, not at step end), `ReadFile` experts from mmap/NVMe into
   pinned ring slots, enqueue H2D copies on the transfer queue. Slot lifecycle
   tracked by timeline values (slot reusable when `gpu_timeline >= slot.last_use`).
3. **CPU expert pool (0–N threads)** — optional PowerInfer-style path: cold
   experts' GEMV on CPU with AVX2/AVX-512 kernels while GPU does hot ones;
   combine on CPU (ik lesson §1.5: combine next to the weights, 9× fewer
   activation copies). Disable initially; keep the seam.
4. **Frontend/scheduler thread** — tokenize, sample (needs one small D2H
   readback of logits through the pinned ring each step), HTTP. Never blocks
   GPU thread; one-step-behind like SGLang's `result_queue`.

Pipeline per decode step (top-8 of 128 experts, 36 effective layer-passes):
GPU does attention+router → router logits → (on-GPU top-k or tiny readback) →
expert residency check → miss list goes to I/O pool *immediately* → GPU
continues with resident experts + next-layer attention while transfers land →
MoE combine runs once expert GEMMs complete → logits → sample on CPU
(one-step-behind). MTP: draft head uses same loop; verify is just a batch.

Key sizing notes for our box: 16 GB VRAM ⇒ attention tensors + KV + ~one
layer-block of hot experts resident; ~30+ GB RAM ⇒ most of 128-expert×3-matrix
working set warm in page cache after first pass; NVMe ~5–7 GB/s ⇒ per-token
expert miss latency is the *defining* constraint — hence the
router-early-dispatch and the ring-buffer design. Both ik_llama's and fastllm's
data say the same: static "which experts live where" beats clever LRU for
throughput; offer both.

## 6. Build vs fork vs embed — honest comparison

| Option | Effort | Control | Perf ceiling | Risk |
|---|---|---|---|---|
| **From scratch** (this design) | High — Vulkan kernels for NVFP4 GEMM/GEMV, FA, RoPE, sampling = weeks of kernel work; but *only* ~6 kernel families + ~3k LOC of orchestration | Total — exact tiering/pipeline for 16GB+48GB+NVMe | Highest potential (we control layout → coopmat-friendly NVFP4 dequant-fused GEMV; no ggml genericity tax) | Writing correct fast Vulkan matmul is hard; no fallback if it's slow |
| **Fork llama.cpp** (ggml+llama-*) | Medium — architecture mostly exists; add `inkling2_loop` model (~1 file), NVFP4/MXFP8 types, and a custom buft for expert ring | Medium — fight generic tensor/graph model, GGUF assumptions, upstream churn | Good — Vulkan backend already does RDNA4 coopmat + fused ops; but generic op granularity blocks best fusion | If `inkling2_loop` is landing upstream anyway, contributing upstream gets free maintenance; forking without upstreaming = permanent merge debt |
| **Embed fastllm** | Medium-low — it literally does tiered MoE | Low — CUDA-centric GPU path (no Vulkan), 700 KB model files, fat `Data`, Chinese-language codebase | Low on RDNA — its GPU path is CUDA; ROCm support secondary | Worst of both: inherit its complexity without its main asset |

**Recommendation: middle path.** Write our own thin runtime (the module map in
§5) but **extract, don't rewrite**: lift ggml-*concepts* (buft/buffer/sched
split/events) and ggml-vulkan *code patterns* (cmd-pool, incremental submit,
transfer-queue+timeline, host-import) — not the codebase. Two concrete
borrow-not-fork decisions:

1. **Use ggml-vulkan's proven kernel approach but with our op set.** Even
   better, spike: link ggml (it's a clean C library) for CPU-side tensors +
   optionally its Vulkan backend as the *first* working GPU path, then replace
   hot ops with our fused NVFP4 kernels under our own scheduler. ggml is
   vendored, not forked — we keep our scheduler/IO/expert-ring, they provide
   math. This de-risks "no working GPU path" day one.
2. **Keep the expert tiering ourselves.** Neither upstream's `-ot`+LRU nor
   fastllm's caches give us router-early-dispatch prefetch into a ring buffer
   with MTP-aware lookahead; that's our differentiator and it's ~1k LOC.

Revisit fork decision only if (a) `inkling2_loop` lands upstream making
maintenance free, or (b) Vulkan kernel work stalls — forking ggml for kernels
while keeping our own sched/io is the fallback.

## 7. Open questions / risks

- **NVFP4 on RDNA4**: ggml-vulkan has no NVFP4 path; we'll need a custom
  dequant+WMMA (coopmat) kernel. MXFP8 (E4M3) maps to `VK_EXT_shader_float8` —
  which is exactly ggml's RDNA4 probe (§1.4), so RDNA4 FP8 is driver-exposed.
- **Vulkan lacks cudaGraph exec objects**: mitigation = cached secondary
  command buffers per shape (§1.4) — unproven perf claim, needs a spike.
  Fallback is ggml's incremental-submit model which is already adequate.
- **mmap vs explicit reads** for experts: mmap+`PrefetchVirtualMemory` is the
  zero-code path; explicit `ReadFile` gives deterministic ordering + progress
  reporting. Start mmap, measure fault-stall distribution, escalate if needed.
- **MTP draft KV**: speculative KV pages need fork/rollback — design the page
  pool refcounting early; retrofitting hurts.
- **48 GB RAM is the real cache**: aim for expert footprint ≤ ~28–30 GB resident
  (page cache counts). If the whole expert set is ~20 GB (NVFP4), everything
  changes — static-resident in RAM with fast H2D staging may beat any cache.
