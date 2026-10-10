# 35B CUDA: decode parity with llama.cpp, hot experts, then prefill

    Document class:     findings (one day's campaign)
    Date:               2026-10-09
    Branch:             grok-wip (fast-forwarded to 7f279eb; lane-l/opus-int same tip)
    Model:              Ornith-1.5-35B-A3B Q2_K_XL (GGUF via GESSO_ORNITH35_GGUF)
    Target:             llama.cpp tg64 32.80 tok/s (30.5 ms/token), --n-cpu-moe 24
    Acceptance:         scripts/gate_35b.jl cuda (exact ids) + targeted CUDA tests

## Result

| tip | what | gate | tg64 / medians | tests |
|---|---|---|---|---|
| 8b32b3f | start of day | 7.7-7.9 tok/s | | |
| db466e1 | parity: cached launches, prefault | 36.64 tok/s, ids ok | 27.3-28.6 ms | 366/366 |
| 58af083 | + hot experts, pinned (opt-in) | 39.9 tok/s, ids ok | best 18.7-19.1 ms, typical 20-26 ms | 366/366 |
| a136560 | hot config retuned (budget 4096, reserve 512) | 42.4 tok/s, ids ok | | |
| e46b4e3 | zero-alloc raw launches, typed caches | 47.53 tok/s, ids ok | | 366/366 |
| 46c94b2 | fused add+rmsnorm, KV append, qk-norm rows | 49.10 tok/s, ids ok | p9419 15.97 ms | |
| 7f279eb | prefill temporaries freed, pools purged on pressure | 45.95 tok/s, ids ok (host swapping) | p9419 15.23 ms, p1000 16.51 ms, 0 GCs | 590/590 |
| 16709c2 | prefill streaming ring, prefill buffers released before decode | 44.8 tok/s, ids ok | | 653/653 |

The gate is a short cold run and moves about +-2 tok/s between runs. The
medians come from `bench_ab`: six warm 32-token generations per prompt.
From 46c94b2 on, every row uses the hot+pin configuration.

On this box Gesso is faster than llama.cpp on the 35B with the default
settings: 27-29 ms/token against llama.cpp's 30.5 ms.

With `GESSO_GPU_HOT_EXPERTS=1 GESSO_GPU_HOT_PIN=1` and a smaller
`GESSO_GPU_BUDGET_MIB`, some prompts run at 52-53 tok/s. How fast a
prompt runs depends on how often its expert picks are already in VRAM.
This is why the median varies so much between prompts.

## The machine sets the floor

- **PCIe.** The GPU link is gen3 x8, about 6.5 GB/s. Uploading the ~173 MB of
  host-layer experts each token costs at least 27 ms, so host experts are
  computed on the CPU, as in llama.cpp's `--n-cpu-moe`.
- **DRAM.** The board has one DDR4 DIMM: a single channel at about 14 GB/s.
  Reading the routed host experts costs about 12 ms per token. Every byte read
  for the CPU GEMVs and every DMA read for a VRAM swap shares that one channel.
  Populating four channels is the largest single speedup available, and it
  needs no code.

## What moved the number (in order)

1. Host experts run on the CPU, with one staged D2H per host layer. Fused
   attention.
2. Pool workers are held across the 1-2 ms GPU gap (`pool_hold!`). A parked
   worker made the next layer's chunks about twice as slow.
3. SwiGLU+Q8_K quantization runs per expert in parallel (123 -> 27 us/layer).
   The output is bit-identical.
4. Warp-per-row dp4a MMVQ for the IQ2_XS/IQ3_XXS/IQ4_XS experts
   (7.35 -> 1.11 ms).
5. `JULIA_CUDA_GC_EARLY=false` by default. With the weights filling the card,
   CUDA.jl ran `GC.gc(false)` on every per-token allocation.
6. One-token kernels replace cuBLAS for the router, GDN a/b, the shared gate
   and the combine. Each cuBLAS call costs 12-20 us of host time.
7. `@fastcuda` caches the compiled kernel per (f, types, context). Plain
   `@cuda` repeats the `cufunction` lookup and the `with_managed`
   capture check on every launch, about 5 us each, over ~1400 ops per
   token. This change reached parity.
8. `madvise(MADV_POPULATE_READ)` on the host expert slabs at load. The
   50-92 ms first tokens were page faults.

## Hot experts (58af083, opt-in)

`ext/cuda_hot_experts.jl` keeps a VRAM pool of experts for each host layer.
Experts are ranked by a routing score that decays by 0.99 per token. Hits run
on the GPU and misses run on the CPU, and a kernel merges the two results.
A swap fills empty slots first and evicts at most one expert per token. The
new expert must outscore the coldest resident one by `GESSO_GPU_HOT_MARGIN`
(default 1.0).

Routing locality: the decayed top 32 per layer covers 62-65% of picks, and the
top 64 covers about 80%.

| GESSO_GPU_BUDGET_MIB | hybrid layers | hit rate | 64-token medians |
|---|---|---|---|
| 6144 | 21 | 0.33-0.56 | 21.3-25.5 ms |
| 4096 | 29 | 0.70-0.89 | 19.1-27.2 ms |
| 3072 | 33 | up to 0.92 | 18.7-25.6 ms |

Settings that did not help:

- **Margin 3 instead of 1.** Swaps fell to 0.13-0.31 per layer-token, but the
  hit rate also fell and the medians did not improve.
- **`GESSO_GPU_HOT_PIN=1`.** This copies the hot layers' slabs into pinned
  anonymous memory, about 7 GB at a 4096 budget. Prefetch host time fell from
  1-2.8 ms to 0.2-0.55 ms, but the medians did not change. Tokens with bursts
  of swaps reached 45-50 ms because the DMA reads compete with the CPU experts
  for DRAM. It is still the best configuration, because the gate improved.
- **Registering the file mmap with `cuMemHostRegister` in place.** It returns
  801 (NOT_SUPPORTED). Only anonymous memory can be pinned.
- **Issuing swaps inside the router wait.** No gain, because the wait is short.

Hot experts are off by default. Combining GPU hits with CPU misses rounds
differently from the all-CPU path: the median relative L2 error is 2% and the
maximum is 19%. Generated ids diverge from the all-CPU run around token 12.
The gate's 9 ids still match.

The recommended opt-in configuration is HOT=1, PIN=1, BUDGET 3072-4096,
MARGIN 1.

## Host overhead and the GC (lane m, 46c94b2 -> 7f279eb)

- **Launch cost.** `@fastcuda` launches now go through
  `unchecked_cuLaunchKernel` with reused argument buffers. Struct fields use the
  concrete `CuArray{T,1,DeviceMemory}`: a `CuVector{T}` field is abstract and
  cost a dynamic dispatch per launch. CPU allocations per token fell from 15.2k
  to 7.9k. Fused add+rmsnorm, KV append and qk-norm rows kernels removed most
  of the remaining GPUArrays broadcasts.
- **Every full GC was CUDA's allocation retry.** A trace of CUDACore's
  `reclaim_step` showed each 270-790 ms full GC coming from `_hot_alloc!`
  refilling a pool that the prefill had released. Prefill temporaries were
  left to the GC, and host allocations were too small to trigger one, so
  device garbage built up until an allocation failed.
- **Freeing the owner is not enough.** A reshape or contiguous view of a
  CuArray is a derived CuArray that holds its own reference to the parent's
  memory. `unsafe_free!(parent)` only drops one reference, so the memory stays
  until the derived array's finalizer runs. A 41-token prompt leaked about
  80 MiB per generation this way. The fix frees the derived array too, as soon
  as its last kernel is queued (`unsafe_free!` is stream-ordered). This covers
  the rmsnorm reshapes, the GDN column views, the rope positions, the embed
  staging and the gated-delta q/k normals. The live device delta after p1000
  x32 fell from 78.6 MiB to 0.1 MiB.
- **Pools are released on memory pressure, not on prefill length.**
  `_HotPurge <: CUDACore.Reclaimable` releases the pools at the PURGE step of
  the allocation retry, before any GC. It is a no-op while a hot token or a
  prefetch is running. A 41-token prompt used to drop all 29 pools and spend
  about 40 tokens refilling them. It now keeps them, and a P=1024 prefill keeps
  them too.
- **Batched host prefill.** Prefill tokens are grouped by expert. The output
  is bit-identical. Prefill runs at about 22-25 ms/token for P=41-1024, which
  is still slow (see next levers).

## Prefill (lane m, 22ebc45 -> ca67e01)

Prompt processing went from below decode speed to above llama.cpp at 256
tokens. Hot+pin configuration throughout. llama.cpp's reference is pp256
238.98 and pp1024 386.00 tok/s.

| tip | what | P256 tok/s (before -> after) |
|---|---|---|
| 22ebc45 | GDN recurrence tiled over value columns in shared memory | 48.5 -> 60.8 |
| 77fb404, 873bf22 | hot host layers' prefill partly on the GPU; all tokens batched on device-resident MoE layers | 62 -> 172 |
| 16f93c5 | dense prefill GEMM: weights dequantized to F16, then cuBLAS | 98 -> 218 |
| 35cec30 | prefill MoE routing on the GPU in one batched kernel | 210.8 -> 220.7 |
| 98628e6 | each layer's expert slabs placed on the device whole or not at all | 220.7 -> 278.7 |
| 75a6783 | GDN gated norm fused into one kernel | 258.8 -> 282.7 |
| ca67e01 | GEMM weights dequantized directly, not through a one-hot GEMV | 282.7 -> 403.8 |

Each before/after pair was measured back to back. The baselines drift
between sessions, so pairs don't chain exactly. At ca67e01: P64 381.6,
P256 403.8, P1024 328.7 tok/s.

- **The GEMM is more accurate than the GEMV it replaced.** Against a Float64
  host dequant, the F16 GEMM's RMS relative error is 1.4e-4 to 3.7e-4 per
  tensor kind. The Q8_1-activation GEMV's is 4.7e-3 to 1.9e-2. On real text
  the two agree on 37 of the first 40 greedy tokens. The split is the GEMV's
  rounding.
- **The one-hot dequant was most of the GEMM.** The first F16 tile ran the
  decode GEMV against one-hot activations, so it read every weight block
  once for each output group. A direct kernel uses one warp per (row,
  256-value superblock) and decodes each byte once. Per tensor at P256: Q5_K
  qkv 3.0 -> 0.26 ms, z 1.13 -> 0.067 ms, Q8_0 shared down 0.148 -> 0.009 ms.
  The whole qkv GEMM went from 2.6 to 0.5 ms. Output equals the old tile up to
  the sign of zero on Q6_K.
- **Hot experts make greedy output depend on history. This is not a race.**
  Hits run the GPU kernels (q8_1 activations) and misses run the CPU ones
  (Q8_K), and residency follows the routing history. So a prompt can decode
  differently after a different earlier prompt. The evidence:
  - With hot experts off, the top-2 logit gaps are bitwise identical after
    every history tried.
  - With hot experts on and `CUDA_LAUNCH_BLOCKING=1`, the dependence remains.
  - The model amplifies rounding through its routing choices: with hot experts
    off, swapping the fused gnorm (relative error 1.9e-7) for the per-head
    loop already moves the first logit gap by 0.1.
- **Ops.** A crashed test run left a 7.8 GB systemd core dump and tripped the
  disk guard. The local job wrapper now runs with `ulimit -c 0`.

Profile at P256, ca67e01 (profiled wall 688 ms, stages serialized):

| stage | ms |
|---|---|
| moe_host (CPU experts on hot host layers) | 165.9 |
| gdn recurrence | 122.9 |
| moe_qgemv (device experts, per-token GEMV) | 117.2 |
| gdn qk + proj + outproj | 47.6 |
| shared expert | 16.7 |
| router | 14.3 |
| moe_h2d | 10.3 |

### Streaming ring (16709c2)

A long prompt's activations purge the hot-expert pools in VRAM. A layer whose
pool was purged and could not be re-allocated fell back to the whole CPU MoE
path. That path costs about 60 us per pair, so each such layer added about
0.5 s at P1024 and about 1 s at P2048. A decode-time OOM had a second effect:
it set a layer's slot count to 0 permanently, so that layer never used the
GPU in prefill again.

The fix is a streaming ring. Layers without a usable pool stream their experts
through one small ring of device slots shared by all layers. The default is 32
slots, about 1.4 MB each (`GESSO_GPU_HOT_PREFILL_STREAM`). The ring reuses the
issue path that resident pools use. It double-buffers its uploads against the
GEMVs, so one expert's copy overlaps the previous expert's compute.

Decode now frees the prefill's grow-only device buffers and the ring before it
re-allocates a pool. The order matters: after a 4096-token prompt the D buffer
alone is 256 MiB, about two pools, so keeping it would starve the refill.
Layers with 0 slots stream too. The default `GESSO_GPU_HOT_PREFILL_MAX_SEQ`
rises from 1024 to 4096.

Results, 35B Q2_K_XL, hot+pin configuration (`GESSO_GPU_HOT_EXPERTS=1
GESSO_GPU_HOT_PIN=1 GESSO_GPU_HOT_RESERVE_MIB=512 GESSO_GPU_BUDGET_MIB=4096`):

| P | before (full-CPU fallbacks) | after (tok/s) | llama.cpp |
|---|---|---|---|
| 512 | n/a | 414 | - |
| 1024 | 324 (2 full-CPU layers) | 437 | 386 |
| 2048 | ~310 | 433 | - |
| 4096 | ~300 | 385 | - |

Full-CPU layers after the change: 0 at every length. GC time: 0. The verify
pass (GPU vs CPU per layer) has a max relative error of 2.7e-2 and a median of
2.3e-2. No layer exceeds 0.05.

Decode gate after the change: 44.8 tok/s, ids ok. The 7f279eb row was 45.95,
and the gate moves about +-2 between runs, so the gap is inside that spread.
Targeted CUDA tests: 653/653.

Open: the pools drop to 0 during a 4096-token prompt. Decode refills them over
the next generations. Warm decode measured 50-52 tok/s after the sweep.
Prompts over 4096 still take the CPU path. MoE token chunking would lift that
limit.

## Where the time goes now

Profiling syncs every stage, so these stage times are serialized: they add to
34 ms, against about 20 ms unprofiled.

| stage (ms) | hot+pin | off |
|---|---|---|
| moe_host | 6.4 | 20.2 |
| moe_qgemv | 4.1 | 0.9 |
| gdn | 7.0 | 6.6 |
| attn | 4.4 | 4.3 |
| shared | 3.3 | 3.2 |
| router | 1.5 | 1.5 |
| other | 3.5 | 2.7 |
| total | 34.1 | 42.8 |

The ~17 ms floor comes from the GPU stages, plus 29 or more router syncs
and serial host work per hybrid layer.

## Next levers, ranked

0. **Prefill past P256.** P256 beats llama.cpp, but P1024 (328.7) still
   trails 386. Three levers:
   - device experts as a grouped GEMM over each expert's tokens instead of a
     per-token GEMV (moe_qgemv, 117 ms at P256);
   - a chunked GDN recurrence on tensor cores instead of the sequential scan
     (123 ms);
   - fewer CPU experts on hot host layers (166 ms).
1. **Four DIMM channels.** About 4x the DRAM bandwidth for every miss and
   every swap. No code needed.
2. **The GPU floor.** Fuse gdn, attention and the shared expert. Cut the
   per-layer router syncs (top-k on the device, with host experts fed by a
   mapped flag instead of a sync).
3. **Swap rate limiting.** Spreads out the burst tokens (45-50 ms) behind
   the median.
4. **Zero-copy misses.** GPU reads of misses directly from pinned memory would
   remove the syncs and allow device-only tokens and CUDA graphs. But PCIe
   (6.5 GB/s) is under half of DRAM (14 GB/s), so this only pays at high hit
   rates. Not built.

## Reproduce

    GESSO_ORNITH35_GGUF=<gguf> julia --project=benchmark -t 16 scripts/gate_35b.jl cuda
    GESSO_GPU_HOT_EXPERTS=1 GESSO_GPU_HOT_PIN=1 GESSO_GPU_BUDGET_MIB=4096 \
      GESSO_ORNITH35_GGUF=<gguf> julia --project=benchmark -t 16 scripts/gate_35b.jl cuda

Raw logs (per-run medians, hit rates, profiles) stay in local scratch. They
are not committed.
