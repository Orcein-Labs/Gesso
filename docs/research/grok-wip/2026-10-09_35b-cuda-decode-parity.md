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
| lane m | tiled prefill attention, P8192 in 26.4 s, split-K decode attention | 61.7 tok/s, ids ok | P4096 43.9 tok/s, P8192 39.9 tok/s (was 20.3, 11.9) | 398/398 |

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

### Long prompts (lane m, 314eef7 -> next commit)

The goal here is the longest context that fits next to the weights, with
decode speed that holds up as the context grows. 314eef7 split the prefill
into passes. This round fixed what the long prompts broke after that.

**Prefill memory.**

- **Attention tiles the keys.** `_attention_prefill_device!` reads the KV
  heads in place (GQA, no repeat) and runs an online softmax over key tiles.
  Its score scratch is (tile, L), whatever K is. The reference path needs the
  KV heads repeated (1 GiB at 32k keys) and two (L, K) score buffers.
- **Passes size themselves.** A pass steps down from 1024 rows until its
  estimate fits in what a purge can free. P4096 in fixed 1024-row passes ran
  out of memory next to hot+pin pools.
- **Gesso keeps its own cuBLAS handle** (`ext/cuda_blas.jl`). A failed
  pool allocation walks CUDA.jl's reclaim ladder to DROP, which destroys
  idle cuBLAS handles. The next `cublasCreate` is a driver allocation with
  no retry, and it failed mid-prefill. The handle is created before the
  weights are placed, and its state follows `CUBLAS.handle()`.
- **A pool purged during a prompt streams through the ring for the rest of
  that prompt.** Re-allocating it took two full GCs (~0.36 s) per layer, and
  a later layer purged it again. At P8192 that was 15-30 s of a 38-54 s
  prefill. P8192 now prefills in 26.4 s with no GC. P256-P1024 fit in one
  pass and never purge, so they don't change.
- **Lazy score buffers, newest hidden row only.** Keeping every hidden row
  cost 8 KiB a token, 256 MiB at 32k.

**Decode after a long prompt.** The fixed tree first measured 21.8 tok/s for
tokens 2-65 of a P4096 generation and 16.4 tok/s for tokens 65-129. Short
prompts ran at 37 tok/s. Four separate causes:

- **GC stalls.** CUDA's allocation retry runs two full GCs. In decode they
  free nothing: after P4096, live memory equalled used memory (6835 MiB) at
  all seven GCs, because that generation's KV holds the memory. They cost
  1.7 s of 64 tokens. Pool allocations now retry with the GC disabled. A pool
  that doesn't fit waits `_HOT_RETRY_TOKENS` decode tokens and then tries
  again. It used to be retired for good, which cost 62 -> 34 tok/s on every
  later generation.
- **The `permutedims!` alias leak.** GPUArrays launches on `vec(src)`, a
  derived array that keeps `src`'s memory alive until a GC. The arena's
  `unsafe_free!` of each prefill FFN input freed nothing, and P4096 carried
  32 MiB into decode. `_transpose_into!` passes an unowned alias instead.
- **Prefill-only scratch.** The device-bank MoE batch and the dense F16 GEMM
  scratch (78 MiB) are now freed before decode, with the D buffer and the ring.
- **Decode attention read the cache 8 times.** The per-head kernel ran one
  block per query head, so each of the 2 KV heads' caches was streamed once
  for each of its 8 query heads, at ~40 GB/s. Split-K (flash-decoding)
  replaces it. A block takes one KV head and a chunk of C keys
  (C in 32/64/128, sized for ~128 blocks). It scores all 8 query heads from
  one read of each key tile staged in shared memory, and writes the chunk's
  unnormalised output, max and sum. A second kernel folds the chunks. Groups
  other than 1/2/4/8 keep the per-head kernel.

One layer at the 35B shape, microbenchmark:

| K | per-head (us) | split-K (us) | split-K GB/s |
|---|---|---|---|
| 1 | 52 | 14 | - |
| 1024 | 325 | 37 | 112 |
| 4096 | 1291 | 92 | 182 |
| 16384 | 11141 | 395 | 170 |
| 32768 | 22226 | 786 | 171 |

The maximum difference from the per-head kernel is 1.1e-6. The ideal at
400 GB/s is 42 us at K=4096, so there's about 2x left. At 0.9 ms per token
over 10 layers, that 2x isn't worth chasing yet.

End to end: one generation of 65 tokens per row, per-token timing (sum of the
synced head and the decode step, tokens 2-64). Both kernels ran in the same
process, with `GESSO_GPU_HOT_EXPERTS_MIB=480` so the pool budget doesn't
vary between rows:

| prompt | per-head tok/s | split-K tok/s | prefill |
|---|---|---|---|
| P1 | 48.8 | 46.5 | - |
| P4096 | 20.3 | 43.9 | 10-11 s |
| P8192 | 11.9 | 39.9 | 22.5 s |

P1 is within run noise. Gate after the change: 61.7 tok/s, ids ok. Targeted
CUDA tests (import_llama, attention_decode, ops, quant_dense, hybrid, moe,
inference): 398/398. The decode-attention oracle gained cases for the 35B
shape at 1 and 4100 keys, a partial last chunk, and a group that keeps the
per-head kernel.

**Measurement caveats.**

- Taking the difference of two generations (n=65 minus n=1, each with its
  own prefill) is not a decode measurement at long prompts. Prefill varies by
  about 1 s, and one such difference read 160 tok/s.
- The hot-pool budget is set from free memory at registration. It came out
  at 13 MiB per layer in one run and 22 MiB in the next, which moved the
  short-prompt baseline from 37 to 45 tok/s. Fix
  `GESSO_GPU_HOT_EXPERTS_MIB` when comparing.

### Quantized KV (lane m, 5af6b9d -> next commit)

16k and 32k prompts ran out of device memory. A probe after load shows
weights at 6097 MiB, with 860 MiB available once the pools are reclaimed.
The F32 KV cache is 40 KiB a token over the 10 attention layers, which is
640 MiB at 16k, so F32 fails at 16384 and passes at 12288.

`GESSO_KV_FORMAT` now picks the cache format for a run. Unset, it is q8
on CUDA and f32 on CPU (f2bcfda). With the variable unset, P16384 decodes
at 22.99 ms/token where f32 runs out of memory. The decode gate with q8
passes with exact ids at 61.1 tok/s. `reference_generate` is the only
path that runs the 35B: `Session` validates for dense FFN blocks and
rejects it.

| format | storage | KiB/token | 16k | 32k |
|---|---|---|---|---|
| f32 | Float32 | 40 | 640 MiB, OOM | 1.25 GiB |
| f16 | Float16 | 20 | 320 MiB | 640 MiB, OOM |
| q8 | Int8, Float16 absmax per 32 | 10.6 | 170 MiB | 340 MiB |
| q4h | 4-bit Lloyd-Max on a Hadamard-rotated row, Float16 RMS per 32 | 5.6 | 90 MiB | 180 MiB |

q8 and q4h complete prefill and decode at 16k, 24k and 32k.

q4h rotates each K and V row by a signed Walsh-Hadamard transform before
quantizing. That spreads outlier channels, so each block is close to
Gaussian. The rotation is orthonormal, so the query takes the same rotation
and the attention output is unrotated once per head. RoPE is untouched.

**Drift.** Teacher-forced: f32 generates 32 greedy tokens from a real-text
prompt (tokenized with the GGUF's own BPE), then each format replays the
same ids, so the logits compare step by step. The control is f32 against
f32. Its KL is GPU run-to-run nondeterminism, the floor any format is
measured against.

| P | format | top-1 | KL mean | KL max |
|---|---|---|---|---|
| 2048 | f32 (control) | 30/32 | 4.1e-3 | 2.0e-2 |
| 2048 | f16 | 30/32 | 6.1-6.3e-3 | 2.7e-2 |
| 2048 | q8 | 30-31/32 | 5.7-7.8e-3 | 2.6-4.9e-2 |
| 2048 | q4h | 29/32 | 1.8-2.1e-2 | 1.9-2.3e-1 |
| 8192 | f16 | 28/32 | 1.4e-2 | 6.7e-2 |
| 8192 | q8 | 31/32 | 1.3e-2 | 7.3e-2 |
| 8192 | q4h | 28/32 | 5.7e-2 | 4.5e-1 |

The ranges span two runs. At P=8192 the f32 control replay itself ran out
of memory, so those rows have no floor. q8 matches f16 at both lengths, and
f16 is llama.cpp's default cache type. q4h sits at about 4-5x the floor. No
task-level score exists yet, so q4h stays opt-in.

**The decode kernel.** The split-K kernel's generic tile loader reads one
element per thread through `getindex`. For a quantized cache that means a
code load, a scale load and a decode per element, so the first q8 decode
ran 4-8x slower than f32 at long K. The loader now has a specialized method
per format. Each thread loads 4 consecutive rows of a column in one
instruction: UInt32 of codes plus one UInt64 for the 4 Float16 scales (q8),
UInt32 for 4 rows × 2 dims (q4h), or UInt64 (f16). Every load is issued
before any store. A tile's 32 dims are exactly one scale block. The cache
allocates its rows padded to a multiple of 4, so every column is 4-row
aligned. Readers take K explicitly, so the tail rows are never read.
Unaligned caches fall back to the generic loader.

One layer, split-K, microseconds. The generic number is the same kernel on
an unaligned cache. The maximum difference between fast and generic is 0:

| K | f32 | f16 (generic) | q8 (generic) | q4h (generic) |
|---|---|---|---|---|
| 4096 | 97 | 89 (96) | 97 (373) | 107 (450) |
| 16384 | 469 | 318 (518) | 293 (2259) | 324 (1159) |
| 32768 | 783 | 595 (916) | 618 (2472) | 641 (2383) |

At small K the quantized formats cost about 28 us more per call than f16,
in both paths. That is launch overhead, about 0.3 ms a token. Not fixed.

**Decode by format.** One generation of 65 tokens per row, per-token
timing over tokens 2-64, `GESSO_GPU_HOT_EXPERTS_MIB=480` with pinning, ms
per token (tok/s):

| prompt | f32 | f16 | q8 | q4h |
|---|---|---|---|---|
| P1 | 21.9 (45.7) | 20.9 (47.9) | 20.0 (49.9) | 20.5 (48.9) |
| P8192 | 25.3 (39.6) | 22.9 (43.7) | 22.0 (45.5) | 21.2 (47.1) |
| P16384 | OOM | 26.7 (37.4) | 24.1 (41.6) | 23.6 (42.4) |
| P32768 | OOM | OOM | 29.3 (34.2) | 29.5 (33.9) |

The same table before the loader change, from an earlier process, had q8
at 25.9 / 32.2 / 40.4 ms (P1 / P8192 / P16384), slower than f32 at 8k. That
process's f32 P1 control read 24.4 ms against 21.9 here, so about 10% of
the gap is run conditions. The rest is the loader. q8 at 32k decodes faster
than llama.cpp's short-context 32.8 tok/s. Prefill at 32k takes 100-130 s.

Gate after the change: 56.8 tok/s, ids ok. Targeted CUDA tests
(import_llama, kv_format, cuda_attention_decode, ops, quant_dense, hybrid,
moe, inference): 996/996. The fused-decode oracle now covers an aligned
cache, an aligned cache with a partial last row group, and an unaligned
cache, for f16, q8 and q4h.

**Measurement caveat.** One gate run read 41.5 tok/s with this code. At the
time the host had 18 GB of swap in use and kswapd0 at 99%, so the CPU
expert path was taking page faults. The rerun with swap settled read 56.8.
Check host memory before reading a gate drop as a regression. Separately,
the synced decode profile serializes the hot-pool copies, so its attn stage
can't be used to attribute attention time with hot pools on. Use the
microbenchmark.

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
5. **Long context.** q8 is at the f16 drift level and decodes faster than
   f32 at every length, so it is now the CUDA default cache. For multi-turn
   use the bigger cost is that `reference_generate` keeps no KV between
   calls: every turn re-prefills the whole prompt. Past 32k, the
   limits are prefill time (130 s at 32k, quadratic attention) and the
   ~860 MiB headroom. The next steps are a task-level quality check before
   q4h goes beyond opt-in, and compaction or subquad attention for prefill.

## Reproduce

    GESSO_ORNITH35_GGUF=<gguf> julia --project=benchmark -t 16 scripts/gate_35b.jl cuda
    GESSO_GPU_HOT_EXPERTS=1 GESSO_GPU_HOT_PIN=1 GESSO_GPU_BUDGET_MIB=4096 \
      GESSO_ORNITH35_GGUF=<gguf> julia --project=benchmark -t 16 scripts/gate_35b.jl cuda

Any run takes `GESSO_KV_FORMAT=f32|f16|q8|q4h` for the KV cache format
(unset: q8 on CUDA, f32 on CPU).

Raw logs (per-run medians, hit rates, profiles) stay in local scratch. They
are not committed.
