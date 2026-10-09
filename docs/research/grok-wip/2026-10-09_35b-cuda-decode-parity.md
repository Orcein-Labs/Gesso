# 35B CUDA decode: parity with llama.cpp, then hot experts

    Document class:     findings (one day's campaign)
    Date:               2026-10-09
    Branch:             grok-wip (fast-forwarded to 58af083; lane-l/opus-int same tip)
    Model:              Ornith-1.5-35B-A3B Q2_K_XL (GGUF via GESSO_ORNITH35_GGUF)
    Target:             llama.cpp tg64 32.80 tok/s (30.5 ms/token), --n-cpu-moe 24
    Acceptance:         scripts/gate_35b.jl cuda (exact ids) + targeted CUDA tests

## Result

| tip | what | gate | tg64 / medians | tests |
|---|---|---|---|---|
| 8b32b3f | start of day | 7.7-7.9 tok/s | | |
| db466e1 | parity: cached launches, prefault | 36.64 tok/s, ids ok | 27.3-28.6 ms | 366/366 |
| 58af083 | + hot experts, pinned (opt-in) | 39.9 tok/s, ids ok | best 18.7-19.1 ms, typical 20-26 ms | 366/366 |

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
