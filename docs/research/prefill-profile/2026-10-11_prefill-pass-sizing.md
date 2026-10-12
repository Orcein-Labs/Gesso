# Prefill pass sizing and hot experts (2026-10-11)

Ornith-35B Q2_K_XL on an RTX 5060 (8 GB), 12 host threads, 4096-row prompt.
This explains why the defaults now run hot+pinned experts with a 4800 MiB
weight budget and a 1024 MiB reserve, and what is still slow.

## The profiler

Set `GESSO_PREFILL_PROFILE=1` (commit 091c7e0). It is off by default.

Each prefill pass prints one line:

    PFPASS i=<pass> n=<rows> K=<keys> wall_ms=... attn_mixer=... moe_host=... moe_h2d=... other=... h2d_mb=... h2d_ms=...

After the last pass it prints one summary line:

    PFSUM passes=<N> rows=<total> wall_ms=... <stage>=... other=...

Stages are disjoint. `other` is wall time minus the stage sum. `h2d_mb` counts
expert bytes streamed from host to device.

## Measurements

| config | chunk 128 | chunk 512 | chunk 1024 |
|---|---|---|---|
| hot=1 pin=1 budget=4800 reserve=1024 | 244.5 tok/s (32 passes) | 405.8 (8 passes) | 467.0 (4 passes) |
| hot=1 pin=1 budget=5632 reserve=1024 | 246.8 (32 passes) | 251.2 (32 passes) | 250.9 (32 passes) |
| hot=0 pin=0 budget=4800 | 66.0 | 64.9 | 59.5 |
| hot=0 pin=0 budget=5632 | 75.0 | 76.0 | 74.5 |

Findings, from PFSUM over the whole prompt:

- Cold path (no hot pool): `moe_host` (CPU experts) is 57,000 to 60,000 ms of
  about 63,000 ms. Prefill is bound by CPU experts.
- Hot+pin, budget 4800, chunk 1024: `moe_host` 4,057 ms, `moe_qgemv` 2,326 ms,
  16.3 GB of expert slabs streamed host to device, wall 9,087 ms.
- At budget 5632 the weights leave about 1005 MiB free. The pass sizer then
  forces 128-row passes whatever chunk is asked for, so streaming never
  amortizes over a large pass.
- 32k prompt (32,740 rows, budget 4800): 256 passes of 128 rows, wall 450 s
  (73 tok/s). `moe_host` took 389 s (86%) and `h2d_mb` was 0. Once the KV cache
  is resident, the free VRAM only allows 128-row passes.

## The pass sizer

`_prefill_pass_rows` (ext/cuda_hot_experts.jl) picks the rows for each pass. It
starts at `min(chunk, rows left)` and steps down by 128 rows while the estimate
for the pass is over budget. The budget is free device memory, plus pool and
prefill buffers that a purge can return, minus the larger of 160 MiB and the
hot reserve.

The estimate in `_prefill_pass_bytes`, for n rows over K keys, is:

    bytes = headroom * n + 4 * n * tile(n, K) + dq_scratch + 2 * n * qdim

- `headroom` is 768 KiB per row by default (`GESSO_GPU_HOT_PREFILL_HEADROOM_KIB`).
  That is the source's own estimate of about 0.75 MiB per prompt token.
- `tile(n, K)` is the attention key tile: 4 MiB divided by n keys, rounded down
  to a multiple of 256, floored at 256 and capped at K (`_attn_prefill_tile` in
  ext/cuda_attention.jl). So `4 * n * tile` is about 16 MiB and does not grow
  with K.
- `dq_scratch` is 32 MiB (`GESSO_DQ_GEMM_SCRATCH_MIB`).
- `qdim` is the attention width, `n_heads * d_head`.

This formula is read from the code, not measured. The 0.75 MiB per row is the
source's estimate, and no run here checked it against the allocator.

The sizer never returns fewer than 128 rows. If even 128 does not fit, it
returns 128 and the purge runs. So the 128-row passes at budget 5632 and at 32k
are a floor the sizer hits, not a size it chose. With the KV resident, the free
memory left after the weights is below what larger passes need.

## Decision

Defaults now in source:

- `GESSO_GPU_HOT_EXPERTS=1`, hot expert pool on.
- `GESSO_GPU_HOT_PIN=1`, pinned slabs.
- `GESSO_GPU_BUDGET_MIB=4800`, leaves room for 1024-row passes and the pool.
- `GESSO_GPU_HOT_RESERVE_MIB=1024`, left free for the desktop or compositor.

The reason is throughput at 4k: 467 tok/s with hot+pin at chunk 1024, against
60 to 75 tok/s on the all-CPU path.

Hot experts change the numbers. Resident picks run the GPU kernels with q8_1
activations, and misses run the CPU kernels with Q8_K, so the output rounds
differently. Decode ids differ from the all-CPU path after about 12 tokens. A
2026-10-11 logit-KL probe found that equivalent prefill paths on this model
already differ by first-token KL 0.03 to 0.56 (single-pass vs chunk=1, and f32
vs q8). So this is precision noise, not a correctness defect. We have not
measured hot-vs-CPU KL directly. The claim is only that the divergence is of
the same kind the model already shows between equivalent paths.

## Next steps

- Long-context prefill is bound by CPU experts because passes shrink as the KV
  cache grows. Two levers:
  - Tiered attention: move old KV to pinned host RAM to free VRAM for larger
    passes.
  - Cut the per-row pass buffers (the headroom estimate and the `2 * n * qdim`
    term), which raises the row count that fits.
- Pinned slabs cost host RAM, about 7 GB at budget 4096 per earlier notes.
- Opt out with `GESSO_GPU_HOT_EXPERTS=0` (all-CPU experts). `GESSO_GPU_HOT_PIN=0`
  keeps the slabs pageable.
