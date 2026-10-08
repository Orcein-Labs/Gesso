# RPDO spiral — Lava decode slice

Owner: Grok. Repo: Gesso. Layer: Gesso mechanism. Date: 2026-10-07.

Problem: Regime I Lava warmed greedy is ~2.5 tok/s vs Gesso CUDA ~30
tok/s on SmolLM2-135M. This spiral asks what to build, then builds
only what the oracle keeps.

## Load-bearing questions

| Q | Cheap experiment | Disposition |
| --- | --- | --- |
| Q1. Are per-op `KA.synchronize` calls most of the 416 ms/token? | Count+time `_lava_sync!` on 8 warmed Hello decodes. | `PARK` "one-wait is the 12×". Count hypothesis kept: 605.125 waits/token (predicted 605). Time hypothesis killed: sync is 10.8% of audited decode wall. Receipt `arm1-hello.json`. |
| Q2. Do `_all_finite` scalar readbacks add a second wait class? | Count+time `_all_finite`. | `BUILD` skip on decode rms/softmax. 332 finites/token, 6.0% of wall. Together with Q1 this is ~17% mechanical, not 12×. |
| Q3. Can decode drop per-op sync and keep greedy IDs exact? | Arm 2: `_lava_after_op!(::DecodeWorkload)=nothing`; boundary waits once; decode rms/softmax `check_finite=false`. Oracle: Hello IDs vs HF + planted NaN in `s.h` after prefill. | `BUILD` retain. `arm2-hello.json`: IDs exact, planted NaN true, waits/token 1.125, **3.097 tok/s** (was 2.537). Sync 0.07% of wall. ~22% mechanical win. Not the 12×. |
| Q4. After one wait, which op bodies own the leftover wall? | Same audited decode: per-op ns for embed/rms/rope/matmul/softmax/swiglu. Residual is attention GEMM + host argmax + GPUArrays outside those helpers. | `BUILD` fused softmax. Softmax **62.1%** of audited wall (2160 calls, ~845 µs each). Residual 22.0%. rms 8.0%. rope 4.9%. matmul 2.0%. swiglu 0.5%. embed 0.1%. |
| Q5. Julia compiler fork? | Only if Q1/Q4 show SPIR-V/compile, not wait/launch. | `PARK` until a named receipt |

## Laws in force

RPDO: candidate → oracle (greedy IDs exact, planted NaN still
`ERR_NUMERICAL_INSTABILITY`) → measure post-warmup → retain or revert.

Caps: no new modules, no new deps, no `libs/` edits.

## Arm 1 evidence (closed)

`docs/archive/2026-10_regime-ii/arm1-hello.json`. RTX 5060, SmolLM2-135M,
Hello 8 tokens, IDs exact.

- `syncs_per_token=605.125` (8×605 + 1 trailing probe sync)
- `finites_per_token=332.0`
- unaudited median 3.153 s / 8 = **2.537 tok/s**
- `sync_fraction=0.1077`, `finite_fraction=0.0599`
- leftover ~83% is op bodies (GPUArrays broadcast, mapreduce, RoPE
  host+upload, softmax index grids, 9-head loop, fill!+mul!, host argmax)

## Arm 2 candidate (RETAIN)

Permitted files: `ext/lava_ops.jl`, `test/test_numeric_lava.jl`,
`test/regime_ii_attribution.jl`. Prefill per-op sync stays.

`arm2-hello.json`: unaudited median 2.583 s / 8 = 3.097 tok/s
(arm 1 was 2.537). IDs exact. Planted NaN at logits. waits/token 1.125
(8 boundary + 1 trailing probe sync). finites/token 1.0 (logits only).

## Arm 2 leftover (Q4)

Audited 8-token Hello, fractions of 2.942 s:

| bucket | fraction | n | notes |
| --- | ---: | ---: | --- |
| softmax | 0.621 | 2160 | 270/token; index grids + 2 scalar dimreduce roundtrips/call |
| residual | 0.220 | — | 9-head score/value GEMM in Inference.jl + host argmax |
| rms | 0.080 | 488 | still scalar `_lava_dimreduce` on 1-row decode |
| rope | 0.049 | 240 | host angle table + upload every layer |
| matmul | 0.020 | 1688 | fill! + mul! |
| swiglu | 0.005 | 240 | broadcast |
| embed | 0.001 | 8 | |
| sync + finite | 0.003 | 9+8 | one-wait did its job |

## Arm 5a candidate (RETAIN): fused softmax

`arm5a-hello.json`: IDs exact, planted NaN true, **7.691 tok/s**
(arm 2 was 3.097; arm 1 was 2.537). Softmax 62.1% → **4.2%**
(845 µs/call → 24 µs). New split of 1.040 s unaudited median:

| bucket | fraction |
| --- | ---: |
| residual (attn GEMM + host argmax) | 0.535 |
| rms | 0.247 |
| rope | 0.106 |
| matmul | 0.048 |
| softmax | 0.042 |
| swiglu | 0.012 |

P vs CUDA 29.93 ≈ 0.257. Still ~3.9×.

## Arm 5b (RETAIN): decode rmsnorm row kernel

`arm5b-hello.json`: **8.657 tok/s**. rms 24.7% → 1.2%. IDs exact.

## Arm 5c (weak retain) / 5d (RETAIN): RoPE kernel then scalar position

`arm5c-hello.json`: 9.32 tok/s but rope_ns stayed ~210 ms — the per-call
4-byte `LavaArray` position upload ate the kernel. `arm5d-hello.json`:
scalar `Int32` position, **11.90 tok/s**, rope 21% → 1.7%. IDs exact.

## Arm 4 / 10F (RETAIN): Lava split/merge/repeat/add + per-head attn kernels

`arm4-hello.json`: **28.67 tok/s**. IDs exact, planted NaN true.
P vs frozen CUDA Hello 29.93 = **0.958**. Residual 74% → still 58% of a
much smaller pie (229 ms / 8 tokens).

Hello unaudited median 0.279 s / 8.

| step | tok/s | P vs CUDA 29.93 |
| --- | ---: | ---: |
| arm 1 (count) | 2.54 | 0.085 |
| arm 2 (one-wait) | 3.10 | 0.103 |
| arm 5a (softmax KA) | 7.69 | 0.257 |
| arm 5b (rms KA) | 8.66 | 0.289 |
| arm 5d (rope scalar) | 11.90 | 0.397 |
| arm 4 (10F attn) | 28.67 | 0.958 |

## Arm 3 (RETAIN): `mul!(C,A,B,α,0)` drop fill!

`arm3-lava-three.json`, IDs exact on all three prompts (27/27).
Decode tok/s vs frozen CUDA arm 11:

| Prompt | Lava | CUDA (frozen) | P |
| --- | ---: | ---: | ---: |
| Hello | 28.72 | 29.93 | 0.960 |
| The quick brown fox | 27.58 | 28.81 | 0.957 |
| Julia is a programming language. | 26.79 | 28.80 | 0.930 |

Median P = 0.957 ≥ 0.90 vs the frozen arm-11 CUDA receipts named in
`docs/goals/REGIME_II.md`.

Same-session CUDA (`arm6-cuda-three.json`) ran hotter on fox
(33.84 decode tok/s vs frozen 28.81):

| Prompt | Lava | CUDA (live) | P live |
| --- | ---: | ---: | ---: |
| Hello | 28.72 | 31.31 | 0.917 |
| The quick brown fox | 27.58 | 33.84 | 0.815 |
| Julia is a programming language. | 26.79 | 29.64 | 0.904 |

Frozen gate is green. Live fox is the leftover. Next spiral: batched
attention (longer K), host alloc 6–8 MB vs CUDA 1 MB, device argmax.

## Closed

Q1–Q4 have dispositions. Q5 compiler fork stays `PARK`. This spiral's
build path is retain-all: one-wait, fused softmax, rms row, scalar
RoPE, 10F attn/layout kernels, β=0 mul!.

Compiler fork, representation, half, GGUF stay `PARK`.
