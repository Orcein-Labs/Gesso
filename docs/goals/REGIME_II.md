# Regime II — Lava decode to CUDA parity

Owner: Grok. Layer: Gesso mechanism. Palette and Cyan stay frozen.
Regime I is closed. This file is the execution plan. It is not a speed
claim.

No Julia compiler fork. No representation planner. No Magenta. No
speculative decoding. No GGUF. No native FP16/BF16. Those stay parked
until this plan's parity gate is green, or until a measured receipt
proves the remaining limiter lives there.

The CUDA path on this box already does ~30 tok/s on SmolLM2-135M with
the same `Session` interpreter. Lava does ~2.5 tok/s on the same
interpreter, same checkpoint, same prompts. The gap is in
`ext/lava_ops.jl` and how decode uses it. Close that gap with ordinary
GPU runtime work: fewer waits, fewer launches, fewer host round-trips,
then the same class of small kernels CUDA already has.

## Target

Same box, SmolLM2-135M, three Regime I prompts, host-visible greedy,
compile outside the timed region:

\[
P = \frac{\text{tok/s}_{\text{Lava}}}{\text{tok/s}_{\text{Gesso CUDA}}}
\]

**Parity gate:** median \(P \ge 0.90\) on all three prompts. Greedy IDs
exact against the independent reference. Report the three rows. Do not
average a dead kernel into a headline.

Comparator is Gesso CUDA on this machine. PyTorch eager stays a third
column. vLLM and vendor peaks are not this gate.

## Frozen baseline (Regime I arm 11)

RTX 5060, Julia 1.12.6, warmed 8-token greedy. Receipts
`docs/archive/2026-10_regime-i/receipts/arm11-lava-after.json` and
`arm11-cuda-after.json`.

| Prompt | Lava | Gesso CUDA | PyTorch eager | Lava host alloc / decode |
| --- | ---: | ---: | ---: | ---: |
| Hello | 2.40 tok/s (3.33 s / 8) | 29.93 | 21.58 | 36.4 MB |
| The quick brown fox | 2.49 | 28.81 | 21.90 | 37.2 MB |
| Julia is a programming language. | 2.49 | 28.80 | 22.26 | 38.0 MB |

That is **~416 ms/token Lava vs ~33 ms/token CUDA**. \(P \approx 0.083\).
Parity is an ~11× Lava decode gain with IDs held.

SmolLM2-135M on this tree: 30 layers, 9 query heads, GQA, dim 576,
vocab 49152 (`test/test_cuda_smollm2.jl`).

Hardware note, not a target: 135M decode is a fraction of a millisecond
of RTX 5060 math. CUDA at 33 ms/token is already launch-bound. Lava is
the same interpreter with a much more expensive device boundary. We
are not chasing a compiler miracle.

## Why Lava is slow (code, not folklore)

The CUDA extension launches kernels and returns. Queue wait happens
when a later host read actually needs the value.

The Lava extension **waits for the entire Vulkan queue after every
public operator**, by design of the Phase 8 seam:

```17:20:ext/lava_ops.jl
# Every op! ends with KA.synchronize on the KA backend: Lava dispatch is
# recorded/streamed, and the interpreter reads results back to the host
# AFTER the op returns — the sync at the op boundary is what makes any
# subsequent host readback correct (§LXXXI sync law; per-op is the
# correctness-first reading for this sprint, Phase 9 tunes).
```

```31:36:ext/lava_ops.jl
function Gesso.Inference._engine_boundary!(::LavaBackend)
    _lava_sync!()
    _lava_sync!()
    return nothing
end
```

CUDA `ext/cuda_ops.jl` has **zero** `synchronize` calls.

One decode token on Lava, 30 layers, from `_session_consume!` plus
`_session_greedy_id!`:

| Public op that calls `_lava_sync!()` | Per layer | ×30 |
| --- | ---: | ---: |
| `rmsnorm!` (attn + ffn) | 2 | 60 |
| `matmul!` (q,k,v,o,gate,up,down) | 7 | 210 |
| `rope!` | 1 | 30 |
| `softmax!` (once per head inside `_attention_heads!`) | 9 | 270 |
| `swiglu!` | 1 | 30 |
| `embedding_lookup!` (once per token) | — | 1 |
| final `rmsnorm!` + `lm_head` `matmul!` | — | 2 |
| `_engine_boundary!` (twice) | — | 2 |

**≥ 605 full-queue waits per token**, before counting implicit
readbacks.

Hidden host round-trips on the same path:

- `_lava_rmsnorm!` and `_lava_softmax!` call `_all_finite`, which
  `mapreduce`s to a scalar and reads it. That is a sync even if we
  delete `_lava_sync!()`.
- `_lava_softmax!` allocates device `Int64` index grids `1:L` and `1:K`
  **every call** to build a causal mask via `ifelse`.
- `_lava_rope!` rebuilds the angle table on the host (`Float64` `rem`,
  then upload, `cos.`, `sin.`) every layer.
- `_lava_embedding_lookup!` does `Lava.LavaArray{Int64}(collect(Int64,
  tokens) .+ 1)` every token.
- `_lava_matmul!` `fill!`s the destination to zero, then `mul!`.
- `_all_finite` allocates a fresh `LavaArray{UInt32}` temp every time.
- `_attention_heads!` is a Julia loop over 9 heads: slice, GEMM,
  softmax, GEMM. CUDA uses the same loop, but without a queue wait
  between them.
- `_greedy_id` / `_require_finite_logits` on Lava run `argmax` /
  `isfinite` on the host (session.jl comment: CPU and Lava reduce on
  the host). That D2Hs the vocab row (49152×F32 ≈ 196 KiB) every token.
  CUDA keeps argmax on device.

36–38 MB host alloc per Lava decode vs ~1 MB CUDA is the same story:
temps, index grids, angle tables, finite-check buffers, broadcast
output.

If each full-queue wait is even 200–500 µs, 605 of them are 120–300 ms.
That is most of the 416 ms token. GPU math is not the bill.

## Sequencing

Run in order. A skip is not a completion gate. Failure preempts the
active arm. Every speed change keeps Regime I greedy IDs exact and the
declared logit tolerances. Lava/Vulkan stays primary. CUDA stays the
comparator. Do not edit `libs/` as a silent fork; Lava pins stay in
the test workspace.

| Arm | Work | Why it is not exotic | Gate |
| --- | --- | --- | --- |
| 1 Count | Instrument decode: `_lava_sync!` count, implicit scalar readbacks, host alloc bytes, submits if Lava exposes them. One Nsight / Vulkan timestamp capture of warmed Hello. Re-run arm 11 harness into a new receipt; do not overwrite arm 11. | Counters. | `docs/archive/` JSON: waits/token, alloc/token, time split (wait / kernel / host). Ranked hypothesis. No speed claim. |
| 2 One wait | Decode-workload methods **must not** call `_lava_sync!()`. `_engine_boundary!` waits **once**. Prefill may keep per-op sync until decode is green. Move `_all_finite` off the per-op decode path: check logits once in `_greedy_id` (already happens). Keep a decode-only test that a planted NaN in rms still fails at the engine boundary or logits check. | Same change CUDA never had to make because it never waited. Phase 8 comment already named this as "Phase 9 tunes." | Wait count per token ≤ 2. IDs exact. Numerics suite still catches planted nonfinite. Publish \(P\). Expected: several× tok/s if arm 1 blamed wait. |
| 3 Stop allocating | Persistent decode workspace for softmax mask/rowmax/den, finite-check temp, RoPE angles (or a KA RoPE kernel that takes `pos` like `_cuda_rope_kernel!`). Embed from `ws.tok_buf` without a new `LavaArray`. `mul!` with β=0, drop `fill!`. | CUDA already has decode-specialized rmsnorm/rope/softmax kernels and no per-op fill+index-grid. | Host alloc/decode ≤ 4 MB. IDs exact. Publish \(P\). |
| 4 Batched attention | Replace the 9-head Julia loop with one batched scores GEMM `(H,1,Dh)×(H,K,Dh)` and one softmax over `(H,K)`, then one values GEMM. Shared by Lava and CUDA if the helper stays in `Inference.jl`. | Algorithmic, not a compiler. Cuts 270 softmax syncs/launches to 30. | Head-loop gone on both device backends. Microbench vs old loop. IDs exact. |
| 5 Decode kernels | Port the CUDA decode kernels that already exist: row rmsnorm, causal softmax, rope, swiglu, as KernelAbstractions kernels in `lava_ops.jl`. No handwritten SPIR-V, no coopmat. | Copy the CUDA seam's kernel set onto Lava storage. GessoLavaExt already said the seam was "not a kernel contest"; this arm is that contest, still inside the extension. | Per-op microbench vs Gesso CUDA on SmolLM2 decode shapes. Class D \(P_i\) reported. No CPU fallback. |
| 6 Parity | Arm 11 harness unchanged except landed arms. | — | Median \(P \ge 0.90\) on all three prompts. Host suite green. Packaging probe still does real Lava inference. |
| 7 Parked | Julia compiler fork, representation, sampling, half, larger models, 8k full-model, AMD, command-buffer graphs. | Opens after arm 6, or if arm 1/5 prove the limiter is pipeline compile or SPIR-V quality rather than wait/launch. | Named receipt. |

Expected shape, not a promise: arm 2 should swallow the 12× if the
wait model is right. Arms 3–5 exist so that after the waits die we
are not left with 200 tiny broadcast kernels per token, which is how
CUDA would still beat a wait-free broadcast interpreter. If arm 2
does not move tok/s, arm 1's capture was wrong and we stop and
re-measure before writing kernels.

## Correctness overlay

- Independent HF/PyTorch greedy IDs exact; logits at declared atol.
- No CPU substitution for unavailable Lava.
- Receipts still split compile, first-use, warmed decode.
- Prefill correctness stays the Regime I suite. Decode-path sync
  changes must not silently weaken prefill.
- Planted nonfinite still fails with `ERR_NUMERICAL_INSTABILITY`.
- Three inherited `@inferred` breaks on `storage::Any` stay broken
  until a later typed-storage packet. They are not this campaign.

## Arm 1 — permitted files

`docs/goals/REGIME_II.md`; `test/regime_ii_attribution.jl`;
`ext/lava_ops.jl` (counter around `_lava_sync!` and `_all_finite`
only); `docs/ARCHITECTURE.md` (pointer); new files under
`docs/archive/2026-10_regime-ii/`.

Runtime edits in arm 1 are counters. No fusion, no kernel, no
algorithm change.

## Arm 2 — permitted files

`ext/lava_ops.jl` (decode methods drop `_lava_sync!`; boundary waits
once; `_all_finite` not on decode rms/softmax); `test/test_numeric_lava.jl`
and a decode planted-NaN case; `src/Inference/session.jl` only if
logits-check must move. Prefill method syncs stay until a later arm
names them.

## Out of scope

`Julia Compiler Optimizations for Gesso/` stays research.
`src/Representation/` stays empty. Palette, Cyan, Luxel. Training.
Beating CUDA by making Lava optional.

## Related

- Regime I report: `docs/archive/2026-10_regime-i/README.md`
- Packaging: `docs/PACKAGING_REGIME_I.md`
- Speed-floor discipline: `docs/research/SPEED_FLOOR.md`
- Parity definition: `Julia Compiler Optimizations for Gesso/docs/the_ancient_texts.md` §35–37
- CUDA kernels to port in arm 5: `ext/cuda_ops.jl`
  (`_cuda_rmsnorm_row_kernel!`, `_cuda_rope_kernel!`,
  `_cuda_softmax_kernel!`, `_cuda_swiglu_kernel!`)
