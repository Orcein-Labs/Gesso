# Regime II — Lava throughput to CUDA parity

Owner: Grok. Layer: Gesso mechanism. Palette and Cyan stay frozen
infrastructure. Regime I is closed (`docs/archive/2026-10_regime-i/`).
This file is the execution roadmap. It is not a speed claim.

## Target

Same box, same SmolLM2-135M checkpoint, same three prompts, same
host-visible greedy workload as Regime I arm 11:

\[
P = \frac{\text{tok/s}_{\text{Lava}}}{\text{tok/s}_{\text{Gesso CUDA}}}
\]

**Parity gate:** median \(P \ge 0.90\) on all three prompts, compile
outside the timed region, greedy IDs exact against the independent
reference. Report the three prompt rows. Do not average a dead kernel
into a headline.

Comparator is Gesso CUDA on this machine, not vLLM and not a vendor
peak. PyTorch eager stays a published third column.

## Frozen baseline (Regime I arm 11, RTX 5060, Julia 1.12.6)

| Prompt | Lava tok/s | Gesso CUDA | PyTorch eager | Lava host alloc / decode |
| --- | ---: | ---: | ---: | ---: |
| Hello | 2.40 | 29.93 | 21.58 | 36.4 MB |
| The quick brown fox | 2.49 | 28.81 | 21.90 | 37.2 MB |
| Julia is a programming language. | 2.49 | 28.80 | 22.26 | 38.0 MB |

Receipts: `docs/archive/2026-10_regime-i/receipts/arm11-lava-after.json`,
`arm11-cuda-after.json`. Lava device snapshot at that run: 855 MB live,
87 pipelines cached, 90 kernels cached. Native Float32 Lava
normalization did not move tok/s. Native CUDA normalization was
rejected. Three inherited `@inferred` breaks on `storage::Any` remain.

Current \(P \approx 0.083\). Closing the gate is an ~11× Lava decode
gain on this workload, with correctness held.

## Sequencing

Run the arms in order. A skip is not a completion gate. Failure
preempts the active arm. No later arm graduates before all preceding
arms pass. No Julia compiler fork, representation planner, speculative
decoding, GGUF, native FP16/BF16, or AMD work until the profile in
arm 1 names it as the remaining limiter.

Every speed change keeps Regime I greedy IDs exact and the declared
logit tolerances. Lava/Vulkan stays primary. CUDA stays the
throughput comparator.

| Arm | Work | Completion gate |
| --- | --- | --- |
| 1 Attribution | Split a warmed Lava decode into host launch, CPU↔GPU sync, host alloc, H2D/D2H, GPU kernel time, and unattributed remainder. Nsight Systems / `NVTX` or equivalent plus Julia alloc profile. | Written receipt: percentage table for one Hello decode, three repeats, compile excluded. Hypothesis list ranked by measured share. No code speed claim. |
| 2 Host tax | Cut the 36–38 MB/decode host allocation and any per-token Julia dispatch that arm 1 shows in the top two buckets. | Same harness: host alloc/decode ≤ 4 MB, IDs exact, \(P\) published. If tok/s does not move, that is a finding and we still close the alloc gate. |
| 3 Launch and sync | Collapse per-op submit/wait. Graph capture, fused decode step, or persistent command buffer — whichever arm 1 says owns the time. | Kernel/pipeline submit count per new token documented before/after. Sync count per token documented. IDs exact. |
| 4 Hot kernels | GEMM, attention, RMSNorm, RoPE, SwiGLU on the Lava path. Match Gesso CUDA algorithms where they already beat us; keep Lava as the device. | Per-op microbench vs Gesso CUDA on the same shapes as SmolLM2 decode. Class D (ML) \(P_i\) reported. No silent CPU fallback. |
| 5 Parity | Re-run the arm 11 harness unchanged except for the landed arms. | Median \(P \ge 0.90\) on all three prompts. Host suite still green. Packaging probe still does real Lava inference. |
| 6 Exotic (parked) | Compiler fork, representation, sampling, half, larger models, 8k full-model. | Opens only after arm 5, or after arm 1 proves the limiter lives there. |

## Correctness overlay

Copied from Regime I and still binding:

- Independent HF/PyTorch greedy IDs exact; logits at declared atol.
- No CPU substitution for an unavailable Lava device.
- Receipts distinguish compile, first-use, and warmed decode.
- `libs/` is not the package. Do not silently vendor a Lava fork
  into the published tree.

## First Grok pass (arm 1 boilerplate)

Concrete start, no kernel heroics:

1. Re-run `test/regime_i_performance.jl` lava/cuda on this box and
   store new JSON next to the archive (do not overwrite arm 11).
2. Add a decode-step attribution probe: count Lava submits, waits,
   and bytes allocated around `_session_greedy_id!` / decode loop
   in `src/Inference/session.jl` and `ext/lava_ops.jl`.
3. One Nsight Systems capture of a warmed 8-token Hello generate.
4. Write `docs/archive/` receipt `regime-ii-arm1-attribution.json`
   with the percentage table and the ranked hypothesis list.

Permitted files for arm 1: this document; a new
`test/regime_ii_attribution.jl`; `docs/ARCHITECTURE.md` (pointer
only); receipts under `docs/archive/`. Runtime edits in arm 1 are
instrumentation only.

## Out of scope until named by a receipt

Julia compiler fork (`Julia Compiler Optimizations for Gesso/`
stays research). Magenta / KV representation. Palette/Cyan. Training.
Matching CUDA by making Lava optional.

## Related

- Regime I report: `docs/archive/2026-10_regime-i/README.md`
- Packaging: `docs/PACKAGING_REGIME_I.md`
- Speed-floor discipline: `docs/research/SPEED_FLOOR.md`
- Parity definition: `Julia Compiler Optimizations for Gesso/docs/the_ancient_texts.md` §35–37
