# Campaign 1: matched GDN gate projection diagnostic

Opened 2026-10-16. Status: active external diagnostic; no source repair selected.
Parent: `CAMPAIGN1_LAVA_CORRECTNESS.md` and the unchanged Campaign 1 gates.

## Objective

Determine whether the first captured CPU-generic/GPU difference in the p64
first-layer `a` projection is reproduced by ordinary Float32 matrix arithmetic
on identical operands, before selecting another runtime change.

## Permitted files

This document, the campaign status document, new immutable external helpers
`gdn_gate_projection_fixture.jl`, `gdn_gate_projection_replay.jl`, and
`validate_gdn_gate_projection.py`, plus new result artifacts and resume-frontier
entries. A process-local Float64 accumulator kernel is an explicit diagnostic
arm. No package source, CPU oracle, precision policy or prior artifact edits.
Model diagnostic envelope: permit new external
`gdn_gate_projection_model.jl` and corresponding result files. Reuse the exact
replayed Float64 accumulator kernel. Register only the actual device storage
objects for each linear-attention block's `wgdn_a` and `wgdn_b`; every other
matrix call must retain the original body. In a single bounded process, require
the unmodified external Q8_0/NoContraction baseline to reproduce fresh p64
65017 first, then run two fresh-state requests with only registered gate
projections substituted. Keep the original CPU target 82202. Require exactly
two substitutions per linear-attention block per request. This is a causal
diagnostic, not a production precision decision or full model acceptance.

## Interfaces

Public CPU embedding, RMS and matrix operators; public Lava matrix operator;
existing FloatControlledKernel API. Source `ef086ef` stays frozen. Read the
GGUF locally; reconstruct the first norm input and require its exact captured
hash before replay. Consume the terminal stage observer v2 and its independently
audited controls. CPU generic prefill remains a same-output proxy, not proof of
default CPU fast-window intermediate identity.

## Invariants

Identical Float32 weights and input bits on CPU/GPU; preserve input and weights.
Reproduce both recorded CPU projections at M64, and both GPU projections before
interpreting a diagnostic arm. Preserve all negative results. A more accurate
operator is not sufficient model admission. No host fallback or precision
policy is introduced into production.

## Tests

Replay both `a` and `b` projections with M1/31/32/64/65 (last shape repeats the
first row). Record stock dispatch, repeat identity with poisoned destinations,
CPU BLAS output and Float64 mathematical reference rounded to Float32. Audit
raw hashes, extents, finite values, source identities and actual emitted shaders.
The diagnostic Float64 accumulator must match independently reconstructed
products; GPU stock must reproduce the captured M64 differences.

## Performance target

No throughput claim. Record cold run/resource limits and bounded artifact size;
selected production-source compile, allocation and operator costs remain a
separate mandatory Campaign 1 gate.

## Expected artifact

A source-bound operand fixture, device replay, independent numerical receipt,
and a demonstrated next action. Model 39/41 remains red until the original
gate and all broader stress, integration, capability and cost gates pass.

## Operator receipt

CPU fixture and GPU replay both terminated with exit 0. The reconstructed
64x2048 norm input exactly matches the original trace. Both CPU and both GPU
M64 projection outputs exactly match the original saved hashes. Independently
decoded native GGUF Q8_0 gate weights match the captured Float32 matrices.
All ten device Float64-accumulator results match independent NumPy Float64
products rounded to Float32 at 0 ULP. Stock GPU error against that reference
reaches 1.5735626220703125e-5. All six emitted shaders validate externally;
the two diagnostic modules contain Float64 arithmetic and preserve modes for
32/64-bit floats. Receipts: `gdn-gate-projection-audit-2.json` and
`gdn-gate-projection-shader-validation-1.json`.

An independent C replay using serial ascending-K Float32 `fmaf` matches every
stock GPU output bitwise across all ten cases. Its separately rounded versus
fused cancellation witness is effective. Thus the observed operator delta is
reproduced by ordinary F32 reduction order; it is not evidence of lost writes
or a compiler arithmetic defect. Receipt directory: `gdn-gate-fma-replay-1`.

What changed: external diagnostics only; package source and oracle unchanged.
Hardware: RTX 5060, Vulkan/Lava, Ornith exact local GGUF. No speed claim;
compilation is included in bounded diagnostic runs. Matrix captures are small;
GPU runs use the existing 20 GiB RAM / 4 GiB swap cap and shared GPU lock.
The three-request registered-gate model diagnostic is running; model causality,
production precision selection, cost and portability remain unproven.

The model diagnostic is now terminal, exit 1. Its stock control reproduces
65017. Both Float64-gate requests return 3476 instead of 82202, with exactly
60 selected calls per request. Independent validation checks all 60 native
GGUF registry weight hashes and exact kernel/dispatch transformations:
`gdn-gate-projection-model-validation-2.json`. Reject the gate-only arm as a
sufficient repair. All 57 pre-link model shaders validate externally. The first
audit's wrong expectation of a repeated summary flag is preserved separately.

Next diagnostic envelope: permit new external `gdn_gate_params_replay.jl`
and `validate_gdn_gate_params.py` plus results, without package changes. Replay
CPU and Lava gate-parameter methods on identical saved first-layer a/b values
and the actual A_log/dt_bias vectors. Include the stock-GPU, CPU-BLAS and
Float64-projection inputs as separately labeled controls. This isolates the
nonlinear seam from projection error. A bounded direct device `log1p` probe
may check whether the current implementation loses small nonzero arguments;
source inspection shows it expands to `log(1+x)`, including a Float64-to-F32
cast for the Float64 overload. Require actual device evidence and quantify
exposure in the real model inputs before selecting any repair. No production
math-library edit or model sufficiency claim follows from a synthetic probe.

Stateful first-block envelope: permit external `gdn_first_block_replay.jl`
and `validate_gdn_first_block.py` with bounded result artifacts. Clone the
existing generic GDN block under a new diagnostic function name; package
methods remain unchanged. Use the actual first-block weights, reconstructed
embedding input, zero convolution/recurrence state and the same Q8_0 diagnostic.
Require unmodified CPU and GPU block outputs to match original trace hashes
before interpretation. Then inject frozen CPU intermediates at projection,
convolution, gate, core and merged boundaries in separately named controls.
Record every injection and compare subsequent stages. Require the CPU merged
injection to reproduce CPU output projection, and a final unmodified GPU repeat
to reproduce all stock stage fields. Cap raw captures at 128 MiB. This isolates
remaining stateful arithmetic seams and cannot graduate the model stress gate.

The matched nonlinear replay is terminal. Three actual GGUF-derived gate cases
lose no nonzero g values, with maximum CPU/GPU g delta 1.52587890625e-5.
The direct log1p probes lose five nonzero outputs per dtype. The authoritative
`gdn-gate-params-validation-5.json` admits structural/provenance integrity and
recomputed metrics only: no numerical accuracy budget is selected from these
observations. It supersedes earlier validators that tuned an absolute bound
after observing device output. Four emitted shader modules validate externally.

The first-block replay is terminal, exit 0. Its independent auditor verifies
all eight routes, 64 stage records, 11 injections, source-clone reconstruction
and inverse, original CPU/GPU sub-output hashes, and stock-repeat bit identity.
Raw bytes total 51,740,672, below the 128 MiB cap. Successive CPU injections
leave convolution, nonlinear gates, core and then merged normalization as
separate remaining arithmetic differences. Injecting CPU core still leaves
merged maxabs 4.76837158203125e-7; injecting CPU merged makes the final output
projection bitwise CPU-exact, while recurrence state retains its own difference.
The projection-only injection leaves the first-block output hash unchanged
from stock. This bounds the significance of the first captured projection
delta; no source repair follows from operator error ordering alone.
Receipts: `gdn-first-block-replay-validation-2.json` and
`gdn-first-block-shader-validation-1.json` (14 emitted modules, two actual linked
Q8_0 modules, inverse word recovery exact).

Next prediction-level discriminator: `CAMPAIGN1_OUTPUT_HEAD_CONTRACT.md`.
Q4_K output-head execution was excluded from the Q8_0 contract arm; compare
its two activation contracts on an identical actual final hidden row before
committing to broader GDN precision changes.
