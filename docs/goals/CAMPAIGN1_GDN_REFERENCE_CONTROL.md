# Campaign 1: complete GDN CPU-reference control

Opened 2026-10-16. Status: bounded external causal diagnostic.
Parent: `CAMPAIGN1_LAVA_CORRECTNESS.md`; no acceptance gate changes.

## Objective

Determine whether complete CPU-reference GDN mixers restore the fresh-p64
decision while the surrounding model continues through the existing Lava
attention, host-expert and output-head paths.

## Evidence and permitted files

Gate-only Float64 arithmetic is insufficient (3476 twice); CPU-head-only
selection is also insufficient (65017 twice on identical GPU-normalized input).
The first-block replay demonstrates separate convolution, gate, recurrence and
merged-normalization precision seams, with exact CPU output projection after
CPU merged injection. This motivates a whole-mixer control before proposing a
device arithmetic repair.

Permit this document, campaign status, external `ornith_cpu_gdn_replay.jl`
and `validate_ornith_cpu_gdn.py`, and new diagnostic result/frontier artifacts.
No package, original helper, CPU oracle or production precision/fallback edits.

## Interfaces

Frozen `ef086ef` with unchanged external Q8_0 v3/NoContraction baseline. Add a
process-local LavaBackend specialization of `_hybrid_linear_block!`. Preserve
its full original signature and bind each device GDN block to the actual host
block by `wgdn_a.storage` identity. Copy input, convolution and recurrent state
to host; call the existing generic CPU GDN body with CPU scratch/fused paths
disabled; copy normed/output/state back. All non-GDN operators stay unchanged.
Record the host-copy control explicitly; it is not a production fallback.

## Invariants and tests

Five fresh p64 states: stock control must reproduce 65017; two complete-CPU-GDN
requests target 82202; two complete-CPU-GDN plus CPU-head requests also target
82202. Keep all failures. Require exactly 30 CPU GDN calls per selected request,
each actual linear-attention layer visited once, and one observed head call.
The normalized final hidden row must repeat bitwise across all four GDN-selected
requests; head selection must leave upstream state unchanged. Save source/helper
identities, registry/layer records, exact head inputs/logits, raw IDs and cache
contracts. Separate structural integrity, causal sufficiency and source/model
admission. No accuracy thresholds are selected from observed errors.

## Performance target and expected artifact

No speed claim; CPU-copy control measures causal behavior only. Use the shared
GPU lock, 4800 MiB device budget, 20 GiB RAM / 4 GiB swap cap, at most 16 threads
and a bounded process timeout. Cap raw head arrays at 16 MiB. Produce a reviewable
negative or positive control and choose the next operator work from its result.
Full final-source stress, cost, capability and integration gates remain open.

## Terminal complete-GDN receipt

The five-request control completed, exit 1. Stock returns 65017. Both
CPU-GDN-only requests return GPU-head 3476, while the CPU head on their same
normalized rows chooses 82202. Both combined CPU-GDN/CPU-head requests return
82202. Thus the combined control repairs this bounded prediction, while GDN
alone remains insufficient. This does not admit a host-copy production path.

Independent audit verifies the exact source-derived specialization and head
callback, 30 native gate-alpha registry weights, 120 GDN calls, repeat identity
of seven state/output hashes at every selected layer, all raw head arrays and
request/cache contracts. The 9,973,760 saved bytes remain within the 16 MiB cap.
All 55 emitted shaders and four linked records (two unique Q8_0 modules)
validate, including exact inverse recovery after removing NoContraction.
Receipts: `ornith-cpu-gdn-validation-3.json` and
`ornith-cpu-gdn-shader-validation-1.json`. Structural validity and combined-arm
success are separate from overall sufficiency and production admission.

The first audit retained a source-slice ambiguity failure. The corrected parser
selects the exact slice using the producer's pinned original-body hash; no
numeric check or token expectation changed. The initially consumed Python file
was overwritten during that correction. Its old hash and failed receipt remain
recorded, and the corrected code is now separately frozen as
`validate_ornith_cpu_gdn_v2.py` (SHA-256
`10886fabfc5c10fe14ca94ce496f98920981f22b78f193e9d5cf8724458f67b9`).
Receipt 3 executes that named immutable version.

## Coupled gate/head discriminator envelope

Permit additional external `ornith_f64_gate_head_replay.jl` and
`validate_ornith_f64_gate_head.py` plus new results. The complete CPU-GDN control
is terminal: GDN-only chooses GPU-head 3476 while its same-input CPU head chooses
82202; combined CPU-GDN/CPU-head requests return 82202 twice. The earlier
verified GPU Float64-gate-only arm also chose 3476. Before rewriting all GDN
arithmetic, test whether that narrower gate arithmetic plus CPU head suffices.
Reusing a smaller intervention is an experiment, not a reduced completion gate.

Keep the exact previously replayed Float64 dot kernel and register only actual
`wgdn_a/wgdn_b` device storage identities (60 weights). Preserve every other
matrix call byte-for-byte. Five fresh states: stock control 65017, two gate-only
controls 3476, then two gate-plus-CPU-head requests targeting unchanged CPU
82202. The 3476 controls retain their CPU-ID mismatch explicitly; they never
become model acceptance targets. Require 60 gate calls per selected request,
one head callback, identical final normalized input across all four selected
requests and exact raw-logit recomputation. No complete GDN CPU copies occur in
this arm. Preserve negative results; no production precision policy or model
admission follows from these five diagnostics.

The narrower discriminator is terminal, exit 1. Stock reproduces 65017.
Gate-only and gate-plus-CPU-head both return 3476 twice. The same-input CPU head
also chooses 3476 on all four gate-selected normalized rows. Thus the identical
GPU ID shared with complete CPU-GDN did not imply an identical hidden row:
complete CPU-GDN/CPU-head succeeds, gate-only/CPU-head does not. Reject the
narrower coupled intervention as sufficient.

Receipt `ornith-f64-gate-head-validation-1.json` independently verifies all 60
native gate weights, 240 selected calls, repeated normalized rows/logits,
reversible kernel/dispatch/head source and the two raw-ID failures. All 54
emitted shaders and four linked records (two unique modules) validate in
`ornith-f64-gate-head-shader-validation-1.json`. Numerical/operator conformance
of the gate kernel remains intact; it is not the complete model repair.

## Matched recurrence arithmetic envelope

Permit external `c1_gdn_reference_recur.jl`,
`c1_gdn_reference_no_contract_link.jl`, `c1_gdn_recur_conformance.jl` and new
versioned auditors/receipts. Replay the independently audited first-block CPU
q/k/v/g/beta inputs and zero initial state, comparing complete core and final
recurrent state. Retain the ordinary device implementation as a measured control.

Use physical Float64 q/k normalization arrays and CPU reduction/state-update
order, preserving Float32 recurrent-state writes. Test both the frozen CPU decay
fixture and ordinary device Float32 exp(g) separately. The former isolates
algebra and rounding only; it is explicitly not a production CPU offload or
complete device repair. Capture all arrays, actual dispatch, float widths and
NoContraction shader inverse. Require exact CPU output/state for the arithmetic
arm; retain any device-transcendental difference without choosing a tolerance
from its observed magnitude. This operator replay does not replace the model
control or full Campaign 1 gates. No broader software math rewrite is admitted.

Additionally permit external `c1_gdn_gnorm_conformance.jl` to replay the same
audited real CPU core/z inputs and native first-block norm weights. Compare the
existing fused GDN normalization with public CPU-matched 3D RMS plus explicit
Float32 gate/merge. Separate frozen CPU SiLU-gate fixtures from ordinary device
SiLU, retaining every mismatch. First reproduce the original CPU merged capture
bitwise. This reuses the current RMS operator without changing package source,
and does not admit a production CPU-gate copy or disable fused decode globally.

## Terminal recurrence and normalization receipts

Recurrence v2 is terminal, exit 1. All eight Float64/CPU-decay repetitions
match complete CPU output and recurrent state exactly; six device-exp
repetitions fail. Independent scalar-order replay, physical Float64 q/k buffers,
repeat identity and shader inverses pass in
`gdn-reference-recur-validation-4.json` (`validate_c1_gdn_recur_v5.py`).
Earlier immutable auditors retain schema/bookkeeping failures. Producer v1
failed inserting a 2D decay descriptor into a dictionary inferred from 3D
outputs; v2 only broadens that metadata dictionary. Raw capture is 43,485,800
bytes. Saved CPU decay remains a fixture, not an independently reimplemented
transcendental in this audit.

For real p64, 985/2048 GPU decay values differ from saved CPU bits, with maxabs
5.960464477539063e-8. All 64 nonzero CPU subnormal decays become GPU zero in
this capture. This is a measured fixture result, not a universal device policy.
The actual driver reports DenormPreserveFloat32=false and
DenormPreserveFloat64=false; do not request unsupported execution modes.
Physical-property receipt: `float-properties-vulkaninfo-1.json` and its raw log.

Gated normalization is terminal, exit 1. Independent replay verifies original
CPU merged output and native norm weights/epsilon. Public CPU-matched 3D RMS
normalization is 0 ULP; both CPU-gate merged repetitions are bit-exact. Both
device-SiLU merged repetitions differ, despite exact normalized inputs. The
stock fused path has 172,434/262,144 differing bits; public RMS plus device gate
has 104,481, with maxabs 4.76837158203125e-7 in both comparisons. All six shaders
validate. Receipt: `gdn-gnorm-validation-1.json`. Ancillary mapreduce/fill dumps
have no preservation modes; keep that shader-scope limitation explicit.

The static review initially confused the CPU seq=1 scratch path with this
seq=64 generic fixture. Review was corrected against the actual call path:
the generic CPU fallback uses public RMSNorm. No reference was changed.

## Bounded Float32 exponential discriminator

Permit external `c1_julia_exp32.jl`, `c1_julia_exp32_replay.jl`,
`c1_julia_exp32_no_contract_link.jl` and versioned validators/receipts. Test only
the Float32 exponential primitive now implicated by the matched recurrence
and gate controls. This does not authorize blanket Base math overrides or a
complete software transcendental library.

Lift the pinned Julia Base Float32 range-reduction and minimax polynomial,
retaining explicit fused operations. Use explicit ties-to-even reduction and
integer IEEE power-of-two scaling if needed to preserve subnormal output bits
without relying on an unsupported DenormPreserve32 mode. Preserve license and
exact Julia source/helper identities. First compare the helper on CPU against
executed Base.exp over all actual g and negative-z inputs, dense finite ranges,
range-reduction ties, overflow/underflow boundaries, signed zeros and random
Float32 bit patterns. Treat NaN handling as classification unless payload
identity is separately proved.

Then compare stock and controlled GPU outputs on those immutable inputs,
checking complete raw bytes, repeated poisoned destinations, input identity,
actual source/shader widths, NoContraction inverse and effective numeric
mutation. Keep strict finite-result bit comparisons; do not tune an error budget
after observing errors. No model/source admission follows from the primitive.

## Exponential CPU proof and first GPU compiler failure

CPU-only receipt `exp32-cpu-mpfr-reference-1/receipt.json` compares all 464,446
inputs used by the GPU replay. The helper matches executed Julia Base exactly
for every non-NaN result, with NaN classification also matching. A separate
header-verified MPFR 4.2.2, 512-bit reference measures a maximum one-ULP Base
error on 440,208 finite pairs (20,221 differ). This is a measurement, not a
new approximation budget or replacement oracle. Raw capture is 7,431,136 bytes.

GPU replay 1 (session 19268) is terminal, exit 1, before numeric GPU output.
`spirv-val` rejects the constant-tuple `evalpoly` call: a Private pointer is
passed to a PhysicalStorageBuffer parameter. The original helper, input/CPU
arrays, fatal event and compiler debug artifacts remain preserved. This is
not numerical disproof or a passing GPU primitive.

Permit new `c1_julia_exp32_v2.jl` and `c1_julia_exp32_replay_v2.jl`, retaining
the original files. Version 2 writes the same seven Horner fused operations
explicitly instead of passing a constant tuple. It changes no coefficient,
round point, input or reference. CPU replay of all 464,446 inputs remains exact.
The existing per-kernel NoContraction hook still preserves ordinary arithmetic
while leaving intentional Fma instructions intact. New versioned validation
must retain the failed run and bind the new helper hashes.

## GPU exponential recurrence and gate discriminator envelope

If the standalone exponential proves exact, permit new external recurrence
and gated-normalization helpers and versioned auditors. Replay the same real
p64 and synthetic grouped/tiled recurrence cases, with at least three poisoned
repetitions of a new GPU-only exponential route. Keep CPU-decay and ordinary
GPU-exp controls as separate recorded arms. Keep the physical Float64 q/k
normalization and recurrence arithmetic and Float32 state stores unchanged.
Require complete CPU core/state/decay bit identity and original CPU capture
reproduction. A new helper may duplicate the frozen kernel definitions with
only the decay primitive replaced; no consumed helper may be edited.

For gated normalization, retain the native weights/epsilon and original CPU
core/z/merged fixtures, public CPU-matched 3D RMS and CPU-gate/ordinary-device
controls. Compare a staged GPU exp32, Float32 denominator and Float32 division
with an additional physical Float64 division followed by Float32 store as a
bounded discriminator. Capture exp/denominator/gate/merged arrays to expose
rounding differences. Require exact CPU results, unchanged inputs and three
poisoned repetitions; do not choose an observed tolerance. Cap each new raw
capture at 64 MiB. A single scoped NoContraction hook may cover the new
experiment's arithmetic kernels, including exp32, with exact shader inverse
and actual linked artifacts. Explicit fused operations remain intentional.

These operator results select subsequent GPU model experiments. They do not
admit CPU fixture copies, replace the full model gate or authorize blanket
Base overrides, broad software math libraries or production source changes.

## Terminal standalone exponential checkpoint

GPU v2/session46577 is terminal, exit 0. All 464,446 inputs match executed
Julia Base in three repetitions (finite bits exact; NaNs classification).
Independent audit `julia-exp32-replay-validation-2.json` reconstructs inputs,
executes the pinned CPU reference and checks all saved arrays and identities.
`julia-exp32-numeric-mutation-validation-2.json` rejects a copied corruption
with updated hashes and green flags. All three emitted SPIR-V modules and one
actual linked record validate in `julia-exp32-shader-validation-2.json`, with
exact inverse recovery after removing NoContraction. No model/source admission.

The user reserves GPU for Qwen retraining and then Claude. Operator replays
using the primitive are prepared, unrun and not queued. The recurrence auditor
is incomplete; the gated-norm auditor is unexecuted. See the complete bounded
resume frontier in `CAMPAIGN1_CHECKPOINT_2026-10-16_EXP32.md`. Final Campaign 1
acceptance is unchanged and remains incomplete.

## Physical division discriminator — 2026-10-18

The user has released the GPU reservation; live checks found no competing
compute job or held shared lock. The GPU-only exp32 recurrence is terminal
and independently exact in all twelve repetitions across four geometries;
its ordinary-exp controls retain six failed repetitions. Raw/source/CPU and
all sixteen emitted/nine linked shaders pass; a copied mutation of all three
new-route outputs with repaired hashes and retained green flags is rejected.

The gated-normalization replay is terminal, exit 1. The custom exponential
and Float32 denominator are CPU-exact in all repetitions, but the gate has
71,856/262,144 differing elements (maximum absolute delta 4.76837158203125e-7).
Both the F32 and intended F64-expression arms produce the same wrong values.
Actual linked SPIR-V contains only F32 division; the independent v3 auditor
correctly rejects the claimed F64 discriminator. Preserve this negative run
and its audit; do not interpret it as a physical F64 division result.

Permit new external `c1_gdn_physical_div_conformance.jl`, a single scoped
`c1_gdn_physical_div_no_contract_link.jl` and versioned independent auditors.
Use separate promotion, divide and rounding kernels: device F32 z/denominator
become physical F64 buffers; divide loads F64 buffers and stores F64 quotient;
a separate kernel rounds to F32 gate. Require actual shader result types and
captures of numerator64/denominator64/quotient64, not source-level casts alone.

Keep the original native epsilon, norm weights, core/z/merged targets, GPU
exp32/denominator stages and public RMS path. Three poisoned repetitions each
of CPU-gate control, ordinary F32 division control and new physical division
route. Record the F32 misses; require CPU-gate and physical-route exact CPU
gate/merged output, normalization and unchanged inputs. Independent CPU Base
must reconstruct exp/gate references, with scalar RMS/merge, physical F64
division replay, exact promotion/rounding and all actual shader/terminal checks.
The unchanged 64 MiB capture cap includes every saved array; project the total
before running. Mutate copied GPU outputs with repaired hashes/green flags.

This is a bounded device-arithmetic discriminator. It does not declare F64
division followed by F32 rounding universally identical to F32 division, choose
a new tolerance, replace model acceptance or admit a production precision
policy. Keep the NoContraction source draft isolated and the complete final
Campaign 1 stress/integration/cost/portability gates unchanged.
