# Campaign 1: identical-input Ornith output-head diagnostic

Opened 2026-10-16. Status: external diagnostic; no runtime repair selected.
Parent: `CAMPAIGN1_LAVA_CORRECTNESS.md`; original acceptance gates unchanged.

## Objective

Determine whether the untouched Q4_K output-head activation contract changes
the failing fresh-p64 decision when CPU and Lava receive the identical final
normalized hidden row.

## Permitted files

This document, campaign status, new external `ornith_output_head_replay.jl`
and `validate_ornith_output_head.py`, and new raw results/resume-frontier entries.
No package source, CPU oracle, original artifact or production fallback change.

Native-oracle envelope: additionally permit external
`replay_ornith_native_head.c` and `validate_ornith_native_head.py` plus new
receipts. Use pinned local GGML base/CPU libraries, independently parsed
GGUF output.weight bytes and the captured normalized row. Quantize that row
with the exported native Q8_K reference quantizer, then replay each Q4_K row
through the selected native CPU dot. Record ABI/layout checks, library/compiler
identities and CPU features. Run bounded synthetic zero/constant/cancellation
witnesses before consuming a full model capture. Preserve compilation and
native arithmetic as a named CPU reference, not platform-independent exact-real
arithmetic. No accuracy budget is chosen after observing output differences.

## Interfaces

Frozen Gesso `ef086ef`, external Q8_0 v3 and NoContraction diagnostic unchanged.
The existing `_last_logits` computes the final normalized row and ordinary
Lava head output. Add one process-local callback after its head matmul, retaining
every original statement. Bind the callback to the actual device lm_head storage
identity. Copy the same row into public CPU `matmul!` with actual host lm_head.
Record actual head storage type, Q4_K format, shape and host/device raw-byte
hash equality. CPU uses Q8_K activations and integer Q4_K dots; Lava's M1
Q4_K GEMV decodes weight elements to Float32 and uses Float32 activations.
The earlier Q8_0 proof explicitly excludes this path.

## Invariants

Three fresh explicit-state p64 requests in one bounded process. First leaves
Lava output unchanged and must reproduce 65017. The next two explicitly copy
the same-input CPU head logits into the diagnostic output; retain CPU target
82202. All three normalized head inputs must match bitwise, with one callback
per request and no upstream substitution. Save complete input/logits bytes,
full token IDs, processed cache IDs, top-k and decision margins. No admitted
numerical error budget and no production CPU fallback follow from this arm.

## Tests

Verify exact original-body inverse, source/helper/fixture/oracle identity,
head raw bytes, complete raw-array hashes/extents/finite values and input
preservation. Independently recompute top-k/argmax/margins and raw-ID outcomes.
Keep structural validity separate from model sufficiency. Capture model shader
validation separately. Any proposed GPU implementation of a selected contract
requires its own operator/native-oracle, portability, cost and full stress gates.

## Performance target

No speed claim. GPU work uses the shared lock, 4800 MiB declared device budget
and existing 20 GiB RAM / 4 GiB swap cap. Bound saved input/logit arrays to 16 MiB;
report the host-copy diagnostic explicitly.

## Expected artifact

A source-bound identical-input head comparison and a causal prediction result.
Campaign 1 remains open until the original full completion contract is proved.

## Terminal receipt

The three-request identical-input head diagnostic is terminal, exit 1.
Stock, CPU-head selection and its repeat all return 65017. Every captured
normalized row, GPU logit array and CPU logit array repeats bitwise. Host/device
Q4_K payload hashes match the independently parsed 286,064,640-byte GGUF head.
CPU/GPU logits differ by maxabs 0.13959717750549316 and relative L2
0.00723763908680243, but both choose 65017 over 82202. Their top-one margins
are 0.19462966918945312 and 0.23735523223876953 respectively. Reject head-only
selection as a sufficient repair; no production head policy changes.

The independent validator v2 derives process exit from raw-ID failure counts;
the original validator's early exit-zero assumption was wrong for a complete
negative diagnostic. Terminal status is retained, never replaced by a mock
zero. Receipt: `ornith-output-head-validation-2.json`. All 51 emitted shader
modules and two actual linked Q8_0 shaders validate externally, with exact word
recovery on decoration removal: `ornith-output-head-shader-validation-1.json`.

Native GGML replay quantizes the same input into eight Q8_K blocks, bitwise
matching an independent Python port of the frozen Julia quantizer. It replays
all 248,320 Q4_K rows through the pinned CPU trait and also chooses 65017,
with margin 0.19463062286376953. Native/Gesso CPU logit maxabs is
2.86102294921875e-6; this is a measured comparison, not a selected error budget.
Six bounded ABI/quantizer/dot witnesses passed before the full replay. Receipts:
`ornith-native-head-selftest-2.json` and
`ornith-native-head-replay-validation-2.json`. Native SIMD/FMA and current-host
limits are recorded. The first native helper was changed after its self-test;
that lineage remains retained, while immutable v2 was separately self-tested.

Next causal control: `CAMPAIGN1_GDN_REFERENCE_CONTROL.md`. The head comparison
constrains the diagnosis to upstream computation for this captured row; it does
not establish which GDN or attention precision seam needs a production repair.

## GPU contract operator envelope

The complete CPU-GDN plus CPU-head control now repairs the bounded fresh-p64
decision twice. Permit external-only Q4_K x Q8_K operator prototypes and
conformance fixtures before proposing source dispatch: `c1_q4k_cpu_fixtures.jl`,
`c1_cpu_q4k_install.jl`, `c1_q4k_no_contract_link.jl`,
`c1_q4k_gpu_conformance.jl`, and versioned independent validators/receipts.
No package or existing helper edits enter this envelope.

Read actual GGUF output.weight rows; preserve byte layout and inputs. Cover
M=1/31/32/64/65 with a shared first row, 65 first/interior/last vocabulary rows,
zero/constant/cancellation, signed-maximum ties and quantizer rounding witnesses.
Compare GPU activation codes, Float32 scales, block sums and complete outputs
bitwise with the frozen executed Gesso CPU contract. Independently replay native
GGML quantization/dots and retain any native SIMD association differences as
measurements. Include the full 248,320-row identical-input head when the sampled
operator passes. Repeat poisoned outputs, GC/shape changes, input preservation,
actual dispatch and emitted/linked SPIR-V validation. Effective numeric mutations
must be rejected even when their metadata and producer flags are updated.

Keep integer accumulation and eight-lane CPU float reduction order explicit.
Source Float64 casts alone do not prove device rounding: inspect actual shaders
and use physical intermediate storage if required. Any NoContraction link
override is diagnostic and recorded; a production implementation requires a
separate reviewed compiler/runtime envelope. Keep finite-input, portability and
resource limits explicit. These operator controls cannot replace the unchanged
model-history, full stress, integration and cost gates.

## GPU contract operator receipt

The sampled CPU fixture completed in `q4k-cpu-fixtures-2`, exit 0: 13 cases,
including the five token counts and eight quantizer witnesses. All public CPU
outputs match the executed scalar Q4_K path bitwise. The independent native
validator parses the GGUF separately, verifies all 65 selected payload rows,
and replays 201 input rows. Every Q8_K code/scale/block-sum structure matches
native packing exactly (469,536 packed bytes). Native SIMD/FMA output deltas
are recorded without a tolerance or pass budget. Receipt:
`q4k-cpu-fixtures-2.native-validation-2.json`. The first fixture attempt failed
because it assumed a source Manifest; v2 records the actual run environment.

GPU prototype v1 failed compilation before numerical output because a
self-referential tuple-update closure introduced dynamic dispatch. Immutable
`c1_cpu_q4k_install_v2.jl` moves the identical eight-lane float update into a
typed inline helper. Its `LavaBlockQuant` specialization preserves Q8_0 generic
dispatch, and it changes only Q4_K in that process.

V2 completed with 42/42 exact repetitions: 13 sampled cases three times each,
then the full 248,320-row head three times. Independent raw-byte validation
confirms Y, Q8_K codes/scales/block sums, physical Float64/F32 division stages,
shape/repeat identity and source/environment/payload bindings. Capture totals
8,527,596 bytes, below 32 MiB. A copied output changed by finite +0.5, with
updated SHA and green producer fields, is rejected against the actual CPU
reference. Receipts: `q4k-gpu-conformance-validation-1.json` and
`q4k-gpu-conformance-validation-1-mutation.json`; authoritative validator
`validate_c1_q4k_gpu_v3.py` preserves the two earlier schema failures separately.

All ten emitted modules and seven actual linked records (four arithmetic
kernel identities, with repeated variants) validate externally. Every added
NoContraction patch has an exact word inverse. Receipt:
`q4k-gpu-shader-validation-3.json`. The earlier auditor incorrectly disallowed
repeated variants; its failure remains preserved, with a new immutable v3.

This proves a bounded external GPU head-contract implementation on the current
device, not an integrated source repair. The Q4_K/Q8_0/GDN link helpers each
replace the same linker method and cannot be stacked as independent includes.
Production work needs an explicit per-kernel compiler contract, correct cache
identity and actual final-shader validation. Existing SignedZeroInfNanPreserve
requirements alone do not prohibit contraction. No runtime source, CPU oracle,
full-model admission, cross-device claim or throughput claim changed here.
