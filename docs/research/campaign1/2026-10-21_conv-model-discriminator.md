# Campaign 1: test the isolated decode convolution correction in Ornith

## Objective
Determine whether the already isolated block22 convolution arithmetic difference contributes to the four remaining hello two-token model failures, on the qualified maintained shared-gate candidate.

## Permitted files
New `gdn_conv_model_*` helpers/providers/auditors and bounded raw results under the campaign scripts/results roots; this work item, current status/diagnosis/frontier. No maintained source, frozen source, isolated supplier, old helpers, oracle, or model edits. The sole process-local override is the convolution launch inside GessoLavaExt's `_gdn_decode_core!` method, installed through one exact source replacement with an exact inverse.

## Interfaces
Same conv_out/state/mixed/weight/k/channels arguments as `_lava_gdn_conv_decode_kernel!`; same state mutation. Use the three already qualified preparation/division/cast kernels from block22_conv_arithmetic_v1.jl, preserving source expressions, WG64, NoContraction and preserve32/64 contracts. Keep the maintained shared gate and every other model computation unchanged.

## Invariants
Frozen original13 profile and CPU oracle; no token/tolerance/quantization changes, no CPU tensor fallback, no broad nonfinite or portability claim. The convolution provider is qualified for the captured real block22 fixture only; a positive model result does not admit it generally. Default20GiB/4GiBswap model cap,1800sec, under-lock5500MiB GPU admission. Respect foreign GPU processes and pause marker; user explicitly requested waiting for VRAM.

## Tests
Prerequisites: qualified matched block22 replay and `block22-conv-arithmetic-1-validation.json`: native3889 and Base-exp/native-div2545 differing elements versus zero for the separated F64 division path,8192outputs,three repeats,actual shaders and effective mutations. Baseline candidate2108748 original13 is independently9/13 with four hello[11,271] vs CPU[11,353], zero integrity/source errors, all actual shared-gate shaders exact after compact-ID-only normalization. The new model arm must run that same13 profile, bind the one exact source inverse and actual override method, count provider calls, archive three actual convolution programs, retain unchanged shared-gate proof, and independently compare actual programs to the qualified fixture through compact-ID-only equality. Effective model-ID and valid shader mutations must reject. Any source/harness failure is separate from numerical model results.

## Performance target
N/A for this discriminator: numerical causality only. Record compile/allocation/resource observations, no throughput claim. The separate shared-gate operator cost capture has completed, but it is not admitted as cost or throughput evidence.

## Result

`gdn-conv-model-2` is terminal exit1. Independent audit
`gdn-conv-model-2-validation.json` qualifies the original13 profile at 9/13:
the same four normal hello requests emit `[11, 271]` instead of CPU `[11, 353]`.
There are zero integrity/source errors. The exact one-launch override is bound,
300 convolution-provider calls are recorded, and all three actual prepare/div/cast
programs validate and match their qualified fixture programs after compact-ID-only
normalization. The override does not repair the four remaining failures.

Both effective mutations pass in
`gdn-conv-model-2-mutation-validation-v1.json`. The model-ID mutation retains
producer pass flags and repairs generated/full IDs coherently; the independent
auditor rejects it against the CPU oracle. A structurally valid div shader
mutation (`OpFDiv`→`OpFMul`) passes `spirv-val`, has repaired receipt descriptors,
and is rejected as `conv shader differs beyond ID renaming`. The mutation run
preserved the original evidence hashes. These results qualify this bounded
diagnostic, not general convolution support or source admission.

## Expected artifact
A qualified model positive or negative deciding whether to generalize this convolution correction. Original full Campaign1 stress and release gates stay open either way.

## Receipt
TERMINAL NEGATIVE FOR THIS DIAGNOSTIC. The convolution correction retains 9/13
with the four hello two-token mismatches. The subsequent projection diagnostic also remains9/13, with zero integrity/source
errors and600model calls. Current control is
`2026-10-21_gdn-route-model-discriminator.md`, running as
`gdn-route-model-1` session70073/Julia2372129 after guarded admission.
Maintained source admission and all original full-stress/release gates remain
open; no inference-throughput claim is made.
