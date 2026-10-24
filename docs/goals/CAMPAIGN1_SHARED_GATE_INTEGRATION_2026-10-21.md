# Campaign 1: integrate the qualified Float32 shared gate

## Objective
Move the independently qualified exact finite Float32 shared-gate provider into an isolated maintained Gesso candidate, prove the actual public entry point, and run the unchanged original model gate.

## Permitted files
Candidate worktree `gesso-shared-gate-integer-20261021`, based on frozen ad47: `ext/lava_hybrid.jl`, new `ext/lava_float32_bits.jl`, `test/test_lava_hybrid.jl`, new `test/test_lava_shared_gate_bits.jl`, `docs/ARCHITECTURE.md`, `vendor/patches/lava/0007-scalar-integer-widths.patch`, `vendor/README.md`, and vendor/Lava regenerated only by scripts/vendor_lava.sh. New candidate-specific helpers/env/source snapshots/results under campaign scripts/results, this item, current status/rounding diagnosis/frontier. Frozen ad47, isolated Lava388, and consumed helpers/results remain unchanged.

## Interfaces
Preserve Gesso.Inference._shared_gate_add!(::LavaBackend,y,x,sd,g) and CPU oracle, device-storage checks, Float32 tensors, same return value and in-place semantics. Internal helper expands binary32 bits and rounds binary64 bits nearest-even; kernels retain F32 operation rounding with physical F64 arithmetic. No CPU tensor fallback or host copy. No dependence from core Gesso onto Lava.

## Invariants
No oracle/token/tolerance change. Explicit failures for unsupported requirements or invalid geometry/type. Carry per-kernel NoContraction and preserve32/64 policies through existing Lava contracts. No universal NaN payload or portability claim. The original model gate remains 9/13; the convolution discriminator is terminal negative, and the projection discriminator is terminal negative and the general GDN route discriminator is running. User instructed to wait for VRAM and leave Spark server on port8996 untouched.

## Tests
Prerequisites: shared-gate-rounded-4-validation-v2.json (576/576 full-provider outputs, 78,591 scalar comparisons; native-dot negative21/576), shared-gate-rounded-4-mutation-validation.json (output corruption, valid helper-op mutation, valid output-store bypass all reject). Actual maintained entry must pass an equivalent raw192-fixture corpus, per-program contracts, source/environment binding and effective mutations. Add maintained exact Float32 CPU-bit regressions covering sigmoid boundary, subnormal operands/results, cancellation/signedzero, carried/repeated calls, empty geometry and seeded widths. Compare old source on the same new regression to prove meaningful failures. Include actual test_lava_hybrid.jl and separately qualified supplier units/regressions as appropriate. Original13 model checks and complete Campaign1 stress suite remain mandatory after source changes; no proxy admission.

## Performance target
Record bounded operator baseline/candidate cost after cold compile and validated warmup, allocation and temporary storage. Cost capture v3 is independently qualified as an external operator capture, but cost admission is false because stock is not numerically admitted; no speed or throughput claim. Full inference throughput remains deferred. No hardware-specific optimizer or fastpath in this item.

## Expected artifact
Reviewable candidate diff, reproducible vendor patch and exact maintained public-entry regression, with honest resource/cost/model limitations. No push, merge or release.

## Receipt
QUALIFIED IMPLEMENTATION CHECKPOINT: the maintained public entry and its regressions qualify, and the unchanged model gate was run and independently remains 9/13. The bounded cost capture qualifies as a capture only. Full Campaign1 source-model, stress, make/test/format and cost-admission gates remain open.


## Candidate checkpoint

Candidate commit `2108748f29b44bbfcba7c992e0f326f0abe2b4ae`, branch
`codex/c1-shared-gate-integer-20261021`: 11 changed files, 916 source files.
Includes regenerated vendor patch0007 from Lava388, with all278 supplier files
exact; helper rounding/codecs, three shared-gate kernels, exact public tests,
and architecture note. Pure helper/scoped kernel/test formatting retains parsed
AST. One trailing blank-line removal is the sole source-v1→v2 delta; source-v3
binds those bytes to the candidate commit. No frozen source was modified.

The actual maintained entry independently passes576/576 outputs/78,591 scalar
comparisons with three actual shader/dataflow contracts. Three effective
maintained-result/program mutations reject. Receipts:
`shared-gate-maintained-1-validation.json` and
`shared-gate-maintained-1-mutation-validation.json`.

Actual `test/test_lava_hybrid.jl` passes1770 checks, zero failures/errors.
The same new exact regression against frozen old source produces1192pass,
466fail, zero errors (1658 total). Structured unit/log/source comparison:
`shared-gate-integration-unit-comparison-1.json`. First wrapper attempt had a
constant-binding setup failure before any tests; corrected wrapper v2 is the
1770-check run. Keep the failed zero-test receipt separately.

The original model control is terminal. Launch1 exited before model requests
because its wrapper referenced `Main.Gesso` without importing it; this retained
harness failure is not a numerical result. Corrected launch2 was independently
audited by `shared-gate-maintained-model-2-validation.json`: 9/13, with the four
normal hello rows producing 11,271 rather than CPU 11,353. Integrity and source
errors are empty. The mutation receipt
`shared-gate-maintained-model-2-id-mutation-validation.json` rejects a coherent
output-ID mutation while retaining producer green flags.

The convolution diagnostic then ran the same original13 profile with the one
process-local launch override. `gdn-conv-model-2-validation.json` is terminal
9/13 with the same four hello two-token mismatches, 300 convolution-provider
calls, zero integrity/source errors, and all three actual convolution programs
exact after compact-ID-only comparison with their qualified fixture programs.
The repaired-ID and valid `OpFDiv`→`OpFMul` mutations both reject in
`gdn-conv-model-2-mutation-validation-v1.json`; the mutated shader passes
`spirv-val` before the auditor rejects its changed semantics. This discriminator
does not repair the model failures and does not qualify the override generally.

Cost capture v3 independently validates 400 measured samples, 8 cold samples,
8 warmup groups/47,364 warmup calls, exact finite candidate outputs and all three
actual shaders after compact-ID-only normalization. Its stock arm differs from
actual Base CPU on fixtures 1, 4, and 188, so baseline correctness and cost
admission remain false. Effective candidate/model throughput and production
admission are not established. The effective output and valid shader mutations
reject; injected measured compile time is classified as timing contamination and
keeps speed claims disabled. See the rounding item for observed operator
statistics and exact limitations.

The candidate remains a local tested branch, without general nonfinite or
multidevice portability admission. The projection diagnostic is terminal at9/13
with600model calls and a raw-exact qualified projection shader; effective ID and
valid shader mutations reject. See its separate work item for receipts.
Current model control is the general-versus-fused GDN route discriminator,
`gdn-route-model-1`, session70073/Julia2372129, with6927MiBfree at guarded
admission. No route-control outcome is claimed. Full Campaign1 stress,
make/test/format, source and cost admission remain open.

Before final source admission, shared-gate scratch allocation must use the input
backend dispatch queue rather than the default context. This remains an explicit
multidevice followup; current finite single-device evidence does not prove it.
