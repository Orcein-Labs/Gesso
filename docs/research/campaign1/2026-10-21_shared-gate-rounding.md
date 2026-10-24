# Campaign 1: preserve shared-gate Float32 operation rounding

## Objective
Qualify a clean GPU shared-gate provider that preserves the frozen CPU Float32 result across subnormal operands/results, rounding and cancellation.

## Permitted files
New shared_gate_rounded_* providers/probes/reference helpers/auditors and bounded result directories under the campaign scripts/results roots; this item and current status/diagnosis/frontier. No consumed-helper, Gesso ad47/vendor or supplier388e6f9 edits. Maintained integration requires a separate item after qualification.

## Interfaces
Same Float32 matrices/vector and apply!(y,x,sd,g) contract as qualified shared_gate_wide_provider_v2.jl. Use the actual Gesso CPU _shared_gate_add! authority plus an independent Base scalar reference. Charter first principle/development law and receipts apply. Physical Float64 arithmetic must retain explicit Float32 rounding between declared Float32 operations; wide accumulation without those boundaries is a different algorithm.

## Invariants
No oracle/tolerance/token changes, CPU tensor fallback, foreign process termination or unsupported-feature silent fallback. Keep original subnormal negative and original model9/13. Reuse frozen source912/env identity and under-lock256MiB GPU admission; bound6GiBRAM/1GiBswap,1200seconds/process,16MiBraw. Respect empty geometry and signedzero. No NaN payload or universal portability claim from a finite-input corpus.

## Tests
Compare two explicit variants: wide rounded epilogue with unchanged native Float32 dot, and a sibling also preserving Float32 product/sum rounding in the dot through physical Float64 operations. Include original32fixtures/real archived block1 endpoint,144epilogue boundary/signedzero/cancellation fixtures,8adversarial subnormal-dot witnesses and8seeded geometries. Record every variant on every fixture for3repeats, including failures rather than aborting at the first numeric mismatch. Preserve expected native-dot negatives. Independently recompute all CPU targets from raw inputs, require actual program inventory and typed LLVM/SPIR-V evidence for multiplication/addition and intermediate rounding, and reject repaired-digest output/intermediate or valid shader mutations. Source or harness errors cannot count as numeric results. No model admission without a later labelled model control and all original stress gates.

## Performance target
This qualification is correctness-first. Record allocations, compile/warm behavior and temporary-buffer cost. Baseline-relative operator costs must follow a numerically qualified result; inference throughput remains deferred. No speed claim from compilation or unit success.

## Expected artifact
Qualified broad provider or retained numeric negative, actual shaders and independent evidence selecting a minimal maintainable source repair. The full Campaign1 goal remains unchanged.

## Receipt
QUALIFIED FOR THE TESTED FINITE OPERATOR CORPUS. The integer-rounded provider matches the independent CPU target on all 576 full-provider outputs and passes effective output/shader mutation checks. This does not admit nonfinite semantics, general portability, cost baseline, model behavior or inference throughput; see the results and bounded limitations below.


## Progress receipt, 2026-10-21

- v1 pair: actual LLVM/SPIR-V narrowed apparent wide expressions to F32. Both variants CPU-exact 393/576 outputs; 183 failures each. Qualified negative: `shared-gate-rounded-1-validation-v2.json`.
- v2 pair: noinline helpers retain actual F64 multiply/add but native conversion paths still lose tiny values. Both exact 402/576; 174 failures each. Qualified negative: `shared-gate-rounded-2-validation.json`; effective repaired-output and valid-shader mutations rejected. CPU target unchanged; diagnostic FTZ simulator v1 correction retained separately.
- Integer rounding codec: exact F32-bit expansion to F64 plus existing integer RTE conversion. CPU 67,080 fixtures in three paths; GPU 603,720 exact scalar outputs over three paths/three repeats. Actual GPU programs use no floating arithmetic/conversions. Receipts `shared-gate-rounded-codec-cpu-2-validation.json`, `shared-gate-rounded-codec-gpu-1-validation.json`.
- First CPU reference harness incorrectly permitted sNaN conversion roundtrip folding; v2 forces actual conversion. First GPU mutation harness assumed a floating bitcast that the compiler had legally removed. Both original failures retained. Corrected mutation helper v2 inserts structurally valid F64 arithmetic and rejects it, and rejects a repaired-digest decoded-output bit mutation: `shared-gate-rounded-codec-gpu-1-mutation-validation-v2.json`.
- v3 pair keeps intermediate Float32 values as integer bits between wide operations. It uses separately qualified isolated Lava 388e6f9; frozen Gesso ad47 and its vendor remained unchanged. The first `shared-gate-rounded-3` launch failed helper parsing before numeric results. The corrected immutable probe completed in `shared-gate-rounded-4`: both full variants are exact on all 576 outputs; the epilogue-only sibling is exact555/576 and retains the seven intended adversarial dot fixtures177–183 failures across three repeats.
- Cost attempts v1/v2 completed their samples but were not admitted because their audit required raw shader equality. Cost capture v3 now qualifies after `spirv-opt --compact-ids`-only equality. It is a capture qualification, not a cost or speed admission; details and effective mutations are recorded below.

The earlier next checkpoint is fulfilled: v3 qualified the same 192-fixture/576-output corpus. Maintained integration and the unchanged original model gate are recorded in `2026-10-21_shared-gate-integration.md`.


## Qualified result and maintained transition

`shared-gate-rounded-4-validation-v2.json` independently confirms 192 fixtures,
576/576 CPU-exact full-provider outputs and 78,591 scalar comparisons. The
native-dot sibling is exact555/576 and fails exactly witnesses177..183 across
three repeats. All six actual programs validate with declared policies; the
rounded epilogue uses F64 arithmetic and integer rounding through output stores.
First v3 audit assumed a final floating bitcast; the actual store is UInt32.
Auditor v4 follows the integer dataflow instead; original failed audit retained.

Composed-provider repaired-output, valid wide multiply-to-add, and valid
output-store-bypass mutations all reject (`shared-gate-rounded-4-mutation-validation.json`).
Mutation helper v2 shadowed its imported auditor only during the third mutation;
v3 fixes that harness error. No numeric targets/artifacts were altered.

Cost capture v3 independently validates 400 measured samples, 8 cold samples,
8 warmup groups/47,364 warmup calls, zero measured compile/recompile/GC
contamination, exact candidate outputs, and all three actual shaders after
`spirv-opt --compact-ids` only. Receipt:
`shared-gate-rounded-cost-3-validation-v3.json`. The capture is qualified, while
`cost_admission=false`, `baseline_numerically_admitted=false`, and
`speed_claim_allowed=false`; the stock arm differs from actual Base CPU on three
of its four fixtures. The model was not loaded.

Effective cost mutations in `shared-gate-rounded-cost-3-mutation-validation-v1.json`
reject a cold candidate output bit flip with repaired descriptors, reject a
structurally valid `OpFMul`→`OpFAdd` mutation with repaired shader descriptors,
and classify an injected measured compile duration as timing contamination with
speed claims disabled. Original producer flags are retained for all three.

The observed finite-fixture timings and allocations are descriptive only. The
candidate is CPU-exact on all four fixtures; stock CPU differences are 1,522,
57, 0, and 319 differing Float32 outputs for fixtures 1, 4, 9, and 188 respectively, so the stock arm
is not an admitted numerical baseline. Candidate/stock medians are in
microseconds and observed allocations in bytes:

| Fixture | Candidate median / allocation | Stock median / allocation | Candidate minus stock median |
|---:|---:|---:|---:|
| 1 | 1,920.2725 µs / 15,584 B | 135.172 µs / 8,320 B | +1,785.1005 µs |
| 4 | 247.408 µs / 15,456 B | 97.6785 µs / 8,272 B | +149.7295 µs |
| 9 | 1,986.561 µs / 15,584 B | 120.4255 µs / 8,320 B | +1,866.1355 µs |
| 188 | 486.763 µs / 15,456 B | 69.133 µs / 8,272 B | +417.63 µs |

Source: `shared-gate-rounded-cost-3-observed-statistics.json`. These values are
not an admitted performance comparison because the stock arm is numerically
wrong on three fixtures. They do not establish end-to-end throughput or cost
admission.

Maintained integration now belongs to
`2026-10-21_shared-gate-integration.md`. The maintained candidate
is independently audited there; the original13 model gate remains 9/13. Full
Campaign1 remains open, and no inference-throughput claim is admitted.
