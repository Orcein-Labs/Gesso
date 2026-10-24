# Campaign 1: imported-policy RoPE source repair

## Objective
Make native Float32 imported-frequency Lava RoPE preserve the executed CPU Float64 coefficient/rotation contract across partitions, then test the unchanged model gate.

## Permitted files
New isolated candidate from480b5a45112b81f2f7ca2deadfb624cf25207305: ext/lava_ops.jl, ext/lava_rope_reference.jl, test/test_lava_rope_reference.jl, test/runtests.jl, docs/ARCHITECTURE.md only. New external versioned tests, environments, source manifests, runners, auditors and receipts; planning status/frontier records. No consumed source/helper/env changes, libs/vendor/dependencies/oracle/target changes or unrelated refactors.

## Interfaces
Public Gesso.rope!, CPUBackend/LavaBackend, unchanged input validation and metadata policy. Public Lava NoContraction/FloatControls32/64. CanonII/III/LXXI/LXXII. Qualified external rope_partition_control_v1.jl arithmetic supplies the kernel, with explicit local rename only.

## Invariants
Native Float32 tensors with non-nothing imported Float64 frequencies only. Float64 phase/coefficient metadata built on CPU; Q/K data and rotation remain Vulkan. Existing unselected paths preserved verbatim. Per-call coefficient buffers, no persistent cache or mutable global resource registry. Logical coefficient bytes=16*seq*(rotary_dim/2), independent of context history. No compiler/link hooks, global precision switch, CPU tensor fallback or admission from helper results. Unsupported arithmetic capabilities fail explicitly through public Lava contracts.

## Tests
Maintained public-entry exact CPU comparisons, whole/partition controls, repeat identity, both layouts, partial-rotation tail preservation, multiple geometries and large positions. Same tests on frozen parent demonstrate numeric old-red/new-green. Exact kernel AST provenance against qualified helper; scoped format AST preservation. Then unchanged41-case original-history model gate. Only if green proceed to original forward/reverse53, sixteen fresh processes, numeric-state/GC, cancellation/recovery/callback/resume, W3/S1+mutations/vocab, Lava/Session/capability/GEMM gates, scalar484/RMS827, make test and final resource/cost/portability checks. No target or tolerance changes.

## Performance target
No inference throughput claim. Explicit per-call allocation policy avoids admitting diagnostic cache lifetime/concurrency assumptions. Measure cold compilation, warm operator time/allocation, GPU/logical coefficient memory and peak resources before final admission. Full inference throughput remains deferred.

## Expected artifact
Isolated reviewable local source commit and exact regression/model receipts, retained parent negatives. No push/merge/deployment. Full Campaign1 stress remains mandatory after the initial source model pass.

## Receipt
OPEN. External diagnostic passes unchanged41/41 with24repeat comparisons, independently recomputed. Separate diagnostic audit checks440calls,13cache entries,163840logical retained coefficient bytes,20480Base coefficient values,fouractualFloat64shader plans and908unchanged tracked source files. External result is diagnostic only. Source candidate not yet created or tested.

### Regression harness correction
Initial candidate73a0dc1 and its consumed test remain frozen. The first parent run errored before numerical checks because the new test omitted the required PrefillWorkload argument from public rope! calls. This is a test-harness defect, not numeric old-red evidence. A new isolated candidate `gesso-rope64-regression-20261018` corrects only those calls; runtime code remains byte-identical to73a0dc1. New environment/helper/runner versions retain both attempts. AST helperv1/v2 syntax errors are retained;v3 verifies the kernel AST against the qualified provider. Coefficient readbackv1 soft-scope error is retained;v2 validates20480values and rejects a copied numeric mutation with repaired raw digest.

### Maintained source qualification
Final isolated candidate6d3683c8e87dce259de0eab8ba4f8d9644e275b5 corrects only test invocation relative to73a0dc1. All909other tracked files match73a0dc1 exactly; parent480b5a4 differs only in the five permitted files. Runtime kernel AST matches the qualified provider after one local rename; scoped formatter2.14.0 preserves parsed ASTs of both new files. `rope64-source-focused-receipt-1.json`:parent83pass/108numericfail/0errors;new191/191 using identical maintained test bytes through publicCPU/Lava entry points. All910tracked source hashes unchanged after qualification. Initial no-workload harness failure is excluded from numeric evidence. No throughput or model admission.

Unchanged41-case source model gate is now running as `rope64-source-original-history-1.jsonl`, session18905. It uses ordinary shipping_stress_entry.jl, source guards and no process-local override. `rope64-source-consumed-run1-source.json` pins910source files, environment,runner,entry/oracle. Source matrix helperv2 adds serialized prompt/full IDs only; generation/ordering/targets remain unchanged. New matrix auditor has16synthetic positive schedules and16effective wrong-output mutations; these are validator tests, not GPU acceptance. Matrix execution is guarded on independent source41/41 and has not begun. Original full source stress, final cost/memory/portability/integration and make test remain mandatory.

### Ordinary-source41gate passes
`rope64-source-original-history-validation-1.json` independently recomputes41passed/0failed,24repeat comparisons; cancellation/recovery correct. `rope64-source-original-history-provenance-1.json` verifies ordinary shipping entry,no process-local operator replacement,all910tracked files and environment/helper/oracle identities unchanged. Frozen source6d3683c is the current candidate. This is an internal Gesso CPU oracle gate, not independent HF/PyStack parity or Campaign1 completion.

Forward53/chunk64 is now running as `rope64-source-boundary-forward-1.jsonl`,session7268. Its exact original cases,2repetitions,bothstate modes,GC/callback/cancellation/recovery settings are retained. New boundary auditor reuses unchangedv3 row/CPU-oracle rules, with explicit original53profile and implicit-state metadata checks. Both directions pass53synthetic-positive controls and reject wrong raw IDs with green flags retained; these are validator controls,notGPUevidence. The fresh16-process matrix runner now additionally refuses execution until BOTH boundary53receipts are independently green. Full original stress/integration/cost/memory/portability/fullmake gates remain open.

### Full boundary stress retains a negative
Forward53/chunk64 independently passes53/53. Reverse53/chunk128 finishes49/53:each of four p127requests returns668vs1503(two repetitions×implicit/state);allother49checks,includingcallback/retry/new-state/cancel/recovery,pass. `rope64-source-boundary-{forward,reverse}-validation-2.json` has zero source/integrity errors and910unchanged files. Source6d3683c remains frozen and is not admitted. Sixteen-process matrix remains gated and unexecuted.

Boundary auditorv1 incorrectly hardcodedp1forreverse callback checks,overcountingthreefailures;V2takes first(cases),as the unchangedshippingdriver does. V1and its incorrect reverse synthetic fixture remain preserved;V2passes correctedpositive controls and rejects wrong raw IDs with green flags retained. Numerical/oracle row logic is unchanged. Follow-on external-only CPU/GPU whole/chunk/history discriminator:`CAMPAIGN1_P127_CHUNK_CONTROL_2026-10-18.md`.
