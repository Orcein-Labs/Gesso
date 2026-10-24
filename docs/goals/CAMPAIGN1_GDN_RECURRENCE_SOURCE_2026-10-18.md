# Campaign 1: CPU-order GDN recurrence source candidate

## Objective
Repair fresh-state model parity with the independently qualified GDN recurrence/Julia-exp32 GPU arithmetic and retain the original full stress finish line.

## Permitted files
New candidate from9335b649fad4b896393b5e8bea6dc4db748265d6 only: ext/lava_hybrid.jl, ext/lava_gdn_reference.jl, ext/lava_exp32.jl, test/test_lava_gdn_reference.jl, test/runtests.jl, docs/ARCHITECTURE.md. New external versioned environments/runners/auditors/receipts and planning status/frontier records. No consumed trees, libs, core dependencies, vendor changes, gate/oracle changes or unrelated refactors.

## Interfaces
Existing Gesso.Inference._gated_delta_rule! Lava backend method; per-kernel public Lava NoContraction/FloatControls; CPU recurrence remains the unchanged internal oracle. Canon II/III/LXXI/LXXII. Base Julia1.12.6 exp32 coefficient provenance retained with MIT attribution.

## Invariants
Only unfused recurrence arithmetic changes: physical Float64 q/k normalization, Float32 exp32 decay, Float64 reductions and Float32 recurrent-state round points. Preserve shape guards/head mappings/empty behavior, host experts, quantization, routing, all state/model semantics and default environment. Keep gated normalization and fused single-token decode unchanged in this bounded candidate. No CPU tensor fallback or compiler/link override. Arithmetic kernel bodies match consumed qualified operators modulo local names/wrapper plumbing; retain negative combined-normalization evidence. Unsupported capabilities fail through existing public contracts.

## Tests
Maintained exact CPU-operator comparisons for carried state, both head mappings, l2 on/off, multiple geometries and input preservation; Base exp32 special/boundary controls and actual GPU decay outputs; immutable real first-block recurrence fixture. Demonstrate numeric old-red/new-green through public recurrence. Then unchanged41 original-history model gate. If fully green, all original forward/reverse53, sixteenfreshprocess64request, GC/numeric repeat/cancellation/recovery/callback/resume, W3/S1+mutations/vocab, Lava/Session/capabilities/GEMM, scalar484/RMS827 and fullmake test/format/finalresource gates remain mandatory. A red model gate is retained and localized, never excused.

## Performance target
No throughput claim. Capture final-source compile/warm allocation/operator cost and peak memory; full inference throughput remains deferred. Target explicit bounded resources and maintainability; investigate material regressions.

## Expected artifact
Reviewable isolated source candidate with exact source receipts and maintained regressions. No model admission until unchanged gates and full stress complete. No push/merge/deployment.

## Receipt
OPEN. External recurrence-onlyfreshp64 reaches82202 with30calls. Independent source/raw-logit/shader audit passes and repaired-hash output mutation is rejected. Combined gatednorm negative7439 retained. Frozen source9335b64 remains39/41. New candidate not yet executed.

### Source and focused checkpoint

New isolated `gesso-gdn-recurrence-20261018`, local commit `480b5a4` (full hash in model source manifest), parent9335b64. Six permitted files changed; no vendor/core dependency/libs changes. Three actual kernel ASTs and the complete exp32 code match qualified helpers modulo explicit local symbol renames and line numbers (`gdn-recurrence-source-arithmetic-AST-2.toml`). V1 AST helper failed on quoted-symbol syntax before comparison; retained unchanged; V2 uses explicit Symbol strings, no source change. Scoped formatter2.14.0 preserved parsed ASTs of all three new files (`gdn-recurrence-source-format-1/receipt.toml`).

Actual Vulkan maintained regression: old source289passed/127failed/0errors, with126 exact output/state failures plus one missing-helper check; new source418/418, including original captured64-token block, three repetitions and carried-state controls. Both use identical maintained test bytes and source/path-checked real Vulkan execution. `gdn-recurrence-source-focused-receipt-1.json` binds the consumed six-file hashes. No source changed during tests.

Unchanged41-case model gate now running as `gdn-recurrence-original-history-1.jsonl`, runner `run_gdn_recurrence_model_v1.sh`, live session64058. `gdn-recurrence-consumed-model1-source.json` pins all908tracked files, helpers/environment/oracle. No model pass is claimed; complete stress, final engineering/resource/portability gates remain open.

### Terminal source result:35/41, not admitted

Model session64058 exited1. Independent `gdn-recurrence-original-history-validation-1.json` recomputes35passed/6failed, with24 repeat-ID comparisons stable. Fresh-state cancellation now emits82202 and recovery3927, both correct. Reused-prefix p64 emits46108 instead of82202 and p127 emits668 instead of1503, in all three repetitions; all other checks pass. `gdn-recurrence-postrun-source-1.json` confirms all908tracked source hashes unchanged. Candidate480b5a45112b81f2f7ca2deadfb624cf25207305 and its environment/helpers remain frozen. No full stress run or admission follows this red model gate.

Inference.jl445 selects a distinct fused GDN path for one-token updates with live scratch/state; this bounded candidate intentionally left that path unchanged. This source observation motivates the external control `CAMPAIGN1_GDN_SINGLE_TOKEN_CONTROL_2026-10-18.md`; it does not yet prove the causal explanation. The control is never default-source acceptance or the intended production repair.
