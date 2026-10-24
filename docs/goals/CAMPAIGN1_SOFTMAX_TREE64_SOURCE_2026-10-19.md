# Campaign 1: stable checked-softmax row reduction

## Objective
Remove the checked Float32 softmax denominator's shape-dependent reduction order, then pass the unchanged original model and stress gates on maintained source.

## Permitted files
Only after the separately qualified model control passes: a new isolated candidate from6d3683c8e87dce259de0eab8ba4f8d9644e275b5, changing ext/lava_ops.jl, a new ext/lava_softmax_reference.jl, test/test_lava_softmax_reference.jl, test/runtests.jl and docs/ARCHITECTURE.md. New external versioned runners/environments/guards/auditors/receipts and planning records. Preserve every consumed source, helper, environment, capture and CPU oracle. No vendor/libs/dependency changes.

## Interfaces
Existing checked public Gesso.softmax! and ordinary reference_generate, Session and Lava backend paths. Exact qualified kernel and launch arithmetic from softmax_tree64_operator_v1.jl, with local names adapted only. Public Lava FloatControls32 and NoContraction. Use the new reduction only for Float32 exponent matrices; retain original non-Float32 behavior and unchecked fused path verbatim.

## Invariants
No mask/max/exponential/division/storage/finite-check change; one64-lane workgroup per row, Float32 lane-strided accumulation and fixed32/16/8/4/2/1 tree, explicit barriers, no atomics or subgroup-size assumption. Per-call denominator allocation, no retained cache. No CPU activation fallback, implicit precision downgrade, ID/prompt special case, chunk cap, oracle/tolerance retarget or compiler hook. Unsupported arithmetic capability fails through the existing explicit Lava contract.

## Tests
Prerequisites: independent operator positive and numeric-mutation receipts, model nine-request positive and raw-ID-mutation receipts, exact model/operator shader identity. Maintained tests through public softmax: causal masking, row normalization and existing CPU tolerance, seeded whole/partition bit identity across row-count and64-lane boundaries, rectangular cached-prefix scores, deterministic repeats and input preservation. Same maintained test bytes on parent establish numeric old-red/new-green. Verify kernel AST provenance and scoped formatter AST preservation.

Then run unchanged original41-case history and both53-case boundary schedules. Both boundaries must pass before16fresh-process matrix (ten16-thread processes and two each1/4/8-thread processes), numeric-state/GC, overlap-off/multi-off, cancellation/recovery/callback/two-token resume, real Lava/Session integration, GEMM capability/registration enabled/disabled, W3/S1 with effective mutations and exact vocabulary, scalar484/RMS827, actual make test, and final resource/capability/cost receipt. Earlier-candidate results do not transfer.

## Performance target
No full inference-throughput claim. Measure operator cold compilation, warmed latency/allocation, logical/device denominator memory and peak resources against the parent after numerical qualification. Check portability through public capability contracts. Retain negative results and explicit unverified hardware limits.

## Expected artifact
Reviewable isolated local source commit, exact provenance and original-scope stress receipts. No push, merge or deployment. Campaign1 remains active until every original gate passes.

## Receipt
OPEN; source candidate not created. External operator qualification and mutation pass. Process-local model control remains running; first five requests match frozen targets, including both p127 chunk policies. Do not create source until full independent model qualification and its mutation gate pass.

### Candidate created after prerequisites
External model positive and mutation gates pass. New isolated candidate ad47cb3241bb56f9289e3d6a10323e38b8566dbc changes only five permitted files. Kernel and launch AST match qualified provider after local name normalization and removal of its diagnostic type capture. Scoped formatter2.14.0 preserves both new-file ASTs. Manifest softmax-tree64-source-consumed-run1-source.json freezes912tracked source files, path-only copied offline environment and helpers. Maintained public-entry regression is running(session61660). Parent same-test negative, model41 and original stress remain unrun.

### New source regression terminal
softmax-tree64-source-focused-1.log is terminal0:718/718 public-entry assertions pass. Same maintained test bytes are running on parent6d as softmax-tree64-parent-focused-1.log(session10452,pid3121762). No source model admission yet. Fresh matrix validator preflight passes16synthetic schedules and rejects16joint-ID mutations; this is not GPU matrix evidence.

### Maintained regression qualification
Parent same-test terminal1:694passed,24numeric partition-identity failures,zeroerrors. Newsource718/718. softmax-tree64-source-focused-receipt-1.json independently confirms both schedules, exact unchanged other runtime paths, only five permitted source changes, path-only offline environment copy, scoped-format and kernel/launch AST provenance,912unchanged source files. Original unchanged41-case model history is running(session99883), ordinary shipping entry with no override. Both53boundaries,16fresh matrix and all remaining original gates remain mandatory and unrun on this source.

### Ordinary source history qualification
softmax-tree64-source-original-history-validation-1.json independently passes41/41, including24recomputed raw-ID repeat comparisons,41GC records and cancellation/callback/retry/new-state recovery. Separate provenance receipt confirms ordinary source execution, no process-local override,912unchanged files and frozen environment/oracle/helpers. Actual copied-event mutation changes a token plus reported generated/expected/stability suffixes while retaining green flags; frozen v3 validator rejects it with3computed failures and originals unchanged. This is not full numeric-state payload repeat qualification or independent HF/PyStack evidence.

### Original stress pipeline
Forward53/chunk64 is live as session77040,pid3127452. Controller softmax_tree64_stress_pipeline_v1.py runs as session72844,pid3128287, state softmax-tree64-stress-pipeline-1.state.json. It observes this existing job without restarting, independently audits it, then runs reverse53/chunk128 and its audit, then the unchanged16fresh-process matrix and aggregate audit. It stops on failed jobs/audits or disappearance without a terminal receipt. All heavy stages use the existing shared locks/budgets; fresh profiles remain ten16-thread and two each1/4/8-thread processes. Do not independently launch stages already owned by this controller. Controller success admits only these stress slices; numeric-state payloads, mode controls, integration, capability/registration/GEMM, W3/S1/vocabulary/mutations, full make test, formatting and final costs remain mandatory.

CPU-only final-suite and actual make-test wrappers are prepared, not executed. They preserve the documented make entry point and avoid a known generic-entry Lava guard after CPU-only helpers. No acceptance evidence is claimed from preparation.

### Original forward boundary qualification
softmax-tree64-source-boundary-forward-validation-1.json independently passes53/53,24repeat comparisons, zero integrity/source errors and912unchanged files. Controller72844 has advanced to reverse53/chunk128 (runner child3131544); the16fresh matrix stays gated. Source and all consumed helpers remain frozen.

Completed numeric payload preparation is documented separately in CAMPAIGN1_NUMERIC_STATE_PAYLOAD_2026-10-19.md. Its CPU-only format preflight passes495assertions; this is neither GPU payload nor model acceptance.

### Original reverse boundary qualification
softmax-tree64-source-boundary-reverse-validation-1.json independently passes53/53 with24repeat comparisons, zero integrity/source errors and912unchanged source files. All four formerly failing p127 requests now match the frozen CPU1503 sequence in implicit/state modes and both repetitions. Controller72844 has started the original16fresh-process matrix (batch runner3136575), not yet qualified. Remaining gates stay open.

### Original fresh-process matrix qualification
Controller72844 is terminal successful. softmax-tree64-source-default-matrix-validation-1.json independently qualifies all16fresh processes:80full sequences,64p64=82202 and16hello=11, ten16-thread/two1-thread/two4-thread/two8-thread profiles, hello ordering controls and912unchanged source files. Both original53boundaries and source41 remain green. This satisfies the full repeated-source target; it does not close remaining numeric-state/control/integration/fullsuite/resource gates.

GPU was rechecked after the long interval: desktop consumers only, no foreign model job/shared lock or pause marker,39GiBhome and1.6TiBD free. Forward numeric-state payload capture started as14347/pid3259094. Controller45362/pid3261832 owns its subsequent audit/mutation and reverse capture/audit/mutation. No source or qualified external environment edits.
