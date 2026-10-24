# Campaign 1: real first-block partition replay

## Objective
Localize the first numerical difference when the same64 real embedding rows traverse the first GDN block whole versus the original1/31/32/33/63/64 prefix history.

## Permitted files
New external versioned helpers/runners/auditors/receipts, this work item and planning status/frontier records only. Source480b5a45112b81f2f7ca2deadfb624cf25207305, all908 tracked files, environments, consumed helpers, oracle and original model gates remain unchanged. No libs, dependency, production/kernel/link/compiler changes.

## Interfaces
Clone the exact current Inference._hybrid_linear_block! under a new process-local name with read-only stage capture calls only. Public CPU and Lava operator paths; no CPU tensor injection or operator substitution inside GPU routes. Verify original captured embedding fixture against current CPU embedding lookup and original p64 oracle prompt before use. Load only first block to device; retain existing host expert setting.

## Invariants
Four observed routes: CPU whole/history, GPU whole/history. History chunks are1,30,1,1,30,1 (cumulative1,31,32,33,63,64). All routes start with zero conv/recurrent state; q/k mapping/spec/epsilon/input weights unchanged. Eight stage captures cover norm/projections/conv/split/gates/core/merged/sub. Capture final state for all observed routes and state at63 for history routes. Compare every observed route's final sub/state with an unobserved ordinary-public-method control using the same stream. Readbacks are diagnostics, never timing evidence. Source and raw hashes pinned, artifact cap128MiB, explicit nonzero execution/integrity failure. CPU/GPU and partition differences are findings, not automatically harness failures or admission.

## Tests
Raw-derived stage comparisons aligned by original token position, including actual last-row64 values; independent shader/source/clone/fixture identities and counter/schema checks; untouched input/weights; observed-vs-unobserved controls. Preserve differences even if one isolated block cannot reproduce the whole-model failure. If first-block stages agree, continue to the next real divergence; never infer whole-model parity from this replay.

## Performance target
N/A; correctness localization. Serialized shared GPU lock,20GiB RAM/4GiB swap,4800MiB GPU budget,1800seconds, sixteen Julia threads, BLAS/OMP1. No throughput claim.

## Expected artifact
Bounded real-input stage ledger and independently recomputed first-difference report, with full source/input/control provenance. No source/model/production admission, push, merge or deployment.

## Receipt
OPEN. Source480b5a4 passes418 focused assertions but model35/41; external single-token query control reproduces p64=46108 and p127=668 so far. Prefill call site does not supply gdn_scratch, so the separate fused-decode branch is not implied by a one-token prefix extension. Finish/validate the control before running this replay.

### Completed diagnostic: GPU first GDN block is partition invariant

Session4554 exited0. `gdn-partition-replay-validation-1.json` independently validates112stage rows/196raw fields,6state boundaries,28chunk rows,8cases,908unchanged source files and55,849,181 artifact bytes. All four ordinary-public-method controls report final sub/conv/recurrent digests equal to independently reread observed raw values. Control payloads themselves are not serialized; before/after weight digest records match, with no full raw weight archive. Observer clone inverses exactly to current source. Its Julia1.12 binding-world warning is preserved; this diagnostic is not a maintained-runtime claim.

GPU whole-versus-history is bitwise identical in every captured GDN stage, all64rows and all fields. CPU partitioning first differs at gdn_proj/a (411values,max1.9073486328125e-6), and wholeCPU/GPU first differs at the same field (1698values,max1.621246337890625e-5). These are findings, not source admission or proof that isolated differences explain model IDs.

`gdn-partition-reference-compiler-shaders-1/receipt.json` archives five stock compiler SPV/LLVM pairs emitted within this isolated run interval, covering all three reference kernels and their workgroup variants. All SPVs independently validate with complete NoContraction coverage and preserve32/64 modes. Public launch-plan/link records were not serialized, so the scope is compiler-emitted identities plus existing supplier/runtime qualification.

`gdn-partition-replay-mutation-validation-1.json` proves rejection of copied final-sub values changed+0.5 with repaired file digest and retained green control flags; original capture unchanged. This checks captured-output/control consistency, not an independent mathematical CPU oracle. The next actual-model trace includes residual/FFN and later block boundaries: `2026-10-18_p64-partition-model-trace.md`. All original full-model/stress gates remain open.
