# Campaign 1: actual-model p64 partition trace

## Objective
Find the first block/stage where current GPU generation differs between fresh64 and the original1/31/32/33/63/64 prefix history.

## Permitted files
New external versioned helpers/runners/auditors/receipts, this work item and planning status/frontier records only. Gesso480b5a45112b81f2f7ca2deadfb624cf25207305, all908tracked files, environments, consumed helpers, original gates and oracle remain frozen. No libs/dependency/kernel/compiler/link/backend changes.

## Interfaces
Clone the exact current reference_generate under a new process-local name, adding read-only stage calls only. Capture token64 rows at block enter, mixer residual/sub, FFN norm/down, finish for all40blocks. All arithmetic stays on the existing source path, including declared host experts. No tensor injection or method replacement of operators/query traits.

## Invariants
Observed fresh64 and original prefix history plus ordinary-public unobserved controls for both. Fresh must reproduce82202; history must reproduce46108 with unchanged CPU target82202. Controls must match each observed route's full IDs, final hidden/logits and complete persistent-state hashes. Same prompt, state semantics, chunk64/default window and source environment. Serialize full hidden/logit rows and observed token64 stage rows;64MiB artifact cap. Readbacks are diagnosis, not timing evidence. No model admission from a trace.

## Tests
Independent raw-derived first-difference report, logit argmax/IDs/prefix consistency, exact clone inverse/source908hashes, control state/logit digest comparison, stage/request schema and vocabulary. Record if observation changes the source failure; do not excuse or reinterpret it. If no difference appears in serialized rows, inspect persistent state/uncaptured seam rather than declaring parity. Preserve negative single-token and GDN-only replay results.

## Performance target
N/A for throughput. Shared GPU/timed locks,20GiB RAM/4GiB swap,4800MiB GPU budget,1800seconds,16Julia threads, BLAS/OMP1. No timing claim.

## Expected artifact
Actual-model failed-history localization and reviewable source/input/control provenance. No source/model/production admission, push, merge or deployment. Complete unchanged41 and original full stress remain required for any eventual source fix.

## Receipt
OPEN. Source480 is35/41. Single-token selection control independently stays35/41. Real first GDN-block replay validates112stage rows,196raw fields and all four observer controls; GPU whole/history is bitwise identical throughout that isolated GDN block. The trace now covers actual residual/FFN and later blocks to locate the model discrepancy.

### Terminal observation qualification
`p64-partition-model-trace-validation-1.json`:PASS_CAPTURE_INTEGRITY,observation_qualified=true.400stage rows/480rawfields,14requests,fourcases. Fresh observed/control82202,history observed/control46108; logits argmax,hidden and reported live-state hashes match ordinary controls. Token64 first visible difference is layer9mixer,all2048residual/sub values; preceding token64finishes1–8match. This does not establish equality for earlier token rows. K-cache hashes differ at attention4/8 while V-cache hashes match there; this motivated the separately qualified RoPE experiment, not a causal proof by itself. Repaired-digest raw-logit mutation is rejected. All908tracked source files unchanged. No source/model/production admission.
