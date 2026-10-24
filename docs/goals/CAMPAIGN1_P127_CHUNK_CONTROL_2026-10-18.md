# Campaign 1: p127 chunk-size discriminator

## Objective
Determine whether the reverse/chunk128 p127 failure is intrinsic to whole-prefix execution, shared request history, or also present in the unchanged CPU reference.

## Permitted files
New external versioned helpers/runners/auditors/captures and planning status/frontier/work-item records only. Frozen source6d3683c8e87dce259de0eab8ba4f8d9644e275b5,910tracked files, qualified provider/supplier, all consumed environments/helpers, original CPU oracle/targets/gates remain unchanged. No libs,source,dependency,compiler/link or precision-policy edits.

## Interfaces
Ordinary public Gesso.reference_generate with CPUBackend or LavaBackend in separate serialized processes; existing public GESSO_PREFILL_CHUNK policy only. Same actual GGUF/tokenizer/canonical corpus and frozen oracle prompt validation. Host experts remain the ordinary model route. No activation injection, observers or method replacement.

## Invariants
Diagnostic only. Same p127 prompt,backend,weights,precision and generation settings; max_new_tokens1,greedy,eos_ids(). Test chunk128isolated,chunk64isolated,chunk128again,then the exact reverse prefix hello/p256/p129/p128/p127 at128,thenp127at64. These are fresh GenerateState calls within one process; shared-context histories are explicitly retained and are not claimed to be fresh processes. Record all prompt/full IDs,oracle comparisons,compile/allocation/RSS and source identities. Do not retarget1503 or excuse668 as a near tie. Unsupported execution fails explicitly; all910source hashes verified before/after.

## Tests
Finish and independently audit the original reverse53run first. Then separate CPU/GPU nine-request controls, raw-ID/ordering/source/helper/environment audit. If needed follow with bounded read-only model/operator trace plus ordinary controls; no source fix from a hypothesis alone. Source41andforward53passes do not admit Campaign1 while reverse fails. Sixteen-fresh matrix stays gated on BOTH boundary53passes.

## Performance target
No throughput claim; timing includes ordinary cold/warm compilation and is diagnostic only. Shared heavy locks,CPU20GiB/4GiB swap or GPU20GiB/4GiB swap/4800MiB budget,2400seconds,16threads,BLAS/OMP1.

## Expected artifact
Explicit whole/chunk/history/CPU-vs-GPU negative ledger and a bounded next localization step. No source/model/fullstress/production admission,push,merge or deployment.

## Receipt
OPEN. Ordinary source41/41 andforward53/53independently pass. Reverse128run currently reports p127=668vs1503in both implicit repetitions and first reused-state repetition;allothercompleted requests pass. Let the run finish without modification.

### Reverse terminal and CPU launch
Corrected independent reverse result49/53,allfour failures p127=668vs1503;zero source/integrity errors and910unchanged tracked files. Boundaryv1callback-case audit defect retained,V2corrected and validated. CPU-only nine-request control now running(session77305),helperp127_chunk_controls_v1.jl,ordinary source/API/chunk policy only. No source fix,CPU reference result or GPU diagnostic outcome yet.

### Terminal ordinary controls qualified
CPU nine-request process and GPU nine-request process both independently pass capture integrity. CPU p127at64=[1503,1503],at128=[1503,1503,1503],all9oracle comparisons pass. GPU p127at64=[1503,1503],at128=[668,668,668];three numeric mismatches,otherrequests correct. Both source manifests910files unchanged. `p127-chunk-controls-{cpu,gpu}-validation-1.json` binds raw sequences,oracle,source,helper/environment and terminal. A copied actual+expectedID mutation retaining green flags is rejected against frozenoracle;originalunchanged. This closes the bounded discriminator only:failure exists without reverse-history priming and CPU does not reproduce it at either chunk size. No source/model/fullstress/production admission.
