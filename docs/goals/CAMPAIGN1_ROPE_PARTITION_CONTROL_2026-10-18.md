# Campaign 1: matched-input imported-policy RoPE partition control

## Objective
Test whether current imported-policy RoPE changes identical Float32 Q/K values when positions are processed whole versus partitioned, and qualify bounded shared-coefficient alternatives before any source change.

## Permitted files
New external versioned helpers/runners/auditors/receipts, this work item and planning status/frontier records only. Frozen source480b5a45112b81f2f7ca2deadfb624cf25207305, environments, all908tracked files, previous captures/helpers, original oracle/gates stay unchanged. No libs/dependency/source edits or compiler/link hooks.

## Interfaces
Current public CPU/Lava rope!; actual GGUF header/spec positional policy. Synthetic matched finite Float32 inputs at actual Qwen head/rotary dimensions. External explicit GPU rotation kernel uses public NoContraction/FloatControls only. Candidate32 uses current single-row Float32 phase/reduction/CPU coefficient semantics for every partition; candidate64 uses exact executed CPU Float64 phase/coefficient/rotation contract. CPU computes positional metadata only; Q/K rotations remain on GPU.

## Invariants
Retain current whole and partitioned arms. Same exact input bytes/positions/layout/policy in every arm. Preserve unrotated tails and input ownership, no CPU activation injection or model-target change. Two imported-policy layouts plus bounded larger-position control; repeated outputs, all raw CPU/GPU arrays and coefficient bytes, actual SPV/LLVM identities and 64MiB cap. Operator evidence is never model admission.

## Tests
Current whole vs original1,30,1,1,30,1 partitions; both shared coefficient alternatives whole/partitioned; compare every raw Q/K output against executed CPU reference and each other. Cover half-split/interleaved and positions through65535, finite/special validation inherited where relevant. Independent raw/source/header/coefficient/shader audit and effective numerical mutation. If qualified, test a targeted model control then new bounded source candidate; unchanged41/full original stress remain mandatory.

## Performance target
No throughput claim. Record bounded resource/compile/allocation limitations; choose production caching only after evidence. Heavy work shared-lock serialized,20GiB/4GiB swap,4800MiB GPU budget,1800seconds,16Julia threads,BLAS/OMP1.

## Expected artifact
Explicit old/new operator partition ledger with genuine CPU reference/raw shader proof and limits. No source/model/production admission, push, merge or deployment.

## Receipt
OPEN. Actual-model trace independently qualifies observation: fresh82202/history46108 reproduced in ordinary controls, IDs/logits/hidden/live-state digests identical to observed routes. Token64 first differs at block9 mixer; earlier cached keys differ at attention4/8 while values4/8 match. Current RoPE uses CPU-built single-row tables but GPU trig for multi-row imported frequencies. This is a hypothesis to test, not an established cause or source fix.

### Terminal qualified operator result
`rope-partition-control-validation-1.json`:PASS_OPERATOR_INTEGRITY,54repetitions;all9current-path whole/partition comparisons differ. Both shared-coefficient alternatives are partition invariant. Float64 alternative is bit-exact to executed CPU reference for every output in all18relevant repetitions. Actual32/64shader modules validate publicNoContraction/FloatControls andspirv-val. Independent NumPy typed arithmetic checks all Q/K values and tail preservation. Copied metadata64outputs+0.5 with repaired digests, unchanged tails/repeats/partition flags are rejected; originals unchanged. All908tracked source hashes remain pinned. This closes the bounded operator control only; it makes no MPFR elementary-function accuracy, HF/PyStack, source/model/production or performance claim. Follow-on external model control passes41/41 separately; maintained source/full stress stay open.
