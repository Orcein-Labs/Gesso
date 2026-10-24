# Campaign 1: fixed-tree softmax model control

## Objective
Test whether the qualified fixed64 Float32 denominator removes the frozen p127 chunk128 model mismatch, while preserving the existing nine-request control schedule.

## Permitted files
New external versioned probe, capture, auditor, mutation test and planning records only. Source6d3683c and all910tracked files, consumed inputs/helpers/environments, model/oracle/targets and public acceptance gates remain unchanged.

## Interfaces
Process-local replacement of checked _lava_softmax! with its exact original body except the denominator call to the qualified fixed64 implementation. Kernel and launch source copied byte-for-byte from the qualified operator helper; no observed readbacks in model execution. Actual shader artifacts archived. Ordinary public reference_generate and unchanged nine-request schedule from p127_chunk_controls_v1.jl.

## Invariants
Require operator positive audit and effective repaired-digest numeric mutation before execution. No other method replacement, CPU tensor injection, precision/global compiler change, mask/exp/division change, prompt/ID retarget or chunk cap. Restore is process exit. Record exact inverse source proof, source hashes before/after, helper/env/oracle identities. This is diagnostic model control; maintained source, full stress, production admission remain false.

## Tests
Nine ordinary public requests: p127 at128/64/128, then hello,p256,p129,p128,p127 at128, then p127 at64. Independently compare full IDs and prompts with frozen CPU oracle; compare baseline GPU control and reproduce shader/kernel provenance. Reject copied numeric ID mutation even with repaired reported expected IDs/green flags. Keep negative results. Only a qualified passing control permits scoping a fresh maintained source candidate and unchanged original stress gates.

## Performance target
No throughput claim. Bounded3600s,16Julia threads,20GiB RAM/4GiBswap,4800MiBGPU,BLAS/OMP1, shared timed-job/GPU locks, raw capture <=8MiB. Report compile/alloc/GC/RSS as observations, not warmed throughput.

## Expected artifact
Qualified positive or negative model control with frozen targets and exact intervention provenance.

## Receipt
OPEN. Operator positive audit passed9/9 and all3 complete probability assemblies are bitwise partition invariant; mutation prerequisite pending. Model probe not yet executed.

### Qualified model control
Runner terminal0. softmax-tree64-model-validation-1.json independently passes9/9 with zero mismatches. p127 is1503 at128 three times and64 twice; hello,p256,p129,p128 retain11,8924,1318,524. Exact actual shader bytes match qualified operator1b59c1c9. All910source files unchanged. softmax-tree64-model-mutation-validation-1.json rejects joint generated/reported-expected ID tampering while green flag is retained. Process-local diagnostic only; maintained source/fullstress/production admission remain false.
