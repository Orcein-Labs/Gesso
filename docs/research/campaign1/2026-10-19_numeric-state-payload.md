# Campaign 1: independently replayable numeric state repeats

## Objective
Verify actual captured Float32 working-state bytes across repeated completed requests, rather than relying only on producer equality flags or digests.

## Permitted files
New external versioned helper, auditor, mutation test and local captures; planning records. Preserve consumed source ad47cb3, its912tracked files, environments/helpers/oracle and every existing artifact. No production, vendor, dependency or configuration changes.

## Interfaces
Ordinary reference_generate with persistent GenerateState(258), same twelve frozen prompt cases and three repetitions as the original history profile. Separate forward/chunk64 and reverse/chunk128 captures. Readbacks only after completed requests; full collection after each request. No method replacement, intermediate observer, priming or numerical intervention.

## Invariants
Capture hidden state, every present GDN convolution/recurrence array and valid-prefix K/V arrays. Record explicit absent fields. Exclude scratch buffers and checkpoint-owned copies from this payload claim. Preserve full generated IDs, prompt IDs and cache-ID checks. Store one binary pack per request with per-array shape/offset/size/digest descriptors. Bounded8GiB raw bytes per process, durable artifacts on the data disk, not the home disk. Default overlap/multi settings and frozen oracle remain unchanged.

## Tests
GPU execution requires independently green ordinary source41, both original53boundaries and16fresh-process matrix. Each profile:36complete raw sequences match frozen oracle;24repeat comparisons of all captured fields are bitwise exact and finite; inventory, offsets, shapes, hashes, terminal status and source/env/helper identities independently checked. Reject an actual copied raw-array mutation with its digest repaired and producer flags left green. Preserve negatives. Numeric payload qualification alone does not graduate Campaign1.

## Performance target
No throughput claim. Each run <=3600seconds,16Julia threads,20GiBRAM/4GiBswap,4800MiBGPU,BLAS/OMP1 and shared locks. Stream array packs; keep descriptors rather than all baseline arrays in RAM. Report readback/I/O overhead separately from normal execution.

## Expected artifact
Two independently qualified payload captures or retained negatives, effective numeric mutation and exact resource/provenance receipts. All other original final gates remain mandatory.

## Receipt
OPEN. Preparation only; source41qualifies, original boundary/fresh pipeline is still running. No payload GPU run yet.

### CPU format preflight
New helper softmax_tree64_state_payload_v1.jl is prepared and syntax checked. CPU-only preflight_state_payload_v1.jl evaluates its exact pack_state/same_payload definitions against40-layer synthetic arrays, passing495assertions:161-field inventories, contiguous offsets, byte extents and digests, equal repeated packs, unequal changed data, and nonfinite rejection. Receipt:state-payload-format-preflight-1/receipt.json; GPU_evidence=false. Helper is now consumed by this preflight and remains frozen. Independent complete-dataset auditor and repaired-digest raw numeric mutation are still unwritten; no GPU run has started.

### Independent reader and mutation preparation
validate_state_payload_v1.py now implements full source/env/oracle/prerequisite/terminal/event-inventory checks and independently rereads complete Float32 pack bytes, checks161-field coverage/presence/extents/digests/finite values, and recomputes all24bitwise working-state comparisons per profile. Its independent binary-reader preflight qualifies equal repeated packs, detects the changed fixture, and detects a one-scalar copied mutation with the digest repaired; originals remain unchanged. Receipt:state-payload-reader-preflight-1.json. GPU_evidence=false and full_dataset_audit_tested=false: this exercises the binary reader/comparator, not a real complete model dataset.

mutate_state_payload_v1.py is prepared and syntax checked, but has not run against a real model dataset. It requires a qualified positive payload receipt, copies the bounded dataset to temporary space on the data disk, changes one repeat2 hidden scalar by one representable Float32 step, repairs its digest while retaining producer green flags, and requires rejection specifically at the raw repeat comparison. Source helper and reader are consumed by their CPU preflights and remain frozen; corrections require new versions.

### Execution naming after matrix qualification
Use generic run_softmax_tree64_source_v1.sh big3600/16 with probe softmax_tree64_state_payload_v1.jl and log softmax-tree64-state-payload-PROFILE-1.log. Set C1_STATE_PAYLOAD_PROFILE=forward or reverse and C1_STATE_PAYLOAD_OUT to the same stem directory under results. Audit with validate_state_payload_v1.py, saving STEM-validation.json; then mutate_state_payload_v1.py --profile PROFILE, which expects that positive receipt and writes STEM-mutation-validation.json. Run one profile through audit/mutation before the next. No GPU payload execution until the original16fresh aggregate receipt independently passes.

### GPU execution started after prerequisites
Full16fresh aggregate receipt qualifies; all original source41 and both53boundary prerequisites pass. Forward payload capture is running(session14347,pid3259094), first request initially compiling, no completed-payload qualification yet. state_payload_pipeline_v1.py runs as session45362,pid3261832, state state-payload-pipeline-1.state.json. It observes the existing forward job without restarting, independently audits and applies repaired-digest numeric mutation, then captures/audits/mutates reverse128. It stops on failure. Do not independently duplicate these stages. CPU mutation copies run under the shared timed-job budget and keep temporary data on the data disk. Success qualifies these two payload slices only.

### Forward GPU payload qualification
softmax-tree64-state-payload-forward-1-validation.json independently qualifies36full CPU-oracle sequences and24bitwise exact, finite repeats, rereading2,485,665,792raw bytes. All912source files remain unchanged. Actual model-dataset mutation changes one hidden scalar by one representable Float32 step, repairs its digest and retains producer green flags; independent audit rejects specifically at the raw-repeat comparison. softmax-tree64-state-payload-forward-1-mutation-validation.json confirms originals unchanged.

Controller45362 has advanced to reverse128 capture (runner3264707). Forward14347 is terminal; do not wait on or relaunch it. Reverse payload qualification and all other final gates remain open.

### Both payload profiles terminal qualified
Reverse validation independently qualifies36CPU-matching full sequences and24bitwise exact finite comparisons across2,485,665,792raw bytes, with912unchanged source files. Reverse copied one-ULP hidden-state mutation with repaired digest/retained green flags is rejected specifically by the raw repeat comparison; originals unchanged. Controller45362 is terminal PASS_BOTH_NUMERIC_PAYLOAD_PROFILES. Combined scope:72sequences,48numeric repeat comparisons,4,971,331,584raw bytes. This closes the specified completed-working-state payload slice only; scratch and checkpoint-owned copies remain excluded from this claim. All other original final gates remain mandatory.
