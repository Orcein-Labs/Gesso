# Campaign 1: external GDN supplier composition discriminator

## Objective
Determine whether qualified recurrence/exp32 and physical-Float64 gated normalization compose through the public Lava arithmetic contract to repair fresh p64 generation.

## Permitted files
New versioned helpers, runners, auditors and receipts under external campaign scripts/results; this work item and campaign status/frontier records only. Frozen Gesso9335b64, its vendored Lava289d069, environments, oracle, consumed helpers and all original worktrees remain unchanged. No libs or dependency changes.

## Interfaces
Existing Gesso Lava GDN recurrence and gated-normalization methods, Julia-exp32 helper and reference recurrence, physical-divide operator kernels. Process-local Gesso method interventions may select these exact GPU operators; public Lava.with_no_contraction/with_float_controls only. No compiler/link/cache monkeypatches. Canon II/III/LXXI/LXXII.

## Invariants
Preserve arithmetic and input/output layouts from consumed successful operator proofs; any wrapper changes are recorded. No CPU tensor evaluation/fallback/injected fixture inside model requests. Model outputs/hidden and logits may be read back only after completed requests. Original CPU oracle and all admission gates stay unchanged. Stock arm must reproduce65017. Every model arm uses a fresh state and the oracle's exact64-token prompt. No selected-arithmetic call may silently skip. Capture actual shader/LLVM identity, raw output/hidden bytes and exact source hashes; 64MiB artifact cap. Preserve failed arms; diagnostics cannot admit production.

## Tests
Stock fresh p64; recurrence/exp32 only; recurrence plus physical-divide gated normalization twice. Count30 GDN blocks per selected request and verify cache IDs. Save full generated IDs and independently re-read/argmax248320-element logits. Require explicit nonzero exit for CPU-ID mismatch. Audit model/controls/source/module identities and repeat equality. If effective, scope a new source candidate and rerun unchanged41 and full stress; if ineffective, retain exact stage evidence and localize the remaining seam.

## Performance target
N/A for throughput; this is a correctness discriminator. Record resource caps, per-request elapsed time and raw artifact bytes; compilation remains included and timings are not warm throughput evidence.

## Expected artifact
Four-request diagnostic ledger with actual public-supplier shader provenance and independently validated numerical/source integrity. No source/model/production admission, push, merge or deployment.

## Receipt
OPEN. Starting from independently verified9335b64 at39/41. Hardware RTX5060, Julia1.12.6, Ornith GSQ-RCO3bit, Lava Vulkan, host experts enabled, GPU budget4800MiB and memory cap20GiB. Expected CPU p64=82202; observed baseline fresh p64=65017. All final-source stress gates remain open.

### Terminal evidence

Session81791 exited1: stock65017, recurrence-only82202, combined7439, combined_repeat7439. Exact selected calls are0/0,30/0,30/30,30/30. Combined logits repeat bitwise. Independent `gdn-supplier-composition-model-validation-1.json` reports PASS_INTEGRITY with nine validated actual NoContraction shaders, physical F64 division, preserved qualified source bodies,905 unchanged tracked source files, raw-logit argmax reproducing all generated IDs and4,345,356 artifact bytes. This is a narrow positive for recurrence-only and a preserved negative for adding gated normalization; it is not a full-model gate.

`gdn-supplier-composition-mutation-validation-1.json` proves rejection of copied recurrence logits whose raw values and digest were changed while claimed IDs/green flags remained. Original bytes unchanged. No CPU fixture was injected into requests; no compiler/link hooks were installed. All original model/stress admission gates remain open. Next bounded source item: `CAMPAIGN1_GDN_RECURRENCE_SOURCE_2026-10-18.md`.
