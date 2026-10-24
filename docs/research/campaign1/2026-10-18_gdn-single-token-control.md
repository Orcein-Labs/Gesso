# Campaign 1: single-token GDN consistency control

## Objective
Test whether selecting the repaired GPU recurrence for every single-token update removes the reused-prefix failures exposed by the unfused-only source candidate.

## Permitted files
New external versioned helper/runner/auditor/receipts, this work item and planning status/frontier records only. Consumed Gesso480b5a4, source test environment, all908 tracked source files, helpers, original gate and CPU oracle remain frozen. No production/source edits.

## Interfaces
Process-local diagnostic override of Gesso.Inference._gdn_fused_decode for LavaBackend only, selecting the existing unfused GPU GDN path. No backend downgrade, CPU tensor injection, compiler/link override or arithmetic rewrite. Original41-case shipping driver and source guards run unchanged.

## Invariants
This is explicitly an unfused control, never the production fix or default-source acceptance. Log and receipt must label the override and prohibit model/source/production admission even if41checks pass. Preserve host-expert/overlap/multi defaults, original oracle, callbacks/recovery and full history/order. Retain red source gate35/41 and all negative experiments.

## Tests
Original full41-case gate at16threads/chunk64; independently rederive counts/IDs/cache-prefix/callback/recovery/repeat metadata with unchanged validator numerical logic. Verify all908 source bytes, exact driver/helper/environment identities and terminal status. If positive, localize/repair single-token GDN arithmetic in a new bounded source candidate with maintained old-red/new-green tests; never land this diagnostic selection override. If negative, capture first divergent inputs/state before choosing another repair.

## Performance target
N/A; correctness discriminator only. Heavy work remains serialized and bounded at20GiB RAM/4GiB swap,4800MiB GPU budget,1800seconds. No throughput claim.

## Expected artifact
Explicitly labelled diagnostic41-case ledger and independent audit; no model admission, push, merge or deployment.

## Receipt
OPEN. Source480b5a4 is35/41: freshcancel82202 and recovery3927 now pass, but reusedp64=46108/p127=668 fail in all three repetitions. Source branch at Inference.jl445 selects the separate fused path on one-token updates; source480 intentionally changed only unfused recurrence.

### Terminal negative result

Session32757 exited1; independent `gdn-recurrence-diagnostic-unfused-history-validation-1.json` recomputes35/41, the same six p64/p127 history failures as unmodified480b5a4. Freshcancel82202/recovery3927 still pass. `gdn-single-token-control-receipt-1.json` verifies908source hashes unchanged. The query-selection control does not repair this mismatch. Source inspection additionally confirms the prefill block call omits gdn_scratch, so the separate fused-decode branch is not selected merely because a prefix extension has one row. No fused source repair is justified by this experiment. Next: real partition capture, preserving this negative.
