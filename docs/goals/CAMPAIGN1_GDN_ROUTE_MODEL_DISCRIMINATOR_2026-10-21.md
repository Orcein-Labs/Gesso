# Campaign 1: compare fused and general GDN model routes

## Objective
Determine whether selecting the already maintained general GDN route changes the remaining original13 token failures that shared-gate, convolution, and wider-projection repairs did not resolve.

## Permitted files
New gdn_route_model_* helpers, runner, preflight, auditor and bounded results under campaign scripts/results; this item, latest status and resume frontier. No maintained/frozen source, supplier, model, oracle or consumed helper changes. One exact process-local capability method override, _gdn_fused_decode(::LavaBackend)=false; no arithmetic or tensor replacement.

## Interfaces
Existing Gesso Inference capability seam and general _hybrid_linear_block! route. The maintained general route uses the already integrated Float64 q/k normalization/reductions, explicit Float32 state rounding, Base-compatible decay and per-kernel arithmetic contracts. Public convolution/gates/gated norm remain their existing implementations. GPU tensors stay on device; host experts remain the declared existing model route.

## Invariants
Same916maintained source bytes, candidate2108748, exact environment, unchanged original13 request/oracle profile and resource guards. No earlier diagnostic conv/projection overrides carried into this arm; baseline is the actual maintained model2. Wait for5500MiBfree under both locks and respect pause/foreign jobs.20GiB RAM/4GiB swap,1800seconds. No oracle/tolerance/token/quantization changes, CPU tensor fallback or hardware-specific BLAS emulation.

## Tests
Strict parse and runtime method-binding preflight before model. Bind original true method source, exact false override text, actual selected method/module/file and before/after capability values. Archive all actual general qk/decay/recur shaders plus LLVM, verify NoContraction/preserve32+64, and require zero fused conv/delta/gated-norm shader launches. Independent original13 row/oracle/source/environment audit; meaningful model-ID mutation. Existing general-operator tests are supporting evidence only; no full stress or production admission from this control.

## Performance target
N/A for this causal diagnostic. Record actual allocations/compile/resource observations; do not claim throughput from this control. A positive result would require a maintainable source fix and every original stress/engineering gate, including cost.

## Expected artifact
Qualified model positive/negative selecting whether to repair the fused route or investigate broader model state differences. Scope remains the full original Campaign1.

## Receipt
QUALIFIED NEGATIVE — 2026-10-22. The terminal `gdn-route-model-1` capture independently audits to 9/13 against the internal Gesso CPU oracle (not independent HF): all four normal `hello` rows emit `[11, 271]` rather than `[11, 353]`, with zero integrity/source errors. The exact process-local capability override selected the maintained general route; its capture contains five qk/decay/recur programs, all actual general shaders plus LLVM, and zero fused conv/delta/gated-norm programs. Selecting the general route does not resolve the failures, so this result does not justify a fusion-disable source fix. Continue broader model-state investigation; Campaign 1 remains open.

CPU-only coherent ID mutation: `gdn_route_model_mutation_v1.py` changed p64 recovery's final full/generated ID 3927→3926 while retaining producer `passed=true` and `exact_ids_match=true`; only the summary failure count changed. The route auditor exited 1 and recomputed 8/13, rejecting both ID fields against the CPU oracle and rejecting the retained flags. Integrity/source errors remained zero. The provider/route sidecars, event-path metadata, memory receipt and run log were copied. All 950 recorded original capture/helper/source/environment hashes were unchanged. Receipt: `gdn-route-model-1-mutation-receipt-v1.json`; report: `gdn-route-model-1-mutation-id-flip-validation-v1.json`.

Auditor gap and repair: v1 accepted a repaired copy of the route sidecar after qk WG32 and its SPIR-V/LLVM files were omitted, reporting four general programs without a route/integrity/source error (the overall model gate remains the original 9/13). New `gdn_route_model_audit_v2.py` requires the exact multiset `(qk,64),(qk,32),(decay,64),(decay,32),(recur,64)` and `y=z=1`; it rejects that omission with `general route program inventory differs from frozen original13 profile`. V2 audits the untouched capture to the same 9/13 negative with all five programs and zero integrity/source errors. V1 remains unchanged. Receipts: `gdn-route-model-1-validation-v2.json` and `gdn-route-model-1-mutation-omit-qk-wg32-receipt-v1.json`; review diff: `gdn-route-model-audit-v2-review.diff`.

Hardware/workload: guarded Lava run on Ornith-1.5 35B A3B, 13 original two-token requests; memory admission observed 6927 MiB free against the 5500 MiB threshold. This causal diagnostic does not admit source/model behavior, full stress, cost, or throughput. Full source-model stress, documented make/test/format, and cost gates remain open. Source identity is bound to the 916-file candidate; the general recurrence/exp32 helpers match the earlier 418-assertion focused source qualification.

Preflightv1 constructed LavaBackend and initialized Vulkan context, contrary to its printed no-device-initialization claim; it dispatched no model/numerical kernels. Corrected immutable preflightv2 checks lowered method returns/binding without backend construction and passed. The actual model wrapper constructed the backend only after guarded admission under both GPU locks.
