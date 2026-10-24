# Campaign 1: qualified NoContraction supplier

Standalone Lava commit `289d0693e84a3ad6ae6869cbf81b47dd4bc20ba7` on `codex/no-contraction-format-20261018`
is clean and local. No Gesso source, vendor or installed runtime was changed.
Campaign 1 remains active: the frozen Gesso source is ef086ef and its model gate
remains **39/41**. This receipt closes the isolated compiler-contract work item,
not model admission or production release.

## Change and evidence

The opt-in `Lava.with_no_contraction` API carries a compile-time Boolean through
the existing FloatControls carrier, Adapt, compiler configuration and cache
identities. It decorates emitted core add/subtract/multiply/divide results
before optimization, checks coverage afterward, and leaves explicit Fma intact.
Required shader artifacts receive fail-closed structural validation before
cache trust or linking. Invalid memory entries are evicted; bad disk entries
miss; invalid stores fail. Default behavior remains unchanged.

| Requirement | Evidence |
| --- | --- |
| Public API, composition, Adapt, optimized shader coverage, actual GPU witness, malformed cache rejection/recovery and integer-only positive | 141/141 maintained assertions on the formatted source, exit 0 |
| Prior FloatControls | 116/116, strict device checks, no skips on the structural predecessor |
| Direct compiler interfaces and configured-job path | 11/11 CPU assertions; no Vulkan context |
| Default shader bytes | Formatted source equals parent exactly; SHA256 `8166451f23abb376be09e6a8ba1cf9e6100f6cf3d9a86494da3a506184997ef5` |
| Q4_K through the actual supplier API | 42/42 exact repetitions, including the full 248,320-output head; independent native/raw/SPIR-V audit and effective mutation rejection |
| Q8_0 through the actual supplier API | 45/45 exact repetitions across 15 frozen cases; independent native pack/CPU output/LLVM/SPIR-V audit and three-repetition mutation rejection |
| Cost capture | All 110 timed outputs independently exact-rational audited; all eight saved SPIR-V modules and actual serialized cache entries verified; 50-output repaired-hash mutation rejected |
| Scoped formatting | JuliaFormatter 2.14.0; eight complete code files have identical parsed expressions after removing line numbers |

The quantized, FloatControls, direct-entrypoint and cost captures ran against
the preserved structural predecessor. The formatted source has the same eight
complete code ASTs, an updated explanatory Markdown contract, a fresh 141/141
GPU run and fresh parent-default-byte identity. This is an explicit preservation
chain; no post-format quantized rerun is claimed. All original captures, failed
helper/auditor versions and corrected successors remain available.

## Numerical, compile and memory measurements

For the finite Float32 contraction witness, separately rounded arithmetic
returns +0 while intentional Fma returns -2^-46. Quantized fixtures match their
unchanged CPU/native references bitwise; no accuracy target or tolerance changed.
The original full-model gate was not rerun and no model repair is inferred.

On RTX 5060, driver 615.78.08, Julia 1.12.6, the 65,536-element cost workload
used at least 25 checked warmups and one second per route, then 50 paired
measurements each. Median call-plus-sync was 0.443244 ms default and 0.440966 ms
opt-in, with 8,288 host-allocated bytes each. Variance does not support a speedup
claim. Direct compiler samples were 62/70 ms default versus 93/86 ms opt-in.
Required shader validation adds observable cold/cache-boundary cost: the one
forced frozen-memory sample was 0.435 ms default versus 8.454 ms opt-in; those
single forced samples are not a distribution or inference benchmark. Frozen
recording was layered over the ordinary serialized-SPIR-V cache, with driver
pipeline caches retained.

Tracked live GPU buffers stayed at 71,565,312 bytes after the first default
dispatch and after adding the opt-in route; cached pipeline count rose from one
to two. Driver pipeline/code memory is not included. Peak whole-process RSS
was 1,399,584 KiB. Raw cost captures occupy 30,146,560 bytes under the 64 MiB cap.

## Limits and next gate

Physical cross-vendor/lavapipe numerical execution is not established. These
finite quantized fixtures and arithmetic witnesses do not establish universal
division/transcendental accuracy or model parity. Full inference throughput
testing remains deferred as requested.

Machine-readable receipt under the campaign result root:
`no-contraction-supplier-qualification-20261018.json`. It binds exact source
files, proof hashes, test scopes and preserved failures. Next: a **new Gesso
candidate**, explicit work-item scope, sufficient numerical repair, the unchanged
41-case model gate, then all original final-source stress/recovery/integration
gates. No push, merge or deployment occurred.
