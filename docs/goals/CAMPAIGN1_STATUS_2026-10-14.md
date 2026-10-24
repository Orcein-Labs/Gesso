# Campaign 1 checkpoint — 2026-10-14

**Open. Shared-store diagnostics are promising; source stress acceptance is pending.**
The alignment change below fixes a confirmed Vulkan conformance defect, but
fresh-process testing rejected it as a sufficient L1 repair. No release,
throughput-parity, complete stress, or Campaign 1 completion claim is made.

## Source state

| Tree | Identity | State |
|---|---|---|
| Original Lava diagnosis | `7a3a58a`, `lava-precompile-20261013` | Preserved clean baseline |
| Isolated Lava conformance repair | `eef73a9`, `lava-coopmat-alignment-20261014` | Two physical coopmat access operands plus regression/registration |
| Isolated accumulator handoff | `a6a2844`, `lava-accumulator-handoff-20261014` | Experimental shared-store candidate; 460 numerical/capability assertions and 12 subgroup checks passed; source model gate pending |
| Combined Gesso draft | W3/S1 plus the three-patch Lava series, `gesso-campaign1-20261014` | Shared-store candidate integrated; prior alignment-only generation red; new source model gate pending |
| Inherited L1 diagnosis | `37d87da`, `gesso-l1-20261013` | Two inherited probes preserved; excluded from combined draft |

The vendor script replayed the three-patch series from upstream
`11c7e31bdf62408d22bf379e9e59510f69d2103e`. All 274 vendored files match the
standalone `a6a2844` checkout by inventory and SHA256, recorded in
`vendor-accumulator-manifest.json`. Local commits only;
no production checkout merge, push, deployment, or user configuration change.

## What the failure evidence now proves

The first real GDN block followed by its shared projections reproduces the
failure without CPU experts, routing, attention blocks, or later model layers.
A gate-projection capture uses correct finite Float16 input and weight buffers.
Before dispatch its partial buffer was filled with marker `0x7fc1dead`.
After failure, exactly 131,072 of 262,144 entries still contained that marker:
rows 4–7 modulo 8, across every column and all eight partial planes.

An independent GPU integer/atomic counter confirmed the same 131,072 entries;
its synthetic positive control returned `[32,32,0,64]`, and its actual coverage
counter returned 262,144. This establishes incomplete writes in device memory,
not merely a host readback artifact. Some written entries are also wrong,
including zero-extended Float16 input-fragment words. It does **not** yet
identify the responsible source or driver behavior.

An offline audit found unique exact contiguous input matches for 26,624 of
32,768 written four-row low16 sequences, all within the corresponding partial
plane's K slab, and none matching weights. Another 8,192 written words equal
addresses in the captured input allocation. The receipt
`block1-gate-poison-3-offline-audit.md` records hashes and interpretation limits;
these are fingerprints, not a complete fragment layout or mechanism.

An earlier down-projection capture independently showed correct operands,
incorrect producer partials, and a reducer faithfully summing those partials.
The shader binaries from healthy and failing processes were identical, including
the subsequently aligned shaders. Source inspection and validator success alone
have therefore not explained the runtime failure.

## Conformance repair and regression

Vulkan requires an `Aligned` memory operand on physical-pointer memory accesses.
The original global cooperative loads/stores omitted it; the isolated repair
adds the scalar component alignment, preserving Workgroup operands and GEMM
arithmetic. See [Vulkan VUID 04708/06314](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-StandaloneSpirv-PhysicalStorageBuffer64-04708).
Actual target pointer/stride alignment also satisfies the separate runtime
16-byte requirement. This is not a universal allocator-alignment claim.

The SPIRV-Tools 2026.4 cooperative instruction validator skips its absent-memory-
operand check, explaining why old modules validated. The new compiler regression
fails all four old Float16/Float32 load/store checks and passes 18 assertions
against the repaired source. Package precompilation took approximately 16 s;
the compiler test took 46.6 s, versus 47.9 s for the old failing test. These are
cold diagnostic measurements, not a steady-performance comparison.

## Result ledger

Raw logs, source metadata, captures and helpers are under
`<pystack-work>/{results,scripts}/campaign1-20261014`.
`results/.../progress.md` retains the full sequence and failed setup attempts.

| Check | Evidence | Result and limit |
|---|---|---|
| Alignment matched first-block test | `block1-alignment-crossover-2.log` | Stock failed pair 17; aligned arm passed 17. Initial promising result, insufficient across processes |
| Alignment-only fresh processes | `aligned-full-{1,2,3}.log` | Six correct calls across two processes, then three nonfinite failures in process 3. Remaining queued repeats cancelled before launch |
| Combined default generation | `candidate-full-1.log` | All three p64 calls failed; actual Gesso and vendored Lava paths asserted |
| Empty private Vulkan cache | `private-cache-full-1.log` | All three p64 calls failed with a new empty application cache. Old application-cache seeding is not required for the fault; driver-internal caches were not controlled |
| W3 CPU subset | `w3-candidate-green-1.log` | 493 passes; two optional Saluki/Kolibri real-model row checks skipped |
| W3 last-column mutation | `w3-candidate-red-1.log` | Nine failures; mutated helper executed 80,218 times. Strict nonzero exit |
| S1 suite | `s1-candidate-green-1.log` | 328 passes |
| S1 tie-order mutation | `s1-candidate-red-1.log` | 24 failures; mutated helper executed 1,074 times. Strict nonzero exit |
| S1 target vocabulary | `s1-candidate-vocab-1.log` | 1,400 exact token comparisons and 700 matching final RNG states, vocabulary 248,320; CPU sampler entry point only |
| No entry bounds guard | `block1-guard-crossover-1.log` | Variant failed pair 31 while stock was healthy; rejected. The intended guard removal was verified in emitted code |
| Fully unrolled K | `block1-full-k-crossover-1.log` | Both arms passed all 48 pairs; inconclusive. Variant has 64 MulAdds and zero cooperative matrix phis in each target shader |
| Fully unrolled K, fresh combined tree | `block1-full-k-candidate-1.log` | Both arms healthy for 36 pairs, both gate/up outputs wrong at pair 37. Variant still has 64 MulAdds, zero cooperative phis, and aligned accesses. Rejected as sufficient repair |
| Alternative matrix-zero encoding | `zero-composite-full-1.log` | All three p64 requests failed. All five cooperative shaders use one-constituent zero composites instead of matrix ConstantNull and validate. Rejected as sufficient repair |
| Pipeline optimization flag control | `pipeline-noopt-full-1.log` | All three p64 requests failed; receipt confirms five cooperative pipelines created with DISABLE_OPTIMIZATION, no retries. The flag is a request, not proof of every native optimization choice |
| Explicit stronger alignment | `aligned16-full-1.log` | All three p64 requests failed; stronger alignment is not a sufficient repair |
| Final-MMA workgroup barrier | `single-mma-barrier-full-{1,2}.log` | First fresh process passed three requests; second failed all three. Rejected as sufficient repair |
| Shared store, paired first-block control | `block1-shared-store-tiled-5.log` | Stock failed twice; shared-store arm passed all 128 pairs. No extra stock-failure capture/readback branch in this repeat |
| Shared store, diagnostic full model | `shared-store-full-{1,2,3}.log` | Nine correct 82202 results in three fresh processes; 780 guarded direct routes per process. Process-local variants, not source acceptance |
| Numerical source contract | `accumulator-contract-{old,source}-1.log` | Each passed 441 assertions / 60 actual-device cases; old small cases do not reproduce the model fault. Source test predates capability-gate additions |
| Source numerical/capability contract | `accumulator-contract-source-2.log` | 460 assertions passed, including exact no-feature strided GEMM and emitted shader capability absence |
| Source subgroup regression | `accumulator-subgroup-source-2.log` | 12 assertions passed; first launch failed on an undeclared test import, corrected to use Lava's existing KernelAbstractions binding |
| Published strict stress driver | `shipping-stress-selfcheck-1.jsonl` | Positive/negative ID and callback gates passed; model, boundary and recovery roles still pending. Usage is in `CAMPAIGN1_STRESS_USAGE.md` |
| Device marker confirmation | `block1-gate-poison-3.log` | Counter positive control passed; host and device agree on half the partials remaining untouched |

The sampler's flat-tie allocation case is adverse: optimized 11,450,880 bytes,
old full sort 4,966,616 bytes. No universal speed or memory improvement is claimed.
No top-k coverage is claimed: the current Sampler API has no top-k parameter.
The auxiliary marker counter originally used a floating `isfinite` test that
failed its own synthetic control; the successful counter uses integer exponent
bits. The failed auxiliary attempt is retained and is not device-state proof.
Stock compilation writes numbered IR/SPIR-V files into `vendor/tmp_kernels`;
the generated untracked directory is retained. Earlier diagnostic descriptions
of no per-compile dumping mean no additional helper-installed dump observer,
not an absence of stock compiler disk writes.

## Environment and remaining gates

Julia 1.12.6; NVIDIA GeForce RTX 5060, driver 615.78.08; at most 16 Julia threads,
OpenBLAS/OMP one thread, 4,800 MiB declared GPU budget. Heavy jobs use the handoff
locks and bounded job runner. The cached and user Ornith GGUF files are both
12,992,461,216 bytes and byte-identical SHA256
`9cae3b59a34308664b53e5b80fd74914cd0c1cf504bcd05609498b0f9126a74f`.

L1 repair remains the first acceptance gate. Still required after a sufficient
repair: five fresh default processes with correct repeated p64 calls; the
64-request/16-process stress target; CPU oracles around prompt/chunk boundaries;
state reuse, collection, cancellation/resume, callback exception and new-state
recovery; overlap/multi controls; actual-device Lava and Session subsets; final
combined-tree local suite and scoped formatting; and valid warmed before/after
runtime and memory receipts. The successful independent W3/S1 checks are
preparation and do not graduate Campaign 1A or bypass these gates.

The inherited 507/507 was a historical aggregate from an eight-file Gesso Lava
subset plus fixture prerequisites, not today's expected assertion count. The
current device selection must also include `test_session_lava.jl`; report actual
per-file totals and skips. See `lava-subset-reconciliation.md`. The active
`env-campaign1` Manifest and entry-point assertions select the combined trees,
and its Project `[sources]` entries have now been reconciled to those same paths.
The Manifest is unchanged; `env-source-reconcile.json` records both identities.
No package resolution was performed during these probes.
