# RPDO spiral 3 — Lava host alloc leftover

Owner: Grok. Repo: Gesso. Layer: Gesso mechanism. Date: 2026-10-08.

Problem: spiral 2 retained batched decode attention and closed the
live-fox CUDA gap (median live P=1.095). Warmed `decode!` host alloc
grew with K: **3.45 / 3.71 / 3.94 MB** vs CUDA **1.06 MB**. Device
argmax was reverted. Compiler fork stayed parked until a named receipt.

## Load-bearing questions

| Q | Cheap experiment | Disposition |
| --- | --- | --- |
| Q1. Where are the 3.4 MB? | `Profile.Allocs` + alloc ladder on Hello. | `BUILD` pin workgroupsize. First decode after prefill is **50 MB of SPIR-V compile** (`lava_disk_cache_key` 12 MB, `compilation.jl` 7.5 MB). Default KA wg is `min(prod(ndrange), 64)`, so batched scores `(H,K)` and GQA repeat `(K, n_kv)` compiled a new LocalSize every token until 9K≥64. Receipts `s3-alloc-hello.json`, `s3-alloc-ladder.json`. |
| Q2. Do remaining generic Lava storage seams (KV copy, hidden-row, swiglu, embed) own the 3.4 MB? | Only if Q1 was GPUArrays broadcast. | `PARK`. Q1 was compiler/launch, not 10F-class wrappers. |
| Q3. Does cutting those bytes move tok/s, or only `@allocated`? | Three-prompt generate after the pin. | Alloc flattened at **3.28 MB** on all three prompts (was growing with K). Decode tok/s held / slightly up. Live median P **1.112**. |
| Q4. Compiler fork? | Only if the leftover is SPIR-V quality, not Gesso launch shape. | `PARK`. The limiter is Lava LocalSize + iterplan keyed on ndrange **value**, plus prefill world-age invalidating decode pipelines. `libs/` stays closed. |

## Laws in force

RPDO: candidate → oracle (greedy IDs exact, planted NaN still
`ERR_NUMERICAL_INSTABILITY`) → measure post-warmup → retain or revert.

Caps: no new modules, no new deps, no `libs/` edits, no Julia compiler
fork, no Magenta, no GGUF/FP16. Palette / Cyan / Luxel frozen. Do not
commit unless asked.

## Baseline (spiral 2 retained)

Receipts `s2-attn-lava-three.json` / `s2-cuda-three.json`.

| Prompt | Lava tok/s | CUDA live | P live | Lava alloc |
| --- | ---: | ---: | ---: | ---: |
| Hello | 36.48 | 32.61 | 1.119 | 3.45 MB |
| The quick brown fox | 36.70 | 35.15 | 1.044 | 3.71 MB |
| Julia is a programming language. | 38.68 | 35.31 | 1.095 | 3.94 MB |

## Q1 evidence

Hello prompt_len=1. After a warmed generate, reset+prefill, then:

| decode | seqlen | alloc before pin | alloc after pin `(64,1)` |
| --- | ---: | ---: | ---: |
| 1 | 2 | 53.8 MB | 50.9 MB |
| 2 | 3 | 7.19 MB | **3.28 MB** |
| 3 | 4 | 7.29 MB | **3.28 MB** |
| 4 | 5 | 7.80 MB | **3.28 MB** |
| 5–6 | 6–7 | 7.45–7.55 MB | **3.28 MB** |

Decode 1 after prefill is still a full SPIR-V walk (prefill bumps
world age; GPUCompiler caches key on world). Decode 2+ stopped
growing with K once LocalSize stopped tracking K.

`pipelines_cached` after generate: 52 before pin, **37** after.

## Candidate 1 (RETAIN): pin KA `workgroupsize`

Vulkan LocalSize is compile-time. Pin `_LAVA_WG1 = (64,)` and
`_LAVA_WG2 = (64, 1)` on every `lava_ops.jl` launch. `(64, 1)` is
Lava's own default once `prod(ndrange) ≥ 64`, so large-K occupancy
matches spiral 2. Small K shares that pipeline.

Split / merge / repeat gained the missing axis bound-check so extra
lanes in a 64-thread group cannot write out of range.

`(8, 8)` tiling was measured and **reverted** (`s3-wg-lava-three.json`):
fox decode 36.70 → 30.43 tok/s. Occupancy change, not an ID miss
(27/27). `(64, 1)` restored fox.

Oracle: Hello IDs exact, planted NaN true (`s3-wg-hello.json`, on
the `(8, 8)` kernel bodies). Three-prompt IDs 27/27 on `(64, 1)`
(`s3-wg64-lava-three.json`).

## Measure (retained `(64, 1)`)

Receipts `s3-wg64-lava-three.json` / `s3-cuda-three.json` (same box,
same night).

| Prompt | Lava | CUDA live | P live | Lava alloc | CUDA alloc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hello | 37.54 | 34.55 | **1.087** | 3.28 MB | 1.06 MB |
| The quick brown fox | 37.35 | 33.58 | **1.112** | 3.28 MB | 1.06 MB |
| Julia is a programming language. | 37.29 | 30.62 | **1.218** | 3.28 MB | 1.06 MB |

Median live P = **1.112**. All three P>1. IDs exact (27/27).
Alloc no longer grows with prompt / K.

## Closed

Q1–Q4 have dispositions. Pinned workgroupsize retained. 10F Lava
copy/swiglu/embed seams stayed parked. Compiler fork stayed `PARK`.

## Leftover (next spiral)

Warmed decode 2+ is **3.28 MB vs CUDA 1.06 MB**. First decode after
prefill is still **~50 MB** of Lava compile (world-age). `generate()`
host alloc stays ~50 MB because it includes that first token.

Gesso-side levers still open: Lava `FROZEN_VERSION` from the
extension (public API, not a `libs/` edit), fewer launches, KA
Kernel object reuse. IterPlan is keyed on the ndrange **value**
inside Lava; that cache is not ours to change.
