# RPDO spiral 4 — Stable Lava decode lifecycle

Owner: Grok. Repo: Gesso. Layer: Gesso mechanism. Date: 2026-10-08.

Problem: spiral 3 flattened warmed decode alloc at 3.28 MB and held
live median P=1.112, but the first `decode!` after `prefill!` still
looked like ~50 MB of compilation. This spiral asked why, whether
compiled specializations survive reset/prefill/seqlen, and what (if
anything) of the remaining 3.28 MB Gesso owns.

## Load-bearing questions

| Q | Cheap experiment | Disposition |
| --- | --- | --- |
| Q1. Why does prefill make the next decode compile? | World counter, Lava pipeline/kernel counts, `@allocated` vs a plain wrapper around shipped `prefill!`/`decode!`/`generate`/`_session_reset!`. | `EXPLAIN`. Prefill does not bump world or kernel counts. The ~50 MB is a fresh Julia 1.12 `@allocated` call site (`Base.allocated` + `@force_compile`) invalidating GPUCompiler's world-keyed cache (~24 decode kernels, 47–51 MB). After one cold `generate`, a later `decode!` / `generate` does not recompile. Receipts `s4-q1-lifecycle.json`, `s4-q1-noalloc.json`. |
| Q1b. `FROZEN_VERSION`? | Same-session `with_frozen_recording("gesso-s4-v1")` then `use_frozen_kernels`; second process disk-only. | `REVERT` for `__init__`. Same-session: 38 hits, kernels stay 40 across a world bump, first `@allocated` decode 7.13 MB not 50 MB (`s4-frozen-session.json`). Disk: 0 hits / 38 misses, first-use still 62.5 s. `module_build_id` of Lava and GessoLavaExt changes every process (neither precompiles — Lava `adapt_structure` overwrite). Recording would write 38 orphan files per session. `use_frozen_kernels` without recording is a miss path and still recompiles. Receipt `s4-frozen-disk.json`. |
| Q2. Do pinned-wg decode kernels reuse across seqlen, reset, repeated generate? | Alloc ladder + kernel/pipeline counts on the named lifecycle. | `HOLD`. Spiral 3 `(64,)` / `(64, 1)` already reuses. After cold generate: world and `kernels_cached=40` stay flat across two more generates, `_session_reset!`, and `prefill!`. A plain `decode!` after that prefill does not add kernels. Seqlen 2–7 on Hello does not grow pipelines. A longer fox prompt adds ndranges (new IterPlans), not new LocalSize. |
| Q3. What is the remaining 3.28 MB? | `Profile.Allocs` sample_rate=1.0 on one warmed Hello `decode!` (no `GC.gc(true)` before it). | `EXPLAIN` + `REVERT` the Gesso-side copy kernels. Profile total **3.11 MB / 40 817 allocs** (`s4-alloc-profile.json`). Named sites: `_lava_matmul!` 728 KB (Lava `mul!`, `libs/` closed), `_split_heads!` 281 KB, generic `_copy_rows_storage!` 263 KB + `_copy_row_storage!` 245 KB, `_lava_swiglu!` 232 KB, rms/repeat/rope/add/attn launches 90–204 KB each. Lava 10F-style copy kernels cut those two copy sites to 217+187 KB (net **−104 KB**, 3.00 MB) and **dropped fox 37.35 → 30.27 tok/s**. Reverted. Receipts `s4-alloc-profile-copy.json`, `s4-copy-lava-three.json`. |
| Q4. DecodePlan? | Only if Q1–Q3 show existing public cache/launch APIs cannot move the leftover. | `PARK`. Q1 is a harness world-age artifact. Q2 already reuses compiled SPIR-V. Q3 leftover is Lava launch metadata (push constants, KA Kernel construction, ndrange tuples) plus GPUArrays broadcast. A Gesso DecodePlan that interned Kernel objects would not remove `launch.jl` bytes and a type-unstable cache would risk tok/s. Extra Gesso launches to replace broadcast already failed the fox canary. |

## Laws in force

RPDO: candidate → oracle (greedy IDs exact, planted NaN still
`ERR_NUMERICAL_INSTABILITY`) → measure post-warmup → retain or revert.

Caps: no new modules, no new deps, no `libs/` edits, no Julia compiler
fork, no Magenta, no GGUF/FP16. Palette / Cyan / Luxel frozen. Spiral 3
pinned workgroup geometry stays. Do not commit unless asked.

## Baseline (spiral 3 retained)

Receipts `s3-wg64-lava-three.json` / `s3-cuda-three.json`.

| Prompt | Lava tok/s | CUDA live | P live | Lava alloc |
| --- | ---: | ---: | ---: | ---: |
| Hello | 37.54 | 34.55 | 1.087 | 3.28 MB |
| The quick brown fox | 37.35 | 33.58 | 1.112 | 3.28 MB |
| Julia is a programming language. | 37.29 | 30.62 | 1.218 | 3.28 MB |

Median live P = 1.112. First decode after prefill ~50 MB (then believed
to be prefill world-age).

## Q1 evidence

`s4-q1-noalloc.json` (probe does not JSON-snap during generate):

| Step | world | kernels | note |
| --- | ---: | ---: | --- |
| after cold generate | 45721 | 40 | first-use compile |
| after 2 more generate | 45721 | 40 | no recompile |
| after reset + prefill | 45721 | 40 | **prefill is innocent** |
| plain `decode!` (no `gc_bytes`) | 45721 | 40 | shipped first decode after prefill does not compile |
| first `gc_bytes`/`@allocated` decode | 45722 | 64 (+24) | 47.0 MB |
| decode 2–6 | 45722 | 64 | 3.28 MB, flat with K |
| a *new* `@allocated` call site | 45726 | 102 (+38) | 47.0 MB again |

Julia 1.12 `@allocated f(x)` becomes `Base.allocated(f, x)` with
`Experimental.@force_compile`. A new measurer call site defines
methods, bumps world, and GPUCompiler misses the decode kernels.
The performance harness measures 3.28 MB on the 2nd decode because
that `@allocated Gesso.decode!(s)` line is compiled once up front
(or after the 6 generates have already compiled at the bumped
world). `generate()` host ~50 MB is 8×3.28 MB plus prefill/reset,
not a 50 MB first-token compile inside a warmed generate.

## Candidate 1 (REVERT): `FROZEN_VERSION` in GessoLavaExt `__init__`

Public Lava API. `frozen_load` keys on `(typeof(f), tt, workgroup_size)`
with no world, so it can survive a measurer bump.

Same-session recording around cold generate (`s4-frozen-session.json`):
38 stores, then 38 hits, `kernels_cached` stays 40 through `@allocated`
decode 1–6 and a second measurer site. Decode-1 alloc 7.13 MB (the
measurer itself plus launch), decode-2+ 3.28 MB. Hello IDs exact.

Second process, `use_frozen_kernels` only (`s4-frozen-disk.json`):
ondisk=38, hits=0, misses=38, first-use 62.5 s, Lava build_id
`…10d89e5a3c22d2a8` → `…2e2b53fee18b80df`, GessoLavaExt
`…11572542902c0865` → `…cb666c604e8b8581`. Disk keys include
`module_build_id`; neither module has a `.ji`.

Do not enable in `__init__`. Same-session recording writes 38 files
that the next process cannot read. `use_frozen_kernels` without
recording is a miss. The shipped `generate`/`decode!` path after one
cold generate already does not recompile.

## Candidate 2 (REVERT): Lava `_copy_row_storage!` / `_copy_rows_storage!` kernels

Profile named 508 KB on the generic `@views .=` seams CUDA 10F already
specialized. Gesso methods + KA kernels, pinned `(64, 1)`, bound-checked
extra lanes. Hello IDs exact, planted NaN true (`s4-copy-hello.json`).

Alloc: 3.11 → 3.00 MB (−104 KB). The copies were already mostly launch
metadata; replacing broadcast with a KA launch saved little.

Three-prompt (`s4-copy-lava-three.json`): Hello 36.87, **fox 30.27**,
Julia 38.70. Fox −19% vs spiral 3 37.35 — the same canary `(8, 8)`
hit in spiral 3. Extra ~150 KA launches/token. Reverted. Post-revert
restored Hello 37.43 / fox 36.94 / Julia 36.78 (`s4-lava-three.json`).

## Measure (no runtime retain)

Receipts `s4-lava-three.json` / `s4-cuda-three.json` / `s4-hello-attr.json`.
27/27 greedy IDs exact. Planted decode NaN is `ERR_NUMERICAL_INSTABILITY`.
`test/test_numeric_lava.jl` was not edited; the SmolLM2 decode planted
case in `regime_ii_attribution.jl` is the oracle used here.

| Prompt | Lava tok/s | CUDA live | P live | Lava alloc | CUDA alloc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hello | 37.43 | 31.38 | **1.193** | 3.28 MB | 1.06 MB |
| The quick brown fox | 36.94 | 28.44 | **1.299** | 3.28 MB | 1.06 MB |
| Julia is a programming language. | 36.78 | 30.10 | **1.222** | 3.28 MB | 1.06 MB |

Median live P = **1.222**. All three P>1.

Lava vs spiral-3 Lava: Hello 37.43 vs 37.54 (−0.3%), fox 36.94 vs
37.35 (−1.1%), Julia 36.78 vs 37.29 (−1.4%). Within night-to-night
noise of the retained pin. No Lava tok/s regression to report against
the spiral-3 Lava baseline.

Live CUDA on this box was slower than spiral 3's CUDA (Hello 31.38 vs
34.55, fox 28.44 vs 33.58). That raises P; it is CUDA variance, not a
Lava gain. Against spiral-3 CUDA the implied P would be ~1.08 / 1.10 /
1.20, still all P>1.

Warmed `@allocated decode!` in the performance harness: **3.28 MB**
on all three prompts, unchanged. First-use Hello generate: 57.0 s
(spiral 3 was 57.4 s).

## Closed

Q1–Q4 have dispositions. No Gesso runtime change retained. Spiral 3
pinned workgroupsize stays. `FROZEN_VERSION` stays off. Copy kernels
reverted. DecodePlan not prototyped. Compiler fork stayed `PARK`.
Orphan `gesso-s4-v1` frozen files under
`~/.julia/scratchspaces/lava_frozen_kernels` were deleted after the
disk probe.

## Leftover (next spiral)

Warmed decode 2+ is still **3.28 MB vs CUDA 1.06 MB**, now attributed:
Lava `mul!` launch (~728 KB) plus KA launch metadata on split / swiglu
broadcast / generic KV copies / rms / rope / attn. Gesso cannot cut
this with more kernels without repeating the fox dispatch regression.
The next intervention that could move it is **inside Lava** (launch /
command-buffer reuse, or making Lava precompile so `FROZEN_VERSION`
disk keys survive). `libs/` stays closed, so that is a Lava-side
experiment, not a Gesso DecodePlan.

First-use compile (~57 s, ~40 kernels) is a process-once cost. Disk
frozen cache cannot amortize it while GessoLavaExt and Lava lack `.ji`
build ids.

`generate()` host ~50 MB remains 8× the warmed decode tax plus
prefill/reset. It is not a hidden 50 MB compile after warmup.
