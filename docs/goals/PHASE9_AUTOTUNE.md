# /goal PHASE 9 — AUTOTUNE (one operator, the full loop)

**For:** Buffy (mechanical implementation)
**From:** Grok (encoding owner)
**Canon:** `docs/Gesso_Stack.md` §LXXXII, §XXVI, §LXX, §VII, §CIX, §LXIX
**Map:** `docs/ARCHITECTURE.md`
**Companion (read, do not copy into `src/`):**
`docs/Native_Julia_Kernel_Autotuning_North_Star_README.md`
(especially the autotuner vs foundry split, §26–§27 there).
**Research (read, do not implement beyond this pick):**
KV program §7 (`docs/research/KV_MEMORY_PROGRAM.md`) — this module is a
physical-realization search engine; first client is kernel schedules;
KV pipelines and quantization programs are later clients. Speculative
plans are not a client.
**Depends on:** Phase 8 complete (`docs/goals/PHASE8_LAVA.md`)
**Packets:** 1 and 2 stay closed.

**Status:** LANDED (Buffy, 2026-10-01). Items A–D committed
(`887648c` Autotune skeleton + CUDA `to_device` attach-or-own,
`b70c665` two gated CUDA `matmul!` candidates, `1d74492` op
consults `select` and dispatches by winner name, `4c01994` maps
+ §LXXXII COMPLETE). 1483 pass / 3 named SmolLM2 skips; both
runners EXIT=0; device-less `CUDA_VISIBLE_DEVICES=-1` 1300 pass /
10 Broken (every CUDA-gated set degrades to a named skip). Isolated
item-C driver EXIT=0 (consult 9/9, regimes 16/16, end-to-end 11/11).
toy2 CUDA fingerprint print `0.00037607177004872483`, llama_micro
`5.5006127839263286e-6` (Phase 8 class); greedy ids exact on both
fixtures; host-Array + CUDA backend still `ERR_INVALID_PLAN`. Bench:
14 rows, `micro_llama_cuda_prefill_012_autotuned` winner
`:cublas_mul` (schema 0.2.0, post-warmup, no speed claim). Autotune
imports neither CUDA nor Lava. `libs/` clean. Selection is the
product. Occupancy search, foundry, disk cache, Lava autotune are
later. Next: speed floor (`docs/goals/PHASE10_SPEED_FLOOR.md`). Canon
§LXXXIII representation stays parked. Do **not** edit `libs/Lava`.

---

## Start condition

Phase 8 is on the tree you inherit (`master` at `5e26ffc` plus the
encoding-owner close of `PHASE8_LAVA.md`):

- Lava weakdep + `GessoLavaExt` + six ops on `LavaArray{Float32}`
- interpreter / `Session` / `fork` on Vulkan; ids match CPU
- 1357 pass / 3 named SmolLM2 skips; both runners EXIT=0; format-check OK
- Gauntlet 100% at HEAD, including device-less simulation
  (`VK_ICD_FILENAMES=nonexistent` → 170 pass / 3 Broken; 17/17 device-free
  seam assertions, constructor `ERR_RESOURCE_LIMIT`)
- `micro_llama_lava_prefill_012` in `benchmark/results/2026-10-01.tsv`
- toy2 CPU fingerprint unchanged (`max|logit| = 1686.4814860783201`)
- JSON is the only core third-party hard dep; CUDA and Lava are weakdeps
- `src/Autotune/Autotune.jl` is still the empty contract module
- `src/Planning.jl` / `Runtime.jl` still empty
- CUDA-after-Lava still clobbers `Gesso.to_device` (`GessoCUDAExt.__init__`
  rebinds unconditionally) — item A of this sprint fixes that

If Phase 8 is unfinished, stop.

## One-sentence objective

`matmul!` on CUDA automatically selects a device-specific winning
implementation under a correctness gate, caches the choice, and the
engine uses the winner. No CUDA device ⇒ named skip. CI never requires
a device.

## Why this phase exists

§LXXXII / §XXVI: autotuning is a first-class Gesso concern. Phases 4 and
8 proved two backend seams with one implementation each. A seam with one
candidate cannot select. This sprint fills `Autotune` with the generic
loop — candidate, compile, gate, bench, winner, cache, invalidate — and
wires **one** operator through it so the exit is real:

    at least one operator automatically selects device-specific
    winning implementation.

Kernel-first (§7.2). The interface vocabulary does not say "kernel";
the first client is `matmul!` realizations. Quantization programs, KV
pipelines, and placement wait for Phases 10–11.

## What this sprint is not

- a second CUDA or Lava seam
- handwritten SPIR-V, FlashAttention, fused kernels, tensor-core contests
- the North Star foundry / kernel-family generator (§26 there)
- occupancy *search* (device identity may sit on the cache key)
- KV realization pipelines, Magenta steps 4–7, cages, ExactBits
- speculative plans (Phase 20)
- scheduler / 5b
- filling `Planning.jl` / `Runtime.jl` / `Representation.jl`
- a new core `[deps]` entry (KernelAbstractions, CUDA, Lava stay out of core)
- disk-persisted autotune cache (process-local cache is the artifact;
  serialize later under a version bump)
- claiming speedups vs CUDA.jl, Lava, or llama.cpp
- changing `reference_*` signatures or the toy2 fingerprint
- widening atol past the Phase 4/8 values
- editing `libs/Lava`

---

## Laws (pin this)

### The loop is the product (§XXVI)

```
semantic operation
      ↓
device capability set
      ↓
workload signature
      ↓
candidate generation
      ↓
compile
      ↓
correctness validation
      ↓
benchmark
      ↓
select winner
      ↓
cache result
```

Reproducible, inspectable, invalidatable, numerically gated. A candidate
that fails the gate is **disqualified**. If every candidate fails, throw
a typed `GessoError` — never silently keep a failing implementation, never
fall back to CPU because search missed.

### Autotune is core machinery, backends register candidates

`src/Autotune/Autotune.jl` owns the search/verify/measure/cache/receipt
loop. It does **not** `using CUDA` or `using Lava`. Extensions register
candidates:

```
Autotune.register!(:matmul!, :cuda, candidate)
```

Core tests cover the loop with pure-Julia stand-ins (no device). CUDA
tests cover the real `matmul!` client (skip-or-green).

### Two candidates, or it is not a search

The CUDA `matmul!` path already has one implementation (`mul!` / CUBLAS
via `_cuda_matmul!`). Item B adds a second **legal** candidate of the
same semantics, same `dst, x, w` contract, same F32 device storage.
Suggested pair (you may swap the second if a cleaner one is already in
the ext; packet if you need a new kernel language):

| name | what |
|------|------|
| `:cublas_mul` | existing `_cuda_matmul!` / `mul!` |
| `:generic_mul` | GPUArrays / generic `mul!`/`*` into `dst` on the same `CuArray{Float32}` |

Both must pass the CPU-F64 oracle at the Phase 4 CUDA atol (`1e-2` for
the op-level compare already in `test_cuda_ops.jl` — do not invent a
third atol). The winner is whichever passing candidate has the lower
post-warmup median. If `:cublas_mul` wins, that is the expected and
acceptable result. The point is that **selection happened**.

Do not add KernelAbstractions to core `[deps]` to invent a third
candidate. One KA kernel inside the CUDA ext is allowed only if
broadcast/`mul!` cannot express the second candidate — same rule as
Phase 8.

### Cache key (this sprint's subset of §XXVI)

```
device identity
backend tag          (:cuda)
operator              (:matmul!)
shape regime          (named bucket, not every (M,K,N))
AUTOTUNE_CACHE_VERSION
```

Add `AUTOTUNE_CACHE_VERSION` next to the other schema versions in
`src/versions.jl`. Start at `v"0.1.0"`. Bump it when cache semantics
change. Miss (re-search) when the key differs. `invalidate!` drops an
entry; `invalidate_all!` drops the table.

Shape regimes for this sprint: `:toy2` and `:llama_micro` (the matrices
those fixtures actually multiply). Do not search a cartesian product of
sizes.

### Timing law (same as the bench harness)

Compile and first-touch are outside the timed region. Warmup, then
measure. No timing claims from runs that included compilation. Never
compare kernel-only time to end-to-end Session time in the same
sentence.

### Receipts, not a new result type

Emit through the existing `Receipt` + sink (`context` carries winner
name, per-candidate medians, rejections, cache hit/miss). Do **not**
invent `ExecutionResult`. Do **not** bump `RECEIPT_SCHEMA_VERSION` or
`BENCH_RESULT_SCHEMA_VERSION`.

### Skip law

```
if CUDA is not in the test env or CUDA.functional() is false
    named skip "no CUDA device — Autotune CUDA client skipped"
```

Core Autotune tests (register, search with two Julia candidates, cache
hit, invalidate, all-fail throws) **always run**. They do not need a
device.

### CUDA-after-Lava `to_device` (item A, required)

`GessoLavaExt.__init__` already attaches by name when `to_device`
exists. `GessoCUDAExt.__init__` still does
`const to_device = $to_device`, which clobbers Lava's methods if CUDA
loads second. Mirror the Lava attach-or-own pattern in the CUDA ext.
One function object, both backends' methods. Test both load orders.

---

## Work items (sequence)

### A — CUDA wart + Autotune skeleton

**Objective.** Mixed-backend `to_device` survives either load order.
`Autotune` has types and a loop that works on two pure-Julia candidates.
No device required.

**Permitted files**

```
ext/GessoCUDAExt.jl          # __init__ attach-or-own only
src/Autotune/Autotune.jl
src/versions.jl              # AUTOTUNE_CACHE_VERSION = v"0.1.0"
src/Gesso.jl                 # re-export if you add public names
test/test_autotune.jl
test/test_to_device_mixed.jl # both load orders, skip if an ext is absent
test/runtests.jl
```

**Tests**

- `AUTOTUNE_CACHE_VERSION` is exported and is `v"0.1.0"`
- register two dummy candidates for `:demo, :cpu`; `search!` returns a
  `TuneResult` whose `winner` is one of the two names
- a candidate that fails the oracle is in `rejected` and is never the
  winner
- if both fail, `search!` throws a typed `GessoError` (pick the existing
  code that means "no legal lowering" — do not add a new family)
- second `select` with the same key is a cache hit (same winner, no
  re-bench required)
- `invalidate!` forces a re-search
- load CUDA then Lava: `to_device` has methods for both backends
  (skip if either ext missing)
- load Lava then CUDA: same (this is the wart)
- `libs/` clean

**Artifact.** The loop exists. `matmul!` is not wired yet.

---

### B — Two CUDA `matmul!` candidates + correctness gate

**Objective.** The CUDA ext registers `:cublas_mul` and `:generic_mul`
(or the pair you documented). Both run on `CuArray{Float32}`. Both are
gated against the CPU F64 oracle at the existing CUDA op atol.

**Permitted files**

```
ext/GessoCUDAExt.jl
ext/cuda_ops.jl
src/Autotune/Autotune.jl     # only if the register API needs a hook
test/test_autotune_cuda.jl
test/runtests.jl
```

**Tests**

- no CUDA: named skip
- with device: each candidate's `matmul!` on a toy2-sized and a
  llama_micro-sized pair matches CPU F64 within the existing CUDA atol
- a deliberately-wrong candidate (if you inject one in the test) is
  rejected
- `quantize!` / `dequantize!` still `LoweringNotImplemented`
- CPU and Lava `matmul!` paths unchanged (no Autotune consult)

**Artifact.** Two gated realizations. Selection not yet consulted by
the engine.

---

### C — Select, cache, the engine uses the winner

**Objective.** CUDA `matmul!` (the existing op name, both workloads)
asks Autotune for the winner for the active shape regime and calls
that candidate. Subsequent calls with the same key do not re-search.
One corpus row records the search (schema `0.2.0`, after warmup).

**Permitted files**

```
ext/cuda_ops.jl
src/Autotune/Autotune.jl
test/test_autotune_cuda.jl
test/test_cuda_ops.jl         # only if a consult hook shows up in existing tests
test/runtests.jl
benchmark/runbenchmarks.jl
benchmark/results/            # append-only TSV
```

Prefer **not** to touch interpreter/session. The op method is the
consult site. Packet if you must change `reference_*`.

**Tests**

- with device: first CUDA `matmul!` for `:toy2` produces a cache entry
  whose winner is a registered name; a second call hits the cache
- generate ids on toy2 and llama_micro still match CPU (greedy)
- host `Array` + `backend=CUDABackend()` still `ERR_INVALID_PLAN`
- CUDA fingerprint print stays inside the existing atol
- no device: named skip; CPU tests unchanged
- SmolLM2: skip-or-green

**Bench.** When CUDA.functional(), append **one** corpus row: the
autotune search for llama_micro `matmul!` (or the winning candidate's
post-warmup matmul), schema `0.2.0`. Note names the winner. No "faster
than Lava/CUDA.jl" claim. No device: no new row required.

**Artifact.** The §LXXXII exit: one operator automatically selects a
device-specific winning implementation, and the engine uses it.

---

### D — maps + §LXXXII status

**Permitted files**

```
README.md
docs/ARCHITECTURE.md
docs/Gesso_Stack.md          # §LXXXII status COMPLETE + date
docs/research/ROADMAP_NOW.md # 9 done; speed floor is next (Phase 10 waits)
docs/research/README.md
scripts/freeze.jl
```

**Artifact.** Docs say Autotune runs the §XXVI loop for `matmul!` on
CUDA, caches the winner, and the engine consults it. They do not say
we built a foundry, tuned Lava, or persisted the cache to disk.

---

## Cross-item invariants

- toy2 CPU fingerprint bit-identical
- `reference_*` public signatures unchanged
- CPU and Lava paths unchanged (no Autotune consult on those backends)
- JSON remains the only core third-party hard dep
- Autotune.jl does not import CUDA or Lava
- Profiling still does not import CUDA or Lava
- `libs/` untouched
- no new §CIX family types
- `Runtime` / `Planning` / `Representation` still empty
- formatter skips `libs/`
- no ExecutionResult, no schema bump of receipt/bench versions
- Receipt: what / why / tests / numerical delta (fingerprint inside
  existing CUDA atol; ids exact) / compile-time / memory / hardware
  (CUDA device name) / workload toy2+micro / backend cuda (+ cpu;
  lava still present, unused by Autotune)

## Escalation (packet, then stop)

- you need a new core dependency to express the second candidate
- you need to change `reference_*` or Session to consult Autotune
- both real CUDA candidates fail the existing atol
- you want to autotune Lava, softmax, or the whole engine this sprint
- you want a disk cache, occupancy search, or the North Star foundry
- you want this sprint to start Phase 10 representation

## Exit checklist

- [x] A, B, C landed
- [x] D maps truthful (ROADMAP/research README: speed floor next; ARCHITECTURE Phase-10 line corrected in this close)
- [x] `make test` green without a CUDA device (named skips; `CUDA_VISIBLE_DEVICES=-1` → 1300 / 10 Broken)
- [x] `make format` run
- [x] CPU fingerprint unchanged
- [x] Lava tests still skip-or-green
- [x] CUDA `matmul!` consults Autotune and uses the cached winner
- [x] generate ids still match CPU on toy2 + llama_micro when a device exists
- [x] mixed-load `to_device` keeps both backends
- [x] no silent fallback (`ERR_VERIFY_MISMATCH` on all-fail / stale winner)
- [x] `libs/` clean
- [x] no Planning/Representation fill, no foundry, no cages
- [x] receipt (below)

---

## Foreman close (encoding owner, 2026-10-01)

Buffy implemented A–D on `master` at `4c01994`. Item D already flipped
canon §LXXXII to COMPLETE. This file was the open recipe (untracked,
checkboxes empty); this close is the encoding-owner receipt. Tests were
not re-run for this close — gate counts are Buffy's commit-body / wrap
receipts plus the TSV row on disk.

### §LXXII receipt

| field | value |
|---|---|
| what changed | Autotune loop in core; CUDA `matmul!` registers `:cublas_mul` + `:generic_mul`; op consults `select` and dispatches by winner name; mixed-load `to_device` attach-or-own |
| why | §LXXXII exit: one operator automatically selects a device-specific winning implementation |
| tests | 1483 pass / 3 named SmolLM2 Broken; bounds runner 1483/3; device-less 1300/10; item-C driver consult 9/9, regimes 16/16, e2e 11/11 |
| numerical delta | toy2 CUDA print 0.00037607177004872483; llama_micro 5.5006127839263286e-6; greedy ids exact; `:cublas_mul` wins as designed |
| before bench | 13 corpus rows (Phase 8 lava row included) |
| after bench | 14 rows; `micro_llama_cuda_prefill_012_autotuned` median 1.0508421e7 ns, 417 samples, winner `:cublas_mul`, commit stamp `b70c665`, `dirty=true` |
| compile-time impact | Autotune registry is runtime `__init__` (never precompile); CUDA ext still loads CUDA |
| memory impact | process-local cache only; no disk cache |
| hardware | CUDA.name(device) on the cache key (RTX 5060 class from prior phases) |
| workload | toy2 + llama_micro; unknown (K,N) attributed to `:toy2` |
| model | toy2, llama_micro |
| backend | cuda (cpu/lava paths unchanged, unconsulted) |

### Known limits (documented, in-scope)

- `:generic_mul` is the deliberate loser on real shapes. Selection happening is the product.
- Cache is process-local.
- Unknown (K,N) → `:toy2` (conservative smaller bucket, not a backend fallback).
- CPU and Lava `matmul!` do not consult Autotune.

### Map split, closed here

Item D left ARCHITECTURE saying "next open recipe: Phase 10" while
ROADMAP/research README already pointed at `SPEED_FLOOR.md`. Encoding
owner aligns ARCHITECTURE with the speed-floor gate. Parallel
workstream files (`SPEED_FLOOR.md`, Phase 5–8 goal closes, RPD_SOP,
ancient texts) stay unstaged.

No escalation packets. Next recipe is the speed floor, not Phase 10.
