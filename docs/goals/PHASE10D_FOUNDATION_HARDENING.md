# /goal PHASE 10D — FOUNDATION HARDENING (audit + RPDO tighten)

**For:** Buffy (mechanical implementation). **Gauntlet this.**
**From:** Grok (encoding owner)
**Depends on:** 10C LANDED on `origin/master` (`2d5111a` class; G2 TSV
`28c9e3f`). Demo-box snapshot still gitignored.
**Canon §LXXXIII stays PARKED.** Do not fill `Representation.jl`,
`Planning.jl`, `Runtime.jl`, `Agents.jl`, `CAPI.jl`, or `Lowering.jl`.
Do not fuse decode. Do not edit `libs/Lava`. Do not fork Julia. Do
not add a Project.toml dep (no JET, SnoopCompile, TimerOutputs,
BenchmarkTools-in-core, torch). Packets 1 and 2 stay closed.
**Mode:** lab MVP is on the board. Slow down. Bulletproof the
foundation. Demo-shaped, not vision-shaped.

**Status:** COMPLETE (2026-10-02, gauntlet close — receipt below)

---

## One-sentence objective

Every public name is pinned to a test or a parked-empty bucket, every
fail-closed hole we can find is closed with a test, the hot path is
idiomatic enough to `@inferred` on the fixtures, and at most five
RPDO-retained mechanical wins land — ids, fingerprints, and fork
bytes unchanged.

## Why this sprint exists

Phases 0–10C built a named-model engine (G1 green, G2 **0.692×**,
G3 share on micro and SmolLM2). The next *product* lever is fused
decode. Before that, the body has to be lean and boring: coverage
holes, skip-shaped fails, unidiomatic Julia, type instability, and
alloc noise will poison the fusion receipt.

This is an **RPDO spiral on the existing machine**, not a new
machine. Propose → test → oracle gate → optional bench → retain or
revert. Taste is not a lowering.

## Start condition

HEAD is 10C + G2 TSV on `origin/master`. Mixed local dirt
(untracked PHASE5/7/8/9, research programs, `RPD_SOP.md`, ancient
texts, `1x`, extra `2026-10-01.tsv` rows, `AGENTS.md` tweaks) is
**encoding-owner freeze work. Do not touch it. Do not commit it.**

Demo box:

```
GESSO_SMOLLM2_DIR  →  <repo>/snapshots/SmolLM2-135M     # gitignored
GESSO_EAGER_PYTHON →  <repo>/snapshots/.venv/bin/python
```

If 10C is unfinished, stop.

## What this sprint is not

- fused decode, page-table attention, FlashAttention
- Autotune new operators or a second candidate family
- `rope_scaling`, SentencePiece, a second `model_type`
- `generate` that does not reset, sampling beyond greedy, 5b serving
- a Julia compiler patchset, PrecompileTools sysimage, a fork
- Magenta topology, cages, ExactBits, MoE routing bodies,
  multi-agent orchestration (those wait on a later fully formed
  handoff). `ExpertWeight` / `RoutingState` stay vocabulary.
- filling parked modules to "make them useful"
- committing `snapshots/`, `.venv`, or the mixed dirt listed above
- claiming "faster than PyTorch" from a micro fix

The 0.692× factor stands unless a retained win is measured on the
**same** SmolLM2 clocks (then append a new TSV row; do not edit
`2026-10-02.tsv`). llama_micro CUDA generate may gain a before/after
row. No kernel-only vs end-to-end mix.

---

## RPDO law (every change)

```
propose one function or one fail-closed hole
  → test (new or extended)
  → oracle: toy2 CPU bit-identical;
            llama_micro CUDA fingerprint class (inside atol=1e-3);
            greedy ids exact; fork unique_kv_bytes
            2048 vs 4096 (micro) and SmolLM2 N = 1_474_560 (CPU)
            when the snapshot is present
  → bench only if you claim speed (schema 0.2.0 append-only)
  → retain Pareto or revert the hunk
```

**Caps**

- Fail-closed **bugs** (silent success, wrong skip, law violation):
  uncapped. Fix all you find, each with a test.
- Mechanical **wins** (idiomatic, type-stable, alloc, tiny hot-path
  tighten): **at most five retained**. Sixth is a packet.
- New modules, new type hierarchies, new public names: **zero**
  unless a test cannot be written without one — then packet.
- New third-party deps: **zero**.

A finding you do not fix is written in the receipt as KEEP or
PACKET, never silently dropped.

---

## Work items (sequence)

### A — Inventory: every public name has a home

**Objective.** A test pins `names(Gesso)` (and the parked
submodules) so a drive-by export fails CI.

**Permitted files**

```
test/test_empty_core.jl          # extend parked-empty fence
test/test_export_inventory.jl    # create
test/runtests.jl                 # include the new file
```

**Parked-empty fence.** These modules stay contract-only (no new
exports, no bodies beyond the existing empty/`end`):

```
Lowering  Representation  Planning  Runtime  Agents  CAPI
```

`test_empty_core.jl` already pins Semantics/ModelIR/Parameters/
Operators. Extend it (or the new inventory file) so the six parked
modules export **nothing** (or exactly what they export today —
pin the set). Growing them is a work item, not a drive-by.

**Export inventory.** For every name in `setdiff(names(Gesso),
[:Gesso])`, one of:

- a test file that exercises it, or
- an explicit parked bucket (`:empty`, `:ext_only`, `:skip_named`)

Missing row ⇒ fail. Extra public name without a row ⇒ fail.

Do not add APIs so the inventory looks full.

**Artifact.** Drive-by exports cannot land. Parked modules cannot
grow in silence.

---

### B — Fail-closed bug hunt

**Objective.** Walk the engine for silent success, skip-shaped
fails, and law violations. Every real bug gets a test and a fix.

**Permitted files**

```
src/            # existing files only; no new modules
test/           # new tests allowed
ext/            # only if a CUDA/Lava cap or to_device lie is proven
```

**Required close (known 10C nit).** If `GESSO_SMOLLM2_DIR` is a
usable snapshot and `test/fixtures/smollm2/expected_logits.toml` is
**missing**, `test_smollm2.jl` and `test_cuda_smollm2.jl` **fail**
(not Broken skip). Unset env is still a named skip. The golden is
frozen; a vanished oracle is a broken tree.

**Hunt list (closed).** For each, KEEP / FIX / PACKET in the
receipt. FIX needs a test.

1. Returning `nothing` or a substitute from a lowering (must throw
   `LoweringNotImplemented` or a typed `GessoError` — §LXX).
2. `@hfallback` / silent backend switch / silent CPU fallback.
3. `supports(:argmax)` / `:attn_gemm` lies (CUDA true, Lava false).
4. Host `Array` under a non-CPU backend accepted (no-copy law).
5. Unknown checkpoint keys other than `*.rotary_emb.inv_freq`.
6. `generate` / `decode!` / `prefill!` error paths that leave the
   Session observably corrupt without a typed error (document if
   already fail-closed; do **not** invent transactional rollback).
7. Type piracy: methods on types we do not own, outside `ext/`.
8. Tests that skip when they should fail (snapshot+missing golden
   is the prototype).
9. `wait(::Process)` / probe lies in the G2 harness (already
   patched — confirm still true; do not re-open 10B).
10. Anything in `src/` that can download or talk to
    huggingface.co.

**Do not** "fix" `generate` resetting the Session. That is a later
product item. Record it as KEEP (known limitation).

**Do not** fill Lowering to route. Record as KEEP (empty by law).

**Artifact.** Known 10C skip-hole closed. Hunt table in the receipt.

---

### C — Unidiomatic / unstable Julia on the hot path

**Objective.** Fixture decode/prefill is type-stable enough to
`@inferred`. Hot structs are concrete where the engine actually
runs. No abstract-field soup, no `Any` bags, no string-key dispatch,
no `eval` / `invokelatest` on decode.

**Permitted files**

```
src/Inference/           # session, kv_manager, llama_import, tokenizer
src/Operators/cpu.jl
src/Profiling/
src/Autotune/            # only if a type-instability is in select()
test/test_type_stability.jl   # create
test/runtests.jl
```

**Required tests** (CPU, always-on, toy2 + llama_micro — no
snapshot required):

```
@inferred reference_prefill(toy2, ts, prompt)
@inferred decode!(session)     # after prefill!; eltype Int
@inferred unique_kv_bytes(mgr) # Profiling
```

If `@inferred` fails: **fix** (concrete types, barrier functions,
`const`) if the hunk is local. If it needs a new type hierarchy,
`@test_broken` with a PACKET — do not invent the hierarchy here.

**Also hunt (receipt KEEP/FIX, each FIX has a test):**

- non-`const` globals
- `Vector{Any}` / `Dict{String,Any}` on the decode path
- string interpolation or `Symbol` construction inside the token loop
- extra `collect` / `copy` / `Array(...)` on CUDA decode after
  `:argmax` (host should receive one Int)
- `isa Array` branches that defeat specialization
- comments that narrate "future fusion" as if it existed

SnoopCompile / Cthulhu / JET may be used **as operator tools** if
already on the box. They are not Project.toml deps. Do not check
in their traces except as a short receipt note.

**Artifact.** `test_type_stability.jl` green or an explicit
`@test_broken` packet. Idiomatic KEEP/FIX table in the receipt.

---

### D — At most five RPDO mechanical wins

**Objective.** Tiny tightenings that survive the oracle and, if
they claim speed, a bench row. Fusion is **not** a win in this
sprint (packet it).

**Permitted files**

```
src/            # existing files; parked modules stay empty
test/           # a test per win
benchmark/runbenchmarks.jl     # only if a win has a row
benchmark/results/             # append-only
```

Legal win shapes:

- remove an alloc from CPU `decode!` after warmup (`@allocated`
  or BenchmarkTools in the **benchmark** env, not a new core dep)
- concrete eltype / barrier that makes `@inferred` pass
- avoid a redundant gather-copy if (and only if) ids + fork bytes
  still hold — **not** a page-table kernel
- fail-closed error message that named the wrong key
- a missing unit test that pins existing law

Illegal win shapes: new Autotune candidates, FlashAttention,
changing greedy, changing page size defaults, F32 CPU oracle,
schema bumps, "while I was here" refactors.

**Measurement.** If you claim llama_micro CUDA generate moved,
append first-token + warmed rows (schema 0.2.0) and cite Phase 10B
warmed ~9.87 ms as the before class — compile-inside first-token
is a different clock. If you re-run G2, append; do not overwrite
`2026-10-02.tsv`. If you do not claim speed, no new TSV is required.

**Attribution (required in the receipt, not a new profiler).**
From **one** warmed run each, write:

```
workload            toy2 / llama_micro / SmolLM2 (if snapshot)
first generate      (compile inside)
second generate     (warmed)
decode! @allocated  after warmup (CPU; CUDA if you have it)
```

That is SPEED_FLOOR §2a.1 in miniature. No TimerOutputs. No Julia
fork. Full compiler-split (LLVM / GPUCompiler / SPIR-V) is a later
pass.

**Artifact.** ≤5 retained hunks, each named in the receipt with
oracle + optional ns/bytes. Fusion still a packet.

---

### E — Full suite, both env shapes

**Objective.** The stack is green with and without the snapshot.
Formatter clean. Fingerprints and fork bytes hold.

**Permitted files:** tests you already touched; no new product
surface.

```
make format
make test                                          # env unset
GESSO_SMOLLM2_DIR=<snapshot> make test             # demo box
```

Unset: named skips only for device/snapshot/Lava absences — counts
must not grow except for new `@inferred` files (all always-on).
Set: **0 Broken** on SmolLM2 CPU golden / Session / CUDA-if-present.
A vanished golden **fails**.

Long tests: `setsid nohup` if the gauntlet kills foreground
`make test`.

**Artifact.** Two suite receipts in the close note.

---

### F — maps; §LXXXIII still parked

**Objective.** Status sentences tell the truth: foundation
hardened; fusion still later; representation still parked.

**Permitted files**

```
README.md
docs/ARCHITECTURE.md
docs/research/ROADMAP_NOW.md
docs/research/README.md
docs/research/SPEED_FLOOR.md      # status sentences only
docs/Gesso_Stack.md               # SPEED FLOOR note under §LXXXIII only
docs/goals/PHASE10D_FOUNDATION_HARDENING.md
scripts/freeze.jl
```

Do not rewrite 10/10B/10C receipts. Do not mark §LXXXIII COMPLETE.
Do not commit mixed dirt.

---

## Cross-item invariants

- toy2 CPU fingerprint bit-identical
- llama_micro CUDA max|Δlogit| inside atol=1e-3 (cite the reprint;
  do not claim bit-identical to 10B's 0.000376 if it reprints)
- greedy ids exact toy2 + llama_micro; SmolLM2 Session ids still
  equal `reference_generate` on `"Hello"` × 8 when snapshot present
- `fork` unique_kv_bytes 2048 vs 4096 micro; SmolLM2 CPU N =
  1_474_560
- JSON only core third-party hard dep
- Autotune.jl imports neither CUDA nor Lava
- CUDA `supports(:argmax)` and `:attn_gemm`; Lava does not
- `libs/` porcelain 0
- parked modules still empty
- no receipt/bench schema bump
- no `generate`-does-not-reset, no fusion, no Julia fork, no MoE body
- `snapshots/` and `.venv` uncommitted
- mixed local dirt uncommitted

## Escalation (stop and write a packet)

- `@inferred` on `decode!` requires a new type hierarchy
- a bug fix moves greedy ids or fork bytes
- you want a page-table / fused attention kernel (that is the
  **next** efficiency recipe, not this one)
- you want to fill Lowering, Representation, or Runtime
- you want JET/SnoopCompile/TimerOutputs as a package dep
- you want to open §LXXXIII, Magenta, MoE routing, or Cyan
  multi-agent
- sixth mechanical win

## Performance target

N/A as a gate. Speed is optional evidence for a retained win.
Default success is **correctness + fences + ≤5 boring tightens**.
Do not treat a G2 re-run as the product unless a win moved the
named-model warmed clock — then publish the new factor beside
0.692×, same stamps.

## Expected artifact

- export inventory + parked-empty fence in CI
- vanished-golden fails
- type-stability tests
- hunt table (KEEP/FIX/PACKET)
- ≤5 RPDO wins named
- maps; §LXXXIII parked
- fusion still a later recipe

## Exit checklist

- [x] A: inventory + parked-empty fence green without snapshot
- [x] B: vanished-golden fails; hunt table in receipt
- [x] C: `@inferred` tests green or packeted `@test_broken`
- [x] D: ≤5 wins, each oracle-gated; attribution table in receipt
- [x] E: `make test` unset green; snapshot-set 0 SmolLM2 Broken
- [x] `make format` / format-check
- [x] fingerprints + fork bytes hold
- [x] mixed dirt and `snapshots/` uncommitted
- [x] F: maps + this file Status COMPLETE + §LXXII receipt
- [x] no Representation fill, no fusion, no Julia fork, no new deps

## Receipt (filled at close — §LXXII, 2026-10-02)

**What changed**

- A: `test/test_export_inventory.jl` (new) — 78-name export inventory
  over `setdiff(names(Gesso), [:Gesso])` (buckets :live / :here / :empty /
  :ext_only; :here exercises GPT2BPE via the gpt2_tiny fixture,
  GessoLogConfig, ReceiptSink, `lowering_not_implemented`) + six-module
  parked-empty fence (Lowering, Representation, Planning, Runtime, Agents,
  CAPI export only their own name) + stale-row and missing-file checks.
  Included from `test/runtests.jl` after `test_speed_floor_harness.jl`.
- B: `test/test_smollm2.jl` + `test/test_cuda_smollm2.jl` — the known 10C
  skip-hole is FAIL-closed: snapshot present + missing golden ⇒ the gate
  errors ("BROKEN TREE … restore it, or regenerate deliberately with
  test/freeze_smollm2_golden.jl --force"), never a Broken skip. Unset env
  stays a named skip. Dead inner if/else removed.
- C: `test/test_type_stability.jl` (new) — 12 pass / 3 `@test_broken`:
  `@inferred` GREEN on `unique_kv_bytes` + `kv_footprint` (toy2 +
  llama_micro); `reference_prefill` / `decode!` @inferred packeted on the
  §CIX `storage::Any` encoding (packet P-1; flip condition in the file
  header). Included from `test/runtests.jl`.
- D: **Win 1** — `src/Inference/kv_manager.jl` pins `page_bytes::Int` on
  `PagedKVManager` (computed from the prototype geometry; flows through the
  one inner-constructor call at kv_manager.jl:165 reached from
  session.jl:126/205) and `kv_bytes` sums it instead of walking
  `sizeof(p.storage)`; `src/Profiling/Profiling.jl` `unique_kv_bytes` sums
  `page_bytes` for distinct storages (docstring updated). Byte values
  identical (test_profiling 58/58, test_kv_manager 124/124 re-run green).
- F: maps (README phase table; ARCHITECTURE; ROADMAP_NOW ×3;
  research/README ×2; SPEED_FLOOR status sentence; canon §LXXXIII note
  only) + this file. `scripts/freeze.jl` needed no edit (owner had already
  staged PHASE10D in the curated list; the test-file section of that list
  is stale for Phases 8+ by convention — adding only 10D's files would be
  misleading, left alone).

**Why** — foundation hardening before fused decode: coverage fences,
fail-closed holes closed, type-stable byte counters, attribution evidence
for the fusion receipt. RPDO spiral on the existing machine; no new
modules, no new public names, no new deps.

**Tests**

- env-unset: `julia scripts/test.jl` → **1560 pass / 6 Broken / 1566
  total, 7m51.2s — passed.** Broken = 3 baseline + exactly the 3 packeted
  `@test_broken` (+34 passes vs the 1526 baseline = 22 inventory + 12
  stability, as designed).
- env-set (`GESSO_SMOLLM2_DIR=snapshots/SmolLM2-135M`): **1590 pass /
  3 Broken / 1593 total, 10m46.4s — passed.** The 3 Broken are exactly the
  packeted `@test_broken`; SmolLM2 gates carry 0 Broken.
- Scoped env-set reprints: test_smollm2 15/15, test_cuda_smollm2 8/8,
  test_session_smollm2 10/10.
- `make format` + `make format-check` → clean.

**Numerical delta**

- toy2 CPU fingerprint: bit-identical (suite oracle atol=0 green in both
  env shapes).
- llama_micro CUDA max|Δlogit| (fresh print this sprint): **8.0539e-5**,
  inside atol=1e-3 — NOT 10B's 0.000376 (different run, same declared
  class; cited, not claimed identical).
- SmolLM2 `unique_kv_bytes`: **N = 1_474_560** (CPU, pinned green) /
  **737_280** (CUDA F32, reprint).
- llama_micro fork `unique_kv_bytes`: **2048 vs 4096** (pinned green).
- Greedy ids: exact (toy2 + llama_micro + SmolLM2 `reference_generate` on
  `"Hello"` × 8), both env shapes.

**Attribution table (item D required artifact)** — one process, fresh
Session per generate (law: generate resets), idle clocks, RTX 5060;
SPEED_FLOOR §2a.1 in miniature, NOT a bench row — no TSV written:

```
workload     backend  first generate (compile inside)  warmed (×2)          decode! @allocated
toy2         CPU      2211.6 ms                        3.96 / 4.10 ms       89,120 B
toy2         CUDA     35,807.8 ms (in-process JIT)     35.31 / 32.69 ms     239,808 B
llama_micro  CPU      118.9 ms                         6.50 / 6.26 ms       139,984 B
llama_micro  CUDA     859.1 ms                         36.45 / 35.55 ms     321,712 B
SmolLM2      CPU      2682.3 ms                        2622.6 / 2576.7 ms   15,864,048 B
SmolLM2      CUDA     1150.2 ms                        565.0 / 595.0 ms     5,986,816 B
```

- **Before benchmark**: G2 **0.692×** (`benchmark/results/2026-10-02.tsv`,
  eager 0.436 s / Gesso CUDA 0.630 s); llama_micro CUDA warmed class
  ~9.87 ms (10B bench harness; the probe's fresh-session shape and idle
  clocks are not comparable to it — that is why no row was written).
- **After benchmark**: none — no win claims speed; both TSVs untouched.
- **Compile-time impact**: none claimed; no sysimage/precompile work; suite
  wall times above.
- **Memory impact**: byte-counter VALUES unchanged (page_bytes reproduces
  the old numbers by construction); decode! host-alloc profile documented,
  not changed: per token toy2 89 KB / llama_micro 140 KB / SmolLM2 15.9 MB
  CPU (§LXXVIII contract shape — per-token Activations + gather-on-read;
  packet P-2).
- **Hardware**: RTX 5060 (CUDA device present); Julia 1.12.6; Linux (WSL).
- **Workload / model / backend**: toy2 + llama_micro fixtures (CPU F64
  oracle; CUDA F32 rows); SmolLM2-135M from the local gitignored snapshot
  (CPU F64 + CUDA F32 rows); Lava loads and skips by name where a device
  requirement is absent.

**Hunt table (item B — 10 rows: 1 FIX / 9 KEEP)**

| # | Finding | Verdict |
|---|---------|---------|
| 1 | Lowerings return nothing/substitute? | KEEP — all stubs throw `LoweringNotImplemented`; Lowering.jl empty by law. Note: stale docstring `-> Nothing` on `lowering_not_implemented` (src/errors.jl:135), cosmetic, left for its next touch |
| 2 | `@hfallback` / silent switch / silent CPU fallback | KEEP — none found; `@gfallback` logs and returns the actual backend (§LXX) |
| 3 | `supports(:argmax)` / `:attn_gemm` lies | KEEP (verified) — CUDA ext declares both; Lava neither; decode gates on `supports` before the fast path |
| 4 | Host `Array` under a non-CPU backend accepted? | KEEP (verified) — `_check_device_storage` throws typed `ERR_INVALID_PLAN` naming `to_device`; the attribution probe hit exactly this when tensors were staged wrong, which is the evidence the error fires |
| 5 | Unknown checkpoint keys | KEEP (verified) — materialize errors WITH the key name; only `*.rotary_emb.inv_freq` ignored (suite green) |
| 6 | Error paths leaving the Session corrupt without a typed error | KEEP (documented) — typed errors on empty prefill / not-ready decode / no-tokenizer / context exhaustion; no transactional rollback by law; `generate` resets the Session (known limitation → later product item) |
| 7 | Type piracy | KEEP (verified) — only `Base.showerror` on our own error types |
| 8 | Tests that skip when they should fail | **FIX** — vanished-golden (snapshot present + missing golden) now FAILs in both SmolLM2 gates; verified by moving the golden away (suite errors) and restoring (green) |
| 9 | `wait(::Process)` / probe lies in the G2 harness | KEEP (confirmed still true) — runbenchmarks.jl documents the quirk; compare_eager.py uses `local_files_only` + a real `--probe`; 10B not re-opened |
| 10 | Downloads / huggingface.co in src/ | KEEP (verified) — none; nothing in src/ fetches |

**Item C idiomatic table** — non-`const` globals in src/: none;
`Vector{Any}`/`Dict{String,Any}` on the decode path: none; Symbol
construction in the token loop: none (only the import-time `Symbol(dt)`
dtype tag, llama_import.jl:218); extra collect/copy after CUDA `:argmax`:
none (host receives one Int, no-copy law green); `isa Array` branches:
the backend branch at decode top level is the declared §XX seam;
fusion-narration comments: none → all KEEP, nothing to fix. `@inferred`
`unique_kv_bytes`/`kv_footprint`: GREEN (Win 1). `@inferred`
`reference_prefill`/`decode!`: `@test_broken` → packet P-1.

**Retained wins (1 of ≤5)**

1. **page_bytes pinned on the KV manager** (kv_manager.jl + Profiling.jl):
   byte math stops walking `sizeof(p.storage::Any)`; values identical;
   makes the required `@inferred unique_kv_bytes` gate green. Oracle:
   full suites both envs; tests: test_type_stability green tier +
   test_profiling/test_kv_manager unchanged-green.

**Packets**

- **P-1 (§CIX re-encoding)**: `storage::Any` on the §XI family structs
  (Parameters.jl) + `Session.{model,tensors,h,tokenizer}::Any` poisons
  inference through `permutedims(seqvocab.storage)` and the decode hot
  path. Flipping the two `@test_broken` in test_type_stability.jl is the
  exit condition. Not done here — the spec forbids inventing the
  hierarchy in this sprint.
- **P-2 (decode! alloc profile)**: per-token Activations + gather-on-read
  copies (SmolLM2 CPU 15.9 MB/token) are the §LXXVIII contract shape;
  removing them needs Session scratch state — evidence handed to the
  fused-decode phase.
- **P-3 (win cap)**: 1 win landed; no sixth win attempted, so no cap
  packet required.

**Escalations** — none triggered (no id/fork-byte moves; no new type
hierarchy; no page-table kernel; no §LXXXIII fill; no tool dep).

**Cross-item invariants** — all hold: JSON still the only core third-party
dep; Autotune imports neither CUDA nor Lava; caps as declared (CUDA
`:argmax`+`:attn_gemm`, Lava neither); libs/ porcelain 0; parked modules
empty (inventory fence); no receipt/bench schema bump; no
`generate`-does-not-reset, no fusion, no Julia fork, no MoE body;
`snapshots/` and the mixed owner dirt remain UNCOMMITTED (nothing in this
sprint was committed — the working tree is the deliverable; the owner
lands the commit).

**Known limitations / unresolved questions** — `@inferred` on the
inference path waits on packet P-1 (§CIX owner); the SmolLM2 CPU warmed
clock is compute-dominated (2622→2577 ms across two warmed runs), i.e. the
alloc profile is the lever, not compile time — that is P-2's evidence;
attribution figures are idle-clock single-process probe numbers, not bench
rows.
