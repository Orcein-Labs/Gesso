# /goal PHASE 10B — SPEED FLOOR CLOSE (harness + measure + skip-or-land G1/G2)

**For:** Buffy (mechanical implementation). **Gauntlet this.** Set-and-forget:
no user sign-off, no download, no waiting on ops. If `GESSO_SMOLLM2_DIR` is
a real snapshot **and** torch imports **and** CUDA.functional(), G1/G2 are
**required green** and the factor row must be a number. Otherwise named
skip, suite still EXIT 0.
**From:** Grok (encoding owner)
**Depends on:** Phase 10 LANDED (`docs/goals/PHASE10_SPEED_FLOOR.md`, HEAD
`04117d9` plus encoding-owner status close)
**Canon §LXXXIII stays PARKED.** Do not fill `Representation.jl`. Do not
mark representation COMPLETE. Do not fork Julia. Do not edit `libs/Lava`.
**Packets 1 and 2 stay closed.**

**Status:** COMPLETE 2026-10-01 (items A–D; receipt below).

---

## Receipt (Phase 10B close, 2026-10-01)

**What changed.**
A: the G2 torch gate is now a REAL dry import — `compare_eager.py --probe`
runs `import torch` + `import transformers` and exits 0/3 accordingly
(`--help` was never a torch probe; it proved the interpreter parses the
script, not that PyTorch exists — the exact hole that let a snapshot with a
missing venv pass the old gate and `error()` mid-suite). `runbenchmarks.jl`
probes with `--probe` and prints the probe's answer; a snapshot gate on the
loader's actual files (config.json + model.safetensors + tokenizer.json)
replaces bare `isdir`, so a junk dir is a named skip, not a `load_llama`
explosion; the `elseif` skip branch now names WHICH gate failed (unset /
not-a-snapshot / probe-failed). Harness tests assert the pair (--help 0 vs
--probe 3 on a torch-less box), the source guard (`--probe` present, no
`--help` command in runbenchmarks.jl), and keep the dep-law assertions.
B: caps are law, not comments — CUDA `supports(:argmax)` and
`supports(:attn_gemm)` asserted on an instance (test_session_cuda.jl);
Lava negatives asserted at type level without a device
(test_lava_inference.jl), plus its six real caps untouched. Contraction
math unchanged (no test proved a cap lie). C: llama_micro CUDA
`Session.generate` measured on the fast path — FIRST-TOKEN row (one
wall-clock generate, compile inside) + WARMED row (untimed warmup, then
BenchmarkTools through the standard suite), ids gated against the CPU
Session BEFORE any row is recorded (§LXX fail closed); notes name the caps
and declare FIXTURE status (llama_micro is not the board factor). SmolLM2
G1/G2: env unset on this box → named skip, no download, factor honestly
NOT MEASURED. D: maps updated (canon §LXXXIII note extended, SPEED_FLOOR
status sentences, ROADMAP_NOW, research README, README, ARCHITECTURE,
freeze list); §LXXXIII still NOT COMPLETE.

**Why.** Phase 10 built the door; this sprint closes the harness so a
later snapshot cannot false-fire the bench, pins the fast-path caps as
tests, and records a fixture-level generate measurement so the project is
not blind until SmolLM2 exists.

**Tests.** `make test` 1505 pass / 3 named SmolLM2 skips (8m52s); second
runner `--check-bounds=yes` 1505/3 (8m56s); harness test 13/13 torch-free;
Lava-negative testset green inside the Vulkan-gated block; item-A
acceptance runs: `make bench` EXIT 0 ×3 — env unset (skip named, 17 rows),
file-shaped snapshot dir + torch missing (probe-failed skip named, 17
rows), junk dir (not-a-snapshot skip named, 17 rows); format-check OK.

**Numerical delta.** Fingerprints bit-identical to Phase 8/9/10: toy2
cuda-vs-cpu max|Δlogit| = 0.00037607177004872483, llama_micro =
5.5006127839263286e-6 (both reprinted this run in the gate logs); greedy
ids exact toy2 + llama_micro; `fork` unique_kv_bytes 2048 vs 4096 BYTE
rows still on disk (no re-bench needed — unchanged rows stand). New
llama_micro generate rows (RTX 5060, CUDA 6.3.1, Julia 1.12.6,
tokens [0,1,2] + 3 greedy steps, page_size 4): FIRST-TOKEN 42.1 ms
(compile inside, 1 sample), WARMED median 9.87 ms / mean 10.86 ms /
min 8.88 ms over 461 samples. Factor: NOT MEASURED (no snapshot, no
torch on this box) — the row says so itself.

**First-token vs warmed.** Both clocks on the fixture path: first-token
carries one-shot kernel compile (~4.3× the warmed median); warmed excludes
it (§XXXIII). SmolLM2 clocks (both sides) remain implemented behind the
G2 gates and land when the box has snapshot + torch.

**Compile-time / memory.** No new packages anywhere (probe is a
subprocess dry import; torch stays external, asserted by the dep-law
test); no schema bump (receipts 0.1.0, bench 0.2.0); bench suite 15 → 17
rows (two llama_micro generate rows).

**Hardware / workload / model / backend.** RTX 5060, CUDA 6.3.1,
Julia 1.12.6, linux x86_64; workload: llama_micro `Session.generate`
3 greedy steps (fixture) + `generate("Hello"; max_new_tokens=8)` on
SmolLM2-135M-when-present (skipped here); backends: CUDA (fast paths
:attn_gemm + :argmax), CPU (oracle + ids gate), Lava (unchanged, negatives
pinned).

**Known limits / unresolved.** G1 and the G2 factor remain ops-owned:
snapshot + torch on one box converts the three SmolLM2 skips to required
green and the factor row to a number (both sides' dtype stamps travel in
the notes — Gesso F32; eager from_pretrained defaults). The page-table
kernel (skipping the gather) is still a packet. §LXXXIII stays parked
until G1∧G2∧G3 are honestly green with a measured SmolLM2 factor; next
recipe is encoding-owner from the TSV (harder fusion vs lab-MVP hold).

---

## Start condition

You inherit `master` at `04117d9` (item D) with living Phase 10 rungs 1–2
on the tree:

- CUDA decode: `_device_greedy_id` (`:argmax`) — host gets one Int
- CUDA attention: device `mul!` over gathered scratch (`:attn_gemm`)
- Lava: no those caps; host argmax + per-head contraction
- `fork` unique_kv_bytes 2048 vs 4096 on llama_micro (TSV BYTE rows)
- fingerprints cited: toy2 `0.00037607177004872483`, llama_micro
  `5.5006127839263286e-6`
- 1497 pass / 3 named SmolLM2 skips; device-less 1310 / 10
- `benchmark/compare_eager.py` exists; G2 factor row is **NOT MEASURED**
- `_p10_torch_ok()` currently runs `compare_eager.py --help` (never
  imports torch). If snapshot dir is set and torch is missing, the G2
  block **`error()`s** instead of skipping.
- `prefill!` still `Array(logits)` on CUDA; `generate` uses prefill then
  `decode!` — generate hot path is already device greedy
- Autotune still 2-D `matmul!` only; attn GEMM is Session-dispatched
- JSON only core third-party hard dep; no torch in any Project.toml
- `libs/Lava` clean
- Parallel workstream files (PHASE5–9 untracked, SPEED_FLOOR, ledger,
  ancient texts, RPD_SOP, `1x`) — **do not stage or rewrite them**

If Phase 10 B/C/D are missing, stop.

## One-sentence objective

The G2 harness skip-or-lands correctly; `:attn_gemm` is tested; llama_micro
CUDA generate is measured on the GEMM path; if a local SmolLM2 snapshot
and torch happen to be present, G1 is green and G2 publishes a factor.
CI never needs any of that. §LXXXIII still parked.

## Why this sprint exists

Phase 10 built the door. Ops has not placed the weights. This sprint
closes the harness so a later snapshot cannot false-fire the bench, pins
the fast-path caps, and records a fixture-level generate measurement so
we are not blind until SmolLM2 exists. If ops drops the snapshot while
you run, you land the board number in the same sprint.

## What this sprint is not

- canon §LXXXIII, Magenta, cages, ExactBits, foundry, Julia fork
- FlashAttention / page-table kernel (still a packet)
- Lava Autotune consult
- 5b serving, C ABI, `generate` no longer RESETS
- downloading SmolLM2 or adding torch/PyCall to Project.toml
- touching parallel untracked docs listed above
- a speed claim vs PyTorch in prose (the factor row is the claim)

---

## Laws

- **Skip-or-land, never error-for-missing-torch.** Probe with a real
  `import torch` / `import transformers` (or `compare_eager.py` mode that
  does that and exits 3). Snapshot xor torch ⇒ named skip + NOT MEASURED
  factor row. Both present + CUDA ⇒ ids gate then rows. Missing CUDA ⇒
  named skip.
- **No downloads.** `local_files_only=True` stays. `GESSO_SMOLLM2_DIR`
  must be a directory of local files.
- **Fail closed on ids.** If SmolLM2 CUDA generate diverges from CPU,
  no G2 rows. Typed error.
- **G3 on fixtures still holds.** toy2 + llama_micro ids exact; fork
  2048 vs 4096 BYTE rows still on disk (re-append if you re-bench).
- **Long tests:** `setsid nohup … &` — gauntlet tool-call kills have
  eaten foreground `make test` before.

---

## Work items (sequence)

### A — G2 probe is a dry import; skip is skip

**Objective.** `_p10_torch_ok()` (or equivalent) actually imports torch
and transformers. Snapshot-present + torch-absent does **not**
`error()` the bench suite. Named skip + NOT MEASURED factor row.

**Permitted files**

```
benchmark/runbenchmarks.jl
benchmark/compare_eager.py
test/test_speed_floor_harness.jl
test/runtests.jl
```

**Tests**

- harness test still 10/10 torch-free (or updated count, still green
  without torch)
- a unit/harness assertion: `--help` is not the torch probe
- `make bench` without snapshot: EXIT 0, factor row NOT MEASURED
- do not add torch to Project.toml (existing assertion stays)

**Artifact.** Demo-box can have a snapshot and a missing venv without
the suite exploding.

### B — Pin the fast-path caps

**Objective.** CUDA `supports(:argmax)` and `supports(:attn_gemm)` are
asserted. Lava `supports` of both is false. Decode still one-Int D2H.
No page-table kernel.

**Permitted files**

```
test/test_session_cuda.jl
test/test_lava_inference.jl     # only if a Lava negative cap test belongs here
test/runtests.jl
src/Inference/session.jl        # only if a probe comment is a lie
```

Prefer tests-only. Do **not** change contraction math unless a test
proves a cap lie.

**Tests**

- CUDA.functional(): `@test Gesso.supports(cuda, :argmax)` and
  `:attn_gemm`
- Lava-when-present: both caps false (or skip if no Lava device)
- existing ids / fingerprint / fork tests still green
- no CUDA: named skip

**Artifact.** Caps are law, not comments.

### C — Measure the path we have; land G1/G2 if the box allows

**Objective.** Two clocks on llama_micro CUDA `Session.generate` **after**
`:attn_gemm` (first-token + warmed, schema 0.2.0, notes name the caps).
If `GESSO_SMOLLM2_DIR` is a snapshot **and** torch imports **and**
CUDA.functional():

- existing SmolLM2 tests required green (the three skip files become
  pass)
- G2 block publishes eager first/warmed + Gesso first/warmed + factor
  declaration from this run's medians
- ids CUDA == CPU on `Hello` × 8

If any of the three is missing: named skip, no download, EXIT 0.

**Permitted files**

```
benchmark/runbenchmarks.jl
benchmark/results/            # append-only TSV
test/test_smollm2.jl          # only if a skip/gate hole appears
test/test_cuda_smollm2.jl
test/test_session_smollm2.jl
test/runtests.jl
```

Do not invent a SmolLM2 golden. Do not time llama_micro as if it were
the board factor. llama_micro rows are fixture measurement; SmolLM2
rows are G2.

**Artifact.** New TSV rows on disk. Factor is either a number or an
honest NOT MEASURED. Both are success.

### D — maps; §LXXXIII still parked

**Permitted files**

```
README.md
docs/ARCHITECTURE.md
docs/research/ROADMAP_NOW.md
docs/research/README.md
docs/research/SPEED_FLOOR.md      # G1/G2/G3 status sentences only
docs/Gesso_Stack.md               # SPEED FLOOR status note under §LXXXIII;
                                  # do NOT mark §LXXXIII COMPLETE
docs/goals/PHASE10B_SPEED_FLOOR_CLOSE.md
scripts/freeze.jl
```

If C landed a SmolLM2 factor, say so with the number and the dtype
stamp. If not, say ops still owns G1. Next recipe is encoding-owner
from the TSV (harder fusion vs lab-MVP hold). You do not open
representation.

**Do not** commit or rewrite: `AGENTS.md`, PHASE4/5/6/7/8/9 extras,
`KV_MEMORY_PROGRAM.md`, `EXOTIC_CAPABILITY_MASTER_LEDGER.md`,
`RPD_SOP.md`, `Julia Compiler Optimizations for Gesso/`, `1x`.

---

## Cross-item invariants

- toy2 CPU fingerprint bit-identical
- CUDA fingerprint prints stay in the Phase 8/9 class (cite the two
  floats if they reprint)
- `reference_*` signatures unchanged
- JSON only core third-party hard dep
- Autotune.jl imports neither CUDA nor Lava
- CPU/Lava paths unchanged in consult behavior
- `libs/` porcelain 0
- `Representation` / `Planning` / `Runtime` still empty
- no ExecutionResult, no schema bump
- Receipt: what / why / tests / numerical delta / first-token vs warmed
  / factor-or-NOT-MEASURED / hardware / llama_micro always / SmolLM2
  if present / backend cuda

## Escalation (packet, then stop)

- you need a page-table kernel or FlashAttention to keep ids
- you need to drop `fork` aliasing
- you need a new core dependency
- you want to fill Representation.jl
- G2 ids diverge on SmolLM2 when snapshot is present
- you want to download weights

## Exit checklist

- [x] A: torch probe is a dry import; missing torch never `error()`s the bench
- [x] B: `:argmax` and `:attn_gemm` asserted on CUDA; Lava negatives when present
- [x] C: llama_micro CUDA generate first-token + warmed rows on disk (GEMM path)
- [x] C: SmolLM2 G1+G2 either **numbers** or **named skip**; never a download
- [x] D: maps truthful; §LXXXIII not COMPLETE
- [x] `make test` EXIT 0 without CUDA and without SmolLM2
- [x] `make format` run
- [x] CPU fingerprint unchanged
- [x] Lava skip-or-green
- [x] `fork` 2048 vs 4096 still evidenced (test or TSV BYTE rows)
- [x] no silent fallback
- [x] `libs/` clean
- [x] parallel workstream files not in your commits
- [x] receipt in this file
