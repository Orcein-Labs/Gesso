# AGENTS.md — Gesso Agent Charter

**Every agent working in this repository operates under this charter.**
It transcribes laws from `docs/Gesso_Stack.md` (canon, sections referenced
as §NNN). Reading the charter is not a substitute for reading the canon —
but the charter is binding even if you have not read the canon.

---

## 0. Read order (before your first edit)

1. This file.
2. `docs/Gesso_Stack.md` — at minimum §II (first principle), §III
   (development law), §LXXI–§LXXII (swarm rules, receipts).
3. `docs/ARCHITECTURE.md` — the module map.
4. The work item you were assigned, which must follow the template in
   `.github/ISSUE_TEMPLATE/work-item.md`.

## 1. What Gesso is (so you do not build the wrong thing)

Gesso is a Julia-native semantic ML and agent execution runtime: it loads
existing models, preserves what they mean, and uses that meaning to decide
how they physically exist and execute.

    MAKE IT WORK → MAKE IT COMPLETE → MAKE IT MEASURABLE → MAKE IT FAST
    (then, much later: MAKE IT WEIRD → MAKE THE WEIRDNESS FAST)

Do not build ahead of the phase plan. Do not build "the whole vision."

## 2. Hard scope fences

* **Training is NOT our problem** (§LVIII). No AD modes, optimizers,
  gradient paths, training loops — not deferred, *out of scope*. If you
  find yourself writing a backward pass, stop.
* **Do not touch `libs/`** — local dev checkouts, never part of the package.
* **Lava and CUDA are extensions, not dependencies** (§VII). Core Gesso
  earns every hard dependency; the dependency-law test enforces this and
  you must edit that test (with a justification) to add anything.
* **Palette/Cyan boundaries** (§XLIII): Gesso owns mechanism. Palette owns
  expression. Cyan owns policy and cognition (internal: NIRA). Do not
  implement policy or cognition here. L3+ memory (semantic/episodic) is a
  seam, not a component.

## 3. How you work (§LXXI)

Every task you receive should be: bounded, independently testable,
benchmarkable where relevant, revertible. If your task cannot be described
in the work-item format, ask for it to be split.

You receive narrow, testable pieces. You do not:

* "make it fast" without a benchmark,
* "support model X" without conformance tests,
* "optimize" by changing semantics,
* refactor beyond the permitted files in your work item,
* introduce a dependency, type hierarchy, or abstraction the task does not
  require ("speculative future implementation" is forbidden — §LXXIII exit
  rule: no speculative future implementation).

**Ground truth before theory.** When observations, tool output, or task notes
seem incoherent, inspect the filesystem and git state first (`git status`,
`ls`, re-read the file) before theorizing. CI paths and dev-loop paths drift
independently: a green CI does not prove `make test` works (this exact gap
was found in the Foundation Hardening sprint — CI ran tests directly while
the documented dev loop was broken). Run the documented commands themselves,
not just their equivalents.

## 4. Correctness and evidence (§V, §LXXII)

* **Agent confidence is not evidence. The harness decides.**
* Every meaningful change produces a receipt (§LXXII):

```
what changed · why · tests · numerical delta ·
before benchmark · after benchmark · compile-time impact ·
memory impact · hardware · workload · model · backend
```

* No timing claims from runs that included compilation (§XXXIII rule 1).
* Never compare kernel-only time to end-to-end time.
* Failing tests are findings, not embarrassments. Record them.

## 5. Failure and fallback (§LXX)

Gesso fails explicitly. No silent representation downgrade, backend switch,
quantization mismatch, memory-plan violation, or kernel substitution. If a
policy permits a fallback, it goes through `@gfallback` — logged, or it
did not happen. Lowerings decline work ONLY by throwing
`LoweringNotImplemented`; returning `nothing` or a substitute result is a
law violation.

## 6. Approximate execution (docs/research/KV_MEMORY_PROGRAM.md §6)

Approximate lowerings are DECLARED contracts
(`BoundedApproximation{metric, ε, oracle_tier}`), never undocumented
optimization flags. Exceeding a declared budget is
`APPROXIMATION_BUDGET_EXCEEDED` and disqualifies the candidate.

## 7. Citations and research claims (KV program §4.3)

In any written artifact, citation = claim of having read it. The
verify-before-citing list exists; do not promote papers from it into canon
without reading them. No public/performance claims beyond what a benchmark
run in this repo produced.

## 8. Where things live

```
src/           package core (see docs/ARCHITECTURE.md for the module map)
test/          test harness — per-area files included from runtests.jl
benchmark/     benchmark harness — results land in benchmark/results/
docs/          Gesso_Stack.md is CANON; docs/research/ is research program
libs/          DO NOT TOUCH (local dev checkouts)
scripts/       dev commands (test/bench/format/freeze)
```

Open architecture questions (decisions deliberately NOT made locally) are
recorded as decision packets in `docs/DECISION_PACKETS.md`. Do not resolve
a packet by implementing one option — resolution happens in canon.
Packets 1 (workload) and 2 (receipt identity) are resolved in
`docs/Gesso_Stack.md` §CIX. Phase 1 and Phase 2 have landed; do not reopen
the object-model encoding.

## 9. Exit checklist for every work item

* [ ] tests pass (`scripts/test.jl` or `make test`)
* [ ] formatter run (`make format`)
* [ ] receipt written (§LXXII fields above) — in the PR, not your head
* [ ] changed-files summary matches the work item's permitted files
* [ ] known limitations + unresolved questions written down
* [ ] no speculative future implementation added
