# GESSO FALLOUT FROM THE FIRST CYAN ENGINEER TRIAL

    Document class:     research handoff / promotion filter
    Status:             accepted as a metal detector, pre-implementation
    Index:              docs/research/README.md
    Source:             first end-to-end Cyan/Palette operator trial
                        (Buffy / GLM-5.3 via Freebuff); Chat extraction
                        handed to Foreman Grok 2026-09-30
    Primary testimony:  Palette repo, docs/research/FIRST_ENGINEER_INTERVIEW.md
                        (testimony + transcripts; not Gesso canon)
    Canon:              docs/Gesso_Stack.md remains canon
    This file is NOT:   a Phase 5 work item, a Magenta program,
                        permission to add ExecutionResult / ExecutionBudget
                        / Snapshot / ProbeSuite to src/

---

## 0. THE BOUNDARY THE TRIAL TESTED

Gesso is mechanism: load models, preserve meaning, decide how they
physically exist and execute. Training stays out (§LVIII). Chat's
wider sentence — "coherent competitor to the practical Python ML
stack" — is the *ecosystem* north star, including the Phase 21
training APB. It is not a license to put training loops, optimizer
checkpoints, or gradient probes in this package.

Palette is the Gesso-native interactive operator surface.
Cyan is the long-horizon harness on that surface.

    A feature belongs in Gesso only if it has clear value outside
    Cyan and Palette to ordinary Julia ML software, runtimes, tools,
    inference systems, or scientific workloads.

Being low-level is not sufficient. The trial overwhelmingly found
Palette/Cyan problems. That is a desirable result. Do not pollute
Gesso because Cyan hurt.

Chat's Cyan-thesis metrics (tokens per outcome, cheap-model +
substrate vs frontier + conventional, WGE as domain machine) are
**Cyan/WGE evaluation targets**. They do not become Gesso KPIs.
Gesso's receipts may later *supply numbers those metrics consume*.

---

## 1. PROMOTION RULE (ALL THREE, AFFIRMATIVE)

Before anything from Cyan/Palette moves into Gesso:

    Generality       would a Julia ML developer want this without
                     Cyan or Palette?
    Composability    can it be expressed without agents, context
                     windows, playbooks, or Palette sessions?
    Ecosystem leverage
                     does centralizing it here cut duplicated glue
                     across real Julia ML components?

Any "no" ⇒ keep it above Gesso.

---

## 2. CANDIDATES (INVESTIGATE, DO NOT BUILD)

Testimony from the trial, not ground truth. Observation → primary
receipt → operator interpretation → architectural hypothesis →
adversarial probe → disposition. Buffy's own controls (real host
backend; matched same-model A/B with/without the desk, graded
blind) are still outstanding. Do not promote on interview prose.

### A. Structured computational result contracts

Machine-consumable ops should return stable structured results
whose semantic fields are independent of presentation. Cyan's
result dictionary is not the protocol.

Gesso already has `Receipt` (§XLII) and `GessoError` / `ErrorCode`
(§LXX). Those are the vocabulary. A second `ExecutionResult` type
is the premature abstraction.

    Disposition:    GESSO LATER
    When:           Phase 6 fills timing / memory / failure on the
                    existing Receipt; Inference Session may emit
                    one per generate
    Do not:         import Cyan's dict; add a parallel result type
    Palette owns:   operator-facing accessors over whatever it runs

### B. Execution budgets as a first-class concept

"How close am I to the limit?" is not `elapsed_ms`.
`limit_margin_ms` is a Palette field.

Gesso already names the failure modes: `ERR_RESOURCE_LIMIT`,
`ERR_TIMEOUT`, `ERR_APPROXIMATION_BUDGET_EXCEEDED`, Magenta
`BoundedApproximation`. Runtime (§XXXII) is still empty.

    Disposition:    GESSO LATER
    When:           Phase 6 observability + Phase 9 autotune +
                    Phase 11 memory planner actually have budgets
                    to report against
    Do not:         build a Gesso scheduler because one Cyan call
                    wanted a timeout
    Palette owns:   live margin on desk operations

### C. Snapshot / clone / restore of computational state

Buffy's ask: a disposable copy of this world for a risky experiment.

That is Palette session cloning. Gesso's nearby object is an
inference `Session` (Phase 5) plus, later, KV CoW (Magenta §9.5
step 2) — a fork of *attention memory*, not a Julia world image.

Recipe reconstruction (ModelIR + checkpoint + Session fields)
beats memory-image cloning. GPU buffers, open files, and RNG
streams do not snapshot as one object.

    Disposition:    PALETTE/CYAN ONLY for interactive world clone
                    GESSO LATER for capture/restore of
                    ModelIR + weights + KV (+ seed), as a
                    *capability protocol*, not a universal
                    serializer
    Do not:         JLD2-the-whole-process; training checkpoints
                    (§LVIII)

### D. Provenance-bearing artifacts

Mordant (Buffy's Lab) registers rules as data and materializes
dispatch whose generated names are evidence. Do not promote
Mordant.

The steal: Gesso-produced artifacts should explain how they were
derived without an informal narrative. Canon already wants this
(§XLII receipts, §LXVIII `explain(plan)`, freeze SHA-256,
Phase 5 KV page provenance hooks).

    Disposition:    GESSO LATER as a design principle
    When:           Phase 6/9/10 attach provenance to kernels,
                    plans, materializations
    Do not:         a provenance framework in Phase 5

### E. Content-verified probe conventions

Health is representative behavior, not "the package loaded."

Gesso already does this locally: toy2 fingerprint, CUDA
`functional()` skip-or-green, SmolLM2 named skip, dependency-law
test, `ERR_VERIFY_MISMATCH`. That is the package's probe style.

A shared cross-package ProbeSuite is premature.

    Disposition:    PALETTE/CYAN ONLY for the live-desk battery
                    GESSO LATER only if multiple Gesso components
                    need a common probe *convention* (Phase 6)
    Do not:         one enormous diagnostic suite; gradient-flow
                    probes (training)

### F. Silence is a bug (conditional)

Every initiated computation terminates in an attributable outcome:
success, structured failure, cancellation, worker loss, or
explicit unknown-state.

This is already §LXX / `@gfallback`. Exact Unicode/parse and
request-id bugs were Palette/CyanTunnel.

    Disposition:    already GESSO NOW as law
    Implementation: none until Gesso owns shared worker/device
                    process machinery (Runtime, Phase 5+ follow-on)
    Do not:         a Gesso RPC framework

---

## 3. STAYS OUT (INSUFFICIENT JUSTIFICATION)

| Finding | Home |
|---|---|
| Operator playbooks / `# %%` | Palette |
| Supervisor `# !!` verbs | Palette |
| Output ring / `ring_id` | Palette |
| Inventory-bearing revival report | Palette |
| Workspace map | Palette |
| `$`-ownership shell UX | Palette |
| Live operator probe command | Palette |
| Interactive multi-block atomicity | Palette |
| Durable session journal | Palette substrate, Cyan consumes |
| Long-horizon continuity after compaction | Cyan |
| Recovery policy / capability governance | Cyan |
| Agent adapters / six-hour unattended | Cyan |
| Mordant / `@mordant` / Mordant revival | Buffy's Lab |
| Model-facing doctrine | Cyan / Palette |
| Cheap-model-bridges-frontier metrics | Cyan evaluation |
| WGE world-construction machinery | WGE |

---

## 4. NEGATIVE FINDING (LOAD-BEARING)

The scarcity of Gesso fallout is data.

The trial did not show that the Julia ML substrate needs a major
redesign. It showed that the young surface is the machine-facing
interactive and long-horizon operator experience.

Do not respond to a successful boundary test by collapsing the
boundary.

Phase 5 stays the boring engine (`docs/goals/PHASE5_ENGINE.md`).
These five spikes do not enter that recipe.

---

## 5. FOREMAN DIRECTIVE

Do not build these findings wholesale into Gesso.

Treat the Cyan trial as an architectural metal detector.

Investigate the candidates above. Search for existing Julia
solutions before inventing Gesso types. Promote an abstraction
only if its usefulness survives removal of Cyan, Palette,
Freebuff, MCP, and the agent.

Desired outcome: less fragmentation at the correct level,
not more machinery.
