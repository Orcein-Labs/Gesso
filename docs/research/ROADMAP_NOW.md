# GESSO ROADMAP NOW

    Document class:     living interpretation of the phase plan
    Status:             working map (subordinate to Gesso_Stack.md)
    Index:              docs/research/README.md
    This file is NOT:   a Buffy /goal, a work item, permission to
                        implement Phases 7–20, or a rewrite of canon

Buffy implements closed recipes under `docs/goals/`.
Phase 6 landed 2026-09-30 (`docs/goals/PHASE6_OBSERVABILITY.md`):
receipts per engine call, Profiling reports, KV bytes from the page
table — attribution, not speed. Phase 5a landed 2026-09-30
(`docs/goals/PHASE5_ENGINE.md`). Next open recipe: Phase 7 (one semantic
win; default pick CoW / identity prefix share) — goal file to come.

Canon still lists Phases 0–22 in `docs/Gesso_Stack.md` §LXXIII–§XCV.
This file says what that list *means* after Phases 0–4 and the first
Cyan engineer trial.

---

## 0. WHERE WE ARE

Phases 0–5a are complete: meaning, CPU F64 oracle, Llama-shaped
import, CUDA.jl as a weakdep, `Session` + paged KV matching the
oracle. `reference_*` remain the oracle. Lowering, Planning,
Autotune, Representation, Runtime, Profiling are contract-only
(Profiling fills in Phase 6).

That is the floor. Exotic Gesso is still the point. The floor has
to exist first.

---

## 1. THREE LAYERS (DO NOT FLATTEN)

    1. Boring machine     Phases 5–6 (5a now, 5b later)
    2. One measured win   Phase 7, then Lava + autotune (8–9)
    3. Exotic compilers   Phases 10–11, then 18–20

Cyan and Palette are not a Gesso layer. They are already being
tortured *above* the machine. Gesso ships a Session and later a
C ABI. Cyan consumes it. Playbooks, revival, Mordant, notes
tiers stay up there (`docs/research/CYAN_TRIAL_GESSO_FALLOUT.md`).

---

## 2. PHASE 5 IS TWO GOALS

Canon §LXXVIII packs prefill, decode, KV, sampling, scheduler,
streaming, and sessions into one exit. That is too much for one
Buffy recipe.

    5a  LANDED  docs/goals/PHASE5_ENGINE.md  (2026-09-30)
                Session, paged KV (Magenta step 1), greedy,
                on_token, string generate. Token ids match
                the oracle. reference_* stay the oracle.

    5b  LATER   a new goal file, still under Phase 5, after 5a
                lands and someone has two concurrent Sessions
                Scheduler, continuous batching, multiplexing,
                sampling beyond greedy.

Phase 5 exit that matters: Gesso no longer relies on another
inference engine. That is 5a. Serving fabric is 5b.

---

## 3. SEQUENCE FROM HERE

```
0–4     floor                         DONE
5a      engine                        DONE  2026-09-30
5b      serving fabric                after two concurrent Sessions
6       measure                       OPEN  PHASE6_OBSERVABILITY.md
7       one semantic win              pick ONE (see below)
8–9     Lava + autotune               second silicon, first search
10–11   weight compiler + memory      REPRESENTATION_PROGRAM + Magenta
12–13   shared model runtime          mechanism for many agents, one model
ABI     Cyan consumes Gesso           Phases 16, then 17 adapter
14–15   canon still names Palette/Cyan as Gesso phases;
        living reading: they already exist above Gesso
18–20   specialize / rematerialize / speculate
21      training APB                  never this package
22      plastic / lifecycle           research
```

Phase 7 pick, given current evidence:

    Default:  CoW + identity prefix share on the paged KV
              (Magenta §9.5 steps 2–3). Same model, same
              hardware, correctness preserved, measurable share.

    Steal:    gauge-compiled / ExactBits only if Measurement 1
              (B1 < B0 on SmolLM2) is interesting. Probe on the
              CPU oracle in parallel as research. Do not make it
              the Phase 7 default until the number exists.

    Needs 6:  phase-specific prefill/decode realization.

Speculation remains Phase 20.

---

## 4. GLUE vs INVENTION

Glue (write our copies; do not pretend they are science):

    paged KV + gather-to-scratch attention
    Session lifecycle, EOS from config, no-copy CUDA pages
    greedy extracted; later sampling with a declared RNG
    Lowering.jl as actual backend routing
    second model_type / rope_scaling / SentencePiece
        after SmolLM2 is green on a real snapshot
    Lava through the existing backend contract
    C ABI over a stable Session
    Phase 6: fill existing Receipt timing/memory/failure;
        attributable kernel / launch / KV counters
        (Cyan's structured-result / budget-margin / silence
         spikes land here as field growth)

Invention (the Gesso-shaped hole):

    working state as a lowering          Magenta after pages
    logical ≠ physical for weights       Phase 10; optional 7
    search with a correctness gate       Phase 9, then 18–19
    declared approximation               named now, enforced
                                         when a lowering can lie
    memory under pressure                Phase 11
    identity-based prefix share          mechanism claim
    generalized speculation              Phase 20

Do not invent in Gesso: ExecutionResult, Julia-world snapshots,
a probe suite, an RPC, playbooks, Mordant, training loops.

---

## 5. HOW TO USE THIS FILE

- Hand Buffy a file under `docs/goals/`. Never this file.
- When a phase opens, write a closed recipe, then implement.
- When a research program is exciting, leave it in this folder
  until that recipe exists.
- Promote a sentence from this file into `Gesso_Stack.md` only
  by editing canon, not by implementing it.
