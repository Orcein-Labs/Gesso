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
(`docs/goals/PHASE5_ENGINE.md`). Phase 7 landed 2026-09-30
(`docs/goals/PHASE7_PREFIX_SHARE.md`): CoW + declared identity prefix
share — `fork` is the only share constructor, and the win is bytes
(`Profiling.unique_kv_bytes`: 2048 vs 4096 on the llama_micro prefill
pair, `benchmark/results/2026-09-30.tsv`). Phase 8 landed 2026-09-30
(`docs/goals/PHASE8_LAVA.md`): Lava/Vulkan as a weakdep extension — seam,
six ops, interpreter/Session/fork on `LavaArray{Float32}`, ids matching the
CPU oracle; portable seam, not tuned. Phase 9 landed 2026-10-01
(`docs/goals/PHASE9_AUTOTUNE.md`): the §XXVI loop — two gated CUDA
`matmul!` candidates, the winner cached per (device, backend, op, regime),
the op consults Autotune and dispatches to it. Selection happened; no
speed claim. Next open recipe: **speed floor**
(`docs/goals/PHASE10_SPEED_FLOOR.md`) — named model + eager-PyTorch
factor + device greedy + device attention over gathered scratch.
Canon §LXXXIII representation waits on G1∧G2∧G3.

Canon still lists Phases 0–22 in `docs/Gesso_Stack.md` §LXXIII–§XCV.
This file says what that list *means* after Phases 0–4 and the first
Cyan engineer trial.

---

## 0. WHERE WE ARE

Phases 0–8 are complete: meaning, CPU F64 oracle, Llama-shaped
import, CUDA.jl as a weakdep, `Session` + paged KV matching the
oracle, receipts + Profiling, the declared-share win
(CoW + identity prefix share), and Lava/Vulkan as a weakdep
extension (`GessoLavaExt`, `LavaArray{Float32}`, ids matching the
CPU oracle). `reference_*` remain the oracle. The CUDA and Lava
*seams* exist; Lowering *routing* is still later. Planning,
Representation, Runtime are contract-only. Autotune is not anymore:
the §XXVI loop runs, one operator (CUDA `matmul!`) selects a
device-specific winner under a correctness gate, caches it, and the
engine uses it (§LXXXII; selection is the product — no speed claim).

That is the floor. Exotic Gesso is still the point. The floor has
to exist first. A board will not accept the exotic on an engine
that is not somewhat close in performance on a named model —
see §3a and `docs/research/SPEED_FLOOR.md`.

---

## 1. FOUR LAYERS (DO NOT FLATTEN)

    1. Boring machine     Phases 5–6 (5a+6 landed, 5b later)
    2. One measured win   Phase 7 (landed 2026-09-30), Phase 8 Lava (landed 2026-09-30), Phase 9 autotune (landed 2026-10-01 — selection, not speed)
    3. Speed floor        G1∧G2∧G3 in SPEED_FLOOR.md (named model,
                          eager-PyTorch factor, fork-preserving fused
                          decode). Gate on layer 4.
    4. Exotic compilers   Phases 10–11, then 18–20

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
5b      serving fabric                after G2 and two concurrent
                                      Sessions on the fast path
6       measure                       DONE  2026-09-30
7       one semantic win              DONE  2026-09-30  PHASE7_PREFIX_SHARE.md
8       Lava / Vulkan                 DONE  2026-09-30  PHASE8_LAVA.md
9       autotune (selection)          DONE  2026-10-01  PHASE9_AUTOTUNE.md
G1–G3   speed floor                   LANDED 2026-10-01  PHASE10_SPEED_FLOOR.md
                                      device greedy (one Int per token) +
                                      device GEMM over gathered scratch;
                                      still forks; G2 harness exists —
                                      SmolLM2 rows land on a box with a
                                      local snapshot + torch (G1 is
                                      ops-blocked here; harness skip-or-green)
                                      §LXXXIII stays parked until G1∧G2∧G3
                                      are honestly green with a measured
                                      SmolLM2 factor on the demo box
10–11   weight compiler + memory      AFTER G1∧G2∧G3
12–13   shared model runtime          mechanism for many agents, one model
ABI     Cyan consumes Gesso           Phases 16, then 17 adapter
14–15   canon still names Palette/Cyan as Gesso phases;
        living reading: they already exist above Gesso
18–20   specialize / rematerialize / speculate
21      training APB                  never this package
22      plastic / lifecycle           research
```

---

## 3a. BOARD CONSTRAINT (2026-10-01)

Exotic capabilities are the product. They are illegal to *sell*
until a named-model decode on this box is somewhat close to naive
PyTorch eager, as a published factor, with `fork` still sharing
pages. Full gate, hot-path inventory, and RPD/RPDO split:
`docs/research/SPEED_FLOOR.md`.

Do not open a Phase 10 Buffy recipe until G1∧G2∧G3. Idle Buffys
take the speed-floor recipe or Llama-family import glue (second
tokenizer, `rope_scaling`), one architecture at a time.

The speed floor has two clocks: application (fusion, D2H, gather)
and Julia compiler (TTFX, invalidation, GPUCompiler, Lava SPIR-V).
Attribute compile vs execute on SmolLM2 before anyone proposes a
compiler fork. Seed:
`Julia Compiler Optimizations for Gesso/docs/the_ancient_texts.md`.

Seriousness bet (`SPEED_FLOOR.md` §2b): stack kernels + Julia
compiler + Lava, together, get us fairly close. After that,
dispatch and metaprogramming specialize parallel *schedules*
(better where parallelism already exists; new where the engine
still serializes — `fork` trees, page CoW). That is Phases
18–20 shaped work. It is not the next Buffy recipe.

Proof ladder and thesis: `SPEED_FLOOR.md` §2c. Two lanes, one
engine — boring Session for everyone (and for Cyan); exotic
schedules for ML/data scientists who can use the meaning. Lane B
exists because Lane A is reliable. Collect "Julia would let us
do *what*?" cases; do not build them from a daydream. First-draft
mine: `docs/research/EXOTIC_CAPABILITY_MASTER_LEDGER.md` — strings,
not sprints. The speed floor is the baseline every exotic schedule
has to race.

"Any currently available model" is the long import program. The
speed floor's model is SmolLM2-135M. Training remains someone
else's package.

Phase 7 pick, LANDED 2026-09-30:

    Default:  CoW + identity prefix share on the paged KV
              (Magenta §9.5 steps 2–3). Same model, same
              hardware, correctness preserved, measured share:
              unique_kv_bytes 2048 vs 4096 (llama_micro
              prefill pair, benchmark/results/2026-09-30.tsv).

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
