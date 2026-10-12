# RPD SOP

    Research / Plan / Develop
    Audience:     team leads (Grok, Codex, Claude) when Matt asks for
                  R&D on a problem
    Copies:       identical files at Gesso/, Palette/,
                  Project Cyan/, and Project Cyan/Cyan Harness/.
                  Edit all four or they will drift.
    This is not:  a Buffy /goal, a work item, or permission to implement.

RPD is a spiral with an exit ramp, not a waterfall and not an infinite
chat. Buffy-class implementation starts only from a closed contract.

RPDO is the same spiral aimed at search: candidate → oracle gate →
measure (post-warmup) → winner → cache → invalidate. Semantic
stubbing (below) is how RPDO retains a capability before the
candidate has a dedicated body.

---

## Trigger

This SOP applies when Matt asks for R&D, research, a spiral, "what
should this be," or "figure out this problem."

It does **not** apply when he hands you a closed recipe, a bug with a
known repair, or "just implement it." Those go straight to Develop
(micro) or to the implementer.

One lead owns a spiral. Name the owner, the repo, and the layer
(Gesso / Palette / Cyan / Lab / WGE) in the first reply.

---

## The spiral

```
    RESEARCH  →  PLAN  →  DEVELOP  →  RESEARCH
         ↑                               │
         └──────── leftover question ────┘
```

Exit this spiral when every **load-bearing** question for **this
problem** has a disposition. The leftover question starts the *next*
spiral. It is not a reason to keep talking.

---

## Eleven laws

1. **Research before architecture.**
   Do not design around assumptions that can cheaply be investigated.
   If the experiment is expensive: name the assumption, proceed, tag
   it for the next spiral.

2. **Research produces evidence, not requirements.**
   A complaint such as "I need a better revival report" is an
   observation. The requirement might actually be "the system needs
   explicit state identity and survivorship semantics."

3. **Planning is compression.**
   Explain many observations with fewer primitives. Seventeen UX
   problems becoming four architectural concepts is progress.

4. **Planning must be adversarial.**
   The same lead red-teams their own abstraction before the roadmap.
   Kill for: seams, measurement, non-composability, scaling,
   hardware/runtime, correctness, decisive cost. Semantic nitpicks
   are not kills. Hold existing canon until evidence forces a change.

5. **Prototype to answer questions, not merely to make progress.**
   A throwaway 100-line experiment that kills a bad abstraction is
   enormously productive. Prototypes stay throwaway. They do not land
   in product `src/` of a layer that does not own the question.
   A prototype dies after it answers the question. A semantic stub
   (below) persists as identity with a cheap lowering.

6. **Development generates research data.**
   Bugs, operator friction, strange usage patterns, performance
   results, and unexpected inventions all flow back into Research.

7. **Real use outranks synthetic reasoning.**
   If the architecture says something should be ergonomic and an
   operator repeatedly trips over it, take that seriously. One trip
   is a receipt, not an automatic rewrite. It becomes architecture
   after it repeats, survives a probe, and still cannot be explained
   by missing skill, docs, or probes. Law 2 still holds: a complaint
   is an observation.

8. **Preserve provenance.**
   Observation → evidence → interpretation → decision →
   implementation → validation. Otherwise six months later you retain
   the strange constraint but forget why it exists.

9. **Promote abstractions slowly.**
   Local fix → repeated pattern → demonstrated generality →
   lower-layer primitive. This is the Gesso / Palette / Cyan
   discipline. See **Layer gate** below. Semantic stubs promote
   the same way, inside their layer, on harness evidence.

10. **Every development loop ends with another research question.**
    That question is the seed of the *next* spiral.

11. **Win condition for this spiral.**
    Every load-bearing question for this problem has a disposition,
    and you have produced a bounded roadmap *for this problem*
    (explicit non-goals, first implementation slice if the answer is
    build). Reject, "stays in Palette/Cyan/Lab", and `STUB` count as
    done. Congratulations.

---

## What you emit (pick one or more)

A spiral produces artifacts. It does not start a PR unless Develop
was in scope **and** a closed contract exists.

| Artifact | When |
|---|---|
| Disposition | always: `BUILD` / `STUB` / `PARK` / `REJECT` / `PALETTE` / `CYAN` / `LAB` / `GESSO LATER` |
| Semantic stub | the capability is real, a dedicated body is unearned, and a cheap honest lowering exists (see below) |
| Parking-lot note | architecture or research that is not this week's code and is not yet runnable |
| Experiment design / prototype | a cheap question is still open |
| Closed recipe | someone is about to implement (owner, permitted files, tests, non-goals) |
| Remaining questions | the next spiral's start |

A roadmap is not a `/goal`. A `/goal` is a closed recipe. A semantic
stub is not a `/goal`.

Precision follows the mode:

    exploration      semantic elasticity
    architecture     explicit boundaries
    experiment       operational definitions
    implementation   exact contracts
    verification     zero ambiguity

---

## Layer gate

Before anything moves **down** a layer, all three must be yes:

    Generality        would a developer want this without Cyan or Palette?
    Composability     can it be expressed without agents, context windows,
                      playbooks, or Palette sessions?
    Ecosystem leverage
                      does centralizing it cut duplicated glue across
                      real components of that layer?

Any no: keep it in the layer that hurt. Being low-level is not
enough. Scarcity of Gesso fallout from a Cyan trial is a successful
boundary.

Semantic stubbing retains **inside a layer**. A Cyan recipe that
uses a Gesso-shaped noun does not become a Gesso export.

---

## Semantic stubbing (RPDO retain)

Preserve the idea semantically. Exercise it procedurally. Let
demonstrated repetition earn machinery.

Do not build machinery merely to preserve an idea.
Stub semantics. Never stub truth.

Semantic stubbing applies when the intended subsystem can be written
as methodology and process in markdown **before dedicated code for
that subsystem is laid down**. The markdown is the cheaper version
you buy: an agent or operator runs the process with tools that
already exist. Helpers and `src/` named after the idea wait on
promotion. If the idea cannot live as a followed document plus
existing machinery, it is `PARK`, a packet, or a closed recipe — not
a stub.

This is the third retain:

| Retain | Keeps | Pays |
|---|---|---|
| `PARK` / parking-lot note | prose | rediscovery risk |
| Packet / `@test_broken` | a failure and a flip condition | a test |
| `STUB` | a capability you can run | the cheapest truthful lowering |

```
IDEA → SEMANTIC IDENTITY → EXPLICIT CONTRACT → TEMPORARY LOWERING
     → REAL USAGE → EVIDENCE → REPEATED PATTERNS → DEDICATED MACHINERY
```

### Eligibility

Mandatory, all four:

1. Authority boundaries can be stated honestly.
2. A useful version can be executed today through existing machinery.
3. The temporary lowering preserves correctness.
4. The eventual body can plausibly remain behind the same conceptual
   model.

Typically also true: identifiable problem, clear architectural role,
dedicated implementation expensive relative to value, use will teach
more than speculation, the concept will recur, losing it is costly.

Do not stub because it sounds interesting, because success is
undefined, because the abstraction is still mush, because correctness
requires machinery that does not exist, because the lowering would
violate authority or safety, because a runtime primitive is required
now, or because the name would imply support the layer does not have.

### Minimum stub

```
name
purpose / problem
operations          (what a caller may say)
honesty status
current lowering    (the markdown + existing tools/recipes that run it)
failure behavior
flip / promotion evidence
provenance          (why it exists, date, owner)
```

That is the birth certificate. Friction ledgers, extension seams,
and performance characterization accrue with use. They are not
required to mint the stub.

Honesty status is fail-closed and layer-specific:

- **Gesso:** implemented (tested) / packeted / skip-or-green /
  `LoweringNotImplemented`
- **Cyan / Palette / WGE:** `supported` (a test in the owning repo
  passes) / `agent_executed` / `partially_supported` /
  `experimental` / `unsupported_pending_X`

`verified` is forbidden unless a harness in the owning repo produced
the receipt. `supported` on a Gesso engine call requires a passing
test. `agent_executed` is illegal on a Gesso engine call.

Gesso stubs live in `docs/` (and, if needed, a skip-or-green,
`@test_broken`, or `LoweringNotImplemented` pin). They do not
appear in `names(Gesso)`, in parked-empty modules, or as methods
that return `nothing`. The first commit of a stub is the markdown.

### Maturity (descriptive)

    SEMANTIC → PROCEDURAL → ASSISTED → NATIVE

NATIVE plus receipts is hardened. Some stubs stay procedural
forever. That is success. Do not use these words aspirationally.

### Friction is observation

```
repeated procedure  →  friction ledger (observation)
                    →  harness / receipt (evidence)
                    →  helper / primitive
```

Agent cognition is a complaint (law 2). Agent confidence is not
evidence. One occurrence is not an extension seam. Usage reveals
the axis: observed variation → stable axis → seam.

### Promotion

Earn dedicated machinery when one or more hold **and** the evidence
is a receipt or a test in the owning repo:

1. The same procedure repeats.
2. The same failure recurs.
3. Latency or performance requires it.
4. Correctness would improve with machine enforcement.
5. The lowering leaks backend complexity to the caller.
6. Repeated projects demand the same extension seam.

Then write a closed recipe. Still subject to the layer gate before
anything moves down.

### Layer rider

| Layer | Legal lowering | Forbidden lowering |
|---|---|---|
| Cyan / Palette / WGE | agent recipe, framework doc, generic tools, fake workers | pretending native runtime; `verified` without a harness |
| Gesso | docs + test/packet + existing `src/` | new public names; empty-module APIs; agent-in-the-kernel |
| WGE runtime-critical (render, input, collision, animation execution, core gameplay) | dedicated machinery | procedural paper-over |
| Gesso runtime-critical (`decode!`, KV pages, greedy ids, CUDA/Lava seams) | cheapest truthful *engine* lowering | agent procedure as the engine |

### Already-run examples

**Gesso fused decode** (`Gesso/docs/goals/PHASE10E_FUSED_DECODE.md`)

- Identity (markdown): warmed `decode!` reuses Session scratch;
  pages remain the cache; attention reads a contiguous `1:K`
  prefix; a page-table kernel is later.
- Cheaper version: the 10E recipe reuses `gather_kv!` and
  `:attn_gemm` — existing engine machinery, not a new attention
  subsystem.
- Honesty: gather-on-read is the contract; FlashAttention is an
  escalation (packet).
- Promotion evidence: alloc ceilings, seqlen independence, G2
  factor next to 0.692×.
- Dedicated thing not yet laid down: page-table / fused attention,
  and only if `fork` bytes still hold.

**Cyan Phase A / MAO v0** (`CYAN_ELASTIC_ORCHESTRATION_v0.md`,
`PHASE_A_WORKTREE_REDUCER.md`; Gesso seam pointer
`Gesso/docs/research/CYAN_ELASTIC_ORCHESTRATION.md`)

- Identity: unfinished work is the durable object; workers are
  disposable `WorkerLease`s over a single-writer `WorkTree`.
- Lowering: fake workers, jobs 1/5/8, journal replay; no LLM.
- Honesty: Gesso `src/Agents.jl` stays empty; the Gesso file is a
  seam pointer, not a `/goal`.
- Promotion evidence: a real reducer that survives kill-every-worker;
  then a Gesso Phase 12 recipe, after the layer gate.

**WGE Reforge** (planning identity; dedicated body unearned)

- Identity: `wge.reforge.{inspect, baseline, model, plan, migrate,
  verify}`.
- Lowering: Cyan + repository tooling + build/profile + existing
  WGE validation + framework procedure.
- Honesty: `agent_executed` / `experimental` until a WGE test owns
  the operation.
- Promotion: dependency archaeology, behavior capture, renderer
  comparison, and similar, when those procedures repeat across
  modernization projects.
- The renderer, input, and collision stay machinery.

---

## Loop sizes

**Micro** — a known bug.

    research bug → plan repair → implement → test → learn

**Meso** — one subsystem question (a Gesso phase, a Palette primitive).

    research question → closed recipe → one slice → operate → learn

**Macro** — a problem space.

    research space → architecture → build subsystem → operate →
    study emergence → redesign

Default to meso. Micro is for repairs. Macro is for "what is this
whole thing." Do not redesign the stack to fix a telemetry field.
A `STUB` is a meso retain: the meaning is stable, the body is later.

---

## Time-box and collisions

Cap a spiral: the load-bearing questions named at kickoff, at most
one throwaway prototype, one written disposition. Overflow is the
next spiral. A semantic stub counts as the disposition; it does not
also consume the prototype slot.

Do not reopen a spiral another lead already closed unless you have
**new primary evidence**. Agent confidence is not evidence. Citation
means having read it. Testimony yields to transcripts and logs. No
performance claim without a receipt from the repo under test.

---

## Handoff into Develop

RPD output does not implement the product.

Implementation starts from a closed contract in the owning repo
(`docs/goals/` in Gesso; the equivalent work-item form elsewhere).
The contract names owner, permitted files, tests, and non-goals.

A `STUB` is markdown plus existing machinery. Growing dedicated
code under it waits on a closed recipe. Reusing an existing engine
path as the cheaper version (10E: `gather_kv!`) is Develop of that
recipe, not the first commit of the intended subsystem.

Until a closed contract exists, stay in Research and Plan.
