# Harpe Decision Packets

Escalations from the Foundation Hardening sprint (2026-09-29), in the
§6 decision-packet format of that sprint's goal. These are the points
where local engineering stops and a stronger architectural reasoning
pass should take over.

Consumption rule: a packet is resolved by writing the decision INTO canon
(`docs/Harpe_Stack.md` or the owning research program), never by silently
implementing one option in code. Until resolved, the safe move is to
keep the seam undecided.

Both packets below are RESOLVED INTO CANON (`docs/Harpe_Stack.md` §CIX,
2026-09-29). The option analysis is kept as the record of the choice.
Code still does not implement either resolution — Phase 1 work items do.

---

## PACKET 1 — Workload/execution-phase vocabulary

**QUESTION**
Should Harpe define an explicit `ExecutionPhase`/workload enum (or type
lattice) now, and if so, what is in it?

**Why does it need deciding now?**
The receipt vocabulary (§XLII) and the benchmark corpus both need a stable
notion of "what kind of action/workload this was" to remain useful as data
accumulates. The longer the repo accrues rows/receipts with ad-hoc workload
tags, the more painful a later normalization becomes. The next sprint
(Phase 1, semantic core) will force this question within weeks.

**CURRENT FACTS**
What canon/repo already constrain:
- Canon §XVI ("Phase-specific materialization"): prefill and decode are
  different workloads; Harpe may choose distinct layouts, kernel families,
  quantization, cache strategies, scheduling, memory policies for:
  *prompt prefill, batch-1 decode, batched decode, long-context execution,
  structured/tool output, swarm inference*. This is a taxonomy of
  CATEGORIES, not a frozen enum.
- Canon §XXX (referenced by ARCHITECTURE.md as "prefill/decode split") is
  the Inference-engine phase; the split is a Phase-5 concern.
- `Harpe_musings.md` muses about workload tags but is parking-lot, not law.
- The repo currently has NO workload type anywhere (deliberately: this
  sprint refused to freeze one — see the gate).
- Receipts are `Any`-typed per field, so no existing persisted type blocks
  any choice.

**OPTION A**
Enum (`@enum WorkloadKind prefill batch1_decode batched_decode ...`)
- Benefits: cheap, closed-world, pattern-matchable, trivially stable as a
  receipt/CSV value; matches §XVI's fixed list; impossible to misuse.
- Costs: open-world hazard — §XVI's list is a snapshot; swarm/agent phases
  (12+) and the KV program (hinting, re-prefill) may need kinds the enum
  lacks; enum extension renumbers nothing but still needs a taxonomy
  review; a closed enum can push legitimate categories into stringly-typed
  side channels (the thing Harpe hates).

**OPTION B**
Free-form symbols/strings with a registry convention (like log `event`:
snake_case, documented in the owning module)
- Benefits: open-world; zero premature commitment; mirrors the existing
  structured-event vocabulary precedent.
- Costs: no exhaustiveness checking; registry drifts (exactly the stale-doc
  class of bug this repo keeps killing); comparison/case analysis on
  Symbols is weaker; receipts lose closed-world classification.

**OTHER OPTIONS**
- Two-level: coarse enum (prefill/decode/tool/swarm/maintenance) + open
  metadata struct with details. More faithful to §XVI, but defines a
  composite object whose shape IS an architecture decision.
- Defer entirely to Phase 5 and let receipts carry `nothing`. Costs a bit
  of corpus usefulness; fully reversible.

**REVERSIBILITY**
What becomes expensive if chosen incorrectly?
- A wrong enum is expensive once receipts/corpus rows persist with it
  (migration of append-only history). A wrong registry is expensive in a
  different currency: silent drift and normalization pain. The two-level
  option is hardest to reverse (it bakes both).
- The DECISION is medium-cost to reverse today (nothing persisted uses it)
  and expensive to reverse after Phase 5 ships.

**RECOMMENDATION**
Defer the type decision; adopt the §XVI six-category LIST as the interim
documented vocabulary (symbols, documented in ARCHITECTURE.md) only if the
next sprint needs workload tags in receipts before Phase 5. Confidence:
moderate. The genuine choice — enum vs registry vs composite — deserves
the reasoning pass.

**RESOLUTION (2026-09-29, §CIX)**
Two-level. PrefillWorkload and DecodeWorkload are TYPES (dispatch from
Phase 1/2; required by §XII and §XXX). The six §XVI names are documented
TAGS for receipts and plans until a lowering actually dispatches on them.
No ExecutionPhase / WorkloadKind mega-enum. Promoting a tag to a type is
a work item. Option A (closed enum of all six) is rejected; Option B
(free-form forever) is rejected for the prefill/decode cut.

---

## PACKET 2 — Receipt identity and parent references

**QUESTION**
What is the global identity scheme for receipts — process-local counters
(today) vs time-ordered ULID-style ids vs UUIDv7 vs content/hybrid — and
what does `parent_dependency` reference exactly (a receipt id, a task id,
or an agent-local sequence)?

**Why does it need deciding now?**
`next_receipt_id()` is live, atomic, and already proven monotonic across
threads; every receipt minted from now on carries these ids. The §XLII
field list includes `parent_dependency` ("receipt id of the parent
action") — as soon as the first cross-process or persisted consumer
appears (swarm coordination, §XLII's stated purpose), the id scheme and
the parent-reference semantics stop being a local detail.

**CURRENT FACTS**
- Canon §XLII: receipts are auditable records; the field list names
  `parent_dependency` as a first-class field; receipts serve "swarm
  coordination" — i.e. multiple emitters, possibly multiple hosts.
- The sink docstring currently says: "receipt ids are atomic and
  independent of this lock; ids stay strictly monotonic across threads"
  and `next_receipt_id` is documented "process-local; global ids are a
  later concern".
- Current ids are `UInt64` counters starting at 0 per process: NOT unique
  across processes, NOT time-ordered, NOT self-describing.
- No receipts have been persisted to disk yet (serialization is a later
  phase), so nothing in the corpus embeds today's scheme.

**OPTION A**
Keep process-local UInt64 counters; define `parent_dependency` as
process-local receipt id; add (host, boot/session, counter) tuple when
persistence/swarm arrives.
- Benefits: maximal simplicity now; zero premature global-identity design;
  the tuple can be introduced as a schema bump (RECEIPT_SCHEMA_VERSION).
- Costs: every persisted receipt is ambiguous without session context;
  parent references across sessions/processes need the tuple anyway;
  consumers must all handle composite keys.

**OPTION B**
ULID/UUIDv7-style ids (time-ordered, globally unique) minted per receipt.
- Benefits: receipts are self-identifying anywhere, any time; sort order
  matches causality approximately; parent references stay single-valued.
- Costs: needs a spec decision (which ULID variant, monotonicity within a
  millisecond, entropy source) — that is real design; ids are no longer
  dense; more allocations per receipt on the hot path; stdlib-only
  constraint means hand-rolling (dependency law §VII) which must itself
  be tested.

**OTHER OPTIONS**
- Hybrid: keep UInt64 counter for in-memory sinks; mint global ids only at
  serialization boundary (schema bump then). Defers all cost without
  closing any door.
- Task-scoped identity: parent_dependency references task ids, not receipt
  ids (§XXXIII agent tasks are the real causal parents; receipts are
  emissions of tasks). Semantically strongest claim; biggest redesign.

**REVERSIBILITY**
What becomes expensive if chosen incorrectly?
- Once receipts are persisted and cross-referenced (parent_dependency
  chains), changing the id scheme is a data migration on append-only
  history — the exact class of thing §LXIX forbids doing silently.
- Choosing Option A now does NOT close Option B (the schema-bump escape
  hatch exists and nothing is persisted yet). Choosing B now commits the
  hot path and adds untested machinery before any consumer exists.

**RECOMMENDATION**
Option A (or the hybrid variant) for now, with the explicit acknowledgment
that the schema-bump path is the pre-planned escape hatch; revisit at the
phase that first persists receipts. Confidence: high that deferring is
safe; low-to-moderate that A is the right end state. The END-STATE identity
scheme (and task-vs-receipt parent semantics) belongs to the reasoning
pass.

**RESOLUTION (2026-09-29, §CIX / §XLII)**
Hybrid, which is Option A plus the pre-planned bump. Process-local UInt64
now; parent_dependency is a receipt id in the same process; `task` stays
its own field. Global time-ordered ids (ULID / UUIDv7 / host-session-counter)
are chosen in the work item that first persists receipts or crosses
process — by bumping RECEIPT_SCHEMA_VERSION, never by silent reinterpret
(§LXIX). Option B in core now is rejected (no consumer, hot-path cost,
stdlib-only ULID would itself be a design). Task-scoped parent_dependency
is rejected: §XLII already names it as a receipt id.

---

## Status

Both packets are RESOLVED INTO CANON (`docs/Harpe_Stack.md` §CIX).
No code in the repo implements either resolution. Phase 1 work items
live in `docs/goals/PHASE1_SEMANTIC_CORE.md` and implement
PrefillWorkload / DecodeWorkload and the semantic family types.
Receipt identity stays process-local UInt64 until the persistence/swarm
phase that owns the schema bump.
