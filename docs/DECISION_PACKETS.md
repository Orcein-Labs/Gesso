# Gesso Decision Packets

Escalations from the Foundation Hardening sprint (2026-09-29), in the
§6 decision-packet format of that sprint's goal. These are the points
where local engineering stops and a stronger architectural reasoning
pass should take over.

Consumption rule: a packet is resolved by writing the decision INTO canon
(`docs/Gesso_Stack.md` or the owning research program), never by silently
implementing one option in code. Until resolved, the safe move is to
keep the seam undecided.

PACKET 1 and PACKET 2 are RESOLVED INTO CANON (`docs/Gesso_Stack.md`
§CIX, 2026-09-29). The option analysis is kept as the record of the choice.
Code still does not implement either resolution — Phase 1 work items do.

PACKET 3 is **OPEN**. It was promoted here on 2026-10-04 out of
`docs/goals/PHASE10D_FOUNDATION_HARDENING.md` §P-1 so that the one
architecture question this repo is currently sitting on lives with the
other packets rather than inside a sprint goal that has already closed.

---

## PACKET 1 — Workload/execution-phase vocabulary

**QUESTION**
Should Gesso define an explicit `ExecutionPhase`/workload enum (or type
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
  different workloads; Gesso may choose distinct layouts, kernel families,
  quantization, cache strategies, scheduling, memory policies for:
  *prompt prefill, batch-1 decode, batched decode, long-context execution,
  structured/tool output, swarm inference*. This is a taxonomy of
  CATEGORIES, not a frozen enum.
- Canon §XXX (referenced by ARCHITECTURE.md as "prefill/decode split") is
  the Inference-engine phase; the split is a Phase-5 concern.
- `Gesso_musings.md` muses about workload tags but is parking-lot, not law.
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
  side channels (the thing Gesso hates).

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

## PACKET 3 — `storage::Any` and what it costs (the 10D packet, P-1)

**OPEN.** Promoted here 2026-10-04 from `docs/goals/PHASE10D_FOUNDATION_HARDENING.md`
(§P-1) so it lives with the other packets instead of inside a sprint goal.
Until it is resolved into canon, the three `@test_broken` gates in
`test/test_type_stability.jl` STAY broken and no field type changes.

**QUESTION**
§CIX says storage pointer and device residency are METADATA, never type
parameters, and every §XI family is written that way: `storage::Any`.
Should that stand as the end state, or should the ARRAY TYPE holding a
tensor become a trait-equivalent carried as ONE type parameter `S` on the
§XI families and on the Session engine buffers? If neither, what is the
third shape — and what is the measured cost of leaving it?

**Why does it need deciding now?**
10E (2026-10-04) took the warmed toy2 CPU `decode!` host allocation from
87,008 B to 12,848 B — an 8.0x cut against the reproducible floor of
10,912 B — **without changing a single field type**. It did it by passing
storage arrays into the op bodies so the arithmetic infers while the fields
stay `::Any`. (10F left the CPU number alone by design and cut the CUDA
side instead, 195,808 → 81,904 B.) 10E's mechanism was correct and it is
now essentially exhausted. It is also the measurement that makes this
packet decidable: the boxing is the dominant remaining term, and the
counterfactual below shows it is the FIELD TYPE, not the code, that is
holding it. The question is no longer "is there a leak" (there is) but
"does canon keep the leak on purpose, and for how long".

### CURRENT FACTS (measured on this box, this tree, 2026-10-04)

Julia 1.12.6, cachyos-x8664, CPU fixture toy2, `context_length = 128`.
Every number below is reproduced by

    julia --project=. scripts/p1_storage_any_cost.jl

That script is EVIDENCE, not a gate: it changes no repository file, asserts
nothing, and is not wired into `scripts/test.jl`. The gates that guard this
packet's invariant stay where they are — the three `@test_broken` in
`test/test_type_stability.jl`, still broken.

**1. The baseline this packet is about.**

```
toy2 CPU warmed decode! @allocated            12,544 B   (first session)
same, min of 5 fresh identical sessions       10,912 B
allocations in one warmed decode!             164 in 10,368 B
```
(The 12,848 B printed in `test/test_decode_scratch.jl`'s header and in the
10E receipt is the same quantity measured on a differently-ordered run;
10,912 B is the reproducible floor and is the number used below. Both are
under the 16 KiB CPU gate, so the gate is unaffected either way.)

**2. What one `::Any` read costs.** Microbenchmark, 1,000 iterations:

```
call on a ::Any field (e.g. length(b.x))      32.0   B/dispatch
call on a typed field                              0.0 B/dispatch
a[i] through a ::Any field                    39.824 B/read
a[i] through a typed field                     0.032 B/read
```

**3. The counterfactual: the FIELD TYPE decides it, not the code.** Three
structs, byte-identical read patterns (`x.storage[1, 1]`), one field-type
difference:

```
real  Gesso.Activation   (storage::Any)     @inferred  -> NOT INFERRED
fake  FakeAct{Matrix{Float64}}              @inferred  -> INFERRED
fake  FakeActAny        (storage::Any)      @inferred  -> NOT INFERRED
read `s[1]` alloc, typed field                        0.032 B/call
read `s[1]` alloc, ::Any field                        32.0   B/call
```
Nothing about the read differs between rows 1 and 2 except the field
declaration. That is the whole argument for Option A, and it is also the
whole argument against it: the fix is a type-system change to the object
model, not a local optimisation.

**4. Where the remaining 10,368 B sits.** `Profile.Allocs`, `sample_rate =
1.0`, attributed to the first frame under `src/`:

```
   B    allocs  site                                    what the ::Any is
1760      6    session.jl:789  bt = tensors.blocks[bi]   getindex on ::Any
 544     16    session.jl:801  kh.storage[1,:,:]          ::Any field + @views
 496      3    session.jl:786  embedding_lookup!(..., tensors.embedding, ...)
 464      2    session.jl:372  dim = s.model.embedding.dim   getproperty chain
 448      1    session.jl:367  haskey(s.tensors, :final_rms)  dynamic call
 352      4    session.jl:790  rmsnorm!(..., bt.attn_rms, ...)
 352      4    session.jl:791  matmul!(..., bt.wq, ...)
 352      4    session.jl:792  matmul!(..., bt.wk, ...)
 352      4    session.jl:793  matmul!(..., bt.wv, ...)
 352      4    session.jl:869  matmul!(..., bt.wo, ...)
 352      4    session.jl:871  rmsnorm!(..., bt.ffn_rms, ...)
 352      4    session.jl:872  matmul!(..., bt.wgate, ...)
 352      4    session.jl:873  matmul!(..., bt.wup, ...)
 352      4    session.jl:875  matmul!(..., bt.wdown, ...)
 344      9    session.jl:366  Activation(; storage=reshape(s.h[...], ...))
 256      7    session.jl:788  for (bi, blk) in enumerate(model.blocks)
 224      6    session.jl:777  view(ws.k_gather.storage, ...)
 224      6    session.jl:778  view(ws.v_gather.storage, ...)
 160     10    session.jl:797  rope!(..., ws.pos_buf, ...)
                                                             = 8,088 B  (78%)
```

**8,088 B of 10,368 B — 78% — sits on lines that read through a `::Any`
field or pass a `::Any` argument.** Split honestly, because four of those
lines do something else as well:

```
purely ::Any boxing                          6,752 B   65%   (789, 786, 372,
                                                          367, 790-793, 869,
                                                          871-873, 875, 788,
                                                          797)
::Any PLUS a view/reshape on the same line   1,336 B   13%   (801, 366, 777,
                                                          778)
                                                -------
                                                8,088 B  78%
```

The remaining 2,280 B is not P-1 and is named here so this packet cannot be
used to claim it: 400 B is `_engine_receipt`'s `nameof(typeof(s.model))`
(session.jl:422 — receipt fields are `::Any` by §XLII design, not by
accident), and the rest is `cpu.jl` / `kv_manager.jl` internals
(`_cpu_matmul!` 240 B, `_cpu_rmsnorm_storage!` 192 B,
`_cpu_embedding_lookup_storage!` 176 B, `append_kv!` 256 B).

**5. The blast radius is 44 field declarations**, not 11:

```
§XI families with storage::Any                 11   (src/Parameters/Parameters.jl)
KVPage.storage / PagedKVManager.prototype::Any  2  (src/Inference/kv_manager.jl)
Session.{model,tensors,tokenizer,h,ws}::Any    5   (src/Inference/session.jl)
DecodeWorkspace.<26 buffers>::Any             26   (src/Inference/session.jl)
                                               ---
                                               44
```

**6. The gates.** `grep -rn "@test_broken" test/` returns exactly THREE
executable ones, all in `test/test_type_stability.jl`:

* line 46 — `@inferred reference_prefill(toy2, ts, prompt) isa Matrix{Float64}`
* line 60 — `@inferred decode!(session) isa Int` (toy2)
* line 85 — `@inferred decode!(s) isa Int` (llama_micro)

The suite's total of 8 broken is these 3 plus 5 named skips. (`docs/goals/
PHASE10F_CUDA_DECODE_ALLOC.md:721` says "still Broken (7)"; the grep is the
authority and the count there is wrong.)

These gates are LOAD-BEARING, not cosmetic. `@test_broken` ERRORS if the
expression starts passing. That is the mechanism that forces this packet
open on purpose if any future change accidentally resolves it — which is
the same law 10E and 10F worked under.

### OPTIONS

**OPTION A — one storage type parameter `S` (what 10H's withdrawn receipt
claimed).** `ProjectionWeight{S}` … `AdapterDelta{S}`, `storage::S`, plus
`Session{S, Ts, M}` and `DecodeWorkspace{S}`. Shape / seqlen / batch stay
fields.

* Benefits: `@inferred decode!` and `@inferred reference_prefill` go green
  by construction, not by hoisting; the three gates flip; the ~8 KB
  residual disappears because there is no dynamic dispatch left to box.
* Costs: `S` must be the FULL array type (`Array{Float64,3}` CPU,
  `CuArray{Float32,3}` on device, whatever Lava's is) because rank differs
  per backend — so the parameter carries rank, eltype and device at once.
  Every construction site, `haskey`, `similar`, the §CIX structural-identity
  `==`, `nameof(typeof(...))` in the receipt, `_session_zeros_like`, the
  `to_device` seam and every `GessoCUDAExt` method all gain an `S`
  dimension. Specialization cost is real: one compiled method instance per
  (array type × backend), which multiplies against a host allocation floor
  10F measured at 688 B for a single `@cuda` launch — and 10H's own premise
  line counts ~9,300 launches per token. It amends a canon clause
  (§CIX) rather than implementing one.
* Does NOT close the door on anything, which is its appeal: §CIX keeps
  device residency and storage pointer as metadata; only the container
  type becomes compile-time.

**OPTION B — keep `storage::Any`, keep hoisting.** Do what 10E did, by
hand, everywhere: bind each `tensors.blocks[bi]`, `bt.wq`, `bt.wk`,
`bt.wv`, `bt.wo`, `bt.wgate`, `bt.wup`, `bt.wdown`, `s.model.embedding.dim`
to a `local` at the top of the block loop and pass storage arrays down.

* Benefits: zero canon change, zero object-model change, zero new type
  parameters. The 8,088 B line-items above are individually removable.
* Costs: it is a manual, uncheckable discipline. Nothing in the type system
  prevents the next `matmul!(cpu, q, normed, bt.wq, wl)` from re-boxing, and
  the three `@test_broken` gates stay broken forever, so the leak has no
  machine-checked end state. Every hoist is a maintenance hazard: a future
  block added to `model.blocks` silently reintroduces the boxing, and the
  only detector is an allocation ceiling nobody has moved in a year. It
  also does not reach the four mixed sites (801, 366, 777, 778), where the
  view/reshape would still be constructed per token.

**OPTION C — a typed engine-side bundle; the §XI families stay `::Any`.**
Leave `storage::Any` on all 11 semantic families (so §CIX's object model
and its identity law are untouched), and give the Session a concretely-typed
`LayerTensors` / `Tensors` struct whose FIELDS are the families, so
`s.tensors.blocks[bi].wq` infers at the call site even though `.storage`
inside the family still does not.

* Benefits: it targets precisely the 8,088 B that is actually hot — the
  reads are on `Session.tensors` and `Session.model`, both of which are
  engine state, not semantic identity. The semantic families keep `::Any`
  and keep `haskey`-style dynamic construction. Much smaller blast radius
  than A: two structs, not 44 fields.
* Costs: it introduces a SECOND parallel shape for the tensor bundle,
  which is exactly the "stacked hierarchy" that 10H was told not to build;
  `to_device` must rebuild the bundle when it converts storage, or the two
  bundles drift; and it still does not make `@inferred decode!` green
  (the return value reads `s.h[row, :]` and `argmax` over a `::Any`), so
  two of the three gates would stay broken.

**OPTION D — leave it, and say so in canon.** Accept ~8 KB/token of host
allocation on the decode path as the declared price of §CIX, write that
sentence into §CIX, and stop spending sprint items on it.

* Benefits: honest, zero risk, and it ends the recurring re-litigation —
  every allocation sprint since 10D has rediscovered this same 8 KB.
* Costs: it is a permanent ~78% overhead on the decode step's host
  allocation, and `docs/research/SPEED_FLOOR.md` names type stability as
  the next ladder rung after the alloc trilogy. Declaring it accepted is a
  decision that should be made once, explicitly, with the number above.

### REVERSIBILITY

Option A is the expensive one to undo. Parameterising the §XI families
touches 44 declarations plus every construction, comparison and extension
method; the diff is large, the identity law (`==` on shape) is load-bearing
for §XIII regression tests, and reverting a partial application leaves a
half-parametric object model that satisfies neither §CIX nor Option B.
Option C is cheaper to undo (two structs) but re-opens if `to_device` ever
has to convert a bundle, which is a normal operation. Options B and D are
both fully reversible — B because it changes no declaration, D because a
canon sentence is a deletion.

The asymmetry matters: A costs the most and B costs nothing, so the only
reason to prefer A over B is that B has no machine-checked end state. If
canon rejects A, B should be paired with a gate (an allocation ceiling that
actually moves) so the boxing stays visible.

### RECOMMENDATION

None, as an agent. AGENTS.md §8 reserves packet resolution for canon, and
the choice is genuinely a canon trade-off: Option A buys a green
`@inferred` gate at the price of a 44-declaration object-model change and a
specialization multiplier on top of a launch count (10H counts ~9,300
`@cuda`/CUBLAS launches per token) that canon has already named KEEP;
Option D buys certainty at the price of a permanent, declared 8 KB.

What the evidence DOES settle, and what should be written into canon
whichever option is chosen:

1. The 8.0x cut 10E got came entirely from hoisting, with zero field-type
   change. That result is not an argument that `storage::Any` is cheap — it
   is an argument that the hoisting is nearly exhausted (78% of what is
   left is one dynamic read per weight per block).
2. The counterfactual in fact 3 isolates the cause to the field declaration
   and nothing else. No amount of further local rewriting reaches it.
3. 78% of a 10,912 B/token budget is not a rounding error. If Option D is
   chosen it must be written down as a DECLARED cost with this number, not
   left as an unexplained residual.

**NOT A DECISION. No field type changes. The three `@test_broken` gates stay
broken.**

---

## Status

PACKET 1 and PACKET 2 are RESOLVED INTO CANON (`docs/Gesso_Stack.md` §CIX).
No code in the repo implements either resolution. Phase 1 work items
live in `docs/goals/PHASE1_SEMANTIC_CORE.md` and implement
PrefillWorkload / DecodeWorkload and the semantic family types.
Receipt identity stays process-local UInt64 until the persistence/swarm
phase that owns the schema bump.

PACKET 3 (`storage::Any`, the 10D packet P-1) is **OPEN**. 10E and 10F cut
warmed `decode!` host allocation 8.0x under it and measured what is left
(8,088 B of 10,368 B, 78%, still boxing through `::Any`). The three
`@test_broken` gates in `test/test_type_stability.jl` remain Broken and
load-bearing. Resolution happens in §CIX; do not resolve it by
parameterising a family.
