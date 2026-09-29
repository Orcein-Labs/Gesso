# /goal PHASE 1 — SEMANTIC CORE

**Status:** LANDED (Buffy, 2026-09-29). Next: `docs/goals/PHASE2_CPU_ORACLE.md`.

**For:** Buffy (mechanical implementation)
**From:** Grok (encoding owner)
**Canon:** `docs/Gesso_Stack.md` §CIX, §XI–§XIII, §VIII, §XXX, §LXXIV
**Map:** `docs/ARCHITECTURE.md` (Object model)
**Packets:** 1 and 2 are already law in §CIX. Do not reopen them.

---

## Start condition

Ready-room is done:

- empty-core fence green
- toy fixture pack landed (`test/fixtures/toy/`, `test/toyfixtures.jl`)
- `make test` / `make format-check` green on the tree you inherit

If the fence is still the empty-module version, this goal is what lifts it.
If ready-room is unfinished, stop and wait.

## One-sentence objective

A tiny reference model (the toy fixture `toy2`) is expressible entirely
through the semantic core, using the encoding already decided in §CIX.

## Encoding (do not choose another)

| Object | Encoding |
|---|---|
| Operator | a function; methods are implementations |
| ModelIR node | an immutable value; composition of primitives |
| SemanticTensor / Parameter | family TYPE + `frozen` TRAIT + runtime METADATA |
| Workload | `PrefillWorkload`, `DecodeWorkload` types |
| Receipt id | still process-local `UInt64` (not this sprint) |

§CIX "WHAT PHASE 1 IMPLEMENTS" / "WHAT PHASE 1 DOES NOT IMPLEMENT" is the
stop line. Read it before the first edit.

## What this sprint is not

- CPU/reference math, logit oracle, generation harness (Phase 2)
- filling `expected_logits.toml`
- CUDA, Lava, safetensors, a real checkpoint importer
- `ExecutionPhase` / `WorkloadKind` mega-enum
- global receipt ids
- `LlamaRuntime`-shaped types
- representation/quantization lattice (no `Quantized{T,Q4}` stack)
- mutable IR handles, Dict-ontology for semantic roles
- storage as the meaning of a tensor (bytes stay `nothing` / unset)

---

## Work items (sequence; land A before B before C)

### A — Vocabulary types

**Objective.** The semantic families, the `frozen` trait, and the two
workload types exist and can be constructed.

**Permitted files**

```
src/Semantics/Semantics.jl
src/Parameters/Parameters.jl
src/Gesso.jl
test/test_semantics.jl
test/test_parameters.jl
test/runtests.jl
docs/ARCHITECTURE.md
```

**Interfaces to expose**

- `Semantics`: `PrefillWorkload`, `DecodeWorkload` (singleton types, §XII / §XXX / §CIX).
- `Parameters`: abstract `SemanticTensor`; concrete families from §XI:

  `ProjectionWeight`, `KVCache`, `EmbeddingTable`, `ExpertWeight`,
  `FrozenParameter`, `QuantizedParameter`, `Activation`,
  `TemporaryWorkspace`, `RoutingState`, `DecodeState`, `AdapterDelta`

- `frozen(::Type)` / `frozen(x)` Holy-trait hook: true for weight-like
  families (`ProjectionWeight`, `EmbeddingTable`, `FrozenParameter`,
  `ExpertWeight`, `AdapterDelta`); false for the rest. That is the only
  trait this item adds.
- Metadata is a small struct or named fields (`shape`, `storage=nothing`).
  Volatile facts are not type parameters.
- Re-export the public names from `Gesso` so `using Gesso` sees them.

**Invariants**

- No `Gradient`, `OptimizerState`, or AD identifiers in `src/` (existing
  training fence stays).
- `QuantizedParameter` exists as a family type; there is no quantization
  lattice and no quantized trait beyond the type's existence.
- `next_receipt_id()` is still `UInt64`.
- `ExecutionPhase` and `WorkloadKind` remain undefined.

**Tests.** Construction, `frozen` on every family, workload types are
distinct, `storage` defaults unset. Include from `test/runtests.jl`.

**Performance.** N/A.

**Artifact.** Types you can construct; fence still forbids ModelIR/Operators
contents until B and C (or update the fence per-module as each item lands —
do not leave a lying fence).

---

### B — ModelIR composition

**Objective.** The toy architecture in `test/fixtures/toy/model.toml` is
buildable as an immutable ModelIR graph of primitives. No per-family
runtime type (`LlamaRuntime` etc.).

**Permitted files**

```
src/ModelIR/ModelIR.jl
src/ModelIR/*.jl          # split files if the single file gets large
src/Gesso.jl
test/test_modelir.jl
test/runtests.jl
docs/ARCHITECTURE.md
```

You may *read* `test/fixtures/toy/` and `test/toyfixtures.jl`. You may
add a builder in `test/` that uses ModelIR constructors. Do not make the
TOML schema a Gesso type. Fixture `kind` strings (`attention`, `mlp`)
stay fixture data; the builder maps them onto primitives.

**Interfaces to expose**

Immutable primitives sufficient for §VIII *and* for `toy2`:

- `Embedding`
- `RMSNorm`
- `RoPE`
- `Attention` (at least `n_heads`; `n_kv_heads` may default to `n_heads`)
- `SwiGLU` / dense FFN (hidden dim)
- a `Block` (or equivalent) that composes primitives
- a `Model` (or equivalent) that is an ordered composition: embedding +
  blocks. Identity is structural.

Rewrites construct a new graph. No mutator that edits a node in place.

**Invariants**

- Two independently built copies of `toy2` compare equal.
- Changing `n_heads` or block order yields a different model.
- No `LlamaModel` / `QwenModel` / `MistralModel` types.
- ModelIR nodes do not own KV contents or workspace buffers.

**Tests.** Primitive construction, `toy2` round-trip from the fixture
builder, equality, immutability (`setfield!` fails).

**Performance.** N/A.

**Artifact.** `toy2` as ModelIR. Still no operator execution.

---

### C — Operators + Phase 1 exit + fence lift

**Objective.** Operators are functions. The existing lowering-stub
vocabulary dispatches on semantic types × workload. The empty-core fence
is rewritten to pin the encoding instead of emptiness. Phase 1 exit is
met: `toy2` is expressible entirely through the semantic core.

**Permitted files**

```
src/Operators/Operators.jl
src/backends.jl            # only if you must add methods / keep stubs coherent
src/Gesso.jl
test/test_operators.jl
test/test_empty_core.jl
test/test_phase1_exit.jl
test/runtests.jl
docs/ARCHITECTURE.md
README.md
scripts/freeze.jl          # add new test files to the curated list
```

**Interfaces**

- Do **not** create a second `matmul!`. The lowering stubs in
  `src/backends.jl` remain the functions. Add methods that dispatch on
  `SemanticTensor` families × `PrefillWorkload`/`DecodeWorkload`.
- Un-specialized calls (`rmsnorm!(cpu, nothing)` etc.) still throw
  `LoweringNotImplemented` — existing backend-interface tests stay green.
- Specialized calls also throw `LoweringNotImplemented` this sprint
  (Phase 2 fills CPU math). They must identify `op` and backend the same
  way the stubs do.
- Operator vocabulary is exactly the current stub list:

  `rmsnorm!`, `rope!`, `softmax!`, `swiglu!`, `matmul!`,
  `embedding_lookup!`, `quantize!`, `dequantize!`

  Do not add ops because canon *might* want them.

**Fence rewrite (`test/test_empty_core.jl`)**

Replace "modules export nothing" with:

- the four modules export the §CIX names this sprint added
- `ExecutionPhase` / `WorkloadKind` still undefined
- `PrefillWorkload` / `DecodeWorkload` **are** defined
- receipt ids still `UInt64`
- training-boundary scan unchanged
- packets file still says `RESOLVED INTO CANON`

**Phase 1 exit test (`test/test_phase1_exit.jl`)**

One test that:

1. loads the toy fixture (data)
2. builds ModelIR from it
3. constructs the semantic tensors the model implies (storage unset)
4. names the operators the blocks would call
5. does **not** run a forward pass and does **not** touch expected logits

That is "expressible through the semantic core." Execution is Phase 2.

**README.** Phase 1 row: types exist, toy model expressible, no execution yet.

**Performance.** N/A.

**Artifact.** Fence pins encoding. Exit test green. `make test` green.

---

## Cross-item invariants

- Zero new hard dependencies. Stdlib already in the allowlist is fine
  (`Dates`, test-env `Random` / `TOML`). Adding a dep means editing the
  dependency-law test *and* a justification; this sprint should not need one.
- `libs/` is untouched.
- No silent fallback. Lowerings decline with `LoweringNotImplemented`.
- Do not resolve architecture by inventing a type §CIX did not name.
  If you hit a missing primitive that `toy2` does not need, skip it.
  If you hit a missing primitive that `toy2` *does* need, add that one
  primitive and record it in the receipt — do not grow a zoo.
- Formatter: `make format` before close.
- Receipt at close (§LXXII): what changed · why · tests · numerical delta
  (none expected) · compile-time impact (note if type explosion appears) ·
  hardware CPU · workload n/a · model `toy2` · backend none.

## Escalation (stop and write a packet)

Write a new packet in `docs/DECISION_PACKETS.md` and stop the item if:

- you need a type parameter beyond `frozen` to make dispatch work
- you need ModelIR to be mutable
- you need a third workload type
- the toy fixture's `kind` strings cannot map onto the primitive list
  without inventing a per-family runtime

Do not "just pick one" in code. That is how Packet 1/2 existed.

## Exit checklist

- [ ] A, B, C landed (or a written split with A green and the rest as
      follow-up work items in this file)
- [ ] `make test` green
- [ ] `make format` run
- [ ] empty-core fence rewritten, not deleted
- [ ] Phase 1 exit test green
- [ ] ARCHITECTURE map matches the tree
- [ ] README Phase 1 status truthful
- [ ] receipt in the PR / close note
- [ ] no CPU oracle, no expected logits, no CUDA/Lava
