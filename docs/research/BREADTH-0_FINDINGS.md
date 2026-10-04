# BREADTH-0 FINDINGS

**Batch:** BREADTH-0 — Universal Model Doorway
**Status:** implemented; regression green except one PRE-EXISTING broken WIP test file
**Canon:** §VIII (model import), §XI (semantic parameters), §LXX (explicit failure), §XIII (specialization), §XXX (prefill/decode)
**Companion:** `test/test_breadth0.jl`, `test/test_tokenizer_protocol.jl`

One line: **stop teaching Gesso individual models; teach Gesso what a model
architecture IS.**

---

## 0. THE HEADLINE

Before BREADTH-0, adding a model family meant editing the inference engine.
After, it means writing a data adapter. Measured on the meta-metric (§XVIII):

> Phi-3 was added end-to-end — fused `qkv_proj`, fused `gate_up_proj`, a
> declared per-KV-group packing — and required **zero edits** to
> `session.jl`, `Inference.jl`, `kv_manager.jl`, or any operator. Its config
> was parsed, its tensors mapped, and it ran `reference_prefill` and
> `reference_generate` on the CPU F64 oracle with a different tensor layout
> and the same set of primitives.

---

## 1. ASSUMPTIONS UNCOVERED

Each row is a Llama-specific assumption that existed in the code before this
batch, and what replaced it.

| # | Assumption (before) | Where | After |
|---|---|---|---|
| 1 | `model_type == "llama"` decides what a model *is* | `load_llama_config` | `ArchitectureSpec` — the engine consumes meaning, never the string. `family` is provenance only |
| 2 | Nine hardcoded tensor spellings bound to nine fields | `_LAYER_TENSOR_NAMES` | Canonical semantic identity + a `FamilyParamMap` per family |
| 3 | Parameter identity is derived by **string surgery** on `model.layers.$i.` | `materialize_llama` | `SemanticParamId`; identity is MEANING, spelling is one adapter's choice |
| 4 | One external tensor = one parameter | `materialize_llama` | One external tensor can carry **N** canonical identities (fused QKV → q/k/v) |
| 5 | Fusion packing would be guessed from a name | (did not exist) | `FusedQKVLayout` is a **declared** layout; Gesso never guesses a packing |
| 6 | `rope_scaling must be null` | `load_llama_config` | `RoPEPolicy` with `:none` / `:linear` / `:llama3`; the default stays bit-identical |
| 7 | Positional semantics live at the **call site** (`theta` kwarg) | `Inference.jl`, `session.jl` | The policy **travels with the model** (`tensors.rope`) |
| 8 | Unsupported model ⇒ `error("…not Llama-shaped")` | `load_llama_config` | Capability lattice: the family imports, then fails at the **named operation** |
| 9 | Tokenizer == GPT-2 byte-level BPE | `gpt2_tokenizer.jl` | `Tokenizer` protocol; GPT-2 is one implementation, metaspace is another |
| 10 | Compatibility is discoverable only by running and hitting a trace | — | `import_report` + a **generated** `compatibility_matrix` |

### The two that mattered most

**#4 (fusion).** The old assumption was so deep it was invisible: it was
implicit in the shape of `_LAYER_TENSOR_NAMES`. Phi-3 stores Q, K and V in
one tensor. Under the old model there was no way to express that, so the
"obvious" fix would have been a `Phi3Runtime` — exactly what §VIII forbids.
Making identity canonical and packing declared solved it with no new type.

**#7 (positional).** `theta` was already threaded cleanly as a keyword, which
made this look like a non-issue. It was one: a model with scaled RoPE would
have run **silently unscaled** — a representation change with no error, a
direct §LXX violation. `Session` now refuses a scaled policy outright, and
the CPU oracle implements it properly.

---

## 2. THE CAPABILITY LATTICE

An architecture family is no longer accepted or rejected. A model travels as
far into Gesso as the implemented semantics allow, and stops at a **named
capability**:

```
checkpoint parsed          READY      tensor names mapped         READY
architecture recognized    READY      semantic parameters built   READY
tokenizer identified      READY      positional policy parsed    READY
                                          ↓
                            attention primitive unsupported  ✕
```

Example — what a Qwen2 config reports today:

```
Execution capabilities:
    rmsnorm                   READY
    attention                 READY
    matmul                    READY
    swiglu_ffn                READY
    qk_norm                   NOT IMPLEMENTED
    rope_linear               NOT IMPLEMENTED

Failure boundary: qk_norm
```

Before, that same config produced: *`model_type "qwen2" is not "llama" —
not Llama-shaped, refusing`*. It told the user nothing actionable and died at
the door. Now it tells them exactly which semantic operation is missing, and
`reference_generate` accepts `model_type = :qwen2` so the phrase is gone.

---

## 3. WHAT LANDED, BY PASS

| Pass | Status | Evidence |
|---|---|---|
| A ArchitectureSpec + adapters | ✅ | `architecture_spec.jl`, `arch_adapters.jl` |
| B Canonical param identity | ✅ | `semantic_params.jl`, `materialize_architecture.jl` |
| C Tokenizer protocol + 2 impls | ✅ | `tokenizer_abstract.jl`, `tokenizer_protocol.jl` |
| D Positional/RoPE policy | ✅ | `RoPEPolicy`, `rope_inv_freq`, `Session` fail-closed |
| E Meaning ≠ materialization | ✅ | `bindings` record; typed seam for future materialization |
| F Capability declaration | ✅ | `capabilities`, `required_semantics` |
| G Family fixtures | ✅ | five configs + a real fused-QKV checkpoint |
| H Second family through exec | ✅ | Phi-3 → CPU F64 prefill/decode/generate |
| I Import report | ✅ | `import_report` |
| J Compatibility matrix | ✅ | `compatibility_matrix` / `compatibility_table` |
| K Regression | ⚠️ | green except pre-existing WIP — see §6 |

`materialize_llama` is now a **thin delegation** to the family-agnostic binder.
There is ONE name-map implementation, not two.

---

## 4. THE META-METRIC, ANSWERED

*"What would it take to add architecture family #6?"*

1. Write a `<Family>Adapter` with `parse_config` and `param_map`.
2. Declare any semantics it needs that Gesso lacks as capabilities.
3. Add a conformance fixture (a config; a checkpoint only if it needs one).
4. Implement only the genuinely new **operators** it requires — `:qk_norm`,
   `:dense_ffn`, `:sliding_window_attention`, `:moe_routing` — each of which
   already has a named slot in `required_semantics`.

No engine edit. No new parameter type. No second name map.

---

## 5. DELIBERATE NON-DECISIONS

* **No new dependency.** The dependency law (§VII) is untouched. SentencePiece
  was implemented natively rather than added, because a tokenizer is
  transport machinery and `JSON` already covers its file format.
* **`Lowering.jl` stays empty.** The route is not a fusion planner.
* **No operator was added.** Pass D reuses `rope!` with frequencies rather than
  introducing a `scaled_rope!`; a policy is a *value*, not a new kernel kind.
* **`supports(backend, cap)` was not extended.** `_implemented_capabilities`
  in `import_report.jl` declares what Gesso *has*; backend reachability stays
  the backend's own answer. Conflating them would have made the matrix lie.

---

## 6. KNOWN LIMITATIONS AND OPEN QUESTIONS

1. **A pre-existing WIP test file fails and is not mine.**
   `test/test_decode_scratch.jl` (untracked, plus its `include` in
   `runtests.jl`) calls `Gesso.Inference._workspace_pointers`, which **does not
   exist anywhere in `src/`**. It fails on a clean stashed tree too. It needs a
   decision: land the implementation, or revert the WIP.
2. **No authoritative tokenizer fixture.** The metaspace fixture is SYNTHETIC,
   authored here: the 256 `<0xNN>` byte fallbacks plus a merge table, with
   expected ids derived from the published algorithm. They are **not** copied
   from a production SentencePiece model, and this document will not pretend
   otherwise. A conformance fixture traced from a real published tokenizer
   remains open — it needs a checkpoint, which is an ops fact (§LXXVI).
3. **Session does not execute scaled RoPE — on purpose.** The CPU oracle does;
   the engine refuses loudly rather than guessing. Threading the policy through
   `Session` is a one-line seam once `Session` carries it, but it needs a
   struct field and its own recipe.
4. **`make format` is not clean on master.** It reformats 14 unrelated files.
   This batch reverted them to keep the diff reviewable; the repo needs a
   separate format-cleanup decision.
5. **Executed-backend coverage is unproven for the new path.** Everything here
   is verified on the CPU oracle. CUDA/Lava reachability for a second family is
   a CUDA/Lava recipe, not a claim this batch makes.
6. **`model_type` strings are still a closed registry.** A family with no
   adapter is a *transport* refusal naming the family — correctly distinct
   from a capability failure, but still a whitelist at the very edge. Widening
   it to a name→adapter convention is future work.

---

## 7. THE RECEIPT (§LXXII)

```
what changed   · 7 new files in src/Inference/, 2 new test files, 1 new
                 fixture; 6 existing files touched (2 functional, 4 wiring)
why            · make model breadth an adapter exercise, not an engine rewrite
tests          · test_breadth0.jl + test_tokenizer_protocol.jl: 233 assertions
                 green; export inventory green; Llama import 91/91 unchanged
numerical      · unscaled RoPE path UNCHANGED BY CONSTRUCTION (the oracle
                 evaluates the same literal expression when inv_freq===nothing);
                 scaled policy provably changes output (asserted)
benchmarks     · none run. No performance claim is made. BREADTH-0 is
                 architectural and its own rule forbids judging it on speed.
compile-time   · not measured
memory         · not applicable (no new data structures on the decode path)
hardware       · CPU oracle; CUDA and Lava suites run and stay green
workload       · toy2, llama_micro, SmolLM2-135M conformance fixtures
model          · llama, qwen2, gemma, mistral, phi3, phi (configs);
                 phi3 executed end-to-end
backend        · cpu (executed), cuda/lava (regression only)
```