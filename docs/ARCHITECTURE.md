# Gesso Architecture Map

> **Upkeep rule: a stale map is worse than no map. Update this file in the
> same PR that changes what it maps.** Verified against the tree by the BONES
> sprint; if this note and the tree disagree, the tree wins and this file is
> a bug (file it).

Gesso is a Julia-native semantic ML and agent execution runtime. Canon:
`docs/Gesso_Stack.md` (Roman-numeral sections, referenced as §NNN). This map
is the fast orientation layer; the canon is the law.

Stack names (§I, §XLIII): **Gesso** = mechanism (this package, formerly Harpe);
**Palette** = expression (palette.jl, formerly NeuraJL); **Cyan** = harness /
policy (internal: NIRA); **Lava** = Vulkan substrate.

## Layer map: module → canon → phase

| Module | Path | Governs (canon) | Phase |
|---|---|---|---|
| `Gesso` (root) | `src/Gesso.jl` | include order, package docstring | — |
| `Log` | `src/logging.jl` | §XLII receipts vocabulary, §LXX fallback recording, §XLIX events | 0 ✓ |
| `versions` | `src/versions.jl` | §LXIX determinism, North Star §34 schema versioning | 0 ✓ |
| `backends` | `src/backends.jl` | §XX capabilities, §XXI tiers, backend contract + lowering stubs | 0 ✓ (draft) |
| `errors` | `src/errors.jl` | §LXX explicit failure, North Star §22 taxonomy + `APPROXIMATION_BUDGET_EXCEEDED` | hardening ✓ |
| `receipts` | `src/receipts.jl` | §XLII audit records + sink interface (thread-safe sink, §CIX identity) | hardening ✓ |
| `Semantics` | `src/Semantics/` | §I, §XI, §XIII, **§CIX** — meaning vocabulary; workload dispatch types (`PrefillWorkload`/`DecodeWorkload`) | 1 — item A ✓ |
| `ModelIR` | `src/ModelIR/` | §VII, §VIII, **§CIX** — immutable semantic composition graph (Embedding/RMSNorm/RoPE/Attention/SwiGLU/Block/Model; structural identity via Tuple composition) | 1 — item B ✓ |
| `Parameters` | `src/Parameters/` | §XI, **§CIX** — §XI family types + `frozen` trait + metadata fields; §LVIII forbids Gradient/OptimizerState | 1/3 — item A ✓ |
| `Operators` | `src/Operators/` | §XII, **§CIX** — operators are functions (owned by `backends.jl`); dispatch methods on `SemanticTensor × workload`; `cpu.jl` = CPU reference math (§LXXV, Float64; `quantize!`/`dequantize!` still decline) | 1 — item C ✓; 2 — item A ✓ |
| `Lowering` | `src/Lowering/` | §XXII–XXIII backend routing; mixed-backend is ordinary | 4 |
| `Inference` | `src/Inference/` | §XXIX engine, §XXX prefill/decode split, KV manager hooks; Phase 2 slice: `reference_prefill` + `reference_generate` CPU oracle (§LXXV) | 2 — items B, C ✓; 5 |
| `Runtime` | `src/Runtime/` | §XXXII scheduler, §XXXIII+ agent mechanism; mechanism-only (§XLIII) | 5+/12 |
| `Profiling` | `src/Profiling/` | §XLIX metrics, §L performance failure taxonomy | 6 |
| `Planning` | `src/Planning/` | §XVII execution synthesis, §XIX memory planning, §LXI policies | 7 |
| `Autotune` | `src/Autotune/` | §XXVI; KV program §7 — realization search (kernel-first) | 9 |
| `Representation` | `src/Representation/` | §XIV materialization, §XV quantization-as-lowering; KV program §5 lattice | 10 |
| `Agents` | `src/Agents/` | §XXXIII–XLII agent primitives; JSON is wire format, not ontology | 12 |
| `CAPI` | `src/CAPI/` | §XLVI–XLVII libgesso; adoption surface, not architecture | 16 |

## Research layer

| Document | Role |
|---|---|
| `docs/Gesso_Stack.md` | **CANON.** Everything else is subordinate. |
| `docs/research/KV_MEMORY_PROGRAM.md` | KV/working-memory research program; extends §XXXI/§X/§LIX; feeds Phases 5/9/10. |
| `docs/Gesso_musings.md` | The dangerous notebook. Parking lot — promote deliberately, never wholesale. |
| `docs/DECISION_PACKETS.md` | Architecture escalations (decision-packet format). Packets 1–2 resolved into §CIX; remaining packets follow the same rule: resolved INTO canon, never in code. |
| `docs/Harpe_Stack_old.md` | Predecessor vision (Harpe-era), archived. Superseded where they disagree. |
| `docs/Native_Julia_Kernel_Autotuning_North_Star_README.md` | Companion project spec (standalone autotuner). Gesso's Phase 9 consumes it. |

## Process surface

| Path | Role |
|---|---|
| `AGENTS.md` | **Binding agent charter** (transcribes §LXXI/§LXXII/§LXX). |
| `.github/ISSUE_TEMPLATE/work-item.md` | §LXXI work-item format. |
| `docs/goals/` | Sprint goals handed to implementing agents. Phase 1: `PHASE1_SEMANTIC_CORE.md` (landed). Phase 2: `PHASE2_CPU_ORACLE.md` (landed). Phase 3: `PHASE3_FIRST_IMPORT.md`. |
| `.github/PULL_REQUEST_TEMPLATE.md` | §LXXII receipt-as-PR. |
| `scripts/` + `Makefile` | `make test / bench / format / format-check / freeze`. |
| `benchmark/results/*.tsv` | Regression corpus (accrues from every bench run). |
| `test/` | Per-area test files, included from `runtests.jl`; dependency-law test lives there. |

## Object model (§CIX)

Phase 1 implements this encoding. It does not choose another.

| Object | Encoding | Owns |
|---|---|---|
| Operator | function; methods are implementations | `Operators` |
| ModelIR node | immutable value; composition of primitives | `ModelIR` |
| SemanticTensor / Parameter | family TYPE + optimization TRAITS + runtime METADATA | `Parameters` (+ `Semantics` vocabulary) |
| Workload | `PrefillWorkload` / `DecodeWorkload` types; §XVI names are tags | dispatch surface; receipts |
| Receipt id | process-local `UInt64` until persistence/swarm schema bump | `receipts.jl` |

The four Phase 1 modules stay contract-only until a Phase 1 work item fills them (`test/test_empty_core.jl`).

Foundation-hardening status: receipts, errors, versions, backends contract,
freeze/bench/CI machinery, and the test-side correctness laboratory are
hardened and pinned by tests (`test/`, per-area). The four Phase 1 modules
are the frontier; the toy fixture pack (`test/fixtures/toy/`) is the
laboratory Phase 2's CPU oracle will consume.

## Hard fences (violations are law violations, not style choices)

* Training is out of scope permanently (§LVIII).
* `libs/` is not part of the package; backends are extensions (§VII).
* No silent fallbacks — `@gfallback` or it did not happen (§LXX).
* No dependency without editing the dependency-law test (§VII).
* L3+ memory is a seam, not a component (KV program §8, §XLIII).
* Semantic-core encoding is §CIX. Do not collapse Operator / ModelIR / SemanticTensor onto one encoding.
