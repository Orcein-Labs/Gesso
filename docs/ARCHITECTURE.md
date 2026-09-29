# Harpe Architecture Map

> **Upkeep rule: a stale map is worse than no map. Update this file in the
> same PR that changes what it maps.** Verified against the tree by the BONES
> sprint; if this note and the tree disagree, the tree wins and this file is
> a bug (file it).

Harpe is a Julia-native semantic ML and agent execution runtime. Canon:
`docs/Harpe_Stack.md` (Roman-numeral sections, referenced as §NNN). This map
is the fast orientation layer; the canon is the law.

## Layer map: module → canon → phase

| Module | Path | Governs (canon) | Phase |
|---|---|---|---|
| `Harpe` (root) | `src/Harpe.jl` | include order, package docstring | — |
| `Log` | `src/logging.jl` | §XLII receipts vocabulary, §LXX fallback recording, §XLIX events | 0 ✓ |
| `versions` | `src/versions.jl` | §LXIX determinism, North Star §34 schema versioning | 0 ✓ |
| `backends` | `src/backends.jl` | §XX capabilities, §XXI tiers, backend contract + lowering stubs | 0 ✓ (draft) |
| `errors` | `src/errors.jl` | §LXX explicit failure, North Star §22 taxonomy + `APPROXIMATION_BUDGET_EXCEEDED` | BONES ✓ |
| `receipts` | `src/receipts.jl` | §XLII audit records + sink interface | BONES ✓ |
| `Semantics` | `src/Semantics/` | §I, §XI, §XIII — meaning vocabulary, traits/types/metadata discipline | 1 |
| `ModelIR` | `src/ModelIR/` | §VII, §VIII — semantic model graph (not a tensor graph) | 1 |
| `Parameters` | `src/Parameters/` | §XI semantic tensor classes; §LVIII forbids Gradient/OptimizerState | 1/3 |
| `Operators` | `src/Operators/` | §XII dispatch-as-execution, §XIX operator semantics | 1/2 |
| `Lowering` | `src/Lowering/` | §XXII–XXIII backend routing; mixed-backend is ordinary | 4 |
| `Inference` | `src/Inference/` | §XXIX engine, §XXX prefill/decode split, KV manager hooks | 5 |
| `Runtime` | `src/Runtime/` | §XXXII scheduler, §XXXIII+ agent mechanism; mechanism-only (§XLIII) | 5+/12 |
| `Profiling` | `src/Profiling/` | §XLIX metrics, §L performance failure taxonomy | 6 |
| `Planning` | `src/Planning/` | §XVII execution synthesis, §XIX memory planning, §LXI policies | 7 |
| `Autotune` | `src/Autotune/` | §XXVI; KV program §7 — realization search (kernel-first) | 9 |
| `Representation` | `src/Representation/` | §XIV materialization, §XV quantization-as-lowering; KV program §5 lattice | 10 |
| `Agents` | `src/Agents/` | §XXXIII–XLII agent primitives; JSON is wire format, not ontology | 12 |
| `CAPI` | `src/CAPI/` | §XLVI–XLVII libharpe; adoption surface, not architecture | 16 |

## Research layer

| Document | Role |
|---|---|
| `docs/Harpe_Stack.md` | **CANON.** Everything else is subordinate. |
| `docs/research/KV_MEMORY_PROGRAM.md` | KV/working-memory research program; extends §XXXI/§X/§LIX; feeds Phases 5/9/10. |
| `Harpe_musings.md` (root) | The dangerous notebook. Parking lot — promote deliberately, never wholesale. |
| `docs/DECISION_PACKETS.md` | Open architecture escalations (decision-packet format). Resolved INTO canon, never in code. |
| `docs/Harpe_Stack_old.md` | Predecessor vision, archived. Superseded where they disagree. |
| `docs/Native_Julia_Kernel_Autotuning_North_Star_README.md` | Companion project spec (standalone autotuner). Harpe's Phase 9 consumes it. |

## Process surface

| Path | Role |
|---|---|
| `AGENTS.md` | **Binding agent charter** (transcribes §LXXI/§LXXII/§LXX). |
| `.github/ISSUE_TEMPLATE/work-item.md` | §LXXI work-item format. |
| `.github/PULL_REQUEST_TEMPLATE.md` | §LXXII receipt-as-PR. |
| `scripts/` + `Makefile` | `make test / bench / format / format-check / freeze`. |
| `benchmark/results/*.tsv` | Regression corpus (accrues from every bench run). |
| `test/` | Per-area test files, included from `runtests.jl`; dependency-law test lives there. |

## Hard fences (violations are law violations, not style choices)

* Training is out of scope permanently (§LVIII).
* `libs/` is not part of the package; backends are extensions (§VII).
* No silent fallbacks — `@hfallback` or it did not happen (§LXX).
* No dependency without editing the dependency-law test (§VII).
* L3+ memory is a seam, not a component (KV program §8, §XLIII).
