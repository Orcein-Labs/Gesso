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
| `Parameters` | `src/Parameters/` | §XI, **§CIX** — §XI family types + `frozen` trait + metadata fields; §LVIII forbids Gradient/OptimizerState. **10H (§CIX amendment): the eleven families carry ONE storage type parameter `S`** (`ProjectionWeight{S}` … `AdapterDelta{S}`), inferred from `storage`, with `S = Nothing` for unset. Shape/seqlen/batch remain fields; no second axis (§XIII); `frozen` stays a trait. | 1/3 — item A ✓; 10H ✓ |
| `Operators` | `src/Operators/` | §XII, **§CIX** — operators are functions (owned by `backends.jl`); dispatch methods on `SemanticTensor × workload`; `cpu.jl` = CPU reference math (§LXXV, Float64; `rmsnorm!`/`rope!` carry `eps`/`theta` keyword defaults per §LXXVI; `quantize!`/`dequantize!` still decline) | 1 — item C ✓; 2 — item A ✓; 3 — item A ✓ |
| `Lowering` | `src/Lowering/` | §XXII–XXIII backend routing; mixed-backend is ordinary | 4 + 8 (seam proven by the GessoCUDAExt and GessoLavaExt extensions — CUDA.jl and Lava/Vulkan backends as weakdep extensions, §LXXVII/§LXXXI; routing lands later) |
| `Inference` | `src/Inference/` | §XXIX engine, §XXX prefill/decode split, KV manager hooks; Phase 2 slice: `reference_prefill` + `reference_generate` CPU oracle (§LXXV); Phase 3: GQA (repeat-for-contraction, cache at `n_kv_heads`), optional `final_rms`, threaded `eps`/`theta` (item A); Llama import — `load_llama` path (`llama_import.jl`: config validation, safetensors reader with exact f64 upcast, closed name map; HF 0-based **and** fixture 1-based layer origins detected from `q_proj.weight`, mixes refused, `*.rotary_emb.inv_freq` the only ignored leftover — fail-closed otherwise (10C); JSON is the ONE sanctioned third-party dep + Mmap, §LXXVI) (item B); GPT-2 byte-level BPE tokenizer (`gpt2_tokenizer.jl`: published algorithm, loud refusals) (item C); SmolLM2 real-model gate (`GESSO_SMOLLM2_DIR` skip-or-green, no downloads) (item D); Phase 4: backend-generic interpreter — `backend=` keyword (default CPU, bit-identical), explicit no-copy law (host Array under a non-CPU backend is `ERR_INVALID_PLAN`), device buffers via `similar`, CPU scalar loops pinned vs device broadcast/CUBLAS forms (§LXXVII); Phase 5: paged KV manager (`kv_manager.jl` — Magenta §9.5 step 1: pages are the cache, gather-on-read, per-page provenance `layer/kind/start_pos/filled`, typed `ERR_RESOURCE_LIMIT` at context exhaustion) + the Session engine (`session.jl` — `prefill!`/`decode!`/`generate`, greedy `_greedy_id`, required `eos_token_id`, streaming `on_token` callback, string prompts via the tokenizer; ids equal the oracle, CPU logits atol=0); Phase 10: device fast paths behind declared capability probes (`Gesso.supports(backend, :argmax / :attn_gemm)`) — CUDA decode! runs argmax ON the device (host receives ONE Int per token, the (vocab,) logits row never crosses back; GPUArrays argmax is first-index-tie deterministic) and the attention contraction (QKᵀ + PV) is one flat device GEMM over the SAME gathered scratch per site (reshape(CuArray) views; pages + gather unchanged, no page-table kernel); Lava keeps its own contraction and the full-row host argmax (§LXXXIII gates G1–G3, PHASE10_SPEED_FLOOR.md) | 2 — items B, C ✓; 3 — items A, B, C ✓, D skip-or-green; 4 — items A–C ✓, D skip-or-green; 5 — items A, B, C, D ✓; 10 — items B, C ✓ |
| `Runtime` | `src/Runtime/` | §XXXII scheduler, §XXXIII+ agent mechanism; mechanism-only (§XLIII) | 5+/12 |
| `Profiling` | `src/Profiling/` | §XLIX metrics, §L performance failure taxonomy; Phase 6: `engine_report`/`print_report` (stable machine-readable attribution from engine receipts), `kv_footprint`/`page_footprint` (derived from the page table via `Inference.kv_bytes`); Phase 7: `unique_kv_bytes` (live storage counted once per distinct page array across managers — the declared-share win metric); no CUDA in this module | 6 — items A, B ✓; 7 — item C ✓ |
| `Planning` | `src/Planning/` | §XVII execution synthesis, §XIX memory planning, §LXI policies | still empty (Phase 7 was prefix share; Phase 8 was the Lava extension) |
| `Autotune` | `src/Autotune/` | §XXVI; KV program §7 — the realization-search LOOP: `Candidate`/`TuneResult`, `register!` (registration order), `search!` (correctness gate before timing; compile+warmup untimed, §XXXIII), `select` (cache hit replays the cached result with `cache_hit = true` and EMITS NOTHING — a hit is not a decision, `TuneResult.cache_hit` is the consult-site signal; Phase 10G), `invalidate!`/`invalidate_all!`, one `:autotune_select` receipt per SEARCH (schema unchanged); imports neither CUDA nor Lava (§VII) — extensions register candidates in `__init__` (Phase 9: `:cublas_mul` + `:generic_mul` for CUDA `matmul!`, gated at the CUDA op atol; the op consults and dispatches to the cached winner, §LXXXII) | 9 — items A, B, C, D ✓; 10G ✓ |
| `benchmark/compare_eager.py` | `benchmark/` | §LXXXIII gate G2 — eager-PyTorch reference timings (first-token / warmed) for the named-model factor: external python3 binary, local weights only (`local_files_only=True`, never downloads), no torch.compile, JSON or key-value output; `--probe` is the REAL torch gate (dry `import torch` + `import transformers`, exit 3 when missing — `--help` is not a probe, Phase 10B); invoked by the G2 block in `benchmark/runbenchmarks.jl` when snapshot + CUDA + torch exist, named skip otherwise — torch is NEVER a Project.toml dep (§VII) | 10 — item D ✓; 10B — item A ✓ |
| `Representation` | `src/Representation/` | §XIV materialization, §XV quantization-as-lowering; research seed: `docs/research/REPRESENTATION_PROGRAM.md` | 10 |
| `Agents` | `src/Agents/` | §XXXIII–XLII agent primitives; JSON is wire format, not ontology | 12 |
| `CAPI` | `src/CAPI/` | §XLVI–XLVII libgesso; adoption surface, not architecture | 16 |

## Research layer

Regime I checkpoint-layout repair (2026-10-07): `load_safetensors` decodes
the format's C/row-major byte ordering into Julia array coordinates through
`_safetensors_array`. `test/test_safetensors_layout.jl` constructs raw bytes
independently of the round-trip writer; `test/reference_hf.py` and
`test/test_hf_parity.jl` compare real-model CPU logits and greedy IDs with
independent eager Hugging Face/PyTorch. Imported positional metadata is retained through `load_llama`, Session and
device transfer; half-split RoPE is explicit. `_attention_heads!` reuses score
scratch per query head and replaces the mathematically invalid shared-softmax
contraction, including the former flat cross-head CUDA GEMM. The previous
speed-factor rows describe the old semantics and do not certify this repair.
This campaign follows `docs/goals/REGIME_I_MATURITY.md`; skips do not graduate
an arm.

| Document | Role |
|---|---|
| `docs/Gesso_Stack.md` | **CANON.** Everything else is subordinate. |
| `docs/research/README.md` | **Index.** Sequencing law: boring stack first, exotic later. Every research program is linked from here. |
| `docs/research/KV_MEMORY_PROGRAM.md` | Magenta Memory Part I — KV/working-state as a lowering; extends §XXXI/§X/§LIX; feeds Phases 5/9/10/11. |
| `docs/research/KV_MEMORY_PROGRAM_part2.md` | Magenta Part II — topology / residency / schedule. Exploratory. After a real KV manager. |
| `docs/research/REPRESENTATION_PROGRAM.md` | Gauge-compiled / caged weights. Phases 7 (candidate) / 10 (host). Not Phase 5. |
| `docs/research/CYAN_TRIAL_GESSO_FALLOUT.md` | Cyan trial metal detector. Promotion filter. Not a work item. |
| `docs/research/CYAN_ELASTIC_ORCHESTRATION.md` | Seam pointer: Cyan MAO freeze. Not a Gesso /goal. `src/Agents.jl` stays empty. |
| `docs/research/ROADMAP_NOW.md` | Living phase map. Not a Buffy goal. |
10F LANDED (`PHASE10F_CUDA_DECODE_ALLOC.md`, 2026-10-03): the CUDA decode broadcast wrappers are gone (device kernels behind `CuArray` specializations of the SAME core copy helpers), §2 row 3 updated again. 10G LANDED (`PHASE10G_AUTOTUNE_RECEIPT.md`, 2026-10-03): miss-only Autotune receipts (a cache hit is not a decision) — SmolLM2 CUDA warmed `decode!` 1,269,616 B → 942,128 B, under item D's 1 MiB gate. 10H LANDED (`PHASE10H_TYPE_STABILITY.md`, 2026-10-03): §CIX amended so storage is ONE type parameter; three `@inferred` gates green; §2a.2 type-stable decode landed. |
| `docs/research/EXOTIC_CAPABILITY_MASTER_LEDGER.md` | Capability mine (meta-primitives, meta-tools). Parked. Not a /goal. |
| `docs/Gesso_musings.md` | The dangerous notebook. Parking lot — promote deliberately, never wholesale. |
| `docs/DECISION_PACKETS.md` | Architecture escalations (decision-packet format). Packets 1–2 resolved into §CIX; remaining packets follow the same rule: resolved INTO canon, never in code. |
| `docs/Harpe_Stack_old.md` | Predecessor vision (Harpe-era), archived. Superseded where they disagree. |
| `docs/Native_Julia_Kernel_Autotuning_North_Star_README.md` | Companion project spec (standalone autotuner). Gesso's Phase 9 consumes it. |

## Process surface

| Path | Role |
|---|---|
| `AGENTS.md` | **Binding agent charter** (transcribes §LXXI/§LXXII/§LXX). |
| `.github/ISSUE_TEMPLATE/work-item.md` | §LXXI work-item format. |
| `docs/goals/` | Sprint goals. Phases 1–9 and living 10/10B/10C/10D/10E/10F/10G/10H landed (10F 2026-10-03: `PHASE10F_CUDA_DECODE_ALLOC.md` — CUDA decode broadcast-wrapper removal via device-storage specialization, restored 256 KiB CUDA alloc gates; 10G 2026-10-03: `PHASE10G_AUTOTUNE_RECEIPT.md` — resolved 10F's escalation packet, miss-only Autotune receipts, SmolLM2 CUDA 1 MiB gate green; 10H 2026-10-03: `PHASE10H_TYPE_STABILITY.md` — closed 10D packet P-1, one storage type parameter, the three `@inferred` gates green). G2 factor 1.452×. Canon §LXXXIII parked. |
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

Regime I transport boundary: safetensors headers now reject duplicate keys,
invalid dimensions and unindexed/overlapping payload ranges before tensor
allocation. Shard index assignments must match the actual containing file.
Config and tokenizer JSON use the same duplicate-key rejecting parser;
tokenizer IDs are unique and contiguous. No additional package dependency.

Regime I positional contract: omitted theta uses imported RoPE metadata; an
explicit positive finite theta overrides its base. Linear scaling divides
inverse frequencies by its factor. CPU and primary Lava honor default, linear
and Llama3 frequency tables. Lava reduces the same Float32 phase constants
from host position metadata before device trig/rotation; tensor execution stays
on Vulkan. Independent HF gates cover real-model prefixes 17/33/65/129 and
rotary positions through 32768; these are not full-model 8k context certification.

Runtime now owns fixed round-robin request scheduling over separate Sessions.
It acquires exclusive nonblocking leases, refuses duplicate ownership and
advances one ordinary decode per request per round. Cancellation is checked
before prefill and between tokens; terminal results retain committed IDs.
CPU independent tasks are tested; primary Lava uses a single host owner for
logical concurrent sequences. This adds no tensor fusion or continuous batching.

Regime I containment: Session validates actual/declared parameter shapes and
model grouping before workspace allocation; token IDs are checked before
lookup. Public Llama/config/safetensors/GPT2 loading errors are typed invalid
plans with original offender detail. Engine failures invalidate ready state,
unknown operation/interruption errors are typed runtime failures, and a new
generate resets state for recovery. Batch callback failures also invalidate
the affected Session without suppressing other sequences.

Receipts now distinguish prefill, all decode work and the first decode step;
TTFT is absent if no token was emitted. Replay digests are named FNV-1a over
64-bit little-endian IDs, with committed IDs retained for partial failures.
Requests report actual backend and native dtype alongside cache accounting.
Private safe delivery protects inference even from faulty custom sinks.

The Regime I performance floor compares warmed, host-visible greedy inference
on the same local checkpoint and RTX 5060. Primary Lava storage and normalization
are native Float32; CPU Float64 remains a correctness oracle. The auxiliary
CUDA comparison retains legacy Float64 normalization intermediates to meet
its tighter small-fixture accuracy budget, with Float32 storage. Unsupported
native epsilon representations fail explicitly. Independent numeric/context
references bound the changed intermediate arithmetic. CUDA remains a
throughput comparator, with the existing candidate choice and reduction
algorithms retained; its ordinary registry lookup no longer copies the list.

Regime I package reproduction pins the exact tested Lava source revision in
its optional test workspace. Inference imports workload/cache types directly
from their owning modules, preserving their identities before parent reexport.
A relocated cache-backed fresh-depot source install is exercised with actual
CPU and Lava inference. See PACKAGING_REGIME_I.md for the exact environment,
source-execution command and Lava precompilation limitation. Campaign receipts
and the verified host report live in `docs/archive/2026-10_regime-i/`.

Fresh source-only Lava execution requires locally precompiled LLVM/GPUCompiler
prerequisites. scripts/prepare_lava_compiler.jl prepares them from the exact
workspace lock in a temporary project, without a core dependency or upgrade.
