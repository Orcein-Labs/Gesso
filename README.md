# Gesso.jl

**Julia-native semantic ML and agent execution runtime.**

Gesso loads existing open-weight models, preserves what they mean, and uses that
information to determine how they should physically exist and execute on the
hardware and workload actually present.

> The model should not lose its meaning before it reaches the machine.
> The runtime should not lose sight of why the model is being invoked.

## Status

**Requires Julia 1.12** (`julia = "1.12"` compat; CI runs 1.12 only).

**Phase 0 — repository foundation: COMPLETE** (incl. the BONES swarm-readiness
sprint and the foundation-hardening sprint: receipts/failure-taxonomy/freeze
infrastructure stress-tested, correctness laboratory established).
**Phase 1 (semantic core) and Phase 2 (CPU oracle) are COMPLETE.**
Next gate: Phase 6 — performance observability (`docs/goals/PHASE6_OBSERVABILITY.md`). `toy2` and `llama_micro` run on `CPUBackend` and `CUDABackend`.

- [x] Clean `Project.toml`, no third-party dependencies (§VII: *Gesso earns every hard dependency*; stdlibs `Dates` and `LinearAlgebra`, enforced by test)
- [x] Package skeleton: logging conventions, backend interface draft
- [x] Test harness with dependency-law enforcement
- [x] CI (tests + format + bench-smoke, dev-loop entry included)
- [x] Benchmark harness with benchmark-integrity conventions (§XXXIII)
- [x] Research parking lot: [docs/research/README.md](docs/research/README.md) (Magenta Memory + representation program; parked until a phase owns them)
- [x] BONES sprint (agent charter, module skeleton, receipts types, tooling)
- [x] Foundation hardening (receipt stress tests, taxonomy pins, freeze evidence, lab fixtures, empty-core fence)
- [x] Phase 1: semantic core per §CIX — types exist, `toy2` expressible end to end, **no execution yet**
- [x] Phase 2: CPU oracle COMPLETE — deterministic prefill with known logits (persisted + provenance) and greedy KV-cached decode; `toy2` runs on `CPUBackend`
- [x] Phase 3: first real model import — SKIP-OR-GREEN (GQA interpreter, Llama import, GPT-2 BPE landed; the real-model gate is a named skip without a local SmolLM2 snapshot, §LXXVI)
- [x] Phase 4: CUDA.jl execution — COMPLETE 2026-09-29 (extension seam, operator methods + `to_device`, backend-generic interpreter; device-vs-oracle gates green on RTX 5060)
- [x] Phase 5: native inference engine — `Session` + `generate` over a paged KV manager (Magenta §9.5 step 1); engine ids equal the oracle on toy2/llama_micro, SmolLM2 gate skip-or-green (§LXXVIII)
- [x] Phase 6: performance observability — every generate emits a receipt (prefill/decode/TTFT, tokens, KV bytes from the page table); `Profiling` renders stable reports; warmed TTFT/decode bench rows (§LXXIX)
- [ ] Telemetry collection (Phase 6+)

## The stack

```julia
using Gesso
using Lava   # Phase 8 — portable/Vulkan path
# or
using CUDA   # Phase 4 — strategic fast path
```

### The engine (Phase 5, §LXXVIII)

```julia
session = Gesso.Session(model, tensors;
                        context_length = 128,
                        eos_token_id = 2)
ids = Gesso.generate(session, [1, 3, 4, 5]; max_new_tokens = 8)
ids = Gesso.generate(session, "Hello";             # requires tokenizer=…
                     max_new_tokens = 8,
                     on_token = id -> println(id)) # streaming callback
```

Greedy only this sprint; the paged KV manager (page_size=16 default) stores
cache rows and the engine gathers pages per attention step.
`reference_prefill` / `reference_generate` remain the oracle the engine is
gated against (token ids equal, CPU logits atol=0).

Every `Session` call leaves a `Receipt` (§XLII fields: prefill/decode/TTFT
timing, token usage, KV bytes + pages derived from the page table, failure)
in a `Gesso.InMemorySink` — pass `sink=` to inject your own — and
`Gesso.Profiling.engine_report` projects receipts into stable
machine-readable rows. This is attribution, not a speed claim: we can
explain prefill vs decode vs KV bytes; we have not made anything fast.

Phase 7 adds **declared** identity prefix share: `child = Gesso.fork(session)`
aliases the prefix KV pages, and the next write into a shared page copies
that page only (copy-on-write). `Gesso.Profiling.unique_kv_bytes(parent.mgr,
child.mgr)` counts the pair's live storage once per shared page. Sharing is
declared, never discovered: two sessions that prefill the same tokens
independently do NOT share. `generate` still resets (a forked child that
calls `generate` drops the share).

Phase 8 adds the **Lava/Vulkan backend** as a package extension
(`ext/GessoLavaExt.jl`, the only file allowed `using Lava`): pass
`backend=Gesso.LavaBackend()` and tensors moved with `Gesso.to_device` to run
the interpreter, `Session`, and `fork`/CoW on `LavaArray{Float32}` storage —
GPUArrays broadcast + Lava's `mul!`, no handwritten kernels. Same API as CPU
and CUDA; greedy ids match the CPU oracle, logits within the declared atol.
Requesting Lava without a usable Vulkan device throws a typed error — never a
silent CPU fallback (§LXX). The extension is portable-first; nothing here is
tuned, and no speed claim is made.

Phase 9 adds **autotuning** (`src/Autotune/`, §XXVI/§LXXXII): the CUDA
`matmul!` operator consults `Gesso.Autotune`, which searches over registered
candidate realizations (`:cublas_mul` — the CUBLAS path — and
`:generic_mul`), disqualifies any candidate that fails the CPU-F64 oracle at
the declared atol, benchmarks the survivors post-warmup, and caches the
winner per (cache version, device, backend, operator, shape regime). The op
dispatches to the cached winner; every selection emits one `Receipt`
(`task=:autotune_select`). The loop is core machinery that imports neither
CUDA nor Lava — backends register candidates at load time. Selection
happened; no tuned-kernel or speed claim is made.

Phase 10 lands the **speed floor's proof-ladder rungs 1–2**
(`docs/goals/PHASE10_SPEED_FLOOR.md`, gates G1–G3 of
`docs/research/SPEED_FLOOR.md`): CUDA decode runs greedy argmax **on the
device** — the host receives one integer per token and the (vocab,) logits
row never crosses back on the hot path — and the attention contraction
(QKᵀ + PV) runs as **one flat device GEMM over the gathered scratch** (pages
remain the cache, gather remains legal, `fork` aliasing untouched; no
page-table kernel — that is a later packet). The G2 comparison harness
(`benchmark/compare_eager.py`) publishes first-token and warmed rows for
SmolLM2-135M against eager PyTorch `generate` on the same checkpoint, prompt,
and length — when a local snapshot, CUDA, and an external `python3` with
torch/transformers are all present; without them it is a named skip, and CI
never requires any of the three (never downloads). §LXXXIII (the
representation planner) stays parked until G1∧G2∧G3 are honestly green.

## Development order (§III)

```
MAKE IT WORK → MAKE IT COMPLETE → MAKE IT MEASURABLE → MAKE IT FAST
```

Current phases (docs/Gesso_Stack.md §LXXIII ff.):

| Phase | Deliverable | Status |
|------:|-------------|--------|
| 0 | Repository foundation | **COMPLETE** |
| 1 | Semantic core (§CIX encoding) | **COMPLETE** — expressible, not executable |
| 2 | Reference execution (CPU oracle) | **COMPLETE** — prefill + greedy KV decode |
| 3 | First real model import | **SKIP-OR-GREEN** — items A/B/C landed (GQA interpreter, Llama import, GPT-2 BPE); item D is skip-or-green pending a local SmolLM2 snapshot |
| 4 | CUDA.jl execution | **COMPLETE 2026-09-29** — extension seam, ops + `to_device`, backend-generic interpreter, device-vs-CPU gates; SmolLM2-CUDA + bench row skip-or-green (§LXXVII) |
| 5 | Native inference engine | **A/B/C/D LANDED 2026-09-30** — paged KV manager + `Session`/`generate` matching the oracle; scheduler/continuous batching is a later goal under this phase (§LXXVIII) |
| 6 | Performance observability | **A/B/C/D LANDED 2026-09-30** — receipts per engine call, Profiling reports, warmed TTFT/decode rows; attribution, not speed (§LXXIX) |
| 7 | First semantic optimization | **A/B/C/D LANDED 2026-09-30** — CoW + declared identity prefix share (`fork`), `Profiling.unique_kv_bytes` byte win (§LXXX) |
| 8 | Lava/Vulkan backend | **A/B/C/D LANDED 2026-09-30** — GessoLavaExt seam + six ops + interpreter/Session/fork on `LavaArray{Float32}` (§LXXXI); portable seam, not tuned; device-less runs skip by name |
| 9 | Autotune | **A/B/C/D LANDED 2026-10-01** — the §XXVI loop: two gated CUDA `matmul!` candidates, winner cached per (device, backend, op, regime), op consults and dispatches to it (§LXXXII); selection, not speed; device-less runs skip by name |
| 10 | Speed floor (proof-ladder rungs 1–2) | **A/B/C/D LANDED 2026-10-01** — device greedy (one Int D2H per token) + device GEMM attention over gathered scratch (§LXXXIII gates G1–G3 harness); G2 eager-PyTorch rows skip-or-land; SmolLM2 gate is ops-blocked without a local snapshot; §LXXXIII representation planner stays PARKED |

Training is **not** part of Gesso — by explicit, permanent decision
([docs/Gesso_Stack.md §LVIII](docs/Gesso_Stack.md)). If you want to contribute
to the Julia ML ecosystem and want a job with real scope: **build the training
stack.** The natural place to start is [WGPU.jl](https://github.com/JuliaGPU/WGPU.jl).
Gesso will load its checkpoints like everyone else's.

## Layout

```
src/           package core (zero deps; logging, receipts, errors, backends)
test/          test harness (workspace member; per-area test files)
benchmark/     benchmark harness (workspace member; results/ accrues)
docs/          architecture: Gesso_Stack.md is canon; ARCHITECTURE.md is the map;
               docs/research/README.md is the research parking lot
libs/          local dev sources (gitignored; Lava lives here) — DO NOT TOUCH
```

## Working here

- `AGENTS.md` is the **binding agent charter** — read it before your first edit.
- `docs/ARCHITECTURE.md` maps every module to its canon section and phase.
- Work items follow `.github/ISSUE_TEMPLATE/work-item.md` (§LXXI format);
  PRs are receipts per `.github/PULL_REQUEST_TEMPLATE.md` (§LXXII).
- Open architecture decisions are quarantined in
  [docs/DECISION_PACKETS.md](docs/DECISION_PACKETS.md) — not settled by
  local engineering alone.

## Commands

```bash
make test          # test suite (workspace root; runs with -t 2)
make bench         # benchmarks (append to benchmark/results/)
make format        # format the repo
make format-check  # CI's formatting gate
make freeze        # curated context-freeze bundle (evidence, not ceremony)
```

### Real-model gate (Phase 3, §LXXVI)

The test suite never downloads and never talks to huggingface.co. To run
the real-model parity gate locally, point `GESSO_SMOLLM2_DIR` at a local
snapshot of [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
containing `config.json`, `model.safetensors` (or the shard index),
`vocab.json`, and `merges.txt`:

```bash
GESSO_SMOLLM2_DIR=/path/to/SmolLM2-135M make test
```

Without the variable the gate is one named skip — CI stays green on a
machine that has never seen SmolLM2 weights. The first successful run on
a snapshot freezes `test/fixtures/smollm2/expected_logits.toml`
(oracle `gesso-cpu`); that committed file is the regression oracle.

CUDA device tests (Phase 4, §LXXVII) follow the same skip law: without an
NVIDIA device they are named skips and CI never requires a GPU. The
SmolLM2 CUDA gate additionally needs `GESSO_SMOLLM2_DIR` — it compares
last-position logits on device to the frozen CPU golden at `atol=1e-2`
(F32 device vs F64 oracle) and never fakes a golden CUDA file.

Under the hood (Julia 1.12 workspace — activate the root, then let Pkg
discover the members; do NOT `Pkg.instantiate("test")`, positional
instantiate is not valid API):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.activate("test"); Pkg.instantiate(); Pkg.activate("benchmark"); Pkg.instantiate()'
julia --project=test --check-bounds=yes test/runtests.jl
julia --project=benchmark benchmark/runbenchmarks.jl
```
