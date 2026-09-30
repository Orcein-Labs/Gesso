# /goal PHASE 6 — PERFORMANCE OBSERVABILITY

**For:** Buffy (mechanical implementation)
**From:** Grok (encoding owner)
**Canon:** `docs/Gesso_Stack.md` §LXXIX, §XLIX, §L, §XLII, §XXXIII, §LXX
**Map:** `docs/ARCHITECTURE.md`
**Research (read, do not implement):** `docs/research/README.md`,
`docs/research/CYAN_TRIAL_GESSO_FALLOUT.md` §2.A/B/F
(grow existing `Receipt` / `GessoError`; do not invent
`ExecutionResult` or a Gesso scheduler)
**Depends on:** Phase 5a complete (`docs/goals/PHASE5_ENGINE.md`)
**Packets:** 1 and 2 stay closed.

**Status:** OPEN. Next after this lands: Phase 7 (one semantic win;
default pick is CoW / identity prefix share — a later closed recipe).
5b (scheduler / batching) is still not this sprint.

---

## Start condition

Phase 5a is on the tree you inherit:

- `Session` / `prefill!` / `decode!` / `generate` exported
- paged KV manager; pages ARE the cache; gather-on-read
- engine ids equal `reference_*` on toy2 and llama_micro
- `make test` green (CPU-only named CUDA/SmolLM2 skips are fine)
- toy2 CPU fingerprint unchanged (`max|logit| = 1686.4814860783201`)
- `Receipt` + `InMemorySink` + `emit!` exist; Session does not emit
- `src/Profiling/Profiling.jl` is still contract-only
- JSON is the only core third-party dep; CUDA is weakdep

If Phase 5a is unfinished, stop.

## One-sentence objective

Every `Session` generate is attributable: a `Receipt` records
prefill vs decode time, tokens, and KV bytes. The Profiling
module can explain those numbers. Nothing gets faster.

## Why this phase exists

§III: MAKE IT MEASURABLE before MAKE IT FAST. Canon §LXXIX exit:
every major latency component attributable, memory accounting
trustworthy, profiler reports stable. The engine is an interpreter
plus pages — "kernel" and "launch" are the same CPU/GPU loop.
Measure what exists. Do not invent occupancy or a scheduler queue.

Cyan trial fallout that belongs here: fill `Receipt` fields the
engine actually has (timing, tokens, KV footprint, context
remaining). `limit_margin_ms` stays Palette.

## What this sprint is not

- speeding anything up
- fused kernels, FlashAttention, paged-attention kernels
- CoW, cages, quantizers, speculation
- scheduler / continuous batching / 5b
- `ExecutionResult`, ProbeSuite, RPC, Julia-world snapshots
- llama.cpp / Python comparison matrix (§LI — later)
- occupancy, swarm GPU, agent idle, tool wait, cache hit rate
  of a cache that is not Magenta yet
- Chrome/tracy UI
- adding Random.jl or any new hard dependency
- changing `reference_*` or the toy2 fingerprint
- bumping `RECEIPT_SCHEMA_VERSION` unless you add a **field**
  to `Receipt` (filling existing `Any` fields does not bump)

---

## Laws (pin this)

### Receipts, not a parallel type

Use `Receipt` as it stands (§XLII). Fill:

```
timing          Dict or named tuple with at least:
                  prefill_ns, decode_ns, total_ns, ttft_ns
token_usage     prompt_tokens, new_tokens, total_tokens
memory_usage    kv_bytes, page_count, kv_len, context_length,
                context_remaining  (= context_length - kv_len)
inference_request  backend, page_size, max_new_tokens, eos_token_id
failure         GessoError if the call threw; else nothing
```

`emit!` still never throws. A telemetry failure must not fail
generate.

Default: a process-level `InMemorySink` (or a Session keyword
`sink=` that defaults to it). Tests inject their own sink.
`generate` / `prefill!` / `decode!` emit one receipt each
successful or failed call (`failure` filled on throw **after**
the error is constructed; the throw still propagates).

Do not add `ExecutionResult`. Do not add Receipt fields unless
you bump `RECEIPT_SCHEMA_VERSION` and pin it in `test_receipts.jl`.

### Timing hygiene (§XXXIII)

No published number from a run that included compilation unless
the receipt/`bench_note` says so. Tests that check structure may
include compile. Benchmarks warm up first, then sample.

CPU: `time_ns()` around prefill and around the decode loop.
TTFT = prefill_ns + first decode step (or prefill_ns if
`max_new_tokens==0`). CUDA: `CUDA.synchronize()` already happens
before host-visible logits/ids; wall clocks after that. Do not
load CUDA in `Profiling` — keep it out of core.

### KV bytes are derived, not estimated

`kv_bytes` is computed from the manager: sum of allocated page
storages (including unused rows in a live page), or a documented
formula pinned by test:

```
n_pages * page_size * n_kv_heads * d_head * sizeof(eltype) * 2 * n_layers
```

Pick one, test it against an actual `sizeof` of the page arrays.
Trustworthy means a test can reconstruct the number from the
manager. Filled length stays derived from pages (Phase 5 law).

### Taxonomy is a label, not a detective

§L classes that this sprint may **attach** when the engine already
knows:

    allocation, memory transfer, backend, hardware limitation,
    algorithm, resource limit  (ERR_RESOURCE_LIMIT / ERR_TIMEOUT)

Do not auto-diagnose "why it is slow." A receipt may carry
`context[:gap_class] = :algorithm` for the interpreter path as a
constant. Expanding the detective is Phase 9+.

### Skip law

CUDA benches and CUDA receipt tests: named skip without a device.
SmolLM2: named skip without `GESSO_SMOLLM2_DIR`. CI never
downloads, never requires a GPU.

---

## Work items (sequence)

### A — Session emits receipts

**Objective.** `generate` on toy2 produces a `Receipt` whose
timing / token / memory fields are present and internally
consistent. Oracle ids still match.

**Permitted files**

```
src/Inference/session.jl
src/Inference/Inference.jl
src/receipts.jl              # only if you must; prefer filling fields
src/Gesso.jl                 # export a default sink getter if needed
test/test_session_receipts.jl
test/runtests.jl
```

**Tests (CPU)**

- `generate(session, PROMPT; max_new_tokens=8)` still equals
  `reference_generate` (ids)
- a sink sees exactly one receipt per `generate`
- `timing.total_ns ≈ timing.prefill_ns + timing.decode_ns`
  (tolerance: 1% or 1e6 ns, whichever larger — clocks are not
  the fingerprint)
- `token_usage.prompt_tokens == 4`, `new_tokens` equals
  `length(ids) - 4`
- `memory_usage.kv_len` equals session seqlen after the call
- `memory_usage.context_remaining == context_length - kv_len`
- empty prompt still throws; the receipt has `failure` set
  (`ERR_INVALID_PLAN`) and the throw still happens
- context exhaustion still throws `ERR_RESOURCE_LIMIT`; receipt
  has `failure` set
- `emit!` failure cannot change generate ids (if you can induce
  it without breaking the sink lock law, do; otherwise document
  in the module comment that `emit!` already swallows)

**Artifact.** The engine is auditable. Fingerprint untouched.

---

### B — Profiling module: KV footprint + stable report

**Objective.** `Profiling` is no longer empty. It can compute
KV bytes from a manager and render a stable text/structured
report from a receipt (or from a sink of them).

**Permitted files**

```
src/Profiling/Profiling.jl
src/Inference/kv_manager.jl  # read-only helpers if footprint lives
                             # next to the pages; prefer Profiling
test/test_profiling.jl
test/runtests.jl
docs/ARCHITECTURE.md
```

**Tests**

- `kv_bytes(mgr)` after N appends equals the documented formula
  (CPU `Array{Float64}` pages). A second test with `page_size=4`
  still matches `sizeof` of live page storage
- report from a toy2 generate receipt contains the keys
  `prefill_ns`, `decode_ns`, `ttft_ns`, `kv_bytes`, `kv_len`
  as machine-readable fields (not only pretty-print)
- two reports from two identical generates after warmup have
  the same **structure** (keys/types); values may differ
- Profiling does not import CUDA
- empty sink / no receipts: report is explicit empty, not a crash

**Artifact.** §LXXIX "profiler reports stable" for the engine we
have. Memory accounting is a function of the page table.

---

### C — Benchmarks: TTFT and decode, warmed

**Objective.** Accrue Session numbers the way the harness already
accrues interpreter numbers. Schema `0.2.0`. Warmup first.

**Permitted files**

```
benchmark/runbenchmarks.jl
benchmark/results/           # append-only TSV from the run
test/test_session_cuda.jl    # only if a CUDA receipt/timing
                             # assertion is missing; prefer skip-or-green
```

**Tests / harness**

- CPU always: toy2 `Session` `generate` `max_new_tokens=8`,
  warmed, named so TTFT and per-generate median are distinct
  rows or a documented pair of notes
- llama_micro Session row may stay; do not remove it
- CUDA: named skip without device; with device, one Session
  generate row after `CUDA.synchronize` / warmup
- CI bench-smoke still only checks the harness runs and persists
- `bench_note` says "includes compile" or "post-warmup"; never
  mix those in one claim

**Artifact.** A corpus row for the engine. No speed claim in
README.

---

### D — maps + §LXXIX status

**Permitted files**

```
README.md
docs/ARCHITECTURE.md
docs/Gesso_Stack.md          # §LXXIX status COMPLETE + date
docs/research/ROADMAP_NOW.md # 6 done; 7 is next open recipe
scripts/freeze.jl
```

**Artifact.** Docs tell the truth: we can attribute prefill vs
decode vs KV bytes. We did not make it fast.

---

## Cross-item invariants

- toy2 CPU fingerprint bit-identical
- `reference_*` public signatures unchanged
- engine ids still equal oracle (re-run the Phase 5 gates)
- JSON remains the only core third-party dep
- `libs/` untouched
- no new §CIX family types, no new fields on `KVCache`
- `Runtime` still empty of scheduler
- Profiling stays out of CUDA
- formatter skips `libs/`
- Receipt: what / why / tests / numerical delta (fingerprint
  unchanged; receipt field presence) / compile-time / memory
  (kv_bytes formula) / hardware / workload / model toy2+micro /
  backend cpu (+ cuda if present)

## Escalation (packet, then stop)

- you think `Receipt` needs new fields (then bump schema, pin
  the test, and say so)
- KV bytes cannot be derived from the page table
- you need CUDA in core to time the GPU
- you want a second result type beside `Receipt`
- generate ids drift once receipts are on the path

## Exit checklist

- [ ] A, B, C landed
- [ ] D maps truthful
- [ ] `make test` green on CPU-only
- [ ] `make format` run
- [ ] `generate` still matches oracle ids
- [ ] toy2 fingerprint unchanged
- [ ] every generate leaves a receipt with prefill/decode/KV
- [ ] no ExecutionResult, no scheduler, no speed claims
- [ ] README does not advertise tok/s as a product number
- [ ] receipt
