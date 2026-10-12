# /goal PHASE 7 — CoW + IDENTITY PREFIX SHARE

**For:** Buffy (mechanical implementation)
**From:** Grok (encoding owner)
**Canon:** `docs/Gesso_Stack.md` §LXXX, §LIV, §XXXI, §LXX, §CIX
**Map:** `docs/ARCHITECTURE.md`
**Research (read, do not implement beyond this pick):**
Magenta §9.4–§9.5 steps 2–3 only
(`docs/research/KV_MEMORY_PROGRAM.md`). Identity-based prefix
share is the declared-sharing claim. Do not discover sharing by
token match. Do not start span classes, tiered precision,
eviction, remat, cages, or speculation.
**Depends on:** Phase 6 complete (`docs/goals/PHASE6_OBSERVABILITY.md`)
**Packets:** 1 and 2 stay closed.

**Status:** LANDED (Buffy, 2026-09-30). Items A–D committed
(`f71af45` CoW manager, `dfae09b` Session `fork`, `c8d8585`
`unique_kv_bytes`, `6ea0be1` maps + §LXXX). 1244 pass + 3 named
SmolLM2 skips (Broken = skip records, not `@test_broken`). Win:
`unique_kv_bytes` 2048 vs 4096 on the llama_micro prefill pair
(`benchmark/results/2026-09-30.tsv`). Two independent Sessions do
not share; `generate` still RESETS. Next:
`docs/goals/PHASE8_LAVA.md`. 5b remains later. Planning.jl stays
empty.

---

## Start condition

Phase 6 is on the tree you inherit:

- `Session` emits one `Receipt` per `generate` / `prefill!` / `decode!`
- `Profiling.engine_report` / `kv_footprint` / `page_footprint`
- `kv_bytes` reconstructs from the page table
- engine ids equal `reference_*` on toy2 and llama_micro
- `make test` green (CPU-only named CUDA/SmolLM2 skips are fine)
- toy2 CPU fingerprint unchanged (`max|logit| = 1686.4814860783201`)
- `generate` still RESETS the Session (Phase 5 law)
- JSON is the only core third-party dep; CUDA is weakdep
- `KVPage` has `layer`, `kind`, `start_pos`, `filled`, `storage`
- `KVCache` still has no extra fields

If Phase 6 is unfinished, stop.

## One-sentence objective

Two Sessions can **declare** they share a KV prefix: complete
prefix pages are the same arrays; the first write into a shared
page copies that page only; token ids still match two independent
Sessions; unique live page storage of the pair is less than twice
an isolated run.

## Why this phase exists

§LIV / §LXXX: one clean, reproducible semantic win. Same model,
same hardware, correctness preserved, measured advantage.

Magenta §9.4: Gesso **declares** sharing by identity. A generic
server discovers it after the fact by exact token match. Two
Sessions that happen to prefill the same tokens do **not** share.
They share because `fork` was called.

Phase 6 made `kv_bytes` a function of one session's page table.
Shared pages would be counted in both receipts. The win lives in
a Profiling helper over **unique** storage objects, not a new
Receipt field.

## What this sprint is not

- radix / n-gram / discovered prefix cache
- span classes, tiered KV precision, meaning-aware eviction, remat
  (Magenta §9.5 steps 4–7)
- cages, quantizers, speculation (Phase 20), fused kernels
- scheduler / continuous batching / 5b
- filling `Planning.jl` / `Runtime.jl`
- `ExecutionResult`, new Receipt fields, schema bump
- changing `reference_*` or the toy2 fingerprint
- claiming tok/s
- CoW of the hidden-state buffer `h` (copy it; KV pages are the win)
- making `generate` share (it still RESETS — two `generate` calls
  stay independent)

---

## Laws (pin this)

### Declaration, not discovery

The only share constructor is `fork(s::Session) -> Session`.

`fork` is legal after `prefill!` (and after subsequent `decode!`).
It is `ERR_INVALID_PLAN` on a Session that is not `ready`.

Two Sessions constructed separately with the same prompt **must**
have distinct `storage` object-ids. A test pins that, or the
differentiator is a lie.

### Pages stay the cache

Physical unit is still a page. Sharing is **pointer identity** of
`KVPage.storage` (and of the `KVPage` object until it is copied).

You **may** add a field to `KVPage` (it is not the §CIX `KVCache`
family type). Suggested pick, pin it in the module comment:

```
shared::Bool    # true after fork; this page's storage is aliased
```

You may **not** add fields to `KVCache`. Page table stays in
`storage`.

Filled length is still derived from pages (`Σ page.filled`).
No shadow counters.

### Copy on write, only the dirty page

On `append_kv!` into the last page of a cache:

- if that page is not shared, write in place (Phase 5 path)
- if it is shared, allocate a new `KVPage`, copy the filled rows
  of `storage`, set `shared=false`, replace **this manager's**
  list entry, then write
- if the last page is full, allocate a **new** unshared page
  (Phase 5 already does this). Full prefix pages stay aliased
  forever until someone would write them, which nobody does

The other manager keeps the original page object. Do not mutate
`filled` or `storage` of a shared page.

### Hidden state

`fork` copies `h` (and `seqlen`, `ready`, model/tensors/backend
fields). Do not invent hidden-state CoW this sprint.

### `generate` still resets

`generate` wipes manager + `h` and runs a fresh prefill/decode.
A forked Session that then calls `generate` **drops the share**.
Test that. The share path is `prefill!` / `fork` / `decode!`.

### Per-session `kv_bytes` vs unique bytes

`kv_bytes(mgr)` stays "sum of this manager's page storages"
(Phase 6). Shared pages appear in both sessions' receipts. That
is honest per-session accounting.

The win metric:

```
unique_kv_bytes(mgrs...) -> Int
```

Sum `sizeof(storage)` once per distinct `objectid(page.storage)`
across the managers. Lives in `Profiling`. No Receipt schema bump.

### CUDA / skip

Same as Phase 5/6: named skip without a device. With a device,
forked CUDA Sessions produce the same token ids as two independent
CUDA Sessions (and as CPU, existing atol/id gates). Pages allocate
via `similar` off device tensors. No silent host copy. Profiling
still does not import CUDA.

### Oracle stays oracle

`reference_prefill` / `reference_generate` unchanged. Engine ids
on an unforked Session still equal the oracle. A forked child's
decode ids equal a **second independent Session** that saw the
same token sequence — not a new oracle.

---

## Public surface

```
fork(s::Session) -> Session
```

Child:

- same `model`, `tensors`, `backend`, `page_size`, `context_length`,
  `eos_token_id`, `tokenizer`, `eps`, `theta`
- **own** `sink` (default process sink is fine; tests inject)
- own `PagedKVManager` whose page **lists** are new vectors
  holding the same `KVPage` objects, each marked `shared=true`
- copied `h`, copied `seqlen`, copied `ready`
- subsequent `decode!` on parent or child CoWs only the page
  being written

Export `fork` from Inference / Gesso.

Do not add `share_prefix(parent, n)` this sprint. Identity share
**is** `prefill!` the declared prefix, then `fork`. Truncating a
longer parent to `n` tokens would split a page; that is a later
item.

Do not add a radix API, a share registry, or a planner type.

---

## Work items (sequence)

### A — CoW in the manager

**Objective.** Two managers can alias the same `KVPage` objects.
`append_kv!` copies a shared last page before writing. Full
shared pages are never mutated.

**Permitted files**

```
src/Inference/kv_manager.jl
src/Inference/Inference.jl
test/test_kv_manager.jl
test/runtests.jl
```

**Tests (CPU, `page_size=4` so you cross a boundary)**

- helper to alias manager B's page lists at manager A's current
  pages and mark them `shared` (Session `fork` will call this;
  the helper may be internal)
- after alias, `objectid(page.storage)` matches for every page
- append one row on B: B's last page is a **new** object; A's
  last page object-id and filled-row bytes are unchanged
- a full shared page stays `===` after B allocates a new page
  for the next row
- filled length still derives from pages; context exhaustion
  still `ERR_RESOURCE_LIMIT`
- two managers filled independently (no alias) never share
  `objectid(storage)`

**Artifact.** Magenta step 2 mechanics. No Session API yet.

---

### B — `Session.fork`

**Objective.** `fork` after toy2 `prefill!` yields a child that
decodes the same next ids as a second independent Session, and
the parent's continued decode matches a third independent Session.

**Permitted files**

```
src/Inference/session.jl
src/Inference/Inference.jl
src/Gesso.jl          # export fork
test/test_session_fork.jl
test/runtests.jl
test/test_session_cuda.jl   # skip-or-green: forked CUDA ids match
```

**Tests (CPU)**

- `fork` before `prefill!` throws `ERR_INVALID_PLAN`
- after `prefill!(s, PROMPT)` (toy2, 4 tokens), `c = fork(s)`:
  `c.seqlen == s.seqlen`, `c.ready`, prefix pages `===`
- `decode!(c)` then `decode!(s)` (or the other order): each
  sequence of ids equals a solo Session that prefills PROMPT
  and decodes the same number of steps
- `generate` on a forked child still RESETS: after `generate`,
  child's pages are not `===` parent's prefix pages
- empty prompt / context limit still throw; receipts still emit
  (one per `prefill!`/`decode!`/`fork` does **not** emit unless
  you have a clean existing field to fill — prefer **no**
  receipt for `fork` itself)
- llama_micro: same id-equality vs two independent Sessions
- CUDA: named skip without device; with device, same id gate

**Artifact.** Magenta step 3 is `prefill!` + `fork`. The
declaration is the function call.

---

### C — unique bytes (the win)

**Objective.** `unique_kv_bytes` proves the pair is cheaper than
two isolated Sessions for a shared prefix, and that the saving
is the aliased full pages.

**Permitted files**

```
src/Profiling/Profiling.jl
test/test_profiling.jl
benchmark/runbenchmarks.jl
benchmark/results/           # append-only TSV
```

**Tests / harness**

- after `prefill!` + `fork`, before any decode:
  `unique_kv_bytes(parent.mgr, child.mgr) == kv_bytes(parent.mgr)`
  (every page aliased)
- after enough child-only decodes to allocate **new** pages
  (`page_size=4`, several steps): unique bytes equal parent
  pages + child's private pages, **less than**
  `kv_bytes(parent) + kv_bytes(child)` if any alias remains
- two independent Sessions (no fork):
  `unique_kv_bytes(a, b) == kv_bytes(a) + kv_bytes(b)`
- CPU bench row, warmed, schema `0.2.0`: unique bytes for
  2× isolated toy2 prefill vs 1× prefill + fork. Notes say
  post-warmup. This is a **byte** row, not tok/s
- CUDA bench: skip without device; do not require a new CUDA
  row if CPU unique-bytes already prove the win

**Artifact.** §LIV win is a number of bytes, not a slogan.

---

### D — maps + §LXXX status

**Permitted files**

```
README.md
docs/ARCHITECTURE.md
docs/Gesso_Stack.md          # §LXXX status COMPLETE + date + pick
docs/research/ROADMAP_NOW.md # 7 done; 5b/8 are later
docs/research/README.md      # Phase 7 pick recorded as landed
scripts/freeze.jl
```

**Artifact.** Docs say we share by declaration. They do not say
we discovered a radix cache. They do not say we got faster.

---

## Cross-item invariants

- toy2 CPU fingerprint bit-identical
- `reference_*` public signatures unchanged
- unforked engine ids still equal the oracle
- forked decode ids equal independent Sessions with the same tokens
- JSON remains the only core third-party dep
- `libs/` untouched
- no new §CIX family types
- no new fields on `KVCache` (KVPage may gain `shared`)
- `Runtime` / `Planning` still empty of scheduler / planner
- Profiling stays out of CUDA
- formatter skips `libs/`
- no ExecutionResult, no schema bump, no speed claims
- Receipt: what / why / tests / numerical delta (fingerprint
  unchanged; unique_kv_bytes vs 2× isolated) / compile-time /
  memory (unique vs summed) / hardware / workload / model
  toy2+micro / backend cpu (+ cuda if present)

## Escalation (packet, then stop)

- you think `KVCache` needs a field
- you think sharing must be discovered by token equality
- you need to split a page so `fork` can truncate to `n < seqlen`
- `generate` must share (it must not)
- unique bytes cannot be defined by `objectid(storage)`
- CUDA fork requires a host copy

## Exit checklist

- [x] A, B, C landed
- [x] D maps truthful (§LXXX in Gesso_Stack.md)
- [x] `make test` green (Buffy receipt: 1244 pass + 3 named SmolLM2 skips)
- [x] `make format` run
- [x] unforked `generate` still matches oracle ids
- [x] toy2 fingerprint unchanged (fixture still `1686.4814860783201`)
- [x] two independent Sessions do not share storage
- [x] `fork` after prefill aliases prefix pages
- [x] CoW copies only the dirty page
- [x] `unique_kv_bytes` after fork (no decode) equals one session (2048 vs 4096 isolated)
- [x] no radix, no cages, no scheduler, no Planning.jl fill
- [x] receipt (Buffy §LXXII, 2026-09-30; Grok QA accepted from the tree + screenshot)
