# /goal PHASE 5 — NATIVE INFERENCE ENGINE

**For:** Buffy (mechanical implementation)
**From:** Grok (encoding owner)
**Canon:** `docs/Gesso_Stack.md` §LXXVIII, §XXIX, §XXX, §XXXI, §LXVII, §CIX, §LXX
**Map:** `docs/ARCHITECTURE.md`
**Research (read, do not implement):** `docs/research/README.md`,
Magenta §9.5 step 1 only (`docs/research/KV_MEMORY_PROGRAM.md`)
**Depends on:** Phase 4 complete (`docs/goals/PHASE4_CUDA.md`)
**Packets:** 1 and 2 stay closed.

**Status:** LANDED (Buffy, 2026-09-30). Items A–D committed
(`15e207e` paged KV, `b38e369` Session, `2c3e996` streaming/CUDA,
`d2ebb1f` llama_micro/SmolLM2/bench, `971eb9b` maps). 926 pass + 3
named skips; both runners green. Engine ids equal the oracle on
toy2 and llama_micro; CUDA Session ids equal CPU on RTX 5060;
SmolLM2 skip-or-green; toy2 fingerprint untouched; `reference_*`
untouched. `generate` resets the Session (multi-turn `prefill!` on
a consumed session is `ERR_INVALID_PLAN` this sprint). Next:
`docs/goals/PHASE6_OBSERVABILITY.md`. Scheduler / continuous
batching remains 5b — a later goal under Phase 5.

---

## Start condition

Phase 4 is on the tree you inherit:

- `make test` green (CPU-only named CUDA skips are fine)
- toy2 CPU fingerprint unchanged (`max|logit| = 1686.4814860783201`)
- `reference_prefill` / `reference_generate` are the public engine slice
- KV is a preallocated `Activation` buffer inside `reference_generate`
- `KVCache` exists as a family type and is unused by the interpreter
- `src/Runtime/Runtime.jl` is still contract-only
- JSON is the only core third-party dep; CUDA is weakdep

If Phase 4 is unfinished, stop.

## One-sentence objective

Gesso generates from a `Session` through a paged KV manager.
`reference_*` stay the oracle. Engine token ids match the oracle.
Nothing else is in the path.

## Why this phase exists

§XXIX: inference is not an adapter around llama.cpp. Phases 2–4
proved meaning, import, and a backend. This sprint is the first
engine that owns prefill, decode, KV, greedy sampling, a session,
and a stream of tokens — on the operators we already have.

Canon exit for Phase 5: *Gesso no longer relies on another inference
engine.* That is this sprint. Serving fabric (scheduler, continuous
batching, multi-session multiplexing) is later.

## What this sprint is not

- CoW, identity prefix share, span classes, tiered KV precision,
  meaning-aware eviction (Magenta §9.5 steps 2–7 — Phase 7+)
- cages, quantizers, Huffman, Unweight, CagedLinear
  (`docs/research/REPRESENTATION_PROGRAM.md`)
- speculative decoding (Phase 20)
- continuous batching, request queues, priorities, cancellation
- FlashAttention / paged-attention kernels / block-sparse scores
- temperature / top-k / nucleus / RNG sampling (greedy only)
- `Gesso.load("model")` Hub magic
- changing §CIX types or adding fields to `KVCache`
- rewriting `reference_prefill` / `reference_generate` into the
  Session path (they remain the oracle; do not delete them)
- batch > 1
- Lava
- downloading weights
- claiming speedups

Attention this sprint may **gather pages into a contiguous scratch**
and reuse the existing score / softmax / value contraction. The
manager owns storage and append. It does not own a new kernel.

---

## Laws (pin this)

### Oracle stays oracle

```
reference_prefill(...)     # unchanged public contract
reference_generate(...)    # unchanged public contract, including TOY_EOS = 2
```

Existing tests stay bit-identical. The engine is a **new** path that
must match them.

Do not route `reference_*` through `Session` "to share code" if that
changes a single existing test. Sharing private helpers
(`_decode_step!`, `_last_logits`, head split) is allowed.

### Paged KV is Magenta step 1, nothing further

Physical unit: a **page** of `page_size` token-rows,
shape `(page_size, n_kv_heads, d_head)`, allocated with `similar`
off the session's tensor storage (CPU `Array{Float64}`, CUDA
`CuArray{Float32}` — same as Phase 4).

Logical cache per layer, K and V separately: an ordered list of
pages plus a filled-token count. Append writes the next row; when
the current page is full, allocate one more page. Never realloc-copy
the whole cache.

Default `page_size = 16`. Tests must also run `page_size = 4` so a
toy2 prompt of 4 plus a few decode steps **crosses a page boundary**.

Each page records provenance, even though nothing consumes it yet:

```
layer::Int
kind::Symbol          # :k or :v
start_pos::Int        # 0-based first token-row this page covers
filled::Int           # rows written in this page, 0 ≤ filled ≤ page_size
```

That is the hook Magenta §12.2 asked for. Do not invent span classes.

Filled length of the cache equals prefix length: after prefill of
`P` tokens and `S` decode steps, `kv_len == P + S`. Same invariant
as `reference_generate`'s `info` channel.

Exceeding `context_length` throws typed `GessoError(ERR_RESOURCE_LIMIT)`.
Do not silently drop tokens or wrap.

`KVCache` is the semantic identity of a layer's K or V. Put the page
table in `storage`. Do **not** add fields to the type (packet if you
think you must). `shape` is the logical filled shape
`(kv_len, n_kv_heads, d_head)`, updated as length grows, or a
fixed `(context_length, n_kv_heads, d_head)` with filled length
held beside it in the manager — pick one, test it, document it in
the module comment.

### Session is the public engine (§LXVII)

```
session = Session(model, tensors;
                  backend = CPUBackend(),
                  page_size = 16,
                  context_length = 128,
                  eos_token_id = 2,
                  tokenizer = nothing)

prefill!(session, tokens::AbstractVector{Int}) -> logits
    # (vocab, P) host-visible; CUDA: synchronize, copy logits to host

decode!(session) -> Int
    # greedy 0-based id of the next token; appends one KV row
    # unless the sampled id is eos (still returned; no append after stop)

generate(session, prompt::AbstractVector{Int};
         max_new_tokens = 8,
         on_token = nothing) -> Vector{Int}

generate(session, text::AbstractString; ...) -> Vector{Int}
    # requires session.tokenizer; uses encode(tokenizer, text)
    # errors typed (ERR_INVALID_PLAN) if tokenizer is nothing
```

`generate` returns prompt + new ids (same shape as
`reference_generate`). `on_token` if set is `on_token(id::Int)`
called once per **new** token, including EOS if produced.
Default `on_token = nothing` means no callback.

Greedy: `argmax` on the last-position logits, 0-based
(`Int(argmax(row)) - 1`). Ties = first index (Julia `argmax`).
No `Random`. Extract the argmax into a tiny helper so the engine
does not bury it; do not add a Sampler type zoo.

`eos_token_id` is a Session field. toy2 uses `2`. SmolLM2 uses
whatever the tokenizer/config says (`0` on that checkpoint). The
engine does not hardcode `TOY_EOS`. The oracle still does.

Empty prompt / empty string throws (same as the oracle).

`backend=CUDABackend()`: tensors MUST already be on device
(`to_device`). Interpreter / engine never copy. Host `Array` under
a non-CPU backend is `ERR_INVALID_PLAN`. Pages allocate on device
via `similar`. `CUDA.synchronize()` before reading logits or ids
back. Token **ids** match CPU (argmax gate). Logits `atol=1e-3`
on toy2 / micro, `atol=1e-2` on SmolLM2 — same as Phase 4.

Batch = 1. One Session, one sequence.

### Prefill / decode stay distinct (§XXX)

Internally, prefill is `PrefillWorkload` over the prompt; decode is
`DecodeWorkload` one token at a time. Do not re-prefill the prompt
on every step. Do not collapse both into one "generic generate"
loop that always runs the full sequence.

### Runtime module

`src/Runtime/Runtime.jl` stays contract-only. Session and the KV
manager live in `Inference`. Scheduler is not this sprint.

### Skip law

CUDA engine tests: named skip without a device, same shape as
Phase 4. SmolLM2 engine tests: named skip without
`GESSO_SMOLLM2_DIR`. CI never downloads, never requires a GPU.

---

## Work items (sequence)

### A — Paged KV manager

**Objective.** Append and read K/V through pages. Page-boundary
append is correct. No Session API yet.

**Permitted files**

```
src/Inference/kv_manager.jl      # new
src/Inference/Inference.jl       # include + export if needed internally
test/test_kv_manager.jl
test/runtests.jl
docs/ARCHITECTURE.md
```

**Tests (CPU)**

- allocate manager for `n_layers`, `n_kv_heads`, `d_head`,
  `page_size=4`, `context_length=16`
- append 1 row, `kv_len == 1`; logical view shape
  `(1, n_kv_heads, d_head)`
- append 4 more rows: a second page exists; `kv_len == 5`
- gathering the logical view equals `vcat` of the written rows
  (bit-identical on CPU)
- append past `context_length` throws `GessoError` with
  `ERR_RESOURCE_LIMIT`
- `page_size < 1` throws at construction
- each page's provenance `(layer, kind, start_pos, filled)` is
  consistent with `kv_len`
- CUDA: when a device exists, the same append/gather on
  `CuArray{Float32}` (allocate pages from a device prototype
  array). Skip without device.

**Artifact.** Magenta step 1 exists. Interpreter still uses
`Activation` caches.

---

### B — Session + generate matches the oracle

**Objective.** toy2 `generate(session, PROMPT)` token ids equal
`reference_generate` token ids. Prefill logits match
`reference_prefill` at `atol=0` on CPU.

**Permitted files**

```
src/Inference/session.jl         # new (or fold into Inference.jl)
src/Inference/Inference.jl
src/Inference/kv_manager.jl
src/Gesso.jl                     # export Session, generate, prefill!, decode!
test/test_session.jl
test/runtests.jl
docs/ARCHITECTURE.md
```

`PROMPT = [1, 3, 4, 5]` (same as `test_reference_generate.jl`).

**Tests (CPU)**

- `generate(session, PROMPT; max_new_tokens=8)` ==
  `reference_generate(model, tensors, PROMPT; max_new_tokens=8)`
- two `Session`s, same prompt ⇒ same ids (deterministic)
- `max_new_tokens=0` returns exactly the prompt
- first generated token == argmax of `reference_prefill` last column
- `prefill!(session, PROMPT)` logits `approx_eq` `reference_prefill`
  at `atol=0`
- decode-via-session matches re-prefill argmax (same cache-
  correctness gate as the oracle tests)
- `page_size=4` still matches the oracle (page boundary is not
  optional)
- `eos_token_id=2` on toy2: if EOS appears, it is last
- empty prompt throws
- `backend=CPUBackend()` + existing CPU tests of `reference_*`
  still `atol=0` in-process

**Artifact.** §LXVII `session = Session(...); generate(session, …)`
works on toy2. Oracle untouched.

---

### C — Streaming + string generate + CUDA engine

**Objective.** Tokens can be observed as they are produced.
String prompts work when a tokenizer is on the session.
CUDA Session matches CPU ids.

**Permitted files**

```
src/Inference/session.jl
src/Inference/Inference.jl
ext/GessoCUDAExt.jl              # only if device page alloc needs a hook;
                                 # prefer `similar` so you do not touch the ext
test/test_session.jl
test/test_session_cuda.jl        # skip without device
test/runtests.jl
```

**Tests**

- `seen = Int[]; generate(session, PROMPT; max_new_tokens=3, on_token = id -> push!(seen, id))`
  ⇒ `seen` equals the generated suffix (not the prompt)
- `generate(session, "hi"; …)` without tokenizer throws
  `GessoError(ERR_INVALID_PLAN)`
- with `load_gpt2_tokenizer` on the tiny fixture: string generate
  runs, returns ids, `on_token` fires; do not pin the actual ids
  beyond "deterministic, two runs equal"
- CUDA (skip without device): toy2 Session `to_device` tensors,
  `generate` token ids **equal** CPU Session ids for
  `max_new_tokens=8`. Logits from `prefill!` vs CPU
  `approx_eq` `atol=1e-3`. Host Array tensors + CUDA backend throws
  `ERR_INVALID_PLAN`.

**Artifact.** Streaming is a callback, not a server. CUDA engine
is the Phase 4 path with pages.

---

### D — micro-llama + SmolLM2 skip-or-green + maps

**Objective.** The engine runs the models we already import.
Docs tell the truth.

**Permitted files**

```
test/test_session_llama.jl
test/test_session_smollm2.jl
test/runtests.jl
benchmark/runbenchmarks.jl       # optional: one toy2 Session generate
                                 # row, schema 0.2.0, after warmup;
                                 # CPU always, CUDA gated on device
README.md
docs/ARCHITECTURE.md
docs/Gesso_Stack.md              # §LXXVIII status COMPLETE + date
scripts/freeze.jl                # new test files
```

**Tests**

- llama_micro: Session generate `max_new_tokens=3` on `[0,1,2]`
  token ids equal `reference_generate` (CPU). CUDA skip-or-green,
  ids equal CPU when a device exists.
- SmolLM2: without `GESSO_SMOLLM2_DIR`, one named skip. With it:
  `generate(session, "Hello"; max_new_tokens=8)` ids equal
  `reference_generate` on the same snapshot (CPU). CUDA: ids equal
  CPU Session when a device exists; do not invent a CUDA golden
  file.

**Artifact.** Phase 5 exit: Gesso is the engine for toy2, micro,
and SmolLM2-when-present. CI green without snapshot and without GPU.

---

## Cross-item invariants

- toy2 CPU fingerprint bit-identical
- `reference_*` public signatures unchanged
- JSON remains the only core third-party dep
- `libs/` untouched
- `quantize!` / `dequantize!` still decline
- Operators module still exports nothing
- no new §CIX family types, no new fields on `KVCache`
- `Runtime` still empty of scheduler
- formatter skips `libs/`
- Receipt: what / why / tests / numerical delta (CPU fingerprint
  unchanged; engine vs oracle token-id equality on toy2) /
  compile-time / memory (page vs contiguous, qualitative is fine) /
  hardware / workload / model toy2+llama_micro / backend cpu
  (+ cuda if present)

## Escalation (packet, then stop)

- `KVCache` cannot hold a page table without a new field
- page gather cannot match contiguous CPU caches bit-for-bit
- you need a new operator or a fused paged-attention kernel
- Session wants to live in `Runtime` or become a SemanticTensor
- greedy argmax on CUDA F32 diverges from CPU F64 on toy2
  (same rule as Phase 4: fix math, don't relax token equality)
- you think temperature sampling is required for the exit

## Exit checklist

- [ ] A, B, C landed
- [ ] D skip-or-green
- [ ] `make test` green on CPU-only
- [ ] `make format` run
- [ ] `Session` + `generate` exported
- [ ] `reference_*` still the oracle, tests bit-identical
- [ ] paged KV used by Session; page_size=4 still matches oracle
- [ ] no CoW, no cages, no scheduler, no Hub
- [ ] README shows `Session` / `generate`
- [ ] ARCHITECTURE / §LXXVIII status truthful
- [ ] receipt
