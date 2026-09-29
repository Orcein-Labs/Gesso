# /goal PHASE 2 — CPU ORACLE

**Status:** LANDED (Buffy, 2026-09-29). Next: `docs/goals/PHASE3_FIRST_IMPORT.md`.

**For:** Buffy (mechanical implementation)
**From:** Grok (encoding owner)
**Canon:** `docs/Gesso_Stack.md` §LXXV, §X, §XII, §XXX, §CIX
**Map:** `docs/ARCHITECTURE.md`
**Depends on:** Phase 1 complete (`docs/goals/PHASE1_SEMANTIC_CORE.md`)
**Packets:** 1 and 2 stay closed. Do not reopen them.

---

## Start condition

Phase 1 is on the tree you inherit:

- `make test` / `make format-check` green
- §CIX fence pins Phase 1 vocabulary (`test/test_empty_core.jl`)
- `toy2` is expressible (`test/test_phase1_exit.jl`)
- `expected_logits.toml` is still an empty slot
- CPU methods still throw `LoweringNotImplemented`

If Phase 1 is unfinished, stop.

## One-sentence objective

`toy2` runs on `CPUBackend`: a deterministic forward pass produces known
logits, and greedy generation is deterministic.

## What this sprint is not

- CUDA, Lava, safetensors, a real checkpoint
- sampling with temperature / top-k / RNG (greedy argmax only)
- batch > 1
- GQA path (toy2 is MHA: `n_kv_heads == n_heads`)
- `quantize!` / `dequantize!` math (still `LoweringNotImplemented`)
- a new operator in the vocabulary (no `add!`, no `residual!`)
- changing ModelIR node types or the §CIX family list
- filling logits by hand without the oracle
- global receipt ids, `ExecutionPhase`, training

---

## Math law (pin this; do not invent a second recipe)

All CPU math is **Float64**. Arrays are dense `Array{Float64}` in `storage`.
`shape` matches `size(storage)`.

### Operator methods

Phase 1's 4-arg methods stay as the generic decline:

```
op!(::AbstractGessoBackend, ::SemanticTensor, ::SemanticTensor, ::Workload)
    -> LoweringNotImplemented
```

Phase 2 adds **more specific** `CPUBackend` methods. Same function names.
No second vocabulary. Extra arguments are allowed; more-specific methods
win. Un-typed calls like `rmsnorm!(cpu, nothing, nothing)` still hit the
stub and still throw `LoweringNotImplemented` (inventory tests stay green).

Signatures (this is the contract):

```
embedding_lookup!(::CPUBackend, dst::Activation, table::EmbeddingTable,
                  tokens::AbstractVector{Int}, ::PrefillWorkload)
    dst[t, :] = table[tokens[t] + 1, :]
    tokens are 0-based fixture ids; Julia arrays are 1-based.

rmsnorm!(::CPUBackend, dst::Activation, x::Activation,
         scale::FrozenParameter, ::PrefillWorkload)
    ε = 1e-6
    rms = sqrt.(mean(x.^2; dims=last) .+ ε)
    dst = (x ./ rms) .* scale
    last dim is the feature dim.

rope!(::CPUBackend, q::Activation, k::Activation,
      positions::AbstractVector{Int}, ::PrefillWorkload)
    LLaMA-style pairwise rotate on the head feature dim.
    θ_i = 10000^{-2i/d_head}, i = 0 .. d_head/2-1
    positions are 0-based.
    q and k are (seq, n_heads, d_head) in storage; rotate in-place.

matmul!(::CPUBackend, dst::Activation, x::Activation,
        w::ProjectionWeight, ::PrefillWorkload)
    W has shape (out, in). x is (seq, in) or flattened equivalent.
    dst = x * transpose(W)

softmax!(::CPUBackend, dst::TemporaryWorkspace, scores::TemporaryWorkspace,
         ::PrefillWorkload)
    row-wise softmax, max-subtract for stability.
    scores are (seq_q, seq_k) per head; apply causal mask BEFORE softmax
    (masked positions = -Inf). Causal: position i may attend to j ≤ i.

swiglu!(::CPUBackend, dst::Activation, gate::Activation, up::Activation,
        ::PrefillWorkload)
    silu(g) = g * sigmoid(g)
    dst = silu(gate) .* up
```

DecodeWorkload: same signatures with `::DecodeWorkload`. Decode math is
the same elementwise/matmul as prefill. KV *append* lives in the
interpreter (item C), not inside `matmul!`.

`quantize!` / `dequantize!`: do not implement. Decline stays.

Residuals (`x = x + sublayer(x)`) are **interpreter-level** `storage`
addition. Do not add an `add!` operator.

### Block recipe (pre-norm transformer)

ModelIR `Block` stores `Attention` + `SwiGLU` only. The interpreter still
runs RMSNorm and RoPE — those primitives already exist; they are not new
node types. Per block:

```
h = rmsnorm(h, attn_scale)
q = matmul(h, Wq); k = matmul(h, Wk); v = matmul(h, Wv)
split heads → rope(q, k, positions) → scores = (q kᵀ) / √d_head
causal softmax → (attn @ v) → merge heads → matmul(O)
h = h + that                                  # residual, interpreter add
h = rmsnorm(h, ffn_scale)
gate = matmul(h, Wgate); up = matmul(h, Wup)
h = h + matmul(swiglu(gate, up), Wdown)       # residual
```

Tied embedding as the output head: `logits = h * transpose(E)` with `E`
the embedding table `(vocab, dim)`. No separate `lm_head`.

### Weight walk (fixture protocol)

`toy_weights(fixture, n)` is the only RNG. Consume the stream in this
order, row-major, `Float64`, no jumps, no per-tensor reseeding:

```
E            (vocab, dim)
for each block, in model order:
    Wq       (dim, dim)
    Wk       (n_kv_heads * d_head, dim)     # toy2 MHA: (dim, dim)
    Wv       (n_kv_heads * d_head, dim)
    Wo       (dim, dim)
    Wgate    (hidden, dim)
    Wup      (hidden, dim)
    Wdown    (dim, hidden)
    attn_rms (dim,)
    ffn_rms  (dim,)
```

`d_head = dim / n_heads`. Integer division must be exact (toy2: 16/2 = 8).

Phase 1's exit test sketched K/V as `(d_head, dim)` as a *representative*
shape. The oracle uses the packed MHA shape above. Do not "fix" Phase 1
shapes in that test except to match this walk if the test would otherwise
lie.

Any change to this order is a fixture protocol break: bump
`gesso-toy-fixture-v1` and the README together.

### Prompt for known logits

Exactly this token-id sequence, 0-based, BOS-prefixed:

```
[1, 3, 4, 5]     # BOS, then tokens 3, 4, 5
```

`seq_len = 4`. Logits shape `(vocab_size, seq_len)` = `(32, 4)`, column
`t` is the logits after consuming tokens `1..t` (standard next-token
head at each position). Persist as `[[value]]` rows in
`expected_logits.toml` per the existing slot comments.

Provenance **must** be filled:

```
oracle = "cpu"
commit = "<git rev-parse --short HEAD at fill time>"
fixture_seed = "<decimal or 0x… of fixture.seed>"
```

### Generation (item C)

Greedy argmax of the last position. Start from `[BOS]`. Stop at `EOS`
or `max_new_tokens = 8`. DecodeWorkload path must use the KV cache
(append the new K/V; do not re-prefill the whole prompt each step).
Same tokens in ⇒ same tokens out, always.

---

## Work items (sequence; land A before B before C)

### A — CPU operator methods

**Objective.** Every implemented op has a `CPUBackend` method that
computes the formula above, with unit tests against the formula on tiny
arrays.

**Permitted files**

```
src/Operators/Operators.jl
src/Operators/*.jl              # split CPU methods out if the file grows
src/backends.jl                 # supports(::CPUBackend, cap) only
src/Gesso.jl
Project.toml                    # LinearAlgebra stdlib, if you `using` it
test/runtests.jl
test/test_cpu_ops.jl
test/test_backends.jl           # supports() fence: implemented caps become true
docs/ARCHITECTURE.md
```

**Dependency.** If you `using LinearAlgebra`, add it to `Project.toml`
**and** the dependency-law allowlist in `test/runtests.jl` with the
justification: CPU reference matmul (§LXXV). No other deps.

**`supports`.** `supports(::CPUBackend, cap)` returns `true` for
`:rmsnorm, :rope, :softmax, :swiglu, :matmul, :embedding_lookup`.
Unknown caps and `:quantize` / `:dequantize` stay `false`. Update the
ready-room "supports stays false" test; do not delete the unknown-cap
case.

**Invariants**

- Stub 3-arg calls still throw `LoweringNotImplemented` with `op` + `:cpu`.
- Vocabulary list in `backends.jl` unchanged.
- `quantize!` / `dequantize!` still decline.
- Float64 only. No Float16/BFloat16 path.

**Tests.** One `@testset` per op: a hand-sized array, compute the formula
in the test (or a comment-local oracle), `approx_eq` with `atol=0, rtol=0`
if bit-identical, otherwise `atol=1e-12` declared in the test. Causal
softmax: a 3×3 scores fixture proving `-Inf` mask. RoPE: even `d_head`,
two positions, check pairwise rotate.

**Performance.** N/A.

**Artifact.** CPU methods exist. No model forward yet.

---

### B — Prefill oracle + known logits

**Objective.** `toy2` prefill on `[1,3,4,5]` produces logits that are
written into `expected_logits.toml` and re-read by tests.

**Permitted files**

```
src/Inference/Inference.jl      # reference prefill interpreter
src/Gesso.jl
test/test_reference_prefill.jl
test/test_toyfixtures.jl        # loader must ACCEPT a filled slot
test/toyfixtures.jl             # stop refusing values; validate schema
test/fixtures/toy/expected_logits.toml
test/fixtures/toy/README.md
test/runtests.jl
docs/ARCHITECTURE.md
README.md
scripts/freeze.jl               # new test files on the curated list
```

You may *read* `test/test_modelir.jl` (`toy2_modelir`) and
`test/toyfixtures.jl` (`toy_weights`). Materialize weights into
`SemanticTensor.storage` in test or in Inference; do not invent a
checkpoint format.

**Interfaces**

```
reference_prefill(model, tensors, tokens) -> Matrix{Float64}
```

- `model` is `Gesso.Model`
- `tokens` is `Vector{Int}` (0-based)
- result shape `(vocab_size, seq_len)`
- uses `CPUBackend()` + `PrefillWorkload()` only
- lives in `Inference` (first slice of the engine; scheduler is still
  Phase 5)

**Loader change.** `load_toy_fixture` currently errors if the logits
slot has values. After fill: if provenance.oracle == `"cpu"` and
`[[value]]` is present, return them as a `(32, 4)` matrix. If the slot
is half-filled (values without provenance, or provenance without
values), error loudly.

**Invariants**

- Two calls of `reference_prefill` on `toy2` + `[1,3,4,5]` are
  `approx_eq` with `atol=0, rtol=0` (same process).
- Persisted logits vs recomputed logits: `approx_eq` with
  `atol=1e-10, rtol=0` (cross-commit regression; BLAS can wiggle).
- `test/test_phase1_exit.jl` still must not *require* a forward pass.
  It may keep asserting whatever is still true; if it asserts
  `expected_logits === nothing`, update that one assertion.

**Tests.** Prefill shape, determinism, persisted-slot round trip,
causal (position 0 cannot see position 3).

**Performance.** N/A. No timing claims.

**Artifact.** `expected_logits.toml` filled. README slot language
updated. Phase 2 "known logits" exit met.

---

### C — Greedy generate + KV decode

**Objective.** Deterministic generation on CPU using `DecodeWorkload`
and a real KV append. Prefill the prompt, then one token at a time.

**Permitted files**

```
src/Inference/Inference.jl
src/Operators/Operators.jl      # DecodeWorkload methods if not already shared
src/Gesso.jl
test/test_reference_generate.jl
test/runtests.jl
docs/ARCHITECTURE.md
README.md
docs/Gesso_Stack.md             # §LXXV status line only: COMPLETE + date
```

**Interfaces**

```
reference_generate(model, tensors, prompt; max_new_tokens=8) -> Vector{Int}
```

- greedy argmax at the last position
- stop on EOS (`2`) or `max_new_tokens`
- returned vector is the full 0-based id sequence (prompt + new)
- KV cache starts empty at prefill, appends on each decode step
- decode of token `t` must not recompute attention over the prompt
  from scratch (assert via a test that KV length equals prefix length)

**Invariants**

- Same prompt ⇒ same output ids, two runs.
- Generate from `[1,3,4,5]` with `max_new_tokens=0` returns exactly
  the prompt (no extra tokens).
- Prefill logits at the last prompt position have the same argmax as
  the first generated token when `max_new_tokens≥1`.

**Tests.** Determinism, EOS/cap stop, KV length, argmax consistency
with prefill.

**Performance.** N/A.

**Artifact.** Phase 2 exit complete: forward pass, known logits,
deterministic generation. README / ARCHITECTURE / §LXXV status
truthful.

---

## Cross-item invariants

- Zero new third-party deps. `LinearAlgebra` stdlib is the only
  expected addition, with the allowlist edit.
- `libs/` untouched.
- No silent fallback. Missing CPU coverage still throws
  `LoweringNotImplemented`.
- §CIX fence: Operators still exports nothing (functions live on
  `Gesso`). Inference may start exporting `reference_prefill` /
  `reference_generate`. Do not punch new names through Semantics /
  ModelIR / Parameters without a packet.
- Formatter: `make format` before close.
- Receipt at close (§LXXII): what changed · why · tests · numerical
  delta (logits vs empty slot: first numbers exist; record
  `max_abs` of the filled matrix as a fingerprint, not a perf claim) ·
  compile-time impact · hardware CPU · workload prefill `[1,3,4,5]` ·
  model `toy2` · backend `cpu`.

## Escalation (stop and write a packet)

Write a new packet in `docs/DECISION_PACKETS.md` and stop the item if:

- the 4-arg Phase 1 surface cannot grow extra arguments without
  breaking dispatch, and you want a different calling convention
- you need a new operator name in `backends.jl`
- you need to mutate ModelIR (`Block` growing RMSNorm/RoPE fields)
- RoPE layout (pair rotate vs GPT-J interleaved) is forced by a
  later real-model importer to change — this sprint stays LLaMA-style
- bit-identical prefill fails across two calls in one process

Do not "just pick one" in code.

## Exit checklist

- [ ] A, B, C landed (or A+B green and C written as a follow-up
      work item in this file — known logits are the hard gate)
- [ ] `make test` green
- [ ] `make format` run
- [ ] `expected_logits.toml` filled with provenance
- [ ] loader accepts the filled slot and rejects half-filled ones
- [ ] `quantize!` / `dequantize!` still decline
- [ ] ARCHITECTURE / README / §LXXV status truthful
- [ ] receipt in the PR / close note
- [ ] no CUDA, no Lava, no real checkpoint
