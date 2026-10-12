# Speculation seam: draft sources and the verify window

Status: contract, 2026-10-08. This doc implements the first step of §LV (Speculative Execution) and
Phase 20 (§XCIII) in `docs/Gesso_Stack.md`. It fixes the boundary so draft sources can be built
apart from the runtime and dropped in without changing either side.

## The split

```
  DraftSource (any number)              Gesso runtime
  ─────────────────────────             ─────────────────────────────────
  symbolic / template / suffix  ──ids──▶ verify window (T tokens, one pass)
  lookup / MSE / grammar        ◀─n────  accepts the longest prefix that
                                         greedy decode would have produced
```

- **Draft sources** propose token ids. They run on the host, never call the model, and never
  touch the GPU. They own their own cleverness: grammars, templates, executable procedures,
  retrieved patterns, MSE, suffix indexes.
- **The runtime** owns verification. A window of `T = 1 + length(draft)` tokens goes through
  every layer in one pass. Token `t`'s argmax is bit-identical to plain greedy decode after token
  `t`. The runtime keeps the longest agreeing prefix, plus its own next token, and rolls the rest
  back. That includes the linear-attention (Gated DeltaNet) recurrent state and the KV cells.

## Acceptance is model agreement only

A draft token is accepted exactly when greedy decode would have emitted it. Semantic correctness
(the symbolic procedure gave a valid result) plays no part in acceptance. It only makes agreement
more likely. So speculation never changes output: the exact-id gate that guards the runtime also
guards every draft source. Sampled decoding (temperature > 0) is a later extension, and it must
keep the output distribution exact.

## Interface (Julia)

```julia
abstract type DraftSource end

# What a source may read. Host-only and immutable for the call.
struct DraftContext
    ids::AbstractVector{Int}   # every token so far: prompt + accepted output
    detok                      # id -> text, for sources that reason in text
    vocab_size::Int
end

struct Draft
    ids::Vector{Int}           # proposed continuation (may be empty)
    origin::Symbol             # e.g. :template, :mse, :suffix (for receipts)
end

# Return at most `max_tokens` ids. An empty Draft means "no opinion".
propose(src::DraftSource, ctx::DraftContext; max_tokens::Int)::Draft

# Feedback after verification: how many of `d.ids` were accepted.
observe!(src::DraftSource, d::Draft, accepted::Int)::Nothing
```

Rules for a `DraftSource`:
- It must be deterministic for a given context and history, so runs are reproducible.
- It must be cheap. Its cost is charged to the token. The runtime records `propose` wall time
  per origin.
- It must not hold references into runtime buffers. Communication is the `DraftContext` and the
  returned ids only.
- It may decline: an empty draft costs one ordinary decode step and nothing else.

## Measuring whether a source pays

A window costs about one decode step plus the union of the drafted tokens' extra work. Strata
measured 1.75x one token's CPU expert misses for T=2, 2.4x for T=3, and 3.05x for T=4 (Strata
`include/strata/core/verify.hpp`, MIT). A source pays only where
`E[accepted + 1] / window_cost > 1`. Every run records, per origin: windows, mean draft length,
mean accepted, propose cost, verify cost, rollback cost, tokens/s against a no-draft arm on the
same prompt, and the exact-id check. A source that does not pay on its target workload is turned
off for that workload. It is not tuned until it looks good.

## Ownership

- Draft sources (including the symbolic proposer): built separately; ported in as `DraftSource`
  implementations under `src/Speculation/`.
- Verify window, state commit and rollback, multi-token kernels with single-token bit parity:
  the runtime throughput campaign (`docs/goals/ARCH_GAUNTLET.md`).
- The first built-in source: a suffix/prompt-lookup drafter. It serves as the baseline every other
  source must beat.
