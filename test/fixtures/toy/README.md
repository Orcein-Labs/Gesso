# Toy fixture pack (laboratory material — NOT Harpe API)

Purpose: give Phase 2's CPU-oracle differential tests a **tiny, fully
deterministic model to reason about** without pre-deciding anything about
Harpe's semantic core. Everything here is **data**. The moment any of this
grows a Julia type outside `test/`, it has left its lane (see
`test/test_empty_core.jl` — the fence applies).

## Files

| File | Schema tag | Role |
|---|---|---|
| `model.toml` | `harpe-toy-fixture-v1` | architecture as data: dims, ordered block list, vocab size |
| `tokenizer.toml` | `harpe-toy-tokenizer-v1` | vocabulary as data: dense ids 0..31, control tokens, placeholder strings |
| `expected_logits.toml` | `harpe-toy-expected-logits-v1` | expected-outputs SLOT (schema + provenance only; values arrive with the Phase 2 oracle) |

## What this is not

* Not a `ModelIR`, not a preview of one, not a semantic-parameter type.
* `kind` strings (`attention`, `mlp`) are fixture data that **Phase 2's
  oracle math interprets**; they name no Harpe operator type.
* The tokenizer is placeholder strings, not a tokenizer design.
* Filling `expected_logits.toml` **requires** the Phase 2 CPU oracle math
  to exist first. Hand-computing logits before an oracle exists would bake
  arithmetic errors into the laboratory. Until then the slot stays empty,
  and the pack tests enforce exactly that.

## Weight derivation protocol (contract)

There are **no stored weight binaries**. Weights are DERIVED, so the pack
is reproducible from three files of text:

1. Derive the master RNG: `deterministic_rng(fix_seed(<fixture seed>))`
   (`HarpeTestHelpers.deterministic_rng`; the `<fixture seed>` is recorded
   in the loaded fixture and echoed by `load_toy_fixture`).
2. Walk the architecture's ordered blocks; for each weight tensor the
   oracle's reference implementation needs, draw `randn(rng)` in
   **row-major index order**.
3. One master stream — no per-tensor reseeding, no jumps, no skipping.
   Any change to the walk order is a **fixture protocol break**: bump the
   fixture `schema` tags, record it in this README, and re-derive the
   expected logits at the same commit.

The loader and its enforcement live in `test/toyfixtures.jl`; the
enforcement tests live in `test/test_toyfixtures.jl`.
