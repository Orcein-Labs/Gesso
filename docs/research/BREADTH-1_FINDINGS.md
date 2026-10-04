# BREADTH-1 — the report stops overstating, and the door opens

**Date:** 2026-10-03
**Preceded by:** [BREADTH-0_FINDINGS.md](BREADTH-0_FINDINGS.md) (`ca5383e`)
**Status:** complete and verified green.

---

## What prompted it

BREADTH-0 landed 6 architecture families and a generated compatibility matrix.
Read on its own, that matrix said:

```
gemma 0 — llama 0 — mistral 0 — phi 0 — phi3 0 — qwen2 0 —
```

**Every family, zero missing capabilities.** That is the number a status
meeting quotes. It was also false of the world.

The same machinery, asked about a real Llama-3 checkpoint, answered correctly:

```
Execution capabilities:
    rope_llama3               NOT IMPLEMENTED
Failure boundary: rope_llama3
```

Both were true. `import_report` was config-aware and honest;
`compatibility_matrix` probed **one default config per family** and presented
itself as the answer to "what can Gesso run?"

This is §LXX in the reporting direction: a report that overstates is worse than
no report, because nobody knows to distrust it. The same failure species found
twice elsewhere in this project — a doc that promises an escaping
kill-switch, a revival report whose labels read like a restore.

---

## Two changes

### 1. The matrix sweeps the config space

`compatibility_matrix()` now probes every capability-relevant configuration of
each family and reports the **worst case** alongside the runnable fraction.

Only five axes can change `required_semantics` — `rope.kind`,
`activation_kind`, `features` (`:qk_norm`, `:moe`), `sliding_window`. Sizes,
head counts, bias flags, norm kind and tied embeddings provably cannot, and
probing them would only multiply identical rows. The axes are read off
`required_semantics` itself rather than restated from memory, and
`_REQUIRED_ORDER` is *derived* from a maximal probe so the "first missing"
ordering cannot drift from the real one.

48 configurations per family. The result:

```
| architecture | base form | runnable configs | first missing across configs |
|---|---|---|---|
| gemma | runs | 1/48 | dense_ffn |
| llama | runs | 1/48 | dense_ffn |
| …      |      |       |             |

unreachable = [:dense_ffn, :qk_norm, :sliding_window_attention,
               :moe_routing, :rope_llama3, :rope_linear]
```

A row reading `first_missing === nothing` now means *every* configuration of
that family runs, not merely the default. `compatibility_table` prints the
runnable fraction, so "all green" is unrepresentable in the artifact a reader
actually sees.

**The fence.** `test/test_breadth0.jl` Pass J now asserts that while any
unimplemented capability is reachable from any family's config space, **no
family row may read total** — and that every row states its runnable
fraction. The old bug cannot recur silently.

### 2. The engine reads the positional policy

`Session` previously refused any scaled RoPE policy **at construction**:

```julia
throw(LoweringNotImplemented(Symbol("rope_", _policy.kind), backend_name(backend)))
```

That was correct while the engine threaded `theta` only — a scaled policy
running unscaled is exactly the silent representation change §LXX forbids. But
the CPU oracle had implemented scaled policies since Pass D, and the CPU
operator already accepted `inv_freq`. **The capability existed; the door was
shut.**

`Session` now reads the policy once (`inv_freq = tensors_rope_inv_freq(tensors,
d_head)`) and threads it to every `rope!`, mirroring the oracle exactly.
`nothing` still means unscaled, so every existing model stays bit-identical.

Refusal did not disappear — it moved to where §LXX wants it:

| policy | before | after |
|---|---|---|
| `:none` | runs, bit-identical | **unchanged** |
| `:linear`, `:llama3` | `LoweringNotImplemented` at construction | runs on CPU, matches the oracle |
| `:linear` on CUDA/Lava | declined at construction | declines at `rope!` |
| `interleaved=true`, partial `rotary_dim` | `LoweringNotImplemented` | **unchanged** — still fails closed |

---

## Tests

`test/test_session.jl` gains a BREADTH-1 testset that asserts four things, and
the last two are the ones that keep the first two honest:

1. `:linear` and `:llama3` sessions match `reference_generate` exactly.
2. The unscaled path is unchanged and `inv_freq === nothing`.
3. **The policy is not decorative** — scaled and unscaled output must differ,
   or "engine matches oracle" could be true because *both* ignore the policy
   and agree on the same wrong answer.
4. Unsupported policy *dimensions* still fail closed.

---

## Receipt (§LXXII)

- **what changed:** `compatibility_matrix` sweeps 48 configs/family and reports
  worst case; `Session` threads `inv_freq`; Pass D's contract updated from
  "refuses at construction" to "reads the policy, fails closed at dimensions".
- **why:** the matrix overstated Gesso's reach (§LXX); the engine refused a
  capability the oracle already had.
- **tests:** full suite `julia scripts/test.jl` — see below. `test_breadth0`
  1081/1081; BREADTH-1 testset 14/14.
- **numerical delta:** none for existing models. Unscaled `:none` returns
  `inv_freq === nothing` and takes the identical `m * theta^(-2i/d)`
  expression — §XIII regression law holds by construction, and asserted.
- **before/after benchmark:** not run. This change touches no kernel and no
  allocation path; a timing claim here would be theatre (§XXXIII).
- **compile-time / memory impact:** one extra `Vector{Float64}` of length
  `d_head ÷ 2` per Session, and only for scaled models. `nothing` for
  unscaled — zero new allocation.
- **hardware / workload / model / backend:** CPU Float64 only, toy2 fixture,
  no device. CUDA and Lava were not exercised and still decline scaled RoPE.
- **measurement environment:** julia 1.12.6, WSL, `-t 2`.

---

## Known limitations

- **CUDA and Lava still cannot run a scaled policy.** They accept the
  `inv_freq` kwarg and refuse `nothing`-not-`nothing` at the operation. Making
  them work is a separate, unstarted task.
- **The matrix does not model quantized checkpoints.** Every real quantization
  format is a bigger gap than any axis here; the matrix's 48 configurations
  describe *architecture* variants, not *representation* variants.
- **The variant set is a declared list, not derived from configs.** It is
  honest about being a sweep of axes that can change the answer, but a family's
  adapter could in principle expose a dimension not in these five.
- **No end-to-end scaled-RoPE checkpoint was run.** The proofs are toy2 with a
  synthetic policy; no Llama-3 weights were available in this environment
  (§LXXVI: an ops fact, not an assumption).