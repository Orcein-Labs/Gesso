# BREADTH-2 — one capability authority

**Date:** 2026-10-03
**Follows:** [BREADTH-1_FINDINGS.md](BREADTH-1_FINDINGS.md) (`2a3136f`)
**Status:** complete and verified green.

---

## The seam

BREADTH-1 fixed a report that **overstated**. In fixing it, and in committing,
I shipped a report that **understated** — and it shipped undetected.

Three declarations of one fact were live:

| layer | vocabulary | answer about `:attention` |
|---|---|---|
| `CPU_SUPPORTED_CAPS` | `:rope`, `:swiglu`, … (operator) | absent → `false` |
| `_implemented_capabilities` | `:rope_none`, `:swiglu_ffn`, … (semantic) | present → `true` |
| `supports(CPU, :rope_none)` | asked under the operator name | `false` (`:rope` is `true`) |

`required_semantics` emits the *semantic* names; the backend sets held the
*operator* names; and the matrix asked a hand-kept set that answered to
neither. `:attention` was in `_implemented_capabilities` and in nobody's
backend set, so the matrix said implemented while `supports` said false.

Then BREADTH-1 made CPU run scaled RoPE. Nothing told the declarations. The
matrix went on reporting `:rope_linear` and `:rope_llama3` as unreachable —
the mirror image of the bug it had just been fixed for, one commit later.

## The fix: `supports()` is the authority

- The backend sets now hold **the semantic symbols `required_semantics`
  emits**, not operator names. `:rope` becomes `:rope_none` / `:rope_linear`
  / `:rope_llama3`, kept distinct because the distinction *is* the information.
  `:attention` — which CPU has implemented all along — is now actually listed.
- `_implemented_capabilities` is **deleted**. One authority, not two.
- `import_report`, `compatibility_matrix` and `compatibility_table` take a
  `backend` keyword (defaulting to `CPUBackend`) and derive every answer from
  `supports(backend, cap)`.
- CUDA and Lava claim only `:rope_none`, because their `rope!` still refuses a
  scaled policy at the operation.

The matrix now has a **backend dimension**, which it needed all along:
"implemented" was never a global property, and a report that cannot express
"CPU has it and CUDA does not" will keep making one of the two mistakes.

```
| architecture | backend | base form | runnable configs | first missing |
|---|---|---|---|---|
| llama | cpu    | runs | 3/48 | dense_ffn |
```

Runnable went **1/48 → 3/48** — the three rope kinds, which CPU genuinely runs.
`unreachable` shrank to the four real gaps:

```
[:dense_ffn, :qk_norm, :moe_routing, :sliding_window_attention]
```

## Two bugs the new tests caught, in my own new code

**`_REQUIRED_ORDER` derived from a single probe.** A `RoPEPolicy` carries one
kind, so deriving the capability order from one spec silently omitted
`:rope_linear` and `:swiglu_ffn`, which then sorted by fallback rank.

**…then the fix encoded the loop.** Deriving it from the sweep's enumeration
order made the order an artifact of `_MATRIX_ACTIVATIONS` — reordering that
tuple would silently change which capability `first_missing` names. `dense_ffn`
sorted *last* purely because `:gelu` is the second activation.

Both were caught by the Pass J fence, not by reading the code. The resolution is
`_REQUIRED_RANK`: a capability's rank is the **minimum position it occupies in
any probed variant**, derived from `required_semantics` and stable under
enumeration order.

```
rmsnorm 1 · attention 2 · matmul 3 · {swiglu,dense}_ffn 4 · optional 5
```

## Fences

- nothing called `unreachable` that the backend can actually run — guards the
  **understating** direction, which is the one I shipped
- while anything unimplemented is reachable, no row may read total — guards
  **overstating**, the original bug
- every row names its backend
- a device-free `ProbeNoScaledRope` backend (CPU's set minus scaled RoPE,
  modelling CUDA) must produce a **strictly narrower** matrix. If the two ever
  read the same, the `backend` argument is decorative and "one authority" is an
  authority that only ever gets asked one question.

---

## Receipt (§LXXII)

- **what changed:** backend capability sets moved to the semantic vocabulary;
  `_implemented_capabilities` deleted; report + matrix take a backend and
  derive from `supports`; capability rank derived from minimum position.
- **why:** three declarations of one fact let a generated report contradict the
  code in both directions, undetected across a commit.
- **tests:** full suite **2731 pass / 0 fail from this change**, 7m47s. Pass J
  951 assertions; `test_breadth0` 1105/1105. The 4 failures and 2 errors are
  the pre-existing untracked `test_decode_scratch.jl` WIP.
- **numerical delta:** none. No kernel, no allocation path, no tensor layout.
  `Session.inv_freq` behaviour is unchanged since BREADTH-1.
- **before/after benchmark:** not run — no hot path touched (§XXXIII).
- **compile-time / memory impact:** nil. Sets are compile-time constants.
- **hardware / workload / model / backend:** CPU Float64, toy2. CUDA and Lava
  capability sets edited but **not executed** — the extension seam tests ran,
  the matrices were not compared against a real device.
- **measurement environment:** julia 1.12.6, WSL, `-t 2`.

## Known limitations

- **CUDA/Lava matrix rows are asserted nowhere.** The per-backend proof uses a
  synthetic probe backend, because `CUDABackend` only exists once CUDA is
  loaded and comparing real device matrices needs a device.
- **Operator-name capabilities are gone from `supports`.** `:rope` and
  `:swiglu` no longer answer — deliberately, one vocabulary. Three test loops
  were updated. Any out-of-tree caller querying those names gets `false`, which
  is the safe default but not a loud failure.
- **`test_decode_scratch.jl` remains the only red.** Untracked, references a
  nonexistent `_workspace_pointers`, still needs a decision.