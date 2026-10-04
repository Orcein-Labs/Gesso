# CAPABILITY-0a — the conformance biconditional

**Date:** 2026-10-03
**Follows:** [BREADTH-2_FINDINGS.md](BREADTH-2_FINDINGS.md) (`39c055a`)
**Status:** complete and verified green.

---

## The seam

BREADTH-2 collapsed three declarations of one fact into one authority
(`supports(backend, cap)`). That is the right shape, and it is still a
**hand-written Set**. Nothing tied the Set to the code it describes. So the
question ChatGPT asked — *"there should be one declaration per fact"* — was
answered halfway: one *place*, no *proof*.

Two live defects were found while writing the proof, both shipped in `39c055a`:

### 1. `import_report` rebuilt a lossy probe (a false NEGATIVE this time)

`_missing_capabilities` took an `ArchitectureCapabilities` and re-derived the
required list from it. That derivation propagated `qk_norm` and dropped `moe` —
a BREADTH-0 overcorrection for an earlier false positive. Result: a `moe`
spec was declared **fully runnable** while the compatibility matrix, correctly,
named `:moe_routing`.

This is the mirror of the bug BREADTH-1 fixed, one commit later, and it was
shipped *by the commit that claimed to close the seam*.

**Fix:** `_missing_capabilities(reqs::AbstractVector{Symbol}, backend)` now
filters the spec's own `required_semantics` list. Re-deriving a list you
already hold is how a declaration drifts from the thing it describes.

### 2. `arm I` iterated the wrong set (and `@test_broken` hid it)

The first version of the biconditional walked only the capabilities a *spec can
require*. `:softmax` and `:embedding_lookup` are claimed by CPU and emitted by
**no** spec — they are engine-internal — so neither was ever executed by the
"claimed ⇒ proven" arm. A claimed capability with no proof, in the arm whose
entire job is preventing exactly that.

It also branched to `@test_broken false` when a claim had no probe. A mutation
check killed that: adding a fabricated `:phantom_capability` to
`CPU_SUPPORTED_CAPS` left the suite **green**, because Julia counts `broken` as
a non-failure. A declared capability with no proof is the defect, so it is now
a plain failure.

## The biconditional

`test/test_capability_conformance.jl`. Five arms, each failing on its own:

| arm | law | catches |
|---|---|---|
| I | `supports(b,cap)` ⇒ the operation **executes** | overstated claim |
| I′ | the operation **executes** ⇒ `supports(b,cap)` | *under*stated claim |
| II | `!supports(b,cap)` ⇒ the report **names** it as the failure boundary | the `moe` bug |
| III | report boundary == first unsupported capability | drift between surfaces |
| IIIb | report == matrix, per variant, matched on the **required list** | label-coincidence hiding drift |

Arm I′ was **missing on the first pass** and only surfaced under mutation:
deleting `:rope_llama3` from `CPU_SUPPORTED_CAPS` — while the CPU `rope!`
plainly executes it — left every other arm green. Arm I skips unclaimed
capabilities, and arm II is *satisfied* by a report that honestly says "blocked
at `:rope_llama3`". Self-consistent, and still wrong: **a model that runs gets
refused.** That is the same false negative that made BREADTH-1 necessary.

Both directions are now mutation-verified in both directions:

| mutation | result |
|---|---|
| add `:phantom_capability` (claims nothing implements) | **FAILS**, arm I |
| delete `:rope_llama3` (CPU executes it) | **FAILS**, arm I′ |

### `:attention` gets a real fixture

`:attention` has **no lowering operator of its own** — the engine composes it
from `matmul` / `softmax` / `matmul`. That makes it the one capability easiest
to assert without proving anything, which is precisely how it stayed split-brain
for two commits.

Its probe now executes the composition through the backend's own operations,
written out in the test rather than delegated to a session — a fixture that
calls the engine proves the engine agrees with itself. It is further checked
against `_attention_reference`, plain-Julia causal attention sharing **no
operator** with the fixture, to `1e-12` (observed `2.2e-16`). A backend that
returned garbage would fail arm I's sibling check, not just "not throw".

### Three probe signatures were wrong before any arm could pass

Not code defects — probe defects, and the distinction matters: a probe calling
an operator with the wrong argument types falls through to the lowering stub and
returns `:refused`, which is *indistinguishable from a missing capability*.
Recorded because it is a trap for anyone adding a probe:

- `matmul!` requires a `ProjectionWeight` W, stored **(out, in)** — `dst = x * Wᵀ`
- `softmax!` requires **both** operands to be `TemporaryWorkspace`
- `matmul!` requires an **Activation destination**; only `softmax!` takes
  workspace. Passing workspace to `matmul!` reads `:refused`.

The file's header states this so the next probe author inherits the trap rather
than the debugging round.

## Fences

- `_conformance_run` rethrows anything that is not `LoweringNotImplemented`. A
  probe that dies of `UndefVarError` is a **broken probe, not a missing
  capability**, and must not be laundered into a green arm.
- every capability in the vocabulary must be accounted for in
  `_REQUIRED_RANK`, and every claimed one must have a probe
- arm IIIb matches matrix variants on the **required list**, not on a shared
  label — a coincidental label match is how drift hides

---

## Receipt (§LXXII)

- **what changed:** `import_report`'s `_missing_capabilities` filters the real
  `required_semantics` list instead of rebuilding a lossy probe (defect 1);
  new `test/test_capability_conformance.jl` with the five arms (49 assertions),
  including a from-scratch `:attention` fixture and a numerically independent
  attention reference.
- **why:** one authority was still one *unverified* hand-written list; a
  committed report overstated a `moe` spec as fully runnable, and the arm meant
  to prevent that was itself blind to two claimed capabilities.
- **tests:** targeted **49 pass / 0 fail**, ~8s (`julia --project=test`, the
  file plus helpers). Full suite `julia scripts/test.jl`: **2780 pass / 4 fail
  / 2 error / 8 broken**, 8m45s — a delta of exactly **+49** over BREADTH-2's
  2731, i.e. this file and nothing else. Every one of the 4 failures and 2
  errors is in the pre-existing untracked `test/test_decode_scratch.jl` WIP
  (lines 71, 76, 110, 127, 152, 178), and all 8 broken are nested inside that
  same file's named skips. Verified twice: pre-format and post-format.
- **numerical delta:** none. No kernel, allocation path, or tensor layout
  touched. The attention fixture is test-local and never runs in production.
- **before/after benchmark:** not run — no hot path touched (§XXXIII).
- **compile-time / memory impact:** nil. Test-only file; the source change is a
  list comprehension over an existing vector.
- **hardware / workload / model / backend:** CPU Float64. Probes are synthetic
  tensors (3 tokens × 2 heads × 4 dims for attention); no checkpoint is loaded.
  CUDA and Lava capability sets are untouched by this change and remain
  unexecuted — the `ProbeNoScaledRope` synthetic backend in `test_breadth0`
  covers the `backend` argument, but no *probe* runs against a real device.
- **formatter:** `julia scripts/format.jl` touched exactly two files —
  `src/Inference/import_report.jl` and `test/test_capability_conformance.jl` —
  verified by md5 before/after across all of `src/` and `test/`. No unrelated
  file was reformatted.
- **measurement environment:** julia 1.12.6, WSL, `-t 2`.

## Known limitations

- **The arms test CPU only.** `ProbeNoScaledRope` (in `test_breadth0`) covers
  the `backend` argument, but no conformance *probe* runs against CUDA or Lava:
  `CUDABackend` requires CUDA loaded, and the operators need a device.
- **`supports` is still a hand-written Set.** This work proves the Set agrees
  with execution; it does not derive the Set from execution. A registration API
  (`register_capability!(CPU, :attention, ...)`) was considered and **rejected**
  for now: it is another declaration maintained the same way, and the biconditional
  is what makes a declaration *checkable*. Deriving from dispatch truth is the
  natural CAPABILITY-0b, once the vocabulary stops growing.
- **The three-axes question ChatGPT raised is unresolved.** `supports` still
  answers SEMANTIC, STRATEGY (`:argmax`, `:attn_gemm`) and REPRESENTATION
  (`:quantize`) axes from one function. The docstring names the axes; splitting
  them into separate vocabularies is CAPABILITY-0b or later, deliberately not
  done here.
- **`test_decode_scratch.jl` remains the only red.** Untracked, references a
  nonexistent `_workspace_pointers`, needs a decision across five commits now.