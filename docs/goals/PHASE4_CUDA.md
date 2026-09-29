# /goal PHASE 4 — CUDA.JL EXECUTION

**For:** Buffy (mechanical implementation)
**From:** Grok (encoding owner)
**Canon:** `docs/Gesso_Stack.md` §LXXVII, §VII, §XX–§XXI, §LXX, §CIX
**Map:** `docs/ARCHITECTURE.md`
**Depends on:** Phase 3 complete (`docs/goals/PHASE3_FIRST_IMPORT.md`)
**Packets:** 1 and 2 stay closed.

---

## Start condition

Phase 3 is on the tree you inherit:

- `make test` green (736 + 1 named skip is the last known count)
- toy2 fingerprint unchanged (`max|logit| = 1686.4814860783201`)
- JSON is the only core third-party dep
- `ext/` does not exist yet; `[weakdeps]` is empty

If Phase 3 is unfinished, stop.

## One-sentence objective

`toy2` and the llama micro-model generate on an NVIDIA GPU through
`CUDABackend`, matching the CPU oracle within declared tolerance.
No GPU on the machine ⇒ named skip. CI never requires a device.

## Why this phase exists

Gesso architecture and Lava/kernel maturity are different problems
(§LII). CUDA.jl is the fast path that proves the backend seam while
Lava is still catching up. This sprint is that seam, not a kernel
contest.

## What this sprint is not

- CUDA as a **core** `[deps]` entry (law violation §VII)
- silent fallback to CPU when CUDA is requested (§LXX)
- FlashAttention, fused kernels, tensor cores, graph capture
- Float16/BFloat16 compute path (storage may originate as BF16 from
  safetensors; math on GPU this sprint is Float32 or Float64 as pinned
  below)
- Lava, quantization math, Magenta Memory, paged KV
- downloading weights
- changing §CIX types or the Llama name map
- claiming speedups vs vLLM / llama.cpp

---

## Laws (pin this)

### Extension, not identity

```
Project.toml
    [weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

    [extensions]
    GessoCUDAExt = "CUDA"
```

`ext/GessoCUDAExt.jl` is the only file allowed to `using CUDA`.
Core Gesso (`src/`) must compile and test with CUDA **not** loaded.

`CUDABackend` lives in the extension, not in `src/backends.jl`.
Core still "knows nothing else about specific vendors" except the
comment that the ext exists. Do not add `struct CUDABackend` to core.

The dependency-law test currently asserts `weakdeps` is empty. **Edit
that test** with the justification: CUDA.jl Phase 4 backend, package
extension, never a core dep. That friction is the point.

`test/Project.toml` may declare CUDA so Pkg.test can load the
extension. That is the test env, not core.

### No silent fallback

`CUDABackend()` with `CUDA.functional() == false` throws a typed
`GessoError` (`ERR_LAUNCH` or `ERR_RESOURCE_LIMIT` — pick one, use it
consistently, put the reason in the diagnostic). It does **not**
return `CPUBackend()`. `@gfallback` is for policy-permitted fallbacks
that a later phase might declare; this sprint has none.

### Same operators

Add more-specific methods on the existing `op!` names, dispatching on
`CUDABackend`. No second vocabulary. `quantize!` / `dequantize!` still
decline.

Math: **CuArray{Float32}** for GPU storage this sprint.

Why F32 not F64: 135M F64 is ~1.08 GiB and a poor GPU path; F32 is
the honest first NVIDIA lowering. CPU oracle stays F64. Compare with
declared atol, not bit-identity.

```
atol = 1e-3    # CUDA F32 vs CPU F64, micro-llama / toy2, seq ≤ 8
```

If a specific op exceeds that on micro-llama, fix the op (usually
softmax / rmsnorm reductions). Do not silently widen atol past `1e-2`
without a packet.

Implement with CuArray broadcasting + CUBLAS (`mul!` / `CUDA.CUBLAS`).
One `CUDA.@cuda` kernel per op is allowed if broadcast cannot express
it (RoPE, causal softmax). No PTX, no CUTLASS, no CUDA.jl extra
packages (KernelAbstractions, GPUArrays extras beyond what CUDA.jl
already reexports).

### Transfer is explicit

```
to_device(::CUDABackend, tensors) -> tensors'
```

Copies `Array` storage to `CuArray{Float32}` (conversion F64→F32 is
part of the lowering). Returns a new named-tuple / struct of the same
shape. Does not mutate the CPU tensors.

`reference_prefill` / `reference_generate` grow:

```
reference_prefill(model, tensors, tokens; backend=CPUBackend())
reference_generate(model, tensors, prompt; backend=CPUBackend(), max_new_tokens=8)
```

Default keeps every existing CPU test bit-identical. CUDA path:
caller passes `backend=CUDABackend()` **and** tensors already on
device. If backend is CUDA and storage is still `Array`, error
loudly (`ERR_INVALID_PLAN`): the interpreter does not copy.

`CUDA.synchronize()` after prefill/generate before reading logits
back to host.

### Skip law (same shape as SmolLM2 item D)

```
if CUDA is not in the test env or !isdefined(Main, :CUDA) or !CUDA.functional()
    named skip that says "no NVIDIA device — CUDA tests skipped"
```

`make test` on a CPU-only box stays green. Never skip the **extension
loads** tests that can run without a device (module exists when CUDA
is in the test env; `CUDABackend` is defined). Split:

| Test | Needs device? |
|---|---|
| `GessoCUDAExt` loads when CUDA is in the env | no |
| `CUDABackend` is not in core `names(Gesso)` without the ext | no |
| requesting CUDA with `!functional()` throws, does not CPU-fallback | no (functional is false) |
| toy2 / micro-llama CUDA prefill vs CPU | yes |
| CUDA generate greedy vs CPU argmax | yes |
| SmolLM2 CUDA | yes **and** `GESSO_SMOLLM2_DIR` |

### Benchmarks

When a device exists, append **one** corpus row: micro-llama CUDA
prefill `[0,1,2]` median ns + allocs, schema 0.2.0, after warmup,
no compile in the timed region. No comparison to CPU in the receipt
as a "speedup" unless both were measured the same way on the same
box — and even then it is a measurement, not a claim.

CPU-only machines: no new bench row required.

---

## Work items (sequence)

### A — Extension skeleton + seam tests

**Objective.** CUDA.jl is a weakdep. `GessoCUDAExt` loads. Core still
has no CUDA. Requesting a device that is not there fails explicitly.

**Permitted files**

```
Project.toml
ext/GessoCUDAExt.jl
src/backends.jl          # comments / docs only; no CUDABackend type
src/errors.jl            # only if you need a clearer diagnostic
src/Gesso.jl             # do not `using CUDA`
test/Project.toml        # CUDA for the test env
test/runtests.jl         # weakdeps allowlist
test/test_cuda_seam.jl
test/runtests.jl
docs/ARCHITECTURE.md
```

**Tests (no device required)**

- `CUDA` is in `[weakdeps]`, not `[deps]`
- core `names(Gesso)` does not include `CUDABackend` until the ext loads
- after `using CUDA` (test env), `CUDABackend <: AbstractGessoBackend`
- `backend_name == :cuda`, `execution_tier == 1` (OPTIMIZED_GENERIC)
- `supports` true for the Phase 2 op caps except quantize/dequantize
- `CUDABackend()` when `!CUDA.functional()` throws `GessoError`
- CPU-only `make test` still 736+skip (plus these seam tests)

**Artifact.** The seam exists.

---

### B — CUDA operator methods + `to_device`

**Objective.** The eight (minus quantize/dequantize) ops run on
`CuArray{Float32}` and match CPU F64 on tiny arrays within `atol=1e-3`.

**Permitted files**

```
ext/GessoCUDAExt.jl
ext/*.jl                 # split kernels if the file grows
src/Inference/Inference.jl   # to_device may live here or in the ext;
                             # if in the ext, export it from there
test/test_cuda_ops.jl
test/runtests.jl
```

Reuse the Phase 2 tiny-array fixtures. Copy to device, run, copy back,
`approx_eq` vs CPU result converted to F32 or vs CPU F64 with atol
1e-3. Skip the whole file when `!CUDA.functional()`.

Causal softmax and RoPE need a device test, not just matmul.

**Artifact.** GPU math exists. Interpreter still CPU-default.

---

### C — CUDA prefill + generate on toy2 / micro-llama

**Objective.** Full generation on NVIDIA for the models we already
own. Reference correctness vs CPU.

**Permitted files**

```
src/Inference/Inference.jl   # backend= keyword, no silent copy
ext/GessoCUDAExt.jl
test/test_cuda_inference.jl
test/runtests.jl
docs/ARCHITECTURE.md
README.md
scripts/freeze.jl            # new test files
```

**Tests (skip without device)**

- toy2 CUDA prefill vs CPU: `approx_eq` logits `atol=1e-3` (CPU F64 vs
  GPU F32 host copy). Fingerprint file on disk stays the CPU one;
  do not rewrite `expected_logits.toml`.
- toy2 CUDA greedy generate, `max_new_tokens=8`: **token ids match
  CPU** (argmax is the correctness gate; logits may differ at 1e-3).
  If argmax diverges, that is a real bug — fix math, don't relax
  tokens.
- llama_micro CUDA prefill `[0,1,2]` vs CPU, same atol
- `backend=CUDABackend()` + CPU `Array` storage throws
- `backend=CPUBackend()` still `atol=0` in-process (regression)

**Artifact.** §LXXVII "full generation on NVIDIA GPU" + "reference
correctness" met for toy2/micro. SmolLM2 not required.

---

### D — Optional SmolLM2 CUDA + bench row

**Objective.** If both a device **and** `GESSO_SMOLLM2_DIR` exist,
prefill `"Hello"` on CUDA and compare last-position logits to the
CPU golden with `atol=1e-2` (F32 vs F64, 135M). If either is missing,
named skip.

When a device exists, one micro-llama CUDA bench row as specified
above.

**Permitted files**

```
test/test_cuda_smollm2.jl
benchmark/runbenchmarks.jl   # only the new CUDA probe, gated
README.md
docs/Gesso_Stack.md          # §LXXVII status COMPLETE + date
docs/ARCHITECTURE.md
```

Do not fake a golden CUDA logits file. CPU golden already exists as
skip-or-green; CUDA compares to that, with wider atol.

**Artifact.** Phase 4 exit complete on a CUDA box; CI complete
without one.

---

## Cross-item invariants

- toy2 CPU fingerprint bit-identical
- JSON remains the only **core** third-party dep; CUDA is weakdep
- `libs/` untouched
- `quantize!` / `dequantize!` decline on CUDA too
- Operators module still exports nothing
- formatter skips `libs/`
- Receipt: what / why / tests / numerical delta (CPU fingerprint
  unchanged; CUDA vs CPU max-abs on toy2 recorded) / compile-time
  (CUDA.jl precompile is expected and noted) / hardware (GPU name
  if present) / workload / model toy2+llama_micro / backend cuda

## Escalation (packet, then stop)

- CUDA.jl cannot be a weakdep/extension on 1.12 without putting it
  in core `[deps]`
- F32 GPU vs F64 CPU argmax diverges on toy2 after the ops are
  faithful
- you need a new operator or a fused kernel to match atol
- `to_device` wants to become implicit inside the interpreter

## Exit checklist

- [ ] A, B, C landed
- [ ] D skip-or-green
- [ ] `make test` green on CPU-only
- [ ] `make format` run
- [ ] CUDA in `[weakdeps]` + `[extensions]`, not `[deps]`
- [ ] no silent CPU fallback
- [ ] toy2 CPU fingerprint unchanged
- [ ] README says how CUDA tests skip
- [ ] ARCHITECTURE / §LXXVII status truthful
- [ ] receipt
- [ ] no Lava, no Magenta, no Hub
