# /goal PHASE 8 — LAVA / VULKAN BACKEND

**For:** Buffy (mechanical implementation)
**From:** Grok (encoding owner)
**Canon:** `docs/Gesso_Stack.md` §LXXXI, §XXIII, §VII, §XX–§XXI, §LXX,
§CIX, §LXXVII (same no-copy / skip laws as CUDA)
**Map:** `docs/ARCHITECTURE.md`
**Depends on:** Phase 7 complete (`docs/goals/PHASE7_PREFIX_SHARE.md`)
**Packets:** 1 and 2 stay closed.

**Status:** LANDED (Buffy, 2026-09-30). Items A–D committed
(`e28be15` GessoLavaExt seam, `fcc8cd3` six ops + `to_device`,
`c9cebbc` interpreter/Session/`fork` + bench row, `5e26ffc` maps
+ §LXXXI COMPLETE). 1357 pass + 3 named SmolLM2 skips (Broken =
skip records, not `@test_broken`); both runners EXIT=0;
format-check OK. toy2 lava-vs-cpu max|Δlogit| = 4.00e-4 (< atol
1e-3); llama_micro 6.61e-6; generate ids exact on both models;
fork aliasing + partial-page CoW byte-identical on device. Bench:
13 rows, `micro_llama_lava_prefill_012` in
`benchmark/results/2026-10-01.tsv` (schema 0.2.0, post-warmup, no
speed claim). `src/` untouched; `libs/` untouched. Portable seam,
not tuned. Next: Phase 9 autotune (`docs/Gesso_Stack.md` §LXXXII).
5b remains later. Planning.jl stays empty. Do **not** edit
`libs/Lava`.

---

## Start condition

Phase 7 is on the tree you inherit:

- `fork` + CoW + `unique_kv_bytes` (2048 vs 4096 on llama_micro)
- `Session` / `prefill!` / `decode!` / `generate` on CPU and CUDA
- engine ids equal `reference_*` on toy2 and llama_micro
- `make test` green (named CUDA/SmolLM2 skips are fine)
- toy2 CPU fingerprint unchanged (`max|logit| = 1686.4814860783201`)
- JSON is the only core third-party hard dep; CUDA is weakdep
- `ext/GessoCUDAExt.jl` is the seam template
- `src/backends.jl` still has no `struct LavaBackend`
- `libs/` is gitignored local checkout — **do not touch it**

If Phase 7 is unfinished, stop.

## One-sentence objective

`toy2` and llama_micro generate on Vulkan through `Gesso.LavaBackend`,
matching the CPU oracle within declared tolerance. No Vulkan device
⇒ named skip. CI never requires a device. Lava stays an extension.

## Why this phase exists

§LXXXI: integrate Lava through the backend contract. Start portable
(GPUArrays / broadcast / `mul!`, the same shape as Phase 4 CUDA).
Then tune (Phase 9). CUDA proved the seam on NVIDIA. Lava is the
portable machine path (§XXIII): NVIDIA, AMD, Intel, MoltenVK,
lavapipe. Gesso owns ML semantics. Lava owns device, memory, kernels,
sync.

## What this sprint is not

- editing `libs/Lava` (local dev checkout; never part of the package)
- Lava as a **core** `[deps]` entry (law violation §VII)
- silent fallback to CPU or CUDA when Lava is requested (§LXX)
- handwritten SPIR-V, coopmat GEMM, graphics, ray tracing
- KernelAbstractions / GPUArrays / GLFW as core deps
- Autotune, result cache, occupancy (Phase 9)
- FlashAttention, fused kernels, cages, speculation
- scheduler / 5b
- filling `Planning.jl` / `Runtime.jl`
- claiming speedups vs CUDA or llama.cpp
- changing `reference_*` signatures or the toy2 fingerprint
- registering Lava in the General registry

---

## Laws (pin this)

### Extension, not identity

```
Project.toml
    [weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Lava = "3a680b1f-cb25-4bee-9cf7-bc880b76dc8c"

    [extensions]
    GessoCUDAExt = "CUDA"
    GessoLavaExt = "Lava"
```

`ext/GessoLavaExt.jl` is the only file allowed to `using Lava`.
Core Gesso (`src/`) must compile and test with Lava **not** loaded.

`Gesso.LavaBackend <: AbstractGessoBackend` lives in the extension,
bound in `__init__` the same way `CUDABackend` is (`Core.eval` at
load time, not at precompile). Do not add `struct LavaBackend` to
`src/backends.jl`.

Edit the dependency-law test: weakdeps may now contain CUDA **and**
Lava, each with a justification. That friction is the point.

### Name clash (load-bearing)

Lava.jl already exports `LavaBackend <: KernelAbstractions.GPU`.
Gesso's tag is **also** named `LavaBackend`, in the `Gesso` module.

```
Gesso.LavaBackend     # intent tag, AbstractGessoBackend
Lava.LavaBackend      # KA GPU backend, used for synchronize / launch
Lava.LavaArray        # storage
```

Inside the extension, alias the KA type (`const KALava = Lava.LavaBackend`).
Never `export` Lava's KA type from Gesso. Tests and Session code use
`Gesso.LavaBackend()`. Canon `using Gesso, Lava` stays valid:
qualify (`Gesso.LavaBackend` vs `Lava.LavaBackend`).

### Lava is not in the General registry

Install in the **test** env (not core):

```
test/Project.toml
    Lava = "3a680b1f-cb25-4bee-9cf7-bc880b76dc8c"

    [sources]
    Lava = {url = "https://github.com/SimonDanisch/Lava.jl"}
```

That is the test env, same role as CUDA.jl there. Do **not**
`path = "../libs/Lava"` in committed Project.toml (libs is
gitignored; CI would fail resolve). Locally you may `Pkg.develop`
the checkout for Lava work — that is operator choice, not this
recipe.

If resolve of SimonDanisch/Lava.jl is blocked (missing UUID, Raycore
unsatisfiable, Julia 1.12 pin), **packet and stop**. Do not vendor
Lava into `src/`.

### No silent fallback

`Gesso.LavaBackend()` with no usable Vulkan device throws typed
`GessoError(ERR_RESOURCE_LIMIT)`. It does **not** return
`CPUBackend()` or `CUDABackend()`.

Lava has no `functional()` twin of CUDA. Probe in the constructor:

```
try
    Lava.vk_context()
catch
    throw ERR_RESOURCE_LIMIT   # "no usable Vulkan device"
end
```

Do not swallow other errors into a CPU path.

### Same operators, portable math

Add more-specific methods on the existing `op!` names, dispatching on
`Gesso.LavaBackend`. No second vocabulary. `quantize!` /
`dequantize!` still decline.

Storage: **`Lava.LavaArray{Float32}`**. CPU oracle stays F64. Compare
with declared atol, not bit-identity.

```
atol = 1e-3    # Lava F32 vs CPU F64, micro-llama / toy2, seq ≤ 8
```

If a specific op exceeds that, fix the op. Do not silently widen past
`1e-2` without a packet.

Implement with GPUArrays broadcasting + Lava's `mul!` (already in
Lava `array/gemm.jl`). One KernelAbstractions `@kernel` per op is
allowed if broadcast cannot express it (RoPE, causal softmax) —
launch on `Lava.LavaBackend()`, not on Gesso's tag. No handwritten
`.comp` / SPIR-V, no coopmat, no graphics, no ray tracing.

### Transfer is explicit

```
to_device(::Gesso.LavaBackend, tensors) -> tensors'
```

Copies `Array` storage to `LavaArray{Float32}` (F64→F32 is part of
the lowering). Returns a new named-tuple / struct of the same shape.
Does not mutate the CPU tensors.

`reference_prefill` / `reference_generate` already take `backend=`.
Default stays CPU. Lava path: caller passes
`backend=Gesso.LavaBackend()` **and** tensors already on device. If
backend is Lava and storage is still `Array`, `ERR_INVALID_PLAN` —
the interpreter does not copy. Same law as CUDA.

Synchronize before reading logits back to host:
`KernelAbstractions.synchronize(Lava.LavaBackend())` (the KA one).

### Skip law

```
if Lava is not in the test env or Gesso.LavaBackend is undefined
    named skip "Lava extension not loaded"
if Lava is loaded but vk_context() fails
    named skip "no Vulkan device — Lava tests skipped"
```

`make test` on a box without Vulkan stays green. Never skip the
**extension loads** tests that can run without a device (module
exists when Lava is in the test env; `Gesso.LavaBackend` is defined;
constructor throws when there is no device).

| Test | Needs device? |
|---|---|
| `GessoLavaExt` loads when Lava is in the env | no |
| `LavaBackend` is not in core `names(Gesso)` without the ext | no |
| requesting Lava with no Vulkan throws, does not CPU-fallback | no (constructor) |
| toy2 / llama_micro Lava prefill vs CPU | yes |
| Lava generate greedy vs CPU argmax | yes |
| Lava `Session` + `fork` ids | yes |
| SmolLM2 Lava | yes **and** `GESSO_SMOLLM2_DIR` — skip-or-green |

### Pages / Session

The engine is already backend-generic. Pages allocate via `similar`
off the tensor prototype. If prototype is `LavaArray{Float32}`,
pages are LavaArray. `fork` CoW uses `similar` + row copy — must
work on LavaArray the same way it does on CuArray. Do not special-case
CoW for Vulkan.

### Benchmarks

When a Vulkan device exists, append **one** corpus row: llama_micro
Lava prefill or 3-token generate, schema `0.2.0`, after warmup, no
compile in the timed region. No "faster than CUDA" claim.

No device: no new bench row required.

---

## Work items (sequence)

### A — Extension skeleton + seam tests

**Objective.** Lava is a weakdep. `GessoLavaExt` loads. Core still
has no Lava. Requesting a device that is not there fails explicitly.

**Permitted files**

```
Project.toml
Manifest.toml                 # if the resolver writes it
ext/GessoLavaExt.jl
test/Project.toml
test/runtests.jl              # weakdeps allowlist + include
test/test_lava_seam.jl
docs/ARCHITECTURE.md          # only the Lowering / ext row if needed
```

**Tests**

- `GessoLavaExt` is listed in `[extensions]`
- `Lava` is in `[weakdeps]`, not `[deps]`
- dependency-law test updated with the Lava justification
- without the ext loaded: `!isdefined(Gesso, :LavaBackend)`
- with Lava in the test env: `isdefined(Gesso, :LavaBackend)`
- `Gesso.LavaBackend()` with failed `vk_context()` throws
  `GessoError` with `ERR_RESOURCE_LIMIT`, not a CPU backend
- `libs/` git status clean of this sprint

**Artifact.** The seam exists. No ops yet.

---

### B — Operators on `LavaArray{Float32}`

**Objective.** The six implemented ops run on Lava storage. Same
names as CPU/CUDA. `to_device` copies host → `LavaArray{Float32}`.

**Permitted files**

```
ext/GessoLavaExt.jl
ext/lava_ops.jl
test/test_lava_ops.jl
test/runtests.jl
```

**Tests**

- CPU-only / no Vulkan: named skip
- with device: each of rmsnorm / rope / softmax / swiglu / matmul /
  embedding_lookup on a tiny array matches CPU F64 within `atol=1e-3`
- `quantize!` / `dequantize!` still `LoweringNotImplemented`
- `to_device` does not mutate the host array

**Artifact.** Portable lowering. Not a kernel contest.

---

### C — Interpreter + Session on Lava

**Objective.** `reference_generate` and `Session.generate` on Lava
match CPU token ids (greedy). Prefill logits within atol. `fork`
after Lava prefill still aliases prefix pages.

**Permitted files**

```
src/Inference/interpreter.jl   # only if a backend-generic hole appears
src/Inference/session.jl       # only if similar()/sync needs a hook
test/test_lava_inference.jl
test/test_session_lava.jl
test/runtests.jl
benchmark/runbenchmarks.jl
benchmark/results/             # append-only TSV
```

Prefer **not** to touch interpreter/session. Packet if you must.

**Tests**

- toy2 Lava generate ids == CPU generate ids
- llama_micro Lava generate ids == CPU
- `Session` on Lava: ids == independent CPU Session; `fork` prefix
  pages `===`; CoW of the dirty page leaves parent storage unmoved
- host `Array` + `backend=Gesso.LavaBackend()` → `ERR_INVALID_PLAN`
- SmolLM2: skip-or-green (`GESSO_SMOLLM2_DIR`)
- CUDA tests unchanged

**Artifact.** Same logical model, same Gesso API, Lava execution
correct on supported Vulkan hardware (§LXXXI exit).

---

### D — maps + §LXXXI status

**Permitted files**

```
README.md
docs/ARCHITECTURE.md
docs/Gesso_Stack.md          # §LXXXI status COMPLETE + date
docs/research/ROADMAP_NOW.md # 8 done; 9 is next open recipe
docs/research/README.md
scripts/freeze.jl
```

**Artifact.** Docs say Lava is an extension that runs the engine.
They do not say we tuned it. They do not say `libs/Lava` is a dep.

---

## Cross-item invariants

- toy2 CPU fingerprint bit-identical
- `reference_*` public signatures unchanged
- CPU and CUDA paths unchanged
- JSON remains the only core third-party hard dep
- `libs/` untouched (no diffs under `libs/`)
- no new §CIX family types
- `Runtime` / `Planning` still empty
- Profiling still does not import CUDA or Lava
- formatter skips `libs/`
- no ExecutionResult, no schema bump, no speed claims
- Receipt: what / why / tests / numerical delta (fingerprint
  unchanged; Lava vs CPU id equality + logit atol) / compile-time /
  memory / hardware (Vulkan device name) / workload toy2+micro /
  backend lava (+ cpu, cuda if present)

## Escalation (packet, then stop)

- SimonDanisch/Lava.jl will not resolve in the test env
- you need to edit `libs/Lava`
- `Gesso.LavaBackend` vs `Lava.LavaBackend` cannot be kept distinct
- you need a handwritten SPIR-V kernel to hit atol
- LavaArray `similar` cannot allocate KV pages
- you want to add KernelAbstractions to core `[deps]`
- you want this sprint to autotune (that is Phase 9)

## Exit checklist

- [x] A, B, C landed
- [x] D maps truthful
- [x] `make test` green without a Vulkan device (named skips; this box also ran *with* a device: 1357 pass / 3 Broken)
- [x] `make format` run
- [x] CPU fingerprint unchanged (CUDA-vs-CPU print unchanged: toy2 0.000376… / llama_micro 5.50e-6)
- [x] CUDA tests still skip-or-green
- [x] Lava generate ids match CPU on toy2 + llama_micro when a device exists
- [x] no silent fallback (`ERR_RESOURCE_LIMIT` on no device)
- [x] `libs/` clean
- [x] no Planning.jl fill, no autotune, no cages
- [x] receipt (below)

---

## Foreman close (encoding owner, 2026-09-30)

Buffy implemented A–D on `master` at `5e26ffc`. Item D already flipped
canon §LXXXI to COMPLETE and the living maps. This file was the open
recipe (untracked, checkboxes empty); this close is the encoding-owner
receipt. Tests were not re-run for this close — gate counts and
numerical deltas are Buffy's commit-body receipts plus the TSV row
on disk.

### §LXXII receipt

| field | value |
|---|---|
| what changed | Lava as weakdep + `ext/GessoLavaExt.jl` + `ext/lava_ops.jl`; six ops on `LavaArray{Float32}`; `to_device`; interpreter/`Session`/`fork` on Vulkan; one corpus row |
| why | §LXXXI start-portable; CUDA already proved the seam |
| tests | 1357 pass / 3 named SmolLM2 Broken; both runners EXIT=0; format-check OK |
| numerical delta | toy2 lava-vs-cpu max\|Δlogit\| = 3.9995e-4; llama_micro 6.614e-6; ids exact; CoW byte-identical on device |
| before bench | 12 corpus rows (Phase 7) |
| after bench | 13 rows; `micro_llama_lava_prefill_012` median 1.376e7 ns, 319 samples, 25464 allocs, commit `fcc8cd3`, `dirty=true` |
| compile-time impact | Lava loads from source on 1.12 (~1 min warning); no core precompile of Lava |
| memory impact | F32 device storage vs F64 CPU oracle; CoW share works on LavaArray with zero special-casing |
| hardware | RTX 5060, Vulkan API 1.4.351, driver 615.71.09, Julia 1.12.6 |
| workload | toy2 + llama_micro (seq ≤ 8); SmolLM2 still skip-or-green |
| model | toy2, llama_micro |
| backend | lava (+ cpu, cuda 6.3.1 still present) |

### Upstream workarounds (documented in `test/Project.toml`, not packets)

1. **Vulkan.jl skew.** Registered 0.6.30 lacks
   `PhysicalDeviceCooperativeMatrixFeaturesKHR`. Lava master calls it on
   any cooperative-matrix device (this RTX 5060 is one). Test/bench envs
   pin Vulkan.jl 0.7.0 + VulkanCore 1.4.0 from unreleased `sd/vk1.4`
   (`03b4ca2` / `1d02829`). Revisit when those ship to the registry.
2. **Lava does not precompile under Julia 1.12** (KA method overwrite) —
   loads from source with a warning. Lava's SPIR-V jll range forces
   GPUCompiler ≤ 2.6.0; the CUDA 6.4 stack wants 2.9. CUDA suite stays
   on 6.3.1 so both extensions load in one env.

### Residual (not this sprint)

`Gesso.to_device` now carries **both** backends' methods when Lava loads
after CUDA (attach-by-name in `GessoLavaExt.__init__`).
`GessoCUDAExt.__init__` still rebinds unconditionally, so **CUDA after
Lava** clobbers Lava's methods. Permitted files for B/C did not include
the CUDA ext. Follow-up: one-line CUDA-ext wart, or a Phase 9/hardening
item.

Device-less skip law is now **evidenced**, not only named: gauntlet at
HEAD `5e26ffc` ran `VK_ICD_FILENAMES=nonexistent` → 170 pass / 3 Broken
(the three named device-file skips); 17/17 device-free seam assertions,
including the constructor `ERR_RESOURCE_LIMIT` branch. Full device gates
also re-ran: 1357 / 3, both runners, format-check, fingerprint gate.

No escalation packets. Tune is Phase 9 (`docs/goals/PHASE9_AUTOTUNE.md`).
