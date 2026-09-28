# Native Julia Kernel Autotuning — North Star Specification
### Private R&D README / Working Design
**Status:** pre-implementation architecture  
**Date:** 2026-09-22  
**Working context:** experimental private fork / companion work around `KernelTuner.jl`, with possible upstream contributions later  
**Primary language:** Julia  
**Initial execution substrate:** `KernelAbstractions.jl`  
**Initial hardware target:** CPU + NVIDIA through `CUDA.jl`  
**Long-term targets:** AMDGPU, Metal, oneAPI, Lava/Vulkan, and any future `KernelAbstractions` / `KernelInterface` backend that can satisfy the execution contract

---

## 0. Executive Summary

This project investigates a **native-Julia, backend-agnostic kernel autotuning substrate**.

The immediate opportunity is unusually crisp:

- Julia already has strong heterogeneous-kernel infrastructure through `KernelAbstractions.jl`.
- Julia already has mature vendor-facing GPU integrations such as `CUDA.jl`, with AMDGPU, Metal, oneAPI, and other backends available through the broader JuliaGPU ecosystem.
- `AcceleratedKernels.jl` explicitly identifies algorithm-agnostic tuning of parameters such as block size as useful future work.
- `KernelTuner.jl` already brings a serious autotuning API into Julia, but its current implementation wraps the Python Kernel Tuner and therefore depends on `PythonCall` / `CondaPkg` and the Python implementation beneath it.
- `KernelForge.jl` already demonstrates that high-performance portable GPU primitives can be written in Julia and already contains tuning-oriented infrastructure/data. Therefore **this project must not duplicate KernelForge's primitive library**.

The gap worth attacking is not "invent GPU programming for Julia."

The gap is:

> **Can Julia own the autotuning control plane itself?**

That means native Julia representations for:

- tunable parameter spaces,
- constraints,
- candidate generation,
- deterministic search,
- compilation/execution orchestration,
- correctness verification,
- benchmarking,
- hardware/workload fingerprinting,
- result caching,
- performance-regression tracking,
- tuning receipts and provenance,
- and eventually higher-level kernel-family generation.

The intended outcome is a reusable tuner that can sit beneath multiple Julia GPU projects rather than becoming another isolated collection of hand-tuned kernels.

The initial project should be developed as a **non-destructive native path beside the existing Python-backed path**, not as an immediate rewrite.

---

# 1. North Star

The North Star is a Julia package/runtime in which a developer can define a parameterized kernel family once and ask:

> "For this operation, input shape, data type, backend, and device, find me the fastest configuration that is correct, record exactly how you found it, and reuse that result safely later."

Conceptually:

```text
Julia kernel family
       │
       ▼
  Tuning Plan
       │
       ├── parameter space
       ├── constraints
       ├── workload signature
       ├── correctness oracle
       ├── search strategy
       └── resource budget
       │
       ▼
 Candidate Generator
       │
       ▼
 Compile / Launch Adapter
       │
       ▼
 Correctness Gate
       │
       ▼
 Benchmark Engine
       │
       ▼
 Search / Selection
       │
       ▼
 Best Configuration
       │
       ├── cache entry
       ├── hardware fingerprint
       ├── software fingerprint
       └── full tuning receipt
```

The user-facing ideal is boring:

```julia
result = tune(plan)
best = result.best
```

The machinery underneath may be sophisticated. The public contract should not be.

---

# 2. What This Project Is Not

This section is as important as the feature list.

This project is **not**:

1. A replacement for `KernelAbstractions.jl`.
2. A replacement for `GPUCompiler.jl`.
3. A new GPU compiler.
4. A new CUDA implementation.
5. A replacement for vendor libraries such as cuBLAS, rocBLAS, CUB, or rocPRIM.
6. A duplicate of `AcceleratedKernels.jl`.
7. A duplicate of `KernelForge.jl`.
8. A generic natural-language-to-GPU-kernel generator.
9. A new ML framework.
10. A reason to rewrite working Python code before a native path has demonstrated parity.
11. A Mojo bridge.
12. A Project Neura subsystem during the early implementation stages.

The project earns each additional layer only after the layer beneath it works.

---

# 3. Why This Project Exists

## 3.1 The problem

Portable kernel code is only half of performance portability.

A single kernel implementation may need different values for:

- workgroup/block size,
- elements per thread,
- tile dimensions,
- vector width,
- unroll factor,
- shared/local-memory tile size,
- staging depth,
- algorithm-switch threshold,
- subgroup strategy,
- reduction topology,
- fusion strategy,
- and backend-specific flags.

The correct answer depends on:

- device architecture,
- backend,
- compiler,
- driver,
- input shape,
- dtype,
- memory layout,
- kernel version,
- and sometimes neighboring operations.

Hardcoding one "good" configuration is therefore brittle.

The purpose of autotuning is to convert this from a manual folklore problem into a reproducible search problem.

## 3.2 Why Julia is well suited

Julia already specializes code aggressively by type and value. Tunable values can naturally become compile-time specializations using mechanisms such as `Val`, parametric types, generated call paths, or explicit configuration structs.

A tuner therefore does not need to invent a foreign source-template language on day one.

A parameterized Julia kernel can itself be the kernel family.

## 3.3 Why native Julia matters

The current Python-backed path is useful and should remain useful.

A native path adds different properties:

- fewer runtime boundaries,
- easier integration with Julia's type system,
- easier package composition,
- no mandatory Python environment for the tuning control plane,
- direct access to Julia-side package metadata and method specialization,
- simpler provenance capture,
- easier agent/tool integration,
- and an opportunity to make tuning a normal Julia programming primitive rather than an external service.

"Native" is not a purity contest. Vendor libraries and backend runtimes remain valid and desirable.

---

# 4. Ecosystem Positioning

The project should deliberately occupy a narrow layer.

```text
┌──────────────────────────────────────────────────────────────┐
│ Applications / ML runtimes / scientific codes / Project Neura│
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ AcceleratedKernels / KernelForge / custom KA kernel families │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│            THIS PROJECT: native autotuning control plane     │
│ search • verify • benchmark • fingerprint • cache • receipts │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│          KernelAbstractions / KernelInterface adapters       │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ CUDA.jl / AMDGPU.jl / Metal.jl / oneAPI / Lava / CPU / ...  │
└──────────────────────────────────────────────────────────────┘
```

### Key rule

**Do not own layers that already have competent owners.**

The project should orchestrate, measure, select, and remember.

It should not become a second GPU compiler stack.

---

# 5. Existing Projects We Must Respect

## 5.1 KernelTuner.jl

`KernelTuner.jl` already exposes serious autotuning functionality to Julia.

Current architecture includes a Python-backed core and dependencies such as `PythonCall` and `CondaPkg`.

This project should initially treat that implementation as:

- a compatibility baseline,
- a behavioral reference,
- a source of useful API concepts,
- and optionally an oracle for differential testing.

The first implementation strategy should therefore be:

> **Add a native execution/tuning path without deleting the existing path.**

Do not begin with a rewrite.

## 5.2 AcceleratedKernels.jl

`AcceleratedKernels.jl` provides portable parallel algorithms from a unified `KernelAbstractions` codebase.

Its public roadmap explicitly includes algorithm-agnostic automated tuning of parameters such as block size and algorithm thresholds, plus performance-regression infrastructure.

This makes it an excellent **consumer target** for a general tuner.

The correct long-term relationship is likely:

```text
AcceleratedKernels
        │
        └── asks tuner for configuration
                │
                └── native tuning substrate
```

not:

```text
new project reimplements AcceleratedKernels
```

## 5.3 KernelForge.jl

`KernelForge.jl` already exists as a pure-Julia high-performance GPU primitives package, with `KernelIntrinsics.jl` beneath it.

It already covers primitives such as reductions, scans, matrix-vector work, GEMM-related functionality, search, sorting, and vectorized memory operations, with CUDA and AMD support.

It also already has tuning-related data/infrastructure and its author has discussed autotuning as a long-term direction.

Therefore:

> **Do not build another KernelForge.**

Instead, a successful native tuner should eventually be something KernelForge *could use*.

That turns apparent overlap into an integration opportunity.

---

# 6. Core Design Principles

## 6.1 Correctness before speed

A candidate that is fast and wrong is rejected.

Every tuning session must support a correctness gate before a timing result can become eligible for selection.

## 6.2 Determinism where practical

Given:

- the same kernel revision,
- same parameter space,
- same constraints,
- same search strategy,
- same seed,
- same hardware/software fingerprint,
- and same workload,

the tuner should produce reproducible candidate ordering and an auditable result.

Performance measurements themselves are noisy; the **decision process** should still be reconstructible.

## 6.3 Explicit budgets

Every tuning run should have bounded resources:

- maximum candidates,
- maximum compilation count,
- maximum elapsed tuning time,
- optional maximum device-memory allocation,
- warmup count,
- measurement count,
- failure limit.

An agent must never be able to accidentally launch an unbounded combinatorial search simply because a parameter grid grew.

## 6.4 Backend neutrality at the control plane

The tuner core should not care whether a candidate eventually runs through CUDA, ROCm, Metal, oneAPI, Vulkan, or CPU.

Backend-specific details belong in adapters.

## 6.5 Backend specificity where performance requires it

Portable does not mean pretending hardware is identical.

A backend adapter may expose:

- maximum workgroup size,
- subgroup/warp size,
- local-memory limits,
- architecture identifier,
- timer capabilities,
- and other constraints.

Search spaces may use those facts.

## 6.6 Provenance is a first-class output

The result is not just:

```text
block_size = 256
```

It is:

```text
block_size = 256
because:
  - 27 candidates were legal
  - 24 compiled
  - 23 passed correctness
  - 23 were benchmarked
  - this candidate had the best score
  - on this hardware
  - with this software stack
  - for this workload signature
  - under this measurement protocol
```

## 6.7 No hidden magic

The tuner may automate search.

It must not make invisible semantic decisions.

All transformations, exclusions, failures, and selections should be inspectable.

---

# 7. Fundamental Data Model

The following names are conceptual. Exact names may change.

## 7.1 `KernelFamily`

Represents a parameterized operation.

Responsibilities:

- identify the kernel family,
- define how a candidate configuration specializes the kernel,
- expose parameter schema,
- expose optional static constraints,
- expose version/hash identity.

Example:

```julia
struct RMSNormFamily <: AbstractKernelFamily
    eps::Float32
end
```

A family is not necessarily one function. It is the stable identity of an operation plus its specialization contract.

## 7.2 `TuneSpace`

Defines legal candidate dimensions before constraints.

Example:

```julia
space = TuneSpace(
    block_size = [64, 128, 256, 512],
    work_per_thread = [1, 2, 4, 8],
    vector_width = [1, 2, 4],
)
```

Initial implementation should strongly prefer ordinary Julia data structures over a macro DSL.

Macros can come later if repeated boilerplate justifies them.

## 7.3 `TuneConstraint`

A predicate over:

- candidate configuration,
- workload,
- and optionally backend/device capabilities.

Examples:

```text
block_size <= device.max_workgroup_size
block_size * local_bytes_per_thread <= device.local_memory_limit
vector_width divides contiguous dimension
tile_m * tile_n remains within resource budget
```

A candidate rejected by constraints should be recorded as **skipped**, not **failed**.

## 7.4 `WorkloadSignature`

Captures performance-relevant workload identity.

Potential fields:

- operation identifier,
- dimensions,
- dtype(s),
- strides/layout,
- contiguity,
- batch size,
- sequence length,
- alignment class,
- flags that change generated code.

The workload signature is part of the cache key.

## 7.5 `DeviceFingerprint`

Captures relevant device identity.

Potential fields:

- backend,
- vendor,
- device model,
- device architecture,
- compute capability / equivalent,
- subgroup size,
- memory size if relevant,
- driver/runtime version,
- optional PCI/device identifier class.

Do not key only on marketing name.

## 7.6 `SoftwareFingerprint`

Potential fields:

- Julia version,
- package version,
- kernel-family source/hash,
- `Manifest.toml` hash or selected dependency versions,
- backend package version,
- compiler/runtime version,
- project commit.

## 7.7 `Candidate`

A concrete assignment of tunable parameters.

```julia
Candidate(
    block_size = 256,
    work_per_thread = 4,
    vector_width = 2,
)
```

Candidates should be immutable and hashable.

## 7.8 `Evaluation`

The complete outcome of one candidate.

Possible states:

```text
SKIPPED_CONSTRAINT
COMPILE_FAILED
LAUNCH_FAILED
VERIFY_FAILED
BENCHMARK_FAILED
VALID
```

A valid evaluation stores timing samples and derived score.

## 7.9 `TuneResult`

Contains:

- selected candidate,
- all evaluations or a reference to persisted evaluations,
- score,
- fingerprints,
- tuning receipt,
- confidence/measurement metadata,
- cache status.

---

# 8. Native Tuning Pipeline

A tuning session should proceed through explicit stages.

## Stage 1 — Normalize the plan

Validate:

- parameter names,
- candidate values,
- workload definition,
- backend,
- budget,
- search strategy,
- correctness oracle,
- timing policy.

Reject malformed plans before compiling anything.

## Stage 2 — Fingerprint environment

Collect hardware and software identity before search begins.

This fingerprint becomes part of every evaluation and the eventual cache key.

## Stage 3 — Generate candidate

Search strategy emits the next candidate.

In v0:

- exhaustive grid search,
- deterministic ordering.

Then:

- seeded random search.

Later:

- successive halving,
- Bayesian optimization,
- evolutionary strategies,
- model-based search,
- transfer from previous tuning records.

## Stage 4 — Apply constraints

Reject impossible or nonsensical candidates before compilation.

Constraint rejection must be cheap.

## Stage 5 — Materialize specialization

Convert candidate values into a concrete Julia specialization.

Possible mechanisms:

- `Val`,
- parametric configuration types,
- closures,
- generated launch wrapper,
- backend workgroup parameters.

Do not begin with source-code rewriting unless a real use case demands it.

## Stage 6 — Compile

Compile lazily through the normal Julia/backend path.

Record:

- compile success/failure,
- compile time,
- diagnostic,
- artifact identity if available.

Compilation cost may itself become a secondary metric later.

## Stage 7 — Warm up

Run a controlled number of warmup iterations so compilation and one-time allocation effects do not pollute timing.

## Stage 8 — Verify

Compare against an oracle.

Possible oracle types:

- CPU reference implementation,
- known expected output,
- differential backend result,
- property-based invariant,
- tolerance-based numerical reference.

A verification failure permanently disqualifies that candidate for the current fingerprint/workload.

## Stage 9 — Benchmark

Collect multiple samples.

Minimum viable policy:

- synchronize before timing,
- launch,
- synchronize after timing,
- store raw samples,
- report median,
- optionally record min / mean / dispersion.

Later, backend-native event timers can improve accuracy.

## Stage 10 — Score

Default objective:

```text
minimize median execution time
```

Later objectives may include:

- throughput,
- latency,
- memory usage,
- energy,
- compile time,
- Pareto objectives,
- tail latency.

## Stage 11 — Persist evaluation

Every candidate outcome is written to the session record.

## Stage 12 — Select winner

The winner must be selected only from:

```text
compiled ∧ launched ∧ verified ∧ benchmarked
```

## Stage 13 — Emit tuning receipt

Store the full decision trail.

---

# 9. Measurement Discipline

Autotuning becomes garbage if benchmarking is sloppy.

The benchmark engine must explicitly address:

## Warmup

Compilation must not be mistaken for kernel execution time.

## Synchronization

GPU launches are asynchronous.

The timing adapter must enforce correct synchronization semantics.

## Multiple samples

One sample is not evidence.

Store raw measurements.

## Outliers

v0 can use median without aggressive filtering.

Later versions may use robust outlier detection, but raw data must remain available.

## Thermal / clock instability

Long tuning sessions can alter clocks.

The tuner should eventually support:

- randomized candidate measurement order,
- repeated champion confirmation,
- final top-K remeasurement.

## Allocation effects

Host-side allocation should be outside the timed region whenever possible.

## End-to-end vs kernel-only timing

The API must distinguish:

- pure kernel execution,
- launch overhead,
- data transfer,
- end-to-end operation latency.

Do not silently mix them.

---

# 10. Correctness Framework

Correctness is not optional.

A tuning plan should define one or more verification policies.

## 10.1 Exact verification

Useful for integer/bitwise operations.

## 10.2 Floating-point tolerance

Support:

- `atol`,
- `rtol`,
- optional NaN policy,
- optional Inf policy.

## 10.3 Property verification

Examples:

- reduction result is within reference tolerance,
- sorted output is monotonic and a permutation,
- normalization output has expected statistical property,
- no out-of-bounds sentinel corruption occurred.

## 10.4 Differential verification

Run:

- CPU reference,
- alternate backend,
- or known-safe implementation,

then compare.

This becomes particularly important for Project Neura, where generated or agent-modified kernel families may eventually be tested.

---

# 11. Search Strategy Architecture

Search should be an interface, not a hardcoded loop.

Conceptually:

```julia
abstract type AbstractSearchStrategy end

struct GridSearch <: AbstractSearchStrategy end
struct RandomSearch <: AbstractSearchStrategy
    seed::UInt64
end
```

A strategy should receive:

- parameter space,
- constraints or constraint callback,
- prior evaluations,
- budget state,

and return the next candidate.

## v0 strategies

1. Exhaustive grid.
2. Seeded random without replacement.

That is enough to validate the architecture.

## Later strategies

- successive halving,
- Hyperband-style resource allocation where meaningful,
- Bayesian optimization,
- evolutionary search,
- local search,
- Latin hypercube / quasi-random sampling,
- transfer learning from related workload/device records.

Do not implement advanced strategies before the evaluator and measurement engine are trustworthy.

---

# 12. Tuning Cache

The cache is one of the highest-value pieces of the project.

A user should not retune an identical workload every process start.

Conceptual cache key:

```text
hash(
  kernel_family_id,
  kernel_family_version,
  workload_signature,
  backend,
  device_fingerprint,
  software_fingerprint,
  tuning_schema_version
)
```

Cached record:

```text
best candidate
score
measurement protocol
raw winning samples
verification metadata
date
full receipt reference
```

## Cache policy

A cache hit is valid only if the compatibility policy says the fingerprints are compatible.

Examples:

- new kernel commit -> invalidate,
- new GPU architecture -> invalidate,
- changed dtype/shape class -> likely invalidate,
- changed driver -> configurable conservative invalidation,
- changed Julia/backend package version -> conservative invalidation initially.

The first implementation should prefer false misses over unsafe false hits.

---

# 13. Tuning Receipts

This is the point where the project naturally aligns with Project Neura's provenance philosophy.

A `TuningReceipt` should answer:

- What was tuned?
- Why was tuning invoked?
- What exact search space was considered?
- Which candidates were skipped?
- Which failed to compile?
- Which failed correctness?
- What were the raw benchmark samples?
- What selection rule was used?
- Which candidate won?
- On what hardware?
- Under what software versions?
- With what random seed?
- Under what resource budget?
- Was the result loaded from cache or newly measured?

Suggested serializations:

- Julia-native struct in memory,
- JSON or JSON3-compatible persistent form,
- optional human-readable Markdown summary.

No opaque pickle-equivalent should be required for core records.

---

# 14. First Public API Sketch

This is intentionally conservative.

```julia
using NativeKernelTuning
using KernelAbstractions
using CUDA

family = RMSNormFamily(eps = 1f-5)

space = TuneSpace(
    block_size = [64, 128, 256, 512],
    work_per_thread = [1, 2, 4],
)

plan = TunePlan(
    family = family,
    backend = CUDABackend(),
    workload = workload,
    space = space,
    strategy = GridSearch(),
    verification = DifferentialCheck(reference_rmsnorm;
        rtol = 1f-4,
        atol = 1f-5,
    ),
    budget = TuneBudget(
        max_candidates = 32,
        max_seconds = 60,
    ),
)

result = tune(plan)

result.best
result.receipt
```

Names are placeholders.

The important point is architectural separation.

---

# 15. Backend Adapter Contract

The native tuner core should depend on a small internal adapter contract.

Conceptually:

```julia
abstract type AbstractExecutionAdapter end
```

Required operations may include:

```text
capabilities(adapter)
prepare(adapter, family, candidate, workload)
compile(adapter, prepared)
launch(adapter, compiled, workload)
synchronize(adapter)
benchmark(adapter, compiled, workload, policy)
device_fingerprint(adapter)
cleanup(adapter, state)
```

The first adapter should be KernelAbstractions-oriented.

Backend-specific extensions can specialize timing and capability reporting.

## Initial backend order

1. CPU.
2. CUDA.
3. AMDGPU.
4. Metal / oneAPI as contributor hardware permits.
5. Lava/Vulkan.
6. Any future KernelInterface-compatible backend.

CPU-first development is valuable because Qwen Studio may not have useful GPU access.

CPU success does **not** prove GPU correctness.

It proves the control-plane architecture.

---

# 16. KernelAbstractions / KernelInterface Strategy

Do not over-bind the core package to one specific internal revision of `KernelAbstractions`.

The Julia GPU stack is evolving.

Therefore:

- isolate KA/KI-specific calls in an adapter module,
- write tests against the supported stable API,
- avoid reaching into undocumented internals unless absolutely necessary,
- feature-detect capabilities where sensible,
- make backend capability objects explicit.

The package should survive upstream API movement by changing one integration layer rather than its entire architecture.

---

# 17. Relationship to the Existing Python Kernel Tuner

The Python implementation is not an enemy.

It is an asset.

## Compatibility mode

During development, retain a path such as:

```text
engine = :python
```

and add:

```text
engine = :native
```

This enables:

- behavioral comparisons,
- benchmark comparisons,
- migration without flag day,
- fallback when native search lacks a feature.

## Differential tuner testing

For search spaces both systems understand:

1. feed equivalent candidate space,
2. use equivalent benchmark workload,
3. compare legal candidate handling,
4. compare best configuration,
5. investigate large discrepancies.

Exact candidate ordering need not match unless intentionally specified.

---

# 18. The First Vertical Slice

The first end-to-end success must be embarrassingly small.

## Slice A — vector addition

Parameterized knobs:

- block/workgroup size,
- work per thread.

Success means:

1. Native tuner enumerates candidates.
2. Constraints work.
3. Each candidate becomes a real Julia specialization.
4. CPU path runs.
5. Correctness is verified.
6. Timings are captured.
7. Winner is selected.
8. Receipt is persisted.
9. Re-running uses cache.

This proves architecture, not usefulness.

## Slice B — reduction

The next kernel should expose a tuning problem that actually matters.

Potential parameters:

- workgroup size,
- elements per thread,
- algorithm threshold.

This aligns conceptually with the kind of tuning `AcceleratedKernels.jl` has identified as valuable.

## Slice C — Neura-relevant RMSNorm

The first Project Neura-oriented experiment should be a small LLM-relevant primitive rather than a giant attention kernel.

Candidate:

```text
RMSNorm
```

Why:

- simple reference implementation,
- memory/performance sensitive,
- amenable to block-size and work-per-thread tuning,
- numerically testable,
- useful in real transformer inference,
- small enough to understand.

Possible later extension:

```text
residual add + RMSNorm fusion
```

Do not begin with custom GEMM.

Vendor libraries and existing Julia projects already own that battlefield far better than a v0 tuner does.

---

# 19. Qwen Coder's Role

Qwen is the **scaffolding and bounded implementation agent**, not the final authority on GPU correctness.

Its environment may be limited.

That is acceptable.

## Qwen is responsible for

### Repository archaeology

- map current `KernelTuner.jl` package structure,
- identify Python boundary,
- identify existing tests,
- identify public API surface,
- identify where a parallel native engine can be inserted with minimum disruption.

### Architecture scaffolding

Create native modules for:

- core types,
- tune spaces,
- candidates,
- constraints,
- budgets,
- search strategies,
- evaluation records,
- receipts,
- cache keys,
- serialization.

### CPU reference implementation

Implement a CPU-only evaluator sufficient to prove:

- candidate enumeration,
- specialization,
- correctness,
- timing,
- selection,
- persistence.

### Tests

Build strong tests for:

- deterministic candidate order,
- constraint rejection,
- budget enforcement,
- result serialization,
- cache invalidation,
- failure classification,
- reproducibility under fixed seed.

### Documentation

Maintain:

- architecture notes,
- decisions,
- known limitations,
- failure log,
- TODOs,
- parity table against Python-backed functionality.

## Qwen is explicitly not responsible for

without a later explicit instruction:

- deleting the Python path,
- rewriting the package API wholesale,
- designing unsafe GPU intrinsics,
- claiming CUDA correctness without CUDA execution,
- inventing benchmark numbers,
- implementing a compiler,
- integrating Project Neura,
- integrating Mojo,
- creating a massive generalized DSL,
- adding advanced Bayesian search before the basic evaluator is sound.

---

# 20. Handoff Roles for Other Models / Local Environment

## Codex or equivalent code agent

Best used when:

- real repository-wide refactors are needed,
- tests must be repaired across many files,
- CUDA path can be executed,
- CI behavior needs to be chased.

## Grok / hostile architecture reviewer

Best used for:

- red-team passes,
- identifying leaky abstractions,
- finding resource-exhaustion paths,
- attacking cache correctness,
- attacking benchmark validity,
- challenging over-generalization.

## Claude / specification and invariant review

Best used for:

- contract review,
- API coherence,
- invariants,
- failure taxonomy,
- documentation precision,
- adversarial edge cases.

## Human owner

Responsible for:

- project scope,
- accepting/rejecting architectural changes,
- deciding what gets upstreamed,
- validating that generated code matches intent,
- running hardware tests,
- preserving the failure record.

---

# 21. Suggested Repository Layout

If this begins as a fork of `KernelTuner.jl`, do **not** immediately rename/restructure everything.

Prefer adding a contained native subsystem.

Possible shape:

```text
src/
  KernelTuner.jl

  native/
    NativeEngine.jl
    types.jl
    spaces.jl
    constraints.jl
    budgets.jl
    evaluate.jl
    verify.jl
    benchmark.jl
    fingerprint.jl
    cache.jl
    receipts.jl

    search/
      grid.jl
      random.jl

    adapters/
      cpu.jl
      kernelabstractions.jl

ext/
  KernelTunerCUDAExt.jl
  KernelTunerAMDGPUExt.jl
  KernelTunerMetalExt.jl
  KernelTunerLavaExt.jl

examples/
  native/
    00_vector_add.jl
    01_reduction.jl
    02_rmsnorm.jl

test/
  native/
    test_spaces.jl
    test_constraints.jl
    test_search.jl
    test_receipts.jl
    test_cache.jl
    test_cpu_eval.jl
    test_failure_modes.jl

docs/
  NORTH_STAR.md
  native_architecture.md
  parity.md
  failures.md
```

If upstream maintainers prefer a separate package rather than a native engine inside `KernelTuner.jl`, the same internal architecture can be extracted later.

Do not prematurely optimize for that fork point.

---

# 22. Failure Taxonomy

Failures must be classified.

At minimum:

```text
INVALID_PLAN
CONSTRAINT_REJECTED
COMPILE_ERROR
RESOURCE_LIMIT
ALLOCATION_ERROR
LAUNCH_ERROR
RUNTIME_ERROR
TIMEOUT
VERIFY_MISMATCH
NUMERICAL_INSTABILITY
BENCHMARK_ERROR
CACHE_ERROR
INTERNAL_ERROR
```

Each failure record should include:

- candidate,
- stage,
- diagnostic,
- backend,
- timestamp,
- whether search may continue.

A bad candidate should usually not kill the entire session.

A corrupt environment may.

---

# 23. Safety and Agentic Use

This matters because the eventual consumer may be Project Neura.

Autotuning can become accidental code-execution roulette if an agent is allowed to mutate arbitrary kernel source and launch it without bounds.

Therefore distinguish:

## Safe tuning mode

Agent may choose among:

- registered kernel family,
- registered tunables,
- bounded legal values,
- fixed correctness oracle,
- fixed resource budget.

No arbitrary source generation.

## Experimental forge mode

Later, a higher-capability mode may permit generated kernel variants.

That requires stronger containment:

- subprocess isolation,
- wall-clock limits,
- memory budgets,
- compile limits,
- crash capture,
- artifact quarantine,
- mandatory reference comparison,
- no automatic promotion to trusted cache.

Do not build forge mode in v0.

---

# 24. Performance Regression System

Once stable tuning records exist, they become a regression corpus.

A regression runner can ask:

> Does the previously best known configuration still perform within tolerance?

Track:

- absolute latency,
- relative change,
- compiler/package changes,
- winner changes,
- newly invalid candidates,
- correctness regressions.

This is directly useful to:

- AcceleratedKernels,
- KernelForge,
- Project Neura,
- and the tuner itself.

Later CI may keep small CPU tests always-on and schedule hardware benchmarks separately.

---

# 25. Cross-Device Knowledge

Long-term, the tuning database can become more intelligent.

Never assume that a winning configuration transfers perfectly.

But records can seed search.

Example:

```text
RTX 4070 winner:
  block=256
  work=4

New Ada-class GPU:
  try nearby configurations first
```

This turns the tuning store from a cache into a prior.

Possible hierarchy:

```text
exact device + exact workload
same architecture + similar workload
same vendor + similar workload
global historical prior
cold search
```

This is a later optimization, not a v0 requirement.

---

# 26. Kernel-Family Generation: The Later "Foundry" Layer

Only after the tuner is trustworthy should the project explore a generator layer.

The important distinction:

## Autotuner

Selects among parameterized variants of a known kernel family.

## Generator / foundry

Produces structurally different kernel implementations.

Examples:

- different reduction tree,
- different tiling topology,
- different fusion boundary,
- different memory staging scheme,
- different vectorized load layout.

A future family generator could emit a set of legal implementations and hand them to the same tuner.

That is where Julia metaprogramming becomes extremely powerful.

But the foundry should consume the tuner.

The tuner must not depend on the foundry.

```text
Kernel family generator
        │
        ▼
candidate implementations
        │
        ▼
native tuner
        │
        ▼
verified best implementation
```

This separation keeps the foundational project useful even if automatic structural generation never matures.

---

# 27. Mojo: Explicitly Out of Scope, Architecturally Allowed

A future Mojo-related backend or tool may become interesting.

Do not design the native tuner around that possibility.

The correct abstraction is:

> if a future execution adapter can present a candidate as a compilable/runnable kernel and provide timing/capability information, the tuner can evaluate it.

That could someday include:

- Mojo,
- Vulkan/SPIR-V,
- a custom compiler,
- or another accelerator stack.

No Mojo SDK dependency belongs in v0.

---

# 28. Lava / Vulkan

Lava is potentially valuable later because it provides a Julia-to-SPIR-V/Vulkan route and a `KernelAbstractions`-compatible compute path.

This makes it useful for:

- portability testing,
- differential execution,
- non-CUDA hardware,
- graphics/compute crossover experiments.

However:

> Lava should be an adapter/integration target, not a dependency of the tuning core.

CUDA first is a practical hardware lab.

Lava later is a portability lab.

---

# 29. Project Neura Integration

Project Neura should consume this system only after the standalone tuner has earned trust.

The eventual relationship could look like:

```text
NeuraBash acceleration trapdoor
            │
            ▼
     capability policy
            │
            ▼
   registered operation
            │
            ▼
      tuning cache lookup
        │           │
      HIT          MISS
        │           │
        │           ▼
        │      bounded tuner
        │           │
        └─────┬─────┘
              ▼
      verified configuration
              │
              ▼
        execution receipt
```

## Neura use cases

- tune a registered RMSNorm kernel for the local machine,
- retune after GPU/driver/package change,
- compare CUDA vs Lava when both are available,
- maintain per-machine performance profiles,
- detect regressions,
- select safe acceleration paths without asking the model to reason about thread blocks directly.

## Agent interface

The model should ask:

```text
"optimize operation X under budget Y"
```

not:

```text
"write arbitrary GPU code and run it until something is fast"
```

That distinction is fundamental.

---

# 30. NeuraBash Surface — Future Sketch Only

Potential conceptual command:

```text
|?> tune rmsnorm --budget 30s
```

or an internal structured operation:

```text
accel.tune(
    operation = "rmsnorm",
    budget_seconds = 30,
    verification = "strict",
)
```

Return:

```text
status
best configuration
speedup over baseline
cache key
receipt handle
backend
hardware fingerprint
```

Again: not v0.

The standalone library must remain useful without Neura.

---

# 31. First Neura-Relevant Demonstrator: RMSNorm

Reference form:

```text
y = x / sqrt(mean(x²) + eps) * weight
```

Implementation goals:

- simple Julia CPU oracle,
- KernelAbstractions implementation,
- block/workgroup tunable,
- work-per-thread tunable,
- optional vector width,
- shape-aware workload signature.

Benchmark cases might include representative hidden sizes.

Do not optimize only one shape and then imply generality.

Measure several.

Success criterion:

> the tuner reliably finds a correct configuration and can reproduce/reuse the result for the same fingerprint.

A speedup is desirable.

Correct tuning behavior is the first objective.

---

# 32. Testing Strategy

## Unit tests

Test pure logic without GPU:

- space generation,
- constraints,
- seeded ordering,
- budgets,
- cache keys,
- serialization,
- failure classification,
- receipt completeness.

## Property tests

Examples:

- no duplicate candidate emitted by random-without-replacement search,
- constrained candidate is never executed,
- same seed yields same order,
- changing kernel version changes cache key,
- changing shape changes workload key,
- invalid candidate can never become winner.

## Integration tests

CPU:

- vector add,
- reduction.

GPU when available:

- same kernels,
- synchronization correctness,
- backend fingerprint.

## Differential tests

Native path vs Python-backed path for overlapping cases.

## Regression tests

Known bug gets a permanent test.

The failure log is part of the engineering process, not an embarrassment.

---

# 33. Benchmark Integrity Rules

1. Never publish timing from a run that included compilation unless explicitly labeled.
2. Always state device.
3. Always state backend and important software versions.
4. Always state input size and dtype.
5. Store raw timing samples.
6. Never compare kernel-only time on one side to end-to-end time on the other.
7. Never hide failed configurations from the tuning record.
8. Re-run the top candidates before declaring the winner in serious benchmarks.
9. Treat tiny timing differences as noise until shown otherwise.
10. Prefer reproducible scripts over screenshots.

---

# 34. Versioning Strategy

Internal schemas need versions.

Examples:

```text
TUNE_PLAN_SCHEMA_VERSION
RECEIPT_SCHEMA_VERSION
CACHE_KEY_SCHEMA_VERSION
DEVICE_FINGERPRINT_VERSION
```

If cache semantics change, bump the relevant version.

Do not silently reinterpret old records.

---

# 35. Licensing / Upstream Discipline

Because this begins around existing open-source projects:

- preserve upstream license files,
- preserve attribution,
- keep upstream history when working in a fork,
- isolate experimental changes cleanly,
- avoid making a giant unreviewable divergence,
- upstream small coherent pieces when appropriate.

If the native engine ultimately becomes a separate package, select a license compatible with intended upstream consumers.

Do not assume a private fork can later copy arbitrary code between differently licensed projects without checking compatibility.

---

# 36. Upstream Communication Strategy

Do not appear with a 20,000-line rewrite and ask maintainers to bless it.

Prefer:

1. understand current design,
2. produce a minimal native prototype,
3. collect measurements,
4. write down the exact seam,
5. ask maintainers whether they want:
   - native engine in `KernelTuner.jl`,
   - shared tuner package,
   - integration in `AcceleratedKernels`,
   - or a different boundary.

A useful opening technical question is approximately:

> "I'm experimenting with a pure-Julia tuning control plane for parameterized `KernelAbstractions` kernels. `KernelTuner.jl` currently provides the mature Python-backed path, while `AcceleratedKernels.jl` explicitly wants algorithm-agnostic tuning. Would you prefer native tuning to live as an engine inside KernelTuner.jl, as a small shared package consumed by both, or somewhere else?"

Do not frame this as replacing other people's work.

Frame it as identifying a reusable seam.

---

# 37. Milestones

## M0 — Reconnaissance

Deliverables:

- architecture map of current repo,
- Python boundary map,
- test map,
- API map,
- written decision on insertion point.

Exit criterion:

> We can explain exactly how a native engine can coexist with the existing engine.

## M1 — Native Core Types

Deliverables:

- `TuneSpace`,
- `Candidate`,
- `TuneConstraint`,
- `TuneBudget`,
- evaluation/result types,
- receipt schema.

Exit criterion:

> Pure unit tests pass with no backend required.

## M2 — Deterministic Search

Deliverables:

- grid search,
- seeded random search,
- budget enforcement.

Exit criterion:

> Search is reproducible and cannot exceed configured bounds.

## M3 — CPU Evaluator

Deliverables:

- specialization,
- launch,
- verification,
- benchmark,
- winner selection,
- cache.

Exit criterion:

> Vector-add vertical slice works end to end with no Python dependency in native mode.

## M4 — KernelAbstractions Adapter

Deliverables:

- KA execution adapter,
- backend capability surface,
- synchronization contract.

Exit criterion:

> Same tuning plan structure can run through KA CPU backend.

## M5 — CUDA

Deliverables:

- CUDA device fingerprint,
- correct GPU timing path,
- vector-add GPU test,
- reduction GPU test.

Exit criterion:

> Native tuning finds a verified winner on real NVIDIA hardware and persists a reusable result.

## M6 — Python Differential Parity

Deliverables:

- comparison harness for overlapping functionality,
- discrepancy report.

Exit criterion:

> Major semantic differences are understood and documented.

## M7 — Neura RMSNorm Demo

Deliverables:

- KA RMSNorm family,
- CPU oracle,
- CUDA tuning run,
- tuning receipt,
- cache reuse.

Exit criterion:

> A genuinely useful ML kernel is tuned by the same generic machinery.

## M8 — AcceleratedKernels Experiment

Deliverables:

- non-invasive proof of tuning one AK-exposed parameterized operation,
- no fork-wide redesign.

Exit criterion:

> Shared tuner concept proves useful outside the toy/demo kernel.

## M9 — Maintainer Conversation

Deliverables:

- small design note,
- benchmarks,
- API sketch,
- upstream question.

Exit criterion:

> We know whether to pursue an upstream engine, shared package, or independent package.

---

# 38. Estimated Effort

These are engineering-effort estimates, not promises.

Agent assistance can compress typing and scaffolding.

It cannot eliminate hardware debugging, benchmark validation, or architecture mistakes.

## Prototype

M0–M3:

**~20–45 supervised engineering hours**

Likely dominated by repo archaeology, API design, and tests rather than code volume.

## Credible CPU + CUDA v0.1

M0–M5:

**~45–100 supervised engineering hours**

Large uncertainty comes from backend timing, compilation behavior, and environment-specific issues.

## Upstream-worthy native engine

M0–M9:

**~120–300+ supervised engineering hours**

This includes:

- documentation,
- compatibility,
- edge cases,
- hardware testing,
- regression work,
- maintainability cleanup.

## Mature multi-backend system

Potentially:

**several hundred additional hours across multiple contributors**

This is why the architecture must allow every intermediate milestone to be independently useful.

---

# 39. Qwen Implementation Sequence

Qwen should receive work in small passes.

## Pass 1 — Read only

Instruction:

- inspect repository,
- produce architecture map,
- make no code changes.

## Pass 2 — Native type skeleton

Instruction:

- add minimal types under isolated native module,
- tests only,
- no Python path changes.

## Pass 3 — Search spaces and constraints

Instruction:

- deterministic grid,
- seeded random,
- budget state,
- tests.

## Pass 4 — Receipts and cache-key model

Instruction:

- serialization,
- fingerprints with mocked device data,
- tests.

## Pass 5 — CPU executor

Instruction:

- one trivial parameterized function,
- correctness gate,
- benchmark,
- end-to-end tune result.

## Pass 6 — Refactor after review

At this point, hand the repo to a stronger full-environment agent or human review before adding GPU complexity.

Every pass should end with:

- tests,
- changed-files summary,
- known limitations,
- unresolved questions,
- no speculative future implementation.

---

# 40. Stop Conditions

The project should pause and re-evaluate if any of the following become true:

1. Upstream maintainers reveal an actively developed pure-Julia core already solving the same seam.
2. Native execution requires duplicating a large fraction of `KernelAbstractions` internals.
3. The proposed interface cannot support existing KernelTuner use cases without becoming more complicated than the Python boundary it replaces.
4. Reproducible GPU measurement cannot be made backend-neutral enough to justify a common core.
5. A better abstraction appears in the JuliaGPU ecosystem that makes the project redundant.
6. The work starts becoming "write a new compiler."

Stopping would be a successful research outcome if it prevents duplicate infrastructure.

---

# 41. Research Questions

The project can generate useful knowledge even before becoming a mature package.

Questions worth measuring:

- How much tuning logic can remain fully backend-agnostic?
- Which workload properties must be part of a tuning cache key?
- How stable are winning configurations across driver/compiler versions?
- How transferable are configurations within one GPU architecture family?
- How many candidate evaluations are typically needed before simple random search matches exhaustive search?
- How frequently does the optimal block size depend on input shape?
- Can prior tuning receipts reduce search cost on similar hardware?
- How often do portable KA kernels need backend-specific parameter spaces?
- Can tuning data predict performance regressions before end-user reports?

These questions are more interesting than "can we write a for loop over block sizes?"

---

# 42. Potential Paper / Technical Report Direction

Do not write a paper before there is data.

But if the project matures, a defensible technical report could focus on:

> **Reproducible backend-agnostic autotuning of Julia GPU kernels with hardware-aware caching and provenance.**

Possible contributions:

- native Julia tuning architecture,
- backend-neutral search/evaluation contract,
- reproducible tuning receipts,
- empirical cross-device tuning study,
- cache transfer experiments,
- integration case studies with existing Julia GPU projects.

The paper-worthy part would be evidence and methodology, not merely the existence of another tuner.

---

# 43. Longer-Term ML Runtime Connection

If this tuner succeeds, it becomes useful infrastructure for the earlier "Julia-native inference systems layer" idea.

A future inference runtime may need optimized families for:

- RMSNorm,
- RoPE,
- softmax,
- attention subkernels,
- quantize/dequantize,
- fused elementwise operations,
- KV-cache movement,
- token sampling,
- small-batch matrix-vector paths.

The inference runtime should **consume** a tuner.

It should not contain a pile of one-off autotuning code.

That is the leverage.

---

# 44. Relationship to a Future Paged KV Project

A future `PagedKV.jl`-style project could use this tuner to select parameters for:

- page copy kernels,
- gather/scatter,
- quantized cache transforms,
- layout conversion,
- prefix-sharing operations,
- compaction.

This means today's small tuner can become tomorrow's low-level optimization service without being ML-specific itself.

---

# 45. Why This Connects to Project Neura

Project Neura's broader thesis is to shift repetitive mechanical reasoning out of the model and into explicit tools.

Kernel autotuning fits that thesis perfectly.

An LLM should not burn reasoning tokens guessing:

- 128 vs 256 threads,
- 2 vs 4 items per thread,
- whether one vector width is faster on this GPU,
- whether yesterday's tuned configuration still applies after a driver change.

The machine can test those questions.

Therefore the Neura-relevant abstraction is:

> **The model chooses the optimization intent and constraints.  
> The symbolic/tooling substrate performs bounded empirical search.  
> The result is verified, cached, and returned with provenance.**

That is a much stronger contract than "let the model write CUDA."

---

# 46. Success Criteria

The project can call v0.1 successful when all of the following are true:

- [ ] native mode runs without Python,
- [ ] existing Python-backed mode remains intact,
- [ ] parameter spaces are explicit,
- [ ] constraints are explicit,
- [ ] search is bounded,
- [ ] grid search is deterministic,
- [ ] random search is seed-reproducible,
- [ ] failures are classified,
- [ ] correctness is mandatory before selection,
- [ ] raw benchmark samples are retained,
- [ ] cache keys include hardware/software/workload identity,
- [ ] cache invalidation is conservative,
- [ ] receipts reconstruct the tuning decision,
- [ ] CPU vertical slice passes,
- [ ] CUDA vertical slice passes on real hardware,
- [ ] reduction tuning works,
- [ ] one Neura-relevant kernel works,
- [ ] no custom compiler was required,
- [ ] no existing Julia GPU project was needlessly reimplemented.

---

# 47. Definition of "Done Enough to Show People"

The project is ready for serious external discussion when this command-equivalent story is true:

> "Here is one Julia `KernelAbstractions` kernel family. Here is its legal parameter space. Here is a CPU reference. The native Julia tuner ran a bounded search on this GPU, rejected incorrect/illegal variants, benchmarked the valid ones, selected the fastest observed configuration, persisted it under a hardware/workload fingerprint, and can show the complete receipt. Re-running the same workload retrieves the verified cached configuration."

That is enough.

Do not wait until there is Bayesian optimization, ten backends, an ML server, or a Neura integration.

---

# 48. Immediate Next Action

The immediate next action is **not code generation**.

It is repo archaeology.

Qwen's first assignment should be:

```text
Read the current KernelTuner.jl repository in full enough to map:
1. public API entry points,
2. Julia ↔ Python boundary,
3. existing parameter-space representation,
4. tuning call flow,
5. benchmark/verification behavior,
6. current backend handling,
7. test layout,
8. serialization/output structures,
9. the smallest insertion point for a native Julia engine.

Do not modify code.

Produce:
- ARCHITECTURE_CURRENT.md
- NATIVE_ENGINE_INSERTION_PLAN.md
- a dependency/call graph in text form
- a risk list
- a list of questions that require human or maintainer decisions.

The design goal is coexistence, not rewrite.
```

Once that report exists, the next implementation pass can be designed against the real repository instead of our assumptions.

---

# 49. Final Architectural Rule

If this project stays disciplined, it can become a high-leverage piece of Julia infrastructure.

The rule that protects it is simple:

> **Own the decision machinery. Reuse the execution machinery.**

Julia already knows how to compile.

The GPU backends already know how to execute.

Kernel libraries already know how to implement operations.

This project should become exceptionally good at:

- exploring legal alternatives,
- proving they are correct,
- measuring them honestly,
- remembering what worked,
- and explaining why a configuration was selected.

That is a sufficiently large project.

Everything else can plug into it later.
