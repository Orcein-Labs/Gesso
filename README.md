# Harpe.jl

**Julia-native semantic ML and agent execution runtime.**

Harpe loads existing open-weight models, preserves what they mean, and uses that
information to determine how they should physically exist and execute on the
hardware and workload actually present.

> The model should not lose its meaning before it reaches the machine.
> The runtime should not lose sight of why the model is being invoked.

## Status

**Phase 0 — repository foundation: COMPLETE.** Currently in the BONES sprint
(swarm-readiness scaffolding: agent charter, module skeleton, receipts types,
dev tooling). Nothing here runs models yet.

- [x] Clean `Project.toml`, no third-party dependencies (§VII: *Harpe earns every hard dependency*; only the `Dates` stdlib, enforced by test)
- [x] Package skeleton: logging conventions, backend interface draft
- [x] Test harness with dependency-law enforcement
- [x] CI (tests + format)
- [x] Benchmark harness with benchmark-integrity conventions (§XXXIII)
- [x] Research program: [docs/research/KV_MEMORY_PROGRAM.md](docs/research/KV_MEMORY_PROGRAM.md)
- [ ] BONES sprint (agent charter, module skeleton, receipts types, tooling)
- [ ] Telemetry collection (Phase 6)

## The stack

```julia
using Harpe
using Lava   # Phase 8 — portable/Vulkan path
# or
using CUDA   # Phase 4 — strategic fast path
```

## Development order (§III)

```
MAKE IT WORK → MAKE IT COMPLETE → MAKE IT MEASURABLE → MAKE IT FAST
```

Current phases (docs/Harpe_Stack.md §LXXIII ff.):

| Phase | Deliverable | Status |
|------:|-------------|--------|
| 0 | Repository foundation | **in progress** |
| 1 | Semantic core | |
| 2 | Reference execution (CPU oracle) | |
| 3 | First real model import | |
| 4 | CUDA.jl execution | |

Training is **not** part of Harpe — by explicit, permanent decision
([docs/Harpe_Stack.md §LVIII](docs/Harpe_Stack.md)). If you want to contribute
to the Julia ML ecosystem and want a job with real scope: **build the training
stack.** The natural place to start is [WGPU.jl](https://github.com/JuliaGPU/WGPU.jl).
Harpe will load its checkpoints like everyone else's.

## Layout

```
src/           package core (zero deps; logging, backend interface)
test/          test harness (workspace member)
benchmark/     benchmark harness (workspace member)
ci/            CI environment (workspace member)
docs/          architecture: Harpe_Stack.md is canon
libs/          local dev sources (gitignored; Lava lives here)
```

## Commands

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.instantiate("test"); Pkg.instantiate("benchmark")'
julia --project=test -e 'using Pkg; Pkg.test()'       # test suite
julia --project=benchmark benchmark/runbenchmarks.jl  # benchmark suite
julia -e 'using JuliaFormatter; format(".", verbose=true)'  # format
```
