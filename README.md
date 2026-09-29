# Harpe.jl

**Julia-native semantic ML and agent execution runtime.**

Harpe loads existing open-weight models, preserves what they mean, and uses that
information to determine how they should physically exist and execute on the
hardware and workload actually present.

> The model should not lose its meaning before it reaches the machine.
> The runtime should not lose sight of why the model is being invoked.

## Status

**Requires Julia 1.12** (`julia = "1.12"` compat; CI runs 1.12 only).

**Phase 0 — repository foundation: COMPLETE** (incl. the BONES swarm-readiness
sprint and the foundation-hardening sprint: receipts/failure-taxonomy/freeze
infrastructure stress-tested, correctness laboratory established).
**Next gate: Phase 1 — the semantic core**, which implements the
object-model encoding decided in canon (§CIX). Nothing here runs models yet.

- [x] Clean `Project.toml`, no third-party dependencies (§VII: *Harpe earns every hard dependency*; only the `Dates` stdlib, enforced by test)
- [x] Package skeleton: logging conventions, backend interface draft
- [x] Test harness with dependency-law enforcement
- [x] CI (tests + format + bench-smoke, dev-loop entry included)
- [x] Benchmark harness with benchmark-integrity conventions (§XXXIII)
- [x] Research program: [docs/research/KV_MEMORY_PROGRAM.md](docs/research/KV_MEMORY_PROGRAM.md)
- [x] BONES sprint (agent charter, module skeleton, receipts types, tooling)
- [x] Foundation hardening (receipt stress tests, taxonomy pins, freeze evidence, lab fixtures, empty-core fence)
- [x] Phase 1: semantic core per §CIX — types exist, `toy2` expressible end to end, **no execution yet**
- [ ] Phase 2: CPU oracle (reference execution) — fills the operators' math, fills `expected_logits.toml`
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
| 0 | Repository foundation | **COMPLETE** |
| 1 | Semantic core (§CIX encoding) | **COMPLETE** — expressible, not executable |
| 2 | Reference execution (CPU oracle) | **next gate** |
| 3 | First real model import | |
| 4 | CUDA.jl execution | |

Training is **not** part of Harpe — by explicit, permanent decision
([docs/Harpe_Stack.md §LVIII](docs/Harpe_Stack.md)). If you want to contribute
to the Julia ML ecosystem and want a job with real scope: **build the training
stack.** The natural place to start is [WGPU.jl](https://github.com/JuliaGPU/WGPU.jl).
Harpe will load its checkpoints like everyone else's.

## Layout

```
src/           package core (zero deps; logging, receipts, errors, backends)
test/          test harness (workspace member; per-area test files)
benchmark/     benchmark harness (workspace member; results/ accrues)
docs/          architecture: Harpe_Stack.md is canon; ARCHITECTURE.md is the map
libs/          local dev sources (gitignored; Lava lives here) — DO NOT TOUCH
```

## Working here

- `AGENTS.md` is the **binding agent charter** — read it before your first edit.
- `docs/ARCHITECTURE.md` maps every module to its canon section and phase.
- Work items follow `.github/ISSUE_TEMPLATE/work-item.md` (§LXXI format);
  PRs are receipts per `.github/PULL_REQUEST_TEMPLATE.md` (§LXXII).
- Open architecture decisions are quarantined in
  [docs/DECISION_PACKETS.md](docs/DECISION_PACKETS.md) — not settled by
  local engineering alone.

## Commands

```bash
make test          # test suite (workspace root; runs with -t 2)
make bench         # benchmarks (append to benchmark/results/)
make format        # format the repo
make format-check  # CI's formatting gate
make freeze        # curated context-freeze bundle (evidence, not ceremony)
```

Under the hood (Julia 1.12 workspace — activate the root, then let Pkg
discover the members; do NOT `Pkg.instantiate("test")`, positional
instantiate is not valid API):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.activate("test"); Pkg.instantiate(); Pkg.activate("benchmark"); Pkg.instantiate()'
julia --project=test --check-bounds=yes test/runtests.jl
julia --project=benchmark benchmark/runbenchmarks.jl
```
