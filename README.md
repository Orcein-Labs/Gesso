# Gesso.jl

A Julia-native inference runtime. It loads open-weight models, keeps their
meaning intact, and runs them on the hardware you actually have — CPU,
CUDA, or Vulkan (Lava) — through one `Session`.

Julia 1.12. Training is out of scope. Gesso loads checkpoints; it does not
make them.

## Head to head

SmolLM2-135M, RTX 5060, Julia 1.12.6, greedy, 8 new tokens, compile
excluded, host-visible token IDs. Same `Session` interpreter on every
backend. 27/27 IDs match the independent Hugging Face reference. A planted
NaN still fails with `ERR_NUMERICAL_INSTABILITY`.

### Gesso Lava vs Gesso CUDA (2026-10-08)

Warmed decode tok/s. Receipts:
[`s4-lava-three.json`](docs/archive/2026-10_regime-ii/s4-lava-three.json),
[`s4-cuda-three.json`](docs/archive/2026-10_regime-ii/s4-cuda-three.json).

| Prompt | Lava | CUDA | Lava/CUDA |
| --- | ---: | ---: | ---: |
| Hello | **37.43** | 31.38 | 1.19 |
| The quick brown fox | **36.94** | 28.44 | 1.30 |
| Julia is a programming language. | **36.78** | 30.10 | 1.22 |

Median Lava/CUDA = **1.22**. Lava decode started this campaign at **2.40 /
2.49 / 2.49** tok/s
([arm 11](docs/archive/2026-10_regime-i/receipts/arm11-lava-after.json)).

Warmed `decode!` host allocation: Lava 3.28 MB, CUDA 1.06 MB.

These are implementation pipelines, not identical floating-point instruction
sequences. CUDA tok/s moves a few points night to night on this box; the
pairs above were taken the same night.

### Gesso CUDA vs eager PyTorch (2026-10-03)

Same checkpoint, prompt `"Hello"`, `max_new_tokens=8`, greedy, batch 1, no
`torch.compile`, no vLLM. Gesso CUDA is F32; eager PyTorch is bfloat16.
Both dtypes travel with the numbers; this is not a cross-dtype speed claim.

| Side | Warmed generate | Factor |
| --- | ---: | ---: |
| Gesso CUDA | 240 ms | **1.45×** vs eager |
| Eager PyTorch | 348 ms | 1× |

Source: [`benchmark/results/2026-10-03.tsv`](benchmark/results/2026-10-03.tsv)
row `smollm2_g2_factor_eager_over_gesso`.

## Run it

```julia
using Gesso
using Lava    # Vulkan path — GessoLavaExt
# or
using CUDA    # NVIDIA path — GessoCUDAExt

model, tensors, cfg = Gesso.load_llama(ENV["GESSO_SMOLLM2_DIR"])
backend = Gesso.LavaBackend()          # or Gesso.CUDABackend()
tensors = Gesso.to_device(backend, tensors)
session = Gesso.Session(model, tensors; backend, context_length=128, eos_token_id=0)
ids = Gesso.generate(session, "Hello"; max_new_tokens=8)
```

Requesting CUDA or Lava without a device throws a typed error. There is no
silent CPU fallback.

Reproduce the table:

```bash
GESSO_SMOLLM2_DIR=/path/to/SmolLM2-135M \
GESSO_HF_REFERENCE=docs/archive/2026-10_regime-i/receipts/hf-reference.json \
julia --project=test --startup-file=no test/regime_i_performance.jl lava /tmp/lava.json

GESSO_SMOLLM2_DIR=/path/to/SmolLM2-135M \
GESSO_HF_REFERENCE=docs/archive/2026-10_regime-i/receipts/hf-reference.json \
julia --project=test --startup-file=no test/regime_i_performance.jl cuda /tmp/cuda.json
```

The test suite never downloads weights. Without `GESSO_SMOLLM2_DIR` the
real-model gates are named skips.

## What is in the tree

One engine: paged KV, `Session` / `generate`, declared prefix share
(`fork` + copy-on-write), receipts on every call. CPU is the oracle.
CUDA is the NVIDIA lowering (device argmax, GEMM attention, autotuned
`matmul!`). Lava is the portable Vulkan lowering. Core Gesso has no
CUDA or Lava dependency.

Landed on SmolLM2-135M. Next work is the same engine across more
architectures and larger models — dense, MoE, and the oddballs — with
the same ID oracle and the same compile-excluded harness.

Regime II notes live in [`docs/goals/REGIME_II.md`](docs/goals/REGIME_II.md)
and [`docs/archive/2026-10_regime-ii/`](docs/archive/2026-10_regime-ii/).

## Phases

```
MAKE IT WORK → MAKE IT COMPLETE → MAKE IT MEASURABLE → MAKE IT FAST
```

| Phase | Status |
|------:|--------|
| 0–2 Foundation, semantic core, CPU oracle | Complete |
| 3 First real model (Llama import, GQA, GPT-2 BPE) | Complete; SmolLM2 gate skip-or-green in CI |
| 4 CUDA.jl | Complete |
| 5 Native engine (`Session`, paged KV) | Complete |
| 6 Observability (receipts, Profiling) | Complete |
| 7 Declared prefix share (`fork`) | Complete |
| 8 Lava/Vulkan | Complete; now at CUDA-parity decode on SmolLM2-135M |
| 9 Autotune (CUDA `matmul!` winner cache) | Complete |
| 10 Speed-floor rungs (device argmax, GEMM attention, alloc, G2) | Complete through 10H |

Training is **not** part of Gesso
([§LVIII](docs/Gesso_Stack.md)). If you want to contribute to the Julia ML
ecosystem and want a job with real scope: **build the training stack.**
Start at [WGPU.jl](https://github.com/JuliaGPU/WGPU.jl). Gesso will load
those checkpoints like everyone else's.

## Layout

```
src/           package core (zero third-party deps)
ext/           CUDA and Lava package extensions
test/          test harness
benchmark/     benchmark harness (results/ accrues)
docs/          Gesso_Stack.md is canon; ARCHITECTURE.md is the map
libs/          local dev sources (gitignored) — DO NOT TOUCH
```

## Working here

- `AGENTS.md` is the binding agent charter — read it before your first edit.
- `docs/ARCHITECTURE.md` maps every module to its canon section.
- Work items follow `.github/ISSUE_TEMPLATE/work-item.md`.
- Open architecture decisions live in
  [docs/DECISION_PACKETS.md](docs/DECISION_PACKETS.md).

## Commands

```bash
make test          # test suite (workspace root; runs with -t 2)
make bench         # benchmarks (append to benchmark/results/)
make format        # format the repo
make format-check  # CI's formatting gate
make freeze        # curated context-freeze bundle
```

```bash
GESSO_SMOLLM2_DIR=/path/to/SmolLM2-135M make test
```

The snapshot needs `config.json`, `model.safetensors` (or the shard index),
`vocab.json`, and `merges.txt` from
[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

Julia 1.12 workspace — activate the root, then let Pkg discover the
members; do not `Pkg.instantiate("test")`:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.activate("test"); Pkg.instantiate(); Pkg.activate("benchmark"); Pkg.instantiate()'
julia --project=test --check-bounds=yes test/runtests.jl
julia --project=benchmark benchmark/runbenchmarks.jl
```
