# Gesso.jl

Gesso is an inference runtime written in Julia. It loads open-weight models
and runs them on CPU, on NVIDIA through CUDA.jl, or on any Vulkan GPU through
Lava, all behind one `Session` API. It also ships an OpenAI-compatible chat
server, so tools that talk to the OpenAI API can use a local model through it.

We test everything against an oracle. The CPU path is the reference, and on
real checkpoints we compare greedy token IDs against Hugging Face or
llama.cpp. A speed number only counts if the IDs still match.

Gesso needs Julia 1.12. It does not train models. It only runs them.

## Results

Two models, measured on one desktop: an RTX 5060 (8 GB) on a PCIe gen3 x8
link, with 31 GiB of single-channel DDR4. SmolLM2-135M is small enough to
show what the engine itself costs. Ornith-1.5-35B-A3B is a model you can
actually work with, and it doesn't fit on the card.

### Ornith-1.5-35B-A3B on an 8 GB card (2026-10-09 to 2026-10-11)

The model is a hybrid MoE (gated DeltaNet plus attention, 256 experts with 8
active per token). We ran Unsloth's `UD-Q2_K_XL` GGUF, 11.4 GiB. Its expert
weights are IQ2_XS, IQ3_XXS and IQ4_XS, and the dense weights are Q5_K to
Q8_0. The weights don't fit in 8 GB, so some experts live in host memory and
run on the CPU. llama.cpp does the same thing with `--n-cpu-moe`.

The baseline is llama.cpp's CUDA build on the same machine:
`-ngl 99 --n-cpu-moe 24 -fa on`, tg64 **32.80 ± 3.20 tok/s**, pp1024
**386 tok/s**. Gesso's greedy IDs match llama.cpp's exactly on the gate
prompt.

**Decode.** Gesso CUDA with default settings runs at 27-29 ms per token,
against llama.cpp's 30.5 ms. Hot experts are opt-in (`GESSO_GPU_HOT_EXPERTS=1
GESSO_GPU_HOT_PIN=1`). They keep the most-used host experts in a VRAM pool,
and decode speed then holds up as the context grows:

| Context | ms/token | tok/s |
| --- | ---: | ---: |
| 1 token | 20.0 | 49.9 |
| 8,192 | 22.0 | 45.5 |
| 16,384 | 24.1 | 41.6 |
| 32,768 | 29.3 | 34.2 |

Each row is one 65-token generation, timed per token over tokens 2-64, with
the default q8 KV cache and the hot-expert budget fixed at 480 MiB
(`GESSO_GPU_HOT_EXPERTS_MIB=480`). At 32k tokens Gesso still decodes faster than
llama.cpp does at short context. An F32 cache runs out of memory at 16k, and
F16 runs out at 32k.

**Prefill**, with hot experts on:

| Prompt | Gesso tok/s | llama.cpp tok/s |
| --- | ---: | ---: |
| 512 | 414 | |
| 1,024 | 437 | 386 |
| 2,048 | 433 | |
| 4,096 | 385 | |

An 8k prompt prefills in about 26 s and a 32k prompt in 100-130 s.

**Multi-turn.** The session keeps its KV cache and DeltaNet state between
calls. On an 8k-token conversation, the second turn takes 1.26 s instead of
the 24.6 s a fresh prefill costs. Reusing prompt rows gives the same tokens as
a fresh run. Rows that were computed during decode round slightly differently
from a batched prefill, so reusing those can flip a near-tied greedy pick
later on. llama.cpp's cache reuse behaves the same way.

Some caveats:

- The gate is a short cold run and moves about ±2 tok/s between runs. The
  tables come from warm generations.
- How fast a prompt decodes with hot experts depends on how often its expert
  picks are already resident. Hits run on the GPU and misses on the CPU,
  and the two round differently. So with hot experts on, greedy output can
  depend on what the session ran earlier. With them off, it doesn't.
- This machine's memory bus limits the result more than any kernel does.
  One DDR4 channel gives about 14 GB/s, and reading the host experts costs
  about 12 ms of every token.
- These numbers are CUDA. Lava support for this model is newer, and we
  haven't published its speed yet.

### SmolLM2-135M (2026-10-03 and 2026-10-08)

Greedy, 8 new tokens, compile time excluded. All 27 IDs match the Hugging
Face reference on every backend. A planted NaN still fails with
`ERR_NUMERICAL_INSTABILITY`.

Lava against CUDA, warmed decode tok/s, both taken the same night:

| Prompt | Lava | CUDA | Lava/CUDA |
| --- | ---: | ---: | ---: |
| Hello | **37.43** | 31.38 | 1.19 |
| The quick brown fox | **36.94** | 28.44 | 1.30 |
| Julia is a programming language. | **36.78** | 30.10 | 1.22 |

Lava started this work at 2.4 tok/s.

Gesso CUDA against eager PyTorch, prompt `"Hello"`, batch 1, no
`torch.compile`: a warmed generate takes 240 ms in Gesso and 348 ms in
PyTorch, so Gesso is 1.45× faster. Gesso ran F32 and PyTorch ran bfloat16,
so read this as a comparison of two pipelines, not of dtypes. The source is
[`benchmark/results/2026-10-03.tsv`](benchmark/results/2026-10-03.tsv), row
`smollm2_g2_factor_eager_over_gesso`.

## Run it

```julia
using Gesso
using Lava    # Vulkan, through GessoLavaExt
# or
using CUDA    # NVIDIA, through GessoCUDAExt

model, tensors, cfg = Gesso.load_llama(ENV["GESSO_SMOLLM2_DIR"])
backend = Gesso.LavaBackend()          # or Gesso.CUDABackend()
tensors = Gesso.to_device(backend, tensors)
session = Gesso.Session(model, tensors; backend, context_length=128, eos_token_id=0)
ids = Gesso.generate(session, "Hello"; max_new_tokens=8)
```

Lava is the default backend. CUDA is there for people who want the vendor
stack, and the scripts pick it with `GESSO_BACKEND=cuda`. If you ask for a GPU
backend and there's no device, you get a typed error. Gesso never falls back
to the CPU without telling you.

GGUF models load with `Gesso.load_gguf`, which also needs `GESSO_GGML_DIR` to
point at a llama.cpp build. The chat server is `scripts/gesso-serve.jl`. It
serves `/v1/models` and `/v1/chat/completions`.

### Reproduce

The 35B gate, which checks the IDs and prints one receipt line:

```bash
GESSO_ORNITH35_GGUF=/path/to/Unsloth-Ornith-1.5-35B-A3B-UD-Q2_K_XL.gguf \
GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
GESSO_BACKEND=cuda \
julia --project=benchmark -t 16 scripts/gate_35b.jl
```

Add `GESSO_GPU_HOT_EXPERTS=1 GESSO_GPU_HOT_PIN=1` for hot experts. Keep the
model file on a fast local disk. Mapping it from a slow network or FUSE
mount made expert reads about 5× slower for us.

The SmolLM2 tables:

```bash
GESSO_SMOLLM2_DIR=/path/to/SmolLM2-135M \
GESSO_HF_REFERENCE=test/fixtures/smollm2/hf-reference.json \
julia --project=test --startup-file=no test/regime_i_performance.jl lava out-lava.json
```

Swap `lava` for `cuda` to get the other column.

The test suite never downloads weights. If `GESSO_SMOLLM2_DIR` isn't set, the
real-model gates are reported as named skips.

## What's in the tree

- One engine with paged KV, `Session` / `generate`, prefix sharing (`fork`
  with copy-on-write), and a receipt on every call.
- Model support: Llama-family safetensors, and GGUF with K-quants and
  IQ-quants, including the Qwen3.5-MoE hybrid architecture.
- KV cache formats f32, f16, q8 and q4h (`GESSO_KV_FORMAT`). CUDA and Lava
  default to q8.
- Backends: CPU, the oracle. CUDA, with device argmax, GEMM attention,
  quantized expert kernels and an autotuned `matmul!`. Lava, the portable
  Vulkan path. The core package depends on neither GPU stack.
- An OpenAI-compatible chat server with streaming and sampling.

## Layout

```
src/           package core
ext/           CUDA and Lava package extensions
scripts/       gates, benchmarks and the chat server
test/          test harness
benchmark/     benchmark harness (results/ accrues)
vendor/        patched upstream packages (Lava)
docs/          reference (environment variables)
libs/          local dev sources (gitignored)
```

Every `GESSO_*` environment variable is indexed in [`docs/env-vars.md`](docs/env-vars.md).
The long-prompt benchmarks read `benchmark/corpus-pride-and-prejudice.txt`, the
opening chapters of Jane Austen's novel (public domain, from Project Gutenberg).

## Commands

```bash
make test          # test suite (workspace root; runs with -t 2)
make bench         # benchmarks (append to benchmark/results/)
make format        # format the repo
make format-check  # CI's formatting gate
make env-vars      # regenerate docs/env-vars.md
```

This is a Julia 1.12 workspace. Activate the root and let Pkg find the
members. Don't run `Pkg.instantiate("test")`.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.activate("test"); Pkg.instantiate(); Pkg.activate("benchmark"); Pkg.instantiate()'
julia --project=test --check-bounds=yes test/runtests.jl
```

The SmolLM2 gates need `config.json`, `model.safetensors` (or the shard index),
`vocab.json` and `merges.txt` from
[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

## License

Apache License 2.0. See [LICENSE](LICENSE).
