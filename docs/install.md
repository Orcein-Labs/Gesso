# Install

## Requirements

- Julia 1.12. `Project.toml` pins `julia = "1.12"`.
- A GGUF model file, or a Llama-family safetensors directory (see [models.md](models.md)).
- Nothing else for the CPU backend. The GPU backends are optional and need
  extra packages and drivers (see [backends.md](backends.md)).

Gesso runs models. It does not train them and it does not download weights.

## Install the package

```julia
using Pkg
Pkg.add(url="https://github.com/Orcein-Labs/Gesso")
```

or, from a checkout:

```bash
git clone https://github.com/Orcein-Labs/Gesso
cd Gesso
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The core package depends on Dates, LinearAlgebra, Libdl, Mmap, JSON and
Sockets. It depends on neither GPU stack, so this install gives you the CPU
backend only.

## GPU backends

`CUDA` and `Lava` are weak dependencies. Loading one activates the matching
package extension (`GessoCUDAExt`, `GessoLavaExt`).

- CUDA: add `CUDA` to your environment, then `using CUDA`.
- Lava: Gesso carries a patched copy of Lava at `vendor/Lava`. Install that
  copy, not a registry release:

  ```julia
  using Pkg
  Pkg.add(url="https://github.com/Orcein-Labs/Gesso", subdir="vendor/Lava")
  ```

  The repository's own `test` and `benchmark` environments load it through
  `[sources]` and also pin Vulkan and VulkanCore from git. If you work inside
  a checkout, use `--project=benchmark` for scripts that need a GPU.

## First run

The CPU backend needs no GPU. With a Llama-family safetensors directory:

```julia
using Gesso

model, tensors, cfg = Gesso.load_llama("/path/to/model-dir")
session = Gesso.Session(model, tensors; backend=Gesso.CPUBackend(),
                        context_length=128, eos_token_id=0)
ids = Gesso.generate(session, "Hello"; max_new_tokens=8)
```

`Session` uses the CPU backend when you pass none. To use a GPU, load its
package, build the backend, and move the weights:

```julia
using Lava
backend = Gesso.LavaBackend()
tensors = Gesso.to_device(backend, tensors)
session = Gesso.Session(model, tensors; backend, context_length=128, eos_token_id=0)
```

GGUF files load with `Gesso.load_gguf(path)`, which returns
`(model, tensors, spec)`. To chat with one over HTTP, see [serving.md](serving.md).

## Check the install

1. `julia --project=. -e 'using Gesso; println("ok")'` loads the package.
2. `julia --project=. scripts/gesso-serve.jl --check` starts the chat server
   with a fake engine on an ephemeral port, asks `/health`, and prints
   `gesso-serve check ok (ephemeral port N, fake engine)`. It needs no model
   and no GPU.
3. With a real GGUF, run the server on the CPU and send one request:

   ```bash
   julia --project=. scripts/gesso-serve.jl --gguf /path/to/model.gguf --cpu --port 8080
   curl -s localhost:8080/v1/models
   ```

   `tools/clean-install-check.sh <model.gguf> <registry-url>` automates this
   in a fresh Julia depot: it installs Gesso from a registry, starts a CPU
   server, sends one chat request, and fails unless a completion comes back.

If you ask for a GPU backend and no usable device exists, Gesso raises a
typed error instead of falling back to the CPU. See [errors.md](errors.md).

## Developing

`make test` runs the suite, `make format-check` the formatting gate, and
`make env-vars` regenerates [env-vars.md](env-vars.md). The repository README
has the workspace setup. Some real-model tests need `GESSO_SMOLLM2_DIR`; the
suite never downloads weights.
