# Backends

Gesso runs the same models on four backends. Pick one by passing it to
`Session` or `to_device`, or for the scripts and chat server with
`GESSO_BACKEND=lava|cpu|cuda`. The chat server also takes `amdgpu`.

| Backend | Constructor | What it needs | Default for |
| --- | --- | --- | --- |
| CPU | `Gesso.CPUBackend()` | Nothing beyond Julia 1.12 | `Session` when no backend is passed |
| Lava (Vulkan) | `Gesso.LavaBackend()` after `using Lava` | The vendored Lava package and a working Vulkan driver | Scripts and the chat server when `GESSO_BACKEND` is unset |
| CUDA | `Gesso.CUDABackend()` after `using CUDA` | CUDA.jl and an NVIDIA driver with a functional device | Nothing; opt-in |
| AMDGPU (ROCm) | `Gesso.AMDGPUBackend()` after `using AMDGPU, KernelAbstractions` | AMDGPU.jl, KernelAbstractions.jl and a ROCm driver with a functional device | Nothing; opt-in, GPU-untested |

## CPU

The correctness oracle. Every GPU result in the project is checked against
it, and it is the only backend with no extra install. It keeps the KV cache in
`f32` unless you set `GESSO_KV_FORMAT`.

## Lava

A Vulkan backend. Install the patched copy shipped in the repository (see
[install.md](install.md)); it is not a registry release. You also need a
Vulkan driver and ICD for your GPU; `vulkaninfo` should list a device.

`Gesso.LavaBackend()` is defined only after `using Lava`. If no Vulkan device
is usable it logs a warning with the underlying exception and raises
`ERR_RESOURCE_LIMIT`. It never switches to the CPU or CUDA on its own.

Lava supports the `f32`, `f16` and `q8` KV formats, not `q4h`. The Gemma 4
routed feed-forward block runs on the host under Lava.

## CUDA

Optional. Install CUDA.jl, run `using CUDA`, then use `Gesso.CUDABackend()`
or `GESSO_BACKEND=cuda`. It raises `ERR_RESOURCE_LIMIT` if `CUDA.functional()`
is false. Compared with Lava it adds device argmax and GEMM attention, and it
supports the `q4h` KV format; it does not implement some model paths that Lava
does, for example the Gemma 4 and short-convolution (LFM2) paths. Check
`Gesso.supports(backend, capability)` for a given capability before relying
on it: an unknown or missing capability returns `false`, and an operation
with no lowering raises `LoweringNotImplemented`.

## AMDGPU

GPU-untested. This backend has never run on AMD hardware. Its kernels are
written in KernelAbstractions, and the test suite runs them on
KernelAbstractions' CPU backend against the CPU oracle and compiles them for
AMD targets. The kernels are written for correctness and are not tuned for
speed.

Install AMDGPU.jl and KernelAbstractions.jl, then
`using AMDGPU, KernelAbstractions, Gesso`; both packages are needed to load
the extension. `Gesso.AMDGPUBackend()` raises `ERR_RESOURCE_LIMIT` when
AMDGPU.jl finds no functional ROCm device, and it never falls back to the CPU.

It covers the Llama-family, Qwen3.5 hybrid, LFM2 and Gemma 4 paths with the
quantized and IQ-quantized weight formats. The KV cache is `f32`; a `q4h`
cache is refused, and `q8` is not yet supported end to end. MoE routing and
the Gemma 4 routed feed-forward block run on the host. Check
`Gesso.supports(Gesso.AMDGPUBackend, capability)` before relying on a
capability.

## Choosing and mixing

- Load only the GPU package you intend to use. Each extension installs its
  own `Gesso.to_device`, and moving weights binds them to that backend.
- A weight budget (`GESSO_GPU_BUDGET_MIB`) decides how much of the model lands
  on the device; the rest stays in host memory. See [memory.md](memory.md).
- Backend-specific environment variables are listed in [env-vars.md](env-vars.md).
