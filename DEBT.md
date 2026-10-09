# Gesso debt register

Known debt, newest first. Each entry names its cause and how it gets paid off. See
`docs/standards/routine-maintenance.md` in the workspace for the cadence.

## 2026-10-08

- **Model files need a managed NVMe cache.** A DISKGUARD WARN fired after about 25 GB of GGUF
  copies to `~/.cache/gesso` in one hour. Mmap from a spinning disk is about 5x slower, so
  benchmarks copy models to NVMe by hand. Pay off: one cache directory with a size budget and
  LRU eviction, used by the gate and bench scripts. Nothing gets copied ad hoc.
- **Runtime dependency on a local llama.cpp build** (`GESSO_GGML_DIR`, ggml `to_float` via
  `dlopen`). It also reads ggml's type-traits struct at fixed byte offsets, which is ABI-fragile.
  Pay off: native dequantization for every format. Native dot products now cover IQ2_XS,
  IQ3_XXS, IQ4_XS, Q2_K-Q6_K and Q8_0. Still missing: IQ2_S, IQ3_S, IQ1_M.
- **The `to_float` fallback in `QuantMatrix` matmul** still serves IQ2_S, IQ3_S and IQ1_M. That
  makes Qwen3.8-27B UD-IQ2_M slow. Pay off: native kernels for those three formats.
- **Host-side MoE expert path is opt-in** (`GESSO_MOE_HOST_EXPERTS=1`). It measured slower than
  uploading experts. Pay off: CPU miss compute that overlaps GPU hits, using the native kernels,
  inside the expert cache. Then remove the switch.
- **Device scratch buffers are shared per device.** Concurrent decodes on one GPU would collide.
  Pay off: per-session workspaces before any multi-session serving.
- **`QuantMatrix` allocates a Q8_K activation buffer per matmul.** Pay off: reuse buffers from
  a session workspace.
- **Lava pirates `KernelAbstractions.adapt_structure`** (upstream `Lava/src/array/ka_backend.jl`).
  Precompilation errors as a result. Pay off: an upstream fix, or a pinned patched revision.
  `benchmark/Project.toml` also pulls Lava unpinned.
- **121 docs contain machine-specific paths** (`/mnt/d`, `/home/mattc`). The repo is public.
  Pay off: a docs sweep. Code paths are already env-only.
