# Gesso debt register

Known debt, newest first. Each entry names its cause and how it gets paid off. See
`docs/standards/routine-maintenance.md` in the workspace for the cadence.

## 2026-10-08

- **Model files on NVMe have no managed home.** A DISKGUARD WARN fired after about 25 GB of
  GGUF copies to `~/.cache/gesso` in one hour. The owner wants the models kept on NVMe, so the
  WARN is accepted. Mmap from a spinning disk is about 5x slower. Pay off: one documented model
  directory, set by env, that the gate and bench scripts use, so files aren't copied ad hoc.
  Do not evict models automatically.
- **Runtime dependency on a local llama.cpp build** (`GESSO_GGML_DIR`, ggml `to_float` via
  `dlopen`). Native matmul kernels now cover every format Gesso ships (IQ2_XS, IQ2_XXS, IQ2_S,
  IQ3_XXS, IQ3_S, IQ4_XS, IQ1_S, IQ1_M, Q2_K-Q6_K, Q8_0), so what is left is the load path:
  GGUF import and expert dequantization still call `to_float`, and any other format falls back
  to it. The type-traits struct is read at fixed byte offsets. Those reads are now checked
  against Gesso's own block table and fail loudly on a mismatch, but a new ggml layout still
  stops the load. Pay off: native dequantization on the load path, then drop the library.
- **Host-side MoE expert path is opt-in** (`GESSO_MOE_HOST_EXPERTS=1`). It measured slower than
  uploading experts. Pay off: CPU miss compute that overlaps GPU hits, using the native kernels,
  inside the expert cache. Then remove the switch.
- **Device scratch buffers are shared per device.** Concurrent decodes on one GPU would collide.
  Pay off: per-session workspaces before any multi-session serving.
- **`QuantMatrix` allocates a Q8_K activation buffer per matmul.** Pay off: reuse buffers from
  a session workspace.
- **Gesso builds against a vendored, patched Lava** (`vendor/Lava`, patches in
  `vendor/patches/lava/`). Patch 0001 drops the `ConstAdaptor` `ReshapedArray` override that
  KernelAbstractions 0.9.43+ also defines; Julia 1.12 refused the overwrite at precompile, which
  broke CI. Pay off: land the patches upstream, then go back to a pinned upstream revision.
- **Lava imports GLFW, and GLFW's `__init__` needs a display.** Every env that loads Lava fails
  to precompile on a headless box. CI and `make test` run under `xvfb-run` to get around it.
  Pay off: make the windowing import lazy in Lava (a weakdep or extension).
- **119 docs contain machine-specific paths** (home directories, drive mounts). The repo is public.
  Pay off: a docs sweep. Code paths are already env-only.
