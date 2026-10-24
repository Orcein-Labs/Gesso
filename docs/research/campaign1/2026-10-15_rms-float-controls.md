# Campaign 1: CPU-contract RMS and explicit floating-point preservation

Status: isolated source experiment implemented; rejected as a sufficient repair.
External and public operators pass, but the unchanged model gate is 39/41: fresh
p64 cancellation and its continuation fail. Normal history passes all 36 rows,
including three p65 cases. See the checkpoint and raw validation receipt; retain
the candidate for diagnosis without admitting it as the production baseline.
Parent: `2026-10-14_campaign1-lava-correctness.md`.

## Objective

Test Float32 squares, Float64 accumulation/normalization and Float32 output
consistently for 2D/3D RMS rows. Make the invalid-row decision independent of
GLSL Sqrt special-value semantics. Require the per-width Vulkan preservation
contract needed by this kernel, with explicit unsupported-device behavior.
Prove this through public operators and unchanged real-model stress before
selecting a repair. Do not equate external operator passes with model parity.

## Evidence and source bases

Working Gesso `5fe6dc13a55b0212d8c26a59f6f8c2d77a83a966` passes 38/41 original
history checks; three stable p65 mismatches remain. Keep it frozen.
Standalone Lava parent is `c5a7ec141f90f7fdbb3261d0b1eed9c8f804568e` at
`lava-accumulator-handoff-20261014`; preserve it too.

External RMS v3 is CPU-exact for 19 finite fixtures but returns zero on F32
square overflow. Adding only `SignedZeroInfNanPreserve` 32/64 declarations
repairs that marker without changing the shader arithmetic/control bytes.
External v4 classifies total rather than Sqrt, passes the same fixtures and
24 additional special-value controls. Raw receipts live in
`<pystack-work>/results/campaign1-20261014`:
`rms-intermediate-trace-validation-1.json`,
`rms-float-controls-both-shader-validation-1.json`, and
`rms-float-controls-total-validation-1.json`.

## Permitted files and ownership

Start new isolated standalone Lava and Gesso candidates. Do not hand-edit the
vendored source; carry the selected standalone delta through the vendor script.

Gesso runtime/tests: `ext/lava_ops.jl`, `test/test_lava_ops.jl`.
Vendor integration: `vendor/patches/lava/0005-*.patch`, `vendor/README.md`,
`scripts/vendor_lava.sh`, and regenerated `vendor/Lava/` from the exact patch.

Lava envelope, limited to the opt-in contract and its cache/capability plumbing:
`src/compiler/target.jl`, `src/compiler/compilation.jl`,
`src/compiler/spirv/module.jl`, `src/compiler/spirv/emit.jl`,
`src/runtime/device.jl`, `src/runtime/pipeline.jl`, `src/runtime/launch.jl`,
`src/runtime/frozen_cache.jl`, `src/array/ka_backend.jl`,
`test/test_float_controls.jl` (new), `test/runtests.jl`.
Use fewer files where the existing interfaces allow it. Planning/receipt files
belong in this planning worktree or the external campaign artifact directory.

## Interfaces and invariants

- Query actual per-device `PhysicalDeviceFloatControlsProperties`; retain support
  per width. Legacy mode support is a property, not the FloatControls2 feature.
- Opt in only the demonstrated RMS kernel. Default compiler behavior and unrelated
  GEMM/quant kernels remain unchanged. No name-substring dispatch in product code.
- Include requirements in every applicable compiled, launch, disk and frozen cache
  contract. A stale artifact must not silently drop a requested preservation mode.
- Missing requested support rejects before pipeline creation with a descriptive
  error. Do not omit required modes, retry after GPU failure or special-case a
  GPU vendor. No new dependency or FloatControls2 enablement is needed.
- For F32 input, retain F32 square rounding, use F64 lane partials/fold/finish,
  then F32 storage. One workgroup owns each row or row/head. Preserve aliasing,
  scale indexing, epsilon validation, input/scale preservation and typed errors.
  Other storage dtypes retain their existing dispatch and arithmetic; include
  regression controls for them. The parallel F64 fold is not universally bitwise
  identical to the sequential CPU sum across every possible exponent range.
- Decide invalidity from total, independently of Sqrt's special-value result.
  Enforce nonempty positive dimensions and finite positive storage-representable
  epsilon. With Int64-bounded feature count, finite F32 squares cannot overflow
  a F64 sum. Preserve guarded failures for invalid totals and outputs.
- Do not change CPU expectations, quant bytes, weight precision, model architecture,
  routing/GDN/attention semantics, checkpoint behavior or generation acceptance.

## Tests and admission gates

1. Host/emitted-shader contracts: default shaders unchanged; requested 32/64 modes
   and capability present exactly; unsupported widths fail before dispatch;
   ordinary compilation remains usable. Validate generated SPIR-V.
2. Cache identity/negative controls: requests with distinct requirements cannot
   share an incompatible artifact; exercise launch, disk and frozen paths that
   are actually available. Preserve missing-runtime limitations explicitly.
3. Actual-device public RMS: real captured input, seeded 2D/3D, one/multiple rows,
   widths below/above and not divisible by 64, wide finite exponent ranges,
   input/scale preservation, in-place equality, shape stability,
   F32 square-overflow, NaN/Inf and epsilon/shape errors. Verify typed exceptions,
   not merely raw NaN markers. Keep old-red/new-green controls.
4. Replay exact vendor provenance; audit source boundaries; rerun affected Lava
   conformance and the scalar-reduction regression. Keep all rejected candidates.
5. Run the unchanged original 41-check model-history gate. Any regression or
   remaining mismatch rejects this as a sufficient repair. Then complete the
   original broader boundary/fresh-process/recovery/integration/full-suite gates.

## Performance target and receipt

Correctness is mandatory. Record bounded before/after operator runtime, cold
compile, allocations and memory costs on unchanged workloads. No full inference
throughput tuning or throughput claim before correctness. The final receipt must
include exact source/patch identities, numerical deltas, failed controls, hardware,
scope/portability limits and the existing material GEMM/reduction costs.

Expected artifact: reviewable isolated source/test patch and terminal receipts,
either admitted by all Campaign 1 gates or explicitly rejected with evidence.
