# Campaign 1: explicit per-kernel NoContraction compiler contract

Opened 2026-10-16. Qualified 2026-10-18 as standalone supplier commit
`289d0693e84a3ad6ae6869cbf81b47dd4bc20ba7`; not Gesso model admission.
See 2026-10-18_no-contraction-receipt.md for exact scopes and limits.
Parent: 2026-10-14_campaign1-lava-correctness.md and OUTPUT_HEAD_CONTRACT.md.

## Objective

Replace the proven external link-time arithmetic decoration experiments with
an opt-in Lava compiler contract that survives compilation and cache reuse.
Start a NEW standalone worktree from Lava 6a3b0e4a; preserve every consumed tree.

## Permitted files

In the new standalone Lava worktree only: src/compiler/target.jl,
src/array/ka_backend.jl, src/compiler/spirv/module.jl,
src/compiler/compilation.jl, src/runtime/launch.jl,
src/runtime/frozen_cache.jl, test/test_no_contraction.jl,
test/test_float_controls.jl, test/runtests.jl and
 docs/NO_CONTRACTION_CONTRACT.md. Plan/status/frontier documents and external
versioned replay/validation artifacts are permitted. No Gesso/vendor/main/libs
edits or production integration in this item.

## Interfaces and invariants

Add Lava.with_no_contraction(KA kernel), composing canonically with existing
with_float_controls in either order. Carry a compile-time Boolean in the
existing callable wrapper, Adapt, compiler parameters/config and cache keys.
Default unwrapped kernel behavior and shader bytes must stay unchanged.

Decorate emitted core OpFAdd, OpFSub, OpFMul and OpFDiv result IDs with
NoContraction before SPIR-V optimization. Verify all remaining covered
operations are decorated after optimization. Explicit source muladd/Fma is
intentional and remains allowed; this contract is not a promise that every
transcendental or other arithmetic operation is exact. Inspect real LLVM and
SPIR-V for contraction-sensitive source witnesses. No context-global policy,
new dependencies or unsupported DenormPreserve execution modes.

Propagate policy through direct compile entry points and launch, disk/frozen
cache identities and artifact checks. A key alone is insufficient: required
arithmetic decorations must be present in loaded shader bytes. Malformed or
incomplete required artifacts must fail or miss explicitly, never be linked
as if conforming. Integer-only shaders may satisfy the arithmetic policy
vacuously; their callable/config/cache identity still distinguishes policy.

Khronos SPIRV-Headers unified1/spirv.hpp was read: NoContraction=42, Restrict=19.
The existing legacy Restrict constant is outside scope; add the correct
NoContraction name without unrelated repair.
Source: https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/main/include/spirv/unified1/spirv.hpp

## Tests

Require wrapper composition/Adapt, stock bytes, config/cache isolation, actual
pre/post optimization coverage, corruption/omission rejection on disk and
frozen cache, public dispatch and explicit-Fma witness. Replay immutable Q8_0
and Q4_K fixtures without link monkeypatching. Run prior float-control tests.
New tests must reject removing an actual required decoration. Record actual
loaded shader provenance, CPU/GPU numeric results, compile/allocation/memory
and operator-cost comparisons. Final Gesso model/stress gates remain separate.

## Performance target and artifact

No speedup target: preserve default path and measure opt-in cost with cold
compilation excluded from warm timing. Expected artifacts: isolated source
commit, focused passing tests, raw shader/numeric/mutation receipts, exact
source diff, measured costs and explicit unresolved portability limits.
Do not push, merge or admit source before these checks.
