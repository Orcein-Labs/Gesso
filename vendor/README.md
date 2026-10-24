# Vendored dependencies

Gesso depends on upstream packages as ordinary installable dependencies. When
we have to change one, it is vendored here instead. That means the upstream
tree at a recorded rev, with our patches applied, so a Gesso checkout always
carries the code it was tested against.

## Lava

| | |
|---|---|
| Upstream | https://github.com/SimonDanisch/Lava.jl (MIT, see `Lava/LICENSE`) |
| Base rev | `vendor/lava-upstream.txt` |
| Our patches | `vendor/patches/lava/*.patch`, applied in order |
| Tree | `vendor/Lava`, which is the base rev with the patches applied. Never edit it by hand. |

The test and benchmark environments load it through `[sources]`
(`Lava = {path = "../vendor/Lava"}`). Lava is a weak dependency, so code
outside those environments gets the same tree with:

```julia
using Pkg
Pkg.add(url="https://github.com/Orcein-Labs/Gesso", subdir="vendor/Lava")
```

The patches, in order:

1. `0001-ka-backend-drop-the-constadaptor-reshapedarray-overr.patch`. Lava's
   `adapt_structure(::ConstAdaptor, ::ReshapedArray)` overwrote the method
   KernelAbstractions already defines, and Julia refuses method overwrites
   during precompilation. As a result, Lava never precompiled. KA's own rule
   produces the same array, so the override is deleted.
2. `0002-cooperative-matrix-physical-alignment.patch`. Add the `Aligned`
   memory operand required by Vulkan VUID 04708/06314 to physical global
   cooperative-matrix loads/stores, using scalar component alignment.
   Workgroup operands and GEMM arithmetic stay unchanged. Includes a compiler
   regression for Float16/Float32 loads and stores. Exported from isolated Lava
   commit `eef73a9`; runtime and stress acceptance is recorded by Campaign 1.
3. `0003-gemm-shared-accumulator-handoff.patch`. Store each direct cooperative
   GEMM accumulator tile through subgroup-disjoint shared memory, then copy its
   elements to the original output/partial plane. Preserve operand precision,
   accumulator order, split-K and convert-before-epilogue semantics. Select full
   workgroups for uniform barriers, including odd subgroup totals. Retain and
   require the enabled Workgroup explicit-layout feature group for shared GEMM;
   ordinary multiplication uses the existing strided kernel without it, while
   explicit cooperative requests reject before dispatch. Includes numerical,
   poisoned-partial and disabled-feature dispatch regressions. Exported from
   isolated Lava commit `a6a2844`; Campaign 1 tracks source-model/stress acceptance
   separately from the passing operator tests.
4. `0004-gate-direct-gemm-test-registration.patch`. Include the direct GEMM
   accumulator-store device suite only when `Lava.coopmat_gemm_available()` is
   true; otherwise record an explicit skipped test. This makes the targeted
   device suite portable without changing Lava runtime dispatch or kernels.
   Exported from isolated Lava commit `c5a7ec1`; this registration guard does
   not establish source-model or GPU runtime acceptance.
5. `0005-per-kernel-float-controls.patch`. Add opt-in per-kernel
   `SignedZeroInfNanPreserve` requirements with per-device support checks before
   dispatch/pipeline creation. Carry the contract through compiler, launch,
   disk and frozen caches, retaining the original kernel module's build identity.
   Ordinary kernels keep their existing modes. Includes capability, stale-cache,
   argument-packing and numerical regressions. Exported from isolated Lava
   commit `6a3b0e4`; Campaign 1 tracks model and stress admission separately.

6. `0006-compute-no-contraction.patch`. Opt-in compute arithmetic contract, strict SPIR-V validation and cache admission; exported from qualified standalone Lava commit `289d0693e84a3ad6ae6869cbf81b47dd4bc20ba7`. Model and stress admission remain separate.

### Changing a patch or rebasing

- `scripts/vendor_lava.sh` with no argument rebuilds the tree at the recorded
  rev. Afterwards, `git status vendor/` must be clean.
- `scripts/vendor_lava.sh main` (or any upstream rev) rebases. It re-applies
  the series onto that rev and updates the recorded rev.
- If a patch no longer applies, the script stops and leaves its clone in
  place. Resolve the conflict there and re-export the series.
- After a rebase, run the Lava tests before committing.
- A patch upstream has taken drops out of the series. Once the series is
  empty, Lava goes back to being a url dependency, and this directory goes
  with it.

7. `0007-scalar-integer-widths.patch`. Preserve scalar logical integer widths
   when mapping LLVM odd-width integers into SPIR-V native buckets: mask
   unsigned values, extend signed values from the logical sign bit, and retain
   low-bit truncation semantics. Unsupported odd-width vector and instruction
   boundaries fail explicitly. Adds integer-width semantic regressions.
   Exported from standalone Lava commit
   `388e6f95b8db99e16bb238ef6827a3768addbf11`; this compiler fix supports
   integer-backed Float32 rounding but does not establish model or stress
   admission.
