# Campaign 1: localize shared-gate subnormal loss

## Objective
Identify the first operation that loses the known CPU subnormal shared-gate result while proving the diagnostic reproduces the unchanged provider endpoint.

## Permitted files
New versioned shared_gate_subnormal_* helpers, auditors, runner and bounded captures under <pystack-work>/scripts/campaign1-20261014 and results/campaign1-20261014; this item, current campaign status/diagnosis and resume-frontier.json. No edits to consumed helpers, frozen Gesso ad47/vendor, or qualified Lava388e6f9.

## Interfaces
Use unchanged shared_gate_wide_provider_v2.jl public apply! and its actual denominator/reciprocal/apply kernels, frozen Gesso CPU _shared_gate_add!, physical intermediate arrays and actual emitted LLVM/SPIR-V. Follow charter first principle/development law and receipt boundaries. This is a diagnostic, not a maintained source repair.

## Invariants
No preferred token, oracle/tolerance changes, host tensor fallback or foreign-process termination. Frozen Gesso912files and env identities remain exact. New observation must match unchanged provider output. Independently separate F64 reciprocal, F64-to-F32 cast/storage, F32 multiplication, and F32 addition. Successful isolated native cast across84,066cases is evidence against assuming casting alone causes loss, not proof for this provider. Keep original19negative.

## Tests
Reproduce fixture19 x=-87.3365478515625,g=1,shared=1,y=0, CPU bits0x007fffe6. Test the original single-row geometry and a boundary batch. Compare unchanged provider with direct unchanged kernels and a staged observer endpoint on3repeats. Archive raw denominator/reciprocal/cast/product/final arrays and actual shaders; independently audit bytes, CPU/reference arithmetic, LLVM/SPIR-V operations, source/helper/env identity, effective repaired-digest intermediate mutation. Capture shaders on failure. Separate harness/resource failure from numeric evidence.

## Performance target
N/A to inference throughput; small model-free probe,256MiB GPU admission, shared locks, bounded1200seconds/6GiB RAM/1GiBswap and16MiBraw capture. No production performance claim.

## Expected artifact
A source-bound, independently qualified first-loss result that chooses the next clean provider experiment. Provider/F32 operation semantics remain unqualified until actually demonstrated.

## Receipt
OPEN. Frozen provider negative and standalone native-cast positive retained. No probe launched at item creation; no Gesso or supplier admission.

## Qualified localization receipt
COMPLETE for this bounded localization, not Campaign1. shared-gate-subnormal-2-validation.json independently qualifies two geometries,6repeated requests,9actual structurally valid shaders and257,355raw bytes. All51observed final scalars reproduce the unchanged provider by both direct unchanged kernels and the observer. There are8subnormal observations per repetition across both geometries (24total): denominator, physicalFloat64 reciprocal, Float32cast, bitcast and plain copy preserve exact CPU bits; Float32multiply by1 and independent Float32add of0 both flush those values tozero. Original fixture19 CPU0x007fffe6 is preserved through cast/copy and first lost at multiplication. Its nativeFloat32-to-Float64 conversion and physicalFloat64multiply/add plusfinalFloat32cast preserve the CPU value. Both wide-control outputs match CPU on all17inputs and3repeats.

Actual shaders prove Float32multiply/add in the failing paths, Float64multiply/add in the successful control, physicalFloat64division in the reciprocal, WG64 and SignedZeroInfNanPreserve32+64 with NoContraction. Separate Base CPU reference reproduces all CPU inputs/denominators/reciprocals/casts/finals. shared-gate-subnormal-2-mutation-validation.json rejects a repaired-digest one-bit reciprocal edit while endpoints remain unchanged, and rejects structurally valid OpFAdd substituted for the isolated OpFMul while output bytes and producer summary remain unchanged.

First launch used a small profile lacking the GPU lock; owned Julia2082894 was terminated before any numerical events. Retained rejected launch profile has exit143 and no numeric authority. Corrected big profile was overridden to6GiB RAM/1GiBswap, verified from live cgroup; both timed/GPU locks held. shared-gate-subnormal-2-resources.json records limits and lock ownership. Corrected run exits0; all jobs terminal. Gesso912source bytes and isolated supplier388e6f9 are unchanged. No maintained source/provider/model admission. Next bounded item must test explicit Float32 operation rounding around wide arithmetic, including subnormal dot inputs/results and cancellation, rather than assume epilogue-only repair is general.
