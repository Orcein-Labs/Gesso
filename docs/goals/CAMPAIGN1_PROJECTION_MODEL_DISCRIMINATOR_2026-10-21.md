# Campaign 1: test accurate GDN gate projections in the model

## Objective
Determine whether the localized alpha/beta projection accumulation difference changes the four remaining original13 model failures, with the qualified shared gate and convolution diagnostic held fixed.

## Permitted files
New gdn_projection_model_* helpers, providers, preflights, auditors and bounded results under campaign scripts/results; this item and current campaign status/diagnosis/frontier. No maintained or frozen source, supplier, consumed helpers, model or oracle edits. One exact process-local insertion before the fused _gdn_decode_core! call inside _hybrid_linear_block!: recompute only a/b projections on Lava using the existing normed input and same dense weights. Keep the previous exact convolution launch override.

## Interfaces
Same Float32 output/input/weight geometry as dense projection. Use the already captured serial64 dot kernel, WG64, NoContraction and preserve32/64. Devices remain devices. CPU path unchanged. Physical F64 accumulation rounds once to output Float32; this is an explicitly labelled diagnostic algorithm, not the existing Float32 reduction or a portable BLAS-bit emulation.

## Invariants
Frozen original13 cases/modes/repeats/GC/cancellation and CPU oracle, unchanged916source bytes and environments. No token, tolerance, quantization or CPU-reference changes. No CPU tensor fallback, hardware-specific BLAS tree, foreign process termination or speed claim. Under-lock5500MiB admission,20GiB RAM/4GiB swap,1800sec and GPU pause/locks respected; wait if resources unavailable.

## Tests
Prerequisites: maintained shared gate and convolution original13 both independently9/13 with zero integrity/source errors; block22-gate-projection-gpu-1-validation.json (24 real-fixture outputs, exact qualified variants) and exact rational reference (64 serial64 results correctly rounded). Cheap parse/exact-inverse preflight before model launch. Inside the same guarded process, replay both archived32-output fixtures three times before model loading, require exact serial64 reference bits and archive outputs. Bind one actual projection shader with compact-ID-only equality to its qualified shader, both exact source inverses, positive model call count, unchanged source/environment and unchanged shared-gate/convolution programs. Independent original13 audit and effective output/shader mutations required. No source admission from this diagnostic.

## Performance target
N/A: causal numerical discriminator. Record resource/compile observations; no inference-throughput claim. Native a/b projections are intentionally overwritten after the original grouped call, so runtime is not a proposed optimized implementation.

## Expected artifact
Qualified original13 positive or negative that decides whether accurate projection summation merits source integration or whether to trace the next earliest model divergence. Full Campaign1 stress acceptance remains unchanged.

## Receipt
TERMINAL QUALIFIED NUMERICAL NEGATIVE. `gdn-projection-model-1-validation-v2.json` independently confirms9/13 with the same four hello[11,271] vsCPU[11,353] failures, zero integrity/source errors, unchanged916source bytes,600model projection calls and6archived fixture outputs/192scalar comparisons. The actual projection shader is raw-byte-identical to the qualified serial64 shader; all three convolution and three shared-gate programs match their qualified programs after ID-only normalization.

Auditorv1 incorrectly constructed the qualified shader filename without its custom_ prefix; the failed audit is retained and is not a numerical result. Immutable auditorv2 reads the reference descriptor from the hash-bound qualified event summary. No model rerun or capture changes were needed.

`gdn-projection-model-1-mutation-validation-v1.json` rejects both coherent passing-row ID corruption and a structurally valid projection OpFMul-to-OpFAdd mutation with repaired descriptors;34original artifact hashes stayed unchanged. This diagnostic does not resolve the model failure or admit a source change.

Next is the separate existing-general-versus-fused GDN route discriminator. Full Campaign1 source integration, stress and engineering gates remain open; no throughput claim.
