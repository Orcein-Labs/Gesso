# Campaign 1: shape-stable projection accumulation discriminator

Status: rejected as a sufficient repair; no production source change or admission.
Parent: 2026-10-14_campaign1-lava-correctness.md. Preserve all previous candidates.

## Evidence and question

The rejected F32 batch experiment has accurate sampled operators but only 33/41
unchanged model checks. The cold p64 observer reproduces GEMV82202, batch46108,
GEMV82202, with every repeated stage/routing/selected-layer expert capture exact.
The first difference is layer1 GDN QKV projection, maximum absolute4.9591e-5;
input RMS and the two dense gate projections are exact. At layer1/token2,
Q8_K activation element1853 crosses from7.49999666 to7.50001717 and rounds7 to8.
The first expert-set change is layer3/token13. Native GGML Q8_K reference replay
matches all2302 quantizations from1151 saved raw pairs, including every observed
scale, integer and folded-sum change. The quantizer is reproducing its canonical
rule; it amplifies upstream differences. Receipts:
cold-p64-precision-observer-validation-2.json,
cold-p64-native-q8-replay-validation-1.json.

Test the question: do fixed-order F64 dot products over unchanged F32 operands,
rounded once to F32 output, eliminate shape-dependent projection differences
and improve the unchanged model gates? This is a precision discriminator, not
a throughput optimization or selected release policy.

## External scope and invariants

No package file is edited. Use guarded process-local helpers in the external
campaign script directory. Base is Gessoef086ef and its exact vendored Lava.
Retain quant bytes/native F32 decode, F32 activation/state/output storage,
RMS source, GDN/attention equations, host expert activation Q8 quantization,
router, sampler, cached-state contract and unchanged CPU IDs.

Derive a uniquely named diagnostic kernel from Lava's existing strided GEMM,
changing only the accumulator to Float64 and opting it into preservation32/64.
Override only Float32-input/output gemmlaunch calls; all other dtypes retain
the existing dispatch body. This applies the same fixed K traversal at every
row/batch size. For dense quant projections use canonical decoded F32 weights
and that same kernel at every token count, including below the old threshold.
This route change is explicit and diagnostic; it is not a hidden shipping flag.
Keep decoded weight chunks bounded in bytes. Require actual F64 capability and
float-control properties rather than silently substituting another kernel.

The F64 sum is not a proof of universally exact real dot products or bitwise
identity to every CPU summation order. It is a strong numerical discriminator
for the actual F32 fixtures. F64 device throughput can be materially lower;
no release-performance claim follows even if the model checks pass.

## Gates

1. Host source/transform guards and capability checks. Preserve old source hashes.
2. Actual-device ordinary and real quant matmul: independent host F64 reference,
   F32 output, one/multiple/ragged rows, transpose/strided operands, nonzero
   alpha/beta, beta0 with poisoned output, input preservation, and shape
   invariance across whole/single-row/partitioned calls. Retain other-dtype
   controls. Independently validate actual generated SPIR-V preservation modes.
3. Run the unchanged original41 history/recovery gate. Any remaining mismatch
   rejects this as sufficient; operator passes never replace model acceptance.
4. A positive external result would require a separately bounded production
   work item, source implementation, cost/portability review, all original C1
   boundary/fresh-process/recovery/integration/full-suite and stress gates.

Artifact: terminal raw diagnostics, source/helper/shader hashes, explicit failed
controls and resume frontier. No source commit, merge, push or precision-policy
cutover is authorized by this diagnostic's positive result alone.

## Executed diagnostic gates

Host transformation guards pass and recover the frozen original bodies
byte-for-byte after reverting only the declared changes. Ordinary actual-device
checks pass 701/701; the baseline arm fails all three cancellation discriminators.
All 16 ordinary SPIR-V modules validate externally. Six whole/single-row/partition
cases agree bitwise, with zero ULP against rounded host Float64 products.

Real sampled-weight v2 passes 40/40 on four Q8_0/Q4_K tensors. Independent
GGUF-index/native-GGML replay checks 665,600 decoded values, and binary-array
replay checks all 20 products at zero ULP. Four shared-first-row groups are
bitwise invariant across M=1/31/32/64/65. All 16 real-weight SPIR-V modules
validate externally. The first real-weight arm's logger hid the wrapped kernel
name and failed 20 route checks; v2 exposes actual callable origin/mask only,
without arithmetic changes. Both raw runs are preserved. Actual Q5_K/Q6_K/IQ4_XS
specimens remain missing.

The unchanged model-history/recovery gate is terminal with exit 1: **33/41**.
All 24 repeated-ID comparisons are stable. In all three normal repetitions,
p64 returns 65017 rather than CPU 82202 and p256 returns 69383 rather than 8924.
Cancellation returns 65017 rather than 82202; recovery returns 71198 rather than
3927. The external raw-ID validator reproduces all eight failures with no
schema/source/oracle/row-integrity errors. This rejects the experiment as a
sufficient repair. It does not justify a source precision-policy change or
operator performance work on this arm. Preserve the complete raw run and
`history-v2-shape-stable-projection-validation-1.json`.

Receipts in the external campaign result directory:
`shape-stable-projection-source-transform-validation-1.json`,
`shape-stable-projection-ops-validation-1.json`,
`shape-stable-realquant-native-validation-2.json`, and
`shape-stable-realquant-shader-validation-2.json`.
