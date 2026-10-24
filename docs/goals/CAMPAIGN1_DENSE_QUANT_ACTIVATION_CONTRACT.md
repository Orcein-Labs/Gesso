# Campaign 1: matched-input dense activation-quantization contract

Status: external diagnostic only; preserve all existing model gates and oracles.
Parent: CAMPAIGN1_LAVA_CORRECTNESS.md.

## Objective and evidence

The fixed-order Float64 projection experiment passes 701 ordinary and 40 sampled
real-weight operator checks, including independent binary replay at zero ULP.
Its unchanged model history/recovery gate remains red at 33/41. This rejects
projection accumulation alone as a sufficient repair.

Source inspection identifies a distinct CPU/Lava contract: resident CPU Q8_0
dense projections quantize each Float32 activation row to Q8_0 before their
integer dot; the current Lava path multiplies unquantized Float32 activations
by decoded weights (and the shipping batch path also casts operands to F16).
Quantization policy and rounding association are separate questions. Measure
the former on actual matched inputs rather than attributing every model
discrepancy to small GPU accumulation errors. This is not yet a causal model
diagnosis or a proposal to change either backend's precision policy.

## Permitted files and interface

New helpers and exclusive receipts in the external Campaign 1 script/result
directories only. This work item and status documentation may change. No
package source, oracle, consumed helper, tensor bytes, importer or installed
runtime edits. Load the frozen ef086ef Gesso through its existing guarded
runner. Exercise the actual public CPU matmul against the same three Q8_0
sampled tensors and 15 saved activation matrices from the real-weight v2 run.
Save CPU output arrays and exact source/fixture identities. No GPU/model load
is needed; mmap only the GGUF index and sampled quant rows.

## Invariants and tests

Keep all existing CPU activation quantization and native integer projection
math. Require finite output, preserved input, all row-count cases, repeated
shared-first-row output equivalence and explicit public operator execution.
Independently quantize each input using native GGML Q8_0 reference, verify the
exact code/scale packing, and replay the CPU scalar Q8_0 block dot arithmetic
against saved CPU outputs. Separately report the difference to the saved
unquantized Float64-dot result. Distinguish quantization error from subsequent
Float32 accumulation; do not assert model causality from an operator result.
The Q4_K head is outside this first bounded Q8_0 contract measurement and must
be reported as an exclusion. A successful measurement is diagnostic evidence,
not replacement model acceptance.

## Performance and artifacts

No speed target or benchmark claim. Record bounded operator wall time and
memory if available, exact helper/source/model-sample hashes, all numeric
deltas and failed attempts. Any later GPU contract experiment requires another
explicit work-item update and the unchanged model gates. Preserve the working
source, rejected precision arm and full original Campaign 1 stress scope.

## Executed host proof and bounded GPU discriminator

The public CPU operator passed all 15 cases. Independent native GGML replay
matches activation packing for 579 rows, and an independent scalar integer
block-dot replay matches every saved CPU output bitwise. Shared first rows are
bitwise invariant across all five M values. The three M=1 fixtures differ from
unquantized products by relative L2 0.00255, 0.00503 and 0.01671. This establishes
the different operator contracts, without proving model causality. See
`dense-quant-cpu-contract-validation-1.json`.

Permit an additional external GPU discriminator for Q8_0 dense weights only.
Keep source ef086ef unchanged. Quantize each Float32 activation row in 32-value
blocks with the existing CPU max/divide/half-scale/ties-away contract. Compute
exact Int32 block dots, then apply the CPU's Float32 scale/product/ascending-block
addition order. The initial source expresses Float64 intermediates rounded to
Float32, but emitted-shader inspection shows LLVM can fold these expressions
back to Float32. They do not by themselves preserve CPU rounding points. Require existing
Float64/float-controls capabilities; this is a diagnostic correctness device,
not a proposed performance implementation. Other quant types and all ordinary
GEMM/RMS/GDN/attention/routing/sampling code remain on their unchanged paths.
Intercept only positive-row Q8_0 public matmul after its existing shape checks.

Require all 15 matched-input GPU outputs bitwise equal to saved public CPU
outputs, actual device activation codes/scales bitwise equal to the executed
CPU pack, finite output/input preservation, exact wrapped kernel dispatch
identity, dumped binary evidence and external SPIR-V validation. Save failures
and reject an insufficient GPU operator before any model run. If these pass,
run the unchanged original 41-check model/recovery gate under this explicit
process-local override. Neither positive result authorizes a production policy
change without separate source, cost, portability and full Campaign 1 gates.

V2 is rejected: all native activation codes match, but 49,403 of 49,408 scales
do not match canonical half precision. Its saved GPU outputs match an independent
fused multiply-add replay using the actual scales exactly; separately rounded
replay differs in 28,087 output elements. V3 stores scales in physical Float16
memory, matching the actual native Q8_0 format. Preserve V2 and V3 separately.

Permit a further external control adding only `NoContraction` decorations to
floating add/subtract/multiply/divide instructions in these two diagnostic
kernels at link time. Preserve every other SPIR-V word and all kernel/dispatch
source arithmetic. Export before/after binaries, record exact target IDs and
prove removal recovers the original binary. Require external spirv-val before
interpreting numerical results. Do not patch unrelated kernels or package files.
This is a rounding-contract control, not a hardware workaround. Vulkan permits
expression rearrangement without that decoration; SignedZeroInfNanPreserve
alone does not prohibit it. See the primary
[Vulkan expression-evaluation rules](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html).

Any production activation-quantization route would be an explicit approximate
lowering. Existing unquantized numerical contracts must not be silently
retargeted to make the internal CPU-ID gate green. Production selection still
requires a declared precision/budget contract, independent reference review,
cost and portability evidence plus the complete original Campaign 1 gates.

V3 confirms the separation: physical half-scale memory matches all 49,408
native scales and every activation code; the remaining 28,200 output differences
match fused arithmetic exactly. V4 adds only seven NoContraction decorations
across the two linked shaders and passes **15/15 GPU operator cases**. Independent
replay verifies all 579 packed activation rows and every saved output bitwise
against the separately replayed CPU results. All original emitted modules and
both actual linked shaders validate externally. Removing the seven decorations
recovers every original SPIR-V word. Receipts:
`cpu-q80-gpu-v3-native-failure-analysis-1.json`,
`q80-no-contraction-offline-preflight-validation-1.json`, and
`cpu-q80-gpu-native-validation-4.json`.

The unchanged 41-check model/recovery gate is terminal, exit 1: **39/41**.
All 36 normal requests, all 24 repeat-ID comparisons, callback/retry and
close/new-state controls pass. Fresh p64 cancellation remains 65017 rather than
82202; recovery remains 71198 rather than 3927. Independent raw-ID validation
reproduces both failures with no source/schema/oracle/integrity defects. Reject
Q8_0 contract matching as a sufficient repair. No package source or CPU oracle
changed; do not create a production precision-policy candidate from this result.
See `history-v2-cpu-q80-no-contract-validation-1.json`.

Permit a bounded fresh-p64 stage comparison under this same external diagnostic
route. Compare an unobserved default CPU control, observed CPU generic prefill
(`GESSO_PREFILL_WINDOW=0`), fresh GPU and repeated fresh GPU. Require CPU controls
82202 and GPU controls 65017 before interpretation. The generic CPU control is
a proxy with the same output, not proof of identical default-window intermediates.
Instrument existing reference/GDN/routing bodies only in memory. Freeze baseline
arrays as copies, bounded to 512 MiB; save at most 16 MiB of first-layer differing
raw pairs. Preserve all source/helper identities. Readbacks alter scheduling;
observed controls cannot replace acceptance.
