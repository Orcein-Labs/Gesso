# Campaign 1: preserve decoded quant precision in batched prefill

Status: rejected as a sufficient repair. External operators pass 40/40 and
all 12 deliberate cast mutations fail, but the unchanged original model gate
passes only 33/41: repeated p127/p256 and fresh p64/recovery mismatches remain.
All 24 repeat-ID comparisons are stable. No production source candidate was
created or admitted. Preserve the experiment, receipts and unrun regression/
cost recipes; further diagnosis compares identical cold p64 operands/paths.
Parent: 2026-10-14_campaign1-lava-correctness.md. This is a separate work item from the
RMS preservation experiment and explicitly evaluates the prefill precision policy.

## Evidence and objective

Frozen Gesso ef086ef34eae6a12f222ba0ddec1d254b9622e12 passes all 36 normal
history requests, but fresh p64 cancellation returns 65017 versus CPU 82202,
and recovery returns 71198 versus 3927. Independent validation gives 39/41.
A terminal 20-request precision diagnostic reproduces fresh p64 65017 under
the default dense quant FP16 prefill route, and 82202 under the existing F32
GEMV control. All other 19 requests match CPU. Default/F32 growing histories
have identical hidden/logit hashes, while fresh p64 hidden/logit relative L2
deltas are 0.2621/0.1466. This implicates the dense prefill arithmetic route;
it does not yet distinguish input/weight rounding from accumulation order.
Receipt: cpu-rms-precision-partitions-validation-1.json.

Test batched prefill with Float32 activations and canonical Float32-decoded
GGUF weights. Retain batching, the existing threshold, device placement,
quant bytes, matrix layout, bounded weight-chunk budget and Lava mul! selection.
Do not substitute per-token GEMV or alter CPU IDs. First use a guarded external
process-local override. Production changes require positive operator and
unchanged-model evidence. There is no performance claim from correctness alone.

## Permitted source and invariants

If the external experiment succeeds, create another isolated Gesso candidate
based on ef086ef. Runtime envelope: ext/lava_quant.jl. Focused tests:
test/test_lava_quant.jl. Keep vendored Lava and the RMS implementation unchanged.
Do not change CPU, MoE, routing, GDN, sampler, model metadata, device capability
policy, importer, API, shader compiler or global precision settings.

The precision change is explicit: the old prefill route rounds decoded weights
and input activations to Float16. The experiment uses Float32 for both and
Float32 accumulation/output. The CPU dynamic-quantized projection and GPU GEMM
remain distinct arithmetic implementations; claim no universal bit identity.
Use 4*K bytes per decoded weight column when computing chunk size rather than
2*K. The configured 32 MiB weight chunk budget is unchanged. Record activation
scratch and persistent scratch-cache costs; no unbounded model materialization.
Public unsupported kinds remain unsupported, with explicit errors unchanged.

## Tests and acceptance

- Verify real sampled Q8_0/Q4_K decoding against the native GGML F32 decoder,
  including exact quant/repack bytes. Preserve existing old F16 controls.
- Public matmul: actual Float32 input/weight scratch and correct batched dispatch
  above threshold, original GEMV below it, finite outputs, inputs preserved,
  no unexpected F16 cooperative dispatch. Test dimensions 1/31/32/64/65 and
  both whole output and multiple weight chunks, with ragged tail columns.
- Require a tighter numerical bound against Float64 products of the original
  F32 inputs/native F32 weights. Include nonrepresentable-in-F16 witnesses so
  the test detects reintroduced casts; preserve old-red/new-green control.
- Run the unchanged original 41-check history/cancel/recovery gate using the
  unchanged CPU oracle. Any regression or remaining mismatch means this is
  insufficient. External override results are diagnostic, not source acceptance.
- If selected, reproduce on the committed source and complete all original
  Campaign 1 boundary, fresh-process, integration, full-suite and portability
  gates. No throughput tuning until correctness passes.

## Engineering receipt

Record source/helper identities, raw failed and successful rows, hardware,
operator errors, compile and warm runtime distributions, allocations and host/
device memory. Compare the old FP16 batch path and Float32 batch path on the
same declared workload. Retain material cost regressions. Full-model PyStack
throughput parity belongs to Campaign 3 and remains unclaimed.
