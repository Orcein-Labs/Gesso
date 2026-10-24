# Campaign 1: correct and stress-tested Lava baseline

Opened 2026-10-14. Status: active; no fix or completion claimed.
Parent: [release campaigns](RELEASE_PARITY_2026-10.md), campaign 1.

## Objective

Fix intermittent Ornith GSQ-RCO p64 corruption, integrate the existing W3,
S1 and vendored Lava changes in an isolated candidate, and prove correctness
under repeated fresh processes, changing request shapes and state recovery.

## Permitted files

For L1 diagnosis/fix, narrow the final delta to the demonstrated cause in:

- `ext/lava_ops.jl`, `ext/lava_hybrid.jl`, `ext/lava_quant.jl`.
- `src/Inference/Inference.jl`, `moe_host.jl`, `gguf_quant_matrix.jl`,
  `quant_cpu_multi.jl`.
- `test/test_lava_l1.jl` (new), `test/test_lava_quant.jl`,
  `test/test_lava_hybrid.jl`, `test/runtests.jl`.
- `scripts/stress_ornith_lava.jl` (new) and its usage/receipt documentation.

For integration, the existing W3 delta is `moe_host.jl` and
`test/test_moe_host_prefill.jl`; S1 is `src/Inference/sampling.jl` and
`test/test_sampling.jl`. Vendoring brings `vendor/Lava/`, provenance/patch
files, and the package/test/benchmark source declarations already present
on the vendor branch.

Fix-envelope update, 2026-10-14: permit one additional Lava compiler repair in
a new isolated checkout based on `7a3a58a`, then carry that exact patch into the
vendored candidate. Permitted Lava files are `src/compiler/spirv/coopmat.jl`,
`test/test_coopmat_alignment.jl` (new), and its `test/runtests.jl` registration.
The change adds the required `Aligned` memory operand to physical global
cooperative-matrix loads/stores, using the scalar component alignment. Keep
Workgroup access operands, GEMM arithmetic, blocking, split-K and dispatch
unchanged. Preserve the existing Lava diagnosis checkout as a clean baseline.
Include the patch and its source identity in vendor provenance.

Evidence for this bounded extension: Vulkan VUID 04708/06314 requires the
missing operand; an emitted-shader regression fails all four old load/store
checks for Float16/Float32. In a matched real-block comparison, stock failed
on pair 17 while the alignment-only arm passed all 17 pairs. Removing only
the added memory operands, debug information and incidental IDs makes the
paired shaders byte-identical. One fresh full-generation process passed
three p64 calls; further fresh-process and stress gates remain outstanding.
Subsequent evidence: the third fresh process failed all three p64 calls, with
the alignment operands present in shaders byte-identical to the healthy run.
The isolated combined draft also failed all three calls. Alignment repairs a
confirmed conformance defect but is **not sufficient to repair L1**. Keep L1
open and retain the failed controls. Independent W3/S1 checks on the draft are
preparation; they do not graduate Campaign 1A or the combined acceptance gates.
Second fix-envelope update, 2026-10-14: permit a bounded direct-GEMM accumulator
handoff repair in a **new** isolated Lava checkout based on `eef73a9`, preserving
both original diagnosis and alignment-only trees. Additional files are
`src/array/gemm.jl`, `test/test_gemm_accumulator_store.jl` (new), and the existing
`test/runtests.jl` registration. Carry the exact selected repair and tests through
the vendor patch script and provenance; do not hand-edit the vendored source.

Evidence: a device-verified marker capture proves incomplete producer writes
with correct captured operands. A 128-pair first-block comparison kept the
shared-memory-store/scalar-copy variant correct while stock failed 93 times;
that run captured extra data after stock failures. A fresh repeat removed those
extra captures: the variant passed all 128 pairs while stock failed twice. The
same variant's first full-model process returned `82202` on all three p64 calls,
with 780 guarded direct GEMM dispatches and the emitted store path audited.
These are promising diagnostics, **not** a source or stress acceptance pass and
not proof of an NVIDIA driver cause. The smaller final-MMA barrier control
failed all three requests in its second fresh process; explicit `Aligned 16`
alone also failed all three requests. The shared-store diagnostic passed a
second fresh process, and the generalized source draft passed 441 assertions
across 60 actual-device GEMM cases. Source model/stress acceptance remains open.

Keep original operand precision, accumulator/partition order, blocking, split-K,
bias initialization and convert-before-epilogue semantics. Generalize a selected
barrier/store path for Float16 and Float32 destinations and odd subgroup totals;
all workgroup barriers must be reached uniformly, using a full 32-lane group
where a 64-lane group would be partial. Keep staged GEMM semantics unchanged.
Cover direct BLK1/2/4, split1/8, odd/even KPER, odd/even subgroup counts,
destination rounding, bias/epilogues and partial-plane coverage with actual
device tests. Preserve real-model red controls and every existing acceptance
gate below. Do not add speculative memory flags, vendor-specific behavior,
fallback after runtime errors, or unrelated allocator/capture repairs to this delta.

Portability-envelope update, 2026-10-14: additionally permit
`src/runtime/device.jl` and `test/test_coopmat_gemm_subgroup.jl`. Retain the
enabled Workgroup explicit-layout feature group on each context and require it
when admitting both cooperative and staged scalar GEMM. Explicit cooperative
GEMM requests must reject missing support before dispatch; ordinary matrix
multiplication must use its existing plain strided kernel when the feature is
absent. This is capability routing, not recovery from a failed GPU operation.
Keep raw cooperative-matrix support unchanged. Test the disabled-feature route
on a real device, including exact numerical output, dispatch identity, and
restoration of context state. No vendor-specific exceptions or global-context
queries enter the repair.

No other Lava changes enter this envelope without another explicit work-item
update.

## Interfaces

Use the current `reference_generate`, `GenerateState`, Lava operator dispatch,
quantized matrix and host-expert contracts. The actual Ornith path is the
generation interpreter; a small-model `Session` pass is not equivalent proof.
CPU generation supplies an internal oracle. Independent model-family admission
remains campaign 2.

## Invariants

Lava stays the default. Fix the source rather than forcing single threading,
priming with a one-token call, disabling overlap or selecting CUDA. Preserve
quantization, routing and sampling semantics. Retain the existing host-expert
option and declared resource budget. Keep the inherited probes as local
evidence, then remove their temporary instrumentation from the fix commit.
No production checkout, user configuration or unrelated worktree changes.

## Tests and stress gates

1. Localize L1 with captured inputs/outputs and preserve failed experiments.
   A caught generation error with process exit zero is a failed request.
2. Test the source repair against an isolated regression that fails on the old
   implementation where feasible. Real-model red control remains mandatory.
3. The mandatory repeated gate is at least five fresh default-environment
   processes, two p64 calls each, all token `82202`; hello `[9419]` gives `11`.
   Match these runs on baseline `37d87da` and require a visible semantic failure.
4. Stress beyond that minimum: target 64 p64 requests across 16 fresh candidate
   processes (ten at 16 threads; two each at 1, 4 and 8 threads), four p64 calls
   each, with hello ordering controls. Every process records every request and
   returns nonzero on any unexpected exception or ID mismatch.
5. Exercise prompt lengths around 32-token GEMM and prefill chunk boundaries,
   forward/reverse case order, explicit state reuse, forced collection between
   completed requests, cancellation/resume, callback exception invalidation and
   retry, and close/new-state recovery. Capture CPU expectations before candidate
   comparison. Investigate numerical mismatches explicitly; never excuse a
   nonfinite output or nondeterministic result as a near tie.
6. Retain overlap-off and multi-off runs as controls, separate from the default
   gate. Include the full Lava subset, reconcile the inherited 507 count, and
   require actual device execution rather than import-error skips.
7. On the combined candidate rerun the repeated gate, Lava subset, W3 and S1
   equivalence tests and their effective mutations, then the documented full
   local suite. Stress S1 with the target vocabulary size as well as fixtures.
   Run formatting checks and report pre-existing failures separately.

The full stress matrix is a target, not a claim it has run. Record exclusions,
failures, timeouts and missing references; do not infer campaign completion from
one green subset. Heavy jobs remain serialized under the handoff lock, with at
most 16 Julia threads, bounded native-library threading and the 4800 MiB GPU
budget. Preserve raw logs and exact source/environment identities.

## Performance target

No speedup target in this campaign. Record warmed runtime and memory before
and after the repair on the same workload. Investigate material regressions;
do not trade correctness for timing. Compilation and diagnostic synchronization
are separate from steady execution measurements.

## Expected artifacts and receipt

Targeted source fix and regression, reviewable integrated branch/diff, strict
stress driver, complete result ledger and §LXXII receipt with numerical,
timing, compilation, memory, workload, hardware and backend fields. The candidate
must load the vendored Lava source explicitly. Local commits only; coupon push
and production history cutover remain Matt's actions.
