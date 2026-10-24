# Campaign 1: correct and stress-tested Lava baseline

Opened 2026-10-14. Status: active; targeted repairs under stress, completion unproven.
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
  `test/test_lava_hybrid.jl`, `test/test_lava_ops.jl`, `test/runtests.jl`.
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

The direct store-contract file remains strict when explicitly invoked by the
acceptance harness. Its upstream Tier 3 registration must run it only when the
device admits cooperative GEMM, otherwise recording a named unsupported-feature
skip. The umbrella runner also targets lavapipe; registering a strict
cooperative-only file unconditionally would break that existing CI path even
when ordinary multiplication correctly selects the scalar kernel. Verify the
registration under both available and unavailable capability states without
weakening the actual-device acceptance preflight or its zero-skip requirement.

No other Lava changes enter this envelope without another explicit work-item
update.

State/partition investigation update, 2026-10-15: permit a demonstrated Gesso
state-resume repair in `src/Inference/generate_state.jl` and focused regression
coverage in `test/test_ctx_checkpoints.jl`, in addition to the inference and
Lava extension files above. Keep the combined `25e7424` candidate frozen while
its queued acceptance jobs consume it; any runtime edit starts in a new isolated
checkout. This extends the investigation envelope, not a diagnosis or permission
to bypass cache reuse, checkpoints, fused kernels, or strict ID checks.

Evidence: forward-order chunk-64 model stress failed five of 53 checks, all
in explicit-state requests; p65 returned 3067 rather than CPU 759, and p127/p256
also had wrong or history-dependent results. Reverse-order chunk-128 stress
failed four of 53 checks, with p65 returning 3067 even for fresh implicit states.
Thus the remaining discrepancy is not exclusive to state reuse. Exact-prefix
extensions do not restore checkpoints. Compare CPU explicit-state execution and
synchronized completed-request state across prompt partitions before selecting
a source repair. Retain both nonzero-exit receipts and all passing controls.

Scalar-reduction envelope update, 2026-10-15: the demonstrated repeatability
defect additionally permits the focused operator regression in the existing
`test/test_lava_ops.jl`, alongside the already permitted `ext/lava_ops.jl`.
Keep this repair in its isolated candidate. Replace the Float32 scalar sum's
native atomic shortcut with the existing fixed-tree reduction; preserve causal
masking, finite checks and typed rejection of all-nonfinite rows. Do not alter
vendored GEMM, quantization or CPU expectations as part of this change.

Evidence: completed-request captures were exact through p64 but first varied
slightly at p65, before later p127/p256 token instability. A stage observer
localized its first differences to attention values after exact QKV/RoPE.
The scalar-sum discriminator observed native atomic variation at lengths 127
and 128, with stable fixed-tree controls. Identical focused tests produced
359 Pass / 125 Fail / zero Error on `d473c3a`, and 484/484 Pass on `5fe6dc1`.
The repaired source's unmodified model-history run passed all 24 repeated-ID
comparisons and 38/41 overall checks. The three remaining failures are p65
3067 rather than CPU 759. This bounds the repair; it does not admit those
failures or graduate Campaign 1. Numeric-state repeat captures, affected
combined-source gates and the numerical/performance receipt remain required.

RMS partition diagnostic envelope, 2026-10-15: permit a separate isolated
candidate in the same two files, generalizing the existing single-row and
single-token/head RMS kernels to one workgroup per row or row/head. Keep their
Float32 lane accumulation and Float64 normalization arithmetic unchanged and
apply it consistently across 2D/3D prompt shapes. Preserve epsilon validation,
scale validation, finite checks and typed failures. Keep weight precision,
quantization, routing, GDN recurrence and CPU expectations unchanged. This is
an experimental candidate, not an admitted repair; require operator numerical
and partition-equivalence tests plus the same model controls before selecting it.

Evidence: repaired scalar reductions make all 24 repeated full numeric states
exact, but the cached p65 history still disagrees with CPU. Existing precision
controls make fresh all-Float32 p65 match CPU in both whole/split execution,
while the all-Float32 cached history still returns 3067. Its first convolution
state differs from fresh all-Float32 by about 1.9e-6. Source inspection confirms
that single-row RMS normalizes in Float64 after a Float32 sum, while multirow
RMS normalizes in Float32 with a different reduction order. The control does
not establish this seam as the complete model cause; preserve negative results
and reject an insufficient variant rather than relaxing the acceptance checks.

Independent-reference diagnostic update, 2026-10-15: retain bounded exact-GGUF
llama.cpp CPU and CUDA/host-expert raw-ID controls outside runtime source.
Six terminal processes keep p64 `82202`, while p65 changes with backend and
physical prompt batch: CPU 128/64 -> `3067`, CPU 1 -> `759`; CUDA 128/64 ->
`759`, CUDA 1 -> `3067`. All prompts were fully evaluated with no cached input,
F32 K/V and flash attention off. Gesso CPU window/per-row controls still return
`759` for p65. These observations constrain reference interpretation; they do
not change the strict acceptance gates or select one native route as universal
truth. Preserve full arguments, source/model/binary identities and raw results.
A synthetic native Q8 activation-quantizer/dot diagnostic may test the CPU port
boundary before selecting another source repair. No quantization, precision
policy or oracle changes are admitted by this diagnostic update.

Prefill RMS axis experiment, 2026-10-15: permit a new isolated candidate based
on working `5fe6dc1`, limited to `ext/lava_ops.jl` and focused coverage in
`test/test_lava_ops.jl`. Test the existing multirow Float32 normalization
arithmetic on one-row **prefill** too. Use the existing AK explicit-axis
reduction with its ordinary output temp and 64-lane block; bypass Lava's
scalar-result dispatch shortcut for that case. Preserve current multirow
reduction selection, all decode arithmetic, epsilon/scale validation, finite
checks, typed failures, default weight/activation quantization, GEMM threshold,
overlap and CPU expectations. Do not admit this candidate from an operator
pass alone. The frozen working and rejected RMS trees remain unchanged.

Evidence: the matched first-token replay reproduced all 13 observed GDN field
deltas by substituting only public row/batch RMS outputs, with exact original
single-row/output/post-state controls. Explicit AK axis reduction plus the
existing Float32 finish reproduces the captured batched result bitwise and
passes seven whole-input multirow controls. Three-dimensional normalization
still changes reduction strategy at output-count/feature-width boundaries;
retain those differences rather than claiming universal partition equality.
The actual cached-prefix FFN trace localizes its first large jump to token3,
layer3: one Q8_K input integer changes from -72 to -71. Native generic and
optimized Q8_K quantizers both reproduce the complete captured structures;
the saved input lies at a rounding threshold, not a demonstrated quantizer
port defect. This supports testing an upstream prefill RMS seam repair,
without changing quantization or selecting a new numerical oracle. Require
focused old-red/new-green controls, unchanged decode controls and all existing
model-history/boundary/stress gates before any final admission.


External Float64 diagnostic envelope, 2026-10-15: after real sampled GPU dense
weight conformance passed 35/35, permit an external helper only (no production
source edits) to test Float64 activations/accumulation over canonical native
Float32 dequantized GGUF values. Override dense quant matmul and routed MoE only
inside that process, streaming at most 1<<20 weight elements per slab. Disable
hybrid CPU scratch inside that process, since it contains forced Float32 buffers.
Preserve architecture, softmax/top-k ordering, quant bytes and importer behavior.
Use serial slab ownership, one BLAS thread, bounded job limits and fresh states.
Verify operator outputs against independent scalar/dense Float64 controls before
model execution, including more than one slab, one/multiple token rows, expert
selection/ties, input preservation and actual dispatch identity. Then limit model
controls to p64/p65 whole/split/cached histories, saving IDs, final logits and
storage/finite/source identities. This is diagnosis, not a production precision
change, independent implementation, arbitrary-precision truth, acceptance pass
or permission to replace the current CPU oracle. Resource costs remain measured
limitations, not performance claims. Preserve all red controls and rejected RMS
candidates. See `float64-diagnostic-feasibility-1.md` in the campaign artifacts.


External matched-CPU RMS discriminator, 2026-10-15: permit a new external helper
only to test the existing CPU RMS arithmetic contract on actual captured and
bounded synthetic Float32 inputs. Square in Float32, accumulate 64-lane partials
and their fold in Float64, normalize/scale in Float64, then store Float32. This
is different from the two rejected RMS candidates, which retained Float32
accumulation. Compare against executed existing CPU RMS, preserving its square
precision; cover 2D/3D and one/multiple leading rows, aliasing, input preservation,
shape consistency and independent numerical/ULP checks. Keep stock and diagnostic
arms explicit, with strict nonzero failure and immutable raw artifacts. No
production-source change is admitted by an operator result. Require another
bounded source envelope and all original model/stress/engineering gates before
selecting any repair. The completed ten-request Float64 diagnostic is evidence
of one stable mathematical execution path, not proof that RMS alone resolves
the production mismatch or permission to substitute that path for production.

RMS/float-controls source-experiment update, 2026-10-15: the external discriminator
now passes finite, shape, alias and invalid-marker controls when its shaders
explicitly require SignedZeroInfNanPreserve 32/64. Adding only those declarations
changes the invalid result from zeros to NaNs; arithmetic/control bytes are
unchanged. A total-based guard additionally passes 24 special-value controls
without depending on GLSL Sqrt special-value semantics. Permit the bounded new
isolated source experiment in [CAMPAIGN1_RMS_FLOAT_CONTROLS.md](CAMPAIGN1_RMS_FLOAT_CONTROLS.md),
including per-kernel compiler requirements, per-device support and cache identity.
This does not admit a runtime repair, change any CPU/model acceptance gate or
permit blanket precision/semantics changes to unrelated kernels.

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
