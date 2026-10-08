# Regime I maturity campaign — 2026-10-07

Owner: Codex. Layer: Gesso mechanism. Authorization: sequential reliability
campaign requested by Matt. Palette and Cyan remain frozen infrastructure.

## Isolation and sequencing

The working package is an isolated copy of the surviving Gesso shadow at
`caa68e218896761c3f049c5cd702ba95a01ac17f`, including its pre-existing
working files. Baseline hashes and original diffs are preserved outside the
package. Campaign patches compare against that preserved baseline; unrelated
changes are not part of the patch. Neither original checkout is edited.

Run the arms in this order. A skip is not a completion gate. Failure preempts
the active arm. No later arm may graduate before all preceding arms pass.
No exotic KV, representation, speculative execution, or tuning work.

| Arm | Completion gate |
| --- | --- |
| 1 Correct inference | Real local SmolLM2 checkpoint; independent eager Hugging Face/PyTorch CPU reference; tokenizer IDs and eight greedy tokens exact, all last-position vocabulary logits within declared absolute tolerance on three prompts; engine and interpreter both tested; corrupted oracle must fail. |
| 2 Loading | Supported safetensors single/sharded inputs and architecture/tokenizer paths preserve values and identities; malformed metadata, missing/extra tensors, unsupported dtypes/features fail explicitly. GGUF is not currently supported; never claim it. |
| 3 Reproducibility | Greedy runs repeat across sessions, fresh processes, thread counts and BLAS settings with pinned checkpoint and environment hashes; no claim of sampling support. |
| 4 Memory lifecycle | Repeated real-model generate/reset/reload yields identical IDs; workspace reused; page accounting consistent; context exhaustion typed, recovery correct; retained memory measured after GC, not process RSS alone. |
| 5 Devices | CPU is usable; NVIDIA model/engine/operators actually execute and match the oracle; explicit unavailable-device and unsupported-lowering errors. AMD/Vulkan only graduates on available hardware. No CPU substitution for an unavailable device. |
| 6 Numerics | Supported arithmetic paths match an independent reference under declared dtype tolerances; nonfinite/normalization edge behavior explicit. Unsupported FP16/BF16 execution must be named, never silently relabeled. |
| 7 Context | Causal masks, RoPE/scaling, cached vs full-prefix logits, page boundaries and context exhaustion agree with independent reference. No exotic cache policy. |
| 8 Scheduling | Ordinary independent sequences and supported concurrency, variable lengths and cancellation have executable conformance gates. Missing scheduling is a finding, not completion. |
| 9 Containment | Malformed model/tensors, operation/callback interruption and partial output yield typed failures/receipts and demonstrated recovery. |
| 10 Observability | Actual engine receipts faithfully report timings, tokens, backend, cache bytes and errors; compilation/warmed measurements separate. |
| 11 Performance floor | Same checkpoint, prompts, arithmetic, workload and hardware: warmed end-to-end timing and allocations against eager reference; investigate avoidable overhead with before/after receipts. |
| 12 Packaging | Fresh writable depot/environment installs the declared package and performs real local inference with a reproducible receipt and explicit optional-device status. |

## Arm 1 closed work item

Objective: establish independent real-model correctness evidence for the current
CPU engine, preserving the existing Gesso regression golden.

Permitted files: this document; `test/reference_hf.py`;
`test/test_hf_parity.jl`; `test/runtests.jl`.
Further fixes require an explicit bounded item added here before editing.

Interfaces: local `config.json`/safetensors/tokenizer checkpoint;
`load_llama`, `load_gpt2_tokenizer`, `reference_prefill`, `Session`, `prefill!`,
`generate`; `GESSO_HF_REFERENCE` supplies a JSON receipt generated externally.
PyTorch/Transformers remain outside Julia package dependencies.

Invariants: no network, no regenerated self-golden, no checkpoint mutation,
CPU Float64 Gesso vs independent eager CPU Float32 PyTorch. Logits
`atol=1e-2`, `rtol=0`; tokenizer/greedy IDs exact. Tolerance set before results.

Tests: three prompts (`Hello`, `The quick brown fox`,
`Julia is a programming language.`), eight greedy output tokens each;
unavailable reference gives a named CI skip but cannot graduate this campaign;
provided invalid reference fails; corruption of one logit must fail.

Performance target: N/A for correctness. No performance claims from these runs.
Expected artifact: reproducible external oracle generator, Julia conformance
test and machine-readable receipt with model/environment hashes.

Receipt: pending measurement. Runtime code unchanged at kickoff; compile-time
and memory impact not measured, no performance claim.

## Arm 1 repair item — checkpoint layout

Observation: independent parity failed 9 assertions across all three prompts;
a 2×3 externally written tensor `[1 2 3; 4 5 6]` loaded as
`[1 3 5; 2 4 6]`. Reader and local fixture writer shared the same layout error.
The safetensors format explicitly requires C/row-major order:
https://github.com/safetensors/safetensors#format (read 2026-10-07).

Objective: preserve checkpoint tensor coordinates when materializing Julia arrays.
Additional permitted files: `src/Inference/llama_import.jl`,
`test/test_import_llama.jl`, `test/test_safetensors_layout.jl`,
`test/fixtures/smollm2/expected_logits.toml` (deliberate refresh only after
independent parity passes), `docs/ARCHITECTURE.md` (importer map correction).
`test/Project.toml` may declare SHA (stdlib) for checkpoint identity validation
in the external oracle gate; this does not add a core dependency.
Interfaces: `load_safetensors`; all four existing dtypes, arbitrary tensor rank.
Invariants: Float64 exact upcast, same tensor shapes/identities, explicit
unsupported dtype refusal, toy oracle unchanged. No new model or backend.
Tests: independently constructed raw-byte matrix, higher-rank, scalar and
empty shapes; existing import round-trips; three-prompt real-model HF parity;
old layout mutation must fail; refreshed golden must match independently
validated corrected CPU behavior.
Performance target: no speed claim; correctness first. Keep dtype dispatch
outside the element loop to avoid boxing per checkpoint value.
Expected artifact: small reader/writer patch, independent tests, new golden
with preserved old golden in the baseline and external reference provenance.
The full suite exposed the same column-order fixture error in the separate
Phi-3 writer. `test/test_breadth0.jl` is additionally permitted to delegate
that writer to the corrected C-order fixture writer; its value assertions
remain unchanged.

## Arm 1 repair item — positional layout

Observation after the checkpoint fix: `Hello` (position zero) max logit error
4.87e-5; longer prompt errors 24.71 and 19.35. CPU RoPE rotates adjacent
features; HF Llama's `rotate_half` couples first and second head halves.
Source read: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

Objective: honor imported Llama's existing `interleaved=false` positional
meaning through interpreter, Session and explicit device transfer.
Additional permitted files: `src/Operators/cpu.jl`,
`src/Inference/architecture_spec.jl`, `src/Inference/Inference.jl`,
`src/Inference/session.jl`, `ext/cuda_ops.jl`, `ext/lava_ops.jl`,
`test/test_rope_layout.jl`.
Invariants: standalone operator/legacy toy defaults retain adjacent-pair
behavior; imported tensors retain their literal weights and positional
policy; no public exported name added; scaled/partial/interleaved unsupported
policies keep their explicit failure. GPU arithmetic is not certified without
an actual device run; Arm 5 remains mandatory.
Tests: independent half-split formula for Q and GQA K; imported layout survives
prefill/decode; real HF parity and legacy toy golden both pass; metadata
survives transfer in existing device tests when hardware is available.
Performance target: N/A. No new kernel strategy, only coordinate selection.

## Arm 1 repair item — independent attention heads

Observation: layout-corrected longer prompts still fail. All interpreter and
Session paths sum QK scores across heads before one shared softmax. Standard
MHA/GQA requires one softmax per query head. The existing Phase 2 contract
already says scores are `(seq_q, seq_k) per head`; the implementation and
test-side formula both violated that contract. Position zero hid the defect
because a one-key softmax is always one.

Objective: compute each head's probabilities independently in prefill/decode.
Additional permitted files: `test/test_attention_heads.jl`, `test/test_gqa.jl`,
`test/fixtures/toy/expected_logits.toml` (deliberate semantic repair refresh).
Previously allowed interpreter/Session files may replace their repeated
aggregate contraction with one private helper reusing the 2D score scratch
one head at a time. No new attention kernel or public surface.
Invariants: pages/GQA mapping/cancellation/receipt ownership unchanged;
independent-head semantics apply to every model including toy2. The old toy
golden encoded the defective attention and cannot truthfully be preserved.
Tests: adversarial opposing-head scores must retain different probabilities;
changing one head cannot affect another; independent HF parity, repeat-first
test formula corrected to the specified per-head contract; regenerate both
self-goldens only after independent gates pass.
Performance target: N/A; former flat cross-head GEMM was mathematically invalid.
Per-head device GEMM is ordinary correct execution; its performance waits for
Arm 11 and real device validation waits for Arm 5.

## Arm 1 completion — 2026-10-07

Independent parity: 45 pass, zero fail; max absolute vocabulary-logit errors
4.87e-5, 4.11e-5, 2.44e-5 under predeclared 1e-2. Both interpreter and Session
produce all eight reference tokens exactly on all three prompts. Corrupted
oracle fails two assertions and exits 1; old transport mutation fails two
assertions and exits 1. CPU full suite: 2607 pass, 20 named skips/broken,
zero errors/failures. Formatting gate passed. Optional devices remain Arm 5.
External receipts preserve hashes and before/after errors. Arm 1 graduates.

## Arm 2 closed work item — strict transport validation

Objective: reject ambiguous or malformed supported checkpoint inputs before
allocation/materialization; preserve exact valid values and shard identity.
Permitted files: this document, `src/Inference/llama_import.jl`,
`src/Inference/architecture_spec.jl`, `src/Inference/arch_adapters.jl`,
`src/Inference/gpt2_tokenizer.jl`, `test/test_loading_boundaries.jl`,
`test/runtests.jl`, `docs/ARCHITECTURE.md`.
Interfaces: existing config/tokenizer/safetensors readers; no new export,
format, dependency or implicit conversion. Strict JSON object uses existing
JSON parser with a private rejecting dictionary; JSON syntax remains JSON's.
Invariants: all four transport dtypes preserve bits under Float64 widening;
scalar/empty tensors valid; offsets cover payload exactly; header bounded by
actual file; shard name is local and index assignment exact. Dtype NaN/Inf
values are format-valid, though execution numerics have a separate gate.
Tests: independently assembled valid/malformed headers; duplicate nested keys,
overlap/hole/trailing/truncated/overflow/noninteger dimensions; malformed
metadata; exact shard assignment; config positivity/finite numeric policy;
unique contiguous tokenizer IDs and duplicate merge rejection; existing
round-trip tests and real checkpoint parity. No performance claim.

## Arm 3 closed work item — deterministic replay

Objective: prove supported greedy CPU behavior across fresh processes and
thread/BLAS configurations against the pinned independent oracle.
Permitted files: this document, `test/regime_i_probe.jl`; receipt files outside
package. No runtime edits. Tests: all three prompts × three fresh Sessions,
three fresh processes (Julia/BLAS 1/1, 2/2, 4/1), exact eight-token IDs,
checkpoint SHA256 identities, version and BLAS stamps. Sampling is not
implemented or certified. No speed claim; elapsed run includes compilation.

Arm 2 completion: 40 strict boundary checks pass; all prior supported loader,
architecture and tokenizer fixtures pass; independent real-model parity
passes unchanged. CPU full suite 2647 pass, zero fail/error, 20 named
skips/broken. Format and diff whitespace checks pass. Only SmolLM2 has real
checkpoint evidence; other architecture fixtures establish import transport,
not a claim of external real-model numerical parity. Arm 2 graduates.

## Arm 4 closed work item — bounded lifecycle and device scratch repair

Objective: prove reset/unload/reload ownership on the real checkpoint and fix
an observed ordinary CUDA scratch regression without raising its allocation
ceiling. The host suite executed CUDA/Vulkan correctly but measured CUDA
SmolLM2 decode at 6,043,584 B against the declared 1,048,576 B ceiling.
Permitted files: this document, `test/regime_i_probe.jl`,
`src/Inference/Inference.jl`, `ext/GessoCUDAExt.jl`, `ext/cuda_ops.jl`,
`docs/ARCHITECTURE.md`; existing tests stay at their declared limits.
Interfaces: existing per-head contraction, Session/reset, manager page/byte
accounting. Device methods may use ordinary in-place GEMM on strided head
views instead of allocating device slices and products. No new kernel,
representation, sharing policy, public type or dependency; CPU reference
order unchanged. Tests: repeated generate/reset and reload exact IDs;
workspace identity preserved; page/byte accounting exact; exhausted context
is ERR_RESOURCE_LIMIT and subsequent reset recovers; retained live bytes after
warmup/GC bounded; actual host CUDA allocation under existing 1 MiB gate plus
independent CPU/HF parity. Before/after allocations recorded separately from
compile times. No throughput tuning or exotic cache work.

Arm 3 completion: each fresh process passed 19 assertions (57 total).
Julia/BLAS thread counts 1/1, 2/2, 4/1 each replayed all three prompts three
times against the independent eight-token oracle; exact IDs and checkpoint
hashes matched. Three environment-stamped JSON receipts preserved. Arm 3
graduates. Arm 4 additionally permits `test/regime_i_cuda.jl` for the actual
hardware allocation/parity probe, without importing CUDA into core.

Arm 4 measured refinements: views reduced allocations to 2,613,152 B;
direct ordinary cuBLAS GEMM reduced them to 1,567,712 B, with all three
independent prompt/token gates still passing. Neither meets 1 MiB. The
allocation profile also attributes 297,680 B to RMSNorm and 209,520 B to
softmax through `storage::Any`. Private storage-argument function barriers
are permitted in those existing CUDA operators and matmul, preserving their
arithmetic statements and existing kernels. This is the same specialization
boundary used by the CPU engine, without changing any semantic type fields.

## Arm 5 closed work item — actual device and storage boundaries

Objective: establish actual CPU/CUDA and available Vulkan execution with the
external checkpoint/oracle, and reject cross-backend storage explicitly.
Permitted files: this document, `src/Inference/Inference.jl`,
`ext/GessoCUDAExt.jl`, `ext/GessoLavaExt.jl`, `ext/cuda_ops.jl`,
`ext/lava_ops.jl`, `test/regime_i_cuda.jl`, `test/regime_i_lava.jl`,
`test/test_device_boundaries.jl`, `test/runtests.jl`, `docs/ARCHITECTURE.md`.
Interfaces: Session/reference constructors and existing operator guards;
private parent-storage inspection, no new public surface or dependency.
Tests: CPU first-class; actual NVIDIA engine and interpreter vs independent
HF Float32 oracle (atol 1e-2, eight tokens exact); existing CUDA operator
gates; available Vulkan backend vs real checkpoint/reference; host views,
other backend arrays, unsupported arithmetic and unavailable-device paths
throw typed failures rather than substituting CPU. Runtime device stamps
record actual hardware. AMD absent on this host, so no AMD certification.

Arm 4 allocation gate remains open at 1,476,112 B after specialization.
The measured RMSNorm library reduction still allocates temporary device
arrays for each single-row decode. Bounded extension: a plain one-row,
Float32 serial reduction inside the existing CUDA RMSNorm operator is now
permitted, retaining the existing apply expression and kernel convention.
This is ordinary normalization, not a new inference representation or exotic
kernel strategy. Direct standard cuBLAS GEMM for projections and GEMV for
single-query attention may replace the generic wrapper. All declared parity
and allocation limits remain fixed; prefill multi-row normalization stays on
the existing reduction. Failed experiments remain separate receipts.

## Backend priority correction — direct user steering, 2026-10-07

Lava is the PRIMARY Gesso execution backend. CUDA is a throughput comparison
backend. The earlier interpretation making CUDA's historical allocation
ceiling a mandatory campaign blocker is superseded by Matt's clarification.
The existing CUDA test stays unchanged and its result stays visible, but it
cannot replace or hold up the Lava acceptance gate. No further CUDA-only
optimization campaign. Retain only validated ordinary comparison repairs;
failed experiments remain receipts, not maturity claims.

Arm 4 now requires the same real-checkpoint lifecycle, scratch ownership,
cache accounting, bounded retained memory and recovery on actual Lava/Vulkan.
Arm 5 requires actual Lava device/operator/engine/reference correctness and
explicit storage/device failures. NVIDIA CUDA correctness is evidence needed
for a trustworthy comparison, rather than the main product acceptance path.
Arm 11 compares primary Lava to Gesso CUDA and independent eager PyTorch CUDA
with the same model, Float32 arithmetic, prompts, output IDs and hardware.
AMD hardware is unavailable on this host; Vulkan on the available NVIDIA
hardware is the verified device identity, not an AMD claim.
The table's backend interpretation is amended accordingly.
Arm 4 additionally permits `test/regime_i_lava.jl` for primary acceptance.

Arm 4 primary run: all real-model parity, tokens, reset/reload/recovery and
page accounting assertions passed; one retained-size assertion failed
(1,641,896 B > 1 MiB). Inspection establishes that `summarysize(Session)`
traverses Lava buffer `last_write`/pool references into the SHARED Vulkan
queue and context caches, so that number is not Session-owned memory.
Refine the measurement (not the threshold): count unique Session-owned
logical GPU arrays and page buffers, and exclude shared Vulkan context,
BatchQueue and PoolBlock from the Julia Session heap traversal. Preserve the
original failed receipt. Also measure whole-process Julia live bytes after
full GC and real device reserved bytes via `gpu_live_bytes` after the
library's existing `trim_gpu_pool!` cleanup (which quiesces the queue).
Owned GPU bytes must be unchanged; owned Julia growth remains <1 MiB;
whole Julia live growth remains <20 MiB; trimmed device reservation growth
must stay within one declared 64 MiB pool-block margin. Repeated hot batches
and scope-end reload cleanup are required. No allocator/source workaround.

Arm 4 completion: CPU lifecycle 67/67; primary Lava lifecycle 40/40.
Lava Session-owned GPU bytes remain exactly 539,395,328; owned Julia growth
7,472 B; whole Julia growth 2,581,960 B; trimmed device growth 0 B.
Three external full-vocabulary logits gates and exact eight-token sequences
passed. Exhaustion is typed, reset and reload recover. Arm 4 graduates;
CUDA historical ceiling remains visible as auxiliary comparison evidence.
Arm 5 probe inventory additionally permits `test/regime_i_device.jl`.
The host probe combines existing CPU/Lava operator tests, explicit host-view,
wrong-device and unsupported-dtype rejection, and independent real-model
interpreter parity. CPU requires its established Float64 materialization;
Lava requires LavaArray Float32, including legitimate device views.

## Arm 6 closed work item — finite supported arithmetic

Objective: independently validate native CPU Float64 and primary Lava
Float32 normalization, large stable scores, zero/small/large activations,
SwiGLU and projections; stop nonfinite logits from becoming ordinary tokens.
Permitted files: this document, `src/Inference/Inference.jl`,
`src/Inference/session.jl`, `src/Operators/cpu.jl`, `ext/lava_ops.jl`,
`test/reference_numerics.py`, `test/regime_i_numerics.jl`,
`test/test_numeric_failures.jl`, `test/runtests.jl`, `docs/ARCHITECTURE.md`.
Interfaces: existing operators and engine; private finite-logit validation,
existing ERR_NUMERICAL_INSTABILITY/ERR_INVALID_PLAN. No new types or deps.
Oracle: independently executed PyTorch functional RMSNorm, SiLU, softmax,
and matrix multiply in native float64/float32. Before-result tolerances:
CPU atol 1e-10/rtol 1e-12; Lava atol 1e-3/rtol 2e-5. Stable softmax at
1e6 scores, RMSNorm zeros/1e-9/1/1e10; native-overflow and nonfinite rows
must fail explicitly, invalid eps must fail, NaN/Inf logits cannot emit IDs.
FP16/BF16 safetensors transport widens exactly; native execution is
unsupported and explicitly rejected by Arm 5, never advertised as native.
Full real-model parity remains at atol 1e-2 and exact output IDs.
Arm 5 absence probe additionally permits `test/regime_i_unavailable.jl`.
Restricted-runtime import failed in GLFW before reaching Gesso. Preserve
that log as harness/environment evidence; it is not a typed Gesso absence
proof. A scoped missing Vulkan ICD environment on the usable host exercises
the constructor's genuine absence branch without changing host settings.

Arm 5 completion: 72 CPU/Lava operator, storage and independent real-model
interpreter assertions plus 3 actual unavailable-driver assertions passed.
All three last-vocabulary-logit deltas remain <= 0.0002251, with eight new
IDs exact. Arm 4 already proves the primary engine path. Host views,
wrong devices and unsupported native dtypes fail with ERR_INVALID_PLAN.
AMD is absent and unclaimed. Arm 5 graduates; next is numerical reliability.

## Arm 7 closed work item — ordinary context/positional correctness

Objective: verify default metadata, linear and Llama3 scaling against the
installed independent HF rotary implementation, primary Lava included;
verify full-prefix vs cached last logits across ordinary page boundaries.
Permitted files: this document, `src/Inference/architecture_spec.jl`,
`src/Inference/Inference.jl`, `src/Inference/session.jl`,
`src/Operators/cpu.jl`, `ext/lava_ops.jl`, `ext/GessoLavaExt.jl`,
`test/reference_context.py`, `test/regime_i_context.jl`,
`test/test_breadth0.jl`, `test/test_rope_metadata.jl`,
`test/runtests.jl`, `docs/ARCHITECTURE.md`.
Interfaces: existing positional policy/rope!, Session/reference defaults;
private theta resolution; no new representation, dependency or KV policy.
Independent source evidence: pinned Transformers 5.18.0 linear scaling
computes the base inverse frequency divided by factor. Gesso instead uses
(theta/factor)^(-2i/d), a self-mirrored defect. Correct that expression and
its old fixture expectation only after the external frequency test fails.
Imported unscaled theta is also ignored unless callers explicitly pass it;
defaults must read metadata while explicit overrides remain authoritative.
Lava may use its existing uploaded angle-frequency table for ordinary
scaled policies; capabilities change only with executable conformance.
Before-result tolerances: frequencies atol 1e-8/rtol 1e-6; native rotated
values atol 1e-3/rtol 1e-5 at positions 0,1,8191,8192,32768; real last logits
atol 1e-2 and next token exact. Prefixes 17,33,65,129 straddle 16-token
pages; full and cached prefixes both match external HF. Causal future-token
mutation preserves earlier positions; cache counts and typed exhaustion
stay exact. This proves the tested model lengths and high-position rotary
operator semantics, not full-model execution at 8192 or extrapolation quality.

Arm 6 primary negative gate: 25 passed, 7 failed. A minimal actual-device
probe confirms `all(isfinite, LavaArray([0, NaN/Inf/-Inf, 1]))` returns true
and ordinary IDs are emitted. This is not an exception-class mismatch.
Separate mapped-predicate and Boolean/integer-reduction probes are required
before repair. If the defect is isolated to the device Boolean reduction,
a private storage-dispatched finite predicate using ordinary integer
map/reduce is permitted in `src/Inference/session.jl` and `ext/lava_ops.jl`.
It must validate on-device and return a scalar decision; no full-logit
CPU transfer, arithmetic fallback, custom performance kernel or library
source edit. All seven negative cases and independent parity stay required.
Also permit format-only changes to the campaign-owned probes introduced by
Arms 3–5 and `test/test_device_boundaries.jl`, as required by `make format`.
These are campaign files, not carried shadow changes.
Arm 6 regression inventory also permits `test/test_numeric_lava.jl`,
included after the existing Lava seam. It exercises poisoned tiny-model
prefill through the real engine and asserts typed numerical failure;
no external checkpoint is needed for that permanent regression.
Arm 6 diagnosis correction: Boolean reduction itself works (true/false/true
reduces false). The GPU `isfinite` predicate maps ALL THREE nonfinite inputs
to true; integer reduction using that predicate also returns zero. An IEEE
exponent-bit classifier on the same real device returns one correctly.
The permitted repair is therefore a private storage-dispatched finite check
using IEEE Float32/Float64 exponent masks and ordinary UInt32 maximum
reduction. Float64 is covered for existing internal normalization temporaries;
this does not advertise native Float64 model execution on Lava. CPU/CUDA
retain their existing predicate; no Lava library edits or transfer fallback.

Arm 6 completion: native CPU and Lava references each passed 47 assertions,
including first/last poisoned elements across reduction sizes 3/64/65/129/1025.
The permanent Lava poisoned-model engine regression passed 15 assertions
(including fixture checks). Real checkpoint replay remained exact on all
three prompts. Full CPU suite 2656 passed, 21 named skips/expected broken,
zero failures; required formatting and diff checks passed. The original
25-pass/7-fail run and both independent primitive diagnostics are retained.
No native FP16/BF16 claim, no underlying library or infrastructure edit.
Arm 6 graduates; immediately proceed to the contextual/positional gate.

Arm 6 boundary follow-up, before any Arm 7 implementation: the CPU engine
rejects half storage, but a direct Float16 `rmsnorm!` probe succeeds and
returns Float16. The earlier unsupported-native statement was broader than
that evidence. Reopen the dtype boundary until all six CPU operator entries
reject unsupported arithmetic consistently. Additional permitted files:
`src/backends.jl` and `test/test_numeric_failures.jl`. Move the identical
private parent-storage walk to the backend seam so CPU operators do not
depend on the higher Inference layer; retain Inference's private forwarding
name for extension compatibility. Guard only materialized storage here;
existing unset-storage behavior stays for the containment arm. No public
exports, semantic type changes or dependency. Negative probes cover all six
ops with Float16 arrays and both workloads; native Float64 stays unchanged.

Arm 6 dtype boundary reclosed: all twelve direct CPU half-precision
operator/workload probes now raise ERR_INVALID_PLAN. Full CPU suite 2668
passed, 21 named skips/expected broken, zero failures; unchanged warmed
allocation ceilings also passed. Formatting and diff checks passed. The
private parent walk is shared at the lower backend seam, with no arithmetic
change on primary Lava. Continue Arm 7 after its independent negative
control produced 2 passed/1 failed and exit 1 against the old linear formula.

## Arm 8 closed work item — ordinary owned sequences and cancellation

Objective: provide an executable fixed round-robin batch mechanism for
independent sessions, variable lengths, EOS, cancellation and contained
request failure. No fused tensor batching or throughput scheduling claim.
Permitted files: this document, `src/Runtime/Runtime.jl`,
`src/Inference/session.jl`, `test/test_batch.jl`,
`test/regime_i_batch.jl`, `test/test_export_inventory.jl`,
`test/runtests.jl`, `docs/ARCHITECTURE.md`.
Interfaces: existing engine, Runtime-owned BatchRequest/BatchResult,
run_batch and cancel!; new exports are within the existing Runtime module,
not the Gesso root. Atomic cancellation flag; Session-owned reentrant lock
and operation-busy flag reject concurrent mutation or recursive engine
calls. Batch holds leases for all its sessions and calls ordinary existing
prefill!/decode!, one token per active request per round. Nonblocking lease
acquisition and explicit duplicate-session refusal avoid deadlock/aliasing.
Private lock fields do not parameterize Session or alter storage::Any (P-1).
Terminal statuses complete/eos/cancelled/failed preserve committed partial
IDs; malformed request/callback failure cannot suppress other sequences.
Batch receipt reports actual request progress and cache usage.
Tests: real HF exact prefixes at budgets 2/5/8; unequal prompt lengths;
cancel before prefill and after two tokens; bad request beside valid ones;
duplicate and cross-task Session ownership refusal; independent concurrent
sessions; callback order shows deterministic round-robin progress. Both CPU
and primary Lava execute. Runtime's parked fence becomes a specific export
inventory, with every new surface covered. No Palette/Cyan/NIRA changes.

Arm 7 initial primary result: all independent real-model full/cached logits,
next IDs, page accounting, causal mutation and exhaustion passed at all four
lengths. Four rotated-value assertions failed (default and Llama3, both
workloads), including 32768-position first-frequency error ~0.00147 above
the fixed 1e-3 tolerance. That frequency and angle are exactly representable;
GPU large-argument trig range reduction is the remaining numerical issue.
Permitted ordinary repair: compute the SAME Float32 angle table from host
position/frequency metadata, reduce those angles modulo 2π in Float64 on the
host, and upload the reduced Float32 constants. GPU sin/cos and Q/K rotation
remain on Lava; no tensor download or inference fallback. Reuse the small
immutable angle/trig table across Q and K within the operator call, since
their head dimension and positions are equal. No custom kernel, library
change, tolerance increase or exotic work. Preserve initial failed receipt;
repeat every rotary and real-model gate after this correction.

Arm 7 completes: CPU and primary Lava each passed 46 independent context/
frequency assertions. Full CPU regression 2673 passed, 21 named skips/broken,
zero failures. The initial high-position Lava failures, wrong-linear negative
control (2 pass/1 fail, exit 1), and one stale-regression expectation failure
remain separately recorded. Corrected test expectations follow HF frequency
math, with unchanged tolerances. Required format and diff checks passed.
Proceed immediately to ordinary ownership, scheduling and cancellation.

Arm 8 execution bounds: primary Lava logical concurrent sequences use one host
owner and separate Session scratch. Independent simultaneous host tasks are
certified on CPU only; no unproven Vulkan queue concurrency claim.

Arm 8 primary blocker: 13 passed / 8 failed. Valid requests fail after a few
rounds with Vulkan host-visible heap exhaustion (246 MiB heap saturated;
main device heap still has ~6 GiB). Lava's KA allocator places tiny <=64-byte
reduction temporaries in individually mapped BAR allocations; Julia's host
allocation heuristic does not account for the driver's padded allocation.
The earlier lifecycle probe collected between repetitions and missed this
multi-session case. Keep that failed run as the negative witness.
Additional permitted repair: `ext/lava_ops.jl` and a private backend boundary
hook in Session. At a COMPLETED engine action, synchronize primary Lava and
perform a minor collection so dead temporary array wrappers retire before
another request advances. CPU boundary is a no-op. No live buffer freeing,
Lava library modification, allocation threshold change, memory fallback or
kernel research. Re-run the identical batch gate and then repeated live
three-session rounds with explicit actual heap/memory observation. Performance
cost of this ordinary temporary lifetime discipline remains visible in Arm 11.
Minor collection at completed actions did not close the blocker: identical
13 pass / 8 fail. The full-collection boundary is now the bounded candidate,
with mapped/live buffer counts recorded after each action. Preserve the minor
collection failure; do not change tests or budgets. If this cannot bound the
mapped heap, replace the relevant ordinary reduction temporary ownership in
the Gesso Lava operator seam rather than modifying global Lava allocation.
Full collection also failed (13 pass / 8 fail). Mapped buffers grew by exactly
331 per completed decode, from 1059 to 3376; collection inside an engine action
cannot reclaim them on this path. The lifetime-only candidates are rejected.
Use the existing public AcceleratedKernels reduction `temp` parameter from
Lava's dependency: allocate ordinary pooled Lava device results/temporary
storage explicitly for finite classification and dimensionwise RMS/softmax
reductions. This bypasses the tiny auto-unified allocation while preserving
Float32 reduction dtype and integer finite decisions; no CPU arithmetic on
activations or new kernel. Revalidate all 62 numeric/poison and 46 contextual
assertions, in addition to the unchanged scheduling gate. Remove the forced
full-collection candidate; retain ordinary completed-action synchronization.
Arm 8 ownership audit also covers existing `fork`: it changes parent sharing
metadata and reads a coherent snapshot. Acquire the same nonblocking Session
lease and reject active/recursive calls; retain fork's no-receipt contract.
Add a cross-task fork refusal alongside the generation ownership test.
The pooled ND candidate bounded mapped allocations at four buffers throughout
all 15 scheduled tokens, but produced wrong greedy IDs (13 pass / 8 fail).
Reject its changed one-row reduction route. Preserve the original reduction
algorithm and synchronization: multi-row calls keep the existing GPUArrays
path; scalar max/abs2 reductions use the SAME AK one-dimensional tree with an
explicit pooled temporary, scalar read and upload to the ordinary result.
Existing Float32 identity sum retains Lava's native sum path. Only temporary
placement changes; no new ND arithmetic candidate is accepted.
Arm 8 completes: primary Lava 21 scheduling assertions pass, CPU real-model
24 pass, full CPU regression 2691 pass / 21 named skips / zero failures.
Independent numeric revalidation: 47 native + 15 poisoned-engine assertions
pass; context revalidation 3 frequency + 43 real/operator assertions pass.
Mapped buffers stay at five across all 15 scheduled tokens, replacing the
331-per-token growth. Original reduction algorithms/Float32 tolerances stay
fixed; only small temporary placement changes. Failed GC/ND candidates and
one isolated-module harness include error are retained distinctly.
Fresh-process final audit will repeat primary gates after packaging.
## Arm 9 closed work item — typed containment and reuse

Objective: reject malformed model/storage/token plans before unsafe work;
contain ordinary interrupted callbacks and operation failures, record typed
failure and forbid continuing partially consumed state; reset/replay recovers.
Permitted: this document, src/Inference/session.jl,
src/Inference/Inference.jl (private entrypoint wrapping seam),
src/Inference/llama_import.jl, src/Inference/gpt2_tokenizer.jl,
src/Runtime/Runtime.jl (request failure invalidates state),
test/test_containment.jl, test/regime_i_containment.jl,
test/test_import_llama.jl, test/test_loading_boundaries.jl,
test/test_tokenizer_gpt2.jl (precise public-load taxonomy expectations),
test/runtests.jl, docs/ARCHITECTURE.md.
Private structural validator covers block presence/counts, heads/grouping,
rotary head dimension, uniform scratch dimensions, actual AND declared tensor
shapes, vocab and EOS. No semantic storage redesign. Validate prompt IDs
before reset/embedding lookup; invalid inputs cannot launch unchecked indexes.
Public checkpoint/config/tokenizer loaders classify malformed content as
ERR_INVALID_PLAN preserving the original offender/cause text; known Gesso
errors retain identity. Engine unknown failures become ERR_RUNTIME,
InterruptException becomes ERR_RUNTIME with interruption context, allocation
failure becomes ERR_ALLOCATION. Failed engine or batch callback invalidates
ready state; decode cannot proceed until a new generate/reset cycle.
Tests: malformed tiny model/tensors; missing checkpoint/invalid header;
negative/out-of-range IDs; callback failure and interruption after two tokens;
typed receipt matching thrown failure, partial token count, invalidation and
same-session recovery; mixed batch request isolation. CPU and real primary
Lava execute exact HF output after failure. No deliberate driver crash, device
loss recovery certification, service changes or exotic scheduling.
Arm 9 pre-launch audit: raw embedding operators also accept unchecked IDs,
including the CPU inbounds row copy and retained CUDA comparison kernel.
Additionally permit `src/backends.jl`, `src/Operators/cpu.jl`,
`ext/lava_ops.jl`, `ext/cuda_ops.jl` for one shared private shape/host-ID
validator before any embedding launch, with CPU/primary Lava raw negative
probes under both workloads. Protect the comparison seam with the same
validator; no CUDA optimization. Also permit Lava's typed allocation-error
classification at the existing private engine failure seam. Public loader
docstrings must remain attached after boundary wrappers are introduced.
Arm 9 also audits the existing String generation entry before tokenization:
missing or failing tokenizers must yield the same typed, single receipt and
ownership/recovery behavior as integer prompts. No tokenizer algorithm change.
An injected host projection wrapper fails after cache writes and proves the
engine invalidates partial state, then exact tiny-model replay recovers.
Arm 9 completes: CPU and primary Lava each passed 29 real-model containment/
recovery assertions. Full CPU regression 2738 pass, 21 named skips/broken,
zero failures. A real host projection failure after KV writes, callback
interruption and allocation-class injection all produce typed failures and
invalidate continuation; reset/replay recovers. Raw embedding IDs are checked
before launches. Public loader offender detail and docs are retained, and
String generation joins the single ownership/audit boundary. Required format
and diff checks pass. Proceed to truthful observability.
## Arm 10 closed work item — truthful action receipts

Objective: meaningful phase timing, first token timing, stable output identity,
partial output on failure and a receipt delivery failure that cannot alter
inference. Permitted: this document, src/Inference/session.jl,
src/Runtime/Runtime.jl (reuse safe delivery and digest seam),
test/test_observability.jl, test/regime_i_observability.jl,
test/reference_receipts.py, test/runtests.jl, docs/ARCHITECTURE.md.
Finding before edit: TTFT currently adds ALL decode time to prefill; output
digest is always nothing. Track only first decode elapsed for TTFT, total
decode separately and committed IDs through the existing private span.
Digest contract: FNV-1a 64 over each nonnegative token ID encoded as an unsigned
64-bit little-endian integer, algorithm/version named; this is replay identity,
not cryptographic authentication. Independent Python computes expected values
from the external HF sequence. Failure receipt includes committed IDs and
counts; invalidation preserves partial state for diagnosis until reset.
Safe private delivery catches even a faulty custom sink; sink failures must
not change IDs, replace inference failures or suppress other batch requests.
Requested and actual backend/dtype are explicit, as are cache page bytes and
remaining capacity. Timings measure completed engine work; first use versus
warmed throughput/compilation is separated by Arm 11. No tracing framework,
external service or new core dependency. Tests include multiple callbacks,
partial interruption, EOS/no new tokens, digest stability across runs,
independent expected digest, and deliberately throwing sink on CPU and Lava.
String request timing also records tokenization explicitly. TTFT includes
that phase plus prefill and only the first decode; absent for zero new IDs.
Vector prompts record tokenize_ns=0. This preserves one audited action per
String request. Batch terminal cancellation fills its existing cancellation
field, with no new receipt schema or vocabulary.
## Arm 11 closed work item — measured ordinary performance floor

Objective: honest warmed end-to-end throughput and host allocation receipts
for primary Lava, Gesso CUDA comparison and independent PyTorch CUDA eager,
on the same RTX 5060/checkpoint/three prompts/eight new greedy IDs. All GPU
native arrays Float32, TF32 disabled on the PyTorch comparator. CPU Float64
is a correctness oracle rather than a same-precision speed competitor.
Permitted: this document, test/regime_i_performance.jl,
test/reference_performance.py, src/Autotune/Autotune.jl,
ext/cuda_ops.jl (ordinary winner lookup only), test/test_autotune.jl,
docs/ARCHITECTURE.md. Optional primary ordinary buffer lifetime repair remains
within the Arm 8 seam and requires revalidation of numeric/context gates.
First use, initialization/loading/transfer and warmed execution are separate.
Two warmups and three independent measurements per shape; exact external IDs
on every call. Fixed repeat-stability gates: max/min timing <2, host allocation
spread <1 MiB. No claim of a universal speed floor or outperforming CUDA.
Investigate measurable avoidable overhead and preserve before/after receipts.
An existing candidate registry copies its entire vector on each winner lookup;
add a private lock-protected lookup retaining public candidates() copy semantics
and replacement-by-name behavior. The CUDA comparison caller can use it with
an unchanged explicit stale-winner failure. No new kernel, cache policy or workload change; comparison-only allocation
ceiling remains unchanged. The bounded normalization amendment below repairs
unintended intermediate precision before the final matched-Float32 comparison.
Tests: private name lookup missing/replace/isolated public copy plus unchanged
full suite. Timed runs must be sequential with no competing GPU measurements.
This is the last performance work before packaging; exotic KV, kernel/fusion,
cooperative matrices and speculative methods remain excluded.

Arm 11 primary overhead audit permits `ext/lava_ops.jl` and
`ext/cuda_ops.jl` for normalization precision alignment only:
Float64 Session epsilon currently promotes ordinary Float32 denominator
broadcasts on primary Lava (and generic retained CUDA prefill). Before/after
receipts must disclose the initial mixed intermediate precision. Cast the
positive finite epsilon to the requested storage dtype, rejecting an
unrepresentable/underflowed epsilon explicitly. This removes an accidental
promotion, adds no native half precision or different tensor representation.
Revalidate all independent numeric, contextual and scheduling gates at the
fixed tolerances and exact IDs before graduating. Supported storage remains
Float32 on devices; CPU Float64 reference arithmetic stays unchanged.

Permitted numeric tests: test/regime_i_numerics.jl and
test/test_numeric_lava.jl, plus test/regime_i_performance.jl metadata.
The before run retains promoted Float64 intermediates and is labeled as such;
only the final comparison runs count toward the matched-Float32 floor gate.

Arm 11 bounded bookkeeping amendment: the comparison remains 43 KB above its
existing allocation ceiling after list-copy removal. A direct metadata profile
finds an 80-byte TuneResult wrapper on every cache hit plus an immutable
candidate return. Permit inlining the existing select and private name lookup
so the consult site can read the winner/run fields without boxing wrappers or
variadic semantic arguments. Public select result/receipt identities, cache
keys, candidate replacement and stale-winner errors remain unchanged. No new
cache policy, device kernel, dtype or numerical algorithm. Recheck existing
Autotune tests and full CPU/host ceilings without changing thresholds.

The inline candidate did not reduce the direct metadata profile and is
rejected. A consult-only private cached-winner read is permitted instead:
read the existing locked key and winner name without passing/boxing semantic
operands or constructing a public TuneResult. Cache misses still call the
unchanged public select with all live operands and emit its mandatory receipt.
Candidate lookup remains fresh by name, so replacements and stale errors keep
their semantics. Test missing/hit/invalidation; no new cache or decision.

Arm 11 receipt-buffer bookkeeping item: new cache-miss evidence shifts the
process sink's vector-growth boundary into an otherwise constant-size decode
allocation probe (110,056 B versus 16,000 B). Permit src/receipts.jl and
test/test_receipts.jl to reserve capacity+one transient incoming slot at sink
construction. Keep the existing push-before-drop policy, dropped counts and
error isolation, with no receipt schema or retention-policy change. Verify
retention contents and stable backing storage across overflow, then unchanged
full allocation gates. Cost: about 2 MB reserved upfront at default capacity;
this is bounded metadata storage, not device or kernel optimization.

Receipt reservation refinement: preserve the reserved storage when dropping the oldest record; shift retained records and pop the last slot. Julia front deletion advances the storage offset and eventually regrows even a reserved vector. Retention order and dropped counts stay unchanged.

Arm 11 final dispatch bookkeeping item: the actual CUDA allocation profile
finds 6,752 bytes copying Candidate values and 20,256 bytes boxing semantic
arguments for erased candidate calls per decode. Permit a private registry
runner lookup (fresh by winner name, no function cache), registering the two
existing named implementations directly, and concrete calls only when the
retrieved function is exactly one of those implementations. Replacement
functions still execute through the ordinary dynamic call; missing entries
still fail. No new kernel, candidate, cache key, tuning decision or arithmetic.
Permit test/test_autotune_cuda.jl replacement-by-name regression. Existing
allocation limits remain fixed and real-reference checks must repeat.

## Arm 12 closed work item — reproducible package entry and final audit

Objective: install/resolve the declared package in a new writable depot and
relocated source path, then run real local CPU and primary Lava inference
without inherited compiled Gesso/Lava caches. Cached dependency sources and
artifacts remain read-only lower-depot inputs, explicitly documented; not a
network clean-room or every-OS/hardware certification.
Permitted: this document, src/Inference/Inference.jl (import owning modules),
test/Project.toml (exact Lava revision matching the tested tree),
test/regime_i_package.jl, docs/ARCHITECTURE.md, docs/PACKAGING_REGIME_I.md.
Resolve and export manifest in the isolated tree only; no upgrade. Lava tested
source tree 795df8f1fb847d261f4da4d944448c4d96683862 equals readonly libs/Lava
commit 11c7e31bdf62408d22bf379e9e59510f69d2103e. Pin this commit, not master.
Remove undeclared imports of PrefillWorkload/DecodeWorkload/KVCache from the
parent before reexport by importing them from their actual owning modules;
identities/exports stay unchanged. Required full CPU and actual-host suites,
format/diff checks and repeat primary lifecycle/independent references.
Archive reviewable source plus exact environment lock; keep models/depot,
credentials and unrelated shadow delta outside campaign diff. Export binary
patch against the preserved full carried baseline; read-only apply checks on
both originals and hash/status preservation audit. Stop before Regime II.

Arm 12 prerequisite finding: with every optional package loaded from source,
LLVM 9.13.2 pointerref/pointerset infer Any and Lava's ordinary shared-memory
reduction fails GPU compilation (4 checks passed, one error). Existing compiled
LLVM infers Float32/Nothing. The same pinned LLVM/GPUCompiler dependencies are
now built in the fresh writable depot; no old compiled cache is copied. Permit
scripts/prepare_lava_compiler.jl to prepare those declared transitive compiler
dependencies using a temporary project and the exact installed workspace lock,
plus documentation of the mandatory preparation before source-only Lava probes.
No compiler package upgrade, Lava patch, new kernel or arithmetic change. The
failed probe remains evidence; actual relocated inference must pass afterward.

Arm 12 host-integration findings: the default actual-host suite reports 3108
passed, five failed, three named skips. Three CUDA toy parity failures have
max delta 0.00171085 above the existing 0.001 ceiling; CPU decode allocates
16,608 above 16,384 bytes only with optional extensions loaded, and CUDA real
decode allocates 1,051,520 above 1,048,576 bytes in that combined environment.
Permitted: ext/cuda_ops.jl to finish the declared native Float32 normalization
(the inherited apply kernel still widens to Float64); src/Inference/session.jl
for measured metadata boxing only; existing numerical/allocation regressions.
Do not relax any tolerance/limit or add a kernel/candidate. Profile the actual
combined-extension context and repeat targeted references before full suites.

The native CUDA apply experiment still misses toy parity (0.00134464 > 0.001)
and is rejected. Restore the original CUDA mixed-precision normalization,
retain explicit configuration/storage guards, and label its Float64
normalization intermediates in comparison receipts. Primary Lava and independent
PyTorch retain native Float32 normalization. Do not claim identical arithmetic
for this auxiliary legacy comparator. The combined-context CPU profile shows
two 192-byte immutable Receipt copies across return boundaries; emit inside the
existing concrete finish barrier, with unchanged single-emission/error/sink
semantics. Permit the existing CUDA storage guard to use an array-only function
barrier so erased semantic fields do not widen guard inference; preserve view,
backend and dtype validation. Actual combined-context allocation gates repeat.

The combined CUDA allocation profile reports 1,051,552 bytes and repeated
boxed shape/grid/argument tuples in the existing rotary wrapper (including
5,760 bytes of 180 two-Int tuples and 1,920 bytes of 60 shape tuples).
Permit an array-only ext/cuda_ops.jl rotary function barrier to specialize
those host launch arguments. Preserve the same kernel, Float64 theta,
position transfer, two launches and numerical output. No pooling, new
kernel, fusion or arithmetic change; retain the fixed allocation ceiling.

The repaired full host suite reports 3,112 passes, one failure and three
named skips: toy CPU decode is 16,416 bytes, only 32 above the unchanged
16,384-byte ceiling. CUDA accuracy and allocation now pass. The combined
allocation profile retains one 192-byte Receipt box at the abstract sink
field call. Permit passing the existing sink as a separate argument to the
concrete receipt finish barrier, allowing specialization of its safe emit.
Keep sink identity, one emission, return values and error behavior unchanged.
Repeat the CPU/full-host gates; do not raise the allocation ceiling.
