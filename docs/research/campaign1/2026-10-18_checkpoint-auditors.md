# Campaign 1 checkpoint: recurrence and physical-division operators verified

**Superseded supplier status:** the standalone compiler contract is now
qualified as local Lava commit `289d0693e84a3ad6ae6869cbf81b47dd4bc20ba7`.
See `2026-10-18_no-contraction-receipt.md` for exact source/AST,
GPU/native/mutation and cost scopes. The pending-source statements below are
historical. Gesso remains ef086ef, 39/41; Campaign 1 stays active.


2026-10-18. Campaign 1 remains active and incomplete. The frozen Gesso source
is still `ef086ef34eae6a12f222ba0ddec1d254b9622e12`, with the unchanged full-model
gate at 39/41. No production source changed. This work began with CPU auditor
preparation while the user's Qwen/Claude GPU reservation remained active.
The user subsequently released the GPU; live checks found no competing compute
job or held shared locks. The sequential GPU results below supersede the
earlier unrun/preparation statements. The previous exponential proof remains
valid; see `2026-10-16_checkpoint-exp32.md`.

## Latest GPU evidence

The GPU-only exp32 recurrence is terminal, exit 0. Across four geometries and
three repetitions each, all twelve new-route outputs, recurrent states and
decay arrays match CPU bitwise. Ordinary GPU-exp controls retain six failed
repetitions. Independent audit validates the raw arrays against a freshly
executed Julia Base reference plus Python scalar recurrence, all sixteen
emitted shaders and nine actual linked records, and complete arithmetic-width
sets. Removing inserted NoContraction decorations restores the original words.
Copied corruption of all three new-route outputs, with updated hashes and
retained green flags, is rejected.

The first gated-normalization experiment is a retained negative, exit 1.
Custom exp and denominator match CPU, while F32 division changes 71,856 of
262,144 gates, maximum two ULP. The intended wider expression compiled to F32
division; its identical output is not evidence about physical F64 arithmetic.
Auditor v3 rejects that missing width. A separate v4 diagnostic verifies all
raw/source/CPU arrays, nine emitted/four linked shaders and counters while
explicitly reporting `operator_exact=false`, three required-route failures and
`intended_F64_gate_arithmetic_observed=false`. No numerical threshold changed.

The subsequent physical-buffer experiment is terminal, exit 0. Separate
promotion, F64 division/store and F32 rounding kernels prevent folding. All
three repetitions match CPU numerator, denominator, F64 quotient, F32 gate,
normalization and original merged output exactly. The F32 control still has
71,856 wrong gate values per repetition. Independent audit verifies all twelve
emitted shaders and seven linked kernels, actual physical pointer/load/store
widths, conversion widths, native GGUF epsilon/weights, exact patch inverse,
terminal exit and every raw byte (65,012,224 bytes). This proves the bounded
real-p64 operator, not universal equivalence of wider division to F32 division.

Two copied mutations are rejected: changed merged outputs with repaired
hashes/green flags across all repetitions, and a one-ULP F64 quotient corruption
that leaves the rounded F32 gate unchanged. The latter tests the hidden widened
arithmetic independently of the final output. Initial physical auditor v1's
scope-string mismatch is retained; the corrected v3 checks actual dispatch
inventory rather than assuming declaration order and binds physical pointers.
No producer, reference, fixture or accuracy target was changed after the run.

Primary immutable checkpoints under R (defined below):
`checkpoint-gdn-operators-20261018.json` and
`checkpoint-gdn-physical-division-20261018.json`. They bind all helper versions,
proof hashes, terminal handles, raw captures and preserved failures.

Primary receipts:

- `gdn-recur-exp32-validation-1.json` and
  `gdn-recur-exp32-mutation-validation-1.json`.
- `gdn-exp32-gnorm-validation-1.json` (strict-width rejection) and
  `gdn-exp32-gnorm-validation-2.json` (negative diagnostic).
- `gdn-physical-div-validation-2.json`,
  `gdn-physical-div-mutation-merged-validation-1.json` and
  `gdn-physical-div-mutation-quotient-validation-1.json`.

## Standalone source contract qualification

The isolated six-file NoContraction draft remains byte-identical to its saved
patch. Actual CPU tests now pass 155 assertions: 110 source parser/emitter
checks on ten real shaders, 30 package wrapper/Adapt/config/key checks, and
15 actual compiler/optimized-SPIR-V/disk-cache checks. Valid SPIR-V mutations
with a required decoration removed are rejected. Missing cache policy flags,
wrong flags and incomplete coverage produce misses or explicit store rejection.
Explicit source Fma survives opt-in compilation. No Vulkan context was created
by these tests. Receipts are `no-contraction-source-spv-cpu-1/receipt.toml`,
`no-contraction-host-api-1/receipt.toml`, and
`no-contraction-compile-cache-1/receipt.toml`.

The prior FloatControls GPU regression is now terminal, exit 0, with 116/116
assertions and no skips. Its eight explicit shader exports validate; its
isolated temporary compiler dumps were removed by the existing test helper.
The representative parent/candidate default shader bytes are identical, SHA256
`8166451f23abb376be09e6a8ba1cf9e6100f6cf3d9a86494da3a506184997ef5`.
These bring the focused source/regression assertion count to 271. A preliminary
post-run inventory expecting deleted temporary dumps is preserved as a failed
audit; the corrected artifact-scope audit passes.

Public source NoContraction dispatch is terminal, exit 0. All six required
repetitions match independent exact-rational IEEE Float32 rounding; three
ordinary diagnostic repetitions instead match fused arithmetic. Three emitted,
runtime-linked and frozen shaders have identical bytes and pass validation.
The policy separates the intended cancellation witness (+0) from explicit Fma
(-2^-46); explicit Fma remains fused. Copied output corruption across all three
separated repetitions, with repaired hashes and green flags, is rejected.
Receipts: `no-contraction-gpu-dispatch-validation-1.json` and
`no-contraction-gpu-dispatch-mutation-validation-1.json`.

Frozen-cache corruption helper v4 is terminal, exit 0. The nested public GPU
witness independently passes in
`no-contraction-frozen-corruption-dispatch-validation-1.json`. Runtime checks
reject a valid shader missing one actual required decoration in memory and
on disk, reject its store without changing the file, recover a good disk entry,
and classify truncated serialization as a miss. No corrupt shader is submitted
by the helper; pipeline keys and dispatch count are checked unchanged. The
receipt has a reporting defect: top-level
`no_contraction_corruption_linked_or_dispatched=true` is hard-coded and
contradicts its detailed tests. Preserve it as a schema defect, not evidence
of execution or a clean independent cache receipt. The previous v3 world-age
harness failure is also retained.

The consumed six-file draft is frozen. A new isolated checkout,
`lava-no-contraction-qualified-20261018`, begins with identical
runtime bytes and adds portable maintained tests/docs and test registration.
The first maintained run (`no-contraction-maintained-1`, session 45992) ended
exit 1: 58 assertions passed, including all six public arithmetic repetitions,
then the direct-compile helper passed `nothing` as a static workgroup size.
The consumed test snapshot is preserved. Its correction uses the actual
KA iteration-space work-item count, matching the runtime. Run 2 (session
29233, `no-contraction-maintained-2`) is terminal, exit 0, with 101/101
assertions. Its nine source/test/doc files are bound by
`no-contraction-qualified-consumed-run2-source.json`; that checkout is now
frozen. A fresh `lava-no-contraction-structural-20261018` worktree and matching
offline environment copy those exact files for the structural correction.
The earlier review also corrected
its workgroup size from 4096 to 64 before run 1.

A read-only review identified a genuine contract gap: the arithmetic parser
checks coverage and word extents, not complete SPIR-V structural validity.
The CPU-only discriminator `no-contraction-malformed-preflight-1/receipt.toml`
(session 93905, exit 0, six assertions) reproduces false acceptance of both a
header-only stream and an actual shader with invalid ID bound. `spirv-val`
rejects both. No malformed shader was sent to the GPU. A fail-closed structural
check at shader trust/link boundaries and maintained malformed-cache cases
are required before supplier admission; a vacuous arithmetic set must remain
valid for genuine integer-only/Fma-only shaders.

The new structural checkout implements a fail-closed validator using the pinned
SPIRV-Tools executable, isolated temporary files and no new dependencies. Tool
timeout/failure surfaces; invalid modules or missing coverage cause cache misses
and invalid in-memory entries are evicted. Required-policy checks precede the
FloatControls parser, so malformed instruction extents cannot bypass recovery.
Default policy remains untouched. The new maintained cases cover a header-only
module, bad ID bound and zero-word-count instructions in serialized/memory
caches, unchanged bad-store files, no dispatch/pipeline changes and a genuine
integer-only positive. The test also stops sweeping the debug directory shared
by sibling worktrees; only its own temporary directories are removed. Root
review corrected the external draft's loader order before execution. Session
64047 is terminal, exit 0: `no-contraction-structural-1` passes 141/141
assertions on the actual RTX 5060. The exact source patch and test file still
match their pre-run snapshots. The nine-file identity is bound by
`no-contraction-structural-consumed-run1-source.json`. This closes the reproduced
malformed-cache gap for the tested cases; the earlier false-pass and harness
failures remain preserved. The implementation invokes a structural validator
at required shader trust boundaries; its cache/compile cost is not yet measured.
Full cost/capability/quantized controls, broader direct compiler entrypoint
coverage and source/model admission remain open. These source
tests are not final Gesso stress evidence.

## What changed and why

The incomplete recurrence auditor now has a new, separate executable version,
`validate_c1_gdn_recur_exp32_v2.py`. It verifies the four geometries and the
2/2/2/3 route repetitions, independently recomputes CPU normalization and
recurrence, checks raw-derived counters and actual terminal exit, and audits
linked/emitted shaders. A separate CPU-only Julia helper regenerates synthetic
inputs from the fixed seed and recomputes every decay with pinned Julia Base.
The reference subprocess imports only standard libraries, not Gesso or Lava.

A supervised read-only review found three substantive gaps in the prepared
gated-normalization auditor: it trusted saved exponential values, did not bind
epsilon to native GGUF metadata, and omitted all per-case arrays from its byte
total. A new `validate_c1_gdn_exp32_gnorm_v2.py` corrects these. Its separate
Julia Base helper computes exp, denominator and both division variants from
the original z values. The auditor checks native epsilon/weight payloads,
actual arithmetic result widths, preservation modes on the compute entrypoint,
the exact link-hook insertion, and emitted/linked shader correspondence.

Both auditors distinguish an internally consistent diagnostic receipt from
operator exactness. A diagnostic `PASS` can preserve a numerically failing
route; `operator_exact` and the raw-derived required failure counts remain
separate. No tolerance or target token changed.

All original helpers and captures are preserved. The new versions were frozen
when consumed by their tests; further corrections require new versions.

## Executed evidence

Scripts root S:
`<pystack-work>/scripts/campaign1-20261014`.
Results root R:
`<pystack-work>/results/campaign1-20261014` (the existing
symlink to the large artifact disk).

- `test_c1_gdn_recur_exp32_auditor_v2.py`: terminal session 28378, exit 0;
  15 preflight checks. Four saved cases reproduce independently, including the
  real p64 CPU output and recurrent state. Both saved CPU-decay GPU repetitions
  per geometry match the newly reconstructed references. Nine copied input,
  decay, output or state mutations are rejected after their hashes are updated.
  The full CLI rejects the old producer as evidence for the unrun new route.
  Receipt: `R/gdn-exp32-auditor-cpu-preflight-1/receipt.json`, SHA256
  `05d7f02f1e4d0b222248c622662100271caf35cad69b61debbd9f28c1bed0a29`.
- `test_c1_gdn_exp32_gnorm_auditor_v2.py`: terminal session 96182, exit 0;
  nine preflight checks. Fresh Julia Base reproduces all 262,144 saved gate
  elements; native epsilon/weights and Python scalar RMS/merge reproduce the
  original output. Four copied reference mutations with updated hashes and a
  changed epsilon are rejected. The real saved exp32 shader passes the F32
  arithmetic-width check and is rejected as a gate requiring F64 division.
  Receipt: `R/gdn-exp32-gnorm-auditor-cpu-preflight-1/receipt.json`, SHA256
  `f75189c6ee41efb6dbb24ee335a241a3465128ee27dffb4a07576970b672bf0a`.

The executed commands used `gjob.sh small 300`, a 6 GiB RAM / 1 GiB swap cap,
one Julia/BLAS/OMP thread, offline standard libraries and no GPU imports.
Host: Intel Xeon E5-2699 v3, Linux 6.18.55-2-cachyos-lts; Julia 1.12.6.
Observed preflight elapsed times were about 9.06 s and 5.19 s. These include
startup/reference work and are not inference or kernel benchmarks. Compile,
warm allocation, peak memory and throughput effects of a production repair
remain unmeasured. The numerical reference deltas above are bit-exact, with
no approximation budget.

Immutable machine-readable checkpoint:
`R/checkpoint-auditor-preflight-20261018.json`. It binds all six new helpers and
both receipts. The declared upcoming captures total 65,800,208 recurrence bytes
and 65,012,224 gated-norm bytes; both fit the unchanged 67,108,864-byte cap.

## Limits and next work after the GPU results

The CPU preflights above were followed by the actual GPU results at the top of
this document. The gnorm byte-accounting correction passed a complete capture.
The exact recurrence and physical-division operators remain external controls;
neither is integrated into the frozen 39/41 source.

The recurrence shader review was addressed before the GPU run: v3 constrains
complete opcode width sets, pins the recorded Random source identity, and
checks kernel names/IR hashes from a new hook and consumer. All original helper
versions remain intact. Current auditors bind modes to the compute entrypoint.

Continue source contract qualification under the existing shared locks and
resource caps, with fresh output/log/shader directories. Preserve all negative
controls and exact copied mutations. Inspect actual shaders and runtime paths;
a passing host wrapper/config test does not prove device dispatch or cache use.

The source draft is uncommitted and is not vendored. Finish its explicit
compiler contract and test
the unchanged quantized controls without stacked link overrides. Integrate in
a new Gesso candidate only; rerun the complete unchanged model gate, then the
original final-source stress, recovery, integration, formatting, capability and
cost matrix. No Campaign 1 acceptance requirement is reduced by this checkpoint.
