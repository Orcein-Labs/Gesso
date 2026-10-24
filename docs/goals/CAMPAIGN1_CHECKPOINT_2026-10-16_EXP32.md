# Campaign 1 checkpoint: GPU Float32 exponential verified

2026-10-16. Campaign 1 is active and incomplete. User requested a reviewable
checkpoint with approximately 6% weekly quota remaining, then reserved the
GPU for Qwen retraining followed by Claude. No new Gesso GPU jobs are queued.
No merge, push, release, deployment or production cutover occurred.

## Verified outcome

The external Julia-matched Float32 exponential passes all 464,446 inputs in
three GPU repetitions: 1,393,338 comparisons, zero non-NaN bit mismatches and
zero NaN-class mismatches. Actual GDN decay and negative-z inputs are included,
with dense ranges, random IEEE bits, boundaries and range-reduction ties.
Physical Float32 stored results match independently recomputed Julia Base.
The input and result SHA values match in every repetition. This proves the
bounded primitive on the tested RTX 5060, not a full model/source repair.

Independent numeric/source/Julia reference audit:
`julia-exp32-replay-validation-2.json`, SHA256
`fbc23fde2d033174809b423a982d7ee530f966857a80ec6f534da22a17e0570f`.
Effective copied numeric mutation updates all result hashes/green flags and
all repeated outputs together; independent recomputation still rejects it:
`julia-exp32-numeric-mutation-validation-2.json`, SHA256
`de18590018467e12226d14aadda8ac89e7e88c0cff131c182e8b458662307d56`.
All three emitted SPIR-V modules and the actual linked module validate;
NoContraction removal restores all original shader words:
`julia-exp32-shader-validation-2.json`, SHA256
`0c52b538c57e51c615ba29e264ae08a52bed36e35c008246394bd39d5c1b6432`.

The separate MPFR512 CPU receipt measures maximum one ULP between Julia Base
and the high-precision reference; this is a measurement, not a changed oracle
or approximation budget. Full CPU/GPU raw capture, shaders and failed run1
remain preserved. Run1 exited 1 at tuple-pointer shader validation; v2 expands
the identical polynomial into explicit Horner Fmas and passes. No coefficient,
input, reference or tolerance was changed to obtain the pass.

Results root (R):
`<pystack-work>/results/campaign1-20261014`, resolving to
`<research-artifacts>/Diskguard/2026-10-15/gesso-campaign1-results`.
Machine-readable checkpoint: `R/checkpoint-exp32-20261016.json`.
Mutable resume index: `R/resume-frontier.json`.
GPU session46577 and CPU auditor session12620 both exited 0. Run1/session19268
exited 1. No Gesso GPU job remains running or waiting from these experiments.

## Frozen source and unfinished work

Current Gesso source remains `ef086ef34eae6a12f222ba0ddec1d254b9622e12` in
`gesso-campaign1-cpu-rms-20261015`. Its tracked source is clean;
only the existing untracked `vendor/tmp_kernels/` remains. Its full model gate
is still 39/41; fresh cancellation/recovery failures remain. The external
Q4_K x Q8_K head prototype passes 42/42 but is not integrated. The original
full Campaign 1 source/stress/integration/cost/portability gates remain open.

Prepared, unrun operator scripts under S =
`<pystack-work>/scripts/campaign1-20261014`:

- `c1_gdn_recur_exp32_conformance.jl`, `c1_gdn_recur_exp32.jl`,
  `c1_gdn_exp32_no_contract_link.jl`: frozen recurrence arithmetic plus
  GPU-only exp32 decay; original stock/CPU-decay/device-exp controls retained.
- `c1_gdn_exp32_gnorm_conformance.jl` and
  `c1_gdn_exp32_gnorm_no_contract_link.jl`: staged exp/denominator/gate/merge
  with CPU-gate and ordinary-device controls and F64-division discriminator.
- `validate_c1_gdn_exp32_gnorm.py` is syntax clean and has a CLI, but has not
  validated a capture. Review all checks and add effective mutation proof.
- `validate_c1_gdn_recur_exp32.py` is an INCOMPLETE draft: only utility/source
  functions exist; no validation loop or CLI. Finish a new version before
  operator admission. Do not mistake Python syntax success for an audit.

Julia parse-tree checks reject error/incomplete nodes and pass for the five
new operator helper files. This is preparation only, not operator execution.
All helper hashes are bound in the machine-readable checkpoint; preserve
those exact files and use new versions for subsequent corrections.

A separate standalone Lava NoContraction source draft exists at
`lava-no-contraction-20261016`, branch
`codex/no-contraction-20261016`, parent
`6a3b0e4a35de6af4587aa4313b33b89c5036478c`. Six source files are modified and
uncommitted. Julia parse trees and git diff whitespace check pass only;
behavior, cache corruption, GPU dispatch, formatting and cost are UNTESTED.
It is not admitted or vendored. Exact draft is preserved as
`R/no-contraction-source-draft-20261016.patch`, SHA256
`47bbb55dd2e215244a6c9c798ecb2a05cb791a74c753b8dc32b8b7483a8ab5ae`.
Its bounded work item is `CAMPAIGN1_NO_CONTRACTION_CONTRACT.md`.

## Resume order

1. Review/complete the prepared independent operator auditors before observing
   new captures. Read `CAMPAIGN1_GDN_REFERENCE_CONTROL.md` for unchanged gates.
2. When the user's GPU queue is clear, run the recurrence and gated-norm
   experiments through the existing bounded `run_cpu_rms_candidate.sh`, with
   new output/log/shader paths, 4800 MiB GPU budget and shared locks. Do not
   queue them ahead of Qwen/Claude. Independently validate all arrays/shaders,
   actual terminal exit and effective numeric mutations; preserve negatives.
3. If these operator primitives close their seams, finish/test the explicit
   NoContraction compiler contract in its isolated tree. Never stack old link
   overrides: each replaces the same Lava.link_kernel method. Revalidate
   immutable Q8_0/Q4_K controls without monkeypatches, then integrate only in a
   NEW Gesso candidate and run the unchanged complete model gate.
4. After a sufficient source repair, run the original final-source stress,
   integration, formatting, capability and cost matrix. No throughput claim
   or Campaign 1 completion follows from this checkpoint.
