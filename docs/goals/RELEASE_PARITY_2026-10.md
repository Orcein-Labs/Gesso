# Qwen3.5 and Gemma4 release campaigns

Updated: 2026-10-14. Status: execution plan; all campaign exits remain open.
Owner: Matt. Implementation lead: Codex, continuing the October 13 handoff.
This plan is subordinate to `docs/Gesso_Stack.md`. It organizes the current
release objective; it does not reopen every phase in the long-term vision.

## Release objective

Ship Qwen3.5 and Gemma4 family inference on Gesso with Lava as the default,
correct against independent references and at least at PyStack performance
parity on the declared release workload matrix. Then open specialized Julia
research against that measured baseline. Performance must come from portable
implementation choices or explicit, tested capability selection.

Ornith-1.5-35B-A3B GSQ-RCO 3-bit is the immediate workload. Configure its
location through `GESSO_ORNITH35_GGUF`; never embed a developer's model path.
Ornith results do not certify every Qwen3.5 variant or Gemma4.

Scope decision pending: text inference first versus multimodal release
support. The campaign decomposition below covers text inference. It does not
claim a text-only scope has been approved. If multimodal support is required,
add explicit model, processor, correctness, serving and performance work items
before freezing campaign 2's matrix. MTP and speculative execution are not
required by this text plan; importing a checkpoint containing extra heads must
still account for them explicitly.

## Which existing plans this continues

| Source | Current use | What needs updating |
|---|---|---|
| `docs/research/ROADMAP_NOW.md` | Long-term phase interpretation and architecture boundaries | Operational status largely stops at October 3; it is not the current release checklist |
| `docs/goals/REGIME_I_MATURITY.md` and `REGIME_II.md` | Reliability and attribution gates; retain completed evidence | Small-model results cannot graduate the new families |
| `docs/goals/ARCH_GAUNTLET.md` | October 8 architecture work and acceptance ladder | Old sequence includes Xing; current release priorities are Qwen3.5 and Gemma4 |
| `research/2026-10-13_pj-close/2026-10-13_codex-handoff.md` | Latest operational frontier, job rules, local branches and coupon procedure | Correctness blocker remains open; no complete family release plan |
| `research/2026-10-13_pj-close/2026-10-13_pea-warm-path-audit.md` | Cost attribution and named findings | Its status table and early L1 localization are superseded by the handoff |
| `DEBT.md` | Candidate release risks to verify | Entries are October 8 observations, not confirmed current defects |

No newer complete release plan was found in the current checkout's docs,
research, README or debt register. This file is the current campaign checklist.
Historical recipes and receipts remain historical evidence.

The October 13 handoff and audit are local ignored research files, absent from
a fresh checkout of tracked master. Their paths above identify the source notes,
not packaged dependencies. The starting ledger and campaign 1 below preserve the
release-critical findings and gates without copying private machine paths into
public documentation. Retain the original notes and raw logs locally.

## Starting ledger

Inspected main at `37d87da`. No inference or performance tests were rerun
while writing this plan. “Reported verified” below refers to the handoff;
branch existence and tips were checked, not the numerical claims themselves.

| Item | State | Next treatment |
|---|---|---|
| L1: Ornith Lava p64 wrong tokens / nonfinite activations | Open correctness blocker; L1 worktree retains two temporary probes | Campaign 1 first; expected p64 token `82202`, t1 token `11` |
| W3 host MoE tiled prefill, `e20e65f` | Reported verified, local branch | Integrate after L1; reproduce on combined tree |
| S1 nucleus sampler, `cf9f691` | Reported verified, local branch | Integrate after L1; retain semantic equivalence and red control |
| Vendored Lava, `faa39d3` | Reported verified, local branch; only approved Lava patch is the ConstAdaptor deletion | Use this fork for release verification; no extra Lava changes implied |
| W4 host-expert/device overhead; W2 one-token prefill; V4 non-Q8 GEMV | Open performance findings | Campaign 3; remeasure after correctness and integration |
| Qwen3.5 dense/MoE GGUF | Import and test code exists on main | Inventory exact variants and prove release matrix in campaign 2 |
| Gemma4 | Main GGUF dispatcher rejects it; separate `lane-gemma` tip `694b36e` has forward/tokenizer work | Inspect ownership, delta and evidence before reuse; do not merge the entire branch blindly |
| Chat serving | HTTP, streaming, sampling and engine adapter exist | Campaign 4 must test actual loaded models and family-specific rendering |
| Public API for hybrid MoE | Handoff says `Session` refuses Ornith; serving uses `reference_generate` and `GenerateState` | Document and test the supported public entry point; do not claim universal Session support |
| Palette shell T1–T5 | Reported local completion; T5 has unowned/unverified additions | Separate supporting track; triage before adopting |
| Palette Julia security-test ports | Handoff's nominal green run actually has 11 failures | Not green; confirm package/setup cause, reconcile red totals before adoption |
| Coupon 2 | Rewrite procedure prepared; nothing pushed according to handoff | Matt-controlled checkpoint after L1 and combined-tree prechecks |

## Acceptance contract

Campaign 2 freezes explicit rows: checkpoint identity/digest, architecture,
format and tensor quant kinds, tokenizer/template, backend, dtype, cache
format, residency/offload policy, context range and supported public API.
Family support is a tested capability matrix, not a promise that every size,
quantization, modality and context length works. Include dense and MoE paths
where the chosen family variants require them; missing specimens stay open.

Each row has two distinct references:

1. Semantic reference: independent model implementation for architecture,
   tokenization, logits, greedy continuation and state behavior. Use HF/PyTorch
   where it supports the exact model; for GGUF use an independent implementation
   that reads the exact file, with llama.cpp as an explicitly named comparator.
   Gesso CPU is an internal isolation oracle, not independent model validation.
2. Performance reference: a pinned PyStack pipeline, including version,
   configuration, precision, placement and request settings. A llama.cpp GGUF
   result and a Gesso CUDA result remain separate columns. Neither substitutes
   for the PyStack release gate. If exact GSQ-RCO execution has no equivalent
   PyStack configuration, record the incompatibility and a separately labeled
   pipeline comparison; do not describe it as identical-weight/precision parity.

Freeze prompt IDs, generation length, sampling settings/seeds, context lengths,
thread counts, memory budget and warm/cold protocol before optimization. Use
short and multi-token first requests (including 64 and 256 tokens), a longer
prefill case, steady decode, sampled decode and multi-turn reuse. Choose the
long-context row from the documented supported range, rather than inheriting a
CUDA-only 32k claim. Store raw outputs and exact commands with receipts.

Correctness requires finite outputs, independent numerical tolerances justified
per path, exact tokenizer IDs, greedy checks and explicit investigation of
near-tie differences. Declare any approximation and its bound before admission.
Nondeterministic corruption never becomes an acceptable tolerance.

Performance parity means Lava/reference throughput >= 1.0 for both prefill and
decode, and reference/Lava warm TTFT >= 1.0, for every mandatory comparable row.
Report distributions from at least five measured warm repeats, not the best run;
an uncertainty interval that straddles parity is inconclusive. Alternating run
order and matching resource limits prevent an accidental advantage. Cold load,
compilation, first request and memory are reported separately; freeze their
release budgets in campaign 3 before tuning. Do not average away a failing
family, prompt length or sampling mode. Historical 90–95% targets do not replace
Matt's current “at least parity” release bar.

## Campaigns and sprint order

| Campaign | Sprints | Exit | Dependency |
|---|---|---|---|
| 1. Trust the Lava baseline | 1A isolate/fix L1; 1B integrate local fixes and retest | Deterministic correct Ornith baseline on the shipping Lava fork | First |
| 2. Complete family coverage | 2A inventory/reference matrix and Gemma recovery; 2B Qwen closure; 2C Gemma closure | Every declared family row passes import, numerical and state gates | Inventory can start now; device graduation follows 1 |
| 3. Reach measured parity | 3A freeze benchmark contract and attribute; 3B ordinary optimizations; 3C full matrix rerun | Mandatory rows meet PyStack parity with correctness intact | 1 and each relevant row of 2 |
| 4. Prove the real workflow | 4A server/family contracts; 4B Cyan + Palette; 4C soak/recovery | Real model conversations, streaming, tools where supported, cancellation and recovery pass | Contract audit can start now; live Lava proof follows 1/2 |
| 5. Make the release reproducible | 5A packaging/dependencies; 5B portability; 5C release review | Reproducible install, bounded support claims and complete release receipts | Audit can start now; final exit requires 1–4 |

These are work streams, not permission to run simultaneous heavy jobs. Execute
one measured job at a time and one writer per worktree. The default execution
frontier is 1A. Campaigns advance only when their predecessor gates have evidence.

### Campaign 1 — trust the Lava baseline

**1A objective:** fix intermittent p64 corruption at its source. Run the existing
single-thread probe first. Passing at one thread is a clue, not proof of a host
race: it also changes scheduling and compilation. If corruption persists,
compare block outputs against CPU, then isolate the earliest bad operation.
Do not default an optimization off or prime the model with t1 as the fix.

Start in the existing L1 worktree, preserving its probe diff. Build against the
environment named in the handoff (`env-l1-lavafix`). Initial permitted code paths
are `ext/lava_ops.jl`, `ext/lava_hybrid.jl`, `ext/lava_quant.jl`,
`src/Inference/Inference.jl`, `src/Inference/moe_host.jl`,
`src/Inference/gguf_quant_matrix.jl`, `src/Inference/quant_cpu_multi.jl` and
`test/test_lava_l1.jl` (new regression if the isolated failure warrants it).
Narrow the final fix to the demonstrated cause; revise the work item before
editing outside this envelope. Persistent experiment scripts/logs stay in the
handoff's work area, not temporary directories.

**Interfaces/invariants:** `reference_generate`, Lava operators, host expert
dispatch; same model, route, quant semantics and default overlap setting.
**Tests/exit:** at least five fresh default-env processes, two p64 calls each,
all `82202`; t1/order controls; matching red baseline runs at `37d87da`; Lava
subset with the handoff's 507/507 reference count reconciled against the current
suite. Remove temporary probes from the fix commit. Performance target: no new
speed claim, record runtime and memory regression checks. Artifact: source fix,
regression/evidence, exact failure ledger and receipt.

**1B objective:** combine L1 with W3, S1 and vendored Lava on an isolated branch.
Preserve their individual controls and rerun combined-tree correctness, sampler,
MoE and Lava tests. Recheck fresh-process L1 after integration. Scope: existing
branch deltas plus necessary conflict resolution, not unrelated cleanup.
Artifact: reviewable combined commit list, diff and receipt. Coupon preparation
may follow this exit under the handoff; pushing/merging still requires Matt.

### Campaign 2 — complete family coverage

**2A objective:** turn “Qwen3.5 and Gemma4 supported” into explicit matrix rows.
Audit available local checkpoints, missing references and `lane-gemma` work
without modifying its checkout. Distinguish GemmaAdapter from Gemma4 support.
Recover only relevant work with ownership and prerequisite checks. Freeze the
input modality decision and exact variants before claiming campaign completion.
Artifacts: support matrix, independent fixtures and narrow implementation items.

**2B/2C objectives:** close Qwen and Gemma rows respectively through transport,
architecture, tensor maps, quantization, tokenizer, chat rendering, EOS, prefill,
decode and state reuse. Test CPU then Lava, including dense/MoE cases as required,
partial chunks, non-square head dimensions, and applicable hybrid/sliding state.
Do not rewrite the engine around a model name. Validate the actual serve adapter,
not merely the underlying forward pass. Reject unsupported configurations loudly.

Candidate code areas: `src/Inference/architecture_spec.jl`, `arch_adapters.jl`,
`materialize_architecture.jl`, GGUF/tokenizer/template and family inference files,
the necessary `ext/lava_*.jl` methods, `src/serve/engine.jl` and focused tests.
Each implementation item must name exact files before edits; this campaign is
not blanket authorization over those directories. Interfaces: existing importer,
operator dispatch, tokenizer and generation contracts. Invariants: semantic
identity, explicit capability refusal, no silent backend or precision switch.

Exit: independent tokenization and numerical/continuation evidence for every
mandatory row; fresh/reused/reset state checks; malformed/unsupported metadata
and nonfinite controls; actual public entry-point proof. Missing models or
references are blocked rows, not successful skips. Performance target here:
establish comparable baselines, do not claim parity yet.

### Campaign 3 — reach measured parity

**3A objective:** implement the acceptance protocol above as a reproducible
benchmark matrix. Attribute load, compilation, host compute, device compute,
transfers, synchronization, sampling and state binding separately. A profiler
that inserts synchronization must be labeled and paired with end-to-end timing.
Compile-time thread sums are not elapsed wall time.

**3B objective:** remove measured costs in bounded, individually revertible
changes. Start with refreshed W4, V4 and W2 evidence; rank by end-to-end impact.
Candidates include device router work, generic K-quant kernels, data movement,
buffer lifetime/reuse and batching already supported by the execution contract.
Hot expert residency requires declared rounding/history behavior and correctness
admission; CUDA results alone do not certify a Lava implementation. Existing
fingerprinting/state validation must not be weakened merely to improve timing.
Compiler forks, new speculative schemes and exotic representations stay parked.

Scope: named benchmark drivers/results, only the source hot spots demonstrated
by 3A, and independent equivalence/regression tests. Interfaces: existing backend
capabilities and operator/workload dispatch. Every optimization item freezes its
file list, before measurement and numerical gate before implementation.

**3C exit:** rerun the complete mandatory matrix on the integrated tree; attach
versions, file digests, raw distributions, correctness outputs and memory/compile
costs. All mandatory comparable PyStack rows meet the acceptance contract.
A faster CUDA column or llama.cpp column cannot graduate the Lava/PyStack gate.
If a reference cannot run within the shared budget, report that row separately
and leave its parity claim unproven rather than assigning an infinite speedup.

### Campaign 4 — prove the real workflow

**4A objective:** validate family-specific chat rendering and the supported
OpenAI-compatible endpoints with real models. Cover streaming boundaries, EOS,
sampling/seed handling, tool serialization where supported, thinking modes where
supported, context exhaustion, interrupted clients and recoverable backend errors.
The current Qwen renderer/parser must not silently serve a Gemma conversation.

**4B objective:** complete a real Cyan conversation using Gesso/Lava with Palette
connected: first multi-token request, streamed response, a permitted local tool
round trip, follow-up turn and cancellation followed by a successful new turn.
Record the actual endpoint, backend and model identities. No health-only proof.
Read user configuration as needed; Matt applies cutover values. Do not send
messages to external recipients as an integration test.

**4C objective/exit:** bounded soak with at least 100 sequential requests plus
cancellation/disconnect/recovery cases, defined before the run; record failures,
host/device memory trends and context limits. Verify serialized GPU ownership
and rejection/queue behavior under overlapping clients. Do not promise concurrent
device execution or continuous batching without separate workspace/ownership proof.

Code envelope: serving files, its launcher and focused server/engine tests;
family changes return to campaign 2. Live integration uses installed entry points.
Palette T9 and its matrices at seeds 7/41/2026 are a supporting track. First
triage the unverified source/ANSI quoting additions and resolve the 11 port-test
failures with their red controls; do not quietly fold them into Gesso changes.
Artifact: API contract, real-workflow transcript/receipt and complete soak ledger.
Performance target: preserve campaign 3 bounds and report serving overhead.

### Campaign 5 — make the release reproducible

**5A objective:** reproduce installation and inference from the release tree in
a fresh environment and relocated checkout. Vendored Lava must carry provenance,
license and the approved patch. Unmodified upstream packages remain installable
dependencies. Inventory all native helpers, including ggml, their versions/ABI
assumptions and failure modes; ship a reproducible dependency path or replace the
required fragile seam before release. “Works with my cached build” is not an exit.

**5B objective:** verify portable capability selection and documented resource
limits. Provide a portable CPU path and fail-closed unavailable Vulkan capability
tests. Exercise the Lava smoke/parity subset on a second GPU/device class where
available, with actual device receipts. Without a second device, label hardware
portability unverified and the release validation range explicitly; do not claim
cross-vendor certification. Avoid fixed device IDs, host thread counts and model
paths in shipping configuration; explicit tunable budgets are allowed.

**5C objective/exit:** package documentation, support matrix, exact reproduction
commands, dependency/license notices, known limits and release evidence. Run the
documented local test entry point and formatting checks in the isolated release
tree; preserve real-model failures and skips by name. Check public changed files
for developer paths, excluding untouched upstream vendor content. Reconcile stale
maps, README claims and debt entries using evidence, not wholesale deletion.

Scope: package/workspace metadata, vendor provenance, installation scripts,
release docs and focused packaging/capability tests; no production credentials,
model deletion or unrelated checkout changes. Artifact: reproducible candidate
plus receipt; release/push is Matt's final action under the coupon rules.

## Milestones, controls and ownership

- **Usable development milestone:** campaigns 1 and 2, plus the real workflow
  from 4B. This enables use/testing; it does not waive the performance release bar.
- **Release candidate:** all five campaign exits, including the declared modality
  scope and every mandatory parity row, have complete evidence.
- **Exotic research opens:** only after the release bar is met; separate campaign,
  same correctness/portability contract, baseline retained for comparison.

All implementation sprints use `.github/ISSUE_TEMPLATE/work-item.md`: objective,
exact permitted files, interfaces, invariants, tests, performance target and
expected artifact. Each close has the §LXXII receipt: what changed, why, tests,
numerical delta, before/after timing, compile/memory impact, hardware, workload,
model and backend. Planning is complete when the campaign order and gates exist;
implementation completion requires final captured test results.

Operational rules from the handoff persist: one job under the timed-job lock,
at most 16 threads, GPU budget 4800 MiB during the current experiments, durable
logs, no sudo, no killing Palette sessions, no broad stash or cleanup deletion.
Keep Palette/Cyan configuration and other owners' worktrees intact. No remote
push, history rewrite cutover, main-checkout reset or worktree removal is implied
by this plan. Preserve Matt's unrelated main-checkout changes. Public commits use
plain messages without coauthor trailers or developer machine paths.

## Next executable step

Active work item: [Campaign 1 correctness and stress](CAMPAIGN1_LAVA_CORRECTNESS.md).

Continue from the [current Campaign 1 checkpoint](CAMPAIGN1_STATUS_2026-10-14.md),
which supersedes the handoff's initial thread-race hypothesis and probe order.
The selected shared-store source candidate still needs the fresh-process,
recovery, boundary, integration and measurement gates in the active work item.
Retain the rejected controls. Do not begin throughput tuning while L1 is open.

## Planning receipt

Change: one current release plan and pointers from the two existing roadmap
entry points. Reason: the phase map and architecture recipe do not capture the
current family/parity release objective. Validation: source/branch reconciliation,
relative-link and machine-path checks, campaign/dependency review and whitespace
check. No runtime source changed; no inference tests or benchmarks were run.
Numerical delta, runtime/compile/memory impact and model/backend measurements:
not applicable to this documentation change. Open decisions and unverified
claims remain explicitly recorded above.
