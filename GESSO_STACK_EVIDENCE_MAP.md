# GESSO_STACK_EVIDENCE_MAP.md

An evidence-backed map of the Gesso stack, its lineage, its current implementation truth, and the boundaries it collapses or preserves.

**Compiled:** 2026-10-03, from the working trees of `Gesso` @ `c68cb7e`, `Palette` @ `1a245ec`, `WGE` @ `c99a069`.
**Method:** read source, tests, benchmark corpora, goal documents, and git history. Where a claim could be checked by running code, I ran it. Claims about hardware performance are quoted from the repo's own receipts and are marked as such, because I had no NVIDIA device attached during this pass.

**Claim classification used throughout:**

| Label | Meaning |
|---|---|
| `DEMONSTRATED` | Code exists and a test or measurement in this repo exercises it |
| `PARTIAL` | Implemented, exercised only on a subset of cases |
| `UNTESTED` | Implemented, no direct test found |
| `PLANNED` | Specified in canon or a goal doc, not implemented |
| `SPECULATIVE` | A hypothesis in a research doc with no implementation |
| `REJECTED` | Attempted, abandoned, and the reason recorded |

---

## I. RESEARCH QUESTION

Gesso's own canon states the question in its first screen, before it defines a single type: it is "no longer being conceived as 'a Julia replacement for PyTorch'" (`docs/Gesso_Stack.md`, STATUS block). The replacement target sits in `§LXIII`, which lists nine things Gesso is not, including PyTorch, llama.cpp, and LangChain rewritten in Julia.

The operational question the project actually runs on is narrower and testable. From `§LXIV`:

> Does Gesso make this a natural system composition rule where PyStack would require bespoke graph rewriting, tensor subclasses, compiler plugins, custom kernels, hooks, metadata plumbing, multiple runtimes, and application-level orchestration?

The second principle (`§LXV`) and the first (`§II`, "Never discard semantic information earlier than necessary") give the mechanism. The doc's own summary line is the memorable version: the model should not lose its meaning before it reaches the machine.

For the larger research thesis this map supports, the question is a counterfactual one about design space rather than about software quality: which architectural boundaries in contemporary ML are load-bearing properties of the problem and which are artifacts of a Python-shaped toolchain. Gesso is the experimental platform used to find out, and the interesting part is that it has produced at least as much evidence about what does *not* work as about what does.

`DEMONSTRATED`: the question is written down, scoped, and gated. `SPEED_FLOOR.md` §1 makes it conditional on evidence rather than on intent.

---

## II. CURRENT STACK MAP

Four repositories and one document umbrella. Only three carry live git history. `Project Cyan` is a directory of specs plus a symlink into Palette.

### Ownership, as declared in `§XLIII`

```
Gesso     = mechanism      Palette = expression      Cyan = policy and cognition
```

The canon is explicit that Gesso owns agent execution primitives, task graphs, inference scheduling, cross-agent batching, budgets, priorities, cancellation, lifecycle, shared model resources, hardware resources, receipts, and runtime semantics. Palette owns a condensed agent-facing surface and must not duplicate Gesso's scheduler. Cyan owns when to spawn agents, roles, modes, cognitive architecture, memory strategy, tool policy, and planning behavior.

The test that makes this boundary real is in `docs/research/CYAN_TRIAL_GESSO_FALLOUT.md`. After a full end-to-end Cyan/Palette operator trial, three candidates had to pass an affirmative promotion rule (generality, composability, ecosystem leverage) before entering Gesso. All three were deferred. The document's §4 is titled "NEGATIVE FINDING (LOAD-BEARING)" and says the scarcity of Gesso fallout is itself the data: the trial did not show the Julia ML substrate needs redesign, it showed the young surface is the machine-facing interactive experience.

### What each repo actually contains

| Repo | Role | Live code size | Notes |
|---|---|---|---|
| `Gesso` | Julia semantic ML and inference runtime | 8,162 lines across `src/` and `ext/` | 5 core deps (4 stdlib + JSON), 2 weakdep extensions |
| `Palette` | Persistent sandboxed Julia operator surface | ~8 `.jl` files in `src/`, Rust host in `host/`, Python authority fence in `security/` | Deps are all stdlib except JSON |
| `WGE` | Game engine: Rust authority + Julia/Lava execution | 8,665 lines of Julia (`graphics_lab` 7,505, `terrain_lab` 1,160) | Julia is deliberately *not* the authority |
| `Project Cyan` | Harness specs and freeze documents | No code of its own | Birth commit `01732e1`, extracted from Palette at `8eeadd6a8` |

### The one architectural fact that matters most for the thesis

WGE inverts the Gesso arrangement on purpose. Its `ACTIVE_ARCHITECTURE.md` sets the envelope as Rust for "semantic authority, identity, contracts, runtime, certification" and Julia for "numerical/spatial execution behind typed packets." The boundary rule is stated bluntly: "Julia does not promote semantic state or invent unbound identity." Graphics packet execution is "coarse-grained; no per-frame semantic FFI choreography."

This matters because it is counter-evidence to a naive reading of the thesis. Within the same project, on a different subsystem, the team chose to *keep* a language boundary rather than collapse it, and gave a reason (authority and identity belong in a language with a strong type system and no ambient mutation). Any argument that Julia removes boundaries should have to explain why WGE re-introduced one.

`DEMONSTRATED`: both boundaries exist in code and in tests.

---

## III. EXISTING JULIA SUBSTRATE

The project runs a hard dependency law (`§VII`: "Gesso earns every hard dependency"), enforced by a test in `test/runtests.jl` that I saw pass in my run. Gesso's total third-party dependency count is one: `JSON`. `Dates`, `LinearAlgebra`, and `Mmap` are stdlibs with written justifications. CUDA.jl and Lava arrive as package extensions, so core compiles and tests with neither loaded.

That is admirable discipline, and it is also a finding about the ecosystem composition question. Counting mentions across Gesso's canon, goal docs, and research docs:

| Julia component | Mentions | How it is used |
|---|---|---|
| CUDA.jl | 32 | The strategic fast path and the G2 speed horse |
| KernelAbstractions | 25 | Named as the *initial execution substrate* of the companion autotuning project, and the type Gesso aliases around when Lava collides |
| Adapt | 18 | Storage/parameter adaptation across backends |
| GPUCompiler | 16 | In the compiler-attribution ladder |
| GPUArrays | 9 | The mechanism that makes the Lava backend work without handwritten kernels |
| SnoopCompile, PrecompileTools | 7 each | Named on the untried rungs of the compiler ladder |
| LLVM | 7 | Same ladder |
| JET | 4 | Static analysis |
| Flux | 1 | Listed in the archived `Harpe_Stack_old.md` as a checkpoint origin |
| Lux | 1 | Same |
| Enzyme | 1 | Listed under things explicitly out of scope |

Two things follow.

First, the honest finding is that the ecosystem composition here is **CUDA.jl and GPUArrays, and very little else**. Flux, Lux, Enzyme, DifferentialEquations, MLIR tooling, and distributed primitives have no presence in the current tree. The companion autotuning spec (`docs/Native_Julia_Kernel_Autotuning_North_Star_README.md`) is where the ecosystem survey actually happened, and it did the survey properly: it identifies `KernelAbstractions`, `AcceleratedKernels`, `KernelForge`, and notes that `KernelTuner.jl` currently wraps the *Python* Kernel Tuner via `PythonCall`/`CondaPkg`. That last observation is a genuinely useful data point for the thesis, since it shows a Python dependency surviving inside a Julia tool through a convenience seam.

Second, Gesso's `Profiling` module carries an explicit law: "no CUDA in this module." The separation of mechanism from backend is real at the module level, not just aspirational.

`PARTIAL`: the substrate story is well argued in the companion spec and thin in Gesso itself. If the thesis claims Julia-ecosystem composition, the evidence for that is currently one autotuning document, not an architecture.

---

## IV. CHRONOLOGICAL TECHNICAL LINEAGE

The name history matters because it records two separations. From `docs/ARCHITECTURE.md` §I and §XLIII:

- **NeuraJL** became **Palette** (operator surface). Commit `54f0728b`, "rename: NeuraJL to Palette."
- **Harpe** became **Gesso** (mechanism). Commit `c65a56c`, 2026-09-29, "Rename the public product from Harpe to Gesso."
- **NIRA** is the internal codename for Cyan policy.

`docs/Harpe_Stack_old.md` (1,481 lines) survives as the predecessor vision and is marked "superseded where they disagree." It contains the checkpoint-transport principle that survives into current canon: parameters from Hugging Face, Lux, Flux, Harpe training, or any other exporter are all equivalent, because "checkpoint format is transport" and "execution semantics are separate."

### The eras

**Era 0, documentation and process (pre-2026-09-28).** `docs/Gesso_musings.md` (1,433 lines) is kept and labeled "the dangerous notebook." `docs/Native_Julia_Kernel_Autotuning_North_Star_README.md` (2,270 lines, dated 2026-09-22) is the companion autotuning project spec. Research programs for KV memory (Magenta) and caged representation exist as design documents with explicit phase gates.

**Era 1, foundation (2026-09-28).** Commit `6b999c6` lands Phase 0 with the canon docs and the KV memory program. `f79fead` adds "BONES sprint: swarm-readiness scaffolding." Then a hardening sequence that is itself evidence about how the work is done: thread-safe receipt sink with stress tests (`422d2a5`), a pinned failure taxonomy (`4d688d4`), freeze machinery described as "evidence, not ceremony" (`aea6d60`), and a correctness laboratory with fixtures (`ed138e3`).

**Era 2, the semantic core (2026-09-28 to 09-29).** `2b00b92` lands the §CIX object-model encoding "from the parallel canon session." Phase 1 follows in three commits (workload types, ModelIR, operator dispatch), then Phase 2 in three (CPU reference ops, prefill oracle with known logits, greedy generate). The rename lands mid-stream.

**Era 3, real models and real backends (09-29 to 09-30).** Phase 3 imports SmolLM2-135M via GQA, a Llama importer, and a GPT-2 BPE tokenizer. Phase 4 adds CUDA as a weakdep extension. Phase 5 builds the paged KV manager and the `Session` engine. Phases 6 through 9 land in about 36 hours: observability, copy-on-write prefix share, the Lava/Vulkan backend, and the autotune loop.

**Era 4, the speed floor (10-01 to 10-03).** This is the densest and most informative stretch. Ten consecutive sprints (10, 10B, 10C, 10D, 10E, 10F, 10G, 10H) plus three BREADTH batches and a capability conformance pass.

**Era 5, model breadth (10-03).** BREADTH-0, 1, and 2 land in three consecutive commits on one day, followed by CAPABILITY-0a.

Total: 78 commits in Gesso between 2026-09-28 and 2026-10-03.

### What the commit rate hides

The history is six days long and contains twenty-three numbered work items. Each is a closed recipe with its own goal document, its own fences, and its own §LXXII receipt. This is the "RPD" method the docs describe as a closed-recipe spiral with receipts, skip-or-green gates, and fail-closed oracles.

`SPEED_FLOOR.md` §0 puts a limit on how far that travels: RPD "is not a performance technology. It is a performance *discipline*." The same file warns against advertising RPD as throughput numbers.

---

## V. MAJOR ARCHITECTURAL PIVOTS

Five pivots, each reconstructed as initial assumption, implementation, observed behavior, friction, revised abstraction, current consequence.

### Pivot 1: the compatibility matrix lied, twice, in both directions

**Assumption.** That a report about model support can be derived from one representative configuration per architecture family.

**Implementation.** BREADTH-0 (`ca5383e`) landed six architecture families and a generated `compatibility_matrix`. Read on its own it said every family had zero missing capabilities.

**Observed behavior.** BREADTH-1 (`2a3136f`) opened with the reason: "That is the number a status meeting quotes. It was also false of the world." The matrix probed one default config per family. The same machinery asked about a real Llama-3 checkpoint answered correctly and named `rope_llama3` as not implemented.

**Friction.** The fix swept 48 configurations per family and reported the worst case alongside the runnable fraction. `gemma` and `llama` both read 1/48 runnable. Then BREADTH-1 made the CPU backend run scaled RoPE, and the matrix kept reporting `:rope_linear` and `:rope_llama3` as unreachable. The report had overcorrected into understating, one commit later.

**Revised abstraction.** BREADTH-2 (`39c055a`) made `supports(backend, capability)` the single authority and deleted a second hand-written capability list that had been contradicting it. The matrix gained a backend dimension, because "implemented" was never a global property. Runnable went from 1/48 to 3/48, which is the honest number, since CPU genuinely runs the three rope kinds.

**Current consequence.** Four real gaps remain: `dense_ffn`, `qk_norm`, `moe_routing`, `sliding_window_attention`. Two fences guard both directions, and a device-free probe backend must produce a strictly narrower matrix than CPU or the backend argument is deemed decorative.

**Then CAPABILITY-0a (`c68cb7e`) found two more defects in the code that claimed to close the seam.** One report function re-derived a capability list it already held, dropping `moe` and declaring a MoE spec fully runnable. And the conformance test's first arm skipped capabilities no spec could require, so two claimed capabilities were never proven. The arm was also branching to `@test_broken` when a claim had no probe, and a mutation check proved the defect: adding a fabricated capability left the suite green, because Julia counts `broken` as a non-failure.

**Classification:** `DEMONSTRATED`. All of it, including the mutation checks, which are the strongest evidence in this repository that the reporting is trustworthy.

### Pivot 2: the capability existed and the door was shut

**Assumption.** That `Session` refusing any scaled RoPE policy at construction was correct.

**Implementation.** BREADTH-0's Pass D made the CPU oracle handle `:linear` and `:llama3` policies, with `Session` throwing `LoweringNotImplemented` at construction.

**Observed behavior.** BREADTH-1 put it plainly: the capability existed and the door was shut. The refusal had been correct while the engine threaded only `theta`, since a scaled policy running unscaled is exactly the silent representation change `§LXX` forbids. Once the oracle had the capability, refusal became over-caution.

**Friction.** The specific danger recorded was a model with scaled RoPE running silently unscaled. No error, wrong numbers, a direct law violation.

**Revised abstraction.** `Session` now reads the policy once and threads `inv_freq` to every `rope!`. Refusal moved to the dimension rather than the policy: `interleaved=true` and partial `rotary_dim` still fail closed.

**Current consequence.** The test suite asserts the policy is not decorative. Scaled and unscaled output must differ, or "engine matches oracle" could pass because both ignore the policy and agree on the same wrong answer.

**Classification:** `DEMONSTRATED`, and I verified the BREADTH-1 testset passes in my run.

### Pivot 3: five sprints of allocation archaeology, ending in an observability bug

This is the best material in the repository for the thesis, because the overhead turned out to be in the measurement apparatus itself.

**Assumption.** That warmed `decode!` allocation was dominated by engine work.

**Implementation.** Phase 10D through 10G, each with a `Profile.Allocs` attribution table.

**Observed behavior, in order.** 10D baseline: SmolLM2 CUDA warmed `decode!` allocated 5,986,816 B. 10E gave the Session an owned decode workspace and made `gather_kv!` write in place, cutting CPU allocation to O(1) in seqlen (15,864,048 to 100,480 B on SmolLM2 CPU) while CUDA barely moved, to 5,094,736 B. 10F found the residual was `@views dst[:, hh, :] .= src[:, slice]` on `CuArray` views, which allocates a fresh `Broadcasted` tree per head per layer per token. Replacing those with `CuArray` specializations of the same core copy helpers took `Inference.jl` to exactly 0 B against a warmed CUDA `decode!` and the total to 1,269,616 B. Still 221,040 B over the declared 1 MiB ceiling.

**Friction.** 10F's receipt states the ceiling was not raised, the miss was carried as a tracked `@test_broken`, and the sprint's own invariant ("0 SmolLM2 Broken beyond the three P-1") was knowingly broken by exactly one. The file says that was recorded rather than papered over.

**Revised abstraction.** 10G (`PHASE10G_AUTOTUNE_RECEIPT.md`) then found the last 320,509 B was not engine work at all. `Autotune.select` re-emitted a full `§LXXII` receipt on every cache *hit*, so one cached decision was being copied 633 times per token. The fix is a sentence: "a cache hit is not a decision." `select` now returns the cached `TuneResult` and emits nothing, and the miss receipt still names the winner.

**Current consequence.** 942,128 B, which is 106,448 B under the declared ceiling, at a ceiling that was never moved.

**Classification:** `DEMONSTRATED`. Then 10H closed a packet about type stability, amending §CIX so the array type holding a tensor is one trait-equivalent type parameter. The measured side effect cut allocation again, to 814,544 B.

### Pivot 4: the board gate reversed the project's own priorities

**Assumption.** That exotic representation research was the point and could proceed in parallel with engine work.

**Implementation.** The Magenta memory program and the caged representation program were written first, in September, with detailed phase gates.

**Observed behavior.** On 2026-10-01 a board constraint landed, recorded verbatim in `SPEED_FLOOR.md`: "Exotic work is the point. A board will not accept exotic work on an engine that is not even somewhat close in performance to the practical Python inference stack on a named model."

**Friction.** Every timing row in the corpus was `toy2` or `llama_micro`, which the document calls "correctness fixtures. They are not a product conversation."

**Revised abstraction.** Three conjuncts, G1 ∧ G2 ∧ G3, each with receipts required. G1: a named model generating through `Session`. G2: same-box eager-PyTorch comparison published as a factor, with first-token and warmed clocks reported separately because they are different clocks. G3: the fast path must still match CPU-oracle greedy ids, fail closed, and preserve `fork` copy-on-write.

**Current consequence.** `§LXXXIII` and every exotic phase stayed parked for the whole run. The gate also specifies that a fused kernel which wins Autotune and breaks `fork` "is not a winner. Disqualify it."

**Classification:** `DEMONSTRATED` as a process. The G2 factor itself is `PARTIAL` and needs care, covered in §XI.

### Pivot 5: the headroom measurement and its thrown-away first attempt

**Assumption.** That at SmolLM2 scale, decode time was dominated by arithmetic or memory traffic.

**Implementation.** A ladder of synthetic but real Llama-shaped checkpoints from 1M to 843M parameters, written as safetensors plus `config.json` by the protocol `test_import_llama.jl` already uses, so the importer takes them with no special casing. Weights are random, and the document says so: this measures time and host allocation only.

**Observed behavior.** A fit of `ms = 0.797 + 0.6528·L + 6.03e-08·L·h²` gave per-session fixed cost 0.797 ms, per-layer launch floor 0.653 ms, and compute 6.03e-08·L·h². At SmolLM2 (h=576, L=30): overhead 19.58 ms against compute 0.60 ms, so **97.0% of decode is per-layer launch latency**. Crossover against compute sits at h ≈ 3291, past any demo-sized model.

**Friction, and this is the part worth keeping.** The first attempt at the table was wrong and was thrown away. Thirty measured tokens per model, taken right after warmup with multi-second gaps between models, read 60 GB/s and 186 GB/s on bandwidth trials at 10.9 W of a 145 W limit. The GPU sits at `pstate P8` and 405 MHz memory clock when briefly loaded and needs seconds of *continuous* work to promote to boost. The fix was a sustained 2 s ramp before each timing, 100 measured tokens, and clocks printed per row as evidence.

**Revised abstraction.** A different question than the ladder rung it sits beside, and the document is explicit that it is not a substitute for it. It asks only whether steady-state decode is arithmetic-bound, bandwidth-bound, or neither.

**Current consequence.** Measured achieved triad bandwidth on this box is 371 GB/s, against a vendor spec near 1.8 TB/s. SmolLM2 decode moves 27.7 GB/s, about 7.5% of achieved. The document says the honest reading is that the real ceiling is at or below 371 GB/s, because decode's access pattern is worse than a streaming triad, so the percentage is an optimistic bound.

**Classification:** `DEMONSTRATED` as a measurement with a stated method and a documented discarded attempt. The inference drawn from it is `SPECULATIVE`, and the document labels it that way.
---

## VI. CURRENT IMPLEMENTATION TRUTH

### Demonstrated

| Capability | Evidence | Verification |
|---|---|---|
| CPU Float64 reference oracle with frozen golden logits | `src/Inference/Inference.jl` (`reference_prefill`, `reference_generate`); `test/fixtures/toy/expected_logits.toml` stamped `oracle = "cpu"`, `commit = "b8a5988"` | Passes in my run |
| Llama-family import with GQA, safetensors via `Mmap`, fail-closed on unrecognized names | `src/Inference/llama_import.jl` | Passes; handles HF 0-based and fixture 1-based origins, refuses mixes |
| Paged KV manager with derived filled length and per-page provenance | `src/Inference/kv_manager.jl` (429 lines) | Passes |
| `Session` engine matching oracle ids | `src/Inference/session.jl` (805 lines) | Passes; ids equal oracle on toy2, llama_micro, SmolLM2 |
| Copy-on-write prefix sharing via `fork` | `src/Autotune/../Inference/session.jl`; `Profiling.unique_kv_bytes` | Passes; 2048 vs 4096 B on llama_micro pair, 1,474,560 B N on SmolLM2 |
| CUDA.jl backend as weakdep extension | `ext/GessoCUDAExt.jl` (160 lines), `ext/cuda_ops.jl` (502 lines) | Seam tests pass; device rows unverified by me |
| Lava/Vulkan backend as weakdep extension | `ext/GessoLavaExt.jl` (188 lines), `ext/lava_ops.jl` (405 lines) | Seam tests pass; device rows unverified by me |
| Autotune as gated candidate selection | `src/Autotune/Autotune.jl` (421 lines) | Passes; CUDA `matmul!` selects `:cublas_mul` |
| Six architecture families through one doorway, Phi-3 executed end to end | `arch_adapters.jl` (636 lines), `architecture_spec.jl` (354 lines) | 1105/1105 assertions in repo run |
| Capability-report biconditional, mutation-verified | `test/test_capability_conformance.jl` | Passes in my run |
| Dependency law: one third-party dep | `Project.toml` + test in `runtests.jl` | Passes in my run |

### Partial

- **Device backends.** CUDA and Lava capability *sets* are exercised by the seam tests and a synthetic probe backend, but no conformance probe runs against a real device. `CUDABackend` requires CUDA loaded and the operators need hardware. The repo says this plainly in three separate receipts.
- **Lava.** A portable seam, not tuned. Its implementation uses GPUArrays broadcasting plus Lava's `mul!` and defines zero `KernelAbstractions` kernels. The prefill row in the 2026-10-03 corpus is 14.0 ms against CUDA's 9.5 ms on the same shape.
- **Second-family execution.** BREADTH-0 verified Phi-3 on the CPU F64 oracle only. CUDA and Lava reachability for a second family is described as a recipe, not a claim.
- **The compat matrix.** Sweeps five declared axes and reports worst case, but the variant set is a declared list rather than derived from configs, and it models no quantized checkpoints.

### Untested or lightly tested

- **`Planned`, not implemented:** `src/Lowering/Lowering.jl` is 13 lines, `Planning` 15, `Runtime` 15, `Agents` 14, `CAPI` 14, `Representation` 20. All six are contract-only. `test/test_empty_core.jl` originally fenced them as empty, then inverted its own job after Phase 1 landed: it now pins exactly what each `§CIX` module contains so nothing can be added to the semantic core outside a work item.
- **`Session` does not execute scaled RoPE on CUDA or Lava.** Both accept the `inv_freq` kwarg and refuse at the operation.
- **No end-to-end scaled-RoPE checkpoint has been run.** Proofs use toy2 with a synthetic policy. No Llama-3 weights were available.
- **The metaspace tokenizer fixture is synthetic.** BREADTH-0 authored it: 256 byte fallbacks plus a merge table, with ids derived from the published algorithm. It is not traced from a production SentencePiece model, and the document declines to pretend otherwise.

### Rejected or abandoned

- **Fixed-capacity KV cache.** Rejected in favor of the logical filled shape, pinned by test.
- **A second contiguous KV buffer.** Rejected on 2026-09-30 as a locked decision: pages are the cache and reads gather.
- **Unsharing on fork-of-fork.** Deliberately conservative, no refcount, so a shared page is never un-shared.
- **A `Phi3Runtime` type.** `§VIII` forbids it. The fused-QKV problem was solved with canonical identity plus a *declared* layout instead.
- **`scaled_rope!` as a new operator.** Rejected: a policy is a value, and `rope!` with frequencies covers it.
- **Registration API for capabilities.** Considered and rejected in CAPABILITY-0a, on the reasoning that it is another declaration maintained the same way, while the biconditional makes a declaration *checkable*.
- **Julia compiler fork.** Explicitly deferred until a measured hole exists that stock Julia cannot close.

---

## VII. DEMONSTRATED EXOTIC CAPABILITIES

Four things in this stack actually work and would be awkward elsewhere. Each has a receipt.

### 1. Device-side greedy argmax: one integer per token

On the hot path the host receives a single `Int` per token. The `(vocab,)` logits row never crosses back. `session.jl` branches on `Gesso.supports(cpu, :argmax)` and calls `_device_greedy_id`, with the host-row path preserved for backends lacking the capability. GPUArrays argmax is first-index-tie deterministic, which is what makes this safe against a CPU oracle.

Why it is interesting beyond Gesso: the boundary crossed per token goes from *vocab-sized* to *one word*. A conventional eager loop materializes and ships that row every step.

`DEMONSTRATED` on CUDA per the corpus, unverified by me.

### 2. Copy-on-write KV sharing with honest accounting

`fork(session)` aliases prefix pages, and the next write into a shared page copies that page alone. Three properties make this trustworthy rather than a demo:

- **Declared, never discovered.** Two sessions that prefill the same tokens independently never share. This is pinned by test.
- **The win is measured in bytes, and the negative case is measured too.** The corpus carries both rows side by side: `kv_bytes_two_isolated_prefill_cpu` = 4096 B and `kv_bytes_prefill_fork_cpu` = 2048 B, with the note that the saving is the difference between the two rows.
- **Measured on the named model too.** SmolLM2 CPU N = 1,474,560 B, where fork equals one session and two isolated sessions equal 2N.

`DEMONSTRATED`.

### 3. One operator vocabulary across three heterogeneous backends, expressed as plain Julia

This is the collapsed-boundary finding with the strongest evidence behind it. Gesso supports CPU `Array`, CUDA `CuArray`, and Lava `LavaArray` behind the *same* operator functions. The Lava backend is the strongest case, because it defines no kernels at all. GPUArrays broadcasting plus Lava's `mul!` expresses all six operators, with RoPE via strided pairwise views and causal softmax via an index mask.

The whole visible seam is one name collision. Lava.jl exports its own `LavaBackend <: KernelAbstractions.GPU`, so Gesso aliases the KA type as `KALava` inside the extension and keeps `using Gesso, Lava` valid. That is a genuinely small amount of glue.

`DEMONSTRATED`: greedy ids match the CPU oracle on all three, within declared tolerances (CPU `atol=0`, Lava declared `1e-3`, CUDA `1e-2` against an F64 oracle).

### 4. Persistent, sandboxed, long-lived Julia as an operator surface (Palette)

Palette runs an unrestricted `julia` inside a bubblewrap sandbox, keeps state across calls, parses each whole call before any of it runs, and kills a call that never yields, then revives the kernel and reports what was lost. Agents reach it through an MCP server.

This is the boundary-collapse case in its purest form. There is no Python orchestration layer, no serialization at the tool boundary beyond the wire, and no subprocess per operation. The host broker sits in a *different* trust domain on purpose, reachable only through a single bind-mounted Unix socket, and it performs broker-mediated effects itself with host privilege so that "the worker cannot grant itself capabilities" is a structural fact rather than a policy the worker is trusted to honor.

What makes this evidence rather than a claim is the honesty ledger. `docs/research/KNOWN_LIMITS.md` has seventeen numbered limits written against their own system, including that value identity is not tracked so a mutation with the same type is invisible to the diff, that imported bindings are entirely absent from the capture eye, that `kernel_pid` is the only generation witness and pid recycling would fool it, and that receipt accumulation is unbounded by design.

`DEMONSTRATED` for the mechanism, with a documented adversarial test program (`CyanTunnel/endurance/rally.py`, roughly six minutes).

---

## VIII. CANDIDATE EXOTIC CAPABILITIES

Everything here is `SPECULATIVE` or `PLANNED`. The `EXOTIC_CAPABILITY_MASTER_LEDGER.md` is explicit that it is "a capability mine, not a roadmap override," and the file states it is not permission to implement Time Machine, Foundry, Magenta, cages, or a Julia fork.

| Capability | Ledger status | Current state |
|---|---|---|
| Logical identity separate from physical representation (`LogicalAttentionMemory`) | Supporting primitive | The paged KV manager is a real instance; the general object is not built |
| Branches as a work DAG | Post-launch R&D | Nothing beyond `fork` |
| Availability separate from residency | Post-launch R&D | Magenta Part II, 1,962 lines, exploratory, pre-data |
| Demand-driven materialization | Post-launch R&D | Concept only |
| Cross-agent batching and shared-prefix agent execution | Named in `§XXXVII`, `§LIV` | Not implemented; `Runtime` is 15 lines |
| Speculative execution as a general runtime capability | `§LV`, Phase 20 | Explicitly "research only until benchmarked" |
| Policy-driven compilation (`MinLatency()`, `AccuracyBound(ε)`) | `§LXI` | Long-term interface sketch |
| Quantization as lowering, gauge-compiled / caged weights | `REPRESENTATION_PROGRAM.md` | Parked behind G1 ∧ G2 ∧ G3; `Representation` is 20 lines |
| Rate-distortion optimization over lawful KV transforms | `KV_MEMORY_PROGRAM.md` Part I | Part I accepts a direction, blocks publication until a composition experiment produces numbers |
| Neurosymbolic parameters | `§LVI` | "Future possibility." Explicitly not "every scalar is a symbolic object" |
| Plastic / lifecycle weights, FluidGEMM seam | `§LVII` | Extension points preserved, no dependency taken |
| Elastically orchestrated multi-agent inference | `CYAN_ELASTIC_ORCHESTRATION.md` | Seam pointer. `src/Agents.jl` stays empty by instruction |

One pattern is worth naming because it recurs across programs. `KV_MEMORY_PROGRAM.md` §1 tracks a single invariant, "X is a lowering decision," against four subsystem decompositions and marks each as confirmed, planned, or starred. The document's own reasoning is that "a thesis that keeps generating valid subsystem decompositions is probably true." That is a weak form of evidence and the doc knows it.

---

## IX. PYTHON-SHAPED BOUNDARIES

For each boundary below: what conventional stacks treat as fixed, what this stack did, and the honest read on whether the boundary was actually load-bearing.

### 9.1 The graph as an optimization boundary

Conventional stacks route model → generic tensor ops → graph → graph analysis → attempt to recover patterns → compiler → backend. Gesso's `§II` names this chain and prefers logical model → semantic object → legal physical alternatives → measured lowering.

Evidence: `ArchitectureSpec` consumes meaning rather than the `model_type` string, which is demoted to provenance. Parameter identity is `SemanticParamId`, and BREADTH-0's central finding is that identity is meaning while spelling is one adapter's choice.

**Read:** partially collapsed, and this is where the strongest architectural claim lives. The evidence is real for model import. Whether it extends to optimization proper is untested, since `Lowering.jl` is empty.

### 9.2 The model-family whitelist

Before BREADTH-0, an unsupported model produced `model_type "qwen2" is not "llama" - not Llama-shaped, refusing`. It died at the door and told the user nothing actionable. Now a model travels as far as the implemented semantics allow and stops at a *named operation*. A Qwen2 config reports its missing `qk_norm` as the failure boundary.

Phi-3, with fused QKV and a declared per-KV-group packing, ran end to end with zero edits to `session.jl`, `Inference.jl`, `kv_manager.jl`, or any operator.

**Read:** genuinely collapsed, and cheaply. Adding family number six is a `<Family>Adapter`, a capability declaration, a conformance fixture, and any genuinely new operators.

### 9.3 Serialization at the operator boundary

Palette's `src/rlm.jl` makes recursive children from Julia code. Each call is one host request through the session's broker, and the shapes follow `prime-agent-runtime/src/rlm/__init__.py` "so both kernels behave alike against one host."

**Read:** partially collapsed, honestly. A wire protocol survives, and the doc says why: two kernels must behave alike against one host. That is a convention-driven boundary, not a fundamental one.

### 9.4 Rust as the host for Julia systems

Both Palette and WGE keep a Rust layer. Palette's host is a kernel launcher and supervisor with a bubblewrap sandbox. WGE puts semantic authority in Rust and Julia behind typed packets, and states the rule "Do NOT rewrite good Rust code merely to become 'all Julia'" (`§XLV`).

**Read:** *preserved*, with reasoning. Rust owns identity, authority, and certification because those need strong types and no ambient mutation. Gesso canon adds the guidance to use a C ABI and `@ccall` rather than a rewrite, and says the goal is "remove unnecessary Python hops, NOT conduct a language purity crusade."

This one cuts against the thesis and should be presented as such.

### 9.5 The tokenizer as a package dependency

SentencePiece was implemented natively rather than added, on the reasoning that a tokenizer is transport machinery and `JSON` already covers its file format.

**Read:** collapsed, though the gain is small. Worth reporting as a composition choice, not a capability.

### 9.6 Capability declaration as a hand-written table

Gesso's own reporting layer had this boundary, and BREADTH-0 through CAPABILITY-0a spent four commits collapsing it. `supports(backend, capability)` is now the single authority, with a biconditional test proving claimed capabilities execute and executing capabilities are claimed.

**Read:** this one is the most transferable finding in the repository. It is language-independent, and the failure mode (a generated report contradicting the code in both directions, undetected across a commit) is exactly the kind of thing the thesis should be able to talk about without naming a language.

---

## X. ACTUAL HARD BARRIERS

Places where Julia does not save anyone, taken from the repo's own measurements.

### 10.1 The launch floor is the binding constraint, and it is architecture-shaped

At SmolLM2 scale, 97.0% of decode time is per-layer launch latency at 0.653 ms per layer, with arithmetic at 0.60 ms. The overhead is linear in layer count and nearly independent of width, so it dominates everything a demo would run. Crossover against compute sits at h ≈ 3291.

Classification: `architectural` and `implementation`, in that order. The host-side interpreter loop in `Inference.jl` orchestrates layers in a serial `for` loop. `SPEED_FLOOR.md` row 5 calls this "fine for correctness. Death for decode tok/s until the inner step is one (or few) device graphs."

**Julia does not fix this.** The barrier is that the host cannot submit the inner step as a graph, and the fix is CUDA graphs or a fused decode step, both of which are on the "later packet" list.

### 10.2 46.9 tok/s at 135M is roughly an order of magnitude below a fused runtime

`SPEED_FLOOR.md` §2a.0 states it directly, and attributes the gap entirely to per-layer overhead rather than to anything semantic.

### 10.3 Torch is not on the demo box, so the factor is one point

The G2 factor could not be re-measured at any size. The only Llama-shaped model present is SmolLM2-135M. The other cached snapshot has `model_type: "needle"` and is not loadable. The document says the factor "is a single point on one model and should be quoted as exactly that."

### 10.4 The compile/execute attribution ladder is still open

Rung 1 of the compiler ladder requires splitting every run into source load, package load, inference, specialization, LLVM, native emit, GPUCompiler, GPU execute, and D2H. `SPEED_FLOOR.md` marks it "STILL OPEN" and states that no Julia compiler fork begins until it exists for SmolLM2 decode on the demo box.

Classification: `implementation` and `language/runtime-induced`. This is the clearest place where the host language's own compiler is the constraint.

### 10.5 Lava does not precompile on Julia 1.12

Noted as "a compiler-layer finding we already paid for," and it is why the Lava load is about a minute from source.

### 10.6 Measurement on this hardware has a trap that already caught one measurement

The GPU sits at `pstate P8` and 405 MHz memory clock when briefly loaded. A first bandwidth attempt read 60 GB/s and 186 GB/s at 10.9 W of a 145 W limit. Every subsequent number uses a sustained ramp with clocks printed per row.

Classification: `hardware/physics`, with a `measurement` failure mode layered on top. Honest methodology note, and one that any replication has to reproduce or it is measuring the wrong thing.

### 10.7 Backend reachability for a second family is unproven

CUDA and Lava capability sets were edited but not executed in the breadth work. The per-backend proof uses a synthetic probe backend, because comparing real device matrices needs a device.

### 10.8 Six days of history, no long-run evidence

78 commits in Gesso. Nothing in this repository speaks to what happens at month six, under model churn, with multiple models resident and a real scheduling load. `§XXXII` scheduler, `§XXXVII` cross-agent batching, and continuous batching are all unimplemented, and `generate` still resets.

---

## XI. PERFORMANCE AND EFFICIENCY EVIDENCE

### The G2 factor, stated carefully

From `benchmark/results/2026-10-03.tsv`, on an RTX 5060 with host `cachyos-x8664`, Julia 1.12.6, one thread:

| Row | Value |
|---|---|
| `smollm2_eager_pytorch_warmed` | 347,842,463 ns (median of 5, after the script's own warmup) |
| `smollm2_gesso_cuda_warmed` | 239,630,452 ns (median of 21) |
| `smollm2_g2_factor_eager_over_gesso` | 1.452× |

Three caveats travel with the number, all recorded in the corpus itself:

1. **Cross-dtype.** Gesso computes in F32, eager PyTorch in bfloat16 from `from_pretrained` defaults. The row states "both stamps travel with the numbers, no cross-dtype claim."
2. **The comparison is eager, not optimized.** No `torch.compile`, no vLLM. `SPEED_FLOOR.md` says the gate is eager deliberately, "because that is the stack a board member can reproduce in an afternoon."
3. **Greedy, batch 1, one prompt.** "Hello", `max_new_tokens=8`, `context_length=128`.

One more thing cuts against a comfortable reading of the factor. The first-token clocks are 5.92 s for eager against 1.02 s for Gesso, which flatters Gesso on a clock that includes kernel compile. That is why the corpus publishes both and `SPEED_FLOOR.md` insists they stay separate: "TTFX is a compiler problem; steady-state tok/s is a kernel/fusion/transfer problem."

**The factor's own trajectory is the most informative number in the corpus:**

| Sprint | Date | Factor |
|---|---|---|
| 10B | 10-02 | 0.692× |
| 10E | 10-02 | 0.737× |
| 10F | 10-03 | 1.452× |

The 10E entry is the one that matters for interpretation. The repo's own note reads "the ratio moved against us while both sides got faster," since Gesso warmed went 630 to 487 ms and eager went 436 to 359 ms in the same run. A factor that moves 2× in a day from allocation work alone is a moving target, and the 1.452× should be quoted as one measurement rather than a level.

### Allocation, which is where the real work went

Warmed `decode!` host allocation, `@allocated`, after warmup:

| Workload | Backend | 10D | 10E | 10F | 10G | 10H | Gate |
|---|---|---|---|---|---|---|---|
| toy2 | CPU | 89,120 | 10,736 | 10,608 | unchanged | unchanged | 16 KiB met |
| llama_micro | CPU | 139,984 | 8,528 | 8,400 | unchanged | unchanged | 16 KiB met |
| SmolLM2 | CPU | 15,864,048 | 100,480 | 98,560 | unchanged | unchanged | 256 KiB met |
| toy2 | CUDA | 233,952* | 219,104 | 97,312 | unchanged | unchanged | 256 KiB met |
| llama_micro | CUDA | 305,712 | 278,112 | 100,736 | unchanged | unchanged | 256 KiB met |
| SmolLM2 | CUDA | 5,986,816 | 5,094,736 | 1,269,616 | 942,128 | 814,544 | 1 MiB met |

`*` measured during 10E, not published by 10D.

The SmolLM2 CUDA row is the whole story of five sprints: a 7.4× cut in seven days, from 5,986,816 B to 814,544 B, with the ceiling at 1 MiB never raised. The mechanism at each step was different. 10E removed per-token rebuilding. 10F removed `Broadcasted` wrapper allocation by specializing core helpers on `CuArray`, taking `Inference.jl` to 0 B. 10G removed an observability bug where a cached decision was receipted 633 times per token. 10H got type stability by making storage one type parameter, and allocation fell again as a side effect.

CPU seqlen independence is also demonstrated: 8,400 B at seqlen 4 and at seqlen 8, and the seqlen-16 figure is a one-time page allocation equal on first and repeat measurement.

**Classification:** `DEMONSTRATED`. This is the strongest quantitative result in the repository and it is fully attributable by `Profile.Allocs` with per-site byte and alloc counts.

### Where measurement is missing

- **The decode ladder rung 1.** Compile/execute decomposition is open. Without it, no claim about TTFX is actionable.
- **Device-verified G2 on anything but one model.** torch is not installed on the demo box.
- **Fused-kernel numbers.** No fused decode has been built, so the 97%-overhead finding has no counterfactual. The best available statement is the one already made: 46.9 tok/s at 135M, an order of magnitude below a fused runtime.
- **Lava tuned at all.** 14.0 ms against CUDA's 9.5 ms on `llama_micro` prefill, with Phase 9 tuning never applied to Lava.
- **Batch, concurrency, multi-model residency.** Unmeasured because unimplemented.

### Architectural reasons for expected wins, not yet measured

Three, in descending order of plausibility.

**Launch-bound overhead is beatable by a host that can express the inner step as one call.** Julia's multiple dispatch lets a whole layer become a single `@cuda` block or a CUDA graph launch, which is exactly what a Python-orchestrated loop cannot do without writing a custom kernel or reaching for a compiler. The measurement says there is 19.58 ms of overhead per token to attack at SmolLM2 scale.

**Persistent-kernel latency is a language property.** Palette demonstrated it as a mechanism: state across calls, no subprocess per operation. Gesso's analogue has not been built, and `§XXXIII` through `§XLII` are all unimplemented, so this is a hypothesis with one adjacent data point.

**Selection among homogeneous candidates is cheaper to express here.** Autotune does gated candidate selection per (device, backend, operator, shape regime) with correctness gating before timing. It currently selects between two CUDA `matmul!` paths, which is not a win. The repo says so: selection happened, and no tuned-kernel or speed claim is made.

---

## XII. RESEARCH BRANCHES

| Branch | Name and location | Maturity | Gate |
|---|---|---|---|
| Attention-memory as a lowering | `KV_MEMORY_PROGRAM.md` (523 lines, "Magenta Memory") | `PLANNED`, accepted direction, pre-data | Part I blocks publication until a composition experiment produces numbers |
| Attention topology, residency, retrieval | `KV_MEMORY_PROGRAM_part2.md` (1,962 lines) | `SPECULATIVE`, exploratory synthesis | Waits on Phase 10 plus a real KV working set on SmolLM2 |
| Gauge-compiled / caged weights (TetCage-derived) | `REPRESENTATION_PROGRAM.md` (347 lines) | `PLANNED`, parked | Behind G1 ∧ G2 ∧ G3. `Representation` is 20 empty lines |
| Exotic capability mine | `EXOTIC_CAPABILITY_MASTER_LEDGER.md` (1,551 lines) | `SPECULATIVE`, parked | "Strings, not sprints" |
| Plastic weights / FluidGEMM seam | `§LVII` | `PLANNED`, extension points only | Gesso "must NOT depend on PlasticWeights" |
| Neurosymbolic parameters | `§LVI` | `SPECULATIVE` | Long-term. Explicitly not per-scalar symbolic objects |
| Elastic multi-agent orchestration | `CYAN_ELASTIC_ORCHESTRATION.md` (50 lines) | `SPECULATIVE`, seam pointer | "Not a Gesso /goal, permission to fill src/Agents.jl" |
| Native kernel autotuning substrate | `Native_Julia_Kernel_Autotuning_North_Star_README.md` (2,270 lines) | `PARTIAL`, spec plus Phase 9 consumption | Gesso's Phase 9 consumes it |
| Structured dynamic models | `§LX` | `PLANNED` | "Do not implement early. Do not architect them out." |
| Capability vocabulary split (CAPABILITY-0b) | Named in `CAPABILITY-0a_FINDINGS.md` | `PLANNED` | `supports()` still answers semantic, strategy, and representation axes from one function |
| Distributed / sharding as a lowering | Musings only | `SPECULATIVE` | "V1.5+; no program yet" |

### The representation program deserves one note

`REPRESENTATION_PROGRAM.md` has the best framing in the research folder. Its thesis is "DO NOT QUANTIZE THE ACCIDENTAL REPRESENTATION," on the argument that training leaves a coordinate system that is not semantics, so the search should be over the exact functional equivalence class for a minimum-description executable form. Its lateral origin is AMD tetrahedral cages from HPG 2026, and the doc is careful to say the steal is not "tetrahedra for matrices" but "separate the coordinate system of the query from the physical representation of the expensive object."

It also has three firewalled correctness ladders, which is unusual discipline: `ExactBits` (bitwise reconstruction), `ExactOperator` (equality under a declared arithmetic contract), and `ReachabilityExact` (equality only on the visited activation manifold), with the explicit law that a result "may not climb a stronger contract than it proved." The doc names the baseline it competes with rather than claiming a moonshot: DFloat11 at roughly 30% size, and Cloudflare's Unweight at 15 to 22% whole-model on Llama 3.1 8B.

`SPECULATIVE`, with an unusually well-built frame and zero measurements.

---

## XIII. OPEN QUESTIONS AND FALSIFICATION TARGETS

Each of these is a place where the thesis could die, with the measurement that would kill it.

### Falsifiers for the stack's own claims

1. **Does the G2 factor survive a fair dtype?** Re-run with Gesso in bfloat16, or eager in F32. If the factor collapses, it was a dtype artifact and the headline result is void. This is the single highest-value next measurement.
2. **Does the factor hold at scale?** The ladder says Gesso stays launch-bound to h ≈ 3291, which would predict the factor holds or improves against eager, whose per-layer dispatch is worse. That is a labeled hypothesis. Measuring it at 1B and 7B with torch installed settles it.
3. **Does a fused inner step close the 19.58 ms gap?** If a CUDA graph or fused decode does not move the launch floor, then `0.653 ms/layer` is a hardware or driver cost, not an orchestration artifact, and the strongest performance argument in this repository is wrong about its own cause.
4. **Does the second family reach the device?** BREADTH-0 proved Phi-3 on CPU. Run Phi-3 on CUDA and Lava. If the adapter story does not survive a backend change, then §VIII's "adapter exercise, not an engine rewrite" is true only at tier 0.
5. **Do the conformance arms catch a real regression on a real device?** The per-backend proof currently uses a synthetic probe backend. A capability mutation on an actual `CUDABackend` would test whether the biconditional works where it matters.

### Falsifiers for the thesis itself

6. **Is the launch floor a Python-shaped boundary or a universal one?** The strongest available evidence cuts *for* it being Python-shaped, because eager PyTorch's per-layer dispatch is worse. But Gesso's own interpreter loop is a serial `for` in Julia, so Julia is not magically exempt. A CUDA-graph experiment in pure Julia with no Gesso involvement would isolate this cleanly.
7. **Would a careful PyTorch extension reach the same places?** This is `§LXIV`'s own question and the repo never answers it with code. `ArchitectureSpec` plus a capability lattice might be buildable as a PyTorch tensor-subclass plus custom-ops stack. If it is, the architectural claim weakens to an ergonomics claim, which is still worth something.
8. **Does the capability-report finding generalize?** The four-commit arc from BREADTH-0 to CAPABILITY-0a is language-independent. Any codebase with a generated support matrix and a hand-maintained capability list has this defect. That is testable without Julia at all.
9. **Is WGE's preserved Rust boundary a refutation or a refinement?** WGE keeps Julia out of authority on the grounds that identity and certification need strong types. If that reasoning holds generally, then the thesis needs a refined claim: Julia removes *numerical and execution* boundaries, not *authority* boundaries.

### Questions the repo itself flags as unresolved

- Whether `supports` should split its three axes into separate vocabularies (CAPABILITY-0b, deliberately deferred).
- Whether deriving capability sets from dispatch truth is possible (CAPABILITY-0a rejects a registration API for now and names derivation as the natural next step).
- Whether quantized checkpoints fit the compatibility matrix at all. The 48 configurations describe architecture variants, not representation variants, and every real quantization format is a bigger gap than any axis there.
- Whether a family's adapter could expose a config dimension not in the five swept axes.
- Whether Palette's capability-grant-per-generation model survives long sessions, given receipts accumulate monotonically with no trimming mechanism.

---

## XIV. CURRENT DEVELOPMENT DIRECTION

The direction is set by three constraints that are fighting each other in a productive way.

**The speed floor gate holds everything exotic shut.** `§LXXXIII` and every exotic phase stay parked behind G1 ∧ G2 ∧ G3. The gate has been open since 10F on the strength of a 1.452× factor and an intact fork-preserving fast path, which means the current work is no longer the floor.

**Model breadth became the active thread.** Three BREADTH batches and a capability conformance pass landed on 2026-10-03. The direction is explicit: stop teaching Gesso individual models and teach it what a model architecture *is*. The remaining named gaps are `dense_ffn`, `qk_norm`, `moe_routing`, and `sliding_window_attention`, each of which already has a slot in `required_semantics`.

**The compiler ladder is the next structural bet, and no fork has been started.** Rung 1 (attribution) is open. Rung 2 has had its first rung landed by 10H. Rungs for bounded specialization, invalidation control, and a persistent worker remain. The strategic bet recorded in `SPEED_FLOOR.md` rung 4 is that Gesso alone does not justify maintaining a compiler, but that Gesso plus WGE together might, since both demand the same hole. That is the most interesting unfunded idea in the repository.

**Where the process is genuinely strong.** The four-commit BREADTH arc and the five-sprint allocation arc both show the same discipline: find the defect, write the receipt, fence against recurrence, and leave the ceiling where it was. The 10F decision to carry a miss as a tracked `@test_broken` while knowingly breaking the sprint's own invariant, and to say so in the file, is the kind of thing that makes a research program trustworthy.

**Where it is weaker.** Six days of history, one model, one GPU, no device-verified second family, and a test file (`test_decode_scratch.jl`) that has been red across five commits awaiting a decision. Palette's `AGENTS.md` is inherited TypeScript boilerplate from the upstream monorepo that mentions `npm run check` and `packages/coding-agent` in a Julia repository, which suggests charter drift at the harness boundary.

---

## XV. EVIDENCE INDEX

### Repositories

| Repo | HEAD | Commits | Note |
|---|---|---|---|
| `Gesso` | `c68cb7e` 2026-10-03 | 78 | Full history from Phase 0 |
| `Palette` | `1a245ec` 2026-10-01 | 5,013 | Julia work sits on top of an inherited TS monorepo history. The TS line runs through 2025-11; the Julia line begins at `f7f3f2ad0`, "Experiment 002: a minimal, real, adversarially-tested authority fence" |
| `WGE` | `c99a069` 2026-10-03 | 76 | Starts 2026-08-02 as docs separated from Codeweald |
| `Project Cyan` | none | none | Directory of specs; `Cyan Harness` born from Palette at `8eeadd6a8` |

### Gesso source, by size

```
src/Inference/Inference.jl            821    src/Inference/session.jl      805
src/Inference/arch_adapters.jl        636    ext/cuda_ops.jl               502
src/Inference/kv_manager.jl           429    src/Autotune/Autotune.jl      421
src/Inference/llama_import.jl         382    src/Inference/arch_spec.jl    354
src/Inference/import_report.jl        345    src/Inference/tokenizer_protocol.jl  276
src/Parameters/Parameters.jl          227    src/Operators/cpu.jl           217
src/Profiling/Profiling.jl            150    ext/lava_ops.jl               405
ext/GessoLavaExt.jl                   188    ext/GessoCUDAExt.jl            160
src/Lowering          13   src/Planning  15   src/Runtime  15
src/Agents            14   src/CAPI      14   src/Representation  20
```

### The four highest-value documents

1. `docs/Gesso_Stack.md` (3,781 lines). Canon, Roman-numeral sections `§I` through `§XCV`.
2. `docs/research/SPEED_FLOOR.md` (603 lines). The board gate, the nine-row "what is actually slow" table, and the §2a.0 ladder with its discarded first attempt.
3. `docs/research/BREADTH-0_FINDINGS.md` through `CAPABILITY-0a_FINDINGS.md`. Four documents, four days, one defect found after each fix. The clearest causal chain in the repository.
4. `docs/goals/PHASE10F_CUDA_DECODE_ALLOC.md` (683 lines) and `PHASE10G_AUTOTUNE_RECEIPT.md` (430 lines). The allocation arc, including the decision to leave a ceiling unmet and record it.

### Benchmarks

`benchmark/results/*.tsv`, schema v0.2.0 since 2026-09-28. Rows per day: 6, 41, 36, 128, 43, 22. Schema carries `julia_version`, `host`, `arch`, `nthreads`, `commit`, `dirty`, plus a free-text `bench_note` per row, which is why rows can be read without consulting the goal docs.

### Tests

`test/runtests.jl` includes 44 tracked per-area files (45 on disk, counting the untracked WIP). Independent verification this pass:

```
Gesso | 2696 pass | 5 fail | 2 error | 9 broken | 2712 total | 6m53.9s
```

Environment: Julia 1.12.6, `GESSO_SMOLLM2_DIR` pointed at a local `snapshots/SmolLM2-135M` snapshot containing `config.json`, `model.safetensors`, `vocab.json`, and `merges.txt`. No CUDA device attached.

All 7 non-passing results come from `test/test_decode_scratch.jl`, which is untracked (`git status` shows `??`) and calls `Gesso.Inference._workspace_pointers`, a symbol that does not exist anywhere in `src/`:

```
toy2 CPU: ≤ 16 KiB per warmed decode!:                Test Failed
toy2 CPU: ≤ 16 KiB per warmed decode!:                Error During Test (UndefVarError)
llama_micro CPU: ≤ 16 KiB, and NOT growing with seqlen: Test Failed  (×2)
SmolLM2 CPU: ≤ 256 KiB:                                Test Failed
CUDA: workspace reused, host allocation bounded:       Error During Test (UndefVarError)
SmolLM2 CUDA: ≤ 1 MiB:                                 Test Failed
```

Filtering that file out leaves **zero failures and zero errors across the tracked suite**. The four findings documents each name this file as pre-existing WIP awaiting a decision, so the claim in the receipts is confirmed independently.

One incidental observation: my first two attempts at running the suite failed with `ENOSPC` during package precompilation, from contention with live Palette sandbox workers on the same depot. That is an environment fact worth recording, because it means a green suite and a red suite on the same commit can differ for reasons that have nothing to do with the code.

### Adjacent research worth reading in the original

- `docs/Harpe_Stack_old.md` (1,481 lines), the predecessor vision, marked superseded where it disagrees.
- `docs/Gesso_musings.md` (1,433 lines), the "dangerous notebook."
- `Julia Compiler Optimizations for Gesso/docs/the_ancient_texts.md` (1,580 lines), the compiler ladder instrument list.
- `docs/Native_Julia_Kernel_Autotuning_North_Star_README.md` (2,270 lines, 2026-09-22), the ecosystem survey that Gesso Phase 9 consumes.
- `Palette/docs/research/KNOWN_LIMITS.md`, seventeen limits written against Palette's own system.
- `Palette/docs/research/RPD_MASTER_LOG.md`, twelve loop closures with dispositions including PARK and MERGED.
- `WGE/ACTIVE_ARCHITECTURE.md` and `WGE/docs/FRICTION_LEDGER.md`.
