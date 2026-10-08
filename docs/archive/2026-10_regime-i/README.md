# Gesso Regime I — verified campaign

Lava/Vulkan is primary. CUDA is a throughput comparator. All 12 sequential
completion gates passed within the tested scope. Regime II was not started.
Palette, Cyan, and the original shadow were preserved during the campaign;
final source hashes and read-only patch checks are recorded in
[the isolation audit](receipts/final-isolation.json). Canonical Gesso now
carries this package.

| Arm | Evidence-backed completion |
| --- | --- |
| 1 Inference | Independent eager Hugging Face/PyTorch full-vocabulary logits and exact eight-token greedy continuations on three prompts. Checkpoint coordinate order, imported RoPE layout and cross-head softmax defects repaired. Corrupted-oracle and old-layout controls failed as intended. |
| 2 Loading | Exact single/sharded safetensors transport, supported F64/F32/F16/BF16 imports and explicit malformed config/tokenizer/tensor refusals. Forty boundary probes. |
| 3 Reproduction | Exact greedy replay across fresh processes and Julia/BLAS thread settings; checkpoint identities pinned. Fifty-seven assertions. |
| 4 Memory | Repeated reset/exhaustion/reload, scratch reuse, owned storage and trimmed reservations. Final real-model Lava lifecycle rerun passed. |
| 5 Devices | Actual CPU and Vulkan execution on RTX 5060, full-logit/token reference gates, strict storage identities and explicit unavailable-driver error. No silent CPU substitution. |
| 6 Numerics | Independent native arithmetic, nonfinite and overflow checks. Gesso's on-device IEEE guard contains Lava's faulty array isfinite predicate. Final Lava numerical revalidation passes 51 checks plus permanent poison/epsilon regressions. |
| 7 Context | Independent full/cached prefixes 17/33/65/129, causal mutations, page edges, scaling and rotary positions through 32768; 46 assertions on each supported path. |
| 8 Scheduling | Ordinary round-robin sequences, variable lengths, cancellation, ownership and isolated callback failures. Mapped-heap exhaustion repaired through pooled temporary placement; the rejected numerical rewrite and unsuccessful GC attempts remain recorded. |
| 9 Containment | Typed malformed inputs, operation/callback interruption, partial-output diagnosis and exact recovery on the same Session. CPU/Lava 29 direct assertions each. |
| 10 Observability | Correct first-token phase timing, tokenization timing, actual backend/dtype, cache accounting, independent replay digests and partial IDs. Faulty or mutating receipt sinks cannot change returned IDs or replace inference errors. Existing allocation ceilings pass. |
| 11 Performance floor | Exact output in every warmed measurement; fixed repeatability gates passed. Same checkpoint, hardware, prompts and host-visible greedy workload. Primary Lava and eager PyTorch normalize in native Float32; the auxiliary CUDA comparator retains legacy Float64 normalization intermediates to meet its stricter toy tolerance. No claim of identical reduction/instruction sequences or a universal throughput target. |
| 12 Packaging | Exact Lava source revision, resolved environment, relocated package and initially empty writable depot; actual CPU and Lava inference. Final full CPU suite: 2772 passed. Actual-host suite: 3113 passed, zero failures, three inherited expected-broken type-inference checks. |

The final primary throughput is **measured and reproducible, but slow**.
Two warmup calls precede three timed samples per prompt; each result is exact
against the independent reference. Loading, transfer and first use are recorded
separately, and compilation is excluded from warmed measurements.

| Prompt | Primary Lava new tokens/sec | Gesso CUDA comparison | PyTorch CUDA eager |
| --- | ---: | ---: | ---: |
| Hello | 2.40 | 29.93 | 21.58 |
| The quick brown fox | 2.49 | 28.81 | 21.90 |
| Julia is a programming language. | 2.49 | 28.80 | 22.26 |

The primary native Float32 normalization correction provides comparable
arithmetic against eager PyTorch; it did not produce a clear primary throughput
improvement. The attempted native CUDA normalization exceeded its unchanged
toy-model tolerance and was rejected; that auxiliary comparator retains
legacy Float64 normalization intermediates, explicitly labeled. Primary host
allocations remain roughly 36–38 MB per warmed decode. These costs are exposed,
not hidden by a speed claim. The ordinary shared registry lookup removes
candidate-list copies and cached-result wrappers while preserving replacement-by-name and explicit stale
winner failures. Receipt storage is reserved before actions and remains bounded
through overflow; the default sink pays roughly 2 MB at construction. The
inherited CUDA allocation ceiling remains unchanged.

Tests use the local SmolLM2-135M checkpoint, Julia 1.12.6 and an RTX 5060.
Fresh installation reuses read-only cached dependency sources/artifacts with
no inherited compiled Gesso/Lava cache; it is not a network clean-room.
The initially source-loaded LLVM pointer intrinsics failed GPU compilation.
Building the same locked LLVM/GPUCompiler prerequisites in the new depot
restores concrete inference; the before/after diagnostic and failed probe
remain recorded. The pinned Lava/KernelAbstractions method overwrite prevents clean automatic
precompilation. Source execution is verified; successful Lava precompilation
is not certified, and the reproduction guide records the explicit command.
Three inherited `@inferred` checks remain expected-broken because the semantic
`storage::Any` hierarchy is preserved. Those are explicit deferred type-stability
limitations, not device skips. Native half/bfloat16 execution, GGUF, sampling, AMD execution, larger-model
coverage, full-model 8k context and multi-host-thread Lava concurrency are not
certified. Lava scheduling is logical sequence concurrency on one host owner;
CPU task contention is tested separately. Standalone rotary-table checks at
32768 are not full-model 32k-context evidence.

[Machine-readable progress and receipt index](receipts/campaign-progress.json)
retain the failed experiments, hardware/environment and source identities.
[Package reproduction](PACKAGING.md) describes the tested environment.
The workspace lock used for that run is archived as [`Gesso-Manifest.toml`](Gesso-Manifest.toml)
(the live package still gitignores Manifest.toml at the repo root).

The campaign-only patch and source tarball remain local transfer artifacts in
`outputs(from chatgpt)/`. They compare the isolated campaign tree against the
carried shadow baseline, not GitHub HEAD. Models, depots, credentials and
`libs/` checkouts are absent. The [change receipts](change-receipts.json)
record the baseline and graduated commit series.
