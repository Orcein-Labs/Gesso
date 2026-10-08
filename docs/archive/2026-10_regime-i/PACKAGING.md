# Regime I package reproduction

Lava/Vulkan is the primary device backend. CUDA is an optional throughput
comparison. Core package dependencies remain Dates, JSON, LinearAlgebra and
Mmap; backend packages stay in the optional test workspace.

Use Julia 1.12.6 and the exported workspace Manifest.toml. The tested Lava
revision is `11c7e31bdf62408d22bf379e9e59510f69d2103e`, whose Git tree is
`795df8f1fb847d261f4da4d944448c4d96683862`. The installed cache was compared
against every one of the commit's 272 tracked files; all match. The Vulkan
and VulkanCore source revisions in test/Project.toml remain pinned.

For the source bundle, extract it and keep Manifest.toml at its root. Set
GESSO_SMOLLM2_DIR to an existing local HuggingFaceTB/SmolLM2-135M checkpoint;
models are intentionally absent from this package. Set GESSO_HF_REFERENCE to
the independent `hf-reference.json` receipt. Then:

```sh
JULIA_PKG_PRECOMPILE_AUTO=0 julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/prepare_lava_compiler.jl
make test
make format-check
```

`make test` covers device tests when hardware is usable and records named
skips otherwise. A skip alone does not constitute device certification.
The explicit campaign probes require successful primary-device execution:

```sh
julia --compiled-modules=existing --project=test test/regime_i_package.jl cpu cpu.json "$PWD"
julia --compiled-modules=existing --project=test test/regime_i_package.jl lava lava.json "$PWD"
```

The campaign's fresh-install gate relocates the package and uses an initially
empty writable depot. Read-only lower-depot inputs expose only cached source,
artifacts, registries and clones, with no inherited compiled Gesso/Lava cache.
Instantiation is offline. The pinned LLVM/GPUCompiler prerequisites are
precompiled into that new writable depot before device probes use
`--compiled-modules=existing`. No inherited compiler cache is copied. Without
this preparation, source-loaded LLVM pointer operations infer Any and the
primary shared-memory reduction cannot compile; that failed probe is retained.
The helper uses a temporary project against the same workspace lock and does
not add core dependencies, resolve versions or precompile Lava itself.
The pinned Lava source overwrites a KernelAbstractions adaptation method;
default automatic precompilation can report that overwrite and fall back to
source loading. Successful Lava precompilation is not certified. The explicit
source-execution command above avoids attempting to generate that cache;
actual inference and the documented full test entry point are verified.
This proves a fresh, cache-backed local install; it is not a network clean-room
or certification of every operating system, AMD hardware, larger model,
8k full-model context, sampling mode or native FP16/BF16 arithmetic.

CPU arithmetic is Float64. Primary Lava storage and normalization are Float32.
The auxiliary CUDA comparison stores Float32 and retains legacy Float64
normalization intermediates; its receipts state that difference.
Greedy output IDs are exact against the independent reference; full-vocabulary
logit comparisons retain atol=0.01, rtol=0. Context prefixes through 129 and
rotary-table positions through 32768 are verified separately.

The final receipt index records successful probes and preserves failed
experiments, before measurements and source identity. No Regime II work
is included, and the canonical/shadow development trees are left untouched.

The full actual-host suite also retains three inherited expected-broken
`@inferred` checks for reference prefill and decode, packeted on the unchanged
semantic `storage::Any` hierarchy. They are deferred type-stability limits,
not unavailable-device skips or newly passing tests.
