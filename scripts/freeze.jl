# scripts/freeze.jl — build a curated context-freeze bundle (ChatGPT handoff
# + baseline snapshot). Entry: `make freeze PHASE=phaseN`.
#
# Reproduces the manual Phase-0 freeze process: stage curated files, write a
# FREEZE.md briefing + SHA-256 manifest, zip. Excluded: libs/ (local dev
# checkouts), .git, .freebuff, Manifest.toml files, previous bundles.

using Dates

phase = get(ENV, "PHASE", "manual")
date = Dates.format(Dates.now(UTC), dateformat"yyyy-mm-dd")
root = dirname(@__DIR__)
stage = joinpath(tempdir(), "harpe-freeze-$(rand(UInt32))")
zip_path = joinpath(root, "Harpe_$(phase)_Freeze_$(date).zip")

curated = String[
    "AGENTS.md",
    "README.md",
    "Project.toml",
    ".gitignore",
    ".JuliaFormatter.toml",
    "docs/Harpe_Stack.md",
    "docs/Harpe_Stack_old.md",
    "docs/Native_Julia_Kernel_Autotuning_North_Star_README.md",
    "docs/ARCHITECTURE.md",
    "docs/research/KV_MEMORY_PROGRAM.md",
    "Harpe_musings.md",
    "src/Harpe.jl",
    "src/logging.jl",
    "src/versions.jl",
    "src/backends.jl",
    "src/errors.jl",
    "src/receipts.jl",
    "test/Project.toml",
    "test/runtests.jl",
    "test/test_foundation.jl",
    "benchmark/Project.toml",
    "benchmark/runbenchmarks.jl",
    "ci/Project.toml",
    ".github/workflows/ci.yml",
]
src_dirs = [
    "src/Semantics",
    "src/ModelIR",
    "src/Parameters",
    "src/Operators",
    "src/Lowering",
    "src/Inference",
    "src/Runtime",
    "src/Profiling",
    "src/Representation",
    "src/Autotune",
    "src/Planning",
    "src/Agents",
    "src/CAPI",
]

mkpath(stage)
missing_files = String[]
for rel in curated
    src = joinpath(root, rel)
    if isfile(src)
        dst = joinpath(stage, rel)
        mkpath(dirname(dst))
        cp(src, dst; force=true)
    else
        push!(missing_files, rel)   # e.g. ARCHITECTURE.md on first run: fine
    end
end
for d in src_dirs
    isdir(joinpath(root, d)) || continue
    for f in readdir(joinpath(root, d); join=true)
        isfile(f) || continue
        dst = joinpath(stage, relpath(f, root))
        mkpath(dirname(dst))
        cp(f, dst; force=true)
    end
end

# FREEZE.md briefing — regenerated each time so it reflects the live state.
briefing = """
# Harpe — Context Freeze: $(phase)

**Freeze date:** $date (UTC)
**Purpose:** drop-in context bundle for external AI consultation and a
per-phase baseline snapshot.

## How to read this bundle

1. `docs/Harpe_Stack.md` — CANON (sections are Roman numerals; §LXXIII+
   is the phase plan).
2. `docs/research/KV_MEMORY_PROGRAM.md` — the KV/working-memory research
   program (extends §XXXI/§X/§LIX; feeds Phases 5/9/10).
3. `docs/ARCHITECTURE.md` — module map for orientation.
4. `Harpe_musings.md` — the dangerous research notebook. Parking lot,
   not canon. Promotions from it are deliberate.
5. `AGENTS.md` — the binding agent charter.
6. `FREEZE_MANIFEST.txt` — file list + SHA-256 checksums.

## Laws any advice must respect

* Training is OUT of scope permanently (§LVIII) — settled product decision.
* Harpe owns mechanism; NeuraJL owns expression; NIRA owns policy (§XLIII).
* NIRA decides what memory means; Harpe decides how it lives (KV program §8).
* No public performance claims without benchmark data from this repo.
* Citation = claim of having read it (KV program §4.3).

## Verify

```
sha256sum -c FREEZE_MANIFEST.txt   # from inside the extracted folder
julia --project=test test/runtests.jl
```
"""
open(joinpath(stage, "FREEZE.md"), "w") do io
    write(io, briefing)
end

# SHA-256 manifest over everything staged so far (manifest excluded itself).
cd(stage) do
    files = String[]
    for (dirpath, dirs, filenames) in walkdir(stage)
        for fname in filenames
            full = joinpath(dirpath, fname)
            basename(full) == "FREEZE_MANIFEST.txt" && continue
            push!(files, relpath(full, stage))
        end
    end
    files = sort(files)
    open("FREEZE_MANIFEST.txt", "w") do io
        for f in files
            write(io, read(`sha256sum $f`, String))
        end
    end
    run(`zip -r -q $zip_path .`)
end

println("freeze:   ", zip_path)
isempty(missing_files) || println("skipped:  ", join(missing_files, ", "))
println("hint:     move the zip wherever you hand context to external models")
