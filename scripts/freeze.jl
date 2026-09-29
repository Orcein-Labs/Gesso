# scripts/freeze.jl — build a curated context-freeze bundle (ChatGPT handoff
# + baseline snapshot). Entry: `make freeze PHASE=phaseN` (FORCE=1 to overwrite
# an existing same-day bundle).
#
# Reproduces the manual Phase-0 freeze process: stage curated files, write a
# FREEZE.md briefing + SHA-256 manifest, zip.
#
# A freeze is EVIDENCE, not ceremony:
#   * it records which commit it was taken from and whether the tree was
#     dirty (with the dirty file list) — an unmarked dirty freeze would lie;
#   * its manifest is `sha256sum -c`-compatible and generated with the stdlib
#     SHA package (no external checksum tool); ordering is sorted, so the
#     manifest itself is reproducible;
#   * it refuses to silently overwrite an existing bundle (same phase+date);
#     pass FORCE=1 to deliberately replace one;
#   * files missing from the curated list are printed AND recorded in the
#     bundle's FREEZE.md, so a reader knows what the snapshot lacks.
#
# Excluded: libs/ (local dev checkouts), .git, .freebuff, Manifest.toml files,
# previous bundles. The zip stores filesystem mtimes, so the ARCHIVE is not
# bit-reproducible — the manifest content is.

using Dates
using SHA

phase = get(ENV, "PHASE", "manual")
force = get(ENV, "FORCE", "0") == "1"
date = Dates.format(Dates.now(UTC), dateformat"yyyy-mm-dd")
root = dirname(@__DIR__)
zip_path = joinpath(root, "Gesso_$(phase)_Freeze_$(date).zip")

# --- duplicate protection: never silently clobber a historical snapshot ----
if isfile(zip_path) && !force
    error(
        "freeze: $zip_path already exists.\n" *
        "  Refusing to overwrite a freeze silently (it is a historical artifact).\n" *
        "  To deliberately replace it:  make freeze PHASE=$phase FORCE=1",
    )
end

# --- git context: what does this freeze actually snapshot? ------------------
function git_context(root::String)
    head = dirty_files = nothing
    try
        head = strip(read(setenv(`git log -1 --format=%h\ %s`; dir=root), String))
        dirty_files = String.(
            split(
                strip(read(setenv(`git status --porcelain`; dir=root), String)),
                '\n';
                keepempty=false,
            ),
        )
    catch
        return nothing, String[]  # git unavailable — say so, don't guess
    end
    return head, dirty_files
end

head, dirty_files = git_context(root)
is_dirty = head !== nothing && !isempty(dirty_files)

# --- stage -------------------------------------------------------------------
curated = String[
    # charter + canon + research + maps
    "AGENTS.md",
    "README.md",
    "Project.toml",
    ".gitignore",
    ".JuliaFormatter.toml",
    "docs/Gesso_Stack.md",
    "docs/Harpe_Stack_old.md",
    "docs/Native_Julia_Kernel_Autotuning_North_Star_README.md",
    "docs/ARCHITECTURE.md",
    "docs/DECISION_PACKETS.md",
    "docs/goals/PHASE1_SEMANTIC_CORE.md",
    "docs/goals/PHASE2_CPU_ORACLE.md",
    "docs/goals/PHASE3_FIRST_IMPORT.md",
    "docs/goals/PHASE4_CUDA.md",
    "docs/research/KV_MEMORY_PROGRAM.md",
    "docs/Gesso_musings.md",
    # package core
    "src/Gesso.jl",
    "src/logging.jl",
    "src/versions.jl",
    "src/backends.jl",
    "src/errors.jl",
    "src/receipts.jl",
    # test harness (per-area files are part of the contract)
    "test/Project.toml",
    "test/runtests.jl",
    "test/test_foundation.jl",
    "test/test_errors.jl",
    "test/test_receipts.jl",
    "test/test_empty_core.jl",
    "test/test_backends.jl",
    "test/test_semantics.jl",
    "test/test_parameters.jl",
    "test/test_modelir.jl",
    "test/test_operators.jl",
    "test/test_phase1_exit.jl",
    "test/test_cpu_ops.jl",
    "test/test_reference_prefill.jl",
    "test/test_reference_generate.jl",
    "test/test_gqa.jl",
    "test/test_import_llama.jl",
    "test/test_tokenizer_gpt2.jl",
    "test/fixtures/gpt2_tiny/vocab.json",
    "test/fixtures/gpt2_tiny/merges.txt",
    "test/test_smollm2.jl",
    "test/testhelpers.jl",
    "test/test_helpers.jl",
    "test/toyfixtures.jl",
    "test/test_toyfixtures.jl",
    "test/fixtures/toy/model.toml",
    "test/fixtures/toy/tokenizer.toml",
    "test/fixtures/toy/expected_logits.toml",
    "test/fixtures/toy/README.md",
    # benchmark harness
    "benchmark/Project.toml",
    "benchmark/runbenchmarks.jl",
    # dev loop (a cold reader must be able to run what we run)
    "Makefile",
    "scripts/test.jl",
    "scripts/bench.jl",
    "scripts/format.jl",
    "scripts/freeze.jl",
    "scripts/fill_logits.jl",
    # process surface
    ".github/workflows/ci.yml",
    ".github/ISSUE_TEMPLATE/work-item.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
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

stage = joinpath(tempdir(), "gesso-freeze-$(rand(UInt32))")
mkpath(stage)
missing_files = String[]
for rel in curated
    src = joinpath(root, rel)
    if isfile(src)
        dst = joinpath(stage, rel)
        mkpath(dirname(dst))
        cp(src, dst; force=true)
    else
        push!(missing_files, rel)   # recorded in FREEZE.md — the snapshot knows its gaps
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

# --- FREEZE.md briefing — regenerated each time so it reflects the live state
git_line = head === nothing ? "UNKNOWN (git not available at freeze time)" : head
dirty_block = if head === nothing
    ""
elseif is_dirty
    "\n**⚠ TREE WAS DIRTY at freeze time.** Uncommitted changes are NOT\n" *
    "captured by the HEAD hash above; they are whatever the staged files show.\n\n" *
    "Dirty files (git status --porcelain):\n\n```\n" *
    join(dirty_files, '\n') *
    "\n```\n"
else
    "\nTree was CLEAN at freeze time (all staged content matches $head).\n"
end
skipped_block =
    isempty(missing_files) ? "" :
    "\n## Files missing from the curated list (skipped)\n\n```\n" *
    join(missing_files, '\n') *
    "\n```\n"

briefing = """
# Gesso — Context Freeze: $(phase)

**Freeze date:** $date (UTC)
**Snapshot of:** $git_line
$dirty_block
**Purpose:** drop-in context bundle for external AI consultation and a
per-phase baseline snapshot.

## How to read this bundle

1. `docs/Gesso_Stack.md` — CANON (sections are Roman numerals; §LXXIII+
   is the phase plan).
2. `docs/research/KV_MEMORY_PROGRAM.md` — the KV/working-memory research
   program (extends §XXXI/§X/§LIX; feeds Phases 5/9/10).
3. `docs/ARCHITECTURE.md` — module map for orientation.
4. `docs/Gesso_musings.md` — the dangerous research notebook. Parking lot,
   not canon. Promotions from it are deliberate.
5. `AGENTS.md` — the binding agent charter.
6. `FREEZE_MANIFEST.txt` — file list + SHA-256 checksums
   (verify: `sha256sum -c FREEZE_MANIFEST.txt` from inside the extracted
   folder).
$skipped_block
## Laws any advice must respect

* Training is OUT of scope permanently (§LVIII) — settled product decision.
* Gesso owns mechanism; Palette owns expression; Cyan owns policy (§XLIII).
* Cyan (NIRA) decides what memory means; Gesso decides how it lives (KV program §8).
* No public performance claims without benchmark data from this repo.
* Citation = claim of having read it (KV program §4.3).
"""
open(joinpath(stage, "FREEZE.md"), "w") do io
    write(io, briefing)
end

# --- SHA-256 manifest over everything staged (manifest excludes itself) -----
# Format matches GNU sha256sum (`<hash>  <path>`) so `sha256sum -c` works.
# Leading `#` lines are comments accepted by `sha256sum -c`.
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
        println(io, "# Gesso freeze manifest — phase: $(phase), date: $(date) UTC")
        println(io, "# snapshot of: $git_line")
        println(io, "# verify: sha256sum -c FREEZE_MANIFEST.txt")
        for f in files
            hash = bytes2hex(open(sha256, f))
            println(io, hash, "  ", f)
        end
    end
    # --- zip -----------------------------------------------------------------
    zip_bin = Sys.which("zip")
    zip_bin === nothing && error(
        "freeze: `zip` was not found on PATH — it is needed to build the bundle.\n" *
        "  Install it (e.g. `sudo apt install zip`) and re-run. Staged files: $stage",
    )
    run(`$zip_bin -r -q $zip_path .`)
end

println("freeze:   ", zip_path)
println("snapshot: ", git_line)
is_dirty &&
    println("WARNING:  tree was DIRTY — dirty file list recorded in the bundle's FREEZE.md")
isempty(missing_files) || println("skipped:  ", join(missing_files, ", "))
println("hint:     move the zip wherever you hand context to external models")
