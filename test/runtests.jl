# Gesso test harness (Gesso_Stack.md §LXXIII).
#
# Structure: this file owns the package-level laws (dependency law, load,
# foundation contracts) and includes per-area test files. Tests are cheap,
# CPU-only, and deterministic — GPU/differential tests arrive with their
# phases, gated on hardware availability.

using Gesso
using Test
# Phase 2 oracle surface, unqualified for the per-area test files
using Gesso: reference_prefill, reference_generate

@testset "Gesso" begin
    include("test_foundation.jl")
    include("test_errors.jl")
    include("test_receipts.jl")
    include("test_semantics.jl")
    include("test_parameters.jl")
    include("test_operators.jl")
    include("test_empty_core.jl")
    include("test_backends.jl")

    # correctness-laboratory fixtures (goal §H): equipment, not API —
    # deterministic seeds, tolerance mechanics, parity diagnostics.
    include("testhelpers.jl")
    include("test_helpers.jl")

    # toy-model fixture pack (Phase 1 ready-room): data-only laboratory
    # material for Phase 2 — see test/fixtures/toy/README.md
    include("toyfixtures.jl")
    include("test_toyfixtures.jl")

    # CPU reference operator methods (§LXXV item A): needs the lab helpers
    include("test_cpu_ops.jl")

    # ModelIR tests build toy2 from the fixture pack, so they come after it;
    # the Phase 1 exit test composes the whole chain (fixture → IR → tensors
    # → named operators) and therefore comes last
    include("test_modelir.jl")
    include("test_phase1_exit.jl")

    # Phase 2 (§LXXV): the prefill oracle over the fixture pack, then greedy
    # generation with a real KV append
    include("test_reference_prefill.jl")
    include("test_reference_generate.jl")

    # Phase 3 (§LXXVI item A): GQA interpreter, final RMSNorm, eps/theta knobs
    # — defaults are the Phase 2 constants, so every earlier gate is unchanged
    include("test_gqa.jl")

    # Phase 3 (§LXXVI item B): config → ModelIR, safetensors reader, Llama
    # name map — the micro checkpoint is generated in-test by a writer that
    # mirrors the reader byte-for-byte
    include("test_import_llama.jl")

    # Phase 3 (§LXXVI item C): GPT-2 byte-level BPE against a tiny data-only
    # fixture (golden ids traced from the merge table in the file header)
    include("test_tokenizer_gpt2.jl")

    # Phase 3 (§LXXVI item D): the REAL-model gate — one named skip unless
    # GESSO_SMOLLM2_DIR points at a local snapshot; it never downloads
    include("test_smollm2.jl")

    # Phase 4 (§LXXVII item A): the CUDA backend seam — manifest law, ext
    # binding, explicit no-device failure. Load-order-sensitive: BEFORE/AFTER
    # the CUDA import. Device tests live in their own files (B/C/D).
    include("test_cuda_seam.jl")

    @testset "package loads" begin
        @test Gesso.Log isa Module
        @test isdefined(Gesso, :CPUBackend)
        @test isdefined(Gesso, :glog)
        @test isdefined(Gesso, Symbol("@gfallback"))
    end

    @testset "dependency law (§VII: Gesso earns every hard dependency)" begin
        # Phase 0 law: the core package has NO third-party hard dependencies.
        # Julia stdlibs are permitted one at a time, each with a justification
        # entry here. Backends arrive as package extensions (CUDA → Phase 4,
        # Lava → Phase 8), never as core deps. Adding a dependency requires
        # editing this test. That friction is the point.
        # §LXXVI sanctioned the ONE third-party exception: JSON, for
        # config.json + safetensors header parsing.
        stdlib_allowlist = Dict{String, String}( # name => justification
            "Dates" => "timestamps for structured log events (§XLII)",
            "LinearAlgebra" => "CPU reference matmul (§LXXV Phase 2)",
            "Mmap" => "safetensors byte region (§LXXVI)",
            "JSON" => "config.json + safetensors header parsing (§LXXVI)",
        )
        project = joinpath(pkgdir(Gesso), "Project.toml")
        section = ""
        declared = Dict{String, Vector{String}}()
        for line in eachline(project)
            if startswith(line, '[')
                section = strip(line, ['[', ']'])
            else
                m = match(r"^\s*([A-Za-z0-9_]+)\s*=", line)
                if m !== nothing && section in ("deps", "weakdeps")
                    push!(get!(declared, section, String[]), String(m.captures[1]))
                end
            end
        end
        unexpected =
            [d for d in get(declared, "deps", String[]) if !haskey(stdlib_allowlist, d)]
        @test isempty(unexpected) ||
              "undeclared hard deps: $unexpected — justify them here or remove them" == ""
        # weakdeps: backends arrive as package extensions (§VII). Each entry
        # is justified here; the extension is the ONLY code allowed to load it.
        weakdeps_allowlist = Dict{String, String}( # name => justification
            "CUDA" => "Phase 4 backend (§LXXVII), package extension GessoCUDAExt — never a core dep",
        )
        unjustified_weak = [
            d for d in get(declared, "weakdeps", String[]) if !haskey(weakdeps_allowlist, d)
        ]
        @test isempty(unjustified_weak) ||
              "unjustified weakdeps: $unjustified_weak — justify them here or remove them" ==
              ""
    end

    @testset "logging conventions (§LXX, §XLII)" begin
        cfg = Gesso.current_config()
        old_io, old_level = cfg.io, cfg.min_level
        buf = IOBuffer()
        try
            cfg.io = buf

            # debug dropped at default min level (§: events below min are gone)
            Gesso.glog(Gesso.Log.LOG_DEBUG, :should_be_dropped; x=1)
            @test isempty(take!(buf))

            # structured events carry event name + key=value context
            Gesso.glog(Gesso.Log.LOG_INFO, :plan_selected; op=:rmsnorm, backend=:cpu)
            s = String(take!(buf))
            @test occursin("[info]", s)
            @test occursin("plan_selected", s)
            @test occursin("op=rmsnorm", s)
            @test occursin("backend=cpu", s)

            # min_level! returns the previous level
            prev = Gesso.min_level!(Gesso.Log.LOG_DEBUG)
            @test prev === Gesso.Log.LOG_INFO
            Gesso.glog(Gesso.Log.LOG_DEBUG, :now_visible; k=42)
            @test occursin("now_visible", String(take!(buf)))
            Gesso.min_level!(Gesso.Log.LOG_INFO)

            # §LXX: fallbacks are recorded, and the macro returns the fallback
            val = Gesso.@gfallback(:lava, :cuda)
            @test val === :cuda
            s = String(take!(buf))
            @test occursin("[warn]", s)
            @test occursin("fallback", s)
            @test occursin("requested=lava", s)
            @test occursin("actual=cuda", s)
        finally
            cfg.io, cfg.min_level = old_io, old_level
        end
    end

    @testset "backend interface (§XX, §XXI, §LXX)" begin
        cpu = Gesso.CPUBackend()
        @test cpu isa Gesso.AbstractGessoBackend
        @test Gesso.backend_name(cpu) === :cpu
        @test Gesso.execution_tier(cpu) == 0   # tier 0: PORTABLE_CORRECTNESS

        # §XX: capability probing is always safe; unknown capabilities are
        # false, never an error
        @test Gesso.supports(cpu, :definitely_unknown_capability) == false

        # §LXX: lowering stubs fail explicitly and identify themselves —
        # no silent substitution, ever
        @test_throws Gesso.LoweringNotImplemented Gesso.rmsnorm!(cpu, nothing)
        @test_throws Gesso.LoweringNotImplemented Gesso.matmul!(cpu, nothing)

        err = try
            Gesso.softmax!(cpu, nothing)
            nothing
        catch e
            e
        end
        @test err isa Gesso.LoweringNotImplemented
        @test err.op === :softmax!
        @test err.backend === :cpu
        @test occursin("explicit failure", sprint(showerror, err))
    end

end
