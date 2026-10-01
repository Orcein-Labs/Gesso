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

    # Phase 4 (§LXXVII item B): CUDA operator methods on CuArray{Float32} vs
    # the CPU oracle at declared atol — one named skip without a device
    include("test_cuda_ops.jl")

    # Phase 4 (§LXXVII item C): full prefill + generate on device vs the CPU
    # oracle — logits at declared atol, token ids EXACT (argmax is the gate)
    include("test_cuda_inference.jl")

    # Phase 4 (§LXXVII item D): the real model on device vs the frozen CPU
    # golden at wider atol — named skip without a device AND GESSO_SMOLLM2_DIR
    include("test_cuda_smollm2.jl")

    # Phase 9 (§LXXXII item B): the two CUDA matmul! candidates — :cublas_mul
    # and :generic_mul — gated against the CPU F64 oracle at the existing CUDA
    # op atol; named skip without a device. Registration happened in
    # GessoCUDAExt.__init__ (the CUDA seam above loaded it).
    include("test_autotune_cuda.jl")

    # Phase 8 (§LXXXI item A): the Lava (Vulkan) backend seam — manifest law,
    # extension binding, name clash, explicit no-device failure. Probes and
    # imports Lava at its own top level, exactly like the CUDA seam above.
    include("test_lava_seam.jl")

    # Phase 8 (§LXXXI item B): Lava operator methods on LavaArray{Float32} vs
    # the CPU F64 oracle at declared atol — named skip without a device. The
    # LAVA_LOADED / VULKAN_OK consts come from the seam file above.
    include("test_lava_ops.jl")

    # Phase 8 (§LXXXI item C): Lava inference — prefill + greedy generate vs
    # the CPU oracle, ids EXACT, logits at declared atol; no-copy law.
    include("test_lava_inference.jl")

    # Phase 5 (§LXXVIII item A): paged KV manager — Magenta §9.5 step 1;
    # pages are the cache, gather-on-read, typed context exhaustion
    include("test_kv_manager.jl")

    # Phase 5 (§LXXVIII item B): the Session engine vs the oracle — exact ids,
    # atol=0 prefill logits, page_size=4 crosses the boundary, EOS, typed errors
    include("test_session.jl")

    # Phase 5 (§LXXVIII item C): CUDA Session — device ids equal CPU ids,
    # logits at declared atol, no-copy law; named skip without a device
    include("test_session_cuda.jl")

    # Phase 5 (§LXXVIII item D): the engine runs the models we already import
    # — llama_micro vs the oracle; SmolLM2 named skip without GESSO_SMOLLM2_DIR
    include("test_session_llama.jl")
    include("test_session_smollm2.jl")

    # Phase 6 (§LXXIX item A): the engine is auditable — one receipt per call,
    # timing/token/KV consistency, failure receipts, ids unchanged
    include("test_session_receipts.jl")

    # Phase 7 (§LXXX item B): Session fork — DECLARED identity prefix share
    # (Magenta §9.5 step 3); forked decode ids equal independent sessions,
    # generate still resets (and drops the share), fork emits no receipt
    include("test_session_fork.jl")

    # Phase 8 (§LXXXI item C): the Lava Session engine — ids equal the CPU
    # engine, fork aliases prefix pages on device, CoW leaves parent storage
    # unmoved; named skip without a Vulkan device.
    include("test_session_lava.jl")

    # Phase 6 (§LXXIX item B): Profiling — KV footprint from the page table,
    # structurally stable reports, empty sink is explicit, no CUDA
    include("test_profiling.jl")

    # Phase 9 (§LXXXII item A): mixed-backend to_device — both extension load
    # orders keep both backends' methods (in-suite order here; Lava-first via
    # a child process). Runs after both seam files so the load-state consts exist.
    include("test_to_device_mixed.jl")

    # Phase 9 (§LXXXII item A): the Autotune loop — register, gate, bench,
    # select, cache, invalidate with pure-Julia candidates. Core machinery:
    # always runs, no device, no backend imports in the loop.
    include("test_autotune.jl")

    # Phase 10 (§LXXXIII gate G2 item D): the eager-PyTorch comparison harness
    # exists and stays honest — source-inspected and --help-probed WITHOUT
    # torch, a snapshot, or a device (CI never requires any of them).
    include("test_speed_floor_harness.jl")

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
            "Lava" => "Phase 8 backend (§LXXXI), package extension GessoLavaExt — never a core dep; not in General registry, test env installs from source",
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
