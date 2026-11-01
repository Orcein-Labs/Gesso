# The precompile workload (src/precompile.jl) runs a synthetic model through
# generate while the package image is built. Whatever state it touches is
# serialized into that image, so a fresh `using Gesso` must see exactly the
# state a never-run package would: lazy caches unresolved, no CPU pool, no
# receipts, default logging.

@testset "precompile workload leaves no state behind" begin
    @testset "fresh process after using Gesso" begin
        script = """
        using Gesso
        const I = Gesso.Inference
        bad = String[]
        I._Q8K_SIMD_MODE[] == -1 || push!(bad, "_Q8K_SIMD_MODE")
        I._GEMV_POOL_MODE[] == -1 || push!(bad, "_GEMV_POOL_MODE")
        I._PREFILL_MULTI[] == -1 || push!(bad, "_PREFILL_MULTI")
        I._GEMV_CHUNKS_PER_THREAD[] == 0 || push!(bad, "_GEMV_CHUNKS_PER_THREAD")
        I._GPU_BUDGET_CTX[] == 0 || push!(bad, "_GPU_BUDGET_CTX")
        I._ROUTING_TRACE_IO[] === :unset || push!(bad, "_ROUTING_TRACE_IO")
        (@atomic Gesso.Runtime._CPU_POOL_SLOT.pool) === nothing || push!(bad, "_CPU_POOL_SLOT")
        isempty(Gesso._PROCESS_SINK.buf) || push!(bad, "_PROCESS_SINK.buf")
        Gesso._PROCESS_SINK.dropped == 0 || push!(bad, "_PROCESS_SINK.dropped")
        c, d = Gesso.Log.GLOBAL_CONFIG, Gesso.Log.GessoLogConfig()
        for f in fieldnames(typeof(c))
            getfield(c, f) == getfield(d, f) || push!(bad, "GLOBAL_CONFIG.\$f")
        end
        println(isempty(bad) ? "PRECOMPILE_STATE_OK" : "PRECOMPILE_STATE_BAD " * join(bad, ","))
        """
        out = try
            read(`$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $script`, String)
        catch e
            "CHILD_FAILED: " * sprint(showerror, e)
        end
        @test occursin("PRECOMPILE_STATE_OK", out)
    end

    @testset "in-process run restores what was there" begin
        I = Gesso.Inference
        # a non-default value proves the run restores the old state, not a reset
        prev = I._PREFILL_MULTI[]
        I._PREFILL_MULTI[] = 7
        n = length(Gesso._PROCESS_SINK.buf)
        try
            Gesso._run_precompile_workload()
            @test I._PREFILL_MULTI[] == 7
            @test length(Gesso._PROCESS_SINK.buf) == n
            @test (@atomic Gesso.Runtime._CPU_POOL_SLOT.pool) === nothing
        finally
            I._PREFILL_MULTI[] = prev
        end
    end
end
