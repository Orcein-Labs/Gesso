# Item C tests: the operator dispatch surface (§CIX, §XII).
#
# Operators are FUNCTIONS owned by src/backends.jl; item C adds METHODS that
# dispatch on SemanticTensor families × workload. Phase 1 does the routing
# and declines explicitly (§LXX); Phase 2 fills the CPU math.

@testset "operator dispatch surface (§CIX): methods exist on the stub vocabulary" begin
    cpu = Harpe.CPUBackend()
    w = Harpe.ProjectionWeight(shape=(4, 4))
    a = Harpe.Activation(shape=(4, 4))

    for op in (
        :rmsnorm!,
        :rope!,
        :softmax!,
        :swiglu!,
        :matmul!,
        :embedding_lookup!,
        :quantize!,
        :dequantize!,
    )
        f = getglobal(Harpe, op)

        # a semantic-tensor × workload method EXISTS and is the one dispatch
        # selects (not the vararg stub) — for both workload cuts
        for wl in (Harpe.PrefillWorkload(), Harpe.DecodeWorkload())
            m = which(f, (typeof(cpu), typeof(w), typeof(a), typeof(wl)))
            @test parentmodule(m) === Harpe.Operators
        end

        # the dispatch path declines EXPLICITLY with full identity (§LXX):
        # Phase 1 routes; Phase 2 does the math
        for wl in (Harpe.PrefillWorkload(), Harpe.DecodeWorkload())
            err = try
                f(cpu, w, a, wl)
                nothing
            catch e
                e
            end
            @test err isa Harpe.LoweringNotImplemented
            @test err.op === op
            @test err.backend === :cpu
        end
    end
end

@testset "operator dispatch surface (§CIX): workload participates in dispatch" begin
    # the two workload cuts are DISTINCT dispatch keys — §CIX Packet 1:
    # types, not tags. Each resolves its own Operators-owned method.
    mp = which(
        Harpe.rmsnorm!,
        (
            Harpe.CPUBackend,
            Harpe.SemanticTensor,
            Harpe.SemanticTensor,
            Harpe.PrefillWorkload,
        ),
    )
    md = which(
        Harpe.rmsnorm!,
        (
            Harpe.CPUBackend,
            Harpe.SemanticTensor,
            Harpe.SemanticTensor,
            Harpe.DecodeWorkload,
        ),
    )
    @test mp !== md
    @test parentmodule(mp) === Harpe.Operators
    @test parentmodule(md) === Harpe.Operators
end

@testset "operator dispatch surface (§CIX): un-specialized stubs unchanged" begin
    # existing backend-interface contract stays green: vararg calls still
    # hit the stubs and throw with the same identity
    cpu = Harpe.CPUBackend()
    @test_throws Harpe.LoweringNotImplemented Harpe.rmsnorm!(cpu, nothing)
    @test_throws Harpe.LoweringNotImplemented Harpe.matmul!(cpu, nothing, nothing, nothing)
    err = try
        Harpe.softmax!(cpu, nothing)
        nothing
    catch e
        e
    end
    @test err isa Harpe.LoweringNotImplemented
    @test err.op === :softmax!
    @test err.backend === :cpu

    # §CIX: no second vocabulary was created — these ARE backends.jl's
    # functions, extended with methods (generic function still owned by the
    # module that defined the stub)
    @test parentmodule(Harpe.rmsnorm!) === Harpe
    @test length(methods(Harpe.rmsnorm!)) == 3   # stub + prefill + decode
end
