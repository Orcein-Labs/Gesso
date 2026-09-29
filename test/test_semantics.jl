# Item A tests: workload dispatch types (§CIX Packet 1; §XII, §XXX).

@testset "workload types (§CIX): construction + dispatch identity" begin
    p = Gesso.PrefillWorkload()
    d = Gesso.DecodeWorkload()

    # singleton types: one value each, dispatch-stable
    @test p === Gesso.PrefillWorkload()
    @test d === Gesso.DecodeWorkload()

    # the two cuts are distinct types — dispatch can separate them
    @test typeof(p) !== typeof(d)
    @test p isa Gesso.Semantics.PrefillWorkload
    @test d isa Gesso.Semantics.DecodeWorkload

    # §CIX: they are TYPES, not enum instances of a mega-vocabulary
    @test !isdefined(Gesso, :ExecutionPhase)
    @test !isdefined(Gesso, :WorkloadKind)

    # reachable unqualified through `using Gesso` (root re-export)
    @test Base.isexported(Gesso, :PrefillWorkload)
    @test Base.isexported(Gesso, :DecodeWorkload)
end
