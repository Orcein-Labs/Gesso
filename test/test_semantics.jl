# Item A tests: workload dispatch types (§CIX Packet 1; §XII, §XXX).

@testset "workload types (§CIX): construction + dispatch identity" begin
    p = Harpe.PrefillWorkload()
    d = Harpe.DecodeWorkload()

    # singleton types: one value each, dispatch-stable
    @test p === Harpe.PrefillWorkload()
    @test d === Harpe.DecodeWorkload()

    # the two cuts are distinct types — dispatch can separate them
    @test typeof(p) !== typeof(d)
    @test p isa Harpe.Semantics.PrefillWorkload
    @test d isa Harpe.Semantics.DecodeWorkload

    # §CIX: they are TYPES, not enum instances of a mega-vocabulary
    @test !isdefined(Harpe, :ExecutionPhase)
    @test !isdefined(Harpe, :WorkloadKind)

    # reachable unqualified through `using Harpe` (root re-export)
    @test Base.isexported(Harpe, :PrefillWorkload)
    @test Base.isexported(Harpe, :DecodeWorkload)
end
