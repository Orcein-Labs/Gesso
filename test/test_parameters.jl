# Item A tests: §XI semantic families + the `frozen` trait (§CIX).

@testset "semantic families (§XI): construction + metadata fields" begin
    families = [
        Harpe.ProjectionWeight,
        Harpe.KVCache,
        Harpe.EmbeddingTable,
        Harpe.ExpertWeight,
        Harpe.FrozenParameter,
        Harpe.QuantizedParameter,
        Harpe.Activation,
        Harpe.TemporaryWorkspace,
        Harpe.RoutingState,
        Harpe.DecodeState,
        Harpe.AdapterDelta,
    ]
    @test length(families) == 11   # the §XI list, complete

    for F in families
        # every family is a SemanticTensor and constructs kwarg-style
        x = F(; shape=(2, 3))
        @test x isa Harpe.SemanticTensor
        @test x isa F
        @test x.shape == (2, 3)

        # §CIX/§XIV: storage starts UNSET — bytes are not the meaning
        @test x.storage === nothing

        # metadata is FIELDS, not type parameters: different shapes share
        # one type (a shape must not become a dispatch axis)
        @test typeof(F(; shape=(4,))) === typeof(F(; shape=(9, 9, 9)))
    end
end

@testset "semantic families (§XI): immutability" begin
    w = Harpe.ProjectionWeight(; shape=(2, 3))
    err = try
        w.shape = (5, 5)
        nothing
    catch e
        e
    end
    @test err isa ErrorException   # setfield! on an immutable struct fails
end

@testset "frozen trait (§CIX: the only trait)" begin
    frozen_families = [
        Harpe.ProjectionWeight,
        Harpe.EmbeddingTable,
        Harpe.FrozenParameter,
        Harpe.ExpertWeight,
        Harpe.AdapterDelta,
    ]
    volatile_families = [
        Harpe.KVCache,
        Harpe.QuantizedParameter,
        Harpe.Activation,
        Harpe.TemporaryWorkspace,
        Harpe.RoutingState,
        Harpe.DecodeState,
    ]
    for F in frozen_families
        @test Harpe.frozen(F) == true      # on the type
        @test Harpe.frozen(F(; shape=(1,))) == true   # on the value
    end
    for F in volatile_families
        @test Harpe.frozen(F) == false
        @test Harpe.frozen(F(; shape=(1,))) == false
    end
    # default for anything else: not frozen (traits default closed)
    @test Harpe.frozen(Harpe.CPUBackend) == false
end

@testset "no quantization lattice beyond the family type (§CIX stop line)" begin
    @test Harpe.QuantizedParameter isa Type
    @test !isdefined(Harpe.Parameters, :Quantized)
    @test !isdefined(Harpe, :Quantized)
    @test !isdefined(Harpe.Parameters, :Representation)
end
