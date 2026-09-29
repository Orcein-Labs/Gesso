# Item A tests: §XI semantic families + the `frozen` trait (§CIX).

@testset "semantic families (§XI): construction + metadata fields" begin
    families = [
        Gesso.ProjectionWeight,
        Gesso.KVCache,
        Gesso.EmbeddingTable,
        Gesso.ExpertWeight,
        Gesso.FrozenParameter,
        Gesso.QuantizedParameter,
        Gesso.Activation,
        Gesso.TemporaryWorkspace,
        Gesso.RoutingState,
        Gesso.DecodeState,
        Gesso.AdapterDelta,
    ]
    @test length(families) == 11   # the §XI list, complete

    for F in families
        # every family is a SemanticTensor and constructs kwarg-style
        x = F(; shape=(2, 3))
        @test x isa Gesso.SemanticTensor
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
    w = Gesso.ProjectionWeight(; shape=(2, 3))
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
        Gesso.ProjectionWeight,
        Gesso.EmbeddingTable,
        Gesso.FrozenParameter,
        Gesso.ExpertWeight,
        Gesso.AdapterDelta,
    ]
    volatile_families = [
        Gesso.KVCache,
        Gesso.QuantizedParameter,
        Gesso.Activation,
        Gesso.TemporaryWorkspace,
        Gesso.RoutingState,
        Gesso.DecodeState,
    ]
    for F in frozen_families
        @test Gesso.frozen(F) == true      # on the type
        @test Gesso.frozen(F(; shape=(1,))) == true   # on the value
    end
    for F in volatile_families
        @test Gesso.frozen(F) == false
        @test Gesso.frozen(F(; shape=(1,))) == false
    end
    # default for anything else: not frozen (traits default closed)
    @test Gesso.frozen(Gesso.CPUBackend) == false
end

@testset "no quantization lattice beyond the family type (§CIX stop line)" begin
    @test Gesso.QuantizedParameter isa Type
    @test !isdefined(Gesso.Parameters, :Quantized)
    @test !isdefined(Gesso, :Quantized)
    @test !isdefined(Gesso.Parameters, :Representation)
end
