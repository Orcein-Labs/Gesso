using Gesso, Lava, Test
@testset "Unavailable primary Lava device is explicit" begin
    err=try
        Gesso.LavaBackend()
        nothing
    catch e
        e
    end
    @test err isa Gesso.GessoError
    @test err.code==Gesso.ERR_RESOURCE_LIMIT
    @test err.detail[:requested_backend]==:lava
end
