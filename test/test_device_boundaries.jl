@testset "Explicit CPU storage boundary" begin
    check=Gesso.Inference._check_device_storage
    cpu=Gesso.CPUBackend()
    @test check(:probe, cpu, Gesso.Activation(; shape=(2, 2), storage=ones(2, 2))) ===
          nothing
    @test check(
        :probe,
        cpu,
        Gesso.Activation(; shape=(2, 2), storage=view(ones(3, 2), 1:2, :)),
    ) === nothing
    for storage in (nothing, ones(Float32, 2, 2))
        t=Gesso.Activation(; shape=(2, 2), storage)
        err=try
            check(:probe, cpu, t)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code==Gesso.ERR_INVALID_PLAN
    end
end
