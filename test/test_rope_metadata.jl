@testset "Imported RoPE metadata and explicit override" begin
    ts=(; rope=Gesso.RoPEPolicy(; theta=100000.0, kind=:linear, factor=8.0))
    @test Gesso.Inference._resolved_rope_theta(ts, nothing)==100000.0
    @test Gesso.Inference._resolved_rope_theta(ts, 10000.0)==10000.0
    @test Gesso.Inference._resolved_rope_theta((;), nothing)==10000.0
    @test Gesso.tensors_rope_inv_freq(ts, 8)==[100000.0^(-2i/8)/8 for i in 0:3]
    @test Gesso.tensors_rope_inv_freq(ts, 8; theta=10000.0)==[
        10000.0^(-2i/8)/8 for i in 0:3
    ]
end
