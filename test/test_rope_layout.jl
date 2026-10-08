using Test

@testset "HF half-split RoPE preserves imported meaning (Regime I arm 1)" begin
    q0 = reshape(Float64.(1:32), 2, 2, 8)
    k0 = reshape(Float64.(33:48), 2, 1, 8)
    q = Gesso.Activation(; shape=size(q0), storage=copy(q0))
    k = Gesso.Activation(; shape=size(k0), storage=copy(k0))
    Gesso.rope!(
        Gesso.CPUBackend(),
        q,
        k,
        [0, 3],
        Gesso.PrefillWorkload();
        interleaved=false,
    )
    for (actual, source) in ((q.storage, q0), (k.storage, k0))
        @test actual[1, :, :] == source[1, :, :]
        for h in axes(source, 2), i in 1:4
            angle = 3 * 10000.0^(-2(i - 1) / 8)
            a, b = source[2, h, i], source[2, h, i+4]
            @test actual[2, h, i] ≈ a * cos(angle) - b * sin(angle)
            @test actual[2, h, i+4] ≈ b * cos(angle) + a * sin(angle)
        end
    end
    @test Gesso.Inference._tensors_rope_interleaved((embedding=nothing,))
    @test !Gesso.Inference._tensors_rope_interleaved((rope=Gesso.RoPEPolicy(),))
end
