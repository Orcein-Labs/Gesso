# Single-token GDN depthwise conv + SiLU (CPU decode): the pooled channel split
# against the serial loop, bit for bit, for outputs and the shifted conv state.
#
#   julia --project=test -t 8 test/test_cpu_gdn_conv_decode.jl

using Gesso
using Test
using Random: Xoshiro

const _GC = Gesso.Inference

_gconv_bits(a::AbstractArray{Float32}) = reinterpret(UInt32, vec(a))

@testset "pooled GDN conv decode equals the serial loop (bit-exact)" begin
    rng = Xoshiro(20261106)
    # Below, at and above the pool threshold, odd channel counts, ksz 2 and 4.
    for (channels, ksz) in [(64, 4), (2047, 4), (2048, 4), (2049, 4), (3001, 2), (6144, 4)]
        weight = randn(rng, Float32, channels, 1, ksz)
        state0 = randn(rng, Float32, channels, ksz - 1)
        s_pool, s_ser = copy(state0), copy(state0)
        o_pool = zeros(Float32, channels)
        o_ser = zeros(Float32, channels)
        for step in 1:3
            x = randn(rng, Float32, channels)
            _GC._gdn_conv_decode!(o_pool, x, weight, s_pool, ksz)
            _GC._gdn_conv_decode_range!(o_ser, x, weight, s_ser, ksz, 1, channels)
            @test _gconv_bits(o_pool) == _gconv_bits(o_ser)
            @test _gconv_bits(s_pool) == _gconv_bits(s_ser)
        end
        @test s_pool[:, ksz-1] != state0[:, ksz-1]
    end
end
