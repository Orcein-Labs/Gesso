# Attention output gate dst = src .* sigmoid(gate): the pooled split for
# prefill-sized arrays against the serial loop, bit for bit, including dst
# aliasing src as the window path calls it.
#
#   julia --project=test -t 8 test/test_cpu_sigmoid_mul.jl

using Gesso
using Test
using Random: Xoshiro

const _SM = Gesso.Inference

@testset "pooled sigmoid-mul equals the serial loop (bit-exact)" begin
    rng = Xoshiro(20261107)
    # Below, at and above the split threshold, and an odd prefill shape.
    for dims in [(4, 2048), (128, 2048), (129, 2048), (512, 4096), (511, 2047)]
        src = randn(rng, Float32, dims) .* 4
        gate = randn(rng, Float32, dims) .* 6
        ref = similar(src)
        _SM._cpu_sigmoid_mul!(ref, src, gate)
        dst = similar(src)
        _SM._sigmoid_mul!(nothing, dst, src, gate)
        @test reinterpret(UInt32, vec(dst)) == reinterpret(UInt32, vec(ref))
        alias = copy(src)
        _SM._sigmoid_mul!(nothing, alias, alias, gate)
        @test reinterpret(UInt32, vec(alias)) == reinterpret(UInt32, vec(ref))
    end
end
