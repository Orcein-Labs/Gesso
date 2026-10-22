# Native F16 / BF16 matmul for QuantMatrix (`_half_matmul!`).
#
#   julia --project=test -t 8 test/test_quant_half.jl
#
# The widening is exact, so both paths are checked against a Float64 product
# of the same widened weights. A row dot must not depend on how rows are split
# across threads or on how many tokens share the call.

using Gesso
using Test
using Random: Xoshiro

const _GH = Gesso.Inference

function _half_random_qm(rng, ty::Int32, nrows::Int, ne0::Int)
    w = randn(rng, Float32, ne0, nrows) .* 0.05f0
    bits = ty == _GH.GGML_TYPE_BF16 ?
        map(v -> UInt16(reinterpret(UInt32, v) >> 16), w) :
        map(v -> reinterpret(UInt16, Float16(v)), w)
    # two bytes of padding in front, so offset0 is exercised
    data = vcat(UInt8[0xaa, 0x55], collect(reinterpret(UInt8, vec(bits))))
    q = _GH.QuantMatrix(data, 2, ty, ne0, nrows, 2 * ne0, C_NULL, Int64(1))
    wide = ty == _GH.GGML_TYPE_BF16 ?
        map(u -> Float64(reinterpret(Float32, UInt32(u) << 16)), bits) :
        map(u -> Float64(reinterpret(Float16, u)), bits)
    return q, wide
end

@testset "F16 / BF16 QuantMatrix matmul" begin
    rng = Xoshiro(20261022)
    ne0, nrows = 640, 203
    for ty in (_GH.GGML_TYPE_BF16, _GH.GGML_TYPE_F16)
        q, wide = _half_random_qm(rng, ty, nrows, ne0)
        for seq in (1, 3, 17, _GH._HALF_ROWDOT_MAX_SEQ + 1, 130)
            x = randn(rng, Float32, seq, ne0)
            dst = Matrix{Float32}(undef, seq, nrows)
            _GH._quant_matmul!(dst, x, q)
            ref = Float64.(x) * wide
            @test maximum(abs.(dst .- ref)) <= 1.0e-4 * maximum(abs.(ref))
        end

        # A row's result is the same for any chunk count and any token count:
        # one serial pass, the threaded pass, and the token alone all agree.
        xs = randn(rng, Float32, 7, ne0)
        xt = zeros(Float32, ne0, 8); xt[:, 1:7] = permutedims(xs)
        d1 = Matrix{Float32}(undef, 7, nrows)
        _GH._half_rows!(d1, xt, q, Val(ty), 1, nrows)
        d2 = Matrix{Float32}(undef, 7, nrows)
        _GH._half_rowdot!(d2, xt, q, Val(ty))
        @test reinterpret(UInt32, d1) == reinterpret(UInt32, d2)
        d3 = Matrix{Float32}(undef, 1, nrows)
        _GH._quant_matmul!(d3, xs[4:4, :], q)
        @test reinterpret(UInt32, vec(d3)) == reinterpret(UInt32, d2[4, :])

        # a contiguous row view reads from its own offset
        qs = q[11:40, :]
        dst = Matrix{Float32}(undef, 7, 30)
        _GH._quant_matmul!(dst, xs, qs)
        @test reinterpret(UInt32, dst) == reinterpret(UInt32, d2[:, 11:40])
    end
    # a row length that is not a multiple of 16 takes the scalar tail
    q, wide = _half_random_qm(rng, _GH.GGML_TYPE_BF16, 37, 650)
    xs = randn(rng, Float32, 6, 650)
    dst = Matrix{Float32}(undef, 6, 37)
    _GH._quant_matmul!(dst, xs, q)
    ref = Float64.(xs) * wide
    @test maximum(abs.(dst .- ref)) <= 1.0e-4 * maximum(abs.(ref))
    d1 = Matrix{Float32}(undef, 1, 37)
    _GH._quant_matmul!(d1, xs[6:6, :], q)
    @test reinterpret(UInt32, vec(d1)) == reinterpret(UInt32, dst[6, :])

    # an empty fallback log: neither type takes the to_float bridge
    @test !(_GH.GGML_TYPE_BF16 in _GH._QUANT_FALLBACK_TYPES)
    @test !(_GH.GGML_TYPE_F16 in _GH._QUANT_FALLBACK_TYPES)
end
