# Element-wise host work that splits across the CPU pool: the window prefill's
# residual add, the zeros factory's fill and the quantized embedding rows.
#
#   julia --project=test -t 8 test/test_pool_elemwise.jl
#
# Sizes sit above the pool threshold, so with more than one thread each call
# takes the split path. Results must equal the one-thread loop bit for bit.

using Gesso
using Test
using Random: Xoshiro

const _PE = Gesso.Inference

@testset "pooled element-wise host ops" begin
    rng = Xoshiro(20261106)
    big = _PE._POOL_ELEMWISE_MIN

    @testset "residual add" begin
        for dims in ((512, 1024), (513, 777), (big + 3,))
            dst = randn(rng, Float32, dims)
            src = randn(rng, Float32, dims)
            ref = copy(dst)
            for i in eachindex(ref)
                ref[i] = ref[i] + src[i]
            end
            @test _PE._add_storage!(dst, src) === dst
            @test reinterpret(UInt32, vec(dst)) == reinterpret(UInt32, vec(ref))
        end
        # mismatched shapes keep the broadcast
        a = zeros(Float32, 600, 600)
        b = ones(Float32, 600, 1)
        _PE._add_storage!(a, b)
        @test all(==(1.0f0), a)
    end

    @testset "zero fill" begin
        for dims in ((512, 2048), (big + 5,), (3, 4))
            x = fill(NaN32, dims)
            @test _PE._fill_zero!(x) === x
            @test all(v -> reinterpret(UInt32, v) == 0x00000000, x)
        end
        e = (; storage=ones(Float32, 2, 2))
        zl = _PE._zeros_like_factory(nothing, (; embedding=e, lm_head=e))
        z = zl((640, 512))
        @test size(z) == (640, 512) && eltype(z) === Float32 && all(iszero, z)
    end

    @testset "quantized embedding rows" begin
        ty = _PE.GGML_TYPE_Q4_K
        ne0, nrows = 512, 300
        rb = _PE._qrow_bytes(ty, ne0)
        data = rand(rng, UInt8, nrows * rb)
        bsz = _PE._iq_block_bytes(ty)
        for blk in 0:(nrows*div(rb, bsz)-1)
            bits = reinterpret(UInt16, Float16(0.005f0 + 0.05f0 * rand(rng, Float32)))
            data[blk*bsz+1] = UInt8(bits & 0xff)
            data[blk*bsz+2] = UInt8(bits >> 8)
            # Q4_K's second f16 is the min scale; keep it finite too
            mbits = reinterpret(UInt16, Float16(0.002f0 * rand(rng, Float32)))
            data[blk*bsz+3] = UInt8(mbits & 0xff)
            data[blk*bsz+4] = UInt8(mbits >> 8)
        end
        q = _PE.QuantMatrix(data, 0, ty, ne0, nrows, rb, Int64(_PE.QK_K))
        for n in (1, 15, 16, 97, 512)
            toks = rand(rng, 0:(nrows-1), n)
            got = fill(NaN32, n, ne0)
            ref = fill(NaN32, n, ne0)
            _PE._quant_embedding_rows!(got, q, toks)
            _PE._quant_embedding_range!(ref, q, toks, 1:n)
            @test reinterpret(UInt32, vec(got)) == reinterpret(UInt32, vec(ref))
            @test all(isfinite, got)
        end
        # a non-vector token iterator
        got = zeros(Float32, 20, ne0)
        ref = zeros(Float32, 20, ne0)
        _PE._quant_embedding_rows!(got, q, (t for t in 0:19))
        _PE._quant_embedding_range!(ref, q, collect(0:19), 1:20)
        @test got == ref
    end
end
