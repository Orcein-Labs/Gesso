# Shared-input quantized projections (CPU decode): one Q8_K quantization of the
# input, one threaded pass over the concatenated rows of several weights.
#
#   julia --project=test -t 8 test/test_cpu_matmul_group.jl
#
# Every output must be bit-identical to the separate `_quant_matmul!` of the
# same weight, which quantizes the same input the same way.

using Gesso
using Test
using Random: Xoshiro

const _GI = Gesso.Inference

# Random valid rows for a weight whose block starts with an f16 super-scale
# (IQ4_XS and Q4_K both do). The scale is finite, so every output is finite.
function _group_random_qm(rng, ty::Int32, nrows::Int, ne0::Int)
    rb = _GI._qrow_bytes(ty, ne0)
    data = rand(rng, UInt8, nrows * rb)
    bsz = _GI._iq_block_bytes(ty)
    for blk in 0:(nrows*div(rb, bsz)-1)
        bits = reinterpret(UInt16, Float16(0.005f0 + 0.05f0 * rand(rng, Float32)))
        data[blk*bsz+1] = UInt8(bits & 0xff)
        data[blk*bsz+2] = UInt8(bits >> 8)
    end
    return _GI.QuantMatrix(data, 0, ty, ne0, nrows, rb, Int64(_GI.QK_K))
end

_bits(a::AbstractArray{Float32}) = reinterpret(UInt32, vec(a))

@testset "shared-input group matmul" begin
    rng = Xoshiro(20261009)
    ne0 = 1024
    for ty in (_GI.GGML_TYPE_IQ4_XS, _GI.GGML_TYPE_Q4_K)
        # uneven row counts, including a weight shorter than one chunk
        ws = (
            _group_random_qm(rng, ty, 37, ne0),
            _group_random_qm(rng, ty, 5, ne0),
            _group_random_qm(rng, ty, 21, ne0),
        )
        x = randn(rng, Float32, 1, ne0)
        xq = _GI.Q8KVec(div(ne0, _GI.QK_K))
        _GI.quantize_q8_K!(xq, vec(x))
        dsts = map(w -> zeros(Float32, 1, w.nrows), ws)
        _GI._quant_matmul_group!(dsts, xq, ws)
        for (d, w) in zip(dsts, ws)
            ref = zeros(Float32, 1, w.nrows)
            _GI._quant_matmul!(ref, x, w)
            @test _bits(d) == _bits(ref)
        end
        # a single weight is the degenerate group
        d1 = (zeros(Float32, 1, ws[1].nrows),)
        _GI._quant_matmul_group!(d1, xq, (ws[1],))
        ref1 = zeros(Float32, 1, ws[1].nrows)
        _GI._quant_matmul!(ref1, x, ws[1])
        @test _bits(d1[1]) == _bits(ref1)
    end

    @testset "decode helper: group when every weight is Q8_K kind, matmul! otherwise" begin
        ne0 = 512
        x = Gesso.Activation(; shape=(1, ne0), storage=randn(rng, Float32, 1, ne0))
        ws = (
            _group_random_qm(rng, _GI.GGML_TYPE_IQ4_XS, 19, ne0),
            _group_random_qm(rng, _GI.GGML_TYPE_IQ4_XS, 8, ne0),
        )
        pw = map(w -> Gesso.ProjectionWeight((w.nrows, ne0), w), ws)
        xq = _GI.Q8KVec(div(ne0, _GI.QK_K))
        outs = map(
            w -> Gesso.Activation(; shape=(1, w.nrows), storage=zeros(Float32, 1, w.nrows)),
            ws,
        )
        _GI._dec_matmul_group!(
            _GI.CPUBackend(),
            outs,
            xq,
            x,
            pw,
            Gesso.Semantics.DecodeWorkload(),
        )
        for (o, w) in zip(outs, ws)
            ref = zeros(Float32, 1, w.nrows)
            _GI._quant_matmul!(ref, x.storage, w)
            @test _bits(o.storage) == _bits(ref)
        end

        # a Q8_0 weight has no Q8_K kernel: the helper falls back, same values
        w80 = _group_random_qm(rng, _GI.GGML_TYPE_Q8_0, 6, ne0)
        pw80 = (Gesso.ProjectionWeight((6, ne0), w80),)
        out80 = (Gesso.Activation(; shape=(1, 6), storage=zeros(Float32, 1, 6)),)
        _GI._dec_matmul_group!(
            _GI.CPUBackend(),
            out80,
            xq,
            x,
            pw80,
            Gesso.Semantics.DecodeWorkload(),
        )
        ref80 = zeros(Float32, 1, 6)
        _GI._quant_matmul!(ref80, x.storage, w80)
        @test _bits(out80[1].storage) == _bits(ref80)
        @test_throws ErrorException _GI._quant_matmul_group!(
            (zeros(Float32, 1, 6),),
            xq,
            (w80,),
        )
    end
end
