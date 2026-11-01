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

        # a Q8_K weight of another width cannot share `xq`: the helper falls
        # back to the per-weight matmul instead of erroring
        x2 = Gesso.Activation(; shape=(1, 2ne0), storage=randn(rng, Float32, 1, 2ne0))
        w2 = _group_random_qm(rng, _GI.GGML_TYPE_IQ4_XS, 7, 2ne0)
        out2 = (Gesso.Activation(; shape=(1, 7), storage=zeros(Float32, 1, 7)),)
        _GI._dec_matmul_group!(
            _GI.CPUBackend(),
            out2,
            xq,
            x2,
            (Gesso.ProjectionWeight((7, 2ne0), w2),),
            Gesso.Semantics.DecodeWorkload(),
        )
        ref2 = zeros(Float32, 1, 7)
        _GI._quant_matmul!(ref2, x2.storage, w2)
        @test _bits(out2[1].storage) == _bits(ref2)
    end
end

# Random K-quant rows with finite Float16 d (and dmin, where the type has one)
# at each type's own block offsets.
function _group_random_kq(rng, ty::Int32, nrows::Int, ne0::Int)
    rb = _GI._qrow_bytes(ty, ne0)
    data = rand(rng, UInt8, nrows * rb)
    bsz = _GI._ggml_traits(ty).type_size
    offs = filter(>=(0), [_GI._q8k_d_offset(ty), _GI._q8k_dmin_offset(ty)])
    for blk in 0:(nrows*div(rb, bsz)-1), off in offs
        bits = reinterpret(UInt16, Float16(0.005f0 + 0.05f0 * rand(rng, Float32)))
        data[blk*bsz+off+1] = UInt8(bits & 0xff)
        data[blk*bsz+off+2] = UInt8(bits >> 8)
    end
    return _GI.QuantMatrix(data, 0, ty, ne0, nrows, rb, Int64(_GI.QK_K))
end

# Dense decode now groups its projections too. A Q4_K_M or Q3_K_S GGUF mixes
# these types within one group (QKV, gate+up), so the bits must hold for each
# type and for a mixed group.
@testset "shared-input group matmul: every K-quant type, mixed groups" begin
    rng = Xoshiro(20261101)
    ne0 = 768
    tys = (
        _GI.GGML_TYPE_Q2_K,
        _GI.GGML_TYPE_Q3_K,
        _GI.GGML_TYPE_Q4_K,
        _GI.GGML_TYPE_Q5_K,
        _GI.GGML_TYPE_Q6_K,
    )
    x = randn(rng, Float32, 1, ne0)
    xq = _GI.Q8KVec(div(ne0, _GI.QK_K))
    _GI.quantize_q8_K!(xq, vec(x))
    for ty in tys
        @test _GI._quant_q8k_kind(_group_random_kq(rng, ty, 1, ne0))
    end
    ws = map((ty, n) -> _group_random_kq(rng, ty, n, ne0), tys, (29, 3, 64, 17, 40))
    dsts = map(w -> zeros(Float32, 1, w.nrows), ws)
    _GI._quant_matmul_group!(dsts, xq, ws)
    for (d, w) in zip(dsts, ws)
        ref = zeros(Float32, 1, w.nrows)
        _GI._quant_matmul!(ref, x, w)
        @test all(isfinite, ref)
        @test _bits(d) == _bits(ref)
    end
    # negative control: a group over a different input must not match
    xo = randn(rng, Float32, 1, ne0)
    xqo = _GI.Q8KVec(div(ne0, _GI.QK_K))
    _GI.quantize_q8_K!(xqo, vec(xo))
    do_ = map(w -> zeros(Float32, 1, w.nrows), ws)
    _GI._quant_matmul_group!(do_, xqo, ws)
    @test all(_bits(a) != _bits(b) for (a, b) in zip(do_, dsts))
end
