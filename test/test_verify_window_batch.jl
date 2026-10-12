# Batched window projections: one weight pass over T rows must give the bits of
# T separate single-row decode products. Included after test_verify_window.jl,
# which provides the hybrid Float32 toy.

using Random: Xoshiro

const _GIB = Gesso.Inference

_bits_vw(a::AbstractArray{Float32}) = reinterpret(UInt32, vec(a))

# Random rows for a block-scaled type whose first two bytes are an f16 super-scale.
function _vwb_random_qm(rng, ty::Int32, nrows::Int, ne0::Int)
    rb = _GIB._qrow_bytes(ty, ne0)
    data = rand(rng, UInt8, nrows * rb)
    bsz = _GIB._iq_block_bytes(ty)
    for blk in 0:(nrows * div(rb, bsz) - 1)
        bits = reinterpret(UInt16, Float16(0.005f0 + 0.05f0 * rand(rng, Float32)))
        data[blk * bsz + 1] = UInt8(bits & 0xff)
        data[blk * bsz + 2] = UInt8(bits >> 8)
    end
    return _GIB.QuantMatrix(data, 0, ty, ne0, nrows, rb, C_NULL, Int64(_GIB.QK_K))
end

@testset "verify window: batched quantized projections" begin
    rng = Xoshiro(20261011)
    ne0 = 1024
    T = 5
    for ty in (_GIB.GGML_TYPE_IQ4_XS, _GIB.GGML_TYPE_Q4_K, _GIB.GGML_TYPE_Q5_K, _GIB.GGML_TYPE_Q6_K)
        qm = _vwb_random_qm(rng, ty, 37, ne0)
        X = randn(rng, Float32, T, ne0)
        dst = zeros(Float32, T, qm.nrows)
        _GIB._window_matmul!(nothing, dst, X, (storage=qm,), nothing, nothing)
        for t in 1:T
            # decode's single-row path: one quantization, then the group kernel
            xq = _GIB.Q8KVec(div(ne0, _GIB.QK_K))
            _GIB.quantize_q8_K!(xq, vec(X[t:t, :]))
            ref = zeros(Float32, 1, qm.nrows)
            _GIB._quant_matmul_group!((ref,), xq, (qm,))
            @test _bits_vw(dst[t:t, :]) == _bits_vw(ref)
        end
    end

    # Q8_0 is native but not Q8_K kind, so decode runs it through the seq=1
    # `_quant_matmul!`; the window must give the same bits row by row.
    qm8 = _vwb_random_qm(rng, _GIB.GGML_TYPE_Q8_0, 23, 512)
    X8 = randn(rng, Float32, 3, 512)
    dst8 = zeros(Float32, 3, qm8.nrows)
    _GIB._window_matmul!(nothing, dst8, X8, (storage=qm8,), nothing, nothing)
    for t in 1:3
        ref = zeros(Float32, 1, qm8.nrows)
        _GIB._quant_matmul!(ref, X8[t:t, :], qm8)
        @test _bits_vw(dst8[t:t, :]) == _bits_vw(ref)
    end
end

@testset "verify window: batched projections of one hybrid block vs decode rows" begin
    ts = _vw_hybrid_f32()
    cpu = Gesso.CPUBackend()
    wl = _GIB.DecodeWorkload()
    zeros_like = _GIB._decode_zeros_like(cpu, ts)
    bt = ts.blocks[1]
    dim = ts.model.embedding.dim
    rng = Xoshiro(7)
    T = 4
    X = randn(rng, Float32, T, dim)
    for w in (bt.wgdn_qkv, bt.wgdn_z)
        out = size(w.storage, 1)
        dst = zeros(Float32, T, out)
        _GIB._window_matmul!(cpu, dst, X, w, wl, zeros_like)
        for t in 1:T
            xa = Gesso.Activation(; shape=(1, dim), storage=Matrix{Float32}(X[t:t, :]))
            ya = Gesso.Activation(; shape=(1, out), storage=zeros_like((1, out)))
            Gesso.matmul!(cpu, ya, xa, w, wl)
            @test _bits_vw(dst[t:t, :]) == _bits_vw(Array{Float32}(ya.storage))
        end
    end
end
