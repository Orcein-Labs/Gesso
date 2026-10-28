# Batched window projections: one weight pass over T rows must give the bits of
# T separate single-row decode products. Included after test_verify_window.jl,
# which provides the hybrid Float32 toy.

using Random: Xoshiro

const _GIB = Gesso.Inference

_bits_vw(a::AbstractArray{Float32}) = reinterpret(UInt32, vec(a))

# Random rows for a block-scaled type, with finite f16 super-scales (d, and dmin
# where the format has one) so no row is NaN: NaN payloads are not bit-stable.
function _vwb_random_qm(rng, ty::Int32, nrows::Int, ne0::Int)
    rb = _GIB._qrow_bytes(ty, ne0)
    data = rand(rng, UInt8, nrows * rb)
    bsz = _GIB._iq_block_bytes(ty)
    offs = filter(>=(0), [_GIB._q8k_d_offset(ty), _GIB._q8k_dmin_offset(ty)])
    for blk in 0:(nrows*div(rb, bsz)-1), off in offs
        bits = reinterpret(UInt16, Float16(0.005f0 + 0.05f0 * rand(rng, Float32)))
        data[blk*bsz+off+1] = UInt8(bits & 0xff)
        data[blk*bsz+off+2] = UInt8(bits >> 8)
    end
    return _GIB.QuantMatrix(data, 0, ty, ne0, nrows, rb, Int64(_GIB.QK_K))
end

@testset "verify window: batched quantized projections" begin
    rng = Xoshiro(20261011)
    ne0 = 1024
    T = 5
    for ty in (
        _GIB.GGML_TYPE_IQ4_XS,
        _GIB.GGML_TYPE_Q4_K,
        _GIB.GGML_TYPE_Q5_K,
        _GIB.GGML_TYPE_Q6_K,
    )
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

# The prompt is built by the window's T-row forward. Its state must be the bits
# that decode leaves when it consumes the same tokens one row at a time. Row 1
# has no earlier state to decode from, so the state is initialised by a one-token
# prefill (a T=1 window); rows 2..P go through `_decode_step!`, the decode path.
function _vw_decode_one_by_one(ts, prompt::Vector{Int})
    model = ts.model
    st = GenerateState(_VW_CTX)
    reference_generate(model, ts, prompt[1:1]; max_new_tokens=0, state=st)
    cpu = Gesso.CPUBackend()
    wd = _GIB.DecodeWorkload()
    n_heads = model.blocks[1].attention.n_heads
    n_kv_heads = model.blocks[1].attention.n_kv_heads
    d_head = _GIB._tensor_head_dim(model, ts)
    group = div(n_heads, n_kv_heads)
    theta = _GIB._resolved_rope_theta(ts, nothing)
    inv_freq = _GIB.tensors_rope_inv_freq(ts, d_head; theta)
    rotary_dim = _GIB._tensor_rotary_dim(ts)
    for k in 2:length(prompt)
        _GIB._decode_step!(
            model,
            ts,
            st.kc,
            st.vc,
            prompt[k],
            k,
            cpu,
            wd,
            st.h;
            on_cpu=true,
            n_heads,
            n_kv_heads,
            d_head,
            group,
            eps=1e-6,
            theta,
            inv_freq,
            rotary_dim,
            gdn_conv=st.gdn_conv,
            gdn_rec=st.gdn_rec,
            gdn_scratch=st.gdn_scratch,
            cpu_scratch=st.cpu_scratch,
            hrow=1,
        )
        push!(st.ids, prompt[k])
    end
    return st
end

@testset "prefill window: state equals decode one row at a time" begin
    ts = _vw_hybrid_f32()
    vocab = size(ts.embedding.storage, 1)
    prompt = Int[mod(7 * i + 3, vocab) for i in 0:8]   # 0-based token ids
    oracle = _vw_decode_one_by_one(ts, prompt)
    try
        # one pass, then several passes of 3 rows each, which exercises the commit
        # of a window that does not start at row 1
        for chunk in ("1024", "3")
            withenv("GESSO_PREFILL_CHUNK" => chunk) do
                st = GenerateState(_VW_CTX)
                try
                    reference_generate(ts.model, ts, prompt; max_new_tokens=0, state=st)
                    @test st.cpu_scratch !== nothing
                    @test _vw_state_diff(st, oracle) == String[]
                finally
                    close(st)
                end
            end
        end
    finally
        close(oracle)
    end
end
