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

# A random GGUF expert slab: `nexp` experts of (ne1, ne0), finite super-scales.
function _vwb_slice(rng, ty::Int32, nexp::Int, ne0::Int, ne1::Int)
    qm = _vwb_random_qm(rng, ty, nexp * ne1, ne0)
    blk = ty == _GIB.GGML_TYPE_Q8_0 ? 32 : _GIB.QK_K
    return Gesso.GgufQuantSlice(qm.data, 0, ty, nexp, ne0, ne1, ne1 * qm.row_bytes, Int64(blk))
end

# The window MoE FFN must give each row the bits of `_decode_ffn_block!` on that
# row alone: K-quant slabs (batched host route), Q8_0 slabs (per-token route),
# and a shared expert with and without its sigmoid gate, quantized or dense.
@testset "verify window: batched MoE FFN vs decode rows" begin
    rng = Xoshiro(20261101)
    h, ff, nexp, T = 256, 256, 4, 5
    cpu = Gesso.CPUBackend()
    wl = _GIB.DecodeWorkload()
    zl = dims -> zeros(Float32, dims)
    pw(a) = Gesso.ProjectionWeight(; shape=size(a), storage=a)
    fz(a) = Gesso.FrozenParameter(; shape=size(a), storage=a)
    cases = (
        (_GIB.GGML_TYPE_Q2_K, _GIB.GGML_TYPE_Q3_K, :quant, true),
        (_GIB.GGML_TYPE_Q4_K, _GIB.GGML_TYPE_Q6_K, :quant, false),
        (_GIB.GGML_TYPE_Q5_K, _GIB.GGML_TYPE_Q4_K, :dense, true),
        (_GIB.GGML_TYPE_Q4_K, _GIB.GGML_TYPE_Q4_K, :none, false),
        (_GIB.GGML_TYPE_Q8_0, _GIB.GGML_TYPE_Q8_0, :quant, true),
    )
    for (tgu, tdown, shared, gated) in cases
        bt = (;
            ffn_rms=fz(1.0f0 .+ 0.1f0 .* randn(rng, Float32, h)),
            wrouter=pw(randn(rng, Float32, nexp, h)),
            n_expert_used=2,
            wexp_gate=(storage=_vwb_slice(rng, tgu, nexp, h, ff),),
            wexp_up=(storage=_vwb_slice(rng, tgu, nexp, h, ff),),
            wexp_down=(storage=_vwb_slice(rng, tdown, nexp, ff, h),),
        )
        if shared !== :none
            sw(rows, cols) =
                shared === :quant ?
                pw(_vwb_random_qm(rng, _GIB.GGML_TYPE_Q4_K, rows, cols)) :
                pw(0.05f0 .* randn(rng, Float32, rows, cols))
            bt = merge(bt, (wshared_gate=sw(ff, h), wshared_up=sw(ff, h), wshared_down=sw(h, ff)))
            gated && (bt = merge(bt, (wshared_gate_inp=fz(0.1f0 .* randn(rng, Float32, h)),)))
        end
        h0 = randn(rng, Float32, T, h)
        subT = randn(rng, Float32, T, h)
        bufs = _GIB._decode_row_buffers(nothing, zl, h, 8, 8, 1, 1, 8, ff)
        W = (normed2=zl((T, h)), ffn_down=zl((T, h)))
        @test _GIB._window_moe_ok(bt, h0, W, cpu)
        hwin = copy(h0)
        _GIB._window_ffn_block!(bt, hwin, subT, W, bufs, cpu, wl, zl; eps=1e-6)
        ref = copy(h0)
        for t in 1:T
            _GIB._win_load!(bufs.hp.storage, ref, t)
            _GIB._win_load!(bufs.sub.storage, subT, t)
            _GIB._decode_ffn_block!(bt, bufs, cpu, wl, zl; eps=1e-6)
            _GIB._win_store!(ref, t, bufs.hp.storage)
        end
        @test all(isfinite, ref)
        @test _bits_vw(hwin) == _bits_vw(ref)
    end
end
