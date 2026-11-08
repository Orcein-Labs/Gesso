# qwen4exp (Qwen3.8 Flash-Next) CPU forward, after llama.cpp
# `src/models/qwen4exp.cpp`.
#
# The residual is `n_hc` parallel streams of width `n_embd`, held as one
# (T, n_hc * n_embd) matrix in ggml's flat order (feature i of stream s sits at
# column i + n_embd * s). Each block reads the streams through a hyper-connection
# mix (a per-stream RMS norm, a low-rank sigmoid gate, the gated mean over
# streams) and writes its output back into every stream scaled by
# 2 * sigmoid(inject / n_hc).
#
# Blocks alternate gated delta-net mixers and full attention
# ((il + 1) % full_attention_interval == 0, 0-based il). Every FFN is a softmax
# top-k MoE with a sigmoid-gated shared expert. The layers listed in
# `ple.layers` first add a per-layer embedding: hashed bigram and trigram rows
# of `per_layer_token_embd`, gated per stream by a key/query score and passed
# through a dilated causal conv.
#
# Attention layers with a compress ratio use sparse attention (QSA) in the
# reference. Its indexer keeps the top `attention.indexer.top_k` keys, so for
# contexts up to that length it is dense attention. This forward is dense only
# and refuses longer contexts on those layers. The MTP head is not wired.
#
# `qwen4exp_forward!` carries every recurrent piece in `Qwen4expState` (the
# GDN conv and delta state, the KV caches, the PLE token and conv history), so
# a prompt fed in chunks gives the same rows as one call.

struct Qwen4expConfig
    n_embd::Int
    n_layer::Int
    n_hc::Int
    n_head::Int
    n_head_kv::Int
    head_dim::Int
    n_rot::Int
    rope_base::Float64
    eps::Float64
    attn_layer::Vector{Bool}   # index il + 1
    compress::Vector{Int}      # index il + 1; > 0 means a QSA layer
    qsa_top_k::Int
    conv_kernel::Int
    ssm_nk::Int
    ssm_dk::Int
    ssm_nv::Int
    ssm_dv::Int
    n_exp::Int
    n_used::Int
    ple_layers::Vector{Int}    # 0-based
    ple_ngram::Int
    ple_hpn::Int
    ple_mult::Vector{UInt64}
    ple_off::Vector{Int}
    ple_voc::Vector{UInt64}
    ple_eos::Int
    ple_kernel::Int
    vocab::Int
end

"""
    qwen4exp_config(kv) -> Qwen4expConfig

Read the architecture from GGUF metadata. Every value comes from the file; a
missing key is an error.
"""
function qwen4exp_config(kv::AbstractDict)
    arch = String(get(kv, "general.architecture", ""))
    arch == "qwen4exp" || error("qwen4exp_config: general.architecture is $(repr(arch))")
    req(key) = (k = "qwen4exp.$key"; haskey(kv, k) ? kv[k] : error("qwen4exp_config: missing $k"))
    ints(v) = Int[Int(x) for x in v]
    nextn = Int(get(kv, "qwen4exp.nextn_predict_layers", 0))
    n_layer = Int(req("block_count")) - nextn
    interval = Int(req("full_attention_interval"))
    compress = ints(req("attention.compress_ratios"))
    n_rot = Int(req("rope.dimension_count"))
    # Text-only M-RoPE: every section reads the same position, which is NeoX
    # over the first n_rot dims.
    sum(ints(req("rope.dimension_sections"))) == n_rot ÷ 2 ||
        error("qwen4exp_config: rope sections do not cover n_rot/2 = $(n_rot ÷ 2)")
    n_head = Int(req("attention.head_count"))
    hd = Int(req("attention.key_length"))
    hd == Int(req("attention.value_length")) ||
        error("qwen4exp_config: key_length ≠ value_length")
    nv = Int(req("ssm.time_step_rank"))
    inner = Int(req("ssm.inner_size"))
    inner % nv == 0 || error("qwen4exp_config: ssm.inner_size $inner not divisible by $nv")
    toks = get(kv, "tokenizer.ggml.tokens", nothing)
    vocab = toks isa Integer ? Int(toks) : toks === nothing ? 0 : length(toks)
    return Qwen4expConfig(
        Int(req("embedding_length")), n_layer, Int(req("hyper_connection.count")),
        n_head, Int(req("attention.head_count_kv")), hd, n_rot,
        Float64(req("rope.freq_base")), Float64(req("attention.layer_norm_rms_epsilon")),
        [(il + 1) % interval == 0 for il in 0:(n_layer-1)],
        compress[1:n_layer], Int(req("attention.indexer.top_k")),
        Int(req("ssm.conv_kernel")), Int(req("ssm.group_count")), Int(req("ssm.state_size")),
        nv, inner ÷ nv,
        Int(req("expert_count")), Int(req("expert_used_count")),
        ints(req("ple.layers")), Int(req("ple.ngram_size")), Int(req("ple.heads_per_ngram")),
        UInt64[UInt64(x) for x in req("ple.layer_multipliers")],
        ints(req("ple.head_offsets")),
        UInt64[UInt64(x) for x in req("ple.head_vocab_sizes")],
        Int(req("ple.eos_token_id")), Int(req("ple.conv_kernel")), vocab,
    )
end

"""
    Qwen4expModel

The config plus a lazy tensor store. GGUF tensors are mapped on first use:
1-D tensors and conv kernels as flat Float32 (ggml order), `ffn_gate_inp` as a
dense (n_exp, n_embd) matrix, 3-D experts as `GgufQuantSlice`, other 2-D
weights as `QuantMatrix` (dense (out, in) for F32). A model built from a
`Dict` of dense arrays uses the same names and layouts, with experts as
(n_exp, out, in) arrays.
"""
struct Qwen4expModel
    cfg::Qwen4expConfig
    index::Dict{String, Tuple{Any, GgufTensorInfo}}
    cache::Dict{String, Any}
    # 1 while a call holds the reused scratch buffers (see _q4x_buf)
    scratch_busy::Threads.Atomic{Int}
end

Qwen4expModel(cfg::Qwen4expConfig, index::Dict{String, Tuple{Any, GgufTensorInfo}}, cache::Dict{String, Any}) =
    Qwen4expModel(cfg, index, cache, Threads.Atomic{Int}(0))

Qwen4expModel(cfg::Qwen4expConfig, tensors::AbstractDict) =
    Qwen4expModel(cfg, Dict{String, Tuple{Any, GgufTensorInfo}}(), Dict{String, Any}(tensors))

"""
    load_qwen4exp(path) -> Qwen4expModel

Open a qwen4exp GGUF. `path` may be any shard of a `-0000N-of-0000M.gguf`
split; the siblings are found next to it.
"""
function load_qwen4exp(path::AbstractString)
    m = match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
    files = m === nothing ? [path] :
        [string(m[1], "-", lpad(i, 5, '0'), "-of-", m[3], ".gguf") for i in 1:parse(Int, m[3])]
    hdrs = [gguf_header(f) for f in files]
    cfg = qwen4exp_config(hdrs[1].kv)
    index = Dict{String, Tuple{Any, GgufTensorInfo}}()
    for h in hdrs, t in h.tensors
        index[t.name] = (h, t)
    end
    return Qwen4expModel(cfg, index, Dict{String, Any}())
end

function _q4x_load(m::Qwen4expModel, name::String)
    haskey(m.index, name) || error("qwen4exp: no tensor $name")
    h, t = m.index[name]
    nd = length(t.dims)
    # ssm_a is stored as A itself; `_materialize_gguf_tensor` would apply log(-A).
    if nd == 1 || occursin("conv1d", name)
        return _dequant_f32(h.data, Int(h.data_start) + Int(t.offset), t.ggml_type, prod(t.dims))
    end
    nd == 3 && return _expert_slice(h, t)
    # The hyper-connection inject weights are (n_hc, n_embd * n_hc): too few
    # rows for the quant GEMV to pay off, so they widen to F32 (exact for BF16),
    # stored input-major so each output row is one contiguous run.
    endswith(name, "_inject.weight") &&
        return transpose(Matrix{Float32}(permutedims(_gguf_array(h, t))))
    (t.ggml_type == GGML_TYPE_F32 || endswith(name, "ffn_gate_inp.weight")) &&
        return _gguf_array(h, t)
    return _quant_matrix(h, t)
end

_q4x_w(m::Qwen4expModel, name::String) = get!(() -> _q4x_load(m, name), m.cache, name)

# x (T, in) -> (T, out)
function _q4x_mm(W::AbstractMatrix, x::AbstractMatrix)
    size(W, 1) <= _Q4X_SMALL_ROWS && x isa Matrix{Float32} &&
        (W isa Matrix{Float32} || W isa Transpose{Float32, Matrix{Float32}}) &&
        return _q4x_small_mm(W, x)
    # convert is a no-op on the Matrix{Float32} BLAS result; Matrix{Float32}(...) copied it.
    return convert(Matrix{Float32}, x * transpose(W))
end

# A handful of output rows: a serial loop reads W and each x column once, with
# no BLAS call (its threads fight the GEMV pool's spinning workers).
const _Q4X_SMALL_ROWS = 8

function _q4x_small_mm(W::Matrix{Float32}, x::Matrix{Float32})
    nout, nin = size(W)
    T = size(x, 1)
    size(x, 2) == nin || error("qwen4exp: x has $(size(x, 2)) features, weight has $nin")
    y = zeros(Float32, T, nout)
    @inbounds for i in 1:nin, o in 1:nout
        w = W[o, i]
        @simd for t in 1:T
            y[t, o] = muladd(x[t, i], w, y[t, o])
        end
    end
    return y
end

# Input-major weights: one token is a contiguous dot per output row; more
# tokens walk W once and keep the token loop contiguous.
function _q4x_small_mm(W::Transpose{Float32, Matrix{Float32}}, x::Matrix{Float32})
    Wt = parent(W)
    nin, nout = size(Wt)
    T = size(x, 1)
    size(x, 2) == nin || error("qwen4exp: x has $(size(x, 2)) features, weight has $nin")
    if T == 1
        y = Matrix{Float32}(undef, 1, nout)
        @inbounds for o in 1:nout
            acc = 0f0
            @simd for i in 1:nin
                acc = muladd(x[i], Wt[i, o], acc)
            end
            y[1, o] = acc
        end
        return y
    end
    y = zeros(Float32, T, nout)
    @inbounds for o in 1:nout, i in 1:nin
        w = Wt[i, o]
        @simd for t in 1:T
            y[t, o] = muladd(x[t, i], w, y[t, o])
        end
    end
    return y
end

function _q4x_mm(W::QuantMatrix, x::AbstractMatrix)
    y = zeros(Float32, size(x, 1), W.nrows)
    # The kernels only read x, so a Float32 matrix goes in as is.
    xf = x isa Matrix{Float32} ? x : Matrix{Float32}(x)
    _quant_half(W.ggml_type) ? _half_matmul!(y, xf, W) : _quant_matmul!(y, xf, W)
    return y
end

# The tensor cache is Dict{String, Any}; the assert gives callers a concrete type
# so their loops compile without dynamic dispatch.
_q4x_mm(m::Qwen4expModel, x::AbstractMatrix, name::String) =
    _q4x_mm(_q4x_w(m, name), x)::Matrix{Float32}

# x (T, in) into a caller buffer y (T, out). Every quant kernel writes each dst
# element once (no accumulate), so y needs no zeroing and the bits match
# _q4x_mm. Other weight types go through _q4x_mm and copy.
function _q4x_mm!(y::Matrix{Float32}, m::Qwen4expModel, x::Matrix{Float32}, name::String)
    W = _q4x_w(m, name)
    W isa QuantMatrix || return copyto!(y, _q4x_mm(W, x)::Matrix{Float32})
    _quant_half(W.ggml_type) ? _half_matmul!(y, x, W) : _quant_matmul!(y, x, W)
    return y
end

_q4x_rows_out(m::Qwen4expModel, name::String) = _q4x_rows_out(_q4x_w(m, name))
_q4x_rows_out(W::QuantMatrix) = W.nrows
_q4x_rows_out(W::AbstractMatrix) = size(W, 1)

# A (T, n) Float32 scratch matrix kept on the model under `slot`, for values
# that never leave the call that fills them. A new shape replaces the buffer.
# A call that did not win m.scratch_busy (own = false) gets a fresh matrix, so
# two forwards on one model never share a buffer.
function _q4x_buf(m::Qwen4expModel, slot::String, T::Int, n::Int, own::Bool)
    own || return Matrix{Float32}(undef, T, n)
    b = get(m.cache, slot, nothing)
    b isa Matrix{Float32} && size(b) == (T, n) && return b
    b = Matrix{Float32}(undef, T, n)
    m.cache[slot] = b
    return b
end

# Rows of an embedding table for 0-based ids, (length(ids), width).
_q4x_rows(W::AbstractMatrix, ids) = Matrix{Float32}(W[ids .+ 1, :])

function _q4x_rows(W::QuantMatrix, ids)
    out = zeros(Float32, length(ids), W.ne0)
    _quant_embedding_rows!(out, W, ids)
    return out
end

_q4x_moe!(y, x, router, g::GgufQuantSlice, u, d, k) =
    _cpu_moe_topk_swiglu_gguf!(y, x, router, g, u, d, k)
_q4x_moe!(y, x, router, g::AbstractArray, u, d, k) =
    _cpu_moe_topk_swiglu!(y, x, router, g, u, d, k)

_q4x_sigm(x) = 1 / (1 + exp(-x))
_q4x_silu(x) = x / (1 + exp(-x))
_q4x_softplus(x) = x > 20 ? x : log1p(exp(x))

# Chunks for an elementwise pass over `work` elements split along a dimension of
# length n: one below 64k elements (decode shapes stay on the calling thread).
_q4x_nchunks(n::Int, work::Int) = work < 65536 ? 1 : clamp(Threads.nthreads(), 1, n)

# Runs f(lo, hi) over 1:n split into nc contiguous ranges on the GEMV pool.
function _q4x_par_ranges(f, n::Int, nc::Int)
    nc <= 1 && return f(1, n)
    _pool_run_chunks!(nc) do c
        lo = div((c - 1) * n, nc) + 1
        hi = div(c * n, nc)
        hi >= lo && f(lo, hi)
        return nothing
    end
    return nothing
end

# RMS norm over contiguous groups of n columns. w has n entries (shared) or one
# per column. Token rows split across the pool; each (group, token) runs the
# same loop as a serial pass, so the bits do not depend on the split. A gate z
# (same shape as x) multiplies each normed value by sigmoid(z), the same
# Float32 product as `group_rms(...) .* sigm.(z)` without the two extra passes.
_q4x_group_rms(x::AbstractMatrix, n::Int, w, eps::Float64, z=nothing) =
    _q4x_group_rms!(Matrix{Float32}(undef, size(x)), x, n, w, eps, z)

function _q4x_group_rms!(y::Matrix{Float32}, x::AbstractMatrix, n::Int, w, eps::Float64, z=nothing)
    T, F = size(x)
    size(y) == (T, F) || error("qwen4exp: rms out $(size(y)) ≠ $((T, F))")
    z === nothing || size(z) == (T, F) ||
        error("qwen4exp: rms gate $(size(z)) ≠ $((T, F))")
    wo = length(w) == n ? 0 : n
    _q4x_par_ranges(T, _q4x_nchunks(T, T * F)) do t0, t1
        _q4x_group_rms_rows!(y, x, n, w, wo, eps, t0, t1, z)
    end
    return y
end

function _q4x_group_rms_rows!(y::Matrix{Float32}, x::AbstractMatrix, n::Int, w, wo::Int, eps::Float64, t0::Int, t1::Int, z)
    F = size(x, 2)
    @inbounds for g in 0:(F÷n-1), t in t0:t1
        acc = 0.0
        @simd for i in 1:n
            acc += Float64(x[t, g*n+i])^2
        end
        r = 1 / sqrt(acc / n + eps)
        o = g * wo
        if z === nothing
            @simd for i in 1:n
                y[t, g*n+i] = Float32(x[t, g*n+i] * r) * w[o+i]
            end
        else
            @simd for i in 1:n
                j = g * n + i
                y[t, j] = Float32(Float32(x[t, j] * r) * w[o+i]) * _q4x_sigm(z[t, j])
            end
        end
    end
    return nothing
end

# Depthwise causal conv over [hist; x]; w is ggml [k, C] flat. hist holds the
# last (k - 1) * dil input rows and is advanced in place. Channels split across
# the pool; each output is the same p-ordered sum, so the split leaves the bits
# alone. `_q4x_causal_conv_silu!` also returns silu of each output, the value
# `_q4x_silu.(y)` gives, from the same pass.
_q4x_causal_conv!(hist::Matrix{Float32}, x::AbstractMatrix, w, k::Int, dil::Int) =
    _q4x_causal_conv_impl!(hist, x, w, k, dil, nothing)

# y, s and xx (the [hist; x] rows) may be caller buffers; every element of
# each is written before it is read.
function _q4x_causal_conv_silu!(hist::Matrix{Float32}, x::AbstractMatrix, w, k::Int, dil::Int,
                                y=nothing, s=nothing, xx=nothing)
    s === nothing && (s = Matrix{Float32}(undef, size(x)))
    y = _q4x_causal_conv_impl!(hist, x, w, k, dil, s, y, xx)
    return y, s
end

function _q4x_causal_conv_impl!(hist::Matrix{Float32}, x::AbstractMatrix, w, k::Int, dil::Int, s,
                                y=nothing, xx=nothing)
    H = size(hist, 1)
    H == (k - 1) * dil || error("qwen4exp: conv history has $H rows, needs $((k - 1) * dil)")
    T, C = size(x)
    if xx === nothing
        xx = vcat(hist, x)
    else
        size(xx) == (H + T, C) || error("qwen4exp: conv buffer is $(size(xx)), needs $((H + T, C))")
        copyto!(view(xx, 1:H, :), hist)
        copyto!(view(xx, H+1:H+T, :), x)
    end
    y === nothing && (y = Matrix{Float32}(undef, T, C))
    size(y) == (T, C) || error("qwen4exp: conv output is $(size(y)), needs $((T, C))")
    _q4x_par_ranges(C, _q4x_nchunks(C, T * C)) do c0, c1
        _q4x_causal_conv_cols!(y, s, xx, w, k, dil, H, T, c0, c1)
    end
    hist .= @view xx[end-H+1:end, :]
    return y
end

function _q4x_causal_conv_cols!(y::Matrix{Float32}, s, xx::Matrix{Float32}, w, k::Int, dil::Int, H::Int, T::Int, c0::Int, c1::Int)
    @inbounds for c in c0:c1, t in 1:T
        acc = 0f0
        for p in 0:(k-1)
            acc += w[p+1+k*(c-1)] * xx[H+t-(k-1-p)*dil, c]
        end
        y[t, c] = acc
        s === nothing || (s[t, c] = _q4x_silu(y[t, c]))
    end
    return nothing
end

"""
    Qwen4expState(cfg)

Decode state: the position, per-GDN-layer conv history and delta state
(stored `(dv, dk, nv)`, head-contiguous), per-attention-layer K/V (post-norm, post-RoPE), the last `ngram - 1` tokens
for the PLE hash (EOS at the start) and the PLE conv history.
"""
mutable struct Qwen4expState
    pos::Int
    conv::Dict{Int, Matrix{Float32}}
    ssm::Dict{Int, Array{Float64, 3}}
    k::Dict{Int, Matrix{Float32}}
    v::Dict{Int, Matrix{Float32}}
    ple_tokens::Vector{Int}
    ple_conv::Dict{Int, Matrix{Float32}}
end

Qwen4expState(cfg::Qwen4expConfig) = Qwen4expState(
    0, Dict{Int, Matrix{Float32}}(), Dict{Int, Array{Float64, 3}}(),
    Dict{Int, Matrix{Float32}}(), Dict{Int, Matrix{Float32}}(),
    fill(cfg.ple_eos, cfg.ple_ngram - 1), Dict{Int, Matrix{Float32}}(),
)

_q4x_tap!(taps, name, x) = (taps === nothing || (taps[name] = copy(x)); x)

# 0-based per_layer_token_embd rows for the last T tokens of `toks`,
# (ngram - 1) * hpn per token, token-major. Context stops at the first EOS
# looking back; positions before it hash as EOS.
function _q4x_ple_rows(cfg::Qwen4expConfig, toks::Vector{Int}, T::Int)
    n = cfg.ple_ngram
    nh = (n - 1) * cfg.ple_hpn
    rows = Vector{Int}(undef, T * nh)
    ctx = Vector{Int}(undef, n)
    L = length(toks)
    for r in 1:T
        i = L - T + r
        ctx[1] = toks[i]
        cut = false
        for s in 1:(n-1)
            j = i - s
            cut = cut || j < 1 || toks[j] == cfg.ple_eos
            ctx[s+1] = cut ? cfg.ple_eos : toks[j]
        end
        for g in 2:n
            mixed = UInt64(0)
            for j in 1:g
                mixed ⊻= UInt64(ctx[j]) * cfg.ple_mult[j]
            end
            for e in 0:(cfg.ple_hpn-1)
                h = (g - 2) * cfg.ple_hpn + e
                rows[(r-1)*nh+h+1] = Int(mixed % cfg.ple_voc[h+1]) + cfg.ple_off[h+1]
            end
        end
    end
    return rows
end

function _q4x_ple_embed(m::Qwen4expModel, st::Qwen4expState, tokens::Vector{Int})
    cfg = m.cfg
    toks = vcat(st.ple_tokens, tokens)
    T = length(tokens)
    rows = _q4x_ple_rows(cfg, toks, T)
    buf = _q4x_rows(_q4x_w(m, "per_layer_token_embd.weight"), rows)::Matrix{Float32}
    nh = length(rows) ÷ T
    d = size(buf, 2)
    out = Matrix{Float32}(undef, T, nh * d)
    @inbounds for t in 1:T, h in 0:(nh-1), i in 1:d
        out[t, i+d*h] = buf[(t-1)*nh+h+1, i]
    end
    st.ple_tokens = toks[end-cfg.ple_ngram+2:end]
    return out
end

function _q4x_ple(m::Qwen4expModel, st::Qwen4expState, il::Int, emb, res, taps)
    cfg = m.cfg
    NE, HC = cfg.n_embd, cfg.n_hc
    p = "blk.$il."
    key = _q4x_group_rms(_q4x_mm(m, emb, p * "ple_key.weight"), NE,
        _q4x_w(m, p * "ple_norm_key.weight"), cfg.eps)
    val = _q4x_mm(m, emb, p * "ple_value.weight")
    qry = _q4x_group_rms(res, NE, _q4x_w(m, p * "ple_norm_query.weight"), cfg.eps)
    T = size(res, 1)
    gate = Matrix{Float32}(undef, T, HC)
    for t in 1:T, s in 0:(HC-1)
        acc = 0.0
        for i in 1:NE
            acc += Float64(key[t, i+NE*s]) * qry[t, i+NE*s]
        end
        sc = acc / sqrt(NE)
        gate[t, s+1] = _q4x_sigm(sign(sc) * sqrt(max(abs(sc), 1e-6)))
    end
    gated = Matrix{Float32}(undef, T, HC * NE)
    for t in 1:T, s in 0:(HC-1), i in 1:NE
        gated[t, i+NE*s] = val[t, i] * gate[t, s+1]
    end
    nrm = _q4x_group_rms(gated, NE, _q4x_w(m, p * "ple_norm_conv.weight"), cfg.eps)
    k, dil = cfg.ple_kernel, cfg.ple_ngram
    hist = get!(() -> zeros(Float32, (k - 1) * dil, HC * NE), st.ple_conv, il)
    conv = _q4x_silu.(_q4x_causal_conv!(hist, nrm, _q4x_w(m, p * "ple_conv1d.weight"), k, dil))
    _q4x_tap!(taps, "ple_gate-$il", gate)
    _q4x_tap!(taps, "ple_gated_value-$il", gated)
    _q4x_tap!(taps, "ple_conv_out-$il", conv)
    return res .+ gated .+ conv
end

# Hyper-connection read: (mixed (T, n_embd), inject (T, n_hc) or nothing).
function _q4x_hc_mix(m::Qwen4expModel, x, pre::String; inject::Bool=true)
    own = Threads.atomic_cas!(m.scratch_busy, 0, 1) == 0
    try
        return _q4x_hc_mix(m, x, pre, inject, own)
    finally
        own && Threads.atomic_xchg!(m.scratch_busy, 0)
    end
end

function _q4x_hc_mix(m::Qwen4expModel, x, pre::String, inject::Bool, own::Bool)
    cfg = m.cfg
    NE, HC = cfg.n_embd, cfg.n_hc
    T = size(x, 1)
    # xn, lo and gate never leave this call, so they live in reused buffers
    # (a fresh 20 MB matrix costs more in page faults and GC than the math).
    xn = _q4x_group_rms!(_q4x_buf(m, "scratch:hc_xn", T, size(x, 2), own), x, NE,
        _q4x_w(m, pre * "_norm.weight"), cfg.eps)
    dn, up = pre * "_down.weight", pre * "_up.weight"
    lo = _q4x_mm!(_q4x_buf(m, "scratch:hc_lo", T, _q4x_rows_out(m, dn), own), m, xn, dn)
    lo .= _q4x_silu.(lo ./ HC)
    gate = _q4x_mm!(_q4x_buf(m, "scratch:hc_gate", T, _q4x_rows_out(m, up), own), m, lo, up)
    # Stream s of a (T, n_hc * n_embd) matrix is the contiguous span
    # (s * T * NE) .+ (1:T * NE), so the mix runs over linear indices.
    TN = T * NE
    mixed = zeros(Float32, T, NE)
    # Each k sums its streams in s order, so splitting k keeps the bits.
    _q4x_par_ranges(TN, _q4x_nchunks(TN, TN * HC)) do k0, k1
        _q4x_hc_mix_range!(mixed, xn, gate, TN, HC, k0, k1)
    end
    return mixed, inject ? _q4x_mm(m, xn, pre * "_inject.weight") : nothing
end

function _q4x_hc_mix_range!(mixed::Matrix{Float32}, xn::Matrix{Float32}, gate::Matrix{Float32}, TN::Int, HC::Int, k0::Int, k1::Int)
    @inbounds for s in 0:(HC-1)
        o = TN * s
        for k in k0:k1
            gate[o+k] = _q4x_sigm(gate[o+k])
        end
    end
    @inbounds for s in 0:(HC-1)
        o = TN * s
        @simd for k in k0:k1
            mixed[k] += xn[o+k] * gate[o+k]
        end
    end
    @inbounds @simd for k in k0:k1
        mixed[k] /= HC
    end
    return nothing
end

# Stream s gets res + out * w with w = 2 sigm(inj[t, s] / n_hc). Columns split
# across the pool and each column runs down the tokens contiguously; every
# element is one add of one product, so the split does not change the bits.
function _q4x_hc_combine(cfg::Qwen4expConfig, res, out, inj)
    NE, HC = cfg.n_embd, cfg.n_hc
    T = size(res, 1)
    y = similar(res)
    wv = Matrix{Float32}(undef, T, HC)
    @inbounds for s in 1:HC, t in 1:T
        wv[t, s] = 2 * _q4x_sigm(inj[t, s] / HC)
    end
    F = NE * HC
    _q4x_par_ranges(F, _q4x_nchunks(F, T * F)) do c0, c1
        _q4x_hc_combine_cols!(y, res, out, wv, NE, T, c0, c1)
    end
    return y
end

function _q4x_hc_combine_cols!(y::AbstractMatrix, res::AbstractMatrix, out::AbstractMatrix, wv::Matrix{Float32}, NE::Int, T::Int, c0::Int, c1::Int)
    @inbounds for c in c0:c1
        s = (c - 1) ÷ NE
        i = c - NE * s
        for t in 1:T
            y[t, c] = res[t, c] + out[t, i] * wv[t, s+1]
        end
    end
    return nothing
end

function _q4x_gdn(m::Qwen4expModel, st::Qwen4expState, il::Int, x, taps)
    # Taps keep references to the intermediates, so a tapped call never uses
    # the shared scratch buffers.
    own = taps === nothing && Threads.atomic_cas!(m.scratch_busy, 0, 1) == 0
    try
        return _q4x_gdn(m, st, il, x, taps, own)
    finally
        own && Threads.atomic_xchg!(m.scratch_busy, 0)
    end
end

function _q4x_gdn(m::Qwen4expModel, st::Qwen4expState, il::Int, x, taps, own::Bool)
    cfg = m.cfg
    p = "blk.$il."
    T = size(x, 1)
    nk, dk, nv, dv = cfg.ssm_nk, cfg.ssm_dk, cfg.ssm_nv, cfg.ssm_dv
    xf = x isa Matrix{Float32} ? x : Matrix{Float32}(x)
    # qkv, z, the conv buffers, o and fo never leave this call (taps aside),
    # so they live in reused buffers; out is returned and stays fresh.
    nqkv = _q4x_rows_out(m, p * "attn_qkv.weight")
    qkv = _q4x_mm!(_q4x_buf(m, "scratch:gdn_qkv", T, nqkv, own), m, xf, p * "attn_qkv.weight")
    z = _q4x_mm!(_q4x_buf(m, "scratch:gdn_z", T, _q4x_rows_out(m, p * "attn_gate.weight"), own),
        m, xf, p * "attn_gate.weight")
    beta = _q4x_sigm.(_q4x_mm(m, x, p * "ssm_beta.weight"))
    alpha = _q4x_mm(m, x, p * "ssm_alpha.weight")
    sp = _q4x_softplus.(alpha .+ transpose(_q4x_w(m, p * "ssm_dt.bias")))
    g = sp .* transpose(_q4x_w(m, p * "ssm_a"))
    k = cfg.conv_kernel
    hist = get!(() -> zeros(Float32, k - 1, size(qkv, 2)), st.conv, il)
    craw, c = _q4x_causal_conv_silu!(hist, qkv, _q4x_w(m, p * "ssm_conv1d.weight"), k, 1,
        _q4x_buf(m, "scratch:gdn_craw", T, nqkv, own), _q4x_buf(m, "scratch:gdn_c", T, nqkv, own),
        _q4x_buf(m, "scratch:gdn_xx", size(hist, 1) + T, nqkv, own))
    nq = nk * dk
    # Views, not copies: the kernel widens q, k, v and beta to Float64 per
    # element, so the values match the old Float64 copies. g stays Float64
    # because the kernel takes exp(g) in g's own type.
    q = PermutedDimsArray(reshape(view(c, :, 1:nq), T, dk, nk), (1, 3, 2))
    kk = PermutedDimsArray(reshape(view(c, :, nq+1:2nq), T, dk, nk), (1, 3, 2))
    v = PermutedDimsArray(reshape(view(c, :, 2nq+1:size(c, 2)), T, dv, nv), (1, 3, 2))
    # The kernel only stores into out, so a Float32 out rounds each value once,
    # the same as Float32.(o) on a Float64 buffer, without the buffer.
    of = _q4x_buf(m, "scratch:gdn_o", T, nv * dv, own)
    o = reshape(of, T, dv, nv)
    # The delta state is stored (dv, dk, nv) so each value head is one
    # contiguous block; the kernel sees it as (nv, dk, dv) through the view.
    # Held (nv, dk, dv) directly, a head's values sit nv apart and the head
    # threads share cache lines (4.3 ms against 0.85 ms per layer on decode).
    raw = get!(() -> zeros(Float64, dv, dk, nv), st.ssm, il)
    _cpu_gated_delta_rule!(PermutedDimsArray(o, (1, 3, 2)), q, kk, v, Float64.(g),
        beta, PermutedDimsArray(raw, (3, 2, 1)); l2norm=true, v_head_tiled=true)
    fo = _q4x_group_rms!(_q4x_buf(m, "scratch:gdn_fo", T, nv * dv, own), of, dv,
        _q4x_w(m, p * "ssm_norm.weight"), cfg.eps, z)
    out = _q4x_mm(m, fo, p * "ssm_out.weight")
    if taps !== nothing
        for (n, a) in (("linear_attn_qkv_mixed", qkv), ("z", z), ("beta_sigmoid", beta),
            ("alpha", alpha), ("a_softplus", sp), ("gate", g), ("conv_output_raw", craw),
            ("conv_output_silu", c), ("attn_output", of), ("final_output", fo),
            ("linear_attn_out", out))
            taps["$n-$il"] = a
        end
    end
    return out
end

# NeoX rotation of the first n_rot dims of each head, at absolute positions
# p0, p0 + 1, ….
function _q4x_rope!(x::Matrix{Float32}, cfg::Qwen4expConfig, nh::Int, p0::Int)
    half = cfg.n_rot ÷ 2
    hd = cfg.head_dim
    @inbounds for i in 0:(half-1)
        f = cfg.rope_base^(-2i / cfg.n_rot)
        for t in 1:size(x, 1)
            θ = (p0 + t - 1) * f
            c, s = cos(θ), sin(θ)
            for h in 0:(nh-1)
                a = h * hd + i + 1
                b = a + half
                x1, x2 = x[t, a], x[t, b]
                x[t, a] = Float32(x1 * c - x2 * s)
                x[t, b] = Float32(x1 * s + x2 * c)
            end
        end
    end
    return x
end

function _q4x_attn(m::Qwen4expModel, st::Qwen4expState, il::Int, x, taps)
    cfg = m.cfg
    p = "blk.$il."
    T = size(x, 1)
    NH, NKV, HD = cfg.n_head, cfg.n_head_kv, cfg.head_dim
    qf = _q4x_mm(m, x, p * "attn_q.weight")
    q = Matrix{Float32}(undef, T, NH * HD)
    gt = Matrix{Float32}(undef, T, NH * HD)
    @inbounds for h in 0:(NH-1), d in 1:HD, t in 1:T
        q[t, h*HD+d] = qf[t, 2h*HD+d]
        gt[t, h*HD+d] = qf[t, 2h*HD+HD+d]
    end
    q = _q4x_group_rms(q, HD, _q4x_w(m, p * "attn_q_norm.weight"), cfg.eps)
    k = _q4x_group_rms(_q4x_mm(m, x, p * "attn_k.weight"), HD,
        _q4x_w(m, p * "attn_k_norm.weight"), cfg.eps)
    v = _q4x_mm(m, x, p * "attn_v.weight")
    _q4x_tap!(taps, "Qcur_normed-$il", q)
    _q4x_tap!(taps, "Kcur_normed-$il", k)
    _q4x_tap!(taps, "Vcur-$il", v)
    p0 = st.pos
    _q4x_rope!(q, cfg, NH, p0)
    _q4x_rope!(k, cfg, NKV, p0)
    K = st.k[il] = vcat(get(st.k, il, zeros(Float32, 0, NKV * HD)), k)
    V = st.v[il] = vcat(get(st.v, il, zeros(Float32, 0, NKV * HD)), v)
    L = size(K, 1)
    o = Matrix{Float32}(undef, T, NH * HD)
    scale = Float32(1 / sqrt(HD))
    grp = NH ÷ NKV
    for h in 0:(NH-1)
        kc = (h ÷ grp) * HD+1:(h ÷ grp + 1) * HD
        S = (@view(q[:, h*HD+1:(h+1)*HD]) * transpose(@view K[:, kc])) .* scale
        @inbounds for t in 1:T
            lim = p0 + t
            mx = -Inf32
            for u in 1:lim
                mx = max(mx, S[t, u])
            end
            z = 0f0
            for u in 1:L
                e = u <= lim ? exp(S[t, u] - mx) : 0f0
                S[t, u] = e
                z += e
            end
            for u in 1:lim
                S[t, u] /= z
            end
        end
        o[:, h*HD+1:(h+1)*HD] = S * @view V[:, kc]
    end
    og = o .* _q4x_sigm.(gt)
    out = _q4x_mm(m, og, p * "attn_output.weight")
    _q4x_tap!(taps, "Qcur-$il", q)
    _q4x_tap!(taps, "Kcur-$il", k)
    _q4x_tap!(taps, "attn_pregate-$il", o)
    _q4x_tap!(taps, "attn_gated-$il", og)
    _q4x_tap!(taps, "attn_output-$il", out)
    return out
end

function _q4x_ffn(m::Qwen4expModel, il::Int, x, taps)
    cfg = m.cfg
    p = "blk.$il."
    router = _q4x_w(m, p * "ffn_gate_inp.weight")
    xf = x isa Matrix{Float32} ? x : Matrix{Float32}(x)
    moe = zeros(Float32, size(xf)...)
    _q4x_moe!(moe, xf, router, _q4x_w(m, p * "ffn_gate_exps.weight"),
        _q4x_w(m, p * "ffn_up_exps.weight"), _q4x_w(m, p * "ffn_down_exps.weight"), cfg.n_used)
    sh = _q4x_mm(m, _q4x_silu.(_q4x_mm(m, xf, p * "ffn_gate_shexp.weight")) .*
                    _q4x_mm(m, xf, p * "ffn_up_shexp.weight"), p * "ffn_down_shexp.weight")
    sg = xf * _q4x_w(m, p * "ffn_gate_inp_shexp.weight")
    shg = sh .* _q4x_sigm.(sg)
    out = moe .+ shg
    if taps !== nothing
        taps["ffn_moe_logits-$il"] = Matrix{Float32}(xf * transpose(router))
        taps["ffn_moe_out-$il"] = moe
        taps["ffn_shexp-$il"] = sh
        taps["shared_expert_gate-$il"] = reshape(sg, :, 1)
        taps["ffn_shexp_gated-$il"] = shg
        taps["ffn_out-$il"] = out
    end
    return out
end

"""
    qwen4exp_forward!(m, st, tokens; n_layers=m.cfg.n_layer, taps=nothing) -> residual

Run `tokens` (0-based ids) at positions `st.pos`, `st.pos + 1`, … and advance
`st`. Returns the residual streams (T, n_hc * n_embd) after `n_layers` blocks.
A truncated run leaves the state of the skipped blocks behind, so use it for
comparisons, not decoding. `taps` (a Dict) collects intermediates under the
llama.cpp eval-callback node names, each (T, features).
"""
function qwen4exp_forward!(m::Qwen4expModel, st::Qwen4expState, tokens::AbstractVector{<:Integer};
                           n_layers::Int=m.cfg.n_layer, taps=nothing)
    cfg = m.cfg
    toks = Vector{Int}(tokens)
    T = length(toks)
    T > 0 || error("qwen4exp_forward!: no tokens")
    0 < n_layers <= cfg.n_layer || error("qwen4exp_forward!: n_layers $n_layers out of 1..$(cfg.n_layer)")
    if any(il -> cfg.attn_layer[il+1] && cfg.compress[il+1] > 0, 0:(n_layers-1)) &&
       st.pos + T > cfg.qsa_top_k
        error("qwen4exp_forward!: context $(st.pos + T) exceeds the QSA top_k $(cfg.qsa_top_k); " *
              "the sparse-attention indexer is not implemented")
    end
    e = _q4x_rows(_q4x_w(m, "token_embd.weight"), toks)::Matrix{Float32}
    res = repeat(e, 1, cfg.n_hc)
    pemb = isempty(cfg.ple_layers) ? nothing : _q4x_ple_embed(m, st, toks)
    for il in 0:(n_layers-1)
        il in cfg.ple_layers && (res = _q4x_ple(m, st, il, pemb, res, taps))
        cur, inj = _q4x_hc_mix(m, res, "blk.$il.hc_attn")
        _q4x_tap!(taps, "hc_mixed-$il", cur)
        cur = cfg.attn_layer[il+1] ? _q4x_attn(m, st, il, cur, taps) : _q4x_gdn(m, st, il, cur, taps)
        res = _q4x_hc_combine(cfg, res, cur, inj)
        _q4x_tap!(taps, "hc_combine-$il", res)
        cur, inj = _q4x_hc_mix(m, res, "blk.$il.hc_ffn")
        cur = _q4x_ffn(m, il, cur, taps)
        res = _q4x_hc_combine(cfg, res, cur, inj)
        _q4x_tap!(taps, "l_last-$il", res)
    end
    st.pos += T
    return res
end

"""
    qwen4exp_logits(m, res; taps=nothing) -> Vector{Float32}

Logits for the last row of the residual from `qwen4exp_forward!`.
"""
function qwen4exp_logits(m::Qwen4expModel, res::AbstractMatrix; taps=nothing)
    h, _ = _q4x_hc_mix(m, res[end:end, :], "output_hc"; inject=false)
    logits = _q4x_mm(m, h, "output.weight")
    _q4x_tap!(taps, "result_norm", h)
    _q4x_tap!(taps, "result_output", logits)
    return vec(logits)
end

"""
    qwen4exp_device(backend, m) -> device model

Backend hook. A backend extension returns a device mirror of `m` whose
`qwen4exp_forward!` and `qwen4exp_logits` match the CPU taps by name.
"""
function qwen4exp_device end

"""
    qwen4exp_device_state(dm) -> device decode state

Backend hook paired with `qwen4exp_device`.
"""
function qwen4exp_device_state end
