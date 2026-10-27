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
end

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
    (t.ggml_type == GGML_TYPE_F32 || endswith(name, "ffn_gate_inp.weight")) &&
        return _gguf_array(h, t)
    return _quant_matrix(h, t)
end

_q4x_w(m::Qwen4expModel, name::String) = get!(() -> _q4x_load(m, name), m.cache, name)

# x (T, in) -> (T, out)
_q4x_mm(W::AbstractMatrix, x::AbstractMatrix) = Matrix{Float32}(x * transpose(W))

function _q4x_mm(W::QuantMatrix, x::AbstractMatrix)
    y = zeros(Float32, size(x, 1), W.nrows)
    xf = Matrix{Float32}(x)
    _quant_half(W.ggml_type) ? _half_matmul!(y, xf, W) : _quant_matmul!(y, xf, W)
    return y
end

_q4x_mm(m::Qwen4expModel, x::AbstractMatrix, name::String) = _q4x_mm(_q4x_w(m, name), x)

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

# RMS norm over contiguous groups of n columns. w has n entries (shared) or one
# per column.
function _q4x_group_rms(x::AbstractMatrix, n::Int, w, eps::Float64)
    T, F = size(x)
    y = Matrix{Float32}(undef, T, F)
    @inbounds for g in 0:(F÷n-1), t in 1:T
        acc = 0.0
        for i in 1:n
            acc += Float64(x[t, g*n+i])^2
        end
        r = 1 / sqrt(acc / n + eps)
        for i in 1:n
            y[t, g*n+i] = Float32(x[t, g*n+i] * r) * w[length(w) == n ? i : g*n+i]
        end
    end
    return y
end

# Depthwise causal conv over [hist; x]; w is ggml [k, C] flat. hist holds the
# last (k - 1) * dil input rows and is advanced in place.
function _q4x_causal_conv!(hist::Matrix{Float32}, x::AbstractMatrix, w, k::Int, dil::Int)
    H = size(hist, 1)
    H == (k - 1) * dil || error("qwen4exp: conv history has $H rows, needs $((k - 1) * dil)")
    T, C = size(x)
    xx = vcat(hist, x)
    y = Matrix{Float32}(undef, T, C)
    @inbounds for c in 1:C, t in 1:T
        acc = 0f0
        for p in 0:(k-1)
            acc += w[p+1+k*(c-1)] * xx[H+t-(k-1-p)*dil, c]
        end
        y[t, c] = acc
    end
    hist .= @view xx[end-H+1:end, :]
    return y
end

"""
    Qwen4expState(cfg)

Decode state: the position, per-GDN-layer conv history and delta state,
per-attention-layer K/V (post-norm, post-RoPE), the last `ngram - 1` tokens
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
    buf = _q4x_rows(_q4x_w(m, "per_layer_token_embd.weight"), rows)
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
    cfg = m.cfg
    NE, HC = cfg.n_embd, cfg.n_hc
    xn = _q4x_group_rms(x, NE, _q4x_w(m, pre * "_norm.weight"), cfg.eps)
    lo = _q4x_silu.(_q4x_mm(m, xn, pre * "_down.weight") ./ HC)
    gate = _q4x_mm(m, lo, pre * "_up.weight")
    T = size(x, 1)
    mixed = Matrix{Float32}(undef, T, NE)
    @inbounds for i in 1:NE, t in 1:T
        acc = 0f0
        for s in 0:(HC-1)
            acc += xn[t, i+NE*s] * _q4x_sigm(gate[t, i+NE*s])
        end
        mixed[t, i] = acc / HC
    end
    return mixed, inject ? _q4x_mm(m, xn, pre * "_inject.weight") : nothing
end

function _q4x_hc_combine(cfg::Qwen4expConfig, res, out, inj)
    NE, HC = cfg.n_embd, cfg.n_hc
    y = copy(res)
    @inbounds for s in 0:(HC-1), t in 1:size(res, 1)
        w = 2 * _q4x_sigm(inj[t, s+1] / HC)
        for i in 1:NE
            y[t, i+NE*s] += out[t, i] * w
        end
    end
    return y
end

function _q4x_gdn(m::Qwen4expModel, st::Qwen4expState, il::Int, x, taps)
    cfg = m.cfg
    p = "blk.$il."
    T = size(x, 1)
    nk, dk, nv, dv = cfg.ssm_nk, cfg.ssm_dk, cfg.ssm_nv, cfg.ssm_dv
    qkv = _q4x_mm(m, x, p * "attn_qkv.weight")
    z = _q4x_mm(m, x, p * "attn_gate.weight")
    beta = _q4x_sigm.(_q4x_mm(m, x, p * "ssm_beta.weight"))
    alpha = _q4x_mm(m, x, p * "ssm_alpha.weight")
    sp = _q4x_softplus.(alpha .+ transpose(_q4x_w(m, p * "ssm_dt.bias")))
    g = sp .* transpose(_q4x_w(m, p * "ssm_a"))
    k = cfg.conv_kernel
    hist = get!(() -> zeros(Float32, k - 1, size(qkv, 2)), st.conv, il)
    craw = _q4x_causal_conv!(hist, qkv, _q4x_w(m, p * "ssm_conv1d.weight"), k, 1)
    c = _q4x_silu.(craw)
    nq = nk * dk
    q = permutedims(reshape(c[:, 1:nq], T, dk, nk), (1, 3, 2))
    kk = permutedims(reshape(c[:, nq+1:2nq], T, dk, nk), (1, 3, 2))
    v = permutedims(reshape(c[:, 2nq+1:end], T, dv, nv), (1, 3, 2))
    o = zeros(Float64, T, nv, dv)
    state = get!(() -> zeros(Float64, nv, dk, dv), st.ssm, il)
    _cpu_gated_delta_rule!(o, Float64.(q), Float64.(kk), Float64.(v), Float64.(g),
        Float64.(beta), state; l2norm=true, v_head_tiled=true)
    of = Float32.(reshape(permutedims(o, (1, 3, 2)), T, nv * dv))
    fo = _q4x_group_rms(of, dv, _q4x_w(m, p * "ssm_norm.weight"), cfg.eps) .* _q4x_sigm.(z)
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
    xf = Matrix{Float32}(x)
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
    e = _q4x_rows(_q4x_w(m, "token_embd.weight"), toks)
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
