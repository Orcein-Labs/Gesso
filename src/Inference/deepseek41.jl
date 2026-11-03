# DeepSeek-V4.1-Flash text CPU forward, after llama.cpp `src/models/deepseek41.cpp`
# and the reference `inference_model.py`. This file holds the metadata reader
# and the spec; the forward lands with `deepseek41_forward!`.
#
# Every layer reads and writes `hc` parallel residual streams of width `n_embd`
# (hyper-connections). Attention is multi-query with a learned sink per head:
# a sliding window of `attention.sliding_window` tokens, plus on compressed
# layers a pooled KV (ratio 1 or 2) and a top-k block selection from an indexer.
# Every layer routes experts by a sqrtsoftplus score with a selection bias
# (`exp_probs_b`); there is no token-id routing table. Layers listed in
# `ple.layers` add an Engram n-gram embedding.
#
# The oracle follows llama.cpp's F32 path: no activation quantization is
# simulated, unlike the fp8/fp4 steps in the reference `model.py`.

struct Deepseek41Config
    n_embd::Int
    n_layer::Int
    n_head::Int
    n_head_kv::Int
    head_dim::Int             # attention.key_length; the single KV head is this wide
    n_rot::Int                # rope.dimension_count, the RoPE tail of each head
    rope_base::Float64        # layers without compression
    compress_rope_base::Float64
    rope_factor::Float64      # YaRN factor on compressed layers
    rope_orig_ctx::Int
    eps::Float64
    q_lora::Int
    window::Int               # attention.sliding_window
    o_groups::Int
    o_lora::Int
    compress::Vector{Int}     # per layer (1-based index il+1): 0 SWA only, 1 or 2 compressed
    kv_src::Vector{Int}       # per layer: 0-based latest KV source at or before it, -1 if none
    idx_src::Vector{Int}      # per layer: 0-based latest indexer source, -1 if none
    idx_n_head::Int
    idx_head_dim::Int
    idx_top_k::Int
    cand_src::Int             # attention.indexer.candidate_source_layer, -1 if none
    cand_top_k_blocks::Int
    cand_block_size::Int
    hc::Int
    sinkhorn_iters::Int
    hc_eps::Float64
    n_exp::Int
    n_used::Int
    exp_ff::Int
    n_shared::Int
    route_scale::Float64
    route_norm::Bool
    clamp_exp::Vector{Float64}
    clamp_shexp::Vector{Float64}
    ple_layers::Vector{Int}   # 0-based Engram layers
    ple_ngram::Int
    ple_hpn::Int
    ple_head_dim::Int
    ple_mult::Vector{UInt64}  # per Engram module, ngram values
    ple_off::Vector{Int}      # per Engram module and head
    ple_voc::Vector{Int}      # per Engram module and head (bucket count)
    engram_pad::Int           # raw engram.pad_token_id; mapped through token_map before use
    token_map::Vector{Int}    # engram.token_map: token id -> compressed id
    vocab::Int
end

# SQRT_SOFTPLUS in llama.cpp's expert_gating_func enum.
const _DSV41_GATING_SQRT_SOFTPLUS = 4

"""
    deepseek41_config(kv) -> Deepseek41Config

Read the DeepSeek-V4.1 architecture from GGUF metadata. Every value comes from
the file; a missing key or inconsistent layout is an error. The checks mirror
llama.cpp's loader: compress ratios 0-2, each compressed layer matched to a KV
source and an indexer source of the same ratio, candidate selection pointing at
an indexer source, and Engram metadata of consistent size.
"""
function deepseek41_config(kv::AbstractDict)
    arch = String(get(kv, "general.architecture", ""))
    arch == "deepseek41" || error("deepseek41_config: general.architecture is $(repr(arch))")
    req(key) = (k = "deepseek41.$key"; haskey(kv, k) ? kv[k] : error("deepseek41_config: missing $k"))
    opt(key, default) = get(kv, "deepseek41.$key", default)
    ints(v) = Int[Int(x) for x in v]
    floats(v, n) = v isa AbstractVector ? Float64[Float64(x) for x in v] : fill(Float64(v), n)

    n_layer = Int(req("block_count"))
    n_layer > 0 || error("deepseek41_config: block_count must be > 0")
    n_embd = Int(req("embedding_length"))
    n_head = Int(req("attention.head_count"))
    head_dim = Int(req("attention.key_length"))
    n_rot = Int(req("rope.dimension_count"))
    n_rot <= head_dim && n_rot % 2 == 0 ||
        error("deepseek41_config: rope.dimension_count $n_rot must be even and at most key_length $head_dim")
    n_head_kv = Int(req("attention.head_count_kv"))
    n_head_kv == 1 || error("deepseek41_config: attention.head_count_kv must be 1 (multi-query), got $n_head_kv")

    compress = ints(req("attention.compress_ratios"))
    length(compress) >= n_layer || error("deepseek41_config: compress_ratios is shorter than block_count")
    compress = compress[1:n_layer]
    any(r -> r < 0 || r > 2, compress) &&
        error("deepseek41_config: compress ratios must be 0, 1 or 2, got $(compress)")

    kv_src_layers = ints(req("attention.kv_source_layers"))
    idx_src_layers = ints(req("attention.indexer.source_layers"))
    any(il -> il < 0 || il >= n_layer, kv_src_layers) &&
        error("deepseek41_config: kv source layer out of range")
    any(il -> il < 0 || il >= n_layer, idx_src_layers) &&
        error("deepseek41_config: indexer source layer out of range")
    kv_src = fill(-1, n_layer)
    idx_src = fill(-1, n_layer)
    last_kv = -1
    last_idx = -1
    for il in 0:(n_layer - 1)
        ratio = compress[il + 1]
        is_kv = il in kv_src_layers
        is_idx = il in idx_src_layers
        (is_kv || is_idx) && ratio == 0 &&
            error("deepseek41_config: source layer $il has compress ratio 0")
        last_kv = is_kv ? il : last_kv
        last_idx = is_idx ? il : last_idx
        ratio == 0 && continue
        (last_kv < 0 || last_idx < last_kv || compress[last_kv + 1] != ratio ||
            compress[last_idx + 1] != ratio) &&
            error("deepseek41_config: layer $il has compress ratio $ratio without a matching kv/indexer source")
        kv_src[il + 1] = last_kv
        idx_src[il + 1] = last_idx
    end

    idx_top_k = Int(req("attention.indexer.top_k"))
    idx_top_k > 0 || error("deepseek41_config: attention.indexer.top_k must be > 0")

    cand_src = Int(opt("attention.indexer.candidate_source_layer", -1))
    cand_top_k_blocks = 0
    cand_block_size = 0
    if cand_src >= 0
        cand_top_k_blocks = Int(req("attention.indexer.candidate_top_k_blocks"))
        cand_block_size = Int(req("attention.indexer.candidate_block_size"))
        (cand_src >= n_layer || idx_src[cand_src + 1] != cand_src || cand_block_size <= 0 ||
            cand_top_k_blocks <= 0) &&
            error("deepseek41_config: invalid candidate selection (the source must be an indexer source, " *
                  "block size and top-k blocks > 0)")
        for il in (cand_src + 1):(n_layer - 1)
            idx_src[il + 1] == il && compress[il + 1] != compress[cand_src + 1] &&
                error("deepseek41_config: candidate consumers must share the source's compress ratio")
        end
    end

    hc = Int(req("hyper_connection.count"))
    hc > 0 || error("deepseek41_config: hyper_connection.count must be > 0")
    sinkhorn_iters = Int(req("hyper_connection.sinkhorn_iterations"))
    hc_eps = Float64(req("hyper_connection.epsilon"))

    gating = Int(req("expert_gating_func"))
    gating == _DSV41_GATING_SQRT_SOFTPLUS ||
        error("deepseek41_config: expert_gating_func must be sqrtsoftplus ($_DSV41_GATING_SQRT_SOFTPLUS), got $gating")
    n_exp = Int(req("expert_count"))
    n_used = Int(req("expert_used_count"))
    n_used <= n_exp || error("deepseek41_config: expert_used_count $n_used exceeds expert_count $n_exp")
    exp_ff = Int(req("expert_feed_forward_length"))
    n_shared = Int(req("expert_shared_count"))
    route_scale = Float64(req("expert_weights_scale"))
    route_norm = Bool(req("expert_weights_norm"))
    clamp_exp = floats(req("swiglu_clamp_exp"), n_layer)
    clamp_shexp = haskey(kv, "deepseek41.swiglu_clamp_shexp") ?
        floats(kv["deepseek41.swiglu_clamp_shexp"], n_layer) : copy(clamp_exp)

    ple_layers = ints(opt("ple.layers", Int[]))
    ple_ngram = 0
    ple_hpn = 0
    ple_head_dim = 0
    ple_mult = UInt64[]
    ple_off = Int[]
    ple_voc = Int[]
    engram_pad = 0
    token_map = Int[]
    if !isempty(ple_layers)
        ple_layers == sort(unique(ple_layers)) && all(il -> 0 <= il < n_layer, ple_layers) ||
            error("deepseek41_config: ple.layers must be ascending layers in range")
        ple_ngram = Int(req("ple.ngram_size"))
        ple_hpn = Int(req("ple.heads_per_ngram"))
        ple_head_dim = Int(req("embedding_length_per_layer"))
        ple_mult = UInt64[UInt64(x) for x in req("ple.layer_multipliers")]
        ple_off = ints(req("ple.head_offsets"))
        ple_voc = ints(req("ple.head_vocab_sizes"))
        engram_pad = Int(req("engram.pad_token_id"))
        token_map = ints(req("engram.token_map"))
        n_e = length(ple_layers)
        n_cols = ple_ngram >= 2 ? (ple_ngram - 1) * ple_hpn : 0
        (n_cols == 0 || ple_head_dim == 0 || length(ple_mult) != n_e * ple_ngram ||
            length(ple_voc) != n_e * n_cols || length(ple_off) != n_e * n_cols ||
            engram_pad < 0 || engram_pad >= length(token_map)) &&
            error("deepseek41_config: engram metadata is inconsistent")
        for c in eachindex(ple_voc)
            (ple_voc[c] < 2 || ple_off[c] < 0 || ple_off[c] + ple_voc[c] > typemax(Int32)) &&
                error("deepseek41_config: engram head $c has a bucket range out of range")
        end
    end

    toks = get(kv, "tokenizer.ggml.tokens", nothing)
    vocab = toks isa Integer ? Int(toks) : toks === nothing ? 0 : length(toks)

    return Deepseek41Config(
        n_embd, n_layer, n_head, n_head_kv, head_dim, n_rot,
        Float64(req("rope.freq_base")),
        Float64(req("attention.compress_rope_freq_base")),
        Float64(req("rope.scaling.factor")),
        Int(req("rope.scaling.original_context_length")),
        Float64(req("attention.layer_norm_rms_epsilon")),
        Int(req("attention.q_lora_rank")),
        Int(req("attention.sliding_window")),
        Int(req("attention.output_group_count")),
        Int(req("attention.output_lora_rank")),
        compress, kv_src, idx_src,
        Int(req("attention.indexer.head_count")),
        Int(req("attention.indexer.key_length")),
        idx_top_k, cand_src, cand_top_k_blocks, cand_block_size,
        hc, sinkhorn_iters, hc_eps,
        n_exp, n_used, exp_ff, n_shared, route_scale, route_norm,
        clamp_exp, clamp_shexp,
        ple_layers, ple_ngram, ple_hpn, ple_head_dim, ple_mult, ple_off, ple_voc,
        engram_pad, token_map, vocab,
    )
end

"""
`deepseek41` (DeepSeek-V4.1-Flash) GGUF spec. It records the shape and the
features so the import report can describe the file. `moe` is left `nothing`:
`MoeSpec` only knows softmax and sigmoid gating, and the sqrtsoftplus router is
recorded as the `:sqrtsoftplus_moe` feature; the forward is
`deepseek41_forward!`, not the generic block runner. The DSpark / MTP draft
head is not part of the text model and is not read.
"""
function _spec_from_gguf_deepseek41(kv::AbstractDict, tensor_names)
    cfg = deepseek41_config(kv)
    cfg.vocab > 0 || throw(_missing_metadata("tokenizer.ggml.tokens"))
    features = Set{Symbol}([:hyper_connections, :sparse_attention, :compressed_kv,
        :sinks, :grouped_output, :sqrtsoftplus_moe])
    isempty(cfg.ple_layers) || push!(features, :ple)
    return ArchitectureSpec(;
        family=:deepseek41,
        hidden_size=cfg.n_embd,
        num_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        n_kv_heads=cfg.n_head_kv,
        vocab_size=cfg.vocab,
        intermediate_size=cfg.exp_ff,
        head_dim=cfg.head_dim,
        rope=RoPEPolicy(;
            theta=cfg.rope_base, rotary_dim=cfg.n_rot, interleaved=true),
        tie_word_embeddings=tensor_names !== nothing && !("output.weight" in tensor_names),
        features,
        rms_norm_eps=cfg.eps,
        moe=nothing,
    )
end

# ------------------------------------------------------------------ weights

"""
    Deepseek41Model

The config plus a lazy tensor store, as in `Qwen4expModel`. GGUF tensors are
mapped on first use: 1-D tensors as flat Float32, 3-D tensors (experts and the
per-group `attn_output_a`) as `GgufQuantSlice` banks, the router, hyper-
connection mixers and Engram gate as dense (out, in) matrices, and other 2-D
weights as `QuantMatrix`. A model built from a `Dict` of dense arrays uses the
same names and layouts, with 3-D banks as (in, out, n_bank) arrays, the `_expert_slice` order.

Tensors the text forward never reads (the DSpark / MTP draft head, vision
experts) are indexed but never mapped, so their presence does not fail a load.
"""
struct Deepseek41Model
    cfg::Deepseek41Config
    index::Dict{String, Tuple{Any, GgufTensorInfo}}
    cache::Dict{String, Any}
end

Deepseek41Model(cfg::Deepseek41Config, tensors::AbstractDict) =
    Deepseek41Model(cfg, Dict{String, Tuple{Any, GgufTensorInfo}}(), Dict{String, Any}(tensors))

# Tensors multiplied as dense Float32 matrices whatever their storage type.
_ds41_dense(name::AbstractString) =
    endswith(name, "ffn_gate_inp.weight") || endswith(name, "hc_attn_fn.weight") ||
    endswith(name, "hc_ffn_fn.weight") || endswith(name, "engram_gate.weight") ||
    endswith(name, "indexer.proj.weight")

function _ds41_load(m::Deepseek41Model, name::String)
    haskey(m.index, name) || error("deepseek41: no tensor $name")
    h, t = m.index[name]
    nd = length(t.dims)
    nd == 1 && return _dequant_f32(h.data, Int(h.data_start) + Int(t.offset), t.ggml_type, prod(t.dims))
    nd == 3 && return _expert_slice(h, t)
    (t.ggml_type == GGML_TYPE_F32 || _ds41_dense(name)) && return _gguf_array(h, t)
    return _quant_matrix(h, t)
end

_ds41_w(m::Deepseek41Model, name::String) = get!(() -> _ds41_load(m, name), m.cache, name)

# x (T, in) -> (T, out)
_ds41_mm(W::AbstractMatrix, x::AbstractMatrix) = Matrix{Float32}(x * transpose(W))

function _ds41_mm(W::QuantMatrix, x::AbstractMatrix)
    y = zeros(Float32, size(x, 1), W.nrows)
    xf = Matrix{Float32}(x)
    _quant_half(W.ggml_type) ? _half_matmul!(y, xf, W) : _quant_matmul!(y, xf, W)
    return y
end

_ds41_mm(m::Deepseek41Model, x::AbstractMatrix, name::String) = _ds41_mm(_ds41_w(m, name), x)

# Rows of an embedding table for 0-based ids, (length(ids), width).
_ds41_rows(W::AbstractMatrix, ids) = Matrix{Float32}(W[ids .+ 1, :])

function _ds41_rows(W::QuantMatrix, ids)
    out = zeros(Float32, length(ids), W.ne0)
    _quant_embedding_rows!(out, W, ids)
    return out
end

# y = W[e, :, :] * x for one bank member.
_ds41_bank_mv!(y::AbstractVector{Float32}, x::AbstractVector{Float32}, W::GgufQuantSlice, e::Int) =
    _qgemv_expert!(y, x, W, e; alpha=1.0f0, beta=0.0f0)

function _ds41_bank_mv!(y::AbstractVector{Float32}, x::AbstractVector{Float32},
                        W::AbstractArray{<:Any, 3}, e::Int)
    # dense banks are (in, out, n_bank), the layout `_expert_slice` gives: y = W[:, :, e]' * x
    y .= Float32.(transpose(@view(W[:, :, e])) * x)
    return y
end

"""
    load_deepseek41(path) -> Deepseek41Model

Open a DeepSeek-V4.1 GGUF. `path` may be any shard of a `-0000N-of-0000M.gguf`
split; every sibling is read so that tensors spread over the shards resolve.
"""
function load_deepseek41(path::AbstractString)
    m = match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
    files = m === nothing ? [path] :
        [string(m[1], "-", lpad(i, 5, '0'), "-of-", m[3], ".gguf") for i in 1:parse(Int, m[3])]
    hdrs = [gguf_header(f) for f in files]
    cfg = deepseek41_config(hdrs[1].kv)
    index = Dict{String, Tuple{Any, GgufTensorInfo}}()
    for h in hdrs, t in h.tensors
        index[t.name] = (h, t)
    end
    return Deepseek41Model(cfg, index, Dict{String, Any}())
end

# ------------------------------------------------------------------ state

"""
    Deepseek41State(cfg)

Decode state. `pos` is the next absolute position. `raw[il]` holds the last
`window` post-RoPE K rows of layer `il` (K = V, one head-width row per token),
ending at `pos - 1`; every layer has its own `attn_kv`. `comp[own]` holds the
completed blocks of KV owner `own` as `[comp | ik]`, one row per block of
`ratio` tokens; `pend_kv[own]` and `pend_sc[own]` hold the pooled inputs of the
partial block since the last boundary. `eng` is the last `ngram - 1` raw tokens
for the Engram hash, `-1` before the start.
"""
mutable struct Deepseek41State
    pos::Int
    raw::Dict{Int, Matrix{Float32}}
    comp::Dict{Int, Matrix{Float32}}
    pend_kv::Dict{Int, Matrix{Float32}}
    pend_sc::Dict{Int, Matrix{Float32}}
    eng::Vector{Int}
end

Deepseek41State(cfg::Deepseek41Config) = Deepseek41State(
    0, Dict{Int, Matrix{Float32}}(), Dict{Int, Matrix{Float32}}(),
    Dict{Int, Matrix{Float32}}(), Dict{Int, Matrix{Float32}}(),
    fill(-1, max(0, cfg.ple_ngram - 1)),
)

# ------------------------------------------------------- hyper-connections

# Stream s, feature i sits at column i + n_embd*s (i 1-based, s 0-based).
# Returns pre (T, hc), post (T, hc) and comb (T, src, dst) for one sublayer.
function _ds41_hc_mixes(m::Deepseek41Model, S::Matrix{Float32}, p::String)
    cfg = m.cfg
    HC, eps, eps_h = cfg.hc, cfg.eps, cfg.hc_eps
    T, F = size(S)
    fn = _ds41_w(m, p * "_fn.weight")
    sc = _ds41_w(m, p * "_scale.weight")
    base = _ds41_w(m, p * "_base.weight")
    proj = _ds41_mm(fn, S)
    mixdim = (2 + HC) * HC
    pre = Matrix{Float32}(undef, T, HC)
    post = Matrix{Float32}(undef, T, HC)
    comb = Array{Float32, 3}(undef, T, HC, HC)
    C = Matrix{Float64}(undef, HC, HC)
    for t in 1:T
        ss = 0.0
        for j in 1:F
            ss += Float64(S[t, j])^2
        end
        r = 1 / sqrt(ss / F + eps)
        mx = Float64[Float64(proj[t, k]) * r for k in 1:mixdim]
        for s in 0:(HC - 1)
            pre[t, s + 1] = Float32(_q4x_sigm(mx[s + 1] * sc[1] + base[s + 1]) + eps_h)
            post[t, s + 1] = Float32(2 * _q4x_sigm(mx[HC + s + 1] * sc[2] + base[HC + s + 1]))
        end
        # raw comb is row-major over (src j, dst k): element j*hc + k, k fastest
        for j in 0:(HC - 1), k in 0:(HC - 1)
            idx = 2HC + j * HC + k + 1
            C[j + 1, k + 1] = mx[idx] * sc[3] + base[idx]
        end
        # Sinkhorn as llama: softmax over dst per src, +eps, column step (sum over src),
        # then (iters - 1) row steps (sum over dst) each followed by a column step.
        for j in 1:HC
            mxr = maximum(view(C, j, :))
            z = 0.0
            for k in 1:HC
                C[j, k] = exp(C[j, k] - mxr)
                z += C[j, k]
            end
            for k in 1:HC
                C[j, k] = C[j, k] / z + eps_h
            end
        end
        for k in 1:HC
            s = sum(view(C, :, k)) + eps_h
            for j in 1:HC
                C[j, k] /= s
            end
        end
        for _ in 2:cfg.sinkhorn_iters
            for j in 1:HC
                s = sum(view(C, j, :)) + eps_h
                for k in 1:HC
                    C[j, k] /= s
                end
            end
            for k in 1:HC
                s = sum(view(C, :, k)) + eps_h
                for j in 1:HC
                    C[j, k] /= s
                end
            end
        end
        comb[t, :, :] .= Float32.(C)
    end
    return pre, post, comb
end

# A[t, i] = sum_s pre[t, s] * S[t, i + n_embd*s]
function _ds41_hc_pre(S::Matrix{Float32}, pre::Matrix{Float32}, NE::Int)
    T, HC = size(pre)
    A = zeros(Float32, T, NE)
    for t in 1:T, i in 1:NE
        acc = 0f0
        for s in 0:(HC - 1)
            acc += pre[t, s + 1] * S[t, i + NE * s]
        end
        A[t, i] = acc
    end
    return A
end

# out[t, i + n_embd*d] = post[t, d] * x[t, i] + sum_src comb[t, src, d] * S[t, i + n_embd*src]
function _ds41_hc_post(x::Matrix{Float32}, S::Matrix{Float32}, post::Matrix{Float32},
                       comb::Array{Float32, 3})
    T, NE = size(x)
    HC = size(post, 2)
    out = zeros(Float32, T, HC * NE)
    for t in 1:T, d in 0:(HC - 1), i in 1:NE
        acc = post[t, d + 1] * x[t, i]
        for j in 0:(HC - 1)
            acc += comb[t, j + 1, d + 1] * S[t, i + NE * j]
        end
        out[t, i + NE * d] = acc
    end
    return out
end

# ------------------------------------------------------------------- RoPE

# Angle of pair k (NORM pairs on the last n_rot dims) is position * mul[k].
# Compressed layers use the compress base with YaRN (freq_scale 1/factor,
# ext_factor 1); the YaRN magnitude correction cancels to 1 in llama.
function _ds41_rope_mul(cfg::Deepseek41Config, ratio::Int)
    half = cfg.n_rot ÷ 2
    base = ratio > 0 ? cfg.compress_rope_base : cfg.rope_base
    corr(r) = cfg.n_rot * log(cfg.rope_orig_ctx / (r * 2π)) / (2 * log(base))
    low = max(0.0, floor(corr(32.0)))
    high = min(cfg.n_rot - 1.0, ceil(corr(1.0)))
    mul = Vector{Float64}(undef, half)
    for k in 0:(half - 1)
        f = base^(-2k / cfg.n_rot)
        if ratio > 0
            ramp = clamp((k - low) / max(0.001, high - low), 0.0, 1.0)
            rm = 1 - ramp
            mul[k + 1] = f * ((1 / cfg.rope_factor) * (1 - rm) + rm)
        else
            mul[k + 1] = f
        end
    end
    return mul
end

# Rotate the tail of each of nh rows-of-heads (width hd) in place; row t sits at
# absolute position pos[t]. inverse=true undoes the rotation (ggml rope_back).
function _ds41_rope!(x::AbstractMatrix{Float32}, nh::Int, hd::Int, mul::Vector{Float64},
                     pos::AbstractVector{Int}; inverse::Bool=false)
    half = length(mul)
    off = hd - 2half
    for h in 0:(nh - 1), t in axes(x, 1), k in 0:(half - 1)
        θ = pos[t] * mul[k + 1]
        c = cos(θ)
        s = inverse ? -sin(θ) : sin(θ)
        a = h * hd + off + 2k + 1
        b = a + 1
        x1 = Float64(x[t, a])
        x2 = Float64(x[t, b])
        x[t, a] = Float32(x1 * c - x2 * s)
        x[t, b] = Float32(x1 * s + x2 * c)
    end
    return x
end

# ------------------------------------------------------------- compressor

# Owner layers (kv_src == il, ratio > 0): pool completed blocks of `ratio`
# tokens per channel with a softmax over the block's scores (ratio 1: score 0),
# normalize, and append [comp | ik] to comp[il]. ik = RMS(indexer.attn_k · comp)
# * k_norm, with the tail RoPE at each block's first position g*ratio.
function _ds41_compress!(m::Deepseek41Model, st::Deepseek41State, il::Int,
                         x::Matrix{Float32}, pos0::Int)
    cfg = m.cfg
    ratio = cfg.compress[il + 1]
    hd, idim, eps = cfg.head_dim, cfg.idx_head_dim, cfg.eps
    T = size(x, 1)
    kvc = _ds41_mm(m, x, "blk.$il.attn_compressor_kv.weight")
    scc = ratio > 1 ? _ds41_mm(m, x, "blk.$il.attn_compressor_gate.weight") :
        zeros(Float32, T, hd)
    pk = get(st.pend_kv, il, zeros(Float32, 0, hd))
    ps = get(st.pend_sc, il, zeros(Float32, 0, hd))
    cn = _ds41_w(m, "blk.$il.attn_compressor_norm.weight")
    wk = _ds41_w(m, "blk.$il.indexer.attn_k.weight")
    kn = _ds41_w(m, "blk.$il.indexer.k_norm.weight")
    mul = _ds41_rope_mul(cfg, ratio)
    rows = get(st.comp, il, zeros(Float32, 0, hd + idim))
    for t in 1:T
        p = pos0 + t - 1
        pk = vcat(pk, kvc[t:t, :])
        ps = vcat(ps, scc[t:t, :])
        (p + 1) % ratio == 0 || continue
        g = (p + 1) ÷ ratio - 1
        pooled = Matrix{Float32}(undef, 1, hd)
        for c in 1:hd
            col = view(ps, :, c)
            wts = exp.(col .- maximum(col))
            pooled[1, c] = sum(view(pk, :, c) .* wts) / sum(wts)
        end
        comp = _q4x_group_rms(pooled, hd, cn, eps)
        ik = _q4x_group_rms(_ds41_mm(wk, comp), idim, kn, eps)
        _ds41_rope!(comp, 1, hd, mul, [g * ratio])
        _ds41_rope!(ik, 1, idim, mul, [g * ratio])
        rows = vcat(rows, hcat(comp, ik))
        pk = zeros(Float32, 0, hd)
        ps = zeros(Float32, 0, hd)
    end
    st.comp[il] = rows
    st.pend_kv[il] = pk
    st.pend_sc[il] = ps
    return nothing
end

# ---------------------------------------------------------------- indexer

# Indexer owners (idx_src == il). Returns per query the selected block indices
# (0-based, ascending) and, on the candidate source layer only, the candidate
# blocks per query. cand_prev holds the candidate source's blocks for level two.
function _ds41_indexer(m::Deepseek41Model, st::Deepseek41State, il::Int, x::Matrix{Float32},
                       qr::Matrix{Float32}, pos0::Int, cand_prev)
    cfg = m.cfg
    ratio = cfg.compress[il + 1]
    hd, idim, ih = cfg.head_dim, cfg.idx_head_dim, cfg.idx_n_head
    T = size(x, 1)
    comp = get(st.comp, cfg.kv_src[il + 1], zeros(Float32, 0, hd + idim))
    iq = _ds41_mm(m, qr, "blk.$il.indexer.attn_q_b.weight")
    _ds41_rope!(iq, ih, idim, _ds41_rope_mul(cfg, ratio), collect(pos0:(pos0 + T - 1)))
    wgt = _ds41_mm(m, x, "blk.$il.indexer.proj.weight") .* Float32(idim^-0.5 * ih^-0.5)
    is_cand = cfg.cand_src == il
    use_cand = !is_cand && cfg.cand_src >= 0 && il > cfg.cand_src && cand_prev !== nothing
    bs = cfg.cand_block_size
    sel = Vector{Vector{Int}}(undef, T)
    cand = is_cand ? Vector{Vector{Int}}(undef, T) : nothing
    for t in 1:T
        p = pos0 + t - 1
        # Causal choice, per query: block g is complete at absolute position p
        # exactly when g < (p+1) ÷ ratio, so only those g are visible. The
        # reference's chunk-end count (end_pos ÷ ratio for every query) would let
        # earlier queries see the future inside a chunk; the per-query count keeps
        # a chunked run equal to one call.
        nv = min((p + 1) ÷ ratio, size(comp, 1))
        sc = fill(-Inf, nv)
        for g in 0:(nv - 1)
            acc = 0.0
            for h in 0:(ih - 1)
                dt = 0.0
                for j in 1:idim
                    dt += Float64(iq[t, h * idim + j]) * Float64(comp[g + 1, hd + j])
                end
                acc += max(dt, 0.0) * Float64(wgt[t, h + 1])
            end
            sc[g + 1] = acc
        end
        if is_cand
            if nv == 0
                cand[t] = Int[]
            else
                nb = cld(nv, bs)
                bsc = fill(-Inf, nb)
                for g in 0:(nv - 1)
                    bsc[g ÷ bs + 1] = max(bsc[g ÷ bs + 1], sc[g + 1])
                end
                # pin the query's newest visible block: finite, so it is always kept
                bsc[(nv - 1) ÷ bs + 1] += 1e30
                ord = sort(collect(1:nb); by = b -> (-bsc[b], b))
                cand[t] = ord[1:min(cfg.cand_top_k_blocks, nb)] .- 1
            end
        end
        if use_cand
            allowed = Set(cand_prev[t])
            for g in 0:(nv - 1)
                (g ÷ bs) in allowed || (sc[g + 1] = -Inf)
            end
        end
        k = min(cfg.idx_top_k, nv)
        picks = Int[]
        for g in sort(collect(0:(nv - 1)); by = g -> (-sc[g + 1], g))
            isfinite(sc[g + 1]) || continue
            push!(picks, g)
            length(picks) == k && break
        end
        sel[t] = sort!(picks)
    end
    return sel, cand
end

# ------------------------------------------------------------- attention

# Multi-query attention over the sliding window (self included) plus, on
# compressed layers, the rows topk[t] of the KV owner. The sink enters the
# denominator only. Output: derope at each query, then wo_a per group and wo_b.
function _ds41_attn(m::Deepseek41Model, st::Deepseek41State, il::Int, x::Matrix{Float32},
                    qr::Matrix{Float32}, pos0::Int, topk)
    cfg = m.cfg
    ratio = cfg.compress[il + 1]
    NH, hd, W, eps = cfg.n_head, cfg.head_dim, cfg.window, cfg.eps
    T = size(x, 1)
    mul = _ds41_rope_mul(cfg, ratio)
    pos = collect(pos0:(pos0 + T - 1))
    q = _ds41_mm(m, qr, "blk.$il.attn_q_b.weight")
    _ds41_rope!(q, NH, hd, mul, pos)
    kv = _q4x_group_rms(_ds41_mm(m, x, "blk.$il.attn_kv.weight"), hd,
        _ds41_w(m, "blk.$il.attn_kv_a_norm.weight"), eps)
    _ds41_rope!(kv, 1, hd, mul, pos)
    old = get(st.raw, il, zeros(Float32, 0, hd))
    base = pos0 - size(old, 1)
    R = vcat(old, kv)
    C = ratio > 0 ? st.comp[cfg.kv_src[il + 1]] : zeros(Float32, 0, hd + cfg.idx_head_dim)
    sinks = _ds41_w(m, "blk.$il.attn_sinks.weight")
    scale = 1 / sqrt(hd)
    o = zeros(Float32, T, NH * hd)
    for t in 1:T
        p = pos0 + t - 1
        lo = max(0, p - W + 1)
        Kw = R[(lo - base + 1):(p - base + 1), :]
        Kq = ratio > 0 ? vcat(Kw, C[topk[t] .+ 1, 1:hd]) : Kw
        nk = size(Kq, 1)
        for h in 0:(NH - 1)
            s = Vector{Float64}(undef, nk)
            for j in 1:nk
                acc = 0.0
                for d in 1:hd
                    acc += Float64(q[t, h * hd + d]) * Float64(Kq[j, d])
                end
                s[j] = acc * scale
            end
            sk = Float64(sinks[h + 1])
            mx = max(maximum(s), sk)
            z = 0.0
            num = zeros(Float64, hd)
            for j in 1:nk
                e = exp(s[j] - mx)
                z += e
                for d in 1:hd
                    num[d] += e * Kq[j, d]
                end
            end
            z += exp(sk - mx)
            for d in 1:hd
                o[t, h * hd + d] = Float32(num[d] / z)
            end
        end
    end
    _ds41_rope!(o, NH, hd, mul, pos; inverse=true)
    G, OL = cfg.o_groups, cfg.o_lora
    ne0 = NH * hd ÷ G
    Wa = _ds41_w(m, "blk.$il.attn_output_a.weight")
    yo = zeros(Float32, T, G * OL)
    buf = Vector{Float32}(undef, OL)
    for t in 1:T, g in 0:(G - 1)
        v = Vector{Float32}(o[t, (g * ne0 + 1):((g + 1) * ne0)])
        _ds41_bank_mv!(buf, v, Wa, g + 1)
        yo[t, (g * OL + 1):((g + 1) * OL)] .= buf
    end
    keep = min(size(R, 1), W)
    st.raw[il] = R[(size(R, 1) - keep + 1):end, :]
    return _ds41_mm(m, yo, "blk.$il.attn_output_b.weight")
end

# -------------------------------------------------------------------- MoE

# SwiGLU with the llama clamp: gate capped above, up clamped both sides; no clamp
# when the limit is not positive.
function _ds41_swiglu(g::Float32, u::Float32, lim::Float32)
    if lim > 1f-6
        g = min(g, lim)
        u = clamp(u, -lim, lim)
    end
    return _q4x_silu(g) * u
end

# Routed experts (sqrtsoftplus gate, exp_probs_b selection bias, unbiased
# weights, route_norm over more than one expert, route_scale) plus the shared
# expert, which uses clamp_shexp and has no gate.
function _ds41_ffn(m::Deepseek41Model, il::Int, y::Matrix{Float32})
    cfg = m.cfg
    T, NE = size(y)
    nE, nU, ff = cfg.n_exp, cfg.n_used, cfg.exp_ff
    logits = _ds41_mm(m, y, "blk.$il.ffn_gate_inp.weight")
    bias = _ds41_w(m, "blk.$il.exp_probs_b.bias")
    gw = _ds41_w(m, "blk.$il.ffn_gate_exps.weight")
    uw = _ds41_w(m, "blk.$il.ffn_up_exps.weight")
    dw = _ds41_w(m, "blk.$il.ffn_down_exps.weight")
    lim = Float32(cfg.clamp_exp[il + 1])
    moe = zeros(Float32, T, NE)
    bg = Vector{Float32}(undef, ff)
    bu = Vector{Float32}(undef, ff)
    bd = Vector{Float32}(undef, NE)
    for t in 1:T
        xt = Vector{Float32}(y[t, :])
        sc = Float64[sqrt(_q4x_softplus(Float64(logits[t, e]))) for e in 1:nE]
        picks = sort(collect(1:nE); by = e -> (-(sc[e] + Float64(bias[e])), e))[1:nU]
        w = Float64[sc[e] for e in picks]
        cfg.route_norm && nU > 1 && (w ./= (sum(w) + 1e-20))
        w .*= cfg.route_scale
        for (k, e) in enumerate(picks)
            _ds41_bank_mv!(bg, xt, gw, e)
            _ds41_bank_mv!(bu, xt, uw, e)
            h = _ds41_swiglu.(bg, bu, lim)
            _ds41_bank_mv!(bd, h, dw, e)
            moe[t, :] .+= Float32(w[k]) .* bd
        end
    end
    lims = Float32(cfg.clamp_shexp[il + 1])
    g = _ds41_mm(m, y, "blk.$il.ffn_gate_shexp.weight")
    u = _ds41_mm(m, y, "blk.$il.ffn_up_shexp.weight")
    h = _ds41_swiglu.(g, u, lims)
    return moe .+ _ds41_mm(m, h, "blk.$il.ffn_down_shexp.weight")
end

# ----------------------------------------------------------------- Engram

# Engram n-gram embedding on Engram layers (ple_layers). The hash runs over the
# raw token history in `st.eng` followed by this chunk, with unmapped or
# not-yet-seen tokens blocking the rest of the window (pad id). Each head's row
# is gathered from per_layer_token_embd, projected by engram_kv, and gated per
# stream by a signed square root of the key/query dot product.
function _ds41_engram(m::Deepseek41Model, st::Deepseek41State, il::Int,
                      S::Matrix{Float32}, toks::Vector{Int})
    cfg = m.cfg
    e = findfirst(==(il), cfg.ple_layers) - 1
    n, hpn, hdp = cfg.ple_ngram, cfg.ple_hpn, cfg.ple_head_dim
    NE, HC, eps = cfg.n_embd, cfg.hc, cfg.eps
    nc = (n - 1) * hpn
    T = length(toks)
    H = length(st.eng)
    seq = vcat(st.eng, toks)
    pad = cfg.token_map[cfg.engram_pad + 1]
    cid(t) = (t < 0 || t >= length(cfg.token_map)) ? -1 : cfg.token_map[t + 1]
    ids = Vector{Int}(undef, T * nc)
    tk = Vector{Int}(undef, n)
    for t in 1:T
        i = H + t
        c = cid(seq[i])
        blocked = c < 0
        tk[1] = blocked ? pad : c
        for s in 1:(n - 1)
            c = cid(seq[i - s])
            blocked = blocked || c < 0
            tk[s + 1] = blocked ? pad : c
        end
        rolling = UInt64(tk[1]) * cfg.ple_mult[e * n + 1]
        for o in 1:(n - 1)
            rolling ⊻= UInt64(tk[o + 1]) * cfg.ple_mult[e * n + o + 1]
            for h in 0:(hpn - 1)
                col = (o - 1) * hpn + h
                c2 = e * nc + col + 1
                ids[(t - 1) * nc + col + 1] = Int(rolling % UInt64(cfg.ple_voc[c2])) + cfg.ple_off[c2]
            end
        end
    end
    rows = _ds41_rows(_ds41_w(m, "per_layer_token_embd.weight"), ids)
    flat = zeros(Float32, T, nc * hdp)
    for t in 1:T, col in 0:(nc - 1), d in 1:hdp
        flat[t, col * hdp + d] = rows[(t - 1) * nc + col + 1, d]
    end
    kv = _ds41_mm(m, flat, "blk.$il.engram_kv.weight")
    G = _ds41_w(m, "blk.$il.engram_gate.weight")
    out = copy(S)
    for t in 1:T, s in 0:(HC - 1)
        sx = 0.0
        sk = 0.0
        for i in 1:NE
            sx += Float64(S[t, i + NE * s])^2
            sk += Float64(kv[t, s * NE + i])^2
        end
        rx = 1 / sqrt(sx / NE + eps)
        rk = 1 / sqrt(sk / NE + eps)
        dt = 0.0
        for i in 1:NE
            dt += Float64(S[t, i + NE * s]) * rx * Float64(kv[t, s * NE + i]) * rk * Float64(G[s + 1, i])
        end
        dt /= sqrt(NE)
        a = sqrt(max(abs(dt), 1e-6))
        gate = _q4x_sigm(dt < 0 ? -a : a)
        for i in 1:NE
            out[t, i + NE * s] += Float32(kv[t, HC * NE + i] * gate)
        end
    end
    return out
end

# ---------------------------------------------------------------- forward

"""
    deepseek41_forward!(m, st, tokens; n_layers=m.cfg.n_layer, taps=nothing) -> (S, pre)

Run `tokens` (0-based ids) at positions `st.pos`, `st.pos + 1`, … and advance
`st`. `S` is the residual (T, hc * n_embd) after `n_layers` blocks and `pre` the
last FFN pre-mix (T, hc), which collapses `S` for the output head. A truncated
run leaves the state of the skipped blocks behind, so use it for comparisons,
not decoding. `taps` (a Dict) collects intermediates under node names
(`attn_out-<il>`, `ffn_out-<il>`, `l_out-<il>`).

Block flow as llama.cpp: Engram first on its layers; attention input is the
previous FFN pre-mix applied to the residual (layer 0 copies stream 0); the FFN
input is the attention pre-mix of the same layer applied to the post-attention
residual.
"""
function deepseek41_forward!(m::Deepseek41Model, st::Deepseek41State, tokens::AbstractVector{<:Integer};
                             n_layers::Int=m.cfg.n_layer, taps=nothing)
    cfg = m.cfg
    toks = Vector{Int}(tokens)
    T = length(toks)
    T > 0 || error("deepseek41_forward!: no tokens")
    0 < n_layers <= cfg.n_layer || error("deepseek41_forward!: n_layers $n_layers out of 1..$(cfg.n_layer)")
    NE, HC = cfg.n_embd, cfg.hc
    pos0 = st.pos
    S = repeat(_ds41_rows(_ds41_w(m, "token_embd.weight"), toks), 1, HC)
    topk_of = Dict{Int, Vector{Vector{Int}}}()
    cand_prev = nothing
    prev = nothing
    for il in 0:(n_layers - 1)
        il in cfg.ple_layers && (S = _ds41_engram(m, st, il, S, toks))
        apre, apost, acomb = _ds41_hc_mixes(m, S, "blk.$il.hc_attn")
        A = prev === nothing ? Matrix{Float32}(S[:, 1:NE]) : _ds41_hc_pre(S, prev, NE)
        x = _q4x_group_rms(A, NE, _ds41_w(m, "blk.$il.attn_norm.weight"), cfg.eps)
        qr = _q4x_group_rms(_ds41_mm(m, x, "blk.$il.attn_q_a.weight"), cfg.q_lora,
            _ds41_w(m, "blk.$il.attn_q_a_norm.weight"), cfg.eps)
        ratio = cfg.compress[il + 1]
        topk = nothing
        if ratio > 0
            cfg.kv_src[il + 1] == il && _ds41_compress!(m, st, il, x, pos0)
            if cfg.idx_src[il + 1] == il
                sel, cand = _ds41_indexer(m, st, il, x, qr, pos0, cand_prev)
                topk_of[il] = sel
                cfg.cand_src == il && (cand_prev = cand)
            end
            topk = topk_of[cfg.idx_src[il + 1]]
        end
        att = _ds41_attn(m, st, il, x, qr, pos0, topk)
        _q4x_tap!(taps, "attn_out-$il", att)
        S_mid = _ds41_hc_post(att, S, apost, acomb)
        fpre, fpost, fcomb = _ds41_hc_mixes(m, S_mid, "blk.$il.hc_ffn")
        F = _ds41_hc_pre(S_mid, apre, NE)
        y = _q4x_group_rms(F, NE, _ds41_w(m, "blk.$il.ffn_norm.weight"), cfg.eps)
        ffn = _ds41_ffn(m, il, y)
        _q4x_tap!(taps, "ffn_out-$il", ffn)
        S = _ds41_hc_post(ffn, S_mid, fpost, fcomb)
        _q4x_tap!(taps, "l_out-$il", S)
        prev = fpre
    end
    if !isempty(cfg.ple_layers)
        hist = vcat(st.eng, toks)
        st.eng = hist[(end - (cfg.ple_ngram - 1) + 1):end]
    end
    st.pos += T
    return (S = S, pre = prev)
end

"""
    deepseek41_logits(m, res; taps=nothing) -> Vector{Float32}

Logits for the last row of `res`, the `(S, pre)` pair from `deepseek41_forward!`.
The output head has no hyper-connection mixer: the last FFN pre-mix collapses
the streams, then output_norm and output.weight.
"""
function deepseek41_logits(m::Deepseek41Model, res; taps=nothing)
    cfg = m.cfg
    r = size(res.S, 1):size(res.S, 1)
    h = _ds41_hc_pre(res.S[r, :], res.pre[r, :], cfg.n_embd)
    x = _q4x_group_rms(h, cfg.n_embd, _ds41_w(m, "output_norm.weight"), cfg.eps)
    _q4x_tap!(taps, "result_norm", x)
    logits = _ds41_mm(m, x, "output.weight")
    _q4x_tap!(taps, "result_output", logits)
    return vec(logits)
end
