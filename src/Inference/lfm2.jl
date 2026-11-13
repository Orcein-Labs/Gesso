# LFM feature table. lfm2, lfm2moe (LFM2.5 MoE) and lfm3 share one importer and one
# forward. A variant is the set of features its metadata and tensors declare; the
# spec, param map and forward read those features, never the family name.
#
#   feature / field          detected from                                        forward effect
#   :qk_norm                 attn_q_norm, attn_k_norm (always)                    per-head RMS norm on Q and K
#   spec.moe                 <arch>.expert_count                                  routed experts after the dense lead blocks
#   spec.sliding_window      <arch>.attention.sliding_window > 0, with the        SWA on the pattern's true layers
#                            sliding_window_pattern array (both or neither)
#   spec.rope_swa            <arch>.rope.freq_base_swa                            SWA layers use this theta, no factors
#   :rope_freq_factors       rope_freqs.weight                                    full layers divide inv_freq by the factors
#   :attn_head_gate          attn_gate.weight on every attention layer            per-head sigmoid gate on the mixer output
#   :sandwich_norm           post_attention_norm and post_ffw_norm, both, on      RMS norm on each sublayer output before its
#                            every layer                                          residual add
#   :output_norm_final       output_norm.weight (else token_embd_norm.weight)     final norm name for the head
#
# Tensor-level features need tensor_names. With tensor_names === nothing (metadata
# only) none of them is declared, so such a spec must not be run.
#
# A metadata key outside _LFM_KEYS is an error that names the key, so no new key
# is ever ignored silently.
const _LFM_KEYS = Set([
    "block_count", "context_length", "embedding_length", "feed_forward_length",
    "vocab_size", "decision_block_count", "attention.head_count", "attention.head_count_kv",
    "attention.key_length", "attention.value_length", "attention.layer_norm_rms_epsilon",
    "attention.causal", "shortconv.l_cache", "rope.freq_base", "rope.dimension_count",
    "expert_count", "expert_feed_forward_length", "expert_used_count", "expert_gating_func",
    "expert_weights_scale", "expert_group_count", "leading_dense_block_count",
    "attention.sliding_window", "attention.sliding_window_pattern", "rope.freq_base_swa",
])

function _spec_from_gguf_lfm(kv::AbstractDict, tensor_names)
    arch = String(kv["general.architecture"])
    pre = arch * "."
    req(key) = get(kv, pre * key) do
        error("spec_from_gguf_kv: missing $pre$key")
    end
    function scalar(key)
        v = req(key)
        v isa AbstractVector && error("LFM: per-layer $pre$key is unsupported")
        return Int(v)
    end
    for k in keys(kv)
        startswith(k, pre) || continue
        k[ncodeunits(pre)+1:end] in _LFM_KEYS ||
            error("LFM: unsupported metadata key $k (no LFM feature declares it)")
    end
    n = Int(req("block_count"))
    h = Int(req("embedding_length"))
    nh = Int(req("attention.head_count"))
    nh > 0 || error("LFM: attention.head_count must be positive")
    raw_kv = req("attention.head_count_kv")
    heads = raw_kv isa AbstractVector ? Int.(raw_kv) : fill(Int(raw_kv), n)
    length(heads) == n || error("LFM: head_count_kv length differs from block_count")
    attn_heads = unique(filter(>(0), heads))
    length(attn_heads) == 1 || error("LFM: expected uniform positive attention KV head counts")
    all(>=(0), heads) || error("LFM: negative KV head count")
    n_attn = count(>(0), heads)
    Int(get(kv, pre * "decision_block_count", 0)) == 0 || error("LFM: decision heads are not causal language models")
    tensor_names !== nothing && "dense_2_out.weight" in tensor_names &&
        error("LFM: dense_2_out embedding heads are not causal language models")
    get(kv, pre * "attention.causal", true) || error("LFM: noncausal short convolution is unsupported")
    hd = Int(get(kv, pre * "attention.key_length", h ÷ nh))
    Int(get(kv, pre * "attention.value_length", hd)) == hd || error("LFM: unequal key/value dimensions")
    nh * hd == h || error("LFM: attention width must equal hidden width")
    if haskey(kv, pre * "rope.dimension_count")
        Int(kv[pre * "rope.dimension_count"]) == hd || error(
            "LFM: rope.dimension_count must equal the head dimension (partial rotary is unsupported)",
        )
    end

    # sliding window: the width and the per-layer pattern are declared together
    window = Int(get(kv, pre * "attention.sliding_window", 0))
    pattern = get(kv, pre * "attention.sliding_window_pattern", nothing)
    (window > 0) == (pattern !== nothing) || error(
        "LFM: attention.sliding_window and sliding_window_pattern must be declared together",
    )
    swa_layers = Bool[]
    if pattern !== nothing
        swa_layers = Bool.(collect(pattern))
        length(swa_layers) == n || error(
            "LFM: sliding_window_pattern length $(length(swa_layers)) differs from block_count $n",
        )
    end
    theta_swa = get(kv, pre * "rope.freq_base_swa", nothing)
    theta_swa === nothing || window > 0 || error("LFM: rope.freq_base_swa without a sliding window")

    features = Set{Symbol}([:qk_norm])
    if tensor_names !== nothing
        tcount(suffix) = count(t -> startswith(t, "blk.") && endswith(t, "." * suffix), tensor_names)
        gate = tcount("attn_gate.weight")
        gate in (0, n_attn) || error(
            "LFM: attn_gate.weight on $gate layers, expected 0 or $n_attn attention layers",
        )
        gate == 0 || push!(features, :attn_head_gate)
        pa = tcount("post_attention_norm.weight")
        pf = tcount("post_ffw_norm.weight")
        (pa == 0) == (pf == 0) || error(
            "LFM: post_attention_norm and post_ffw_norm must both be present (sandwich norm needs both)",
        )
        pa == pf && pa in (0, n) || error(
            "LFM: post_attention_norm on $pa and post_ffw_norm on $pf layers, expected both 0 or $n",
        )
        pa == 0 || push!(features, :sandwich_norm)
        "rope_freqs.weight" in tensor_names && push!(features, :rope_freq_factors)
        if "output_norm.weight" in tensor_names
            push!(features, :output_norm_final)
        else
            "token_embd_norm.weight" in tensor_names ||
                error("LFM: no final norm tensor (output_norm.weight or token_embd_norm.weight)")
        end
    end

    mixer = ShortConvSpec([k == 0 ? :short_convolution : :full_attention for k in heads], Int(req("shortconv.l_cache")))
    moe = if haskey(kv, pre * "expert_count")
        gating = Int(req("expert_gating_func"))
        gating in (1, 2) || error("LFM: unsupported expert_gating_func $gating")
        get(kv, pre * "expert_weights_scale", 0) in (0, 1) || error("LFM: scaled expert weights are unsupported")
        Int(get(kv, pre * "expert_group_count", 0)) <= 1 || error("LFM: grouped expert routing is unsupported")
        MoeSpec(; n_experts=Int(req("expert_count")), n_expert_used=scalar("expert_used_count"),
            expert_ff=scalar("expert_feed_forward_length"),
            dense_lead=Int(get(kv, pre * "leading_dense_block_count", 0)),
            gating=gating == 2 ? :sigmoid : :softmax, selection_bias=true)
    else
        nothing
    end
    tokens = get(kv, "tokenizer.ggml.tokens", nothing)
    token_count = if tokens isa Integer
        Int(tokens)
    elseif tokens isa AbstractVector
        length(tokens)
    elseif tokens === nothing
        nothing
    else
        error("LFM: tokenizer.ggml.tokens must be an array or token count")
    end
    declared_vocab = get(kv, pre * "vocab_size", nothing)
    vocab = declared_vocab === nothing ? token_count : Int(declared_vocab)
    vocab === nothing && error("LFM: missing vocabulary size and tokenizer.ggml.tokens")
    token_count === nothing || token_count == vocab || error(
        "LFM: $pre" * "vocab_size=$vocab differs from tokenizer.ggml.tokens count $token_count",
    )
    return ArchitectureSpec(; family=Symbol(arch), hidden_size=h, num_layers=n, n_heads=nh,
        n_kv_heads=only(attn_heads), head_dim=hd, vocab_size=Int(vocab),
        intermediate_size=scalar("feed_forward_length"),
        rope=RoPEPolicy(; theta=Float64(req("rope.freq_base"))),
        rope_swa=theta_swa === nothing ? nothing : RoPEPolicy(; theta=Float64(theta_swa)),
        sliding_window=window > 0 ? window : nothing, swa_layers,
        tie_word_embeddings=tensor_names === nothing || "output.weight" ∉ tensor_names,
        features, mixer, moe,
        rms_norm_eps=Float64(req("attention.layer_norm_rms_epsilon")))
end

function _gguf_lfm_param_map(spec::ArchitectureSpec)
    ref(role, name, kind=:projection) = ParamRef(SemanticParamId(role, 0, kind), name)
    norms = [ref(ROLE_NORM_PRE_ATTN, "attn_norm.weight", :frozen), ref(ROLE_NORM_POST_ATTN, "ffn_norm.weight", :frozen)]
    if :sandwich_norm in spec.features
        append!(norms, [ref(ROLE_ATTN_POST_NORM, "post_attention_norm.weight", :frozen),
            ref(ROLE_FFN_POST_NORM, "post_ffw_norm.weight", :frozen)])
    end
    ffn = [ref(ROLE_GATE, "ffn_gate.weight"), ref(ROLE_UP, "ffn_up.weight"), ref(ROLE_DOWN, "ffn_down.weight")]
    if spec.moe !== nothing
        append!(ffn, [ref(ROLE_ROUTER, "ffn_gate_inp.weight"),
            ref(ROLE_EXPERT_GATE, "ffn_gate_exps.weight", :expert), ref(ROLE_EXPERT_UP, "ffn_up_exps.weight", :expert),
            ref(ROLE_EXPERT_DOWN, "ffn_down_exps.weight", :expert)])
        spec.moe.selection_bias && push!(ffn, ref(ROLE_ROUTER_BIAS, "exp_probs_b.bias", :frozen))
    end
    attn = [ref(ROLE_Q, "attn_q.weight"), ref(ROLE_K, "attn_k.weight"), ref(ROLE_V, "attn_v.weight"),
        ref(ROLE_O, "attn_output.weight"), ref(ROLE_Q_NORM, "attn_q_norm.weight", :frozen), ref(ROLE_K_NORM, "attn_k_norm.weight", :frozen)]
    :attn_head_gate in spec.features && push!(attn, ref(ROLE_ATTN_HEAD_GATE, "attn_gate.weight"))
    conv = [ref(ROLE_SHORT_IN, "shortconv.in_proj.weight"), ref(ROLE_SHORT_CONV, "shortconv.conv.weight", :frozen),
        ref(ROLE_SHORT_OUT, "shortconv.out_proj.weight")]
    final_norm = :output_norm_final in spec.features ? "output_norm.weight" : "token_embd_norm.weight"
    rope_freqs = :rope_freq_factors in spec.features ? "rope_freqs.weight" : nothing
    return FamilyParamMap(spec.family, "blk.", vcat(norms, attn, ffn), "token_embd.weight",
        final_norm, spec.tie_word_embeddings ? nothing : "output.weight";
        linear_refs=vcat(norms, conv, ffn), rope_freqs_external=rope_freqs)
end

# The sandwich post-norm: an RMS norm on a sublayer output before its residual
# add, with the slot named by `slot`. Only blocks that carry the :sandwich_norm
# flag (set by materialize for specs declaring that feature) run it.
function _sandwich_post!(cpu, x, bt, slot::Symbol, wl, eps, zeros_like)
    haskey(bt, :sandwich_norm) || return x
    scale = getproperty(bt, slot)
    tmp = Activation(; shape=size(x.storage), storage=zeros_like(size(x.storage)))
    rmsnorm!(cpu, tmp, x, scale, wl; eps)
    copyto!(x.storage, tmp.storage)
    return x
end

function _short_convolution_block!(cpu, h, normed, sub, bt, wl, eps, zeros_like; state=nothing, xq=nothing)
    seq, dim = size(h.storage)
    width = size(bt.wshort_conv.storage, 2)
    hist = state === nothing ? zeros_like((dim, width - 1)) : state
    size(hist) == (dim, width - 1) || error("short convolution: invalid history shape")
    rmsnorm!(cpu, normed, h, bt.attn_rms, wl; eps)
    bcx = Activation(; shape=(seq, 3dim), storage=zeros_like((seq, 3dim)))
    @decode_sub :conv_in cpu _conv_proj!(cpu, bcx, normed, bt.wshort_in, wl, xq)
    mixed = Activation(; shape=(seq, dim), storage=zeros_like((seq, dim)))
    @decode_sub :conv_mix cpu _short_conv_mix!(cpu, mixed.storage, bcx.storage, bt.wshort_conv.storage, hist)
    @decode_sub :conv_out cpu _conv_proj!(cpu, sub, mixed, bt.wshort_out, wl, xq)
    return sub
end

# One projection of the conv block. A decode row (seq = 1) with the decode
# scratch's Q8_K column `xq` (`_decode_xq`) takes `_dec_matmul_group!`, the
# route the attention and FFN projections use: K-quant weights then run the
# prepped multi-row kernel, which is bit-equal to the per-row `_dot_q8k` that
# `matmul!` runs on the same Q8_K quantization. Prefill, the reference path and
# weights without a Q8_K kernel keep `matmul!`.
function _conv_proj!(cpu, dst, x, w, wl, xq)
    if xq !== nothing && size(x.storage, 1) == 1 && w.storage isa QuantMatrix &&
       _quant_q8k_kind(w.storage) && w.storage.ne0 == length(xq) &&
       dst.storage isa Matrix{Float32}
        _dec_matmul_group!(cpu, (dst,), xq, x, (w,), wl)
        return dst
    end
    return matmul!(cpu, dst, x, w, wl)
end

# The gated short convolution between the two projections. bcx holds B, C and X
# side by side; hist holds the last width-1 B*X values per channel and is
# advanced in place. Channels are independent, so each channel walks its own
# tokens in order (bcx is column-major, so that reads it contiguously) and a
# prompt splits the channels across threads. Every output keeps the same
# operations in the same order as the token-major loop, so results are
# bit-identical to it.
function _short_conv_mix!(backend, mixed, bcx, w, hist)
    backend_name(backend) === :cpu || throw(LoweringNotImplemented(:short_convolution, backend_name(backend)))
    seq, dim = size(mixed)
    if seq > 1 && Threads.nthreads() > 1
        nc = min(Threads.nthreads(), cld(dim, 64))
        _run_row_chunks!(_ShortConvChunk(mixed, bcx, w, hist, cld(dim, nc), dim), nc, seq * dim)
    else
        _short_conv_cols!(mixed, bcx, w, hist, 1, dim)
    end
    return mixed
end

struct _ShortConvChunk{M,B,W,H}
    mixed::M
    bcx::B
    w::W
    hist::H
    per::Int
    dim::Int
end

(g::_ShortConvChunk)(k::Int) = _short_conv_cols!(g.mixed, g.bcx, g.w, g.hist,
    (k - 1) * g.per + 1, min(k * g.per, g.dim))

function _short_conv_cols!(mixed, bcx, w, hist, c0::Int, c1::Int)
    seq, dim = size(mixed)
    width = size(w, 2)
    # GGUF taps run oldest to newest; the current B*X is the final tap.
    @inbounds for c in c0:c1, t in 1:seq
        bx = bcx[t, c] * bcx[t, 2dim+c]
        y = zero(eltype(mixed))
        for p in 1:(width-1)
            y += hist[c, p] * w[c, p]
        end
        y += bx * w[c, width]
        mixed[t, c] = bcx[t, dim+c] * y
        for p in 1:(width-2)
            hist[c, p] = hist[c, p+1]
        end
        hist[c, width-1] = bx
    end
    return nothing
end

function _selected_expert_mul!(dst, bank, e, x)
    if bank isa GgufQuantSlice
        _qgemv_expert!(dst, x, bank, e)
    else
        LinearAlgebra.mul!(dst, view(bank, e, :, :), x)
    end
    return dst
end

function _biased_moe_ffn!(cpu, down, normed, bt)
    backend_name(cpu) === :cpu || throw(LoweringNotImplemented(:sigmoid_moe_routing, backend_name(cpu)))
    x, y = normed.storage, down.storage
    router = bt.wrouter.storage
    ne, dim = size(router)
    ff = size(bt.wexp_gate.storage, 2)
    T = eltype(y)
    logits = Vector{T}(undef, ne)
    probs = similar(logits)
    order = collect(1:ne)
    bias = haskey(bt, :wrouter_bias) ? bt.wrouter_bias.storage : zeros(T, ne)
    gate_s, up_s, down_s = bt.wexp_gate.storage, bt.wexp_up.storage, bt.wexp_down.storage
    if T === Float32 && gate_s isa GgufQuantSlice && up_s isa GgufQuantSlice && down_s isa GgufQuantSlice &&
       _moe_host_batchable(gate_s, up_s, down_s, dim, ff)
        return _biased_moe_batched!(down, x, router, bias, bt, logits, probs, order, gate_s, up_s, down_s)
    end
    xr, g, u, a, out = Vector{T}(undef, dim), Vector{T}(undef, ff), Vector{T}(undef, ff), Vector{T}(undef, ff), Vector{T}(undef, dim)
    fill!(y, zero(T))
    for t in axes(x, 1)
        m = _cpu_router_logits!(logits, x, t, router)
        if bt.router_gating === :sigmoid
            probs .= _cpu_sigmoid.(logits)
        else
            probs .= exp.(logits .- m)
            probs ./= sum(probs)
        end
        sort!(order; by=e -> (-Float64(probs[e] + bias[e]), e))
        selected = view(order, 1:bt.n_expert_used)
        denom = max(sum(probs[e] for e in selected), T(6.103515625e-5))
        xr .= view(x, t, :)
        for e in selected
            _selected_expert_mul!(g, bt.wexp_gate.storage, e, xr)
            _selected_expert_mul!(u, bt.wexp_up.storage, e, xr)
            @. a = _cpu_silu(g) * u
            _selected_expert_mul!(out, bt.wexp_down.storage, e, a)
            weight = T(probs[e] / denom)
            @inbounds for j in 1:dim
                y[t, j] += weight * out[j]
            end
        end
    end
    return down
end

# Router logits for every token of a prompt, (n_exp, seq). Each logit sums
# over i in order exactly as _cpu_router_logits! does, so the values are
# bit-identical; walking t inside i reads the column-major x contiguously
# instead of one strided row per token, and threads split the tokens.
function _router_logits_all(x, router, ::Type{T}) where {T}
    seq = size(x, 1)
    L = zeros(T, size(router, 1), seq)
    nc = min(Threads.nthreads(), cld(seq, 16))
    _run_row_chunks!(_RouterLogitsChunk(L, x, router, cld(seq, nc), seq), nc, seq * size(router, 1) * size(router, 2))
    return L
end

struct _RouterLogitsChunk{LT,X,R}
    L::LT
    x::X
    router::R
    per::Int
    seq::Int
end

function (g::_RouterLogitsChunk)(k::Int)
    t0, t1 = (k - 1) * g.per + 1, min(k * g.per, g.seq)
    L, x, router = g.L, g.x, g.router
    T = eltype(L)
    n_exp, h = size(router)
    @inbounds for i in 1:h, t in t0:t1
        xi = T(x[t, i])
        @simd for e in 1:n_exp
            L[e, t] += xi * router[e, i]
        end
    end
    return nothing
end

# y[1, :] .= v behind a call boundary: `down.storage` is not concretely typed
# where it is read, so the store loop inline would box every element.
function _store_row1!(y::AbstractMatrix, v::AbstractVector)
    @inbounds for i in eachindex(v)
        y[1, i] = v[i]
    end
    return y
end

# K-quant banks route every token first, then run the shared host MoE: the
# batched Q8_K kernels for one token, the grouped multi-row tiles for a prompt.
function _biased_moe_batched!(down, x, router, bias, bt, logits, probs, order, gate_s, up_s, down_s)
    y = down.storage
    seq, dim = size(x)
    n_used = bt.n_expert_used
    experts = Matrix{Int}(undef, n_used, seq)
    wts = Matrix{Float32}(undef, n_used, seq)
    L = seq > 1 ? _router_logits_all(x, router, eltype(logits)) : nothing
    @decode_sub :moe_route nothing for t in 1:seq
        if L === nothing
            m = _cpu_router_logits!(logits, x, t, router)
        else
            @inbounds for e in eachindex(logits)
                logits[e] = L[e, t]
            end
            m = maximum(logits)
        end
        if bt.router_gating === :sigmoid
            probs .= _cpu_sigmoid.(logits)
        else
            probs .= exp.(logits .- m)
            probs ./= sum(probs)
        end
        sort!(order; by=e -> (-Float64(probs[e] + bias[e]), e))
        denom = max(sum(probs[order[k]] for k in 1:n_used), 6.103515625f-5)
        @inbounds for k in 1:n_used
            e = order[k]
            experts[k, t] = e
            wts[k, t] = probs[e] / denom
        end
    end
    @decode_sub :moe_experts nothing begin
        if seq == 1
            xf = Vector{Float32}(undef, dim)
            yrow = Vector{Float32}(undef, dim)
            @inbounds for i in 1:dim
                xf[i] = x[1, i]
            end
            _moe_host_experts_auto!(yrow, xf, view(experts, :, 1), view(wts, :, 1), n_used, gate_s, up_s, down_s)
            _store_row1!(y, yrow)
        else
            X = x isa Matrix{Float32} ? x : Matrix{Float32}(x)
            Y = y isa Matrix{Float32} ? y : Matrix{Float32}(undef, seq, dim)
            _moe_host_prefill!(Y, X, experts, wts, fill(true, seq), length(logits), gate_s, up_s, down_s)
            Y === y || copyto!(y, Y)
        end
    end
    return down
end
