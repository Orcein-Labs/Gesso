function _spec_from_gguf_lfm(kv::AbstractDict, tensor_names)
    arch = String(kv["general.architecture"])
    req(key) = get(kv, "$arch.$key") do
        error("spec_from_gguf_kv: missing $arch.$key")
    end
    function scalar(key)
        v = req(key)
        v isa AbstractVector && error("LFM: per-layer $arch.$key is unsupported")
        return Int(v)
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
    Int(get(kv, "$arch.decision_block_count", 0)) == 0 || error("LFM: decision heads are not causal language models")
    tensor_names !== nothing && "dense_2_out.weight" in tensor_names &&
        error("LFM: dense_2_out embedding heads are not causal language models")
    get(kv, "$arch.attention.causal", true) || error("LFM: noncausal short convolution is unsupported")
    window = Int(get(kv, "$arch.attention.sliding_window", 0))
    window == 0 || error("LFM: attention.sliding_window is unsupported by the CPU reference path")
    hd = Int(get(kv, "$arch.attention.key_length", h ÷ nh))
    Int(get(kv, "$arch.attention.value_length", hd)) == hd || error("LFM: unequal key/value dimensions")
    nh * hd == h || error("LFM: attention width must equal hidden width")
    unsupported_rope_keys = (
        "$arch.rope.dimension_count", "$arch.rope.dimension_count_swa",
        "$arch.rope.scaling.type", "$arch.rope.scaling.factor", "$arch.rope.scale_linear",
        "$arch.rope.scaling.original_context_length", "$arch.rope.scaling.finetuned",
        "$arch.rope.scaling.attn_factor", "$arch.rope.scaling.alpha",
        "$arch.rope.scaling.low_freq_factor", "$arch.rope.scaling.high_freq_factor",
    )
    present_rope_keys = filter(k -> haskey(kv, k), unsupported_rope_keys)
    isempty(present_rope_keys) || error(
        "LFM: unsupported RoPE metadata: $(join(present_rope_keys, ", "))",
    )
    mixer = ShortConvSpec([k == 0 ? :short_convolution : :full_attention for k in heads], Int(req("shortconv.l_cache")))
    moe = if arch == "lfm2moe"
        gating = Int(req("expert_gating_func"))
        gating in (1, 2) || error("LFM: unsupported expert_gating_func $gating")
        get(kv, "$arch.expert_weights_scale", 0) in (0, 1) || error("LFM: scaled expert weights are unsupported")
        Int(get(kv, "$arch.expert_group_count", 0)) <= 1 || error("LFM: grouped expert routing is unsupported")
        MoeSpec(; n_experts=Int(req("expert_count")), n_expert_used=scalar("expert_used_count"),
            expert_ff=scalar("expert_feed_forward_length"),
            dense_lead=Int(get(kv, "$arch.leading_dense_block_count", 0)),
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
    declared_vocab = get(kv, "$arch.vocab_size", nothing)
    vocab = declared_vocab === nothing ? token_count : Int(declared_vocab)
    vocab === nothing && error("LFM: missing vocabulary size and tokenizer.ggml.tokens")
    token_count === nothing || token_count == vocab || error(
        "LFM: $arch.vocab_size=$vocab differs from tokenizer.ggml.tokens count $token_count",
    )
    return ArchitectureSpec(; family=Symbol(arch), hidden_size=h, num_layers=n, n_heads=nh,
        n_kv_heads=only(attn_heads), head_dim=hd, vocab_size=Int(vocab),
        intermediate_size=scalar("feed_forward_length"),
        rope=RoPEPolicy(; theta=Float64(req("rope.freq_base"))),
        tie_word_embeddings=tensor_names === nothing || "output.weight" ∉ tensor_names,
        features=Set([:qk_norm]), mixer, moe,
        rms_norm_eps=Float64(req("attention.layer_norm_rms_epsilon")))
end

function _gguf_lfm_param_map(spec::ArchitectureSpec)
    ref(role, name, kind=:projection) = ParamRef(SemanticParamId(role, 0, kind), name)
    norms = [ref(ROLE_NORM_PRE_ATTN, "attn_norm.weight", :frozen), ref(ROLE_NORM_POST_ATTN, "ffn_norm.weight", :frozen)]
    ffn = [ref(ROLE_GATE, "ffn_gate.weight"), ref(ROLE_UP, "ffn_up.weight"), ref(ROLE_DOWN, "ffn_down.weight")]
    if spec.moe !== nothing
        append!(ffn, [ref(ROLE_ROUTER, "ffn_gate_inp.weight"),
            ref(ROLE_EXPERT_GATE, "ffn_gate_exps.weight", :expert), ref(ROLE_EXPERT_UP, "ffn_up_exps.weight", :expert),
            ref(ROLE_EXPERT_DOWN, "ffn_down_exps.weight", :expert)])
        spec.moe.selection_bias && push!(ffn, ref(ROLE_ROUTER_BIAS, "exp_probs_b.bias", :frozen))
    end
    attn = [ref(ROLE_Q, "attn_q.weight"), ref(ROLE_K, "attn_k.weight"), ref(ROLE_V, "attn_v.weight"),
        ref(ROLE_O, "attn_output.weight"), ref(ROLE_Q_NORM, "attn_q_norm.weight", :frozen), ref(ROLE_K_NORM, "attn_k_norm.weight", :frozen)]
    conv = [ref(ROLE_SHORT_IN, "shortconv.in_proj.weight"), ref(ROLE_SHORT_CONV, "shortconv.conv.weight", :frozen),
        ref(ROLE_SHORT_OUT, "shortconv.out_proj.weight")]
    return FamilyParamMap(spec.family, "blk.", vcat(norms, attn, ffn), "token_embd.weight",
        "token_embd_norm.weight", spec.tie_word_embeddings ? nothing : "output.weight";
        linear_refs=vcat(norms, conv, ffn))
end

function _short_convolution_block!(cpu, h, normed, sub, bt, wl, eps, zeros_like; state=nothing)
    seq, dim = size(h.storage)
    width = size(bt.wshort_conv.storage, 2)
    hist = state === nothing ? zeros_like((dim, width - 1)) : state
    size(hist) == (dim, width - 1) || error("short convolution: invalid history shape")
    rmsnorm!(cpu, normed, h, bt.attn_rms, wl; eps)
    bcx = Activation(; shape=(seq, 3dim), storage=zeros_like((seq, 3dim)))
    matmul!(cpu, bcx, normed, bt.wshort_in, wl)
    mixed = Activation(; shape=(seq, dim), storage=zeros_like((seq, dim)))
    _short_conv_mix!(cpu, mixed.storage, bcx.storage, bt.wshort_conv.storage, hist)
    matmul!(cpu, sub, mixed, bt.wshort_out, wl)
    return sub
end

# The gated short convolution between the two projections. bcx holds B, C and X
# side by side; hist holds the last width-1 B*X values per channel and is
# advanced in place.
function _short_conv_mix!(backend, mixed, bcx, w, hist)
    backend_name(backend) === :cpu || throw(LoweringNotImplemented(:short_convolution, backend_name(backend)))
    seq, dim = size(mixed)
    width = size(w, 2)
    # GGUF taps run oldest to newest; the current B*X is the final tap.
    @inbounds for t in 1:seq, c in 1:dim
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
    return mixed
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
    xr, g, u, a, out = Vector{T}(undef, dim), Vector{T}(undef, ff), Vector{T}(undef, ff), Vector{T}(undef, ff), Vector{T}(undef, dim)
    bias = haskey(bt, :wrouter_bias) ? bt.wrouter_bias.storage : zeros(T, ne)
    gate_s, up_s, down_s = bt.wexp_gate.storage, bt.wexp_up.storage, bt.wexp_down.storage
    if T === Float32 && gate_s isa GgufQuantSlice && up_s isa GgufQuantSlice && down_s isa GgufQuantSlice &&
       _moe_host_batchable(gate_s, up_s, down_s, dim, ff)
        return _biased_moe_batched!(down, x, router, bias, bt, logits, probs, order, gate_s, up_s, down_s)
    end
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

# K-quant banks route every token first, then run the shared host MoE: the
# batched Q8_K kernels for one token, the grouped multi-row tiles for a prompt.
function _biased_moe_batched!(down, x, router, bias, bt, logits, probs, order, gate_s, up_s, down_s)
    y = down.storage
    seq, dim = size(x)
    n_used = bt.n_expert_used
    experts = Matrix{Int}(undef, n_used, seq)
    wts = Matrix{Float32}(undef, n_used, seq)
    for t in 1:seq
        m = _cpu_router_logits!(logits, x, t, router)
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
    if seq == 1
        xf = Vector{Float32}(undef, dim)
        yrow = Vector{Float32}(undef, dim)
        @inbounds for i in 1:dim
            xf[i] = x[1, i]
        end
        _moe_host_experts_auto!(yrow, xf, view(experts, :, 1), view(wts, :, 1), n_used, gate_s, up_s, down_s)
        @inbounds for i in 1:dim
            y[1, i] = yrow[i]
        end
        return down
    end
    X = x isa Matrix{Float32} ? x : Matrix{Float32}(x)
    Y = y isa Matrix{Float32} ? y : Matrix{Float32}(undef, seq, dim)
    _moe_host_prefill!(Y, X, experts, wts, fill(true, seq), length(logits), gate_s, up_s, down_s)
    Y === y || copyto!(y, Y)
    return down
end
