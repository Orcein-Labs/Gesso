# Gemma 4 shared-spec architecture, GGUF import, and CPU lowering.

const _GEMMA4_WINDOW_MASK_ENABLED = Ref(true)
const _GEMMA4_SOFTCAP_ENABLED = Ref(true)

const _G4_SQRT_2_OVER_PI = Float32(sqrt(2.0 / pi))
const _G4_TOPK_CLAMP = 6.103515625f-5

@inline _g4_gelu(x::Float32) =
    0.5f0 * x * (1.0f0 + tanh(_G4_SQRT_2_OVER_PI * x * (1.0f0 + 0.044715f0 * x * x)))

Base.@kwdef mutable struct _Gemma4Scratch
    xn::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    A::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    attn_out::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    xf::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    mlp::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    rin::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    x2::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    moe::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    gate::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    up::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    gu::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    act_exp::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    y_exp::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    Q::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    K::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    V::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    O::Matrix{Float32} = Matrix{Float32}(undef, 0, 0)
    scores::Vector{Float32} = Float32[]
    logits::Vector{Float32} = Float32[]
    probs::Vector{Float32} = Float32[]
    order::Vector{Int} = Int[]
end

const _GEMMA4_SCRATCH = _Gemma4Scratch()

@inline function _g4_mat!(s::_Gemma4Scratch, field::Symbol, m::Int, n::Int)
    M = getfield(s, field)
    if size(M) != (m, n)
        M = Matrix{Float32}(undef, m, n)
        setfield!(s, field, M)
    end
    return M
end

@inline function _g4_vec!(v::Vector{T}, n::Int) where {T}
    length(v) < n && resize!(v, n)
    return view(v, 1:n)
end

# Row-wise RMSNorm: Y = X / rms(X) * w. w = nothing gives weightless norm.
function _g4_rmsnorm_rows!(Y::AbstractMatrix{Float32}, X::AbstractMatrix{Float32}, w, eps::Real)
    E = size(X, 2)
    eps_f = Float32(eps)
    w_vec = w isa FrozenParameter ? w.storage : w
    @inbounds for t in axes(X, 1)
        ss = 0.0f0
        for i in 1:E
            ss += X[t, i] * X[t, i]
        end
        r = 1.0f0 / sqrt(ss / E + eps_f)
        for i in 1:E
            Y[t, i] = X[t, i] * r * (w_vec === nothing ? 1.0f0 : Float32(w_vec[i]))
        end
    end
    return Y
end

# Per-head RMSNorm in place: each head segment of length hd is normalised.
function _g4_head_norm!(A::AbstractMatrix{Float32}, nheads::Int, hd::Int, w, eps::Real)
    eps_f = Float32(eps)
    w_vec = w isa FrozenParameter ? w.storage : w
    @inbounds for t in axes(A, 1), h in 1:nheads
        o = (h - 1) * hd
        ss = 0.0f0
        for d in 1:hd
            ss += A[t, o + d] * A[t, o + d]
        end
        r = 1.0f0 / sqrt(ss / hd + eps_f)
        for d in 1:hd
            A[t, o + d] = A[t, o + d] * r * (w_vec === nothing ? 1.0f0 : Float32(w_vec[d]))
        end
    end
    return A
end

# Matrix multiplication helper: handles QuantMatrix or standard Matrix (stored out, in).
function _g4_gemm!(dst::AbstractMatrix, x::AbstractMatrix, w)
    w_stor = w isa Union{ProjectionWeight, FrozenParameter, ExpertWeight} ? w.storage : w
    if w_stor isa QuantMatrix
        _quant_matmul!(dst, x, w_stor)
    else
        LinearAlgebra.mul!(dst, x, transpose(w_stor))
    end
    return dst
end

function _g4_expert_matrix(s::GgufQuantSlice, e::Int)
    rem(s.bytes_per_expert, s.ne1) == 0 || error("gemma4: expert row bytes not integral")
    rb = div(s.bytes_per_expert, s.ne1)
    return QuantMatrix(
        s.data, s.offset0 + (e - 1) * s.bytes_per_expert, s.ggml_type,
        s.ne0, s.ne1, rb, s.blck_size,
    )
end

function _g4_eval_gate_up!(s::_Gemma4Scratch, w, e::Int, x::AbstractMatrix{Float32})
    stor = w isa ExpertWeight ? w.storage : w
    if stor isa GgufQuantSlice
        qgu = _g4_expert_matrix(stor, e)
        gu = _g4_mat!(s, :gu, 1, qgu.nrows)
        _quant_matmul!(gu, x, qgu)
        return gu, qgu.nrows ÷ 2
    else
        W = view(stor, e, :, :)
        gu = _g4_mat!(s, :gu, 1, size(W, 1))
        LinearAlgebra.mul!(gu, x, transpose(W))
        return gu, size(W, 1) ÷ 2
    end
end

function _g4_eval_down!(s::_Gemma4Scratch, w, e::Int, a::AbstractMatrix{Float32})
    stor = w isa ExpertWeight ? w.storage : w
    if stor isa GgufQuantSlice
        qd = _g4_expert_matrix(stor, e)
        y = _g4_mat!(s, :y_exp, 1, qd.nrows)
        _quant_matmul!(y, a, qd)
        return y
    else
        W = view(stor, e, :, :)
        y = _g4_mat!(s, :y_exp, 1, size(W, 1))
        LinearAlgebra.mul!(y, a, transpose(W))
        return y
    end
end

# --- Named capability-checked lowering functions ------------------------------

# NeoX RoPE: pair (d, d + n_rot/2) rotates by pos * inv_freq (with optional rope_freqs folded in).
function _apply_rope_neox!(backend, A::AbstractMatrix{Float32}, positions, nheads::Int, hd::Int, n_rot::Int, inv_freq::Vector{Float64})
    backend_name(backend) === :cpu || throw(LoweringNotImplemented(:rope_freq_factors, backend_name(backend)))
    half = n_rot ÷ 2
    N = size(A, 1)
    @inbounds for t in 1:N
        pos = Float64(positions[t])
        for i in 1:half
            theta = pos * inv_freq[i]
            c = Float32(cos(theta))
            s = Float32(sin(theta))
            for h in 1:nheads
                o = (h - 1) * hd
                x0 = A[t, o + i]
                x1 = A[t, o + i + half]
                A[t, o + i] = x0 * c - x1 * s
                A[t, o + i + half] = x0 * s + x1 * c
            end
        end
    end
    return A
end

function _apply_embedding_scale!(backend, h, tensors)
    backend_name(backend) === :cpu || throw(LoweringNotImplemented(:embedding_scale, backend_name(backend)))
    scale = eltype(h.storage)(tensors.spec.embedding_scale)
    h.storage .*= scale
    return h
end

function _apply_logit_softcap!(backend, logits, tensors)
    backend_name(backend) === :cpu || throw(LoweringNotImplemented(:logit_softcap, backend_name(backend)))
    if _GEMMA4_SOFTCAP_ENABLED[] && tensors.spec.final_logit_softcap > 0
        cap = eltype(logits.storage)(tensors.spec.final_logit_softcap)
        @inbounds for i in eachindex(logits.storage)
            logits.storage[i] = cap * tanh(logits.storage[i] / cap)
        end
    end
    return logits
end

function _gemma4_v_norm!(backend, vh::AbstractMatrix{Float32}, eps; nheads::Int=1, hd::Int=size(vh, 2))
    backend_name(backend) === :cpu || throw(LoweringNotImplemented(:v_norm_weightless, backend_name(backend)))
    eps_f = Float32(eps)
    @inbounds for t in axes(vh, 1), h in 1:nheads
        o = (h - 1) * hd
        ss = 0.0f0
        for d in 1:hd
            ss += vh[t, o + d] * vh[t, o + d]
        end
        r = 1.0f0 / sqrt(ss / hd + eps_f)
        for d in 1:hd
            vh[t, o + d] *= r
        end
    end
    return vh
end

function _gemma4_k_only_v!(backend, k)
    backend_name(backend) === :cpu || throw(LoweringNotImplemented(:k_only_global_attention, backend_name(backend)))
    return copy(k)
end

function _gemma4_layer_output_scale!(backend, out, scale_vec)
    backend_name(backend) === :cpu || throw(LoweringNotImplemented(:layer_output_scale, backend_name(backend)))
    s = scale_vec isa AbstractArray ? scale_vec[1] : scale_vec
    out .*= s
    return out
end

function _gemma4_parallel_moe_ffn!(backend, attn, bt, eps::Real; scratch::_Gemma4Scratch=_GEMMA4_SCRATCH)
    backend_name(backend) === :cpu || throw(LoweringNotImplemented(:gemma4_parallel_moe, backend_name(backend)))
    X = attn isa Activation ? attn.storage : attn
    N, E = size(X)
    eps_f = Float32(eps)

    # 1. Dense FFN (reuse hoisted N x E and N x n_ff buffers)
    xf = _g4_mat!(scratch, :xf, N, E)
    _g4_rmsnorm_rows!(xf, X, bt.ffn_rms, eps_f)
    n_ff = bt.wgate.shape[1]
    gate = _g4_mat!(scratch, :gate, N, n_ff)
    up = _g4_mat!(scratch, :up, N, n_ff)
    _g4_gemm!(gate, xf, bt.wgate)
    _g4_gemm!(up, xf, bt.wup)
    @inbounds for i in eachindex(gate)
        gate[i] = _g4_gelu(gate[i]) * up[i]
    end
    mlp = _g4_mat!(scratch, :mlp, N, E)
    _g4_gemm!(mlp, gate, bt.wdown)
    _g4_rmsnorm_rows!(mlp, mlp, bt.ffn_post_rms_1, eps_f)

    # 2. Routed MoE (reuse hoisted N x E buffers)
    inv_sqrt_e = 1.0f0 / sqrt(Float32(E))
    rin = _g4_mat!(scratch, :rin, N, E)
    _g4_rmsnorm_rows!(rin, X, nothing, eps_f)
    r_scale = bt.wrouter_scale isa FrozenParameter ? bt.wrouter_scale.storage : bt.wrouter_scale
    @inbounds for t in 1:N, i in 1:E
        rin[t, i] = rin[t, i] * inv_sqrt_e * Float32(r_scale[i])
    end
    x2 = _g4_mat!(scratch, :x2, N, E)
    _g4_rmsnorm_rows!(x2, X, bt.ffn_pre_rms_2, eps_f)

    r_stor = bt.wrouter.storage
    n_expert = r_stor isa QuantMatrix ? r_stor.nrows : size(r_stor, 1)
    n_used = bt.n_expert_used
    down_scale = bt.wexp_down_scale isa FrozenParameter ? bt.wexp_down_scale.storage : bt.wexp_down_scale

    moe = _g4_mat!(scratch, :moe, N, E)
    fill!(moe, 0.0f0)
    logits = _g4_vec!(scratch.logits, n_expert)
    probs = _g4_vec!(scratch.probs, n_expert)
    order = _g4_vec!(scratch.order, n_expert)

    for t in 1:N
        if r_stor isa QuantMatrix
            _quant_matmul!(reshape(logits, 1, n_expert), view(rin, t:t, :), r_stor)
        else
            @inbounds for e in 1:n_expert
                s = 0.0f0
                for i in 1:E
                    s += r_stor[e, i] * rin[t, i]
                end
                logits[e] = s
            end
        end
        mx = maximum(logits)
        den = 0.0f0
        @inbounds for e in 1:n_expert
            v = exp(logits[e] - mx)
            probs[e] = v
            den += v
        end
        probs ./= den
        partialsortperm!(order, probs, 1:n_used; rev=true)
        wsum = 0.0f0
        @inbounds for k in 1:n_used
            wsum += probs[order[k]]
        end
        denom = max(wsum, _G4_TOPK_CLAMP)
        x2_row = view(x2, t:t, :)

        for k in 1:n_used
            e = order[k]
            w = probs[e] / denom
            gu, n_ff_exp = _g4_eval_gate_up!(scratch, bt.wexp_gate_up, e, x2_row)
            a = _g4_mat!(scratch, :act_exp, 1, n_ff_exp)
            @inbounds for i in 1:n_ff_exp
                a[1, i] = _g4_gelu(gu[1, i]) * gu[1, n_ff_exp + i]
            end
            y = _g4_eval_down!(scratch, bt.wexp_down, e, a)
            sc = w * Float32(down_scale[e])
            @inbounds for i in 1:E
                moe[t, i] += sc * y[1, i]
            end
        end
    end
    _g4_rmsnorm_rows!(moe, moe, bt.ffn_post_rms_2, eps_f)

    # 3. Sum dense and moe in place into mlp, then post_ffw_norm
    mlp .+= moe
    _g4_rmsnorm_rows!(mlp, mlp, bt.ffn_post_rms, eps_f)
    return mlp
end

# --- Block prefill and generate -----------------------------------------------

function _gemma4_prefill_block!(backend, h, bt, tensors, positions, wl, eps, zeros_like; scratch::_Gemma4Scratch=_GEMMA4_SCRATCH)
    X = h isa Activation ? h.storage : h
    N, E = size(X)
    hd = bt.d_head_layer
    nkv = bt.n_kv_heads_layer
    nh = tensors.spec.n_heads
    qdim = nh * hd
    kvdim = nkv * hd
    window = bt.attn_window
    win = (_GEMMA4_WINDOW_MASK_ENABLED[] && window > 0) ? window : 0
    eps_f = Float32(eps)

    # Attention pre-norm
    xn = _g4_mat!(scratch, :xn, N, E)
    _g4_rmsnorm_rows!(xn, X, bt.attn_rms, eps_f)

    # Q, K, V projections
    Q = _g4_mat!(scratch, :Q, N, qdim)
    _g4_gemm!(Q, xn, bt.wq)
    K = _g4_mat!(scratch, :K, N, kvdim)
    _g4_gemm!(K, xn, bt.wk)
    V = _g4_mat!(scratch, :V, N, kvdim)
    if haskey(bt, :wv) && bt.wv !== nothing
        _g4_gemm!(V, xn, bt.wv)
    else
        V .= _gemma4_k_only_v!(backend, K)
    end

    # Q/K/V norms and RoPE
    _g4_head_norm!(Q, nh, hd, bt.wq_norm, eps_f)
    _g4_head_norm!(K, nkv, hd, bt.wk_norm, eps_f)
    _gemma4_v_norm!(backend, V, eps_f; nheads=nkv, hd=hd)

    n_rot = bt.rope_layer.rotary_dim == 0 ? hd : bt.rope_layer.rotary_dim
    _apply_rope_neox!(backend, Q, positions, nh, hd, n_rot, bt.rope_layer.inv_freq)
    _apply_rope_neox!(backend, K, positions, nkv, hd, n_rot, bt.rope_layer.inv_freq)

    # Causal attention with scale 1.0 (scores buffer hoisted outside loops)
    O = _g4_mat!(scratch, :O, N, qdim)
    rep = nh ÷ nkv
    pos0 = first(positions)
    for t in 1:N
        p = pos0 + t - 1
        lo = win > 0 ? max(0, p - (win - 1)) : 0
        n_keys = p - lo + 1
        scores = _g4_vec!(scratch.scores, n_keys)
        for h_idx in 1:nh
            kv_h = (h_idx - 1) ÷ rep
            qo = (h_idx - 1) * hd
            kvo = kv_h * hd
            @inbounds for i in 1:n_keys
                k_row = lo + i - 1 - pos0 + 1
                s = 0.0f0
                for d in 1:hd
                    s += Q[t, qo + d] * K[k_row, kvo + d]
                end
                scores[i] = s
            end
            mx = maximum(scores)
            den = 0.0f0
            @inbounds for i in 1:n_keys
                v = exp(scores[i] - mx)
                scores[i] = v
                den += v
            end
            inv_den = 1.0f0 / den
            @inbounds for d in 1:hd
                acc = 0.0f0
                for i in 1:n_keys
                    k_row = lo + i - 1 - pos0 + 1
                    acc += (scores[i] * inv_den) * V[k_row, kvo + d]
                end
                O[t, qo + d] = acc
            end
        end
    end

    # Post attention norm & residual
    A = _g4_mat!(scratch, :A, N, E)
    _g4_gemm!(A, O, bt.wo)
    _g4_rmsnorm_rows!(A, A, bt.attn_post_rms, eps_f)
    attn_out = _g4_mat!(scratch, :attn_out, N, E)
    attn_out .= A .+ X

    # Parallel MoE FFN & optional layer output scale
    cur = _gemma4_parallel_moe_ffn!(backend, attn_out, bt, eps_f; scratch=scratch)
    cur .+= attn_out
    if haskey(bt, :layer_output_scale) && bt.layer_output_scale !== nothing
        _gemma4_layer_output_scale!(backend, cur, bt.layer_output_scale.storage)
    end
    X .= cur
    return h
end

function _gemma4_generate_block!(backend, hp, kc_layer, vc_layer, bt, tensors, s0, s1, wl, eps, zeros_like; scratch::_Gemma4Scratch=_GEMMA4_SCRATCH)
    X = hp isa Activation ? hp.storage : hp
    N, E = size(X)
    hd = bt.d_head_layer
    nkv = bt.n_kv_heads_layer
    nh = tensors.spec.n_heads
    qdim = nh * hd
    kvdim = nkv * hd
    window = bt.attn_window
    win = (_GEMMA4_WINDOW_MASK_ENABLED[] && window > 0) ? window : 0
    eps_f = Float32(eps)

    # Attention pre-norm
    xn = _g4_mat!(scratch, :xn, N, E)
    _g4_rmsnorm_rows!(xn, X, bt.attn_rms, eps_f)

    # Q, K, V projections
    Q = _g4_mat!(scratch, :Q, N, qdim)
    _g4_gemm!(Q, xn, bt.wq)
    K = _g4_mat!(scratch, :K, N, kvdim)
    _g4_gemm!(K, xn, bt.wk)
    V = _g4_mat!(scratch, :V, N, kvdim)
    if haskey(bt, :wv) && bt.wv !== nothing
        _g4_gemm!(V, xn, bt.wv)
    else
        V .= _gemma4_k_only_v!(backend, K)
    end

    # Q/K/V norms and RoPE
    _g4_head_norm!(Q, nh, hd, bt.wq_norm, eps_f)
    _g4_head_norm!(K, nkv, hd, bt.wk_norm, eps_f)
    _gemma4_v_norm!(backend, V, eps_f; nheads=nkv, hd=hd)

    n_rot = bt.rope_layer.rotary_dim == 0 ? hd : bt.rope_layer.rotary_dim
    positions = (s0 - 1):(s1 - 1)
    _apply_rope_neox!(backend, Q, positions, nh, hd, n_rot, bt.rope_layer.inv_freq)
    _apply_rope_neox!(backend, K, positions, nkv, hd, n_rot, bt.rope_layer.inv_freq)

    # Write K and V into per-layer KV cache for rows s0:s1
    for t in 1:N
        p = s0 + t - 1
        for h in 1:nkv
            kvo = (h - 1) * hd
            kc_layer.storage[p, h, :] .= view(K, t, (kvo+1):(kvo+hd))
            vc_layer.storage[p, h, :] .= view(V, t, (kvo+1):(kvo+hd))
        end
    end

    # Attend over keys in 1:s1 with scale 1.0 and sliding-window mask (scores hoisted)
    O = _g4_mat!(scratch, :O, N, qdim)
    rep = nh ÷ nkv
    for t in 1:N
        pos = s0 + t - 1
        lo = win > 0 ? max(1, pos - win + 1) : 1
        n_keys = pos - lo + 1
        scores = _g4_vec!(scratch.scores, n_keys)
        for h_idx in 1:nh
            kv_h = (h_idx - 1) ÷ rep + 1
            qo = (h_idx - 1) * hd
            @inbounds for i in 1:n_keys
                k_idx = lo + i - 1
                s = 0.0f0
                for d in 1:hd
                    s += Q[t, qo + d] * kc_layer.storage[k_idx, kv_h, d]
                end
                scores[i] = s
            end
            mx = maximum(scores)
            den = 0.0f0
            @inbounds for i in 1:n_keys
                v = exp(scores[i] - mx)
                scores[i] = v
                den += v
            end
            inv_den = 1.0f0 / den
            @inbounds for d in 1:hd
                acc = 0.0f0
                for i in 1:n_keys
                    k_idx = lo + i - 1
                    acc += (scores[i] * inv_den) * vc_layer.storage[k_idx, kv_h, d]
                end
                O[t, qo + d] = acc
            end
        end
    end

    # Post attention norm & residual
    A = _g4_mat!(scratch, :A, N, E)
    _g4_gemm!(A, O, bt.wo)
    _g4_rmsnorm_rows!(A, A, bt.attn_post_rms, eps_f)
    attn_out = _g4_mat!(scratch, :attn_out, N, E)
    attn_out .= A .+ X

    # Parallel MoE FFN & optional layer output scale
    cur = _gemma4_parallel_moe_ffn!(backend, attn_out, bt, eps_f; scratch=scratch)
    cur .+= attn_out
    if haskey(bt, :layer_output_scale) && bt.layer_output_scale !== nothing
        _gemma4_layer_output_scale!(backend, cur, bt.layer_output_scale.storage)
    end
    X .= cur
    return hp
end

# --- GGUF Spec and Param Map --------------------------------------------------

function _spec_from_gguf_gemma4(kv::AbstractDict, tensor_names)
    arch = "gemma4"
    miss(k) = gesso_error(
        ERR_RUNTIME,
        "spec_from_gguf_kv: GGUF metadata has no $k, which this architecture needs. " *
        "The file was not written by a converter Gesso supports.";
        reason=:missing_metadata,
        key=String(k),
    )
    unsup(msg) = gesso_error(
        ERR_RUNTIME,
        msg;
        reason=:unsupported_architecture,
        architecture=arch,
    )
    req(key) = (v = get(kv, "$arch.$key", nothing); v === nothing ? throw(miss("$arch.$key")) : v)

    Int(get(kv, "$arch.embedding_length_per_layer_input", 0)) == 0 ||
        throw(unsup("gemma4: embedding_length_per_layer_input != 0 (PLE) is unsupported"))
    Int(get(kv, "$arch.attention.shared_kv_layers", 0)) == 0 ||
        throw(unsup("gemma4: attention.shared_kv_layers != 0 is unsupported"))

    n_layers = Int(req("block_count"))
    h = Int(req("embedding_length"))
    nh = Int(req("attention.head_count"))
    raw_kv = req("attention.head_count_kv")
    kv_heads = raw_kv isa AbstractVector ? Int.(raw_kv) : fill(Int(raw_kv), n_layers)
    length(kv_heads) == n_layers ||
        throw(unsup("gemma4: head_count_kv length differs from block_count"))

    swa_pattern = collect(Bool, req("attention.sliding_window_pattern"))
    length(swa_pattern) == n_layers ||
        throw(unsup("gemma4: sliding_window_pattern length differs from block_count"))

    hd_global = Int(req("attention.key_length"))
    hd_swa = Int(req("attention.key_length_swa"))
    Int(req("attention.value_length")) == hd_global ||
        throw(unsup("gemma4: key_length != value_length (global)"))
    Int(req("attention.value_length_swa")) == hd_swa ||
        throw(unsup("gemma4: key_length_swa != value_length_swa"))

    rot_global = Int(req("rope.dimension_count"))
    rot_swa = Int(get(kv, "$arch.rope.dimension_count_swa", hd_swa))

    n_expert = Int(get(kv, "$arch.expert_count", 0))
    n_expert > 0 ||
        throw(unsup("gemma4: dense variants (expert_count == 0) are not yet supported"))
    n_used = Int(req("expert_used_count"))
    n_ff_exp = Int(req("expert_feed_forward_length"))
    moe = MoeSpec(; n_experts=n_expert, n_expert_used=n_used, expert_ff=n_ff_exp)

    tokens = get(kv, "tokenizer.ggml.tokens", nothing)
    token_count = if tokens isa Integer
        Int(tokens)
    elseif tokens isa AbstractVector
        length(tokens)
    elseif tokens === nothing
        0
    else
        throw(miss("tokenizer.ggml.tokens"))
    end
    declared_vocab = get(kv, "$arch.vocab_size", nothing)
    vocab = declared_vocab === nothing ? token_count : Int(declared_vocab)
    vocab > 0 || throw(miss("tokenizer.ggml.tokens"))

    features = Set{Symbol}([
        :qk_norm,
        :embedding_scale,
        :logit_softcap,
        :v_norm_weightless,
        :k_only_global_attention,
        :gemma4_parallel_moe,
    ])
    if tensor_names === nothing || any(n -> endswith(n, ".layer_output_scale.weight"), tensor_names)
        push!(features, :layer_output_scale)
    end
    if tensor_names === nothing || "rope_freqs.weight" in tensor_names
        push!(features, :rope_freq_factors)
    end

    return ArchitectureSpec(;
        family=:gemma4,
        hidden_size=h,
        num_layers=n_layers,
        n_heads=nh,
        n_kv_heads=maximum(kv_heads),
        head_dim=hd_global,
        head_dim_swa=hd_swa,
        layer_kv_heads=kv_heads,
        vocab_size=vocab,
        intermediate_size=Int(req("feed_forward_length")),
        activation_kind=:geglu,
        rope=RoPEPolicy(;
            theta=Float64(req("rope.freq_base")),
            rotary_dim=rot_global == hd_global ? 0 : rot_global,
            interleaved=false,
        ),
        rope_swa=RoPEPolicy(;
            theta=Float64(get(kv, "$arch.rope.freq_base_swa", 10000.0)),
            rotary_dim=rot_swa == hd_swa ? 0 : rot_swa,
            interleaved=false,
        ),
        sliding_window=Int(req("attention.sliding_window")),
        swa_layers=swa_pattern,
        tie_word_embeddings=tensor_names === nothing || !("output.weight" in tensor_names),
        features=features,
        moe=moe,
        rms_norm_eps=Float64(req("attention.layer_norm_rms_epsilon")),
        embedding_scale=sqrt(Float64(h)),
        final_logit_softcap=Float64(get(kv, "$arch.final_logit_softcapping", 0.0)),
    )
end

function _gguf_gemma4_param_map(spec::ArchitectureSpec)
    ref(role, name, kind=:projection) = ParamRef(SemanticParamId(role, 0, kind), name)
    layer_refs = ParamRef[
        ref(ROLE_NORM_PRE_ATTN, "attn_norm.weight", :frozen),
        ref(ROLE_Q, "attn_q.weight"),
        ref(ROLE_K, "attn_k.weight"),
        ref(ROLE_V, "attn_v.weight"),
        ref(ROLE_O, "attn_output.weight"),
        ref(ROLE_Q_NORM, "attn_q_norm.weight", :frozen),
        ref(ROLE_K_NORM, "attn_k_norm.weight", :frozen),
        ref(ROLE_ATTN_POST_NORM, "post_attention_norm.weight", :frozen),
        ref(ROLE_NORM_POST_ATTN, "ffn_norm.weight", :frozen),
        ref(ROLE_GATE, "ffn_gate.weight"),
        ref(ROLE_UP, "ffn_up.weight"),
        ref(ROLE_DOWN, "ffn_down.weight"),
        ref(ROLE_FFN_POST_NORM_1, "post_ffw_norm_1.weight", :frozen),
        ref(ROLE_FFN_PRE_NORM_2, "pre_ffw_norm_2.weight", :frozen),
        ref(ROLE_ROUTER, "ffn_gate_inp.weight"),
        ref(ROLE_ROUTER_SCALE, "ffn_gate_inp.scale", :frozen),
        ref(ROLE_EXPERT_GATE_UP, "ffn_gate_up_exps.weight", :expert),
        ref(ROLE_EXPERT_DOWN, "ffn_down_exps.weight", :expert),
        ref(ROLE_EXPERT_DOWN_SCALE, "ffn_down_exps.scale", :frozen),
        ref(ROLE_FFN_POST_NORM_2, "post_ffw_norm_2.weight", :frozen),
        ref(ROLE_FFN_POST_NORM, "post_ffw_norm.weight", :frozen),
    ]
    if :layer_output_scale in spec.features
        push!(layer_refs, ref(ROLE_LAYER_OUTPUT_SCALE, "layer_output_scale.weight", :frozen))
    end
    return FamilyParamMap(
        :gemma4,
        "blk.",
        layer_refs,
        "token_embd.weight",
        "output_norm.weight",
        spec.tie_word_embeddings ? nothing : "output.weight";
        rope_freqs_external=(:rope_freq_factors in spec.features ? "rope_freqs.weight" : nothing),
    )
end
