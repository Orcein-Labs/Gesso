# CPU Gated DeltaNet + the hybrid full-attention extras (qk_norm is rmsnorm
# on the head axis; attn_output_gate is a sigmoid multiply). These are the
# PORTABLE_CORRECTNESS bodies for Qwen3.5 / Ornith. They live next to cpu.jl
# so the interpreter can compose them without a second operator vocabulary.

function _cpu_silu(x::T) where {T<:AbstractFloat}
    return x / (one(T) + exp(-x))
end

function _cpu_sigmoid(x::T) where {T<:AbstractFloat}
    return one(T) / (one(T) + exp(-x))
end

function _cpu_softplus(x::T) where {T<:AbstractFloat}
    # log(1+exp(x)) with the usual split so large |x| stays finite
    return x > T(20) ? x : log1p(exp(x))
end

function _cpu_l2norm!(x::AbstractArray, dim::Int, eps::Float64=1e-6)
    # Normalize along `dim` (1-based). Matches HF `l2norm` / FLA rsqrt form.
    n = size(x, dim)
    lead = ntuple(i -> i == dim ? 1 : size(x, i), ndims(x))
    for I in CartesianIndices(lead)
        acc = 0.0
        for k in 1:n
            idx = ntuple(d -> d == dim ? k : I[d], ndims(x))
            acc += abs2(x[idx...])
        end
        s = sqrt(acc + eps)
        inv = 1.0 / s
        for k in 1:n
            idx = ntuple(d -> d == dim ? k : I[d], ndims(x))
            x[idx...] *= inv
        end
    end
    return x
end

function _cpu_sigmoid_mul!(dst::AbstractArray, src::AbstractArray, gate::AbstractArray)
    for i in eachindex(dst, src, gate)
        dst[i] = src[i] * _cpu_sigmoid(gate[i])
    end
    return dst
end

"""
Causal depthwise conv1d + SiLU.

`x` is (seq, channels), `weight` is (channels, 1, kernel) as stored by the
safetensors loader. `conv_state` is (channels, kernel-1): the left-pad for
decode, or zeros for a fresh prefill. Mutates `conv_state` to the last
`kernel-1` inputs (post-concat, pre-activation).
"""
function _cpu_causal_conv1d_silu!(
    dst::AbstractArray,
    x::AbstractArray,
    weight::AbstractArray,
    conv_state::AbstractArray,
)
    seq, channels = size(x)
    k = size(weight, 3)
    size(weight, 1) == channels || error("causal_conv1d: weight channels $(size(weight,1)) ≠ $channels")
    size(conv_state, 1) == channels && size(conv_state, 2) == k - 1 || error(
        "causal_conv1d: conv_state must be (channels, kernel-1) = ($channels, $(k-1)), got $(size(conv_state))",
    )
    # padded[t] for t=1:(k-1) is conv_state; t=k:(k-1+seq) is x
    for t in 1:seq, c in 1:channels
        acc = 0.0
        for p in 1:k
            src = t - (k - p)
            v = if src <= 0
                conv_state[c, src+(k-1)]
            else
                x[src, c]
            end
            acc += weight[c, 1, p] * v
        end
        dst[t, c] = _cpu_silu(acc)
    end
    # update conv state: last k-1 inputs of the concatenated stream
    if seq >= k - 1
        for p in 1:(k-1), c in 1:channels
            conv_state[c, p] = x[seq-(k-1)+p, c]
        end
    else
        # shift old state left by seq, append x
        keep = (k - 1) - seq
        for p in 1:keep, c in 1:channels
            conv_state[c, p] = conv_state[c, p+seq]
        end
        for p in 1:seq, c in 1:channels
            conv_state[c, keep+p] = x[p, c]
        end
    end
    return dst
end

"""
Recurrent gated delta rule (HF `torch_recurrent_gated_delta_rule`).

Q/K: (seq, n_k_heads, k_dim); V/out: (seq, n_v_heads, v_dim);
g, beta: (seq, n_v_heads); state: (n_v_heads, k_dim, v_dim) mutated in place.
When n_v_heads > n_k_heads, Q/K are repeated (GQA-style) to n_v_heads.
`v_head_tiled=false` groups V heads by K head (HuggingFace `[K0,K0,K1,K1]`);
`v_head_tiled=true` tiles them (llama.cpp GGUF `[K0,K1,K0,K1]`). Equal head
counts make the two pairings identical.
"""
function _cpu_gated_delta_rule!(
    out::AbstractArray,
    q::AbstractArray,
    k::AbstractArray,
    v::AbstractArray,
    g::AbstractArray,
    beta::AbstractArray,
    state::AbstractArray;
    l2norm::Bool=true,
    v_head_tiled::Bool=false,
)
    seq = size(q, 1)
    nk, dk = size(q, 2), size(q, 3)
    nv, dv = size(v, 2), size(v, 3)
    nv % nk == 0 || error("gated_delta_rule: n_v_heads $nv not divisible by n_k_heads $nk")
    group = div(nv, nk)
    size(state) == (nv, dk, dv) ||
        error("gated_delta_rule: state $(size(state)) ≠ ($nv, $dk, $dv)")
    qn = Array{Float64}(undef, seq, nv, dk)
    kn = Array{Float64}(undef, seq, nv, dk)
    for t in 1:seq, h in 1:nk, ghead in 0:(group-1)
        hh = v_head_tiled ? (ghead * nk + h) : ((h - 1) * group + ghead + 1)
        for d in 1:dk
            qn[t, hh, d] = q[t, h, d]
            kn[t, hh, d] = k[t, h, d]
        end
    end
    if l2norm
        _cpu_l2norm!(qn, 3)
        _cpu_l2norm!(kn, 3)
    end
    scale = 1.0 / sqrt(Float64(dk))
    for t in 1:seq, hh in 1:nv, d in 1:dk
        qn[t, hh, d] *= scale
    end
    for t in 1:seq
        for hh in 1:nv
            decay = exp(g[t, hh])
            for i in 1:dk, j in 1:dv
                state[hh, i, j] *= decay
            end
            bt = beta[t, hh]
            for j in 1:dv
                kv_mem = 0.0
                for i in 1:dk
                    kv_mem += state[hh, i, j] * kn[t, hh, i]
                end
                delta = (v[t, hh, j] - kv_mem) * bt
                for i in 1:dk
                    state[hh, i, j] += kn[t, hh, i] * delta
                end
            end
            for j in 1:dv
                acc = 0.0
                for i in 1:dk
                    acc += state[hh, i, j] * qn[t, hh, i]
                end
                out[t, hh, j] = acc
            end
        end
    end
    return out
end

"""
Softmax router, top-`n_used` (renormalized), SwiGLU experts.

`x`/`y` are (seq, hidden). `router` is (n_experts, hidden). `gate3`/`up3`
are (n_experts, expert_ff, hidden). `down3` is (n_experts, hidden, expert_ff).
Matches llama.cpp `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` with top-k
renormalize (`norm=true` in `build_moe_ffn`).
"""
function _cpu_moe_topk_swiglu!(
    y::AbstractMatrix,
    x::AbstractMatrix,
    router::AbstractMatrix,
    gate3::AbstractArray,
    up3::AbstractArray,
    down3::AbstractArray,
    n_used::Int,
)
    seq, h = size(x)
    n_exp = size(router, 1)
    ff = size(gate3, 2)
    0 < n_used <= n_exp || error("_cpu_moe_topk_swiglu!: n_used $n_used out of 1..$n_exp")
    size(y) == (seq, h) || error("_cpu_moe_topk_swiglu!: y shape $(size(y)) ≠ ($seq, $h)")
    size(router) == (n_exp, h) ||
        error("_cpu_moe_topk_swiglu!: router $(size(router)) ≠ ($n_exp, $h)")
    size(gate3) == (n_exp, ff, h) ||
        error("_cpu_moe_topk_swiglu!: gate3 $(size(gate3)) ≠ ($n_exp, $ff, $h)")
    size(up3) == (n_exp, ff, h) ||
        error("_cpu_moe_topk_swiglu!: up3 $(size(up3)) ≠ ($n_exp, $ff, $h)")
    size(down3) == (n_exp, h, ff) ||
        error("_cpu_moe_topk_swiglu!: down3 $(size(down3)) ≠ ($n_exp, $h, $ff)")
    T = eltype(y)
    fill!(y, zero(T))
    logits = Vector{T}(undef, n_exp)
    order = Vector{Int}(undef, n_exp)
    for t in 1:seq
        m = typemin(T)
        for e in 1:n_exp
            acc = zero(T)
            @inbounds for i in 1:h
                acc += x[t, i] * router[e, i]
            end
            logits[e] = acc
            acc > m && (m = acc)
        end
        s = zero(T)
        for e in 1:n_exp
            v = exp(logits[e] - m)
            logits[e] = v
            s += v
        end
        invs = one(T) / s
        for e in 1:n_exp
            logits[e] *= invs
            order[e] = e
        end
        # highest weight first; ties keep the lower expert index (stable)
        sort!(order; by=e -> (-logits[e], e))
        wsum = zero(T)
        for k in 1:n_used
            wsum += logits[order[k]]
        end
        wsum > 0 || continue
        invw = one(T) / wsum
        for k in 1:n_used
            e = order[k]
            w = logits[e] * invw
            for j in 1:ff
                gacc = zero(T)
                uacc = zero(T)
                @inbounds for i in 1:h
                    gacc += gate3[e, j, i] * x[t, i]
                    uacc += up3[e, j, i] * x[t, i]
                end
                hact = _cpu_silu(gacc) * uacc
                @inbounds for i in 1:h
                    y[t, i] += w * down3[e, i, j] * hact
                end
            end
        end
    end
    return y
end
