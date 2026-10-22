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

# erf in Float64 without a special-functions dependency (Gesso takes only
# stdlibs): the Maclaurin series below |x| = 2, the erfc continued fraction
# from there to 6, and ±1 beyond, where erf is 1 to double precision.
function _cpu_erf(x::Float64)
    z = abs(x)
    z >= 6.0 && return copysign(1.0, x)
    if z < 2.0
        z2 = z * z
        term = z
        acc = z
        n = 0
        while abs(term) > 1e-17 * acc
            n += 1
            term *= -z2 / n
            acc += term / (2n + 1)
        end
        return copysign(acc * (2 / sqrt(pi)), x)
    end
    return copysign(1.0 - _cpu_erfc_cf(z), x)
end

# exp(-c·x²) with x² split as xh² + (x - xh)(x + xh), xh = x rounded to
# Float32: xh² is exact in Float64, so the rounding of x² (relative error
# ~x²·eps in the result) never reaches the exponent.
function _cpu_exp_neg_sq(x::Float64, c::Float64)
    xh = Float64(Float32(x))
    return exp(-c * (xh * xh)) * exp(-c * ((x - xh) * (x + xh)))
end

# erfc(z) for z ≥ 1.25 by the continued fraction
# exp(-z²) / (√π (z + (1/2)/(z + 1/(z + (3/2)/(z + …))))), from a fixed tail.
# `ez` is exp(-z²); callers that know it more exactly than z pass it in.
function _cpu_erfc_cf(z::Float64, ez::Float64=_cpu_exp_neg_sq(z, 1.0))
    t = z
    for n in 120:-1:1
        t = z + (n / 2) / t
    end
    return ez / (sqrt(pi) * t)
end

# Exact (erf) GELU, the form Hugging Face's `gelu` activation computes. Far
# left, 1 + erf cancels, so that tail uses erfc(-x/√2) directly, with
# exp(-x²/2) taken from x itself so the rounding of x/√2 stays out of it.
function _cpu_gelu_erf(x::Float64)
    z = -x / sqrt(2.0)
    z >= 1.25 && return 0.5 * x * _cpu_erfc_cf(z, _cpu_exp_neg_sq(x, 0.5))
    return 0.5 * x * (1.0 + _cpu_erf(-z))
end

# GEGLU over storage: dst = gelu(gate) .* up, in Float64 like the SwiGLU kernel.
function _cpu_geglu!(dst::AbstractArray, gate::AbstractArray, up::AbstractArray)
    for i in eachindex(dst, gate, up)
        dst[i] = _cpu_gelu_erf(Float64(gate[i])) * up[i]
    end
    return dst
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

# Router logits for token t; returns their max. The router is (n_exp, h)
# column-major, so walking e innermost reads it contiguously and vectorizes
# across experts. Each logit still sums over i in order, so the result is
# bit-identical to the per-expert dot product: on the 35B (256 x 2048, 40
# layers) that strided form cost 60 ms per token against 6.5 ms for this one.
function _cpu_router_logits!(logits::AbstractVector{T}, x::AbstractMatrix, t::Int,
    router::AbstractMatrix) where {T}
    n_exp, h = size(router)
    fill!(logits, zero(T))
    @inbounds for i in 1:h
        xi = T(x[t, i])
        @simd for e in 1:n_exp
            logits[e] += xi * router[e, i]
        end
    end
    return maximum(logits)
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
        m = _cpu_router_logits!(logits, x, t, router)
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
