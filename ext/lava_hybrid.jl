# Lava hybrid-decode seams for qwen35moe (L2): the gated-delta-net mixer and
# the gating ops the engine's unfused path calls on every mixer layer, plus
# the host-expert MoE route.
#
# The engine's default bodies are CPU Array code. On a LavaArray they would
# scalar-index the device, so each seam below is a device method on the Lava
# tag. Storage must already be on the device: a host array here is a plan
# error, never copied silently.
#
# Precision. The kernels run in F32 with F32 recurrent state. The CPU oracle
# is F64. The F32 error bound is declared in test/test_lava_hybrid.jl.
#
# Not here yet: the fused single-token GDN decode (`_gdn_decode_core!` stays
# unimplemented, so `_gdn_fused_decode(::LavaBackend)` stays false and the
# unfused path runs) and device expert routing (L3).

const _LH_WG = (64,)

# Device storage check. Lava never copies a host array implicitly.
function _lh_device(x, op::Symbol)
    x isa Lava.LavaArray && return x
    throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "$op: storage is $(typeof(x)), not a LavaArray; call to_device first " *
            "(Lava never copies host storage implicitly)",
        ),
    )
end

_lh_host(x::Lava.LavaArray) = Array(x)
_lh_host(x) = x

# Causal depthwise conv1d + SiLU, one thread per (t, channel). Reads only the
# old conv_state and x, so the state update is a separate launch.
@kernel cpu=false function _lava_conv_silu_kernel!(dst, x, weight, state, channels::Int, k::Int, n::Int)
    i = @index(Global)
    i <= n || return
    t = (i - 1) ÷ channels + 1
    c = (i - 1) % channels + 1
    acc = 0.0f0
    for p in 1:k
        src = t - (k - p)
        v = src <= 0 ? state[c, src + (k - 1)] : x[src, c]
        acc += weight[c, 1, p] * v
    end
    dst[t, c] = acc / (1.0f0 + exp(-acc))
end

# Last k-1 inputs of the concatenated stream. `old` is a copy of the state
# whenever the shift reads it, so no thread reads a slot another thread writes.
@kernel cpu=false function _lava_conv_state_kernel!(state, old, x, seq::Int, channels::Int, k::Int, n::Int)
    i = @index(Global)
    i <= n || return
    c = (i - 1) % channels + 1
    p = (i - 1) ÷ channels + 1
    if seq >= k - 1
        state[c, p] = x[seq - (k - 1) + p, c]
    else
        keep = (k - 1) - seq
        if p <= keep
            state[c, p] = old[c, p + seq]
        else
            state[c, p] = x[p - keep, c]
        end
    end
end

# beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias). Softplus uses
# the same x > 20 split as the CPU oracle.
@kernel cpu=false function _lava_gdn_gates_kernel!(g, beta, a, b, A_log, dt_bias, nv::Int, n::Int)
    i = @index(Global)
    i <= n || return
    hh = (i - 1) % nv + 1
    t = (i - 1) ÷ nv + 1
    beta[t, hh] = 1.0f0 / (1.0f0 + exp(-b[t, hh]))
    x = a[t, hh] + dt_bias[hh]
    sp = x > 20.0f0 ? x : log(1.0f0 + exp(x))
    g[t, hh] = -exp(A_log[hh]) * sp
end

# Repeat K heads to V heads (HF grouping, or llama.cpp tiling when
# v_head_tiled), L2-normalize q and k, and scale q by 1/sqrt(dk). One thread
# per (t, value head); the head mapping and eps match `_cpu_gated_delta_rule!`.
@kernel cpu=false function _lava_gdn_qk_kernel!(
    qn, kn, q, k, nk::Int, group::Int, tiled::Bool, l2norm::Bool,
    scale::Float32, eps::Float32, nv::Int, dk::Int, n::Int,
)
    i = @index(Global)
    i <= n || return
    hh = (i - 1) % nv + 1
    t = (i - 1) ÷ nv + 1
    h = tiled ? (hh - 1) % nk + 1 : (hh - 1) ÷ group + 1
    sq = 0.0f0
    sk = 0.0f0
    for d in 1:dk
        sq += abs2(q[t, h, d])
        sk += abs2(k[t, h, d])
    end
    iq = l2norm ? 1.0f0 / sqrt(sq + eps) : 1.0f0
    ik = l2norm ? 1.0f0 / sqrt(sk + eps) : 1.0f0
    for d in 1:dk
        qn[t, hh, d] = q[t, h, d] * iq * scale
        kn[t, hh, d] = k[t, h, d] * ik
    end
end

# Recurrent delta rule. One thread per (value head, value column j) owns
# state[hh, :, j] exclusively, so there are no cross-thread writes. Time is
# sequential inside the thread, as the recurrence requires.
@kernel cpu=false function _lava_gdn_recur_kernel!(
    out, qn, kn, v, g, beta, state, nv::Int, dk::Int, dv::Int, seq::Int, n::Int,
)
    i = @index(Global)
    i <= n || return
    hh = (i - 1) % nv + 1
    j = (i - 1) ÷ nv + 1
    for t in 1:seq
        decay = exp(g[t, hh])
        for ii in 1:dk
            state[hh, ii, j] *= decay
        end
        bt = beta[t, hh]
        kv = 0.0f0
        for ii in 1:dk
            kv += state[hh, ii, j] * kn[t, hh, ii]
        end
        delta = (v[t, hh, j] - kv) * bt
        for ii in 1:dk
            state[hh, ii, j] += kn[t, hh, ii] * delta
        end
        acc = 0.0f0
        for ii in 1:dk
            acc += state[hh, ii, j] * qn[t, hh, ii]
        end
        out[t, hh, j] = acc
    end
end

function Gesso.Inference._causal_conv1d_silu!(::LavaBackend, dst, x, weight, conv_state)
    seq, channels = size(x)
    k = size(weight, 3)
    size(weight, 1) == channels ||
        error("causal_conv1d: weight channels $(size(weight, 1)) ≠ $channels")
    size(conv_state) == (channels, k - 1) || error(
        "causal_conv1d: conv_state must be ($channels, $(k - 1)), got $(size(conv_state))",
    )
    d = _lh_device(dst, :causal_conv1d)
    xs = _lh_device(x, :causal_conv1d)
    ws = _lh_device(weight, :causal_conv1d)
    st = _lh_device(conv_state, :causal_conv1d)
    seq == 0 && return dst
    n_out = seq * channels
    kout = _lava_conv_silu_kernel!(KA.get_backend(xs))
    kout(d, xs, ws, st, Int(channels), Int(k), Int(n_out); ndrange=n_out, workgroupsize=_LH_WG)
    # The output kernel has read the old state, so the update runs after it on
    # the same queue. A copy is needed only when the shift reads the old state.
    n_st = (k - 1) * channels
    old = if seq < k - 1
        o2 = Lava.LavaArray{Float32}(undef, size(st))
        copyto!(o2, st)
        o2
    else
        st
    end
    kst = _lava_conv_state_kernel!(KA.get_backend(xs))
    kst(st, old, xs, Int(seq), Int(channels), Int(k), Int(n_st); ndrange=n_st, workgroupsize=_LH_WG)
    return dst
end

function Gesso.Inference._gdn_gate_params!(::LavaBackend, g, beta, a, b, A_log, dt_bias)
    seq, nv = size(beta)
    gd = _lh_device(g, :gdn_gate_params)
    bd = _lh_device(beta, :gdn_gate_params)
    ad = _lh_device(a, :gdn_gate_params)
    bb = _lh_device(b, :gdn_gate_params)
    al = _lh_device(A_log, :gdn_gate_params)
    dt = _lh_device(dt_bias, :gdn_gate_params)
    n = seq * nv
    n == 0 && return g, beta
    kern = _lava_gdn_gates_kernel!(KA.get_backend(ad))
    kern(gd, bd, ad, bb, al, dt, Int(nv), Int(n); ndrange=n, workgroupsize=_LH_WG)
    return g, beta
end

function Gesso.Inference._gated_delta_rule!(
    ::LavaBackend,
    out,
    q,
    k,
    v,
    g,
    beta,
    state;
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
    o = _lh_device(out, :gated_delta_rule)
    qs = _lh_device(q, :gated_delta_rule)
    ks = _lh_device(k, :gated_delta_rule)
    vs = _lh_device(v, :gated_delta_rule)
    gs = _lh_device(g, :gated_delta_rule)
    bs = _lh_device(beta, :gated_delta_rule)
    st = _lh_device(state, :gated_delta_rule)
    seq == 0 && return out
    qn = Lava.LavaArray{Float32}(undef, (seq, nv, dk))
    kn = Lava.LavaArray{Float32}(undef, (seq, nv, dk))
    scale = 1.0f0 / sqrt(Float32(dk))
    n_qk = seq * nv
    kqk = _lava_gdn_qk_kernel!(KA.get_backend(qs))
    kqk(qn, kn, qs, ks, Int(nk), Int(group), v_head_tiled, l2norm, scale, 1.0f-6,
        Int(nv), Int(dk), Int(n_qk); ndrange=n_qk, workgroupsize=_LH_WG)
    n_rec = nv * dv
    krec = _lava_gdn_recur_kernel!(KA.get_backend(qs))
    krec(o, qn, kn, vs, gs, bs, st, Int(nv), Int(dk), Int(dv), Int(seq), Int(n_rec);
        ndrange=n_rec, workgroupsize=_LH_WG)
    return out
end

function Gesso.Inference._sigmoid_mul!(::LavaBackend, dst, src, gate)
    d = _lh_device(dst, :sigmoid_mul)
    s = _lh_device(src, :sigmoid_mul)
    gt = _lh_device(gate, :sigmoid_mul)
    d .= s .* (1.0f0 ./ (1.0f0 .+ exp.(-gt)))
    return dst
end

function Gesso.Inference._silu_mul!(::LavaBackend, dst, src, gate)
    d = _lh_device(dst, :silu_mul)
    s = _lh_device(src, :silu_mul)
    gt = _lh_device(gate, :silu_mul)
    d .= s .* (gt ./ (1.0f0 .+ exp.(-gt)))
    return dst
end

# Routed experts on the host (L2). The expert banks stay GgufQuantSlice (or
# host Array) and are never uploaded; x and the router come down, the CPU
# route runs, and y goes back up. Device expert routing is L3.
#
# GESSO_MOE_HOST_EXPERTS=0 means "upload the experts" on CUDA. There is no
# device expert path on Lava yet, so 0 is refused rather than ignored.
function Gesso.Inference._moe_topk_swiglu!(
    ::LavaBackend,
    y,
    x,
    router,
    gate,
    up,
    down,
    n_used::Int;
    overlap=nothing,
)
    get(ENV, "GESSO_MOE_HOST_EXPERTS", "1") == "1" || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "GESSO_MOE_HOST_EXPERTS=0 has no Lava path yet (device expert routing is L3); " *
            "leave it unset on Lava",
        ),
    )
    for (nm, w) in (("gate", gate), ("up", up), ("down", down))
        w isa Lava.AnyLavaArray && throw(
            gesso_error(
                ERR_INVALID_PLAN,
                "_moe_topk_swiglu!: expert bank $nm is a device array; the Lava hybrid " *
                "keeps expert banks on the host",
            ),
        )
    end
    overlap === nothing || overlap()
    xh = _lh_host(x)
    rh = _lh_host(router)
    yh = zeros(Float32, size(y))
    Gesso.Inference._moe_topk_swiglu!(Gesso.CPUBackend(), yh, xh, rh, gate, up, down, n_used)
    copyto!(y, yh)
    return y
end
