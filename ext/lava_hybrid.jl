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
# Not here yet: device expert routing (L3).

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

# Host copies of frozen device weights the host MoE route reads every token.
# The router alone is 2 MB per layer, ~86 ms/token on the GSQ 35B when it was
# downloaded per call. Keyed by identity: the WeakRef check rejects a recycled
# objectid, and dead entries are pruned on the next miss.
const _LH_HOST_WEIGHTS = Dict{UInt, Tuple{WeakRef, Array}}()
const _LH_HOST_WEIGHTS_LOCK = ReentrantLock()

function _lh_host_weight(w::Lava.LavaArray)
    k = objectid(w)
    lock(_LH_HOST_WEIGHTS_LOCK) do
        e = get(_LH_HOST_WEIGHTS, k, nothing)
        e !== nothing && e[1].value === w && return e[2]
        filter!(p -> p.second[1].value !== nothing, _LH_HOST_WEIGHTS)
        h = Array(w)
        _LH_HOST_WEIGHTS[k] = (WeakRef(w), h)
        return h
    end
end
_lh_host_weight(w) = w

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

const _LH_RED_LANES = 64

# Gated RMS norm, SiLU gate and head merge in one launch, one workgroup per
# (token, value head). The engine's unfused fallback issues about five ops per
# head, roughly 160 launches per GDN layer. Same arithmetic as the CUDA kernel.
@kernel cpu=false function _lava_gdn_gnorm_kernel!(merged, core, z, norm_w, nv::Int, dv::Int, eps::Float32)
    tl = @index(Local, Linear)
    gi = @index(Group, Linear)
    part = @localmem Float32 (_LH_RED_LANES,)
    hh = (gi - 1) % nv + 1
    t = (gi - 1) ÷ nv + 1
    acc = 0.0f0
    for j in tl:_LH_RED_LANES:dv
        c = core[t, hh, j]
        acc += c * c
    end
    part[tl] = acc
    # every lane reads all partials after this barrier
    @synchronize
    ss = 0.0f0
    for i in 1:_LH_RED_LANES
        ss += part[i]
    end
    inv = 1.0f0 / sqrt(ss / Float32(dv) + eps)
    base = (hh - 1) * dv
    for j in tl:_LH_RED_LANES:dv
        zv = z[t, base + j]
        merged[t, base + j] = (core[t, hh, j] * inv) * norm_w[j] * (zv / (1.0f0 + exp(-zv)))
    end
end

function Gesso.Inference._gdn_gnorm_merge!(::LavaBackend, merged::Lava.LavaArray{Float32, 2},
                                           core::Lava.LavaArray{Float32, 3},
                                           z::Lava.LavaArray{Float32, 2}, scale, eps)
    seq, nv, dv = size(core)
    seq == 0 && return true
    w = _lh_device(scale.storage, :gdn_gnorm_merge)
    kern = _lava_gdn_gnorm_kernel!(KA.get_backend(core))
    kern(merged, core, z, w, Int(nv), Int(dv), Float32(eps);
        ndrange=seq * nv * _LH_RED_LANES, workgroupsize=(_LH_RED_LANES,))
    return true
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

# --- Qwen3.5 GDN single-token decode (seq = 1), fused ---------------------------
#
# Port of the CUDA fused decode (cuda_ops.jl). Three launches replace the
# unfused block's dozen allocations and ~15 launches per layer: conv + SiLU +
# state shift, the delta rule with gates and q/k norms folded in, and the gated
# RMSNorm. Same layout contract: conv_out holds q, k, v head-major, and core and
# merged are head-major with index (hh-1)*dv + j.

Gesso.Inference._gdn_fused_decode(::LavaBackend) = true

# Thread c owns channel c alone, so its state column is read before it is
# overwritten and no other thread touches it.
@kernel cpu=false function _lava_gdn_conv_decode_kernel!(conv_out, state, mixed, weight, k::Int, channels::Int)
    c = @index(Global)
    c <= channels || return
    acc = 0.0f0
    for p in 1:(k - 1)
        acc += weight[c, 1, p] * state[c, p]
    end
    xv = mixed[1, c]
    acc += weight[c, 1, k] * xv
    conv_out[1, c] = acc / (1.0f0 + exp(-acc))
    for p in 1:(k - 2)
        state[c, p] = state[c, p + 1]
    end
    state[c, k - 1] = xv
end

# Lava workgroups are 1-D and local memory is sized at compile time, so the
# CUDA (nv, Y) block becomes a fixed 256-lane group with lane = hh + (y-1)*nv.
# Lanes past nv*Y idle but still reach every barrier.
const _LH_DELTA_WG = 256
const _LH_DELTA_RED = 3 * _LH_DELTA_WG

# Group j is state column j of every head; lane (hh, y) takes the state rows
# i ≡ y (mod Y) of head hh. Consecutive lanes are consecutive heads, which are
# adjacent in the [nv, dk, dv] state. The q and k norms scale the sums rather
# than every element: the same math to rounding.
@kernel cpu=false function _lava_gdn_delta_kernel!(
    core, conv_out, a, b, A_log, dt_bias, state,
    nv::Int, nk::Int, group::Int, tiled::Bool, dk::Int, dv::Int, Y::Int,
    scale::Float32, eps::Float32,
)
    tl = @index(Local, Linear)
    j = @index(Group, Linear)
    red = @localmem Float32 (_LH_DELTA_RED,)
    active = tl <= nv * Y
    hh = (tl - 1) % nv + 1
    y = (tl - 1) ÷ nv + 1
    key_dim = nk * dk
    h = tiled ? (hh - 1) % nk + 1 : (hh - 1) ÷ group + 1
    qoff = (h - 1) * dk
    koff = key_dim + qoff
    beta = 0.0f0
    decay = 0.0f0
    sq = 0.0f0
    sk = 0.0f0
    sm = 0.0f0
    if active
        beta = 1.0f0 / (1.0f0 + exp(-b[1, hh]))
        av = a[1, hh] + dt_bias[hh]
        sp = av > 20.0f0 ? av : log(1.0f0 + exp(av))
        decay = exp(-exp(A_log[hh]) * sp)
        for i in y:Y:dk
            qv = conv_out[1, qoff + i]
            kv = conv_out[1, koff + i]
            sq += qv * qv
            sk += kv * kv
            sm += (state[hh, i, j] * decay) * kv
        end
    end
    red[tl] = sq
    red[_LH_DELTA_WG + tl] = sk
    red[2 * _LH_DELTA_WG + tl] = sm
    @synchronize
    acc = 0.0f0
    invq = 0.0f0
    if active
        sq = 0.0f0
        sk = 0.0f0
        sm = 0.0f0
        for t in 0:(Y - 1)
            r = hh + t * nv
            sq += red[r]
            sk += red[_LH_DELTA_WG + r]
            sm += red[2 * _LH_DELTA_WG + r]
        end
        invq = scale / sqrt(sq + eps)
        invk = 1.0f0 / sqrt(sk + eps)
        delta = (conv_out[1, 2 * key_dim + (hh - 1) * dv + j] - sm * invk) * beta
        kd = invk * delta
        for i in y:Y:dk
            sn = state[hh, i, j] * decay + conv_out[1, koff + i] * kd
            state[hh, i, j] = sn
            acc += sn * conv_out[1, qoff + i]
        end
    end
    # every lane must be done reading the first reduction before it is reused
    @synchronize
    red[tl] = acc
    @synchronize
    if active && y == 1
        tot = 0.0f0
        for t in 0:(Y - 1)
            tot += red[hh + t * nv]
        end
        core[1, (hh - 1) * dv + j] = tot * invq
    end
end

# Decode form of _lava_gdn_gnorm_kernel!: one group per value head over the
# head-major core the delta kernel writes.
@kernel cpu=false function _lava_gdn_gnorm_decode_kernel!(merged, core, z, norm_w, dv::Int, eps::Float32)
    tl = @index(Local, Linear)
    hh = @index(Group, Linear)
    part = @localmem Float32 (_LH_RED_LANES,)
    base = (hh - 1) * dv
    acc = 0.0f0
    for j in tl:_LH_RED_LANES:dv
        c = core[1, base + j]
        acc += c * c
    end
    part[tl] = acc
    @synchronize
    ss = 0.0f0
    for i in 1:_LH_RED_LANES
        ss += part[i]
    end
    inv = 1.0f0 / sqrt(ss / Float32(dv) + eps)
    for j in tl:_LH_RED_LANES:dv
        zv = z[1, base + j]
        merged[1, base + j] = (core[1, base + j] * inv) * norm_w[j] * (zv / (1.0f0 + exp(-zv)))
    end
end

function Gesso.Inference._gdn_decode_core!(
    backend::LavaBackend, s, bt, spec, eps, conv_state, rec_state,
)
    mx = spec.mixer
    nk, nv = mx.num_key_heads, mx.num_value_heads
    dk, dv = mx.key_head_dim, mx.value_head_dim
    ksz = mx.conv_kernel_dim
    key_dim = nk * dk
    conv_dim = 2 * key_dim + nv * dv
    nv % nk == 0 || error("gdn decode: n_v_heads $nv not divisible by n_k_heads $nk")
    nv <= _LH_DELTA_WG || error("gdn decode: $nv value heads exceed the $_LH_DELTA_WG-lane group")
    size(rec_state) == (nv, dk, dv) ||
        error("gdn decode: state $(size(rec_state)) ≠ ($nv, $dk, $dv)")
    size(conv_state) == (conv_dim, ksz - 1) ||
        error("gdn decode: conv state $(size(conv_state)) ≠ ($conv_dim, $(ksz - 1))")
    weight = _lh_device(bt.wgdn_conv.storage, :gdn_decode)
    A_log = _lh_device(bt.wgdn_A_log.storage, :gdn_decode)
    dt_bias = _lh_device(bt.wgdn_dt_bias.storage, :gdn_decode)
    norm_w = _lh_device(bt.wgdn_norm.storage, :gdn_decode)
    cst = _lh_device(conv_state, :gdn_decode)
    rst = _lh_device(rec_state, :gdn_decode)
    scale = 1.0f0 / sqrt(Float32(dk))
    eps32 = Float32(eps)
    Y = clamp(_LH_DELTA_WG ÷ nv, 1, dk)
    kb = KA.get_backend(s.conv)
    Gesso.Inference.@decode_sub :gdn_conv backend begin
        _lava_gdn_conv_decode_kernel!(kb)(s.conv, cst, s.mixed, weight, Int(ksz), Int(conv_dim);
            ndrange=conv_dim, workgroupsize=_LH_WG)
    end
    Gesso.Inference.@decode_sub :gdn_recur backend begin
        _lava_gdn_delta_kernel!(kb)(s.core, s.conv, s.a, s.b, A_log, dt_bias, rst,
            Int(nv), Int(nk), Int(div(nv, nk)), mx.v_head_tiled, Int(dk), Int(dv), Int(Y),
            scale, eps32; ndrange=dv * _LH_DELTA_WG, workgroupsize=(_LH_DELTA_WG,))
    end
    Gesso.Inference.@decode_sub :gdn_gnorm backend begin
        _lava_gdn_gnorm_decode_kernel!(kb)(s.merged, s.core, s.z, norm_w, Int(dv), eps32;
            ndrange=nv * _LH_RED_LANES, workgroupsize=(_LH_RED_LANES,))
    end
    return s
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
    backend::LavaBackend,
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
    # Lava only records dispatches until something flushes, so the shared
    # expert is submitted after x is down and runs on the GPU while the CPU runs
    # the routed experts. It writes neither y nor anything they read.
    # GESSO_MOE_OVERLAP=0 runs it before the download instead, as CUDA does.
    ov = overlap !== nothing && get(ENV, "GESSO_MOE_OVERLAP", "1") == "1"
    ov || overlap === nothing || overlap()
    xh = Gesso.Inference.@decode_stage :moe_d2h backend _lh_host(x)
    if ov
        overlap()
        Lava.submit!(KALava().dispatch_bq)
    end
    rh = _lh_host_weight(router)
    yh = zeros(Float32, size(y))
    Gesso.Inference.@decode_stage :moe_host nothing Gesso.Inference._moe_topk_swiglu!(
        Gesso.CPUBackend(), yh, xh, rh, gate, up, down, n_used)
    Gesso.Inference.@decode_stage :moe_h2d backend copyto!(y, yh)
    return y
end

# Shared-expert sigmoid gate, y += sigmoid(x . g) * sd, one workgroup per
# token. The host version paid four transfers per layer.
@kernel cpu=false function _lava_shared_gate_kernel!(y, x, sd, g, h::Int)
    tl = @index(Local, Linear)
    t = @index(Group, Linear)
    part = @localmem Float32 (_LH_RED_LANES,)
    acc = 0.0f0
    for i in tl:_LH_RED_LANES:h
        acc += x[t, i] * g[i]
    end
    part[tl] = acc
    # every lane reads all partials after this barrier
    @synchronize
    s = 0.0f0
    for i in 1:_LH_RED_LANES
        s += part[i]
    end
    s = 1.0f0 / (1.0f0 + exp(-s))
    for i in tl:_LH_RED_LANES:h
        y[t, i] += s * sd[t, i]
    end
end

function Gesso.Inference._shared_gate_add!(::LavaBackend, y, x, sd, gvec)
    yd = _lh_device(y, :shared_gate_add)
    xd = _lh_device(x, :shared_gate_add)
    sdd = _lh_device(sd, :shared_gate_add)
    gd = _lh_device(gvec, :shared_gate_add)
    seq, h = size(xd)
    seq == 0 && return y
    kern = _lava_shared_gate_kernel!(KA.get_backend(xd))
    kern(yd, xd, sdd, gd, Int(h); ndrange=seq * _LH_RED_LANES, workgroupsize=(_LH_RED_LANES,))
    return y
end
