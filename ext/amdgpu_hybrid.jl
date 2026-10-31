# Qwen3.5 hybrid seams on KADeviceBackend: the gated-delta-net mixer (conv,
# gates, q/k norms, recurrence, gated RMSNorm), the fused single-token decode,
# the sigmoid and SiLU gates, and the shared-expert gate. Ports
# ext/lava_hybrid.jl. Storage is Float32; every reduction accumulates in
# Float64, as the CPU oracle in src/Operators/cpu_linear.jl does.
#
# Expert routing stays on the host, as on Lava. Device expert routing is not
# ported.

@inline _amdh_sigmoid(x::Float64) = 1.0 / (1.0 + exp(-x))
@inline _amdh_silu(x::Float64) = x / (1.0 + exp(-x))
@inline _amdh_softplus(x::Float64) = x > 20.0 ? x : log1p(exp(x))

# Per-head gate terms of the decode step. The delta kernel recomputes them
# after each barrier rather than carrying them across it.
@inline _amdh_gdn_beta(b, hh::Int) = _amdh_sigmoid(Float64(b[1, hh]))
@inline function _amdh_gdn_decay(a, dt_bias, A_log, hh::Int)
    x = Float64(a[1, hh]) + Float64(dt_bias[hh])
    return exp(-exp(Float64(A_log[hh])) * _amdh_softplus(x))
end

function _amdh_dev(bk::KADeviceBackend, x, op::Symbol)
    _ka_on_device(bk, x) && eltype(x) === Float32 && return x
    throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "$op: $(backend_name(bk)) backend requires Float32 storage on its own device; " *
            "call to_device first (host storage is never copied implicitly)",
        ),
    )
end

# --- causal depthwise conv1d + SiLU --------------------------------------------

# One thread per (t, channel). Reads only the old state and x, so the state
# update is a separate launch after this one.
@kernel function _ka_gdn_conv_silu_kernel!(
    dst,
    @Const(x),
    @Const(weight),
    @Const(state),
    channels::Int,
    k::Int,
    n::Int,
)
    i = @index(Global, Linear)
    if i <= n
        t = (i - 1) ÷ channels + 1
        c = (i - 1) % channels + 1
        acc = 0.0
        for p in 1:k
            src = t - (k - p)
            v = src <= 0 ? Float64(state[c, src+(k-1)]) : Float64(x[src, c])
            acc += Float64(weight[c, 1, p]) * v
        end
        dst[t, c] = Float32(_amdh_silu(acc))
    end
end

# Shifts the last k-1 inputs into the state. `old` must hold the pre-update
# state whenever seq < k-1, because the shift then reads it at a lower index.
@kernel function _ka_gdn_conv_state_kernel!(
    state,
    @Const(old),
    @Const(x),
    seq::Int,
    channels::Int,
    k::Int,
    n::Int,
)
    i = @index(Global, Linear)
    if i <= n
        c = (i - 1) % channels + 1
        p = (i - 1) ÷ channels + 1
        if seq >= k - 1
            state[c, p] = x[seq-(k-1)+p, c]
        else
            keep = (k - 1) - seq
            state[c, p] = p <= keep ? old[c, p+seq] : x[p-keep, c]
        end
    end
end

function _ka_causal_conv1d_silu!(ka, dst, x, weight, state; wg::Int=_KA_OP_WG)
    _ka_check_wg(wg)
    seq, channels = size(x)
    k = size(weight, 3)
    size(weight, 1) == channels ||
        error("causal_conv1d: weight channels $(size(weight, 1)) ≠ $channels")
    size(state) == (channels, k - 1) || error(
        "causal_conv1d: conv_state must be ($channels, $(k - 1)), got $(size(state))",
    )
    seq == 0 && return dst
    n_out = seq * channels
    _ka_gdn_conv_silu_kernel!(ka, wg)(
        dst, x, weight, state, channels, k, n_out; ndrange=n_out
    )
    n_st = (k - 1) * channels
    if n_st > 0
        old = seq < k - 1 ? copy(state) : state
        _ka_gdn_conv_state_kernel!(ka, wg)(
            state, old, x, seq, channels, k, n_st; ndrange=n_st
        )
    end
    return dst
end

# Decode: one thread per channel. Each thread owns its channel's state column,
# so it reads the window, writes the output and shifts the state in order.
@kernel function _ka_gdn_conv_decode_kernel!(
    conv_out,
    state,
    @Const(mixed),
    @Const(weight),
    k::Int,
    channels::Int,
)
    c = @index(Global, Linear)
    if c <= channels
        acc = 0.0
        for p in 1:(k-1)
            acc += Float64(weight[c, 1, p]) * Float64(state[c, p])
        end
        xv = mixed[1, c]
        acc += Float64(weight[c, 1, k]) * Float64(xv)
        conv_out[1, c] = Float32(_amdh_silu(acc))
        if k > 1
            for p in 1:(k-2)
                state[c, p] = state[c, p+1]
            end
            state[c, k-1] = xv
        end
    end
end

function _ka_gdn_conv_decode!(ka, conv_out, state, mixed, weight, k::Int, channels::Int; wg::Int=_KA_OP_WG)
    _ka_check_wg(wg)
    _ka_gdn_conv_decode_kernel!(ka, wg)(
        conv_out, state, mixed, weight, k, channels; ndrange=channels
    )
    return conv_out
end

# --- gates ----------------------------------------------------------------------

# beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias), one thread per
# (t, head).
@kernel function _ka_gdn_gates_kernel!(
    g,
    beta,
    @Const(a),
    @Const(b),
    @Const(A_log),
    @Const(dt_bias),
    nv::Int,
    n::Int,
)
    i = @index(Global, Linear)
    if i <= n
        hh = (i - 1) % nv + 1
        t = (i - 1) ÷ nv + 1
        beta[t, hh] = Float32(_amdh_sigmoid(Float64(b[t, hh])))
        x = Float64(a[t, hh]) + Float64(dt_bias[hh])
        g[t, hh] = Float32(-exp(Float64(A_log[hh])) * _amdh_softplus(x))
    end
end

function _ka_gdn_gate_params!(ka, g, beta, a, b, A_log, dt_bias; wg::Int=_KA_OP_WG)
    _ka_check_wg(wg)
    seq, nv = size(beta)
    n = seq * nv
    n == 0 && return g, beta
    _ka_gdn_gates_kernel!(ka, wg)(g, beta, a, b, A_log, dt_bias, nv, n; ndrange=n)
    return g, beta
end

# --- recurrence -----------------------------------------------------------------

# Q/K heads map to value heads as the CPU oracle does: `tiled` cycles the K
# heads, otherwise each K head covers a contiguous group of V heads. The norms
# are per (t, head) with the same 1e-6 epsilon, and q also carries 1/sqrt(dk).
@kernel function _ka_gdn_qk_kernel!(
    qn,
    kn,
    @Const(q),
    @Const(k),
    nk::Int,
    nv::Int,
    group::Int,
    tiled::Bool,
    dk::Int,
    scale::Float64,
    eps::Float64,
    l2norm::Bool,
    n::Int,
)
    i = @index(Global, Linear)
    if i <= n
        hh = (i - 1) % nv + 1
        t = (i - 1) ÷ nv + 1
        h = tiled ? (hh - 1) % nk + 1 : (hh - 1) ÷ group + 1
        sq = 0.0
        sk = 0.0
        for d in 1:dk
            sq += abs2(Float64(q[t, h, d]))
            sk += abs2(Float64(k[t, h, d]))
        end
        iq = l2norm ? 1.0 / sqrt(sq + eps) : 1.0
        ik = l2norm ? 1.0 / sqrt(sk + eps) : 1.0
        for d in 1:dk
            qn[t, hh, d] = Float32(Float64(q[t, h, d]) * iq * scale)
            kn[t, hh, d] = Float32(Float64(k[t, h, d]) * ik)
        end
    end
end

# One thread per (head, value column). The thread owns its state column, so the
# sequential sweep over t needs no barrier. The state is rounded to Float32
# after every write, as the CPU oracle stores it.
@kernel function _ka_gdn_recur_kernel!(
    out,
    state,
    @Const(qn),
    @Const(kn),
    @Const(v),
    @Const(g),
    @Const(beta),
    nv::Int,
    dk::Int,
    dv::Int,
    seq::Int,
    n::Int,
)
    i = @index(Global, Linear)
    if i <= n
        hh = (i - 1) % nv + 1
        j = (i - 1) ÷ nv + 1
        for t in 1:seq
            decay = exp(Float64(g[t, hh]))
            for d in 1:dk
                state[hh, d, j] = Float32(Float64(state[hh, d, j]) * decay)
            end
            kv = 0.0
            for d in 1:dk
                kv += Float64(state[hh, d, j]) * Float64(kn[t, hh, d])
            end
            delta = (Float64(v[t, hh, j]) - kv) * Float64(beta[t, hh])
            for d in 1:dk
                state[hh, d, j] = Float32(
                    Float64(state[hh, d, j]) + Float64(kn[t, hh, d]) * delta
                )
            end
            acc = 0.0
            for d in 1:dk
                acc += Float64(state[hh, d, j]) * Float64(qn[t, hh, d])
            end
            out[t, hh, j] = Float32(acc)
        end
    end
end

function _ka_gdn_recurrence!(
    ka, out, state, q, k, v, g, beta; l2norm::Bool, v_head_tiled::Bool, wg::Int=_KA_OP_WG
)
    _ka_check_wg(wg)
    seq = size(q, 1)
    nk, dk = size(q, 2), size(q, 3)
    nv, dv = size(v, 2), size(v, 3)
    nv % nk == 0 || error("gated_delta_rule: n_v_heads $nv not divisible by n_k_heads $nk")
    size(state) == (nv, dk, dv) ||
        error("gated_delta_rule: state $(size(state)) ≠ ($nv, $dk, $dv)")
    seq == 0 && return out
    group = div(nv, nk)
    qn = similar(state, Float32, seq, nv, dk)
    kn = similar(state, Float32, seq, nv, dk)
    scale = 1.0 / sqrt(Float64(dk))
    n_qk = seq * nv
    _ka_gdn_qk_kernel!(ka, wg)(
        qn, kn, q, k, nk, nv, group, v_head_tiled, dk, scale, 1.0e-6, l2norm, n_qk;
        ndrange=n_qk,
    )
    n_rec = nv * dv
    _ka_gdn_recur_kernel!(ka, wg)(
        out, state, qn, kn, v, g, beta, nv, dk, dv, seq, n_rec; ndrange=n_rec
    )
    return out
end

# --- gated RMSNorm -------------------------------------------------------------

# Prefill: one workgroup per (t, head), the core row read as core[t, hh, j].
# The reduction is a tree in local memory, as in _ka_rmsnorm_rows_kernel!.
@kernel function _ka_gdn_gnorm_kernel!(
    merged,
    @Const(core),
    @Const(z),
    @Const(norm_w),
    nv::Int,
    dv::Int,
    eps::Float64,
)
    wg = @uniform @groupsize()[1]
    part = @localmem Float64 (_KA_MAX_WG,)
    g = @index(Group, Linear)
    lane = @index(Local, Linear)
    t = (g - 1) ÷ nv + 1
    hh = (g - 1) % nv + 1
    acc = 0.0
    j = lane
    while j <= dv
        cv = Float64(core[t, hh, j])
        acc += cv * cv
        j += wg
    end
    part[lane] = acc
    @synchronize
    steps = @uniform trailing_zeros(wg)
    for k in 1:steps
        lane = @index(Local, Linear)
        h = wg >> k
        if lane <= h
            part[lane] += part[lane+h]
        end
        @synchronize
    end
    g = @index(Group, Linear)
    lane = @index(Local, Linear)
    t = (g - 1) ÷ nv + 1
    hh = (g - 1) % nv + 1
    invr = 1.0 / sqrt(part[1] / dv + eps)
    base = (hh - 1) * dv
    j = lane
    while j <= dv
        zv = Float64(z[t, base+j])
        merged[t, base+j] = Float32(
            Float64(core[t, hh, j]) * invr * Float64(norm_w[j]) * _amdh_silu(zv)
        )
        j += wg
    end
end

# Decode: one workgroup per head over the head-major row core[1, (hh-1)*dv+j].
@kernel function _ka_gdn_gnorm_decode_kernel!(
    merged,
    @Const(core),
    @Const(z),
    @Const(norm_w),
    dv::Int,
    eps::Float64,
)
    wg = @uniform @groupsize()[1]
    part = @localmem Float64 (_KA_MAX_WG,)
    hh = @index(Group, Linear)
    lane = @index(Local, Linear)
    acc = 0.0
    j = lane
    while j <= dv
        cv = Float64(core[1, (hh-1)*dv+j])
        acc += cv * cv
        j += wg
    end
    part[lane] = acc
    @synchronize
    steps = @uniform trailing_zeros(wg)
    for k in 1:steps
        lane = @index(Local, Linear)
        h = wg >> k
        if lane <= h
            part[lane] += part[lane+h]
        end
        @synchronize
    end
    hh = @index(Group, Linear)
    lane = @index(Local, Linear)
    invr = 1.0 / sqrt(part[1] / dv + eps)
    base = (hh - 1) * dv
    j = lane
    while j <= dv
        zv = Float64(z[1, base+j])
        merged[1, base+j] = Float32(
            Float64(core[1, base+j]) * invr * Float64(norm_w[j]) * _amdh_silu(zv)
        )
        j += wg
    end
end

function _ka_gdn_gnorm!(ka, merged, core, z, norm_w, eps; wg::Int=_KA_OP_WG)
    _ka_check_wg(wg)
    seq, nv, dv = size(core)
    seq == 0 && return merged
    n_groups = seq * nv
    _ka_gdn_gnorm_kernel!(ka, wg)(
        merged, core, z, norm_w, nv, dv, Float64(eps); ndrange=n_groups * wg
    )
    return merged
end

function _ka_gdn_gnorm_decode!(ka, merged, core, z, norm_w, nv::Int, dv::Int, eps; wg::Int=_KA_OP_WG)
    _ka_check_wg(wg)
    _ka_gdn_gnorm_decode_kernel!(ka, wg)(
        merged, core, z, norm_w, dv, Float64(eps); ndrange=nv * wg
    )
    return merged
end

# --- fused single-token delta rule ---------------------------------------------

# One workgroup per value column j. Lane tl = (y-1)*nv + hh: lanes of one head
# split the key dimension in strides of Y, and the Y partial sums fold through
# local memory. Y is a power of two, so the fold is a plain halving tree. The
# output is written only by the y == 1 lane of each head, after the second fold.
@kernel function _ka_gdn_delta_kernel!(
    core,
    @Const(conv),
    @Const(a),
    @Const(b),
    @Const(A_log),
    @Const(dt_bias),
    state,
    nv::Int,
    nk::Int,
    group::Int,
    tiled::Bool,
    dk::Int,
    dv::Int,
    Y::Int,
    scale::Float64,
    eps::Float64,
)
    rsq = @localmem Float64 (_KA_MAX_WG,)
    rsk = @localmem Float64 (_KA_MAX_WG,)
    rsm = @localmem Float64 (_KA_MAX_WG,)
    racc = @localmem Float64 (_KA_MAX_WG,)
    j = @index(Group, Linear)
    tl = @index(Local, Linear)
    key_dim = nk * dk
    lanes = nv * Y
    hh = (tl - 1) % nv + 1
    y = (tl - 1) ÷ nv + 1
    h = tiled ? (hh - 1) % nk + 1 : (hh - 1) ÷ group + 1
    qoff = (h - 1) * dk
    koff = key_dim + qoff
    sq = 0.0
    sk = 0.0
    sm = 0.0
    if tl <= lanes
        decay = _amdh_gdn_decay(a, dt_bias, A_log, hh)
        i = y
        while i <= dk
            qv = Float64(conv[1, qoff+i])
            kv = Float64(conv[1, koff+i])
            sv = Float64(state[hh, i, j])
            sq += qv * qv
            sk += kv * kv
            sm += sv * decay * kv
            i += Y
        end
    end
    rsq[tl] = sq
    rsk[tl] = sk
    rsm[tl] = sm
    @synchronize
    steps = @uniform trailing_zeros(Y)
    for k in 1:steps
        tl = @index(Local, Linear)
        lanes = nv * Y
        yk = (tl - 1) ÷ nv + 1
        hs = Y >> k
        if tl <= lanes && yk <= hs
            rsq[tl] += rsq[tl+hs*nv]
            rsk[tl] += rsk[tl+hs*nv]
            rsm[tl] += rsm[tl+hs*nv]
        end
        @synchronize
    end
    tl = @index(Local, Linear)
    j = @index(Group, Linear)
    lanes = nv * Y
    key_dim = nk * dk
    hh = (tl - 1) % nv + 1
    y = (tl - 1) ÷ nv + 1
    h = tiled ? (hh - 1) % nk + 1 : (hh - 1) ÷ group + 1
    qoff = (h - 1) * dk
    koff = key_dim + qoff
    acc = 0.0
    if tl <= lanes
        decay = _amdh_gdn_decay(a, dt_bias, A_log, hh)
        beta = _amdh_gdn_beta(b, hh)
        invk = 1.0 / sqrt(rsk[hh] + eps)
        delta = (Float64(conv[1, 2*key_dim+(hh-1)*dv+j]) - rsm[hh] * invk) * beta
        kd = invk * delta
        i = y
        while i <= dk
            s32 = Float32(Float64(state[hh, i, j]) * decay + Float64(conv[1, koff+i]) * kd)
            state[hh, i, j] = s32
            acc += Float64(s32) * Float64(conv[1, qoff+i])
            i += Y
        end
    end
    racc[tl] = acc
    @synchronize
    steps = @uniform trailing_zeros(Y)
    for k in 1:steps
        tl = @index(Local, Linear)
        lanes = nv * Y
        yk = (tl - 1) ÷ nv + 1
        hs = Y >> k
        if tl <= lanes && yk <= hs
            racc[tl] += racc[tl+hs*nv]
        end
        @synchronize
    end
    tl = @index(Local, Linear)
    j = @index(Group, Linear)
    lanes = nv * Y
    hh = (tl - 1) % nv + 1
    y = (tl - 1) ÷ nv + 1
    if tl <= lanes && y == 1
        invq = scale / sqrt(rsq[hh] + eps)
        core[1, (hh-1)*dv+j] = Float32(racc[hh] * invq)
    end
end

function _ka_gdn_delta!(
    ka, core, conv, a, b, A_log, dt_bias, state, nv::Int, nk::Int, dk::Int, dv::Int,
    tiled::Bool, eps; wg::Int=_KA_MAX_WG,
)
    _ka_check_wg(wg)
    nv <= wg || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "gdn decode: $nv value heads exceed the $wg-lane workgroup",
        ),
    )
    Y = prevpow(2, clamp(div(wg, nv), 1, dk))
    group = div(nv, nk)
    scale = 1.0 / sqrt(Float64(dk))
    _ka_gdn_delta_kernel!(ka, wg)(
        core, conv, a, b, A_log, dt_bias, state,
        nv, nk, group, tiled, dk, dv, Y, scale, Float64(eps);
        ndrange=dv * wg,
    )
    return core
end

# --- elementwise gates and the shared-expert gate ------------------------------

@kernel function _ka_gate_sigmoid_mul_kernel!(dst, src, @Const(gate), n::Int)
    i = @index(Global, Linear)
    if i <= n
        dst[i] = src[i] * (1.0f0 / (1.0f0 + exp(-gate[i])))
    end
end

@kernel function _ka_gate_silu_mul_kernel!(dst, src, @Const(gate), n::Int)
    i = @index(Global, Linear)
    if i <= n
        gv = gate[i]
        dst[i] = src[i] * (gv / (1.0f0 + exp(-gv)))
    end
end

# One workgroup per row. The dot product folds through local memory, then each
# lane adds its share of the gated shared-expert row.
@kernel function _ka_shared_gate_kernel!(
    y,
    @Const(x),
    @Const(sd),
    @Const(gvec),
    h::Int,
)
    wg = @uniform @groupsize()[1]
    part = @localmem Float64 (_KA_MAX_WG,)
    row = @index(Group, Linear)
    lane = @index(Local, Linear)
    acc = 0.0
    j = lane
    while j <= h
        acc += Float64(x[row, j]) * Float64(gvec[j])
        j += wg
    end
    part[lane] = acc
    @synchronize
    steps = @uniform trailing_zeros(wg)
    for k in 1:steps
        lane = @index(Local, Linear)
        hs = wg >> k
        if lane <= hs
            part[lane] += part[lane+hs]
        end
        @synchronize
    end
    row = @index(Group, Linear)
    lane = @index(Local, Linear)
    s = _amdh_sigmoid(part[1])
    i = lane
    while i <= h
        y[row, i] = Float32(Float64(y[row, i]) + s * Float64(sd[row, i]))
        i += wg
    end
end

function _ka_elementwise_gate!(ka, kernel, dst, src, gate; wg::Int=_KA_OP_WG)
    _ka_check_wg(wg)
    n = length(dst)
    n == 0 && return dst
    kernel(ka, wg)(dst, src, gate, n; ndrange=n)
    return dst
end

function _ka_shared_gate!(ka, y, x, sd, gvec; wg::Int=_KA_OP_WG)
    _ka_check_wg(wg)
    seq, h = size(y)
    seq == 0 && return y
    _ka_shared_gate_kernel!(ka, wg)(y, x, sd, gvec, h; ndrange=seq * wg)
    return y
end

# --- host-routed experts -------------------------------------------------------

# Host copies of frozen device weights that the host MoE route reads every
# token. Keyed by identity: the WeakRef check rejects a recycled objectid, and
# dead entries are pruned on the next miss.
const _AMDH_HOST_WEIGHTS = Dict{UInt, Tuple{WeakRef, Array}}()
const _AMDH_HOST_WEIGHTS_LOCK = ReentrantLock()

function _amdh_host_weight(w::AbstractArray)
    KA.get_backend(w) isa KA.CPU && return w
    k = objectid(w)
    lock(_AMDH_HOST_WEIGHTS_LOCK) do
        e = get(_AMDH_HOST_WEIGHTS, k, nothing)
        e !== nothing && e[1].value === w && return e[2]
        filter!(p -> p.second[1].value !== nothing, _AMDH_HOST_WEIGHTS)
        h = Array(w)
        _AMDH_HOST_WEIGHTS[k] = (WeakRef(w), h)
        return h
    end
end
_amdh_host_weight(w) = w

# --- dispatch ------------------------------------------------------------------

Gesso.Inference._gdn_fused_decode(::KADeviceBackend) = true

function Gesso.Inference._causal_conv1d_silu!(bk::KADeviceBackend, dst, x, weight, conv_state)
    d = _amdh_dev(bk, dst, :causal_conv1d)
    xs = _amdh_dev(bk, x, :causal_conv1d)
    ws = _amdh_dev(bk, weight, :causal_conv1d)
    st = _amdh_dev(bk, conv_state, :causal_conv1d)
    return _ka_causal_conv1d_silu!(bk.ka, d, xs, ws, st)
end

function Gesso.Inference._gdn_gate_params!(bk::KADeviceBackend, g, beta, a, b, A_log, dt_bias)
    gd = _amdh_dev(bk, g, :gdn_gate_params)
    bd = _amdh_dev(bk, beta, :gdn_gate_params)
    ad = _amdh_dev(bk, a, :gdn_gate_params)
    bb = _amdh_dev(bk, b, :gdn_gate_params)
    al = _amdh_dev(bk, A_log, :gdn_gate_params)
    dt = _amdh_dev(bk, dt_bias, :gdn_gate_params)
    _ka_gdn_gate_params!(bk.ka, gd, bd, ad, bb, al, dt)
    return g, beta
end

function Gesso.Inference._gated_delta_rule!(
    bk::KADeviceBackend,
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
    o = _amdh_dev(bk, out, :gated_delta_rule)
    qs = _amdh_dev(bk, q, :gated_delta_rule)
    ks = _amdh_dev(bk, k, :gated_delta_rule)
    vs = _amdh_dev(bk, v, :gated_delta_rule)
    gs = _amdh_dev(bk, g, :gated_delta_rule)
    bs = _amdh_dev(bk, beta, :gated_delta_rule)
    st = _amdh_dev(bk, state, :gated_delta_rule)
    _ka_gdn_recurrence!(
        bk.ka, o, st, qs, ks, vs, gs, bs; l2norm, v_head_tiled
    )
    return out
end

function Gesso.Inference._gdn_gnorm_merge!(
    bk::KADeviceBackend,
    merged,
    core,
    z,
    scale,
    eps,
)
    seq = size(core, 1)
    seq == 0 && return true
    md = _amdh_dev(bk, merged, :gdn_gnorm_merge)
    cd = _amdh_dev(bk, core, :gdn_gnorm_merge)
    zd = _amdh_dev(bk, z, :gdn_gnorm_merge)
    w = _amdh_dev(bk, scale.storage, :gdn_gnorm_merge)
    _ka_gdn_gnorm!(bk.ka, md, cd, zd, w, eps)
    return true
end

function Gesso.Inference._gdn_decode_core!(
    bk::KADeviceBackend,
    s,
    bt,
    spec,
    eps,
    conv_state,
    rec_state,
)
    mx = spec.mixer
    nk, nv = mx.num_key_heads, mx.num_value_heads
    dk, dv = mx.key_head_dim, mx.value_head_dim
    ksz = mx.conv_kernel_dim
    key_dim = nk * dk
    conv_dim = 2 * key_dim + nv * dv
    nv % nk == 0 || error("gdn decode: n_v_heads $nv not divisible by n_k_heads $nk")
    size(rec_state) == (nv, dk, dv) ||
        error("gdn decode: state $(size(rec_state)) ≠ ($nv, $dk, $dv)")
    size(conv_state) == (conv_dim, ksz - 1) ||
        error("gdn decode: conv state $(size(conv_state)) ≠ ($conv_dim, $(ksz - 1))")
    weight = _amdh_dev(bk, bt.wgdn_conv.storage, :gdn_decode)
    A_log = _amdh_dev(bk, bt.wgdn_A_log.storage, :gdn_decode)
    dt_bias = _amdh_dev(bk, bt.wgdn_dt_bias.storage, :gdn_decode)
    norm_w = _amdh_dev(bk, bt.wgdn_norm.storage, :gdn_decode)
    cst = _amdh_dev(bk, conv_state, :gdn_decode)
    rst = _amdh_dev(bk, rec_state, :gdn_decode)
    kb = bk.ka
    Gesso.Inference.@decode_sub :gdn_conv bk begin
        _ka_gdn_conv_decode!(kb, s.conv, cst, s.mixed, weight, Int(ksz), Int(conv_dim))
    end
    Gesso.Inference.@decode_sub :gdn_recur bk begin
        _ka_gdn_delta!(
            kb, s.core, s.conv, s.a, s.b, A_log, dt_bias, rst,
            Int(nv), Int(nk), Int(dk), Int(dv), mx.v_head_tiled, eps,
        )
    end
    Gesso.Inference.@decode_sub :gdn_gnorm bk begin
        _ka_gdn_gnorm_decode!(kb, s.merged, s.core, s.z, norm_w, Int(nv), Int(dv), eps)
    end
    return s
end

function Gesso.Inference._sigmoid_mul!(bk::KADeviceBackend, dst, src, gate)
    d = _amdh_dev(bk, dst, :sigmoid_mul)
    s = _amdh_dev(bk, src, :sigmoid_mul)
    gt = _amdh_dev(bk, gate, :sigmoid_mul)
    length(d) == length(s) == length(gt) ||
        throw(gesso_error(ERR_INVALID_PLAN, "sigmoid_mul: operand lengths disagree"))
    _ka_elementwise_gate!(bk.ka, _ka_gate_sigmoid_mul_kernel!, d, s, gt)
    return dst
end

function Gesso.Inference._silu_mul!(bk::KADeviceBackend, dst, src, gate)
    d = _amdh_dev(bk, dst, :silu_mul)
    s = _amdh_dev(bk, src, :silu_mul)
    gt = _amdh_dev(bk, gate, :silu_mul)
    length(d) == length(s) == length(gt) ||
        throw(gesso_error(ERR_INVALID_PLAN, "silu_mul: operand lengths disagree"))
    _ka_elementwise_gate!(bk.ka, _ka_gate_silu_mul_kernel!, d, s, gt)
    return dst
end

function Gesso.Inference._shared_gate_add!(bk::KADeviceBackend, y, x, sd, gvec)
    yd = _amdh_dev(bk, y, :shared_gate_add)
    xd = _amdh_dev(bk, x, :shared_gate_add)
    sdd = _amdh_dev(bk, sd, :shared_gate_add)
    gd = _amdh_dev(bk, gvec, :shared_gate_add)
    seq, h = size(yd)
    size(xd) == (seq, h) && size(sdd) == (seq, h) && length(gd) == h ||
        throw(gesso_error(ERR_INVALID_PLAN, "shared_gate_add storage dimensions disagree"))
    _ka_shared_gate!(bk.ka, yd, xd, sdd, gd)
    return y
end

# The routed experts run on the host, as on Lava. Expert banks must be host
# storage (GgufQuantSlice or host Arrays); a device bank is refused rather than
# downloaded every token.
function Gesso.Inference._moe_topk_swiglu!(
    bk::KADeviceBackend,
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
            "GESSO_MOE_HOST_EXPERTS=0 has no $(backend_name(bk)) path yet " *
            "(device expert routing is not ported); leave it unset",
        ),
    )
    for (nm, w) in (("gate", gate), ("up", up), ("down", down))
        w isa AbstractArray && !(KA.get_backend(w) isa KA.CPU) && throw(
            gesso_error(
                ERR_INVALID_PLAN,
                "_moe_topk_swiglu!: expert bank $nm is a device array; the " *
                "$(backend_name(bk)) hybrid keeps expert banks on the host",
            ),
        )
    end
    overlap === nothing || overlap()
    xh = Gesso.Inference.@decode_stage :moe_d2h bk Array(x)
    rh = _amdh_host_weight(router)
    yh = zeros(Float32, size(y))
    Gesso.Inference.@decode_stage :moe_host bk Gesso.Inference._moe_topk_swiglu!(
        Gesso.CPUBackend(),
        yh,
        xh,
        rh,
        gate,
        up,
        down,
        n_used,
    )
    Gesso.Inference.@decode_stage :moe_h2d bk copyto!(y, yh)
    return y
end
