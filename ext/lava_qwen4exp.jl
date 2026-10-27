# qwen4exp (Qwen3.8 Flash-Next) forward on Lava. A device mirror of
# qwen4exp_forward! and qwen4exp_logits in src/Inference/qwen4exp.jl: the same
# block order, the same tap names and shapes, F32 device math against the
# F64-leaning CPU oracle. Dense weights are uploaded lazily, one tensor at a
# time, under the device weight budget. Routed experts never leave the host:
# the FFN input is downloaded, the host MoE route runs, and the result is
# uploaded. Token embedding and PLE row hashing stay on the host with one
# upload per forward.
#
# `fault` is a negative-control hook for the test suite. Production code only
# ever passes :none.

const _Q4X_FAULTS = (none=0, gdn_silu=1, ple_dil1=2, hc_swap=3, hc_no_scale=4)

function _q4x_fault(f::Symbol)
    haskey(_Q4X_FAULTS, f) || throw(ArgumentError("unknown qwen4exp fault $f"))
    return getproperty(_Q4X_FAULTS, f)
end

# Device model: the host model (for lazy weight reads and the host MoE route),
# the device weight cache, the RoPE frequencies, and uploaded bytes so far.
struct LavaQwen4exp
    cpu::Gesso.Inference.Qwen4expModel
    dev::Dict{String,Any}
    freq::Any
    bytes::Base.RefValue{Int64}
end

# Device decode state. `host` keeps only the PLE token history, which the CPU
# routine reads and writes.
mutable struct LavaQwen4expState
    pos::Int
    host::Gesso.Inference.Qwen4expState
    conv::Dict{Int,Any}
    ssm::Dict{Int,Any}
    k::Dict{Int,Any}
    v::Dict{Int,Any}
    ple_conv::Dict{Int,Any}
end

function Gesso.Inference.qwen4exp_device(::LavaBackend, m::Gesso.Inference.Qwen4expModel)
    cfg = m.cfg
    freq = Float32[cfg.rope_base^(-2 * i / cfg.n_rot) for i in 0:(cfg.n_rot÷2-1)]
    # Pools are keyed by the previous model's banks; its layer count does not apply.
    _lhot_register!(())
    return LavaQwen4exp(m, Dict{String,Any}(), Lava.LavaArray{Float32}(freq), Ref(Int64(0)))
end

function Gesso.Inference.qwen4exp_device_state(dm::LavaQwen4exp)
    return LavaQwen4expState(0, Gesso.Inference.Qwen4expState(dm.cpu.cfg),
        Dict{Int,Any}(), Dict{Int,Any}(), Dict{Int,Any}(), Dict{Int,Any}(), Dict{Int,Any}())
end

# --- device weights ---------------------------------------------------------

function _q4x_budget!(dm::LavaQwen4exp, bytes::Integer)
    dm.bytes[] += Int64(bytes)
    cap = Gesso.Inference._gpu_budget_bytes()
    dm.bytes[] <= cap || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "qwen4exp: device weights reach $(dm.bytes[]) B, over the budget $cap B " *
            "(GESSO_GPU_BUDGET_MIB)",
        ),
    )
    return nothing
end

function _q4x_upload(dm::LavaQwen4exp, h::Gesso.Inference.QuantMatrix)
    if _lq_eligible(h)
        _q4x_budget!(dm, _lq_device_bytes(h))
        return _lq_upload(h)
    end
    host = Gesso.Inference._quant_host_f32(h)
    _q4x_budget!(dm, sizeof(host))
    return Lava.LavaArray{Float32}(host)
end

function _q4x_upload(dm::LavaQwen4exp, h::AbstractArray)
    host = Float32.(h)
    _q4x_budget!(dm, sizeof(host))
    return Lava.LavaArray{Float32}(host)
end

# Device copy of a named weight, uploaded on first use.
function _q4x_dev(dm::LavaQwen4exp, name::String)
    return get!(dm.dev, name) do
        _q4x_upload(dm, Gesso.Inference._q4x_w(dm.cpu, name))
    end
end

# Tensors the forward reads straight from device memory, in the kernels that
# take F32 values (norms, biases, the conv taps, ssm_a). These never reach a
# matmul, so a half or IQ matrix stored under one of these names is widened
# here, and the native bytes are kept only for the matmul names.
_q4x_direct_read(name::AbstractString) =
    endswith(name, "norm.weight") || endswith(name, "_conv1d.weight") ||
    endswith(name, "ssm_dt.bias") || endswith(name, "ssm_a")

function _q4x_dev_f32(dm::LavaQwen4exp, name::String)
    return get!(dm.dev, name * "#f32") do
        h = Gesso.Inference._q4x_w(dm.cpu, name)
        host = h isa Gesso.Inference.QuantMatrix ?
            Gesso.Inference._quant_host_f32(h) : Float32.(h)
        _q4x_budget!(dm, sizeof(host))
        Lava.LavaArray{Float32}(host)
    end
end

# A dense vector as a (1, n) row, for the shared-expert gate.
function _q4x_dev_row(dm::LavaQwen4exp, name::String)
    return get!(dm.dev, name * "#row") do
        h = Gesso.Inference._q4x_w(dm.cpu, name)
        host = reshape(h isa Gesso.Inference.QuantMatrix ?
            Gesso.Inference._quant_host_f32(h) : Float32.(h), 1, :)
        _q4x_budget!(dm, sizeof(host))
        Lava.LavaArray{Float32}(host)
    end
end

# Dense bytes of the layer tensors below n_layers, read before any upload, with
# the size each one takes on the device: matmul matrices keep their native
# bytes when eligible, direct reads and ineligible tensors are F32. Routed
# expert banks are excluded because they stay on the host.
function _q4x_dense_bytes(m::Gesso.Inference.Qwen4expModel, n_layers::Int)
    total = 0
    for name in keys(m.index)
        startswith(name, "blk.") || continue
        il = tryparse(Int, split(name, '.')[2])
        (il === nothing || il >= n_layers) && continue
        occursin("_exps", name) && continue
        h = Gesso.Inference._q4x_w(m, name)
        native = h isa Gesso.Inference.QuantMatrix && _lq_eligible(h) &&
            !_q4x_direct_read(name)
        total += native ? _lq_device_bytes(h) :
            (h isa Gesso.Inference.QuantMatrix ? Int64(h.nrows) * h.ne0 * 4 : sizeof(Float32.(h)))
    end
    return total
end

# --- device matmul ---------------------------------------------------------

# Y = X * W' with W (out, in), dense F32 or a Lava quant matrix.
function _q4x_gemm(X::Lava.LavaArray, W)
    Y = Lava.LavaArray{Float32}(undef, size(X, 1), size(W, 1))
    if W isa LavaDenseQuant
        _lq_matmul!(Y, X, W)
    else
        mul!(Y, X, transpose(W), 1.0f0, 0.0f0)
    end
    return Y
end

_q4x_dmm(dm::LavaQwen4exp, X::Lava.LavaArray, name::String) = _q4x_gemm(X, _q4x_dev(dm, name))

# Launch helper: kernel f on the backend of ref, n work items, n appended as
# the kernel's last argument for its bounds guard.
function _q4x_run!(f, ref, n::Int, args...)
    n > 0 || return nothing
    f(KA.get_backend(ref))(args..., n; ndrange=n, workgroupsize=_LH_WG)
    return nothing
end

_q4x_dtap!(::Nothing, name::String, x) = nothing
_q4x_dtap!(taps, name::String, x) = (taps[name] = Array(x); nothing)

# --- kernels ----------------------------------------------------------------

# RMS norm over contiguous groups of nw columns, one work item per (token, group).
@kernel cpu=false function _q4x_group_rms_kernel!(y, x, w, nw::Int, T::Int, eps::Float32, shared::Bool, n::Int)
    i = @index(Global, Linear)
    i <= n || return
    t = (i - 1) % T + 1
    base = ((i - 1) ÷ T) * nw
    acc = 0.0f0
    for c in 1:nw
        acc += x[t, base+c] * x[t, base+c]
    end
    r = 1.0f0 / sqrt(acc / Float32(nw) + eps)
    for c in 1:nw
        y[t, base+c] = (x[t, base+c] * r) * w[shared ? c : base+c]
    end
end

# mixed[t, c] = sum_s xn[t, c+NE*s] * sigm(gate[t, c+NE*s]) / HC
@kernel cpu=false function _q4x_hc_mix_kernel!(mixed, xn, gate, T::Int, NE::Int, HC::Int, n::Int)
    i = @index(Global, Linear)
    i <= n || return
    t = (i - 1) % T + 1
    c = (i - 1) ÷ T + 1
    acc = 0.0f0
    for s in 0:(HC-1)
        j = c + NE * s
        acc += xn[t, j] * (1.0f0 / (1.0f0 + exp(-gate[t, j])))
    end
    mixed[t, c] = acc / Float32(HC)
end

# y[t, c+NE*s] = res + out[t, c] * 2*sigm(inj[t, s+1] / HC). fc = 3 swaps the
# stream index of inj, fc = 4 drops the factor 2 (negative controls only).
@kernel cpu=false function _q4x_hc_combine_kernel!(y, res, out, inj, T::Int, NE::Int, HC::Int, fc::Int, n::Int)
    i = @index(Global, Linear)
    i <= n || return
    t = (i - 1) % T + 1
    r = (i - 1) ÷ T + 1
    c = (r - 1) % NE + 1
    s = (r - 1) ÷ NE
    ss = fc == 3 ? HC - 1 - s : s
    sg = 1.0f0 / (1.0f0 + exp(-inj[t, ss+1] / Float32(HC)))
    w = fc == 4 ? sg : 2.0f0 * sg
    y[t, r] = res[t, r] + out[t, c] * w
end

# PLE gate per (token, stream): sc = <key_s, qry_s> / sqrt(NE),
# gate = sigm(sign(sc) * sqrt(max(|sc|, 1e-6))), gated = val * gate.
@kernel cpu=false function _q4x_ple_gate_kernel!(gate, gated, key, qry, val, T::Int, NE::Int, HC::Int, n::Int)
    i = @index(Global, Linear)
    i <= n || return
    t = (i - 1) % T + 1
    s = (i - 1) ÷ T
    acc = 0.0f0
    for c in 1:NE
        j = c + NE * s
        acc += key[t, j] * qry[t, j]
    end
    sc = acc / sqrt(Float32(NE))
    gv = 1.0f0 / (1.0f0 + exp(-(sign(sc) * sqrt(max(abs(sc), 1.0f-6)))))
    gate[t, s+1] = gv
    for c in 1:NE
        gated[t, c+NE*s] = val[t, c] * gv
    end
end

# Depthwise causal conv over [hist; x] with dilation dil: y[t, c] reads
# xx[H+t-(k-1-p)*dil, c], where xx = [hist; x] and H = (k-1)*dil. The old
# history is only read here, so the caller writes the new one separately.
@kernel cpu=false function _q4x_conv_kernel!(y, hist, x, w, H::Int, T::Int, C::Int, k::Int, dil::Int, n::Int)
    i = @index(Global, Linear)
    i <= n || return
    t = (i - 1) % T + 1
    c = (i - 1) ÷ T + 1
    acc = 0.0f0
    for p in 0:(k-1)
        j = H + t - (k - 1 - p) * dil
        # Both loads are clamped and taken unconditionally, then the scalar is
        # selected. A select between two different buffers does not compile
        # to a valid device access (it lost the Vulkan device on Lava).
        hv = hist[min(max(j, 1), H), c]
        xv = x[min(max(j - H, 1), T), c]
        acc += w[p+1+k*(c-1)] * (j <= H ? hv : xv)
    end
    y[t, c] = acc
end

# Last H rows of [hist; x]: the next conv history.
@kernel cpu=false function _q4x_hist_kernel!(newh, hist, x, H::Int, T::Int, C::Int, n::Int)
    i = @index(Global, Linear)
    i <= n || return
    r = (i - 1) % H + 1
    c = (i - 1) ÷ H + 1
    j = T + r
    hv = hist[min(max(j, 1), H), c]
    xv = x[min(max(j - H, 1), T), c]
    newh[r, c] = j <= H ? hv : xv
end

# GDN log decay with the qwen4exp convention: sp = softplus(alpha + dt),
# g = sp * A with A used directly (the qwen35 path uses -exp(A_log)).
@kernel cpu=false function _q4x_gdn_gate_kernel!(sp, g, alpha, dt, A, T::Int, n::Int)
    i = @index(Global, Linear)
    i <= n || return
    t = (i - 1) % T + 1
    h = (i - 1) ÷ T + 1
    x = alpha[t, h] + dt[h]
    s = x > 20.0f0 ? x : log(1.0f0 + exp(x))
    sp[t, h] = s
    g[t, h] = s * A[h]
end

# Conv output columns to q (T, nk, dk), k (T, nk, dk), v (T, nv, dv); column
# index d + dk*(h-1) within each block.
@kernel cpu=false function _q4x_qkv_split_kernel!(qd, kd, vd, c, T::Int, nk::Int, dk::Int, nv::Int, dv::Int, n::Int)
    i = @index(Global, Linear)
    i <= n || return
    t = (i - 1) % T + 1
    col = (i - 1) ÷ T + 1
    nq = nk * dk
    if col <= nq
        h = (col - 1) ÷ dk + 1
        d = (col - 1) % dk + 1
        qd[t, h, d] = c[t, col]
    elseif col <= 2 * nq
        cc = col - nq
        h = (cc - 1) ÷ dk + 1
        d = (cc - 1) % dk + 1
        kd[t, h, d] = c[t, col]
    else
        cc = col - 2 * nq
        h = (cc - 1) ÷ dv + 1
        d = (cc - 1) % dv + 1
        vd[t, h, d] = c[t, col]
    end
end

# Interleaved q/gate split: head h uses columns 2h*HD+1 .. 2h*HD+HD for q and
# the next HD for the output gate.
@kernel cpu=false function _q4x_qgate_split_kernel!(qd, gt, qf, HD::Int, T::Int, n::Int)
    i = @index(Global, Linear)
    i <= n || return
    t = (i - 1) % T + 1
    col = (i - 1) ÷ T + 1
    h = (col - 1) ÷ HD
    d = (col - 1) % HD + 1
    qd[t, col] = qf[t, 2 * h * HD + d]
    gt[t, col] = qf[t, 2 * h * HD + HD + d]
end

# NeoX rotation of the first n_rot dims of each head, positions p0+t-1.
@kernel cpu=false function _q4x_rope_kernel!(x, freq, nh::Int, hd::Int, half::Int, T::Int, p0::Int, n::Int)
    i = @index(Global, Linear)
    i <= n || return
    t = (i - 1) % T + 1
    r = (i - 1) ÷ T
    hh = r % nh
    ii = r ÷ nh
    θ = Float32(p0 + t - 1) * freq[ii+1]
    c = cos(θ)
    s = sin(θ)
    a = hh * hd + ii + 1
    b = a + half
    x1 = x[t, a]
    x2 = x[t, b]
    x[t, a] = x1 * c - x2 * s
    x[t, b] = x1 * s + x2 * c
end

# Causal GQA attention for one (token, head): query row t at absolute position
# p0+t-1 sees cache rows 1..p0+t. Three passes (max, sum, output) keep each
# work item free of scratch arrays.
@kernel cpu=false function _q4x_attn_kernel!(o, q, K, V, scale::Float32, grp::Int, HD::Int, p0::Int, T::Int, n::Int)
    i = @index(Global, Linear)
    i <= n || return
    t = (i - 1) % T + 1
    h = (i - 1) ÷ T
    kb = (h ÷ grp) * HD
    qb = h * HD
    lim = p0 + t
    mx = -Inf32
    for u in 1:lim
        sc = 0.0f0
        for d in 1:HD
            sc += q[t, qb+d] * K[u, kb+d]
        end
        mx = max(mx, sc * scale)
    end
    z = 0.0f0
    for u in 1:lim
        sc = 0.0f0
        for d in 1:HD
            sc += q[t, qb+d] * K[u, kb+d]
        end
        z += exp(sc * scale - mx)
    end
    for d in 1:HD
        o[t, qb+d] = 0.0f0
    end
    for u in 1:lim
        sc = 0.0f0
        for d in 1:HD
            sc += q[t, qb+d] * K[u, kb+d]
        end
        w = exp(sc * scale - mx) / z
        for d in 1:HD
            o[t, qb+d] += w * V[u, kb+d]
        end
    end
end

# Row-wise concatenation of the KV cache with the new rows.
@kernel cpu=false function _q4x_cat_rows_kernel!(dst, old, new, nold::Int, C::Int, T::Int, n::Int)
    i = @index(Global, Linear)
    i <= n || return
    R = nold + T
    r = (i - 1) % R + 1
    c = (i - 1) ÷ R + 1
    ov = old[min(max(r, 1), nold), c]
    nv = new[min(max(r - nold, 1), T), c]
    dst[r, c] = r <= nold ? ov : nv
end

# Shared expert: shg = sh * sigm(sg[t]), out = moe + shg.
@kernel cpu=false function _q4x_shgate_kernel!(shg, out, moe, sh, sg, T::Int, n::Int)
    i = @index(Global, Linear)
    i <= n || return
    t = (i - 1) % T + 1
    c = (i - 1) ÷ T + 1
    g = sh[t, c] * (1.0f0 / (1.0f0 + exp(-sg[t, 1])))
    shg[t, c] = g
    out[t, c] = moe[t, c] + g
end

# --- device ops (also called directly by the per-op tests) -------------------

function _q4x_dev_group_rms(x::Lava.LavaArray, nw::Int, w, eps)
    T, F = size(x)
    y = Lava.LavaArray{Float32}(undef, T, F)
    shared = size(w, 1) == nw
    _q4x_run!(_q4x_group_rms_kernel!, x, T * (F ÷ nw), y, x, w, nw, T, Float32(eps), shared)
    return y
end

# Returns (mixed (T, NE), inj (T, HC) or nothing).
function _q4x_dev_hc_mix(dm::LavaQwen4exp, x::Lava.LavaArray, pre::String; inject::Bool=true)
    cfg = dm.cpu.cfg
    NE, HC = cfg.n_embd, cfg.n_hc
    T = size(x, 1)
    xn = _q4x_dev_group_rms(x, NE, _q4x_dev_f32(dm, pre * "_norm.weight"), cfg.eps)
    lo = _q4x_dmm(dm, xn, pre * "_down.weight")
    lo .= (lo ./ Float32(HC)) ./ (1.0f0 .+ exp.(-lo ./ Float32(HC)))
    gate = _q4x_dmm(dm, lo, pre * "_up.weight")
    mixed = Lava.LavaArray{Float32}(undef, T, NE)
    _q4x_run!(_q4x_hc_mix_kernel!, mixed, T * NE, mixed, xn, gate, T, NE, HC)
    inj = inject ? _q4x_dmm(dm, xn, pre * "_inject.weight") : nothing
    return mixed, inj
end

function _q4x_dev_hc_combine(cfg::Gesso.Inference.Qwen4expConfig, res, out, inj, fc::Int=0)
    NE, HC = cfg.n_embd, cfg.n_hc
    T = size(res, 1)
    y = Lava.LavaArray{Float32}(undef, T, NE * HC)
    _q4x_run!(_q4x_hc_combine_kernel!, y, T * NE * HC, y, res, out, inj, T, NE, HC, fc)
    return y
end

# Returns (y (T, C), newhist (H, C)); hist is read, never written.
function _q4x_dev_conv(hist, x, w, k::Int, dil::Int)
    H = size(hist, 1)
    H >= (k - 1) * dil || error("qwen4exp: conv history has $H rows, needs $((k - 1) * dil)")
    T, C = size(x)
    y = Lava.LavaArray{Float32}(undef, T, C)
    _q4x_run!(_q4x_conv_kernel!, y, T * C, y, hist, x, w, H, T, C, k, dil)
    newh = Lava.LavaArray{Float32}(undef, H, C)
    _q4x_run!(_q4x_hist_kernel!, newh, H * C, newh, hist, x, H, T, C)
    return y, newh
end

# Returns (gate (T, HC), gated (T, HC*NE)).
function _q4x_dev_ple_gate(key, qry, val, NE::Int, HC::Int)
    T = size(qry, 1)
    gate = Lava.LavaArray{Float32}(undef, T, HC)
    gated = Lava.LavaArray{Float32}(undef, T, HC * NE)
    _q4x_run!(_q4x_ple_gate_kernel!, gated, T * HC, gate, gated, key, qry, val, T, NE, HC)
    return gate, gated
end

# Returns (sp (T, nv), g (T, nv)).
function _q4x_dev_gdn_gate(alpha, dt, A)
    T, nv = size(alpha)
    sp = Lava.LavaArray{Float32}(undef, T, nv)
    g = Lava.LavaArray{Float32}(undef, T, nv)
    _q4x_run!(_q4x_gdn_gate_kernel!, alpha, T * nv, sp, g, alpha, dt, A, T)
    return sp, g
end

# Causal GQA core: q (T, NH*HD) roped, K and V (L, NKV*HD) with L = p0 + T.
function _q4x_dev_attn_core(q, K, V, p0::Int, NH::Int, NKV::Int, HD::Int)
    T = size(q, 1)
    o = Lava.LavaArray{Float32}(undef, T, NH * HD)
    scale = Float32(1 / sqrt(HD))
    _q4x_run!(_q4x_attn_kernel!, o, T * NH, o, q, K, V, scale, NH ÷ NKV, HD, p0, T)
    return o
end

function _q4x_dev_rope!(dm::LavaQwen4exp, x::Lava.LavaArray, nh::Int, p0::Int)
    cfg = dm.cpu.cfg
    half = cfg.n_rot ÷ 2
    T = size(x, 1)
    _q4x_run!(_q4x_rope_kernel!, x, T * nh * half, x, dm.freq, nh, cfg.head_dim, half, T, p0)
    return x
end

_q4x_cat(::Nothing, new) = new
function _q4x_cat(old::Lava.LavaArray, new::Lava.LavaArray)
    nold = size(old, 1)
    T, C = size(new)
    dst = Lava.LavaArray{Float32}(undef, nold + T, C)
    _q4x_run!(_q4x_cat_rows_kernel!, dst, (nold + T) * C, dst, old, new, nold, C, T)
    return dst
end

# --- blocks -----------------------------------------------------------------

function _q4x_dev_ple(dm::LavaQwen4exp, st::LavaQwen4expState, il::Int, emb, res, taps, fc::Int)
    cfg = dm.cpu.cfg
    NE, HC = cfg.n_embd, cfg.n_hc
    p = "blk.$il."
    T = size(res, 1)
    key = _q4x_dev_group_rms(_q4x_dmm(dm, emb, p * "ple_key.weight"), NE,
        _q4x_dev_f32(dm, p * "ple_norm_key.weight"), cfg.eps)
    val = _q4x_dmm(dm, emb, p * "ple_value.weight")
    qry = _q4x_dev_group_rms(res, NE, _q4x_dev_f32(dm, p * "ple_norm_query.weight"), cfg.eps)
    gate, gated = _q4x_dev_ple_gate(key, qry, val, NE, HC)
    nrm = _q4x_dev_group_rms(gated, NE, _q4x_dev_f32(dm, p * "ple_norm_conv.weight"), cfg.eps)
    k = cfg.ple_kernel
    dil = fc == 2 ? 1 : cfg.ple_ngram
    H = (k - 1) * cfg.ple_ngram
    hist = get!(() -> Lava.LavaArray{Float32}(zeros(Float32, H, HC * NE)), st.ple_conv, il)
    conv, newh = _q4x_dev_conv(hist, nrm, _q4x_dev_f32(dm, p * "ple_conv1d.weight"), k, dil)
    st.ple_conv[il] = newh
    conv .= conv ./ (1.0f0 .+ exp.(-conv))
    out = Lava.LavaArray{Float32}(undef, T, HC * NE)
    out .= res .+ gated .+ conv
    _q4x_dtap!(taps, "ple_gate-$il", gate)
    _q4x_dtap!(taps, "ple_gated_value-$il", gated)
    _q4x_dtap!(taps, "ple_conv_out-$il", conv)
    return out
end

function _q4x_dev_gdn(dm::LavaQwen4exp, st::LavaQwen4expState, il::Int, x, taps, fc::Int)
    cfg = dm.cpu.cfg
    p = "blk.$il."
    T = size(x, 1)
    nk, dk, nv, dv = cfg.ssm_nk, cfg.ssm_dk, cfg.ssm_nv, cfg.ssm_dv
    qkv = _q4x_dmm(dm, x, p * "attn_qkv.weight")
    z = _q4x_dmm(dm, x, p * "attn_gate.weight")
    beta = _q4x_dmm(dm, x, p * "ssm_beta.weight")
    beta .= 1.0f0 ./ (1.0f0 .+ exp.(-beta))
    alpha = _q4x_dmm(dm, x, p * "ssm_alpha.weight")
    sp, g = _q4x_dev_gdn_gate(alpha, _q4x_dev_f32(dm, p * "ssm_dt.bias"), _q4x_dev_f32(dm, p * "ssm_a"))
    k = cfg.conv_kernel
    C = size(qkv, 2)
    hist = get!(() -> Lava.LavaArray{Float32}(zeros(Float32, k - 1, C)), st.conv, il)
    craw, newh = _q4x_dev_conv(hist, qkv, _q4x_dev_f32(dm, p * "ssm_conv1d.weight"), k, 1)
    st.conv[il] = newh
    c = Lava.LavaArray{Float32}(undef, T, C)
    c .= craw ./ (1.0f0 .+ exp.(-craw))
    nq = nk * dk
    qd = Lava.LavaArray{Float32}(undef, T, nk, dk)
    kd = Lava.LavaArray{Float32}(undef, T, nk, dk)
    vd = Lava.LavaArray{Float32}(undef, T, nv, dv)
    _q4x_run!(_q4x_qkv_split_kernel!, c, T * C, qd, kd, vd, c, T, nk, dk, nv, dv)
    state = get!(() -> Lava.LavaArray{Float32}(zeros(Float32, nv, dk, dv)), st.ssm, il)
    core = Lava.LavaArray{Float32}(undef, T, nv, dv)
    _lava_gdn_reference!(core, qd, kd, vd, g, beta, state; l2norm=true, v_head_tiled=true)
    merged = Lava.LavaArray{Float32}(undef, T, nv * dv)
    gnorm = _lava_gdn_gnorm_kernel!(KA.get_backend(core))
    gnorm(merged, core, z, _q4x_dev_f32(dm, p * "ssm_norm.weight"), nv, dv,
        Float32(cfg.eps), fc != 1; ndrange=T * nv * _LH_RED_LANES, workgroupsize=(_LH_RED_LANES,))
    out = _q4x_dmm(dm, merged, p * "ssm_out.weight")
    if taps !== nothing
        _q4x_dtap!(taps, "linear_attn_qkv_mixed-$il", qkv)
        _q4x_dtap!(taps, "z-$il", z)
        _q4x_dtap!(taps, "beta_sigmoid-$il", beta)
        _q4x_dtap!(taps, "alpha-$il", alpha)
        _q4x_dtap!(taps, "a_softplus-$il", sp)
        _q4x_dtap!(taps, "gate-$il", g)
        _q4x_dtap!(taps, "conv_output_raw-$il", craw)
        _q4x_dtap!(taps, "conv_output_silu-$il", c)
        _q4x_dtap!(taps, "attn_output-$il", reshape(permutedims(Array(core), (1, 3, 2)), T, nv * dv))
        _q4x_dtap!(taps, "final_output-$il", merged)
        _q4x_dtap!(taps, "linear_attn_out-$il", out)
    end
    return out
end

function _q4x_dev_attn(dm::LavaQwen4exp, st::LavaQwen4expState, il::Int, x, taps)
    cfg = dm.cpu.cfg
    p = "blk.$il."
    T = size(x, 1)
    NH, NKV, HD = cfg.n_head, cfg.n_head_kv, cfg.head_dim
    qf = _q4x_dmm(dm, x, p * "attn_q.weight")
    qraw = Lava.LavaArray{Float32}(undef, T, NH * HD)
    gt = Lava.LavaArray{Float32}(undef, T, NH * HD)
    _q4x_run!(_q4x_qgate_split_kernel!, qraw, T * NH * HD, qraw, gt, qf, HD, T)
    q = _q4x_dev_group_rms(qraw, HD, _q4x_dev_f32(dm, p * "attn_q_norm.weight"), cfg.eps)
    k = _q4x_dev_group_rms(_q4x_dmm(dm, x, p * "attn_k.weight"), HD,
        _q4x_dev_f32(dm, p * "attn_k_norm.weight"), cfg.eps)
    v = _q4x_dmm(dm, x, p * "attn_v.weight")
    _q4x_dtap!(taps, "Qcur_normed-$il", q)
    _q4x_dtap!(taps, "Kcur_normed-$il", k)
    _q4x_dtap!(taps, "Vcur-$il", v)
    p0 = st.pos
    _q4x_dev_rope!(dm, q, NH, p0)
    _q4x_dev_rope!(dm, k, NKV, p0)
    K = st.k[il] = _q4x_cat(get(st.k, il, nothing), k)
    V = st.v[il] = _q4x_cat(get(st.v, il, nothing), v)
    o = _q4x_dev_attn_core(q, K, V, p0, NH, NKV, HD)
    og = Lava.LavaArray{Float32}(undef, T, NH * HD)
    og .= o .* (1.0f0 ./ (1.0f0 .+ exp.(-gt)))
    out = _q4x_dmm(dm, og, p * "attn_output.weight")
    _q4x_dtap!(taps, "Qcur-$il", q)
    _q4x_dtap!(taps, "Kcur-$il", k)
    _q4x_dtap!(taps, "attn_pregate-$il", o)
    _q4x_dtap!(taps, "attn_gated-$il", og)
    _q4x_dtap!(taps, "attn_output-$il", out)
    return out
end

# Routed experts run on the host (the router logits are recomputed on the
# host from the downloaded input); the shared expert runs on the device.
function _q4x_dev_ffn(dm::LavaQwen4exp, il::Int, x, taps)
    cfg = dm.cpu.cfg
    m = dm.cpu
    p = "blk.$il."
    T, NE = size(x)
    router = Gesso.Inference._q4x_w(m, p * "ffn_gate_inp.weight")
    xh = Array(x)
    ge = Gesso.Inference._q4x_w(m, p * "ffn_gate_exps.weight")
    ue = Gesso.Inference._q4x_w(m, p * "ffn_up_exps.weight")
    de = Gesso.Inference._q4x_w(m, p * "ffn_down_exps.weight")
    # Decode only: prefill keeps the all-host route, since a prompt's picks
    # spread over every row and the pool would refill more than it hits.
    if _lhot_wants(xh, ge, ue, de, cfg.n_layer)
        moe = Lava.LavaArray{Float32}(undef, T, NE)
        _lhot_moe_decode!(moe, x, xh, router, ge, ue, de, cfg.n_used)
    else
        moe_h = zeros(Float32, T, NE)
        Gesso.Inference._q4x_moe!(moe_h, xh, router, ge, ue, de, cfg.n_used)
        moe = Lava.LavaArray{Float32}(moe_h)
    end
    gs = _q4x_dmm(dm, x, p * "ffn_gate_shexp.weight")
    us = _q4x_dmm(dm, x, p * "ffn_up_shexp.weight")
    act = Lava.LavaArray{Float32}(undef, size(gs)...)
    act .= (gs ./ (1.0f0 .+ exp.(-gs))) .* us
    sh = _q4x_dmm(dm, act, p * "ffn_down_shexp.weight")
    sg = _q4x_gemm(x, _q4x_dev_row(dm, p * "ffn_gate_inp_shexp.weight"))
    shg = Lava.LavaArray{Float32}(undef, T, NE)
    out = Lava.LavaArray{Float32}(undef, T, NE)
    _q4x_run!(_q4x_shgate_kernel!, out, T * NE, shg, out, moe, sh, sg, T)
    if taps !== nothing
        _q4x_dtap!(taps, "ffn_moe_logits-$il", _q4x_dmm(dm, x, p * "ffn_gate_inp.weight"))
        _q4x_dtap!(taps, "ffn_moe_out-$il", moe)
        _q4x_dtap!(taps, "ffn_shexp-$il", sh)
        _q4x_dtap!(taps, "shared_expert_gate-$il", sg)
        _q4x_dtap!(taps, "ffn_shexp_gated-$il", shg)
        _q4x_dtap!(taps, "ffn_out-$il", out)
    end
    return out
end

# --- forward and logits -----------------------------------------------------

"""
    qwen4exp_forward!(dm::LavaQwen4exp, st::LavaQwen4expState, tokens; n_layers, taps, fault)

Device counterpart of the CPU forward. Returns the device residual (T, n_hc *
n_embd). `taps` receives host Matrix{Float32} copies under the CPU names.
"""
function Gesso.Inference.qwen4exp_forward!(dm::LavaQwen4exp, st::LavaQwen4expState,
        tokens::AbstractVector{<:Integer}; n_layers::Int=dm.cpu.cfg.n_layer,
        taps=nothing, fault::Symbol=:none)
    cfg = dm.cpu.cfg
    toks = Vector{Int}(tokens)
    T = length(toks)
    T > 0 || error("qwen4exp_forward!: no tokens")
    0 < n_layers <= cfg.n_layer || error("qwen4exp_forward!: n_layers $n_layers out of 1..$(cfg.n_layer)")
    fc = _q4x_fault(fault)
    if any(il -> cfg.attn_layer[il+1] && cfg.compress[il+1] > 0, 0:(n_layers-1)) &&
       st.pos + T > cfg.qsa_top_k
        error("qwen4exp_forward!: context $(st.pos + T) exceeds the QSA top_k $(cfg.qsa_top_k); " *
              "the sparse-attention indexer is not implemented")
    end
    e = Gesso.Inference._q4x_rows(Gesso.Inference._q4x_w(dm.cpu, "token_embd.weight"), toks)
    res = Lava.LavaArray{Float32}(repeat(e, 1, cfg.n_hc))
    pemb = isempty(cfg.ple_layers) ? nothing :
        Lava.LavaArray{Float32}(Gesso.Inference._q4x_ple_embed(dm.cpu, st.host, toks))
    for il in 0:(n_layers-1)
        il in cfg.ple_layers && (res = _q4x_dev_ple(dm, st, il, pemb, res, taps, fc))
        cur, inj = _q4x_dev_hc_mix(dm, res, "blk.$il.hc_attn")
        _q4x_dtap!(taps, "hc_mixed-$il", cur)
        cur = cfg.attn_layer[il+1] ? _q4x_dev_attn(dm, st, il, cur, taps) :
            _q4x_dev_gdn(dm, st, il, cur, taps, fc)
        res = _q4x_dev_hc_combine(cfg, res, cur, inj, fc == 3 || fc == 4 ? fc : 0)
        _q4x_dtap!(taps, "hc_combine-$il", res)
        cur, inj = _q4x_dev_hc_mix(dm, res, "blk.$il.hc_ffn")
        cur = _q4x_dev_ffn(dm, il, cur, taps)
        res = _q4x_dev_hc_combine(cfg, res, cur, inj, fc == 3 || fc == 4 ? fc : 0)
        _q4x_dtap!(taps, "l_last-$il", res)
    end
    st.pos += T
    return res
end

"""
    qwen4exp_logits(dm::LavaQwen4exp, res; taps=nothing) -> Vector{Float32}

Logits for the last row of the device residual.
"""
function Gesso.Inference.qwen4exp_logits(dm::LavaQwen4exp, res::Lava.LavaArray; taps=nothing)
    last = Lava.LavaArray{Float32}(Array(res)[end:end, :])
    h, _ = _q4x_dev_hc_mix(dm, last, "output_hc"; inject=false)
    logits = _q4x_dmm(dm, h, "output.weight")
    _q4x_dtap!(taps, "result_norm", h)
    _q4x_dtap!(taps, "result_output", logits)
    return vec(Array(logits))
end
