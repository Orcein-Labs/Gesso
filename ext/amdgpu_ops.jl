# Operator methods for KADeviceBackend: embedding lookup, RoPE, softmax,
# SwiGLU, GEGLU, matmul, and the row RMS-norm entry point over the kernel in
# amdgpu_kernels.jl. Every kernel is one-dimensional, so the launch shape is a
# single ndrange and the workgroup size is one parameter; test/amdgpu_compile.jl
# compiles each kernel for every AMD target at two workgroup sizes.
#
# Math: Float32 storage. The angles of RoPE and the softmax sums accumulate in
# Float64, as the CPU oracle does, so device error is storage rounding and not
# phase drift. GEGLU evaluates GELU with Lava's Float32 erf polynomial, which
# measures 4.1e-7 absolute against the Float64 oracle on the test grid.

const _KA_OP_WG = 64
const _KA_WORKLOAD = Union{Gesso.PrefillWorkload, Gesso.DecodeWorkload}

# --- kernels ----------------------------------------------------------------

# out[i, j] = table[tokens[i] + 1, j]; tokens are 0-based, validated on the host.
@kernel function _ka_embedding_kernel!(dst, @Const(table), @Const(toks), rows::Int)
    idx = @index(Global, Linear)
    i = (idx - 1) % rows + 1
    j = (idx - 1) ÷ rows + 1
    dst[i, j] = table[Int(toks[i]) + 1, j]
end

# One thread per (position, head, pair). Pair i (1-based) rotates features
# (2i-1, 2i) when interleaved, else (i, i + half). Angles are Float64.
@kernel function _ka_rope_kernel!(x, @Const(pos), @Const(freq), interleaved::Bool, half::Int, nheads::Int)
    n = @index(Global, Linear)
    i = (n - 1) % half + 1
    r = (n - 1) ÷ half
    h = r % nheads + 1
    t = r ÷ nheads + 1
    θ = Float64(pos[t]) * freq[i]
    c = cos(θ)
    s = sin(θ)
    a = interleaved ? 2 * i - 1 : i
    b = interleaved ? 2 * i : i + half
    x1 = Float64(x[t, h, a])
    x2 = Float64(x[t, h, b])
    x[t, h, a] = Float32(x1 * c - x2 * s)
    x[t, h, b] = Float32(x1 * s + x2 * c)
end

# One thread per row. Columns j ≤ offset + i are unmasked (causal, with the key
# window ending at the last query). A row whose unmasked maximum is not finite
# is written as NaN, so the caller's finite check reports it.
@kernel function _ka_softmax_rows_kernel!(dst, @Const(s), K::Int, offset::Int)
    i = @index(Global, Linear)
    lim = min(K, offset + i)
    m = -Inf32
    for j in 1:lim
        m = max(m, s[i, j])
    end
    if isfinite(m)
        acc = 0.0
        for j in 1:lim
            acc += exp(Float64(s[i, j]) - Float64(m))
        end
        for j in 1:K
            dst[i, j] = j <= lim ? Float32(exp(Float64(s[i, j]) - Float64(m)) / acc) : 0.0f0
        end
    else
        for j in 1:K
            dst[i, j] = NaN32
        end
    end
end

@kernel function _ka_swiglu_kernel!(dst, @Const(g), @Const(u))
    i = @index(Global, Linear)
    v = g[i]
    dst[i] = (v / (1.0f0 + exp(-v))) * u[i]
end

# Exact (erf) GELU in Float32: gelu(x) = 0.5 x (1 + erf(x/√2)). The erf is a
# short polynomial on [0, 0.5], and erfc polynomials on [0.5, 1.25), [1.25, 2.5)
# and [2.5, 5), so the left tail (1 + erf(z) = erfc(-z)) is not a cancellation.
@inline function _ka_gelu_erf32(x::Float32)
    z = x * 0.70710678118654752f0
    t = abs(z)
    if t < 0.5f0
        s = z * z
        p = muladd(s, -0.0008394848f0, 0.005218856f0)
        p = muladd(s, p, -0.026865287f0)
        p = muladd(s, p, 0.11283784f0)
        p = muladd(s, p, -0.37612638f0)
        p = muladd(s, p, 0.12837917f0)
        one_plus = 1.0f0 + muladd(z, p, z)
    else
        if t < 1.25f0
            u = t - 0.5f0
            q = -0.0003758171f0
            q = muladd(u, q, 0.00073027064f0)
            q = muladd(u, q, 0.003422347f0)
            q = muladd(u, q, -0.011067663f0)
            q = muladd(u, q, -0.0048868246f0)
            q = muladd(u, q, 0.04987047f0)
            q = muladd(u, q, -0.007286422f0)
            q = muladd(u, q, -0.18308467f0)
            q = muladd(u, q, 0.14646415f0)
            q = muladd(u, q, 0.4393913f0)
            q = muladd(u, q, -0.8787826f0)
            ec = muladd(u, q, 0.47950011f0)
        elseif t < 2.5f0
            u = t - 1.25f0
            p = 7.4290696f-6
            p = muladd(p, u, -7.185403f-5)
            p = muladd(p, u, 0.00035109207f0)
            p = muladd(p, u, -0.0012127459f0)
            p = muladd(p, u, 0.0034827239f0)
            p = muladd(p, u, -0.0090524733f0)
            p = muladd(p, u, 0.022003610f0)
            p = muladd(p, u, -0.050217031f0)
            p = muladd(p, u, 0.10679546f0)
            p = muladd(p, u, -0.20882187f0)
            p = muladd(p, u, 0.36782292f0)
            ec = exp(-t * t) * p
        elseif t < 5.0f0
            u = t - 2.5f0
            p = 2.4104783f-9
            p = muladd(p, u, -4.8156740f-8)
            p = muladd(p, u, 4.6148624f-7)
            p = muladd(p, u, -2.8993439f-6)
            p = muladd(p, u, 1.3911895f-5)
            p = muladd(p, u, -5.6550075f-5)
            p = muladd(p, u, 0.00020915569f0)
            p = muladd(p, u, -0.00073255685f0)
            p = muladd(p, u, 0.0024666584f0)
            p = muladd(p, u, -0.0080014843f0)
            p = muladd(p, u, 0.024937987f0)
            p = muladd(p, u, -0.074347346f0)
            p = muladd(p, u, 0.21080636f0)
            ec = exp(-t * t) * p
        else
            ec = 0.0f0
        end
        one_plus = z < 0.0f0 ? ec : 2.0f0 - ec
    end
    return 0.5f0 * x * one_plus
end

@kernel function _ka_geglu_kernel!(dst, @Const(g), @Const(u))
    i = @index(Global, Linear)
    dst[i] = _ka_gelu_erf32(g[i]) * u[i]
end

# --- host-side launch helpers ------------------------------------------------

_ka_after_op!(b::KADeviceBackend, ::Gesso.PrefillWorkload) = _ka_sync!(b)
_ka_after_op!(::KADeviceBackend, ::Gesso.DecodeWorkload) = nothing

function _ka_upload_host(ka, ::Type{T}, host::AbstractArray) where {T}
    dev = KA.allocate(ka, T, size(host))
    copyto!(dev, convert(Array{T}, host))
    return dev
end

function _ka_positive_dims(xs, op::Symbol)
    all(n -> n > 0, size(xs)) || throw(
        gesso_error(ERR_INVALID_PLAN, "$op: dimensions must be positive"; op=op),
    )
    return nothing
end

# --- embedding ----------------------------------------------------------------

function _ka_embedding!(b::KADeviceBackend, dst, table, tokens)
    ds = _ka_device_storage!(:embedding_lookup!, b, dst)
    ts = _ka_device_storage!(:embedding_lookup!, b, table)
    Gesso._validate_embedding_inputs(dst, table, tokens)
    n = length(ds)
    n == 0 && return dst
    toks = _ka_upload_host(b.ka, Int32, Int32.(tokens))
    _ka_embedding_kernel!(b.ka, _KA_OP_WG)(ds, ts, toks, size(ds, 1); ndrange=n)
    return dst
end

function embedding_lookup!(
    b::KADeviceBackend,
    dst::Activation,
    table::EmbeddingTable,
    tokens::AbstractVector{Int},
    workload::_KA_WORKLOAD,
)
    _ka_embedding!(b, dst, table, tokens)
    _ka_after_op!(b, workload)
    return dst
end

# --- row RMS-norm ---------------------------------------------------------------

function _ka_rmsnorm!(b::KADeviceBackend, dst, x, scale; eps::Real, check_finite::Bool)
    isfinite(eps) && eps > 0 ||
        throw(gesso_error(ERR_INVALID_PLAN, "rmsnorm!: eps must be finite and positive"; op=:rmsnorm!))
    e32 = Float32(eps)
    isfinite(e32) && e32 > 0 || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "rmsnorm!: eps must be representable, finite and positive in storage dtype";
            op=:rmsnorm!,
        ),
    )
    xs = _ka_device_storage!(:rmsnorm!, b, x)
    ds = _ka_device_storage!(:rmsnorm!, b, dst)
    ws = _ka_device_storage!(:rmsnorm!, b, scale)
    ndims(xs) > 0 || throw(
        gesso_error(ERR_INVALID_PLAN, "rmsnorm!: input must have a nonempty feature dimension"; op=:rmsnorm!),
    )
    _ka_positive_dims(xs, :rmsnorm!)
    size(ds) == size(xs) || throw(
        gesso_error(ERR_INVALID_PLAN, "rmsnorm!: destination and input shapes must match"; op=:rmsnorm!),
    )
    d = size(xs, ndims(xs))
    length(ws) == d || throw(
        gesso_error(ERR_INVALID_PLAN, "rmsnorm!: scale length $(length(ws)) ≠ feature dim $d"; op=:rmsnorm!),
    )
    _ka_rmsnorm_rows!(b.ka, ds, xs, ws, eps; wg=_KA_OP_WG)
    if check_finite
        Gesso.Inference._all_finite(ds) || throw(
            gesso_error(
                ERR_NUMERICAL_INSTABILITY,
                "rmsnorm!: nonfinite normalization denominator";
            ),
        )
    end
    return dst
end

function rmsnorm!(
    b::KADeviceBackend,
    dst::Activation,
    x::Activation,
    scale::FrozenParameter,
    workload::Gesso.PrefillWorkload;
    eps::Real=1e-6,
)
    _ka_rmsnorm!(b, dst, x, scale; eps, check_finite=true)
    _ka_after_op!(b, workload)
    return dst
end

function rmsnorm!(
    b::KADeviceBackend,
    dst::Activation,
    x::Activation,
    scale::FrozenParameter,
    workload::Gesso.DecodeWorkload;
    eps::Real=1e-6,
)
    _ka_rmsnorm!(b, dst, x, scale; eps, check_finite=false)
    _ka_after_op!(b, workload)
    return dst
end

# --- RoPE -------------------------------------------------------------------------

function _ka_rope!(b::KADeviceBackend, q, k, positions; theta, inv_freq, interleaved, rotary_dim)
    qs = _ka_device_storage!(:rope!, b, q)
    ks = _ka_device_storage!(:rope!, b, k)
    Gesso._validate_rope_inputs(q, k, positions, theta, inv_freq, rotary_dim)
    size(qs, 3) == size(ks, 3) || throw(
        gesso_error(ERR_INVALID_PLAN, "rope!: q d_head ≠ k d_head"; op=:rope!),
    )
    all(p -> p <= typemax(Int32), positions) || throw(
        gesso_error(ERR_INVALID_PLAN, "rope!: positions exceed the device Int32 range"; op=:rope!),
    )
    d = size(qs, 3)
    rd = rotary_dim == 0 ? d : Int(rotary_dim)
    half = rd ÷ 2
    tθ = Float64(theta)
    freq_host = inv_freq === nothing ?
        Float64[tθ^(Float64(-2i) / rd) for i in 0:(half-1)] :
        Float64.(inv_freq)
    freq = _ka_upload_host(b.ka, Float64, freq_host)
    pos = _ka_upload_host(b.ka, Int32, Int32.(positions))
    for x in (qs, ks)
        n = size(x, 1) * size(x, 2) * half
        n == 0 && continue
        _ka_rope_kernel!(b.ka, _KA_OP_WG)(x, pos, freq, interleaved, half, size(x, 2); ndrange=n)
    end
    return q
end

function rope!(
    b::KADeviceBackend,
    q::Activation,
    k::Activation,
    positions::AbstractVector{Int},
    workload::_KA_WORKLOAD;
    theta::Real=10000.0,
    inv_freq=nothing,
    interleaved::Bool=true,
    rotary_dim::Integer=0,
)
    _ka_rope!(b, q, k, positions; theta, inv_freq, interleaved, rotary_dim)
    _ka_after_op!(b, workload)
    return q
end

# --- softmax ------------------------------------------------------------------

function _ka_softmax!(b::KADeviceBackend, dst, scores; check_finite::Bool)
    ds = _ka_device_storage!(:softmax!, b, dst)
    ss = _ka_device_storage!(:softmax!, b, scores)
    ndims(ss) == 2 && size(ds) == size(ss) || throw(
        gesso_error(ERR_INVALID_PLAN, "softmax!: destination and scores must be matching matrices"; op=:softmax!),
    )
    L, K = size(ss)
    L == 0 && return dst
    _ka_softmax_rows_kernel!(b.ka, _KA_OP_WG)(ds, ss, K, K - L; ndrange=L)
    if check_finite
        Gesso.Inference._all_finite(ds) || throw(
            gesso_error(ERR_NUMERICAL_INSTABILITY, "softmax!: no finite unmasked maximum"),
        )
    end
    return dst
end

function softmax!(
    b::KADeviceBackend,
    dst::TemporaryWorkspace,
    scores::TemporaryWorkspace,
    workload::Gesso.PrefillWorkload,
)
    _ka_softmax!(b, dst, scores; check_finite=true)
    _ka_after_op!(b, workload)
    return dst
end

function softmax!(
    b::KADeviceBackend,
    dst::TemporaryWorkspace,
    scores::TemporaryWorkspace,
    workload::Gesso.DecodeWorkload,
)
    _ka_softmax!(b, dst, scores; check_finite=false)
    _ka_after_op!(b, workload)
    return dst
end

# --- SwiGLU and GEGLU -----------------------------------------------------------

function _ka_elementwise_check(b::KADeviceBackend, op::Symbol, dst, a, c)
    ds = _ka_device_storage!(op, b, dst)
    as = _ka_device_storage!(op, b, a)
    cs = _ka_device_storage!(op, b, c)
    size(ds) == size(as) == size(cs) || throw(
        gesso_error(ERR_INVALID_PLAN, "$op: operand shapes must match"; op=op),
    )
    return ds, as, cs
end

function swiglu!(
    b::KADeviceBackend,
    dst::Activation,
    gate::Activation,
    up::Activation,
    workload::_KA_WORKLOAD,
)
    ds, gs, us = _ka_elementwise_check(b, :swiglu!, dst, gate, up)
    n = length(ds)
    n > 0 && _ka_swiglu_kernel!(b.ka, _KA_OP_WG)(ds, gs, us; ndrange=n)
    _ka_after_op!(b, workload)
    return dst
end

# GEGLU FFN activation: act = gelu(gate) .* up. Like the Lava path, it does not
# sync: the FFN caller synchronizes at its own boundary.
function Gesso.Inference._geglu!(b::KADeviceBackend, act, gate, up)
    ds, gs, us = _ka_elementwise_check(b, :geglu!, act, gate, up)
    n = length(ds)
    n > 0 && _ka_geglu_kernel!(b.ka, _KA_OP_WG)(ds, gs, us; ndrange=n)
    return act
end

# --- matmul -------------------------------------------------------------------
# Dense Float32 GEMM goes to the vendor BLAS on the device array (rocBLAS,
# expected and not device-measured) and to OpenBLAS on KA CPU, as the Lava path
# does. A projection kept quantized on device runs its GEMV in amdgpu_quant.jl.

function _ka_matmul!(b::KADeviceBackend, dst, x, w)
    ds = _ka_device_storage!(:matmul!, b, dst)
    xs = _ka_device_storage!(:matmul!, b, x)
    ws = _ka_device_storage!(:matmul!, b, w)
    ndims(ds) == ndims(xs) == ndims(ws) == 2 || throw(
        gesso_error(ERR_INVALID_PLAN, "matmul!: expected matrices"; op=:matmul!),
    )
    size(xs, 2) == size(ws, 2) && size(ds) == (size(xs, 1), size(ws, 1)) || throw(
        gesso_error(ERR_INVALID_PLAN, "matmul!: shape mismatch"; op=:matmul!),
    )
    mul!(ds, xs, transpose(ws), one(Float32), zero(Float32))
    return dst
end

function matmul!(
    b::KADeviceBackend,
    dst::Activation,
    x::Activation,
    w::Union{ProjectionWeight, EmbeddingTable},
    workload::_KA_WORKLOAD,
)
    if w isa ProjectionWeight && w.storage isa AmdDenseQuant
        _amdq_matmul_op!(b, dst, x, w)
    else
        _ka_matmul!(b, dst, x, w)
    end
    _ka_after_op!(b, workload)
    return dst
end
