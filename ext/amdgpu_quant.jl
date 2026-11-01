# Quantized dense GEMV on device (AMDGPU / KADeviceBackend).
#
# A supported GGUF projection stays quantized on device: the block bytes are
# uploaded as they are and decoded in the kernel with plain integer ops and
# Float32 math. A format without a tested kernel never reaches this file;
# to_device dequantizes it on the host to Float32 instead.
#
# Activations. Q8_0 and Q4_K quantize the activation on device first (Q8_0
# blocks, and Q8_K-style blocks of 256), the way ggml's CPU path does, then
# run one workgroup per (token, output row) with a local-memory tree reduction.
# The other K-quants (Q2_K, Q3_K, Q5_K, Q6_K, kernels in amdgpu_kquant.jl) keep
# the one-work-item-per-output CPU-order launch. The IQ formats
# (amdgpu_iquant.jl) keep the activation in Float32, one work item per output.

const _AMDQ_IQ_SUPERBLOCK_TYPES = (
    Gesso.Inference.GGML_TYPE_IQ4_XS,
    Gesso.Inference.GGML_TYPE_IQ2_XXS,
    Gesso.Inference.GGML_TYPE_IQ2_XS,
    Gesso.Inference.GGML_TYPE_IQ2_S,
    Gesso.Inference.GGML_TYPE_IQ3_XXS,
    Gesso.Inference.GGML_TYPE_IQ3_S,
)

const _AMDQ_TYPES = (
    Gesso.Inference.GGML_TYPE_Q8_0,
    Gesso.Inference.GGML_TYPE_Q2_K,
    Gesso.Inference.GGML_TYPE_Q3_K,
    Gesso.Inference.GGML_TYPE_Q4_K,
    Gesso.Inference.GGML_TYPE_Q5_K,
    Gesso.Inference.GGML_TYPE_Q6_K,
    Gesso.Inference.GGML_TYPE_IQ4_NL,
    Gesso.Inference.GGML_TYPE_IQ4_XS,
    Gesso.Inference.GGML_TYPE_IQ2_XXS,
    Gesso.Inference.GGML_TYPE_IQ2_XS,
    Gesso.Inference.GGML_TYPE_IQ2_S,
    Gesso.Inference.GGML_TYPE_IQ3_XXS,
    Gesso.Inference.GGML_TYPE_IQ3_S,
)

# Device-resident dense quant weight: `data` holds the GGUF block bytes of
# every row back to back, row r at byte (r-1)*row_bytes.
struct AmdDenseQuant{V<:AbstractVector{UInt8}}
    data::V
    ggml_type::Int32
    ne0::Int
    nrows::Int
    row_bytes::Int
end

Base.size(q::AmdDenseQuant) = (q.nrows, q.ne0)
Base.size(q::AmdDenseQuant, d::Integer) = size(q)[d]
Base.ndims(::AmdDenseQuant) = 2
Base.length(q::AmdDenseQuant) = q.nrows * q.ne0
Base.sizeof(q::AmdDenseQuant) = sizeof(q.data)

_amdq_block_values(t::Int32) =
    t == Gesso.Inference.GGML_TYPE_Q8_0 ? 32 :
    t == Gesso.Inference.GGML_TYPE_IQ4_NL ? 32 :
    t in (
        Gesso.Inference.GGML_TYPE_Q2_K,
        Gesso.Inference.GGML_TYPE_Q3_K,
        Gesso.Inference.GGML_TYPE_Q4_K,
        Gesso.Inference.GGML_TYPE_Q5_K,
        Gesso.Inference.GGML_TYPE_Q6_K,
    ) ? 256 :
    t in _AMDQ_IQ_SUPERBLOCK_TYPES ? 256 : 0

function _amdq_eligible(s)
    s isa Gesso.Inference.QuantMatrix || return false
    s.ggml_type in _AMDQ_TYPES || return false
    bv = _amdq_block_values(s.ggml_type)
    return s.ne0 % bv == 0 && s.row_bytes == Gesso.Inference._qrow_bytes(s.ggml_type, s.ne0)
end

function _amdq_upload(b::KADeviceBackend, s::Gesso.Inference.QuantMatrix)
    bytes = collect(Gesso.Inference._quant_bytes(s))
    dev = KA.allocate(b.ka, UInt8, length(bytes))
    copyto!(dev, bytes)
    return AmdDenseQuant(dev, s.ggml_type, s.ne0, s.nrows, s.row_bytes)
end

# --- element helpers (absolute 0-based byte offsets) ---------------------------

@inline _amdq_u8(W, i::Int) = Int32(W[i+1])
@inline _amdq_i8(W, i::Int) = Int32(reinterpret(Int8, W[i+1]))
@inline _amdq_f16(W, i::Int) =
    Float32(reinterpret(Float16, UInt16(W[i+1]) | (UInt16(W[i+2]) << 8)))

# --- Q8_0 (34-byte block: f16 scale, 32 int8) ---------------------------------
#
# The activation is quantized per (token, 32-block) on device, round half away
# from zero on the scaled magnitude, with the scale rounded through Float16 as
# the block format stores it. The dot runs integer products per block and folds
# each block's term into a Float32 accumulator in block order.

@kernel function _amdq_q8_act_pack_kernel!(Q, D, @Const(X), M::Int, K::Int, total::Int)
    lin = @index(Global, Linear)
    if lin <= total
        t = (lin - 1) % M + 1
        b = (lin - 1) ÷ M
        amax = 0.0f0
        @inbounds for j in 1:32
            a = abs(X[t, b*32+j])
            if a > amax
                amax = a
            end
        end
        d = amax / 127.0f0
        inv = d != 0.0f0 ? 1.0f0 / d : 0.0f0
        @inbounds D[t, b+1] = Float32(Float16(d))
        @inbounds for j in 1:32
            Q[t, b*32+j] = unsafe_trunc(Int32, round(X[t, b*32+j] * inv))
        end
    end
end

# One workgroup per (token, output row). Lane `lane` takes blocks lane-1,
# lane-1+wg, ... so neighbouring lanes read neighbouring 34-byte blocks of the
# same row. The lane partials fold through local memory; only that fold order
# depends on the workgroup size, the per-block integer sums do not.
@kernel function _amdq_q8_dot_kernel!(Y, @Const(Q), @Const(D), @Const(W), M::Int, K::Int, row_bytes::Int)
    wg = @uniform @groupsize()[1]
    part = @localmem Float32 (_KA_MAX_WG,)
    grp = @index(Group, Linear)
    lane = @index(Local, Linear)
    nb = K ÷ 32
    base = ((grp - 1) ÷ M) * row_bytes
    t = (grp - 1) % M + 1
    acc = 0.0f0
    b = lane - 1
    while b < nb
        blk = base + b * 34
        sumi = Int32(0)
        for j in 0:31
            wi = _amdq_i8(W, blk + 2 + j)
            sumi += wi * Q[t, b*32+j+1]
        end
        wd = _amdq_f16(W, blk)
        scale = Float32(Float64(wd) * Float64(D[t, b+1]))
        term = Float32(Float64(Float32(sumi)) * Float64(scale))
        acc = Float32(Float64(acc) + Float64(term))
        b += wg
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
    lane = @index(Local, Linear)
    grp = @index(Group, Linear)
    if lane == 1
        @inbounds Y[(grp - 1) % M + 1, (grp - 1) ÷ M + 1] = part[1]
    end
end

# --- Q4_K (144-byte block of 256: f16 d, f16 dmin, 12 packed scale/min bytes,
# 128 bytes of nibbles) and its Q8_K activation -----------------------------------
#
# The activation is quantized per (token, 256-block) with the scale taken from
# the signed element of largest magnitude, so the codes span [-127, 127]. The
# dot sums nibble*code and code per 32-element sub-block in integers, then folds
# each superblock into a Float32 accumulator, the way the CPU path orders it.

@kernel function _amdq_q8k_act_pack_kernel!(Q, D, @Const(X), M::Int, K::Int, total::Int)
    lin = @index(Global, Linear)
    if lin <= total
        t = (lin - 1) % M + 1
        b = (lin - 1) ÷ M
        amax = 0.0f0
        mx = 0.0f0
        @inbounds for j in 1:256
            v = X[t, b*256+j]
            if abs(v) > amax
                amax = abs(v)
                mx = v
            end
        end
        iscale = amax == 0.0f0 ? 0.0f0 : -127.0f0 / mx
        @inbounds D[t, b+1] = iscale == 0.0f0 ? 0.0f0 : 1.0f0 / iscale
        @inbounds for j in 1:256
            Q[t, b*256+j] = min(Int32(127), unsafe_trunc(Int32, round(iscale * X[t, b*256+j])))
        end
    end
end

# Scale and min of sub-block j (0..7) from the 12 packed bytes at offset s.
@inline function _amdq_scale_min(W, s::Int, j::Int)
    if j < 4
        sc = _amdq_u8(W, s + j) & Int32(63)
        mn = _amdq_u8(W, s + j + 4) & Int32(63)
    else
        hi_sc = _amdq_u8(W, s + j + 4)
        sc = (hi_sc & Int32(15)) | ((_amdq_u8(W, s + j - 4) >> 6) << 4)
        mn = (hi_sc >> 4) | ((_amdq_u8(W, s + j) >> 6) << 4)
    end
    return sc, mn
end

# One workgroup per (token, output row), lanes stride over superblocks as in the
# Q8_0 kernel above.
@kernel function _amdq_q4k_dot_kernel!(Y, @Const(Q), @Const(D), @Const(W), M::Int, K::Int, row_bytes::Int)
    wg = @uniform @groupsize()[1]
    part = @localmem Float32 (_KA_MAX_WG,)
    grp = @index(Group, Linear)
    lane = @index(Local, Linear)
    nsb = K ÷ 256
    base = ((grp - 1) ÷ M) * row_bytes
    t = (grp - 1) % M + 1
    acc = 0.0f0
    sb = lane - 1
    while sb < nsb
        blk = base + sb * 144
        isc = Int32(0)
        imn = Int32(0)
        for g in 0:7
            sc, mn = _amdq_scale_min(W, blk + 4, g)
            sub = Int32(0)
            bs = Int32(0)
            for l in 0:31
                byte = _amdq_u8(W, blk + 16 + 32 * (g >> 1) + l)
                nib = iseven(g) ? (byte & Int32(15)) : (byte >> 4)
                qv = Q[t, sb*256+g*32+l+1]
                sub += nib * qv
                bs += qv
            end
            isc += sc * sub
            imn += mn * bs
        end
        dd = _amdq_f16(W, blk) * D[t, sb+1]
        dm = _amdq_f16(W, blk + 2) * D[t, sb+1]
        acc += dd * Float32(isc) - dm * Float32(imn)
        sb += wg
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
    lane = @index(Local, Linear)
    grp = @index(Group, Linear)
    if lane == 1
        @inbounds Y[(grp - 1) % M + 1, (grp - 1) ÷ M + 1] = part[1]
    end
end

# --- launchers ------------------------------------------------------------------

# Quantize the activation on device with `pack` (blocks of `bv` values), then run
# `dot` over every (token, output row). The dot kernels share one argument list.
function _amdq_cpu_order_matmul!(Y, X, q::AmdDenseQuant, pack, dot, bv::Int; wg::Int)
    _ka_check_wg(wg)
    m, k = size(X)
    ka = KA.get_backend(X)
    nblk = k ÷ bv
    total_act = m * nblk
    Q = KA.allocate(ka, Int32, (m, k))
    D = KA.allocate(ka, Float32, (m, nblk))
    pack(ka, wg)(Q, D, X, m, k, total_act; ndrange=total_act)
    total = m * q.nrows
    dot(ka, wg)(Y, Q, D, q.data, m, k, q.row_bytes, total; ndrange=total)
    return Y
end

# Q8_0 and Q4_K: quantize the activation with `pack`, then run `dot` with one
# workgroup of `wg` lanes per (token, output row).
function _amdq_wg_matmul!(Y, X, q::AmdDenseQuant, pack, dot, bv::Int; wg::Int)
    _ka_check_wg(wg)
    m, k = size(X)
    ka = KA.get_backend(X)
    nblk = k ÷ bv
    total_act = m * nblk
    Q = KA.allocate(ka, Int32, (m, k))
    D = KA.allocate(ka, Float32, (m, nblk))
    pack(ka, wg)(Q, D, X, m, k, total_act; ndrange=total_act)
    dot(ka, wg)(Y, Q, D, q.data, m, k, q.row_bytes; ndrange=wg * m * q.nrows)
    return Y
end

_amdq_q8_matmul!(Y, X, q::AmdDenseQuant; wg::Int=64) =
    _amdq_wg_matmul!(Y, X, q, _amdq_q8_act_pack_kernel!, _amdq_q8_dot_kernel!, 32; wg)

_amdq_q4k_matmul!(Y, X, q::AmdDenseQuant; wg::Int=64) =
    _amdq_wg_matmul!(Y, X, q, _amdq_q8k_act_pack_kernel!, _amdq_q4k_dot_kernel!, 256; wg)

# Dispatch: one entry for every supported format, so the op surface does not
# know which kernel a weight runs.
function _amdq_gemv!(Y, X, q::AmdDenseQuant; wg::Int=64)
    q.ggml_type == Gesso.Inference.GGML_TYPE_Q8_0 && return _amdq_q8_matmul!(Y, X, q; wg)
    q.ggml_type == Gesso.Inference.GGML_TYPE_Q4_K && return _amdq_q4k_matmul!(Y, X, q; wg)
    q.ggml_type == Gesso.Inference.GGML_TYPE_Q2_K && return _amdq_q2k_matmul!(Y, X, q; wg)
    q.ggml_type == Gesso.Inference.GGML_TYPE_Q3_K && return _amdq_q3k_matmul!(Y, X, q; wg)
    q.ggml_type == Gesso.Inference.GGML_TYPE_Q5_K && return _amdq_q5k_matmul!(Y, X, q; wg)
    q.ggml_type == Gesso.Inference.GGML_TYPE_Q6_K && return _amdq_q6k_matmul!(Y, X, q; wg)
    q.ggml_type == Gesso.Inference.GGML_TYPE_IQ4_NL && return _amdq_iq4nl_matmul!(Y, X, q; wg)
    q.ggml_type in _AMDQ_IQ_SUPERBLOCK_TYPES && return _amdq_iq_matmul!(Y, X, q; wg)
    throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "amdgpu quant GEMV: no kernel for ggml type $(q.ggml_type)";
            ggml_type=q.ggml_type,
        ),
    )
end

# dst[t, r] = sum_e x[t, e] * W[r, e] for a device quant weight.
function _amdq_matmul_op!(b::KADeviceBackend, dst::Gesso.Activation, x::Gesso.Activation, w::Gesso.ProjectionWeight)
    _ka_device_storage!(:matmul!, b, dst)
    _ka_device_storage!(:matmul!, b, x)
    q = w.storage
    ntok = size(x.storage, 1)
    size(x.storage, 2) == q.ne0 || throw(
        DimensionMismatch("amdgpu quant GEMV: x has $(size(x.storage, 2)) columns, weight has $(q.ne0)"),
    )
    size(dst.storage) == (ntok, q.nrows) || throw(
        DimensionMismatch("amdgpu quant GEMV: dst is $(size(dst.storage)), want $((ntok, q.nrows))"),
    )
    _amdq_gemv!(dst.storage, x.storage, q)
    return dst
end
