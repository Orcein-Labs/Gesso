# Quantized dense GEMV on device (AMDGPU / KADeviceBackend).
#
# A supported GGUF projection stays quantized on device: the block bytes are
# uploaded as they are and decoded in the kernel with plain integer ops and
# Float32 math. A format without a tested kernel never reaches this file;
# to_device dequantizes it on the host to Float32 instead.
#
# Activations. Q8_0 and Q4_K weights take the CPU-order kernels, which quantize
# the activation on device first (Q8_0 blocks, and Q8_K-style blocks of 256 for
# Q4_K), the way ggml's CPU path does. Every other format keeps the activation
# in Float32 and accumulates in Float32 through a local-memory tree.

const _AMDQ_TYPES = (
    Gesso.Inference.GGML_TYPE_Q8_0,
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

_amdq_block_values(t::Int32) = t == Gesso.Inference.GGML_TYPE_Q8_0 ? 32 : 0

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
            amax = max(amax, abs(X[t, b*32+j]))
        end
        d = Float32(Float64(amax) / 127.0)
        inv = d != 0.0f0 ? Float32(1.0 / Float64(d)) : 0.0f0
        @inbounds D[t, b+1] = Float32(Float16(d))
        @inbounds for j in 1:32
            scaled = Float32(Float64(X[t, b*32+j]) * Float64(inv))
            a = abs(scaled)
            whole = unsafe_trunc(Int32, floor(a))
            rounded = whole + Int32(a - Float32(whole) >= 0.5f0)
            Q[t, b*32+j] = scaled < 0.0f0 ? -rounded : rounded
        end
    end
end

@kernel function _amdq_q8_dot_kernel!(
    Y,
    @Const(Q),
    @Const(D),
    @Const(W),
    M::Int,
    K::Int,
    row_bytes::Int,
    total::Int,
)
    lin = @index(Global, Linear)
    if lin <= total
        t = (lin - 1) % M + 1
        r = (lin - 1) ÷ M
        nb = K ÷ 32
        base = r * row_bytes
        acc = 0.0f0
        for b in 0:(nb-1)
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
        end
        @inbounds Y[t, r+1] = acc
    end
end

# --- launchers ------------------------------------------------------------------

function _amdq_q8_matmul!(Y, X, q::AmdDenseQuant; wg::Int=64)
    _ka_check_wg(wg)
    m, k = size(X)
    ka = KA.get_backend(X)
    total_act = m * (k ÷ 32)
    Q = KA.allocate(ka, Int32, (m, k))
    D = KA.allocate(ka, Float32, (m, k ÷ 32))
    _amdq_q8_act_pack_kernel!(ka, wg)(Q, D, X, m, k, total_act; ndrange=total_act)
    total = m * q.nrows
    _amdq_q8_dot_kernel!(ka, wg)(Y, Q, D, q.data, m, k, q.row_bytes, total; ndrange=total)
    return Y
end

# Dispatch: one entry for every supported format, so the op surface does not
# know which kernel a weight runs.
function _amdq_gemv!(Y, X, q::AmdDenseQuant; wg::Int=64)
    q.ggml_type == Gesso.Inference.GGML_TYPE_Q8_0 && return _amdq_q8_matmul!(Y, X, q; wg)
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

function matmul!(
    b::KADeviceBackend,
    dst::Gesso.Activation,
    x::Gesso.Activation,
    w::Gesso.ProjectionWeight,
    ::Gesso.PrefillWorkload,
)
    w.storage isa AmdDenseQuant || Gesso.lowering_not_implemented(:matmul!, b)
    return _amdq_matmul_op!(b, dst, x, w)
end

function matmul!(
    b::KADeviceBackend,
    dst::Gesso.Activation,
    x::Gesso.Activation,
    w::Gesso.ProjectionWeight,
    ::Gesso.DecodeWorkload,
)
    w.storage isa AmdDenseQuant || Gesso.lowering_not_implemented(:matmul!, b)
    return _amdq_matmul_op!(b, dst, x, w)
end
