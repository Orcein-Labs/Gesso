# Lava quant GEMV: the GGUF bytes of a projection or an expert bank stay on the
# device and the matmul dequantizes them inside the kernel. The CUDA path
# (cuda_quant_dense.jl, cuda_quant_gemm.jl) quantizes the activation to Q8_1 and
# runs an integer dot with dp4a; Lava has no dp4a, and this port keeps the
# activation in F32, so every product is exact up to F32 accumulation. The
# weight decode follows the same GGML block layouts as the CUDA kernels.

using Lava.KernelAbstractions: @kernel, @index, @localmem, @synchronize

# Kind codes: the element decoder branches on these at run time, so one
# kernel covers every supported type.
const _LQ_KIND_Q4_K = 1
const _LQ_KIND_Q5_K = 2
const _LQ_KIND_Q6_K = 3
const _LQ_KIND_IQ4_XS = 4
const _LQ_KIND_Q8_0 = 5
const _LQ_KIND_F32 = 6
const _LQ_KIND_IQ2_XS = 7
const _LQ_KIND_IQ3_XXS = 8

# Lanes per output row. Fixed at compile time: Lava bakes the SPIR-V
# LocalSize from workgroupsize, and a shape-dependent size would recompile
# the pipeline for every projection.
const _LQ_LANES = 64

# The IQ4_NL codebook used by IQ4_XS (ggml kvalues_iq4nl).
const _LQ_KVALUES_IQ4NL = Float32[-127, -104, -83, -65, -49, -35, -22, -10,
                                  1, 13, 25, 38, 53, 69, 89, 113]

# Device-resident dense quant weight. `data` holds rows 1..nrows of the GGUF
# file in block order, row r starting at byte (r-1)*row_bytes.
struct LavaDenseQuant
    data::Lava.LavaArray{UInt8}
    ggml_type::Int32
    kind::Int
    ne0::Int
    nrows::Int
    row_bytes::Int
end

Base.size(q::LavaDenseQuant) = (q.nrows, q.ne0)
Base.size(q::LavaDenseQuant, d::Integer) = size(q)[d]
Base.ndims(::LavaDenseQuant) = 2
Base.length(q::LavaDenseQuant) = q.nrows * q.ne0
Base.sizeof(q::LavaDenseQuant) = sizeof(q.data)

function _lq_kind(t::Int32)
    I = Gesso.Inference
    t == I.GGML_TYPE_Q4_K && return _LQ_KIND_Q4_K
    t == I.GGML_TYPE_Q5_K && return _LQ_KIND_Q5_K
    t == I.GGML_TYPE_Q6_K && return _LQ_KIND_Q6_K
    t == I.GGML_TYPE_IQ4_XS && return _LQ_KIND_IQ4_XS
    t == I.GGML_TYPE_Q8_0 && return _LQ_KIND_Q8_0
    return nothing
end

# Expert banks cover the same types as the CUDA expert path.
function _lq_expert_kind(t::Int32)
    I = Gesso.Inference
    t == I.GGML_TYPE_F32 && return _LQ_KIND_F32
    t == I.GGML_TYPE_IQ2_XS && return _LQ_KIND_IQ2_XS
    t == I.GGML_TYPE_IQ3_XXS && return _LQ_KIND_IQ3_XXS
    t == I.GGML_TYPE_IQ4_XS && return _LQ_KIND_IQ4_XS
    return nothing
end

# Same rule as the CUDA side: K-quant superblocks are 256 values, Q8_0 blocks 32.
function _lq_eligible(s)
    s isa Gesso.Inference.QuantMatrix || return false
    kind = _lq_kind(s.ggml_type)
    kind === nothing && return false
    s.ne0 % 32 == 0 || return false
    kind == _LQ_KIND_Q8_0 || s.ne0 % 256 == 0 || return false
    return true
end

_lq_device_bytes(s::Gesso.Inference.QuantMatrix) = Int64(s.nrows) * Int64(s.row_bytes)

function _lq_upload(s::Gesso.Inference.QuantMatrix)
    bytes = Lava.LavaArray{UInt8}(collect(Gesso.Inference._quant_bytes(s)))
    return LavaDenseQuant(bytes, s.ggml_type, _lq_kind(s.ggml_type), s.ne0, s.nrows, s.row_bytes)
end

# --- element decode -----------------------------------------------------------
# Every reader takes an absolute byte index and returns Int32, UInt32 or
# Float32. The index is 0-based, so `W[i + 1]` is the byte at offset `i`.

@inline _lq_u8(W, i::Int) = Int32(W[i + 1])
@inline _lq_u16(W, i::Int) = UInt32(W[i + 1]) | (UInt32(W[i + 2]) << 8)
@inline _lq_u32(W, i::Int) = UInt32(W[i + 1]) | (UInt32(W[i + 2]) << 8) |
                             (UInt32(W[i + 3]) << 16) | (UInt32(W[i + 4]) << 24)
@inline _lq_f16(W, i::Int) = Float32(reinterpret(Float16, UInt16(W[i + 1]) | (UInt16(W[i + 2]) << 8)))
@inline _lq_f32(W, i::Int) = reinterpret(Float32, _lq_u32(W, i))

# get_scale_min_k4: the 6-bit scale and minimum of sub-block j (0..7) of a
# K-quant superblock whose 12 scale bytes start at byte `s`.
@inline function _lq_scale_min(W, s::Int, j::Int)
    if j < 4
        return (_lq_u8(W, s + j) & Int32(63), _lq_u8(W, s + j + 4) & Int32(63))
    end
    sc = (_lq_u8(W, s + j + 4) & Int32(15)) | ((_lq_u8(W, s + j - 4) >> 6) << 4)
    mn = (_lq_u8(W, s + j + 4) >> 4) | ((_lq_u8(W, s + j) >> 6) << 4)
    return (sc, mn)
end

# Q4_K (144-byte superblock): d, dmin f16, 12 scale bytes, 128 nibble bytes.
# Chunk j (64 values) reads bytes 32j..32j+31; low nibbles give the first 32
# values, high nibbles the next 32.
@inline function _lq_q4k(W, b::Int, e::Int)
    d = _lq_f16(W, b)
    dmin = _lq_f16(W, b + 2)
    j = e >> 6
    r = e & 63
    hi = r >> 5
    l = r & 31
    sc, mn = _lq_scale_min(W, b + 4, 2 * j + hi)
    byte = _lq_u8(W, b + 16 + 32 * j + l)
    nib = hi == 0 ? (byte & Int32(15)) : (byte >> 4)
    return d * Float32(sc) * Float32(nib) - dmin * Float32(mn)
end

# Q5_K (176-byte superblock): as Q4_K, plus a 32-byte qh plane at byte 16
# that contributes bit 2j+hi as the fifth value bit.
@inline function _lq_q5k(W, b::Int, e::Int)
    d = _lq_f16(W, b)
    dmin = _lq_f16(W, b + 2)
    j = e >> 6
    r = e & 63
    hi = r >> 5
    l = r & 31
    sc, mn = _lq_scale_min(W, b + 4, 2 * j + hi)
    byte = _lq_u8(W, b + 48 + 32 * j + l)
    nib = hi == 0 ? (byte & Int32(15)) : (byte >> 4)
    bit = (_lq_u8(W, b + 16 + l) >> (2 * j + hi)) & Int32(1)
    q = nib + (bit << 4)
    return d * Float32(sc) * Float32(q) - dmin * Float32(mn)
end

# Q6_K (210-byte superblock): ql 128 bytes, qh 64 bytes, 16 int8 scales, d.
# Each half (128 values) has four 32-value quarters; the low nibble of
# ql[l] or ql[l+32] and two bits of qh[l] form a 6-bit value, centred at 32.
@inline function _lq_q6k(W, b::Int, e::Int)
    d = _lq_f16(W, b + 208)
    h = e >> 7
    r = e & 127
    quad = r >> 5
    l = r & 31
    byte = _lq_u8(W, b + 64 * h + l + (quad & 1) * 32)
    nib = quad < 2 ? (byte & Int32(15)) : (byte >> 4)
    hb = (_lq_u8(W, b + 128 + 32 * h + l) >> (2 * quad)) & Int32(3)
    q = (nib | (hb << 4)) - Int32(32)
    sc = Int32(reinterpret(Int8, W[b + 192 + 8 * h + (l >> 4) + 2 * quad + 1]))
    return d * Float32(sc) * Float32(q)
end

# IQ4_XS (136-byte superblock): d f16, 16-bit scales_h, 4 scales_l bytes, 128
# nibble bytes. Sub-block ib (32 values) has a 6-bit signed scale and indexes
# the IQ4_NL codebook with its nibbles (low nibbles first 16, high nibbles next 16).
@inline function _lq_iq4xs(W, b::Int, e::Int, kv)
    d = _lq_f16(W, b)
    sh = _lq_u8(W, b + 2) | (_lq_u8(W, b + 3) << 8)
    ib = e >> 5
    r = e & 31
    sl = _lq_u8(W, b + 4 + (ib >> 1))
    ls = ((sl >> (4 * (ib & 1))) & Int32(15)) | (((sh >> (2 * ib)) & Int32(3)) << 4)
    dl = d * Float32(ls - Int32(32))
    qb = _lq_u8(W, b + 8 + 16 * ib + (r & 15))
    nib = r < 16 ? (qb & Int32(15)) : (qb >> 4)
    return dl * kv[Int(nib) + 1]
end

# IQ2_XS (74-byte superblock): d f16, 32 u16 entries (9-bit grid index, 7-bit
# sign index), 8 scale bytes. Each 8-value group takes its scale nibble from
# the half of the sub-block it sits in. The 64-bit grid entry is read as two
# UInt32 words because the device path has no UInt64 buffers.
@inline function _lq_iq2xs(W, b::Int, e::Int, g2, ks)
    d = _lq_f16(W, b)
    ib32 = e >> 5
    r = e & 31
    l = r >> 3
    t = r & 7
    qv = _lq_u16(W, b + 2 + 2 * (4 * ib32 + l))
    gi = Int(qv & UInt32(0x01ff))
    signs = UInt32(ks[Int(qv >> 9) + 1])
    sc = UInt32(W[b + 66 + ib32 + 1])
    s4 = l < 2 ? (sc & UInt32(0x0f)) : (sc >> 4)
    db = d * (0.5f0 + Float32(s4)) * 0.25f0
    word = t < 4 ? g2[2 * gi + 1] : g2[2 * gi + 2]
    g = (word >> (8 * (t & 3))) & UInt32(0xff)
    sgn = ((signs >> t) & UInt32(1)) == UInt32(1) ? -1.0f0 : 1.0f0
    return db * Float32(g) * sgn
end

# IQ3_XXS (98-byte superblock): d f16, 64 grid bytes, then per sub-block a u32
# holding a 4-bit scale (top bits) and four 7-bit sign indices. Values 0..3 of
# a group come from the first grid byte, 4..7 from the second.
@inline function _lq_iq3xxs(W, b::Int, e::Int, g3, ks)
    d = _lq_f16(W, b)
    ib32 = e >> 5
    r = e & 31
    l = r >> 3
    u = r & 7
    aux = _lq_u32(W, b + 66 + 4 * ib32)
    db = d * (0.5f0 + Float32(aux >> 28)) * 0.5f0
    signs = UInt32(ks[Int((aux >> (7 * l)) & UInt32(0x7f)) + 1])
    qbase = b + 2 + 8 * ib32 + 2 * l
    gi = Int(W[(u < 4 ? qbase : qbase + 1) + 1])
    word = g3[gi + 1]
    g = (word >> (8 * (u & 3))) & UInt32(0xff)
    sgn = ((signs >> u) & UInt32(1)) == UInt32(1) ? -1.0f0 : 1.0f0
    return db * Float32(g) * sgn
end

# Q8_0 (34-byte block): f16 d, 32 int8 values.
@inline function _lq_q8_0(W, b::Int, e::Int)
    d = _lq_f16(W, b)
    q = Float32(reinterpret(Int8, W[b + 2 + (e & 31) + 1]))
    return d * q
end

# One weight element at 0-based position e within a row starting at byte rowoff.
@inline function _lq_elem(kind::Int, W, kv, g2, g3, ks, rowoff::Int, e::Int)
    if kind == _LQ_KIND_F32
        return _lq_f32(W, rowoff + 4 * e)
    elseif kind == _LQ_KIND_Q8_0
        return _lq_q8_0(W, rowoff + (e >> 5) * 34, e)
    end
    sb = e >> 8
    es = e & 255
    if kind == _LQ_KIND_Q4_K
        return _lq_q4k(W, rowoff + sb * 144, es)
    elseif kind == _LQ_KIND_Q5_K
        return _lq_q5k(W, rowoff + sb * 176, es)
    elseif kind == _LQ_KIND_Q6_K
        return _lq_q6k(W, rowoff + sb * 210, es)
    elseif kind == _LQ_KIND_IQ2_XS
        return _lq_iq2xs(W, rowoff + sb * 74, es, g2, ks)
    elseif kind == _LQ_KIND_IQ3_XXS
        return _lq_iq3xxs(W, rowoff + sb * 98, es, g3, ks)
    else
        return _lq_iq4xs(W, rowoff + sb * 136, es, kv)
    end
end

# --- GEMV ----------------------------------------------------------------------
# One workgroup of _LQ_LANES lanes per output row. Lane l accumulates elements
# l, l+64, ... of the row for each token, the partial sums meet in local
# memory, and lane 1 writes Y[t, row]. X and Y are indexed linearly (column
# major, token fastest): reshaped LavaArrays cannot be indexed inside a kernel.
# The row r starts at byte base + (r-1)*row_bytes, so an expert bank passes its
# expert's byte offset as base.

@kernel cpu=false function _lq_gemv_kernel!(Y, W, X, kv, g2, g3, ks, kind::Int, base::Int,
                                            ntok::Int, ne0::Int, row_bytes::Int,
                                            alpha::Float32, beta::Float32)
    tl = @index(Local, Linear)
    r = @index(Group, Linear)
    # One slot per subgroup; the subgroup size is only known at run time.
    part = @localmem Float32 (_LQ_LANES,)
    sgs = Lava.subgroup_size()
    sgid = (tl - 1) ÷ sgs
    nsg = _LQ_LANES ÷ sgs
    lane0 = Lava.subgroup_lane() == 0
    rowoff = base + (r - 1) * row_bytes
    for t in 1:ntok
        acc = 0.0f0
        e = tl - 1
        while e < ne0
            acc += _lq_elem(kind, W, kv, g2, g3, ks, rowoff, e) * X[e * ntok + t]
            e += _LQ_LANES
        end
        # subgroup_add must run with every lane active, so it sits outside the branch.
        red = Lava.subgroup_add(acc)
        if lane0
            part[sgid + 1] = red
        end
        @synchronize
        if tl == 1
            s = 0.0f0
            for i in 1:nsg
                s += part[i]
            end
            idx = (r - 1) * ntok + t
            # beta == 0 must not read Y: the caller may hand over uninitialized memory.
            Y[idx] = beta == 0.0f0 ? alpha * s : alpha * s + beta * Y[idx]
        end
        @synchronize
    end
end

# Device-resident lookup tables, uploaded once. g2 holds each IQ2_XS grid entry
# as two UInt32 words (low, then high); g3 is the IQ3_XXS grid; ks the sign table.
const _LQ_TABLES = Ref{Any}(nothing)
function _lq_tables()
    tb = _LQ_TABLES[]
    if tb === nothing
        I = Gesso.Inference
        g2 = Vector{UInt32}(undef, 2 * length(I.IQ2XS_GRID))
        for i in eachindex(I.IQ2XS_GRID)
            g2[2i - 1] = UInt32(I.IQ2XS_GRID[i] & 0xffffffff)
            g2[2i] = UInt32(I.IQ2XS_GRID[i] >> 32)
        end
        tb = (
            kv=Lava.LavaArray{Float32}(_LQ_KVALUES_IQ4NL),
            g2=Lava.LavaArray{UInt32}(g2),
            g3=Lava.LavaArray{UInt32}(collect(I.IQ3XXS_GRID)),
            ks=Lava.LavaArray{UInt8}(collect(I.KSIGNS_IQ2XS)),
        )
        _LQ_TABLES[] = tb
    end
    return tb
end

function _lq_launch!(Y, X, W, base::Int, kind::Int, ntok::Int, ne0::Int, nrows::Int,
                     row_bytes::Int, alpha::Float32, beta::Float32)
    tb = _lq_tables()
    kern = _lq_gemv_kernel!(Lava.KernelAbstractions.get_backend(X))
    kern(Y, W, X, tb.kv, tb.g2, tb.g3, tb.ks, kind, base, ntok, ne0, row_bytes, alpha, beta;
        ndrange=_LQ_LANES * nrows, workgroupsize=(_LQ_LANES,))
    return Y
end

function _lq_matmul!(Y, X, q::LavaDenseQuant)
    ntok = size(X, 1)
    size(X, 2) == q.ne0 || throw(DimensionMismatch("lava quant GEMV: x has $(size(X, 2)) columns, weight has $(q.ne0)"))
    size(Y) == (ntok, q.nrows) || throw(DimensionMismatch("lava quant GEMV: dst is $(size(Y)), want $((ntok, q.nrows))"))
    return _lq_launch!(Y, X, q.data, 0, q.kind, ntok, q.ne0, q.nrows, q.row_bytes, 1.0f0, 0.0f0)
end

# Single-token expert GEMV: y = alpha * W_e * x + beta * y, with W_e the
# expert's (ne1 x ne0) block of the device bank. Same contract as the CUDA
# method in cuda_quant_gemm.jl.
function Gesso.Inference._qgemv_expert!(
    y::Lava.LavaArray{Float32,1},
    x::Lava.LavaArray{Float32,1},
    bank::Gesso.Inference.GpuQuantSlice{<:Lava.LavaArray},
    e::Int;
    alpha::Float32=1.0f0,
    beta::Float32=0.0f0,
)
    I = Gesso.Inference
    kind = _lq_expert_kind(bank.ggml_type)
    kind === nothing && error("lava qgemv: no expert kernel for ggml type $(bank.ggml_type)")
    length(x) == bank.ne0 || throw(DimensionMismatch("lava expert GEMV: x has $(length(x)) elements, bank has $(bank.ne0) columns"))
    length(y) == bank.ne1 || throw(DimensionMismatch("lava expert GEMV: y has $(length(y)) elements, bank has $(bank.ne1) rows"))
    kind == _LQ_KIND_F32 || bank.ne0 % 256 == 0 ||
        error("lava expert GEMV: ne0 = $(bank.ne0) is not a multiple of the 256-value superblock")
    row_bytes = bank.bytes_per_expert ÷ bank.ne1
    base = I._quant_expert_off0(bank, e)
    _lq_launch!(y, x, bank.data, base, kind, 1, bank.ne0, bank.ne1, row_bytes, alpha, beta)
    return y
end
