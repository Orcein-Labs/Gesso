# Dense GEMV for the IQ GGUF formats on the AMDGPU (KADeviceBackend) backend.
#
# The activation stays Float32, as in the Lava IQ path, because the IQ codebooks
# are not on a Q8_K grid and the tested reference is the unquantized activation.
# A 256-value superblock is 16 units of 16 values; each unit shares one Float32
# scale, so a unit is a Float32 sum times that scale and the superblock is the
# sum of its units. IQ4_NL is a 32-value block with one f16 scale.
#
# The codebooks and grids are uploaded per launch and passed as kernel
# arguments; a kernel never closes over a host array.

# Kind codes the 256-superblock dot receives.
#   1 IQ4_XS, 2 IQ2_XS, 3 IQ2_XXS, 4 IQ2_S, 5 IQ3_XXS, 6 IQ3_S
function _amdq_iq_kind(t::Int32)
    t == Gesso.Inference.GGML_TYPE_IQ4_XS && return 1
    t == Gesso.Inference.GGML_TYPE_IQ2_XS && return 2
    t == Gesso.Inference.GGML_TYPE_IQ2_XXS && return 3
    t == Gesso.Inference.GGML_TYPE_IQ2_S && return 4
    t == Gesso.Inference.GGML_TYPE_IQ3_XXS && return 5
    t == Gesso.Inference.GGML_TYPE_IQ3_S && return 6
    throw(gesso_error(ERR_INVALID_PLAN, "amdgpu IQ GEMV: no kernel for ggml type $t"; ggml_type=t))
end

@inline _amdq_u16(W, i::Int) = UInt32(W[i+1]) | (UInt32(W[i+2]) << 8)
@inline _amdq_u32(W, i::Int) =
    UInt32(W[i+1]) | (UInt32(W[i+2]) << 8) | (UInt32(W[i+3]) << 16) | (UInt32(W[i+4]) << 24)

# F32 scale of unit u (0..15) of the superblock at byte blk. Units 2i and 2i+1
# share sub-block i; the formats whose sub-block scale covers two halves pick the
# half by parity of u.
@inline function _amdq_iq_unit_scale(kind::Int, W, blk::Int, u::Int)::Float32
    ib = u >> 1
    hi = (u & 1) == 1
    d = _amdq_f16(W, blk)
    if kind == 1
        sh = _amdq_u16(W, blk + 2)
        sl = _amdq_u8(W, blk + 4 + (ib >> 1))
        ls = ((sl >> (4 * (ib & 1))) & Int32(15)) | (((Int32(sh) >> (2 * ib)) & Int32(3)) << 4)
        return d * Float32(ls - Int32(32))
    elseif kind == 2
        sc = _amdq_u8(W, blk + 66 + ib)
        s4 = hi ? (sc >> 4) : (sc & Int32(15))
        return d * (0.5f0 + Float32(s4)) * 0.25f0
    elseif kind == 3
        aux = _amdq_u32(W, blk + 2 + 8 * ib + 4)
        return d * (0.5f0 + Float32(aux >> 28)) * 0.25f0
    elseif kind == 4
        sc = _amdq_u8(W, blk + 74 + ib)
        s4 = hi ? (sc >> 4) : (sc & Int32(15))
        return d * (0.5f0 + Float32(s4)) * 0.25f0
    elseif kind == 5
        aux = _amdq_u32(W, blk + 66 + 4 * ib)
        return d * (0.5f0 + Float32(aux >> 28)) * 0.5f0
    else
        scb = _amdq_u8(W, blk + 106 + (ib >> 1))
        s4 = iseven(ib) ? (scb & Int32(15)) : (scb >> 4)
        return d * Float32(2 * s4 + Int32(1))
    end
end

# Integer weight of element e (0..255) of the superblock at byte blk, sign applied.
# The 64-bit grid entries are stored as two UInt32 words, low then high, which is
# how the device table lays them out.
@inline function _amdq_iq_ival(kind::Int, W, blk::Int, e::Int, G2, G3, KS, KV)::Int32
    ib32 = e >> 5
    r = e & 31
    l = r >> 3
    t = r & 7
    if kind == 1
        qb = _amdq_u8(W, blk + 8 + 16 * ib32 + (e & 15))
        nib = (e & 16) == 0 ? (qb & Int32(15)) : (qb >> 4)
        return KV[Int(nib)+1]
    elseif kind == 2
        qv = _amdq_u16(W, blk + 2 + 2 * (4 * ib32 + l))
        gi = Int(qv & UInt32(0x01ff))
        signs = UInt32(KS[Int(qv >> 9)+1])
        word = t < 4 ? G2[2*gi+1] : G2[2*gi+2]
        g = Int32((word >> (8 * (t & 3))) & UInt32(0xff))
        return ((signs >> t) & UInt32(1)) == UInt32(1) ? -g : g
    elseif kind == 3
        q = blk + 2 + 8 * ib32
        aux = _amdq_u32(W, q + 4)
        gi = Int(W[q+l+1]) + 512
        signs = UInt32(KS[Int((aux >> (7 * l)) & UInt32(0x7f))+1])
        word = t < 4 ? G2[2*gi+1] : G2[2*gi+2]
        g = Int32((word >> (8 * (t & 3))) & UInt32(0xff))
        return ((signs >> t) & UInt32(1)) == UInt32(1) ? -g : g
    elseif kind == 4
        qh = _amdq_u8(W, blk + 66 + ib32)
        gi = Int(_amdq_u8(W, blk + 2 + 4 * ib32 + l)) | Int((qh << (8 - 2 * l)) & Int32(0x300))
        sb = UInt32(_amdq_u8(W, blk + 34 + 4 * ib32 + l))
        word = t < 4 ? G2[2*(gi+768)+1] : G2[2*(gi+768)+2]
        g = Int32((word >> (8 * (t & 3))) & UInt32(0xff))
        return ((sb >> t) & UInt32(1)) == UInt32(1) ? -g : g
    elseif kind == 5
        aux = _amdq_u32(W, blk + 66 + 4 * ib32)
        signs = UInt32(KS[Int((aux >> (7 * l)) & UInt32(0x7f))+1])
        gi = Int(W[blk+2+8*ib32+2*l+(t < 4 ? 0 : 1)+1])
        word = G3[gi+1]
        g = Int32((word >> (8 * (t & 3))) & UInt32(0xff))
        return ((signs >> t) & UInt32(1)) == UInt32(1) ? -g : g
    else
        qh = _amdq_u8(W, blk + 66 + ib32)
        shift = t < 4 ? (8 - 2 * l) : (7 - 2 * l)
        gbyte = blk + 2 + 8 * ib32 + 2 * l + (t < 4 ? 0 : 1)
        gi = Int(_amdq_u8(W, gbyte)) | Int((qh << shift) & Int32(256))
        sb = UInt32(_amdq_u8(W, blk + 74 + 4 * ib32 + l))
        word = G3[(gi+256)+1]
        g = Int32((word >> (8 * (t & 3))) & UInt32(0xff))
        return ((sb >> t) & UInt32(1)) == UInt32(1) ? -g : g
    end
end

@kernel function _amdq_iq_dot_kernel!(
    Y,
    @Const(X),
    @Const(W),
    M::Int,
    K::Int,
    row_bytes::Int,
    total::Int,
    kind::Int,
    @Const(KV),
    @Const(G2),
    @Const(G3),
    @Const(KS),
)
    lin = @index(Global, Linear)
    if lin <= total
        t = (lin - 1) % M + 1
        r = (lin - 1) ÷ M
        nsb = K ÷ 256
        bs = row_bytes ÷ nsb
        base = r * row_bytes
        acc = 0.0f0
        for sb in 0:(nsb-1)
            blk = base + sb * bs
            for u in 0:15
                sc = _amdq_iq_unit_scale(kind, W, blk, u)
                s = 0.0f0
                for l in 0:15
                    e = 16 * u + l
                    s += Float32(_amdq_iq_ival(kind, W, blk, e, G2, G3, KS, KV)) * X[t, sb*256+e+1]
                end
                acc += sc * s
            end
        end
        @inbounds Y[t, r+1] = acc
    end
end

# IQ4_NL: 18-byte block of 32 values, f16 d and 16 nibble bytes. Low nibbles give
# the first 16 values, high nibbles the next 16, both through the IQ4_NL codebook.
@kernel function _amdq_iq4nl_dot_kernel!(
    Y,
    @Const(X),
    @Const(W),
    M::Int,
    K::Int,
    row_bytes::Int,
    total::Int,
    @Const(KV),
)
    lin = @index(Global, Linear)
    if lin <= total
        t = (lin - 1) % M + 1
        r = (lin - 1) ÷ M
        nb = K ÷ 32
        base = r * row_bytes
        acc = 0.0f0
        for b in 0:(nb-1)
            blk = base + b * 18
            d = _amdq_f16(W, blk)
            s = 0.0f0
            for j in 0:31
                qb = _amdq_u8(W, blk + 2 + (j & 15))
                nib = j < 16 ? (qb & Int32(15)) : (qb >> 4)
                s += Float32(KV[Int(nib)+1]) * X[t, b*32+j+1]
            end
            acc += d * s
        end
        @inbounds Y[t, r+1] = acc
    end
end

# Codebooks and grids in the device layout. g2 holds the 64-bit entries of
# IQ2_XS, IQ2_XXS and IQ2_S back to back (low word, then high), g3 the 32-bit
# entries of IQ3_XXS then IQ3_S, ks the IQ2 sign table, kv the IQ4 codebook.
const _AMDQ_IQ_HOST = Ref{Any}(nothing)
function _amdq_iq_host_tables()
    tb = _AMDQ_IQ_HOST[]
    if tb === nothing
        I = Gesso.Inference
        grids64 = (I.IQ2XS_GRID, I.IQ2XXS_GRID, I.IQ2S_GRID)
        g2 = Vector{UInt32}(undef, 2 * sum(length, grids64))
        p = 0
        for grid in grids64, v in grid
            g2[p+1] = UInt32(v & 0xffffffff)
            g2[p+2] = UInt32(v >> 32)
            p += 2
        end
        g3 = vcat(collect(UInt32, I.IQ3XXS_GRID), collect(UInt32, I.IQ3S_GRID))
        kv = Int32[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
        tb = (kv=kv, g2=g2, g3=g3, ks=collect(UInt8, I.KSIGNS_IQ2XS))
        _AMDQ_IQ_HOST[] = tb
    end
    return tb
end

# Uploaded once per backend: a GEMV runs per projection per token, and four
# allocations plus copies on each call would dominate a decode step.
const _AMDQ_IQ_DEVICE = Dict{Any,Any}()
const _AMDQ_IQ_DEVICE_LOCK = ReentrantLock()

function _amdq_iq_device_tables(ka)
    lock(_AMDQ_IQ_DEVICE_LOCK) do
        get!(_AMDQ_IQ_DEVICE, ka) do
            h = _amdq_iq_host_tables()
            up(v) = (d = KA.allocate(ka, eltype(v), length(v)); copyto!(d, v); d)
            (kv=up(h.kv), g2=up(h.g2), g3=up(h.g3), ks=up(h.ks))
        end
    end
end

function _amdq_iq_matmul!(Y, X, q::AmdDenseQuant; wg::Int=64)
    _ka_check_wg(wg)
    m, k = size(X)
    ka = KA.get_backend(X)
    kind = _amdq_iq_kind(q.ggml_type)
    tb = _amdq_iq_device_tables(ka)
    total = m * q.nrows
    _amdq_iq_dot_kernel!(ka, wg)(
        Y, X, q.data, m, k, q.row_bytes, total, kind, tb.kv, tb.g2, tb.g3, tb.ks; ndrange=total,
    )
    return Y
end

function _amdq_iq4nl_matmul!(Y, X, q::AmdDenseQuant; wg::Int=64)
    _ka_check_wg(wg)
    m, k = size(X)
    ka = KA.get_backend(X)
    tb = _amdq_iq_device_tables(ka)
    total = m * q.nrows
    _amdq_iq4nl_dot_kernel!(ka, wg)(
        Y, X, q.data, m, k, q.row_bytes, total, tb.kv; ndrange=total,
    )
    return Y
end
