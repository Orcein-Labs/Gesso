# GGUF element decode shared by the Lava and AMDGPU extensions. Each extension
# includes this file into its own module, so the functions compile for that
# backend's arrays. `_lq_elem(kind, W, kv, g2, g3, ks, rowoff, e)` returns
# element e of the row whose bytes start at 0-based offset rowoff, as Float32.
# The tables are the device layout built by _lq_tables (Lava) and
# _amdq_iq_device_tables (AMDGPU): the same entries in the same order.

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
const _LQ_KIND_IQ4_NL = 9
const _LQ_KIND_Q2_0 = 10
const _LQ_KIND_IQ2_XXS = 11
const _LQ_KIND_IQ2_S = 12
const _LQ_KIND_IQ3_S = 13
# Half-precision kinds have one value per element and no blocks. The bytes are
# read as they are in the GGUF, so a BF16 weight never widens on the host.
const _LQ_KIND_BF16 = 14
const _LQ_KIND_F16 = 15
const _LQ_KIND_Q2_K = 16
const _LQ_KIND_Q3_K = 17

# Entry offsets of each grid in the concatenated device tables (see _lq_tables).
const _LQ_G2_XS = 0
const _LQ_G2_XXS = 512
const _LQ_G2_S = 768
const _LQ_G3_XXS = 0
const _LQ_G3_S = 256
# --- element decode -----------------------------------------------------------
# Every reader takes an absolute byte index and returns Int32, UInt32 or
# Float32. The index is 0-based, so `W[i + 1]` is the byte at offset `i`.

@inline _lq_u8(W, i::Int) = Int32(W[i+1])
@inline _lq_u16(W, i::Int) = UInt32(W[i+1]) | (UInt32(W[i+2]) << 8)
@inline _lq_u32(W, i::Int) =
    UInt32(W[i+1]) | (UInt32(W[i+2]) << 8) | (UInt32(W[i+3]) << 16) | (UInt32(W[i+4]) << 24)
@inline _lq_f16(W, i::Int) =
    Float32(reinterpret(Float16, UInt16(W[i+1]) | (UInt16(W[i+2]) << 8)))
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
    sc = Int32(reinterpret(Int8, W[b+192+8*h+(l>>4)+2*quad+1]))
    return d * Float32(sc) * Float32(q)
end

# Q2_K (84-byte superblock): scales[16] with the scale in the low nibble and the
# min in the high nibble, qs[64] of 2-bit values, d and dmin f16 at 80 and 82.
# Scale byte e >> 4 covers 16 values, the same index the CPU reference uses.
@inline function _lq_q2k(W, b::Int, e::Int)
    d = _lq_f16(W, b + 80)
    dmin = _lq_f16(W, b + 82)
    h = e >> 7
    r = e & 127
    t = r >> 5
    l = r & 31
    sc = _lq_u8(W, b + (e >> 4))
    q = (_lq_u8(W, b + 16 + 32 * h + l) >> (2 * t)) & Int32(3)
    return d * Float32(sc & Int32(15)) * Float32(q) - dmin * Float32(sc >> 4)
end

# Q3_K (110-byte superblock): hmask[32], qs[64] of 2-bit values, scales[12] of
# packed 6-bit signed scales, d f16 at 108. A clear hmask bit subtracts 4 from
# the 2-bit value rather than leaving it unsigned, so the bit sense is what the
# negative control checks.
@inline function _lq_q3k_scale(W, b::Int, i::Int)
    m = i & 3
    k = i >> 2
    a0 = _lq_u8(W, b + 96 + m)
    a1 = _lq_u8(W, b + 100 + m)
    a2 = _lq_u8(W, b + 104 + m)
    if k == 0
        lo = a0 & Int32(15)
        hi = a2 & Int32(3)
    elseif k == 1
        lo = a1 & Int32(15)
        hi = (a2 >> 2) & Int32(3)
    elseif k == 2
        lo = a0 >> 4
        hi = (a2 >> 4) & Int32(3)
    else
        lo = a1 >> 4
        hi = (a2 >> 6) & Int32(3)
    end
    return (lo | (hi << 4)) - Int32(32)
end

@inline function _lq_q3k(W, b::Int, e::Int)
    d = _lq_f16(W, b + 108)
    h = e >> 7
    r = e & 127
    t = r >> 5
    l = r & 31
    c = 4 * h + t
    sc = _lq_q3k_scale(W, b, e >> 4)
    q2 = (_lq_u8(W, b + 32 + 32 * h + l) >> (2 * t)) & Int32(3)
    hm = (_lq_u8(W, b + l) >> c) & Int32(1)
    return d * Float32(sc) * Float32(q2 - Int32(4) * (Int32(1) - hm))
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
    return dl * kv[Int(nib)+1]
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
    gi = Int(qv & UInt32(0x01ff)) + _LQ_G2_XS
    signs = UInt32(ks[Int(qv>>9)+1])
    sc = UInt32(W[b+66+ib32+1])
    s4 = l < 2 ? (sc & UInt32(0x0f)) : (sc >> 4)
    db = d * (0.5f0 + Float32(s4)) * 0.25f0
    word = t < 4 ? g2[2*gi+1] : g2[2*gi+2]
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
    signs = UInt32(ks[Int((aux>>(7*l))&UInt32(0x7f))+1])
    qbase = b + 2 + 8 * ib32 + 2 * l
    gi = Int(W[(u<4 ? qbase : qbase+1)+1]) + _LQ_G3_XXS
    word = g3[gi+1]
    g = (word >> (8 * (u & 3))) & UInt32(0xff)
    sgn = ((signs >> u) & UInt32(1)) == UInt32(1) ? -1.0f0 : 1.0f0
    return db * Float32(g) * sgn
end

# Q8_0 (34-byte block): f16 d, 32 int8 values.
@inline function _lq_q8_0(W, b::Int, e::Int)
    d = _lq_f16(W, b)
    q = Float32(reinterpret(Int8, W[b+2+(e&31)+1]))
    return d * q
end

# IQ4_NL (18-byte block): f16 d, 16 nibble bytes. Low nibbles give the first
# 16 values, high nibbles the next 16, both through the IQ4_NL codebook.
@inline function _lq_iq4nl(W, b::Int, e::Int, kv)
    d = _lq_f16(W, b)
    qb = _lq_u8(W, b + 2 + (e & 15))
    nib = e < 16 ? (qb & Int32(15)) : (qb >> 4)
    return d * kv[Int(nib)+1]
end

# Q2_0 (18-byte block of 64 values): f16 d, 16 bytes of four 2-bit codes each.
# Code c maps to c - 1, so the values are -d, 0, d, 2d.
@inline function _lq_q2_0(W, b::Int, e::Int)
    d = _lq_f16(W, b)
    qs = _lq_u8(W, b + 2 + (e >> 2))
    code = (qs >> (2 * (e & 3))) & Int32(3)
    return d * Float32(code - Int32(1))
end

# IQ2_XXS (66-byte superblock): d f16, then per 32-value sub-block four grid
# bytes and a u32 holding the 4-bit scale (top bits) and four 7-bit sign indices.
@inline function _lq_iq2xxs(W, b::Int, e::Int, g2, ks)
    d = _lq_f16(W, b)
    ib32 = e >> 5
    r = e & 31
    l = r >> 3
    t = r & 7
    q = b + 2 + 8 * ib32
    aux = _lq_u32(W, q + 4)
    gi = Int(W[q+l+1]) + _LQ_G2_XXS
    signs = UInt32(ks[Int((aux >> (7 * l)) & UInt32(0x7f))+1])
    db = d * (0.5f0 + Float32(aux >> 28)) * 0.25f0
    word = t < 4 ? g2[2*gi+1] : g2[2*gi+2]
    g = (word >> (8 * (t & 3))) & UInt32(0xff)
    sgn = ((signs >> t) & UInt32(1)) == UInt32(1) ? -1.0f0 : 1.0f0
    return db * Float32(g) * sgn
end

# IQ2_S (82-byte superblock): d f16, 32 value bytes, 8 high-bit bytes, 8 scale
# bytes and 32 sign bytes. The grid index is 10 bits: one value byte plus two
# bits from qh. Signs are literal bits of the sign byte, not a table lookup.
@inline function _lq_iq2s(W, b::Int, e::Int, g2)
    d = _lq_f16(W, b)
    ib32 = e >> 5
    r = e & 31
    l = r >> 3
    t = r & 7
    qh = _lq_u8(W, b + 66 + ib32)
    gi = (Int(W[b+2+4*ib32+l+1]) | (Int(qh << (8 - 2 * l)) & 0x300)) + _LQ_G2_S
    sb = UInt32(W[b+34+4*ib32+l+1])
    sc = UInt32(W[b+74+ib32+1])
    s4 = l < 2 ? (sc & UInt32(0x0f)) : (sc >> 4)
    db = d * (0.5f0 + Float32(s4)) * 0.25f0
    word = t < 4 ? g2[2*gi+1] : g2[2*gi+2]
    g = (word >> (8 * (t & 3))) & UInt32(0xff)
    sgn = ((sb >> t) & UInt32(1)) == UInt32(1) ? -1.0f0 : 1.0f0
    return db * Float32(g) * sgn
end

# IQ3_S (110-byte superblock): d f16, 64 grid bytes, 8 high-bit bytes, 32 sign
# bytes, 4 scale bytes. Sub-block k (32 values) takes its 4-bit scale from
# nibble k of the scale bytes (low nibble for even k). Within each 8-value group
# the first four values come from one 9-bit grid entry and the last four from another.
@inline function _lq_iq3s(W, b::Int, e::Int, g3)
    d = _lq_f16(W, b)
    k = e >> 5
    r = e & 31
    l = r >> 3
    u = r & 7
    qh = _lq_u8(W, b + 66 + k)
    shift = u < 4 ? (8 - 2 * l) : (7 - 2 * l)
    gbyte = b + 2 + 8 * k + 2 * l + (u < 4 ? 0 : 1)
    gi = (Int(W[gbyte+1]) | (Int(qh << shift) & 256)) + _LQ_G3_S
    sb = UInt32(W[b+74+4*k+l+1])
    scb = UInt32(W[b+106+(k>>1)+1])
    s4 = (k & 1) == 0 ? (scb & UInt32(0x0f)) : (scb >> 4)
    db = d * Float32(2 * s4 + 1)
    g = (g3[gi+1] >> (8 * (u & 3))) & UInt32(0xff)
    sgn = ((sb >> u) & UInt32(1)) == UInt32(1) ? -1.0f0 : 1.0f0
    return db * Float32(g) * sgn
end

# One weight element at 0-based position e within a row starting at byte rowoff.
@inline function _lq_elem(kind::Int, W, kv, g2, g3, ks, rowoff::Int, e::Int)
    if kind == _LQ_KIND_F32
        return _lq_f32(W, rowoff + 4 * e)
    elseif kind == _LQ_KIND_BF16
        # BF16 is the top half of an F32, so widening is a shift, not a table.
        return reinterpret(Float32, _lq_u16(W, rowoff + 2 * e) << 16)
    elseif kind == _LQ_KIND_F16
        return _lq_f16(W, rowoff + 2 * e)
    elseif kind == _LQ_KIND_Q8_0
        return _lq_q8_0(W, rowoff + (e >> 5) * 34, e)
    elseif kind == _LQ_KIND_IQ4_NL
        return _lq_iq4nl(W, rowoff + (e >> 5) * 18, e & 31, kv)
    elseif kind == _LQ_KIND_Q2_0
        return _lq_q2_0(W, rowoff + (e >> 6) * 18, e & 63)
    end
    sb = e >> 8
    es = e & 255
    if kind == _LQ_KIND_Q4_K
        return _lq_q4k(W, rowoff + sb * 144, es)
    elseif kind == _LQ_KIND_Q5_K
        return _lq_q5k(W, rowoff + sb * 176, es)
    elseif kind == _LQ_KIND_Q6_K
        return _lq_q6k(W, rowoff + sb * 210, es)
    elseif kind == _LQ_KIND_Q2_K
        return _lq_q2k(W, rowoff + sb * 84, es)
    elseif kind == _LQ_KIND_Q3_K
        return _lq_q3k(W, rowoff + sb * 110, es)
    elseif kind == _LQ_KIND_IQ2_XS
        return _lq_iq2xs(W, rowoff + sb * 74, es, g2, ks)
    elseif kind == _LQ_KIND_IQ3_XXS
        return _lq_iq3xxs(W, rowoff + sb * 98, es, g3, ks)
    elseif kind == _LQ_KIND_IQ2_XXS
        return _lq_iq2xxs(W, rowoff + sb * 66, es, g2, ks)
    elseif kind == _LQ_KIND_IQ2_S
        return _lq_iq2s(W, rowoff + sb * 82, es, g2)
    elseif kind == _LQ_KIND_IQ3_S
        return _lq_iq3s(W, rowoff + sb * 110, es, g3)
    else
        return _lq_iq4xs(W, rowoff + sb * 136, es, kv)
    end
end

# GGUF type to element-decode kind, or nothing when the type has no decoder.
function _lq_kind(t::Int32)
    I = Gesso.Inference
    t == I.GGML_TYPE_Q4_K && return _LQ_KIND_Q4_K
    t == I.GGML_TYPE_Q5_K && return _LQ_KIND_Q5_K
    t == I.GGML_TYPE_Q6_K && return _LQ_KIND_Q6_K
    t == I.GGML_TYPE_IQ4_XS && return _LQ_KIND_IQ4_XS
    t == I.GGML_TYPE_Q8_0 && return _LQ_KIND_Q8_0
    t == I.GGML_TYPE_IQ4_NL && return _LQ_KIND_IQ4_NL
    t == I.GGML_TYPE_Q2_0 && return _LQ_KIND_Q2_0
    t == I.GGML_TYPE_Q2_K && return _LQ_KIND_Q2_K
    t == I.GGML_TYPE_Q3_K && return _LQ_KIND_Q3_K
    t == I.GGML_TYPE_IQ2_XS && return _LQ_KIND_IQ2_XS
    t == I.GGML_TYPE_IQ3_XXS && return _LQ_KIND_IQ3_XXS
    t == I.GGML_TYPE_IQ2_XXS && return _LQ_KIND_IQ2_XXS
    t == I.GGML_TYPE_IQ2_S && return _LQ_KIND_IQ2_S
    t == I.GGML_TYPE_IQ3_S && return _LQ_KIND_IQ3_S
    t == I.GGML_TYPE_BF16 && return _LQ_KIND_BF16
    t == I.GGML_TYPE_F16 && return _LQ_KIND_F16
    return nothing
end
