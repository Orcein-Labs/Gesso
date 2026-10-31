# K-quant dense GEMV on device (AMDGPU / KADeviceBackend): Q2_K, Q3_K, Q5_K, Q6_K.
#
# These take the same Q8_K activation as Q4_K (`_amdq_q8k_act_pack_kernel!`, in
# amdgpu_quant.jl), because the CPU path quantizes the input to Q8_K for every
# K-quant weight (`_quant_q8k_native`). The weight decode follows the GGML block
# layouts documented in lava_quant.jl.
#
# Thread layout: one workgroup per (token, output row). Lane `l` of the group
# walks superblocks l, l+wg, l+2wg, ... and folds each one into a lane-local
# Float32 accumulator. The lane partials are then summed through local memory as
# a tree (the same reduction as `_ka_rmsnorm_rows_kernel!`), so the superblock
# order differs from the CPU's serial fold by a reassociation of Float32 sums.
# Each dot sums integer products per sub-block. `wg` is a power of two of at most
# `_KA_MAX_WG`, checked by `_ka_check_wg`.

# Q2_K (84-byte superblock): scales[16] (scale low nibble, min high nibble),
# qs[64] of 2-bit values, d and dmin f16 at 80 and 82. Group g (16 values)
# carries scale byte g.
@kernel function _amdq_q2k_dot_kernel!(
    Y,
    @Const(Q),
    @Const(D),
    @Const(W),
    M::Int,
    K::Int,
    row_bytes::Int,
)
    wg = @uniform @groupsize()[1]
    part = @localmem Float32 (_KA_MAX_WG,)
    grp = @index(Group, Linear)
    lane = @index(Local, Linear)
    t = (grp - 1) % M + 1
    r = (grp - 1) ÷ M
    nsb = K ÷ 256
    base = r * row_bytes
    acc = 0.0f0
    sb = lane - 1
    while sb < nsb
        blk = base + sb * 84
        isc = Int32(0)
        imn = Int32(0)
        for g in 0:15
            sc = _amdq_u8(W, blk + g)
            sub = Int32(0)
            bs = Int32(0)
            for i in 0:15
                e = g * 16 + i
                byte = _amdq_u8(W, blk + 16 + 32 * (e >> 7) + (e & 31))
                q2 = (byte >> (2 * ((e >> 5) & 3))) & Int32(3)
                qv = Q[t, sb*256+e+1]
                sub += q2 * qv
                bs += qv
            end
            isc += (sc & Int32(15)) * sub
            imn += (sc >> 4) * bs
        end
        dd = _amdq_f16(W, blk + 80) * D[t, sb+1]
        dm = _amdq_f16(W, blk + 82) * D[t, sb+1]
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
    if @index(Local, Linear) == 1
        grp = @index(Group, Linear)
        t = (grp - 1) % M + 1
        r = (grp - 1) ÷ M
        @inbounds Y[t, r+1] = part[1]
    end
end

# Q3_K (110-byte superblock): hmask[32], qs[64] of 2-bit values, scales[12] of
# packed 6-bit signed scales, d f16 at 108. A clear hmask bit subtracts 4 from
# the 2-bit value, so the decoded value is signed.
@inline function _amdq_q3k_scale(W, b::Int, i::Int)
    m = i & 3
    k = i >> 2
    a0 = _amdq_u8(W, b + 96 + m)
    a1 = _amdq_u8(W, b + 100 + m)
    a2 = _amdq_u8(W, b + 104 + m)
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

@kernel function _amdq_q3k_dot_kernel!(
    Y,
    @Const(Q),
    @Const(D),
    @Const(W),
    M::Int,
    K::Int,
    row_bytes::Int,
)
    wg = @uniform @groupsize()[1]
    part = @localmem Float32 (_KA_MAX_WG,)
    grp = @index(Group, Linear)
    lane = @index(Local, Linear)
    t = (grp - 1) % M + 1
    r = (grp - 1) ÷ M
    nsb = K ÷ 256
    base = r * row_bytes
    acc = 0.0f0
    sb = lane - 1
    while sb < nsb
        blk = base + sb * 110
        isum = Int32(0)
        for g in 0:15
            sc = _amdq_q3k_scale(W, blk, g)
            sub = Int32(0)
            for i in 0:15
                e = g * 16 + i
                h = e >> 7
                tt = (e >> 5) & 3
                ll = e & 31
                q2 = (_amdq_u8(W, blk + 32 + 32 * h + ll) >> (2 * tt)) & Int32(3)
                hm = (_amdq_u8(W, blk + ll) >> (4 * h + tt)) & Int32(1)
                qv = Q[t, sb*256+e+1]
                sub += (q2 - Int32(4) * (Int32(1) - hm)) * qv
            end
            isum += sc * sub
        end
        dd = _amdq_f16(W, blk + 108) * D[t, sb+1]
        acc += dd * Float32(isum)
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
    if @index(Local, Linear) == 1
        grp = @index(Group, Linear)
        t = (grp - 1) % M + 1
        r = (grp - 1) ÷ M
        @inbounds Y[t, r+1] = part[1]
    end
end

# Q5_K (176-byte superblock): as Q4_K, plus a 32-byte qh plane at byte 16 whose
# bit g supplies the fifth value bit of sub-block g.
@kernel function _amdq_q5k_dot_kernel!(
    Y,
    @Const(Q),
    @Const(D),
    @Const(W),
    M::Int,
    K::Int,
    row_bytes::Int,
)
    wg = @uniform @groupsize()[1]
    part = @localmem Float32 (_KA_MAX_WG,)
    grp = @index(Group, Linear)
    lane = @index(Local, Linear)
    t = (grp - 1) % M + 1
    r = (grp - 1) ÷ M
    nsb = K ÷ 256
    base = r * row_bytes
    acc = 0.0f0
    sb = lane - 1
    while sb < nsb
        blk = base + sb * 176
        isc = Int32(0)
        imn = Int32(0)
        for g in 0:7
            sc, mn = _amdq_scale_min(W, blk + 4, g)
            sub = Int32(0)
            bs = Int32(0)
            for l in 0:31
                byte = _amdq_u8(W, blk + 48 + 32 * (g >> 1) + l)
                nib = iseven(g) ? (byte & Int32(15)) : (byte >> 4)
                bit = (_amdq_u8(W, blk + 16 + l) >> g) & Int32(1)
                qv = Q[t, sb*256+g*32+l+1]
                sub += (nib + (bit << 4)) * qv
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
    if @index(Local, Linear) == 1
        grp = @index(Group, Linear)
        t = (grp - 1) % M + 1
        r = (grp - 1) ÷ M
        @inbounds Y[t, r+1] = part[1]
    end
end

# Q6_K (210-byte superblock): ql 128 bytes, qh 64 bytes, 16 int8 scales, d f16
# at 208. Group g (16 values) carries int8 scale g; the value is 6 bits centred
# at 32, with the low nibble or high nibble of ql and two bits of qh.
@kernel function _amdq_q6k_dot_kernel!(
    Y,
    @Const(Q),
    @Const(D),
    @Const(W),
    M::Int,
    K::Int,
    row_bytes::Int,
)
    wg = @uniform @groupsize()[1]
    part = @localmem Float32 (_KA_MAX_WG,)
    grp = @index(Group, Linear)
    lane = @index(Local, Linear)
    t = (grp - 1) % M + 1
    r = (grp - 1) ÷ M
    nsb = K ÷ 256
    base = r * row_bytes
    acc = 0.0f0
    sb = lane - 1
    while sb < nsb
        blk = base + sb * 210
        isum = Int32(0)
        for g in 0:15
            sc = _amdq_i8(W, blk + 192 + g)
            sub = Int32(0)
            for i in 0:15
                e = g * 16 + i
                h = e >> 7
                quad = (e >> 5) & 3
                l = e & 31
                byte = _amdq_u8(W, blk + 64 * h + l + (quad & 1) * 32)
                nib = quad < 2 ? (byte & Int32(15)) : (byte >> 4)
                hb = (_amdq_u8(W, blk + 128 + 32 * h + l) >> (2 * quad)) & Int32(3)
                q6 = (nib | (hb << 4)) - Int32(32)
                sub += q6 * Q[t, sb*256+e+1]
            end
            isum += sc * sub
        end
        dd = _amdq_f16(W, blk + 208) * D[t, sb+1]
        acc += dd * Float32(isum)
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
    if @index(Local, Linear) == 1
        grp = @index(Group, Linear)
        t = (grp - 1) % M + 1
        r = (grp - 1) ÷ M
        @inbounds Y[t, r+1] = part[1]
    end
end

# --- launchers ------------------------------------------------------------------

# Quantize the activation to Q8_K on device, then run `dot` with one workgroup of
# `wg` lanes per (token, output row): ndrange is wg * m * nrows.
function _amdq_kquant_matmul!(dot, Y, X, q::AmdDenseQuant, wg::Int)
    _ka_check_wg(wg)
    m, k = size(X)
    ka = KA.get_backend(X)
    nblk = k ÷ 256
    total_act = m * nblk
    Q = KA.allocate(ka, Int32, (m, k))
    D = KA.allocate(ka, Float32, (m, nblk))
    _amdq_q8k_act_pack_kernel!(ka, wg)(Q, D, X, m, k, total_act; ndrange=total_act)
    dot(ka, wg)(Y, Q, D, q.data, m, k, q.row_bytes; ndrange=wg * m * q.nrows)
    return Y
end

_amdq_q2k_matmul!(Y, X, q::AmdDenseQuant; wg::Int=64) =
    _amdq_kquant_matmul!(_amdq_q2k_dot_kernel!, Y, X, q, wg)

_amdq_q3k_matmul!(Y, X, q::AmdDenseQuant; wg::Int=64) =
    _amdq_kquant_matmul!(_amdq_q3k_dot_kernel!, Y, X, q, wg)

_amdq_q5k_matmul!(Y, X, q::AmdDenseQuant; wg::Int=64) =
    _amdq_kquant_matmul!(_amdq_q5k_dot_kernel!, Y, X, q, wg)

_amdq_q6k_matmul!(Y, X, q::AmdDenseQuant; wg::Int=64) =
    _amdq_kquant_matmul!(_amdq_q6k_dot_kernel!, Y, X, q, wg)
