# Native CPU quantized dot products for IQ2_XXS, IQ2_S, IQ3_S, IQ1_S and IQ1_M
# against Q8_K activations, plus the F32 dequant references the kernels are
# checked against.
# Plugs into the GEMV and dispatch in quant_cpu_simd.jl (_dot_q8k).
#
# Attribution: the scalar references are Julia ports of ggml's generic CPU
# kernels ggml_vec_dot_iq2_xxs_q8_K_generic, ggml_vec_dot_iq2_s_q8_K_generic,
# ggml_vec_dot_iq3_s_q8_K_generic, ggml_vec_dot_iq1_s_q8_K_generic and
# ggml_vec_dot_iq1_m_q8_K_generic (ggml/src/ggml-cpu/quants.c). The F32
# references are ports of dequantize_row_iq2_xxs, dequantize_row_iq2_s,
# dequantize_row_iq3_s, dequantize_row_iq1_s and dequantize_row_iq1_m
# (ggml/src/ggml-quants.c). The grids are in quant_iq_s_tables.jl. ggml is MIT licensed, Copyright (c) 2023-2026 The ggml
# authors. No ggml code is linked or called.
#
# Float order: as in quant_cpu_simd.jl, the integer sub-block sums are exact,
# and the float operations run in the generic kernel's order. The AVX2 kernels
# compute only integer sums, so they equal the scalar references bit for bit.
#
# Block layouts (ggml, little-endian):
#   IQ2_S  (82 B)  d f16 @0, qs[64] @2 (grid bytes 2..33, sign bytes 34..65),
#                  qh[8] @66, scales[8] @74
#   IQ3_S  (110 B) d f16 @0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106
#   IQ1_M  (56 B)  qs[32] @0, qh[16] @32, scales[8] @48. There is no separate d:
#                  the f16 super-scale is four 4-bit nibbles, nibble k being the
#                  top nibble of the 16-bit word scales[2k..2k+1].
#   IQ2_XXS (66 B) d f16 @0, then 8 groups of 8 bytes at 2+8g: a u32 of four
#                  grid bytes, and a u32 of four 7-bit sign indices with the
#                  4-bit scale in its top nibble.
#   IQ1_S  (50 B)  d f16 @0, qs[32] @2, qh[8] as u16 @34. Sub-block s has a
#                  3-bit scale (bits 12-14), a delta sign (bit 15), and 3 grid
#                  high bits per group (bits 3l..3l+2).

include("quant_iq_s_tables.jl")

# Full 8-bit sign masks: entry s holds 8 bytes, 0xff where bit j of s is set
# (negate value j) and 0x01 where it is not. IQ2_S and IQ3_S sign bytes are
# full masks, unlike the 7-bit KSIGNS_IQ2XS indices of IQ2_XS and IQ3_XXS.
const _IQ_SIGN8_FULL_TAB = UInt64[_iq_sign8(UInt8(s)) for s in 0:255]

# Two int32 accumulators for IQ1_M, packed into one vector: lanes 1-8 hold the
# sumi1 partials, lanes 9-16 the sumi2 partials.
const _IQ1M_ACC_ZERO = ntuple(_ -> VecElement(Int32(0)), 16)

# f16 super-scale of an IQ1_M block from its four scale words.
@inline function _iq1m_scale_bits(s0::UInt16, s1::UInt16, s2::UInt16, s3::UInt16)::UInt16
    return (s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000)
end

# Writes d into the four scale nibbles of an IQ1_M block at byte offset boff
# (0-based). The 3-bit sub-block scales in the low bits of each word are kept.
function _iq1m_write_scale!(w::AbstractVector{UInt8}, boff::Int, d::Float32)
    bits = reinterpret(UInt16, Float16(d))
    for k in 0:3
        nib = UInt8((bits >> (4 * k)) & 0x0f)
        i = boff + 48 + 2 * k + 1
        w[i + 1] = (w[i + 1] & 0x0f) | (nib << 4)
    end
    return w
end

# ------------------------------------------------------------- scalar references

# Port of ggml_vec_dot_iq2_s_q8_K_generic.
function _dot_iq2s_q8k_ref(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    sumf = 0.0f0
    @inbounds for ib in 0:(nb - 1)
        b = woff + ib * 82
        d = _ld_f16(w, b) * x.d[ib + 1]
        bsum = Int32(0)
        for ib32 in 0:7
            sc = _ld_u8(w, b + 74 + ib32)
            ls1 = Int32(2 * (sc & 0x0f) + 1)
            ls2 = Int32(2 * (sc >> 4) + 1)
            qh = Int(_ld_u8(w, b + 66 + ib32))
            ao = ib * QK_K + 32 * ib32
            sumi1 = Int32(0)
            sumi2 = Int32(0)
            for l in 0:3
                gi = Int(_ld_u8(w, b + 2 + 4 * ib32 + l)) | ((qh << (8 - 2 * l)) & 0x300)
                grid = IQ2S_GRID[gi + 1]
                sb = _ld_u8(w, b + 34 + 4 * ib32 + l)
                for j in 0:7
                    g = Int32((grid >> (8 * j)) & 0xff)
                    sg = (sb & KMASK_IQ2XS[j + 1]) != 0x00 ? Int32(-1) : Int32(1)
                    v = g * Int32(x.qs[ao + 8 * l + j + 1]) * sg
                    if l < 2
                        sumi1 += v
                    else
                        sumi2 += v
                    end
                end
            end
            bsum += ls1 * sumi1 + ls2 * sumi2
        end
        sumf += d * Float32(bsum)
    end
    return 0.125f0 * sumf
end

# Port of ggml_vec_dot_iq3_s_q8_K_generic. Sub-block k (0:7) uses qh[k], the
# grid bytes qs[8k..8k+7], the signs signs[4k..4k+3] and the scale nibble
# scales[k>>1] (low nibble for even k, high for odd k).
function _dot_iq3s_q8k_ref(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    sumf = 0.0f0
    @inbounds for ib in 0:(nb - 1)
        b = woff + ib * 110
        d = _ld_f16(w, b) * x.d[ib + 1]
        bsum = Int32(0)
        for m in 0:3
            scb = _ld_u8(w, b + 106 + m)
            for half in 0:1
                k = 2 * m + half
                ls = Int32(2 * ((scb >> (4 * half)) & 0x0f) + 1)
                qh = Int(_ld_u8(w, b + 66 + k))
                ao = ib * QK_K + 32 * k
                sumi = Int32(0)
                for l in 0:3
                    i1 = Int(_ld_u8(w, b + 2 + 8 * k + 2 * l)) | ((qh << (8 - 2 * l)) & 256)
                    i2 = Int(_ld_u8(w, b + 2 + 8 * k + 2 * l + 1)) | ((qh << (7 - 2 * l)) & 256)
                    g1 = IQ3S_GRID[i1 + 1]
                    g2 = IQ3S_GRID[i2 + 1]
                    sb = _ld_u8(w, b + 74 + 4 * k + l)
                    for j in 0:3
                        s1 = (sb & KMASK_IQ2XS[j + 1]) != 0x00 ? Int32(-1) : Int32(1)
                        s2 = (sb & KMASK_IQ2XS[j + 5]) != 0x00 ? Int32(-1) : Int32(1)
                        sumi += Int32((g1 >> (8 * j)) & 0xff) * Int32(x.qs[ao + 8 * l + j + 1]) * s1
                        sumi += Int32((g2 >> (8 * j)) & 0xff) * Int32(x.qs[ao + 8 * l + j + 5]) * s2
                    end
                end
                bsum += sumi * ls
            end
        end
        sumf += d * Float32(bsum)
    end
    return sumf
end

# Port of ggml_vec_dot_iq1_m_q8_K_generic. The super-scale is applied after the
# integer sums, with the delta term 0.125 * sumi2 added in float.
function _dot_iq1m_q8k_ref(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    sumf = 0.0f0
    @inbounds for ib in 0:(nb - 1)
        b = woff + ib * 56
        s0 = _ld_u16(w, b + 48)
        s1 = _ld_u16(w, b + 50)
        s2 = _ld_u16(w, b + 52)
        s3 = _ld_u16(w, b + 54)
        dy = Float32(reinterpret(Float16, _iq1m_scale_bits(s0, s1, s2, s3))) * x.d[ib + 1]
        sumi1 = Int32(0)
        sumi2 = Int32(0)
        for s in 0:7
            scw = Int(_ld_u16(w, b + 48 + 2 * (s >> 1)))
            ls1 = Int32(2 * ((scw >> (6 * (s & 1))) & 0x7) + 1)
            ls2 = Int32(2 * ((scw >> (6 * (s & 1) + 3)) & 0x7) + 1)
            qh0 = Int(_ld_u8(w, b + 32 + 2 * s))
            qh1 = Int(_ld_u8(w, b + 33 + 2 * s))
            sum1a = Int32(0)
            sum1b = Int32(0)
            sum2a = Int32(0)
            sum2b = Int32(0)
            for l in 0:3
                qhv = l < 2 ? qh0 : qh1
                idx = Int(_ld_u8(w, b + 4 * s + l)) | ((qhv << (8 - 4 * (l & 1))) & 0x700)
                grid = IQ1S_GRID[idx + 1]
                dbit = iseven(l) ? 0x08 : 0x80
                delta = (qhv & dbit) != 0 ? Int32(-1) : Int32(1)
                lsum1 = Int32(0)
                lsum2 = Int32(0)
                ao = ib * QK_K + 32 * s + 8 * l
                for j in 0:7
                    q = Int32(x.qs[ao + j + 1])
                    gv = Int32(reinterpret(Int8, UInt8((grid >> (8 * j)) & 0xff)))
                    lsum1 += q * gv
                    lsum2 += q
                end
                if l < 2
                    sum1a += lsum1
                    sum2a += lsum2 * delta
                else
                    sum1b += lsum1
                    sum2b += lsum2 * delta
                end
            end
            sumi1 += sum1a * ls1 + sum1b * ls2
            sumi2 += sum2a * ls1 + sum2b * ls2
        end
        sumf += dy * (Float32(sumi1) + 0.125f0 * Float32(sumi2))
    end
    return sumf
end

# Port of ggml_vec_dot_iq2_xxs_q8_K_generic. Each 32-value group has one scale
# (top nibble of its second u32) shared by its four 8-value grid entries.
function _dot_iq2xxs_q8k_ref(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    sumf = 0.0f0
    @inbounds for ib in 0:(nb - 1)
        b = woff + ib * 66
        d = _ld_f16(w, b) * x.d[ib + 1]
        bsum = Int32(0)
        for ib32 in 0:7
            q = b + 2 + 8 * ib32
            aux = _ld_u32(w, q + 4)
            ls = Int32(2 * Int(aux >> 28) + 1)
            ao = ib * QK_K + 32 * ib32
            sumi = Int32(0)
            for l in 0:3
                grid = IQ2XXS_GRID[Int(_ld_u8(w, q + l)) + 1]
                signs = KSIGNS_IQ2XS[Int((aux >> (7 * l)) & 0x7f) + 1]
                for j in 0:7
                    g = Int32((grid >> (8 * j)) & 0xff)
                    sg = (signs & KMASK_IQ2XS[j + 1]) != 0x00 ? Int32(-1) : Int32(1)
                    sumi += g * Int32(x.qs[ao + 8 * l + j + 1]) * sg
                end
            end
            bsum += sumi * ls
        end
        sumf += d * Float32(bsum)
    end
    return 0.125f0 * sumf
end

# Port of ggml_vec_dot_iq1_s_q8_K_generic. The delta term uses the Q8_K bsums:
# bsums holds the sum of each 16 quants, so a 32-value sub-block is two entries.
# The super-scale is applied after the integer sums, as in the generic kernel.
function _dot_iq1s_q8k_ref(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    sumf = 0.0f0
    @inbounds for ib in 0:(nb - 1)
        b = woff + ib * 50
        d = _ld_f16(w, b) * x.d[ib + 1]
        sumi = Int32(0)
        sumi1 = Int32(0)
        bo = 16 * ib
        for s in 0:7
            qh = Int(_ld_u16(w, b + 34 + 2 * s))
            ls = Int32(2 * ((qh >> 12) & 7) + 1)
            delta = (qh & 0x8000) != 0 ? Int32(-1) : Int32(1)
            lsum = Int32(0)
            ao = ib * QK_K + 32 * s
            for l in 0:3
                idx = Int(_ld_u8(w, b + 2 + 4 * s + l)) | (((qh >> (3 * l)) & 7) << 8)
                grid = IQ1S_GRID[idx + 1]
                for j in 0:7
                    gv = Int32(reinterpret(Int8, UInt8((grid >> (8 * j)) & 0xff)))
                    lsum += gv * Int32(x.qs[ao + 8 * l + j + 1])
                end
            end
            sumi += ls * lsum
            sumi1 += ls * delta * (Int32(x.bsums[bo + 2 * s + 1]) + Int32(x.bsums[bo + 2 * s + 2]))
        end
        sumf += d * (Float32(sumi) + 0.125f0 * Float32(sumi1))
    end
    return sumf
end

# -------------------------------------------------------------- AVX2 primitives

# IQ1_M sub-block step. g holds the four grid groups of the sub-block (values
# in {-1, 0, +1}, group l at bytes 8l..8l+7), q8 the 32 Q8_K quants, and w the
# weights: lanes 1-8 scale each group's grid dot (pair-summed lane 2l, 2l+1 for
# group l), lanes 9-16 scale each group's quant sum. psign by the grid gives
# |g| and sign(g) * q, so pmaddubsw is exact here: |g| <= 1 and |q| <= 127.
const _IQ1M_IR = """
declare <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8>, <32 x i8>)
declare <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8>, <32 x i8>)
declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>)
define <16 x i32> @entry(<16 x i32> %acc, <4 x i64> %g, ptr %q8, <16 x i32> %w) #0 {
  %gb = bitcast <4 x i64> %g to <32 x i8>
  %a = load <32 x i8>, ptr %q8, align 1
  %ab = call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %gb, <32 x i8> %gb)
  %sy = call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %a, <32 x i8> %gb)
  %p16 = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %ab, <32 x i8> %sy)
  %p32 = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %p16, <16 x i16> splat (i16 1))
  %q16 = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> splat (i8 1), <32 x i8> %a)
  %q32 = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %q16, <16 x i16> splat (i16 1))
  %w1 = shufflevector <16 x i32> %w, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %w2 = shufflevector <16 x i32> %w, <16 x i32> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %m1 = mul <8 x i32> %p32, %w1
  %m2 = mul <8 x i32> %q32, %w2
  %m = shufflevector <8 x i32> %m1, <8 x i32> %m2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %r = add <16 x i32> %acc, %m
  ret <16 x i32> %r
}
attributes #0 = { alwaysinline }
"""

@inline function _iq1m_sub_acc(
    acc::NTuple{16,VecElement{Int32}},
    g::NTuple{4,VecElement{UInt64}},
    q8::Ptr{Int8},
    w::NTuple{16,VecElement{Int32}},
)::NTuple{16,VecElement{Int32}}
    Base.llvmcall((_IQ1M_IR, "entry"), NTuple{16,VecElement{Int32}},
        Tuple{NTuple{16,VecElement{Int32}}, NTuple{4,VecElement{UInt64}}, Ptr{Int8},
            NTuple{16,VecElement{Int32}}},
        acc, g, q8, w)
end

# ---------------------------------------------------------------- AVX2 kernels
#
# Same loop structure and float operations as the scalar references. Only the
# integer sub-block sums come from the vector primitives (quant_cpu_simd.jl).

@noinline function _dot_iq2s_q8k_avx2(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    _q8k_check_row(w, woff, 82 * nb)
    sumf = 0.0f0
    nb == 0 && return sumf
    GC.@preserve w x begin
        pw = pointer(w, woff + 1)
        pq = pointer(x.qs)
        pg = pointer(IQ2S_GRID)
        ps = pointer(_IQ_SIGN8_FULL_TAB)
        @inbounds for ib in 0:(nb - 1)
            b = pw + 82 * ib
            d = _iq_f16_at(b) * x.d[ib + 1]
            acc = _IQ_ACC_ZERO
            for ib32 in 0:7
                sc = unsafe_load(b + 74 + ib32)
                ls1 = Int32(2 * (sc & 0x0f) + 1)
                ls2 = Int32(2 * (sc >> 4) + 1)
                qh = Int(unsafe_load(b + 66 + ib32))
                qb = b + 2 + 4 * ib32
                sb = b + 34 + 4 * ib32
                i0 = Int(unsafe_load(qb)) | ((qh << 8) & 0x300)
                i1 = Int(unsafe_load(qb + 1)) | ((qh << 6) & 0x300)
                i2 = Int(unsafe_load(qb + 2)) | ((qh << 4) & 0x300)
                i3 = Int(unsafe_load(qb + 3)) | ((qh << 2) & 0x300)
                g = (
                    VecElement(unsafe_load(pg, i0 + 1)),
                    VecElement(unsafe_load(pg, i1 + 1)),
                    VecElement(unsafe_load(pg, i2 + 1)),
                    VecElement(unsafe_load(pg, i3 + 1)),
                )
                s = (
                    VecElement(unsafe_load(ps, Int(unsafe_load(sb)) + 1)),
                    VecElement(unsafe_load(ps, Int(unsafe_load(sb + 1)) + 1)),
                    VecElement(unsafe_load(ps, Int(unsafe_load(sb + 2)) + 1)),
                    VecElement(unsafe_load(ps, Int(unsafe_load(sb + 3)) + 1)),
                )
                acc = _iq2_sub_acc(acc, g, s, pq + (ib * QK_K + 32 * ib32), ls1, ls2)
            end
            sumf += d * Float32(_iq_hsum8(acc))
        end
    end
    return 0.125f0 * sumf
end

@noinline function _dot_iq3s_q8k_avx2(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    _q8k_check_row(w, woff, 110 * nb)
    sumf = 0.0f0
    nb == 0 && return sumf
    GC.@preserve w x begin
        pw = pointer(w, woff + 1)
        pq = pointer(x.qs)
        pg = pointer(IQ3S_GRID)
        ps = pointer(_IQ_SIGN8_FULL_TAB)
        @inbounds for ib in 0:(nb - 1)
            b = pw + 110 * ib
            d = _iq_f16_at(b) * x.d[ib + 1]
            acc = _IQ_ACC_ZERO
            for m in 0:3
                scb = Int(unsafe_load(b + 106 + m))
                for half in 0:1
                    k = 2 * m + half
                    ls = Int32(2 * ((scb >> (4 * half)) & 0x0f) + 1)
                    qh = Int(unsafe_load(b + 66 + k))
                    qb = b + 2 + 8 * k
                    sb = b + 74 + 4 * k
                    g = (
                        VecElement(unsafe_load(pg, (Int(unsafe_load(qb)) | ((qh << 8) & 256)) + 1)),
                        VecElement(unsafe_load(pg, (Int(unsafe_load(qb + 1)) | ((qh << 7) & 256)) + 1)),
                        VecElement(unsafe_load(pg, (Int(unsafe_load(qb + 2)) | ((qh << 6) & 256)) + 1)),
                        VecElement(unsafe_load(pg, (Int(unsafe_load(qb + 3)) | ((qh << 5) & 256)) + 1)),
                        VecElement(unsafe_load(pg, (Int(unsafe_load(qb + 4)) | ((qh << 4) & 256)) + 1)),
                        VecElement(unsafe_load(pg, (Int(unsafe_load(qb + 5)) | ((qh << 3) & 256)) + 1)),
                        VecElement(unsafe_load(pg, (Int(unsafe_load(qb + 6)) | ((qh << 2) & 256)) + 1)),
                        VecElement(unsafe_load(pg, (Int(unsafe_load(qb + 7)) | ((qh << 1) & 256)) + 1)),
                    )
                    s = (
                        VecElement(unsafe_load(ps, Int(unsafe_load(sb)) + 1)),
                        VecElement(unsafe_load(ps, Int(unsafe_load(sb + 1)) + 1)),
                        VecElement(unsafe_load(ps, Int(unsafe_load(sb + 2)) + 1)),
                        VecElement(unsafe_load(ps, Int(unsafe_load(sb + 3)) + 1)),
                    )
                    acc = _iq3_sub_acc(acc, g, s, pq + (ib * QK_K + 32 * k), ls)
                end
            end
            sumf += d * Float32(_iq_hsum8(acc))
        end
    end
    return sumf
end

@noinline function _dot_iq1m_q8k_avx2(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    _q8k_check_row(w, woff, 56 * nb)
    sumf = 0.0f0
    nb == 0 && return sumf
    GC.@preserve w x begin
        pw = pointer(w, woff + 1)
        pq = pointer(x.qs)
        pg = pointer(IQ1S_GRID)
        @inbounds for ib in 0:(nb - 1)
            b = pw + 56 * ib
            s0 = ltoh(unsafe_load(Ptr{UInt16}(b + 48)))
            s1 = ltoh(unsafe_load(Ptr{UInt16}(b + 50)))
            s2 = ltoh(unsafe_load(Ptr{UInt16}(b + 52)))
            s3 = ltoh(unsafe_load(Ptr{UInt16}(b + 54)))
            dy = Float32(reinterpret(Float16, _iq1m_scale_bits(s0, s1, s2, s3))) * x.d[ib + 1]
            acc = _IQ1M_ACC_ZERO
            for s in 0:7
                scw = Int(ltoh(unsafe_load(Ptr{UInt16}(b + 48 + 2 * (s >> 1)))))
                ls1 = Int32(2 * ((scw >> (6 * (s & 1))) & 0x7) + 1)
                ls2 = Int32(2 * ((scw >> (6 * (s & 1) + 3)) & 0x7) + 1)
                qh0 = Int(unsafe_load(b + 32 + 2 * s))
                qh1 = Int(unsafe_load(b + 33 + 2 * s))
                qb = b + 4 * s
                i0 = Int(unsafe_load(qb)) | ((qh0 << 8) & 0x700)
                i1 = Int(unsafe_load(qb + 1)) | ((qh0 << 4) & 0x700)
                i2 = Int(unsafe_load(qb + 2)) | ((qh1 << 8) & 0x700)
                i3 = Int(unsafe_load(qb + 3)) | ((qh1 << 4) & 0x700)
                g = (
                    VecElement(unsafe_load(pg, i0 + 1)),
                    VecElement(unsafe_load(pg, i1 + 1)),
                    VecElement(unsafe_load(pg, i2 + 1)),
                    VecElement(unsafe_load(pg, i3 + 1)),
                )
                e0 = (qh0 & 0x08) != 0 ? -ls1 : ls1
                e1 = (qh0 & 0x80) != 0 ? -ls1 : ls1
                e2 = (qh1 & 0x08) != 0 ? -ls2 : ls2
                e3 = (qh1 & 0x80) != 0 ? -ls2 : ls2
                wt = (
                    VecElement(ls1), VecElement(ls1), VecElement(ls1), VecElement(ls1),
                    VecElement(ls2), VecElement(ls2), VecElement(ls2), VecElement(ls2),
                    VecElement(e0), VecElement(e0), VecElement(e1), VecElement(e1),
                    VecElement(e2), VecElement(e2), VecElement(e3), VecElement(e3),
                )
                acc = _iq1m_sub_acc(acc, g, pq + (ib * QK_K + 32 * s), wt)
            end
            sumi1 = Int32(0)
            sumi2 = Int32(0)
            for l in 1:8
                sumi1 += acc[l].value
                sumi2 += acc[l + 8].value
            end
            sumf += dy * (Float32(sumi1) + 0.125f0 * Float32(sumi2))
        end
    end
    return sumf
end

@noinline function _dot_iq2xxs_q8k_avx2(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    _q8k_check_row(w, woff, 66 * nb)
    sumf = 0.0f0
    nb == 0 && return sumf
    GC.@preserve w x begin
        pw = pointer(w, woff + 1)
        pq = pointer(x.qs)
        pg = pointer(IQ2XXS_GRID)
        ps = pointer(_IQ_SIGN8_TAB)
        @inbounds for ib in 0:(nb - 1)
            b = pw + 66 * ib
            d = _iq_f16_at(b) * x.d[ib + 1]
            acc = _IQ_ACC_ZERO
            for ib32 in 0:7
                # low 32 bits: four grid bytes; high 32 bits: four 7-bit sign
                # indices (bits 0-6, 7-13, ...) and the scale nibble (bits 28-31)
                qw = ltoh(unsafe_load(Ptr{UInt64}(b + 2 + 8 * ib32)))
                ls = Int32(2 * Int((qw >> 60) & 0xf) + 1)
                g = (
                    VecElement(unsafe_load(pg, Int(qw & 0xff) + 1)),
                    VecElement(unsafe_load(pg, Int((qw >> 8) & 0xff) + 1)),
                    VecElement(unsafe_load(pg, Int((qw >> 16) & 0xff) + 1)),
                    VecElement(unsafe_load(pg, Int((qw >> 24) & 0xff) + 1)),
                )
                s = (
                    VecElement(unsafe_load(ps, Int((qw >> 32) & 0x7f) + 1)),
                    VecElement(unsafe_load(ps, Int((qw >> 39) & 0x7f) + 1)),
                    VecElement(unsafe_load(ps, Int((qw >> 46) & 0x7f) + 1)),
                    VecElement(unsafe_load(ps, Int((qw >> 53) & 0x7f) + 1)),
                )
                acc = _iq2_sub_acc(acc, g, s, pq + (ib * QK_K + 32 * ib32), ls, ls)
            end
            sumf += d * Float32(_iq_hsum8(acc))
        end
    end
    return 0.125f0 * sumf
end

@noinline function _dot_iq1s_q8k_avx2(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    _q8k_check_row(w, woff, 50 * nb)
    sumf = 0.0f0
    nb == 0 && return sumf
    GC.@preserve w x begin
        pw = pointer(w, woff + 1)
        pq = pointer(x.qs)
        pg = pointer(IQ1S_GRID)
        @inbounds for ib in 0:(nb - 1)
            b = pw + 50 * ib
            d = _iq_f16_at(b) * x.d[ib + 1]
            acc = _IQ1M_ACC_ZERO
            for s in 0:7
                qh = Int(ltoh(unsafe_load(Ptr{UInt16}(b + 34 + 2 * s))))
                ls = Int32(2 * ((qh >> 12) & 7) + 1)
                ld = (qh & 0x8000) != 0 ? -ls : ls
                qb = b + 2 + 4 * s
                i0 = Int(unsafe_load(qb)) | ((qh & 7) << 8)
                i1 = Int(unsafe_load(qb + 1)) | (((qh >> 3) & 7) << 8)
                i2 = Int(unsafe_load(qb + 2)) | (((qh >> 6) & 7) << 8)
                i3 = Int(unsafe_load(qb + 3)) | (((qh >> 9) & 7) << 8)
                g = (
                    VecElement(unsafe_load(pg, i0 + 1)),
                    VecElement(unsafe_load(pg, i1 + 1)),
                    VecElement(unsafe_load(pg, i2 + 1)),
                    VecElement(unsafe_load(pg, i3 + 1)),
                )
                # lanes 1-8 weight the grid dots by ls, lanes 9-16 the quant sums by ls * delta
                wt = (
                    VecElement(ls), VecElement(ls), VecElement(ls), VecElement(ls),
                    VecElement(ls), VecElement(ls), VecElement(ls), VecElement(ls),
                    VecElement(ld), VecElement(ld), VecElement(ld), VecElement(ld),
                    VecElement(ld), VecElement(ld), VecElement(ld), VecElement(ld),
                )
                acc = _iq1m_sub_acc(acc, g, pq + (ib * QK_K + 32 * s), wt)
            end
            sumi = Int32(0)
            sumi1 = Int32(0)
            for l in 1:8
                sumi += acc[l].value
                sumi1 += acc[l + 8].value
            end
            sumf += d * (Float32(sumi) + 0.125f0 * Float32(sumi1))
        end
    end
    return sumf
end

# ------------------------------------------------------------- F32 dequant

# Ports of dequantize_row_iq2_s / _iq3_s / _iq1_m: one row of ne0 values
# starting at byte woff of w, the reference for the integer kernels above.
function _iqs_dequant_row!(out::AbstractVector{Float32}, w::AbstractVector{UInt8}, woff::Int,
        ggml_type::Int32, ne0::Int)
    length(out) == ne0 || error("iq_s dequant: out length $(length(out)) ≠ ne0 $ne0")
    ne0 % QK_K == 0 || error("iq_s dequant: ne0 $ne0 is not a multiple of $QK_K")
    nb = div(ne0, QK_K)
    @inbounds for ib in 0:(nb - 1)
        y = ib * QK_K
        if ggml_type == GGML_TYPE_IQ2_XXS
            b = woff + ib * 66
            d = _ld_f16(w, b)
            for ib32 in 0:7
                q = b + 2 + 8 * ib32
                aux = _ld_u32(w, q + 4)
                db = d * (0.5f0 + Float32(aux >> 28)) * 0.25f0
                for l in 0:3
                    grid = IQ2XXS_GRID[Int(_ld_u8(w, q + l)) + 1]
                    signs = KSIGNS_IQ2XS[Int((aux >> (7 * l)) & 0x7f) + 1]
                    for j in 0:7
                        g = Float32((grid >> (8 * j)) & 0xff)
                        sgn = (signs & KMASK_IQ2XS[j + 1]) != 0x00 ? -1.0f0 : 1.0f0
                        out[y + 32 * ib32 + 8 * l + j + 1] = db * g * sgn
                    end
                end
            end
        elseif ggml_type == GGML_TYPE_IQ1_S
            b = woff + ib * 50
            d = _ld_f16(w, b)
            for s in 0:7
                qh = Int(_ld_u16(w, b + 34 + 2 * s))
                dl = d * Float32(2 * ((qh >> 12) & 7) + 1)
                delta = (qh & 0x8000) != 0 ? -0.125f0 : 0.125f0
                for l in 0:3
                    idx = Int(_ld_u8(w, b + 2 + 4 * s + l)) | (((qh >> (3 * l)) & 7) << 8)
                    grid = IQ1S_GRID[idx + 1]
                    for j in 0:7
                        gv = Float32(reinterpret(Int8, UInt8((grid >> (8 * j)) & 0xff)))
                        out[y + 32 * s + 8 * l + j + 1] = dl * (gv + delta)
                    end
                end
            end
        elseif ggml_type == GGML_TYPE_IQ2_S
            b = woff + ib * 82
            d = _ld_f16(w, b)
            for ib32 in 0:7
                sc = _ld_u8(w, b + 74 + ib32)
                db1 = d * (0.5f0 + Float32(sc & 0x0f)) * 0.25f0
                db2 = d * (0.5f0 + Float32(sc >> 4)) * 0.25f0
                qh = Int(_ld_u8(w, b + 66 + ib32))
                for l in 0:3
                    dl = l < 2 ? db1 : db2
                    gi = Int(_ld_u8(w, b + 2 + 4 * ib32 + l)) | ((qh << (8 - 2 * l)) & 0x300)
                    grid = IQ2S_GRID[gi + 1]
                    sb = _ld_u8(w, b + 34 + 4 * ib32 + l)
                    for j in 0:7
                        g = Float32((grid >> (8 * j)) & 0xff)
                        sgn = (sb & KMASK_IQ2XS[j + 1]) != 0x00 ? -1.0f0 : 1.0f0
                        out[y + 32 * ib32 + 8 * l + j + 1] = dl * g * sgn
                    end
                end
            end
        elseif ggml_type == GGML_TYPE_IQ3_S
            b = woff + ib * 110
            d = _ld_f16(w, b)
            for m in 0:3
                scb = _ld_u8(w, b + 106 + m)
                for half in 0:1
                    k = 2 * m + half
                    db = d * Float32(1 + 2 * ((scb >> (4 * half)) & 0x0f))
                    qh = Int(_ld_u8(w, b + 66 + k))
                    for l in 0:3
                        i1 = Int(_ld_u8(w, b + 2 + 8 * k + 2 * l)) | ((qh << (8 - 2 * l)) & 256)
                        i2 = Int(_ld_u8(w, b + 2 + 8 * k + 2 * l + 1)) | ((qh << (7 - 2 * l)) & 256)
                        g1 = IQ3S_GRID[i1 + 1]
                        g2 = IQ3S_GRID[i2 + 1]
                        sb = _ld_u8(w, b + 74 + 4 * k + l)
                        for j in 0:3
                            s1 = (sb & KMASK_IQ2XS[j + 1]) != 0x00 ? -1.0f0 : 1.0f0
                            s2 = (sb & KMASK_IQ2XS[j + 5]) != 0x00 ? -1.0f0 : 1.0f0
                            out[y + 32 * k + 8 * l + j + 1] = db * Float32((g1 >> (8 * j)) & 0xff) * s1
                            out[y + 32 * k + 8 * l + j + 5] = db * Float32((g2 >> (8 * j)) & 0xff) * s2
                        end
                    end
                end
            end
        elseif ggml_type == GGML_TYPE_IQ1_M
            b = woff + ib * 56
            dsc = Float32(reinterpret(Float16, _iq1m_scale_bits(
                _ld_u16(w, b + 48), _ld_u16(w, b + 50), _ld_u16(w, b + 52), _ld_u16(w, b + 54))))
            for s in 0:7
                scw = Int(_ld_u16(w, b + 48 + 2 * (s >> 1)))
                dl1 = dsc * Float32(2 * ((scw >> (6 * (s & 1))) & 0x7) + 1)
                dl2 = dsc * Float32(2 * ((scw >> (6 * (s & 1) + 3)) & 0x7) + 1)
                qh0 = Int(_ld_u8(w, b + 32 + 2 * s))
                qh1 = Int(_ld_u8(w, b + 33 + 2 * s))
                for l in 0:3
                    qhv = l < 2 ? qh0 : qh1
                    idx = Int(_ld_u8(w, b + 4 * s + l)) | ((qhv << (8 - 4 * (l & 1))) & 0x700)
                    grid = IQ1S_GRID[idx + 1]
                    delta = (qhv & (iseven(l) ? 0x08 : 0x80)) != 0 ? -0.125f0 : 0.125f0
                    dl = l < 2 ? dl1 : dl2
                    for j in 0:7
                        gv = Float32(reinterpret(Int8, UInt8((grid >> (8 * j)) & 0xff)))
                        out[y + 32 * s + 8 * l + j + 1] = dl * (gv + delta)
                    end
                end
            end
        else
            error("iq_s dequant: no reference for ggml type $ggml_type")
        end
    end
    return out
end

# ------------------------------------------------------------- dispatch helpers

_q8k_iq_s_type(t::Int32) =
    t == GGML_TYPE_IQ2_XXS || t == GGML_TYPE_IQ2_S || t == GGML_TYPE_IQ3_S ||
    t == GGML_TYPE_IQ1_S || t == GGML_TYPE_IQ1_M

@inline function _dot_iq_s_q8k(ggml_type::Int32, w::AbstractVector{UInt8}, woff::Int, x::Q8KVec, simd::Bool)::Float32
    if ggml_type == GGML_TYPE_IQ2_XXS
        return simd ? _dot_iq2xxs_q8k_avx2(w, woff, x) : _dot_iq2xxs_q8k_ref(w, woff, x)
    elseif ggml_type == GGML_TYPE_IQ1_S
        return simd ? _dot_iq1s_q8k_avx2(w, woff, x) : _dot_iq1s_q8k_ref(w, woff, x)
    elseif ggml_type == GGML_TYPE_IQ2_S
        return simd ? _dot_iq2s_q8k_avx2(w, woff, x) : _dot_iq2s_q8k_ref(w, woff, x)
    elseif ggml_type == GGML_TYPE_IQ3_S
        return simd ? _dot_iq3s_q8k_avx2(w, woff, x) : _dot_iq3s_q8k_ref(w, woff, x)
    elseif ggml_type == GGML_TYPE_IQ1_M
        return simd ? _dot_iq1m_q8k_avx2(w, woff, x) : _dot_iq1m_q8k_ref(w, woff, x)
    end
    error("iq_s dot: no kernel for ggml type $ggml_type")
end
