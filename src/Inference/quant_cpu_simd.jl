# Native CPU quantized dot products: Q8_K activations against IQ4_XS, IQ2_XS,
# IQ3_XXS weight rows, plus the GEMV built on them. IQ2_XXS, IQ2_S, IQ3_S,
# IQ1_S and IQ1_M are in quant_cpu_iq_s.jl (included at the end of this file),
# and the K-quants
# in quant_cpu_kquants.jl.
#
# Attribution: the scalar references are Julia ports of ggml's generic CPU
# kernels ggml_vec_dot_iq4_xs_q8_K_generic, ggml_vec_dot_iq2_xs_q8_K_generic
# and ggml_vec_dot_iq3_xxs_q8_K_generic (ggml/src/ggml-cpu/quants.c). The
# lookup tables are in gguf_quant_tables.jl, and the quantizer is in
# quant_q8k.jl. Both are ports of ggml under the MIT License, Copyright
# (c) 2023-2026 The ggml authors. No ggml code is linked or called.
#
# The AVX2 kernels are new Julia code, not copies of ggml's AVX2 paths. ggml's
# AVX2 kernels accumulate floats in a different order from the generic code,
# so they cannot be the bit-exact target. Here only the int32 sub-block sums
# are vectorized (exact integer arithmetic), and the float operations run in
# the reference's order. The AVX2 result therefore equals the scalar result bit
# for bit, and test/test_quant_cpu_simd.jl checks that.
#
# Dependency law: only Base.llvmcall and LLVM x86 intrinsics. The AVX2 kernels
# are @noinline, so their intrinsics are compiled only when a kernel runs, and
# the dispatcher runs one only when CPUID reports AVX2 and the OS saves YMM
# state. On other CPUs and architectures the scalar path runs.
# GESSO_QUANT_SIMD=0 forces the scalar path.

using ..Gesso.Log: glog, LOG_INFO

# Dequantized values per 4-bit nibble, duplicated across both 128-bit lanes so
# one vpshufb serves a full 32-byte vector.
const _Q8K_KV_IQ4NL_X2 = ntuple(i -> VecElement(KVALUES_IQ4NL[((i - 1) & 15) + 1]), 32)

# ---------------------------------------------------------------- CPU features

# CPUID leaves as single-register asm calls. Each cpuid clobbers the other
# three registers, so every call declares them clobbered.
@noinline _cpuid_eax(leaf::UInt32, sub::UInt32)::UInt32 = Base.llvmcall(("""
define i32 @entry(i32 %leaf, i32 %sub) #0 {
  %r = call i32 asm sideeffect "cpuid", "={ax},{ax},{cx},~{bx},~{cx},~{dx},~{dirflag},~{fpsr},~{flags}"(i32 %leaf, i32 %sub)
  ret i32 %r
}
attributes #0 = { alwaysinline }
""", "entry"), UInt32, Tuple{UInt32,UInt32}, leaf, sub)

@noinline _cpuid_ebx(leaf::UInt32, sub::UInt32)::UInt32 = Base.llvmcall(("""
define i32 @entry(i32 %leaf, i32 %sub) #0 {
  %r = call i32 asm sideeffect "cpuid", "={bx},{ax},{cx},~{ax},~{cx},~{dx},~{dirflag},~{fpsr},~{flags}"(i32 %leaf, i32 %sub)
  ret i32 %r
}
attributes #0 = { alwaysinline }
""", "entry"), UInt32, Tuple{UInt32,UInt32}, leaf, sub)

@noinline _cpuid_ecx(leaf::UInt32, sub::UInt32)::UInt32 = Base.llvmcall(("""
define i32 @entry(i32 %leaf, i32 %sub) #0 {
  %r = call i32 asm sideeffect "cpuid", "={cx},{ax},{cx},~{ax},~{bx},~{dx},~{dirflag},~{fpsr},~{flags}"(i32 %leaf, i32 %sub)
  ret i32 %r
}
attributes #0 = { alwaysinline }
""", "entry"), UInt32, Tuple{UInt32,UInt32}, leaf, sub)

@noinline _xgetbv_lo()::UInt32 = Base.llvmcall(("""
define i32 @entry() #0 {
  %r = call i32 asm sideeffect "xgetbv", "={ax},{cx},~{dx},~{dirflag},~{fpsr},~{flags}"(i32 0)
  ret i32 %r
}
attributes #0 = { alwaysinline }
""", "entry"), UInt32, Tuple{})

# AVX2 needs CPUID.7:EBX[5], AVX and OSXSAVE in CPUID.1:ECX[28,27], and
# XCR0 bits 1 and 2 (the OS saves XMM and YMM state).
function _cpu_has_avx2()::Bool
    Sys.ARCH === :x86_64 || return false
    _cpuid_eax(UInt32(0), UInt32(0)) >= UInt32(7) || return false
    ecx1 = _cpuid_ecx(UInt32(1), UInt32(0))
    (ecx1 >> 27) & 1 == 1 && (ecx1 >> 28) & 1 == 1 || return false
    (_xgetbv_lo() & UInt32(6)) == UInt32(6) || return false
    return (_cpuid_ebx(UInt32(7), UInt32(0)) >> 5) & 1 == 1
end

# -1 unresolved, 0 scalar, 1 AVX2. The result is cached on first use, not at
# precompile time, so a precompiled image never carries the build machine's CPU.
const _Q8K_SIMD_MODE = Ref{Int8}(-1)

"""
    q8k_simd_enabled() -> Bool

True when the AVX2 kernels are selected. Resolved once per process: false if
`GESSO_QUANT_SIMD=0`, or if the CPU or OS lacks AVX2 (logged once).
"""
function q8k_simd_enabled()::Bool
    m = _Q8K_SIMD_MODE[]
    if m < 0
        if get(ENV, "GESSO_QUANT_SIMD", "1") == "0"
            m = Int8(0)
        elseif _cpu_has_avx2()
            m = Int8(1)
        else
            m = Int8(0)
            glog(LOG_INFO, :quant_simd_unavailable; kernel=:q8k_scalar, reason=:no_avx2)
        end
        _Q8K_SIMD_MODE[] = m
    end
    return m == 1
end

# ------------------------------------------------------------- scalar reference

# Port of ggml_vec_dot_iq4_xs_q8_K_generic. `w` holds the row's IQ4_XS blocks
# starting at 0-based byte `woff`. `x` holds length(x.d) Q8_K blocks.
function _dot_iq4_xs_q8k_ref(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    sumf = 0.0f0
    @inbounds for ib in 0:(nb - 1)
        b = woff + ib * 136
        d4d8 = _ld_f16(w, b) * x.d[ib + 1]
        h = _ld_u16(w, b + 2)
        ao = ib * QK_K
        for s in 0:7
            sl = _ld_u8(w, b + 4 + (s >> 1))
            ls = (Int32((sl >> (4 * (s & 1))) & 0x0f) | (Int32((h >> (2 * s)) & 0x0003) << 4)) -
                 Int32(32)
            dl = d4d8 * Float32(ls)
            t = Int32(0)
            qb = b + 8 + 16 * s
            for j in 0:15
                q = _ld_u8(w, qb + j)
                t += Int32(x.qs[ao + 32 * s + j + 1]) * Int32(KVALUES_IQ4NL[(q & 0x0f) + 1])
                t += Int32(x.qs[ao + 32 * s + j + 17]) * Int32(KVALUES_IQ4NL[(q >> 4) + 1])
            end
            sumf += dl * Float32(t)
        end
    end
    return sumf
end

# Port of ggml_vec_dot_iq2_xs_q8_K_generic.
function _dot_iq2_xs_q8k_ref(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    sumf = 0.0f0
    @inbounds for ib in 0:(nb - 1)
        b = woff + ib * 74
        d = _ld_f16(w, b) * x.d[ib + 1]
        bsum = Int32(0)
        for ib32 in 0:7
            sc = _ld_u8(w, b + 66 + ib32)
            ls1 = Int32(2 * (sc & 0x0f) + 1)
            ls2 = Int32(2 * (sc >> 4) + 1)
            ao = ib * QK_K + 32 * ib32
            sumi1 = Int32(0)
            sumi2 = Int32(0)
            for l in 0:3
                qv = _ld_u16(w, b + 2 + 8 * ib32 + 2 * l)
                grid = IQ2XS_GRID[Int(qv & 0x01ff) + 1]
                signs = KSIGNS_IQ2XS[Int(qv >> 9) + 1]
                for j in 0:7
                    g = Int32((grid >> (8 * j)) & 0xff)
                    sg = (signs & KMASK_IQ2XS[j + 1]) != 0x00 ? Int32(-1) : Int32(1)
                    v = g * Int32(x.qs[ao + 8 * l + j + 1]) * sg
                    if l < 2
                        sumi1 += v
                    else
                        sumi2 += v
                    end
                end
            end
            bsum += sumi1 * ls1
            bsum += sumi2 * ls2
        end
        sumf += d * Float32(bsum)
    end
    return 0.125f0 * sumf
end

# Port of ggml_vec_dot_iq3_xxs_q8_K_generic.
function _dot_iq3_xxs_q8k_ref(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    sumf = 0.0f0
    @inbounds for ib in 0:(nb - 1)
        b = woff + ib * 98
        d = _ld_f16(w, b) * x.d[ib + 1]
        bsum = Int32(0)
        for ib32 in 0:7
            aux = _ld_u32(w, b + 66 + 4 * ib32)
            ls = Int32(2 * (aux >> 28) + 1)
            ao = ib * QK_K + 32 * ib32
            sumi = Int32(0)
            for l in 0:3
                g1 = IQ3XXS_GRID[Int(_ld_u8(w, b + 2 + 8 * ib32 + 2 * l)) + 1]
                g2 = IQ3XXS_GRID[Int(_ld_u8(w, b + 2 + 8 * ib32 + 2 * l + 1)) + 1]
                signs = KSIGNS_IQ2XS[Int((aux >> (7 * l)) & UInt32(127)) + 1]
                for j in 0:3
                    s0 = (signs & KMASK_IQ2XS[j + 1]) != 0x00 ? Int32(-1) : Int32(1)
                    s1 = (signs & KMASK_IQ2XS[j + 5]) != 0x00 ? Int32(-1) : Int32(1)
                    sumi += Int32((g1 >> (8 * j)) & 0xff) * Int32(x.qs[ao + 8 * l + j + 1]) * s0
                    sumi += Int32((g2 >> (8 * j)) & 0xff) * Int32(x.qs[ao + 8 * l + j + 5]) * s1
                end
            end
            bsum += sumi * ls
        end
        sumf += d * Float32(bsum)
    end
    return 0.25f0 * sumf
end

# -------------------------------------------------------------- AVX2 primitives
#
# Each primitive returns exact int32 sub-block sums. No float arithmetic is
# inside it.
#
# psign/pmaddubsw/pmaddwd are exact here because every pairwise sum fits in
# int16 (|value| ≤ 127 for Q8_K quants, ≤ 127 for the IQ4_XS table, ≤ 43 for
# IQ2_XS grid bytes, and ≤ 62 for IQ3_XXS grid bytes), so the saturating
# pmaddubsw never saturates. Q8_K quants never reach -128: the quantizer caps
# them at 127 and its scale is -127/max.

# Sign masks for the IQ2_XS and IQ3_XXS kernels. Entry s (0:127) holds 8 bytes,
# one per value of an 8-value group: 0xff where bit j of KSIGNS_IQ2XS[s] is set
# (negate) and 0x01 where it is not (keep). psign with these bytes applies the
# reference's sign to each Q8_K quant.
function _iq_sign8(s::UInt8)::UInt64
    v = UInt64(0)
    for j in 0:7
        byte = ((s >> j) & 0x01) == 0x01 ? UInt64(0xff) : UInt64(0x01)
        v |= byte << (8 * j)
    end
    return v
end
const _IQ_SIGN8_TAB = UInt64[_iq_sign8(s) for s in KSIGNS_IQ2XS]

_q8k_hsum4_ir(src::AbstractString, dst::AbstractString) = """
  $(dst)_1 = shufflevector <4 x i32> $src, <4 x i32> poison, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
  $(dst)_2 = add <4 x i32> $src, $(dst)_1
  $(dst)_3 = shufflevector <4 x i32> $(dst)_2, <4 x i32> poison, <4 x i32> <i32 1, i32 0, i32 2, i32 3>
  $(dst)_4 = add <4 x i32> $(dst)_2, $(dst)_3
  $dst = extractelement <4 x i32> $(dst)_4, i32 0
"""

const _Q8K_GRID_DECLS = """
declare <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8>, <32 x i8>)
declare <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8>, <32 x i8>)
declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>)
"""

# IQ2_XS and IQ3_XXS sub-block step. %g holds the 32 grid bytes of the sub-block
# (group l at bytes 8l..8l+7), %s the matching 32 sign bytes (+-1 as 0x01/0xff),
# %q8 the 32 Q8_K quants. psign applies the signs, pmaddubsw forms int16 pair
# sums, and pmaddwd with the int16 multiplier vector %m scales each pair sum and
# folds it into the int32 accumulator %acc. Each int32 lane holds the sum over
# four bytes; lanes 0-3 are groups 0-1 (16 values) and lanes 4-7 groups 2-3.
# The multiplier is the sub-block's scale for that half, so the lane sums are
# exact integers and their total equals the reference's bsum.
const _Q8K_IR_IQ2 = """
$(_Q8K_GRID_DECLS)
define <8 x i32> @entry(<8 x i32> %acc, <4 x i64> %g, <4 x i64> %s, ptr %q8, i32 %ls1, i32 %ls2) #0 {
  %gb = bitcast <4 x i64> %g to <32 x i8>
  %sb = bitcast <4 x i64> %s to <32 x i8>
  %a = load <32 x i8>, ptr %q8, align 1
  %bs = call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %a, <32 x i8> %sb)
  %p16 = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %gb, <32 x i8> %bs)
  %m1 = trunc i32 %ls1 to i16
  %m2 = trunc i32 %ls2 to i16
  %v1 = insertelement <16 x i16> poison, i16 %m1, i64 0
  %v2 = insertelement <16 x i16> poison, i16 %m2, i64 0
  %m = shufflevector <16 x i16> %v1, <16 x i16> %v2, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %p32 = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %p16, <16 x i16> %m)
  %r = add <8 x i32> %acc, %p32
  ret <8 x i32> %r
}
attributes #0 = { alwaysinline }
"""

# IQ3_XXS: one scale per sub-block, so the multiplier is a splat.
const _Q8K_IR_IQ3 = """
$(_Q8K_GRID_DECLS)
define <8 x i32> @entry(<8 x i32> %acc, <8 x i32> %g, <4 x i64> %s, ptr %q8, i32 %ls) #0 {
  %gb = bitcast <8 x i32> %g to <32 x i8>
  %sb = bitcast <4 x i64> %s to <32 x i8>
  %a = load <32 x i8>, ptr %q8, align 1
  %bs = call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %a, <32 x i8> %sb)
  %p16 = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %gb, <32 x i8> %bs)
  %m1 = trunc i32 %ls to i16
  %v1 = insertelement <16 x i16> poison, i16 %m1, i64 0
  %m = shufflevector <16 x i16> %v1, <16 x i16> poison, <16 x i32> zeroinitializer
  %p32 = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %p16, <16 x i16> %m)
  %r = add <8 x i32> %acc, %p32
  ret <8 x i32> %r
}
attributes #0 = { alwaysinline }
"""

# IQ4_XS: the 16 low and 16 high nibbles of one 16-byte group. lanes 0-15 hold
# the low nibbles against q8[0..15], lanes 16-31 the high nibbles against
# q8[16..31]. The pshufb table maps nibble to value, then psign applies
# the value's sign to q8 and maddubs multiplies by |value|.
const _Q8K_IR_IQ4 = """
declare <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8>, <32 x i8>)
declare <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8>, <32 x i8>)
declare <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8>, <32 x i8>)
declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>)
define i32 @entry(<32 x i8> %tab, ptr %qs, ptr %q8) #0 {
  %x128 = load <16 x i8>, ptr %qs, align 1
  %x = shufflevector <16 x i8> %x128, <16 x i8> %x128, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %lo = and <32 x i8> %x, splat (i8 15)
  %x16 = bitcast <32 x i8> %x to <16 x i16>
  %sh16 = lshr <16 x i16> %x16, splat (i16 4)
  %sh8 = bitcast <16 x i16> %sh16 to <32 x i8>
  %hi = and <32 x i8> %sh8, splat (i8 15)
  %nib = shufflevector <32 x i8> %lo, <32 x i8> %hi, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47>
  %kv = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %tab, <32 x i8> %nib)
  %a = load <32 x i8>, ptr %q8, align 1
  %ax = call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %kv, <32 x i8> %kv)
  %sy = call <32 x i8> @llvm.x86.avx2.psign.b(<32 x i8> %a, <32 x i8> %kv)
  %p16 = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %ax, <32 x i8> %sy)
  %p32 = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %p16, <16 x i16> splat (i16 1))
  %pl = shufflevector <8 x i32> %p32, <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %ph = shufflevector <8 x i32> %p32, <8 x i32> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %pa = add <4 x i32> %pl, %ph
$(_q8k_hsum4_ir("%pa", "%tot"))
  ret i32 %tot
}
attributes #0 = { alwaysinline }
"""

function _iq4_sub_avx2(tab::NTuple{32,VecElement{Int8}}, qs::Ptr{UInt8}, q8::Ptr{Int8})::Int32
    Base.llvmcall((_Q8K_IR_IQ4, "entry"), Int32,
        Tuple{NTuple{32,VecElement{Int8}}, Ptr{UInt8}, Ptr{Int8}}, tab, qs, q8)
end

const _IQ_ACC_ZERO = ntuple(_ -> VecElement(Int32(0)), 8)

@inline function _iq2_sub_acc(
    acc::NTuple{8,VecElement{Int32}},
    g::NTuple{4,VecElement{UInt64}},
    s::NTuple{4,VecElement{UInt64}},
    q8::Ptr{Int8},
    ls1::Int32,
    ls2::Int32,
)::NTuple{8,VecElement{Int32}}
    Base.llvmcall((_Q8K_IR_IQ2, "entry"), NTuple{8,VecElement{Int32}},
        Tuple{NTuple{8,VecElement{Int32}}, NTuple{4,VecElement{UInt64}}, NTuple{4,VecElement{UInt64}},
            Ptr{Int8}, Int32, Int32},
        acc, g, s, q8, ls1, ls2)
end

@inline function _iq3_sub_acc(
    acc::NTuple{8,VecElement{Int32}},
    g::NTuple{8,VecElement{UInt32}},
    s::NTuple{4,VecElement{UInt64}},
    q8::Ptr{Int8},
    ls::Int32,
)::NTuple{8,VecElement{Int32}}
    Base.llvmcall((_Q8K_IR_IQ3, "entry"), NTuple{8,VecElement{Int32}},
        Tuple{NTuple{8,VecElement{Int32}}, NTuple{8,VecElement{UInt32}}, NTuple{4,VecElement{UInt64}},
            Ptr{Int8}, Int32},
        acc, g, s, q8, ls)
end

# Horizontal sum of the eight int32 lanes. Integer, so the order is immaterial.
@inline function _iq_hsum8(acc::NTuple{8,VecElement{Int32}})::Int32
    return acc[1].value + acc[2].value + acc[3].value + acc[4].value +
           acc[5].value + acc[6].value + acc[7].value + acc[8].value
end

# f16 super-scale at a raw byte pointer, the same value as _ld_f16.
@inline _iq_f16_at(p::Ptr{UInt8})::Float32 =
    Float32(reinterpret(Float16, ltoh(unsafe_load(Ptr{UInt16}(p)))))

function _q8k_check_row(w::AbstractVector{UInt8}, woff::Int, nbytes::Int)
    (woff >= 0 && woff + nbytes <= length(w)) ||
        error("q8k row: bytes $(woff)..$(woff + nbytes) exceed weight length $(length(w))")
    return nothing
end

# ---------------------------------------------------------------- AVX2 kernels
#
# Same loop structure and float operations as the scalar reference, with the
# integer sub-block sums taken from the primitives above.

@noinline function _dot_iq4_xs_q8k_avx2(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    sumf = 0.0f0
    GC.@preserve w x begin
        pq = pointer(x.qs)
        @inbounds for ib in 0:(nb - 1)
            b = woff + ib * 136
            d4d8 = _ld_f16(w, b) * x.d[ib + 1]
            h = _ld_u16(w, b + 2)
            ao = ib * QK_K
            for s in 0:7
                sl = _ld_u8(w, b + 4 + (s >> 1))
                ls = (Int32((sl >> (4 * (s & 1))) & 0x0f) | (Int32((h >> (2 * s)) & 0x0003) << 4)) -
                     Int32(32)
                dl = d4d8 * Float32(ls)
                t = _iq4_sub_avx2(_Q8K_KV_IQ4NL_X2, pointer(w, b + 9 + 16 * s), pq + (ao + 32 * s))
                sumf += dl * Float32(t)
            end
        end
    end
    return sumf
end

# The row is read through raw pointers after one bounds check on the whole row;
# the per-byte loaders in gguf_quants.jl check every byte, which the hot loop
# cannot afford. Each block's int32 sum is accumulated in a vector across its
# eight sub-blocks, and reduced once per block.
@noinline function _dot_iq2_xs_q8k_avx2(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    _q8k_check_row(w, woff, 74 * nb)
    sumf = 0.0f0
    nb == 0 && return sumf
    GC.@preserve w x begin
        pw = pointer(w, woff + 1)
        pq = pointer(x.qs)
        pg = pointer(IQ2XS_GRID)
        ps = pointer(_IQ_SIGN8_TAB)
        @inbounds for ib in 0:(nb - 1)
            b = pw + 74 * ib
            d = _iq_f16_at(b) * x.d[ib + 1]
            acc = _IQ_ACC_ZERO
            for ib32 in 0:7
                sc = unsafe_load(b + 66 + ib32)
                ls1 = Int32(2 * (sc & 0x0f) + 1)
                ls2 = Int32(2 * (sc >> 4) + 1)
                # four qv = (grid index 9 bits | sign index 7 bits) for groups 0..3
                qw = ltoh(unsafe_load(Ptr{UInt64}(b + 2 + 8 * ib32)))
                g = (
                    VecElement(unsafe_load(pg, Int(qw & 0x1ff) + 1)),
                    VecElement(unsafe_load(pg, Int((qw >> 16) & 0x1ff) + 1)),
                    VecElement(unsafe_load(pg, Int((qw >> 32) & 0x1ff) + 1)),
                    VecElement(unsafe_load(pg, Int((qw >> 48) & 0x1ff) + 1)),
                )
                s = (
                    VecElement(unsafe_load(ps, Int((qw >> 9) & 0x7f) + 1)),
                    VecElement(unsafe_load(ps, Int((qw >> 25) & 0x7f) + 1)),
                    VecElement(unsafe_load(ps, Int((qw >> 41) & 0x7f) + 1)),
                    VecElement(unsafe_load(ps, Int((qw >> 57) & 0x7f) + 1)),
                )
                acc = _iq2_sub_acc(acc, g, s, pq + (ib * QK_K + 32 * ib32), ls1, ls2)
            end
            sumf += d * Float32(_iq_hsum8(acc))
        end
    end
    return 0.125f0 * sumf
end

@noinline function _dot_iq3_xxs_q8k_avx2(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec)::Float32
    nb = length(x.d)
    _q8k_check_row(w, woff, 98 * nb)
    sumf = 0.0f0
    nb == 0 && return sumf
    GC.@preserve w x begin
        pw = pointer(w, woff + 1)
        pq = pointer(x.qs)
        pg = pointer(IQ3XXS_GRID)
        ps = pointer(_IQ_SIGN8_TAB)
        @inbounds for ib in 0:(nb - 1)
            b = pw + 98 * ib
            d = _iq_f16_at(b) * x.d[ib + 1]
            acc = _IQ_ACC_ZERO
            for ib32 in 0:7
                aux = ltoh(unsafe_load(Ptr{UInt32}(b + 66 + 4 * ib32)))
                ls = Int32(2 * (aux >> 28) + 1)
                # eight grid indices, one byte each, in group order g1_0 g2_0 g1_1 ...
                qi = ltoh(unsafe_load(Ptr{UInt64}(b + 2 + 8 * ib32)))
                g = (
                    VecElement(unsafe_load(pg, Int(qi & 0xff) + 1)),
                    VecElement(unsafe_load(pg, Int((qi >> 8) & 0xff) + 1)),
                    VecElement(unsafe_load(pg, Int((qi >> 16) & 0xff) + 1)),
                    VecElement(unsafe_load(pg, Int((qi >> 24) & 0xff) + 1)),
                    VecElement(unsafe_load(pg, Int((qi >> 32) & 0xff) + 1)),
                    VecElement(unsafe_load(pg, Int((qi >> 40) & 0xff) + 1)),
                    VecElement(unsafe_load(pg, Int((qi >> 48) & 0xff) + 1)),
                    VecElement(unsafe_load(pg, Int((qi >> 56) & 0xff) + 1)),
                )
                # four 7-bit sign indices, one per group
                s = (
                    VecElement(unsafe_load(ps, Int(aux & 0x7f) + 1)),
                    VecElement(unsafe_load(ps, Int((aux >> 7) & 0x7f) + 1)),
                    VecElement(unsafe_load(ps, Int((aux >> 14) & 0x7f) + 1)),
                    VecElement(unsafe_load(ps, Int((aux >> 21) & 0x7f) + 1)),
                )
                acc = _iq3_sub_acc(acc, g, s, pq + (ib * QK_K + 32 * ib32), ls)
            end
            sumf += d * Float32(_iq_hsum8(acc))
        end
    end
    return 0.25f0 * sumf
end

# ---------------------------------------------------------------- dispatch

"""
    _dot_q8k(ggml_type, w, woff, x, simd) -> Float32

Dot product of one weight row (IQ4_XS, IQ2_XS or IQ3_XXS, starting at 0-based
byte `woff` of `w`) with the Q8_K activation `x`. `simd=true` takes the AVX2
kernel, `false` the scalar reference. The two agree bit for bit.
"""
@inline function _dot_q8k(ggml_type::Int32, w::AbstractVector{UInt8}, woff::Int, x, simd::Bool)::Float32
    if ggml_type == GGML_TYPE_IQ4_XS
        return simd ? _dot_iq4_xs_q8k_avx2(w, woff, x) : _dot_iq4_xs_q8k_ref(w, woff, x)
    elseif ggml_type == GGML_TYPE_IQ2_XS
        return simd ? _dot_iq2_xs_q8k_avx2(w, woff, x) : _dot_iq2_xs_q8k_ref(w, woff, x)
    elseif ggml_type == GGML_TYPE_IQ3_XXS
        return simd ? _dot_iq3_xxs_q8k_avx2(w, woff, x) : _dot_iq3_xxs_q8k_ref(w, woff, x)
    elseif _q8k_iq_s_type(ggml_type)
        return _dot_iq_s_q8k(ggml_type, w, woff, x, simd)
    end
    return _dot_kq(ggml_type, w, woff, x, simd)
end

function _q8k_check_gemv(y, ggml_type::Int32, ne0::Int, ne1::Int, q)
    _q8k_check_type(ggml_type)
    ne0 % _q8k_block_values(ggml_type) == 0 ||
        error("q8k gemv: ne0 $ne0 is not a multiple of the block size $(_q8k_block_values(ggml_type))")
    length(y) == ne1 || error("q8k gemv: y length $(length(y)) ≠ ne1 $ne1")
    need = ne1 * _qrow_bytes(ggml_type, ne0)
    length(q) >= need || error("q8k gemv: weight bytes $(length(q)) < required $need")
    return nothing
end

# Weight types with a native Q8_K (or Q8_0) integer kernel: the i-quants here
# and in quant_cpu_iq_s.jl, and the K-quants and Q8_0 in quant_cpu_kquants.jl.
_q8k_supported_type(t::Int32) =
    t in (GGML_TYPE_IQ4_XS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS) || _q8k_iq_s_type(t) || _kq_supported(t)

function _q8k_check_type(ggml_type::Int32)
    _q8k_supported_type(ggml_type) || error("q8k gemv: unsupported ggml type $ggml_type")
    return nothing
end

# ---------------------------------------------------------------- GEMV

# y[j] = alpha * dot(row j, x) + beta * y[j] for rows j0:j1, with the same
# beta==0 rule as _qgemv_bytes!.
function _qgemv_q8k_rows!(
    y::AbstractVector{Float32},
    xq,
    q::AbstractVector{UInt8},
    ggml_type::Int32,
    ne0::Int,
    j0::Int,
    j1::Int,
    alpha::Float32,
    beta::Float32,
    simd::Bool,
)
    rb = _qrow_bytes(ggml_type, ne0)
    @inbounds for j in j0:j1
        acc = _dot_q8k(ggml_type, q, (j - 1) * rb, xq, simd)
        y[j] = alpha * acc + (beta == 0.0f0 ? 0.0f0 : beta * y[j])
    end
    return y
end

"""
    _qgemv_q8k!(y, xq, q, ggml_type, ne0, ne1; alpha, beta, simd) -> y

Single-threaded `y = alpha * W * x + beta * y` for `W` an `(ne1, ne0)` matrix
of IQ4_XS, IQ2_XS or IQ3_XXS rows (row j at byte `(j-1) * row_bytes`), with `x`
already quantized to Q8_K.
"""
function _qgemv_q8k!(
    y::AbstractVector{Float32},
    xq,
    q::AbstractVector{UInt8},
    ggml_type::Int32,
    ne0::Int,
    ne1::Int;
    alpha::Float32=1.0f0,
    beta::Float32=0.0f0,
    simd::Bool=q8k_simd_enabled(),
)
    _q8k_check_gemv(y, ggml_type, ne0, ne1, q)
    length(xq) == ne0 || error("q8k gemv: xq length $(length(xq)) ≠ ne0 $ne0")
    return _qgemv_q8k_rows!(y, xq, q, ggml_type, ne0, 1, ne1, alpha, beta, simd)
end

"""
    _qgemv_q8k_threaded!(y, xq, q, ggml_type, ne0, ne1; ...) -> y

Row-chunked `_qgemv_q8k!`. Each output row is computed by exactly one task
with the same arithmetic, so the result is bit-identical for any chunk count.
"""
function _qgemv_q8k_threaded!(
    y::AbstractVector{Float32},
    xq,
    q::AbstractVector{UInt8},
    ggml_type::Int32,
    ne0::Int,
    ne1::Int;
    alpha::Float32=1.0f0,
    beta::Float32=0.0f0,
    simd::Bool=q8k_simd_enabled(),
    nchunks::Int=min(Threads.nthreads(), max(1, cld(ne1, _QGEMV_MIN_ROWS))),
)
    _q8k_check_gemv(y, ggml_type, ne0, ne1, q)
    length(xq) == ne0 || error("q8k gemv: xq length $(length(xq)) ≠ ne0 $ne0")
    nc = clamp(nchunks, 1, ne1)
    _run_row_chunks!(nc, ne1 * ne0) do c
        j0 = div((c - 1) * ne1, nc) + 1
        j1 = div(c * ne1, nc)
        _qgemv_q8k_rows!(y, xq, q, ggml_type, ne0, j0, j1, alpha, beta, simd)
    end
    return y
end

"""
    _qgemv_q8k_f32!(y, x::AbstractVector{Float32}, q, ggml_type, ne0, ne1; threaded=true, ...) -> y

Quantizes `x` to Q8_K once, then runs the row dot products, threaded by default.
"""
function _qgemv_q8k_f32!(
    y::AbstractVector{Float32},
    x::AbstractVector{Float32},
    q::AbstractVector{UInt8},
    ggml_type::Int32,
    ne0::Int,
    ne1::Int;
    alpha::Float32=1.0f0,
    beta::Float32=0.0f0,
    simd::Bool=q8k_simd_enabled(),
    threaded::Bool=true,
)
    length(x) == ne0 || error("q8k gemv: x length $(length(x)) ≠ ne0 $ne0")
    _q8k_check_type(ggml_type)
    ne0 % _q8k_block_values(ggml_type) == 0 ||
        error("q8k gemv: ne0 $ne0 is not a multiple of the block size $(_q8k_block_values(ggml_type))")
    xq = _q8k_quantize_act(ggml_type, x)
    if threaded
        return _qgemv_q8k_threaded!(y, xq, q, ggml_type, ne0, ne1; alpha, beta, simd)
    end
    return _qgemv_q8k!(y, xq, q, ggml_type, ne0, ne1; alpha, beta, simd)
end

# IQ2_XXS, IQ2_S, IQ3_S, IQ1_S and IQ1_M kernels. They use the AVX2 primitives
# above and are dispatched from _dot_q8k.
include("quant_cpu_iq_s.jl")
