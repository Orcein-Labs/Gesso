# Native CPU dot products for the ggml K-quant family (Q2_K, Q3_K, Q4_K, Q5_K,
# Q6_K), Q8_0 and Q4_0, against Q8_K or Q8_0 activations. Plugs into the same GEMV
# and dispatch as quant_cpu_simd.jl (_dot_q8k).
#
# Attribution: the scalar references are Julia ports of ggml's generic CPU
# kernels ggml_vec_dot_{q2_K,q3_K,q4_K,q5_K,q6_K}_q8_K_generic and
# ggml_vec_dot_{q8_0,q4_0}_q8_0_generic (ggml/src/ggml-cpu/quants.c). The F32 dequant
# references are ports of dequantize_row_{q2_K,...,q6_K,q8_0,q4_0} (ggml/src/
# ggml-quants.c). ggml is MIT licensed, Copyright (c) 2023-2026 The ggml
# authors. The kernels link and call no ggml code.
#
# Float order: the scalar references reproduce the generic code's float
# operations exactly (per-lane float sums in the same order, and the same
# operand order in every product). The AVX2 kernels compute only the integer
# sub-block sums, which are exact, and then share the same float code. The
# AVX2 result therefore equals the scalar result bit for bit. The AVX2 kernels
# are new Julia code, not copies of ggml's AVX2 paths.
#
# Lane convention: ggml's generic kernels keep eight float accumulators
# (sums[l]) and integer accumulators (aux32[l]). In those kernels, value
# index k contributes to lane k mod 8. The int16 pair sums in the kernels fit
# without overflow: K-quant weights are at most 63 in magnitude, Q8_K quants
# are at most 127 in magnitude, and the folded sums are at most 4 products.
# Q8_0 weights can be -128, so the Q8_0 kernel widens to int32 before summing.

# ------------------------------------------------------------- block layouts
#
# Byte offsets within one ggml block (all little-endian):
#   Q2_K (84 B)   scales[16]@0 (low nibble scale, high nibble min), qs[64]@16,
#                 d@80, dmin@82
#   Q3_K (110 B)  hmask[32]@0, qs[64]@32, scales[12]@96 (6-bit), d@108
#   Q4_K (144 B)  d@0, dmin@2, scales[12]@4 (6-bit), qs[128]@16
#   Q5_K (176 B)  d@0, dmin@2, scales[12]@4 (6-bit), qh[32]@16, qs[128]@48
#   Q6_K (210 B)  ql[128]@0, qh[64]@128, scales int8[16]@192, d@208
#   Q8_0 (34 B)   d@0 (Float16), qs[32]@2 (32 values)
#   Q4_0 (18 B)   d@0 (Float16), qs[16]@2: value j is (qs[j] & 15) - 8 and
#                 value j + 16 is (qs[j] >> 4) - 8

const _KQ_D_OFFSET = Dict{Int32, Int}(
    GGML_TYPE_Q2_K => 80,
    GGML_TYPE_Q3_K => 108,
    GGML_TYPE_Q4_K => 0,
    GGML_TYPE_Q5_K => 0,
    GGML_TYPE_Q6_K => 208,
    GGML_TYPE_Q8_0 => 0,
    GGML_TYPE_Q4_0 => 0,
)

# Byte offset of the Float16 super-scale d inside a block (test and bench
# generators write finite values there).
_q8k_d_offset(ggml_type::Int32) = get(_KQ_D_OFFSET, ggml_type, 0)

# Float16 min (dmin) offset for Q4_K, Q5_K and Q2_K; -1 when the format has none.
_q8k_dmin_offset(ggml_type::Int32) =
    ggml_type == GGML_TYPE_Q2_K ? 82 :
    (ggml_type == GGML_TYPE_Q4_K || ggml_type == GGML_TYPE_Q5_K) ? 2 : -1

# ------------------------------------------------- unchecked loads and row checks
#
# Bounds are checked once per row (_kq_check_row) before the block loop. The
# helpers below carry their own @inbounds: an @inbounds in the caller does not
# reach a helper loader, which is why the checked form was costing time.

@inline _kq_u8(w, off0::Int) = @inbounds w[off0+1]
@inline _kq_u16(w, off0::Int) = UInt16(_kq_u8(w, off0)) | (UInt16(_kq_u8(w, off0 + 1)) << 8)
@inline _kq_u32(w, off0::Int) =
    UInt32(_kq_u8(w, off0)) | (UInt32(_kq_u8(w, off0 + 1)) << 8) |
    (UInt32(_kq_u8(w, off0 + 2)) << 16) | (UInt32(_kq_u8(w, off0 + 3)) << 24)
@inline _kq_f16(w, off0::Int) = Float32(reinterpret(Float16, _kq_u16(w, off0)))

# w must hold nb blocks of bytes_per_block from byte woff (0-based); x must hold
# the same nb blocks in every field the loop reads, because the loops index x
# without a check.
function _kq_check_row(
    w::AbstractVector{UInt8},
    woff::Int,
    nb::Int,
    bytes_per_block::Int,
    x,
)
    nbytes = nb * bytes_per_block
    woff >= 0 && woff + nbytes <= length(w) || throw(BoundsError(w, (woff+1):(woff+nbytes)))
    _kq_x_holds(x, nb) ||
        throw(DimensionMismatch("activation vector does not hold $nb blocks"))
    return nothing
end

_kq_x_holds(x::Q8KVec, nb::Int) =
    length(x.d) >= nb && length(x.qs) >= QK_K * nb && length(x.bsums) >= (QK_K ÷ 16) * nb
_kq_x_holds(x::Q80Vec, nb::Int) = length(x.d) >= nb && length(x.qs) >= 32 * nb

# -------------------------------------------------------- scale and min decode

# 8 bytes of a 2-word pair: bytes 0-3 of p, then bytes 0-3 of q.
@inline _kq_bytes8(p::UInt32, q::UInt32) =
    ntuple(j -> Int32(((j <= 4 ? p : q) >> (8 * ((j - 1) & 3))) & 0xff), Val(8))

# Q4_K and Q5_K: the 12 packed bytes at byte offset sb hold 8 six-bit scales
# and 8 six-bit mins (ggml's kmask unpacking, get_scale_min_k4 equivalent).
@inline function _kq_scale_min_k4(w, sb::Int)
    u0 = _kq_u32(w, sb)
    u1 = _kq_u32(w, sb + 4)
    u2 = _kq_u32(w, sb + 8)
    m1 = 0x3f3f3f3f
    m2 = 0x0f0f0f0f
    m3 = 0x03030303
    t3 = ((u2 >> 4) & m2) | (((u1 >> 6) & m3) << 4)
    t2 = u1 & m1
    t1 = (u2 & m2) | (((u0 >> 6) & m3) << 4)
    t0 = u0 & m1
    return _kq_bytes8(t0, t1), _kq_bytes8(t2, t3)
end

# Q3_K: 16 signed scales (6-bit value minus 32) from the 12 packed bytes.
@inline function _kq_scales_q3(w, sb::Int)::NTuple{16, Int32}
    a0 = _kq_u32(w, sb)
    a1 = _kq_u32(w, sb + 4)
    a2 = _kq_u32(w, sb + 8)
    k1 = 0x03030303
    k2 = 0x0f0f0f0f
    n2 = ((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4)
    n3 = ((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4)
    n0 = (a0 & k2) | (((a2 >> 0) & k1) << 4)
    n1 = (a1 & k2) | (((a2 >> 2) & k1) << 4)
    ws = (n0, n1, n2, n3)
    return ntuple(
        j -> Int32((ws[((j-1)>>2)+1] >> (8 * ((j - 1) & 3))) & 0xff) - Int32(32),
        Val(16),
    )
end

@inline _kq_fsum(s::NTuple{8, Float32}, d::Float32, a::NTuple{8, Int32}) =
    ntuple(l -> s[l] + d * Float32(a[l]), Val(8))

@inline _kq_vec8(sc::NTuple{N, Int32}, off::Int) where {N} =
    ntuple(l -> VecElement(sc[off+l]), Val(8))

@inline _kq_unvec8(v::NTuple{8, VecElement{Int32}}) = ntuple(l -> v[l].value, Val(8))

# sum_j bsums[16 ib + j] * mins[j / 2] (Q4_K and Q5_K)
@inline function _kq_mins_dot(x::Q8KVec, ib::Int, mn::NTuple{8, Int32})::Int32
    s = Int32(0)
    @inbounds for j in 0:15
        s += Int32(x.bsums[16*ib+j+1]) * mn[(j>>1)+1]
    end
    return s
end

# ---------------------------------------------- scalar lane sums (references)
#
# Each returns the eight int32 lane accumulators aux32[l] of the generic
# kernel, computed directly from the packed fields. Lane l sums the values
# with index k ≡ l (mod 8) inside each sub-block, times the sub-block scale.

@inline function _q4k_value(w, b::Int, k::Int)::Int32
    s = k >> 5
    byte = _kq_u8(w, b + 16 + 32 * (s >> 1) + (k & 31))
    return Int32(iseven(s) ? (byte & 0x0f) : (byte >> 4))
end

@inline function _q5k_value(w, b::Int, k::Int)::Int32
    s = k >> 5
    i = k & 31
    byte = _kq_u8(w, b + 48 + 32 * (s >> 1) + i)
    nib = iseven(s) ? (byte & 0x0f) : (byte >> 4)
    hb = (_kq_u8(w, b + 16 + i) >> s) & 0x01
    return Int32(nib | (hb << 4))
end

@inline function _q6k_value(w, b::Int, k::Int)::Int32
    h = k >> 7
    r = k & 127
    c = r >> 5
    l = r & 31
    ql = _kq_u8(w, b + 64 * h + l + 32 * (c & 1))
    nib = c < 2 ? (ql & 0x0f) : (ql >> 4)
    hb = (_kq_u8(w, b + 128 + 32 * h + l) >> (2 * c)) & 0x03
    return Int32(nib | (hb << 4)) - Int32(32)
end

@inline function _q3k_value(w, b::Int, k::Int)::Int32
    h = k >> 7
    r = k & 127
    t = r >> 5
    l = r & 31
    c = 4 * h + t
    q2 = (_kq_u8(w, b + 32 + 32 * h + l) >> (2 * t)) & 0x03
    hm = (_kq_u8(w, b + l) >> c) & 0x01
    return Int32(q2 | (hm << 2)) - Int32(4)
end

@inline function _q2k_value(w, b::Int, k::Int)::Int32
    h = k >> 7
    r = k & 127
    t = r >> 5
    l = r & 31
    return Int32((_kq_u8(w, b + 16 + 32 * h + l) >> (2 * t)) & 0x03)
end

function _q4k_ints_ref(
    w,
    b::Int,
    x::Q8KVec,
    ib::Int,
    sc::NTuple{8, Int32},
)::NTuple{8, Int32}
    ao = ib * QK_K
    return ntuple(Val(8)) do l
        lane = l - 1
        acc = Int32(0)
        for s in 0:7
            sub = Int32(0)
            for i in (lane, lane + 8, lane + 16, lane + 24)
                k = 32 * s + i
                sub += _q4k_value(w, b, k) * Int32(x.qs[ao+k+1])
            end
            acc += sc[s+1] * sub
        end
        acc
    end
end

function _q5k_ints_ref(
    w,
    b::Int,
    x::Q8KVec,
    ib::Int,
    sc::NTuple{8, Int32},
)::NTuple{8, Int32}
    ao = ib * QK_K
    return ntuple(Val(8)) do l
        lane = l - 1
        acc = Int32(0)
        for s in 0:7
            sub = Int32(0)
            for i in (lane, lane + 8, lane + 16, lane + 24)
                k = 32 * s + i
                sub += _q5k_value(w, b, k) * Int32(x.qs[ao+k+1])
            end
            acc += sc[s+1] * sub
        end
        acc
    end
end

function _q6k_ints_ref(w, b::Int, x::Q8KVec, ib::Int)::NTuple{8, Int32}
    ao = ib * QK_K
    return ntuple(Val(8)) do l
        lane = l - 1
        acc = Int32(0)
        for j in 0:15
            sub = Int32(0)
            for i in (lane, lane + 8)
                k = 16 * j + i
                sub += _q6k_value(w, b, k) * Int32(x.qs[ao+k+1])
            end
            acc += Int32(reinterpret(Int8, _kq_u8(w, b + 192 + j))) * sub
        end
        acc
    end
end

function _q3k_ints_ref(
    w,
    b::Int,
    x::Q8KVec,
    ib::Int,
    sc::NTuple{16, Int32},
)::NTuple{8, Int32}
    ao = ib * QK_K
    return ntuple(Val(8)) do l
        lane = l - 1
        acc = Int32(0)
        for j in 0:15
            sub = Int32(0)
            for i in (lane, lane + 8)
                k = 16 * j + i
                sub += _q3k_value(w, b, k) * Int32(x.qs[ao+k+1])
            end
            acc += sc[j+1] * sub
        end
        acc
    end
end

function _q2k_ints_ref(w, b::Int, x::Q8KVec, ib::Int)::NTuple{8, Int32}
    ao = ib * QK_K
    return ntuple(Val(8)) do l
        lane = l - 1
        acc = Int32(0)
        for j in 0:15
            sub = Int32(0)
            for i in (lane, lane + 8)
                k = 16 * j + i
                sub += _q2k_value(w, b, k) * Int32(x.qs[ao+k+1])
            end
            acc += Int32(_kq_u8(w, b + j) & 0x0f) * sub
        end
        acc
    end
end

# ------------------------------------------------------------- AVX2 IR builders
#
# The integer sub-block sums are built from LLVM vector IR inside Base.llvmcall.
# Only generic vector operations are used (mul, shuffle, extension); the
# instruction selection comes from the host target, as in quant_cpu_simd.jl.

_kq_idx(lo::Int, n::Int) = join(("i32 $(lo + i)" for i in 0:(n-1)), ", ")

_kq_ld(io::IOBuffer, dst::String, base::String, off::Int) = print(
    io,
    "  %$(dst)_p = getelementptr i8, ptr %$base, i64 $off\n",
    "  %$dst = load <32 x i8>, ptr %$(dst)_p, align 1\n",
)

# Half h (0 or 1) of a <32 x i8> value as <16 x i8>.
_kq_half(io::IOBuffer, dst::String, src::String, h::Int) = print(
    io,
    "  %$dst = shufflevector <32 x i8> %$src, <32 x i8> poison, ",
    "<16 x i32> <$(_kq_idx(16 * h, 16))>\n",
)

# n copies of one constant, as an LLVM vector literal.
_kq_rep(ty::String, v::Int, n::Int) = "<" * join(("$ty $v" for _ in 1:n), ", ") * ">"

# Q4_0 weights of the block at %w as %wl16 (values 0-15) and %wh16 (values
# 16-31), each <16 x i16> in -8..7, in the same lanes as the Q8_0 kernel's.
_kq_q40_weights(io::IOBuffer) = print(
    io,
    "  %wp = getelementptr i8, ptr %w, i64 2\n",
    "  %wb = load <16 x i8>, ptr %wp, align 1\n",
    "  %wlo = and <16 x i8> %wb, $(_kq_rep("i8", 15, 16))\n",
    "  %whi = lshr <16 x i8> %wb, $(_kq_rep("i8", 4, 16))\n",
    "  %wlz = zext <16 x i8> %wlo to <16 x i16>\n",
    "  %whz = zext <16 x i8> %whi to <16 x i16>\n",
    "  %wl16 = sub <16 x i16> %wlz, $(_kq_rep("i16", 8, 16))\n",
    "  %wh16 = sub <16 x i16> %whz, $(_kq_rep("i16", 8, 16))\n",
)

# Fold a <16 x i16> product to <8 x i32>: lane l = p[l] + p[l+8], which is
# the sum over the values with index ≡ l (mod 8).
function _kq_fold(io::IOBuffer, dst::String, p::String)
    print(
        io,
        "  %$(dst)_fa = shufflevector <16 x i16> %$p, <16 x i16> poison, <8 x i32> <$(_kq_idx(0, 8))>\n",
        "  %$(dst)_fb = shufflevector <16 x i16> %$p, <16 x i16> poison, <8 x i32> <$(_kq_idx(8, 8))>\n",
        "  %$(dst)_fc = add <8 x i16> %$(dst)_fa, %$(dst)_fb\n",
        "  %$dst = sext <8 x i16> %$(dst)_fc to <8 x i32>\n",
    )
end

# Lane sums of 16 weight x activation products (<16 x i8> operands).
function _kq_lanes16(io::IOBuffer, dst::String, w::String, q::String, wext::String)
    print(
        io,
        "  %$(dst)_we = $wext <16 x i8> %$w to <16 x i16>\n",
        "  %$(dst)_qe = sext <16 x i8> %$q to <16 x i16>\n",
        "  %$(dst)_pr = mul <16 x i16> %$(dst)_we, %$(dst)_qe\n",
    )
    _kq_fold(io, dst, "$(dst)_pr")
end

# Lane sums of 32 products (<32 x i8> operands).
function _kq_lanes32(io::IOBuffer, dst::String, w::String, q::String, wext::String)
    for h in 0:1
        _kq_half(io, "$(dst)_w$h", w, h)
        _kq_half(io, "$(dst)_q$h", q, h)
        print(
            io,
            "  %$(dst)_we$h = $wext <16 x i8> %$(dst)_w$h to <16 x i16>\n",
            "  %$(dst)_qe$h = sext <16 x i8> %$(dst)_q$h to <16 x i16>\n",
            "  %$(dst)_pr$h = mul <16 x i16> %$(dst)_we$h, %$(dst)_qe$h\n",
        )
    end
    print(io, "  %$(dst)_pr = add <16 x i16> %$(dst)_pr0, %$(dst)_pr1\n")
    _kq_fold(io, dst, "$(dst)_pr")
end

# acc + lanes * sc (sc is an i32 SSA name); acc is an operand, e.g. "%A3" or
# "zeroinitializer".
function _kq_acc(io::IOBuffer, dst::String, acc::String, L::String, sc::String)
    print(
        io,
        "  %$(dst)_s0 = insertelement <8 x i32> poison, i32 %$sc, i32 0\n",
        "  %$(dst)_s = shufflevector <8 x i32> %$(dst)_s0, <8 x i32> poison, <8 x i32> zeroinitializer\n",
        "  %$(dst)_m = mul <8 x i32> %$L, %$(dst)_s\n",
        "  %$dst = add <8 x i32> $acc, %$(dst)_m\n",
    )
end

const _KQ_ATTRS = "\nattributes #0 = { alwaysinline }\n"

function _kq_ir_q4k()
    io = IOBuffer()
    print(io, "define <8 x i32> @entry(ptr %w, ptr %q, <8 x i32> %sca) #0 {\n")
    for j in 0:3
        _kq_ld(io, "qv$j", "w", 16 + 32 * j)
        print(
            io,
            "  %lo$j = and <32 x i8> %qv$j, splat (i8 15)\n",
            "  %hs$j = lshr <32 x i8> %qv$j, splat (i8 4)\n",
            "  %hi$j = and <32 x i8> %hs$j, splat (i8 15)\n",
        )
    end
    acc = "zeroinitializer"
    for s in 0:7
        _kq_ld(io, "xq$s", "q", 32 * s)
        _kq_lanes32(io, "L$s", iseven(s) ? "lo$(s >> 1)" : "hi$(s >> 1)", "xq$s", "zext")
        print(io, "  %sc$s = extractelement <8 x i32> %sca, i32 $s\n")
        _kq_acc(io, "A$s", acc, "L$s", "sc$s")
        acc = "%A$s"
    end
    print(io, "  ret <8 x i32> $acc\n}\n", _KQ_ATTRS)
    return String(take!(io))
end

function _kq_ir_q5k()
    io = IOBuffer()
    print(io, "define <8 x i32> @entry(ptr %w, ptr %q, <8 x i32> %sca) #0 {\n")
    _kq_ld(io, "qh", "w", 16)
    for j in 0:3
        _kq_ld(io, "qv$j", "w", 48 + 32 * j)
        print(
            io,
            "  %lo$j = and <32 x i8> %qv$j, splat (i8 15)\n",
            "  %hs$j = lshr <32 x i8> %qv$j, splat (i8 4)\n",
            "  %hi$j = and <32 x i8> %hs$j, splat (i8 15)\n",
        )
    end
    acc = "zeroinitializer"
    for s in 0:7
        nib = iseven(s) ? "lo$(s >> 1)" : "hi$(s >> 1)"
        print(
            io,
            "  %bs$s = lshr <32 x i8> %qh, splat (i8 $s)\n",
            "  %bo$s = and <32 x i8> %bs$s, splat (i8 1)\n",
            "  %bn$s = shl <32 x i8> %bo$s, splat (i8 4)\n",
            "  %wv$s = or <32 x i8> %$nib, %bn$s\n",
        )
        _kq_ld(io, "xq$s", "q", 32 * s)
        _kq_lanes32(io, "L$s", "wv$s", "xq$s", "zext")
        print(io, "  %sc$s = extractelement <8 x i32> %sca, i32 $s\n")
        _kq_acc(io, "A$s", acc, "L$s", "sc$s")
        acc = "%A$s"
    end
    print(io, "  ret <8 x i32> $acc\n}\n", _KQ_ATTRS)
    return String(take!(io))
end

function _kq_ir_q6k()
    io = IOBuffer()
    print(
        io,
        "define <8 x i32> @entry(ptr %w, ptr %q) #0 {\n",
        "  %scbp = getelementptr i8, ptr %w, i64 192\n",
        "  %scb = load <16 x i8>, ptr %scbp, align 1\n",
    )
    acc = "zeroinitializer"
    for h in 0:1
        _kq_ld(io, "qh$h", "w", 128 + 32 * h)
        for c in 0:3
            _kq_ld(io, "ql$(h)$c", "w", 64 * h + 32 * (c & 1))
            if c < 2
                print(io, "  %nb$(h)$c = and <32 x i8> %ql$(h)$c, splat (i8 15)\n")
            else
                print(
                    io,
                    "  %nh$(h)$c = lshr <32 x i8> %ql$(h)$c, splat (i8 4)\n",
                    "  %nb$(h)$c = and <32 x i8> %nh$(h)$c, splat (i8 15)\n",
                )
            end
            print(
                io,
                "  %hs$(h)$c = lshr <32 x i8> %qh$h, splat (i8 $(2c))\n",
                "  %hm$(h)$c = and <32 x i8> %hs$(h)$c, splat (i8 3)\n",
                "  %hb$(h)$c = shl <32 x i8> %hm$(h)$c, splat (i8 4)\n",
                "  %vv$(h)$c = or <32 x i8> %nb$(h)$c, %hb$(h)$c\n",
                "  %wv$(h)$c = sub <32 x i8> %vv$(h)$c, splat (i8 32)\n",
            )
            _kq_ld(io, "xq$(h)$c", "q", 128 * h + 32 * c)
            _kq_half(io, "wa$(h)$c", "wv$(h)$c", 0)
            _kq_half(io, "wb$(h)$c", "wv$(h)$c", 1)
            _kq_half(io, "xa$(h)$c", "xq$(h)$c", 0)
            _kq_half(io, "xb$(h)$c", "xq$(h)$c", 1)
            for half in 0:1
                i = 8 * h + 2 * c + half
                print(
                    io,
                    "  %scb$i.e = extractelement <16 x i8> %scb, i32 $i\n",
                    "  %sc$i = sext i8 %scb$i.e to i32\n",
                )
                w = half == 0 ? "wa$(h)$c" : "wb$(h)$c"
                x = half == 0 ? "xa$(h)$c" : "xb$(h)$c"
                _kq_lanes16(io, "L$i", w, x, "sext")
                _kq_acc(io, "A$i", acc, "L$i", "sc$i")
                acc = "%A$i"
            end
        end
    end
    print(io, "  ret <8 x i32> $acc\n}\n", _KQ_ATTRS)
    return String(take!(io))
end

function _kq_ir_q3k()
    io = IOBuffer()
    print(
        io,
        "define <8 x i32> @entry(ptr %w, ptr %q, <8 x i32> %sca, <8 x i32> %scb) #0 {\n",
    )
    _kq_ld(io, "hm", "w", 0)
    for h in 0:1
        _kq_ld(io, "qs$h", "w", 32 + 32 * h)
    end
    acc = "zeroinitializer"
    for h in 0:1, t in 0:3
        c = 4 * h + t
        print(
            io,
            "  %qa$(h)$t = lshr <32 x i8> %qs$h, splat (i8 $(2t))\n",
            "  %qb$(h)$t = and <32 x i8> %qa$(h)$t, splat (i8 3)\n",
            "  %hc$(h)$t = lshr <32 x i8> %hm, splat (i8 $c)\n",
            "  %hd$(h)$t = and <32 x i8> %hc$(h)$t, splat (i8 1)\n",
            "  %he$(h)$t = shl <32 x i8> %hd$(h)$t, splat (i8 2)\n",
            "  %uu$(h)$t = or <32 x i8> %qb$(h)$t, %he$(h)$t\n",
            "  %wv$(h)$t = sub <32 x i8> %uu$(h)$t, splat (i8 4)\n",
        )
        _kq_ld(io, "xq$(h)$t", "q", 32 * c)
        _kq_half(io, "wa$(h)$t", "wv$(h)$t", 0)
        _kq_half(io, "wb$(h)$t", "wv$(h)$t", 1)
        _kq_half(io, "xa$(h)$t", "xq$(h)$t", 0)
        _kq_half(io, "xb$(h)$t", "xq$(h)$t", 1)
        for half in 0:1
            i = 8 * h + 2 * t + half
            src = i < 8 ? "sca" : "scb"
            print(io, "  %sc$i = extractelement <8 x i32> %$src, i32 $(i & 7)\n")
            w = half == 0 ? "wa$(h)$t" : "wb$(h)$t"
            x = half == 0 ? "xa$(h)$t" : "xb$(h)$t"
            _kq_lanes16(io, "L$i", w, x, "sext")
            _kq_acc(io, "A$i", acc, "L$i", "sc$i")
            acc = "%A$i"
        end
    end
    print(io, "  ret <8 x i32> $acc\n}\n", _KQ_ATTRS)
    return String(take!(io))
end

function _kq_ir_q2k()
    io = IOBuffer()
    print(
        io,
        "define <8 x i32> @entry(ptr %w, ptr %q) #0 {\n",
        "  %scb = load <16 x i8>, ptr %w, align 1\n",
        "  %scl = and <16 x i8> %scb, splat (i8 15)\n",
    )
    for k in 0:1
        _kq_ld(io, "qv$k", "w", 16 + 32 * k)
    end
    acc = "zeroinitializer"
    for k in 0:1, t in 0:3
        print(
            io,
            "  %qa$(k)$t = lshr <32 x i8> %qv$k, splat (i8 $(2t))\n",
            "  %qb$(k)$t = and <32 x i8> %qa$(k)$t, splat (i8 3)\n",
        )
        _kq_ld(io, "xq$(k)$t", "q", 128 * k + 32 * t)
        _kq_half(io, "wa$(k)$t", "qb$(k)$t", 0)
        _kq_half(io, "wb$(k)$t", "qb$(k)$t", 1)
        _kq_half(io, "xa$(k)$t", "xq$(k)$t", 0)
        _kq_half(io, "xb$(k)$t", "xq$(k)$t", 1)
        for half in 0:1
            i = 8 * k + 2 * t + half
            print(
                io,
                "  %sc$i.e = extractelement <16 x i8> %scl, i32 $i\n",
                "  %sc$i = zext i8 %sc$i.e to i32\n",
            )
            w = half == 0 ? "wa$(k)$t" : "wb$(k)$t"
            x = half == 0 ? "xa$(k)$t" : "xb$(k)$t"
            _kq_lanes16(io, "L$i", w, x, "zext")
            _kq_acc(io, "A$i", acc, "L$i", "sc$i")
            acc = "%A$i"
        end
    end
    print(io, "  ret <8 x i32> $acc\n}\n", _KQ_ATTRS)
    return String(take!(io))
end

const _KQ_IR_Q4K = _kq_ir_q4k()
const _KQ_IR_Q5K = _kq_ir_q5k()
const _KQ_IR_Q6K = _kq_ir_q6k()
const _KQ_IR_Q3K = _kq_ir_q3k()
const _KQ_IR_Q2K = _kq_ir_q2k()

const _KQ_IR_Q80 = """
declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>)
define i32 @entry(ptr %w, ptr %q) #0 {
  %wp = getelementptr i8, ptr %w, i64 2
  %wv = load <32 x i8>, ptr %wp, align 1
  %qv = load <32 x i8>, ptr %q, align 1
  %wl = shufflevector <32 x i8> %wv, <32 x i8> poison, <16 x i32> <$(_kq_idx(0, 16))>
  %wh = shufflevector <32 x i8> %wv, <32 x i8> poison, <16 x i32> <$(_kq_idx(16, 16))>
  %ql = shufflevector <32 x i8> %qv, <32 x i8> poison, <16 x i32> <$(_kq_idx(0, 16))>
  %qh = shufflevector <32 x i8> %qv, <32 x i8> poison, <16 x i32> <$(_kq_idx(16, 16))>
  %wl16 = sext <16 x i8> %wl to <16 x i16>
  %wh16 = sext <16 x i8> %wh to <16 x i16>
  %ql16 = sext <16 x i8> %ql to <16 x i16>
  %qh16 = sext <16 x i8> %qh to <16 x i16>
  %pl = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %wl16, <16 x i16> %ql16)
  %ph = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %wh16, <16 x i16> %qh16)
  %p = add <8 x i32> %pl, %ph
  %e0 = extractelement <8 x i32> %p, i32 0
  %e1 = extractelement <8 x i32> %p, i32 1
  %e2 = extractelement <8 x i32> %p, i32 2
  %e3 = extractelement <8 x i32> %p, i32 3
  %e4 = extractelement <8 x i32> %p, i32 4
  %e5 = extractelement <8 x i32> %p, i32 5
  %e6 = extractelement <8 x i32> %p, i32 6
  %e7 = extractelement <8 x i32> %p, i32 7
  %s1 = add i32 %e0, %e1
  %s2 = add i32 %s1, %e2
  %s3 = add i32 %s2, %e3
  %s4 = add i32 %s3, %e4
  %s5 = add i32 %s4, %e5
  %s6 = add i32 %s5, %e6
  %s7 = add i32 %s6, %e7
  ret i32 %s7
}
attributes #0 = { alwaysinline }
"""

# Q4_0: the Q8_0 kernel with nibble weights. Products are at most 8 * 128 in
# magnitude, so the pair sums of vpmaddwd cannot overflow.
function _kq_ir_q40()
    io = IOBuffer()
    print(
        io,
        "declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>)\n",
        "define i32 @entry(ptr %w, ptr %q) #0 {\n",
    )
    _kq_q40_weights(io)
    _kq_ld(io, "qv", "q", 0)
    _kq_half(io, "ql", "qv", 0)
    _kq_half(io, "qh", "qv", 1)
    print(
        io,
        "  %ql16 = sext <16 x i8> %ql to <16 x i16>\n",
        "  %qh16 = sext <16 x i8> %qh to <16 x i16>\n",
        "  %pl = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %wl16, <16 x i16> %ql16)\n",
        "  %ph = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %wh16, <16 x i16> %qh16)\n",
        "  %p = add <8 x i32> %pl, %ph\n",
    )
    for l in 0:7
        print(io, "  %e$l = extractelement <8 x i32> %p, i32 $l\n")
    end
    print(io, "  %s0 = add i32 %e0, 0\n")
    for l in 1:7
        print(io, "  %s$l = add i32 %s$(l - 1), %e$l\n")
    end
    print(io, "  ret i32 %s7\n}\n", _KQ_ATTRS)
    return String(take!(io))
end

const _KQ_IR_Q40 = _kq_ir_q40()

# Each AVX2 block kernel: the same lane sums as the scalar reference, for one
# block. @noinline so the AVX2 code is compiled only when a kernel runs.

@noinline function _q4k_ints_avx2(
    w::AbstractVector{UInt8},
    b::Int,
    x::Q8KVec,
    ib::Int,
    sc::NTuple{8, Int32},
)::NTuple{8, Int32}
    v = GC.@preserve w x Base.llvmcall(
        (_KQ_IR_Q4K, "entry"),
        NTuple{8, VecElement{Int32}},
        Tuple{Ptr{UInt8}, Ptr{Int8}, NTuple{8, VecElement{Int32}}},
        pointer(w, b + 1),
        pointer(x.qs, ib * QK_K + 1),
        _kq_vec8(sc, 0),
    )
    return _kq_unvec8(v)
end

@noinline function _q5k_ints_avx2(
    w::AbstractVector{UInt8},
    b::Int,
    x::Q8KVec,
    ib::Int,
    sc::NTuple{8, Int32},
)::NTuple{8, Int32}
    v = GC.@preserve w x Base.llvmcall(
        (_KQ_IR_Q5K, "entry"),
        NTuple{8, VecElement{Int32}},
        Tuple{Ptr{UInt8}, Ptr{Int8}, NTuple{8, VecElement{Int32}}},
        pointer(w, b + 1),
        pointer(x.qs, ib * QK_K + 1),
        _kq_vec8(sc, 0),
    )
    return _kq_unvec8(v)
end

@noinline function _q6k_ints_avx2(
    w::AbstractVector{UInt8},
    b::Int,
    x::Q8KVec,
    ib::Int,
)::NTuple{8, Int32}
    v = GC.@preserve w x Base.llvmcall(
        (_KQ_IR_Q6K, "entry"),
        NTuple{8, VecElement{Int32}},
        Tuple{Ptr{UInt8}, Ptr{Int8}},
        pointer(w, b + 1),
        pointer(x.qs, ib * QK_K + 1),
    )
    return _kq_unvec8(v)
end

@noinline function _q3k_ints_avx2(
    w::AbstractVector{UInt8},
    b::Int,
    x::Q8KVec,
    ib::Int,
    sc::NTuple{16, Int32},
)::NTuple{8, Int32}
    v = GC.@preserve w x Base.llvmcall(
        (_KQ_IR_Q3K, "entry"),
        NTuple{8, VecElement{Int32}},
        Tuple{
            Ptr{UInt8},
            Ptr{Int8},
            NTuple{8, VecElement{Int32}},
            NTuple{8, VecElement{Int32}},
        },
        pointer(w, b + 1),
        pointer(x.qs, ib * QK_K + 1),
        _kq_vec8(sc, 0),
        _kq_vec8(sc, 8),
    )
    return _kq_unvec8(v)
end

@noinline function _q2k_ints_avx2(
    w::AbstractVector{UInt8},
    b::Int,
    x::Q8KVec,
    ib::Int,
)::NTuple{8, Int32}
    v = GC.@preserve w x Base.llvmcall(
        (_KQ_IR_Q2K, "entry"),
        NTuple{8, VecElement{Int32}},
        Tuple{Ptr{UInt8}, Ptr{Int8}},
        pointer(w, b + 1),
        pointer(x.qs, ib * QK_K + 1),
    )
    return _kq_unvec8(v)
end

@noinline function _q80_sumi_avx2(
    w::AbstractVector{UInt8},
    b::Int,
    x::Q80Vec,
    ib::Int,
)::Int32
    return GC.@preserve w x Base.llvmcall(
        (_KQ_IR_Q80, "entry"),
        Int32,
        Tuple{Ptr{UInt8}, Ptr{Int8}},
        pointer(w, b + 1),
        pointer(x.qs, 32 * ib + 1),
    )
end

@noinline function _q40_sumi_avx2(
    w::AbstractVector{UInt8},
    b::Int,
    x::Q80Vec,
    ib::Int,
)::Int32
    return GC.@preserve w x Base.llvmcall(
        (_KQ_IR_Q40, "entry"),
        Int32,
        Tuple{Ptr{UInt8}, Ptr{Int8}},
        pointer(w, b + 1),
        pointer(x.qs, 32 * ib + 1),
    )
end

# ------------------------------------------------------------- scalar Q8_0, Q4_0

@inline function _q80_sumi_ref(w::AbstractVector{UInt8}, b::Int, x::Q80Vec, ib::Int)::Int32
    s = Int32(0)
    qo = 32 * ib
    for j in 0:31
        s += Int32(reinterpret(Int8, _kq_u8(w, b + 2 + j))) * Int32(x.qs[qo+j+1])
    end
    return s
end

# sumi0 + sumi1 as in the generic kernel (integer, so the order is free).
@inline function _q40_sumi_ref(w::AbstractVector{UInt8}, b::Int, x::Q80Vec, ib::Int)::Int32
    s0 = Int32(0)
    s1 = Int32(0)
    qo = 32 * ib
    for j in 0:15
        q = _kq_u8(w, b + 2 + j)
        s0 += (Int32(q & 0x0f) - Int32(8)) * Int32(x.qs[qo+j+1])
        s1 += (Int32(q >> 4) - Int32(8)) * Int32(x.qs[qo+j+17])
    end
    return s0 + s1
end

# ------------------------------------------------------------- dot products
#
# Each dot reads one row's blocks and one Q8_K activation vector. The integer
# lanes come from the reference or from the AVX2 kernel, and the float code is
# shared. The float operations follow the generic kernel's order exactly.

function _dot_q4k_q8k(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec, simd::Bool)::Float32
    nb = length(x.d)
    _kq_check_row(w, woff, nb, 144, x)
    sumf = 0.0f0
    sums = ntuple(_ -> 0.0f0, Val(8))
    @inbounds for ib in 0:(nb-1)
        b = woff + ib * 144
        d = _kq_f16(w, b) * x.d[ib+1]
        dmin = _kq_f16(w, b + 2) * x.d[ib+1]
        sc, mn = _kq_scale_min_k4(w, b + 4)
        aux = simd ? _q4k_ints_avx2(w, b, x, ib, sc) : _q4k_ints_ref(w, b, x, ib, sc)
        sums = _kq_fsum(sums, d, aux)
        sumf -= dmin * Float32(_kq_mins_dot(x, ib, mn))
    end
    for l in 1:8
        sumf += sums[l]
    end
    return sumf
end

function _dot_q5k_q8k(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec, simd::Bool)::Float32
    nb = length(x.d)
    _kq_check_row(w, woff, nb, 176, x)
    sumf = 0.0f0
    sums = ntuple(_ -> 0.0f0, Val(8))
    @inbounds for ib in 0:(nb-1)
        b = woff + ib * 176
        d = _kq_f16(w, b) * x.d[ib+1]
        dmin = _kq_f16(w, b + 2) * x.d[ib+1]
        sc, mn = _kq_scale_min_k4(w, b + 4)
        aux = simd ? _q5k_ints_avx2(w, b, x, ib, sc) : _q5k_ints_ref(w, b, x, ib, sc)
        sums = _kq_fsum(sums, d, aux)
        sumf -= dmin * Float32(_kq_mins_dot(x, ib, mn))
    end
    for l in 1:8
        sumf += sums[l]
    end
    return sumf
end

function _dot_q6k_q8k(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec, simd::Bool)::Float32
    nb = length(x.d)
    _kq_check_row(w, woff, nb, 210, x)
    sumf = 0.0f0
    sums = ntuple(_ -> 0.0f0, Val(8))
    @inbounds for ib in 0:(nb-1)
        b = woff + ib * 210
        d = _kq_f16(w, b + 208) * x.d[ib+1]
        aux = simd ? _q6k_ints_avx2(w, b, x, ib) : _q6k_ints_ref(w, b, x, ib)
        sums = _kq_fsum(sums, d, aux)
    end
    for l in 1:8
        sumf += sums[l]
    end
    return sumf
end

function _dot_q3k_q8k(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec, simd::Bool)::Float32
    nb = length(x.d)
    _kq_check_row(w, woff, nb, 110, x)
    sumf = 0.0f0
    sums = ntuple(_ -> 0.0f0, Val(8))
    @inbounds for ib in 0:(nb-1)
        b = woff + ib * 110
        d = _kq_f16(w, b + 108) * x.d[ib+1]
        sc = _kq_scales_q3(w, b + 96)
        aux = simd ? _q3k_ints_avx2(w, b, x, ib, sc) : _q3k_ints_ref(w, b, x, ib, sc)
        sums = _kq_fsum(sums, d, aux)
    end
    for l in 1:8
        sumf += sums[l]
    end
    return sumf
end

function _dot_q2k_q8k(w::AbstractVector{UInt8}, woff::Int, x::Q8KVec, simd::Bool)::Float32
    nb = length(x.d)
    _kq_check_row(w, woff, nb, 84, x)
    sumf = 0.0f0
    @inbounds for ib in 0:(nb-1)
        b = woff + ib * 84
        dall = x.d[ib+1] * _kq_f16(w, b + 80)
        dmin = x.d[ib+1] * _kq_f16(w, b + 82)
        aux = simd ? _q2k_ints_avx2(w, b, x, ib) : _q2k_ints_ref(w, b, x, ib)
        isum = Int32(0)
        for l in 1:8
            isum += aux[l]
        end
        summs = Int32(0)
        for j in 0:15
            summs += Int32(x.bsums[16*ib+j+1]) * Int32(_kq_u8(w, b + j) >> 4)
        end
        sumf += dall * Float32(isum) - dmin * Float32(summs)
    end
    return sumf
end

function _dot_q8_0_q8_0(w::AbstractVector{UInt8}, woff::Int, x::Q80Vec, simd::Bool)::Float32
    nb = length(x.d)
    _kq_check_row(w, woff, nb, 34, x)
    sumf = 0.0f0
    @inbounds for ib in 0:(nb-1)
        b = woff + ib * 34
        sumi = simd ? _q80_sumi_avx2(w, b, x, ib) : _q80_sumi_ref(w, b, x, ib)
        sumf += Float32(sumi) * (_kq_f16(w, b) * x.d[ib+1])
    end
    return sumf
end

# The generic Q4_0 kernel has no parentheses: sumi * dw * dx, left to right.
function _dot_q4_0_q8_0(w::AbstractVector{UInt8}, woff::Int, x::Q80Vec, simd::Bool)::Float32
    nb = length(x.d)
    _kq_check_row(w, woff, nb, 18, x)
    sumf = 0.0f0
    @inbounds for ib in 0:(nb-1)
        b = woff + ib * 18
        sumi = simd ? _q40_sumi_avx2(w, b, x, ib) : _q40_sumi_ref(w, b, x, ib)
        sumf += (Float32(sumi) * _kq_f16(w, b)) * x.d[ib+1]
    end
    return sumf
end

# ------------------------------------------------------------- F32 dequant
#
# Ports of dequantize_row_* from ggml-quants.c, one row of ne0 values starting
# at byte woff of w. These are the F32 references the integer kernels are
# checked against (and, when libggml is loaded, against ggml's to_float).

@inline function _kq_get_scale_min_k4(w, b::Int, j::Int)
    # j is 0-based (0..7); get_scale_min_k4 on the packed 12 bytes at b+4.
    q(i) = _kq_u8(w, b + 4 + i)
    if j < 4
        return q(j) & 63, q(j + 4) & 63
    else
        return (q(j + 4) & 0x0f) | ((q(j - 4) >> 6) << 4),
        (q(j + 4) >> 4) | ((q(j) >> 6) << 4)
    end
end

function _kq_dequant_row!(
    out::AbstractVector{Float32},
    w::AbstractVector{UInt8},
    woff::Int,
    ggml_type::Int32,
    ne0::Int,
)
    length(out) == ne0 || error("kq dequant: out length $(length(out)) ≠ ne0 $ne0")
    if ggml_type == GGML_TYPE_Q8_0
        nb = div(ne0, 32)
        for ib in 0:(nb-1)
            b = woff + ib * 34
            d = _kq_f16(w, b)
            for j in 0:31
                out[32*ib+j+1] = d * Float32(reinterpret(Int8, _kq_u8(w, b + 2 + j)))
            end
        end
        return out
    elseif ggml_type == GGML_TYPE_Q4_0
        nb = div(ne0, 32)
        for ib in 0:(nb-1)
            b = woff + ib * 18
            d = _kq_f16(w, b)
            for j in 0:15
                q = _kq_u8(w, b + 2 + j)
                out[32*ib+j+1] = Float32(Int32(q & 0x0f) - Int32(8)) * d
                out[32*ib+j+17] = Float32(Int32(q >> 4) - Int32(8)) * d
            end
        end
        return out
    end
    nb = div(ne0, QK_K)
    for ib in 0:(nb-1)
        b = woff + ib * _iq_block_bytes(ggml_type)
        y = 256 * ib
        if ggml_type == GGML_TYPE_Q4_K
            d = _kq_f16(w, b)
            mn = _kq_f16(w, b + 2)
            for j in 0:3
                sc1, m1 = _kq_get_scale_min_k4(w, b, 2j)
                sc2, m2 = _kq_get_scale_min_k4(w, b, 2j + 1)
                d1 = d * Float32(sc1)
                mm1 = mn * Float32(m1)
                d2 = d * Float32(sc2)
                mm2 = mn * Float32(m2)
                for l in 0:31
                    q = _kq_u8(w, b + 16 + 32j + l)
                    out[y+64j+l+1] = d1 * Float32(q & 0x0f) - mm1
                    out[y+64j+32+l+1] = d2 * Float32(q >> 4) - mm2
                end
            end
        elseif ggml_type == GGML_TYPE_Q5_K
            d = _kq_f16(w, b)
            mn = _kq_f16(w, b + 2)
            for j in 0:3
                sc1, m1 = _kq_get_scale_min_k4(w, b, 2j)
                sc2, m2 = _kq_get_scale_min_k4(w, b, 2j + 1)
                d1 = d * Float32(sc1)
                mm1 = mn * Float32(m1)
                d2 = d * Float32(sc2)
                mm2 = mn * Float32(m2)
                u1 = UInt8(1) << (2j)
                u2 = UInt8(2) << (2j)
                for l in 0:31
                    q = _kq_u8(w, b + 48 + 32j + l)
                    qh = _kq_u8(w, b + 16 + l)
                    v1 = Float32((q & 0x0f) + ((qh & u1) != 0 ? 16 : 0))
                    v2 = Float32((q >> 4) + ((qh & u2) != 0 ? 16 : 0))
                    out[y+64j+l+1] = d1 * v1 - mm1
                    out[y+64j+32+l+1] = d2 * v2 - mm2
                end
            end
        elseif ggml_type == GGML_TYPE_Q6_K
            d = _kq_f16(w, b + 208)
            for h in 0:1, l in 0:31
                ql0 = _kq_u8(w, b + 64h + l)
                ql1 = _kq_u8(w, b + 64h + 32 + l)
                qh = _kq_u8(w, b + 128 + 32h + l)
                q1 = Int32((ql0 & 0x0f) | (((qh >> 0) & 3) << 4)) - Int32(32)
                q2 = Int32((ql1 & 0x0f) | (((qh >> 2) & 3) << 4)) - Int32(32)
                q3 = Int32((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - Int32(32)
                q4 = Int32((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - Int32(32)
                is = 8h + (l >> 4)
                s0 = Float32(reinterpret(Int8, _kq_u8(w, b + 192 + is)))
                s2 = Float32(reinterpret(Int8, _kq_u8(w, b + 192 + is + 2)))
                s4 = Float32(reinterpret(Int8, _kq_u8(w, b + 192 + is + 4)))
                s6 = Float32(reinterpret(Int8, _kq_u8(w, b + 192 + is + 6)))
                out[y+128h+l+1] = d * s0 * Float32(q1)
                out[y+128h+32+l+1] = d * s2 * Float32(q2)
                out[y+128h+64+l+1] = d * s4 * Float32(q3)
                out[y+128h+96+l+1] = d * s6 * Float32(q4)
            end
        elseif ggml_type == GGML_TYPE_Q3_K
            dall = _kq_f16(w, b + 108)
            sc = _kq_scales_q3(w, b + 96)
            for h in 0:1
                for t in 0:3
                    c = 4h + t
                    for half in 0:1
                        i = 8h + 2t + half
                        dl = dall * Float32(sc[i+1])
                        for l in 16half:(16half+15)
                            q2 = (_kq_u8(w, b + 32 + 32h + l) >> (2t)) & 0x03
                            hm = (_kq_u8(w, b + l) >> c) & 0x01
                            v = Int32(q2) - (hm != 0 ? 0 : 4)
                            out[y+128h+32t+l+1] = dl * Float32(v)
                        end
                    end
                end
            end
        elseif ggml_type == GGML_TYPE_Q2_K
            dall = _kq_f16(w, b + 80)
            dmin = _kq_f16(w, b + 82)
            for h in 0:1
                for t in 0:3
                    for half in 0:1
                        i = 8h + 2t + half
                        sc = _kq_u8(w, b + i)
                        dl = dall * Float32(sc & 0x0f)
                        ml = dmin * Float32(sc >> 4)
                        for l in 16half:(16half+15)
                            q2 = (_kq_u8(w, b + 16 + 32h + l) >> (2t)) & 0x03
                            out[y+128h+32t+l+1] = dl * Float32(q2) - ml
                        end
                    end
                end
            end
        else
            error("kq dequant: no F32 reference for ggml type $ggml_type")
        end
    end
    return out
end

# ------------------------------------------------------------- dispatch helpers

# True for the weight types that take a Q8_0 activation (32-value blocks)
# rather than Q8_K, so a shared Q8_K input cannot serve them.
_q80_act_type(ggml_type::Int32) = ggml_type == GGML_TYPE_Q8_0 || ggml_type == GGML_TYPE_Q4_0

# Activation block size and quantizer for a weight type.
_q8k_block_values(ggml_type::Int32) = _q80_act_type(ggml_type) ? 32 : QK_K

function _q8k_quantize_act(ggml_type::Int32, x::AbstractVector{Float32})
    if _q80_act_type(ggml_type)
        return quantize_q8_0!(Q80Vec(div(length(x), 32)), x)
    end
    return quantize_q8_K!(Q8KVec(div(length(x), QK_K)), x)
end

_kq_supported(ggml_type::Int32) = ggml_type in (
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
)

# Called from _dot_q8k in quant_cpu_simd.jl.
@inline function _dot_kq(
    ggml_type::Int32,
    w::AbstractVector{UInt8},
    woff::Int,
    x,
    simd::Bool,
)::Float32
    if ggml_type == GGML_TYPE_Q5_K
        return _dot_q5k_q8k(w, woff, x, simd)
    elseif ggml_type == GGML_TYPE_Q6_K
        return _dot_q6k_q8k(w, woff, x, simd)
    elseif ggml_type == GGML_TYPE_Q4_K
        return _dot_q4k_q8k(w, woff, x, simd)
    elseif ggml_type == GGML_TYPE_Q8_0
        return _dot_q8_0_q8_0(w, woff, x, simd)
    elseif ggml_type == GGML_TYPE_Q4_0
        return _dot_q4_0_q8_0(w, woff, x, simd)
    elseif ggml_type == GGML_TYPE_Q3_K
        return _dot_q3k_q8k(w, woff, x, simd)
    elseif ggml_type == GGML_TYPE_Q2_K
        return _dot_q2k_q8k(w, woff, x, simd)
    end
    error("kquant dot: no kernel for ggml type $ggml_type")
end
