# Q8_K activation quantization: the int8 side of the native CPU quantized
# dot products (quant_cpu_simd.jl).
#
# Port of ggml's quantize_row_q8_K_ref (ggml/src/ggml-quants.c, MIT License,
# Copyright (c) 2023-2026 The ggml authors). No ggml code is linked or called.
#
# Block layout is ggml's block_q8_K, stored as struct-of-arrays so the SIMD
# kernels can load quants directly:
#   d      one Float32 scale per 256 values         (ggml: float d)
#   qs     256 int8 quants per block                (ggml: int8_t qs[256])
#   bsums  16 int16 sums of 16 quants per block     (ggml: int16_t bsums[16])
# One ggml block is 4 + 256 + 32 = 292 bytes; Q8K_BLOCK_BYTES records that.

const Q8K_BLOCK_BYTES = 292

"""
    Q8KVec(nb)

Q8_K quantized row of `nb` blocks (`256 * nb` values). Fields are the
struct-of-arrays form of ggml's `block_q8_K`: `d[i]`, `qs[256(i-1)+1:256i]`,
`bsums[16(i-1)+1:16i]`.
"""
struct Q8KVec
    d::Vector{Float32}
    qs::Vector{Int8}
    bsums::Vector{Int16}
end

Q8KVec(nb::Integer) =
    Q8KVec(zeros(Float32, nb), zeros(Int8, QK_K * nb), zeros(Int16, (QK_K ÷ 16) * nb))

Base.length(x::Q8KVec) = QK_K * length(x.d)

# ggml nearest_int: round-half-to-even via the 2^23 magic-number trick. The
# float add is a separate IEEE operation (Julia never contracts to FMA).
@inline function _q8k_nearest_int(f::Float32)::Int32
    v = f + 12582912.0f0
    return (reinterpret(Int32, v) & Int32(0x007fffff)) - Int32(0x00400000)
end

"""
    quantize_q8_K!(y::Q8KVec, x::AbstractVector{Float32}) -> y

Quantize `x` (finite values, `length(x) == length(y)`) to Q8_K blocks, bit
for bit as ggml's `quantize_row_q8_K_ref`. For an all-zero block ggml leaves
`bsums` unset; here they are written as 0, which is the sum of the zero quants.
"""
function quantize_q8_K!(y::Q8KVec, x::AbstractArray{Float32})
    nb = length(y.d)
    length(x) == QK_K * nb ||
        error("quantize_q8_K!: x length $(length(x)) ≠ QK_K * nb = $(QK_K * nb)")
    for i in 1:nb
        _quantize_q8_K_block!(y, i, x, (i - 1) * QK_K)
    end
    return y
end

# Block i of quantize_q8_K!, from the values x[xo+1:xo+QK_K].
@inline function _quantize_q8_K_block!(y::Q8KVec, i::Int, x::AbstractArray{Float32}, xo::Int)
    @inbounds begin
        qo = (i - 1) * QK_K
        bo = (i - 1) * 16
        # The serial reference keeps the first x[j] whose |x[j]| beats every
        # earlier one, skipping NaN. On the magnitude bits of a non-NaN float
        # the order is the unsigned integer order, so a vector max of those
        # bits (NaN mapped to 0) gives amax, and the first j holding it gives mx.
        ab = UInt32(0)
        for j in 1:QK_K
            b = reinterpret(UInt32, x[xo+j]) & 0x7fffffff
            ab = max(ab, ifelse(b > 0x7f800000, UInt32(0), b))
        end
        amax = reinterpret(Float32, ab)
        mx = 0.0f0
        for j in 1:QK_K
            if reinterpret(UInt32, x[xo+j]) & 0x7fffffff == ab
                mx = x[xo+j]
                break
            end
        end
        if amax == 0.0f0
            y.d[i] = 0.0f0
            for j in 1:QK_K
                y.qs[qo+j] = 0
            end
            for j in 1:16
                y.bsums[bo+j] = 0
            end
            return y
        end
        iscale = -127.0f0 / mx
        # Int8(v) checks every value and keeps the loop scalar; checking the
        # block minimum once throws for the same blocks (non-finite input).
        vmin = typemax(Int32)
        for j in 1:QK_K
            v = min(Int32(127), _q8k_nearest_int(iscale * x[xo+j]))
            vmin = min(vmin, v)
            y.qs[qo+j] = v % Int8
        end
        vmin < -128 && throw(InexactError(:trunc, Int8, vmin))
        for j in 1:16
            s = Int32(0)
            for ii in 1:16
                s += Int32(y.qs[qo+(j-1)*16+ii])
            end
            y.bsums[bo+j] = Int16(s)
        end
        y.d[i] = 1.0f0 / iscale
    end
    return y
end

"""
    dequantize_q8_K(y::Q8KVec) -> Vector{Float32}

`d * qs` per value, the reference for checking the quantizer's error budget.
"""
function dequantize_q8_K(y::Q8KVec)
    out = Vector{Float32}(undef, length(y))
    @inbounds for i in eachindex(out)
        b = (i - 1) ÷ QK_K + 1
        out[i] = y.d[b] * Float32(y.qs[i])
    end
    return out
end

# Q8_0 activation quantization (ggml block_q8_0), the int8 side of the Q8_0
# weight kernel.
#
# Port of ggml's quantize_row_q8_0_ref (ggml/src/ggml-quants.c, MIT License,
# Copyright (c) 2023-2026 The ggml authors). No ggml code is linked or called.
#   d   one Float32 scale per 32 values, rounded through Float16 as ggml stores
#       it (ggml: ggml_half d)
#   qs  32 int8 quants per block (ggml: int8_t qs[32]), rounded half away from
#       zero as roundf does
# One ggml block is 2 + 32 = 34 bytes.

"""
    Q80Vec(nb)

Q8_0 quantized row of `nb` blocks (`32 * nb` values), struct-of-arrays form of
ggml's `block_q8_0`: `d[i]`, `qs[32(i-1)+1:32i]`.
"""
struct Q80Vec
    d::Vector{Float32}
    qs::Vector{Int8}
end

Q80Vec(nb::Integer) = Q80Vec(zeros(Float32, nb), zeros(Int8, 32 * nb))

Base.length(x::Q80Vec) = 32 * length(x.d)

"""
    quantize_q8_0!(y::Q80Vec, x::AbstractVector{Float32}) -> y

Quantize `x` (finite values, `length(x) == length(y)`) to Q8_0 blocks, as
ggml's `quantize_row_q8_0_ref`.
"""
function quantize_q8_0!(y::Q80Vec, x::AbstractVector{Float32})
    nb = length(y.d)
    length(x) == 32 * nb ||
        error("quantize_q8_0!: x length $(length(x)) ≠ 32 * nb = $(32 * nb)")
    for i in 1:nb
        _quantize_q8_0_block!(y, i, x, (i - 1) * 32)
    end
    return y
end

# Block i of quantize_q8_0!, from the values x[xo+1:xo+32].
@inline function _quantize_q8_0_block!(y::Q80Vec, i::Int, x::AbstractArray{Float32}, xo::Int)
    @inbounds begin
        qo = (i - 1) * 32
        amax = 0.0f0
        for j in 1:32
            amax = max(amax, abs(x[xo+j]))
        end
        d = amax / 127.0f0
        id = d != 0.0f0 ? 1.0f0 / d : 0.0f0
        y.d[i] = Float32(Float16(d))
        # As in _quantize_q8_K_block!, one range check per block rather than
        # a checked Int8 per value; NaN fails both comparisons.
        ok = true
        for j in 1:32
            r = round(x[xo+j] * id, RoundNearestTiesAway)
            ok &= (r >= -128.0f0) & (r <= 127.0f0)
            y.qs[qo+j] = unsafe_trunc(Int32, r) % Int8
        end
        ok || throw(InexactError(:trunc, Int8, maximum(j -> abs(x[xo+j] * id), 1:32)))
    end
    return y
end
