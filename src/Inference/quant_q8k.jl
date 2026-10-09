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

Q8KVec(nb::Integer) = Q8KVec(
    zeros(Float32, nb),
    zeros(Int8, QK_K * nb),
    zeros(Int16, (QK_K ÷ 16) * nb),
)

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
function quantize_q8_K!(y::Q8KVec, x::AbstractVector{Float32})
    nb = length(y.d)
    length(x) == QK_K * nb ||
        error("quantize_q8_K!: x length $(length(x)) ≠ QK_K * nb = $(QK_K * nb)")
    @inbounds for i in 1:nb
        xo = (i - 1) * QK_K
        bo = (i - 1) * 16
        amax = 0.0f0
        mx = 0.0f0
        for j in 1:QK_K
            ax = abs(x[xo + j])
            if ax > amax
                amax = ax
                mx = x[xo + j]
            end
        end
        if amax == 0.0f0
            y.d[i] = 0.0f0
            for j in 1:QK_K
                y.qs[xo + j] = 0
            end
            for j in 1:16
                y.bsums[bo + j] = 0
            end
            continue
        end
        iscale = -127.0f0 / mx
        for j in 1:QK_K
            v = _q8k_nearest_int(iscale * x[xo + j])
            y.qs[xo + j] = Int8(min(Int32(127), v))
        end
        for j in 1:16
            s = Int32(0)
            for ii in 1:16
                s += Int32(y.qs[xo + (j - 1) * 16 + ii])
            end
            y.bsums[bo + j] = Int16(s)
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
