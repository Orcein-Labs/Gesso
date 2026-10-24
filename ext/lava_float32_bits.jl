# Binary32 round-to-nearest-even boundaries carried through integer encodings.
# Native GPU Float32 arithmetic may flush subnormals. These helpers allow the
# shared gate to execute each Float32 operation in Float64 and explicitly round
# before the next operation, including signed zero and subnormal values.
# Pure scalar Julia; loading this internal module never initializes a device.
module LavaFloat32Bits
@inline function round_shift_even(m::UInt64, s::Int32)::UInt64
    q=m>>s
    lost=m&((UInt64(1)<<s)-UInt64(1))
    half=UInt64(1)<<(s-Int32(1))
    q+UInt64(lost>half || (lost==half && (q&UInt64(1))!=0))
end
# Integer-only binary64 -> binary32 RTE. No floating operation may flush a tiny result.
@inline function f64_to_f32_bits(u::UInt64)::UInt32
    sign=unsafe_trunc(UInt32, u>>32)&UInt32(0x80000000)
    exponent=Int32((u>>52)&UInt64(0x7ff))
    fraction=u&UInt64(0x000fffffffffffff)
    if exponent==2047
        payload=fraction==0 ? UInt32(0) :
                unsafe_trunc(UInt32, fraction>>29)|UInt32(0x00400000)
        return sign|UInt32(0x7f800000)|payload
    end
    exponent==0&&return sign
    e=exponent-Int32(1023)
    e>127&&return sign|UInt32(0x7f800000)
    m=fraction|UInt64(0x0010000000000000)
    if e>=-126
        q=round_shift_even(m, Int32(29))
        if q==UInt64(0x01000000)
            q>>=1
            e+=1
        end
        e>127&&return sign|UInt32(0x7f800000)
        return sign|(UInt32(e+127)<<23)|(unsafe_trunc(UInt32, q)&UInt32(0x007fffff))
    end
    shift=-e-Int32(97)
    shift>53&&return sign
    sign|unsafe_trunc(UInt32, round_shift_even(m, shift))
end


# Exact IEEE binary32 expansion, including normalization of subnormals.
# NaNs keep their sign/high payload and are quieted, matching CPU conversion.
@inline function f32_to_f64_bits(u::UInt32)::UInt64
    sign=UInt64(u&UInt32(0x80000000))<<32
    exponent=(u>>23)&UInt32(0xff)
    fraction=u&UInt32(0x007fffff)
    if exponent==UInt32(0xff)
        payload=UInt64(fraction)<<29
        fraction!=0&&(payload|=UInt64(0x0008000000000000))
        return sign|UInt64(0x7ff0000000000000)|payload
    elseif exponent==0
        fraction==0&&return sign
        top=UInt64(fraction)
        position=Int32(0)
        while top>=UInt64(2)
            top>>=1
            position+=Int32(1)
        end
        normalized=UInt64(fraction)<<(Int32(52)-position)
        return sign|(UInt64(position+Int32(874))<<52)|(
            normalized&UInt64(0x000fffffffffffff)
        )
    end
    sign|((UInt64(exponent)+UInt64(896))<<52)|(UInt64(fraction)<<29)
end

@inline widen32(x::Float32) = reinterpret(Float64, f32_to_f64_bits(reinterpret(UInt32, x)))
@inline round32_bits(x::Float64) = f64_to_f32_bits(reinterpret(UInt64, x))
@inline rounded32_wide(x::Float64) = reinterpret(Float64, f32_to_f64_bits(round32_bits(x)))
end
