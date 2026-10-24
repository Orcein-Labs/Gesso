# Float32 exp range reduction and polynomial adapted from Julia 1.12.6
# base/special/exp.jl, MIT license (Julia contributors).
# Scoped to the CPU-order GDN recurrence; explicit Fmas are intentional.
@inline function _lava_gdn_exp32_bits(x::Float32)::UInt32
    isnan(x) && return reinterpret(UInt32, x) | UInt32(0x00400000)
    x > 88.72284f0 && return UInt32(0x7f800000)
    x < -103.97208f0 && return UInt32(0)
    scaled=x*1.442695f0
    # |scaled| < 151 here. Float32 add/sub implements ties-to-even without
    # relying on the device's implementation-defined GLSL Round instruction.
    nf=(scaled+12582912.0f0)-12582912.0f0
    n=unsafe_trunc(Int32, nf)
    r=muladd(nf, -0.69314575f0, x)
    r=muladd(nf, -1.4286068f-6, r)
    # Identical Horner order to Base.evalpoly; explicit scalars avoid the
    # currently invalid Private/PhysicalStorageBuffer constant-tuple call.
    s=muladd(r, 0.00019924171f0, 0.001394858f0)
    s=muladd(r, s, 0.008333249f0)
    s=muladd(r, s, 0.041666217f0)
    s=muladd(r, s, 0.16666667f0)
    s=muladd(r, s, 0.5f0)
    s=muladd(r, s, 1.0f0)
    s=muladd(r, s, 1.0f0)
    bits=reinterpret(UInt32, s)
    exponent=Int32((bits>>23)&UInt32(255))+n
    exponent>=255 && return UInt32(0x7f800000)
    if exponent>0
        return (bits&UInt32(0x007fffff)) | (UInt32(exponent)<<23)
    end
    # Exact power-of-two scaling, with one integer RTE at the subnormal
    # boundary. No floating-point arithmetic produces a subnormal here.
    shift=Int32(1)-exponent
    shift>24 && return UInt32(0)
    mantissa=(bits&UInt32(0x007fffff)) | UInt32(0x00800000)
    whole=mantissa>>shift
    lost=mantissa&((UInt32(1)<<shift)-UInt32(1))
    half=UInt32(1)<<(shift-Int32(1))
    increment=lost>half || (lost==half && (whole&UInt32(1))!=0)
    return whole+UInt32(increment)
end
@inline _lava_gdn_exp32(x::Float32) = reinterpret(Float32, _lava_gdn_exp32_bits(x))
