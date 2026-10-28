# Lava storage for the quantized KV cache. A QuantKV{:q8} on the device keeps
# its Int8 codes and Float16 scales in LavaArrays; the write quantizes the rows
# in a kernel. The fused decode (lava_attention.jl) dequantizes in place; the
# prefill and the unfused fallback read a dense Float32 copy of rows 1:K, so
# every reader sees the same values the host dequantizer produces.
#
# Each kernel item owns one (row, head, block) triple and loads its 32 inputs
# again for the second pass instead of keeping them in registers. The rounding
# is round-half-to-even, as Base.round is on the host; the q8 write must match
# _kv_quantize_row! bit for bit, which the tests check on exact ties too.

using Lava.KernelAbstractions: @kernel, @index

import Gesso.Inference:
    QuantKV,
    _KV_BLOCK,
    _kv_alloc,
    _kv_append_storage!,
    _kv_dense,
    _kv_write_rows!,
    _attention_decode_device!

const _LavaQ8 = QuantKV{:q8,<:Lava.LavaArray,<:Lava.LavaArray}

# Vulkan lets OpFDiv miss by 2.5 ulp, and the host quotient is correctly
# rounded, so a code next to a half step can round the other way. The fma
# residual orders the floats near the quotient exactly; walk to the nearest.
# A quotient never falls exactly between two floats, so there is no tie.
@inline function _lava_div_rn(a::Float32, b::Float32)
    q = a / b
    r = abs(fma(-q, b, a))
    for _ in 1:4
        lo, hi = prevfloat(q), nextfloat(q)
        rlo, rhi = abs(fma(-lo, b, a)), abs(fma(-hi, b, a))
        if rlo < r && rlo <= rhi
            q, r = lo, rlo
        elseif rhi < r
            q, r = hi, rhi
        end
    end
    return q
end

# Float32(Float16(x)) on the device came back as x: the driver folds the
# narrowing and widening conversions, so inv was taken from the unrounded scale
# and codes near a half step rounded the other way. Widening the stored bits
# with integer ops leaves no conversion pair to fold. Every step is exact: an
# 11-bit integer times a power of two in [2^-24, 2^6].
@inline function _lava_f16_widen(s::Float16)
    b = reinterpret(UInt16, s)
    e = Int32((b >> 10) & 0x1f)
    m = Int32(b & 0x03ff)
    m = e == Int32(0) ? m : m | Int32(0x0400)
    p = reinterpret(Float32, UInt32(max(e, Int32(1)) - Int32(25) + Int32(127)) << 23)
    v = Float32(m) * p
    return (b & 0x8000) == 0x0000 ? v : -v
end

@kernel cpu=false function _lava_kv_q8_write_kernel!(codes, scales, src, row0::Int, n_kv::Int, nb::Int)
    i = @index(Global, Linear)
    b = (i - 1) % nb + 1
    t = (i - 1) ÷ nb
    h = t % n_kv + 1
    r = t ÷ n_kv + 1
    u = row0 + r - 1
    base = (b - 1) * _KV_BLOCK
    amax = 0.0f0
    for k in 1:_KV_BLOCK
        amax = max(amax, abs(Float32(src[r, h, base+k])))
    end
    s = Float16(_lava_div_rn(amax, 127.0f0))
    inv = s == Float16(0) ? 0.0f0 : _lava_div_rn(1.0f0, _lava_f16_widen(s))
    scales[u, h, b] = s
    for k in 1:_KV_BLOCK
        j = base + k
        q = clamp(round(Float32(src[r, h, j]) * inv), -127.0f0, 127.0f0)
        codes[u, h, j] = unsafe_trunc(Int8, q)
    end
end

@kernel cpu=false function _lava_kv_q8_dense_kernel!(dst, codes, scales, n_kv::Int, d::Int)
    i = @index(Global, Linear)
    j = (i - 1) % d + 1
    t = (i - 1) ÷ d
    h = t % n_kv + 1
    u = t ÷ n_kv + 1
    dst[u, h, j] = Float32(codes[u, h, j]) * Float32(scales[u, h, (j-1)÷_KV_BLOCK+1])
end

function _kv_write_rows!(
    dst::_LavaQ8,
    src::Lava.LavaArray{Float32},
    row0::Int,
)
    n, n_kv, d = size(src)
    nb = d ÷ _KV_BLOCK
    kern = _lava_kv_q8_write_kernel!(KA.get_backend(dst.codes))
    kern(dst.codes, dst.scales, src, Int(row0), Int(n_kv), Int(nb); ndrange=n * n_kv * nb)
    return nothing
end

function _kv_append_storage!(
    kc::_LavaQ8,
    vc::_LavaQ8,
    kh::Lava.LavaArray{Float32},
    vh::Lava.LavaArray{Float32},
    row::Int,
)
    _kv_write_rows!(kc, kh, row)
    _kv_write_rows!(vc, vh, row)
    return nothing
end

# Dense Float32 copy of rows 1:K, for the readers that take a dense cache.
function _lava_kv_q8_rows(x::_LavaQ8, K::Int)
    n_kv = size(x.codes, 2)
    dst = Lava.LavaArray{Float32}(undef, K, n_kv, x.d)
    kern = _lava_kv_q8_dense_kernel!(KA.get_backend(dst))
    kern(dst, x.codes, x.scales, Int(n_kv), Int(x.d); ndrange=K * n_kv * x.d)
    return dst
end

_kv_dense(x::_LavaQ8, K::Int) = _lava_kv_q8_rows(x, K)
_kv_dense(x::SubArray{Float32,3,<:_LavaQ8}, K::Int) = _lava_kv_q8_rows(parent(x), K)

_attention_decode_device!(
    ::LavaBackend,
    attn::Lava.LavaArray,
    q::Lava.LavaArray,
    kc::_LavaQ8,
    vc::_LavaQ8,
    n_heads::Int,
    d_head::Int,
    group::Int,
    K::Int,
) = _lava_attn_decode!(
    attn,
    q,
    kc.codes,
    kc.scales,
    vc.codes,
    vc.scales,
    n_heads,
    d_head,
    group,
    K,
)

# q4h needs a Hadamard rotation and nibble kernels that Lava does not have yet.
# Refusing it here keeps a Lava run from silently falling back to a host copy.
function _kv_alloc(fmt::Symbol, like::Lava.LavaArray, rows::Int, n_kv::Int, d::Int)
    fmt === :q4h && throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "GESSO_KV_FORMAT=q4h is not implemented on Lava; use q8 or f32",
        ),
    )
    return invoke(
        _kv_alloc,
        Tuple{Symbol,AbstractArray,Int,Int,Int},
        fmt,
        like,
        rows,
        n_kv,
        d,
    )
end
