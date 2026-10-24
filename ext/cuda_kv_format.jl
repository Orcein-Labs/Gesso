# Device side of the KV cache formats (src/Inference/kv_format.jl).
#
# A QuantKV adapts to device arrays inside it, so the decode kernel and the
# prefill tile loader index it like a dense cache. Writes and the q4h rotation
# take one block per (row, head) with one thread per channel: d_head <= 1024
# and a multiple of 32, so each 32-value scale block is exactly one warp and
# its absmax or RMS is a warp reduction.

import Gesso.Inference:
    QuantKV, _kv_write_rows!, _kv_rotate!, _kv_sign, _q4_code, _Q4_LEVELS

_CC.Adapt.adapt_structure(to, x::QuantKV{F}) where {F} =
    QuantKV{F}(_CC.adapt(to, x.codes), _CC.adapt(to, x.scales), x.d)

const _CuKV = Union{CuArray, QuantKV{<:Any, <:CuArray}}

# Every lane ends with the reduction over its warp.
@inline function _kv_warp_allreduce(op, v)
    off = 16
    while off >= 1
        v = op(v, CUDA.shfl_xor_sync(0xffffffff, v, off))
        off >>= 1
    end
    return v
end

# The host's butterflies in the host's order, so a rotated row matches
# _kv_rotate_row! bit for bit.
@inline function _cuda_kv_fwht!(sh, j::Int, d::Int)
    CUDA.sync_threads()
    w = 1
    while w < d
        p = j - 1
        if p & w == 0
            a, b = sh[p+1], sh[p+w+1]
            sh[p+1], sh[p+w+1] = a + b, a - b
        end
        CUDA.sync_threads()
        w <<= 1
    end
    return
end

function _cuda_kv_rotate_kernel!(x, forward::Bool, d::Int)
    sh = CUDA.@cuDynamicSharedMem(Float32, d)
    j = Int(threadIdx().x)
    r = Int(blockIdx().x)
    h = Int(blockIdx().y)
    sh[j] = forward ? x[r, h, j] * _kv_sign(j) : x[r, h, j]
    _cuda_kv_fwht!(sh, j, d)
    s = 1.0f0 / sqrt(Float32(d))
    x[r, h, j] = sh[j] * (forward ? s : s * _kv_sign(j))
    return
end

function _kv_rotate!(x::CuArray{Float32, 3}, forward::Bool)
    rows, heads, d = size(x)
    rows == 0 && return x
    @fastcuda threads = d blocks = (rows, heads) shmem = d * sizeof(Float32) _cuda_kv_rotate_kernel!(
        x,
        forward,
        d,
    )
    return x
end

function _cuda_kv_quant_kernel!(dst::QuantKV{F}, src, row0::Int, d::Int) where {F}
    sh = CUDA.@cuDynamicSharedMem(Float32, d)
    j = Int(threadIdx().x)
    r = Int(blockIdx().x)
    h = Int(blockIdx().y)
    u = row0 + r - 1
    lane = (j - 1) & 31
    b = ((j - 1) >> 5) + 1
    x = Float32(src[r, h, j])
    if F === :q8
        amax = _kv_warp_allreduce(max, abs(x))
        s = Float16(amax / 127)
        inv = s == 0 ? 0.0f0 : 1.0f0 / Float32(s)
        lane == 0 && (dst.scales[u, h, b] = s)
        dst.codes[u, h, j] = unsafe_trunc(Int8, clamp(round(x * inv), -127.0f0, 127.0f0))
    else
        sh[j] = x * _kv_sign(j)
        _cuda_kv_fwht!(sh, j, d)
        x = sh[j] * (1.0f0 / sqrt(Float32(d)))
        s = Float16(sqrt(_kv_warp_allreduce(+, x * x) / 32))
        inv = s == 0 ? 0.0f0 : 1.0f0 / Float32(s)
        lane == 0 && (dst.scales[u, h, b] = s)
        # channel j + 1's code is the high nibble of j's byte
        c = UInt32(_q4_code(x * inv))
        hi = CUDA.shfl_down_sync(0xffffffff, c, 1)
        lane & 1 == 0 && (dst.codes[u, h, (j+1)>>1] = (c | (hi << 4)) % UInt8)
    end
    return
end

function _kv_write_rows!(
    dst::QuantKV{F, <:CuArray},
    src::CUDA.StridedCuArray,
    row0::Int,
) where {F}
    n, n_kv, d = size(src)
    n == 0 && return nothing
    @fastcuda threads = d blocks = (n, n_kv) shmem = d * sizeof(Float32) _cuda_kv_quant_kernel!(
        dst,
        src,
        row0,
        d,
    )
    return nothing
end

# Rows t0+1 .. t0+Tn of KV head hk as dense Float32, for the prefill GEMMs.
function _cuda_kv_tile_kernel!(dst, src, t0::Int, hk::Int, Tn::Int)
    i = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    j = Int(blockIdx().y)
    i <= Tn && (dst[i, j] = Float32(src[t0+i, hk, j]))
    return
end

function _kv_load_tile!(dst::CuArray{Float32, 2}, src, t0::Int, hk::Int, Tn::Int)
    d = size(dst, 2)
    @fastcuda threads = 256 blocks = (cld(Tn, 256), d) _cuda_kv_tile_kernel!(
        dst,
        src,
        t0,
        hk,
        Tn,
    )
    return view(dst, 1:Tn, :)
end
