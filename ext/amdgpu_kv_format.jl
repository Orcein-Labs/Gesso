# Quantized KV cache (q8) on device (AMDGPU / KADeviceBackend). A QuantKV{:q8}
# whose codes and scales are ROCArrays keeps its Int8 codes and Float16 scales
# on the device; the write quantizes rows in a KA kernel, and the fused decode
# (amdgpu_attention.jl) dequantizes in place. Readers that take a dense cache
# (prefill, the per-head path) get a Float32 copy of rows 1:K made here.
#
# The write kernel is the one in ext/lava_kv_format.jl with AMD names. It keeps
# the fma-corrected division, so codes match the host quantizer _kv_quantize_row!
# bit for bit, including on ties. The Float16 scale widens with a plain Float32
# conversion, which is exact and compiles for every AMD target.
# Nothing here runs on ROCm hardware yet (GPU-UNTESTED); the tests run the same
# kernels on KA's CPU backend and compile them for the AMD targets.

import Gesso.Inference: QuantKV, _KV_BLOCK, _kv_dense, _kv_write_rows!

const _AmdQ8 = QuantKV{:q8,<:AMDGPU.ROCArray,<:AMDGPU.ROCArray}

# Quotient correctly rounded: the fma residual orders the floats near a / b.
@inline function _ka_kv_div_rn(a::Float32, b::Float32)
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

# One item per (row, head, block) triple of the source rows.
@kernel function _ka_kv_q8_write_kernel!(codes, scales, @Const(src), row0::Int, n_kv::Int, nb::Int)
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
    s = Float16(_ka_kv_div_rn(amax, 127.0f0))
    inv = s == Float16(0) ? 0.0f0 : _ka_kv_div_rn(1.0f0, Float32(s))
    scales[u, h, b] = s
    for k in 1:_KV_BLOCK
        j = base + k
        q = clamp(round(Float32(src[r, h, j]) * inv), -127.0f0, 127.0f0)
        codes[u, h, j] = unsafe_trunc(Int8, q)
    end
end

@kernel function _ka_kv_q8_dense_kernel!(dst, @Const(codes), @Const(scales), n_kv::Int, d::Int)
    i = @index(Global, Linear)
    j = (i - 1) % d + 1
    t = (i - 1) ÷ d
    h = t % n_kv + 1
    u = t ÷ n_kv + 1
    dst[u, h, j] = Float32(codes[u, h, j]) * Float32(scales[u, h, (j-1)÷_KV_BLOCK+1])
end

# Backend-explicit forms: `ka` is any KA backend, so the tests run these on
# KA.CPU() with host arrays.
function _ka_kv_write_rows!(ka, dst::QuantKV{:q8}, src::AbstractArray, row0::Int; wg::Int=64)
    _ka_check_wg(wg)
    n, n_kv, d = size(src)
    d == dst.d || throw(ArgumentError("KV write needs d=$(dst.d), got $d"))
    n == 0 && return nothing
    nb = d ÷ _KV_BLOCK
    _ka_kv_q8_write_kernel!(ka, wg)(
        dst.codes, dst.scales, src, Int(row0), Int(n_kv), Int(nb); ndrange=n * n_kv * nb,
    )
    return nothing
end

function _ka_kv_dense(ka, x::QuantKV{:q8}, K::Int; wg::Int=64)
    _ka_check_wg(wg)
    1 <= K <= size(x.codes, 1) || throw(ArgumentError("KV dense copy needs 1 <= K <= rows, got K=$K"))
    n_kv = size(x.codes, 2)
    dst = KA.allocate(ka, Float32, K, n_kv, x.d)
    _ka_kv_q8_dense_kernel!(ka, wg)(dst, x.codes, x.scales, n_kv, x.d; ndrange=K * n_kv * x.d)
    return dst
end

# A source that is not already a device Float32 array (a permuted view, say) is
# copied to the device first, so the kernel sees plain device memory.
function _ka_kv_on_device(ka, src::AbstractArray)
    src isa AMDGPU.ROCArray{Float32} && return src
    tmp = KA.allocate(ka, Float32, size(src))
    if src isa Array
        copyto!(tmp, Array{Float32}(src))
    else
        tmp .= src
    end
    return tmp
end

# Production entry points, reached through the generic writers in Inference.
function _kv_write_rows!(dst::_AmdQ8, src::AbstractArray, row0::Int)
    ka = KA.get_backend(dst.codes)
    return _ka_kv_write_rows!(ka, dst, _ka_kv_on_device(ka, src), row0)
end

_kv_dense(x::_AmdQ8, K::Int) = _ka_kv_dense(KA.get_backend(x.codes), x, K)
_kv_dense(x::SubArray{Float32,3,<:_AmdQ8}, K::Int) = _kv_dense(parent(x), K)
