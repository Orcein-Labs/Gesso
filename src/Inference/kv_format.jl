# KV cache storage formats, picked per generate call by GESSO_KV_FORMAT
# (unset: _kv_default_format):
#
#   f32  the activations as computed: 40 KiB a token on the 35B (1.25 GiB at 32k)
#   f16  half precision, llama.cpp's default cache type: half of f32
#   q8   int8 codes, one Float16 absmax scale per 32 values: ~27% of f32
#   q4h  4-bit codes, one Float16 RMS scale per 32 values, on rows rotated by a
#        signed Walsh-Hadamard transform: ~14% of f32
#
# A quantized cache is an AbstractArray{Float32,3} whose getindex dequantizes,
# so a reader that indexes elements (the CPU decode, the CUDA decode kernel,
# the prefill tile loader) reads it unchanged. A code depends on its whole
# 32-value block, so writers go through _kv_write_rows! instead of setindex!.
#
# q4h rotates each K and V row by R = H*D/sqrt(d) before quantizing (D a fixed
# sign pattern, H the Hadamard matrix). Rotation spreads an outlier channel
# over the whole row, so the 32-value blocks are near-Gaussian and the
# Lloyd-Max levels for N(0, 1) fit them. R is orthonormal, so scores need only
# the query rotated, (Rq)'(Rk) = q'k, and the attention output comes back with
# R' applied once per head: sum_u p_u Rv_u = R sum_u p_u v_u.

const KV_FORMATS = (:f32, :f16, :q8, :q4h)
const _KV_BLOCK = 32

function _kv_format(default::Symbol=:f32)
    haskey(ENV, "GESSO_KV_FORMAT") || return default
    raw = lowercase(strip(ENV["GESSO_KV_FORMAT"]))
    f = Symbol(raw)
    f in KV_FORMATS || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "GESSO_KV_FORMAT=$raw: expected one of $(join(KV_FORMATS, ", "))",
        ),
    )
    return f
end

# Unset GESSO_KV_FORMAT means q8 on CUDA and Lava. On CUDA its logit drift on
# the 35B matches f16 (llama.cpp's default) and its decode beats f32 at every
# length, at 27% of the memory. Lava stores the same codes; its fused decode
# reads them in place and its other readers take a dense Float32 copy of the
# rows they attend (see ext/lava_kv_format.jl).
# The CPU decode reads a quantized cache one getindex per element and has no
# vectorized reader, so it stays f32 until one is measured. A head too narrow
# for one scale block also stays f32.
_kv_default_format(backend::Symbol, d_head::Int) =
    backend in (:cuda, :lava) && d_head % _KV_BLOCK == 0 ? :q8 : :f32

# A device reader that cannot take a cache in its stored format returns a dense
# Float32 view of rows 1:K here. Other backends read the cache as it is.
_kv_dense(x::AbstractArray, K::Int) = x

# The repeat and contraction take an Activation, so a dense reader is wrapped here.
function _kv_readable(x::AbstractArray, K::Int)
    dense = _kv_dense(x, K)
    return Activation(; shape=size(dense), storage=dense)
end

struct QuantKV{F, C <: AbstractArray, S <: AbstractArray} <: AbstractArray{Float32, 3}
    codes::C   # q8: (rows, n_kv, d) Int8; q4h: (rows, n_kv, d ÷ 2) UInt8, two codes a byte
    scales::S  # (rows, n_kv, d ÷ 32) Float16
    d::Int
end
QuantKV{F}(codes::C, scales::S, d::Int) where {F, C, S} = QuantKV{F, C, S}(codes, scales, d)

Base.size(x::QuantKV) = (size(x.codes, 1), size(x.codes, 2), x.d)
Base.IndexStyle(::Type{<:QuantKV}) = IndexCartesian()

# Lloyd-Max reconstruction levels for a unit Gaussian at 16 levels (the
# positive half; codes 8-15 are the negatives of 0-7)
const _Q4_LEVELS =
    (0.1284f0, 0.3881f0, 0.6568f0, 0.9424f0, 1.2562f0, 1.6181f0, 2.0690f0, 2.7326f0)

@inline function _q4_level(c::UInt8)
    a = _Q4_LEVELS[(c&0x07)+1]
    return c >= 0x08 ? -a : a
end

# Nearest level's code for x already in units of its block's RMS. The
# thresholds are the midpoints between adjacent levels.
@inline function _q4_code(x::Float32)
    a = abs(x)
    c = 0x00
    for t in (0.25825f0, 0.52245f0, 0.79960f0, 1.0993f0, 1.43715f0, 1.84355f0, 2.4008f0)
        c += a > t ? 0x01 : 0x00
    end
    return x < 0.0f0 ? c | 0x08 : c
end

Base.@propagate_inbounds function Base.getindex(x::QuantKV{:q8}, u::Int, h::Int, j::Int)
    return Float32(x.codes[u, h, j]) * Float32(x.scales[u, h, ((j-1)>>5)+1])
end

Base.@propagate_inbounds function Base.getindex(x::QuantKV{:q4h}, u::Int, h::Int, j::Int)
    b = x.codes[u, h, ((j-1)>>1)+1]
    c = isodd(j) ? b & 0x0f : b >> 4
    return _q4_level(c) * Float32(x.scales[u, h, ((j-1)>>5)+1])
end

_kv_rotated(::Any) = false
_kv_rotated(::QuantKV{:q4h}) = true

# The sign pattern D: fixed per channel, the same for every row, head and layer.
@inline _kv_sign(j::Int) = ((UInt32(j) * 0x9e3779b1) >> 16) & 0x01 == 0x01 ? -1.0f0 : 1.0f0

function _kv_alloc(fmt::Symbol, like::AbstractArray, rows::Int, n_kv::Int, d::Int)
    zeros_of(T, dims) = fill!(similar(like, T, dims), zero(T))
    fmt === :f32 && return zeros_of(eltype(like), (rows, n_kv, d))
    fmt === :f16 && return zeros_of(Float16, (rows, n_kv, d))
    d % _KV_BLOCK == 0 || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "GESSO_KV_FORMAT=$fmt needs d_head divisible by $_KV_BLOCK, got $d",
        ),
    )
    scales = zeros_of(Float16, (rows, n_kv, d ÷ _KV_BLOCK))
    fmt === :q8 && return QuantKV{:q8}(zeros_of(Int8, (rows, n_kv, d)), scales, d)
    ispow2(d) || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "GESSO_KV_FORMAT=q4h needs a power-of-two d_head for the Hadamard rotation, got $d",
        ),
    )
    return QuantKV{:q4h}(zeros_of(UInt8, (rows, n_kv, d ÷ 2)), scales, d)
end

_release_storage!(x::QuantKV) =
    (_release_storage!(x.codes); _release_storage!(x.scales); nothing)

# Cache rows row0 .. row0 + n - 1 take the n rows of src, (n, n_kv, d).
function _kv_write_rows!(dst::AbstractArray, src::AbstractArray, row0::Int)
    n = size(src, 1)
    @views dst[row0:(row0+n-1), :, :] .= src
    return nothing
end

function _kv_write_rows!(dst::QuantKV, src::AbstractArray, row0::Int)
    n, n_kv, d = size(src)
    buf = Vector{Float32}(undef, d)
    for h in 1:n_kv, r in 1:n
        for j in 1:d
            buf[j] = Float32(src[r, h, j])
        end
        _kv_quantize_row!(dst, buf, row0 + r - 1, h)
    end
    return nothing
end

function _kv_quantize_row!(dst::QuantKV{:q8}, buf::Vector{Float32}, u::Int, h::Int)
    for b in 1:(dst.d÷_KV_BLOCK)
        js = ((b-1)*_KV_BLOCK+1):(b*_KV_BLOCK)
        amax = maximum(abs, view(buf, js))
        s = Float16(amax / 127)
        inv = s == 0 ? 0.0f0 : 1.0f0 / Float32(s)
        dst.scales[u, h, b] = s
        for j in js
            dst.codes[u, h, j] =
                unsafe_trunc(Int8, clamp(round(buf[j] * inv), -127.0f0, 127.0f0))
        end
    end
    return nothing
end

function _kv_quantize_row!(dst::QuantKV{:q4h}, buf::Vector{Float32}, u::Int, h::Int)
    _kv_rotate_row!(buf, true)
    for b in 1:(dst.d÷_KV_BLOCK)
        js = ((b-1)*_KV_BLOCK+1):(b*_KV_BLOCK)
        rms = sqrt(sum(abs2, view(buf, js)) / _KV_BLOCK)
        s = Float16(rms)
        inv = s == 0 ? 0.0f0 : 1.0f0 / Float32(s)
        dst.scales[u, h, b] = s
        for j in first(js):2:last(js)
            dst.codes[u, h, (j+1)>>1] =
                _q4_code(buf[j] * inv) | (_q4_code(buf[j+1] * inv) << 4)
        end
    end
    return nothing
end

# In-place R (forward) or R' (inverse) on one row of power-of-two length.
function _kv_rotate_row!(x::AbstractVector{Float32}, forward::Bool)
    d = length(x)
    if forward
        for j in 1:d
            x[j] *= _kv_sign(j)
        end
    end
    w = 1
    while w < d
        for i in 0:(2w):(d-1), j in 1:w
            a, b = x[i+j], x[i+j+w]
            x[i+j], x[i+j+w] = a + b, a - b
        end
        w <<= 1
    end
    s = 1.0f0 / sqrt(Float32(d))
    for j in 1:d
        x[j] *= forward ? s : s * _kv_sign(j)
    end
    return x
end

# R or R' along the last dim of a (rows, heads, d) array, in place: the query
# before attention reads a rotated cache, the output after it.
function _kv_rotate!(x::AbstractArray, forward::Bool)
    rows, heads, d = size(x)
    buf = Vector{Float32}(undef, d)
    for hh in 1:heads, r in 1:rows
        for j in 1:d
            buf[j] = x[r, hh, j]
        end
        _kv_rotate_row!(buf, forward)
        for j in 1:d
            x[r, hh, j] = buf[j]
        end
    end
    return x
end

# Decode's one-row append for a quantized cache.
function _kv_append_storage!(
    kc::QuantKV,
    vc::QuantKV,
    kh::AbstractArray,
    vh::AbstractArray,
    row::Int,
)
    _kv_write_rows!(kc, kh, row)
    _kv_write_rows!(vc, vh, row)
    return nothing
end
