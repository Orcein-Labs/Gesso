# Small one-token F32 products on the decode path: the MoE router, the GDN a/b
# projections, the shared-expert gate and the expert combine. cuBLAS ran these
# as gemvx + splitK, dot + reduce and gemv, and each call cost 12 to 20 us of
# host time against about 4 us for a launch: about 160 calls, 2 to 3 ms, per
# 35B token. Fixed summation order, so results are deterministic.

# Lanes cover adjacent output rows, which are adjacent in a column-major W, so
# every warp load is one contiguous 128-byte span. The k slices give a 256-row
# router enough loads in flight; their partials meet in shared memory.
const _XWT_KSPLIT = 32

@inline function _xwt_row(x, w, o::Int, s::Int, m::Int, k::Int)
    acc = 0.0f0
    if o <= m
        i = s
        @inbounds while i <= k
            acc += w[o, i] * x[1, i]
            i += _XWT_KSPLIT
        end
    end
    return acc
end

@inline function _xwt_store!(dst, acc, lane, s, o::Int, m::Int)
    part = CUDA.@cuStaticSharedMem(Float32, (32, _XWT_KSPLIT))
    @inbounds part[lane, s] = acc
    CUDA.sync_threads()
    if s == 1 && o <= m
        tot = 0.0f0
        @inbounds for q in 1:_XWT_KSPLIT
            tot += part[lane, q]
        end
        @inbounds dst[1, o] = tot
    end
    return
end

# dst[1, o] = sum_i x[1, i] * W[o, i]; blocks.y picks the pair, so two products
# that read the same x share one launch.
function _cuda_xwt_kernel!(d1, w1, d2, w2, x, k::Int)
    lane = Int(threadIdx().x)
    s = Int(threadIdx().y)
    o = (Int(blockIdx().x) - 1) * 32 + lane
    if blockIdx().y == 1
        m = size(w1, 1)
        _xwt_store!(d1, _xwt_row(x, w1, o, s, m, k), lane, s, o, m)
    else
        m = size(w2, 1)
        _xwt_store!(d2, _xwt_row(x, w2, o, s, m, k), lane, s, o, m)
    end
    return
end

_xwt_ok(dst, x, w) = dst isa DenseCuArray{Float32} && x isa DenseCuArray{Float32} &&
    w isa DenseCuArray{Float32} && ndims(w) == 2 && ndims(x) == 2 && size(x, 1) == 1 &&
    length(dst) == size(w, 1) && length(x) == size(w, 2)

_xwt_mat(a) = reshape(a, 1, length(a))

function _cuda_xwt!(dst, x, w)
    m, k = size(w)
    m == 0 && return dst
    @cuda threads = (32, _XWT_KSPLIT) blocks = (cld(m, 32), 1) _cuda_xwt_kernel!(
        _xwt_mat(dst), w, _xwt_mat(dst), w, x, k)
    return dst
end

function _cuda_xwt2!(d1, d2, x, w1, w2)
    k = size(w1, 2)
    size(w2, 2) == k || throw(DimensionMismatch("xwt pair: W1 is $(size(w1)), W2 is $(size(w2))"))
    m = max(size(w1, 1), size(w2, 1))
    @cuda threads = (32, _XWT_KSPLIT) blocks = (cld(m, 32), 2) _cuda_xwt_kernel!(
        _xwt_mat(d1), w1, _xwt_mat(d2), w2, x, k)
    return d1, d2
end

# logits = x * Wᵀ for the router, any token count.
function _cuda_router_logits!(logits, x, w)
    _xwt_ok(logits, x, w) && return _cuda_xwt!(logits, x, w)
    CUDA.CUBLAS.gemm!('N', 'T', _cu_scalar(one(Float32)), x, w, _cu_scalar(zero(Float32)), logits)
    return logits
end

# y[i] = sum_e D[i, e] * ws[e], e in 1:n_used. Each thread owns one row and
# walks the experts in order.
function _cuda_combine_kernel!(y, D, ws, h::Int, n::Int)
    i = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    i <= h || return
    acc = 0.0f0
    @inbounds for e in 1:n
        acc += D[i, e] * ws[e]
    end
    @inbounds y[i] = acc
    return
end

function _cuda_combine!(y, D::CuMatrix{Float32}, ws::CuVector{Float32})
    h, n = size(D)
    length(y) == h && length(ws) == n || throw(DimensionMismatch("combine: y $(size(y)), D $(size(D)), ws $(size(ws))"))
    @cuda threads = 256 blocks = cld(h, 256) _cuda_combine_kernel!(y, D, ws, h, n)
    return y
end

# y[i] += sigmoid(x . g) * sd[i] for one token in one block: the dot reduces
# through warp shuffles and shared memory, then every thread reads the gate.
const _SGATE_THREADS = 1024

function _cuda_shared_gate1_kernel!(y, sd, x, g, h::Int)
    tid = Int(threadIdx().x)
    acc = 0.0f0
    i = tid
    @inbounds while i <= h
        acc += x[i] * g[i]
        i += _SGATE_THREADS
    end
    off = 16
    while off >= 1
        acc += CUDA.shfl_down_sync(0xffffffff, acc, off)
        off >>= 1
    end
    warps = CUDA.@cuStaticSharedMem(Float32, 32)
    lane = (tid - 1) % 32
    wid = (tid - 1) ÷ 32 + 1
    lane == 0 && (@inbounds warps[wid] = acc)
    CUDA.sync_threads()
    if wid == 1
        acc = @inbounds warps[lane + 1]
        off = 16
        while off >= 1
            acc += CUDA.shfl_down_sync(0xffffffff, acc, off)
            off >>= 1
        end
        lane == 0 && (@inbounds warps[1] = acc)
    end
    CUDA.sync_threads()
    sg = 1.0f0 / (1.0f0 + exp(-@inbounds warps[1]))
    i = tid
    @inbounds while i <= h
        y[i] += sg * sd[i]
        i += _SGATE_THREADS
    end
    return
end

function _cuda_shared_gate1!(y, x, sd, g)
    h = length(x)
    @cuda threads = _SGATE_THREADS _cuda_shared_gate1_kernel!(y, sd, x, g, h)
    return y
end
