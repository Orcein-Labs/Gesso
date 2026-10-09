# Fused single-query decode attention on device.
#
# One block per query head. The block reads cache rows 1:K in place, with the
# GQA mapping kv head = div(h-1, group)+1, so the per-layer repeat of the whole
# cache and the per-head GEMV loop both disappear. Keys are processed in tiles
# of _ATTN_TILE rows with an online softmax (running max and sum, rescaled
# accumulator), so K is not bounded by shared memory. The only per-call sizes
# are d_head and the tile, both in dynamic shared memory.
#
# Memory layout: kc is (rows, n_kv_heads, d_head), column-major, so for a fixed
# head and dim the rows are contiguous. Score work maps one thread per key
# (consecutive keys are consecutive addresses across a warp). The value
# accumulation maps one warp per output dim with lanes striding over keys, so
# both phases read the cache coalesced.

import Gesso.Inference: _attention_decode_device!

const _ATTN_TILE = 1024
const _ATTN_THREADS = 256

@inline function _attn_warp_reduce(op, v)
    off = 16
    while off >= 1
        v = op(v, CUDA.shfl_down_sync(0xffffffff, v, off))
        off >>= 1
    end
    return v
end

# Block-wide reduction through a 33-slot shared region: slots 1..nw hold the
# per-warp partials, slot 33 the result. The second barrier keeps the result
# slot from being overwritten before every thread has read it.
@inline function _attn_block_reduce(op, v, ident, sh, red, tid, nw)
    lane = (tid - 1) % 32
    wid = (tid - 1) ÷ 32
    v = _attn_warp_reduce(op, v)
    if lane == 0
        sh[red + wid + 1] = v
    end
    CUDA.sync_threads()
    if wid == 0
        x = lane < nw ? sh[red + lane + 1] : ident
        x = _attn_warp_reduce(op, x)
        if lane == 0
            sh[red + 33] = x
        end
    end
    CUDA.sync_threads()
    return sh[red + 33]
end

function _cuda_attn_decode_kernel!(attn, q, kc, vc, group::Int, K::Int, scale::Float32, d::Int)
    # shared layout: scores/probs [1, TILE] | q [1, d] | accumulator [1, d] | reduce [33]
    sh = CUDA.@cuDynamicSharedMem(Float32, _ATTN_TILE + 2 * d + 33)
    qo = _ATTN_TILE
    ao = _ATTN_TILE + d
    ro = _ATTN_TILE + 2 * d
    tid = Int(threadIdx().x)
    nt = Int(blockDim().x)
    nw = nt ÷ 32
    wid = (tid - 1) ÷ 32
    lane = (tid - 1) % 32
    h = Int(blockIdx().x)
    hk = div(h - 1, group) + 1

    for j in tid:nt:d
        sh[qo + j] = q[1, h, j]
        sh[ao + j] = 0.0f0
    end
    CUDA.sync_threads()

    m = -Inf32
    l = 0.0f0
    for t0 in 0:_ATTN_TILE:(K - 1)
        n = min(_ATTN_TILE, K - t0)

        lmax = -Inf32
        for i in tid:nt:n
            u = t0 + i
            s = 0.0f0
            for j in 1:d
                s += sh[qo + j] * kc[u, hk, j]
            end
            s *= scale
            sh[i] = s
            lmax = max(lmax, s)
        end
        tmax = _attn_block_reduce(max, lmax, -Inf32, sh, ro, tid, nw)

        # exp(-Inf - finite) is 0, so the first tile rescales the empty
        # accumulator to zero without a special case
        m_new = max(m, tmax)
        alpha = exp(m - m_new)
        lsum = 0.0f0
        for i in tid:nt:n
            p = exp(sh[i] - m_new)
            sh[i] = p
            lsum += p
        end
        tsum = _attn_block_reduce(+, lsum, 0.0f0, sh, ro, tid, nw)
        l = l * alpha + tsum
        m = m_new

        for j in (wid + 1):nw:d
            part = 0.0f0
            for i in (lane + 1):32:n
                part += sh[i] * vc[t0 + i, hk, j]
            end
            part = _attn_warp_reduce(+, part)
            if lane == 0
                sh[ao + j] = sh[ao + j] * alpha + part
            end
        end
        CUDA.sync_threads()
    end

    inv = 1.0f0 / l
    for j in tid:nt:d
        attn[1, h, j] = sh[ao + j] * inv
    end
    return
end

function _attention_decode_device!(
    ::CUDABackend,
    attn::CuArray,
    q::CuArray,
    kc::CuArray,
    vc::CuArray,
    n_heads::Int,
    d_head::Int,
    group::Int,
    K::Int,
)
    K >= 1 || throw(ArgumentError("decode attention needs at least one cache row, got K=$K"))
    shmem = (_ATTN_TILE + 2 * d_head + 33) * sizeof(Float32)
    @cuda threads = _ATTN_THREADS blocks = n_heads shmem = shmem _cuda_attn_decode_kernel!(
        attn, q, kc, vc, group, K, Float32(1 / sqrt(d_head)), d_head,
    )
    return true
end
