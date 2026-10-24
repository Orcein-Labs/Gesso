# Fused single-query decode attention on Lava, split over key chunks.
#
# Same contract as the CUDA kernel (cuda_attention.jl). The cache rows 1:K are
# read in place with the GQA mapping kv head = div(h-1, group)+1, so the
# per-layer repeat of the whole cache and the per-head loop both go away.
# kc and vc are (rows, n_kv_heads, d_head), column-major.
#
# One workgroup covers one (query head, key chunk) pair, so the grid is
# n_heads * nsplit workgroups instead of n_heads. Each chunk keeps its own
# running max, running sum and unnormalised accumulator (online softmax over
# tiles of _LAVA_ATTN_TILE rows). A second kernel merges the chunks per head
# with the usual rescale by exp(m_s - M), which is exact up to rounding.
#
# Each tile costs three workgroup barriers. The cross-lane max and sum are
# read by every lane from its own slot, so no broadcast barrier is needed.

const _LAVA_ATTN_LANES = 64
const _LAVA_ATTN_TILE = 512
const _LAVA_ATTN_DMAX = 512

import Gesso.Inference: _attention_decode_device!

@kernel cpu=false function _lava_attn_split_kernel!(
    part,
    q,
    kc,
    vc,
    group::Int,
    K::Int,
    scale::Float32,
    d::Int,
    nsplit::Int,
)
    tl = @index(Local, Linear)
    g = @index(Group, Linear)
    s = (g - 1) % nsplit + 1
    h = (g - 1) ÷ nsplit + 1
    hk = div(h - 1, group) + 1
    k0 = (s - 1) * _LAVA_ATTN_TILE
    kn = clamp(K - k0, 0, _LAVA_ATTN_TILE)
    # static sizes: local memory cannot take a run-time dimension, so the
    # launcher declines any d_head above _LAVA_ATTN_DMAX
    sq = @localmem Float32 (_LAVA_ATTN_DMAX,)
    acc = @localmem Float32 (_LAVA_ATTN_DMAX,)
    sp = @localmem Float32 (_LAVA_ATTN_TILE,)
    pmax = @localmem Float32 (_LAVA_ATTN_LANES,)
    psum = @localmem Float32 (_LAVA_ATTN_LANES,)
    for j in tl:_LAVA_ATTN_LANES:d
        sq[j] = q[1, h, j]
        acc[j] = Float32(0)
    end
    @synchronize
    m = Float32(-Inf)
    l = Float32(0)
    for t0 in 0:_LAVA_ATTN_TILE:(kn-1)
        n = min(_LAVA_ATTN_TILE, kn - t0)
        lmax = Float32(-Inf)
        for i in tl:_LAVA_ATTN_LANES:n
            u = k0 + t0 + i
            x = Float32(0)
            for j in 1:d
                x += sq[j] * kc[u, hk, j]
            end
            x *= scale
            sp[i] = x
            lmax = max(lmax, x)
        end
        pmax[tl] = lmax
        @synchronize
        tmax = pmax[1]
        for i in 2:_LAVA_ATTN_LANES
            tmax = max(tmax, pmax[i])
        end
        # exp(-Inf - finite) is 0, so the first tile rescales the empty
        # accumulator to zero without a special case
        m_new = max(m, tmax)
        alpha = exp(m - m_new)
        lsum = Float32(0)
        for i in tl:_LAVA_ATTN_LANES:n
            p = exp(sp[i] - m_new)
            sp[i] = p
            lsum += p
        end
        psum[tl] = lsum
        @synchronize
        tsum = psum[1]
        for i in 2:_LAVA_ATTN_LANES
            tsum += psum[i]
        end
        l = l * alpha + tsum
        m = m_new
        for j in tl:_LAVA_ATTN_LANES:d
            piece = Float32(0)
            for i in 1:n
                piece += sp[i] * vc[k0+t0+i, hk, j]
            end
            acc[j] = acc[j] * alpha + piece
        end
        @synchronize
    end
    for j in tl:_LAVA_ATTN_LANES:d
        part[s, h, 2+j] = acc[j]
    end
    if tl == 1
        part[s, h, 1] = m
        part[s, h, 2] = l
    end
end

@kernel cpu=false function _lava_attn_combine_kernel!(attn, part, nsplit::Int, d::Int)
    tl = @index(Local, Linear)
    h = @index(Group, Linear)
    M = Float32(-Inf)
    for s in 1:nsplit
        M = max(M, part[s, h, 1])
    end
    L = Float32(0)
    for s in 1:nsplit
        L += exp(part[s, h, 1] - M) * part[s, h, 2]
    end
    inv = Float32(1) / L
    for j in tl:_LAVA_ATTN_LANES:d
        num = Float32(0)
        for s in 1:nsplit
            num += exp(part[s, h, 1] - M) * part[s, h, 2+j]
        end
        attn[1, h, j] = num * inv
    end
end

function _attention_decode_device!(
    ::LavaBackend,
    attn::Lava.LavaArray,
    q::Lava.LavaArray,
    kc::Lava.LavaArray,
    vc::Lava.LavaArray,
    n_heads::Int,
    d_head::Int,
    group::Int,
    K::Int,
)
    K >= 1 ||
        throw(ArgumentError("decode attention needs at least one cache row, got K=$K"))
    d_head <= _LAVA_ATTN_DMAX || return false
    nsplit = cld(K, _LAVA_ATTN_TILE)
    part = Lava.LavaArray{Float32}(undef, nsplit, n_heads, d_head + 2)
    backend = KA.get_backend(attn)
    scale = Float32(1 / sqrt(d_head))
    kern_s = _lava_attn_split_kernel!(backend)
    kern_s(
        part,
        q,
        kc,
        vc,
        Int(group),
        Int(K),
        scale,
        Int(d_head),
        Int(nsplit);
        ndrange=n_heads * nsplit * _LAVA_ATTN_LANES,
        workgroupsize=(_LAVA_ATTN_LANES,),
    )
    kern_c = _lava_attn_combine_kernel!(backend)
    kern_c(
        attn,
        part,
        Int(nsplit),
        Int(d_head);
        ndrange=n_heads * _LAVA_ATTN_LANES,
        workgroupsize=(_LAVA_ATTN_LANES,),
    )
    return true
end
