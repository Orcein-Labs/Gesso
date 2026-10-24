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

import Gesso.Inference:
    _attention_decode_device!,
    _attention_prefill_device!,
    _attention_partial_device!,
    _partial_download!

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

# every lane gets the result
@inline function _attn_warp_allreduce(op, v)
    off = 16
    while off >= 1
        v = op(v, CUDA.shfl_xor_sync(0xffffffff, v, off))
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
        sh[red+wid+1] = v
    end
    CUDA.sync_threads()
    if wid == 0
        x = lane < nw ? sh[red+lane+1] : ident
        x = _attn_warp_reduce(op, x)
        if lane == 0
            sh[red+33] = x
        end
    end
    CUDA.sync_threads()
    return sh[red+33]
end

function _cuda_attn_decode_kernel!(
    attn,
    q,
    kc,
    vc,
    group::Int,
    K::Int,
    scale::Float32,
    d::Int,
)
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
        sh[qo+j] = q[1, h, j]
        sh[ao+j] = 0.0f0
    end
    CUDA.sync_threads()

    m = -Inf32
    l = 0.0f0
    for t0 in 0:_ATTN_TILE:(K-1)
        n = min(_ATTN_TILE, K - t0)

        lmax = -Inf32
        for i in tid:nt:n
            u = t0 + i
            s = 0.0f0
            for j in 1:d
                s += sh[qo+j] * kc[u, hk, j]
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

        for j in (wid+1):nw:d
            part = 0.0f0
            for i in (lane+1):32:n
                part += sh[i] * vc[t0+i, hk, j]
            end
            part = _attn_warp_reduce(+, part)
            if lane == 0
                sh[ao+j] = sh[ao+j] * alpha + part
            end
        end
        CUDA.sync_threads()
    end

    inv = 1.0f0 / l
    for j in tid:nt:d
        attn[1, h, j] = sh[ao+j] * inv
    end
    return
end

function _attention_decode_device!(
    ::CUDABackend,
    attn::CuArray,
    q::CuArray,
    kc::_CuKV,
    vc::_CuKV,
    n_heads::Int,
    d_head::Int,
    group::Int,
    K::Int,
)
    K >= 1 ||
        throw(ArgumentError("decode attention needs at least one cache row, got K=$K"))
    n_kv = div(n_heads, group)
    scale = Float32(1 / sqrt(d_head))
    group == 8 &&
        return _attn_decode_split!(attn, q, kc, vc, n_kv, d_head, K, scale, Val(8))
    group == 4 &&
        return _attn_decode_split!(attn, q, kc, vc, n_kv, d_head, K, scale, Val(4))
    group == 2 &&
        return _attn_decode_split!(attn, q, kc, vc, n_kv, d_head, K, scale, Val(2))
    group == 1 &&
        return _attn_decode_split!(attn, q, kc, vc, n_kv, d_head, K, scale, Val(1))
    shmem = (_ATTN_TILE + 2 * d_head + 33) * sizeof(Float32)
    @fastcuda threads = _ATTN_THREADS blocks = n_heads shmem = shmem _cuda_attn_decode_kernel!(
        attn,
        q,
        kc,
        vc,
        group,
        K,
        Float32(1 / sqrt(d_head)),
        d_head,
    )
    return true
end

# Split-K decode. The per-head kernel above runs n_heads blocks, each reading
# its kv head's whole cache: at K=4096 on the 35B (16 heads over 2 kv heads)
# that was 1.3-3.4 ms a layer, the cache read 8 times at 40 GB/s. Here a block
# takes one kv head and a chunk of C keys, scores all G query heads of the
# group from a single read of each row, and leaves the chunk's unnormalised
# output with its max and sum; _cuda_attn_combine_kernel! folds the chunks.
# C is picked for about _ATTN_SPLIT_BLOCKS blocks.
const _ATTN_SPLIT_BLOCKS = 128
const _ATTN_SPLIT_W = 32

mutable struct _AttnSplitScratch
    o::Union{Nothing, CuVector{Float32}}
    ml::Union{Nothing, CuVector{Float32}}
end
const _ATTN_SPLIT = _AttnSplitScratch(nothing, nothing)

function _attn_split_buf(buf, n::Int)
    buf !== nothing && length(buf) >= n && return buf
    buf === nothing || CUDA.unsafe_free!(buf)
    return CuVector{Float32}(undef, cld(n, 1 << 14) << 14)
end

# The cache is (rows, kv heads, d): a key's dims are rows*kv_heads apart, so
# the block stages tiles of W dims over its chunk, read along the keys, into
# shared rows padded to W+1 (key-major writes and dim-major reads both miss
# bank conflicts). Every load of the thread's share is issued before the
# first store: one at a time, each waited out its own DRAM latency.
@inline function _attn_tile_load!(
    sh,
    to::Int,
    src,
    t0::Int,
    hk::Int,
    j0::Int,
    n::Int,
    W::Int,
    tid::Int,
    ::Val{C},
    ::Val{R},
    ::Val,
) where {C, R}
    vals = ntuple(Val(R)) do r
        x = tid - 1 + (r - 1) * _ATTN_THREADS
        jj = x ÷ C
        i = x - jj * C
        (i < n && jj < W) ? Float32(@inbounds(src[t0+i+1, hk, j0+jj+1])) : 0.0f0
    end
    for r in 1:R
        x = tid - 1 + (r - 1) * _ATTN_THREADS
        jj = x ÷ C
        i = x - jj * C
        @inbounds sh[to+i*(_ATTN_SPLIT_W+1)+jj+1] = vals[r]
    end
    return
end

# The f16 and quantized caches load a 4-row group per thread instead of one
# element: the generic path above dequantized each element on its own, 4x the
# loads and a scale fetch per value, and read q8 at 13 GB/s and q4h at 7 against
# 150+ for f32. A column's 4 consecutive rows are adjacent in memory (rows %
# 4 == 0 and t0 % 4 == 0 keep the group aligned), so one load fetches them.
# A tile's 32 dims are one scale block, so a thread's 4 row scales are one
# 8-byte load per tile. Units are (row group, column) with the group fastest:
# a warp reads contiguous runs of the column, and a thread's group stays put
# across its units because 256 is a multiple of C ÷ 4.
@inline _attn_vload(::Type{T}, a, lin::Int, ::Val{A}) where {T, A} =
    unsafe_load(reinterpret(Core.LLVMPtr{T, 1}, pointer(a, lin)), 1, Val(A))

@inline _attn_f16_bits(w::UInt64, k::Int) =
    Float32(reinterpret(Float16, (w >> (16 * k)) % UInt16))

@inline function _attn_tile_store4!(
    sh,
    to::Int,
    i0::Int,
    jj::Int,
    n::Int,
    v::NTuple{4, Float32},
)
    for k in 0:3
        i = i0 + k
        @inbounds sh[to+i*(_ATTN_SPLIT_W+1)+jj+1] = i < n ? v[k+1] : 0.0f0
    end
    return
end

# select chain, not a tuple index: a dynamic index into _Q4_LEVELS can spill
# the table to local memory
@inline function _attn_q4_level(c::UInt32)
    m = c & 0x07
    a =
        m < 0x04 ?
        (
            m < 0x02 ? (m == 0x00 ? _Q4_LEVELS[1] : _Q4_LEVELS[2]) :
            (m == 0x02 ? _Q4_LEVELS[3] : _Q4_LEVELS[4])
        ) :
        (
            m < 0x06 ? (m == 0x04 ? _Q4_LEVELS[5] : _Q4_LEVELS[6]) :
            (m == 0x06 ? _Q4_LEVELS[7] : _Q4_LEVELS[8])
        )
    return c >= 0x08 ? -a : a
end

@inline function _attn_tile_scales(scales, t0::Int, i0::Int, hk::Int, j0::Int, n::Int)
    rows, nkv = size(scales, 1), size(scales, 2)
    i0 < n || return (0.0f0, 0.0f0, 0.0f0, 0.0f0)
    w = _attn_vload(
        UInt64,
        scales,
        t0 + i0 + (hk - 1) * rows + (j0 >> 5) * rows * nkv + 1,
        Val(8),
    )
    return ntuple(k -> _attn_f16_bits(w, k - 1), Val(4))
end

@inline function _attn_tile_load!(
    sh,
    to::Int,
    src::QuantKV{:q8, <:CuDeviceArray},
    t0::Int,
    hk::Int,
    j0::Int,
    n::Int,
    W::Int,
    tid::Int,
    ::Val{C},
    ::Val,
    ::Val{true},
) where {C}
    CG = C ÷ 4
    NR = CG * _ATTN_SPLIT_W ÷ _ATTN_THREADS
    codes = src.codes
    rows, nkv = size(codes, 1), size(codes, 2)
    i0 = 4 * ((tid - 1) % CG)
    sc = _attn_tile_scales(src.scales, t0, i0, hk, j0, n)
    base = t0 + i0 + (hk - 1) * rows + 1
    ws = ntuple(Val(NR)) do r
        jj = (tid - 1 + (r - 1) * _ATTN_THREADS) ÷ CG
        i0 < n ? _attn_vload(UInt32, codes, base + (j0 + jj) * rows * nkv, Val(4)) :
        UInt32(0)
    end
    for r in 1:NR
        jj = (tid - 1 + (r - 1) * _ATTN_THREADS) ÷ CG
        w = ws[r]
        v = ntuple(
            k -> Float32(reinterpret(Int8, (w >> (8 * (k - 1))) % UInt8)) * sc[k],
            Val(4),
        )
        _attn_tile_store4!(sh, to, i0, jj, n, v)
    end
    return
end

# a byte holds channels 2b-1 (low nibble) and 2b, so a UInt32 is 4 rows x 2
# dims and a tile is half the units of q8: at C = 32 half the threads idle
@inline function _attn_tile_load!(
    sh,
    to::Int,
    src::QuantKV{:q4h, <:CuDeviceArray},
    t0::Int,
    hk::Int,
    j0::Int,
    n::Int,
    W::Int,
    tid::Int,
    ::Val{C},
    ::Val,
    ::Val{true},
) where {C}
    CG = C ÷ 4
    NU = CG * (_ATTN_SPLIT_W ÷ 2)
    NR = cld(NU, _ATTN_THREADS)
    codes = src.codes
    rows, nkv = size(codes, 1), size(codes, 2)
    i0 = 4 * ((tid - 1) % CG)
    sc = _attn_tile_scales(src.scales, t0, i0, hk, j0, n)
    base = t0 + i0 + (hk - 1) * rows + 1
    ws = ntuple(Val(NR)) do r
        u = tid - 1 + (r - 1) * _ATTN_THREADS
        jb = u ÷ CG
        (u < NU && i0 < n) ?
        _attn_vload(UInt32, codes, base + ((j0 >> 1) + jb) * rows * nkv, Val(4)) :
        UInt32(0)
    end
    for r in 1:NR
        u = tid - 1 + (r - 1) * _ATTN_THREADS
        u < NU || continue
        jb = u ÷ CG
        w = ws[r]
        lo = ntuple(k -> _attn_q4_level((w >> (8 * (k - 1))) & 0x0f) * sc[k], Val(4))
        hi = ntuple(k -> _attn_q4_level((w >> (8 * (k - 1) + 4)) & 0x0f) * sc[k], Val(4))
        _attn_tile_store4!(sh, to, i0, 2 * jb, n, lo)
        _attn_tile_store4!(sh, to, i0, 2 * jb + 1, n, hi)
    end
    return
end

@inline function _attn_tile_load!(
    sh,
    to::Int,
    src::CuDeviceArray{Float16, 3},
    t0::Int,
    hk::Int,
    j0::Int,
    n::Int,
    W::Int,
    tid::Int,
    ::Val{C},
    ::Val,
    ::Val{true},
) where {C}
    CG = C ÷ 4
    NR = CG * _ATTN_SPLIT_W ÷ _ATTN_THREADS
    rows, nkv = size(src, 1), size(src, 2)
    i0 = 4 * ((tid - 1) % CG)
    base = t0 + i0 + (hk - 1) * rows + 1
    ws = ntuple(Val(NR)) do r
        jj = (tid - 1 + (r - 1) * _ATTN_THREADS) ÷ CG
        (i0 < n && jj < W) ?
        _attn_vload(UInt64, src, base + (j0 + jj) * rows * nkv, Val(8)) : UInt64(0)
    end
    for r in 1:NR
        jj = (tid - 1 + (r - 1) * _ATTN_THREADS) ÷ CG
        jj < W || continue
        w = ws[r]
        _attn_tile_store4!(sh, to, i0, jj, n, ntuple(k -> _attn_f16_bits(w, k - 1), Val(4)))
    end
    return
end

# whether a cache takes the 4-row loads; f32 keeps the generic path
_attn_vec4(::Any) = false
_attn_vec4(x::QuantKV{<:Any, <:CuArray}) = size(x, 1) % 4 == 0
_attn_vec4(x::CuArray{Float16, 3}) = size(x, 1) % 4 == 0

# Rows r0+1:K are attended, so r0 only moves where the chunks start. The
# default decode passes r0 = 0 and gets the same tiles as before.
function _cuda_attn_split_kernel!(
    op,
    ml,
    q,
    kc,
    vc,
    r0::Int,
    K::Int,
    scale::Float32,
    d::Int,
    ::Val{G},
    ::Val{C},
    ::Val{V},
) where {G, C, V}
    R = C * _ATTN_SPLIT_W ÷ _ATTN_THREADS
    WP = _ATTN_SPLIT_W + 1
    # shared layout: q [d, G] | scores, then probs [C, G] | tile [W+1, C]
    sh = CUDA.@cuDynamicSharedMem(Float32, G * d + G * C + WP * C)
    po = G * d
    to = po + G * C
    tid = Int(threadIdx().x)
    nt = _ATTN_THREADS
    nw = nt ÷ 32
    wid = (tid - 1) ÷ 32
    lane = (tid - 1) % 32
    hk = Int(blockIdx().x)
    s = Int(blockIdx().y)
    t0 = r0 + (s - 1) * C
    n = min(C, K - t0)
    h0 = (hk - 1) * G
    nh = Int(gridDim().x) * G

    @inbounds begin
        for g in 1:G, j in tid:nt:d
            sh[(g-1)*d+j] = q[1, h0+g, j]
        end
        # the score loop below keeps this mapping, so each thread clears what it adds to
        for x in tid:nt:(G*C)
            sh[po+x] = 0.0f0
        end

        for j0 in 0:_ATTN_SPLIT_W:(d-1)
            W = min(_ATTN_SPLIT_W, d - j0)
            CUDA.sync_threads()
            _attn_tile_load!(sh, to, kc, t0, hk, j0, n, W, tid, Val(C), Val(R), Val(V))
            CUDA.sync_threads()
            for x in tid:nt:(G*C)
                g0 = (x - 1) ÷ C
                i = x - g0 * C
                acc = 0.0f0
                for jj in 1:W
                    acc += sh[g0*d+j0+jj] * sh[to+(i-1)*WP+jj]
                end
                sh[po+x] += acc
            end
        end
        CUDA.sync_threads()

        # one warp per query head: the chunk's max, then probabilities in place
        for g in (wid+1):nw:G
            b = po + (g - 1) * C
            mx = -Inf32
            for i in (lane+1):32:n
                mx = max(mx, sh[b+i] * scale)
            end
            mx = _attn_warp_allreduce(max, mx)
            sm = 0.0f0
            for i in (lane+1):32:n
                p = exp(sh[b+i] * scale - mx)
                sh[b+i] = p
                sm += p
            end
            sm = _attn_warp_reduce(+, sm)
            if lane == 0
                ml[(s-1)*2*nh+h0+g] = mx
                ml[(s-1)*2*nh+nh+h0+g] = sm
            end
        end

        for j0 in 0:_ATTN_SPLIT_W:(d-1)
            W = min(_ATTN_SPLIT_W, d - j0)
            CUDA.sync_threads()
            _attn_tile_load!(sh, to, vc, t0, hk, j0, n, W, tid, Val(C), Val(R), Val(V))
            CUDA.sync_threads()
            for x in tid:nt:(G*_ATTN_SPLIT_W)
                g0 = (x - 1) ÷ _ATTN_SPLIT_W
                jj = x - g0 * _ATTN_SPLIT_W
                jj <= W || continue
                acc = 0.0f0
                for i in 1:n
                    acc += sh[po+g0*C+i] * sh[to+(i-1)*WP+jj]
                end
                op[((s-1)*nh+h0+g0)*d+j0+jj] = acc
            end
        end
    end
    return
end

# The chunk weights exp(m_s - m) are computed once per head into shared
# memory; every output dim then reads its S partials independently.
function _cuda_attn_combine_kernel!(attn, op, ml, S::Int, d::Int, nh::Int)
    sh = CUDA.@cuDynamicSharedMem(Float32, S + 34)
    h = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    nt = Int(blockDim().x)
    nw = nt ÷ 32
    @inbounds begin
        mx = -Inf32
        for s in tid:nt:S
            mx = max(mx, ml[(s-1)*2*nh+h])
        end
        mx = _attn_block_reduce(max, mx, -Inf32, sh, S, tid, nw)
        l = 0.0f0
        for s in tid:nt:S
            w = exp(ml[(s-1)*2*nh+h] - mx)
            sh[s] = w
            l += ml[(s-1)*2*nh+nh+h] * w
        end
        inv = 1.0f0 / _attn_block_reduce(+, l, 0.0f0, sh, S, tid, nw)
        for j in tid:nt:d
            acc = 0.0f0
            for s in 1:S
                acc += op[((s-1)*nh+h-1)*d+j] * sh[s]
            end
            attn[1, h, j] = acc * inv
        end
    end
    return
end

# Chunk size for about _ATTN_SPLIT_BLOCKS blocks over nrows rows.
function _attn_split_chunk(n_kv::Int, nrows::Int)
    want = cld(nrows * n_kv, _ATTN_SPLIT_BLOCKS)
    want <= 32 && return Val(32)
    want <= 64 && return Val(64)
    return Val(128)
end

function _attn_decode_split!(
    attn,
    q,
    kc,
    vc,
    n_kv::Int,
    d::Int,
    K::Int,
    scale::Float32,
    vg::Val,
)
    return _attn_split_launch!(
        attn,
        q,
        kc,
        vc,
        n_kv,
        d,
        0,
        K,
        scale,
        vg,
        _attn_split_chunk(n_kv, K),
    )
end

# The vector loads take 4-row groups that start on a multiple of 4, so they
# are only valid for r0 % 4 == 0; other ranges take the per-element path.
function _attn_split_pass!(
    q,
    kc,
    vc,
    n_kv::Int,
    d::Int,
    r0::Int,
    r1::Int,
    scale::Float32,
    ::Val{G},
    ::Val{C},
) where {G, C}
    S = cld(r1 - r0, C)
    nh = n_kv * G
    sc = _ATTN_SPLIT
    sc.o = _attn_split_buf(sc.o, d * nh * S)
    sc.ml = _attn_split_buf(sc.ml, 2 * nh * S)
    shmem = (G * d + G * C + (_ATTN_SPLIT_W + 1) * C) * sizeof(Float32)
    vv = r0 % 4 == 0 && _attn_vec4(kc) && _attn_vec4(vc)
    @fastcuda threads = _ATTN_THREADS blocks = (n_kv, S) shmem = shmem _cuda_attn_split_kernel!(
        sc.o,
        sc.ml,
        q,
        kc,
        vc,
        r0,
        r1,
        scale,
        d,
        Val(G),
        Val(C),
        Val(vv),
    )
    return S, nh
end

function _attn_split_launch!(
    attn,
    q,
    kc,
    vc,
    n_kv::Int,
    d::Int,
    r0::Int,
    r1::Int,
    scale::Float32,
    ::Val{G},
    ::Val{C},
) where {G, C}
    S, nh = _attn_split_pass!(q, kc, vc, n_kv, d, r0, r1, scale, Val(G), Val(C))
    @fastcuda threads = _ATTN_THREADS blocks = nh shmem = (S + 34) * sizeof(Float32) _cuda_attn_combine_kernel!(
        attn,
        _ATTN_SPLIT.o,
        _ATTN_SPLIT.ml,
        S,
        d,
        nh,
    )
    return true
end

function _attn_partial_launch!(
    pm,
    pl,
    pacc,
    q,
    kc,
    vc,
    n_kv::Int,
    d::Int,
    r0::Int,
    r1::Int,
    scale::Float32,
    ::Val{G},
    ::Val{C},
) where {G, C}
    S, nh = _attn_split_pass!(q, kc, vc, n_kv, d, r0, r1, scale, Val(G), Val(C))
    @fastcuda threads = _ATTN_THREADS blocks = nh shmem = (S + 34) * sizeof(Float32) _cuda_attn_partial_combine_kernel!(
        pm,
        pl,
        pacc,
        _ATTN_SPLIT.o,
        _ATTN_SPLIT.ml,
        S,
        d,
        nh,
    )
    return true
end

# Same merge as _cuda_attn_combine_kernel!, but it keeps the joint max, the
# total sum and the unnormalised output, so a host merge can fold this range
# together with others.
function _cuda_attn_partial_combine_kernel!(pm, pl, pacc, op, ml, S::Int, d::Int, nh::Int)
    sh = CUDA.@cuDynamicSharedMem(Float32, S + 34)
    h = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    nt = Int(blockDim().x)
    nw = nt ÷ 32
    @inbounds begin
        mx = -Inf32
        for s in tid:nt:S
            mx = max(mx, ml[(s-1)*2*nh+h])
        end
        mx = _attn_block_reduce(max, mx, -Inf32, sh, S, tid, nw)
        l = 0.0f0
        for s in tid:nt:S
            w = exp(ml[(s-1)*2*nh+h] - mx)
            sh[s] = w
            l += ml[(s-1)*2*nh+nh+h] * w
        end
        tot = _attn_block_reduce(+, l, 0.0f0, sh, S, tid, nw)
        if tid == 1
            pm[h] = mx
            pl[h] = tot
        end
        for j in tid:nt:d
            acc = 0.0f0
            for s in 1:S
                acc += op[((s-1)*nh+h-1)*d+j] * sh[s]
            end
            pacc[h, j] = acc
        end
    end
    return
end

# Row-range partial for the tiered attention: the same chunk pass as decode,
# over rows first(rows):last(rows), merged to a per-head max, sum and
# unnormalised output. Scores use the same 1/sqrt(d) scale as decode, so the
# result merges by log-sum-exp with a CPU partial (Inference/kv_tier.jl).
function _attention_partial_device!(
    ::CUDABackend,
    pm::CuVector{Float32},
    pl::CuVector{Float32},
    pacc::CuMatrix{Float32},
    q::CuArray,
    kc::_CuKV,
    vc::_CuKV,
    rows::UnitRange{Int},
    n_heads::Int,
    d_head::Int,
    group::Int,
)
    n_kv = div(n_heads, group)
    n_kv * group == n_heads || throw(
        ArgumentError(
            "partial attention: n_heads=$n_heads is not a multiple of group=$group",
        ),
    )
    length(pm) == n_heads && length(pl) == n_heads && size(pacc) == (n_heads, d_head) ||
        throw(
            ArgumentError(
                "partial attention: buffers do not match n_heads=$n_heads, d_head=$d_head",
            ),
        )
    if isempty(rows)
        fill!(pm, -Inf32)
        fill!(pl, 0.0f0)
        fill!(pacc, 0.0f0)
        return true
    end
    first(rows) >= 1 && last(rows) <= size(kc, 1) || throw(
        ArgumentError("partial attention: rows $rows outside the cache's 1:$(size(kc, 1))"),
    )
    group in (1, 2, 4, 8) ||
        throw(ArgumentError("partial attention: group $group is not 1, 2, 4 or 8"))
    r0 = first(rows) - 1
    r1 = last(rows)
    scale = Float32(1 / sqrt(d_head))
    C = _attn_split_chunk(n_kv, length(rows))
    group == 8 && return _attn_partial_launch!(
        pm,
        pl,
        pacc,
        q,
        kc,
        vc,
        n_kv,
        d_head,
        r0,
        r1,
        scale,
        Val(8),
        C,
    )
    group == 4 && return _attn_partial_launch!(
        pm,
        pl,
        pacc,
        q,
        kc,
        vc,
        n_kv,
        d_head,
        r0,
        r1,
        scale,
        Val(4),
        C,
    )
    group == 2 && return _attn_partial_launch!(
        pm,
        pl,
        pacc,
        q,
        kc,
        vc,
        n_kv,
        d_head,
        r0,
        r1,
        scale,
        Val(2),
        C,
    )
    return _attn_partial_launch!(
        pm,
        pl,
        pacc,
        q,
        kc,
        vc,
        n_kv,
        d_head,
        r0,
        r1,
        scale,
        Val(1),
        C,
    )
end

# Copies a device partial into the host KVPartial. A separate path from the
# decode output: three small arrays, no normalisation.
function _partial_download!(
    part::Gesso.Inference.KVPartial,
    pm::CuVector{Float32},
    pl::CuVector{Float32},
    pacc::CuMatrix{Float32},
)
    part.m .= Array(pm)
    part.l .= Array(pl)
    part.acc .= Array(pacc)
    return part
end

# Prefill keys are taken in tiles with an online softmax, so the score scratch
# is (tile, L) whatever K is: 16 MiB, against 2*4*L*K bytes for the reference
# buffers (256 MiB for a 1024-row pass at 32k keys).
const _ATTN_PF_TILE_BYTES = 16 << 20
const _ATTN_PF_TILE_MIN = 256

_attn_prefill_tile(L::Int, K::Int) = min(
    K,
    max(
        _ATTN_PF_TILE_MIN,
        fld(_ATTN_PF_TILE_BYTES ÷ 4, max(L, 1)) ÷ _ATTN_PF_TILE_MIN * _ATTN_PF_TILE_MIN,
    ),
)

# One block per query row over one key tile. St is (tile, L) so a row's scores
# are contiguous. Keys past the row's own position are masked: row i sees key
# u <= K - L + i. Leaves the unnormalised probabilities in St, folds the tile
# into the running max m and sum l, and rescales the row's accumulated output
# by exp(m_old - m_new) before the tile's P*V lands on it. The first tile has
# no accumulator yet: its GEMM writes with beta = 0.
function _cuda_attn_pf_tile_kernel!(
    St,
    m,
    l,
    attn,
    hh::Int,
    g::Int,
    Tn::Int,
    lim0::Int,
    first::Bool,
    d::Int,
)
    sh = CUDA.@cuStaticSharedMem(Float32, 33)
    tid = Int(threadIdx().x)
    nt = Int(blockDim().x)
    nw = nt ÷ 32
    i = Int(blockIdx().x)
    lim = lim0 + i

    lmax = -Inf32
    for j in tid:nt:Tn
        lmax = max(lmax, j <= lim ? St[j, i] : -Inf32)
    end
    tmax = _attn_block_reduce(max, lmax, -Inf32, sh, 0, tid, nw)
    # key 1 is in the first tile and visible to every row, so m is finite
    # from then on and a fully masked later tile leaves it alone
    mo = first ? -Inf32 : m[i, g]
    mn = max(mo, tmax)
    lsum = 0.0f0
    for j in tid:nt:Tn
        p = j <= lim ? exp(St[j, i] - mn) : 0.0f0
        St[j, i] = p
        lsum += p
    end
    tsum = _attn_block_reduce(+, lsum, 0.0f0, sh, 0, tid, nw)
    alpha = exp(mo - mn)
    if !first
        for c in tid:nt:d
            attn[i, hh, c] *= alpha
        end
    end
    if tid == 1
        l[i, hh] = first ? tsum : l[i, hh] * alpha + tsum
        m[i, g] = mn
    end
    return
end

function _cuda_attn_pf_norm_kernel!(attn, l, d::Int)
    i = Int(blockIdx().x)
    h = Int(blockIdx().y)
    inv = 1.0f0 / l[i, h]
    for c in Int(threadIdx().x):Int(blockDim().x):d
        attn[i, h, c] *= inv
    end
    return
end

function _attention_prefill_device!(
    ::CUDABackend,
    attn::CuArray{Float32},
    q::CuArray{Float32},
    k,
    v,
    n_heads::Int,
    d_head::Int,
    group::Int,
    L::Int,
    K::Int,
)
    L == 0 && return true
    T = _attn_prefill_tile(L, K)
    # A cache in another format (f16, quantized) is converted a key tile at a
    # time into dense scratch, once per KV head and shared by its group.
    dense = k isa StridedCuArray{Float32} && v isa StridedCuArray{Float32}
    St = CuArray{Float32}(undef, T, L)
    m = CuArray{Float32}(undef, L, group)
    l = CuArray{Float32}(undef, L, n_heads)
    Kt = dense ? St : CuArray{Float32}(undef, T, d_head)
    Vt = dense ? St : CuArray{Float32}(undef, T, d_head)
    try
        scale = _cu_scalar(Float32(1 / sqrt(d_head)))
        one_, zero_ = _cu_scalar(1.0f0), _cu_scalar(0.0f0)
        for hk in 1:(n_heads÷group), t0 in 0:T:(K-1)
            Tn = min(T, K - t0)
            rows = (t0+1):(t0+Tn)
            kt = dense ? view(k, rows, hk, :) : _kv_load_tile!(Kt, k, t0, hk, Tn)
            vt = dense ? view(v, rows, hk, :) : _kv_load_tile!(Vt, v, t0, hk, Tn)
            S = view(St, 1:Tn, :)
            for g in 1:group
                hh = (hk - 1) * group + g
                _blas_gemm!('N', 'T', scale, kt, view(q, :, hh, :), zero_, S)
                @fastcuda threads = _ATTN_THREADS blocks = L _cuda_attn_pf_tile_kernel!(
                    St,
                    m,
                    l,
                    attn,
                    hh,
                    g,
                    Tn,
                    K - L - t0,
                    t0 == 0,
                    d_head,
                )
                _blas_gemm!(
                    'T',
                    'N',
                    one_,
                    S,
                    vt,
                    t0 == 0 ? zero_ : one_,
                    view(attn, :, hh, :),
                )
            end
        end
        @fastcuda threads = min(_ATTN_THREADS, d_head) blocks = (L, n_heads) _cuda_attn_pf_norm_kernel!(
            attn,
            l,
            d_head,
        )
    finally
        CUDA.unsafe_free!(St)
        CUDA.unsafe_free!(m)
        CUDA.unsafe_free!(l)
        dense || (CUDA.unsafe_free!(Kt); CUDA.unsafe_free!(Vt))
    end
    return true
end
