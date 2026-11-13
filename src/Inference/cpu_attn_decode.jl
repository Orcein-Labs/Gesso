# Float32 attention over dense (ctx, n_kv, d_head) caches with explicit 8-wide
# vectors. Every element keeps the arithmetic of the generic
# `_attention_row_head!`: a score sums its head dims in order from zero, the
# softmax is the same Float64-total pass, and an output sums its keys in order
# from zero, each step a multiply then an add (no fma). Only the grouping of
# independent elements into lanes and threads changes, so the bits match.
#
# Decode runs in three pool passes so a kv head's keys and values are read once
# for all the query heads that share it, instead of once per head:
#   1. scores, one task per (kv head, key range), every head of the group;
#   2. softmax, one task per head;
#   3. outputs, one task per (kv head, dim range), every head of the group.

@inline _v8fmul(a::_V8, b::_V8) = Base.llvmcall(
    """
%r = fmul <8 x float> %0, %1
ret <8 x float> %r""",
    _V8,
    Tuple{_V8, _V8},
    a,
    b,
)

@generated function _v8shuf(a::_V8, b::_V8, ::Val{M}) where {M}
    mask = join(("i32 $m" for m in M), ", ")
    ir = """
%r = shufflevector <8 x float> %0, <8 x float> %1, <8 x i32> <$mask>
ret <8 x float> %r"""
    return :(Base.llvmcall($ir, _V8, Tuple{_V8, _V8}, a, b))
end

# 8x8 transpose: r_k holds 8 keys of dim k, output k holds 8 dims of key k.
@inline function _v8transpose(r1, r2, r3, r4, r5, r6, r7, r8)
    lo = Val((0, 8, 1, 9, 4, 12, 5, 13))
    hi = Val((2, 10, 3, 11, 6, 14, 7, 15))
    t1 = _v8shuf(r1, r2, lo); t2 = _v8shuf(r1, r2, hi)
    t3 = _v8shuf(r3, r4, lo); t4 = _v8shuf(r3, r4, hi)
    t5 = _v8shuf(r5, r6, lo); t6 = _v8shuf(r5, r6, hi)
    t7 = _v8shuf(r7, r8, lo); t8 = _v8shuf(r7, r8, hi)
    a = Val((0, 1, 8, 9, 4, 5, 12, 13))
    b = Val((2, 3, 10, 11, 6, 7, 14, 15))
    s1 = _v8shuf(t1, t3, a); s2 = _v8shuf(t1, t3, b)
    s3 = _v8shuf(t2, t4, a); s4 = _v8shuf(t2, t4, b)
    s5 = _v8shuf(t5, t7, a); s6 = _v8shuf(t5, t7, b)
    s7 = _v8shuf(t6, t8, a); s8 = _v8shuf(t6, t8, b)
    x = Val((0, 1, 2, 3, 8, 9, 10, 11))
    y = Val((4, 5, 6, 7, 12, 13, 14, 15))
    return (
        _v8shuf(s1, s5, x), _v8shuf(s2, s6, x), _v8shuf(s3, s7, x), _v8shuf(s4, s8, x),
        _v8shuf(s1, s5, y), _v8shuf(s2, s6, y), _v8shuf(s3, s7, y), _v8shuf(s4, s8, y),
    )
end

# Eight keys into the 8 dim lanes of acc: values at pv (dim stride sd floats),
# probabilities at pp, keys in order.
@inline function _attn_vstep(acc::_V8, pv::Ptr{Float32}, sd::Int, pp::Ptr{Float32})
    o = _v8transpose(
        _v8load(pv), _v8load(pv + 4sd), _v8load(pv + 8sd), _v8load(pv + 12sd),
        _v8load(pv + 16sd), _v8load(pv + 20sd), _v8load(pv + 24sd), _v8load(pv + 28sd),
    )
    Base.Cartesian.@nexprs 8 k -> acc = _v8muladd(acc, _v8splat(unsafe_load(pp, k)), o[k])
    return acc
end

# The last n < 8 keys, lane by lane, in key order.
@inline function _attn_vtail(acc::_V8, pv::Ptr{Float32}, sd::Int, pp::Ptr{Float32}, n::Int)
    Base.Cartesian.@nexprs 8 i -> a_i = acc[i].value
    for t in 0:(n-1)
        p = unsafe_load(pp, t + 1)
        Base.Cartesian.@nexprs 8 i -> a_i = a_i + p * unsafe_load(pv, t + 1 + (i - 1) * sd)
    end
    return (Base.Cartesian.@ntuple 8 i -> VecElement(a_i))
end

# 8DB output dims starting at pvj: n keys, the first nt of them in 8-key steps.
@generated function _attn_vblock!(pa::Ptr{Float32}, sa::Int, pvj::Ptr{Float32}, sd::Int,
    pp::Ptr{Float32}, n::Int, nt::Int, ::Val{DB}) where {DB}
    acc(b) = Symbol(:acc_, b)
    init = [:($(acc(b)) = _v8splat(0.0f0)) for b in 1:DB]
    step = [:($(acc(b)) = _attn_vstep($(acc(b)), pvj + 4 * (u + $(8(b - 1)) * sd), sd, pp + 4u)) for b in 1:DB]
    tail = [:($(acc(b)) = _attn_vtail($(acc(b)), pvj + 4 * (nt + $(8(b - 1)) * sd), sd, pp + 4nt, n - nt)) for b in 1:DB]
    store = [:(for i in 1:8
        unsafe_store!(pa, $(acc(b))[i].value, 1 + ($(8(b - 1)) + i - 1) * sa)
    end) for b in 1:DB]
    return quote
        $(init...)
        u = 0
        while u < nt
            $(step...)
            u += 8
        end
        nt < n && begin
            $(tail...)
        end
        $(store...)
        return nothing
    end
end

# attn[row, hh, j] for j in j0:j1, over keys lo:K with probs[off+lo:off+K].
function _attn_values_f32!(attn::Array{Float32,3}, row::Int, vc::Array{Float32,3},
    probs::Vector{Float32}, off::Int, hh::Int, hk::Int, j0::Int, j1::Int, lo::Int, K::Int)
    n = K - lo + 1
    nt = n & ~7
    sd = size(vc, 1) * size(vc, 2)
    sa = size(attn, 1) * size(attn, 2)
    GC.@preserve attn vc probs begin
        pv0 = pointer(vc, lo + (hk - 1) * size(vc, 1))
        pp = pointer(probs, off + lo)
        pa0 = pointer(attn, row + (hh - 1) * size(attn, 1))
        j = j0
        while j + 15 <= j1
            _attn_vblock!(pa0 + 4 * (j - 1) * sa, sa, pv0 + 4 * (j - 1) * sd, sd, pp, n, nt, Val(2))
            j += 16
        end
        while j + 7 <= j1
            _attn_vblock!(pa0 + 4 * (j - 1) * sa, sa, pv0 + 4 * (j - 1) * sd, sd, pp, n, nt, Val(1))
            j += 8
        end
        while j <= j1
            a = 0.0f0
            pvj = pv0 + 4 * (j - 1) * sd
            for t in 1:n
                a += unsafe_load(pp, t) * unsafe_load(pvj, t)
            end
            unsafe_store!(pa0, a, 1 + (j - 1) * sa)
            j += 1
        end
    end
    return nothing
end

# probs[off+u] for keys u0:u1 of head hh against kv head hk: from zero, head
# dims in order. A power-of-two scale stays in Float32 (see `_score_add`), so
# eight keys share a lane vector and 64 keys stay in registers across the dims.
function _attn_scores_f32!(probs::Vector{Float32}, q::Array{Float32,3}, row::Int,
    kc::Array{Float32,3}, off::Int, o::_ScoreMul, hh::Int, hk::Int, d_head::Int,
    u0::Int, u1::Int)
    n = u1 - u0 + 1
    sd = size(kc, 1) * size(kc, 2)
    sq = size(q, 1) * size(q, 2)
    r = Float32(o.r)
    rs = _v8splat(r)
    GC.@preserve probs q kc begin
        pk0 = pointer(kc, u0 + (hk - 1) * size(kc, 1))
        pp = pointer(probs, off + u0)
        pq = pointer(q, row + (hh - 1) * size(q, 1))
        u = 0
        while u + 64 <= n
            Base.Cartesian.@nexprs 8 i -> a_i = _v8splat(0.0f0)
            pk = pk0 + 4u
            for j in 0:(d_head-1)
                qs = _v8splat(unsafe_load(pq, 1 + j * sq))
                pkj = pk + 4 * j * sd
                Base.Cartesian.@nexprs 8 i ->
                    a_i = _v8add(a_i, _v8fmul(_v8fmul(qs, _v8load(pkj + 32(i - 1))), rs))
            end
            Base.Cartesian.@nexprs 8 i -> _v8store(pp + 4u + 32(i - 1), a_i)
            u += 64
        end
        while u + 8 <= n
            a = _v8splat(0.0f0)
            pk = pk0 + 4u
            for j in 0:(d_head-1)
                qs = _v8splat(unsafe_load(pq, 1 + j * sq))
                a = _v8add(a, _v8fmul(_v8fmul(qs, _v8load(pk + 4 * j * sd)), rs))
            end
            _v8store(pp + 4u, a)
            u += 8
        end
        while u < n
            a = 0.0f0
            for j in 0:(d_head-1)
                a += (unsafe_load(pq, 1 + j * sq) * unsafe_load(pk0, 1 + u + j * sd)) * r
            end
            unsafe_store!(pp, a, u + 1)
            u += 1
        end
    end
    return nothing
end

# Any other scale: the generic loop over keys u0:u1.
function _attn_scores_f32!(probs::Vector{Float32}, q::Array{Float32,3}, row::Int,
    kc::Array{Float32,3}, off::Int, scale, hh::Int, hk::Int, d_head::Int, u0::Int, u1::Int)
    @inbounds begin
        for u in u0:u1
            probs[off+u] = 0.0f0
        end
        for j in 1:d_head
            qv = q[row, hh, j]
            for u in u0:u1
                probs[off+u] = _score_add(probs[off+u], qv * kc[u, hk, j], scale)
            end
        end
    end
    return nothing
end

# The softmax of `_attention_row_head!`, over probs[off+lo:off+K].
function _attn_softmax_f32!(probs::Vector{Float32}, off::Int, lo::Int, K::Int)
    @inbounds begin
        row_max = probs[off+lo]
        for u in (lo+1):K
            row_max = max(row_max, probs[off+u])
        end
        isfinite(row_max) || throw(
            gesso_error(ERR_NUMERICAL_INSTABILITY, "softmax!: no finite unmasked maximum"),
        )
        total = 0.0
        for u in lo:K
            probs[off+u] = _cpu_exp(probs[off+u] - row_max)
        end
        for u in lo:K
            total += probs[off+u]
        end
        for u in lo:K
            probs[off+u] /= total
        end
    end
    return nothing
end

@inline function _attn_dense_check(attn, q, row, kc, vc, probs, off, hh, hk, d_head, lo, K)
    (1 <= lo <= K <= size(kc, 1) && K <= size(vc, 1) && 0 <= off &&
     off + K <= length(probs) && 1 <= hk <= min(size(kc, 2), size(vc, 2)) &&
     1 <= hh <= min(size(q, 2), size(attn, 2)) && 1 <= row <= min(size(q, 1), size(attn, 1)) &&
     d_head <= min(size(q, 3), size(attn, 3), size(kc, 3), size(vc, 3))) ||
        throw(BoundsError(probs, off + K))
    return nothing
end

# Dense Float32 caches: the same head as the generic method, vectorized.
function _attention_row_head!(attn::Array{Float32,3}, q::Array{Float32,3}, row::Int,
    kc::Array{Float32,3}, vc::Array{Float32,3}, probs::Vector{Float32}, off::Int, scale,
    hh::Int, hk::Int, d_head::Int, lo::Int, K::Int)
    _attn_dense_check(attn, q, row, kc, vc, probs, off, hh, hk, d_head, lo, K)
    _attn_scores_f32!(probs, q, row, kc, off, scale, hh, hk, d_head, lo, K)
    _attn_softmax_f32!(probs, off, lo, K)
    _attn_values_f32!(attn, row, vc, probs, off, hh, hk, 1, d_head, lo, K)
    return nothing
end

# Probs slots of `ctx` rows in the decode scratch: one per head, so the grouped
# passes can run, unless that passes 64 MiB of Float32; then one per thread.
_attn_probs_slots(ctx::Int, n_heads::Int) =
    ctx * n_heads <= (1 << 24) ? max(1, n_heads) : max(1, min(n_heads, Threads.nthreads()))

# Keys per score block: a block's keys stay in L2 while each head of the group
# scores them. The register kernel takes 64 keys at a time.
_attn_key_block(::_ScoreMul) = 64
# Output dims per value block: a block's values stay in L1/L2 across the group.
const _ATTN_GROUP_DIM_BLOCK = 16

struct _AttnGroupPass{A, Q, KC, VC, S}
    attn::A
    q::Q
    kc::KC
    vc::VC
    probs::Vector{Float32}
    scale::S
    n_heads::Int
    d_head::Int
    group::Int
    lo::Int
    K::Int
    phase::Int
    nsplit::Int   # key ranges (phase 1) or dim ranges (phase 3) per kv head
    span::Int     # keys or dims per range
    nc::Int       # tasks
end

function (g::_AttnGroupPass)(c::Int)
    K = g.K
    if g.phase == 2
        for hh in c:g.nc:g.n_heads
            _attn_softmax_f32!(g.probs, (hh - 1) * K, g.lo, K)
        end
        return nothing
    end
    hk, s = divrem(c - 1, g.nsplit)
    hk += 1
    h0 = (hk - 1) * g.group + 1
    h1 = hk * g.group
    if g.phase == 1
        a = g.lo + s * g.span
        b = min(K, a + g.span - 1)
        kb = _attn_key_block(g.scale)
        for ub in a:kb:b
            ue = min(b, ub + kb - 1)
            for hh in h0:h1
                _attn_scores_f32!(g.probs, g.q, 1, g.kc, (hh - 1) * K, g.scale, hh, hk,
                    g.d_head, ub, ue)
            end
        end
    else
        a = 1 + s * g.span
        b = min(g.d_head, a + g.span - 1)
        for jb in a:_ATTN_GROUP_DIM_BLOCK:b
            je = min(b, jb + _ATTN_GROUP_DIM_BLOCK - 1)
            for hh in h0:h1
                _attn_values_f32!(g.attn, 1, g.vc, g.probs, (hh - 1) * K, hh, hk, jb, je,
                    g.lo, K)
            end
        end
    end
    return nothing
end

function _attention_decode_cpu!(
    attn::Array{Float32,3},
    q::Array{Float32,3},
    kc::Array{Float32,3},
    vc::Array{Float32,3},
    probs::Vector{Float32},
    n_heads::Int,
    d_head::Int,
    group::Int,
    K::Int;
    window::Int=0,
)
    lo = window > 0 ? max(1, K - window + 1) : 1
    n_kv = div(n_heads, group)
    # Every head needs its own probs slot. The Float64-corrected scales are
    # compute-bound, so sharing a kv head's reads gains nothing there (measured
    # 0.7-1.0x on Rome). Both cases run head by head.
    (group >= 1 && n_kv * group == n_heads && length(probs) >= n_heads * K &&
     _attention_score_scale(d_head) isa _ScoreMul) ||
        return _attention_decode_cpu_heads!(attn, q, kc, vc, probs, n_heads, d_head, group, K, lo)
    fill!(attn, 0.0f0)
    for hh in 1:n_heads
        _attn_dense_check(attn, q, 1, kc, vc, probs, (hh - 1) * K, hh, cld(hh, group), d_head, lo, K)
    end
    _attn_group_run!(attn, q, kc, vc, probs, _ScoreMul(inv(sqrt(d_head))), n_heads, d_head,
        group, n_kv, lo, K)
    return attn
end

# Behind a function barrier: the scale type is only known at run time.
function _attn_group_run!(attn, q, kc, vc, probs, scale, n_heads::Int, d_head::Int,
    group::Int, n_kv::Int, lo::Int, K::Int)
    nt = Threads.nthreads()
    n = K - lo + 1
    pool = nt > 1 && n * d_head * n_heads >= _ATTN_DECODE_POOL_MIN
    kb = _attn_key_block(scale)
    ns1 = pool ? max(1, min(cld(nt, n_kv), cld(n, kb))) : 1
    span1 = cld(cld(n, ns1), kb) * kb
    ns1 = cld(n, span1)
    ns3 = pool ? max(1, min(cld(nt, n_kv), cld(d_head, _ATTN_GROUP_DIM_BLOCK))) : 1
    span3 = cld(cld(d_head, ns3), 8) * 8
    ns3 = cld(d_head, span3)
    _attn_group_pass!(_AttnGroupPass(attn, q, kc, vc, probs, scale, n_heads, d_head, group, lo,
        K, 1, ns1, span1, n_kv * ns1), pool)
    _attn_group_pass!(_AttnGroupPass(attn, q, kc, vc, probs, scale, n_heads, d_head, group, lo,
        K, 2, 0, 0, pool ? min(n_heads, nt) : 1), pool)
    _attn_group_pass!(_AttnGroupPass(attn, q, kc, vc, probs, scale, n_heads, d_head, group, lo,
        K, 3, ns3, span3, n_kv * ns3), pool)
    return nothing
end

function _attn_group_pass!(f::_AttnGroupPass, pool::Bool)
    if pool && f.nc > 1
        _pool_run_chunks!(f, f.nc)
    else
        for c in 1:f.nc
            f(c)
        end
    end
    return nothing
end
