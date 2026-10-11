# Exact attention split over row ranges of one KV cache, merged by log-sum-exp.
#
# A range's partial holds, per query head, the running max m of its scores,
# l = sum exp(s - m), and acc = sum exp(s - m) v. Two partials over disjoint
# ranges merge exactly: rescale each side to the joint max and add. Nothing is
# approximated, so the split is the same attention as one pass over all rows,
# up to rounding. This is the arithmetic that lets a host-resident prefix of the
# cache be attended on the CPU while the device attends the newest rows.
#
# The reference decode divides each product term by sqrt(d) as it is
# accumulated. Here q is scaled once into a Float32 copy, so the division moves
# onto q. That changes rounding by a few ulps per term, which is inside the 1e-5
# split tolerance. The 1/sqrt(d) factor still lands on the scores, not the cache.
#
# All query heads of one KV head are computed together, a row block at a time.
# A block's cache column is small enough to stay in L1 while every head in the
# group sweeps it, so memory sees the cache once per KV head, and the innermost
# loop runs down the contiguous rows, where it vectorizes. A q8 cache takes a fast path that
# reads codes and scales directly (value = code * scale, one scale per 32
# values); the other formats go through getindex, so they read what the
# reference reads.
#
# Inside a row block the scores, weights and weighted sum are Float32, which is
# safe at _KV_TIER_BLOCK_ROWS rows. Each block's m, l and acc are widened to
# Float64 for the cross-block merge.
#
# int_scores scores q8 keys with Int16 query codes and Int32 accumulation, one
# scale product per 32-value block. Int16 is needed: Int8 query codes miss the
# 5e-3 gate at unit scale. It is off by default: the rows-innermost layout keeps
# LLVM from emitting vpmaddwd, and on this Haswell the integer path's median was
# slower than float in all four timed cells.
#
# A q4h cache stores rows in the Hadamard-rotated basis. The caller rotates q
# before the partial and rotates the finished output back, as the decode step
# does; the partial itself stays format-agnostic.

const _KV_TIER_BLOCK_ROWS = 1024
const _KV_TIER_INT_SCORES = false

struct KVPartial
    m::Vector{Float64}
    l::Vector{Float64}
    acc::Matrix{Float64}   # (n_heads, d_head)
end

# The identity element: m = -Inf and l = acc = 0 make a merge a no-op.
function _kv_partial(n_heads::Int, d_head::Int)
    return KVPartial(fill(-Inf, n_heads), zeros(n_heads), zeros(n_heads, d_head))
end

function _partial_reset!(part::KVPartial)
    fill!(part.m, -Inf)
    fill!(part.l, 0.0)
    fill!(part.acc, 0.0)
    return part
end

# Scratch for one row block. Blocks index their own scratch by block number, so
# concurrent blocks never share a buffer.
struct KVTierBlock
    qs::Matrix{Float32}    # (n_heads, d_head): q / sqrt(d_head)
    s::Matrix{Float32}     # (block_rows, group): scores, then softmax weights
    t::Matrix{Float32}     # (block_rows, group): per-32-block temp, see the q8 methods
    acc::Matrix{Float32}   # (group, d_head)
    mx::Vector{Float32}    # (group)
    l::Vector{Float32}     # (group)
    qi::Matrix{Int16}      # (n_heads, d_head): qs as codes in [-32767, 32767], see _quantize_query!
    qsc::Matrix{Float32}   # (n_heads, d_head ÷ 32): scale of each qi block
    ti::Matrix{Int32}      # (block_rows, group): per-32-block integer sums
end

function _kv_tier_block(block_rows::Int, n_heads::Int, d_head::Int, group::Int)
    return KVTierBlock(zeros(Float32, n_heads, d_head), zeros(Float32, block_rows, group),
        zeros(Float32, block_rows, group), zeros(Float32, group, d_head),
        zeros(Float32, group), zeros(Float32, group),
        zeros(Int16, n_heads, d_head), zeros(Float32, n_heads, d_head ÷ _KV_BLOCK),
        zeros(Int32, block_rows, group))
end

# Preallocated partials and block scratch for a decode loop. Build it once for
# the largest K it will see and pass it to _attention_split_cpu!.
struct KVTierWorkspace
    parts::Vector{KVPartial}
    blocks::Vector{KVTierBlock}
    max_rows::Int
    n_heads::Int
    d_head::Int
    group::Int
    block_rows::Int
end

function KVTierWorkspace(max_rows::Int, n_heads::Int, d_head::Int, group::Int;
    block_rows::Int=_KV_TIER_BLOCK_ROWS)
    max_rows >= 1 || throw(gesso_error(ERR_INVALID_PLAN,
        "kv tier workspace: max_rows must be >= 1, got $max_rows"))
    group >= 1 && n_heads % group == 0 || throw(gesso_error(ERR_INVALID_PLAN,
        "kv tier workspace: n_heads=$n_heads is not a multiple of group=$group"))
    n_blocks = cld(max_rows, block_rows)
    parts = [_kv_partial(n_heads, d_head) for _ in 1:n_blocks]
    blocks = [_kv_tier_block(block_rows, n_heads, d_head, group) for _ in 1:n_blocks]
    return KVTierWorkspace(parts, blocks, max_rows, n_heads, d_head, group, block_rows)
end

function _check_workspace(ws::KVTierWorkspace, K::Int, n_heads::Int, d_head::Int,
    group::Int, block_rows::Int)
    ok = K <= ws.max_rows && ws.n_heads == n_heads && ws.d_head == d_head &&
         ws.group == group && ws.block_rows == block_rows
    ok || throw(gesso_error(ERR_INVALID_PLAN, "attention split: workspace (max_rows=$(ws.max_rows), " *
        "n_heads=$(ws.n_heads), d_head=$(ws.d_head), group=$(ws.group), " *
        "block_rows=$(ws.block_rows)) does not fit K=$K, n_heads=$n_heads, " *
        "d_head=$d_head, group=$group, block_rows=$block_rows"))
    return ws
end

function _scale_query!(qs::Matrix{Float32}, q, d_head::Int)
    inv_sqd = Float32(1 / sqrt(d_head))
    @inbounds for j in 1:d_head, hh in axes(qs, 1)
        qs[hh, j] = Float32(q[1, hh, j]) * inv_sqd
    end
    return qs
end

# The query as Int16 codes, one scale per query head and 32-value block. Eight
# bits were tried first and left the score error near 1e-2 relative: that error
# is multiplied by the score, so large-score rows lose the most.
function _quantize_query!(blk::KVTierBlock, d_head::Int)
    qs = blk.qs
    @inbounds for b in 1:(d_head ÷ _KV_BLOCK), hh in axes(qs, 1)
        js = ((b - 1) * _KV_BLOCK + 1):(b * _KV_BLOCK)
        sc = maximum(abs, view(qs, hh, js)) / 32767.0f0
        inv = sc == 0 ? 0.0f0 : 1.0f0 / sc
        blk.qsc[hh, b] = sc
        for j in js
            blk.qi[hh, j] = unsafe_trunc(Int16, clamp(round(qs[hh, j] * inv), -32767.0f0, 32767.0f0))
        end
    end
    return blk
end

# Integer scores for a q8 cache: the per-block sums of code products are exact in
# Int32, and both scales multiply the block sum once.
function _score_block_int!(blk::KVTierBlock, kc::QuantKV{:q8}, hk::Int, r0::Int,
    nr::Int, d_head::Int, group::Int)
    s = blk.s
    ti = blk.ti
    qi = blk.qi
    codes = kc.codes
    scales = kc.scales
    hh0 = (hk - 1) * group
    @inbounds for g in 1:group, i in 1:nr
        s[i, g] = 0.0f0
    end
    @inbounds for b in 1:(d_head ÷ _KV_BLOCK)
        for g in 1:group, i in 1:nr
            ti[i, g] = 0
        end
        for j in ((b - 1) * _KV_BLOCK + 1):(b * _KV_BLOCK), g in 1:group
            qv = qi[hh0 + g, j]
            @simd for i in 1:nr
                ti[i, g] += Int32(qv) * Int32(codes[r0 + i, hk, j])
            end
        end
        for g in 1:group
            qsc = blk.qsc[hh0 + g, b]
            @simd for i in 1:nr
                s[i, g] += Float32(ti[i, g]) * qsc * Float32(scales[r0 + i, hk, b])
            end
        end
    end
    return s
end

# Scores of one KV head's query group over rows r0+1:r0+nr.
function _score_block!(blk::KVTierBlock, kc::AbstractArray, hk::Int, r0::Int,
    nr::Int, d_head::Int, group::Int)
    s = blk.s
    qs = blk.qs
    hh0 = (hk - 1) * group
    @inbounds for g in 1:group, i in 1:nr
        s[i, g] = 0.0f0
    end
    @inbounds for j in 1:d_head, g in 1:group
        qv = qs[hh0 + g, j]
        @simd for i in 1:nr
            s[i, g] += qv * Float32(kc[r0 + i, hk, j])
        end
    end
    return s
end

# Per 32-value block the codes are summed against q in Float32, and the row's
# scale multiplies that sum once, so the scale is read once per row per block
# and never per element.
function _score_block!(blk::KVTierBlock, kc::QuantKV{:q8}, hk::Int, r0::Int,
    nr::Int, d_head::Int, group::Int)
    s = blk.s
    t = blk.t
    qs = blk.qs
    codes = kc.codes
    scales = kc.scales
    hh0 = (hk - 1) * group
    @inbounds for g in 1:group, i in 1:nr
        s[i, g] = 0.0f0
    end
    @inbounds for b in 1:(d_head ÷ _KV_BLOCK)
        for g in 1:group, i in 1:nr
            t[i, g] = 0.0f0
        end
        for j in ((b - 1) * _KV_BLOCK + 1):(b * _KV_BLOCK), g in 1:group
            qv = qs[hh0 + g, j]
            @simd for i in 1:nr
                t[i, g] += qv * Float32(codes[r0 + i, hk, j])
            end
        end
        for g in 1:group, i in 1:nr
            s[i, g] += t[i, g] * Float32(scales[r0 + i, hk, b])
        end
    end
    return s
end

# Per row block: m is the block max, weights replace the scores in place, and
# l is their sum. A block with no finite score cannot be normalised.
function _softmax_block!(blk::KVTierBlock, nr::Int, group::Int)
    s = blk.s
    @inbounds for g in 1:group
        mx = s[1, g]
        for i in 2:nr
            mx = max(mx, s[i, g])
        end
        isfinite(mx) || throw(gesso_error(ERR_NUMERICAL_INSTABILITY,
            "attention partial: no finite score in this row block"))
        l = 0.0f0
        for i in 1:nr
            w = exp(s[i, g] - mx)
            s[i, g] = w
            l += w
        end
        blk.mx[g] = mx
        blk.l[g] = l
    end
    return blk
end

function _value_block!(blk::KVTierBlock, vc::AbstractArray, hk::Int, r0::Int,
    nr::Int, d_head::Int, group::Int)
    s = blk.s
    acc = blk.acc
    @inbounds for j in 1:d_head, g in 1:group
        acc[g, j] = 0.0f0
    end
    @inbounds for j in 1:d_head, g in 1:group
        a = 0.0f0
        @simd for i in 1:nr
            a += s[i, g] * Float32(vc[r0 + i, hk, j])
        end
        acc[g, j] = a
    end
    return acc
end

# For each 32-value block the weights are multiplied by that block's scale of
# each row once, into t, so the code loop is a plain dot product down the rows.
function _value_block!(blk::KVTierBlock, vc::QuantKV{:q8}, hk::Int, r0::Int,
    nr::Int, d_head::Int, group::Int)
    s = blk.s
    t = blk.t
    acc = blk.acc
    codes = vc.codes
    scales = vc.scales
    @inbounds for j in 1:d_head, g in 1:group
        acc[g, j] = 0.0f0
    end
    @inbounds for b in 1:(d_head ÷ _KV_BLOCK)
        for g in 1:group, i in 1:nr
            t[i, g] = s[i, g] * Float32(scales[r0 + i, hk, b])
        end
        for j in ((b - 1) * _KV_BLOCK + 1):(b * _KV_BLOCK), g in 1:group
            a = 0.0f0
            @simd for i in 1:nr
                a += t[i, g] * Float32(codes[r0 + i, hk, j])
            end
            acc[g, j] += a
        end
    end
    return acc
end

# Partial over rows `rows` of K and V caches shaped (rows, n_kv, d). Passing a
# block scratch avoids allocation; otherwise one is made for this range.
function _attention_partial_cpu!(part::KVPartial, q, kc, vc, rows::UnitRange{Int},
    n_heads::Int, d_head::Int, group::Int; blk::Union{KVTierBlock,Nothing}=nothing,
    int_scores::Bool=_KV_TIER_INT_SCORES)
    isempty(rows) && return _partial_reset!(part)
    n_kv = size(kc, 2)
    n_heads == n_kv * group || throw(gesso_error(ERR_INVALID_PLAN,
        "attention partial: n_heads=$n_heads is not n_kv=$n_kv times group=$group"))
    nr = length(rows)
    r0 = first(rows) - 1
    scratch = blk === nothing ? _kv_tier_block(nr, n_heads, d_head, group) : blk
    size(scratch.s, 1) >= nr || throw(gesso_error(ERR_INVALID_PLAN,
        "attention partial: block scratch holds $(size(scratch.s, 1)) rows, need $nr"))
    _scale_query!(scratch.qs, q, d_head)
    use_int = int_scores && kc isa QuantKV{:q8}
    use_int && _quantize_query!(scratch, d_head)
    for hk in 1:n_kv
        if use_int
            _score_block_int!(scratch, kc, hk, r0, nr, d_head, group)
        else
            _score_block!(scratch, kc, hk, r0, nr, d_head, group)
        end
        _softmax_block!(scratch, nr, group)
        _value_block!(scratch, vc, hk, r0, nr, d_head, group)
        hh0 = (hk - 1) * group
        for g in 1:group
            hh = hh0 + g
            part.m[hh] = Float64(scratch.mx[g])
            part.l[hh] = Float64(scratch.l[g])
            for j in 1:d_head
                part.acc[hh, j] = Float64(scratch.acc[g, j])
            end
        end
    end
    return part
end

# dst <- dst (+) src. An empty side contributes weight zero, so -Inf maxima
# never reach exp(-Inf - -Inf).
function _partial_merge!(dst::KVPartial, src::KVPartial)
    for hh in eachindex(dst.m)
        m1 = dst.m[hh]
        m2 = src.m[hh]
        m = max(m1, m2)
        w1 = m1 == -Inf ? 0.0 : exp(m1 - m)
        w2 = m2 == -Inf ? 0.0 : exp(m2 - m)
        dst.l[hh] = w1 * dst.l[hh] + w2 * src.l[hh]
        for j in axes(dst.acc, 2)
            dst.acc[hh, j] = w1 * dst.acc[hh, j] + w2 * src.acc[hh, j]
        end
        dst.m[hh] = m
    end
    return dst
end

# attn (1, n_heads, d) <- acc / l. An all-empty partial has no mass to normalize.
function _partial_finish!(attn, part::KVPartial)
    for hh in eachindex(part.m)
        part.l[hh] > 0 || throw(gesso_error(ERR_INVALID_PLAN,
            "attention finish: no rows were attended"))
        for j in axes(part.acc, 2)
            attn[1, hh, j] = Float32(part.acc[hh, j] / part.l[hh])
        end
    end
    return attn
end

# Block partials are computed in parallel into slots indexed by block, never by
# threadid, so the scratch each block uses does not depend on the scheduler.
function _partials_threaded!(parts::Vector{KVPartial}, blocks::Vector{KVTierBlock},
    n_blocks::Int, q, kc, vc, K::Int, block_rows::Int, n_heads::Int, d_head::Int, group::Int,
    int_scores::Bool)
    Threads.@threads for b in 1:n_blocks
        r0 = (b - 1) * block_rows + 1
        r1 = min(K, b * block_rows)
        _attention_partial_cpu!(parts[b], q, kc, vc, r0:r1, n_heads, d_head, group;
            blk=blocks[b], int_scores)
    end
    return parts
end

# Single-query attention over rows 1:K, split into row blocks computed in
# parallel and merged in ascending block order. The output does not depend on
# the thread count: each block's partial is fixed, and the merge order is fixed.
# Pass a KVTierWorkspace to reuse its buffers across calls.
function _attention_split_cpu!(attn, q, kc, vc, K::Int, n_heads::Int, d_head::Int,
    group::Int; block_rows::Int=_KV_TIER_BLOCK_ROWS, ws::Union{KVTierWorkspace,Nothing}=nothing,
    int_scores::Bool=_KV_TIER_INT_SCORES)
    K >= 1 || throw(gesso_error(ERR_INVALID_PLAN, "attention split: K must be >= 1, got $K"))
    ws === nothing && (ws = KVTierWorkspace(K, n_heads, d_head, group; block_rows))
    _check_workspace(ws, K, n_heads, d_head, group, block_rows)
    n_blocks = cld(K, block_rows)
    _partials_threaded!(ws.parts, ws.blocks, n_blocks, q, kc, vc, K, block_rows,
        n_heads, d_head, group, int_scores)
    total = ws.parts[1]
    for b in 2:n_blocks
        _partial_merge!(total, ws.parts[b])
    end
    return _partial_finish!(attn, total)
end
