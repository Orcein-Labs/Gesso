# Exact attention split over row ranges of one KV cache, merged by log-sum-exp.
#
# A range's partial holds, per query head, the running max m of its scores,
# l = sum exp(s - m), and acc = sum exp(s - m) v. Two partials over disjoint
# ranges merge exactly: rescale each side to the joint max and add. Nothing is
# approximated, so the split is the same attention as one pass over all rows,
# up to rounding. This is the arithmetic that lets a host-resident prefix of the
# cache be attended on the CPU while the device attends the newest rows.
#
# Scores follow the reference decode: each term is divided by sqrt(d) as it is
# accumulated, so the 1/sqrt(d) scale lands on the scores, not on q. Cache
# elements go through the existing `getindex` dequant (QuantKV for q8 and q4h),
# so every format reads the values the reference reads.
#
# A q4h cache stores rows in the Hadamard-rotated basis. The caller rotates q
# before the partial and rotates the finished output back, as the decode step
# does; the partial itself stays format-agnostic.

const _KV_TIER_BLOCK_ROWS = 1024

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

# Partial over rows `rows` of K and V caches shaped (rows, n_kv, d). `scores`
# (Float32) and `weights` (Float64) are per-call scratch of length >= length(rows).
function _attention_partial_cpu!(part::KVPartial, q, kc, vc, rows::UnitRange{Int},
    n_heads::Int, d_head::Int, group::Int;
    scores::Vector{Float32}=Vector{Float32}(undef, length(rows)),
    weights::Vector{Float64}=Vector{Float64}(undef, length(rows)))
    isempty(rows) && return _partial_reset!(part)
    n_kv = size(kc, 2)
    n_heads == n_kv * group || throw(gesso_error(ERR_INVALID_PLAN,
        "attention partial: n_heads=$n_heads is not n_kv=$n_kv times group=$group"))
    nr = length(rows)
    r0 = first(rows) - 1
    sqd = sqrt(d_head)
    @inbounds for hk in 1:n_kv, hh in ((hk - 1) * group + 1):(hk * group)
        for i in 1:nr
            scores[i] = 0.0f0
        end
        for j in 1:d_head
            qv = q[1, hh, j]
            for i in 1:nr
                scores[i] += qv * kc[r0 + i, hk, j] / sqd
            end
        end
        mx = scores[1]
        for i in 2:nr
            mx = max(mx, scores[i])
        end
        isfinite(mx) || throw(gesso_error(ERR_NUMERICAL_INSTABILITY,
            "attention partial: no finite score in rows $rows"))
        mx64 = Float64(mx)
        l = 0.0
        for i in 1:nr
            w = exp(Float64(scores[i]) - mx64)
            weights[i] = w
            l += w
        end
        for j in 1:d_head
            a = 0.0
            for i in 1:nr
                a += weights[i] * Float64(vc[r0 + i, hk, j])
            end
            part.acc[hh, j] = a
        end
        part.m[hh] = mx64
        part.l[hh] = l
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
function _partials_threaded!(parts::Vector{KVPartial}, scores::Vector{Vector{Float32}},
    weights::Vector{Vector{Float64}}, q, kc, vc, K::Int, block_rows::Int,
    n_heads::Int, d_head::Int, group::Int)
    Threads.@threads for b in eachindex(parts)
        r0 = (b - 1) * block_rows + 1
        r1 = min(K, b * block_rows)
        _attention_partial_cpu!(parts[b], q, kc, vc, r0:r1, n_heads, d_head, group;
            scores=scores[b], weights=weights[b])
    end
    return parts
end

# Single-query attention over rows 1:K, split into row blocks computed in
# parallel and merged in ascending block order. The output does not depend on
# the thread count: each block's partial is fixed, and the merge order is fixed.
function _attention_split_cpu!(attn, q, kc, vc, K::Int, n_heads::Int, d_head::Int,
    group::Int; block_rows::Int=_KV_TIER_BLOCK_ROWS)
    K >= 1 || throw(gesso_error(ERR_INVALID_PLAN, "attention split: K must be >= 1, got $K"))
    n_blocks = cld(K, block_rows)
    parts = [_kv_partial(n_heads, d_head) for _ in 1:n_blocks]
    scores = [Vector{Float32}(undef, block_rows) for _ in 1:n_blocks]
    weights = [Vector{Float64}(undef, block_rows) for _ in 1:n_blocks]
    _partials_threaded!(parts, scores, weights, q, kc, vc, K, block_rows,
        n_heads, d_head, group)
    total = parts[1]
    for b in 2:n_blocks
        _partial_merge!(total, parts[b])
    end
    return _partial_finish!(attn, total)
end
