# Opt-in sampling for reference_generate. Greedy stays the default: a caller
# that passes no sampler never reaches this file.
#
# The generator is a fixed splitmix64, not Base's default RNG, so one seed gives
# the same ids on every run and every Julia version.

"""
    Sampler(; temperature, top_p=1.0, seed)

Temperature and nucleus (top-p) sampling, one token per logits row. The
temperature must be above 0; greedy decoding is what runs without a sampler.
"""
mutable struct Sampler
    temperature::Float64
    top_p::Float64
    state::UInt64

    function Sampler(temperature::Real, top_p::Real, seed::Integer)
        temperature > 0 ||
            throw(ArgumentError("sampler temperature must be > 0, got $temperature"))
        0 < top_p <= 1 ||
            throw(ArgumentError("sampler top_p must be in (0, 1], got $top_p"))
        seed >= 0 || throw(ArgumentError("sampler seed must be >= 0, got $seed"))
        return new(Float64(temperature), Float64(top_p), UInt64(seed))
    end
end

Sampler(; temperature::Real, top_p::Real=1.0, seed::Integer) =
    Sampler(temperature, top_p, seed)

# One splitmix64 step; a Float64 in [0, 1) from 53 bits.
function _sampler_unit!(s::Sampler)
    s.state += 0x9e3779b97f4a7c15
    z = s.state
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z = z ⊻ (z >> 31)
    return Float64(z >> 11) * 0x1p-53
end

# 0-based id drawn from softmax(logits / temperature), cut to the smallest set of
# most likely ids whose mass reaches top_p. Exactly one draw per call.
function _sample_id(s::Sampler, logits_row)
    logits = _require_finite_logits(Array(logits_row))
    n = length(logits)
    hi = maximum(Float64, logits)
    w = Vector{Float64}(undef, n)
    total = 0.0
    @inbounds for i in 1:n
        w[i] = exp((Float64(logits[i]) - hi) / s.temperature)
        total += w[i]
    end
    if s.top_p >= 1
        r = _sampler_unit!(s) * total
        acc = 0.0
        for i in 1:n
            acc += w[i]
            r < acc && return i - 1
        end
        return n - 1
    end
    order, keep = _nucleus_order(w, s.top_p * total)
    cut = 0.0
    for k in 1:keep
        cut += w[order[k]]
    end
    r = _sampler_unit!(s) * cut
    acc = 0.0
    for k in 1:keep
        acc += w[order[k]]
        r < acc && return order[k] - 1
    end
    return order[keep] - 1
end

# The head of the descending order, long enough to hold the nucleus, and the
# nucleus size. A full sortperm of a 248k vocab costs 25-40 ms per token, and
# the nucleus is usually a few hundred ids. One histogram pass finds the
# smallest band of top weights whose mass reaches the target, and only that
# band is sorted. Positive Float64 bit patterns order like their values, so the
# band is every id above a weight threshold, ties included, and its stable sort
# is the prefix the full stable sortperm(w; rev=true) gave: acc, keep and the
# draw are unchanged.
const _NUCLEUS_BUCKETS = 1 << 15   # exponent and top 5 mantissa bits of a weight in [0, 1]

_nucleus_bucket(x::Float64) = Int(reinterpret(UInt64, x) >> 47) + 1

function _nucleus_order(w::Vector{Float64}, target::Float64)
    n = length(w)
    mass = zeros(Float64, _NUCLEUS_BUCKETS)
    count = zeros(Int, _NUCLEUS_BUCKETS)
    @inbounds for x in w
        b = _nucleus_bucket(x)
        mass[b] += x
        count[b] += 1
    end
    band = _NUCLEUS_BUCKETS
    acc = 0.0
    m = 0
    @inbounds while band > 1
        acc += mass[band]
        m += count[band]
        acc >= target && break
        band -= 1
    end
    band == 1 && (m = n)
    # Gathered into dense vectors: sorting through a view of w cost more than
    # the full sortperm on flat rows where the band is most of the vocab.
    ids = Vector{Int}(undef, m)
    wb = Vector{Float64}(undef, m)
    j = 0
    @inbounds for i in 1:n
        if _nucleus_bucket(w[i]) >= band
            j += 1
            ids[j] = i
            wb[j] = w[i]
        end
    end
    head = ids[sortperm(wb; rev=true)]
    acc = 0.0
    for j in eachindex(head)
        acc += @inbounds w[head[j]]
        acc >= target && return head, j
    end
    length(head) == n && return head, n
    # The histogram sums in another order, so rounding can stop it one band
    # short of the true boundary; the full order is then the only safe answer.
    order = sortperm(w; rev=true)
    acc = 0.0
    for k in 1:n
        acc += @inbounds w[order[k]]
        acc >= target && return order, k
    end
    return order, n
end
