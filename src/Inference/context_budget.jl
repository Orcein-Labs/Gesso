# Context budget and compaction policy: how long a conversation may grow, and
# when a harness should compact it. Pure Julia, no device. The inputs are a
# model's KV geometry, a memory figure and measured per-token costs.
#
# Three ceilings bound a context, and the smallest is the safe length:
#   memory   the KV cache that fits beside the weights and scratch
#   quality  a length the caller judges the model should not exceed
#   latency  decode stays fast enough, and a cold re-prefill stays short
#
# Compaction is a ski-rental choice. A long context pays a decode slowdown on
# every turn (rent). Compacting pays once: a summary, then a cold prefill of
# the shorter prompt (buy). Renting until the rent paid equals the price is
# 2-competitive against an offline choice that knew the future turns.

"""
    kv_bytes_per_token(spec::ArchitectureSpec, kv_format::Symbol) -> Int

Bytes of KV cache one token costs, summed over the layers that hold KV. A
linear attention (GDN) layer keeps a fixed-size recurrent state instead, so a
hybrid family counts only its full-attention layers. Each layer contributes its
own KV head count and head dimension (`layer_n_kv_heads`, `layer_head_dim`),
as the cache allocator sizes it, so Gemma 4 sliding and global layers and
Spark's per-layer geometry are counted exactly. Quantized formats need a head
dimension that fills whole scale blocks, as the cache allocator requires.
"""
function kv_bytes_per_token(spec::ArchitectureSpec, kv_format::Symbol)
    kv_format in KV_FORMATS || throw(
        gesso_error(ERR_INVALID_PLAN, "kv_bytes_per_token: unknown KV format $kv_format"),
    )
    total = 0
    for l in 0:(spec.num_layers-1)
        layer_mixer(spec, l) === :full_attention || continue
        d = layer_head_dim(spec, l)
        _check_kv_head_dim(d, kv_format)
        total += 2 * layer_n_kv_heads(spec, l) * _kv_head_bytes(d, kv_format)  # K and V
    end
    return total
end

function _check_kv_head_dim(d::Integer, kv_format::Symbol)
    if kv_format in (:q8, :q4h)
        d % _KV_BLOCK == 0 || throw(
            gesso_error(
                ERR_INVALID_PLAN,
                "kv_bytes_per_token: $kv_format needs head_dim divisible by $_KV_BLOCK, got $d",
            ),
        )
    end
    kv_format === :q4h &&
        !ispow2(d) &&
        throw(
            gesso_error(
                ERR_INVALID_PLAN,
                "kv_bytes_per_token: q4h needs a power-of-two head_dim, got $d",
            ),
        )
    return nothing
end

# One head's row of K (or V) in a cache of this format.
function _kv_head_bytes(d::Integer, kv_format::Symbol)
    kv_format === :f32 && return d * sizeof(Float32)
    kv_format === :f16 && return d * sizeof(Float16)
    kv_format === :q8 && return d * sizeof(Int8) + (d ÷ _KV_BLOCK) * sizeof(Float16)
    return (d ÷ 2) * sizeof(UInt8) + (d ÷ _KV_BLOCK) * sizeof(Float16)
end

"""
    memory_ceiling_tokens(; budget_bytes, resident_bytes, scratch_bytes=0,
                            kv_bytes_per_token) -> Int

Largest context whose KV cache fits in what the weights and scratch leave of
`budget_bytes`. Zero when nothing fits; never negative.
"""
function memory_ceiling_tokens(;
    budget_bytes::Integer,
    resident_bytes::Integer,
    scratch_bytes::Integer=0,
    kv_bytes_per_token::Integer,
)
    budget_bytes >= 0 && resident_bytes >= 0 && scratch_bytes >= 0 ||
        throw(ArgumentError("memory_ceiling_tokens: byte counts must be nonnegative"))
    kv_bytes_per_token > 0 ||
        throw(ArgumentError("memory_ceiling_tokens: kv_bytes_per_token must be positive"))
    free = budget_bytes - resident_bytes - scratch_bytes
    return free <= 0 ? 0 : fld(free, kv_bytes_per_token)
end

# Piecewise-linear between samples and flat outside them. Past the last sample
# the last measurement is the best estimate; below the first there is no data,
# so the first sample stands in for it.
function _interp(xs::Vector{Float64}, ys::Vector{Float64}, x::Real)
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]
    i = searchsortedlast(xs, x)
    t = (x - xs[i]) / (xs[i+1] - xs[i])
    return ys[i] + t * (ys[i+1] - ys[i])
end

# Integral of _interp over [a, b]: the cost of tokens a+1..b, each paying the
# rate of the length it lands at. Exact for the piecewise-linear model.
function _integrate(xs::Vector{Float64}, ys::Vector{Float64}, a::Float64, b::Float64)
    total = 0.0
    lo = a
    for x in xs
        x <= lo && continue
        x >= b && break
        total += (x - lo) * (_interp(xs, ys, lo) + _interp(xs, ys, x)) / 2
        lo = x
    end
    return total + (b - lo) * (_interp(xs, ys, lo) + _interp(xs, ys, b)) / 2
end

function _cost_samples(name::AbstractString, samples)
    isempty(samples) && throw(ArgumentError("CostModel: $name needs at least one sample"))
    lens = Float64[]
    secs = Float64[]
    for (n, s) in samples
        n >= 0 || throw(ArgumentError("CostModel: $name length $n is negative"))
        isfinite(s) && s > 0 ||
            throw(ArgumentError("CostModel: $name seconds $s must be finite and positive"))
        !isempty(lens) &&
            n <= lens[end] &&
            throw(
                ArgumentError(
                    "CostModel: $name lengths must strictly increase (got $n after $(lens[end]))",
                ),
            )
        push!(lens, n)
        push!(secs, s)
    end
    return lens, secs
end

"""
    CostModel(; prefill, decode)

Measured seconds per token as a function of context length. Each argument is a
vector of `context_length => seconds_per_token` pairs: lengths nonnegative and
strictly increasing, seconds finite and positive. A cold prefill starts at
length 0, so the length-0 prefill sample is the cold rate.
"""
struct CostModel
    prefill_len::Vector{Float64}
    prefill_s::Vector{Float64}
    decode_len::Vector{Float64}
    decode_s::Vector{Float64}
end

function CostModel(; prefill, decode)
    pl, ps = _cost_samples("prefill", prefill)
    dl, ds = _cost_samples("decode", decode)
    return CostModel(pl, ps, dl, ds)
end

function _check_span(n_tokens::Real, at_len::Real)
    n_tokens >= 0 || throw(ArgumentError("token count $n_tokens is negative"))
    at_len >= 0 || throw(ArgumentError("context length $at_len is negative"))
    return nothing
end

"""
    prefill_seconds(cm, n_tokens, at_len) -> Float64

Seconds to prefill `n_tokens` appended at context length `at_len`. The rate
is integrated over the span, so a long cold prefill pays for the lengths it
grows through, not the length-0 rate throughout.
"""
function prefill_seconds(cm::CostModel, n_tokens::Real, at_len::Real)
    _check_span(n_tokens, at_len)
    return _integrate(
        cm.prefill_len,
        cm.prefill_s,
        Float64(at_len),
        Float64(at_len) + n_tokens,
    )
end

"""
    decode_seconds(cm, n_tokens, at_len) -> Float64

Seconds to decode `n_tokens` starting at context length `at_len`, integrated
over the span as `prefill_seconds` is.
"""
function decode_seconds(cm::CostModel, n_tokens::Real, at_len::Real)
    _check_span(n_tokens, at_len)
    return _integrate(
        cm.decode_len,
        cm.decode_s,
        Float64(at_len),
        Float64(at_len) + n_tokens,
    )
end

"""
    ContextBudget(; memory_tokens, quality_tokens, latency_tokens)

The three ceilings on context length, each an `Int` with `typemax(Int)` meaning
unconstrained. `safe_tokens` is their minimum; `binding` names the ceiling that
sets it (`:memory`, `:quality` or `:latency`), or `:none` when all three are
unconstrained. A tie goes to memory, then quality.
"""
struct ContextBudget
    memory_tokens::Int
    quality_tokens::Int
    latency_tokens::Int
    safe_tokens::Int
    binding::Symbol
end

function ContextBudget(;
    memory_tokens::Integer=typemax(Int),
    quality_tokens::Integer=typemax(Int),
    latency_tokens::Integer=typemax(Int),
)
    m, q, l = Int(memory_tokens), Int(quality_tokens), Int(latency_tokens)
    m >= 0 && q >= 0 && l >= 0 ||
        throw(ArgumentError("ContextBudget: ceilings must be nonnegative"))
    safe = min(m, q, l)
    binding =
        safe == typemax(Int) ? :none : m == safe ? :memory : q == safe ? :quality : :latency
    return ContextBudget(m, q, l, safe, binding)
end

"""
    latency_ceiling_tokens(cm; min_decode_tok_s=nothing, max_reprefill_s=nothing,
                           max_len) -> Int

Largest length `L <= max_len` at which decode runs at least `min_decode_tok_s`
tokens per second and a cold prefill of `L` tokens takes at most
`max_reprefill_s`. A `nothing` constraint is ignored. Zero when no positive
length qualifies.

The answer is found from the sampled lengths, the crossings of the decode
constraint between samples, and the prefill bound, so the scan never invents
cost data between or beyond the samples.
"""
function latency_ceiling_tokens(
    cm::CostModel;
    min_decode_tok_s=nothing,
    max_reprefill_s=nothing,
    max_len::Integer,
)
    max_len >= 0 ||
        throw(ArgumentError("latency_ceiling_tokens: max_len must be nonnegative"))
    max_dec_s = nothing
    if min_decode_tok_s !== nothing
        min_decode_tok_s > 0 || throw(
            ArgumentError("latency_ceiling_tokens: min_decode_tok_s must be positive"),
        )
        max_dec_s = 1 / min_decode_tok_s
    end
    if max_reprefill_s !== nothing
        isfinite(max_reprefill_s) && max_reprefill_s >= 0 || throw(
            ArgumentError(
                "latency_ceiling_tokens: max_reprefill_s must be finite and nonnegative",
            ),
        )
    end
    feasible(L) =
        (max_dec_s === nothing || _interp(cm.decode_len, cm.decode_s, L) <= max_dec_s) &&
        (max_reprefill_s === nothing || prefill_seconds(cm, L, 0) <= max_reprefill_s)

    cands = Int[0, max_len]
    for x in cm.decode_len
        x <= max_len && push!(cands, Int(x))
    end
    for x in cm.prefill_len
        x <= max_len && push!(cands, Int(x))
    end
    if max_dec_s !== nothing
        xs, ys = cm.decode_len, cm.decode_s
        for i in 1:(length(xs)-1)
            ys[i] == ys[i+1] && continue
            (ys[i] - max_dec_s) * (ys[i+1] - max_dec_s) <= 0 || continue
            x = xs[i] + (max_dec_s - ys[i]) * (xs[i+1] - xs[i]) / (ys[i+1] - ys[i])
            push!(cands, floor(Int, x), ceil(Int, x))
        end
    end
    if max_reprefill_s !== nothing
        # Cold prefill time grows strictly with length: bisect for the largest fit.
        lo, hi = 0, max_len
        while lo < hi
            mid = hi - (hi - lo) ÷ 2
            prefill_seconds(cm, mid, 0) <= max_reprefill_s ? (lo = mid) : (hi = mid - 1)
        end
        push!(cands, lo)
    end

    best = 0
    for c in cands
        0 <= c <= max_len && c > best && feasible(c) && (best = c)
    end
    return best
end

"""
    CompactionPolicy(; rule=:rent_or_buy, reserve_tokens=4096, target_fraction=0.5,
                       threshold_fraction=0.9, summary_tokens=1024)

How a harness decides to compact. `rule` is `:rent_or_buy` (the default),
`:threshold` or `:never`. `reserve_tokens` is room kept for the next turn's
input and reply. A compaction lands at `target_fraction` of the safe length and
the summary is expected to run `summary_tokens` long. The target must sit below
the threshold, or a compacted context would have no room to grow before it
compacts again.
"""
struct CompactionPolicy
    rule::Symbol
    reserve_tokens::Int
    target_fraction::Float64
    threshold_fraction::Float64
    summary_tokens::Int

    function CompactionPolicy(;
        rule::Symbol=:rent_or_buy,
        reserve_tokens::Integer=4096,
        target_fraction::Real=0.5,
        threshold_fraction::Real=0.9,
        summary_tokens::Integer=1024,
    )
        rule in (:rent_or_buy, :threshold, :never) ||
            throw(ArgumentError("CompactionPolicy: unknown rule $(repr(rule))"))
        reserve_tokens >= 0 ||
            throw(ArgumentError("CompactionPolicy: reserve_tokens must be nonnegative"))
        summary_tokens >= 1 ||
            throw(ArgumentError("CompactionPolicy: summary_tokens must be positive"))
        isfinite(target_fraction) && isfinite(threshold_fraction) ||
            throw(ArgumentError("CompactionPolicy: fractions must be finite"))
        0 < target_fraction < threshold_fraction <= 1 || throw(
            ArgumentError(
                "CompactionPolicy: need 0 < target_fraction ($target_fraction) < " *
                "threshold_fraction ($threshold_fraction) <= 1",
            ),
        )
        return new(
            rule,
            Int(reserve_tokens),
            Float64(target_fraction),
            Float64(threshold_fraction),
            Int(summary_tokens),
        )
    end
end

"""
    CompactionTracker()

Rent paid since the last compaction. Mutable: `decide_compaction` advances it
on every call, and the caller clears it with `reset!(tracker, new_len)` after
compacting.
"""
mutable struct CompactionTracker
    accumulated_slowdown_s::Float64
    turns_since_compaction::Int
    last_compacted_len::Int
end

CompactionTracker() = CompactionTracker(0.0, 0, 0)

"""
    reset!(tracker, new_len)

Clear the rent paid and record the length the context was compacted to. Call
it after every compaction the caller performs.
"""
function reset!(t::CompactionTracker, new_len::Integer)
    new_len >= 0 || throw(ArgumentError("reset!: new_len must be nonnegative"))
    t.accumulated_slowdown_s = 0.0
    t.turns_since_compaction = 0
    t.last_compacted_len = Int(new_len)
    return t
end

_tokens_of(fraction::Float64, n::Int) =
    n == typemax(Int) ? typemax(Int) : floor(Int, fraction * n)

"""
    decide_compaction(policy, budget, cm, tracker, ctx_len;
                      next_turn_tokens=policy.reserve_tokens,
                      new_tokens_per_turn=256)
        -> (compact, reason, target_tokens, compaction_cost_s, accumulated_slowdown_s)

Decide whether to compact a context of `ctx_len` tokens. Call it once per turn:
each call advances `tracker`.

- `reason = :hard_ceiling` fires when `ctx_len + next_turn_tokens` exceeds
  `budget.safe_tokens`, under every rule including `:never`. Overflow fails the
  request outright, so compacting is the only move left. `:never` means never
  compact by choice, not never compact.
- `:threshold` compacts once `ctx_len` reaches `threshold_fraction` of the safe
  length. Otherwise `reason = :wait`.
- `:rent_or_buy` pays rent each turn: the decode slowdown from carrying
  `ctx_len` instead of `target_tokens`, over `new_tokens_per_turn`. It compacts
  when that rent reaches the compaction cost, which is writing the summary at
  the current length plus a cold prefill of the compacted prompt, since KV reuse
  is lost on compaction. This is the ski-rental 2-competitive rule. Otherwise
  `reason = :wait`.
- `reason = :below_target` means the context is already at or under the target,
  so compaction would not shrink it.
- `reason = :cannot_fit` means the context is at or under the target and the
  next turn still overflows. Compaction cannot help, so the caller must shrink
  the reply budget.

When `compact` is true, `target_tokens < ctx_len` always holds.
"""
function decide_compaction(
    policy::CompactionPolicy,
    budget::ContextBudget,
    cm::CostModel,
    tracker::CompactionTracker,
    ctx_len::Integer;
    next_turn_tokens::Integer=policy.reserve_tokens,
    new_tokens_per_turn::Integer=256,
)
    ctx_len >= 0 || throw(ArgumentError("decide_compaction: ctx_len must be nonnegative"))
    next_turn_tokens >= 0 ||
        throw(ArgumentError("decide_compaction: next_turn_tokens must be nonnegative"))
    new_tokens_per_turn >= 0 ||
        throw(ArgumentError("decide_compaction: new_tokens_per_turn must be nonnegative"))
    ctx = Int(ctx_len)
    safe = budget.safe_tokens
    target = _tokens_of(policy.target_fraction, safe)
    cost =
        decode_seconds(cm, policy.summary_tokens, ctx) +
        prefill_seconds(cm, Float64(target) + policy.summary_tokens, 0)
    tracker.turns_since_compaction += 1
    out(compact, reason) = (
        compact=compact,
        reason=reason,
        target_tokens=target,
        compaction_cost_s=cost,
        accumulated_slowdown_s=tracker.accumulated_slowdown_s,
    )

    # Subtraction, not addition: safe may be typemax(Int).
    if ctx > safe - Int(next_turn_tokens)
        return ctx <= target ? out(false, :cannot_fit) : out(true, :hard_ceiling)
    end
    policy.rule === :never && return out(false, :never)
    ctx <= target && return out(false, :below_target)

    if policy.rule === :threshold
        fire = ctx >= policy.threshold_fraction * safe
        return out(fire, fire ? :threshold : :wait)
    end

    rent = max(
        0.0,
        decode_seconds(cm, new_tokens_per_turn, ctx) -
        decode_seconds(cm, new_tokens_per_turn, target),
    )
    tracker.accumulated_slowdown_s += rent
    fire = tracker.accumulated_slowdown_s >= cost
    return out(fire, fire ? :rent_or_buy : :wait)
end
