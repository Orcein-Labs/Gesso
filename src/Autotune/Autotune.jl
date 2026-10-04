# Autotune — bounded physical-realization search (§XXVI; Phase 9;
# KV memory program §7).
#
# Owns: the search/verify/measure/cache/receipt machinery — bounded budgets,
# deterministic candidate ordering, correctness gates (oracle ladder), cost
# measurement (M, L, C), conservative cache invalidation, full receipts.
#
# Framing (KV program §7): this is a PHYSICAL-REALIZATION SEARCH ENGINE whose
# first client is kernel schedules (North Star project) and whose later
# clients include quantization programs, KV realization pipelines, and
# placement. The build stays kernel-first; the generalization is earned at
# Phase 10. Speculative plans are explicitly NOT a client (KV program §7.1).
#
# Integration stance: survey/integrate existing Julia autotuning work before
# building redundant infrastructure (§XXVI) — the North Star companion spec
# is that survey's starting point.
#
# Phase 9 (§LXXXII): the LOOP is the product. This module owns the generic
# candidate → compile → correctness gate → benchmark → winner → cache cycle
# and is backend-agnostic BY LAW: it does not import CUDA or Lava. Backend
# extensions register candidates (`register!`) and consult the winner
# (`select`); the first real client is `matmul!` on CUDA (§LXXXII exit:
# at least one operator automatically selects a device-specific winning
# implementation).
#
# Laws encoded here:
#   §LXX   a candidate that fails the correctness gate is DISQUALIFIED, never
#          retried into the winner slot; if EVERY candidate fails, search!
#          throws a typed GessoError (ERR_VERIFY_MISMATCH — the existing
#          "correctness gate failed" code) — never a silent keep, never a
#          CPU fallback because search missed.
#   §LXIX  the cache key carries AUTOTUNE_CACHE_VERSION; a differing version
#          is a miss (re-search), never a reinterpretation.
#   §XXXIII timing law: compile + first-touch happen OUTSIDE the timed
#          region (one compile call, warmup calls, then samples); medians
#          are post-warmup. No timing claim is made here — the numbers are
#          selection input and receipt payload, not prose.
#   §XLII  every search/cache-hit emits ONE Receipt through the existing
#          sink; context carries winner, per-candidate medians, rejections,
#          cache hit/miss. No new result type (§LXXXII: no ExecutionResult).
module Autotune

using ..Gesso:
    AUTOTUNE_CACHE_VERSION,
    Receipt,
    ReceiptSink,
    InMemorySink,
    emit!,
    new_receipt,
    default_receipt_sink,
    GessoError,
    gesso_error,
    ERR_VERIFY_MISMATCH

using Dates

export Candidate,
    TuneResult,
    register!,
    search!,
    select,
    invalidate!,
    invalidate_all!,
    candidates,
    cached_result

# --- vocabulary ---------------------------------------------------------------

"""
    Candidate(name, run!, validate)

One registered physical realization of one operator on one backend.

- `name::Symbol` — the realization's identity in receipts and cache entries
  (e.g. `:cublas_mul`).
- `run!(args...)` — executes the realization on the LIVE call arguments.
  Must be idempotent with respect to its destination argument (the search
  benchmarks call it repeatedly).
- `validate(args...) -> Bool` — the correctness gate, run BEFORE any timing.
  Receives the same live arguments and must run the realization itself (into
  its own scratch, if the destination must stay clean) and compare against
  the oracle the backend declared. Returning `false`, or throwing, means
  DISQUALIFIED (§LXX) — the candidate is recorded in `rejected` and can
  never win.

Register from the owning backend extension at load time (`__init__`), never
at precompile: the registry is runtime state, and precompile-time mutation
of another module's state breaks incremental compilation.
"""
Base.@kwdef struct Candidate
    name::Symbol
    run!::Function
    validate::Function
end

"""
    TuneResult

The outcome of one bounded search: the winning candidate name, per-candidate
post-warmup medians in nanoseconds (passing candidates only), the rejected
candidates with reasons, and whether this result came from the cache. Stored
under the cache key and replayed verbatim on a hit.
"""
struct TuneResult
    op::Symbol
    backend::Symbol
    regime::Symbol
    winner::Symbol
    medians::Dict{Symbol, Float64}
    rejected::Vector{Pair{Symbol, String}}
    cache_hit::Bool
    key::NamedTuple
end

# --- state (process-local; disk persistence is a later phase) ------------------

const _LOCK = ReentrantLock()
# (op, backend) => candidates in REGISTRATION ORDER — deterministic ordering
# is part of the loop's contract (§XXVI): ties break toward the earlier-
# registered candidate.
const _REGISTRY = Dict{Tuple{Symbol, Symbol}, Vector{Candidate}}()
const _CACHE = Dict{NamedTuple, TuneResult}()

_cache_key(op::Symbol, backend::Symbol, regime::Symbol, device::AbstractString) = (
    version=AUTOTUNE_CACHE_VERSION,
    device=String(device),
    backend=backend,
    op=op,
    regime=regime,
)

"""
    register!(op::Symbol, backend::Symbol, c::Candidate)

Register a candidate realization for `(op, backend)`. Re-registering a name
REPLACES the earlier entry (idempotent across extension reloads). Candidate
order = registration order; search ties break toward the earlier one.
"""
function register!(op::Symbol, backend::Symbol, c::Candidate)
    lock(_LOCK) do
        list = get!(() -> Candidate[], _REGISTRY, (op, backend))
        i = findfirst(x -> x.name === c.name, list)
        i === nothing ? push!(list, c) : (list[i] = c)
    end
    return nothing
end

"""
    candidates(op::Symbol, backend::Symbol) -> Vector{Candidate}

The registered candidates in deterministic (registration) order. A copy —
callers cannot mutate the registry.
"""
function candidates(op::Symbol, backend::Symbol)
    lock(_LOCK) do
        return copy(get(_REGISTRY, (op, backend), Candidate[]))
    end
end

# --- the loop (§XXVI) ----------------------------------------------------------

"""
    search!(op, backend, regime, device, args...; sink, samples, warmup) -> TuneResult

Run the full loop against the LIVE call arguments `args`:

1. candidate generation — the registered `(op, backend)` candidates, in
   registration order;
2. compile + first-touch — one untimed `run!`, then `warmup` untimed calls
   (§XXXIII: nothing that compiles is ever timed);
3. correctness validation — `validate(args...)` BEFORE any timing; a `false`
   or a throw DISQUALIFIES the candidate (recorded with a reason, §LXX);
4. benchmark — `samples` timed calls, post-warmup; the per-candidate median
   (ns) is the cost;
5. select winner — lowest median among passing candidates; ties break toward
   the earlier-registered candidate;
6. cache + receipt — one receipt through `sink` (winner, medians, rejects).

If NO candidate passes the gate, throws `GessoError(ERR_VERIFY_MISMATCH)` —
the existing "correctness gate failed" code — because no legal lowering
exists for `(op, backend, regime)` on this device (§LXX: never silently keep
a failing implementation, never fall back to CPU because search missed).
"""
function search!(
    op::Symbol,
    backend::Symbol,
    regime::Symbol,
    device::AbstractString,
    args...;
    sink::ReceiptSink=default_receipt_sink(),
    samples::Int=16,
    warmup::Int=2,
)
    cands = candidates(op, backend)
    isempty(cands) && throw(
        gesso_error(
            ERR_VERIFY_MISMATCH,
            "search!: no registered candidates for (:$(op), :$(backend)) — " *
            "no legal lowering exists to select among (§LXX)",
            op=op,
            backend=backend,
            regime=regime,
        ),
    )

    medians = Dict{Symbol, Float64}()
    rejected = Pair{Symbol, String}[]

    for c in cands
        gate_ok = try
            c.validate(args...) === true
        catch e
            push!(rejected, c.name => "gate threw: " * _brief(e))
            continue
        end
        gate_ok || begin
            push!(rejected, c.name => "correctness gate failed")
            continue
        end

        bench_ok = try
            c.run!(args...)                       # compile + first-touch: OUTSIDE timing (§XXXIII)
            for _ in 1:warmup
                c.run!(args...)
            end
            times = Float64[]
            for _ in 1:samples
                push!(times, @elapsed c.run!(args...))
            end
            medians[c.name] = median(times) * 1e9 # ns, post-warmup
            true
        catch e
            push!(rejected, c.name => "bench threw: " * _brief(e))
            false
        end
        bench_ok || continue
    end

    isempty(medians) && throw(
        gesso_error(
            ERR_VERIFY_MISMATCH,
            "search!: every candidate for (:$(op), :$(backend), :$(regime)) " *
            "failed the correctness gate — no legal lowering exists (§LXX: " *
            "never silently keep a failing implementation, never fall back " *
            "to another backend because search missed)",
            op=op,
            backend=backend,
            regime=regime,
            device=String(device),
            rejected=String[p[2] for p in rejected],
        ),
    )

    # deterministic winner: lowest median; ties break toward the
    # earlier-registered candidate (argmin returns the first minimal element
    # of the registration-ordered passing list)
    passing = filter(c -> haskey(medians, c.name), cands)
    winner = argmin(c -> medians[c.name], passing).name

    result = TuneResult(
        op,
        backend,
        regime,
        winner,
        medians,
        rejected,
        false,
        _cache_key(op, backend, regime, device),
    )
    _emit(op, backend, regime, device, winner, medians, rejected, false, sink)
    return result
end

"""
    select(op, backend, regime, device, args...; sink, samples, warmup) -> TuneResult

The consult site's entry point: return the cached `TuneResult` for the key
`(AUTOTUNE_CACHE_VERSION, device, backend, op, regime)` when present (a
cache HIT — no re-search and NO receipt; the returned value carries
`cache_hit === true` and shares every other field with the stored search
record), else run the full `search!` (a MISS), emit the one
`:autotune_select` receipt, and cache the result.
"""
function select(
    op::Symbol,
    backend::Symbol,
    regime::Symbol,
    device::AbstractString,
    args...;
    sink::ReceiptSink=default_receipt_sink(),
    samples::Int=16,
    warmup::Int=2,
)
    key = _cache_key(op, backend, regime, device)
    hit = lock(_LOCK) do
        get(_CACHE, key, nothing)
    end
    if hit isa TuneResult
        # Phase 10G item A — A CACHE HIT IS NOT A DECISION. §LXXII records a
        # CHANGE; the miss receipt already named the winner, so emitting the
        # same 8-entry Dict on every hit is the same decision copied once per
        # consult (633 consults per SmolLM2 token). Return the cached entry
        # with the ONE consult-site field flipped and emit nothing.
        #
        # The flip is a new immutable sharing every decision-bearing field —
        # same `medians` / `rejected` OBJECTS, not copies — because
        # `TuneResult` is a non-isbits struct (Dict + Vector fields) for which
        # `===` is FIELD-WISE. So a hit is NOT `===` the stored entry (the
        # `cache_hit` field differs) but every other field is identical by
        # object identity; tests pin that rather than `===`. Reconstructing
        # the wrapper allocates nothing.
        #
        # The STORED `_CACHE` entry keeps `cache_hit = false`: it is the
        # search record, and it is what `cached_result` hands back.
        return TuneResult(
            hit.op,
            hit.backend,
            hit.regime,
            hit.winner,
            hit.medians,
            hit.rejected,
            true,
            hit.key,
        )
    end
    result = search!(op, backend, regime, device, args...; sink, samples, warmup)
    lock(_LOCK) do
        _CACHE[key] = result
    end
    return result
end

"""
    cached_result(op, backend, regime; device) -> Union{TuneResult, Nothing}

The cached result for the key, if any. Inspection surface for tests and
tooling — never mutates.
"""
function cached_result(
    op::Symbol,
    backend::Symbol,
    regime::Symbol;
    device::Union{Nothing, AbstractString}=nothing,
)
    lock(_LOCK) do
        for (key, result) in _CACHE
            key.op === op && key.backend === backend && key.regime === regime || continue
            device === nothing && return result
            key.device == String(device) && return result
        end
        return nothing
    end
end

"""
    invalidate!(op, backend, regime; device = nothing) -> Int

Drop cache entries for `(op, backend, regime)` — for one `device`, or across
all devices when `device` is not given. Returns the number of entries
dropped. The next `select` is a miss and re-searches (conservative
invalidation, §XXVI).
"""
function invalidate!(
    op::Symbol,
    backend::Symbol,
    regime::Symbol;
    device::Union{Nothing, AbstractString}=nothing,
)
    return lock(_LOCK) do
        dropped = 0
        for key in collect(keys(_CACHE))
            key.op === op && key.backend === backend && key.regime === regime || continue
            device === nothing || key.device == String(device) || continue
            delete!(_CACHE, key)
            dropped += 1
        end
        dropped
    end
end

"""
    invalidate_all!()

Drop the entire cache table. The nuclear option — used by tests and by
operators who know every key is stale.
"""
function invalidate_all!()
    lock(_LOCK) do
        empty!(_CACHE)
    end
    return nothing
end

# --- internals ------------------------------------------------------------------

_brief(e) = sprint(showerror, e)[1:min(end, 120)]

# Local median: honest two-middle convention for even lengths. Defined here
# (rather than importing Statistics) to keep the module's dependency surface
# at stdlib Dates only.
function median(v::Vector{Float64})
    s = sort(v)
    n = length(s)
    n == 0 && throw(ArgumentError("median of empty collection"))
    isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2
end

function _emit(
    op::Symbol,
    backend::Symbol,
    regime::Symbol,
    device::AbstractString,
    winner::Symbol,
    medians::Dict{Symbol, Float64},
    rejected::Vector{Pair{Symbol, String}},
    cache_hit::Bool,
    sink::ReceiptSink,
)
    r = new_receipt(;
        task=:autotune_select,
        context=Dict{Symbol, Any}(
            :op => op,
            :backend => backend,
            :regime => regime,
            :device => String(device),
            :cache_version => string(AUTOTUNE_CACHE_VERSION),
            :winner => winner,
            :medians => Dict{Symbol, Float64}(medians),
            :rejected => [(p[1], p[2]) for p in rejected],
            :cache_hit => cache_hit,
        ),
    )
    emit!(sink, r)
    return r
end

end # module Autotune
