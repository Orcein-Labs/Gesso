# Test helpers — the correctness laboratory's reusable fixtures (goal §H).
#
# These live in test/ deliberately: they are LABORATORY EQUIPMENT, not Harpe
# API. Nothing here decides ModelIR shape, tolerance POLICY, or the KV
# program's oracle ladder — it only provides deterministic mechanics that
# Phase 2's CPU-oracle differential tests (and any earlier parity checks)
# will reuse. When Harpe adopts declared approximation contracts
# (BoundedApproximation{metric, ε, oracle_tier}, KV program §6), the
# default tolerances here should be replaced by values THOSE contracts
# declare — the helpers take tolerances as arguments precisely so no
# default can quietly become policy.
#
# Stability note: Random.Xoshiro's stream is fixed by its algorithm for a
# given seed (Julia ≥ 1.7 RNG), so fixtures seeded through deterministic_rng
# are reproducible across runs and, short of a documented stdlib RNG change,
# across Julia versions. If Julia ever changes Xoshiro's stream, seeded
# fixture tests fail loudly and this file is the place to record the bump.

module HarpeTestHelpers

using Random

export deterministic_rng, approx_eq, parity_report, assert_parity

"""
    deterministic_rng([seed]) -> Random.Xoshiro

A seeded RNG for fixture data. Same seed ⇒ same stream, always. Called with
no argument it uses a fixed default seed, so "just give me deterministic
randoms" is one call and needs no magic numbers at call sites.
"""
deterministic_rng(seed::Integer=0x6f11406b13a90d0f) = Random.Xoshiro(UInt64(seed))

"""
    approx_eq(a, b; rtol=0, atol=0) -> Bool

Elementwise agreement under `|a[i] - b[i]| ≤ atol + rtol * max(|a[i]|, |b[i]|)`,
with exact agreement (`isequal`) short-circuiting first — so `NaN === NaN`
pairs pass, `Inf == Inf` pairs pass, and any NaN-vs-finite pair fails.
`0.0` vs `-0.0` passes (their difference is exactly zero).

Requires matching axes and matching element types: converting between float
widths is a semantics decision a test must make explicitly, not silently.
Default is EXACT equality (`rtol=0, atol=0`): loosen only with values a
declared contract justifies.
"""
function approx_eq(
    a::AbstractArray{F},
    b::AbstractArray{F};
    rtol::Real=zero(F),
    atol::Real=zero(F),
) where {F <: AbstractFloat}
    axes(a) == axes(b) || return false
    for i in eachindex(a)
        isequal(a[i], b[i]) && continue
        tol = atol + rtol * max(abs(a[i]), abs(b[i]))
        abs(a[i] - b[i]) <= tol || return false
    end
    return true
end

function approx_eq(
    a::F,
    b::F;
    rtol::Real=zero(F),
    atol::Real=zero(F),
) where {F <: AbstractFloat}
    isequal(a, b) && return true
    tol = atol + rtol * max(abs(a), abs(b))
    return abs(a - b) <= tol
end

"""
    parity_report(a, b; name=:unnamed, rtol=0, atol=0) -> NamedTuple

Diagnostics for a comparison that is allowed to fail — for reports, logs,
and `assert_parity`. Never throws on disagreement; describes it:

    (name, kind, agree, max_abs_diff, rel_at_max, n_mismatch, n_exact,
     first_mismatch)

`kind` is `:shape_mismatch`, `:exact` (every element `isequal`),
`:tolerance` (agrees but not exactly), or `:mismatch`.
"""
function parity_report(
    a::AbstractArray{F},
    b::AbstractArray{F};
    name::Symbol=:unnamed,
    rtol::Real=zero(F),
    atol::Real=zero(F),
) where {F <: AbstractFloat}
    if axes(a) != axes(b)
        return (
            name=name,
            kind=:shape_mismatch,
            agree=false,
            max_abs_diff=NaN,
            rel_at_max=NaN,
            n_mismatch=0,
            n_exact=0,
            first_mismatch=nothing,
        )
    end
    n_mismatch = 0
    n_exact = 0
    max_diff = zero(F)
    first_mismatch = nothing
    max_abs_diff_idx = first(axes(a))[1]  # placeholder, corrected below
    for i in eachindex(a)
        if isequal(a[i], b[i])
            n_exact += 1
            continue
        end
        d = abs(a[i] - b[i])
        tol = atol + rtol * max(abs(a[i]), abs(b[i]))
        if d > tol
            n_mismatch += 1
            first_mismatch === nothing && (first_mismatch = i)
        end
        if d > max_diff
            max_diff = d
            max_abs_diff_idx = i
        end
    end
    agree = n_mismatch == 0
    rel =
        max_diff == zero(F) ? zero(F) :
        max_diff / max(abs(a[max_abs_diff_idx]), abs(b[max_abs_diff_idx]))
    kind = !agree ? :mismatch : (n_exact == length(a) ? :exact : :tolerance)
    return (
        name=name,
        kind=kind,
        agree=agree,
        max_abs_diff=max_diff,
        rel_at_max=rel,
        n_mismatch=n_mismatch,
        n_exact=n_exact,
        first_mismatch=first_mismatch,
    )
end

"""
    assert_parity(report; context=()) -> report

Throw a diagnostic-rich error if `report` (from [`parity_report`](@ref))
describes disagreement; return the report otherwise. `context` is echoed
into the message so the failure points at the case, not just the numbers.
"""
function assert_parity(rep; context=())
    rep.agree && return rep
    msg = sprint() do io
        println(io, "parity FAILED: ", rep.name)
        println(io, "  kind           = ", rep.kind)
        println(io, "  max_abs_diff   = ", rep.max_abs_diff)
        println(io, "  rel_at_max     = ", rep.rel_at_max)
        println(io, "  n_mismatch     = ", rep.n_mismatch)
        println(io, "  first_mismatch = ", rep.first_mismatch)
        println(io, "  context        = ", context)
    end
    error(msg)
end

end # module HarpeTestHelpers
