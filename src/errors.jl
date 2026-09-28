# Explicit-failure vocabulary (Harpe_Stack.md §LXX; North Star §22 taxonomy).
#
# Law: Harpe fails explicitly. No silent representation downgrade, backend
# switch, quantization mismatch, memory-plan violation, or kernel substitution
# unless policy explicitly permits it. If fallback occurs, it is recorded
# (see logging.jl @hfallback).
#
# All Harpe exceptions live here so every module throws typed errors instead
# of ad-hoc strings. Error CODES mirror the failure taxonomy so records can
# classify failures uniformly.

export HarpeException,
    HarpeError, ErrorCode, LoweringNotImplemented, lowering_not_implemented, harpe_error
export ERR_INTERNAL,
    ERR_INVALID_PLAN,
    ERR_CONSTRAINT_REJECTED,
    ERR_COMPILE,
    ERR_RESOURCE_LIMIT,
    ERR_ALLOCATION,
    ERR_LAUNCH,
    ERR_RUNTIME,
    ERR_TIMEOUT,
    ERR_VERIFY_MISMATCH,
    ERR_NUMERICAL_INSTABILITY,
    ERR_BENCHMARK,
    ERR_CACHE,
    ERR_APPROXIMATION_BUDGET_EXCEEDED

"Supertype of all Harpe exceptions (§LXX: failures are typed and explicit)."
abstract type HarpeException <: Exception end

"""
    ErrorCode

Failure taxonomy (North Star §22, extended by the KV memory program §6).
Failure records carry one code, a diagnostic, and whether the surrounding
search/session may continue.
"""
@enum ErrorCode begin
    ERR_INTERNAL = 0                 # Harpe bug; session should not continue
    ERR_INVALID_PLAN = 1             # malformed plan rejected before any work
    ERR_CONSTRAINT_REJECTED = 2      # candidate legally skipped, not failed
    ERR_COMPILE = 3                  # specialization/compilation failed
    ERR_RESOURCE_LIMIT = 4           # budget exhausted (time/candidates/memory)
    ERR_ALLOCATION = 5               # device/host allocation failure
    ERR_LAUNCH = 6                   # kernel launch failure
    ERR_RUNTIME = 7                  # execution-time failure
    ERR_TIMEOUT = 8                  # wall-clock budget exceeded mid-flight
    ERR_VERIFY_MISMATCH = 9          # correctness gate failed
    ERR_NUMERICAL_INSTABILITY = 10   # NaN/Inf/overflow-class failure
    ERR_BENCHMARK = 11               # measurement could not be produced
    ERR_CACHE = 12                   # cache read/write/invalidation failure
    ERR_APPROXIMATION_BUDGET_EXCEEDED = 13   # bounded approximation exceeded
    # its declared ε at its declared oracle
    # tier (KV memory program §6)
end

"""
    HarpeError <: HarpeException

The general Harpe failure. `code` classifies it per the taxonomy, `detail`
carries structured context that must survive into receipts.
"""
struct HarpeError <: HarpeException
    code::ErrorCode
    message::String
    detail::Dict{Symbol, Any}
end

harpe_error(code::ErrorCode, message::AbstractString; kw...) =
    HarpeError(code, String(message), Dict{Symbol, Any}(kw...))

function Base.showerror(io::IO, e::HarpeError)
    print(io, "HarpeError(", e.code, "): ", e.message)
    isempty(e.detail) || print(io, "  detail = ", e.detail)
end

"""
    LoweringNotImplemented <: HarpeException

Raised when an operation is lowered to a backend that does not implement it
(§LXX: failure must be explicit — no silent substitution, no silent fallback).

This is the *only* legal way for a lowering method to decline work. Returning
`nothing`, a sibling backend's result, or a CPU result instead is a law
violation. Policy-permitted degradation is a planner decision, recorded via
`@hfallback` — never a lowering-side surprise.
"""
struct LoweringNotImplemented <: HarpeException
    op::Symbol
    backend::Symbol
end

Base.showerror(io::IO, e::LoweringNotImplemented) = print(
    io,
    "LoweringNotImplemented: operation :",
    e.op,
    " has no lowering for backend :",
    e.backend,
    " (explicit failure per Harpe_Stack.md §LXX — no silent substitution)",
)

"""
    lowering_not_implemented(op::Symbol, backend) -> Nothing

Throw [`LoweringNotImplemented`](@ref) for `op` on `backend`. Standard body of
every lowering stub.
"""
function lowering_not_implemented(op::Symbol, backend::AbstractHarpeBackend)
    throw(LoweringNotImplemented(op, backend_name(backend)))
end
