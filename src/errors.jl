# Explicit-failure vocabulary (Gesso_Stack.md §LXX; North Star §22 taxonomy).
#
# Law: Gesso fails explicitly. No silent representation downgrade, backend
# switch, quantization mismatch, memory-plan violation, or kernel substitution
# unless policy explicitly permits it. If fallback occurs, it is recorded
# (see logging.jl @gfallback).
#
# All Gesso exceptions live here so every module throws typed errors instead
# of ad-hoc strings. Error CODES mirror the failure taxonomy so records can
# classify failures uniformly.

export GessoException,
    GessoError, ErrorCode, LoweringNotImplemented, lowering_not_implemented, gesso_error
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

"Supertype of all Gesso exceptions (§LXX: failures are typed and explicit)."
abstract type GessoException <: Exception end

"""
    ErrorCode

Failure taxonomy (North Star §22, extended by the KV memory program §6).
Failure records carry one code, a diagnostic, and whether the surrounding
search/session may continue.
"""
@enum ErrorCode begin
    ERR_INTERNAL = 0                 # Gesso bug; session should not continue
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

# Intended producers — documentation of ownership, not a registry. Which
# module raises a code is settled by the phase that owns the failing concept
# (docs/ARCHITECTURE.md); adding a code means editing this taxonomy, its
# stability test (test/test_errors.jl), and the owning phase's work item.
#
#   ERR_INTERNAL                     any module, on an invariant violation
#   ERR_INVALID_PLAN                 Planning (§XVII)
#   ERR_CONSTRAINT_REJECTED          Planning/Autotune candidate filters
#   ERR_COMPILE                      Lowering/Autotune (§XXII–XXIII)
#   ERR_RESOURCE_LIMIT               Planning/Autotune budgets
#   ERR_ALLOCATION                   Inference/Runtime (device + host)
#   ERR_LAUNCH                       backend extensions (CUDA/Lava)
#   ERR_RUNTIME                      backends + Inference (§XXIX)
#   ERR_TIMEOUT                      Runtime budgets (§XXXII)
#   ERR_VERIFY_MISMATCH              correctness gates (KV program oracle ladder)
#   ERR_NUMERICAL_INSTABILITY        Operators/backends
#   ERR_BENCHMARK                    benchmark harness (§XXXIII)
#   ERR_CACHE                        Inference/Runtime cache + KV layer (§XXXI)
#   ERR_APPROXIMATION_BUDGET_EXCEEDED declared BoundedApproximation contracts
#                                    (KV memory program §6)
#
# The integer values are the persisted identity of a code (failure records
# classify and may persist by code; §LXIX). Renumbering, reordering, or
# inserting codes without a deliberate taxonomy review is a compatibility
# break — test/test_errors.jl pins them.

"""
    GessoError <: GessoException

The general Gesso failure. `code` classifies it per the taxonomy, `detail`
carries structured context that must survive into receipts.
"""
struct GessoError <: GessoException
    code::ErrorCode
    message::String
    detail::Dict{Symbol, Any}
end

gesso_error(code::ErrorCode, message::AbstractString; kw...) =
    GessoError(code, String(message), Dict{Symbol, Any}(kw...))

function Base.showerror(io::IO, e::GessoError)
    print(io, "GessoError(", e.code, "): ", e.message)
    isempty(e.detail) || print(io, "  detail = ", e.detail)
end

"""
    LoweringNotImplemented <: GessoException

Raised when an operation is lowered to a backend that does not implement it
(§LXX: failure must be explicit — no silent substitution, no silent fallback).

This is the *only* legal way for a lowering method to decline work. Returning
`nothing`, a sibling backend's result, or a CPU result instead is a law
violation. Policy-permitted degradation is a planner decision, recorded via
`@gfallback` — never a lowering-side surprise.
"""
struct LoweringNotImplemented <: GessoException
    op::Symbol
    backend::Symbol
end

Base.showerror(io::IO, e::LoweringNotImplemented) = print(
    io,
    "LoweringNotImplemented: operation :",
    e.op,
    " has no lowering for backend :",
    e.backend,
    " (explicit failure per Gesso_Stack.md §LXX — no silent substitution)",
)

"""
    lowering_not_implemented(op::Symbol, backend) -> Nothing

Throw [`LoweringNotImplemented`](@ref) for `op` on `backend`. Standard body of
every lowering stub.
"""
function lowering_not_implemented(op::Symbol, backend::AbstractGessoBackend)
    throw(LoweringNotImplemented(op, backend_name(backend)))
end
