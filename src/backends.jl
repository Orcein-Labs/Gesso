# Backend interface (Harpe_Stack.md §XX, §XXI, §XXII-XXIII; §LXXIII draft).
#
# This is the seam every backend lowering implements — CUDA.jl in Phase 4,
# Lava in Phase 8 — written now so later phases cannot quietly design their
# own incompatible notions of "a backend".
#
# Laws encoded here:
#
#   §I / §XXII-XXIII  Harpe decides what should execute; the backend determines
#                     how. Backends are capability surfaces, not identities.
#                     "What can this device legally and efficiently execute?",
#                     not "Is this NVIDIA?"
#
#   §XXI              Execution tiers. Every higher tier must retain a
#                     lower-tier fallback. Tier 0 (PORTABLE_CORRECTNESS) is
#                     always available because it is the CPU reference.
#
#   §LXX              Failure must be explicit. Missing backend or unsupported
#                     operation raises (see errors.jl) — never silently
#                     substitutes.
#
# Scope here: backend types, traits, and the lowering-operation contract
# signatures only. Concrete execution arrives with Phase 2 (CPU reference).

export AbstractHarpeBackend, CPUBackend
export backend_name, execution_tier, supports

"""
    AbstractHarpeBackend

Supertype of Harpe backend tags. A backend instance is an *intent* — the
concrete device is resolved from it at lowering time.

Concrete implementations live in package extensions (`HarpeCUDAExt`,
`HarpeLavaExt`) or in core for the CPU reference path. Core Harpe defines only
[`CPUBackend`](@ref) and knows nothing else about specific vendors (§I).
"""
abstract type AbstractHarpeBackend end

"""
    CPUBackend <: AbstractHarpeBackend

The Tier 0 backend (§XXI, PORTABLE_CORRECTNESS). Always available, always
correct, performance explicitly secondary. The bottom of every fallback chain.
"""
struct CPUBackend <: AbstractHarpeBackend end

"""
    backend_name(b) -> Symbol

Stable identifier for a backend, e.g. `:cpu`. Extension-defined backends
return their own symbols (`:cuda`, `:lava`).
"""
backend_name(::CPUBackend) = :cpu

"""
    execution_tier(b) -> Int

The execution tier this backend realizes (§XXI):

    0  PORTABLE_CORRECTNESS          generic legal execution
    1  OPTIMIZED_GENERIC             broadly tuned common-GPU implementations
    2  ARCHITECTURE_SPECIALIZED      vendor/architecture-family specialization
    3  DEVICE+MODEL+WORKLOAD         profile-derived specialization

CPU reference is tier 0 by definition.
"""
execution_tier(::CPUBackend) = 0

"""
    supports(backend, capability::Symbol) -> Bool

Capability query (§XX). Answers "can this backend legally execute X?" — never
"what vendor is this?". Capability symbol vocabulary is established per
operator family in later phases; unknown capabilities must return `false`
rather than throw, so capability probing is always safe.
"""
# Phase 2 (§LXXV): CPU reference coverage. `true` only where a CPU method
# actually computes; :quantize/:dequantize stay false until Representation
# lands their math; unknown capabilities stay false (probing is always safe).
const CPU_SUPPORTED_CAPS =
    Set([:rmsnorm, :rope, :softmax, :swiglu, :matmul, :embedding_lookup],)
supports(::CPUBackend, cap::Symbol) = cap in CPU_SUPPORTED_CAPS

# --- Lowering operation contract -------------------------------------------
#
# Signatures only; every stub fails explicitly via errors.jl. Concrete methods
# arrive with Phase 2 (CPU reference execution). Method names are the
# contract; argument order follows the convention `op!(dst..., src...;
# workload)` where `dst`/`src` are semantic tensors (§XI) and workload
# identifies the execution phase (§XXX).
#
# Add new operations here as the vocabulary grows. An operation missing from
# this list does not exist for planners; an operation present here but
# unimplemented for a backend raises LoweringNotImplemented (errors.jl).

for op in (
    :rmsnorm!,
    :rope!,
    :softmax!,
    :swiglu!,
    :matmul!,
    :embedding_lookup!,
    :quantize!,
    :dequantize!,
)
    @eval function $op(args...; kwargs...)
        lowering_not_implemented($(QuoteNode(op)), args[1])
    end
end

# `args[1]` is the destination/workload-led first argument in the op! convention;
# the stub resolves the backend from it. Documented contract: first positional
# argument of every lowering op is backend-carrying (a backend tag or a semantic
# object that chains to one). Phase 2 replaces stubs with real CPU methods.
