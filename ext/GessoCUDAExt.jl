# GessoCUDAExt — the CUDA.jl backend as a package extension (§LXXVII, §VII).
#
# This file is the ONLY place in Gesso allowed to `using CUDA`. Core Gesso
# compiles and tests with CUDA not loaded; `CUDABackend` lives HERE, never
# in `src/backends.jl` (§VII: backends are extensions, not dependencies).
#
# Laws:
#   * NO silent fallback (§LXX): `CUDABackend()` with no functional device
#     throws a typed GessoError (ERR_RESOURCE_LIMIT, used consistently for
#     device absence). It never returns a CPUBackend.
#   * Same operators (§CIX): the methods below extend the EXISTING `op!`
#     functions from `src/backends.jl` — no second vocabulary. Dispatch is
#     on `CUDABackend` × semantic families × workload, exactly like the CPU
#     reference. `quantize!` / `dequantize!` still decline.
#   * Math: CuArray{Float32} this sprint (§LXXVII: F64 135M is ~1.08 GiB and
#     a poor GPU path; F32 is the honest first NVIDIA lowering). The CPU
#     oracle stays F64; parity gates compare with the declared atol, not
#     bit-identity.
#   * Implementation is CuArray broadcasting + CUBLAS (`mul!`) — no fused
#     kernels, no PTX, no CUTLASS. This sprint is the backend SEAM, not a
#     kernel contest (§LXXVII).
module GessoCUDAExt

# the bare `import Gesso` binds the MODULE NAME (needed for the Core.eval
# namespace binding below); the named imports extend the existing vocabulary
import Gesso
import Gesso:
    rmsnorm!,
    rope!,
    softmax!,
    swiglu!,
    matmul!,
    embedding_lookup!,
    backend_name,
    execution_tier,
    supports

using Gesso:
    AbstractGessoBackend,
    Activation,
    EmbeddingTable,
    ProjectionWeight,
    FrozenParameter,
    TemporaryWorkspace,
    PrefillWorkload,
    DecodeWorkload,
    GessoError,
    gesso_error,
    ERR_RESOURCE_LIMIT,
    ERR_INVALID_PLAN

using CUDA

# --- backend tag ------------------------------------------------------------

"""
    CUDABackend <: AbstractGessoBackend

Tier 1 backend (§XXI, OPTIMIZED_GENERIC): CUDA.jl execution on the device
CUDA considers current. Constructing one WITHOUT a functional device throws
`GessoError(ERR_RESOURCE_LIMIT)` — an explicit failure (§LXX), never a
silent CPU fallback.
"""
struct CUDABackend <: AbstractGessoBackend
    function CUDABackend()
        CUDA.functional() || throw(
            gesso_error(
                ERR_RESOURCE_LIMIT,
                "CUDABackend: no functional NVIDIA device — requesting CUDA " *
                "without a device fails explicitly (§LXX); Gesso never falls " *
                "back to CPU silently. Check the driver, or stay on " *
                "CPUBackend (tier 0).";
                requested_backend = :cuda,
                functional = false,
            ),
        )
        return new()
    end
end

backend_name(::CUDABackend) = :cuda

execution_tier(::CUDABackend) = 1   # OPTIMIZED_GENERIC (§XXI)

# Type-level traits: a device-less machine can still probe the seam's
# identity without constructing (the constructor is the ONLY thing the
# missing device blocks — §LXXVII "never skip the extension-loads tests").
backend_name(::Type{CUDABackend}) = :cuda
execution_tier(::Type{CUDABackend}) = 1

# Same capability coverage as the CPU reference: the six implemented ops.
# :quantize / :dequantize stay false (Representation-phase concern), and
# unknown capabilities stay false — probing is always safe (§XX).
const CUDA_SUPPORTED_CAPS = Set([
    :rmsnorm,
    :rope,
    :softmax,
    :swiglu,
    :matmul,
    :embedding_lookup,
])

supports(::CUDABackend, cap::Symbol) = cap in CUDA_SUPPORTED_CAPS
supports(::Type{CUDABackend}, cap::Symbol) = cap in CUDA_SUPPORTED_CAPS

# Phase 4 item B (§LXXVII): the CUDA operator methods on CuArray{Float32}
# + explicit to_device transfer. Same op names, more-specific methods (§CIX).
include("cuda_ops.jl")

# Bind the backend into the package's namespace: after this ext triggers,
# `Gesso.CUDABackend` resolves to the type defined HERE — while a CUDA-less
# load of core leaves the name entirely absent (both directions are pinned
# by the seam test). No `struct CUDABackend` exists in core source (§VII).
#
# The binding happens in __init__ (load time), NOT at top level: eval into
# another package's module during precompilation breaks incremental
# compilation (the closed-module rule) — but method attachments to Gesso's
# functions above are the sanctioned extension mechanism and precompile
# fine. The binding is a per-session side effect, which is exactly what
# __init__ is for.
#
# `to_device` attach-or-own (Phase 9 item A, §LXXXII — fixes the wart the
# Phase 8 receipts documented): when another backend extension (GessoLavaExt)
# already installed `Gesso.to_device`, THIS extension defines its method BY
# NAME against that existing binding instead of replacing the function
# object. One function object, both backends' methods, either load order.
# (Mirrors GessoLavaExt.__init__; see there for the empirical note on why
# the definition must go through the NAME — interpolating the function
# object into the definition head is not a legal method definition.)
function __init__()
    Core.eval(Gesso, :(const CUDABackend = $CUDABackend))
    _autotune_register!()   # §LXXXII: backend extensions register candidates (runtime state)
    if isdefined(Gesso, :to_device)
        Core.eval(
            Gesso,
            quote
                function to_device(b::CUDABackend, tensors)
                    $(_to_device_cuda)(tensors)
                end
            end,
        )
    else
        Core.eval(Gesso, :(const to_device = $to_device))
    end
    nothing
end

export CUDABackend

end # module GessoCUDAExt
