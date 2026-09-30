# GessoLavaExt — the Lava.jl (Vulkan) backend as a package extension
# (§LXXXI, §VII, §XXIII).
#
# This file is the ONLY place in Gesso allowed to `using Lava`. Core Gesso
# compiles and tests with Lava not loaded; `LavaBackend` lives HERE, never in
# `src/backends.jl` (§VII: backends are extensions, not dependencies).
#
# Laws:
#   * NO silent fallback (§LXX): `LavaBackend()` with no usable Vulkan device
#     throws a typed GessoError (ERR_RESOURCE_LIMIT, used consistently for
#     device absence). It never returns a CPUBackend or a CUDABackend. Lava
#     has no `functional()` twin of CUDA.jl — the constructor probes
#     `Lava.vk_context()` and fails explicitly through the same typed error.
#   * Name clash (§LXXXI, load-bearing): Lava.jl exports its own
#     `LavaBackend <: KernelAbstractions.GPU`. Inside this extension the KA
#     type is aliased `KALava` and is NEVER exported from Gesso; Gesso's tag
#     is the `LavaBackend` defined HERE and bound into Gesso's namespace in
#     `__init__`. `using Gesso, Lava` stays valid — callers qualify
#     (`Gesso.LavaBackend` vs `Lava.LavaBackend`).
#   * Same operators (§CIX): more-specific methods extend the EXISTING `op!`
#     functions from `src/backends.jl` — no second vocabulary. The operator
#     methods arrive with Phase 8 item B (ext/lava_ops.jl); `quantize!` /
#     `dequantize!` keep declining.
#   * Math: `Lava.LavaArray{Float32}` storage (§LXXXI). The CPU oracle stays
#     F64; parity gates compare with the declared atol (1e-3, micro models,
#     seq ≤ 8), not bit-identity. Never silently widen past 1e-2 without a
#     decision packet.
#   * Implementation is GPUArrays broadcasting + Lava's `mul!` (Lava
#     array/gemm.jl) — at most one KernelAbstractions `@kernel` per op where
#     broadcast cannot express it (RoPE pairwise rotate, causal softmax),
#     launched on the KA backend (`KALava`), never on Gesso's tag. No
#     handwritten SPIR-V, no coopmat, no graphics, no ray tracing. This
#     sprint is the portable SEAM (§LXXXI), not a kernel contest.
module GessoLavaExt

# the bare `import Gesso` binds the MODULE NAME (needed for the Core.eval
# namespace binding below); the named imports extend the existing vocabulary
import Gesso
import Gesso:
    backend_name,
    execution_tier,
    supports

using Gesso:
    AbstractGessoBackend,
    GessoError,
    gesso_error,
    ERR_RESOURCE_LIMIT

using Lava

# --- backend tag ------------------------------------------------------------

# Lava's OWN KernelAbstractions GPU backend (kernels, synchronize, launch).
# Aliased so nothing below mistakes it for Gesso's tag (§LXXXI name clash).
const KALava = Lava.LavaBackend

"""
    LavaBackend <: AbstractGessoBackend

Tier 1 backend (§XXI, OPTIMIZED_GENERIC; §XXIII: the portable machine path —
NVIDIA, AMD, Intel, MoltenVK, lavapipe) executing on Lava.jl's Vulkan runtime.

Constructing one WITHOUT a usable Vulkan device throws
`GessoError(ERR_RESOURCE_LIMIT)` — an explicit failure (§LXX), never a silent
CPU or CUDA fallback. The probe is the constructor itself: Lava exposes no
`functional()` twin, so `Lava.vk_context()` is asked and any failure becomes
the one typed device-absence error.
"""
struct LavaBackend <: AbstractGessoBackend
    function LavaBackend()
        ok = true
        try
            Lava.vk_context()
        catch
            ok = false
        end
        ok || throw(
            gesso_error(
                ERR_RESOURCE_LIMIT,
                "LavaBackend: no usable Vulkan device — requesting Lava " *
                "without a device fails explicitly (§LXX); Gesso never falls " *
                "back to CPU or CUDA silently. Check vulkaninfo / the ICD, " *
                "or stay on CPUBackend (tier 0).";
                requested_backend = :lava,
                vk_context_failed = true,
            ),
        )
        return new()
    end
end

backend_name(::LavaBackend) = :lava

execution_tier(::LavaBackend) = 1   # OPTIMIZED_GENERIC (§XXI)

# Type-level traits: a device-less machine can still probe the seam's
# identity without constructing (the constructor is the ONLY thing the
# missing device blocks — §LXXXI "never skip the extension-loads tests").
backend_name(::Type{LavaBackend}) = :lava
execution_tier(::Type{LavaBackend}) = 1

# Same capability coverage as the CPU reference: the six implemented ops.
# :quantize / :dequantize stay false (Representation-phase concern), and
# unknown capabilities stay false — probing is always safe (§XX).
const LAVA_SUPPORTED_CAPS = Set([
    :rmsnorm,
    :rope,
    :softmax,
    :swiglu,
    :matmul,
    :embedding_lookup,
])

supports(::LavaBackend, cap::Symbol) = cap in LAVA_SUPPORTED_CAPS
supports(::Type{LavaBackend}, cap::Symbol) = cap in LAVA_SUPPORTED_CAPS

# Bind the backend into the package's namespace: after this ext triggers,
# `Gesso.LavaBackend` resolves to the type defined HERE — while a Lava-less
# load of core leaves the name entirely absent (both directions are pinned
# by the seam test). No `struct LavaBackend` exists in core source (§VII).
#
# The binding happens in __init__ (load time), NOT at top level: eval into
# another package's module during precompilation breaks incremental
# compilation (the closed-module rule) — but method attachments to Gesso's
# functions above are the sanctioned extension mechanism and precompile
# fine. The binding is a per-session side effect, which is exactly what
# __init__ is for. (Same pattern as GessoCUDAExt; `to_device` joins the
# binding with Phase 8 item B, when the operator methods land.)
function __init__()
    Core.eval(Gesso, :(const LavaBackend = $LavaBackend))
    nothing
end

export LavaBackend

end # module GessoLavaExt
