# GessoAMDGPUExt — the AMDGPU.jl (ROCm) backend as a package extension (§VII).
#
# GPU-UNTESTED. No AMD GPU has run this code. The suite runs every kernel here
# on KernelAbstractions' CPU backend against the CPU oracle and compiles it for
# gfx1030, gfx1100, gfx90a and gfx942 with no device. That is the whole claim.
#
# This file is the ONLY place in Gesso allowed to `using AMDGPU`. Core Gesso
# compiles and tests with AMDGPU not loaded; `AMDGPUBackend` lives HERE, never
# in `src/backends.jl`.
#
# Laws:
#   * NO silent fallback (§LXX): `AMDGPUBackend()` with no functional ROCm
#     device throws GessoError(ERR_RESOURCE_LIMIT), as CUDA and Lava do.
#   * Same operators (§CIX): methods extend the existing `op!` functions and
#     the core storage seams. Where a core seam's generic AbstractArray body
#     is already correct on ROCArray, it is left alone; a ROCArray method is
#     added only where a kernel is faster.
#   * One kernel source. Kernels are KernelAbstractions, written against the
#     generic `KADeviceBackend{B}` so that `KADeviceBackend(KA.CPU())` runs the
#     same kernels on the host for the parity tests. `AMDGPUBackend` is
#     `KADeviceBackend{ROCBackend}`.
#   * Portable kernels only: no wavefront-size assumption (gfx10/11 run wave32,
#     CDNA wave64), power-of-two workgroups of at most 256, tree reductions
#     in local memory with uniform trip counts, no vendor intrinsics.
#   * Math: Float32 device storage. The CPU oracle stays F64.
module GessoAMDGPUExt

import Gesso
import Gesso: backend_name, execution_tier, supports
import Gesso: rmsnorm!, rope!, softmax!, swiglu!, matmul!, embedding_lookup!

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
    ERR_INVALID_PLAN,
    ERR_ALLOCATION,
    ERR_RUNTIME,
    ERR_NUMERICAL_INSTABILITY

using LinearAlgebra: mul!

using AMDGPU
using AMDGPU: ROCBackend
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @Const, @index, @localmem, @synchronize, @uniform, @groupsize

# --- backend tag ------------------------------------------------------------

"""
    KADeviceBackend{B} <: AbstractGessoBackend

Gesso backend running KernelAbstractions kernels on the KA backend `ka::B`.
`AMDGPUBackend === KADeviceBackend{ROCBackend}` is the shipped one;
`KADeviceBackend(KA.CPU())` runs the same kernels on the host and exists for
the parity tests. Constructing an AMDGPU one without a functional ROCm device
throws `GessoError(ERR_RESOURCE_LIMIT)`.
"""
struct KADeviceBackend{B} <: AbstractGessoBackend
    ka::B
    KADeviceBackend(ka::B) where {B} = (_require_functional(ka); new{B}(ka))
end

const AMDGPUBackend = KADeviceBackend{ROCBackend}

KADeviceBackend{ROCBackend}() = KADeviceBackend(ROCBackend())

_require_functional(::Any) = nothing
function _require_functional(::ROCBackend)
    AMDGPU.functional() && return nothing
    throw(
        gesso_error(
            ERR_RESOURCE_LIMIT,
            "AMDGPUBackend: no functional ROCm device — requesting AMDGPU " *
            "without a device fails explicitly; Gesso never falls back to " *
            "CPU silently. Check the ROCm driver, or stay on CPUBackend (tier 0).";
            requested_backend=:amdgpu,
            functional=false,
        ),
    )
end

# :ka is the host test backend (or any other KA backend); only ROCm is :amdgpu.
backend_name(::KADeviceBackend) = :ka
backend_name(::KADeviceBackend{ROCBackend}) = :amdgpu
execution_tier(::KADeviceBackend) = 1   # OPTIMIZED_GENERIC (§XXI)

backend_name(::Type{<:KADeviceBackend}) = :ka
backend_name(::Type{KADeviceBackend{ROCBackend}}) = :amdgpu
execution_tier(::Type{<:KADeviceBackend}) = 1

# Grows only as ops land with a CPU-oracle test and an AMD compile test.
const AMDGPU_SUPPORTED_CAPS = Set{Symbol}([
    :rmsnorm,
    :matmul,
    :softmax,
    :embedding_lookup,
    :swiglu_ffn,
    :geglu_ffn,
    :rope_none,
    :rope_linear,
    :rope_llama3,
    :rope_partial_dim,
    :rope_interleaved,
    :attention,
    :sliding_window_attention,
])

supports(::KADeviceBackend, cap::Symbol) = cap in AMDGPU_SUPPORTED_CAPS
supports(::Type{<:KADeviceBackend}, cap::Symbol) = cap in AMDGPU_SUPPORTED_CAPS

include("amdgpu_device.jl")
include("amdgpu_kernels.jl")
include("amdgpu_ops.jl")
include("amdgpu_attention.jl")

# Bind the tag and attach to_device at load time, as GessoCUDAExt and
# GessoLavaExt do (see GessoLavaExt.__init__ for why the method goes through
# the NAME `to_device` and the binding cannot happen at top level).
function __init__()
    Core.eval(Gesso, :(const AMDGPUBackend = $AMDGPUBackend))
    if isdefined(Gesso, :to_device)
        Core.eval(Gesso, quote
            function to_device(b::$KADeviceBackend, tensors)
                $(_ka_to_device)(b, tensors)
            end
        end)
    else
        Core.eval(Gesso, :(const to_device = $to_device))
    end
    nothing
end

export AMDGPUBackend

end # module GessoAMDGPUExt
