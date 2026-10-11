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

# Phase 10F (item A): the CORE already exposes the head split/merge/repeat,
# residual add, score scale, score-tail zero, hidden-row write and the two KV
# row copies as named STORAGE-level seams (BREADTH-0 named the first four;
# 10E named the KV copies and the contraction loops). This extension adds
# `CuArray` METHODS on those same functions — ordinary dispatch on the storage
# argument, the sanctioned extension mechanism, adding no type to the §CIX
# hierarchy (P-1 stays packeted). The generic `AbstractArray` bodies in core
# keep serving CPU, Lava and the oracle unchanged.
import Gesso.Inference:
    _attention_scores_device!,
    _attention_values_device!,
    _split_heads!,
    _merge_heads!,
    _repeat_heads!,
    _add_storage!,
    _add_rmsnorm!,
    _kv_append_storage!,
    _scale_storage!,
    _zero_tail_storage!,
    _write_hidden_row_storage!,
    _copy_row_storage!,
    _copy_rows_storage!,
    _gated_delta_rule!,
    _causal_conv1d_silu!,
    _sigmoid_mul!,
    _silu_mul!,
    _gdn_gate_params!,
    _gdn_gnorm_merge!,
    _moe_topk_swiglu!,
    _fetch_expert!,
    _shared_gate_add!,
    _fallback_zeros_factory

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

# Semantic coverage matches the CPU hybrid extras that now execute on
# CuArray{Float32}: partial / inv_freq RoPE, qk_norm (3-D rmsnorm),
# attn_output_gate (sigmoid mul), Gated DeltaNet, mRoPE-as-1D-text.
# :quantize / :dequantize stay false (Representation-phase concern).
# :moe_routing executes via fused IQ2_XS / IQ3_XXS / IQ4_XS expert GEMV
# (GpuQuantSlice when the slab fits, else H2D of one quantized expert).
# :argmax / :attn_gemm are STRATEGY gates.
const CUDA_SUPPORTED_CAPS = Set([
    :rmsnorm,
    :attention,
    :matmul,
    :softmax,
    :embedding_lookup,
    :swiglu_ffn,
    :rope_none,
    :rope_linear,
    :rope_llama3,
    :qk_norm,
    :attn_output_gate,
    :linear_attention,
    :mrope,
    :rope_partial_dim,
    :rope_interleaved,
    :moe_routing,
    :argmax,
    :attn_gemm,
])

supports(::CUDABackend, cap::Symbol) = cap in CUDA_SUPPORTED_CAPS
supports(::Type{CUDABackend}, cap::Symbol) = cap in CUDA_SUPPORTED_CAPS

# Phase 4 item B (§LXXVII): the CUDA operator methods on CuArray{Float32}
# + explicit to_device transfer. Same op names, more-specific methods (§CIX).
include("cuda_launch.jl")
include("cuda_blas.jl")
include("cuda_ops.jl")
include("cuda_quant_gemm.jl")
include("cuda_quant_dense.jl")
include("cuda_decode_small.jl")
include("cuda_hot_experts.jl")
include("cuda_attention.jl")

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
    # Resident weights keep device pressure above 0.9, so CUDA.jl's early GC
    # runs a full GC.gc(false) inside nearly every synchronize: one 3.5 ms
    # pause per decode token, and it frees nothing because the weights and
    # decode arena stay live. An OOM still collects in CUDA.jl's alloc retry.
    # CUDA.jl reads the variable on its first allocation; a user's own setting wins.
    haskey(ENV, "JULIA_CUDA_GC_EARLY") || (ENV["JULIA_CUDA_GC_EARLY"] = "false")
    Core.eval(Gesso, :(const CUDABackend = $CUDABackend))
    CUDA.CUDACore.register_reclaimable!(_HotPurge())
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
