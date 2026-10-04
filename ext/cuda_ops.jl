# CUDA operator methods + device transfer (§LXXVII item B).
#
# Same operators, more-specific methods (§CIX): these extend the EXISTING
# op! functions from src/backends.jl, dispatching on CUDABackend × semantic
# families × workload — exactly the CPU reference's shape, on
# CuArray{Float32} storage (§LXXVII math law: F32 this sprint; the CPU
# oracle stays F64 and parity is declared atol, not bit-identity).
#
# Implementation is CuArray broadcasting + CUBLAS mul!, with two small
# @cuda kernels where broadcast cannot express the op (RoPE pairwise
# rotate; causal softmax with the last-blocks-are-keys offset). No fused
# kernels, no PTX, no CUTLASS — this sprint is the SEAM (§LXXVII).
#
# Transfer is explicit (§LXXVII): to_device copies Array storage to
# CuArray{Float32} (F64→F32 conversion is part of the lowering) and returns
# a new named tuple of the same shape. It NEVER mutates the CPU tensors.

using LinearAlgebra: mul!
using Gesso: ERR_VERIFY_MISMATCH   # the all-fail / stale-winner code (§LXX, §LXXXII)

# --- device guard (the interpreter does not copy) -----------------------------

# every op first checks that storage actually lives on the device: handing
# GPU ops CPU memory is ERR_INVALID_PLAN — the caller skipped to_device
# (§LXXVII: "the interpreter does not copy"; §LXX: loud, typed failure)
function _cuda_device_storage!(op::Symbol, t)
    s = t.storage
    s === nothing && throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "$op: storage is unset — materialize (and for CUDA, to_device) " *
            "before lowering; nothing is never silently treated as data";
            op = op,
        ),
    )
    # the guard exists to catch a skipped to_device: HOST memory is an
    # Array. Device-side views (SubArray over CuArray) are legal — the
    # interpreter legitimately slices device buffers (generate's prefill
    # rows). Only the host/device BOUNDARY matters (§LXXVII).
    s isa Array || return s
    throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "$op: CUDABackend received host Array storage — " *
            "call to_device(tensors) first; the interpreter does not copy " *
            "host memory to device implicitly (§LXXVII)";
            op = op,
            storage_type = string(typeof(s)),
        ),
    )
end

# --- op bodies -----------------------------------------------------------------

function _cuda_embedding_lookup!(dst, table, tokens)
    tab = table.storage
    for (t, tok) in enumerate(tokens)
        dst.storage[t, :] .= tab[tok+1, :]        # 0-based token ids (§LXXV)
    end
    return dst
end

function _cuda_rmsnorm!(dst, x, scale; eps = 1e-6)
    xs = x.storage
    d = size(xs, ndims(xs))                       # last dim is the feature dim
    length(scale.storage) == d ||
        throw(gesso_error(ERR_INVALID_PLAN, "rmsnorm!: scale length $(length(scale.storage)) ≠ feature dim $d"; op = :rmsnorm!))
    rms = sqrt.(sum(abs2, xs; dims = ndims(xs)) ./ d .+ eps)
    # scale indexes the LAST axis: trailing 1s so it broadcasts on device too
    stail = reshape(scale.storage, (ntuple(_ -> 1, ndims(xs) - 1)..., d))
    dst.storage .= (xs ./ rms) .* stail
    return dst
end

# RoPE: pairwise rotate on the head feature dim (LLaMA-style, §LXXV math
# law), Q over its head axis and K over its own (GQA-safe, Phase 3 fix).
# One @cuda kernel — broadcast cannot express the pair coupling.
function _cuda_rope_kernel!(x, positions, theta)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x    # sequence index
    h = blockIdx().y                                          # head index
    seq, nheads, d = size(x)
    t <= seq || return
    m = Float32(positions[t])
    for i in 0:(d÷2-1)
        θv = m * theta^(-2i / d)
        c, s = cos(θv), sin(θv)
        x1 = x[t, h, 2i+1]
        x2 = x[t, h, 2i+2]
        x[t, h, 2i+1] = x1 * c - x2 * s
        x[t, h, 2i+2] = x1 * s + x2 * c
    end
    return
end

function _cuda_rope!(q, k, positions; theta = 10000.0)
    size(q.storage, 3) == size(k.storage, 3) ||
        throw(gesso_error(ERR_INVALID_PLAN, "rope!: q d_head ≠ k d_head"; op = :rope!))
    tθ = Float64(theta)
    pos_d = CuArray{Int}(positions)               # kernels never auto-transfer
    for x in (q.storage, k.storage)
        seq, nheads, d = size(x)
        nthreads = min(256, seq)
        nblocks = cld(seq, nthreads)
        # 2-D grid: x = sequence tiles, y = head (K rotates its OWN heads)
        @cuda threads = (nthreads,) blocks = (nblocks, nheads) _cuda_rope_kernel!(x, pos_d, tθ)
    end
    return q
end

# Causal softmax over (L, K) score rows, rows aligned to the LAST K keys
# (key offset = K − L; prefill L==K masks the upper triangle, a decode row
# attends to everything) — same contract as the CPU kernel, §LXXV.
function _cuda_softmax_kernel!(s)
    i = blockIdx().x                              # query row
    K = size(s, 2)
    offset = K - gridDim().x                      # K − L (grid is L wide)
    j0 = offset + i
    # mask BEFORE softmax
    for j in (j0+1):K
        s[i, j] = Float32(-Inf)
    end
    # max-subtract for stability
    row_max = Float32(-Inf)
    for j in 1:K
        row_max = max(row_max, s[i, j])
    end
    acc = Float32(0)
    for j in 1:K
        s[i, j] = exp(s[i, j] - row_max)
        acc += s[i, j]
    end
    inv = 1 / acc
    for j in 1:K
        s[i, j] *= inv
    end
    return
end

function _cuda_softmax!(dst, scores)
    L, K = size(scores.storage)
    @cuda threads = 1 blocks = L _cuda_softmax_kernel!(scores.storage)
    dst.storage .= scores.storage
    return dst
end

function _cuda_swiglu!(dst, gate, up)
    g = gate.storage
    dst.storage .= (g ./ (1f0 .+ exp.(-g))) .* up.storage
    return dst
end

function _cuda_matmul!(dst, x, w)
    mul!(dst.storage, x.storage, transpose(w.storage))   # CUBLAS
    return dst
end

# --- dispatch surface: CUDABackend × families × workload ------------------------

function embedding_lookup!(
    ::CUDABackend,
    dst::Activation,
    table::EmbeddingTable,
    tokens::AbstractVector{Int},
    ::Gesso.PrefillWorkload,
)
    _cuda_device_storage!(:embedding_lookup!, dst)
    _cuda_device_storage!(:embedding_lookup!, table)
    return _cuda_embedding_lookup!(dst, table, tokens)
end

function embedding_lookup!(
    ::CUDABackend,
    dst::Activation,
    table::EmbeddingTable,
    tokens::AbstractVector{Int},
    ::Gesso.DecodeWorkload,
)
    _cuda_device_storage!(:embedding_lookup!, dst)
    _cuda_device_storage!(:embedding_lookup!, table)
    return _cuda_embedding_lookup!(dst, table, tokens)
end

function rmsnorm!(
    ::CUDABackend,
    dst::Activation,
    x::Activation,
    scale::FrozenParameter,
    ::Gesso.PrefillWorkload;
    eps::Real = 1e-6,
)
    _cuda_device_storage!(:rmsnorm!, dst)
    _cuda_device_storage!(:rmsnorm!, x)
    return _cuda_rmsnorm!(dst, x, scale; eps)
end

function rmsnorm!(
    ::CUDABackend,
    dst::Activation,
    x::Activation,
    scale::FrozenParameter,
    ::Gesso.DecodeWorkload;
    eps::Real = 1e-6,
)
    _cuda_device_storage!(:rmsnorm!, dst)
    _cuda_device_storage!(:rmsnorm!, x)
    return _cuda_rmsnorm!(dst, x, scale; eps)
end

function rope!(
    ::CUDABackend,
    q::Activation,
    k::Activation,
    positions::AbstractVector{Int},
    ::Gesso.PrefillWorkload;
    theta::Real = 10000.0,
    inv_freq = nothing,
)
    # BREADTH-0 Pass D: a scaled positional policy has NO CUDA lowering yet.
    # It must be REFUSED, not silently run unscaled (§LXX: no silent
    # representation change). The CPU oracle implements these policies today.
    inv_freq === nothing ||
        Gesso.lowering_not_implemented(:rope!, Gesso.CUDABackend())
    _cuda_device_storage!(:rope!, q)
    _cuda_device_storage!(:rope!, k)
    return _cuda_rope!(q, k, positions; theta)
end

function rope!(
    ::CUDABackend,
    q::Activation,
    k::Activation,
    positions::AbstractVector{Int},
    ::Gesso.DecodeWorkload;
    theta::Real = 10000.0,
    inv_freq = nothing,
)
    # BREADTH-0 Pass D: a scaled positional policy has NO CUDA lowering yet.
    # It must be REFUSED, not silently run unscaled (§LXX: no silent
    # representation change). The CPU oracle implements these policies today.
    inv_freq === nothing ||
        Gesso.lowering_not_implemented(:rope!, Gesso.CUDABackend())
    _cuda_device_storage!(:rope!, q)
    _cuda_device_storage!(:rope!, k)
    return _cuda_rope!(q, k, positions; theta)
end

# --- the §LXXXII exit: the op consults Autotune --------------------------------

# Cache-key device identity (§XXVI): a hardware-stable string, not the
# process-global singleton. `CUDA.name(CUDA.device())` is the GPU's
# marketing name (e.g. "NVIDIA GeForce RTX 5060") — different cards tune
# independently, the same card re-uses its cache across processes.
_autotune_device_id() = CUDA.name(CUDA.device())

# Shape regime (§LXXXII: named buckets, not every (M, K, N)): look the live
# matmul's (K, N) up against the two fixture tables item B pinned. A (K, N)
# that matches no known pair is attributed to :toy2 — the SMALLER bucket —
# so unknown tiny shapes tune once under a conservative regime instead of
# widening the search space (and prefill/decode share a regime: the winner
# is selected per (K, N) table, not per sequence length).
function _autotune_regime(K::Integer, N::Integer)
    (K, N) in ((32, 32), (32, 16), (32, 64), (64, 32)) && return :llama_micro
    return :toy2
end

# Dispatch through the Autotune registry by WINNER NAME (a Symbol in the
# cache entry and every receipt), never by holding a function object —
# re-registration of a name must be able to replace the realization the
# winner points at (idempotent across extension reloads, §LXXXII).
# A name that is registered but somehow not dispatchable means the registry
# and the winner were written by different eras — that is a hard error
# (§LXX: no silent keep of a stale plan).
function _autotune_dispatch!(dst, x, w, regime)
    A = Gesso.Autotune
    result = A.select(
        :matmul!,
        :cuda,
        regime,
        _autotune_device_id(),
        dst,
        x,
        w,
    )
    cands = A.candidates(:matmul!, :cuda)
    i = findfirst(c -> c.name === result.winner, cands)
    i === nothing && throw(
        gesso_error(
            ERR_VERIFY_MISMATCH,
            "matmul!: autotune winner :$(result.winner) has no registered " *
            "candidate — registry and cache disagree (§LXX: no silent keep)",
            op = :matmul!,
            regime = regime,
            winner = string(result.winner),
        ),
    )
    return cands[i].run!(dst, x, w)
end

function matmul!(
    ::CUDABackend,
    dst::Activation,
    x::Activation,
    w::ProjectionWeight,
    ::Gesso.PrefillWorkload,
)
    _cuda_device_storage!(:matmul!, dst)
    _cuda_device_storage!(:matmul!, x)
    return _autotune_dispatch!(dst, x, w, _autotune_regime(size(x.storage, 2), size(w.storage, 1)))
end

function matmul!(
    ::CUDABackend,
    dst::Activation,
    x::Activation,
    w::ProjectionWeight,
    ::Gesso.DecodeWorkload,
)
    _cuda_device_storage!(:matmul!, dst)
    _cuda_device_storage!(:matmul!, x)
    return _autotune_dispatch!(dst, x, w, _autotune_regime(size(x.storage, 2), size(w.storage, 1)))
end

# tied head (§LXXV): the embedding table IS the lm_head — same function,
# no second vocabulary, no second table
function matmul!(
    ::CUDABackend,
    dst::Activation,
    x::Activation,
    w::EmbeddingTable,
    ::Gesso.PrefillWorkload,
)
    _cuda_device_storage!(:matmul!, dst)
    _cuda_device_storage!(:matmul!, x)
    return _autotune_dispatch!(dst, x, w, _autotune_regime(size(x.storage, 2), size(w.storage, 1)))
end

function matmul!(
    ::CUDABackend,
    dst::Activation,
    x::Activation,
    w::EmbeddingTable,
    ::Gesso.DecodeWorkload,
)
    _cuda_device_storage!(:matmul!, dst)
    _cuda_device_storage!(:matmul!, x)
    return _autotune_dispatch!(dst, x, w, _autotune_regime(size(x.storage, 2), size(w.storage, 1)))
end

function softmax!(
    ::CUDABackend,
    dst::TemporaryWorkspace,
    scores::TemporaryWorkspace,
    ::Gesso.PrefillWorkload,
)
    _cuda_device_storage!(:softmax!, dst)
    _cuda_device_storage!(:softmax!, scores)
    return _cuda_softmax!(dst, scores)
end

function softmax!(
    ::CUDABackend,
    dst::TemporaryWorkspace,
    scores::TemporaryWorkspace,
    ::Gesso.DecodeWorkload,
)
    _cuda_device_storage!(:softmax!, dst)
    _cuda_device_storage!(:softmax!, scores)
    return _cuda_softmax!(dst, scores)
end

function swiglu!(
    ::CUDABackend,
    dst::Activation,
    gate::Activation,
    up::Activation,
    ::Gesso.PrefillWorkload,
)
    _cuda_device_storage!(:swiglu!, dst)
    _cuda_device_storage!(:swiglu!, gate)
    _cuda_device_storage!(:swiglu!, up)
    return _cuda_swiglu!(dst, gate, up)
end

function swiglu!(
    ::CUDABackend,
    dst::Activation,
    gate::Activation,
    up::Activation,
    ::Gesso.DecodeWorkload,
)
    _cuda_device_storage!(:swiglu!, dst)
    _cuda_device_storage!(:swiglu!, gate)
    _cuda_device_storage!(:swiglu!, up)
    return _cuda_swiglu!(dst, gate, up)
end

# quantize! / dequantize!: NO CUDA method — the generic decline still fires
# (their math is a Representation-phase concern, §LXXVII).

# --- Autotune candidates for :matmul! (§LXXXII item B) -------------------------
#
# Two legal realizations of the SAME semantics on CuArray{Float32}:
#
#   :cublas_mul   the Phase 4 implementation — fill! + CUBLAS `mul!` with the
#                 transpose unwrapped by the GEMM call.
#   :generic_mul  GPUArrays-broadcast outer-product accumulation:
#                 dst += x[:,k] ⊗ w[:,k] for k in 1:K. Same contract, same
#                 storage, no new kernel language — deliberately the loser on
#                 any real shape. §LXXXII: "If :cublas_mul wins, that is the
#                 expected and acceptable result. The point is that selection
#                 happened."
#
# Phase 10 C note: the Session ATTENTION contraction (QKᵀ and PV over the
# gathered scratch) is NOT registered here — it is head-structured (3-D
# storages), not matmul!-shaped under this two-candidate (dst, x, w) 2-D
# contract, so it lives in Session backend-dispatched (its capability gate
# is supports(::CUDABackend, :attn_gemm)); every matmul! on the path still
# consults the search above (§LXXXII unchanged).
#
# Both are gated against the CPU F64 oracle at the EXISTING Phase 4 CUDA op
# atol (1e-2, the op-level compare in test_cuda_ops.jl — no third atol is
# invented). The gate runs the candidate into a scratch (never the live dst)
# and compares in F64; a NaN anywhere fails the gate (NaN <= atol is false).
# Registration happens at __init__ (runtime state, never precompile); the
# candidates themselves never run unless a search consults them.

function _cuda_generic_matmul!(dst, x, w)
    fill!(dst.storage, 0)
    xs, ws = x.storage, w.storage
    for k in 1:size(xs, 2)
        dst.storage .+= view(xs, :, k) .* transpose(view(ws, :, k))
    end
    return dst
end

# the correctness gate both candidates share: F64 host oracle vs a scratch
# run of `impl`, at the Phase 4 CUDA op atol
function _cuda_matmul_gate(impl, dst, x, w; atol = 1e-2)
    ref = Float64.(Array(x.storage)) * Float64.(Array(w.storage))'
    sc = typeof(dst)(; shape = dst.shape, storage = similar(dst.storage))
    impl(sc, x, w)
    return maximum(abs, Float64.(Array(sc.storage)) .- ref) <= atol
end

function _autotune_register!()
    A = Gesso.Autotune
    A.register!(
        :matmul!,
        :cuda,
        A.Candidate(
            :cublas_mul,
            (dst, x, w) -> _cuda_matmul!(dst, x, w),
            (dst, x, w) -> _cuda_matmul_gate(_cuda_matmul!, dst, x, w),
        ),
    )
    A.register!(
        :matmul!,
        :cuda,
        A.Candidate(
            :generic_mul,
            (dst, x, w) -> _cuda_generic_matmul!(dst, x, w),
            (dst, x, w) -> _cuda_matmul_gate(_cuda_generic_matmul!, dst, x, w),
        ),
    )
    return nothing
end

# --- explicit transfer (§LXXVII: "the interpreter does not copy") ---------------

"""
    to_device(::CUDABackend, tensors) -> tensors'

Copy `Array` storage to `CuArray{Float32}` (F64→F32 conversion is part of
the lowering) and return a new named tuple of the same shape — `embedding`,
`blocks`, `lm_head`, and `final_rms` when present. NEVER mutates the CPU
tensors. Tied heads stay tied on device: `lm_head === embedding`.
"""
function _to_device_cuda(tensors)
    _to_f32(s) = s isa CuArray ? s : CuArray{Float32}(s)
    # tensors in the map carry (shape, storage); blocks are NAMED TUPLES of
    # tensors, so their fields convert field-wise
    _conv_t(t) = begin
        t.storage === nothing && return t            # unset storage stays unset
        typeof(t)(; shape = t.shape, storage = _to_f32(t.storage))
    end
    _conv_blk(b) = NamedTuple{keys(b)}(map(_conv_t, values(b)))
    embedding = _conv_t(tensors.embedding)
    blocks = [_conv_blk(b) for b in tensors.blocks]
    lm_head = tensors.lm_head === tensors.embedding ? embedding : _conv_t(tensors.lm_head)
    fr = haskey(tensors, :final_rms) ? tensors.final_rms : nothing
    final_rms = fr === nothing ? nothing : _conv_t(fr)
    return (; embedding, blocks, lm_head, final_rms)
end

# ext-local dispatch wrapper (bound as Gesso.to_device when this extension is
# the first backend extension to load; otherwise __init__ attaches the worker
# to the already-installed function object — see GessoLavaExt for the full
# cross-extension rationale and the empirical note on why the definition
# must go through the NAME in the eval target's scope)
function to_device(::CUDABackend, tensors)
    return _to_device_cuda(tensors)
end
