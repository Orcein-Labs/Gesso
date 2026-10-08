# Lava operator methods + device transfer (§LXXXI item B).
#
# Same operators, more-specific methods (§CIX): these extend the EXISTING
# op! functions from src/backends.jl, dispatching on Gesso's LavaBackend ×
# semantic families × workload — exactly the CPU reference's and the CUDA
# extension's shape, on Lava.LavaArray{Float32} storage (§LXXXI math law:
# F32 on device; the CPU oracle stays F64 and parity is the declared
# atol=1e-3 for the micro models, not bit-identity).
#
# Implementation is GPUArrays broadcasting + Lava's `mul!` (Lava
# array/gemm.jl) ONLY — no KernelAbstractions kernels were needed: RoPE is
# strided views over the pairwise feature planes with a small angle table,
# causal softmax is an index-comparison mask plus dims-mapreduce max/sum.
# Fewer moving parts than the CUDA path (which needed two raw kernels) and
# no handwritten SPIR-V, no coopmat, no graphics (§LXXXI "not this sprint").
#
# Every op! ends with KA.synchronize on the KA backend: Lava dispatch is
# recorded/streamed, and the interpreter reads results back to the host
# AFTER the op returns — the sync at the op boundary is what makes any
# subsequent host readback correct (§LXXXI sync law; per-op is the
# correctness-first reading for this sprint, Phase 9 tunes).

using LinearAlgebra: mul!
using Gesso: ERR_NUMERICAL_INSTABILITY

const KA = Lava.KernelAbstractions
const KALava = Lava.LavaBackend   # the KA backend instance type (§LXXXI alias)

_lava_sync!() = KA.synchronize(KALava())

# A completed engine action leaves all primary device work synchronized.
function Gesso.Inference._engine_boundary!(::LavaBackend)
    _lava_sync!()
    _lava_sync!()
    return nothing
end

function Gesso.Inference._engine_failure(err::Lava.LavaError)
    code=err.operation in ("memory allocation","pool block allocation") ? Gesso.ERR_ALLOCATION : Gesso.ERR_RUNTIME
    return gesso_error(code,"primary Lava operation failed";cause=sprint(showerror,err),cause_type=string(typeof(err)),interrupted=false,backend=:lava)
end

# --- device guard (the interpreter does not copy) -----------------------------

# every op first checks that storage actually lives on the device: handing
# Lava ops CPU memory is ERR_INVALID_PLAN — the caller skipped to_device
# (§LXXXI: "the interpreter does not copy"; §LXX: loud, typed failure).
# Mirrors the CUDA extension's guard: only the host/device BOUNDARY matters;
# device-side views are legal (the interpreter legitimately slices device
# buffers — generate's prefill rows).
function _lava_device_storage!(op::Symbol, t)
    s = t.storage
    Gesso.Inference._storage_root(s) isa Lava.LavaArray && eltype(s) === Float32 && return s
    throw(gesso_error(ERR_INVALID_PLAN,
        "$op: LavaBackend requires its own Float32 device storage; host Array, " *
        "host views, other devices and unsupported arithmetic are rejected; call to_device explicitly";
        op=op, storage_type=string(typeof(s))))
end
Gesso.Inference._check_device_storage(op::Symbol, backend::LavaBackend, t) =
    _lava_device_storage!(op, t)

# On the tested Vulkan compiler path, the floating `isfinite` predicate
# reports NaN and Inf as finite. Classify their IEEE exponent bits instead;
# all data stays on-device and only the UInt32 reduction result is read.
_lava_nonfinite_flag(x::Float32) = UInt32((reinterpret(UInt32, x) & 0x7f800000) == 0x7f800000)
_lava_nonfinite_flag(x::Float64) = UInt32((reinterpret(UInt64, x) & 0x7ff0000000000000) == 0x7ff0000000000000)
function Gesso.Inference._all_finite(a::Lava.AnyLavaArray{T}) where {T <: Union{Float32,Float64}}
    # Explicit ordinary device temporary avoids KA's tiny BAR allocation.
    temp=Lava.LavaArray{UInt32}(undef,(max(2,2*cld(length(a),128)),))
    return Lava.AK.mapreduce(_lava_nonfinite_flag,max,a,KA.get_backend(a);
        init=UInt32(0),neutral=UInt32(0),temp,block_size=64,switch_below=0)==UInt32(0)
end

function _lava_dimreduce(f,op,a,dim,init)
    shape=ntuple(i -> i==dim ? 1 : size(a,i),ndims(a))
    prod(shape)>1 && return mapreduce(f,op,a;dims=dim)
    # Preserve Lava's native scalar Float32 sum and the legacy 1D tree.
    f===identity && op===(+) && eltype(a)===Float32 && return sum(a;dims=dim)
    result=Lava.LavaArray{typeof(init)}(undef,shape)
    temp=Lava.LavaArray{typeof(init)}(undef,(max(2,2*cld(length(a),128)),))
    value=Lava.AK.mapreduce(f,op,a,KA.get_backend(a);init,neutral=init,temp,block_size=64,switch_below=0)
    copyto!(result,1,[value],1,1)
    return result
end

# --- op bodies -----------------------------------------------------------------

function _lava_embedding_lookup!(dst, table, tokens)
    Gesso._validate_embedding_inputs(dst,table,tokens)
    tab = table.storage
    # 0-based token ids (§LXXV) → 1-based rows on the HOST (cheap), then one
    # broadcast gather on device: rows land in dst in token order.
    tok = Lava.LavaArray{Int64}(collect(Int64, tokens) .+ 1)
    dst.storage .= tab[tok, :]
    return dst
end

function _lava_rmsnorm!(dst, x, scale; eps = 1e-6)
    isfinite(eps) && eps > 0 || throw(gesso_error(ERR_INVALID_PLAN,
        "rmsnorm!: eps must be finite and positive"))
    xs = x.storage
    native_eps=eltype(xs)(eps)
    isfinite(native_eps) && native_eps>0 || throw(gesso_error(ERR_INVALID_PLAN,
        "rmsnorm!: eps must be representable, finite and positive in storage dtype"))
    d = size(xs, ndims(xs))                       # last dim is the feature dim
    length(scale.storage) == d ||
        throw(gesso_error(ERR_INVALID_PLAN, "rmsnorm!: scale length $(length(scale.storage)) ≠ feature dim $d"; op = :rmsnorm!))
    rms = sqrt.(_lava_dimreduce(abs2,+,xs,ndims(xs),zero(eltype(xs))) ./ eltype(xs)(d) .+ native_eps)
    Gesso.Inference._all_finite(rms) || throw(gesso_error(ERR_NUMERICAL_INSTABILITY,
        "rmsnorm!: nonfinite normalization denominator"))
    # scale indexes the LAST axis: trailing 1s so it broadcasts on device too
    stail = reshape(scale.storage, (ntuple(_ -> 1, ndims(xs) - 1)..., d))
    dst.storage .= (xs ./ rms) .* stail
    return dst
end

# RoPE: adjacent pairs by default; imported HF policy selects half-split
# law), Q over its head axis and K over its own (GQA-safe, Phase 3 fix).
# Broadcast cannot express the PAIR COUPLING with a single fused statement,
# but strided views over the odd/even feature planes express it exactly:
# rotate by an angle table that depends on (position, feature pair) — never
# on the head — so it broadcasts across heads (GQA for free).
#
# x1/x2 are views into the SAME array; both outputs read the ORIGINAL
# values, so the second result is staged in a temporary before either view
# is written. Angles are computed in F32 (the declared device math, §LXXXI;
# the F64 CPU oracle differs by ~1e-7 relative here, far inside atol=1e-3).
function _lava_rope!(q, k, positions; theta = 10000.0, inv_freq = nothing, interleaved = true)
    size(q.storage, 3) == size(k.storage, 3) ||
        throw(gesso_error(ERR_INVALID_PLAN, "rope!: q d_head ≠ k d_head"; op = :rope!))
    Gesso._validate_rope_inputs(q,k,positions,theta,inv_freq)
    tθ = Float64(theta)
    seq, _, d = size(q.storage)
    half = d ÷ 2
    freq = Float32.(inv_freq === nothing ? tθ .^ (-(0:2:(d-2)) ./ d) : inv_freq)
    # Match the native Float32 phase, then reduce metadata angles accurately.
    # Only derived constants are uploaded; Q/K and their rotations stay GPU.
    phase = reshape(Float32.(positions), seq, 1) .* reshape(freq, 1, half)
    reduced = Float32.(rem.(Float64.(phase), 2π))
    ang = Lava.LavaArray{Float32}(reshape(reduced, seq, 1, half))
    c = cos.(ang)
    s = sin.(ang)
    for x in (q.storage, k.storage)
        x1 = view(x, :, :, interleaved ? (1:2:d) : (1:half))                                      # odd features  (2i+1)
        x2 = view(x, :, :, interleaved ? (2:2:d) : ((half+1):d))                                      # even features (2i+2)
        n2 = x1 .* s .+ x2 .* c                                        # staged: reads ORIGINAL x1, x2
        x1 .= x1 .* c .- x2 .* s
        x2 .= n2
    end
    return q
end

# Causal softmax over (L, K) score rows, rows aligned to the LAST K keys
# (key offset = K − L; prefill L==K masks the upper triangle, a decode row
# attends to everything) — same contract as the CPU kernel, §LXXV.
# max-subtract for stability; exp(−Inf) = 0 makes the mask implicit in the
# exponential; every row has at least one unmasked key (its own), so the
# denominator is never zero.
function _lava_softmax!(dst, scores)
    s = scores.storage
    L, K = size(s)
    offset = K - L
    qi = reshape(Lava.LavaArray{Int64}(1:L), L, 1)
    kj = reshape(Lava.LavaArray{Int64}(1:K), 1, K)
    masked = ifelse.(kj .> (qi .+ offset), -Inf32, s)
    rowmax = _lava_dimreduce(identity,max,masked,2,-Inf32)
    Gesso.Inference._all_finite(rowmax) || throw(gesso_error(ERR_NUMERICAL_INSTABILITY,
        "softmax!: no finite unmasked maximum"))
    e = exp.(masked .- rowmax)
    den = _lava_dimreduce(identity,+,e,2,0f0)
    dst.storage .= e ./ den
    return dst
end

function _lava_swiglu!(dst, gate, up)
    g = gate.storage
    dst.storage .= (g ./ (1f0 .+ exp.(-g))) .* up.storage
    return dst
end

function _lava_matmul!(dst, x, w)
    # fill! first: mul!'s overwrite-at-β=0 contract must not depend on the
    # destination's prior contents (the interpreter may hand fresh `similar`
    # pages), and the seam sprint buys determinism with one cheap kernel.
    fill!(dst.storage, 0)
    mul!(dst.storage, x.storage, transpose(w.storage))   # Lava gemm unwraps the transpose
    return dst
end

# --- dispatch surface: LavaBackend × families × workload ------------------------

function embedding_lookup!(
    ::LavaBackend,
    dst::Activation,
    table::EmbeddingTable,
    tokens::AbstractVector{Int},
    ::Gesso.PrefillWorkload,
)
    _lava_device_storage!(:embedding_lookup!, dst)
    _lava_device_storage!(:embedding_lookup!, table)
    r = _lava_embedding_lookup!(dst, table, tokens)
    _lava_sync!()
    return r
end

function embedding_lookup!(
    ::LavaBackend,
    dst::Activation,
    table::EmbeddingTable,
    tokens::AbstractVector{Int},
    ::Gesso.DecodeWorkload,
)
    _lava_device_storage!(:embedding_lookup!, dst)
    _lava_device_storage!(:embedding_lookup!, table)
    r = _lava_embedding_lookup!(dst, table, tokens)
    _lava_sync!()
    return r
end

function rmsnorm!(
    ::LavaBackend,
    dst::Activation,
    x::Activation,
    scale::FrozenParameter,
    ::Gesso.PrefillWorkload;
    eps::Real = 1e-6,
)
    _lava_device_storage!(:rmsnorm!, dst)
    _lava_device_storage!(:rmsnorm!, x)
    r = _lava_rmsnorm!(dst, x, scale; eps)
    _lava_sync!()
    return r
end

function rmsnorm!(
    ::LavaBackend,
    dst::Activation,
    x::Activation,
    scale::FrozenParameter,
    ::Gesso.DecodeWorkload;
    eps::Real = 1e-6,
)
    _lava_device_storage!(:rmsnorm!, dst)
    _lava_device_storage!(:rmsnorm!, x)
    r = _lava_rmsnorm!(dst, x, scale; eps)
    _lava_sync!()
    return r
end

function rope!(
    ::LavaBackend,
    q::Activation,
    k::Activation,
    positions::AbstractVector{Int},
    ::Gesso.PrefillWorkload;
    theta::Real = 10000.0,
    inv_freq = nothing,
    interleaved::Bool = true,
)
    # Imported default, linear and Llama3 frequency metadata travels with Q/K.
    # Validate and rotate on the requested Lava storage (§LXX: no silent
    # representation change). The CPU oracle implements these policies today.
    _lava_device_storage!(:rope!, q)
    _lava_device_storage!(:rope!, k)
    r = _lava_rope!(q, k, positions; theta, inv_freq, interleaved)
    _lava_sync!()
    return r
end

function rope!(
    ::LavaBackend,
    q::Activation,
    k::Activation,
    positions::AbstractVector{Int},
    ::Gesso.DecodeWorkload;
    theta::Real = 10000.0,
    inv_freq = nothing,
    interleaved::Bool = true,
)
    # Imported default, linear and Llama3 frequency metadata travels with Q/K.
    # Validate and rotate on the requested Lava storage (§LXX: no silent
    # representation change). The CPU oracle implements these policies today.
    _lava_device_storage!(:rope!, q)
    _lava_device_storage!(:rope!, k)
    r = _lava_rope!(q, k, positions; theta, inv_freq, interleaved)
    _lava_sync!()
    return r
end

function matmul!(
    ::LavaBackend,
    dst::Activation,
    x::Activation,
    w::ProjectionWeight,
    ::Gesso.PrefillWorkload,
)
    _lava_device_storage!(:matmul!, dst)
    _lava_device_storage!(:matmul!, x)
    r = _lava_matmul!(dst, x, w)
    _lava_sync!()
    return r
end

function matmul!(
    ::LavaBackend,
    dst::Activation,
    x::Activation,
    w::ProjectionWeight,
    ::Gesso.DecodeWorkload,
)
    _lava_device_storage!(:matmul!, dst)
    _lava_device_storage!(:matmul!, x)
    r = _lava_matmul!(dst, x, w)
    _lava_sync!()
    return r
end

# tied head (§LXXV): the embedding table IS the lm_head — same function,
# no second vocabulary, no second table
function matmul!(
    ::LavaBackend,
    dst::Activation,
    x::Activation,
    w::EmbeddingTable,
    ::Gesso.PrefillWorkload,
)
    _lava_device_storage!(:matmul!, dst)
    _lava_device_storage!(:matmul!, x)
    r = _lava_matmul!(dst, x, w)
    _lava_sync!()
    return r
end

function matmul!(
    ::LavaBackend,
    dst::Activation,
    x::Activation,
    w::EmbeddingTable,
    ::Gesso.DecodeWorkload,
)
    _lava_device_storage!(:matmul!, dst)
    _lava_device_storage!(:matmul!, x)
    r = _lava_matmul!(dst, x, w)
    _lava_sync!()
    return r
end

function softmax!(
    ::LavaBackend,
    dst::TemporaryWorkspace,
    scores::TemporaryWorkspace,
    ::Gesso.PrefillWorkload,
)
    _lava_device_storage!(:softmax!, dst)
    _lava_device_storage!(:softmax!, scores)
    r = _lava_softmax!(dst, scores)
    _lava_sync!()
    return r
end

function softmax!(
    ::LavaBackend,
    dst::TemporaryWorkspace,
    scores::TemporaryWorkspace,
    ::Gesso.DecodeWorkload,
)
    _lava_device_storage!(:softmax!, dst)
    _lava_device_storage!(:softmax!, scores)
    r = _lava_softmax!(dst, scores)
    _lava_sync!()
    return r
end

function swiglu!(
    ::LavaBackend,
    dst::Activation,
    gate::Activation,
    up::Activation,
    ::Gesso.PrefillWorkload,
)
    _lava_device_storage!(:swiglu!, dst)
    _lava_device_storage!(:swiglu!, gate)
    _lava_device_storage!(:swiglu!, up)
    r = _lava_swiglu!(dst, gate, up)
    _lava_sync!()
    return r
end

function swiglu!(
    ::LavaBackend,
    dst::Activation,
    gate::Activation,
    up::Activation,
    ::Gesso.DecodeWorkload,
)
    _lava_device_storage!(:swiglu!, dst)
    _lava_device_storage!(:swiglu!, gate)
    _lava_device_storage!(:swiglu!, up)
    r = _lava_swiglu!(dst, gate, up)
    _lava_sync!()
    return r
end

# quantize! / dequantize!: NO Lava method — the generic decline still fires
# (their math is a Representation-phase concern, §LXXXI).

# --- explicit transfer (§LXXXI: "the interpreter does not copy") ---------------

"""
    to_device(::LavaBackend, tensors) -> tensors'

Copy `Array` storage to `LavaArray{Float32}` (F64→F32 conversion is part of
the lowering) and return a new named tuple of the same shape — `embedding`,
`blocks`, `lm_head`, and `final_rms` when present. NEVER mutates the CPU
tensors. Tied heads stay tied on device: `lm_head === embedding`.
"""
function _lava_to_device(tensors)
    _to_f32(s) = s isa Lava.LavaArray ? s : Lava.LavaArray{Float32}(s)
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
    _lava_sync!()
    result = (; embedding, blocks, lm_head, final_rms)
    return haskey(tensors, :rope) ? merge(result, (; rope = tensors.rope)) : result
end

# ext-local dispatch wrapper (bound as Gesso.to_device when this extension is
# the first backend extension to load; otherwise __init__ attaches the worker
# to the already-installed function object)
function to_device(::LavaBackend, tensors)
    return _lava_to_device(tensors)
end
