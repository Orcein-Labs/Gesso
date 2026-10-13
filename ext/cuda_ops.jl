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

# cuBLAS runs in device pointer mode, so a Number alpha or beta becomes a fresh
# CuRef per call: a pool allocation (which can run CUDACore's early GC), a
# pageable 4-byte upload and a finalizer. About 1200 of those per 35B token, so
# the constants are kept on the device, one box per context and value.
const _CU_SCALARS = Dict{Tuple{CUDA.CuContext, Float32}, CUDA.CUDACore.CuRefValue{Float32}}()
const _CU_SCALARS_LOCK = ReentrantLock()
_cu_scalar(x::Number) = x
function _cu_scalar(x::Float32)
    key = (CUDA.context(), x)
    @lock _CU_SCALARS_LOCK get!(_CU_SCALARS, key) do
        r = CUDA.CuRef{Float32}(x)
        # the box is shared across streams, so its upload must land first
        CUDA.synchronize()
        r
    end
end

# --- device guard (the interpreter does not copy) -----------------------------

# every op first checks that storage actually lives on the device: handing
# GPU ops CPU memory is ERR_INVALID_PLAN — the caller skipped to_device
# (§LXXVII: "the interpreter does not copy"; §LXX: loud, typed failure)
function _cuda_device_storage!(op::Symbol, t)
    return _cuda_checked_storage(op,t.storage)
end
function _cuda_checked_storage(op::Symbol, s)
    Gesso.Inference._storage_root(s) isa CUDA.CuArray && eltype(s) === Float32 && return s
    throw(gesso_error(ERR_INVALID_PLAN,
        "$op: CUDABackend requires its own Float32 device storage; host Array, " *
        "host views, other devices and unsupported arithmetic are rejected; call to_device explicitly";
        op=op, storage_type=string(typeof(s))))
end
function Gesso.Inference._check_device_storage(op::Symbol, backend::CUDABackend, t)
    s = t.storage
    (s isa Gesso.Inference.HostOffload ||
     s isa Gesso.Inference.GgufQuantSlice ||
     s isa Gesso.Inference.GpuQuantSlice ||
     s isa Gesso.Inference.CachedExpertBank) &&
        return nothing
    s isa CuArray{Float16} && return nothing   # device F16 weight (matmul / table)
    s isa GpuDenseQuant && return nothing      # native dense quant weight (matmul)
    # Untied GGUF embedding gathered on the host. Projections are excluded so
    # a QuantMatrix the device conversion missed still fails here.
    t isa Gesso.EmbeddingTable && s isa Gesso.Inference.QuantMatrix && return nothing
    _cuda_device_storage!(op, t)
    return nothing
end

# Small HostOffload weights H2D for an op. Embedding / lm_head / expert
# slabs have dedicated host paths and must not go through this helper.
function _cuda_materialize_small(s)
    s isa CuArray && return s
    s isa Gesso.Inference.HostOffload && return CuArray{Float32}(s.data)
    return s
end

# --- op bodies -----------------------------------------------------------------

function _cuda_embedding_lookup!(dst, table, tokens)
    Gesso._validate_embedding_inputs(dst,table,tokens)
    tab = table.storage
    for (t, tok) in enumerate(tokens)
        d = size(dst.storage, 2)                  # 0-based token ids (§LXXV)
        d == 0 && continue
        nthreads = min(256, d)
        @fastcuda threads = (nthreads,) blocks = (cld(d, nthreads),) _cuda_embed_kernel!(
            dst.storage, tab, t, tok + 1)
    end
    return dst
end

# dst[t, f] = tab[tok0, f] — a row copy. `tokens` stays a HOST vector (the
# kernel never auto-transfers it, §LXXVII), so the loop over rows stays on the
# host and only the row copy moves to the device.
function _cuda_embed_kernel!(dst2, tab, t::Int, tok0::Int)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j <= size(dst2, 2) || return
    dst2[t, j] = tab[tok0, j]
    return
end

function _cuda_rmsnorm!(dst, x, scale; eps=1e-6)
    ss = _cuda_materialize_small(scale.storage)
    _cuda_rmsnorm_storage!(dst.storage, x.storage, ss, eps)
    return dst
end

# q and k norms alternate row counts every layer, so one resizing scratch slot
# would reallocate on every call; keep one buffer per row count instead. Decode
# row counts are head counts; prefill's grow with the prompt and are not kept.
const _RMSNORM_SUMSQ = Dict{Tuple{CUDA.CuContext, Int}, CuArray{Float32, 2}}()
function _rmsnorm_sumsq(rows::Int)
    rows <= 256 || return CuArray{Float32}(undef, rows, 1)
    get!(() -> CUDA.zeros(Float32, rows, 1), _RMSNORM_SUMSQ, (CUDA.context(), rows))
end

# A reshape or contiguous view of a CuArray holds its own reference to the
# parent's memory, so a prefill temporary that was reshaped stays allocated
# after its owner frees it, until the GC finalizes the derived array. Drop the
# derived reference as soon as its last kernel is queued.
_drop_derived!(a, parent) = (a === parent || CUDA.unsafe_free!(a); nothing)

function _cuda_rmsnorm_storage!(dsts::AbstractArray, xs::AbstractArray, scales::AbstractArray, eps)
    native_eps=eltype(xs)(eps)
    isfinite(native_eps) && native_eps>0 || throw(gesso_error(ERR_INVALID_PLAN,
        "rmsnorm!: eps must be representable, finite and positive in storage dtype"))
    nd = ndims(xs)
    d = size(xs, nd)                              # last dim is the feature dim
    length(scales) == d ||
        throw(gesso_error(ERR_INVALID_PLAN, "rmsnorm!: scale length $(length(scales)) ≠ feature dim $d"; op = :rmsnorm!))
    if nd == 2 && size(xs, 1) == 1
        @fastcuda threads=256 blocks=1 _cuda_rmsnorm_row_kernel!(dsts, xs, scales, eps, d)
        return dsts
    end
    # Retain the legacy mixed-precision normalization to meet its fixed
    # small-fixture accuracy budget. Comparison receipts name this explicitly.
    # The per-head q/k norm arrives 3-D (seq, heads, d_head); folding the
    # leading axes into rows reduces the same elements in the same order, and
    # with a Float64 eps the apply kernel's arithmetic is the broadcast's, so
    # both ranks share the allocation-free path.
    if (nd == 2 || (nd == 3 && eps isa Float64)) && xs isa CuArray && dsts isa CuArray
        d == 0 && return dsts
        rows = length(xs) ÷ d
        x2 = reshape(xs, rows, d)
        d2 = reshape(dsts, rows, d)
        if d <= 256 && rows <= 8192
            @fastcuda threads = 32cld(d, 32) blocks = rows _cuda_rmsnorm_rows_kernel!(
                d2, x2, scales, eps, d)
        else
            ss = _rmsnorm_sumsq(rows)
            sum!(abs2, ss, x2)
            nthreads = min(256, d)
            @fastcuda threads = (nthreads,) blocks = (rows,) _cuda_rmsnorm_apply_kernel!(
                d2, x2, ss, scales, eps, rows, d)
            rows > 256 && CUDA.unsafe_free!(ss)
        end
        _drop_derived!(x2, xs)
        _drop_derived!(d2, dsts)
        return dsts
    end
    rms = sqrt.(sum(abs2, xs; dims = nd) ./ d .+ eps)
    if nd == 2
        rows = size(dsts, 1)
        d == 0 && return dsts
        nthreads = min(256, d)
        @fastcuda threads = (nthreads,) blocks = (rows,) _cuda_rmsnorm_apply_kernel!(
            dsts, xs, rms, scales, rows, d)
        return dsts
    end
    # generic rank: the previous chain, unchanged
    stail = reshape(scales, (ntuple(_ -> 1, nd - 1)..., d))
    dsts .= (xs ./ rms) .* stail
    return dsts
end

@inline function _warp_sum_f32(v::Float32)
    off = 0x00000001
    while off < 0x00000020
        v += CUDA.shfl_down_sync(0xffffffff, v, off)
        off <<= 1
    end
    return v
end

# Sum of squares and apply in one launch, one block per row. The reduction
# reproduces the order of CUDACore's single-block shuffle mapreduce, which
# sum!(abs2, ss, x2) takes when threads = nextwarp(d) and the row count sits
# below its serial-kernel threshold, so the result is bit-identical to the
# two-launch path (checked over seven shapes x 20 magnitudes). That path also
# queries occupancy on every call; this one is a plain launch.
function _cuda_rmsnorm_rows_kernel!(dst2, xs2, scales, eps, d::Int)
    r = blockIdx().x
    t = Int(threadIdx().x)
    sh = CUDA.CuStaticSharedArray(Float32, 33)
    v = 0.0f0
    @inbounds begin
        j = t
        while j <= d
            v += abs2(xs2[r, j])
            j += blockDim().x
        end
        wid, lane = fldmod1(t, 32)
        v = _warp_sum_f32(v)
        lane == 1 && (sh[wid] = v)
        CUDA.sync_threads()
        v = t <= fld1(Int(blockDim().x), 32) ? sh[lane] : 0.0f0
        if wid == 1
            v = _warp_sum_f32(v)
            t == 1 && (sh[33] = v)
        end
        CUDA.sync_threads()
    end
    _cuda_rmsnorm_apply_row!(dst2, xs2, Float64(sqrt(sh[33] / d + eps)), scales, r, d)
    return
end

# Single-row decode normalization: one block of 256 threads (launch width is
# fixed at 256 for the shared-memory tree). Each thread sums a strided share of
# the squares, the block reduces them in shared memory, then every thread
# scales its share. One launch per call; the earlier single-thread loop walked
# the feature axis serially in global memory (~1 ms per call at d = 2048).
function _cuda_rmsnorm_row_kernel!(dst, xs, scales, eps, d::Int)
    tid = threadIdx().x
    part = CUDA.@cuStaticSharedMem(Float32, 256)
    acc = 0.0f0
    j = tid
    while j <= d
        acc += abs2(xs[1, j])
        j += blockDim().x
    end
    part[tid] = acc
    CUDA.sync_threads()
    stride = 128
    while stride > 0
        if tid <= stride
            part[tid] += part[tid + stride]
        end
        CUDA.sync_threads()
        stride >>= 1
    end
    rms = sqrt(Float64(part[1]) / d + eps)
    j = tid
    while j <= d
        dst[1, j] = (xs[1, j] / rms) * scales[j]
        j += blockDim().x
    end
    return
end

# The row kernel above with the residual add folded into its first pass.
# Each thread stores `h + src` and reads back only the columns it wrote, so
# the sum and the norm are the same bits as the add followed by the norm.
function _cuda_add_rmsnorm_row_kernel!(dst, hs, src, scales, eps, d::Int)
    tid = threadIdx().x
    part = CUDA.@cuStaticSharedMem(Float32, 256)
    acc = 0.0f0
    j = tid
    while j <= d
        x = hs[1, j] + src[j]
        hs[1, j] = x
        acc += abs2(x)
        j += blockDim().x
    end
    part[tid] = acc
    CUDA.sync_threads()
    stride = 128
    while stride > 0
        if tid <= stride
            part[tid] += part[tid + stride]
        end
        CUDA.sync_threads()
        stride >>= 1
    end
    rms = sqrt(Float64(part[1]) / d + eps)
    j = tid
    while j <= d
        dst[1, j] = (hs[1, j] / rms) * scales[j]
        j += blockDim().x
    end
    return
end

function _add_rmsnorm!(
    be::CUDABackend,
    dst::Activation,
    h::Activation,
    src::CuArray,
    scale::FrozenParameter,
    wl::DecodeWorkload;
    eps::Real = 1e-6,
)
    hs = h.storage
    ds = dst.storage
    if hs isa CuArray && ds isa CuArray && ndims(hs) == 2 && size(hs, 1) == 1 && length(src) == length(hs)
        ss = _cuda_materialize_small(scale.storage)
        d = size(hs, 2)
        native_eps = eltype(hs)(eps)
        isfinite(native_eps) && native_eps > 0 || throw(gesso_error(ERR_INVALID_PLAN,
            "rmsnorm!: eps must be representable, finite and positive in storage dtype"))
        length(ss) == d ||
            throw(gesso_error(ERR_INVALID_PLAN, "rmsnorm!: scale length $(length(ss)) ≠ feature dim $d"; op = :rmsnorm!))
        @fastcuda threads=256 blocks=1 _cuda_add_rmsnorm_row_kernel!(ds, hs, src, ss, eps, d)
        return dst
    end
    _add_storage!(hs, src)
    return rmsnorm!(be, dst, h, scale, wl; eps)
end

# ONE BLOCK PER ROW, one thread per feature. `sum(abs2, xs; dims=ndims(xs))`
# reduces the FEATURE axis, so `rms` is (rows, 1) and is read as `rms[r, 1]`;
# the scale is reshaped to trailing 1s so it broadcasts along the LAST axis,
# and is read as `scale1[f]`.
#
# Grid mapping is deliberately the same shape as `_cuda_softmax_kernel!`
# (which has always been right): the row comes from `blockIdx` and the feature
# from `threadIdx`. An earlier version indexed the destination LINEARLY and
# recovered (r, f) with `÷` and `%`; on the device that recovery came back
# PERMUTED — every stored value was a correct `(x/rms)*scale` product, but
# attached to the wrong cell (measured: 3x2 input, 2 of 6 cells correct).
# Every other kernel in this file indexes multi-dimensional arrays
# explicitly for the same reason.
function _cuda_rmsnorm_apply_kernel!(dst2, xs2, rms, scale1, rows::Int, d::Int)
    r = blockIdx().x
    r > rows && return
    _cuda_rmsnorm_apply_row!(dst2, xs2, Float64(rms[r, 1]), scale1, r, d)
    return
end

# Same, from the row's sum of squares: `sumsq / d + eps` is the expression the
# host broadcast evaluated per element, in the same types.
function _cuda_rmsnorm_apply_kernel!(dst2, xs2, sumsq, scale1, eps, rows::Int, d::Int)
    r = blockIdx().x
    r > rows && return
    _cuda_rmsnorm_apply_row!(dst2, xs2, Float64(sqrt(sumsq[r, 1] / d + eps)), scale1, r, d)
    return
end

@inline function _cuda_rmsnorm_apply_row!(dst2, xs2, nrm::Float64, scale1, r, d::Int)
    f = threadIdx().x
    while f <= d
        dst2[r, f] = Float32((Float64(xs2[r, f]) / nrm) * Float64(scale1[f]))
        f += blockDim().x
    end
    return
end

# RoPE: adjacent pairs by default; imported HF policy selects half-split
# law), Q over its head axis and K over its own (GQA-safe, Phase 3 fix).
# One @cuda kernel — broadcast cannot express the pair coupling.
function _cuda_rope_kernel!(x, positions, theta, interleaved::Bool, rd::Int, use_inv::Bool, inv_freq)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x    # sequence index
    h = blockIdx().y                                          # head index
    seq, nheads, d = size(x)
    t <= seq || return
    m = Float32(positions[t])
    n = rd ÷ 2
    for i in 0:(n - 1)
        θv = use_inv ? m * inv_freq[i + 1] : m * theta^(-2i / rd)
        c, s = cos(θv), sin(θv)
        a, b = interleaved ? (2i + 1, 2i + 2) : (i + 1, i + 1 + n)
        x1 = x[t, h, a]
        x2 = x[t, h, b]
        x[t, h, a] = x1 * c - x2 * s
        x[t, h, b] = x1 * s + x2 * c
    end
    return
end

function _cuda_rope!(
    q,
    k,
    positions;
    theta = 10000.0,
    interleaved = true,
    rotary_dim = 0,
    inv_freq = nothing,
)
    _cuda_rope_storage!(q.storage, k.storage, positions, theta, interleaved, rotary_dim, inv_freq)
    return q
end

struct _OnePos
    p::Int
end
Base.getindex(o::_OnePos, ::Integer) = o.p

# The frequency table is fixed per model, but arrives as a host vector on every
# call; upload each distinct table once per context.
const _ROPE_FREQ = Dict{Tuple{CUDA.CuContext, Union{Nothing, Vector{Float64}}}, CuArray{Float32, 1}}()
const _ROPE_FREQ_LOCK = ReentrantLock()
function _rope_freq_device(inv_freq)
    ctx = CUDA.context()
    host = inv_freq === nothing || inv_freq isa Vector{Float64} ? inv_freq : Vector{Float64}(inv_freq)
    @lock _ROPE_FREQ_LOCK begin
        dev = get(_ROPE_FREQ, (ctx, host), nothing)
        dev === nothing || return dev
        dev = host === nothing ? CUDA.zeros(Float32, 1) : CuArray{Float32}(Float32.(host))
        # the key is a private copy: the caller's vector may be reused or mutated
        _ROPE_FREQ[(ctx, host === nothing ? nothing : copy(host))] = dev
        return dev
    end
end

# Keep launch arguments concrete across semantic tensors' erased storage fields.
function _cuda_rope_storage!(
    qs::AbstractArray,
    ks::AbstractArray,
    positions,
    theta,
    interleaved,
    rotary_dim,
    inv_freq,
)
    size(qs, 3) == size(ks, 3) ||
        throw(gesso_error(ERR_INVALID_PLAN, "rope!: q d_head ≠ k d_head"; op = :rope!))
    tθ = Float64(theta)
    d = size(qs, 3)
    rd = rotary_dim == 0 ? Int(d) : Int(rotary_dim)
    # kernels never auto-transfer; a decode step's single position rides in
    # the launch parameters instead of a fresh device array
    pos_d = length(positions) == 1 ? _OnePos(Int(first(positions))) : CuArray{Int}(positions)
    use_inv = inv_freq !== nothing
    freq_d = _rope_freq_device(inv_freq)
    for x in (qs, ks)
        seq, nheads, _ = size(x)
        nthreads = min(256, seq)
        nblocks = cld(seq, nthreads)
        # 2-D grid: x = sequence tiles, y = head (K rotates its OWN heads)
        @fastcuda threads = (nthreads,) blocks = (nblocks, nheads) _cuda_rope_kernel!(
            x, pos_d, tθ, interleaved, rd, use_inv, freq_d)
    end
    pos_d isa CuArray && CUDA.unsafe_free!(pos_d)
    return nothing
end

# Causal softmax over (L, K) score rows, rows aligned to the LAST K keys
# (key offset = K − L; prefill L==K masks the upper triangle, a decode row
# attends to everything) — same contract as the CPU kernel, §LXXV.
function _cuda_softmax_kernel!(s, d)
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
    # write the destination IN THE SAME PASS (10F): `dst.storage .=
    # scores.storage` was a second broadcast over the whole buffer, and the
    # values copied are the ones already in hand here.
    for j in 1:K
        s[i, j] *= inv
        d[i, j] = s[i, j]
    end
    return
end

function _cuda_softmax_storage!(scores::AbstractArray, dst::AbstractArray)
    L, K = size(scores)
    L == 0 && return dst
    @fastcuda threads = 1 blocks = L _cuda_softmax_kernel!(scores, dst)
    return dst
end
function _cuda_softmax!(dst, scores)
    _cuda_softmax_storage!(scores.storage, dst.storage)
    return dst
end

function _cuda_swiglu_kernel!(dst, g, u)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= length(dst) || return
    gv = g[i]
    dst[i] = (gv / (1f0 + exp(-gv))) * u[i]
    return
end

function _cuda_swiglu!(dst, gate, up)
    n = length(dst.storage)
    n == 0 && return dst
    @fastcuda threads = (min(256, n),) blocks = (cld(n, 256),) _cuda_swiglu_kernel!(
        dst.storage, gate.storage, up.storage)
    return dst
end

function _cuda_matmul_storage!(dst::AbstractArray, x::AbstractArray, w::AbstractArray)
    _xwt_ok(dst, x, w) && return _cuda_xwt!(dst, x, w)
    _blas_gemm!('N', 'T', _cu_scalar(one(eltype(dst))), x, w, _cu_scalar(zero(eltype(dst))), dst)
    return dst
end

function _cuda_matmul!(dst, x, w)
    _cuda_matmul_storage!(dst.storage, x.storage, w.storage)
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
    table.storage isa Gesso.Inference.HostOffload &&
        return _cuda_host_offload_embed!(dst, table, tokens)
    table.storage isa Gesso.Inference.QuantMatrix &&
        return _cuda_host_quant_embed!(dst, table, tokens)
    _cuda_embed_table_check!(table)
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
    table.storage isa Gesso.Inference.HostOffload &&
        return _cuda_host_offload_embed!(dst, table, tokens)
    table.storage isa Gesso.Inference.QuantMatrix &&
        return _cuda_host_quant_embed!(dst, table, tokens)
    _cuda_embed_table_check!(table)
    return _cuda_embedding_lookup!(dst, table, tokens)
end

function _cuda_host_offload_embed!(dst, table, tokens)
    Gesso._validate_embedding_inputs(dst, table, tokens)
    copyto!(dst.storage, _host_embed_rows(table.storage.data, tokens))
    return dst
end

function _cuda_host_quant_embed!(dst, table, tokens)
    Gesso._validate_embedding_inputs(dst, table, tokens)
    q = table.storage
    host = Matrix{Float32}(undef, length(tokens), q.ne0)
    Gesso.Inference._quant_embedding_rows!(host, q, tokens)
    # prefill writes through a row view of h, and copyto! into a view stages
    # the host rows in a device array it leaves to the GC
    staged = CuArray(host)
    copyto!(dst.storage, staged)
    CUDA.unsafe_free!(staged)
    return dst
end

# EmbeddingTable.storage is untyped, so the gather runs behind this barrier:
# inline, every element read boxed and allocated, ~6k allocations per token.
function _host_embed_rows(tab::AbstractMatrix, tokens)
    d = size(tab, 2)
    host = Matrix{Float32}(undef, length(tokens), d)
    @inbounds for (t, tok) in enumerate(tokens), j in 1:d
        host[t, j] = Float32(tab[tok + 1, j])
    end
    return host
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
    interleaved::Bool = true,
    rotary_dim::Integer = 0,
)
    # Partial rotary and an explicit inv_freq table (linear / Llama3 / Qwen3.5
    # mRoPE-as-1D-text) rotate on device. Frequencies stay host metadata and
    # are uploaded; Q/K stay on the device (§LXX: no silent unscaled run).
    _cuda_device_storage!(:rope!, q)
    _cuda_device_storage!(:rope!, k)
    Gesso._validate_rope_inputs(q, k, positions, theta, inv_freq, rotary_dim)
    return _cuda_rope!(q, k, positions; theta, interleaved, rotary_dim, inv_freq)
end

function rope!(
    ::CUDABackend,
    q::Activation,
    k::Activation,
    positions::AbstractVector{Int},
    ::Gesso.DecodeWorkload;
    theta::Real = 10000.0,
    inv_freq = nothing,
    interleaved::Bool = true,
    rotary_dim::Integer = 0,
)
    _cuda_device_storage!(:rope!, q)
    _cuda_device_storage!(:rope!, k)
    Gesso._validate_rope_inputs(q, k, positions, theta, inv_freq, rotary_dim)
    return _cuda_rope!(q, k, positions; theta, interleaved, rotary_dim, inv_freq)
end

# --- the §LXXXII exit: the op consults Autotune --------------------------------

# Cache-key device identity (§XXVI): a hardware-stable string, not the
# process-global singleton. `CUDA.name(CUDA.device())` is the GPU's
# marketing name (e.g. "NVIDIA GeForce RTX 5060") — different cards tune
# independently, the same card re-uses its cache across processes.
#
# 10F: `CUDA.name` builds a FRESH String on every call, and the decode
# consults this once per autotuned `matmul!` — 45 times per token on
# llama_micro (measured: 5,280 B/token, `Profile.Allocs` at
# cuda_ops.jl:253). The memo is keyed on the ACTIVE DEVICE OBJECT, so a card
# swap misses and re-reads the name; it is a cache of a fact about the
# hardware, not a cache of a decision.
const _DEVICE_ID_CACHE = Ref{Any}(nothing)

function _autotune_device_id()
    dev = CUDA.device()
    cached = _DEVICE_ID_CACHE[]
    if cached !== nothing && cached[1] === dev
        return cached[2]::String
    end
    id = CUDA.name(dev)
    _DEVICE_ID_CACHE[] = (dev, id)
    return id
end

# --- Phase 10F (item A): the storage seams, as CuArray kernels -----------------
#
# Every body below is a pure COPY or the SAME scalar expression the broadcast
# chain evaluated, so nothing here changes a single stored bit (§LXXVII
# declares CPU/CUDA parity at atol; these bodies do not even spend that).
#
# Why kernels and not `copyto!`: measured on the RTX 5060 (2026-10-04),
# `copyto!(view(CuArray3,:,h,:), view(CuArray2,:,a:b))` allocates 3,152 B —
# barely under the 3,568 B of the `.=` broadcast — because both sides are
# `SubArray`s and the generic path degrades. One `@cuda` launch allocates
# 688 B, five times less, and is a real kernel rather than a scalar loop.
#
# Grid conventions: `x` is the linear element / sequence index (CUDA.jl grids
# are x-major, so this is the coalesced axis), `y` and `z` are head and
# feature. Every launch guards its own bound; none of them writes outside the
# destination.

function _cuda_split_kernel!(dst3, src2, d_head::Int)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    h = blockIdx().y
    r > size(dst3, 1) && return
    coloff = (h - 1) * d_head
    for f in 1:d_head
        dst3[r, h, f] = src2[r, coloff + f]
    end
    return
end

function _split_heads!(dst::CuArray, src::AbstractArray, n_heads, d_head)
    rows = size(dst, 1)
    rows == 0 && return dst
    nthreads = min(256, rows)
    @fastcuda threads = (nthreads,) blocks = (cld(rows, nthreads), n_heads) _cuda_split_kernel!(
        dst, src, Int(d_head))
    return dst
end

function _cuda_merge_kernel!(dst2, src3, d_head::Int)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    h = blockIdx().y
    r > size(dst2, 1) && return
    coloff = (h - 1) * d_head
    for f in 1:d_head
        dst2[r, coloff + f] = src3[r, h, f]
    end
    return
end

function _merge_heads!(dst::CuArray, src::CuArray, n_heads, d_head)
    rows = size(dst, 1)
    rows == 0 && return dst
    nthreads = min(256, rows)
    @fastcuda threads = (nthreads,) blocks = (cld(rows, nthreads), n_heads) _cuda_merge_kernel!(
        dst, src, Int(d_head))
    return dst
end

# kv head `sh` fills query-head slots (sh-1)*group+1 … sh*group, rows 1:K
function _cuda_repeat_kernel!(dst3, src3, group::Int, K::Int)
    r = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    sh = blockIdx().y
    r > K && return
    for g in 1:group, f in 1:size(dst3, 3)
        dst3[r, (sh - 1) * group + g, f] = src3[r, sh, f]
    end
    return
end

function _repeat_heads!(dst::CuArray, src::CuArray, group::Int, K::Int)
    group == 1 && return dst
    K == 0 && return dst
    nkv = size(src, 2)
    nthreads = min(256, K)
    @fastcuda threads = (nthreads,) blocks = (cld(K, nthreads), nkv) _cuda_repeat_kernel!(
        dst, src, group, K)
    return dst
end

# one KV token-row appended at 1-based row `i`; `src` is the (n_kv_heads,
# d_head) head slice the engine hands in
function _cuda_copy_row_kernel!(dst3, src2, i::Int)
    h = blockIdx().x
    f = blockIdx().y
    dst3[i, h, f] = src2[h, f]
    return
end

function _copy_row_storage!(dst::CuArray, src::AbstractArray, i::Int)
    @fastcuda threads = (1,) blocks = (size(dst, 2), size(dst, 3)) _cuda_copy_row_kernel!(dst, src, i)
    return dst
end

# decode's K and V row append at 1-based cache row `row`: one block per KV
# head, threads stride the head's features
function _cuda_kv_append_kernel!(kc, vc, kh, vh, row::Int)
    h = blockIdx().x
    f = threadIdx().x
    while f <= size(kc, 3)
        kc[row, h, f] = kh[1, h, f]
        vc[row, h, f] = vh[1, h, f]
        f += blockDim().x
    end
    return
end

function _kv_append_storage!(kc::CuArray, vc::CuArray, kh::CuArray, vh::CuArray, row::Int)
    @fastcuda threads = (min(256, size(kc, 3)),) blocks = (size(kc, 2),) _cuda_kv_append_kernel!(kc, vc, kh, vh, row)
    return nothing
end

# `take` token-rows from a page's origin into dest at 1-based row `r1`
function _cuda_copy_rows_kernel!(dst3, src3, r1::Int, take::Int)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    h = blockIdx().y
    f = blockIdx().z
    t > take && return
    dst3[r1 + t - 1, h, f] = src3[t, h, f]
    return
end

function _copy_rows_storage!(dest::CuArray, src::CuArray, r1::Int, take::Int)
    take <= 0 && return dest
    nthreads = min(256, take)
    @fastcuda threads = (nthreads,) blocks = (cld(take, nthreads), size(dest, 2), size(dest, 3)) _cuda_copy_rows_kernel!(
        dest, src, r1, take)
    return dest
end

# dst[i] = dst[i] + src[i] — the interpreter's residual add (§LXXV: still
# storage addition; there is no `add!` operator and this does not add one)
function _cuda_add_kernel!(dst, src)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= length(dst) || return
    dst[i] += src[i]
    return
end

function _add_storage!(dst::CuArray, src::CuArray)
    n = length(dst)
    n == 0 && return dst
    @fastcuda threads = (min(256, n),) blocks = (cld(n, 256),) _cuda_add_kernel!(dst, src)
    return dst
end

# dst[i] = dst[i] / s — the 1/sqrt(d_head) attention score scale. Division,
# not multiplication by a reciprocal, because the broadcast it replaces was
# `./=` and a multiply would move bits (§LXXIII hygiene).
function _cuda_scale_kernel!(a, s)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= length(a) || return
    a[i] = Float32(Float64(a[i]) / s)
    return
end

function _scale_storage!(dst::CuArray, s)
    n = length(dst)
    n == 0 && return dst
    @fastcuda threads = (min(256, n),) blocks = (cld(n, 256),) _cuda_scale_kernel!(dst, s)
    return dst
end

function _zero_tail_storage!(dst::CuArray, from::Int)
    from > size(dst, 2) && return dst
    fill!(view(dst, :, from:size(dst, 2)), zero(eltype(dst)))
    return dst
end

# write one row of the hidden-state matrix
function _cuda_hidden_row_kernel!(dst2, src2, row::Int)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j <= size(dst2, 2) || return
    dst2[row, j] = src2[1, j]
    return
end

function _write_hidden_row_storage!(dst::CuArray, src::CuArray, row::Int)
    d = size(dst, 2)
    d == 0 && return dst
    nthreads = min(256, d)
    @fastcuda threads = (nthreads,) blocks = (cld(d, nthreads),) _cuda_hidden_row_kernel!(dst, src, row)
    return dst
end

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
    device=_autotune_device_id()
    winner=A._cached_winner(:matmul!,:cuda,regime,device)
    if winner===nothing
        winner=A.select(:matmul!,:cuda,regime,device,dst,x,w).winner
    end
    runner = A._candidate_runner(:matmul!, :cuda, winner)
    runner === nothing && throw(
        gesso_error(
            ERR_VERIFY_MISMATCH,
            "matmul!: autotune winner :$(winner) has no registered " *
            "candidate — registry and cache disagree (§LXX: no silent keep)",
            op = :matmul!,
            regime = regime,
            winner = string(winner),
        ),
    )
    # Exact function identity preserves replacement-by-name while allowing the
    # existing implementations to receive unboxed semantic arguments.
    runner === _cuda_matmul! && return _cuda_matmul!(dst, x, w)
    runner === _cuda_generic_matmul! && return _cuda_generic_matmul!(dst, x, w)
    return runner(dst, x, w)
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
    w.storage isa GpuDenseQuant && return _cuda_dq_matmul!(dst, x, w)
    w.storage isa CuArray{Float16} && return _cuda_f16w_matmul!(dst, x, w)
    w.storage isa Gesso.Inference.HostOffload &&
        return _cuda_host_offload_matmul!(dst, x, w)
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
    w.storage isa GpuDenseQuant && return _cuda_dq_matmul!(dst, x, w)
    w.storage isa CuArray{Float16} && return _cuda_f16w_matmul!(dst, x, w)
    w.storage isa Gesso.Inference.HostOffload &&
        return _cuda_host_offload_matmul!(dst, x, w)
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
    w.storage isa CuArray{Float16} && return _cuda_f16w_matmul!(dst, x, w)
    w.storage isa Gesso.Inference.HostOffload &&
        return _cuda_host_offload_matmul!(dst, x, w)
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
    w.storage isa CuArray{Float16} && return _cuda_f16w_matmul!(dst, x, w)
    w.storage isa Gesso.Inference.HostOffload &&
        return _cuda_host_offload_matmul!(dst, x, w)
    return _autotune_dispatch!(dst, x, w, _autotune_regime(size(x.storage, 2), size(w.storage, 1)))
end

# F16 device weights, F32 activations and accumulate. A table with F16 storage
# is still a device table, so the same device checks apply; only the element
# type differs.
function _cuda_embed_table_check!(table)
    table.storage isa CuArray{Float16} && return table
    return _cuda_device_storage!(:embedding_lookup!, table)
end

# dst[j, o] = Σᵢ x[j, i] · Float32(w[o, i]), W column-major [m, k].
#
# Split-K GEMV. A block is 32 lanes (output rows o0+1..o0+32, one warp) by
# _F16W_KSPLIT warps (k slices), for one token j. Lanes that share a k index
# read adjacent rows of W, which are adjacent in memory, so each warp load is a
# contiguous 64-byte span: the column-major layout is already coalesced. The
# old kernel had one thread per row walking all of k serially (512 threads for
# m = 512), so decode was latency-bound. The k slices recover parallelism and
# are summed through shared memory. Any n works (prefill loops tokens via
# blocks.y), with F32 accumulation.
const _F16W_KSPLIT = 8

function _cuda_f16w_kernel!(dst, x, w, k::Int, m::Int)
    lane = threadIdx().x
    s = threadIdx().y
    o = (blockIdx().x - 1) * 32 + lane
    j = blockIdx().y
    acc = 0.0f0
    if o <= m
        i = s
        while i <= k
            acc += Float32(w[o, i]) * x[j, i]
            i += _F16W_KSPLIT
        end
    end
    part = CUDA.@cuStaticSharedMem(Float32, (32, 8))
    part[lane, s] = acc
    CUDA.sync_threads()
    if s == 1 && o <= m
        tot = 0.0f0
        for q in 1:_F16W_KSPLIT
            tot += part[lane, q]
        end
        dst[j, o] = tot
    end
    return
end

function _cuda_f16w_matmul!(dst, x, w)
    ws = w.storage
    xs = x.storage
    ds = dst.storage
    m, k = size(ws)
    n = size(xs, 1)
    size(xs, 2) == k || throw(DimensionMismatch("f16 matmul: x is $(size(xs)), W is $(size(ws))"))
    size(ds) == (n, m) || throw(DimensionMismatch("f16 matmul: dst is $(size(ds)), want ($n, $m)"))
    n == 0 && return dst
    @fastcuda threads = (32, _F16W_KSPLIT) blocks = (cld(m, 32), n) _cuda_f16w_kernel!(ds, xs, ws, k, m)
    return dst
end

# dst = x * Wᵀ with W host-resident. Used for overflow lm_head / projections
# (35B embedding+head ~1.89 GiB each) — D2H hidden, host GEMM, H2D result.
function _cuda_host_offload_matmul!(dst, x, w)
    W = w.storage.data
    xh = Array(x.storage)
    yh = xh * transpose(W)
    copyto!(dst.storage, yh)
    return dst
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
            _cuda_matmul!,
            (dst, x, w) -> _cuda_matmul_gate(_cuda_matmul!, dst, x, w),
        ),
    )
    A.register!(
        :matmul!,
        :cuda,
        A.Candidate(
            :generic_mul,
            _cuda_generic_matmul!,
            (dst, x, w) -> _cuda_matmul_gate(_cuda_generic_matmul!, dst, x, w),
        ),
    )
    return nothing
end

# --- explicit transfer (§LXXVII: "the interpreter does not copy") ---------------

"""
    to_device(::CUDABackend, tensors) -> tensors'

Copy `Array` storage to `CuArray{Float32}` until `GESSO_GPU_BUDGET_MIB`
(default 4800) is exhausted, smallest first. Overflow becomes `HostOffload`
(explicit host residency — a forgotten raw Array still fails the CUDA
check). Remaining budget then uploads `GgufQuantSlice` expert slabs as
`GpuQuantSlice` (quantized bytes, no F32 dump). Slabs that still do not
fit stay mmap'd and the MoE path H2Ds one quantized expert at a time.
NEVER mutates the CPU tensors. Tied heads stay tied: `lm_head === embedding`.

Config-agnostic: geometry is whatever the tensors carry, so Ornith
256/8, Xing 64/4, and Gemma4 128/8 share this path.
"""
# Device element type for dense projection and tied-table weights.
# GESSO_GPU_WEIGHT_DTYPE=f16 (default) stores them as CuArray{Float16} and the
# matmul accumulates in Float32; f32 keeps the exact F32 device path. Norms,
# biases and other small parameters stay F32 either way. F16 exists only on
# the device: the host copy is always F32.
_moe_host_experts_enabled() = get(ENV, "GESSO_MOE_HOST_EXPERTS", "1") == "1"

function _gpu_weight_eltype()
    raw = lowercase(strip(get(ENV, "GESSO_GPU_WEIGHT_DTYPE", "f16")))
    raw == "f16" && return Float16
    raw == "f32" && return Float32
    throw(ArgumentError("GESSO_GPU_WEIGHT_DTYPE must be f16 or f32, got '$raw'"))
end

# Matrices below this many elements keep F32 (small routers and gates; their
# values feed top-k / gating decisions, where rounding is not worth the bytes).
const _GPU_F16_MIN_ELEMS = 1 << 20

# Upload F32, convert on the device, free the F32 copy. No F16 array is built
# on the host.
function _f16_device_copy(s)
    d = CuArray{Float32}(s)
    h = Float16.(d)
    CUDA.unsafe_free!(d)
    return h
end

# Host F32 view of a weight for upload. Quantized storage dequantizes here;
# an Array is returned as it is (converted to Float32 like before).
_host_f32(s::Gesso.Inference.QuantMatrix) = Gesso.Inference._quant_host_f32(s)
_host_f32(s) = convert(Array{Float32}, s)

function _to_device_cuda(tensors)
    _blas_reserve!()   # before the weights and pools leave no room for it
    @debug "vram before dense placement" free_mib=div(Int64(CUDA.free_memory()), Int64(1024 * 1024)) dense_budget_mib=div(Gesso.Inference._gpu_budget_bytes(), Int64(1024 * 1024))
    budget = Gesso.Inference._gpu_budget_bytes()
    wdt = _gpu_weight_eltype()
    used = Int64(0)
    conv = IdDict{Any, Any}()
    pending = Any[]
    qpending = Any[]
    f16set = Set{UInt}()
    nativeset = Set{UInt}()

    function consider!(s; matmul::Bool=false, native::Bool=false)
        s === nothing && return
        haskey(conv, s) && return
        if s isa CuArray
            conv[s] = s
            used += Int64(sizeof(s))
        elseif s isa Gesso.Inference.HostOffload
            conv[s] = s
        elseif s isa Gesso.Inference.GpuQuantSlice
            conv[s] = s
            used += Int64(sizeof(s.data))
        elseif s isa Gesso.Inference.GgufQuantSlice
            push!(qpending, s)
        else
            push!(pending, s)
            native && _dq_eligible(s) && push!(nativeset, objectid(s))
            if matmul && (s isa Array || s isa Gesso.Inference.QuantMatrix) &&
               ndims(s) == 2 && length(s) >= _GPU_F16_MIN_ELEMS
                push!(f16set, objectid(s))
            end
        end
        return
    end
    function consider_t!(t)
        hasproperty(t, :storage) || return
        # A distinct lm_head is a ProjectionWeight and may stay quantized on
        # the device. A tied table is also the embedding lookup, which reads
        # F16 device rows, so it keeps the F16 path.
        consider!(t.storage;
            matmul = t isa Gesso.ProjectionWeight || t isa Gesso.EmbeddingTable,
            native = t isa Gesso.ProjectionWeight)
        return
    end
    # Untied embedding: a row-gather table only. It stays on the host, so it
    # never competes for GPU budget and never has a device copy. A GGUF table
    # stays quantized: its rows are contiguous per token, while the F32
    # (vocab, dim) copy spreads one row over dim pages (2 GB on the 35B, and
    # 13-18 ms/token of page faults once swap holds it).
    function host_table_t!(t)
        hasproperty(t, :storage) || return
        s = t.storage
        (s === nothing || haskey(conv, s)) && return
        conv[s] = s isa Gesso.Inference.HostOffload || s isa Gesso.Inference.QuantMatrix ? s :
            Gesso.Inference.HostOffload(_host_f32(s))
        return
    end

    tied = tensors.lm_head === tensors.embedding
    if tied
        consider_t!(tensors.embedding)
    else
        host_table_t!(tensors.embedding)
        consider_t!(tensors.lm_head)
    end
    for b in tensors.blocks
        for v in values(b)
            consider_t!(v)
        end
    end
    fr0 = haskey(tensors, :final_rms) ? tensors.final_rms : nothing
    fr0 === nothing || consider_t!(fr0)

    seen = Set{UInt}()
    uniq = Any[]
    for s in pending
        id = objectid(s)
        id in seen && continue
        push!(seen, id)
        push!(uniq, s)
    end
    sort!(uniq; by=s -> Int(prod(size(s))))

    remaining = max(Int64(0), budget - used)
    for s in uniq
        # Dense quant weights of a supported type stay in their GGUF bytes on
        # the device (GpuDenseQuant). When that does not fit the budget, or the
        # type is unsupported, the weight is dequantized here, per tensor, to
        # host F32 and takes the same F16/F32/offload path as an Array.
        if objectid(s) in nativeset
            nb = _dq_device_bytes(s)
            if nb <= remaining
                conv[s] = _to_gpu_dense_quant(s)
                remaining -= nb
                continue
            end
        end
        arr = s isa Gesso.Inference.QuantMatrix ? Gesso.Inference._quant_host_f32(s) : s
        n = Int64(prod(size(s)))
        if wdt === Float16 && objectid(s) in f16set && 2n <= remaining
            conv[s] = _f16_device_copy(arr)
            remaining -= 2n
            continue
        end
        nbytes = n * Int64(sizeof(Float32))
        if nbytes <= remaining
            conv[s] = CuArray{Float32}(arr)
            remaining -= nbytes
        else
            data = eltype(arr) === Float32 && arr isa Array ? arr : convert(Array{Float32}, arr)
            conv[s] = Gesso.Inference.HostOffload(data)
        end
    end

    # Complete gate/up/down expert triples take the per-expert cache below,
    # never the whole-slab path, so one layer cannot end up mixed.
    @debug "vram after dense placement" free_mib=div(Int64(CUDA.free_memory()), Int64(1024 * 1024))
    # Opt-in (GESSO_GPU_EXPERT_CACHE=1) until misses run on the CPU: with
    # misses uploaded, the cache measured slower than whole-slab placement.
    use_cache = _expert_cache_enabled()
    all_triples = _moe_expert_triples(tensors.blocks)
    triples = use_cache ? all_triples : Any[]
    expert_ids = Set{UInt}()
    for (_, g, u, d) in triples
        union!(expert_ids, (objectid(g), objectid(u), objectid(d)))
    end
    # A layer's gate/up/down slabs go up together or not at all. A layer left
    # half on the device matches neither the host-expert path nor the device
    # prefill path and falls to the per-token upload loop: one such layer at
    # the 35B's 4 GiB budget cost ~370 ms of a 256-token prefill.
    group_of = Dict{UInt, Any}()
    for (_, g, u, d) in all_triples
        for x in (g, u, d)
            group_of[objectid(x)] = (g, u, d)
        end
    end
    qseen = Set{UInt}()
    for s in qpending
        id = objectid(s)
        id in qseen && continue
        if id in expert_ids
            push!(qseen, id)
            conv[s] = s
            continue
        end
        grp = get(group_of, id, (s,))
        foreach(x -> push!(qseen, objectid(x)), grp)
        sizes = map(Gesso.Inference._quant_nbytes, grp)
        nbytes = sum(sizes)
        # A slab of a type the CUDA expert qgemv has no kernel for stays
        # host-resident and runs on the CPU expert path, instead of failing at
        # the first MoE.
        if nbytes <= remaining && all(>(0), sizes) &&
           all(x -> x.ggml_type in _CUDA_QGEMV_TYPES, grp)
            for x in grp
                packed = Gesso.Inference.packed_quant_bytes(x)
                conv[x] = Gesso.Inference.gpu_quant_slice(x, CuArray{UInt8}(packed))
            end
            remaining -= nbytes
        else
            foreach(x -> conv[x] = x, grp)
        end
    end

    use_cache && _place_expert_caches!(conv, triples)

    _conv_t(t) = begin
        hasproperty(t, :storage) || return t         # mixer tags and other metadata
        t.storage === nothing && return t            # unset storage stays unset
        typeof(t)(; shape = t.shape, storage = conv[t.storage])
    end
    _conv_blk(b) = NamedTuple{keys(b)}(map(_conv_t, values(b)))
    embedding = _conv_t(tensors.embedding)
    blocks = [_conv_blk(b) for b in tensors.blocks]
    lm_head = tensors.lm_head === tensors.embedding ? embedding : _conv_t(tensors.lm_head)
    fr = haskey(tensors, :final_rms) ? tensors.final_rms : nothing
    final_rms = fr === nothing ? nothing : _conv_t(fr)
    cached_n = 0
    slab_n = 0
    host_n = 0
    for bt in blocks, k in (:wexp_gate, :wexp_up, :wexp_down)
        haskey(bt, k) || continue
        s = bt[k].storage
        if s isa Gesso.Inference.CachedExpertBank
            cached_n += 1
        elseif s isa Gesso.Inference.GpuQuantSlice
            slab_n += 1
        elseif s isa Gesso.Inference.GgufQuantSlice
            host_n += 1
        end
    end
    _prefault_host_experts!(blocks)
    _hot_experts_enabled() ? _hot_register!(blocks) : _hot_release_all!()
    @debug "expert storage census" cached_projections=cached_n whole_slab_projections=slab_n host_projections=host_n vram_free_mib=div(Int64(CUDA.free_memory()), Int64(1024 * 1024))
    result = (; embedding, blocks, lm_head, final_rms)
    return merge(result, _keep_tensor_meta(tensors))
end

# Host-resident expert slabs are views into the mmap'd GGUF, mapped but not
# yet faulted in. The CPU expert path then takes its first-touch page faults in
# the middle of decode: the first generation's tokens ran 50 to 90 ms instead
# of 32 while new experts were routed. MADV_POPULATE_READ (Linux 5.14) maps the
# pages here instead, at load; where it is missing, decode just faults lazily
# as before.
const _MADV_POPULATE_READ = Cint(22)

function _prefault_host_experts!(blocks)
    Sys.islinux() || return
    slabs = Gesso.Inference.GgufQuantSlice[]
    for bt in blocks, k in (:wexp_gate, :wexp_up, :wexp_down)
        haskey(bt, k) || continue
        s = bt[k].storage
        s isa Gesso.Inference.GgufQuantSlice && push!(slabs, s)
    end
    isempty(slabs) && return
    page = UInt(ccall(:getpagesize, Cint, ()))
    failed = Threads.Atomic{Int}(0)
    Threads.@threads for i in eachindex(slabs)
        s = slabs[i]
        GC.@preserve s begin
            lo = UInt(pointer(s.data)) + UInt(s.offset0)
            hi = lo + UInt(s.n_exp * s.bytes_per_expert)
            lo &= ~(page - 1)
            rc = ccall(:madvise, Cint, (Ptr{Cvoid}, Csize_t, Cint), Ptr{Cvoid}(lo), hi - lo,
                _MADV_POPULATE_READ)
            rc == 0 || Threads.atomic_add!(failed, 1)
        end
    end
    failed[] == 0 || @debug "host expert prefault skipped" slabs=length(slabs) failed=failed[]
    return
end

# --- per-expert device residency (see src/Inference/expert_residency.jl) ----

const _EXPERT_RESERVE_DEFAULT_MIB = 1024

_expert_cache_enabled() = get(ENV, "GESSO_GPU_EXPERT_CACHE", "0") == "1"

# Device bytes for expert slot pools: GESSO_GPU_EXPERT_CACHE_MIB when set (0
# disables residency, every expert misses), else free device memory now minus
# GESSO_GPU_RESERVE_MIB (default 1024) for activations, KV and scratch. Called
# after dense placement, so the free figure already reflects dense weights.
function _expert_cache_budget_bytes()
    mib = Int64(1024 * 1024)
    raw = strip(get(ENV, "GESSO_GPU_EXPERT_CACHE_MIB", ""))
    if !isempty(raw)
        v = tryparse(Int, raw)
        v === nothing && error("GESSO_GPU_EXPERT_CACHE_MIB must be an integer MiB, got '$raw'")
        return Int64(max(0, v)) * mib
    end
    rraw = strip(get(ENV, "GESSO_GPU_RESERVE_MIB", string(_EXPERT_RESERVE_DEFAULT_MIB)))
    reserve = tryparse(Int, rraw)
    reserve === nothing && error("GESSO_GPU_RESERVE_MIB must be an integer MiB, got '$rraw'")
    free_raw = Int64(CUDA.free_memory())
    # GC level: a DROP would also destroy library handles that must be re-created later
    CUDA.reclaim(CUDA.CUDACore.RECLAIM_GC)   # return the allocator's cached blocks so the free figure is real
    free = Int64(CUDA.free_memory())
    budget = _expert_budget_from_free(free, reserve)
    @debug "expert cache budget" free_before_reclaim_mib=div(free_raw, mib) free_mib=div(free, mib) reserve_mib=reserve budget_mib=div(budget, mib)
    return budget
end

# Pure budget arithmetic: free device bytes minus the reserve, never negative.
function _expert_budget_from_free(free_bytes::Integer, reserve_mib::Integer)
    free_bytes >= 0 || error("_expert_budget_from_free: negative free bytes $free_bytes")
    return max(Int64(0), Int64(free_bytes) - Int64(max(0, reserve_mib)) * Int64(1024 * 1024))
end

# (layer, gate, up, down) for every block whose three expert banks are all
# host GgufQuantSlice. `layer` is the 1-based block index: the routing trace
# and the profile use it.
function _moe_expert_triples(blocks)
    out = Any[]
    for (li, bt) in enumerate(blocks)
        (haskey(bt, :wexp_gate) && haskey(bt, :wexp_up) && haskey(bt, :wexp_down)) || continue
        g = bt.wexp_gate.storage
        u = bt.wexp_up.storage
        d = bt.wexp_down.storage
        (g isa Gesso.Inference.GgufQuantSlice && u isa Gesso.Inference.GgufQuantSlice &&
         d isa Gesso.Inference.GgufQuantSlice) || continue
        push!(out, (li, g, u, d))
    end
    return out
end

# Fill each layer's pools from the profile (GESSO_EXPERT_PROFILE) or by id,
# then bind the three projections of every layer to one residency table.
function _place_expert_caches!(conv, triples; budget::Int64=_expert_cache_budget_bytes())
    isempty(triples) && return conv
    Inf_ = Gesso.Inference
    prof_path = get(ENV, "GESSO_EXPERT_PROFILE", "")
    profile = isempty(prof_path) ? nothing : Inf_.read_expert_profile(prof_path)
    per_layer = div(budget, Int64(length(triples)))
    slots_log = Int[]
    for (li, g, u, d) in triples
        banks = convert(NTuple{3,Inf_.GgufQuantSlice}, (g, u, d))
        n_exp = g.n_exp
        slots = Inf_.expert_layer_quota(per_layer, banks, n_exp)
        counts = profile === nothing ? nothing : get(profile, li, nothing)
        order = Inf_.expert_fill_order(counts, n_exp)
        pools = ntuple(p -> CuArray{UInt8}(undef, slots * banks[p].bytes_per_expert), 3)
        cache = Inf_.ExpertLayerCache(li, banks, pools, n_exp, slots)
        Inf_.expert_fill!(cache, order)
        for s in 1:slots
            e = Int(cache.slot_expert[s])
            for p in 1:3
                bank = banks[p]
                nb = bank.bytes_per_expert
                copyto!(pools[p], (s - 1) * nb + 1, bank.data, Inf_._quant_expert_off0(bank, e) + 1, nb)
            end
        end
        conv[g] = Inf_.CachedExpertBank(cache, 1)
        conv[u] = Inf_.CachedExpertBank(cache, 2)
        conv[d] = Inf_.CachedExpertBank(cache, 3)
        push!(slots_log, slots)
    end
    @debug "expert cache placed" budget_mib=div(budget, Int64(1024 * 1024)) layers=length(triples) slots=slots_log profile=(isempty(prof_path) ? "none" : prof_path)
    return conv
end

# ArchitectureSpec, RoPEPolicy, mixer tags — anything without .storage.
const _TO_DEVICE_STORAGE_KEYS = (:embedding, :blocks, :lm_head, :final_rms)

function _keep_tensor_meta(tensors)
    ks = Symbol[]
    vs = Any[]
    for k in keys(tensors)
        k in _TO_DEVICE_STORAGE_KEYS && continue
        push!(ks, k)
        push!(vs, getproperty(tensors, k))
    end
    isempty(ks) && return (;)
    return NamedTuple{Tuple(ks)}(Tuple(vs))
end

# ext-local dispatch wrapper (bound as Gesso.to_device when this extension is
# the first backend extension to load; otherwise __init__ attaches the worker
# to the already-installed function object — see GessoLavaExt for the full
# cross-extension rationale and the empirical note on why the definition
# must go through the NAME in the eval target's scope)
function to_device(::CUDABackend, tensors)
    return _to_device_cuda(tensors)
end

# Ordinary strided per-head GEMM into existing score/attention scratch.
# No copied head slices or product arrays survive between tokens.
function _attention_scores_device!(::CUDABackend, sc, q::AbstractArray, k::AbstractArray, hh, d_head, K)
    qh = view(q, :, hh, :)
    kh = view(k, 1:K, hh, :)
    if size(q, 1) == 1
        _blas_gemv!('N', _cu_scalar(eltype(sc)(1 / sqrt(d_head))), kh, view(q, 1, hh, :), _cu_scalar(zero(eltype(sc))), view(sc, 1, :))
    else
        _blas_gemm!('N', 'T', _cu_scalar(eltype(sc)(1 / sqrt(d_head))), qh, kh, _cu_scalar(zero(eltype(sc))), sc)
    end
    return sc
end
function _attention_values_device!(::CUDABackend, attn::AbstractArray, probs, v::AbstractArray, hh, K)
    if size(attn, 1) == 1
        _blas_gemv!('T', _cu_scalar(one(eltype(attn))), view(v, 1:K, hh, :), view(probs, 1, :), _cu_scalar(zero(eltype(attn))), view(attn, 1, hh, :))
    else
        _blas_gemm!('N', 'N', _cu_scalar(one(eltype(attn))), probs, view(v, 1:K, hh, :), _cu_scalar(zero(eltype(attn))), view(attn, :, hh, :))
    end
    return attn
end

# --- Qwen3.5 hybrid extras: GDN, causal conv1d, sigmoid/silu gates --------------
#
# Inference-internal dispatch (same form as `_attention_scores_device!`).
# One block per value head for the recurrent GDN step; conv/gate/silu are
# elementwise. F32 on device; the CPU oracle stays F64 (atol, not bits).

function _cuda_sigmoid_mul_kernel!(dst, src, gate)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= length(dst) || return
    gv = gate[i]
    dst[i] = src[i] / (Float32(1) + exp(-gv))
    return
end

function _sigmoid_mul!(::CUDABackend, dst::AbstractArray, src::AbstractArray, gate::AbstractArray)
    n = length(dst)
    n == 0 && return dst
    @fastcuda threads = (min(256, n),) blocks = (cld(n, 256),) _cuda_sigmoid_mul_kernel!(dst, src, gate)
    return dst
end

function _cuda_silu_mul_kernel!(dst, src, gate)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= length(dst) || return
    gv = gate[i]
    dst[i] = src[i] * (gv / (Float32(1) + exp(-gv)))
    return
end

function _silu_mul!(::CUDABackend, dst::AbstractArray, src::AbstractArray, gate::AbstractArray)
    n = length(dst)
    n == 0 && return dst
    @fastcuda threads = (min(256, n),) blocks = (cld(n, 256),) _cuda_silu_mul_kernel!(dst, src, gate)
    return dst
end

function _cuda_gdn_gates_kernel!(g, beta, a, b, A_log, dt_bias)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    hh = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    seq = size(g, 1)
    nv = size(g, 2)
    (t > seq || hh > nv) && return
    bv = b[t, hh]
    beta[t, hh] = Float32(1) / (Float32(1) + exp(-bv))
    av = a[t, hh] + dt_bias[hh]
    sp = av > Float32(20) ? av : log1p(exp(av))
    g[t, hh] = -exp(A_log[hh]) * sp
    return
end

function _gdn_gate_params!(::CUDABackend, g, beta, a, b, A_log, dt_bias)
    A_log = _cuda_materialize_small(A_log)
    dt_bias = _cuda_materialize_small(dt_bias)
    seq, nv = size(beta)
    seq == 0 && return g, beta
    tx = min(16, seq)
    ty = min(16, nv)
    @fastcuda threads = (tx, ty) blocks = (cld(seq, tx), cld(nv, ty)) _cuda_gdn_gates_kernel!(
        g, beta, a, b, A_log, dt_bias)
    return g, beta
end

# One thread per token and a block column per head: consecutive threads read
# consecutive tokens, so every core/z/merged access is coalesced, and a head's
# dv values stay in one thread so no block reduction is needed. Replaces about
# 5 launches per head per layer (57 ms of a 35B P256 prefill).
# GESSO_GDN_PREFILL_GNORM=0 runs the per-head loop again.
function _cuda_gdn_gnorm_prefill_kernel!(merged, core, z, norm_w, dv::Int, eps::Float32)
    t = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    hh = Int(blockIdx().y)
    t <= size(core, 1) || return
    ss = 0.0f0
    for j in 1:dv
        c = core[t, hh, j]
        ss += c * c
    end
    inv = 1.0f0 / sqrt(ss / dv + eps)
    base = (hh - 1) * dv
    for j in 1:dv
        zv = z[t, base + j]
        merged[t, base + j] = (core[t, hh, j] * inv) * norm_w[j] * (zv / (1.0f0 + exp(-zv)))
    end
    return
end

function _gdn_gnorm_merge!(::CUDABackend, merged::CuArray{Float32, 2}, core::CuArray{Float32, 3},
                           z::CuArray{Float32, 2}, scale, eps)
    get(ENV, "GESSO_GDN_PREFILL_GNORM", "1") == "1" || return false
    seq, nv, dv = size(core)
    seq == 0 && return true
    norm_w = _cuda_materialize_small(scale.storage)
    tx = min(128, seq)
    @fastcuda threads = tx blocks = (cld(seq, tx), nv) _cuda_gdn_gnorm_prefill_kernel!(
        merged, core, z, norm_w, dv, Float32(eps))
    return true
end

function _cuda_conv1d_silu_kernel!(dst, x, weight, conv_state, k::Int)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    c = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    seq = size(dst, 1)
    channels = size(dst, 2)
    (t > seq || c > channels) && return
    acc = zero(eltype(dst))
    for p in 1:k
        src = t - (k - p)
        v = src <= 0 ? conv_state[c, src + (k - 1)] : x[src, c]
        acc += weight[c, 1, p] * v
    end
    dst[t, c] = acc / (Float32(1) + exp(-acc))
    return
end

function _cuda_conv_state_update_kernel!(conv_state, x, seq::Int, k::Int)
    c = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    channels = size(conv_state, 1)
    c > channels && return
    k1 = k - 1
    if seq >= k1
        for p in 1:k1
            conv_state[c, p] = x[seq - k1 + p, c]
        end
    else
        keep = k1 - seq
        for p in 1:keep
            conv_state[c, p] = conv_state[c, p + seq]
        end
        for p in 1:seq
            conv_state[c, keep + p] = x[p, c]
        end
    end
    return
end

function _causal_conv1d_silu!(::CUDABackend, dst, x, weight, conv_state)
    weight = _cuda_materialize_small(weight)
    conv_state = _cuda_materialize_small(conv_state)
    seq, channels = size(x)
    k = size(weight, 3)
    if seq > 0
        tx = min(16, seq)
        ty = min(16, channels)
        @fastcuda threads = (tx, ty) blocks = (cld(seq, tx), cld(channels, ty)) _cuda_conv1d_silu_kernel!(
            dst, x, weight, conv_state, Int(k))
    end
    nthreads = min(256, channels)
    @fastcuda threads = (nthreads,) blocks = (cld(channels, nthreads),) _cuda_conv_state_update_kernel!(
        conv_state, x, Int(seq), Int(k))
    return dst
end

function _cuda_gdn_qk_prep_kernel!(
    qn, kn, q, k, nk::Int, group::Int, tiled::Bool, do_l2::Bool, scale::Float32, eps::Float32,
)
    t = Int(blockIdx().x)
    hh = Int(blockIdx().y)
    seq = size(q, 1)
    nv = size(qn, 2)
    dk = size(q, 3)
    (t > seq || hh > nv) && return
    h = tiled ? ((hh - 1) % nk) + 1 : ((hh - 1) ÷ group) + 1
    accq = zero(Float32)
    acck = zero(Float32)
    for d in 1:dk
        qv = q[t, h, d]
        kv = k[t, h, d]
        qn[t, hh, d] = qv
        kn[t, hh, d] = kv
        accq += qv * qv
        acck += kv * kv
    end
    if do_l2
        invq = Float32(1) / sqrt(accq + eps)
        invk = Float32(1) / sqrt(acck + eps)
        for d in 1:dk
            qn[t, hh, d] *= invq * scale
            kn[t, hh, d] *= invk
        end
    else
        for d in 1:dk
            qn[t, hh, d] *= scale
        end
    end
    return
end

# One block per (value head, tile of value columns), one thread per column.
# The block's state tile lives in shared memory for the whole prompt, and
# token t's k and q are staged once per block and read as broadcasts. Global
# state in [nv, dk, dv] order puts neighbouring columns nv*dk apart, so a
# per-token pass over it never coalesced; that pass was 9 ms of every prefill
# token on Ornith-35B. Each column runs the same arithmetic in the same order as the
# untiled kernel it replaced, so the tile width does not change the result.
const _GDN_RECUR_TILE = 32

function _cuda_gdn_recurrent_kernel!(out, qn, kn, v, g, beta, state)
    T = eltype(state)
    dk = size(state, 2)
    dv = size(state, 3)
    seq = size(out, 1)
    hh = Int(blockIdx().y)
    jl = Int(threadIdx().x)
    tj = Int(blockDim().x)
    j = (Int(blockIdx().x) - 1) * tj + jl
    st = CUDA.CuDynamicSharedArray(T, (tj, dk))
    kq = CUDA.CuDynamicSharedArray(T, (dk, 2), tj * dk * sizeof(T))
    live = j <= dv
    @inbounds begin
        if live
            for i in 1:dk
                st[jl, i] = state[hh, i, j]
            end
        end
        for t in 1:seq
            # the previous token's k/q reads finish before the overwrite
            CUDA.sync_threads()
            i = jl
            while i <= dk
                kq[i, 1] = kn[t, hh, i]
                kq[i, 2] = qn[t, hh, i]
                i += tj
            end
            CUDA.sync_threads()
            if live
                decay = exp(g[t, hh])
                bt = beta[t, hh]
                kv_mem = zero(T)
                for i in 1:dk
                    sv = st[jl, i] * decay
                    st[jl, i] = sv
                    kv_mem += sv * kq[i, 1]
                end
                delta = (v[t, hh, j] - kv_mem) * bt
                acc = zero(eltype(out))
                for i in 1:dk
                    sv = st[jl, i] + kq[i, 1] * delta
                    st[jl, i] = sv
                    acc += sv * kq[i, 2]
                end
                out[t, hh, j] = acc
            end
        end
        if live
            for i in 1:dk
                state[hh, i, j] = st[jl, i]
            end
        end
    end
    return
end

# Register-resident GDN recurrence, v2: four lanes per value column, each
# holding dk/4 rows of that column's state for the whole prompt in registers.
# q, k and v are staged in shared memory in chunks of T tokens, and the next
# chunk is prefetched into registers while the current one is computed. The
# q and k l2 norms come from a separate coalesced kernel, not the recurrence.
# The sums run in a different order from the CPU reference, so results agree
# to rounding, not bitwise. GESSO_GDN_RECUR_REG=0 restores the shared-memory
# kernel above.
_gdn_recur_reg_enabled() = get(ENV, "GESSO_GDN_RECUR_REG", "1") == "1"

# One block per (32-token tile, k head), 32 x 8 threads, coalesced along t.
# The 8 threads of a token split the head's dk columns of k and q.
function _cuda_gdn_norm_kernel!(invk, invq, q, k, do_l2::Bool, scale::Float32, eps::Float32)
    seq, dk = size(q, 1), size(q, 3)
    tx = Int(threadIdx().x)
    ty = Int(threadIdx().y)
    t = (Int(blockIdx().x) - 1) * 32 + tx
    h = Int(blockIdx().y)
    pk = CuStaticSharedArray(Float32, (32, 8))
    pq = CuStaticSharedArray(Float32, (32, 8))
    sk = 0.0f0
    sq = 0.0f0
    if t <= seq
        @inbounds for d in ty:8:dk
            a = Float32(k[t, h, d])
            b = Float32(q[t, h, d])
            sk = fma(a, a, sk)
            sq = fma(b, b, sq)
        end
    end
    @inbounds pk[tx, ty] = sk
    @inbounds pq[tx, ty] = sq
    sync_threads()
    if ty == 1 && t <= seq
        a = 0.0f0
        b = 0.0f0
        @inbounds for y in 1:8
            a += pk[tx, y]
            b += pq[tx, y]
        end
        @inbounds invk[t, h] = do_l2 ? 1.0f0 / sqrt(a + eps) : 1.0f0
        @inbounds invq[t, h] = (do_l2 ? 1.0f0 / sqrt(b + eps) : 1.0f0) * scale
    end
    return
end

# Four accumulators keep four FMA chains in flight to hide add latency.
@inline function _gdn_dot4(a::NTuple{N,Float32}, b::NTuple{N,Float32}) where {N}
    s1 = 0.0f0
    s2 = 0.0f0
    s3 = 0.0f0
    s4 = 0.0f0
    @inbounds for c in 0:4:N-1
        s1 = fma(a[c+1], b[c+1], s1)
        s2 = fma(a[c+2], b[c+2], s2)
        s3 = fma(a[c+3], b[c+3], s3)
        s4 = fma(a[c+4], b[c+4], s4)
    end
    return (s1 + s2) + (s3 + s4)
end

# Xor 1 and 2 sum the four lanes of one value column.
@inline function _gdn_sum4(x)
    x += CUDA.shfl_xor_sync(0xffffffff, x, 1)
    x += CUDA.shfl_xor_sync(0xffffffff, x, 2)
    return x
end

# Top level, not a closure: S is reassigned in the token loop, and a closure
# capturing it would box S and fail with "unsupported dynamic function invocation".
@inline _gdn_step(S::NTuple{M,Float32}, kk::NTuple{M,Float32}, coef, decay) where {M} =
    ntuple(m -> fma(kk[m], coef, S[m] * decay), Val(M))

# Lane r holds float4 chunks r, r+4, ...; register m is element (m-1) % 4 of
# chunk (m-1) ÷ 4 within that lane's set, so a lane's state rows are contiguous.
@inline _gdn_srow(r, m) = 4 * (r + 4 * ((m - 1) >> 2)) + ((m - 1) & 3) + 1

# Chunk loads and stores are top-level for the same reason as _gdn_step: inside
# a closure the chunk counts are captured fields, so Val(NKQ) is not static.
@inline function _gdn_v2_load(k, q, v, g, beta, invk, invq, tid, t0, seq, h, hh, j0, dv,
        ::Val{NT}, ::Val{T}, ::Val{NKQ}, ::Val{NV}) where {NT,T,NKQ,NV}
    @inbounds begin
        kr = ntuple(i -> (e = tid + NT * (i - 1); t = t0 + e % T + 1;
            t <= seq ? Float32(k[t, h, e ÷ T + 1]) : 0.0f0), Val(NKQ))
        qr = ntuple(i -> (e = tid + NT * (i - 1); t = t0 + e % T + 1;
            t <= seq ? Float32(q[t, h, e ÷ T + 1]) : 0.0f0), Val(NKQ))
        vr = ntuple(i -> (e = tid + NT * (i - 1); t = t0 + e % T + 1; jj = j0 + e ÷ T + 1;
            (t <= seq && jj <= dv) ? Float32(v[t, hh, jj]) : 0.0f0), Val(NV))
        sr = (tid < T && t0 + tid < seq) ?
            (Float32(g[t0+tid+1, hh]), Float32(beta[t0+tid+1, hh]),
             invk[t0+tid+1, h], invq[t0+tid+1, h]) :
            (0.0f0, 0.0f0, 0.0f0, 0.0f0)
    end
    return kr, qr, vr, sr
end

@inline function _gdn_v2_store(ks, qs, vs, sm, kr, qr, vr, sr, tid,
        ::Val{NT}, ::Val{T}, ::Val{KP}, ::Val{COLS}) where {NT,T,KP,COLS}
    @inbounds begin
        for i in 1:length(kr)
            e = tid + NT * (i - 1)
            o = (e % T) * KP + e ÷ T + 1
            ks[o] = kr[i]
            qs[o] = qr[i]
        end
        for i in 1:length(vr)
            e = tid + NT * (i - 1)
            vs[(e % T) * COLS + e ÷ T + 1] = vr[i]
        end
        if tid < T
            for c in 1:4
                sm[tid+1, c] = sr[c]
            end
        end
    end
    return
end

function _cuda_gdn_recur_reg_kernel!(out, q, k, v, g, beta, invk, invq, state,
        nk::Int, group::Int, tiled::Bool, ::Val{COLS}, ::Val{T}, ::Val{DK}) where {COLS,T,DK}
    NT = 4 * COLS
    M = DK ÷ 4
    # padded to KP so each row is 16-byte aligned and the transposed stores
    # spread across banks
    KP = DK + 4
    NKQ = (T * DK) ÷ NT
    NV = (T * COLS) ÷ NT
    seq = size(out, 1)
    dv = size(state, 3)
    tid = Int(threadIdx().x) - 1
    cl = tid >> 2
    r = tid & 3
    hh = Int(blockIdx().y)
    j0 = (Int(blockIdx().x) - 1) * COLS
    j = j0 + cl + 1
    live = j <= dv
    # a dead column still joins every shuffle, so it reads column 1 and never writes
    jr = live ? j : 1
    h = tiled ? ((hh - 1) % nk) + 1 : ((hh - 1) ÷ group) + 1
    ks = CuStaticSharedArray(Float32, T * KP)
    qs = CuStaticSharedArray(Float32, T * KP)
    vs = CuStaticSharedArray(Float32, T * COLS)
    os = CuStaticSharedArray(Float32, T * COLS)
    sm = CuStaticSharedArray(Float32, (T, 4))
    @inbounds begin
        S = ntuple(m -> Float32(state[hh, _gdn_srow(r, m), jr]), Val(M))
        LA = (Val(NT), Val(T), Val(NKQ), Val(NV))
        SA = (Val(NT), Val(T), Val(KP), Val(COLS))
        kr, qr, vr, sr = _gdn_v2_load(k, q, v, g, beta, invk, invq, tid, 0, seq, h, hh, j0, dv, LA...)
        _gdn_v2_store(ks, qs, vs, sm, kr, qr, vr, sr, tid, SA...)
        sync_threads()
        nch = cld(seq, T)
        for c in 0:nch-1
            t0 = c * T
            pk, pq, pv, ps = _gdn_v2_load(k, q, v, g, beta, invk, invq, tid, t0 + T, seq, h, hh, j0, dv, LA...)
            nt = min(T, seq - t0)
            for tt in 0:nt-1
                b = tt * KP
                kk = ntuple(m -> ks[b + _gdn_srow(r, m)], Val(M))
                decay = exp(sm[tt+1, 1])
                ik = sm[tt+1, 3]
                sk = _gdn_sum4(_gdn_dot4(S, kk))
                delta = (vs[tt*COLS+cl+1] - decay * sk * ik) * sm[tt+1, 2]
                coef = delta * ik
                S = _gdn_step(S, kk, coef, decay)
                qq = ntuple(m -> qs[b + _gdn_srow(r, m)], Val(M))
                acc = _gdn_sum4(_gdn_dot4(S, qq)) * sm[tt+1, 4]
                r == 0 && (os[tt*COLS+cl+1] = acc)
            end
            # ends this chunk's reads of ks/qs/vs/sm before the prefetch overwrites them
            sync_threads()
            for i in 1:NV
                e = tid + NT * (i - 1)
                t = t0 + e % T + 1
                jj = j0 + e ÷ T + 1
                (t <= seq && jj <= dv) && (out[t, hh, jj] = os[(e % T) * COLS + e ÷ T + 1])
            end
            c + 1 < nch && _gdn_v2_store(ks, qs, vs, sm, pk, pq, pv, ps, tid, SA...)
            # publishes the prefetched chunk to the whole block
            sync_threads()
        end
        if live
            for m in 1:M
                state[hh, _gdn_srow(r, m), j] = S[m]
            end
        end
    end
    return
end

function _gated_delta_rule!(
    backend::CUDABackend,
    out,
    q,
    k,
    v,
    g,
    beta,
    state;
    l2norm::Bool=true,
    v_head_tiled::Bool=false,
)
    seq = size(q, 1)
    nk, dk = size(q, 2), size(q, 3)
    nv, dv = size(v, 2), size(v, 3)
    nv % nk == 0 || error("gated_delta_rule: n_v_heads $nv not divisible by n_k_heads $nk")
    group = div(nv, nk)
    size(state) == (nv, dk, dv) ||
        error("gated_delta_rule: state $(size(state)) ≠ ($nv, $dk, $dv)")
    seq == 0 && return out
    scale = Float32(1) / sqrt(Float32(dk))
    # dk/4 state values per lane in registers; float4 chunks split evenly over
    # the four lanes and dot4 needs a multiple of 4, hence dk % 16. The cap is
    # where the prototype was checked and timed (32 values per lane).
    if _gdn_recur_reg_enabled() && dk % 16 == 0 && dk <= 128
        cols = 32
        invk = similar(out, Float32, (seq, Int(nk)))
        invq = similar(out, Float32, (seq, Int(nk)))
        Gesso.Inference.@decode_sub :gdn_qk backend begin
            @fastcuda threads = (32, 8) blocks = (cld(seq, 32), Int(nk)) _cuda_gdn_norm_kernel!(
                invk, invq, q, k, l2norm, scale, Float32(1e-6))
        end
        Gesso.Inference.@decode_sub :gdn_recur backend begin
            @fastcuda threads = 4 * cols blocks = (cld(dv, cols), nv) _cuda_gdn_recur_reg_kernel!(
                out, q, k, v, g, beta, invk, invq, state, Int(nk), Int(group), v_head_tiled,
                Val(cols), Val(8), Val(dk))
        end
        CUDA.unsafe_free!(invk)
        CUDA.unsafe_free!(invq)
        return out
    end
    qn = similar(out, eltype(out), (seq, nv, dk))
    kn = similar(out, eltype(out), (seq, nv, dk))
    Gesso.Inference.@decode_sub :gdn_qk backend begin
        @fastcuda threads = 1 blocks = (seq, nv) _cuda_gdn_qk_prep_kernel!(
            qn, kn, q, k, Int(nk), Int(group), v_head_tiled, l2norm, scale, Float32(1e-6))
    end
    tj = min(_GDN_RECUR_TILE, dv)
    shmem = (tj + 2) * dk * sizeof(eltype(state))
    Gesso.Inference.@decode_sub :gdn_recur backend begin
        @fastcuda threads = (tj,) blocks = (cld(dv, tj), nv) shmem = shmem _cuda_gdn_recurrent_kernel!(
            out, qn, kn, v, g, beta, state)
    end
    CUDA.unsafe_free!(qn)
    CUDA.unsafe_free!(kn)
    return out
end

# --- Qwen3.5 GDN single-token decode (seq = 1), fused ---------------------------
#
# Three launches replace the per-head loop of the unfused block (about 190
# launches and ~13 temporaries per layer):
#   1. conv1d + SiLU + conv-state shift: one thread per channel.
#   2. delta rule: one block per value-head column j, one thread per value head.
#      Thread hh computes its head's gates and q/k L2 norms itself. The state is
#      [nv, dk, dv] column-major, so the nv threads of a block read consecutive
#      addresses; a block-per-head layout strides by nv*dk instead.
#   3. gated RMSNorm times SiLU(z): one block per value head. The norm needs the
#      whole head, so it cannot be folded into (2) without a cross-block reduce.
# The rmsnorm, the four projections and the output projection stay in the
# generic decode code.
#
# Layout contract (same as the unfused path): conv_out holds q, k, v, each
# head-major; value head hh reads K head h = tiled ? (hh-1)%nk+1 : (hh-1)÷group+1;
# core and merged are head-major with index (hh-1)*dv + j.

Gesso.Inference._gdn_fused_decode(::CUDABackend) = true

# Stream-ordered: kernels already queued on this task's stream still see it.
Gesso.Inference._release_storage!(x::CuArray) = (CUDA.unsafe_free!(x); nothing)

# conv1d (kernel k) on one token + SiLU, then the shift of the conv state.
# Thread c owns channel c alone, so its state column is read before it is
# overwritten and no other thread touches it.
function _cuda_gdn_conv_decode_kernel!(conv_out, state, mixed, weight, k::Int)
    c = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    channels = size(state, 1)
    c > channels && return
    acc = zero(eltype(conv_out))
    for p in 1:(k - 1)
        acc += weight[c, 1, p] * state[c, p]
    end
    xv = mixed[1, c]
    acc += weight[c, 1, k] * xv
    conv_out[1, c] = acc / (one(acc) + exp(-acc))
    for p in 1:(k - 2)
        state[c, p] = state[c, p + 1]
    end
    state[c, k - 1] = xv
    return
end

# Block-wide sum via shared memory; T is a power of two. Every thread gets the
# total. The leading barrier protects `red` from the previous reduction's reads.
@inline function _cuda_gdn_block_sum(red, x, tid::Int, T::Int)
    CUDA.sync_threads()
    red[tid] = x
    CUDA.sync_threads()
    s = T >> 1
    while s > 0
        if tid <= s
            red[tid] += red[tid + s]
        end
        CUDA.sync_threads()
        s >>= 1
    end
    return red[1]
end

# Block j is state column j of every head. Thread (hh, y) is value head hh and
# takes the state rows i ≡ y (mod Y); the Y partial sums of each head meet in
# shared memory. A thread per head walking all dk rows serially left the state
# reads latency-bound (one 4-byte load in flight per thread). The q and k norms
# are applied to the sums instead of to every element, which is the same math
# to rounding.
function _cuda_gdn_delta_kernel!(
    core, conv_out, a, b, A_log, dt_bias, state,
    nk::Int, group::Int, tiled::Bool, dk::Int, dv::Int, scale::Float32, eps::Float32,
)
    hh = Int(threadIdx().x)
    y = Int(threadIdx().y)
    nvh = Int(blockDim().x)
    Y = Int(blockDim().y)
    j = Int(blockIdx().x)
    red = CUDA.CuDynamicSharedArray(Float32, (nvh, Y, 3))
    key_dim = nk * dk
    h = tiled ? ((hh - 1) % nk) + 1 : ((hh - 1) ÷ group) + 1
    qoff = (h - 1) * dk
    koff = key_dim + (h - 1) * dk

    bv = b[1, hh]
    beta = 1.0f0 / (1.0f0 + exp(-bv))
    av = a[1, hh] + dt_bias[hh]
    sp = av > 20.0f0 ? av : log1p(exp(av))
    decay = exp(-exp(A_log[hh]) * sp)

    sq = 0.0f0
    sk = 0.0f0
    sm = 0.0f0
    i = y
    while i <= dk
        qv = conv_out[1, qoff + i]
        kv = conv_out[1, koff + i]
        sq += qv * qv
        sk += kv * kv
        sm += (state[hh, i, j] * decay) * kv
        i += Y
    end
    red[hh, y, 1] = sq
    red[hh, y, 2] = sk
    red[hh, y, 3] = sm
    CUDA.sync_threads()
    sq = 0.0f0
    sk = 0.0f0
    sm = 0.0f0
    for t in 1:Y
        sq += red[hh, t, 1]
        sk += red[hh, t, 2]
        sm += red[hh, t, 3]
    end
    invq = scale / sqrt(sq + eps)
    invk = 1.0f0 / sqrt(sk + eps)
    delta = (conv_out[1, 2 * key_dim + (hh - 1) * dv + j] - sm * invk) * beta
    kd = invk * delta
    acc = 0.0f0
    i = y
    while i <= dk
        sn = state[hh, i, j] * decay + conv_out[1, koff + i] * kd
        state[hh, i, j] = sn
        acc += sn * conv_out[1, qoff + i]
        i += Y
    end
    # Every thread must be done reading the first reduction before it is reused.
    CUDA.sync_threads()
    red[hh, y, 1] = acc
    CUDA.sync_threads()
    if y == 1
        tot = 0.0f0
        for t in 1:Y
            tot += red[hh, t, 1]
        end
        core[1, (hh - 1) * dv + j] = tot * invq
    end
    return
end

# One block per value head: gated RMSNorm of the head's output, times SiLU(z).
function _cuda_gdn_gnorm_kernel!(merged, core, z, norm_w, dv::Int, eps::Float32)
    hh = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    T = Int(blockDim().x)
    red = CUDA.CuDynamicSharedArray(Float32, (T,), 0)
    base = (hh - 1) * dv
    ss = 0.0f0
    for j in tid:T:dv
        c = core[1, base + j]
        ss += c * c
    end
    tot = _cuda_gdn_block_sum(red, ss, tid, T)
    inv = 1.0f0 / sqrt(tot / dv + eps)
    for j in tid:T:dv
        zv = z[1, base + j]
        merged[1, base + j] = (core[1, base + j] * inv) * norm_w[j] * (zv / (1.0f0 + exp(-zv)))
    end
    return
end

function Gesso.Inference._gdn_decode_core!(
    backend::CUDABackend, s, bt, spec, eps, conv_state, rec_state,
)
    mx = spec.mixer
    nk, nv = mx.num_key_heads, mx.num_value_heads
    dk, dv = mx.key_head_dim, mx.value_head_dim
    ksz = mx.conv_kernel_dim
    key_dim = nk * dk
    conv_dim = 2 * key_dim + nv * dv
    nv % nk == 0 || error("gdn decode: n_v_heads $nv not divisible by n_k_heads $nk")
    size(rec_state) == (nv, dk, dv) ||
        error("gdn decode: state $(size(rec_state)) ≠ ($nv, $dk, $dv)")
    size(conv_state) == (conv_dim, ksz - 1) ||
        error("gdn decode: conv state $(size(conv_state)) ≠ ($conv_dim, $(ksz - 1))")
    weight = _cuda_materialize_small(bt.wgdn_conv.storage)
    A_log = _cuda_materialize_small(bt.wgdn_A_log.storage)
    dt_bias = _cuda_materialize_small(bt.wgdn_dt_bias.storage)
    norm_w = _cuda_materialize_small(bt.wgdn_norm.storage)
    scale = Float32(1) / sqrt(Float32(dk))
    eps32 = Float32(eps)
    T = min(256, nextpow(2, max(32, dv)))
    Y = clamp(256 ÷ nv, 1, dk)
    Gesso.Inference.@decode_sub :gdn_conv backend begin
        @fastcuda threads = 256 blocks = cld(conv_dim, 256) _cuda_gdn_conv_decode_kernel!(
            s.conv, conv_state, s.mixed, weight, Int(ksz))
    end
    Gesso.Inference.@decode_sub :gdn_recur backend begin
        @fastcuda threads = (nv, Y) blocks = dv shmem = nv * Y * 3 * sizeof(Float32) _cuda_gdn_delta_kernel!(
            s.core, s.conv, s.a, s.b, A_log, dt_bias, rec_state,
            Int(nk), Int(div(nv, nk)), mx.v_head_tiled, Int(dk), Int(dv), scale, eps32)
        @fastcuda threads = T blocks = nv shmem = T * sizeof(Float32) _cuda_gdn_gnorm_kernel!(
            s.merged, s.core, s.z, norm_w, Int(dv), eps32)
    end
    return s
end

# --- MoE expert offload (config-agnostic) --------------------------------------
#
# Router GEMM on GPU, softmax+top-k on host (n_exp is tens-to-hundreds, seq
# is small), then for each of n_used experts: fused quantized GEMV when the
# bank is GpuQuantSlice / GgufQuantSlice, else fetch into (ff,h)/(h,ff)
# scratch and CUBLAS gemv. Same API for 256/8, 64/4, 128/8.

function Gesso.Inference._fallback_zeros_factory(::CUDABackend)
    return (dims::Tuple{Vararg{Int}}) -> CUDA.zeros(Float32, dims)
end

Gesso.Inference._profile_sync(::CUDABackend) = CUDA.synchronize()

# A decode step requests the same sequence of activation shapes every token, so
# the n-th request is served from the n-th buffer the previous step used. Each
# request within a step still gets its own buffer, and every decode kernel is
# ordered on one stream, so a buffer is never rewritten while an earlier step
# reads it. Without this the ~185 fresh CuArrays per 35B token keep VRAM near
# full and CUDA.jl's allocator runs a GC every few tokens.
#
# The buffers are views into one slab, so zeroing them (callers expect
# `zeros_like`) is one memset per step instead of one per buffer. A step whose
# requests differ from the layout gets fresh buffers for the requests that do
# not fit, and the next step lays the slab out again from what it asked for.
#
# A request the slab cannot hold (rank above four, or empty) keeps its slot in
# the layout with no slab space and a `nothing` view. It is served loose, and
# keeping the slot means one such request does not drop the layout for the
# other requests in the step.
mutable struct ArenaSlab
    slab::Union{Nothing, CuVector{Float32}}
    views::Vector{Union{Nothing, CuArray{Float32}}}
    layout::Vector{NTuple{5, Int}}
    requested::Vector{NTuple{5, Int}}
    loose::Vector{CuArray{Float32}}
end

struct DecodeArena
    s::ArenaSlab
    next::Base.RefValue{Int}
end

# (ndims, dims padded to four); 0 for a rank the slab cannot hold.
_arena_key(dims::NTuple{N, Int}) where {N} =
    N <= 4 ? (N, ntuple(k -> k <= N ? dims[k] : 0, 4)...) : (0, 0, 0, 0, 0)

# 64 Float32s: views start on 256-byte boundaries, as separate allocations would.
_arena_round(n::Int) = cld(n, 64) * 64

function _arena_relayout!(s::ArenaSlab)
    for v in s.views
        v === nothing || CUDA.unsafe_free!(v)
    end
    s.slab === nothing || CUDA.unsafe_free!(s.slab)
    empty!(s.views)
    copy!(s.layout, s.requested)
    total = 0
    for key in s.layout
        key[1] == 0 || (total += _arena_round(prod(key[2:(key[1] + 1)])))
    end
    s.slab = CuVector{Float32}(undef, max(total, 1))
    off = 0
    for key in s.layout
        if key[1] == 0
            push!(s.views, nothing)
            continue
        end
        dims = key[2:(key[1] + 1)]
        n = prod(dims)
        push!(s.views, reshape(view(s.slab, (off + 1):(off + n)), dims))
        off += _arena_round(n)
    end
    return s
end

function (a::DecodeArena)(dims::NTuple{N, Int}) where {N}
    s = a.s
    i = (a.next[] += 1)
    key = _arena_key(dims)
    push!(s.requested, key)
    if key[1] != 0 && i <= length(s.layout) && s.layout[i] == key
        return s.views[i]::CuArray{Float32, N, CUDA.DeviceMemory}
    end
    buf = CUDA.zeros(Float32, dims)
    push!(s.loose, buf)
    return buf
end

const _DECODE_ARENAS = Dict{Int, ArenaSlab}()

function Gesso.Inference._decode_zeros_like(::CUDABackend, tensors)
    s = get!(_DECODE_ARENAS, Int(CUDA.deviceid(CUDA.device()))) do
        ArenaSlab(nothing, Union{Nothing, CuArray{Float32}}[], NTuple{5, Int}[], NTuple{5, Int}[], CuArray{Float32}[])
    end
    # `loose` holds only the previous step's overflow buffers, so it is always
    # released here; the layout is rebuilt only when the requests changed.
    for b in s.loose
        CUDA.unsafe_free!(b)
    end
    empty!(s.loose)
    if !isempty(s.requested) && s.requested != s.layout
        _arena_relayout!(s)
    end
    empty!(s.requested)
    s.slab === nothing || fill!(s.slab, 0.0f0)
    return DecodeArena(s, Ref(0))
end

# Stage backend tag for CUDA work issued outside a CUDABackend method.
struct _CUDASync end
Gesso.Inference._profile_sync(::_CUDASync) = CUDA.synchronize()

# Decode scratch: one reusable device buffer per (device, call site). A buffer
# is allocated (zeroed) on first use and again only when the requested shape
# changes; a reused buffer keeps its old contents, so each call site writes
# every element it reads. Allocations are counted in the decode profile, so a
# steady-state token reports zero. Calls are sequential within a decode, so one
# buffer per key is enough; concurrent decodes on one device would share it.
struct DeviceScratch
    bufs::Dict{Tuple{Int, Symbol}, CuArray{Float32}}
end
const _DEV_SCRATCH = DeviceScratch(Dict{Tuple{Int, Symbol}, CuArray{Float32}}())

function _dev_scratch(key::Symbol, dims::Integer...)
    k = (Int(CUDA.deviceid(CUDA.device())), key)
    buf = get(_DEV_SCRATCH.bufs, k, nothing)
    if buf === nothing || size(buf) != dims
        Gesso.Inference._profile_count!(:cuda_alloc_calls, 1)
        Gesso.Inference._profile_count!(:cuda_alloc_bytes, 4 * prod(dims))
        # Every prefill/decode switch reshapes each key. Left to the GC, the
        # old buffers fill the device until an allocation fails, and the
        # retry runs a full collection (300+ ms on the model's heap). The free
        # is stream-ordered, so kernels already queued on it still finish.
        buf === nothing || CUDA.unsafe_free!(buf)
        buf = CUDA.zeros(Float32, dims...)
        _DEV_SCRATCH.bufs[k] = buf
    end
    return buf
end

# The tied head's row, normed row and logits, reused across tokens.
Gesso.Inference._head_scratch(::CUDABackend, dim::Int, vocab::Int) = (
    lastrow=_dev_scratch(:head_row, 1, dim),
    final_normed=_dev_scratch(:head_normed, 1, dim),
    logits=_dev_scratch(:head_logits, 1, vocab),
    xq_dim=nothing,
)

function _cuda_take_row_kernel!(dst2, src2, row::Int)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j <= size(dst2, 2) || return
    dst2[1, j] = src2[row, j]
    return
end

function Gesso.Inference._copy_row!(dst::CuMatrix{Float32}, src::CuMatrix{Float32}, row::Int)
    d = size(dst, 2)
    d == 0 && return dst
    nthreads = min(256, d)
    @fastcuda threads = nthreads blocks = cld(d, nthreads) _cuda_take_row_kernel!(dst, src, row)
    return dst
end

# Greedy id on the device in one pass: the first index of the maximum under
# `isless` (what host `argmax` returns), or -1 when any logit is non-finite. The
# generic path ran `all(isfinite, ·)` and `argmax` as two GPUArrays reductions,
# each allocating its partials and each waiting on the device.
const _ARGMAX_THREADS = 256

@inline _argmax_better(v, i, bv, bi) = isless(bv, v) || (!isless(v, bv) && i < bi)

# Block tree over (value, index, bad) in shared memory; thread 1 holds the result.
@inline function _argmax_block_reduce!(sv, si, sb, tid, bv, bi, bad)
    sv[tid] = bv
    si[tid] = bi
    sb[tid] = bad
    CUDA.sync_threads()
    stride = _ARGMAX_THREADS >> 1
    while stride > 0
        if tid <= stride
            v, i = sv[tid + stride], si[tid + stride]
            if _argmax_better(v, i, sv[tid], si[tid])
                sv[tid] = v
                si[tid] = i
            end
            sb[tid] |= sb[tid + stride]
        end
        CUDA.sync_threads()
        stride >>= 1
    end
    return
end

function _cuda_argmax_partial_kernel!(pv, pi, pb, x, n::Int)
    tid = Int(threadIdx().x)
    sv = CUDA.@cuStaticSharedMem(Float32, _ARGMAX_THREADS)
    si = CUDA.@cuStaticSharedMem(Int32, _ARGMAX_THREADS)
    sb = CUDA.@cuStaticSharedMem(Int32, _ARGMAX_THREADS)
    bv, bi, bad = -Inf32, typemax(Int32), Int32(0)
    i = (Int(blockIdx().x) - 1) * _ARGMAX_THREADS + tid
    step = _ARGMAX_THREADS * Int(gridDim().x)
    while i <= n
        v = x[i]
        if !isfinite(v)
            bad = Int32(1)
        elseif _argmax_better(v, Int32(i), bv, bi)
            bv, bi = v, Int32(i)
        end
        i += step
    end
    _argmax_block_reduce!(sv, si, sb, tid, bv, bi, bad)
    if tid == 1
        b = Int(blockIdx().x)
        pv[b] = sv[1]
        pi[b] = si[1]
        pb[b] = sb[1]
    end
    return
end

function _cuda_argmax_final_kernel!(out, pv, pi, pb, nparts::Int)
    tid = Int(threadIdx().x)
    sv = CUDA.@cuStaticSharedMem(Float32, _ARGMAX_THREADS)
    si = CUDA.@cuStaticSharedMem(Int32, _ARGMAX_THREADS)
    sb = CUDA.@cuStaticSharedMem(Int32, _ARGMAX_THREADS)
    bv, bi, bad = -Inf32, typemax(Int32), Int32(0)
    if tid <= nparts
        bv, bi, bad = pv[tid], pi[tid], pb[tid]
    end
    _argmax_block_reduce!(sv, si, sb, tid, bv, bi, bad)
    tid == 1 && (out[1] = sb[1] != 0 ? Int32(-1) : si[1])
    return
end

struct ArgmaxBuffers
    pv::_CuVec{Float32}
    pi::_CuVec{Int32}
    pb::_CuVec{Int32}
    out::_CuVec{Int32}
    host::Vector{Int32}
end
const _ARGMAX_BUFS = Dict{Int, ArgmaxBuffers}()

function Gesso.Inference._greedy_id(logits::CuVector{Float32})
    n = length(logits)
    0 < n < typemax(Int32) || throw(ArgumentError("greedy id: logits length $n out of range"))
    b = get!(_ARGMAX_BUFS, Int(CUDA.deviceid(CUDA.device()))) do
        ArgmaxBuffers(CuVector{Float32}(undef, _ARGMAX_THREADS), CuVector{Int32}(undef, _ARGMAX_THREADS),
            CuVector{Int32}(undef, _ARGMAX_THREADS), CuVector{Int32}(undef, 1), Vector{Int32}(undef, 1))
    end
    # ~8 logits per thread; the final pass takes one partial per thread.
    nparts = clamp(cld(n, 8 * _ARGMAX_THREADS), 1, _ARGMAX_THREADS)
    @fastcuda threads = _ARGMAX_THREADS blocks = nparts _cuda_argmax_partial_kernel!(b.pv, b.pi, b.pb, logits, n)
    @fastcuda threads = _ARGMAX_THREADS blocks = 1 _cuda_argmax_final_kernel!(b.out, b.pv, b.pi, b.pb, nparts)
    copyto!(b.host, b.out)
    id = b.host[1]
    id < 0 && throw(Gesso.Inference._nonfinite_logits())
    return Int(id) - 1
end

function _fetch_expert!(::CUDABackend, W::CuMatrix{Float32}, bank::CuArray{Float32, 3}, e::Int)
    copyto!(W, view(bank, e, :, :))
    return W
end

function _fetch_expert!(::CUDABackend, W::CuMatrix{Float32}, bank, e::Int)
    hostW = Matrix{Float32}(undef, size(W, 1), size(W, 2))
    Gesso.Inference._fetch_expert!(Gesso.CPUBackend(), hostW, bank, e)
    copyto!(W, hostW)
    return W
end

function _cuda_shared_gate_kernel!(y, sd, acc)
    t = Int((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    i = Int((blockIdx().y - 1) * blockDim().y + threadIdx().y)
    seq = size(y, 1)
    h = size(y, 2)
    (t > seq || i > h) && return
    s = Float32(1) / (Float32(1) + exp(-acc[t]))
    y[t, i] += s * sd[t, i]
    return
end

function _shared_gate_add!(::CUDABackend, y, x, sd, gvec)
    g = _cuda_materialize_small(gvec)
    g isa CuArray || (g = CuArray{Float32}(g))
    seq = size(x, 1)
    h = size(x, 2)
    seq == 1 && x isa DenseCuArray && y isa DenseCuArray && sd isa DenseCuArray &&
        return _cuda_shared_gate1!(y, x, sd, g)
    acc = _dev_scratch(:shared_gate_acc, seq)
    _blas_gemv!('N', _cu_scalar(one(eltype(y))), x, g, _cu_scalar(zero(eltype(y))), acc)
    tx = min(16, seq)
    ty = min(16, h)
    @fastcuda threads = (tx, ty) blocks = (cld(seq, tx), cld(h, ty)) _cuda_shared_gate_kernel!(
        y, sd, acc)
    return y
end

# Softmax over one router row in place, then the top `n_used` experts by
# (probability descending, index ascending) and their renormalised weights.
# That key is a strict total order, so a partial sort selects exactly the
# prefix a full sort would. Returns the weight sum before renormalisation.
function _moe_route_row!(order, experts, wbuf, row, n_exp::Int, n_used::Int)
    m = typemin(Float32)
    @inbounds for e in 1:n_exp
        row[e] > m && (m = row[e])
    end
    s = zero(Float32)
    @inbounds for e in 1:n_exp
        v = exp(row[e] - m)
        row[e] = v
        s += v
    end
    invs = one(Float32) / s
    @inbounds for e in 1:n_exp
        row[e] *= invs
        order[e] = e
    end
    partialsort!(order, 1:n_used; by=e -> (-row[e], e))
    wsum = zero(Float32)
    @inbounds for k in 1:n_used
        wsum += row[order[k]]
    end
    invw = wsum > 0 ? one(Float32) / wsum : zero(Float32)
    @inbounds for k in 1:n_used
        experts[k] = order[k]
        wbuf[k] = row[order[k]] * invw
    end
    return wsum
end

# Page-locked host staging for the host-expert MoE path. Pinned memory lets the
# router logits and the activation come back in one async D2H pair with one
# wait, and the result go up as one async H2D the stream orders before the
# next kernel. Capacity grows with `seq` and is never shrunk, so decode reuses
# one registration.
mutable struct _MoeHostXfer
    cap::Int
    n_exp::Int
    h::Int
    logits::Vector{Float32}
    x::Vector{Float32}
    y::Vector{Float32}
    xrow::Vector{Float32}
    yrow::Vector{Float32}
    order::Vector{Int}
    experts::Vector{Int}
    wbuf::Vector{Float32}
    E::Matrix{Int}
    W::Matrix{Float32}
    live::Vector{Bool}
    route::Vector{Int32}
end

const _MOE_HOST_XFER = Ref{Union{Nothing,_MoeHostXfer}}(nothing)

function _moe_host_xfer(seq::Int, n_exp::Int, h::Int, n_used::Int)
    b = _MOE_HOST_XFER[]
    if b === nothing || b.cap < seq || b.n_exp != n_exp || b.h != h || length(b.experts) != n_used
        # The previous buffers may still feed an in-flight H2D; unpinning them
        # (their finalizer) must not race it.
        b === nothing || _cuda_stream_wait()
        cap = b === nothing ? seq : max(seq, b.cap)
        lg = Vector{Float32}(undef, cap * n_exp)
        xx = Vector{Float32}(undef, cap * h)
        yy = Vector{Float32}(undef, cap * h)
        rt = Vector{Int32}(undef, cap * (2n_used + 1))
        CUDA.pin(lg)
        CUDA.pin(xx)
        CUDA.pin(yy)
        CUDA.pin(rt)
        b = _MoeHostXfer(cap, n_exp, h, lg, xx, yy, Vector{Float32}(undef, h),
            Vector{Float32}(undef, h), Vector{Int}(undef, n_exp),
            Vector{Int}(undef, n_used), Vector{Float32}(undef, n_used),
            Matrix{Int}(undef, n_used, cap), Matrix{Float32}(undef, n_used, cap),
            Vector{Bool}(undef, cap), rt)
        _MOE_HOST_XFER[] = b
    end
    return b
end

# Waits for the stream by polling it. CUDA.synchronize first runs CUDA.jl's
# early-GC heuristic, which fires on nearly every call once a large model
# fills the device; this wait is on the critical path of every host layer.
function _cuda_stream_wait(s::CuStream=CUDA.stream())
    while !CUDA.isdone(s)
        ccall(:jl_cpu_pause, Cvoid, ())
    end
    return nothing
end

function _cuda_event_wait(e::CuEvent)
    while !CUDA.isdone(e)
        ccall(:jl_cpu_pause, Cvoid, ())
    end
    return nothing
end

const _MOE_HOST_EVENT = Dict{Int,CuEvent}()
_moe_host_event() = get!(() -> CuEvent(CUDA.EVENT_DISABLE_TIMING), _MOE_HOST_EVENT, CUDA.deviceid(CUDA.device()))

# GESSO_MOE_OVERLAP=0 runs the shared expert after the copies land instead, for A/B.
_moe_overlap_enabled() = get(ENV, "GESSO_MOE_OVERLAP", "1") == "1"

# How long the CPU pool stays hot after a host-expert layer. It must cover the
# GPU work up to the next host-expert layer (about 1 to 2 ms on the 35B); a
# worker that parks in that gap makes the next layer's experts about twice as
# slow. See the hold notes in src/Runtime/cpu_pool.jl.
const _MOE_HOST_HOLD_NS = 5_000_000

# Top-k SwiGLU for one MoE layer whose expert slabs stay mmap'd on the host.
# The GPU computes the router logits; the CPU routes and runs the experts; the
# result returns to `y`. Per layer: one D2H wait and one async H2D.
function _moe_topk_host!(backend::CUDABackend, y, x, router, gate, up, down, n_used::Int;
    overlap=nothing)
    seq, h = size(x)
    n_exp = size(router, 1)
    b = _moe_host_xfer(seq, n_exp, h, n_used)
    logits = _dev_scratch(:moe_logits, seq, n_exp)
    xd = x
    if !(x isa DenseCuArray)
        xd = _dev_scratch(:moe_xall, seq, h)
        copyto!(xd, x)
    end
    s = CUDA.stream()
    hot = _hot_layer(gate)
    hot !== nothing && _hot_enter!(hot, seq)
    # Pools otherwise leave when memory runs short (_HotPurge).
    hot !== nothing && seq > _HOT_CFG[].keep_seq && _hot_release!(hot)
    batch = seq > 1 && Gesso.Inference._moe_host_batchable(gate, up, down, h, gate.ne1)
    routed = batch && _moe_prefill_route_ok(n_exp, n_used)
    Gesso.Inference.@decode_stage :moe_router backend begin
        rd = router isa CuArray ? router : CuArray{Float32}(Gesso.Inference._offload_host_array(router))
        _cuda_router_logits!(logits, x, rd)
        Gesso.Inference._profile_copy!(true)
        if routed
            _moe_route_rows_launch!(b, logits, seq, n_exp, n_used, s)
        else
            unsafe_copyto!(pointer(b.logits), pointer(logits), seq * n_exp; async=true, stream=s)
        end
        unsafe_copyto!(pointer(b.x), pointer(xd), seq * h; async=true, stream=s)
    end
    # Work queued behind the copies runs on the GPU while the CPU runs the
    # experts, so wait on the copies alone.
    copied = _moe_host_event()
    CUDA.record(copied, s)
    prefetch = hot !== nothing && seq == 1
    if overlap !== nothing && _moe_overlap_enabled()
        overlap()
        prefetch && _hot_prefetch!(hot, gate, up, down)
        Gesso.Inference.@decode_stage :moe_router backend _cuda_event_wait(copied)
    else
        prefetch && _hot_prefetch!(hot, gate, up, down)
        Gesso.Inference.@decode_stage :moe_router backend _cuda_event_wait(copied)
        overlap === nothing || overlap()
    end
    L = unsafe_wrap(Array, pointer(b.logits), (seq, n_exp))
    X = unsafe_wrap(Array, pointer(b.x), (seq, h))
    Y = unsafe_wrap(Array, pointer(b.y), (seq, h))
    gpu_add = false
    if batch
        gpu_add = _moe_host_prefill_layer!(backend, b, routed, L, X, Y, y, xd, hot, gate, up, down,
            seq, n_exp, n_used, s)
    else
        for t in 1:seq
            wsum = Gesso.Inference.@decode_stage :moe_router nothing _moe_route_row!(
                b.order, b.experts, b.wbuf, view(L, t, :), n_exp, n_used,
            )
            if !(wsum > 0)
                @inbounds for i in 1:h
                    Y[t, i] = 0.0f0
                end
                continue
            end
            @inbounds for i in 1:h
                b.xrow[i] = X[t, i]
            end
            if hot !== nothing
                if seq == 1 && y isa DenseCuArray
                    _HOT_BUSY[] = true
                    done = try
                        _hot_alloc!(hot, gate, up, n_used) &&
                            _moe_hot_token!(backend, hot, y, xd, b, gate, up, down, n_used, s)
                    finally
                        _HOT_BUSY[] = false
                    end
                    if done !== false
                        _hot_leave!(hot)
                        return done
                    end
                end
                _hot_observe!(hot, b.experts, n_used)
            end
            Gesso.Inference.@decode_stage :moe_host nothing Gesso.Inference._moe_host_experts_auto!(
                b.yrow, b.xrow, b.experts, b.wbuf, n_used, gate, up, down,
            )
            @inbounds for i in 1:h
                Y[t, i] = b.yrow[i]
            end
        end
    end
    Gesso.Runtime.pool_hold!(_MOE_HOST_HOLD_NS)
    Gesso.Inference.@decode_stage :moe_h2d backend begin
        Gesso.Inference._profile_copy!(false)
        if y isa DenseCuArray
            unsafe_copyto!(pointer(y), pointer(b.y), seq * h; async=true, stream=s)
        else
            copyto!(y, Y)
        end
    end
    if gpu_add
        Gesso.Inference.@decode_stage :moe_qgemv nothing _hot_prefill_add!(y, seq, h, n_used)
        _HOT_VERIFY[] && _hot_prefill_verify!(y, b, X, gate, up, down, n_exp)
    end
    hot !== nothing && _hot_leave!(hot)
    return y
end

# A prompt's tokens through one host layer: route every token, then run the
# experts grouped by expert so each weight row leaves DRAM once per layer
# instead of once per token that picked it. Bit-identical to the token loop
# unless the layer is hot, when the GPU takes part of the pairs and the result
# is true: Y then holds the CPU's partial sum and the caller adds the GPU's.
function _moe_host_prefill_layer!(backend, b::_MoeHostXfer, routed::Bool, L, X, Y, y, xd, hot, gate, up, down,
    seq::Int, n_exp::Int, n_used::Int, s)
    if routed
        Gesso.Inference.@decode_stage :moe_router nothing _moe_route_rows_unpack!(b, seq, n_used)
    end
    for t in 1:seq
        ex = view(b.E, :, t)
        if !routed
            wsum = Gesso.Inference.@decode_stage :moe_router nothing _moe_route_row!(
                b.order, ex, view(b.W, :, t), view(L, t, :), n_exp, n_used,
            )
            b.live[t] = wsum > 0
        end
        b.live[t] && hot !== nothing && _hot_observe!(hot, ex, n_used)
    end
    hot !== nothing && y isa DenseCuArray &&
        _moe_hot_prefill!(backend, hot, xd, b, X, Y, gate, up, down, seq, n_exp, n_used, s) &&
        return true
    Gesso.Inference.@decode_stage :moe_host nothing Gesso.Inference._moe_host_prefill!(
        Y, X, b.E, b.W, b.live, n_exp, gate, up, down,
    )
    return false
end

# Device routing for decode on device-resident banks. Routing on the host meant
# a logits D2H the host waits on, then the expert table, the weights and three
# metadata blocks back up, all in every GPU MoE layer. That is 19 host
# round trips per Ornith-35B token, and each one stops the host queueing work
# ahead of the GPU. GESSO_MOE_DEVICE_ROUTE=0 restores the host route.
_moe_device_route_enabled() = get(ENV, "GESSO_MOE_DEVICE_ROUTE", "1") == "1"

const _MOE_ROUTE_MAX_EXP = 1024
const _MOE_ROUTE_MAX_USED = 32

# Per-slab launch metadata, [offsets; columns] as _cuda_launch_qgemv! reads it.
struct _MoeRouteMeta
    g::_CuVec{Int32}
    u::_CuVec{Int32}
    d::_CuVec{Int32}
end
const _MOE_ROUTE_META = Dict{Tuple{Int,Int},_MoeRouteMeta}()

function _moe_route_meta(n_used::Int)
    k = (Int(CUDA.deviceid(CUDA.device())), n_used)
    get!(_MOE_ROUTE_META, k) do
        _MoeRouteMeta(CUDA.zeros(Int32, 2n_used), CUDA.zeros(Int32, 2n_used), CUDA.zeros(Int32, 2n_used))
    end
end

# One block routes one row and matches _moe_route_row! bit for bit. The max is
# order-free, so it reduces in a tree. The exp sum stays serial in index order.
# exp is Base's own Float32 kernel, not the __nv_expf CUDA substitutes for it;
# an exhaustive check over every non-positive Float32 down to -110 found no
# mismatch. The top-k takes n_used argmax rounds keyed on (probability
# descending, index ascending), the same strict order the host partial sort
# uses. Taken experts drop to -1, below any probability. Row t of `logits`
# leaves its top k in si (expert) and sv (probability), best first. Needs a
# 256-thread block.
@inline function _moe_route_block!(p, bv, bi, sv, si, logits, t::Int, n_exp::Int32, n_used::Int32)
    tid = threadIdx().x
    nt = blockDim().x
    @inbounds begin
        m = typemin(Float32)
        e = tid
        while e <= n_exp
            v = logits[t, e]
            p[e] = v
            v > m && (m = v)
            e += nt
        end
        bv[tid] = m
        CUDA.sync_threads()
        stride = nt >> 1
        while stride > 0
            if tid <= stride
                o = bv[tid + stride]
                o > bv[tid] && (bv[tid] = o)
            end
            CUDA.sync_threads()
            stride >>= 1
        end
        m = bv[1]
        e = tid
        while e <= n_exp
            p[e] = Base.Math.exp_impl(p[e] - m, Val(:ℯ))
            e += nt
        end
        CUDA.sync_threads()
        if tid == 1
            s = 0.0f0
            for j in 1:n_exp
                s += p[j]
            end
            bv[1] = 1.0f0 / s
        end
        CUDA.sync_threads()
        invs = bv[1]
        e = tid
        while e <= n_exp
            p[e] *= invs
            e += nt
        end
        CUDA.sync_threads()
        for k in 1:n_used
            best = -2.0f0
            besti = typemax(Int32)
            e = tid
            while e <= n_exp
                v = p[e]
                if v > best
                    best = v
                    besti = Int32(e)
                end
                e += nt
            end
            bv[tid] = best
            bi[tid] = besti
            CUDA.sync_threads()
            stride = nt >> 1
            while stride > 0
                if tid <= stride
                    ov = bv[tid + stride]
                    oi = bi[tid + stride]
                    if ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])
                        bv[tid] = ov
                        bi[tid] = oi
                    end
                end
                CUDA.sync_threads()
                stride >>= 1
            end
            if tid == 1
                w = bi[1]
                si[k] = w
                sv[k] = p[w]
                p[w] = -1.0f0
            end
            CUDA.sync_threads()
        end
    end
    return nothing
end

macro _moe_route_shared()
    return esc(quote
        p = CUDA.@cuStaticSharedMem(Float32, _MOE_ROUTE_MAX_EXP)
        bv = CUDA.@cuStaticSharedMem(Float32, 256)
        bi = CUDA.@cuStaticSharedMem(Int32, 256)
        sv = CUDA.@cuStaticSharedMem(Float32, _MOE_ROUTE_MAX_USED)
        si = CUDA.@cuStaticSharedMem(Int32, _MOE_ROUTE_MAX_USED)
    end)
end

function _moe_route_kernel!(ws, mg, mu, md, logits, n_exp::Int32, n_used::Int32,
                            bpe_g::Int32, bpe_u::Int32, bpe_d::Int32)
    @_moe_route_shared
    _moe_route_block!(p, bv, bi, sv, si, logits, 1, n_exp, n_used)
    @inbounds begin
        if threadIdx().x == 1
            wsum = 0.0f0
            for k in 1:n_used
                wsum += sv[k]
            end
            invw = wsum > 0 ? 1.0f0 / wsum : 0.0f0
            for k in 1:n_used
                ws[k] = sv[k] * invw
                e0 = si[k] - Int32(1)
                mg[k] = e0 * bpe_g
                mu[k] = e0 * bpe_u
                md[k] = e0 * bpe_d
                mg[n_used + k] = k
                mu[n_used + k] = k
                md[n_used + k] = k
            end
        end
    end
    return
end

# Prefill routing: block t routes token t into column t of R as
# [experts; weight bits; wsum bits], the same values _moe_route_row! leaves in
# E, W and its return. The host spent about 5 us of exp and sort per token per
# layer on this (170 to 340 ms of a P256 prefill on the Ornith-35B), and the
# table is a sixteenth the size of the logits it replaces on the bus.
# GESSO_MOE_PREFILL_ROUTE=0 routes on the host again.
function _moe_route_rows_kernel!(R, logits, n_exp::Int32, n_used::Int32)
    @_moe_route_shared
    t = Int(blockIdx().x)
    _moe_route_block!(p, bv, bi, sv, si, logits, t, n_exp, n_used)
    @inbounds if threadIdx().x == 1
        wsum = 0.0f0
        for k in 1:n_used
            wsum += sv[k]
        end
        invw = wsum > 0 ? 1.0f0 / wsum : 0.0f0
        for k in 1:n_used
            R[k, t] = si[k]
            R[n_used + k, t] = reinterpret(Int32, sv[k] * invw)
        end
        R[2n_used + 1, t] = reinterpret(Int32, wsum)
    end
    return
end

_moe_prefill_route_ok(n_exp::Int, n_used::Int) = get(ENV, "GESSO_MOE_PREFILL_ROUTE", "1") == "1" &&
    n_exp <= _MOE_ROUTE_MAX_EXP && n_used <= _MOE_ROUTE_MAX_USED

const _MOE_ROUTE_ROWS = Ref{Union{Nothing,_CuVec{Int32}}}(nothing)

# Queues the route table and its copy into b.route on stream s.
function _moe_route_rows_launch!(b::_MoeHostXfer, logits, seq::Int, n_exp::Int, n_used::Int, s)
    w = 2n_used + 1
    R = _MOE_ROUTE_ROWS[] = _hotp_dev(_MOE_ROUTE_ROWS[], Int32, w * seq)
    Rm = unsafe_wrap(CuArray, pointer(R), (w, seq))
    @fastcuda threads = 256 blocks = seq _moe_route_rows_kernel!(Rm, logits, Int32(n_exp), Int32(n_used))
    unsafe_copyto!(pointer(b.route), pointer(R), w * seq; async=true, stream=s)
    return nothing
end

# After the copy lands: the table into b.E, b.W and b.live.
function _moe_route_rows_unpack!(b::_MoeHostXfer, seq::Int, n_used::Int)
    w = 2n_used + 1
    R = b.route
    @inbounds for t in 1:seq
        o = (t - 1) * w
        for k in 1:n_used
            b.E[k, t] = R[o + k]
            b.W[k, t] = reinterpret(Float32, R[o + n_used + k])
        end
        b.live[t] = reinterpret(Float32, R[o + w]) > 0
    end
    return nothing
end

function _moe_device_route_ok(y, x, router, gate, up, down, n_used::Int)
    G = Gesso.Inference.GpuQuantSlice
    return _moe_device_route_enabled() && size(x, 1) == 1 &&
           x isa DenseCuArray && y isa DenseCuArray && router isa CuArray &&
           gate isa G && up isa G && down isa G &&
           gate.data isa CuVector && up.data isa CuVector && down.data isa CuVector &&
           size(router, 1) <= _MOE_ROUTE_MAX_EXP && n_used <= _MOE_ROUTE_MAX_USED &&
           down.n_exp * down.bytes_per_expert <= typemax(Int32) &&
           gate.n_exp * max(gate.bytes_per_expert, up.bytes_per_expert) <= typemax(Int32)
end

# Decode MoE for one token with routing, expert projections and the weighted
# combine all queued on the stream. Same kernels and cuBLAS calls as the
# host-routed quant path below, so the output is bit-identical.
function _moe_topk_device!(backend::CUDABackend, y, x, router, gate, up, down, n_used::Int)
    h = size(x, 2)
    n_exp = size(router, 1)
    ff = size(gate, 2)
    Gesso.Inference.@decode_stage :moe_alloc _CUDASync() begin
        logits = _dev_scratch(:moe_logits, 1, n_exp)
        Gbat = _dev_scratch(:moe_gbat, ff, n_used)
        Ubat = _dev_scratch(:moe_ubat, ff, n_used)
        Hbat = _dev_scratch(:moe_hbat, ff, n_used)
        Dbat = _dev_scratch(:moe_dbat, h, n_used)
        ws = _dev_scratch(:moe_ws, n_used)
    end
    rt = _moe_route_meta(n_used)
    Gesso.Inference.@decode_stage :moe_router backend begin
        _cuda_router_logits!(logits, x, router)
        @fastcuda threads = 256 _moe_route_kernel!(
            ws, rt.g, rt.u, rt.d, logits, Int32(n_exp), Int32(n_used),
            Int32(gate.bytes_per_expert), Int32(up.bytes_per_expert), Int32(down.bytes_per_expert),
        )
    end
    xt = reshape(x, h, 1)
    Gesso.Inference.@decode_stage :moe_qgemv backend begin
        _cuda_launch_qgemv!(Gbat, xt, gate.data, gate.ggml_type, rt.g, n_used, n_used, gate.ne0, gate.ne1, 1.0f0, 0.0f0)
        # gate and up read the same x, so up reuses the q8_1 copy gate just made
        _cuda_launch_qgemv!(Ubat, xt, up.data, up.ggml_type, rt.u, n_used, n_used, up.ne0, up.ne1, 1.0f0, 0.0f0;
            quantize=up.ne0 != gate.ne0 || gate.ggml_type == Gesso.Inference.GGML_TYPE_F32)
        _silu_mul!(backend, Hbat, Ubat, Gbat)
        _cuda_launch_qgemv!(Dbat, Hbat, down.data, down.ggml_type, rt.d, n_used, n_used, down.ne0, down.ne1, 1.0f0, 0.0f0)
        _cuda_combine!(reshape(y, h), Dbat, ws)
    end
    return y
end

# Prefill on a device-resident MoE layer. The token loop below spends about
# 230 us per token per layer on the Ornith-35B, almost all of it host launch
# and metadata traffic. Here the host routes every token once, then each slab
# runs as one launch over all (token, expert) pairs sorted by expert, so an
# expert's weights stay in L2 across the tokens that picked it. Each pair goes
# through the same kernels as the token loop and the combine sums a token's
# experts in k order, so the result matches the loop bit for bit.
# GESSO_MOE_DEVICE_PREFILL=0 restores the token loop.
_moe_device_prefill_enabled() = get(ENV, "GESSO_MOE_DEVICE_PREFILL", "1") == "1"

# Tokens per launch group. Bounds the D buffer at h * chunk * n_used floats.
_moe_device_prefill_chunk() = parse(Int, get(ENV, "GESSO_MOE_DEVICE_PREFILL_CHUNK", "512"))

const _MOE_PAIR_SEG = 1024

mutable struct _MoePairBatch
    meta_h::Vector{Int32}    # pinned: per chunk, per launch [up off; down off; gate off; columns; tokens], then the combine table
    meta_d::Union{Nothing,_CuVec{Int32}}
    xt::Union{Nothing,_CuVec{Float32}}
    g::Union{Nothing,_CuVec{Float32}}
    u::Union{Nothing,_CuVec{Float32}}
    a::Union{Nothing,_CuVec{Float32}}
    d::Union{Nothing,_CuVec{Float32}}
    cnt::Vector{Int}
    first::Vector{Int}
    fill::Vector{Int}
end

const _MOE_PAIR_BATCH = _MoePairBatch(Int32[], nothing, nothing, nothing, nothing, nothing, nothing,
    Int[], Int[], Int[])

# GPUArrays' permutedims! launches on vec(src), a derived array that holds a
# reference to src's memory until a GC finalizes it, so the arena's unsafe_free!
# of a prefill FFN input freed nothing: a 4096-token prompt carried all 32 MiB
# of them into decode, a pool and a half. An unowned alias leaves vec nothing
# to hold.
function _transpose_into!(dst::DenseCuArray, x::DenseCuArray)
    GC.@preserve x permutedims!(dst, unsafe_wrap(CuArray, pointer(x), size(x)), (2, 1))
    return dst
end
_transpose_into!(dst, x) = permutedims!(dst, x, (2, 1))

function _moe_device_prefill_ok(y, x, router, gate, up, down, n_used::Int)
    G = Gesso.Inference.GpuQuantSlice
    I = Gesso.Inference
    _moe_device_prefill_enabled() && size(x, 1) > 1 || return false
    (x isa DenseCuArray && y isa DenseCuArray) || return false
    (gate isa G && up isa G && down isa G) || return false
    (gate.data isa CuVector && up.data isa CuVector && down.data isa CuVector) || return false
    for sl in (gate, up, down)
        sl.ggml_type in (
            I.GGML_TYPE_IQ2_XS, I.GGML_TYPE_IQ3_XXS, I.GGML_TYPE_IQ4_XS,
            I.GGML_TYPE_Q2_K, I.GGML_TYPE_Q3_K, I.GGML_TYPE_Q4_K,
        ) || return false
        sl.n_exp * sl.bytes_per_expert <= typemax(Int32) || return false
    end
    # up reuses the q8_1 activations gate quantized
    return up.ne0 == gate.ne0
end

function _moe_pair_combine_kernel!(y, D, meta, gpoff::Int, woff::Int, n_used::Int, h::Int, t0::Int)
    i = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    t = Int(blockIdx().y)
    i <= h || return
    base = (t - 1) * n_used
    acc = 0.0f0
    @inbounds for k in 1:n_used
        g = Int(meta[gpoff + base + k])
        g > 0 && (acc += D[i, g] * reinterpret(Float32, meta[woff + base + k]))
    end
    @inbounds y[t0 + t, i] = acc
    return
end

function _moe_topk_device_prefill!(backend::CUDABackend, y, x, router, gate, up, down, n_used::Int)
    seq, h = size(x)
    n_exp = size(router, 1)
    ff = size(gate, 2)
    tc = min(seq, max(1, _moe_device_prefill_chunk()))
    seg = _MOE_PAIR_SEG
    s = CUDA.stream()
    b = _moe_host_xfer(seq, n_exp, h, n_used)
    B = _MOE_PAIR_BATCH
    logits = _dev_scratch(:moe_logits, seq, n_exp)
    routed = _moe_prefill_route_ok(n_exp, n_used)
    Gesso.Inference.@decode_stage :moe_router backend begin
        rd = router isa CuArray ? router : CuArray{Float32}(Gesso.Inference._offload_host_array(router))
        _cuda_router_logits!(logits, x, rd)
        Gesso.Inference._profile_copy!(true)
        if routed
            _moe_route_rows_launch!(b, logits, seq, n_exp, n_used, s)
        else
            unsafe_copyto!(pointer(b.logits), pointer(logits), seq * n_exp; async=true, stream=s)
        end
        copied = _moe_host_event()
        CUDA.record(copied, s)
        _cuda_event_wait(copied)
    end
    # The stream has drained up to the copy, so nothing queued still reads the
    # buffers that grow below, or the pinned metadata the previous layer sent.
    np_max = min(seg, tc * n_used)
    nmeta = 7 * seq * n_used
    Gesso.Inference.@decode_stage :moe_alloc nothing begin
        if length(B.meta_h) < nmeta
            B.meta_h = Vector{Int32}(undef, 2nmeta)
            CUDA.pin(B.meta_h)
        end
        B.meta_d = _hotp_dev(B.meta_d, Int32, nmeta)
        B.xt = _hotp_dev(B.xt, Float32, h * seq)
        B.g = _hotp_dev(B.g, Float32, ff * np_max)
        B.u = _hotp_dev(B.u, Float32, ff * np_max)
        B.a = _hotp_dev(B.a, Float32, ff * np_max)
        B.d = _hotp_dev(B.d, Float32, h * tc * n_used)
        _cuda_q8_buffers!(_CUDA_Q8_EXPERT, max(tc * (h ÷ 32), np_max * (ff ÷ 32)))
    end

    if routed
        Gesso.Inference.@decode_stage :moe_router nothing _moe_route_rows_unpack!(b, seq, n_used)
    else
        L = unsafe_wrap(Array, pointer(b.logits), (seq, n_exp))
        Gesso.Inference.@decode_stage :moe_router nothing for t in 1:seq
            wsum = _moe_route_row!(b.order, view(b.E, :, t), view(b.W, :, t), view(L, t, :), n_exp, n_used)
            b.live[t] = wsum > 0
        end
    end

    xt = unsafe_wrap(CuArray, pointer(B.xt), (h, seq))
    _transpose_into!(xt, x)
    meta = B.meta_h
    cnt = resize!(B.cnt, n_exp)
    pfirst = resize!(B.first, n_exp)
    fillc = resize!(B.fill, n_exp)
    bg = gate.bytes_per_expert
    bu = up.bytes_per_expert
    bd = down.bytes_per_expert
    mo = 0
    GC.@preserve meta for t0 in 1:tc:seq
        t1 = min(seq, t0 + tc - 1)
        nt = t1 - t0 + 1
        fill!(cnt, 0)
        @inbounds for t in t0:t1
            b.live[t] || continue
            for k in 1:n_used
                cnt[b.E[k, t]] += 1
            end
        end
        g = 1
        @inbounds for e in 1:n_exp
            pfirst[e] = g
            fillc[e] = 0
            g += cnt[e]
        end
        np = g - 1
        gpoff = mo + 5np
        woff = gpoff + nt * n_used
        @inbounds for t in t0:t1, k in 1:n_used
            i = (t - t0) * n_used + k
            meta[woff + i] = reinterpret(Int32, b.W[k, t])
            if b.live[t]
                e = b.E[k, t]
                gi = pfirst[e] + fillc[e]
                fillc[e] += 1
                meta[gpoff + i] = Int32(gi)
                # the launch holding pair gi starts at pair a and has n pairs
                a = ((gi - 1) ÷ seg) * seg + 1
                n = min(seg, np - a + 1)
                m0 = mo + 5(a - 1)
                l = gi - a + 1
                e0 = e - 1
                meta[m0 + l] = Int32(e0 * bu)
                meta[m0 + n + l] = Int32(e0 * bd)
                meta[m0 + 2n + l] = Int32(e0 * bg)
                meta[m0 + 3n + l] = Int32(l)
                meta[m0 + 4n + l] = Int32(t - t0 + 1)
            else
                meta[gpoff + i] = 0
            end
        end
        nm = woff + nt * n_used - mo
        unsafe_copyto!(pointer(B.meta_d, mo + 1), pointer(meta, mo + 1), nm; async=true, stream=s)
        xc = unsafe_wrap(CuArray, pointer(B.xt, (t0 - 1) * h + 1), (h, nt))
        Gesso.Inference.@decode_stage :moe_qgemv backend begin
            a = 1
            while a <= np
                n = min(seg, np - a + 1)
                m0 = mo + 5(a - 1)
                # One block of 5n per launch, ordered so every slab's table
                # starts at its own offsets and reads the shared columns at
                # positive distances.
                mu = unsafe_wrap(CuArray, pointer(B.meta_d, m0 + 1), 5n)
                md = unsafe_wrap(CuArray, pointer(B.meta_d, m0 + n + 1), 4n)
                mg = unsafe_wrap(CuArray, pointer(B.meta_d, m0 + 2n + 1), 3n)
                G = unsafe_wrap(CuArray, pointer(B.g), (ff, n))
                U = unsafe_wrap(CuArray, pointer(B.u), (ff, n))
                A = unsafe_wrap(CuArray, pointer(B.a), (ff, n))
                D = unsafe_wrap(CuArray, pointer(B.d, (a - 1) * h + 1), (h, n))
                _cuda_launch_qgemv!(G, xc, gate.data, gate.ggml_type, mg, n, n, gate.ne0, gate.ne1,
                    1.0f0, 0.0f0; xcoloff=2n)
                _cuda_launch_qgemv!(U, xc, up.data, up.ggml_type, mu, 3n, n, up.ne0, up.ne1,
                    1.0f0, 0.0f0; quantize=false, xcoloff=4n)
                _silu_mul!(backend, A, U, G)
                _cuda_launch_qgemv!(D, A, down.data, down.ggml_type, md, 2n, n, down.ne0, down.ne1,
                    1.0f0, 0.0f0)
                a += n
            end
            Dall = unsafe_wrap(CuArray, pointer(B.d), (h, max(np, 1)))
            @fastcuda threads = 256 blocks = (cld(h, 256), nt) _moe_pair_combine_kernel!(
                y, Dall, B.meta_d, gpoff, woff, n_used, h, t0 - 1)
        end
        mo += nm
    end
    return y
end

function _moe_topk_swiglu!(
    backend::CUDABackend,
    y::AbstractMatrix,
    x::AbstractMatrix,
    router,
    gate,
    up,
    down,
    n_used::Int;
    overlap=nothing,
)
    seq, h = size(x)
    n_exp = size(router, 1)
    ff = size(gate, 2)
    0 < n_used <= n_exp || error("_moe_topk_swiglu!: n_used $n_used out of 1..$n_exp")
    size(y) == (seq, h) || error("_moe_topk_swiglu!: y shape $(size(y)) ≠ ($seq, $h)")
    size(router) == (n_exp, h) ||
        error("_moe_topk_swiglu!: router $(size(router)) ≠ ($n_exp, $h)")
    size(gate) == (n_exp, ff, h) ||
        error("_moe_topk_swiglu!: gate $(size(gate)) ≠ ($n_exp, $ff, $h)")
    size(up) == (n_exp, ff, h) ||
        error("_moe_topk_swiglu!: up $(size(up)) ≠ ($n_exp, $ff, $h)")
    size(down) == (n_exp, h, ff) ||
        error("_moe_topk_swiglu!: down $(size(down)) ≠ ($n_exp, $h, $ff)")

    # Banks that are mmap'd on the host (GgufQuantSlice in all three slabs)
    # run on the CPU; only the activation goes D2H and the output goes H2D.
    # Default on; GESSO_MOE_HOST_EXPERTS=0 uploads the active experts instead.
    # Ornith-35B Q2_K_XL, warm profile: host mode 132 ms/token (moe_host 35.9 ms
    # over 21 calls) against 477 ms/token before the batched Q8_K host path.
    # Gate decode: 8.48 tok/s host vs 6.80 tok/s upload, both ids_ok=true.
    if _moe_host_experts_enabled() &&
       gate isa Gesso.Inference.GgufQuantSlice &&
       up isa Gesso.Inference.GgufQuantSlice &&
       down isa Gesso.Inference.GgufQuantSlice
        return _moe_topk_host!(backend, y, x, router, gate, up, down, n_used; overlap)
    end
    overlap === nothing || overlap()
    if _moe_device_route_ok(y, x, router, gate, up, down, n_used)
        return _moe_topk_device!(backend, y, x, router, gate, up, down, n_used)
    end
    if _moe_device_prefill_ok(y, x, router, gate, up, down, n_used)
        return _moe_topk_device_prefill!(backend, y, x, router, gate, up, down, n_used)
    end

    Gesso.Inference.@decode_stage :moe_alloc _CUDASync() begin
        logits = _dev_scratch(:moe_logits, seq, n_exp)
        fill!(y, zero(eltype(y)))
        gacc = _dev_scratch(:moe_gacc, ff)
        uacc = _dev_scratch(:moe_uacc, ff)
        hact = _dev_scratch(:moe_hact, ff)
        yrow = _dev_scratch(:moe_yrow, h)
        xt = _dev_scratch(:moe_xt, h)
    end

    Gesso.Inference.@decode_stage :moe_router backend begin
        rd = router isa CuArray ? router : CuArray{Float32}(Gesso.Inference._offload_host_array(router))
        _cuda_router_logits!(logits, x, rd)
        Gesso.Inference._profile_copy!(true)
        logits_h = Array(logits)
    end

    order = Vector{Int}(undef, n_exp)
    quant = Gesso.Inference._is_quant_expert_bank(gate)
    cached = gate isa Gesso.Inference.CachedExpertBank
    Wg = Wu = Wd = nothing
    Gbat = Ubat = Hbat = Dbat = ws = nothing
    experts = Vector{Int}(undef, n_used)
    wbuf = Vector{Float32}(undef, n_used)
    hit_rank = Vector{Int32}(undef, n_used)
    hit_slot = Vector{Int32}(undef, n_used)
    miss_rank = Vector{Int32}(undef, n_used)
    miss_exp = Vector{Int}(undef, n_used)
    if quant || cached
        Gesso.Inference.@decode_stage :moe_alloc _CUDASync() begin
            Gbat = _dev_scratch(:moe_gbat, ff, n_used)
            Ubat = _dev_scratch(:moe_ubat, ff, n_used)
            Hbat = _dev_scratch(:moe_hbat, ff, n_used)
            Dbat = _dev_scratch(:moe_dbat, h, n_used)
            ws = _dev_scratch(:moe_ws, n_used)
        end
    else
        Gesso.Inference.@decode_stage :moe_alloc _CUDASync() begin
            Wg = _dev_scratch(:moe_wg, ff, h)
            Wu = _dev_scratch(:moe_wu, ff, h)
            Wd = _dev_scratch(:moe_wd, h, ff)
        end
    end

    for t in 1:seq
        row = view(logits_h, t, :)
        wsum = Gesso.Inference.@decode_stage :moe_router nothing _moe_route_row!(
            order, experts, wbuf, row, n_exp, n_used,
        )
        invw = one(Float32) / wsum
        wsum > 0 || continue
        fill!(yrow, zero(Float32))
        if cached
            copyto!(xt, view(x, t, :))
            layer = gate.layer
            nh = 0
            nm = 0
            for k in 1:n_used
                e = experts[k]
                s = layer.slot_of[e]
                if s > 0
                    nh += 1
                    hit_rank[nh] = Int32(k)
                    hit_slot[nh] = s
                else
                    nm += 1
                    miss_rank[nm] = Int32(k)
                    miss_exp[nm] = e
                end
            end
            layer.hits += nh
            layer.misses += nm
            Gesso.Inference._profile_count!(:expert_hits, nh)
            Gesso.Inference._profile_count!(:expert_misses, nm)
            io = Gesso.Inference.routing_trace_io()
            io === nothing || Gesso.Inference.routing_trace_record!(io, layer, view(experts, 1:n_used))
            xg = reshape(xt, h, 1)
            _cuda_cached_project!(Gbat, xg, layer, 1, hit_rank, hit_slot, nh, miss_rank, miss_exp, nm)
            _cuda_cached_project!(Ubat, xg, layer, 2, hit_rank, hit_slot, nh, miss_rank, miss_exp, nm)
            Gesso.Inference.@decode_stage :moe_qgemv backend _silu_mul!(backend, Hbat, Ubat, Gbat)
            _cuda_cached_project!(Dbat, Hbat, layer, 3, hit_rank, hit_slot, nh, miss_rank, miss_exp, nm)
            Gesso.Inference.@decode_stage :moe_qupload backend begin
                Gesso.Inference._profile_copy!(false)
                copyto!(ws, wbuf)
            end
            Gesso.Inference.@decode_stage :moe_qgemv backend _cuda_combine!(yrow, Dbat, ws)
        elseif quant
            copyto!(xt, view(x, t, :))
            _cuda_qgemv_experts!(Gbat, xt, gate, experts)
            _cuda_qgemv_experts!(Ubat, xt, up, experts)
            Gesso.Inference.@decode_stage :moe_qgemv backend _silu_mul!(backend, Hbat, Ubat, Gbat)
            _cuda_qgemv_experts!(Dbat, Hbat, down, experts)
            Gesso.Inference.@decode_stage :moe_qupload backend begin
                Gesso.Inference._profile_copy!(false)
                copyto!(ws, wbuf)
            end
            Gesso.Inference.@decode_stage :moe_qgemv backend _cuda_combine!(yrow, Dbat, ws)
        else
            copyto!(xt, view(x, t, :))
            for k in 1:n_used
                e = order[k]
                w = row[e] * invw
                _fetch_expert!(backend, Wg, gate, e)
                _fetch_expert!(backend, Wu, up, e)
                _fetch_expert!(backend, Wd, down, e)
                _blas_gemv!('N', _cu_scalar(one(Float32)), Wg, xt, _cu_scalar(zero(Float32)), gacc)
                _blas_gemv!('N', _cu_scalar(one(Float32)), Wu, xt, _cu_scalar(zero(Float32)), uacc)
                _silu_mul!(backend, hact, uacc, gacc)
                _blas_gemv!('N', w, Wd, hact, one(Float32), yrow)
            end
        end
        view(y, t, :) .= yrow
    end
    return y
end
