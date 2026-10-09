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
        @cuda threads = (nthreads,) blocks = (cld(d, nthreads),) _cuda_embed_kernel!(
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

function _cuda_rmsnorm_storage!(dsts::AbstractArray, xs::AbstractArray, scales::AbstractArray, eps)
    native_eps=eltype(xs)(eps)
    isfinite(native_eps) && native_eps>0 || throw(gesso_error(ERR_INVALID_PLAN,
        "rmsnorm!: eps must be representable, finite and positive in storage dtype"))
    nd = ndims(xs)
    d = size(xs, nd)                              # last dim is the feature dim
    length(scales) == d ||
        throw(gesso_error(ERR_INVALID_PLAN, "rmsnorm!: scale length $(length(scales)) ≠ feature dim $d"; op = :rmsnorm!))
    if nd == 2 && size(xs, 1) == 1
        @cuda threads=256 blocks=1 _cuda_rmsnorm_row_kernel!(dsts, xs, scales, eps, d)
        return dsts
    end
    # Retain the legacy mixed-precision normalization to meet its fixed
    # small-fixture accuracy budget. Comparison receipts name this explicitly.
    rms = sqrt.(sum(abs2, xs; dims = nd) ./ d .+ eps)
    if nd == 2
        rows = size(dsts, 1)
        d == 0 && return dsts
        nthreads = min(256, d)
        @cuda threads = (nthreads,) blocks = (rows,) _cuda_rmsnorm_apply_kernel!(
            dsts, xs, rms, scales, rows, d)
        return dsts
    end
    # generic rank: the previous chain, unchanged (nothing in Gesso is 3-D)
    stail = reshape(scales, (ntuple(_ -> 1, nd - 1)..., d))
    dsts .= (xs ./ rms) .* stail
    return dsts
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
    nrm = Float64(rms[r, 1])
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
    pos_d = CuArray{Int}(positions)               # kernels never auto-transfer
    use_inv = inv_freq !== nothing
    freq_d = use_inv ? CuArray{Float32}(Float32.(inv_freq)) : CUDA.zeros(Float32, 1)
    for x in (qs, ks)
        seq, nheads, _ = size(x)
        nthreads = min(256, seq)
        nblocks = cld(seq, nthreads)
        # 2-D grid: x = sequence tiles, y = head (K rotates its OWN heads)
        @cuda threads = (nthreads,) blocks = (nblocks, nheads) _cuda_rope_kernel!(
            x, pos_d, tθ, interleaved, rd, use_inv, freq_d)
    end
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
    @cuda threads = 1 blocks = L _cuda_softmax_kernel!(scores, dst)
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
    @cuda threads = (min(256, n),) blocks = (cld(n, 256),) _cuda_swiglu_kernel!(
        dst.storage, gate.storage, up.storage)
    return dst
end

function _cuda_matmul_storage!(dst::AbstractArray, x::AbstractArray, w::AbstractArray)
    CUDA.CUBLAS.gemm!('N', 'T', one(eltype(dst)), x, w, zero(eltype(dst)), dst)
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
    _cuda_embed_table_check!(table)
    return _cuda_embedding_lookup!(dst, table, tokens)
end

function _cuda_host_offload_embed!(dst, table, tokens)
    Gesso._validate_embedding_inputs(dst, table, tokens)
    tab = table.storage.data
    d = size(tab, 2)
    host = Matrix{Float32}(undef, length(tokens), d)
    @inbounds for (t, tok) in enumerate(tokens), j in 1:d
        host[t, j] = Float32(tab[tok + 1, j])
    end
    copyto!(dst.storage, host)
    return dst
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
    @cuda threads = (nthreads,) blocks = (cld(rows, nthreads), n_heads) _cuda_split_kernel!(
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
    @cuda threads = (nthreads,) blocks = (cld(rows, nthreads), n_heads) _cuda_merge_kernel!(
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
    @cuda threads = (nthreads,) blocks = (cld(K, nthreads), nkv) _cuda_repeat_kernel!(
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
    @cuda threads = (1,) blocks = (size(dst, 2), size(dst, 3)) _cuda_copy_row_kernel!(dst, src, i)
    return dst
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
    @cuda threads = (nthreads,) blocks = (cld(take, nthreads), size(dest, 2), size(dest, 3)) _cuda_copy_rows_kernel!(
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
    @cuda threads = (min(256, n),) blocks = (cld(n, 256),) _cuda_add_kernel!(dst, src)
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
    @cuda threads = (min(256, n),) blocks = (cld(n, 256),) _cuda_scale_kernel!(dst, s)
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
    @cuda threads = (nthreads,) blocks = (cld(d, nthreads),) _cuda_hidden_row_kernel!(dst, src, row)
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
    @cuda threads = (32, _F16W_KSPLIT) blocks = (cld(m, 32), n) _cuda_f16w_kernel!(ds, xs, ws, k, m)
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
(default 6144) is exhausted, smallest first. Overflow becomes `HostOffload`
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
_moe_host_experts_enabled() = get(ENV, "GESSO_MOE_HOST_EXPERTS", "0") == "1"

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
    # never competes for GPU budget and never has a device copy.
    function host_table_t!(t)
        hasproperty(t, :storage) || return
        s = t.storage
        (s === nothing || haskey(conv, s)) && return
        conv[s] = s isa Gesso.Inference.HostOffload ? s :
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
    triples = use_cache ? _moe_expert_triples(tensors.blocks) : Any[]
    expert_ids = Set{UInt}()
    for (_, g, u, d) in triples
        union!(expert_ids, (objectid(g), objectid(u), objectid(d)))
    end
    qseen = Set{UInt}()
    for s in qpending
        id = objectid(s)
        id in qseen && continue
        push!(qseen, id)
        if id in expert_ids
            conv[s] = s
            continue
        end
        nbytes = Gesso.Inference._quant_nbytes(s)
        if nbytes <= remaining && nbytes > 0
            packed = Gesso.Inference.packed_quant_bytes(s)
            conv[s] = Gesso.Inference.gpu_quant_slice(s, CuArray{UInt8}(packed))
            remaining -= nbytes
        else
            conv[s] = s
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
    @debug "expert storage census" cached_projections=cached_n whole_slab_projections=slab_n host_projections=host_n vram_free_mib=div(Int64(CUDA.free_memory()), Int64(1024 * 1024))
    result = (; embedding, blocks, lm_head, final_rms)
    return merge(result, _keep_tensor_meta(tensors))
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
    CUDA.reclaim()   # return the allocator's cached blocks so the free figure is real
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
        CUDA.CUBLAS.gemv!('N', eltype(sc)(1 / sqrt(d_head)), kh, view(q, 1, hh, :), zero(eltype(sc)), view(sc, 1, :))
    else
        CUDA.CUBLAS.gemm!('N', 'T', eltype(sc)(1 / sqrt(d_head)), qh, kh, zero(eltype(sc)), sc)
    end
    return sc
end
function _attention_values_device!(::CUDABackend, attn::AbstractArray, probs, v::AbstractArray, hh, K)
    if size(attn, 1) == 1
        CUDA.CUBLAS.gemv!('T', one(eltype(attn)), view(v, 1:K, hh, :), view(probs, 1, :), zero(eltype(attn)), view(attn, 1, hh, :))
    else
        CUDA.CUBLAS.gemm!('N', 'N', one(eltype(attn)), probs, view(v, 1:K, hh, :), zero(eltype(attn)), view(attn, :, hh, :))
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
    @cuda threads = (min(256, n),) blocks = (cld(n, 256),) _cuda_sigmoid_mul_kernel!(dst, src, gate)
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
    @cuda threads = (min(256, n),) blocks = (cld(n, 256),) _cuda_silu_mul_kernel!(dst, src, gate)
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
    @cuda threads = (tx, ty) blocks = (cld(seq, tx), cld(nv, ty)) _cuda_gdn_gates_kernel!(
        g, beta, a, b, A_log, dt_bias)
    return g, beta
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
        @cuda threads = (tx, ty) blocks = (cld(seq, tx), cld(channels, ty)) _cuda_conv1d_silu_kernel!(
            dst, x, weight, conv_state, Int(k))
    end
    nthreads = min(256, channels)
    @cuda threads = (nthreads,) blocks = (cld(channels, nthreads),) _cuda_conv_state_update_kernel!(
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

function _cuda_gdn_recurrent_kernel!(out, qn, kn, v, g, beta, state)
    hh = Int(blockIdx().x)
    nv = size(state, 1)
    dk = size(state, 2)
    dv = size(state, 3)
    seq = size(out, 1)
    hh > nv && return
    j = Int(threadIdx().x)
    while j <= dv
        for t in 1:seq
            decay = exp(g[t, hh])
            bt = beta[t, hh]
            for i in 1:dk
                state[hh, i, j] *= decay
            end
            kv_mem = zero(eltype(state))
            for i in 1:dk
                kv_mem += state[hh, i, j] * kn[t, hh, i]
            end
            delta = (v[t, hh, j] - kv_mem) * bt
            for i in 1:dk
                state[hh, i, j] += kn[t, hh, i] * delta
            end
            acc = zero(eltype(out))
            for i in 1:dk
                acc += state[hh, i, j] * qn[t, hh, i]
            end
            out[t, hh, j] = acc
        end
        j += Int(blockDim().x)
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
    qn = similar(out, eltype(out), (seq, nv, dk))
    kn = similar(out, eltype(out), (seq, nv, dk))
    scale = Float32(1) / sqrt(Float32(dk))
    Gesso.Inference.@decode_sub :gdn_qk backend begin
        @cuda threads = 1 blocks = (seq, nv) _cuda_gdn_qk_prep_kernel!(
            qn, kn, q, k, Int(nk), Int(group), v_head_tiled, l2norm, scale, Float32(1e-6))
    end
    nthreads = min(256, dv)
    Gesso.Inference.@decode_sub :gdn_recur backend begin
        @cuda threads = (nthreads,) blocks = (nv,) _cuda_gdn_recurrent_kernel!(
            out, qn, kn, v, g, beta, state)
    end
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

# Thread hh is value head hh; block j is state column j of every head.
function _cuda_gdn_delta_kernel!(
    core, conv_out, a, b, A_log, dt_bias, state,
    nk::Int, group::Int, tiled::Bool, dk::Int, dv::Int, scale::Float32, eps::Float32,
)
    hh = Int(threadIdx().x)
    j = Int(blockIdx().x)
    key_dim = nk * dk
    h = tiled ? ((hh - 1) % nk) + 1 : ((hh - 1) ÷ group) + 1
    qoff = (h - 1) * dk
    koff = key_dim + (h - 1) * dk
    sq = 0.0f0
    sk = 0.0f0
    for i in 1:dk
        qv = conv_out[1, qoff + i]
        kv = conv_out[1, koff + i]
        sq += qv * qv
        sk += kv * kv
    end
    invq = scale / sqrt(sq + eps)
    invk = 1.0f0 / sqrt(sk + eps)

    bv = b[1, hh]
    beta = 1.0f0 / (1.0f0 + exp(-bv))
    av = a[1, hh] + dt_bias[hh]
    sp = av > 20.0f0 ? av : log1p(exp(av))
    decay = exp(-exp(A_log[hh]) * sp)

    kv_mem = 0.0f0
    for i in 1:dk
        st = state[hh, i, j] * decay
        state[hh, i, j] = st
        kv_mem += st * (conv_out[1, koff + i] * invk)
    end
    delta = (conv_out[1, 2 * key_dim + (hh - 1) * dv + j] - kv_mem) * beta
    acc = 0.0f0
    for i in 1:dk
        sn = state[hh, i, j] + (conv_out[1, koff + i] * invk) * delta
        state[hh, i, j] = sn
        acc += sn * (conv_out[1, qoff + i] * invq)
    end
    core[1, (hh - 1) * dv + j] = acc
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
    Gesso.Inference.@decode_sub :gdn_conv backend begin
        @cuda threads = 256 blocks = cld(conv_dim, 256) _cuda_gdn_conv_decode_kernel!(
            s.conv, conv_state, s.mixed, weight, Int(ksz))
    end
    Gesso.Inference.@decode_sub :gdn_recur backend begin
        @cuda threads = nv blocks = dv _cuda_gdn_delta_kernel!(
            s.core, s.conv, s.a, s.b, A_log, dt_bias, rec_state,
            Int(nk), Int(div(nv, nk)), mx.v_head_tiled, Int(dk), Int(dv), scale, eps32)
        @cuda threads = T blocks = nv shmem = T * sizeof(Float32) _cuda_gdn_gnorm_kernel!(
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
        buf = CUDA.zeros(Float32, dims...)
        _DEV_SCRATCH.bufs[k] = buf
    end
    return buf
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
    acc = similar(y, eltype(y), seq)
    CUDA.CUBLAS.gemv!('N', one(eltype(y)), x, g, zero(eltype(y)), acc)
    tx = min(16, seq)
    ty = min(16, h)
    @cuda threads = (tx, ty) blocks = (cld(seq, tx), cld(h, ty)) _cuda_shared_gate_kernel!(
        y, sd, acc)
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
    n_used::Int,
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
        CUDA.CUBLAS.gemm!('N', 'T', one(Float32), x, rd, zero(Float32), logits)
        Gesso.Inference._profile_copy!(true)
        logits_h = Array(logits)
    end

    order = Vector{Int}(undef, n_exp)
    quant = Gesso.Inference._is_quant_expert_bank(gate)
    cached = gate isa Gesso.Inference.CachedExpertBank
    # Banks that are mmap'd on the host (GgufQuantSlice in all three slabs)
    # can run on the CPU; only the activation goes D2H and the output goes H2D.
    # Opt-in (GESSO_MOE_HOST_EXPERTS=1): measured slower than uploading the
    # active experts (0.573 vs 1.226 tok/s on Ornith-35B) until the host
    # kernels are reworked.
    host_all = _moe_host_experts_enabled() &&
               gate isa Gesso.Inference.GgufQuantSlice &&
               up isa Gesso.Inference.GgufQuantSlice &&
               down isa Gesso.Inference.GgufQuantSlice
    Wg = Wu = Wd = nothing
    Gbat = Ubat = Hbat = Dbat = ws = nothing
    experts = Vector{Int}(undef, n_used)
    wbuf = Vector{Float32}(undef, n_used)
    hit_rank = Vector{Int32}(undef, n_used)
    hit_slot = Vector{Int32}(undef, n_used)
    miss_rank = Vector{Int32}(undef, n_used)
    miss_exp = Vector{Int}(undef, n_used)
    xh = yh = nothing
    gh = uh = hh = dh = nothing
    if host_all
        xh = Vector{Float32}(undef, h)
        yh = Vector{Float32}(undef, h)
        gh = Vector{Float32}(undef, ff)
        uh = Vector{Float32}(undef, ff)
        hh = Vector{Float32}(undef, ff)
        dh = Vector{Float32}(undef, h)
    elseif quant || cached
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
        Gesso.Inference.@decode_stage :moe_router nothing begin
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
            sort!(order; by=e -> (-row[e], e))
            wsum = zero(Float32)
            @inbounds for k in 1:n_used
                wsum += row[order[k]]
            end
            invw = wsum > 0 ? one(Float32) / wsum : zero(Float32)
            @inbounds for k in 1:n_used
                experts[k] = order[k]
                wbuf[k] = row[order[k]] * invw
            end
        end
        wsum > 0 || continue
        fill!(yrow, zero(Float32))
        if host_all
            Gesso.Inference.@decode_stage :moe_d2h backend begin
                copyto!(xt, view(x, t, :))
                Gesso.Inference._profile_copy!(true)
                copyto!(xh, xt)
            end
            Gesso.Inference.@decode_stage :moe_host nothing Gesso.Inference._moe_host_experts!(
                yh, xh, experts, wbuf, n_used, gate, up, down, gh, uh, hh, dh,
            )
            Gesso.Inference.@decode_stage :moe_h2d backend begin
                Gesso.Inference._profile_copy!(false)
                copyto!(yrow, yh)
            end
        elseif cached
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
            Gesso.Inference.@decode_stage :moe_qgemv backend CUDA.CUBLAS.gemv!(
                'N', one(Float32), Dbat, ws, zero(Float32), yrow,
            )
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
            Gesso.Inference.@decode_stage :moe_qgemv backend CUDA.CUBLAS.gemv!(
                'N', one(Float32), Dbat, ws, zero(Float32), yrow,
            )
        else
            copyto!(xt, view(x, t, :))
            for k in 1:n_used
                e = order[k]
                w = row[e] * invw
                _fetch_expert!(backend, Wg, gate, e)
                _fetch_expert!(backend, Wu, up, e)
                _fetch_expert!(backend, Wd, down, e)
                CUDA.CUBLAS.gemv!('N', one(Float32), Wg, xt, zero(Float32), gacc)
                CUDA.CUBLAS.gemv!('N', one(Float32), Wu, xt, zero(Float32), uacc)
                _silu_mul!(backend, hact, uacc, gacc)
                CUDA.CUBLAS.gemv!('N', w, Wd, hact, one(Float32), yrow)
            end
        end
        view(y, t, :) .= yrow
    end
    return y
end
