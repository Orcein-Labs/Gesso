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

function _cuda_rmsnorm!(dst, x, scale; eps = 1e-6)
    xs = x.storage
    nd = ndims(xs)
    d = size(xs, nd)                              # last dim is the feature dim
    length(scale.storage) == d ||
        throw(gesso_error(ERR_INVALID_PLAN, "rmsnorm!: scale length $(length(scale.storage)) ≠ feature dim $d"; op = :rmsnorm!))
    # The REDUCTION stays GPUArrays' `sum` (one device temporary, unchanged)
    # and the APPLY becomes a kernel. Before 10F this line built a
    # Float64-promoted chain — `sum`, `./d .+eps`, `sqrt.`, `xs ./ rms`,
    # `.* stail` — and the last two were Float64 temporaries the size of the
    # whole activation. The kernel evaluates the SAME expression, in the SAME
    # precision, over the SAME operands, so the stored Float32 is
    # bit-identical (§LXXVII declares parity at atol, and this does not even
    # spend that).
    rms = sqrt.(sum(abs2, xs; dims = nd) ./ d .+ eps)
    if nd == 2
        rows = size(dst.storage, 1)
        d == 0 && return dst
        nthreads = min(256, d)
        @cuda threads = (nthreads,) blocks = (rows,) _cuda_rmsnorm_apply_kernel!(
            dst.storage, xs, rms, scale.storage, rows, d)
        return dst
    end
    # generic rank: the previous chain, unchanged (nothing in Gesso is 3-D)
    stail = reshape(scale.storage, (ntuple(_ -> 1, nd - 1)..., d))
    dst.storage .= (xs ./ rms) .* stail
    return dst
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

function _cuda_softmax!(dst, scores)
    L, K = size(scores.storage)
    L == 0 && return dst
    @cuda threads = 1 blocks = L _cuda_softmax_kernel!(scores.storage, dst.storage)
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

function _split_heads!(dst::CuArray, src::CuArray, n_heads, d_head)
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
