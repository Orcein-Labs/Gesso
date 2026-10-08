# CPU reference operator methods (§LXXV Phase 2; math law transcribed from
# docs/goals/PHASE2_CPU_ORACLE.md).
#
# Laws:
#   * All CPU math is Float64 on dense Array{Float64} in `storage`;
#     `shape == size(storage)`. No Float16/BFloat16 path exists.
#   * These are MORE SPECIFIC METHODS on the same functions the stubs own
#     (§CIX: no second vocabulary). The Phase-1 generic methods remain the
#     decline for non-CPU backends; 3-arg un-typed calls still hit the stub.
#   * `quantize!` / `dequantize!` have NO CPU method: they decline (their
#     math is a Representation-phase concern).
#   * Prefill and Decode share bodies via the `_cpu_*` internals — the goal
#     fixes decode math as identical elementwise/matmul work. KV append is
#     the interpreter's job (item C), never matmul!'s.
#
# Phase 3 (§LXXVI): `rmsnorm!` and `rope!` gained keyword knobs — `eps` and
# `theta`. Defaults are exactly the Phase 2 constants, so toy2 arithmetic is
# bit-identical; real checkpoints thread their config values through the
# interpreter. No SmolLM2 constant is hardcoded here. RoPE also rotates K
# over K's OWN head axis (GQA: fewer K heads than Q heads) — for MHA the
# counts are equal and the arithmetic is unchanged.
#
# Residuals are interpreter-level storage addition; there is no `add!`.

using LinearAlgebra: mul!
using ..Gesso: CPUBackend, gesso_error, ERR_INVALID_PLAN, ERR_NUMERICAL_INSTABILITY
using ..Parameters:
    Activation, EmbeddingTable, FrozenParameter, ProjectionWeight, TemporaryWorkspace

function _cpu_storage!(op::Symbol, tensors...)
    for t in tensors
        s=t.storage
        s===nothing && continue
        _backend_storage_root(s) isa Array && eltype(s)===Float64 && continue
        throw(
            gesso_error(
                ERR_INVALID_PLAN,
                "$op: CPUBackend requires host Float64 storage; unsupported native arithmetic is rejected";
                storage_type=string(typeof(s)),
            ),
        )
    end
    return nothing
end

_unmaterialized(op::Symbol, what::Symbol) = error(
    "$op: $what.storage is unset — CPU math needs materialized " *
    "Array{Float64} storage (§LXXV math law); nothing means the caller " *
    "skipped materialization",
)

# --- Phase 10E fence expansion: STORAGE-level bodies -------------------------
#
# `SemanticTensor.storage` is `::Any` (P-1 stays packeted), so an operator body
# that reaches its numbers through `dst.storage` / `scale.storage` pays a
# dynamic `getindex` per element and a broadcast temporary per intermediate.
# Measured with `Profile.Allocs` on a warmed toy2 CPU `decode!`, the RoPE body
# alone was ~9 KB of boxing per token, and the rmsnorm/swiglu/softmax
# broadcast chains another ~6 KB of (1, d) temporaries.
#
# The bodies below therefore take the STORAGE ARRAYS directly. The public
# operator methods stay the surface and stay MORE SPECIFIC on CPUBackend
# (§LXXV); they resolve their `.storage` fields once and hand them over, so the
# arithmetic inside is inferred and every dispatch happens ONCE per call
# instead of once per element. Julia specializes on the concrete storage type,
# so the same body serves Array / CuArray / LavaArray. This is the same
# expansion BREADTH-0 applied to `_split_heads!` / `_merge_heads!` /
# `_repeat_heads!` / `_add_storage!` in Inference.jl.
#
# The arithmetic is unchanged, statement for statement: every rewrite below
# evaluates the SAME operations in the SAME order on the SAME operands, so
# the toy2 / llama fingerprints still hold at atol=0 (§XIII). Nothing here
# resolves P-1: no field type moves, no type is added to the §CIX hierarchy,
# and the P-1 `@test_broken` gates in test/test_type_stability.jl stay broken.

# embedding lookup over storage. `enumerate` order is unchanged.
function _cpu_embedding_lookup_storage!(dsts::AbstractArray, tabs::AbstractArray, tokens)
    for (t, tok) in enumerate(tokens)
        dsts[t, :] .= tabs[tok+1, :]   # tokens are 0-based fixture ids
    end
    return dsts
end

# RMS norm over storage, feature axis LAST (as before). The per-row sum of
# squares is a sequential left fold, which is exactly what
# `sum(abs2, xs; dims=ndims(xs))` performs for a reduction over the leading
# axis, so the row rms is bit-identical; the broadcast chain
# `sqrt.(./d .+ eps)` then `(xs ./ rms) .* stail` becomes scalar stores, which
# is the same value per element.
function _cpu_rmsnorm_storage!(
    dsts::AbstractArray,
    xs::AbstractArray,
    ss::AbstractArray,
    eps,
)
    nd = ndims(xs)
    d = size(xs, nd)                     # last dim is the feature dim
    length(ss) == d || error("rmsnorm!: scale length $(length(ss)) ≠ feature dim $d")
    stail = reshape(ss, (ntuple(_ -> 1, nd - 1)..., d))
    lead = CartesianIndices(size(xs)[1:(nd-1)])
    for I in lead
        acc = 0.0
        for f in axes(xs, nd)
            acc += abs2(xs[I, f])
        end
        r = sqrt(acc / d + eps)
        isfinite(r) && r > 0 || throw(
            gesso_error(
                ERR_NUMERICAL_INSTABILITY,
                "rmsnorm!: nonfinite normalization denominator",
            ),
        )
        for f in axes(xs, nd)
            dsts[I, f] = (xs[I, f] / r) * stail[f]
        end
    end
    return dsts
end

# RoPE over storage. Called once for Q and once for K, in that order, so the
# rotation order is the same as the old `for x in (q, k)` loop.
#
# BREADTH-0 Pass D: `inv_freq` carries an EXPLICIT positional policy.
# `inv_freq === nothing` is the unscaled policy and MUST stay the literal
# expression — it is what the Llama/CPU-oracle fingerprints pin at atol=0, so
# the default path is bit-identical by construction (regression law §XIII).
function _cpu_rope_storage!(
    storage::AbstractArray,
    positions::AbstractVector{Int},
    tθ,
    inv_freq,
    interleaved::Bool=true,
)
    for t in axes(storage, 1), h in axes(storage, 2)
        m = Float64(positions[t])           # 0-based position
        d = size(storage, 3)                # d_head, even by contract
        for i in 0:(d÷2-1)
            θ = inv_freq === nothing ? m * tθ^(-2i / d) : m * inv_freq[i+1]
            c, s = cos(θ), sin(θ)
            a, b = interleaved ? (2i + 1, 2i + 2) : (i + 1, i + 1 + d ÷ 2)
            x1, x2 = storage[t, h, a], storage[t, h, b]
            storage[t, h, a] = x1 * c - x2 * s
            storage[t, h, b] = x1 * s + x2 * c
        end
    end
    return storage
end

# swiglu over storage: dst = (g / (1 + exp(-g))) .* up, elementwise, so the
# scalar loop stores the same value the broadcast chain did.
function _cpu_swiglu_storage!(dsts::AbstractArray, gs::AbstractArray, ups::AbstractArray)
    for i in eachindex(dsts, gs, ups)
        g = gs[i]
        dsts[i] = (g / (1.0 + exp(-g))) * ups[i]
    end
    return dsts
end

# softmax over storage, mutating the score row in place exactly as before
# (causal mask, max-subtract, normalize, then copy into dst).
function _cpu_softmax_storage!(dsts::AbstractArray, s::AbstractArray)
    L = size(s, 1)                              # seq_q
    K = size(s, 2)                              # seq_k
    # causal mask BEFORE softmax (§LXXV). Rows align to the LAST K keys:
    # key offset = K - L (keys that precede the query block). For prefill
    # (L == K) the offset is 0 and this is exactly "position i may attend
    # to j ≤ i". For a decode step (L == 1, K = cached + 1) the single
    # query is the last position and legitimately attends to the whole
    # cache — nothing is masked.
    offset = K - L
    for i in 1:L
        for j in (offset+i+1):K
            s[i, j] = -Inf
        end
        # max-subtract for stability; a fully-masked row is a caller bug
        # (a causal row always attends to at least the first key) and
        # NaNs out loudly rather than silently passing
        row_max = maximum(view(s, i, 1:K))
        isfinite(row_max) || throw(
            gesso_error(ERR_NUMERICAL_INSTABILITY, "softmax!: no finite unmasked maximum"),
        )
        acc = 0.0
        for j in 1:K
            s[i, j] = exp(s[i, j] - row_max)
            acc += s[i, j]
        end
        for j in 1:K
            s[i, j] /= acc
        end
    end
    dsts .= s
    return dsts
end

# --- internals (shared by both workload cuts) -------------------------------

function _cpu_embedding_lookup!(dst, table, tokens)
    _cpu_storage!(:embedding_lookup!, dst, table)
    tstorage, tab = dst.storage, table.storage
    tstorage === nothing && _unmaterialized(:embedding_lookup!, :dst)
    tab === nothing && _unmaterialized(:embedding_lookup!, :table)
    _validate_embedding_inputs(dst, table, tokens)
    _cpu_embedding_lookup_storage!(tstorage, tab, tokens)
    return dst
end

function _cpu_rmsnorm!(dst, x, scale; eps=1e-6)
    _cpu_storage!(:rmsnorm!, dst, x, scale)
    isfinite(eps) && eps > 0 ||
        throw(gesso_error(ERR_INVALID_PLAN, "rmsnorm!: eps must be finite and positive"))
    xs = x.storage
    d = size(xs, ndims(xs))                     # last dim is the feature dim
    scale.storage === nothing && _unmaterialized(:rmsnorm!, :scale)
    length(scale.storage) == d ||
        error("rmsnorm!: scale length $(length(scale.storage)) ≠ feature dim $d")
    _cpu_rmsnorm_storage!(dst.storage, xs, scale.storage, Float64(eps))
    return dst
end

function _validate_rope_inputs(q, k, positions, theta, inv_freq)
    qs, ks=q.storage, k.storage
    ndims(qs)==3 && ndims(ks)==3 || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "rope!: expected (sequence, heads, features) storage",
        ),
    )
    d=size(qs, 3)
    d>0 && iseven(d) && size(ks, 3)==d && size(qs, 1)==size(ks, 1) ||
        throw(gesso_error(ERR_INVALID_PLAN, "rope!: invalid head/sequence geometry"))
    (_backend_storage_root(positions) isa Array || positions isa AbstractRange) &&
    length(positions)==size(qs, 1) &&
    all(>=(0), positions) || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "rope!: positions must match sequence and be nonnegative",
        ),
    )
    isfinite(theta) && theta>0 ||
        throw(gesso_error(ERR_INVALID_PLAN, "rope!: theta must be finite and positive"))
    if inv_freq!==nothing
        inv_freq isa AbstractVector &&
        (_backend_storage_root(inv_freq) isa Array || inv_freq isa AbstractRange) &&
        length(inv_freq)==d÷2 &&
        all(x->isfinite(x) && x>0, inv_freq) ||
            throw(gesso_error(ERR_INVALID_PLAN, "rope!: invalid inverse frequencies"))
    end
    return nothing
end

function _cpu_rope!(q, k, positions; theta=10000.0, inv_freq=nothing, interleaved=true)
    _cpu_storage!(:rope!, q, k)
    _validate_rope_inputs(q, k, positions, theta, inv_freq)
    tθ = Float64(theta)
    size(q.storage, 3) == size(k.storage, 3) ||
        error("rope!: q d_head $(size(q.storage, 3)) ≠ k d_head $(size(k.storage, 3))")
    # Q rotates over Q's head axis, K over K's own — under GQA, K has fewer
    # heads than Q; the old shared loop silently skipped K heads (MHA never
    # noticed because the counts are equal). Q is rotated first, then K, as
    # before.
    _cpu_rope_storage!(q.storage, positions, tθ, inv_freq, interleaved)
    _cpu_rope_storage!(k.storage, positions, tθ, inv_freq, interleaved)
    return q
end

function _cpu_matmul!(dst, x, w)
    _cpu_storage!(:matmul!, dst, x, w)
    # W is (out, in): dst = x * transpose(W)
    mul!(dst.storage, x.storage, transpose(w.storage))
    return dst
end

function _cpu_softmax!(dst, scores)
    _cpu_storage!(:softmax!, dst, scores)
    _cpu_softmax_storage!(dst.storage, scores.storage)
    return dst
end

function _cpu_swiglu!(dst, gate, up)
    _cpu_storage!(:swiglu!, dst, gate, up)
    _cpu_swiglu_storage!(dst.storage, gate.storage, up.storage)
    return dst
end

# --- dispatch surface: CPUBackend × families × workload ----------------------
#
# More specific than the Phase-1 generic methods, so these win on CPU while
# other backends keep the explicit decline.

for wl in (:PrefillWorkload, :DecodeWorkload)
    @eval begin
        function embedding_lookup!(
            ::CPUBackend,
            dst::Activation,
            table::EmbeddingTable,
            tokens::AbstractVector{Int},
            ::Semantics.$wl,
        )
            return _cpu_embedding_lookup!(dst, table, tokens)
        end

        function rmsnorm!(
            ::CPUBackend,
            dst::Activation,
            x::Activation,
            scale::FrozenParameter,
            ::Semantics.$wl;
            eps::Real=1e-6,
        )
            return _cpu_rmsnorm!(dst, x, scale; eps)
        end

        function rope!(
            ::CPUBackend,
            q::Activation,
            k::Activation,
            positions::AbstractVector{Int},
            ::Semantics.$wl;
            theta::Real=10000.0,
            inv_freq=nothing,
            interleaved::Bool=true,
        )
            return _cpu_rope!(q, k, positions; theta, inv_freq, interleaved)
        end

        function matmul!(
            ::CPUBackend,
            dst::Activation,
            x::Activation,
            w::ProjectionWeight,
            ::Semantics.$wl,
        )
            return _cpu_matmul!(dst, x, w)
        end

        # tied output head (§LXXV): the embedding table IS the lm_head —
        # its (vocab, dim) bytes viewed as a (out, in) projection. Same
        # function, no second vocabulary, no second table.
        function matmul!(
            ::CPUBackend,
            dst::Activation,
            x::Activation,
            w::EmbeddingTable,
            ::Semantics.$wl,
        )
            return _cpu_matmul!(dst, x, w)
        end

        function softmax!(
            ::CPUBackend,
            dst::TemporaryWorkspace,
            scores::TemporaryWorkspace,
            ::Semantics.$wl,
        )
            return _cpu_softmax!(dst, scores)
        end

        function swiglu!(
            ::CPUBackend,
            dst::Activation,
            gate::Activation,
            up::Activation,
            ::Semantics.$wl,
        )
            return _cpu_swiglu!(dst, gate, up)
        end
    end
end
