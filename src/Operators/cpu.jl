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
using ..Gesso: CPUBackend
using ..Parameters:
    Activation, EmbeddingTable, FrozenParameter, ProjectionWeight, TemporaryWorkspace

_unmaterialized(op::Symbol, what::Symbol) = error(
    "$op: $what.storage is unset — CPU math needs materialized " *
    "Array{Float64} storage (§LXXV math law); nothing means the caller " *
    "skipped materialization",
)

# --- internals (shared by both workload cuts) -------------------------------

function _cpu_embedding_lookup!(dst, table, tokens)
    tstorage, tab = dst.storage, table.storage
    tstorage === nothing && _unmaterialized(:embedding_lookup!, :dst)
    tab === nothing && _unmaterialized(:embedding_lookup!, :table)
    for (t, tok) in enumerate(tokens)
        dst.storage[t, :] .= tab[tok+1, :]   # tokens are 0-based fixture ids
    end
    return dst
end

function _cpu_rmsnorm!(dst, x, scale; eps=1e-6)
    xs = x.storage
    d = size(xs, ndims(xs))                     # last dim is the feature dim
    scale.storage === nothing && _unmaterialized(:rmsnorm!, :scale)
    length(scale.storage) == d ||
        error("rmsnorm!: scale length $(length(scale.storage)) ≠ feature dim $d")
    # rms per row (Base-only: sum of squares / width — no Statistics dep)
    rms = sqrt.(sum(abs2, xs; dims=ndims(xs)) ./ d .+ eps)
    # scale indexes FEATURES (the last axis): reshape so it broadcasts
    # along the trailing axis — a bare vector would broadcast along axis 1
    stail = reshape(scale.storage, (ntuple(_ -> 1, ndims(xs) - 1)..., d))
    dst.storage .= (xs ./ rms) .* stail
    return dst
end

function _cpu_rope!(q, k, positions; theta=10000.0)
    tθ = Float64(theta)
    size(q.storage, 3) == size(k.storage, 3) ||
        error("rope!: q d_head $(size(q.storage, 3)) ≠ k d_head $(size(k.storage, 3))")
    # Q rotates over Q's head axis, K over K's own — under GQA, K has fewer
    # heads than Q; the old shared loop silently skipped K heads (MHA never
    # noticed because the counts are equal).
    for x in (q, k)
        for t in axes(x.storage, 1), h in axes(x.storage, 2)
            m = Float64(positions[t])           # 0-based position
            d = size(x.storage, 3)              # d_head, even by contract
            for i in 0:(d÷2-1)
                θ = m * tθ^(-2i / d)
                c, s = cos(θ), sin(θ)
                x1, x2 = x.storage[t, h, 2i+1], x.storage[t, h, 2i+2]
                x.storage[t, h, 2i+1] = x1 * c - x2 * s
                x.storage[t, h, 2i+2] = x1 * s + x2 * c
            end
        end
    end
    return q
end

function _cpu_matmul!(dst, x, w)
    # W is (out, in): dst = x * transpose(W)
    mul!(dst.storage, x.storage, transpose(w.storage))
    return dst
end

function _cpu_softmax!(dst, scores)
    s = scores.storage
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
        row_max = maximum(view(s, i, 1:K))
        # max-subtract for stability; a fully-masked row is a caller bug
        # (a causal row always attends to at least the first key) and
        # NaNs out loudly rather than silently passing
        acc = 0.0
        for j in 1:K
            s[i, j] = exp(s[i, j] - row_max)
            acc += s[i, j]
        end
        for j in 1:K
            s[i, j] /= acc
        end
    end
    dst.storage .= s
    return dst
end

function _cpu_swiglu!(dst, gate, up)
    g = gate.storage
    dst.storage .= (g ./ (1.0 .+ exp.(-g))) .* up.storage   # silu(gate) .* up
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
        )
            return _cpu_rope!(q, k, positions; theta)
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
