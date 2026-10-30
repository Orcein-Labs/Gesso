# Device weight budget: what the card can give the weights once the memory
# allocated after placement is set aside. The decision is a pure function of
# two numbers so the CPU tests cover it; each backend supplies the probed free
# bytes and its own prefill and BLAS figures.

const _GPU_BUDGET_FALLBACK_MIB = 4800
const _GPU_BUDGET_DEFAULT_CTX = 65536     # gesso-serve's --n-ctx default
const _GPU_RESERVE_SLACK = Int64(128) << 20

const _GPU_BUDGET_CTX = Ref(0)
const _GPU_BUDGET_WARNED = Ref(false)

"""
    gpu_budget_context!(n)

Context length the weight budget keeps KV room for. A server calls it with its
`--n-ctx` before `to_device`; 0 restores the default
(`GESSO_GPU_BUDGET_CTX`, else $(_GPU_BUDGET_DEFAULT_CTX)).
"""
gpu_budget_context!(n::Integer) = (_GPU_BUDGET_CTX[] = max(0, Int(n)); nothing)

function _gpu_budget_ctx()
    _GPU_BUDGET_CTX[] > 0 && return _GPU_BUDGET_CTX[]
    raw = get(ENV, "GESSO_GPU_BUDGET_CTX", "")
    n = something(tryparse(Int, raw), 0)
    return n > 0 ? n : _GPU_BUDGET_DEFAULT_CTX
end

# Rows of the largest prefill pass the sizer can pick: the chunk, capped by the context.
function _gpu_budget_pass_rows(ctx::Int)
    c = something(tryparse(Int, get(ENV, "GESSO_PREFILL_CHUNK", "1024")), 1024)
    return min(ctx, c <= 0 ? 1024 : c)
end

"""
    _gpu_geometry(tensors) -> NamedTuple

Model sizes the reserve needs, from `tensors.spec` when the loader attached
one. A hand-built toy has no spec, or a stub without model sizes, and reserves
no KV.
"""
function _gpu_geometry(tensors)
    spec = tensors isa NamedTuple && haskey(tensors, :spec) ? tensors.spec : nothing
    spec isa NamedTuple && !haskey(spec, :moe) && (spec = nothing)
    spec === nothing &&
        return (; spec=nothing, hidden=0, qdim=0, kvdim=0, ffn=0, group=1, head_dim=0)
    ffn = spec.moe === nothing ? spec.intermediate_size :
          spec.moe.n_expert_used * spec.moe.expert_ff + spec.moe.shared_ff
    return (;
        spec,
        hidden=spec.hidden_size,
        qdim=spec.n_heads * spec.head_dim,
        kvdim=spec.n_kv_heads * spec.head_dim,
        ffn,
        group=max(1, spec.n_heads ÷ max(1, spec.n_kv_heads)),
        head_dim=spec.head_dim,
    )
end

# KV bytes per token in the format a generation on `backend` allocates; f32 when
# the geometry does not fit the quantized layout.
function _gpu_kv_per_token(geo, backend::Symbol)
    geo.spec === nothing && return 0
    fmt = _kv_format(_kv_default_format(backend, geo.head_dim))
    return try
        kv_bytes_per_token(geo.spec, fmt)
    catch
        kv_bytes_per_token(geo.spec, :f32)
    end
end

# Activation bytes one prefill row holds across a layer's projections, the
# attention output and the FFN, in Float32. A backend with a measured figure
# passes its own.
_gpu_prefill_row_bytes(geo) =
    4 * (8 * geo.hidden + 2 * geo.qdim + 2 * geo.kvdim + 3 * geo.ffn)

"""
    _gpu_reserve(; kv_per_token, ctx, prefill, floor, blas) -> NamedTuple

Bytes set aside beside the weights: the KV cache for `ctx` tokens, the largest
prefill pass (`prefill`), the allocator's floor that must stay free after a
pass (`floor`), the BLAS handle and a fixed slack. `total` sums them.
"""
function _gpu_reserve(; kv_per_token::Integer, ctx::Integer, prefill::Integer, floor::Integer, blas::Integer)
    kv = Int64(kv_per_token) * Int64(ctx)
    parts = (; kv, prefill=Int64(prefill), floor=Int64(floor), blas=Int64(blas), slack=_GPU_RESERVE_SLACK)
    return (; parts..., total=kv + parts.prefill + parts.floor + parts.blas + parts.slack)
end

"""
    _gpu_budget_decide(free, reserve) -> (budget::Int64, source::Symbol)

`GESSO_GPU_BUDGET_MIB` wins (`:env`). With no probe (`free === nothing`) the
budget is $(_GPU_BUDGET_FALLBACK_MIB) MiB (`:fallback`). Otherwise it is
`free - reserve`, zero when the reserve alone does not fit (`:probe`): the
weights then stay on the host.
"""
function _gpu_budget_decide(free::Union{Nothing,Integer}, reserve::Integer)
    haskey(ENV, "GESSO_GPU_BUDGET_MIB") && return (_gpu_budget_bytes(), :env)
    free === nothing &&
        return (Int64(_GPU_BUDGET_FALLBACK_MIB) * Int64(1 << 20), :fallback)
    return (max(Int64(0), Int64(free) - Int64(reserve)), :probe)
end

"""
    _gpu_budget_resolve(backend, free, reserve) -> Int64

`_gpu_budget_decide` plus the one info line naming free, reserve and budget in
MiB and where the budget came from. Warns once when there is no probe.
"""
function _gpu_budget_resolve(backend::Symbol, free::Union{Nothing,Integer}, reserve)
    budget, source = _gpu_budget_decide(free, reserve.total)
    mib(x) = x === nothing ? "n/a" : div(Int64(x), Int64(1 << 20))
    if source === :fallback && !_GPU_BUDGET_WARNED[]
        _GPU_BUDGET_WARNED[] = true
        @warn "$backend: no device memory probe, weight budget fixed at $(_GPU_BUDGET_FALLBACK_MIB) MiB"
    end
    @info "$backend weight budget" budget_mib = mib(budget) free_mib = mib(free) reserve_mib =
        mib(reserve.total) source
    return budget
end
