# Runtime settings from the environment.
#
# Most of what Gesso can be told at run time sits under three list variables, so
# the everyday settings stay a short list (docs/env-vars.md):
#
#   GESSO_DISABLE=a,b        turn off default-on paths, for A/B runs and as an
#                            escape hatch: `GESSO_DISABLE=prefill_window,quant_simd`
#   GESSO_PROFILE=a,b=PATH   diagnostics: `GESSO_PROFILE=decode,expert=/tmp/e.tsv`
#   GESSO_TUNE=k=v,k=v       advanced tuning: `GESSO_TUNE=dq_gemm_min=64,hot_fill=4`
#
# Items are comma separated and surrounding spaces are ignored. Names are the
# lowercase words listed in docs/env-vars.md. A GESSO_TUNE value that does not
# parse as the key's type is an error, never a silent default.
#
# Each lookup reads the variable again (tests set it with `withenv`), but the
# parsed list is cached against the raw string, so a call costs one getenv and
# one string compare.

mutable struct _EnvList
    raw::String
    items::Dict{Symbol, String}   # name => value ("" when the item has no `=`)
end

const _ENV_LISTS = Dict{String, Base.RefValue{_EnvList}}(
    "GESSO_DISABLE" => Ref(_EnvList("", Dict{Symbol, String}())),
    "GESSO_PROFILE" => Ref(_EnvList("", Dict{Symbol, String}())),
    "GESSO_TUNE" => Ref(_EnvList("", Dict{Symbol, String}())),
)

function _env_parse_list(raw::String)
    items = Dict{Symbol, String}()
    for part in split(raw, ',')
        s = strip(part)
        isempty(s) && continue
        i = findfirst('=', s)
        if i === nothing
            items[Symbol(lowercase(s))] = ""
        else
            items[Symbol(lowercase(strip(s[1:prevind(s, i)])))] = String(strip(s[nextind(s, i):end]))
        end
    end
    return items
end

function _env_list(var::String)
    slot = _ENV_LISTS[var]
    raw = get(ENV, var, "")
    e = slot[]
    e.raw == raw && return e.items
    e = _EnvList(raw, _env_parse_list(raw))
    slot[] = e        # a mutable object, so the store swaps one pointer
    return e.items
end

"""
    env_disabled(name::Symbol) -> Bool

True when `name` is listed in `GESSO_DISABLE`.
"""
env_disabled(name::Symbol)::Bool = haskey(_env_list("GESSO_DISABLE"), name)

"""
    env_profile(name::Symbol) -> Union{Nothing, String}

`nothing` when `name` is not listed in `GESSO_PROFILE`; otherwise its value,
`""` for a bare name. `GESSO_PROFILE=1` means `decode`.
"""
function env_profile(name::Symbol)::Union{Nothing, String}
    items = _env_list("GESSO_PROFILE")
    name === :decode && haskey(items, Symbol("1")) && return ""
    return get(items, name, nothing)
end

env_profile_on(name::Symbol)::Bool = env_profile(name) !== nothing

"""
    env_tune(name::Symbol, default)

The `GESSO_TUNE` value for `name` parsed as `typeof(default)`, or `default` when
the key is absent or empty. A value that does not parse is an error.
"""
function env_tune(name::Symbol, default::T)::T where {T}
    raw = get(_env_list("GESSO_TUNE"), name, "")
    isempty(raw) && return default
    return _env_tune_parse(T, name, raw)
end

_env_tune_parse(::Type{String}, ::Symbol, raw) = raw
function _env_tune_parse(::Type{Bool}, name::Symbol, raw)
    raw in ("1", "true", "on") && return true
    raw in ("0", "false", "off") && return false
    error("GESSO_TUNE $name=$raw: expected 1 or 0")
end
function _env_tune_parse(::Type{T}, name::Symbol, raw) where {T<:Real}
    v = tryparse(T, raw)
    v === nothing && error("GESSO_TUNE $name=$raw: expected a $(T <: Integer ? "whole number" : "number")")
    return v
end

# Variables replaced by the lists above, with what to set instead. Gesso warns
# once at load when one is still set, so an old run script says what changed
# instead of silently running the default.
const _ENV_REPLACED = (
    "GESSO_GDN_PREFILL_GNORM" => "GESSO_DISABLE=gdn_prefill_gnorm",
    "GESSO_GDN_RECUR_REG" => "GESSO_DISABLE=gdn_recur_reg",
    "GESSO_GPU_DENSE_QUANT" => "GESSO_DISABLE=gpu_dense_quant",
    "GESSO_GPU_HOT_EXPERTS" => "GESSO_DISABLE=hot_experts (CUDA) or GESSO_TUNE=lava_hot_experts=1 (Lava)",
    "GESSO_GPU_HOT_PIN" => "GESSO_DISABLE=hot_pin",
    "GESSO_GPU_HOT_PREFILL" => "GESSO_DISABLE=hot_prefill",
    "GESSO_LAVA_ROPE_TABLE_CACHE" => "GESSO_DISABLE=rope_table_cache",
    "GESSO_LAVA_WINDOW_FUSED" => "GESSO_DISABLE=window_fused",
    "GESSO_MOE_DEVICE_PREFILL" => "GESSO_DISABLE=moe_device_prefill",
    "GESSO_MOE_DEVICE_ROUTE" => "GESSO_DISABLE=moe_device_route",
    "GESSO_MOE_HOST_EXPERTS" => "GESSO_DISABLE=moe_host_experts",
    "GESSO_MOE_OVERLAP" => "GESSO_DISABLE=moe_overlap",
    "GESSO_MOE_PREFILL_ROUTE" => "GESSO_DISABLE=moe_prefill_route",
    "GESSO_PRECOMPILE_WORKLOAD" => "GESSO_DISABLE=precompile_workload",
    "GESSO_PREFAULT" => "GESSO_DISABLE=prefault",
    "GESSO_PREFILL_MULTI" => "GESSO_DISABLE=prefill_multi",
    "GESSO_PREFILL_WINDOW" => "GESSO_DISABLE=prefill_window",
    "GESSO_QUANT_SIMD" => "GESSO_DISABLE=quant_simd",
    "GESSO_PREFILL_PROFILE" => "GESSO_PROFILE=prefill",
    "GESSO_TIMING" => "GESSO_PROFILE=timing",
    "GESSO_GPU_HOT_STATS" => "GESSO_PROFILE=hot_stats",
    "GESSO_EXPERT_PROFILE" => "GESSO_PROFILE=expert=PATH",
    "GESSO_ROUTING_TRACE" => "GESSO_PROFILE=routing=PATH",
    "GESSO_CPU_POOL_SPIN_US" => "GESSO_TUNE=cpu_pool_spin_us=N",
    "GESSO_GEMV_CHUNKS_PER_THREAD" => "GESSO_TUNE=gemv_chunks_per_thread=N",
    "GESSO_GEMV_POOL" => "GESSO_TUNE=gemv_pool=auto|pool|threads",
    "GESSO_DQ_GEMM_MIN" => "GESSO_TUNE=dq_gemm_min=N",
    "GESSO_DQ_GEMM_SCRATCH_MIB" => "GESSO_TUNE=dq_gemm_scratch_mib=N",
    "GESSO_LAVA_CPU_ORDER_WG_MAX" => "GESSO_TUNE=cpu_order_wg_max=N",
    "GESSO_MOE_DEVICE_PREFILL_CHUNK" => "GESSO_TUNE=moe_device_prefill_chunk=N",
    "GESSO_EXPERT_CACHE" => "GESSO_TUNE=expert_cache=N",
    "GESSO_GGUF_UNREORDER_V" => "GESSO_TUNE=gguf_unreorder_v=1",
    "GESSO_GPU_WEIGHT_DTYPE" => "GESSO_TUNE=cuda_weight_dtype=f16|f32",
    "GESSO_GPU_EXPERT_CACHE" => "GESSO_TUNE=cuda_expert_cache=1",
    "GESSO_GPU_EXPERT_CACHE_MIB" => "GESSO_TUNE=cuda_expert_cache_mib=N",
    "GESSO_GPU_HOT_EXPERTS_MIB" => "GESSO_TUNE=hot_mib=N",
    "GESSO_GPU_HOT_RESERVE_MIB" => "GESSO_GPU_RESERVE_MIB",
    "GESSO_GPU_HOT_DECAY" => "GESSO_TUNE=hot_decay=X",
    "GESSO_GPU_HOT_FILL" => "GESSO_TUNE=hot_fill=N",
    "GESSO_GPU_HOT_MARGIN" => "GESSO_TUNE=hot_margin=X",
    "GESSO_GPU_HOT_PREFILL_KEEP" => "GESSO_TUNE=hot_prefill_keep=N",
    "GESSO_GPU_HOT_PREFILL_MAX_SEQ" => "GESSO_TUNE=hot_prefill_max_seq=N",
    "GESSO_GPU_HOT_PREFILL_ROUND" => "GESSO_TUNE=hot_prefill_round=N",
    "GESSO_GPU_HOT_PREFILL_SEG" => "GESSO_TUNE=hot_prefill_seg=N",
    "GESSO_GPU_HOT_PREFILL_GBPS" => "GESSO_TUNE=hot_prefill_gbps=X",
    "GESSO_GPU_HOT_PREFILL_EXPERT_US" => "GESSO_TUNE=hot_prefill_expert_us=X",
    "GESSO_GPU_HOT_PREFILL_PAIR_US" => "GESSO_TUNE=hot_prefill_pair_us=X",
    "GESSO_GPU_HOT_PREFILL_HEADROOM_KIB" => "GESSO_TUNE=hot_prefill_headroom_kib=N",
    "GESSO_GPU_HOT_PREFILL_STREAM" => "GESSO_TUNE=hot_prefill_stream=N",
)

function _env_warn_replaced()
    set = [(old, new) for (old, new) in _ENV_REPLACED if haskey(ENV, old)]
    isempty(set) && return nothing
    lines = join(("  $old: set $new instead" for (old, new) in set), "\n")
    @warn "Gesso no longer reads these environment variables (docs/env-vars.md):\n$lines"
    return nothing
end
