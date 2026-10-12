# Per-expert device residency for MoE expert banks (v1, static fill).
#
# Unit: one expert is its gate, up and down quant bytes together. A layer's
# slot pools hold whole experts, so an expert is resident in all three
# projections or in none. Mixed residency cannot occur.
#
# Quotas: equal device bytes per MoE layer. A single global pool let early
# hits concentrate in two layers in Strata's measurement (2.97% hit rate
# against about 21% with per-layer quotas), and nothing in a routing profile
# argues for one layer over another, so each layer gets the same byte budget.
#
# Fill: with a routing profile, the highest-count experts of each layer. Without
# one, the lowest expert ids. Under uniform routing any fixed subset has the
# same expected hit rate, so the id rule is a neutral default, not a guess.
#
# Adaptive swapping is out of scope for v1. The hook is `expert_fill!`, which
# rewrites a layer's slot tables from any ranking.

"""
    ExpertLayerCache{P}

Residency state of one MoE layer. `banks` are the host mmap'd GGUF slabs
(gate, up, down). `pools` holds one slot pool per projection; slot `s` of
projection `p` occupies bytes `(s-1)*bytes_per_expert_p + 1 : s*bytes_per_expert_p`
of `pools[p]`. The CUDA extension sets `P` to a tuple of `CuVector{UInt8}`.
`slot_of` (expert -> slot, 0 when not resident) is host-only: the kernels
receive byte offsets computed on the host, so no device mirror is needed.
"""
mutable struct ExpertLayerCache{P}
    layer::Int
    banks::NTuple{3,GgufQuantSlice}
    pools::P
    slot_of::Vector{Int32}
    slot_expert::Vector{Int32}
    hits::Int
    misses::Int
    traced::Int
end

function ExpertLayerCache(layer::Int, banks::NTuple{3,GgufQuantSlice}, pools, n_exp::Int, n_slots::Int)
    n_exp >= 0 || error("ExpertLayerCache: n_exp $n_exp < 0")
    0 <= n_slots <= n_exp || error("ExpertLayerCache: n_slots $n_slots outside 0..$n_exp")
    return ExpertLayerCache(
        layer, banks, pools, zeros(Int32, n_exp), zeros(Int32, n_slots), 0, 0, 0,
    )
end

"""
    CachedExpertBank{P}

One projection (`role` 1 gate, 2 up, 3 down) of a cached MoE layer. Holds the
layer cache by reference, so the three projections share one residency table.
It reports the size of the host bank it wraps, so the MoE shape checks apply.
"""
struct CachedExpertBank{P}
    layer::ExpertLayerCache{P}
    role::Int
end

Base.size(b::CachedExpertBank) = size(b.layer.banks[b.role])
Base.size(b::CachedExpertBank, d::Int) = size(b)[d]
Base.ndims(::CachedExpertBank) = 3
Base.eltype(::CachedExpertBank) = Float32
Base.length(b::CachedExpertBank) = prod(size(b))

"""
    expert_slot_bytes(banks) -> Int

Device bytes of one resident expert: its gate, up and down quant bytes.
"""
expert_slot_bytes(banks::NTuple{3,GgufQuantSlice}) =
    sum(b.bytes_per_expert for b in banks)

"""
    expert_layer_quota(layer_budget_bytes, banks, n_exp) -> Int

Slots for one layer: the layer's byte budget over one resident expert,
capped at `n_exp`. The caller divides the total budget equally across layers.
"""
function expert_layer_quota(layer_budget_bytes::Integer, banks::NTuple{3,GgufQuantSlice}, n_exp::Integer)
    layer_budget_bytes >= 0 || error("expert_layer_quota: negative budget $layer_budget_bytes")
    slot = expert_slot_bytes(banks)
    slot > 0 || return 0
    return Int(min(Int64(n_exp), div(Int64(layer_budget_bytes), Int64(slot))))
end

"""
    expert_fill_order(counts, n_exp) -> Vector{Int}

Experts ordered for filling slots. `counts` is a `Dict{Int,Int}` from expert
to routing count, or `nothing`. With counts: descending count, ties by lower
id. Without: ascending id.
"""
function expert_fill_order(counts::Union{Nothing,AbstractDict}, n_exp::Integer)
    order = collect(1:n_exp)
    counts === nothing && return order
    sort!(order; by=e -> (-get(counts, e, 0), e))
    return order
end

"""
    expert_fill!(cache, order)

Set the slot tables so the first `length(cache.slot_expert)` experts of
`order` are resident, in that order. The device copy is the caller's job.
"""
function expert_fill!(cache::ExpertLayerCache, order::AbstractVector{<:Integer})
    n_slots = length(cache.slot_expert)
    length(order) >= n_slots || error("expert_fill!: order has $(length(order)) < $n_slots slots")
    fill!(cache.slot_of, 0)
    fill!(cache.slot_expert, 0)
    for s in 1:n_slots
        e = Int(order[s])
        1 <= e <= length(cache.slot_of) || error("expert_fill!: expert $e outside 1:$(length(cache.slot_of))")
        cache.slot_of[e] = Int32(s)
        cache.slot_expert[s] = Int32(e)
    end
    return cache
end

"""
    read_expert_profile(path) -> Dict{Int,Dict{Int,Int}}

Parse a routing profile written by `scripts/routing_profile.jl`: lines of
`layer expert count` (whitespace separated, 1-based), `#` lines ignored.
Returns layer -> (expert -> count).
"""
function read_expert_profile(path::AbstractString)
    isfile(path) || error("read_expert_profile: no such file $path")
    prof = Dict{Int,Dict{Int,Int}}()
    for (ln, raw) in enumerate(eachline(path))
        line = strip(raw)
        (isempty(line) || startswith(line, '#')) && continue
        f = split(line)
        length(f) == 3 || error("read_expert_profile: $path:$ln expects 3 fields, got $(length(f))")
        l = parse(Int, f[1])
        e = parse(Int, f[2])
        c = parse(Int, f[3])
        d = get!(prof, l, Dict{Int,Int}())
        d[e] = get(d, e, 0) + c
    end
    return prof
end

# --- routing trace (GESSO_ROUTING_TRACE=<path>) ----------------------------
#
# One line per (layer, token row): `layer token_index id_1 ... id_k`, ids in
# routing rank order. `token_index` counts rows per layer. Opt-in and recorded
# only on the device path, where the layer index is known. The file is opened
# once per process on first use and flushed per record.

const _ROUTING_TRACE_IO = Ref{Any}(:unset)

"""
    routing_trace_io() -> IO or nothing

The trace stream when `GESSO_ROUTING_TRACE` is set, else `nothing`.
"""
function routing_trace_io()
    io = _ROUTING_TRACE_IO[]
    io === :unset || return io
    path = get(ENV, "GESSO_ROUTING_TRACE", "")
    io = isempty(path) ? nothing : open(path, "w")
    _ROUTING_TRACE_IO[] = io
    return io
end

"""
    routing_trace_record!(io, cache, experts)

Append one routing row for `cache` and advance its row counter.
"""
function routing_trace_record!(io::IO, cache::ExpertLayerCache, experts::AbstractVector{<:Integer})
    cache.traced += 1
    print(io, cache.layer, ' ', cache.traced)
    for e in experts
        print(io, ' ', e)
    end
    println(io)
    flush(io)
    return nothing
end

"""
    expert_hit_summary(blocks) -> Vector{NamedTuple}

Per-layer residency and hit counters for every cached MoE block in `blocks`
(a device `tensors.blocks`). Layers without a cache are skipped.
"""
function expert_hit_summary(blocks)
    out = NamedTuple{(:layer, :slots, :hits, :misses),Tuple{Int,Int,Int,Int}}[]
    for (li, bt) in enumerate(blocks)
        haskey(bt, :wexp_gate) || continue
        s = bt.wexp_gate.storage
        s isa CachedExpertBank || continue
        c = s.layer
        push!(out, (layer=li, slots=length(c.slot_expert), hits=c.hits, misses=c.misses))
    end
    return out
end
