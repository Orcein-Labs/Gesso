# Hot host experts on Lava: a device pool per host MoE layer.
#
# Host MoE layers run their routed experts on the CPU, and on a single-channel
# host that is bound by memory bandwidth. Routing is local in time, so each
# layer keeps its hottest experts in a device pool. For one decode token the
# hit experts run on the device while the host runs the misses, and the host
# partial sum is added on the device. The policy is the CUDA one
# (ext/cuda_hot_experts.jl): an exponentially decayed pick count per expert, and
# a missed expert that outscores the coldest resident replaces it, at most one
# eviction per layer per token plus fills while slots are empty.
#
# Fills are synchronous. Lava's host-to-device copy flushes the open batch, so a
# swap waits for the device, and the copy allocates a temporary per call. That
# is the v1 cost; the fill count is in the stats so its price can be read off.
#
# Prefill (T > 1) keeps the all-host route: a prompt's experts are spread over
# every row, so the pool would be a small share of the work and the fills would
# cost more than they save. The pool serves decode only.
#
# Hits use the device expert GEMV with F32 activations, misses the host Q8_K
# route, so a pool changes the rounding of the output. Zero slots leave the
# output bit-identical to the all-host route, because the layer then never
# leaves the host path.

const _LHOT_DECAY = 0.99f0
const _LHOT_FILL = 2
const _LHOT_MARGIN = 1.0f0
const _LHOT_RESERVE_MIB = 1024
const _LHOT_FALLBACK_MIB = 2048

_lhot_enabled() = get(ENV, "GESSO_GPU_HOT_EXPERTS", "0") == "1"

function _lhot_env_int(name::AbstractString, default::Int)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    v = tryparse(Int, raw)
    v === nothing && error("$name must be an integer, got '$raw'")
    return v
end

mutable struct _LavaHotLayer
    id::Int
    gate::Gesso.Inference.GgufQuantSlice
    up::Gesso.Inference.GgufQuantSlice
    down::Gesso.Inference.GgufQuantSlice
    nslots::Int
    pg::Any                       # device GpuQuantSlice per projection, nothing when nslots == 0
    pu::Any
    pd::Any
    slot_of::Vector{Int}          # expert -> slot, 0 when not resident
    expert_in::Vector{Int}        # slot -> expert, 0 when empty
    free::Vector{Int}
    score::Vector{Float32}
    seen::Vector{Int}
    t::Int
    fills::Int
    evictions::Int
    hits::Int
    picks::Int
    tokens::Int
    experts::Matrix{Int}
    wts::Matrix{Float32}
    logits::Vector{Float32}
    order::Vector{Int}
    hit_s::Vector{Int}
    hit_w::Vector{Float32}
    miss_e::Vector{Int}
    miss_w::Vector{Float32}
    xf::Vector{Float32}
    yrow::Vector{Float32}
end

const _LHOT = IdDict{Any, _LavaHotLayer}()
const _LHOT_BUDGET = Ref{Int64}(-1)     # bytes for all pools, set at the first decode token
const _LHOT_LEFT = Ref{Int64}(0)        # bytes not yet given to a layer
const _LHOT_NLAYERS = Ref(0)            # pool-eligible expert layers of the loaded model
const _LHOT_WARNED = Ref(false)

# Pool budget in bytes. GESSO_GPU_HOT_EXPERTS_MIB pins it. Otherwise it is the
# device-local heap budget minus usage, less GESSO_GPU_HOT_RESERVE_MIB. When the
# driver reports no budget (VK_EXT_memory_budget missing), a fixed 2048 MiB is
# used: the probe cannot tell a full card from an empty one, and 2048 MiB is
# well inside the 8 GB card.
function _lhot_budget_bytes()
    mib = Int64(1) << 20
    fixed = _lhot_env_int("GESSO_GPU_HOT_EXPERTS_MIB", -1)
    fixed >= 0 && return Int64(fixed) * mib
    rows = Lava.probe_device_memory_budget(Lava.vk_context())
    free = Int64(0)
    for r in rows
        r.device_local && r.budget > 0 && (free += max(Int64(0), Int64(r.budget - r.usage)))
    end
    if free == 0
        if !_LHOT_WARNED[]
            _LHOT_WARNED[] = true
            @warn "lava hot experts: driver reports no memory budget, using a fixed $(_LHOT_FALLBACK_MIB) MiB pool budget"
        end
        return Int64(_LHOT_FALLBACK_MIB) * mib
    end
    reserve = Int64(_lhot_env_int("GESSO_GPU_HOT_RESERVE_MIB", _LHOT_RESERVE_MIB)) * mib
    return max(Int64(0), free - reserve)
end

function _lhot_budget_init!()
    _LHOT_BUDGET[] >= 0 && return _LHOT_BUDGET[]
    _LHOT_BUDGET[] = _lhot_budget_bytes()
    _LHOT_LEFT[] = _LHOT_BUDGET[]
    return _LHOT_BUDGET[]
end

# The pool is eligible only for expert types the Lava expert GEMV decodes, and
# only when the three banks share their shape, the way the host route assumes.
function _lhot_eligible(g, u, d)
    all(s -> _lq_expert_kind(s.ggml_type) !== nothing && s.bytes_per_expert > 0, (g, u, d)) ||
        return false
    g.n_exp == u.n_exp == d.n_exp || return false
    g.ne0 == u.ne0 && g.ne1 == u.ne1 || return false
    return d.ne0 == g.ne1 && d.ne1 == g.ne0
end

function _lhot_pool(bank, nslots::Int)
    data = Lava.LavaArray{UInt8}(undef, nslots * bank.bytes_per_expert)
    return Gesso.Inference.GpuQuantSlice(
        data,
        bank.ggml_type,
        nslots,
        bank.ne0,
        bank.ne1,
        bank.bytes_per_expert,
        bank.blck_size,
    )
end

# Counts the layers that can take a pool, at to_device time, the way CUDA's
# `_hot_register!` does. Layers are created lazily in decode order, so without
# the count the hybrid's first layers would take the whole budget and the rest
# none. The previous model's pools go first: they are keyed by its gate banks.
function _lhot_register!(blocks)
    release_lava_hot_experts!()
    n = 0
    for bt in blocks
        (haskey(bt, :wexp_gate) && haskey(bt, :wexp_up) && haskey(bt, :wexp_down)) || continue
        g, u, d = bt.wexp_gate.storage, bt.wexp_up.storage, bt.wexp_down.storage
        all(s -> s isa Gesso.Inference.GgufQuantSlice, (g, u, d)) || continue
        _lhot_eligible(g, u, d) && (n += 1)
    end
    _LHOT_NLAYERS[] = n
    return n
end

# Creates a layer's state the first time its gate bank is seen. The budget is
# split evenly over `nlayers` when the caller knows the count (qwen4exp), else
# over the count `_lhot_register!` took; with neither, layers take what they
# need in the order they are first seen.
function _lhot_layer!(gate, up, down, nlayers::Int)
    hot = get(_LHOT, gate, nothing)
    hot === nothing || return hot
    I = Gesso.Inference
    total = _lhot_budget_init!()
    bpe = gate.bytes_per_expert + up.bytes_per_expert + down.bytes_per_expert
    n = nlayers > 0 ? nlayers : _LHOT_NLAYERS[]
    share = n > 0 ? min(_LHOT_LEFT[], div(total, n)) : _LHOT_LEFT[]
    # Slot offsets into a pool stay inside Int32, as in the CUDA pool.
    nslots = _lhot_eligible(gate, up, down) ?
        min(gate.n_exp, div(Int(share), bpe), div(Int(typemax(Int32)), bpe)) : 0
    pg = pu = pd = nothing
    if nslots > 0
        try
            pg = _lhot_pool(gate, nslots)
            pu = _lhot_pool(up, nslots)
            pd = _lhot_pool(down, nslots)
        catch err
            @warn "lava hot experts: pool allocation failed, this layer stays on the host" layer=objectid(gate) nslots exception=err
            nslots = 0
            pg = pu = pd = nothing
        end
    end
    nexp = gate.n_exp
    h = gate.ne0
    ff = gate.ne1
    hot = _LavaHotLayer(
        length(_LHOT) + 1,
        gate,
        up,
        down,
        nslots,
        pg,
        pu,
        pd,
        zeros(Int, nexp),
        zeros(Int, nslots),
        collect(nslots:-1:1),
        zeros(Float32, nexp),
        zeros(Int, nexp),
        0,
        0,
        0,
        0,
        0,
        0,
        Matrix{Int}(undef, 0, 0),
        Matrix{Float32}(undef, 0, 0),
        Vector{Float32}(undef, nexp),
        Vector{Int}(undef, nexp),
        Int[],
        Float32[],
        Int[],
        Float32[],
        Vector{Float32}(undef, h),
        Vector{Float32}(undef, h),
    )
    _LHOT_LEFT[] -= Int64(nslots) * bpe
    _LHOT[gate] = hot
    return hot
end

# Whether this token takes the device pool: decode only, every bank a host
# GgufQuantSlice, and a layer with at least one slot. Creates the layer state.
function _lhot_wants(xh, gate, up, down, nlayers::Int)
    _lhot_enabled() || return false
    size(xh, 1) == 1 || return false
    (gate isa Gesso.Inference.GgufQuantSlice && up isa Gesso.Inference.GgufQuantSlice &&
     down isa Gesso.Inference.GgufQuantSlice) || return false
    return _lhot_layer!(gate, up, down, nlayers).nslots > 0
end

_lhot_score(hot::_LavaHotLayer, e::Int) = begin
    dt = hot.t - hot.seen[e]
    dt == 0 ? hot.score[e] : hot.score[e] * _LHOT_DECAY ^ dt
end

function _lhot_observe!(hot::_LavaHotLayer, picks, n_used::Int)
    hot.t += 1
    @inbounds for k in 1:n_used
        e = picks[k]
        hot.score[e] = _lhot_score(hot, e) + 1.0f0
        hot.seen[e] = hot.t
    end
    return
end

# Copies one expert's gate, up and down bytes into a slot. The host bank holds
# the bytes at a 0-based offset; the pool's slot s starts at (s - 1) * bpe.
function _lhot_fill!(hot::_LavaHotLayer, slot::Int, e::Int)
    I = Gesso.Inference
    for (pool, bank) in ((hot.pg, hot.gate), (hot.pu, hot.up), (hot.pd, hot.down))
        b = bank.bytes_per_expert
        copyto!(pool.data, (slot - 1) * b + 1, bank.data, I._quant_expert_off0(bank, e) + 1, b)
    end
    return
end

# Starts at most _LHOT_FILL copies for this token's misses: into empty slots
# first, then at most one eviction of the coldest resident the token did not
# pick. A miss must outscore the victim by the margin, so experts at the edge of
# residency do not trade places every token.
function _lhot_plan!(hot::_LavaHotLayer, picks, n_used::Int)
    evicted = false
    for _ in 1:_LHOT_FILL
        best = 0
        bs = -1.0f0
        @inbounds for k in 1:n_used
            e = picks[k]
            hot.slot_of[e] == 0 || continue
            v = _lhot_score(hot, e)
            if v > bs
                best = e
                bs = v
            end
        end
        best == 0 && break
        if !isempty(hot.free)
            slot = pop!(hot.free)
        elseif !evicted
            slot = 0
            vs = typemax(Float32)
            @inbounds for s in 1:hot.nslots
                e = hot.expert_in[s]
                (e > 0 && hot.seen[e] != hot.t) || continue
                v = _lhot_score(hot, e)
                if v < vs
                    slot = s
                    vs = v
                end
            end
            (slot == 0 || vs + _LHOT_MARGIN >= bs) && break
            hot.slot_of[hot.expert_in[slot]] = 0
            hot.expert_in[slot] = 0
            evicted = true
            hot.evictions += 1
        else
            break
        end
        _lhot_fill!(hot, slot, best)
        hot.expert_in[slot] = best
        hot.slot_of[best] = slot
        hot.fills += 1
    end
    return
end

# A slice nobody registered takes the host route with no pool. It is not
# registered here: registration fixes the layer's slot count, and only
# _lhot_wants is allowed to decide that.
function _lhot_host_decode!(y, xh, router, gate, up, down, n_used::Int)
    I = Gesso.Inference
    h = gate.ne0
    experts = Matrix{Int}(undef, n_used, 1)
    wts = Matrix{Float32}(undef, n_used, 1)
    logits = Vector{Float32}(undef, gate.n_exp)
    order = Vector{Int}(undef, gate.n_exp)
    yrow = zeros(Float32, h)
    xf = Float32[xh[1, i] for i in 1:h]
    if I._moe_route_token!(experts, wts, 1, xh, router, logits, order, n_used)
        I._moe_host_experts_auto!(yrow, xf, view(experts, :, 1), view(wts, :, 1), n_used, gate, up, down)
    end
    copyto!(y, 1, yrow, 1, h)
    return y
end

# One decode token through a layer's pool. `y` is the device (1, h) output and
# is written in full. The hybrid and qwen4exp routes call `_lhot_wants` first,
# which registers the layer; a slice that was never registered takes the host
# route instead of throwing.
function _lhot_moe_decode!(y, x, xh, router, gate, up, down, n_used::Int)
    I = Gesso.Inference
    hot = get(_LHOT, gate, nothing)
    hot === nothing && return _lhot_host_decode!(y, xh, router, gate, up, down, n_used)
    h = gate.ne0
    ff = gate.ne1
    hot.tokens += 1
    if size(hot.experts, 1) != n_used
        hot.experts = Matrix{Int}(undef, n_used, 1)
        hot.wts = Matrix{Float32}(undef, n_used, 1)
        resize!(hot.hit_s, n_used)
        resize!(hot.hit_w, n_used)
        resize!(hot.miss_e, n_used)
        resize!(hot.miss_w, n_used)
    end
    live = I._moe_route_token!(hot.experts, hot.wts, 1, xh, router, hot.logits, hot.order, n_used)
    fill!(hot.yrow, 0.0f0)
    nh = 0
    nm = 0
    if live
        ex = view(hot.experts, :, 1)
        wt = view(hot.wts, :, 1)
        _lhot_observe!(hot, ex, n_used)
        # Classify before planning: a fill this token cannot serve it, and an
        # eviction only ever takes an expert this token did not pick.
        @inbounds for k in 1:n_used
            e = ex[k]
            s = hot.slot_of[e]
            if s > 0
                nh += 1
                hot.hit_s[nh] = s
                hot.hit_w[nh] = wt[k]
            else
                nm += 1
                hot.miss_e[nm] = e
                hot.miss_w[nm] = wt[k]
            end
        end
        hot.hits += nh
        hot.picks += n_used
        _lhot_plan!(hot, ex, n_used)
    end

    # The hit GEMVs go into the queue first, so the device runs them while the
    # host works through the misses below.
    acc = _lq_scratch(Float32, :lhot_acc, (h,))
    if nh > 0
        xv = _lq_scratch(Float32, :lhot_x, (h,))
        copyto!(xv, 1, x, 1, h)
        g = _lq_scratch(Float32, :lhot_g, (ff,))
        u = _lq_scratch(Float32, :lhot_u, (ff,))
        a = _lq_scratch(Float32, :lhot_a, (ff,))
        for j in 1:nh
            s = hot.hit_s[j]
            I._qgemv_expert!(g, xv, hot.pg, s; alpha=1.0f0, beta=0.0f0)
            I._qgemv_expert!(u, xv, hot.pu, s; alpha=1.0f0, beta=0.0f0)
            a .= (g ./ (1.0f0 .+ exp.(-g))) .* u
            I._qgemv_expert!(acc, a, hot.pd, s; alpha=hot.hit_w[j], beta=j == 1 ? 0.0f0 : 1.0f0)
        end
        Lava.submit!(KALava().dispatch_bq)
    end

    if nm > 0
        @inbounds for i in 1:h
            hot.xf[i] = xh[1, i]
        end
        I._moe_host_experts_auto!(
            hot.yrow,
            hot.xf,
            view(hot.miss_e, 1:nm),
            view(hot.miss_w, 1:nm),
            nm,
            gate,
            up,
            down,
        )
    end

    ym = _lq_scratch(Float32, :lhot_ym, (h,))
    copyto!(ym, 1, hot.yrow, 1, h)
    nh > 0 && (ym .+= acc)
    copyto!(y, 1, ym, 1, h)
    return y
end

# Cumulative counters for every layer that has a state, in the order first seen.
function lava_hot_experts_stats()
    layers = sort!(collect(values(_LHOT)); by=l -> l.id)
    return [
        (
            layer=l.id,
            slots=l.nslots,
            tokens=l.tokens,
            picks=l.picks,
            hits=l.hits,
            fills=l.fills,
            evictions=l.evictions,
            hit_rate=l.picks == 0 ? 0.0 : l.hits / l.picks,
        ) for l in layers
    ]
end

# Drops every layer's pool. The device memory goes with the last reference;
# the budget is probed again on the next decode token.
function release_lava_hot_experts!()
    if get(ENV, "GESSO_GPU_HOT_STATS", "0") == "1"
        for s in lava_hot_experts_stats()
            println(stderr, "lava hot experts layer $(s.layer): slots=$(s.slots) tokens=$(s.tokens) " *
                "picks=$(s.picks) hits=$(s.hits) hit_rate=$(round(s.hit_rate; digits=3)) " *
                "fills=$(s.fills) evictions=$(s.evictions)")
        end
    end
    empty!(_LHOT)
    _LHOT_BUDGET[] = -1
    _LHOT_LEFT[] = 0
    return nothing
end
