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
# cost more than they save. The pool serves decode only, and lives only during
# decode: each multi-row prefill pass frees it so the pass has the device to
# itself, and the next decode token allocates it again against the memory then
# free, refilled with the experts that scored highest.
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
    live::Bool                    # false until the next decode token gives it a pool
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
    restores::Int                 # pools refilled from scores after a prefill freed them
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

# Device-local heap budget minus usage, summed over the heaps that report a
# budget; `nothing` when the driver reports none or the probe fails.
function _lava_device_free_bytes()
    rows = try
        Lava.probe_device_memory_budget(Lava.vk_context())
    catch
        return nothing
    end
    free = Int64(0)
    reported = false
    for r in rows
        r.device_local && r.budget > 0 || continue
        reported = true
        free += max(Int64(0), Int64(r.budget - r.usage))
    end
    return reported ? free : nothing
end

# Pool budget in bytes. GESSO_GPU_HOT_EXPERTS_MIB pins it. Otherwise it is the
# device-local heap budget minus usage, less GESSO_GPU_HOT_RESERVE_MIB. When the
# driver reports no budget (VK_EXT_memory_budget missing), a fixed 2048 MiB is
# used: the probe cannot tell a full card from an empty one, and 2048 MiB is
# well inside the 8 GB card.
function _lhot_budget_bytes()
    mib = Int64(1) << 20
    fixed = _lhot_env_int("GESSO_GPU_HOT_EXPERTS_MIB", -1)
    fixed >= 0 && return Int64(fixed) * mib
    free = something(_lava_device_free_bytes(), Int64(0))
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

# Creates a layer's state the first time its gate bank is seen, then gives it
# a pool whenever it has none: on the first decode token, and on the first one
# after a prefill pass freed it. The budget is split evenly over `nlayers` when
# the caller knows the count (qwen4exp), else over the count `_lhot_register!`
# took; with neither, layers take what they need in the order they are first
# seen.
function _lhot_layer!(gate, up, down, nlayers::Int)
    hot = get(_LHOT, gate, nothing)
    if hot === nothing
        hot = _lhot_state(gate, up, down)
        _LHOT[gate] = hot
    end
    hot.live || _lhot_alloc!(hot, nlayers)
    return hot
end

function _lhot_state(gate, up, down)
    nexp = gate.n_exp
    h = gate.ne0
    return _LavaHotLayer(
        length(_LHOT) + 1,
        gate,
        up,
        down,
        false,
        0,
        nothing,
        nothing,
        nothing,
        zeros(Int, nexp),
        Int[],
        Int[],
        zeros(Float32, nexp),
        zeros(Int, nexp),
        0,
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
end

function _lhot_alloc!(hot::_LavaHotLayer, nlayers::Int)
    gate, up, down = hot.gate, hot.up, hot.down
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
            @warn "lava hot experts: pool allocation failed, this layer stays on the host" layer=hot.id nslots exception=err
            for p in (pg, pu, pd)
                p === nothing || Lava.unsafe_free!(p.data)
            end
            nslots = 0
            pg = pu = pd = nothing
        end
    end
    hot.nslots = nslots
    hot.pg, hot.pu, hot.pd = pg, pu, pd
    fill!(hot.slot_of, 0)
    hot.expert_in = zeros(Int, nslots)
    hot.free = collect(nslots:-1:1)
    hot.live = true
    _LHOT_LEFT[] -= Int64(nslots) * bpe
    nslots > 0 && _lhot_restore!(hot)
    return hot
end

# Refills a fresh pool with the experts that scored highest before the last
# prefill freed it, so decode does not start cold. Scores survive a suspend;
# on a layer's first pool every score is zero and nothing is copied.
# The chosen experts take slots 1..k, so each projection goes up as one run of
# consecutive slots, staged on the host in chunks of at most _LHOT_STAGE_MIB.
function _lhot_restore!(hot::_LavaHotLayer)
    nexp = length(hot.score)
    v = Float32[_lhot_score(hot, e) for e in 1:nexp]
    es = Int[]
    for e in partialsortperm(v, 1:min(hot.nslots, nexp); rev=true)
        v[e] > 0.0f0 || break
        push!(es, e)
    end
    k = length(es)
    k == 0 && return 0
    for (pool, bank) in ((hot.pg, hot.gate), (hot.pu, hot.up), (hot.pd, hot.down))
        _lhot_upload_run!(pool, bank, es)
    end
    for (slot, e) in enumerate(es)
        hot.expert_in[slot] = e
        hot.slot_of[e] = slot
    end
    hot.free = collect(hot.nslots:-1:k+1)
    hot.restores += 1
    return k
end

const _LHOT_STAGE_MIB = 64
const _LHOT_STAGE = Ref(UInt8[])

function _lhot_upload_run!(pool, bank, es::Vector{Int};
                           per::Int=max(1, (_LHOT_STAGE_MIB << 20) ÷ bank.bytes_per_expert))
    I = Gesso.Inference
    b = bank.bytes_per_expert
    stage = _LHOT_STAGE[]
    i = 1
    while i <= length(es)
        n = min(per, length(es) - i + 1)
        length(stage) < n * b && resize!(stage, n * b)
        for j in 0:n-1
            copyto!(stage, j * b + 1, bank.data, I._quant_expert_off0(bank, es[i+j]) + 1, b)
        end
        GC.@preserve stage _lhot_upload!(pool, i, pointer(stage), n * b)
        i += n
    end
    return
end

# Host bytes straight into a pool at a slot. Lava's copyto! from an Array
# allocates a host copy of the source first; this goes through its staging
# buffer alone.
_lhot_upload!(pool, slot::Int, src::Ptr{UInt8}, nbytes::Int) =
    Lava.copy_buffer!(:upload, pool.data.buf[], src, nbytes;
                      offset=pool.data.offset + (slot - 1) * pool.bytes_per_expert)

# Frees every layer's pool and gives the empty blocks back to the driver. A
# multi-row prefill pass calls it first: its activations grow with the pass,
# and a pool sized at an earlier decode token, after a shorter prefill, leaves
# a longer one too little room. Scores are kept for the restore.
function _lhot_suspend!()
    any(h -> h.live && h.nslots > 0, values(_LHOT)) || return false
    # A decode token's hit GEMVs may still be queued against the pools.
    KA.synchronize(KALava())
    for hot in values(_LHOT)
        for p in (hot.pg, hot.pu, hot.pd)
            p === nothing || Lava.unsafe_free!(p.data)
        end
        hot.pg = hot.pu = hot.pd = nothing
        hot.nslots = 0
        fill!(hot.slot_of, 0)
        hot.expert_in = Int[]
        hot.free = Int[]
        hot.live = false
    end
    _LHOT_BUDGET[] = -1
    _LHOT_LEFT[] = 0
    Lava.trim_gpu_pool!()
    return true
end

Gesso.Inference._backend_prefill_pass!(::LavaBackend, n::Int) = (n > 1 && _lhot_suspend!(); nothing)

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
        GC.@preserve bank _lhot_upload!(pool, slot, pointer(bank.data, I._quant_expert_off0(bank, e) + 1),
                                        bank.bytes_per_expert)
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

# --- Batched hits ---------------------------------------------------------------
# A token's hits go to the device as two dispatches per layer, not four per hit:
# recording a dispatch costs about as much as running a small expert GEMV, so
# one launch per projection per hit spent more host time than the hits saved.
# Up to _LHOT_BATCH hits go per launch. Their slots and routing weights travel
# as scalar kernel arguments, so nothing is copied to the device and the open
# batch is not flushed. Each workgroup is _LQ_LANES lanes on one output row, the
# layout of _lq_gemv_kernel!.

const _LHOT_BATCH = 8

@inline _lhot_pick(j, a1, a2, a3, a4, a5, a6, a7, a8) =
    j == 1 ? a1 : j == 2 ? a2 : j == 3 ? a3 : j == 4 ? a4 :
    j == 5 ? a5 : j == 6 ? a6 : j == 7 ? a7 : a8

# A[(j-1)*ff + r] = silu(Wg_j[r, :] x) * (Wu_j[r, :] x) for hit j, row r.
@kernel cpu=false function _lhot_gateup_kernel!(
    A, Wg, Wu, X, kv, g2, g3, ks,
    kind_g::Int, kind_u::Int, ne0::Int, ff::Int,
    bpe_g::Int, bpe_u::Int, rb_g::Int, rb_u::Int,
    s1::Int, s2::Int, s3::Int, s4::Int, s5::Int, s6::Int, s7::Int, s8::Int,
)
    tl = @index(Local, Linear)
    r = @index(Group, Linear)
    part = @localmem Float32 (2 * _LQ_LANES,)
    sgs = Lava.subgroup_size()
    sgid = (tl - 1) ÷ sgs
    nsg = _LQ_LANES ÷ sgs
    lane0 = Lava.subgroup_lane() == 0
    j = (r - 1) ÷ ff + 1
    row = (r - 1) % ff
    s = _lhot_pick(j, s1, s2, s3, s4, s5, s6, s7, s8)
    og = (s - 1) * bpe_g + row * rb_g
    ou = (s - 1) * bpe_u + row * rb_u
    accg = 0.0f0
    accu = 0.0f0
    e = tl - 1
    while e < ne0
        xe = X[e+1]
        accg += _lq_elem(kind_g, Wg, kv, g2, g3, ks, og, e) * xe
        accu += _lq_elem(kind_u, Wu, kv, g2, g3, ks, ou, e) * xe
        e += _LQ_LANES
    end
    redg = Lava.subgroup_add(accg)
    redu = Lava.subgroup_add(accu)
    if lane0
        part[sgid+1] = redg
        part[_LQ_LANES+sgid+1] = redu
    end
    @synchronize
    if tl == 1
        g = 0.0f0
        u = 0.0f0
        for i in 1:nsg
            g += part[i]
            u += part[_LQ_LANES+i]
        end
        A[r] = g / (1.0f0 + exp(-g)) * u
    end
end

# Y[r] = sum_j w_j * (Wd_j[r, :] A_j) + beta * Y[r], over nh hits.
@kernel cpu=false function _lhot_down_kernel!(
    Y, Wd, A, kv, g2, g3, ks,
    kind::Int, ff::Int, bpe::Int, rb::Int, nh::Int, beta::Float32,
    s1::Int, s2::Int, s3::Int, s4::Int, s5::Int, s6::Int, s7::Int, s8::Int,
    w1::Float32, w2::Float32, w3::Float32, w4::Float32,
    w5::Float32, w6::Float32, w7::Float32, w8::Float32,
)
    tl = @index(Local, Linear)
    r = @index(Group, Linear)
    part = @localmem Float32 (_LQ_LANES,)
    sgs = Lava.subgroup_size()
    sgid = (tl - 1) ÷ sgs
    nsg = _LQ_LANES ÷ sgs
    lane0 = Lava.subgroup_lane() == 0
    acc = 0.0f0
    for j in 1:nh
        s = _lhot_pick(j, s1, s2, s3, s4, s5, s6, s7, s8)
        w = _lhot_pick(j, w1, w2, w3, w4, w5, w6, w7, w8)
        off = (s - 1) * bpe + (r - 1) * rb
        a0 = (j - 1) * ff
        part_j = 0.0f0
        e = tl - 1
        while e < ff
            part_j += _lq_elem(kind, Wd, kv, g2, g3, ks, off, e) * A[a0+e+1]
            e += _LQ_LANES
        end
        acc += w * part_j
    end
    red = Lava.subgroup_add(acc)
    if lane0
        part[sgid+1] = red
    end
    @synchronize
    if tl == 1
        t = 0.0f0
        for i in 1:nsg
            t += part[i]
        end
        # beta == 0 must not read Y: the scratch may be uninitialized.
        Y[r] = beta == 0.0f0 ? t : t + beta * Y[r]
    end
end

# acc = sum over the token's hits of w_j * expert_j(xv), on the device.
function _lhot_hits!(acc, xv, hot::_LavaHotLayer, nh::Int)
    pg, pu, pd = hot.pg, hot.pu, hot.pd
    h = pg.ne0
    ff = pg.ne1
    tb = _lq_tables()
    be = Lava.KernelAbstractions.get_backend(xv)
    kg = _lq_expert_kind(pg.ggml_type)
    ku = _lq_expert_kind(pu.ggml_type)
    kd = _lq_expert_kind(pd.ggml_type)
    A = _lq_scratch(Float32, :lhot_a, (ff * _LHOT_BATCH,))
    gateup = _lhot_gateup_kernel!(be)
    down = _lhot_down_kernel!(be)
    j0 = 0
    while j0 < nh
        n = min(_LHOT_BATCH, nh - j0)
        s = ntuple(k -> k <= n ? hot.hit_s[j0+k] : 1, _LHOT_BATCH)
        w = ntuple(k -> k <= n ? hot.hit_w[j0+k] : 0.0f0, _LHOT_BATCH)
        gateup(
            A, pg.data, pu.data, xv, tb.kv, tb.g2, tb.g3, tb.ks,
            kg, ku, h, ff,
            pg.bytes_per_expert, pu.bytes_per_expert,
            pg.bytes_per_expert ÷ ff, pu.bytes_per_expert ÷ ff,
            s...;
            ndrange=_LQ_LANES * ff * n,
            workgroupsize=(_LQ_LANES,),
        )
        down(
            acc, pd.data, A, tb.kv, tb.g2, tb.g3, tb.ks,
            kd, ff, pd.bytes_per_expert, pd.bytes_per_expert ÷ h, n,
            j0 == 0 ? 0.0f0 : 1.0f0,
            s..., w...;
            ndrange=_LQ_LANES * h,
            workgroupsize=(_LQ_LANES,),
        )
        j0 += n
    end
    return acc
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
        _lhot_hits!(acc, xv, hot, nh)
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
            restores=l.restores,
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
                "fills=$(s.fills) evictions=$(s.evictions) restores=$(s.restores)")
        end
    end
    empty!(_LHOT)
    _LHOT_STAGE[] = UInt8[]
    _LHOT_BUDGET[] = -1
    _LHOT_LEFT[] = 0
    return nothing
end
