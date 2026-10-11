# Hot host experts in VRAM (GESSO_GPU_HOT_EXPERTS=1).
#
# MoE layers whose expert slabs stay on the host run their experts on the CPU.
# On Ornith-35B that is about 16 ms of a 28 ms token, and it is bound by the
# host's memory bandwidth: every token reads about 165 MB of expert weights.
# Routing is local in time. On natural text, a decayed top-32 of each layer's
# experts covers about 62% of the picks, and a top-64 about 80%. So each host
# layer keeps its currently hottest experts in a VRAM pool. For one token the
# GPU runs the resident picks while the CPU runs the rest, and the GPU adds the
# CPU's partial sum in.
#
# Residency follows each expert's exponentially decayed pick count. A missed
# expert that outscores the coldest resident replaces it. That is at most one
# eviction per layer per token, plus a few fills while slots are still empty.
# The new weights go up on a side stream through pinned staging, and a slot
# serves only once its copy is known to be complete, so the compute stream
# never waits on a swap. The host plans and issues each layer's copies while it
# waits on that layer's router logits, so the staging memcpy fills time the
# host would otherwise spend idle.
#
# With GESSO_GPU_HOT_PIN=1 each hot layer's slabs move into page-locked
# anonymous memory (a file mmap cannot be registered with the driver). A swap
# is then three async DMA copies with no host memcpy. On a single-channel host
# the staged copy read and wrote each expert through DRAM before the DMA read
# it a third time, while the CPU experts competed for the same bus.
#
# Hits use the GPU kernels (q8_1 activations) and misses the CPU ones (Q8_K),
# so the output rounds differently from the all-CPU path.

_hot_experts_enabled() = get(ENV, "GESSO_GPU_HOT_EXPERTS", "0") == "1"

# Decode needs about 50 MiB beyond the weights. A prefill's activations peak
# near 0.7 MiB per token on Ornith-35B (177 MiB at 256 tokens, 617 at 1024). A
# prefill that runs out releases pools (_HotPurge), so the reserve only has to
# cover decode and short prompts.
const _HOT_RESERVE_DEFAULT_MIB = 384

function _hot_env_int(name, default)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    v = tryparse(Int, raw)
    v === nothing && error("$name must be an integer, got '$raw'")
    return v
end

function _hot_env_float(name, default)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    v = tryparse(Float64, raw)
    v === nothing && error("$name must be a number, got '$raw'")
    return v
end

# GESSO_GPU_HOT_EXPERTS_MIB when set, else the free device memory after
# placement minus GESSO_GPU_HOT_RESERVE_MIB.
function _hot_budget_bytes()
    mib = Int64(1024 * 1024)
    fixed = _hot_env_int("GESSO_GPU_HOT_EXPERTS_MIB", -1)
    fixed >= 0 && return Int64(fixed) * mib
    reserve = _hot_env_int("GESSO_GPU_HOT_RESERVE_MIB", _HOT_RESERVE_DEFAULT_MIB)
    CUDA.reclaim(CUDA.CUDACore.RECLAIM_GC)
    return _expert_budget_from_free(Int64(CUDA.free_memory()), reserve)
end

struct _HotConfig
    decay::Float32     # per-token decay of the pick counts
    fill::Int          # copies a layer may start per token
    keep_seq::Int      # longest prefill that keeps the pools; default: until memory runs out
    margin::Float32    # score lead a miss needs over the coldest resident
end

_hot_config() = _HotConfig(
    Float32(_hot_env_float("GESSO_GPU_HOT_DECAY", 0.99)),
    max(1, _hot_env_int("GESSO_GPU_HOT_FILL", 2)),
    _hot_env_int("GESSO_GPU_HOT_PREFILL_KEEP", typemax(Int)),
    Float32(_hot_env_float("GESSO_GPU_HOT_MARGIN", 1.0)),
)

# One slot holds an expert's gate, up and down rows back to back. The three
# projections then read the pool through three views offset by the gate and
# up sizes, and one offset table serves all three launches.
mutable struct _HotLayer
    layer::Int
    nslots::Int
    stride::Int
    pool::Union{Nothing,_CuVec{UInt8}}
    views::Union{Nothing,NTuple{3,_CuVec{UInt8}}}
    base::CuPtr{UInt8}
    slot_of::Vector{Int32}     # expert -> slot, 0 when not resident
    expert_in::Vector{Int32}   # slot -> expert, 0 empty, -1 copy in flight
    free::Vector{Int}
    score::Vector{Float32}
    seen::Vector{Int}
    t::Int
    # [offsets; columns; weight bits], pinned so one async copy carries it
    meta_h::Vector{Int32}
    meta_d::Union{Nothing,_CuVec{Int32}}
    pend_slot::Vector{Int}
    pend_exp::Vector{Int}
    copying::Bool
    direct::Bool               # slabs page-locked: copies skip the staging
    ready::CuEvent
    used::CuEvent              # recorded after the layer's last hybrid token
    last::Vector{Int}          # the experts picked at token t
    staging::Vector{UInt8}
    hits::Int
    picks::Int
    swaps::Int
    retry_t::Int               # decode skips the pool allocation until t reaches it
end

const _HOT = IdDict{Any,_HotLayer}()
const _HOT_CFG = Ref(_HotConfig(0.99f0, 2, typemax(Int), 1.0f0))
const _HOT_COPY_STREAM = Ref{Union{Nothing,CuStream}}(nothing)
const _HOT_MISS_E = Int[]
const _HOT_MISS_W = Float32[]

# Test hook: when set, every hybrid token also runs all of its experts on the
# CPU and records the relative L2 error of the hybrid result. A slot holding
# the wrong expert shows up as an error near 1; rounding alone stays near 1e-2.
const _HOT_VERIFY = Ref(false)
const _HOT_VERIFY_ERR = Float64[]

_hot_layer(gate) = isempty(_HOT) ? nothing : get(_HOT, gate, nothing)

function _hot_copy_stream()
    s = _HOT_COPY_STREAM[]
    if s === nothing
        s = CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
        _HOT_COPY_STREAM[] = s
    end
    return s
end

const _HOT_TYPES = (Gesso.Inference.GGML_TYPE_IQ2_XS, Gesso.Inference.GGML_TYPE_IQ3_XXS,
    Gesso.Inference.GGML_TYPE_IQ4_XS)

function _hot_supported(g, u, d)
    all(s -> s.ggml_type in _HOT_TYPES && s.bytes_per_expert % 4 == 0, (g, u, d)) || return false
    return g.n_exp == u.n_exp == d.n_exp && g.ne0 == u.ne0 && g.ne1 == u.ne1
end

function _hot_release!(hot::_HotLayer)
    hot.pool === nothing && return
    hot.copying && _cuda_event_wait(hot.ready)
    hot.copying = false
    empty!(hot.pend_slot)
    empty!(hot.pend_exp)
    fill!(hot.slot_of, 0)
    fill!(hot.expert_in, 0)
    hot.free = collect(hot.nslots:-1:1)
    CUDA.unsafe_free!(hot.pool)
    hot.pool = nothing
    hot.views = nothing
    return
end

# The pools take nearly all free VRAM, so a long prompt's activations do not fit
# beside them. CUDA's allocator calls purge! when an allocation fails, before it
# escalates to a full GC, so the pools go only when memory is actually short and
# the next decode token refills them from the scores the prefill has warmed. A
# fixed prompt-length threshold either dropped the pools for short prompts that
# fit, costing ~40 tokens of refill, or kept them for long ones that then ran
# out. _HOT_BUSY covers a layer from its pool allocation to its last launch, so
# the allocations inside that span cannot free the pool they are about to read.
#
# Releasing every pool cost a P1024 prefill 29 of its 30: the layers it had not
# reached yet then ran their experts on the CPU (~0.32 s each) or re-allocated
# under _HOT_BUSY, where CUDA's retry escalates to a GC. Layers run in a cycle
# (prefill, then decode from layer 1 again), so the pool needed furthest ahead
# belongs to the layer just finished; purge! releases in that order and stops
# once the prompt's activations fit.
struct _HotPurge <: CUDA.CUDACore.Reclaimable end
const _HOT_BUSY = Ref(false)
const _HOT_NEXT = Ref(1)       # the hot layer the model reaches next
const _HOT_SEQ = Ref(1)        # rows in the current forward pass
const _HOT_PURGE_FLOOR = Int64(512) << 20
const _HOT_ALLOC_MARGIN = Int64(64) << 20
# Decode tokens a layer waits after its pool did not fit. A long prompt's KV
# cache holds the memory until the generation ends, so retrying every token
# would only repeat the failure; the next generation gets the pool back.
const _HOT_RETRY_TOKENS = 64

# Entering a hot layer's MoE, and leaving it. Between the two the layer itself
# is next; afterwards it is the furthest away.
_hot_enter!(hot::_HotLayer, seq::Int) = (_HOT_NEXT[] = hot.layer; _HOT_SEQ[] = seq; nothing)
_hot_leave!(hot::_HotLayer) = (_HOT_NEXT[] = hot.layer + 1; nothing)

function CUDA.CUDACore.purge!(::_HotPurge)
    _HOT_BUSY[] && return
    live = [hot for hot in values(_HOT) if hot.pool !== nothing]
    isempty(live) && return
    span = maximum(hot -> hot.layer, values(_HOT)) + 1
    next = _HOT_NEXT[]
    sort!(live; by=hot -> mod(hot.layer - next, span), rev=true)
    target = max(_HOT_PURGE_FLOOR, Int64(_HOT_PF_CFG[].headroom) * _HOT_SEQ[])
    freed = Int64(0)
    for hot in live
        freed >= target && break
        freed += Int64(hot.nslots) * hot.stride
        _hot_release!(hot)
    end
    # The frees are stream-ordered; without this the retry that follows can
    # miss them and CUDA climbs to its next rung anyway.
    CUDA.device_synchronize()
    return
end

function _hot_release_all!()
    for hot in values(_HOT)
        _hot_release!(hot)
    end
    empty!(_HOT)
    return
end

# Called at the end of to_device. The budget splits evenly over the host
# layers; pools are allocated on the first decode token, not here.
function _hot_register!(blocks)
    _hot_release_all!()
    _HOT_CFG[] = _hot_config()
    _HOT_PF_CFG[] = _hot_prefill_config()
    triples = filter(t -> _hot_supported(t[2], t[3], t[4]), _moe_expert_triples(blocks))
    isempty(triples) && return
    per_layer = div(_hot_budget_bytes(), Int64(length(triples)))
    pin = get(ENV, "GESSO_GPU_HOT_PIN", "0") == "1"
    slots = Int[]
    for (li, g, u, d) in triples
        stride = g.bytes_per_expert + u.bytes_per_expert + d.bytes_per_expert
        # the qgemv offsets are Int32
        nslots = min(g.n_exp, Int(div(per_layer, stride)), div(typemax(Int32), stride))
        push!(slots, nslots)
        nslots > 0 || continue
        n_exp = g.n_exp
        _HOT[g] = _HotLayer(li, nslots, stride, nothing, nothing, CuPtr{UInt8}(0),
            zeros(Int32, n_exp), zeros(Int32, nslots), collect(nslots:-1:1),
            zeros(Float32, n_exp), zeros(Int, n_exp), 0,
            Int32[], nothing, Int[], Int[], false, pin && _hot_pin_slabs!((g, u, d)),
            CuEvent(CUDA.EVENT_DISABLE_TIMING),
            CuEvent(CUDA.EVENT_DISABLE_TIMING), Int[], UInt8[], 0, 0, 0, 0)
    end
    @debug "hot experts registered" layers=length(_HOT) slots=slots per_layer_mib=div(per_layer, Int64(1024 * 1024))
    return
end

# The CPU path reads the copies from then on, so the mmap pages behind them go
# cold and the page cache can drop them: the copy moves the bytes rather than
# doubling them. Returns false, and leaves the slabs alone, if any pin fails.
function _hot_pin_slabs!(slabs)
    copies = Vector{Vector{UInt8}}(undef, length(slabs))
    for (i, s) in enumerate(slabs)
        n = s.n_exp * s.bytes_per_expert
        v = Vector{UInt8}(undef, n)
        copyto!(v, 1, s.data, s.offset0 + 1, n)
        try
            CUDA.pin(v)
        catch err
            @debug "hot experts: page-locking a host slab failed, swaps stage" exception = err
            return false
        end
        copies[i] = v
    end
    for (s, v) in zip(slabs, copies)
        s.data = v
        s.offset0 = 0
    end
    return true
end

# Pools, metadata and staging, allocated on the compute stream the kernels
# run on, so the views' launches take the @fastcuda fast path. A layer that
# does not fit runs on the CPU-only path for _HOT_RETRY_TOKENS decode tokens,
# then tries again. Retiring it for good cost decode 62 -> 34 tok/s for every
# generation after one long prompt.
function _hot_alloc!(hot::_HotLayer, g, u, n_used::Int)
    hot.pool === nothing || return true
    hot.nslots >= 1 || return false
    hot.t < hot.retry_t && return false
    _hot_prefill_release!()
    # A pool that fits by a few MiB still failed to allocate after a trim; the
    # driver wants room of its own. Only DROP, behind two GCs, then rescued it.
    bytes = Int64(hot.nslots) * hot.stride + _HOT_ALLOC_MARGIN
    # a failed allocation under _HOT_BUSY costs a full GC; this check costs none
    if _hot_device_avail() < bytes
        hot.retry_t = hot.t + _HOT_RETRY_TOKENS
        return false
    end
    if CUDA.free_memory() < bytes
        # This runs under _HOT_BUSY, so CUDA's retry cannot purge and its next
        # rescue is a full GC (~0.25 s). What blocks the pool is usually CUDA's
        # own cache, fragmented by the prompt's activations rather than live:
        # handing it back to the driver frees the same bytes with no GC.
        CUDA.synchronize()
        dev = CUDA.device()
        CUDA.CUDACore.stream_ordered(dev) && CUDA.CUDACore.trim(CUDA.CUDACore.pool_create(dev))
        # what the trim could not hand back is fragmented; the allocation would
        # fail only after the GC, and decode retries every _HOT_RETRY_TOKENS
        if CUDA.free_memory() < bytes
            hot.retry_t = hot.t + _HOT_RETRY_TOKENS
            return false
        end
    end
    # CUDA's retry for a failed allocation purges, syncs and trims, then runs
    # two full GCs. In decode they free nothing: after a P4096 prompt, live
    # equalled used (6835 MiB) at all seven of them, because what holds the
    # memory is that generation's KV and buffers. The GCs cost 1.7 s of 64
    # tokens. With GC disabled they return at once; purge, sync and trim
    # still run, and a pool that does not fit waits _HOT_RETRY_TOKENS.
    gc_on = GC.enable(false)
    try
        pool = CuVector{UInt8}(undef, hot.nslots * hot.stride)
        base = pointer(pool)
        n = length(pool)
        o2 = g.bytes_per_expert
        o3 = o2 + u.bytes_per_expert
        hot.views = (pool,
            unsafe_wrap(CuArray, base + o2, n - o2),
            unsafe_wrap(CuArray, base + o3, n - o3))
        hot.pool = pool
        hot.base = base
    catch err
        err isa CUDA.OutOfGPUMemoryError || rethrow()
        @debug "hot experts: pool does not fit" layer=hot.layer
        hot.retry_t = hot.t + _HOT_RETRY_TOKENS
        return false
    finally
        GC.enable(gc_on)
    end
    if length(hot.meta_h) != 3n_used
        hot.meta_h = Vector{Int32}(undef, 3n_used)
        CUDA.pin(hot.meta_h)
        hot.meta_d = CuVector{Int32}(undef, 3n_used)
    end
    nstage = hot.direct ? 0 : _HOT_CFG[].fill * hot.stride
    if length(hot.staging) != nstage
        hot.staging = Vector{UInt8}(undef, nstage)
        CUDA.pin(hot.staging)
    end
    # the copy stream must not write the pool before its allocation lands
    _cuda_stream_wait()
    return true
end

@inline function _hot_score(hot::_HotLayer, e::Int, decay::Float32)
    dt = hot.t - hot.seen[e]
    return dt == 0 ? hot.score[e] : hot.score[e] * decay^dt
end

function _hot_observe!(hot::_HotLayer, experts, n_used::Int)
    decay = _HOT_CFG[].decay
    hot.t += 1
    resize!(hot.last, n_used)
    @inbounds for k in 1:n_used
        e = experts[k]
        hot.score[e] = _hot_score(hot, e, decay) + 1.0f0
        hot.seen[e] = hot.t
        hot.last[k] = e
    end
    return
end

function _hot_commit!(hot::_HotLayer)
    (hot.copying && CUDA.isdone(hot.ready)) || return
    @inbounds for j in eachindex(hot.pend_slot)
        s = hot.pend_slot[j]
        e = hot.pend_exp[j]
        hot.slot_of[e] = s
        hot.expert_in[s] = e
    end
    hot.swaps += length(hot.pend_slot)
    empty!(hot.pend_slot)
    empty!(hot.pend_exp)
    hot.copying = false
    return
end

# Picks the copies from the last token's misses: the best-scoring ones, into
# empty slots first, then at most one eviction of the coldest resident that
# the last token did not pick. Each swap costs a host memcpy of the expert, so
# a miss must outscore the victim by the margin; without it, experts near the
# residency edge trade places every few tokens.
function _hot_plan!(hot::_HotLayer, experts, n_used::Int)
    isempty(hot.pend_slot) || return
    decay = _HOT_CFG[].decay
    evicted = false
    for _ in 1:_HOT_CFG[].fill
        best = 0
        bs = -1.0f0
        @inbounds for k in 1:n_used
            e = experts[k]
            (hot.slot_of[e] == 0 && !(e in hot.pend_exp)) || continue
            if hot.score[e] > bs
                best = e
                bs = hot.score[e]
            end
        end
        best == 0 && break
        if !isempty(hot.free)
            slot = pop!(hot.free)
        elseif !evicted
            slot = 0
            vs = typemax(Float32)
            @inbounds for s in 1:hot.nslots
                e = Int(hot.expert_in[s])
                (e > 0 && hot.seen[e] != hot.t) || continue
                v = _hot_score(hot, e, decay)
                if v < vs
                    slot = s
                    vs = v
                end
            end
            (slot == 0 || vs + _HOT_CFG[].margin >= bs) && break
            hot.slot_of[hot.expert_in[slot]] = 0
            evicted = true
        else
            break
        end
        hot.expert_in[slot] = -1
        push!(hot.pend_slot, slot)
        push!(hot.pend_exp, best)
    end
    return
end

# Starts the planned copies. A victim slot may still be read by this layer's
# last hybrid token, so the copy stream waits for that token's kernels on the
# device; the host does not block.
function _hot_issue!(hot::_HotLayer, g, u, d)
    (isempty(hot.pend_slot) || hot.copying) && return
    cs = _hot_copy_stream()
    CUDA.wait(hot.used, cs)
    st = hot.stride
    if hot.direct
        GC.@preserve g u d for j in eachindex(hot.pend_slot)
            e = hot.pend_exp[j]
            dst = hot.base + (hot.pend_slot[j] - 1) * st
            o = 0
            for sl in (g, u, d)
                nb = sl.bytes_per_expert
                src = pointer(sl.data) + Gesso.Inference._quant_expert_off0(sl, e)
                unsafe_copyto!(dst + o, src, nb; async=true, stream=cs)
                o += nb
            end
        end
        CUDA.record(hot.ready, cs)
        hot.copying = true
        return
    end
    GC.@preserve g u d begin
        stg = pointer(hot.staging)
        for j in eachindex(hot.pend_slot)
            e = hot.pend_exp[j]
            dst = stg + (j - 1) * st
            o = 0
            for sl in (g, u, d)
                nb = sl.bytes_per_expert
                src = pointer(sl.data) + Gesso.Inference._quant_expert_off0(sl, e)
                unsafe_copyto!(dst + o, src, nb)
                o += nb
            end
        end
        for j in eachindex(hot.pend_slot)
            unsafe_copyto!(hot.base + (hot.pend_slot[j] - 1) * st, stg + (j - 1) * st, st;
                async=true, stream=cs)
        end
    end
    CUDA.record(hot.ready, cs)
    hot.copying = true
    return
end

# y = sum over the hits of w[k] * D[:, k], plus the CPU's partial sum. The
# weights ride in the launch metadata as Float32 bits.
function _hot_combine_kernel!(y, D, meta, ymiss, h::Int, nh::Int, wbase::Int)
    i = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    i <= h || return
    acc = 0.0f0
    @inbounds for k in 1:nh
        acc += D[i, k] * reinterpret(Float32, meta[wbase + k])
    end
    @inbounds y[i] = acc + ymiss[i]
    return
end

function _hot_miss_bufs(n_used::Int)
    length(_HOT_MISS_E) == n_used || (resize!(_HOT_MISS_E, n_used); resize!(_HOT_MISS_W, n_used))
    return _HOT_MISS_E, _HOT_MISS_W
end

# Called while the host waits on a decode token's router logits, before the
# token's picks are known: lands finished copies and starts the next ones.
function _hot_prefetch!(hot::_HotLayer, g, u, d)
    hot.pool === nothing && return
    _HOT_BUSY[] = true
    try
        _hot_commit!(hot)
        _hot_plan!(hot, hot.last, length(hot.last))
        _hot_issue!(hot, g, u, d)
    finally
        _HOT_BUSY[] = false
    end
    return
end

# One decode token of a host MoE layer, routed already (b.experts, b.wbuf).
function _moe_hot_token!(backend, hot::_HotLayer, y, xd, b, gate, up, down, n_used::Int, s)
    h = length(b.xrow)
    _hot_observe!(hot, b.experts, n_used)
    me, mw = _hot_miss_bufs(n_used)
    meta = hot.meta_h
    nh = 0
    nm = 0
    @inbounds for k in 1:n_used
        e = b.experts[k]
        sl = Int(hot.slot_of[e])
        if sl > 0
            nh += 1
            meta[nh] = Int32((sl - 1) * hot.stride)
            meta[n_used + nh] = Int32(nh)
            meta[2n_used + nh] = reinterpret(Int32, b.wbuf[k])
        else
            nm += 1
            me[nm] = e
            mw[nm] = b.wbuf[k]
        end
    end
    hot.hits += nh
    hot.picks += n_used
    ff = gate.ne1
    if nh > 0
        Gesso.Inference.@decode_stage :moe_qgemv backend begin
            Gbat = _dev_scratch(:moe_gbat, ff, n_used)
            Ubat = _dev_scratch(:moe_ubat, ff, n_used)
            Hbat = _dev_scratch(:moe_hbat, ff, n_used)
            Dbat = _dev_scratch(:moe_dbat, h, n_used)
            md = hot.meta_d
            unsafe_copyto!(pointer(md), pointer(meta), 3n_used; async=true, stream=s)
            gv, uv, dv = hot.views
            xt = reshape(xd, h, 1)
            _cuda_launch_qgemv!(Gbat, xt, gv, gate.ggml_type, md, n_used, nh, gate.ne0, gate.ne1, 1.0f0, 0.0f0)
            _cuda_launch_qgemv!(Ubat, xt, uv, up.ggml_type, md, n_used, nh, up.ne0, up.ne1, 1.0f0, 0.0f0;
                quantize=false)
            _silu_mul!(backend, Hbat, Ubat, Gbat)
            _cuda_launch_qgemv!(Dbat, Hbat, dv, down.ggml_type, md, n_used, nh, down.ne0, down.ne1, 1.0f0, 0.0f0)
        end
    end
    if nm > 0
        Gesso.Inference.@decode_stage :moe_host nothing Gesso.Inference._moe_host_experts_auto!(
            b.yrow, b.xrow, me, mw, nm, gate, up, down,
        )
    else
        fill!(b.yrow, 0.0f0)
    end
    copyto!(b.y, 1, b.yrow, 1, h)
    Gesso.Runtime.pool_hold!(_MOE_HOST_HOLD_NS)
    Gesso.Inference.@decode_stage :moe_h2d backend begin
        Gesso.Inference._profile_copy!(false)
        if nh == 0
            unsafe_copyto!(pointer(y), pointer(b.y), h; async=true, stream=s)
        else
            ymiss = _dev_scratch(:hot_ymiss, h)
            unsafe_copyto!(pointer(ymiss), pointer(b.y), h; async=true, stream=s)
            @fastcuda threads = 256 blocks = cld(h, 256) _hot_combine_kernel!(
                y, _dev_scratch(:moe_dbat, h, n_used), hot.meta_d, ymiss, h, nh, 2n_used)
        end
    end
    CUDA.record(hot.used, s)
    _HOT_VERIFY[] && _hot_verify!(y, b, gate, up, down, n_used)
    return y
end

# Prefill of a hot layer.
#
# The grouped CPU prefill spends about 114 ms per hot layer on a 256-token
# prompt of the 35B, and that time is arithmetic: each (token, expert) pair
# costs about 45 us of the host's 16 threads, while the expert's bytes cost
# about 155 us to send over PCIe once for all of its pairs. So the GPU runs the
# pairs of every resident expert, plus those of as many other experts as the
# host can upload in the time the CPU takes over the rest. The CPU keeps the
# experts with the fewest pairs, where an upload buys the least.
#
# Uploads cycle through the pool's slots in rounds: free slots first, then
# residents this prompt does not pick, then the picked residents once their
# pairs are done. A round's compute waits on its copies and a copy into a slot
# waits on the compute that last read it, so copies and compute overlap with
# each other and with the CPU. The uploads go in ascending pair count, so the
# busiest experts are the ones left resident for decode.
#
# A layer whose pool a long prompt's activations have purged, with no room to
# re-allocate it, streams its experts through a small ring shared by all
# layers (_STREAM slots) instead. It keeps no residents for decode, but its
# pairs leave the CPU: on a 1024-token prompt the CPU takes ~0.5 s over such a
# layer, the uploads and GEMVs tens of ms.
#
# GESSO_GPU_HOT_PREFILL=0 keeps hot-layer prefill on the CPU. The cost model's
# rates are tunable: _GBPS (PCIe), _EXPERT_US and _PAIR_US (CPU).
struct _HotPrefillConfig
    on::Bool
    max_seq::Int        # longer prompts stay on the CPU; D holds h floats per pair
    round::Int          # uploads per round
    seg::Int            # pairs per launch
    upload_s::Float64   # seconds per byte
    cpu_expert_s::Float64
    cpu_pair_s::Float64
    headroom::Int       # device bytes per prompt token: a purge frees at least this, a pass is sized by it
    stream::Int         # slots in the shared ring of a layer without a pool, 0 none
end

_hot_prefill_config() = _HotPrefillConfig(
    get(ENV, "GESSO_GPU_HOT_PREFILL", "1") == "1",
    min(_hot_env_int("GESSO_GPU_HOT_PREFILL_MAX_SEQ", 4096), 65535),
    max(1, _hot_env_int("GESSO_GPU_HOT_PREFILL_ROUND", 16)),
    clamp(_hot_env_int("GESSO_GPU_HOT_PREFILL_SEG", 1024), 1, 65535),
    1 / (1e9 * _hot_env_float("GESSO_GPU_HOT_PREFILL_GBPS", 6.5)),
    1e-6 * _hot_env_float("GESSO_GPU_HOT_PREFILL_EXPERT_US", 70),
    1e-6 * _hot_env_float("GESSO_GPU_HOT_PREFILL_PAIR_US", 45),
    1024 * max(0, _hot_env_int("GESSO_GPU_HOT_PREFILL_HEADROOM_KIB", 768)),
    max(0, _hot_env_int("GESSO_GPU_HOT_PREFILL_STREAM", 32)),
)

const _HOT_PF_CFG = Ref(_HotPrefillConfig(true, 4096, 16, 1024, 1 / 6.5e9, 70e-6, 45e-6, 768 * 1024, 32))

const _HOT_STREAM = Ref{Union{Nothing,_HotLayer}}(nothing)

# The shared ring dressed as `hot`, empty. Allocated before _HOT_BUSY, so a
# short device may purge pools for it; nothing when it still does not fit.
function _hot_stream_layer(hot::_HotLayer, g, u, n_exp::Int, cfg::_HotPrefillConfig)
    (hot.direct && cfg.stream > 0) || return nothing
    sl = _HOT_STREAM[]
    if sl === nothing
        sl = _HotLayer(0, 0, 0, nothing, nothing, CuPtr{UInt8}(0), Int32[], Int32[], Int[],
            Float32[], Int[], 0, Int32[], nothing, Int[], Int[], false, true,
            CuEvent(CUDA.EVENT_DISABLE_TIMING), CuEvent(CUDA.EVENT_DISABLE_TIMING), Int[], UInt8[], 0, 0, 0, 0)
        _HOT_STREAM[] = sl
    end
    need = cfg.stream * hot.stride
    if sl.pool === nothing || length(sl.pool) < need
        _hot_stream_release!()
        try
            sl.pool = CuVector{UInt8}(undef, need)
        catch err
            err isa CUDA.OutOfGPUMemoryError || rethrow()
            return nothing
        end
    end
    pool = sl.pool
    base = pointer(pool)
    n = length(pool)
    o2 = g.bytes_per_expert
    o3 = o2 + u.bytes_per_expert
    sl.views = (pool, unsafe_wrap(CuArray, base + o2, n - o2), unsafe_wrap(CuArray, base + o3, n - o3))
    sl.base = base
    sl.layer = hot.layer
    sl.stride = hot.stride
    sl.nslots = n ÷ hot.stride
    fill!(resize!(sl.slot_of, n_exp), 0)
    fill!(resize!(sl.expert_in, sl.nslots), 0)
    empty!(sl.free)
    append!(sl.free, sl.nslots:-1:1)
    return sl
end

# Every read and write of the ring is ordered before the main stream's last
# prefill launch, so a stream-ordered free is safe.
function _hot_stream_release!()
    sl = _HOT_STREAM[]
    (sl === nothing || sl.pool === nothing) && return
    CUDA.unsafe_free!(sl.pool)
    sl.pool = nothing
    sl.views = nothing
    return
end

# Decode wants the prefill's device buffers back for its pools: after a
# 4096-token prompt D alone holds 256 MiB, two pools. The device-bank MoE batch
# and the dense F16 GEMM scratch are prefill-only too, another 78 MiB there.
function _hot_prefill_release!()
    _hot_stream_release!()
    _free_fields!(_HOT_PREFILL, (:meta_d, :xt, :g, :u, :a, :d))
    _free_fields!(_MOE_PAIR_BATCH, (:meta_d, :xt, :g, :u, :a, :d))
    _free_fields!(_DQ_GEMM, (:w16, :x16))
    return
end

function _free_fields!(s, fields)
    for f in fields
        buf = getfield(s, f)
        buf === nothing && continue
        CUDA.unsafe_free!(buf)
        setfield!(s, f, nothing)
    end
    return
end

# Free device memory plus what CUDA's pool holds that no array uses. With
# JULIA_CUDA_MEMORY_POOL=none there is no pool and cached_memory() is missing.
function _hot_device_avail()
    CC = CUDA.CUDACore
    cached = coalesce(CC.cached_memory(), 0)
    return Int64(CUDA.free_memory()) + max(Int64(0), Int64(cached) - Int64(CC.used_memory()))
end

# Device memory a prefill pass can claim: purge! hands back the pools and the
# prefill buffers regrow to the pass.
function _prefill_device_avail()
    pools = sum(hot -> hot.pool === nothing ? Int64(0) : Int64(length(hot.pool)), values(_HOT); init=Int64(0))
    P = _HOT_PREFILL
    bufs = sum(f -> (b = getfield(P, f); b === nothing ? Int64(0) : Int64(sizeof(b))),
        (:meta_d, :xt, :g, :u, :a, :d); init=Int64(0))
    return _hot_device_avail() + pools + bufs
end

# A pass of n rows over K keys: per-row activations (the prefill headroom
# estimate, ~0.75 MiB on the 35B MoE) and the attention's key-tile scores.
# Attention reads the KV heads in place and tiles the keys
# (_attention_prefill_device!), so nothing grows with K past the tile. The
# dense dequant GEMM's scratch is freed at every prefill's end and regrown by
# the next (a fixed weight tile plus the pass's Float16 activations); P16384
# ran out of memory on that regrowth while the estimate said it fit. The
# margin is the shared ring and the driver's room to place the rest.
const _PREFILL_PASS_MARGIN = Int64(160) << 20
const _PREFILL_PASS_STEP = 128

_prefill_pass_bytes(n::Int, K::Int, qdim::Int, group::Int) =
    Int64(_HOT_PF_CFG[].headroom) * n + 4 * Int64(n) * _attn_prefill_tile(n, K) +
    (Int64(_dq_gemm_scratch_mib()) << 20) + 2 * Int64(n) * qdim

# P4096 in passes of 1024 ran out of memory next to hot+pin pools; a fixed
# smaller chunk costs short prompts throughput they could have had. The pass
# steps down by _PREFILL_PASS_STEP rows until its estimate fits.
function Gesso.Inference._prefill_pass_rows(::CUDABackend, left::Int, done::Int, chunk::Int,
    qdim::Int, group::Int)
    n = min(chunk, left)
    budget = _prefill_device_avail() - _PREFILL_PASS_MARGIN
    while n > _PREFILL_PASS_STEP && _prefill_pass_bytes(n, done + n, qdim, group) > budget
        n = (n - 1) ÷ _PREFILL_PASS_STEP * _PREFILL_PASS_STEP
    end
    return n
end

Gesso.Inference._prefill_end!(::CUDABackend) = _hot_prefill_release!()

# Grow-only buffers shared by every hot layer. GPU pairs are numbered in round
# order, expert by expert, token order within an expert; D's column g is pair g.
mutable struct _HotPrefill
    meta_h::Vector{Int32}    # pinned: per launch [offsets; columns; tokens], then the combine table
    meta_d::Union{Nothing,_CuVec{Int32}}
    xt::Union{Nothing,_CuVec{Float32}}
    g::Union{Nothing,_CuVec{Float32}}
    u::Union{Nothing,_CuVec{Float32}}
    a::Union{Nothing,_CuVec{Float32}}
    d::Union{Nothing,_CuVec{Float32}}
    ev::Vector{CuEvent}
    ngpu::Int
    gpoff::Int
    woff::Int
    cnt::Vector{Int}
    rnd::Vector{Int}         # expert -> round, -1 on the CPU
    slot::Vector{Int}        # expert -> slot read
    first::Vector{Int}       # expert -> its first pair
    fill::Vector{Int}
    order::Vector{Int}       # GPU experts in pair order
    cand::Vector{Int}
    ring::Vector{Int}
    lastr::Vector{Int}       # slot -> last round that reads it, -1 none
    rstart::Vector{Int}      # round r -> first pair, at index r + 1
    cpu::Vector{Bool}
end

const _HOT_PREFILL = _HotPrefill(Int32[], nothing, nothing, nothing, nothing, nothing, nothing,
    CuEvent[], 0, 0, 0, Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[], Bool[])

function _hotp_dev(buf, ::Type{T}, n::Int) where {T}
    buf !== nothing && length(buf) >= n && return buf
    buf === nothing || CUDA.unsafe_free!(buf)
    return CuVector{T}(undef, cld(max(n, 1), 1 << 14) << 14)
end

function _hotp_event(P::_HotPrefill, i::Int)
    while length(P.ev) < i
        push!(P.ev, CuEvent(CUDA.EVENT_DISABLE_TIMING))
    end
    return P.ev[i]
end

# Pairs a:b (global numbering), in launches of at most `seg` pairs.
function _hotp_compute!(backend, hot::_HotLayer, P::_HotPrefill, xt, gate, up, down,
    a::Int, b::Int, seg::Int, h::Int, ff::Int)
    gv, uv, dv = hot.views
    while a <= b
        n = min(seg, b - a + 1)
        md = unsafe_wrap(CuArray, pointer(P.meta_d, 3(a - 1) + 1), 3n)
        G = unsafe_wrap(CuArray, pointer(P.g), (ff, n))
        U = unsafe_wrap(CuArray, pointer(P.u), (ff, n))
        A = unsafe_wrap(CuArray, pointer(P.a), (ff, n))
        D = unsafe_wrap(CuArray, pointer(P.d, (a - 1) * h + 1), (h, n))
        _cuda_launch_qgemv!(G, xt, gv, gate.ggml_type, md, n, n, gate.ne0, gate.ne1, 1.0f0, 0.0f0;
            xcoloff=2n)
        _cuda_launch_qgemv!(U, xt, uv, up.ggml_type, md, n, n, up.ne0, up.ne1, 1.0f0, 0.0f0;
            quantize=false, xcoloff=2n)
        _silu_mul!(backend, A, U, G)
        _cuda_launch_qgemv!(D, A, dv, down.ggml_type, md, n, n, down.ne0, down.ne1, 1.0f0, 0.0f0)
        a += n
    end
    return
end

# A routed prompt (b.E, b.W, b.live) through hot layer `hot`. Returns false, with
# nothing issued, when the layer must take the CPU path. On true, Y holds the
# CPU experts' partial sums and _hot_prefill_add! adds the GPU's once Y is on
# the device.
function _moe_hot_prefill!(backend, hot::_HotLayer, xd, b, X, Y, gate, up, down,
    seq::Int, n_exp::Int, n_used::Int, s)
    cfg = _HOT_PF_CFG[]
    (cfg.on && seq <= cfg.max_seq) || return false
    h = size(X, 2)
    ff = gate.ne1
    P = _HOT_PREFILL
    cnt = resize!(P.cnt, n_exp)
    fill!(cnt, 0)
    npairs = 0
    @inbounds for t in 1:seq
        b.live[t] || continue
        for k in 1:n_used
            cnt[b.E[k, t]] += 1
        end
        npairs += n_used
    end
    npairs == 0 && return false
    seg = cfg.seg
    nmeta = 3npairs + 2seq * n_used
    # Every device allocation comes before _HOT_BUSY: one that runs short may
    # purge the pools, which is safe only while no launch reads them.
    P.meta_d = _hotp_dev(P.meta_d, Int32, nmeta)
    if length(P.meta_h) < nmeta
        P.meta_h = Vector{Int32}(undef, 2nmeta)
        CUDA.pin(P.meta_h)
    end
    P.xt = _hotp_dev(P.xt, Float32, h * seq)
    P.g = _hotp_dev(P.g, Float32, ff * min(seg, npairs))
    P.u = _hotp_dev(P.u, Float32, ff * min(seg, npairs))
    P.a = _hotp_dev(P.a, Float32, ff * min(seg, npairs))
    P.d = _hotp_dev(P.d, Float32, h * npairs)
    _cuda_q8_buffers!(_CUDA_Q8_EXPERT, max(seq * (h ÷ 32), min(seg, npairs) * (ff ÷ 32)))
    hot.copying && (_cuda_event_wait(hot.ready); _hot_commit!(hot))
    # A pool purged earlier in this prompt streams through the shared ring
    # rather than being allocated again. The allocation runs under _HOT_BUSY;
    # on long prompts its first attempt failed although the precheck passed,
    # and CUDA's retry paid two full GCs (~0.36 s) per layer before it landed,
    # only for a later layer to purge it again. At P8192 that was 15-30 s of a
    # 38-54 s prefill; without it, 26.4 s and no GC. Prompts that fit in one
    # pass never purge, so P256-P1024 do not change. Decode allocates it back.
    lay = hot
    if hot.pool === nothing
        lay = _hot_stream_layer(hot, gate, up, n_exp, cfg)
        lay === nothing && return false
    end
    _HOT_BUSY[] = true
    try
        Gesso.Inference.@decode_stage :moe_qgemv nothing _hot_prefill_issue!(
            backend, lay, P, cfg, xd, b, gate, up, down, seq, n_exp, n_used, npairs, s)
    finally
        _HOT_BUSY[] = false
    end
    cpu = resize!(P.cpu, n_exp)
    @inbounds for e in 1:n_exp
        cpu[e] = P.rnd[e] < 0
    end
    Gesso.Inference.@decode_stage :moe_host nothing Gesso.Inference._moe_host_prefill!(
        Y, X, b.E, b.W, b.live, n_exp, gate, up, down; cpu=cpu,
    )
    return true
end

function _hot_prefill_issue!(backend, hot::_HotLayer, P::_HotPrefill, cfg::_HotPrefillConfig,
    xd, b, gate, up, down, seq::Int, n_exp::Int, n_used::Int, npairs::Int, s)
    h = size(xd, 2)
    ff = gate.ne1
    cnt = P.cnt
    cand = P.cand
    empty!(cand)
    @inbounds for e in 1:n_exp
        cnt[e] > 0 && hot.slot_of[e] == 0 && push!(cand, e)
    end
    sort!(cand; by=e -> cnt[e], rev=true)
    nup = 0
    if hot.direct && !isempty(cand)
        beta = hot.stride * cfg.upload_s
        rest = 0.0
        @inbounds for e in cand
            rest += cfg.cpu_expert_s + cfg.cpu_pair_s * cnt[e]
        end
        best = rest
        @inbounds for j in eachindex(cand)
            rest -= cfg.cpu_expert_s + cfg.cpu_pair_s * cnt[cand[j]]
            v = max(j * beta, rest)
            v < best && (best = v; nup = j)
        end
    end
    # uploads in ascending pair count
    reverse!(cand, 1, nup)

    ring = P.ring
    empty!(ring)
    append!(ring, hot.free)
    @inbounds for sl in 1:hot.nslots
        e = Int(hot.expert_in[sl])
        e > 0 && cnt[e] == 0 && push!(ring, sl)
    end
    @inbounds for sl in 1:hot.nslots
        e = Int(hot.expert_in[sl])
        e > 0 && cnt[e] > 0 && push!(ring, sl)
    end
    N = length(ring)
    C = min(cfg.round, N)
    R = nup == 0 ? 0 : cld(nup, C)

    rnd = resize!(P.rnd, n_exp)
    slot = resize!(P.slot, n_exp)
    pfirst = resize!(P.first, n_exp)
    fillc = resize!(P.fill, n_exp)
    fill!(rnd, -1)
    fill!(fillc, 0)
    lastr = resize!(P.lastr, hot.nslots)
    fill!(lastr, -1)
    order = P.order
    empty!(order)
    @inbounds for e in 1:n_exp
        sl = Int(hot.slot_of[e])
        (cnt[e] > 0 && sl > 0) || continue
        rnd[e] = 0
        slot[e] = sl
        lastr[sl] = 0
        push!(order, e)
    end
    nres = length(order)
    @inbounds for j in 1:nup
        e = cand[j]
        rnd[e] = cld(j, C)
        slot[e] = ring[(j - 1) % N + 1]
        push!(order, e)
    end
    rstart = resize!(P.rstart, R + 2)
    g = 1
    rstart[1] = 1
    @inbounds for (idx, e) in enumerate(order)
        idx > nres && (idx - nres - 1) % C == 0 && (rstart[rnd[e] + 1] = g)
        pfirst[e] = g
        g += cnt[e]
    end
    rstart[R + 2] = g
    ngpu = g - 1

    meta = P.meta_h
    gpoff = 3ngpu
    woff = gpoff + seq * n_used
    @inbounds for t in 1:seq, k in 1:n_used
        i = (t - 1) * n_used + k
        e = b.E[k, t]
        meta[woff + i] = reinterpret(Int32, b.W[k, t])
        if b.live[t] && rnd[e] >= 0
            gi = pfirst[e] + fillc[e]
            fillc[e] += 1
            meta[gpoff + i] = Int32(gi)
            # the launch holding pair gi starts at pair a and has n pairs
            r = rnd[e]
            a0 = rstart[r + 1]
            a = a0 + ((gi - a0) ÷ cfg.seg) * cfg.seg
            n = min(cfg.seg, rstart[r + 2] - a)
            m0 = 3(a - 1)
            l = gi - a + 1
            meta[m0 + l] = Int32((slot[e] - 1) * hot.stride)
            meta[m0 + n + l] = Int32(l)
            meta[m0 + 2n + l] = Int32(t)
        else
            meta[gpoff + i] = 0
        end
    end
    P.ngpu = ngpu
    P.gpoff = gpoff
    P.woff = woff
    unsafe_copyto!(pointer(P.meta_d), pointer(meta), woff + seq * n_used; async=true, stream=s)
    xt = unsafe_wrap(CuArray, pointer(P.xt), (h, seq))
    _transpose_into!(xt, xd)

    _hotp_compute!(backend, hot, P, xt, gate, up, down, rstart[1], rstart[2] - 1, cfg.seg, h, ff)
    CUDA.record(_hotp_event(P, 1), s)
    if R > 0
        cs = _hot_copy_stream()
        CUDA.wait(hot.used, cs)
        st = hot.stride
        GC.@preserve gate up down for r in 1:R
            j0 = (r - 1) * C + 1
            j1 = min(r * C, nup)
            ml = -1
            for j in j0:j1
                ml = max(ml, lastr[slot[cand[j]]])
            end
            ml >= 0 && CUDA.wait(_hotp_event(P, 2ml + 1), cs)
            for j in j0:j1
                e = cand[j]
                dst = hot.base + (slot[e] - 1) * st
                o = 0
                for sl in (gate, up, down)
                    nb = sl.bytes_per_expert
                    src = pointer(sl.data) + Gesso.Inference._quant_expert_off0(sl, e)
                    unsafe_copyto!(dst + o, src, nb; async=true, stream=cs)
                    o += nb
                end
                lastr[slot[e]] = r
            end
            CUDA.record(_hotp_event(P, 2r + 2), cs)
            CUDA.wait(_hotp_event(P, 2r + 2), s)
            _hotp_compute!(backend, hot, P, xt, gate, up, down, rstart[r + 1], rstart[r + 2] - 1,
                cfg.seg, h, ff)
            CUDA.record(_hotp_event(P, 2r + 1), s)
        end
    end
    CUDA.record(hot.used, s)

    @inbounds for j in 1:nup
        e = cand[j]
        sl = slot[e]
        old = Int(hot.expert_in[sl])
        old > 0 && (hot.slot_of[old] = 0)
        hot.expert_in[sl] = e
        hot.slot_of[e] = sl
    end
    empty!(hot.free)
    @inbounds for sl in hot.nslots:-1:1
        hot.expert_in[sl] == 0 && push!(hot.free, sl)
    end
    hot.hits += rstart[2] - 1
    hot.picks += npairs
    hot.swaps += nup
    return
end

# y[t, :] += the GPU pairs of token t, weighted; Y's H2D has landed y's CPU part.
function _hot_prefill_add_kernel!(y, D, meta, gpoff::Int, woff::Int, n_used::Int, h::Int)
    i = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    t = Int(blockIdx().y)
    i <= h || return
    base = (t - 1) * n_used
    acc = 0.0f0
    @inbounds for k in 1:n_used
        g = Int(meta[gpoff + base + k])
        g > 0 && (acc += reinterpret(Float32, meta[woff + base + k]) * D[i, g])
    end
    @inbounds y[t, i] += acc
    return
end

function _hot_prefill_add!(y, seq::Int, h::Int, n_used::Int)
    P = _HOT_PREFILL
    P.ngpu > 0 || return y
    D = unsafe_wrap(CuArray, pointer(P.d), (h, P.ngpu))
    @fastcuda threads = 256 blocks = (cld(h, 256), seq) _hot_prefill_add_kernel!(
        y, D, P.meta_d, P.gpoff, P.woff, n_used, h)
    return y
end

function _hot_verify!(y, b, gate, up, down, n_used::Int)
    _cuda_stream_wait()
    got = Array(y)
    ref = similar(b.yrow)
    Gesso.Inference._moe_host_experts_auto!(ref, b.xrow, b.experts, b.wbuf, n_used, gate, up, down)
    num = 0.0
    den = 0.0
    @inbounds for i in eachindex(ref)
        num += (Float64(got[i]) - ref[i])^2
        den += Float64(ref[i])^2
    end
    push!(_HOT_VERIFY_ERR, sqrt(num / max(den, eps())))
    return
end

# _HOT_VERIFY for a hot-layer prefill: the whole layer's relative L2 error
# against every pair on the CPU.
function _hot_prefill_verify!(y, b, X, gate, up, down, n_exp::Int)
    _cuda_stream_wait()
    got = Array(y)
    ref = similar(got)
    Gesso.Inference._moe_host_prefill!(ref, X, b.E, b.W, b.live, n_exp, gate, up, down)
    num = 0.0
    den = 0.0
    @inbounds for i in eachindex(ref)
        num += (Float64(got[i]) - ref[i])^2
        den += Float64(ref[i])^2
    end
    push!(_HOT_VERIFY_ERR, sqrt(num / max(den, eps())))
    return
end

"""
    _hot_stats(; reset=false)

Per-layer hit counts of the hot expert pools since the last reset:
`(layer, slots, hits, picks, swaps)` rows.
"""
function _hot_stats(; reset::Bool=false)
    rows = [(h.layer, h.nslots, h.hits, h.picks, h.swaps) for h in values(_HOT)]
    sort!(rows)
    if reset
        for h in values(_HOT)
            h.hits = h.picks = h.swaps = 0
        end
    end
    return rows
end
