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

# Decode needs about 50 MiB beyond the weights, and a 256-token prefill about
# 600 MiB at its peak. Prefills longer than the keep threshold release the
# pools, so the reserve only has to cover short prompts.
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
    CUDA.reclaim()
    return _expert_budget_from_free(Int64(CUDA.free_memory()), reserve)
end

struct _HotConfig
    decay::Float32     # per-token decay of the pick counts
    fill::Int          # copies a layer may start per token
    keep_seq::Int      # longest prefill that keeps the pools resident
    margin::Float32    # score lead a miss needs over the coldest resident
end

_hot_config() = _HotConfig(
    Float32(_hot_env_float("GESSO_GPU_HOT_DECAY", 0.99)),
    max(1, _hot_env_int("GESSO_GPU_HOT_FILL", 2)),
    _hot_env_int("GESSO_GPU_HOT_PREFILL_KEEP", 32),
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
end

const _HOT = IdDict{Any,_HotLayer}()
const _HOT_CFG = Ref(_HotConfig(0.99f0, 2, 32, 1.0f0))
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
            CuEvent(CUDA.EVENT_DISABLE_TIMING), Int[], UInt8[], 0, 0, 0)
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
# does not fit goes back to the CPU-only path.
function _hot_alloc!(hot::_HotLayer, g, u, n_used::Int)
    hot.pool === nothing || return true
    hot.nslots >= 1 || return false
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
        @debug "hot experts: pool does not fit, layer stays on the CPU" layer=hot.layer
        hot.nslots = 0
        return false
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
    _hot_commit!(hot)
    _hot_plan!(hot, hot.last, length(hot.last))
    _hot_issue!(hot, g, u, d)
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
