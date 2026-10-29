# Batched host MoE for one token over host-resident GgufQuantSlice banks.
#
# `_moe_host_experts!` (gguf_quants.jl) runs three threaded GEMVs per selected
# expert, and every chunk of each GEMV quantizes x again. Here x is quantized to
# Q8_K once per token, gate and up for all selected experts share one pool
# dispatch, and down for all experts shares a second. Each output row is the
# same `_dot_q8k` call the per-expert path makes, and the weighted sum runs in
# k order, so K-quant slabs agree with the per-expert path bit for bit.

"""
    MoeHostScratch(h, ff, n_used)

Buffers for one `(h, ff, n_used)` expert shape. Built once and reused across
tokens and layers, so the token loop allocates nothing. The gate/up columns are
interleaved per expert: column `2k-1` is gate of selected expert `k`, `2k` is up.
"""
mutable struct MoeHostScratch
    h::Int
    ff::Int
    n_used::Int
    xq::Q8KVec
    hq::Vector{Q8KVec}
    gu::Matrix{Float32}
    dn::Matrix{Float32}
    hacts::Matrix{Float32}
    hact::Vector{Float32}
    gacc::Vector{Float32}
    uacc::Vector{Float32}
    ybuf::Vector{Float32}
end

function MoeHostScratch(h::Int, ff::Int, n_used::Int)
    return MoeHostScratch(
        h,
        ff,
        n_used,
        Q8KVec(div(h, QK_K)),
        [Q8KVec(div(ff, QK_K)) for _ in 1:n_used],
        zeros(Float32, ff, 2 * n_used),
        zeros(Float32, h, n_used),
        zeros(Float32, ff, n_used),
        zeros(Float32, ff),
        zeros(Float32, ff),
        zeros(Float32, ff),
        zeros(Float32, h),
    )
end

# A single global buffer: every layer of one model shares its expert shape, so
# the cache does not thrash. A shape change rebuilds it. Callers may run fewer
# than n_used experts (the GPU hot-expert path sends only its misses), so a
# larger buffer serves a smaller count.
const _MOE_HOST_SCRATCH = Ref{MoeHostScratch}(MoeHostScratch(0, 0, 0))

function _moe_host_scratch(h::Int, ff::Int, n_used::Int)
    s = _MOE_HOST_SCRATCH[]
    if s.h != h || s.ff != ff || s.n_used < n_used
        s = MoeHostScratch(h, ff, n_used)
        _MOE_HOST_SCRATCH[] = s
    end
    return s
end

# The 32-value block types need a Q8_0 activation, so they stay on the per-expert path.
_moe_host_q8k_type(t::Int32) = _quant_q8k_native(t) && !_q80_act_type(t)

function _moe_host_batchable(
    gate_s::GgufQuantSlice,
    up_s::GgufQuantSlice,
    down_s::GgufQuantSlice,
    h::Int,
    ff::Int,
)
    return _moe_host_q8k_type(gate_s.ggml_type) &&
           _moe_host_q8k_type(up_s.ggml_type) &&
           _moe_host_q8k_type(down_s.ggml_type) &&
           h % QK_K == 0 &&
           ff % QK_K == 0 &&
           gate_s.ne0 == h &&
           gate_s.ne1 == ff &&
           up_s.ne0 == h &&
           up_s.ne1 == ff &&
           down_s.ne0 == ff &&
           down_s.ne1 == h
end

_moe_host_nchunks(total::Int) = _qgemv_nchunks(total)

"""
    _moe_host_experts_auto!(yrow, xf, order, weights, n_used, gate_s, up_s, down_s) -> yrow

Top-k SwiGLU for one token on the host. Takes the batched Q8_K path when every
slab has a Q8_K kernel, and the per-expert `_moe_host_experts!` otherwise.
"""
function _moe_host_experts_auto!(
    yrow::AbstractVector{Float32},
    xf::AbstractVector{Float32},
    order,
    weights,
    n_used::Int,
    gate_s::GgufQuantSlice,
    up_s::GgufQuantSlice,
    down_s::GgufQuantSlice,
)
    h = length(yrow)
    ff = gate_s.ne1
    scr = _moe_host_scratch(h, ff, n_used)
    if _moe_host_batchable(gate_s, up_s, down_s, h, ff)
        return _moe_host_experts_batched!(
            yrow,
            xf,
            order,
            weights,
            n_used,
            gate_s,
            up_s,
            down_s,
            scr,
        )
    end
    return _moe_host_experts!(
        yrow,
        xf,
        order,
        weights,
        n_used,
        gate_s,
        up_s,
        down_s,
        scr.gacc,
        scr.uacc,
        scr.hact,
        scr.ybuf,
    )
end

function _moe_host_experts_batched!(
    yrow::AbstractVector{Float32},
    xf::AbstractVector{Float32},
    order,
    weights,
    n_used::Int,
    gate_s::GgufQuantSlice,
    up_s::GgufQuantSlice,
    down_s::GgufQuantSlice,
    scr::MoeHostScratch,
)
    h = scr.h
    ff = scr.ff
    length(yrow) == h || error("moe host: yrow length $(length(yrow)) ≠ h $h")
    length(xf) == h || error("moe host: xf length $(length(xf)) ≠ h $h")
    scr.n_used >= n_used ||
        error("moe host: scratch built for $(scr.n_used) experts, got $n_used")
    simd = q8k_simd_enabled()

    quantize_q8_K!(scr.xq, xf)

    gu_total = 2 * n_used * ff
    gu_nc = _moe_host_nchunks(gu_total)
    _run_row_chunks!(
        _MoeGateUpChunk(gate_s, up_s, order, scr.gu, scr.xq, gu_total, gu_nc, ff, simd),
        gu_nc,
        gu_total * length(scr.xq),
    )

    _run_row_chunks!(_MoeSiluQuantChunk(scr.gu, scr.hacts, scr.hq, ff), n_used, n_used * ff)

    dn_total = n_used * h
    dn_nc = _moe_host_nchunks(dn_total)
    _run_row_chunks!(
        _MoeDownChunk(down_s, order, scr.hq, scr.dn, h, dn_total, dn_nc, simd),
        dn_nc,
        dn_total * ff,
    )

    fill!(yrow, 0.0f0)
    @inbounds for k in 1:n_used
        w = weights[k]
        for i in 1:h
            yrow[i] += w * scr.dn[i, k]
        end
    end
    return yrow
end

# One chunk of the gate/up pass. Global rows are numbered slot by slot: slot
# `s` (0-based) is gate of selected expert `s÷2+1` when `s` is even and its up
# when odd, and holds `ff` rows. A chunk's row range can span several slots.
struct _MoeGateUpChunk
    gate::GgufQuantSlice
    up::GgufQuantSlice
    order::Vector{Int}
    gu::Matrix{Float32}
    xq::Q8KVec
    total::Int
    nc::Int
    ff::Int
    simd::Bool
end

function (g::_MoeGateUpChunk)(c::Int)
    r0 = div((c - 1) * g.total, g.nc) + 1
    r1 = div(c * g.total, g.nc)
    r = r0
    while r <= r1
        s = (r - 1) ÷ g.ff
        j = (r - 1) % g.ff + 1
        jend = min(g.ff, j + (r1 - r))
        sl = iseven(s) ? g.gate : g.up
        e = g.order[s÷2+1]
        t = sl.ggml_type
        rb = _qrow_bytes(t, sl.ne0)
        off0 = _quant_expert_off0(sl, e)
        col = s + 1
        @inbounds for jj in j:jend
            g.gu[jj, col] =
                1.0f0 * _dot_q8k(t, sl.data, off0 + (jj - 1) * rb, g.xq, g.simd) + 0.0f0
        end
        r += jend - j + 1
    end
    return nothing
end

# SwiGLU activation and its Q8_K quantization for selected expert `k`. Base
# Float32 `exp` does not vectorize, so this costs about 5 us per expert serially;
# a faster exp would change the values the exact-ids gate checks, so the experts
# run in parallel instead. Each chunk owns column `k` of `hacts`.
struct _MoeSiluQuantChunk
    gu::Matrix{Float32}
    hacts::Matrix{Float32}
    hq::Vector{Q8KVec}
    ff::Int
end

function (g::_MoeSiluQuantChunk)(k::Int)
    @inbounds for j in 1:g.ff
        g.hacts[j, k] = _cpu_silu(g.gu[j, 2k-1]) * g.gu[j, 2k]
    end
    quantize_q8_K!(g.hq[k], view(g.hacts, :, k))
    return nothing
end

# One chunk of the down pass. Row `j` of selected expert `k` reads its own
# hidden vector `hq[k]`, so each expert keeps a separate Q8_K input.
struct _MoeDownChunk
    down::GgufQuantSlice
    order::Vector{Int}
    hq::Vector{Q8KVec}
    dn::Matrix{Float32}
    h::Int
    total::Int
    nc::Int
    simd::Bool
end

function (g::_MoeDownChunk)(c::Int)
    r0 = div((c - 1) * g.total, g.nc) + 1
    r1 = div(c * g.total, g.nc)
    r = r0
    while r <= r1
        k = (r - 1) ÷ g.h + 1
        j = (r - 1) % g.h + 1
        jend = min(g.h, j + (r1 - r))
        e = g.order[k]
        t = g.down.ggml_type
        rb = _qrow_bytes(t, g.down.ne0)
        off0 = _quant_expert_off0(g.down, e)
        xq = g.hq[k]
        @inbounds for jj in j:jend
            g.dn[jj, k] =
                1.0f0 * _dot_q8k(t, g.down.data, off0 + (jj - 1) * rb, xq, g.simd) + 0.0f0
        end
        r += jend - j + 1
    end
    return nothing
end

# Prefill: a batch of tokens through one layer's host experts.
#
# Token by token, every token streams its selected experts' slabs from DRAM, so
# a P-token prompt reads each active expert about P * n_used / n_exp times.
# Here the tokens are grouped by expert first, so each weight row is read once
# and dotted against every token that picked it while the row sits in L1. The
# row's tokens go through `_dot_q8k_multi!` in tiles, as dense prefill does, so
# the row's unpacking is shared across a tile; each column of that kernel is
# bit-equal to `_dot_q8k` on the same Q8_K activation (quant_cpu_multi.jl). Each
# token's weighted sum still runs in k order, so the result matches the token
# loop bit for bit. GESSO_PREFILL_MULTI=0 restores one `_dot_q8k` per pair.

"""
    MoePrefillScratch

Buffers for a batch of tokens. Pair `p = (t - 1) * n_used + k` is token `t`'s
selected slot `k`. Grow-only, and shared by every layer of one model. `X` and
`H` are the activation blocks (`Q8KVec` or `Q80Vec`) that the gate/up and down
weights dot against.
"""
mutable struct MoePrefillScratch{X, H}
    h::Int
    ff::Int
    xq::Vector{X}               # per token
    hq::Vector{H}               # per pair
    gu::Array{Float32, 3}        # ff x 2 x pairs (gate, up); gate becomes the activation
    dn::Matrix{Float32}         # h x pairs
    cnt::Vector{Int}            # pairs per expert
    start::Vector{Int}          # first index into `pairs` per expert
    pairs::Vector{Int}          # pair ids grouped by expert, token order within one
    active::Vector{Int}         # experts with at least one pair
    cost::Vector{Int}           # prefix dot count over `active` for the current pass
    xg::Vector{X}               # xq of each entry of `pairs`, so one expert's tokens are contiguous
    hg::Vector{H}               # hq of each entry of `pairs`
    cbuf::Vector{Vector{Float32}}  # per chunk, one row's dots over an expert's pairs
    cdec::Vector{Vector{UInt8}}    # per chunk, one decoded IQ2_XS row
    edt::Vector{Vector{Float32}}   # per expert, the d table of its pairs
    eqt::Vector{Vector{Int8}}      # per expert, the packed quants of its pairs
    est::Vector{Vector{Float32}}   # per expert, the s table of its pairs (Q8_0 activations)
end

MoePrefillScratch{X, H}() where {X, H} = MoePrefillScratch{X, H}(
    0,
    0,
    X[],
    H[],
    zeros(Float32, 0, 2, 0),
    zeros(Float32, 0, 0),
    Int[],
    Int[],
    Int[],
    Int[],
    Int[],
    X[],
    H[],
    Vector{Float32}[],
    Vector{UInt8}[],
    Vector{Float32}[],
    Vector{Int8}[],
    Vector{Float32}[],
)
MoePrefillScratch() = MoePrefillScratch{Q8KVec, Q8KVec}()

const _MOE_PREFILL_SCRATCH = Ref{MoePrefillScratch}(MoePrefillScratch())

_act_vec(::Type{Q8KVec}, n::Int) = Q8KVec(div(n, QK_K))
_act_vec(::Type{Q80Vec}, n::Int) = Q80Vec(div(n, 32))
_act_vec(::Type{Q6KPrepVec}, n::Int) = _q6k_prep(Q8KVec(div(n, QK_K)))

_quantize_act!(y::Q8KVec, x) = quantize_q8_K!(y, x)
_quantize_act!(y::Q80Vec, x) = quantize_q8_0!(y, x)
_quantize_act!(y::Q6KPrepVec, x) = (quantize_q8_K!(y.q, x); _q6k_prep!(y))

# The Q8_K activation vector for weights of types `ts`. The K-quant tiles read
# pair-ordered columns, so prepping each token once spares every row of every
# expert the reorder; a mix of types keeps plain Q8_K.
_moe_act_type(simd::Bool, ts::Int32...) =
    simd && _prefill_multi_enabled() && all(_q6k_prep_type, ts) ? Q6KPrepVec : Q8KVec

function _moe_prefill_scratch(
    h::Int,
    ff::Int,
    seq::Int,
    n_used::Int,
    n_exp::Int,
    ::Type{X}=Q8KVec,
    ::Type{H}=Q8KVec,
) where {X, H}
    s = _MOE_PREFILL_SCRATCH[]
    np = seq * n_used
    same = s isa MoePrefillScratch{X, H} && s.h == h && s.ff == ff
    if !same || length(s.hq) < np || length(s.xq) < seq
        ntok = same ? max(seq, length(s.xq)) : seq
        cap = same ? max(np, length(s.hq)) : np
        s = MoePrefillScratch{X, H}(
            h,
            ff,
            [_act_vec(X, h) for _ in 1:ntok],
            [_act_vec(H, ff) for _ in 1:cap],
            zeros(Float32, ff, 2, cap),
            zeros(Float32, h, cap),
            Int[],
            Int[],
            Vector{Int}(undef, cap),
            Int[],
            Int[],
            Vector{X}(undef, cap),
            Vector{H}(undef, cap),
            Vector{Float32}[],
            Vector{UInt8}[],
            Vector{Float32}[],
            Vector{Int8}[],
            Vector{Float32}[],
        )
        _MOE_PREFILL_SCRATCH[] = s
    end
    resize!(s.cnt, n_exp)
    resize!(s.start, n_exp + 1)
    return s::MoePrefillScratch{X, H}
end

# `cpu` empty means every expert; otherwise only experts with cpu[e] take pairs.
@inline _moe_on_cpu(cpu::Vector{Bool}, e::Int) = isempty(cpu) || @inbounds cpu[e]

function _moe_prefill_group!(
    s::MoePrefillScratch,
    experts::AbstractMatrix{Int},
    live::AbstractVector{Bool},
    seq::Int,
    n_used::Int,
    n_exp::Int,
    cpu::Vector{Bool}=Bool[],
)
    fill!(s.cnt, 0)
    @inbounds for t in 1:seq
        live[t] || continue
        for k in 1:n_used
            e = experts[k, t]
            _moe_on_cpu(cpu, e) && (s.cnt[e] += 1)
        end
    end
    s.start[1] = 1
    @inbounds for e in 1:n_exp
        s.start[e+1] = s.start[e] + s.cnt[e]
    end
    fill!(s.cnt, 0)
    @inbounds for t in 1:seq
        live[t] || continue
        for k in 1:n_used
            e = experts[k, t]
            _moe_on_cpu(cpu, e) || continue
            s.pairs[s.start[e]+s.cnt[e]] = (t - 1) * n_used + k
            s.cnt[e] += 1
        end
    end
    empty!(s.active)
    @inbounds for e in 1:n_exp
        s.cnt[e] > 0 && push!(s.active, e)
    end
    # hq's vectors exist before Silu fills them, so hg can point at them now.
    @inbounds for q in 1:(s.start[n_exp+1]-1)
        p = s.pairs[q]
        s.xg[q] = s.xq[(p-1)÷n_used+1]
        s.hg[q] = s.hq[p]
    end
    return s
end

# Chunk c's row buffer. Two chunks can split one expert's rows, so a buffer
# shared by expert would race.
function _moe_prefill_bufs!(s::MoePrefillScratch, nc::Int, n::Int)
    while length(s.cbuf) < nc
        push!(s.cbuf, Float32[])
        push!(s.cdec, UInt8[])
    end
    for c in 1:nc
        length(s.cbuf[c]) < n && resize!(s.cbuf[c], n)
    end
    return s
end

# Gate/up and down chunks per thread. An expert's rows cost its pair count,
# which varies several-fold, so one chunk per thread left threads idle at the
# end of each pass; four and eight measured the same.
const _MOE_CHUNKS_PER_THREAD = 4

# Raw dots of the weight row at `woff` with xv[1:n], tiled, into buf[1:n]. An
# IQ2_XS row with a packed table and more than one tile decodes once into u.
@inline function _moe_prefill_row_multi!(
    buf::Vector{Float32},
    t::Int32,
    w,
    woff::Int,
    xv,
    n::Int,
    simd::Bool,
    dt=nothing,
    u::Union{Nothing, Vector{UInt8}}=nothing,
)
    u !== nothing && dt isa _MultiPacked && t == GGML_TYPE_IQ2_XS && n > _MULTI_TILE &&
        return _dot_iq2xs_decoded_row!(buf, u, w, woff, xv, n, dt)
    i0 = 1
    while i0 <= n
        K = min(_MULTI_TILE, n - i0 + 1)
        _dot_q8k_multi!(buf, t, w, woff, xv, i0, K, simd, dt)
        i0 += K
    end
    return buf
end

# The types whose tiles take a packed table, built once per expert piece: the
# f16-d block types and IQ2_XS. The K-quant tiles read prepped columns instead,
# which the scratch holds when `_moe_act_type` picks Q6KPrepVec.
_moe_prefill_dt(t::Int32, simd::Bool) = simd && _multi_pack_type(t)

# Expert e's packed table, as `_MoePrefillPack` last built it.
_moe_prefill_table(s::MoePrefillScratch, e::Int) = _MultiPacked(s.edt[e], s.eqt[e], s.est[e])

# Packs each active expert's columns of `cols` (xg or hg) once per pass, so the
# chunks that split an expert's rows share one table. Chunk c packs the experts
# whose first pair falls in its share of the pairs.
struct _MoePrefillPack{S <: MoePrefillScratch, V <: AbstractVector}
    s::S
    cols::V
    nc::Int
end

function (g::_MoePrefillPack)(c::Int)
    s = g.s
    np = s.cost[end]
    w0 = div((c - 1) * np, g.nc)
    w1 = div(c * np, g.nc)
    @inbounds for a in eachindex(s.active)
        w0 <= s.cost[a] < w1 || continue
        e = s.active[a]
        p0 = s.start[e]
        _multi_packed!(s.edt[e], s.eqt[e], view(g.cols, p0:(p0+s.cnt[e]-1)), s.est[e])
    end
    return nothing
end

function _moe_prefill_pack!(s::MoePrefillScratch, cols::AbstractVector, n_exp::Int)
    while length(s.edt) < n_exp
        push!(s.edt, Float32[])
        push!(s.eqt, Int8[])
        push!(s.est, Float32[])
    end
    np = _moe_prefill_cost!(s, 1)
    nc = clamp(_MOE_CHUNKS_PER_THREAD * Threads.nthreads(), 1, length(s.active))
    _run_row_chunks!(_MoePrefillPack(s, cols, nc), nc, np * length(cols[1].qs))
    return s
end

function _moe_prefill_cost!(s::MoePrefillScratch, rows::Int)
    na = length(s.active)
    resize!(s.cost, na + 1)
    s.cost[1] = 0
    @inbounds for a in 1:na
        s.cost[a+1] = s.cost[a] + rows * s.cnt[s.active[a]]
    end
    return s.cost[na+1]
end

# Chunks split the pass into equal dot counts, since a row of expert e costs
# cnt[e] dots and the counts vary several-fold. Dots are numbered from 0;
# chunk c holds [w0, w1), row r of the a-th active expert holds dots from
# cost[a] + (r - 1) * cnt, and a row belongs to the chunk holding its first dot.
# `f(e, r0, r1)` runs rows r0:r1 of expert e.
@inline function _moe_prefill_pieces(f, s::MoePrefillScratch, rows::Int, nc::Int, c::Int)
    tot = s.cost[end]
    w0 = div((c - 1) * tot, nc)
    w1 = div(c * tot, nc)
    na = length(s.active)
    a = max(1, searchsortedlast(s.cost, w0))
    @inbounds while a <= na && s.cost[a] < w1
        e = s.active[a]
        n = s.cnt[e]
        base = s.cost[a]
        r0 = max(1, cld(w0 - base, n) + 1)
        r1 = min(rows, cld(w1 - base, n))
        r0 <= r1 && f(e, r0, r1)
        a += 1
    end
    return nothing
end

struct _MoePrefillQuantX{M <: AbstractMatrix{Float32}, V}
    X::M                         # seq x h, the pinned staging layout
    xrow::Vector{Vector{Float32}}
    xq::Vector{V}
    live::Vector{Bool}
    seq::Int
    nc::Int
end

function (g::_MoePrefillQuantX)(c::Int)
    t0 = div((c - 1) * g.seq, g.nc) + 1
    t1 = div(c * g.seq, g.nc)
    xr = g.xrow[c]
    h = length(xr)
    @inbounds for t in t0:t1
        g.live[t] || continue
        for i in 1:h
            xr[i] = g.X[t, i]
        end
        _quantize_act!(g.xq[t], xr)
    end
    return nothing
end

# `up_row0` is where the up rows start inside `up`'s expert slab: 0 for a
# separate up tensor, ff when gate and up share one fused [gate; up] slab.
struct _MoePrefillGateUp{S <: MoePrefillScratch}
    s::S
    gate::GgufQuantSlice
    up::GgufQuantSlice
    up_row0::Int
    nc::Int
    simd::Bool
    multi::Bool
    packed::Bool
end

function (g::_MoePrefillGateUp)(c::Int)
    s = g.s
    ff = s.ff
    buf = g.multi ? s.cbuf[c] : Float32[]
    u = g.multi ? s.cdec[c] : nothing
    _moe_prefill_pieces(s, 2ff, g.nc, c) do e, r0, r1
        p0 = s.start[e]
        n = s.cnt[e]
        p1 = p0 + n - 1
        xv = view(s.xg, p0:p1)
        dt = g.packed ? _moe_prefill_table(s, e) : nothing
        for (sl, col, lo, hi, row0) in (
            (g.gate, 1, r0, min(r1, ff), 0),
            (g.up, 2, max(r0, ff + 1) - ff, r1 - ff, g.up_row0),
        )
            lo <= hi || continue
            t = sl.ggml_type
            rb = _qrow_bytes(t, sl.ne0)
            off0 = _quant_expert_off0(sl, e)
            @inbounds for jj in lo:hi
                woff = off0 + (row0 + jj - 1) * rb
                if g.multi
                    _moe_prefill_row_multi!(buf, t, sl.data, woff, xv, n, g.simd, dt, u)
                    for i in 1:n
                        # Same expression as the per-pair path, so -0.0 becomes +0.0 identically.
                        s.gu[jj, col, s.pairs[p0+i-1]] = 1.0f0 * buf[i] + 0.0f0
                    end
                else
                    for q in p0:p1
                        s.gu[jj, col, s.pairs[q]] =
                            1.0f0 * _dot_q8k(t, sl.data, woff, s.xg[q], g.simd) + 0.0f0
                    end
                end
            end
        end
    end
    return nothing
end

# act(gate) * up for each pair, then the down pass's activation quantization.
struct _MoePrefillGlu{S <: MoePrefillScratch, F}
    s::S
    act::F
    np::Int
    nc::Int
end

function (g::_MoePrefillGlu)(c::Int)
    s = g.s
    q0 = div((c - 1) * g.np, g.nc) + 1
    q1 = div(c * g.np, g.nc)
    @inbounds for q in q0:q1
        p = s.pairs[q]
        for j in 1:s.ff
            s.gu[j, 1, p] = g.act(s.gu[j, 1, p]) * s.gu[j, 2, p]
        end
        _quantize_act!(s.hq[p], view(s.gu, :, 1, p))
    end
    return nothing
end

struct _MoePrefillDown{S <: MoePrefillScratch}
    s::S
    down::GgufQuantSlice
    nc::Int
    simd::Bool
    multi::Bool
    packed::Bool
end

function (g::_MoePrefillDown)(c::Int)
    s = g.s
    sl = g.down
    t = sl.ggml_type
    rb = _qrow_bytes(t, sl.ne0)
    buf = g.multi ? s.cbuf[c] : Float32[]
    _moe_prefill_pieces(s, s.h, g.nc, c) do e, r0, r1
        p0 = s.start[e]
        n = s.cnt[e]
        p1 = p0 + n - 1
        hv = view(s.hg, p0:p1)
        dt = g.packed ? _moe_prefill_table(s, e) : nothing
        off0 = _quant_expert_off0(sl, e)
        @inbounds for jj in r0:r1
            woff = off0 + (jj - 1) * rb
            if g.multi
                _moe_prefill_row_multi!(buf, t, sl.data, woff, hv, n, g.simd, dt)
                for i in 1:n
                    s.dn[jj, s.pairs[p0+i-1]] = 1.0f0 * buf[i] + 0.0f0
                end
            else
                for q in p0:p1
                    s.dn[jj, s.pairs[q]] =
                        1.0f0 * _dot_q8k(t, sl.data, woff, s.hg[q], g.simd) + 0.0f0
                end
            end
        end
    end
    return nothing
end

struct _MoePrefillCombine{S <: MoePrefillScratch, M <: AbstractMatrix{Float32}, IE, FW}
    s::S
    Y::M                         # seq x h
    experts::IE
    weights::FW                  # n_used x seq
    cpu::Vector{Bool}
    live::Vector{Bool}
    seq::Int
    n_used::Int
    nc::Int
end

function (g::_MoePrefillCombine)(c::Int)
    s = g.s
    t0 = div((c - 1) * g.seq, g.nc) + 1
    t1 = div(c * g.seq, g.nc)
    @inbounds for t in t0:t1
        for i in 1:s.h
            g.Y[t, i] = 0.0f0
        end
        g.live[t] || continue
        for k in 1:g.n_used
            _moe_on_cpu(g.cpu, g.experts[k, t]) || continue
            w = g.weights[k, t]
            p = (t - 1) * g.n_used + k
            for i in 1:s.h
                g.Y[t, i] += w * s.dn[i, p]
            end
        end
    end
    return nothing
end

"""
    _moe_host_prefill!(Y, X, experts, weights, live, gate_s, up_s, down_s; cpu=Bool[]) -> Y

Top-k SwiGLU for `seq` tokens on the host. `X` and `Y` are `seq x h`;
`experts` and `weights` are `n_used x seq` as `_moe_route_row!` fills them;
tokens with `live[t] == false` get a zero row. Callers check
`_moe_host_batchable` first. A non-empty `cpu` (length `n_exp`) limits the sum
to the experts it marks, so `Y` holds the host's partial sum and the GPU adds
the rest.
"""
function _moe_host_prefill!(
    Y::Matrix{Float32},
    X::Matrix{Float32},
    experts::Matrix{Int},
    weights::Matrix{Float32},
    live::Vector{Bool},
    n_exp::Int,
    gate_s::GgufQuantSlice,
    up_s::GgufQuantSlice,
    down_s::GgufQuantSlice;
    cpu::Vector{Bool}=Bool[],
)
    simd = q8k_simd_enabled()
    XT = _moe_act_type(simd, gate_s.ggml_type, up_s.ggml_type)
    HT = _moe_act_type(simd, down_s.ggml_type)
    s = _moe_prefill_scratch(size(X, 2), gate_s.ne1, size(X, 1), size(experts, 1), n_exp, XT, HT)
    return _moe_prefill_run!(s, Y, X, experts, weights, live, n_exp, gate_s, up_s, 0, down_s, _cpu_silu, cpu)
end

# The grouped pass over one batch of tokens, with `s` already sized for it:
# quantize X, group pairs by expert, gate/up rows, act(gate) * up, down rows,
# then each token's weighted sum in k order.
function _moe_prefill_run!(
    s::MoePrefillScratch,
    Y::AbstractMatrix{Float32},
    X::AbstractMatrix{Float32},
    experts::AbstractMatrix{Int},
    weights::AbstractMatrix{Float32},
    live::Vector{Bool},
    n_exp::Int,
    gate_s::GgufQuantSlice,
    up_s::GgufQuantSlice,
    up_row0::Int,
    down_s::GgufQuantSlice,
    act,
    cpu::Vector{Bool},
)
    seq, h = size(X)
    n_used = size(experts, 1)
    ff = s.ff
    simd = q8k_simd_enabled()
    nt = Threads.nthreads()

    tnc = clamp(nt, 1, seq)
    xrow = [Vector{Float32}(undef, h) for _ in 1:tnc]
    _run_row_chunks!(_MoePrefillQuantX(X, xrow, s.xq, live, seq, tnc), tnc, seq * h)

    _moe_prefill_group!(s, experts, live, seq, n_used, n_exp, cpu)
    np = s.start[n_exp+1] - 1
    if np > 0
        multi = _prefill_multi_enabled()
        cnc = _MOE_CHUNKS_PER_THREAD * nt
        multi && _moe_prefill_bufs!(s, cnc, maximum(s.cnt))
        packed =
            multi && _moe_prefill_dt(gate_s.ggml_type, simd) &&
            _moe_prefill_dt(up_s.ggml_type, simd)
        packed && _moe_prefill_pack!(s, s.xg, n_exp)
        tot = _moe_prefill_cost!(s, 2ff)
        nc = clamp(cnc, 1, tot)
        _run_row_chunks!(
            _MoePrefillGateUp(s, gate_s, up_s, up_row0, nc, simd, multi, packed), nc, tot * h)
        snc = clamp(nt, 1, np)
        _run_row_chunks!(_MoePrefillGlu(s, act, np, snc), snc, np * ff)
        packed = multi && _moe_prefill_dt(down_s.ggml_type, simd)
        packed && _moe_prefill_pack!(s, s.hg, n_exp)
        tot = _moe_prefill_cost!(s, h)
        nc = clamp(cnc, 1, tot)
        _run_row_chunks!(_MoePrefillDown(s, down_s, nc, simd, multi, packed), nc, tot * ff)
    end
    _run_row_chunks!(
        _MoePrefillCombine(s, Y, experts, weights, cpu, live, seq, n_used, tnc),
        tnc,
        seq * h,
    )
    return Y
end
