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
        h, ff, n_used,
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

# Q8_0 needs a Q8_0 activation, so it stays on the per-expert path.
_moe_host_q8k_type(t::Int32) = _quant_q8k_native(t) && t != GGML_TYPE_Q8_0

function _moe_host_batchable(gate_s::GgufQuantSlice, up_s::GgufQuantSlice,
    down_s::GgufQuantSlice, h::Int, ff::Int)
    return _moe_host_q8k_type(gate_s.ggml_type) &&
           _moe_host_q8k_type(up_s.ggml_type) &&
           _moe_host_q8k_type(down_s.ggml_type) &&
           h % QK_K == 0 && ff % QK_K == 0 &&
           gate_s.ne0 == h && gate_s.ne1 == ff &&
           up_s.ne0 == h && up_s.ne1 == ff &&
           down_s.ne0 == ff && down_s.ne1 == h
end

_moe_host_nchunks(total::Int) = clamp(min(Threads.nthreads(), cld(total, _QGEMV_MIN_ROWS)), 1, total)

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
        return _moe_host_experts_batched!(yrow, xf, order, weights, n_used, gate_s, up_s, down_s, scr)
    end
    return _moe_host_experts!(yrow, xf, order, weights, n_used, gate_s, up_s, down_s,
        scr.gacc, scr.uacc, scr.hact, scr.ybuf)
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
    scr.n_used >= n_used || error("moe host: scratch built for $(scr.n_used) experts, got $n_used")
    simd = q8k_simd_enabled()

    quantize_q8_K!(scr.xq, xf)

    gu_total = 2 * n_used * ff
    gu_nc = _moe_host_nchunks(gu_total)
    _run_row_chunks!(
        _MoeGateUpChunk(gate_s, up_s, order, scr.gu, scr.xq, gu_total, gu_nc, ff, simd),
        gu_nc, gu_total * length(scr.xq),
    )

    _run_row_chunks!(_MoeSiluQuantChunk(scr.gu, scr.hacts, scr.hq, ff), n_used, n_used * ff)

    dn_total = n_used * h
    dn_nc = _moe_host_nchunks(dn_total)
    _run_row_chunks!(
        _MoeDownChunk(down_s, order, scr.hq, scr.dn, h, dn_total, dn_nc, simd),
        dn_nc, dn_total * ff,
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
        e = g.order[s ÷ 2 + 1]
        t = sl.ggml_type
        rb = _qrow_bytes(t, sl.ne0)
        off0 = _quant_expert_off0(sl, e)
        col = s + 1
        @inbounds for jj in j:jend
            g.gu[jj, col] = 1.0f0 * _dot_q8k(t, sl.data, off0 + (jj - 1) * rb, g.xq, g.simd) + 0.0f0
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
        g.hacts[j, k] = _cpu_silu(g.gu[j, 2k - 1]) * g.gu[j, 2k]
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
            g.dn[jj, k] = 1.0f0 * _dot_q8k(t, g.down.data, off0 + (jj - 1) * rb, xq, g.simd) + 0.0f0
        end
        r += jend - j + 1
    end
    return nothing
end
