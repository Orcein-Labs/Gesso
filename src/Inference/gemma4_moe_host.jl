# Gemma 4 decode experts over host-resident quantized slabs, batched across the
# top-k. Each token runs three pool dispatches (fused gate_up rows for every
# selected expert, GELU + activation quantization per expert, down rows for
# every selected expert) and quantizes its input once. The per-expert loop
# took two dispatches and one quantization per expert. Every row is the same
# `_dot_q8k` call the per-expert QuantMatrix path makes, so the output is
# bit-identical to it.

const _GEMMA4_BATCHED_EXPERTS = Ref(true)

mutable struct _G4MoeScratch
    xf::Vector{Float32}
    xk::Vector{Q8KVec}
    x0::Vector{Q80Vec}
    ak::Vector{Q8KVec}
    a0::Vector{Q80Vec}
    gu::Matrix{Float32}
    act::Matrix{Float32}
    dn::Matrix{Float32}
end

function _G4MoeScratch(E::Int, ff::Int, n_used::Int, gu_t::Int32, dn_t::Int32)
    actvecs(t, n, m) = _q80_act_type(t) ? (Q8KVec[], [Q80Vec(div(n, 32)) for _ in 1:m]) :
                                          ([Q8KVec(div(n, QK_K)) for _ in 1:m], Q80Vec[])
    xk, x0 = actvecs(gu_t, E, 1)
    ak, a0 = actvecs(dn_t, ff, n_used)
    return _G4MoeScratch(
        Vector{Float32}(undef, E), xk, x0, ak, a0,
        Matrix{Float32}(undef, 2ff, n_used),
        Matrix{Float32}(undef, ff, n_used),
        Matrix{Float32}(undef, E, n_used),
    )
end

function _g4_moe_scratch_fits(s::_G4MoeScratch, E::Int, ff::Int, n_used::Int, gu_t::Int32, dn_t::Int32)
    size(s.dn) == (E, n_used) && size(s.act) == (ff, n_used) || return false
    x_ok = _q80_act_type(gu_t) ? length(s.x0) == 1 : length(s.xk) == 1
    a_ok = _q80_act_type(dn_t) ? length(s.a0) == n_used : length(s.ak) == n_used
    return x_ok && a_ok
end

function _g4_moe_scratch!(scratch, E::Int, ff::Int, n_used::Int, gu_t::Int32, dn_t::Int32)
    s = scratch.experts
    if !(s isa _G4MoeScratch && _g4_moe_scratch_fits(s, E, ff, n_used, gu_t, dn_t))
        s = _G4MoeScratch(E, ff, n_used, gu_t, dn_t)
        scratch.experts = s
    end
    return s::_G4MoeScratch
end

# True when both expert banks are quantized slabs whose rows `_dot_q8k` reads
# directly and whose shapes are Gemma's fused [gate; up] and down layout.
function _g4_moe_batched_ok(gu, dn, E::Int)
    _GEMMA4_BATCHED_EXPERTS[] || return false
    gu isa GgufQuantSlice && dn isa GgufQuantSlice || return false
    for s in (gu, dn)
        t = s.ggml_type
        _quant_q8k_native(t) || return false
        s.ne0 % _q8k_block_values(t) == 0 || return false
        _qrow_bytes(t, s.ne0) * s.ne1 == s.bytes_per_expert || return false
    end
    return gu.ne0 == E && dn.ne1 == E && gu.ne1 == 2 * dn.ne0
end

# Rows r0:r1 of the selected experts' stacked outputs. Global row r is row j
# of slot k. Slot k reads xs[k], or xs[1] when the input is shared.
struct _G4ExpertRows{V}
    slab::GgufQuantSlice
    order::Vector{Int}
    xs::Vector{V}
    shared::Bool
    out::Matrix{Float32}
    rows::Int
    total::Int
    nc::Int
    simd::Bool
end

function (g::_G4ExpertRows)(c::Int)
    r0 = div((c - 1) * g.total, g.nc) + 1
    r1 = div(c * g.total, g.nc)
    t = g.slab.ggml_type
    rb = _qrow_bytes(t, g.slab.ne0)
    r = r0
    while r <= r1
        k = (r - 1) ÷ g.rows + 1
        j = (r - 1) % g.rows + 1
        jend = min(g.rows, j + (r1 - r))
        off0 = _quant_expert_off0(g.slab, g.order[k])
        xq = g.xs[g.shared ? 1 : k]
        @inbounds for jj in j:jend
            g.out[jj, k] = 1.0f0 * _dot_q8k(t, g.slab.data, off0 + (jj - 1) * rb, xq, g.simd) + 0.0f0
        end
        r += jend - j + 1
    end
    return nothing
end

function _g4_expert_rows!(slab::GgufQuantSlice, order::Vector{Int}, xs::Vector, shared::Bool,
                          out::Matrix{Float32}, n_used::Int, simd::Bool)
    rows = slab.ne1
    total = n_used * rows
    nc = _qgemv_nchunks(total)
    _run_row_chunks!(_G4ExpertRows(slab, order, xs, shared, out, rows, total, nc, simd), nc, total)
    return out
end

# Chunk k: GELU(gate) * up for slot k, then quantize it for the down pass.
struct _G4GeluQuant{V}
    gu::Matrix{Float32}
    act::Matrix{Float32}
    aq::Vector{V}
    ff::Int
end

_g4_quantize_act!(y::Q80Vec, x) = quantize_q8_0!(y, x)
_g4_quantize_act!(y::Q8KVec, x) = quantize_q8_K!(y, x)

function (g::_G4GeluQuant)(k::Int)
    @inbounds for i in 1:g.ff
        g.act[i, k] = _g4_gelu(g.gu[i, k]) * g.gu[g.ff + i, k]
    end
    _g4_quantize_act!(g.aq[k], view(g.act, :, k))
    return nothing
end

function _g4_moe_token_batched!(moe::Matrix{Float32}, t::Int, x2::Matrix{Float32}, order::Vector{Int},
                                probs::AbstractVector{Float32}, denom::Float32, down_scale,
                                n_used::Int, gu_s::GgufQuantSlice, dn_s::GgufQuantSlice,
                                s::_G4MoeScratch)
    E = size(x2, 2)
    ff = dn_s.ne0
    simd = q8k_simd_enabled()
    @inbounds for i in 1:E
        s.xf[i] = x2[t, i]
    end
    if _q80_act_type(gu_s.ggml_type)
        _g4_quantize_act!(s.x0[1], s.xf)
        _g4_expert_rows!(gu_s, order, s.x0, true, s.gu, n_used, simd)
    else
        _g4_quantize_act!(s.xk[1], s.xf)
        _g4_expert_rows!(gu_s, order, s.xk, true, s.gu, n_used, simd)
    end
    if _q80_act_type(dn_s.ggml_type)
        _run_row_chunks!(_G4GeluQuant(s.gu, s.act, s.a0, ff), n_used, n_used * ff)
        _g4_expert_rows!(dn_s, order, s.a0, false, s.dn, n_used, simd)
    else
        _run_row_chunks!(_G4GeluQuant(s.gu, s.act, s.ak, ff), n_used, n_used * ff)
        _g4_expert_rows!(dn_s, order, s.ak, false, s.dn, n_used, simd)
    end
    # Same order and rounding as the per-expert loop: scale, then add slot by slot.
    @inbounds for k in 1:n_used
        e = order[k]
        sc = (probs[e] / denom) * Float32(down_scale[e])
        for i in 1:E
            moe[t, i] += sc * s.dn[i, k]
        end
    end
    return moe
end
