# Multi-column Q8_K dot products: one weight row against K activation vectors
# in one pass (prefill and any seq > 1 projection).
#
# The single-column kernels spend most of their time unpacking a sub-block of
# weights (grid gathers, sign masks, scales). Here that unpacking runs once per
# sub-block and each column then feeds the same unpacked operands through its
# own accumulator and activation pointer. Column c therefore executes the same
# integer instructions with the same operands as `_dot_q8k` on its vector, so
# the result is bit-identical (saturating vpmaddubsw is operand-sensitive, which
# is why the activation stays the signed operand in every column).
#
# Tiles hold at most _MULTI_TILE columns; a remainder of 1..3 columns runs as a
# smaller tile, so any token count works.

const _MULTI_TILE = 4

# GESSO_PREFILL_MULTI=0 selects the per-token path. Read once per process,
# lazily: a module-level const would freeze the value into the precompile cache.
const _PREFILL_MULTI = Ref(Int8(-1))

function _prefill_multi_enabled()::Bool
    s = _PREFILL_MULTI[]
    if s < 0
        s = get(ENV, "GESSO_PREFILL_MULTI", "1") == "0" ? Int8(0) : Int8(1)
        _PREFILL_MULTI[] = s
    end
    return s == 1
end

# ---------------------------------------------------------------- helpers

@inline _multi_k(::Val{K}) where {K} = K

# Per-column block scale: the same product the single kernel forms, dw * x.d.
@inline _multi_d(dw::Float32, dv::NTuple{K,Vector{Float32}}, ib::Int) where {K} =
    ntuple(c -> dw * dv[c][ib + 1], Val(K))

# sumf[c] += d[c] * Float32(hsum(acc[c])), in block order as the single kernel.
@inline _multi_fadd(sumf::NTuple{K,Float32}, d::NTuple{K,Float32},
    acc::NTuple{K,NTuple{8,VecElement{Int32}}}) where {K} =
    ntuple(c -> sumf[c] + d[c] * Float32(_iq_hsum8(acc[c])), Val(K))

@inline _multi_scale(sumf::NTuple{K,Float32}, s::Float32) where {K} =
    ntuple(c -> s * sumf[c], Val(K))

@inline _multi_iq2_step(acc::NTuple{K,NTuple{8,VecElement{Int32}}}, g, s,
    pq::NTuple{K,Ptr{Int8}}, off::Int, ls1::Int32, ls2::Int32) where {K} =
    ntuple(c -> _iq2_sub_acc(acc[c], g, s, pq[c] + off, ls1, ls2), Val(K))

@inline _multi_iq3_step(acc::NTuple{K,NTuple{8,VecElement{Int32}}}, g, s,
    pq::NTuple{K,Ptr{Int8}}, off::Int, ls::Int32) where {K} =
    ntuple(c -> _iq3_sub_acc(acc[c], g, s, pq[c] + off, ls), Val(K))

# Column count of a tile after checking that every column has the row's block
# count; a mismatch would read past the end of a shorter activation.
function _multi_check(xqs, t0::Int, K::Int, nb::Int)
    for c in 1:K
        length(xqs[t0 + c - 1].d) == nb ||
            error("q8k multi: activation $(t0 + c - 1) has $(length(xqs[t0 + c - 1].d)) blocks, row has $nb")
    end
    return nothing
end

# ---------------------------------------------------------------- IQ2_XXS

# Grid and sign operands and scale of sub-block ib32 (0..7) of the IQ2_XXS
# block at pointer b. Same unpacking as _dot_iq2xxs_q8k_avx2.
@inline function _iq2xxs_sub_ops(pg, ps, b::Ptr{UInt8}, ib32::Int)
    qw = ltoh(unsafe_load(Ptr{UInt64}(b + 2 + 8 * ib32)))
    ls = Int32(2 * Int((qw >> 60) & 0xf) + 1)
    g = (
        VecElement(unsafe_load(pg, Int(qw & 0xff) + 1)),
        VecElement(unsafe_load(pg, Int((qw >> 8) & 0xff) + 1)),
        VecElement(unsafe_load(pg, Int((qw >> 16) & 0xff) + 1)),
        VecElement(unsafe_load(pg, Int((qw >> 24) & 0xff) + 1)),
    )
    s = (
        VecElement(unsafe_load(ps, Int((qw >> 32) & 0x7f) + 1)),
        VecElement(unsafe_load(ps, Int((qw >> 39) & 0x7f) + 1)),
        VecElement(unsafe_load(ps, Int((qw >> 46) & 0x7f) + 1)),
        VecElement(unsafe_load(ps, Int((qw >> 53) & 0x7f) + 1)),
    )
    return g, s, ls
end

# Writes out[t0 + c - 1] for the K = `v` columns starting at t0, each the
# raw dot (no alpha/beta), bit-equal to _dot_iq2xxs_q8k_avx2 on column c.
@noinline function _dot_iq2xxs_multi!(out::AbstractVector{Float32}, w::AbstractVector{UInt8},
    woff::Int, xqs, t0::Int, v::Val{K}) where {K}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, 66 * nb)
    _multi_check(xqs, t0, K, nb)
    if nb == 0
        for c in 1:K
            out[t0 + c - 1] = 0.0f0
        end
        return out
    end
    GC.@preserve w xqs begin
        pw = pointer(w, woff + 1)
        pg = pointer(IQ2XXS_GRID)
        ps = pointer(_IQ_SIGN8_TAB)
        pq = ntuple(c -> pointer(xqs[t0 + c - 1].qs), Val(K))
        dv = ntuple(c -> xqs[t0 + c - 1].d, Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb - 1)
            b = pw + 66 * ib
            dw = _iq_f16_at(b)
            acc = ntuple(_ -> _IQ_ACC_ZERO, Val(K))
            for ib32 in 0:7
                g, s, ls = _iq2xxs_sub_ops(pg, ps, b, ib32)
                acc = _multi_iq2_step(acc, g, s, pq, ib * QK_K + 32 * ib32, ls, ls)
            end
            sumf = _multi_fadd(sumf, _multi_d(dw, dv, ib), acc)
        end
        res = _multi_scale(sumf, 0.125f0)
        for c in 1:K
            out[t0 + c - 1] = res[c]
        end
    end
    return out
end

# ---------------------------------------------------------------- IQ3_S

# Grid and sign operands of IQ3_S sub-block k (0..7) of the block at pointer b.
# Same unpacking as _dot_iq3s_q8k_avx2.
@inline function _iq3s_sub_ops(pg, ps, b::Ptr{UInt8}, k::Int)
    qh = Int(unsafe_load(b + 66 + k))
    qb = b + 2 + 8 * k
    sb = b + 74 + 4 * k
    g = (
        VecElement(unsafe_load(pg, (Int(unsafe_load(qb)) | ((qh << 8) & 256)) + 1)),
        VecElement(unsafe_load(pg, (Int(unsafe_load(qb + 1)) | ((qh << 7) & 256)) + 1)),
        VecElement(unsafe_load(pg, (Int(unsafe_load(qb + 2)) | ((qh << 6) & 256)) + 1)),
        VecElement(unsafe_load(pg, (Int(unsafe_load(qb + 3)) | ((qh << 5) & 256)) + 1)),
        VecElement(unsafe_load(pg, (Int(unsafe_load(qb + 4)) | ((qh << 4) & 256)) + 1)),
        VecElement(unsafe_load(pg, (Int(unsafe_load(qb + 5)) | ((qh << 3) & 256)) + 1)),
        VecElement(unsafe_load(pg, (Int(unsafe_load(qb + 6)) | ((qh << 2) & 256)) + 1)),
        VecElement(unsafe_load(pg, (Int(unsafe_load(qb + 7)) | ((qh << 1) & 256)) + 1)),
    )
    s = (
        VecElement(unsafe_load(ps, Int(unsafe_load(sb)) + 1)),
        VecElement(unsafe_load(ps, Int(unsafe_load(sb + 1)) + 1)),
        VecElement(unsafe_load(ps, Int(unsafe_load(sb + 2)) + 1)),
        VecElement(unsafe_load(ps, Int(unsafe_load(sb + 3)) + 1)),
    )
    return g, s
end

@noinline function _dot_iq3s_multi!(out::AbstractVector{Float32}, w::AbstractVector{UInt8},
    woff::Int, xqs, t0::Int, v::Val{K}) where {K}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, 110 * nb)
    _multi_check(xqs, t0, K, nb)
    if nb == 0
        for c in 1:K
            out[t0 + c - 1] = 0.0f0
        end
        return out
    end
    GC.@preserve w xqs begin
        pw = pointer(w, woff + 1)
        pg = pointer(IQ3S_GRID)
        ps = pointer(_IQ_SIGN8_FULL_TAB)
        pq = ntuple(c -> pointer(xqs[t0 + c - 1].qs), Val(K))
        dv = ntuple(c -> xqs[t0 + c - 1].d, Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb - 1)
            b = pw + 110 * ib
            dw = _iq_f16_at(b)
            acc = ntuple(_ -> _IQ_ACC_ZERO, Val(K))
            for m in 0:3
                scb = Int(unsafe_load(b + 106 + m))
                for half in 0:1
                    k = 2 * m + half
                    ls = Int32(2 * ((scb >> (4 * half)) & 0x0f) + 1)
                    g, s = _iq3s_sub_ops(pg, ps, b, k)
                    acc = _multi_iq3_step(acc, g, s, pq, ib * QK_K + 32 * k, ls)
                end
            end
            sumf = _multi_fadd(sumf, _multi_d(dw, dv, ib), acc)
        end
        for c in 1:K
            out[t0 + c - 1] = sumf[c]
        end
    end
    return out
end

# ---------------------------------------------------------------- dispatch

# True for the types with a multi-column kernel. Every other type takes the
# per-column loop in _dot_q8k_multi_tile!.
_q8k_multi_type(t::Int32) = t == GGML_TYPE_IQ2_XXS || t == GGML_TYPE_IQ3_S

# Writes out[t0 + c - 1] = _dot_q8k(ggml_type, w, woff, xqs[t0 + c - 1], simd)
# for the K columns of the tile starting at t0. The AVX2 multi kernels are used
# only when simd is true, and the per-column loop otherwise.
function _dot_q8k_multi_tile!(out::AbstractVector{Float32}, ggml_type::Int32, w::AbstractVector{UInt8},
    woff::Int, xqs, t0::Int, v::Val{K}, simd::Bool) where {K}
    if simd && _q8k_multi_type(ggml_type)
        if ggml_type == GGML_TYPE_IQ2_XXS
            return _dot_iq2xxs_multi!(out, w, woff, xqs, t0, v)
        else
            return _dot_iq3s_multi!(out, w, woff, xqs, t0, v)
        end
    end
    for c in 1:K
        out[t0 + c - 1] = _dot_q8k(ggml_type, w, woff, xqs[t0 + c - 1], simd)
    end
    return out
end

"""
    _dot_q8k_multi!(out, ggml_type, w, woff, xqs, t0, ntile, simd) -> out

Raw dots of the weight row at 0-based byte `woff` of `w` with the `ntile`
Q8_K activations `xqs[t0:t0+ntile-1]`, written to `out[t0:...]`. Column c is
bit-equal to `_dot_q8k(ggml_type, w, woff, xqs[t0+c-1], simd)`. `ntile` is
1..8 so that any remainder is a tile.
"""
function _dot_q8k_multi!(out::AbstractVector{Float32}, ggml_type::Int32, w::AbstractVector{UInt8},
    woff::Int, xqs, t0::Int, ntile::Int, simd::Bool)
    1 <= ntile <= 8 || error("q8k multi: tile of $ntile columns (1..8)")
    t0 + ntile - 1 <= length(xqs) || error("q8k multi: columns $t0:$(t0 + ntile - 1) exceed $(length(xqs))")
    ntile == 1 && return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(1), simd)
    ntile == 2 && return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(2), simd)
    ntile == 3 && return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(3), simd)
    ntile == 4 && return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(4), simd)
    ntile == 5 && return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(5), simd)
    ntile == 6 && return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(6), simd)
    ntile == 7 && return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(7), simd)
    return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(8), simd)
end
