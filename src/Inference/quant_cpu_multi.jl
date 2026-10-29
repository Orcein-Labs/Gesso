# Multi-column Q8_K dot products: one weight row against K activation vectors
# in one pass (prefill and any seq > 1 projection).
#
# The single-column kernels spend most of their time unpacking a sub-block of
# weights (grid gathers, sign masks, scales). Here that unpacking runs once per
# sub-block and each column then feeds the same unpacked operands through its
# own accumulator and activation pointer. Column c therefore executes the same
# integer instructions with the same operands as `_dot_q8k` on its vector, and
# adds its block sums in the same float order, so the result is bit-identical
# (saturating vpmaddubsw is operand-sensitive, which is why the activation stays
# the signed operand in every column).
#
# Tiles hold at most _MULTI_TILE columns; a remainder of 1..7 columns runs as a
# smaller tile, so any token count works.
#
# IQ4_XS is the exception to shared unpacking: its nibble-to-table lookup lives
# inside the llvmcall, so each column repeats that step. Its scale decode and
# the row pointers are still shared.
#
# Q2_K shares its unpack through a different integer core; Q4_K, Q5_K, Q6_K and
# Q8_0 share the rebuilt weights. Each is explained in its section below.

const _MULTI_TILE = 8

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

# Per-column block scale: the same product the single kernel forms, dw * x.d.
@inline _multi_d(dw::Float32, dv::NTuple{K, Vector{Float32}}, ib::Int) where {K} =
    ntuple(c -> dw * dv[c][ib+1], Val(K))

# sumf[c] += d[c] * Float32(hsum(acc[c])), in block order as the single kernel.
@inline _multi_fadd(
    sumf::NTuple{K, Float32},
    d::NTuple{K, Float32},
    acc::NTuple{K, NTuple{8, VecElement{Int32}}},
) where {K} = ntuple(c -> sumf[c] + d[c] * Float32(_iq_hsum8(acc[c])), Val(K))

# The 16-lane IQ1 accumulators: lanes 1-8 and 9-16 are summed separately.
@inline function _iq1_pair(a::NTuple{16, VecElement{Int32}})
    s1 =
        a[1].value +
        a[2].value +
        a[3].value +
        a[4].value +
        a[5].value +
        a[6].value +
        a[7].value +
        a[8].value
    s2 =
        a[9].value +
        a[10].value +
        a[11].value +
        a[12].value +
        a[13].value +
        a[14].value +
        a[15].value +
        a[16].value
    return s1, s2
end

@inline _multi_fadd_pair(
    sumf::NTuple{K, Float32},
    d::NTuple{K, Float32},
    acc::NTuple{K, NTuple{16, VecElement{Int32}}},
) where {K} = ntuple(c -> begin
    s1, s2 = _iq1_pair(acc[c])
    sumf[c] + d[c] * (Float32(s1) + 0.125f0 * Float32(s2))
end, Val(K))

@inline _multi_scale(sumf::NTuple{K, Float32}, s::Float32) where {K} =
    ntuple(c -> s * sumf[c], Val(K))

@inline _multi_iq2_step(
    acc::NTuple{K, NTuple{8, VecElement{Int32}}},
    g,
    s,
    pq::NTuple{K, Ptr{Int8}},
    off::Int,
    ls1::Int32,
    ls2::Int32,
) where {K} = ntuple(c -> _iq2_sub_acc(acc[c], g, s, pq[c] + off, ls1, ls2), Val(K))

@inline _multi_iq3_step(
    acc::NTuple{K, NTuple{8, VecElement{Int32}}},
    g,
    s,
    pq::NTuple{K, Ptr{Int8}},
    off::Int,
    ls::Int32,
) where {K} = ntuple(c -> _iq3_sub_acc(acc[c], g, s, pq[c] + off, ls), Val(K))

@inline _multi_iq1m_step(
    acc::NTuple{K, NTuple{16, VecElement{Int32}}},
    g,
    pq::NTuple{K, Ptr{Int8}},
    off::Int,
    wt::NTuple{16, VecElement{Int32}},
) where {K} = ntuple(c -> _iq1m_sub_acc(acc[c], g, pq[c] + off, wt), Val(K))

@inline _multi_iq4_t(pq::NTuple{K, Ptr{Int8}}, qs::Ptr{UInt8}, off::Int) where {K} =
    ntuple(c -> _iq4_sub_avx2(_Q8K_KV_IQ4NL_X2, qs, pq[c] + off), Val(K))

@inline _multi_iq4_acc(
    sumf::NTuple{K, Float32},
    d4d8::NTuple{K, Float32},
    ls::Int32,
    t::NTuple{K, Int32},
) where {K} = ntuple(c -> sumf[c] + (d4d8[c] * Float32(ls)) * Float32(t[c]), Val(K))

# Column count of a tile after checking that every column has the row's block
# count; a mismatch would read past the end of a shorter activation.
# A prefill's d table: per tile of _MULTI_TILE columns and per block, the
# columns' d values side by side, so the Q8_0 and Q6_K tile kernels read them as
# one vector instead of gathering one per column. Tile i (0-based, columns
# T*i + 1 ...) keeps block ib's column c at index T*(nb*i + ib) + c.
function _multi_d_table(xqs::AbstractVector)
    T = _MULTI_TILE
    nb = length(xqs[1].d)
    dt = zeros(Float32, T * nb * cld(length(xqs), T))
    @inbounds for t in eachindex(xqs)
        i, c = divrem(t - 1, T)
        d = xqs[t].d
        length(d) == nb || error("q8k multi: column $t has $(length(d)) blocks, not $nb")
        for ib in 0:(nb-1)
            dt[T*(nb*i+ib)+c+1] = d[ib+1]
        end
    end
    return dt
end

# Offset of the tile at t0 in a d table, after checking that the tile is one of
# the table's.
function _multi_d_base(dt::Vector{Float32}, t0::Int, K::Int, nb::Int)
    T = _MULTI_TILE
    i, r = divrem(t0 - 1, T)
    r == 0 && K <= T && length(dt) >= T * nb * (i + 1) ||
        error("q8k multi: tile $t0:$(t0 + K - 1) is not in the d table")
    return T * nb * i
end
_multi_d_base(::Nothing, t0::Int, K::Int, nb::Int) = 0

# dw * d for each tile column at block ib: the same products either way.
Base.@propagate_inbounds _multi_tile_d(
    ::Nothing,
    xv::NTuple{K, Any},
    i0::Int,
    ib::Int,
    dw::Float32,
) where {K} = ntuple(c -> dw * xv[c].d[ib+1], Val(K))
Base.@propagate_inbounds _multi_tile_d(
    dt::Vector{Float32},
    xv::NTuple{K, Any},
    i0::Int,
    ib::Int,
    dw::Float32,
) where {K} = ntuple(c -> dw * dt[i0+_MULTI_TILE*ib+c], Val(K))

function _multi_check(xqs, t0::Int, K::Int, nb::Int)
    for c in 1:K
        length(xqs[t0+c-1].d) == nb || error(
            "q8k multi: activation $(t0 + c - 1) has $(length(xqs[t0 + c - 1].d)) blocks, row has $nb",
        )
    end
    return nothing
end

function _multi_zero!(out::AbstractVector{Float32}, t0::Int, K::Int)
    for c in 1:K
        out[t0+c-1] = 0.0f0
    end
    return out
end

# ---------------------------------------------------------------- 8-lane families
#
# Shared drivers for the IQ2 and IQ3 layouts. `subops(pg, ps, b, k)` returns the
# operands of sub-block k (0..7) and its scale. Sub-blocks run in order 0..7 as
# in the single kernels, so each column's accumulator sees the same sequence.

# IQ2 sub-block operands: (g, s, ls1, ls2).
@noinline function _iq2_multi_core!(
    out::AbstractVector{Float32},
    w::AbstractVector{UInt8},
    woff::Int,
    xqs,
    t0::Int,
    v::Val{K},
    rb::Int,
    grid,
    signs,
    subops::F,
    scale::Float32,
) where {K, F}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, rb * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    GC.@preserve w xqs grid signs begin
        pw = pointer(w, woff + 1)
        pg = pointer(grid)
        ps = pointer(signs)
        pq = ntuple(c -> pointer(xqs[t0+c-1].qs), Val(K))
        dv = ntuple(c -> xqs[t0+c-1].d, Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb-1)
            b = pw + rb * ib
            dw = _iq_f16_at(b)
            acc = ntuple(_ -> _IQ_ACC_ZERO, Val(K))
            for ib32 in 0:7
                g, s, ls1, ls2 = subops(pg, ps, b, ib32)
                acc = _multi_iq2_step(acc, g, s, pq, ib * QK_K + 32 * ib32, ls1, ls2)
            end
            sumf = _multi_fadd(sumf, _multi_d(dw, dv, ib), acc)
        end
        res = _multi_scale(sumf, scale)
        for c in 1:K
            out[t0+c-1] = res[c]
        end
    end
    return out
end

# IQ3 sub-block operands: (g, s, ls). `scale` is the format's final factor
# (0.25 for IQ3_XXS, 1 for IQ3_S, which has none).
@noinline function _iq3_multi_core!(
    out::AbstractVector{Float32},
    w::AbstractVector{UInt8},
    woff::Int,
    xqs,
    t0::Int,
    v::Val{K},
    rb::Int,
    grid,
    signs,
    subops::F,
    scale::Float32,
) where {K, F}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, rb * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    GC.@preserve w xqs grid signs begin
        pw = pointer(w, woff + 1)
        pg = pointer(grid)
        ps = pointer(signs)
        pq = ntuple(c -> pointer(xqs[t0+c-1].qs), Val(K))
        dv = ntuple(c -> xqs[t0+c-1].d, Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb-1)
            b = pw + rb * ib
            dw = _iq_f16_at(b)
            acc = ntuple(_ -> _IQ_ACC_ZERO, Val(K))
            for k in 0:7
                g, s, ls = subops(pg, ps, b, k)
                acc = _multi_iq3_step(acc, g, s, pq, ib * QK_K + 32 * k, ls)
            end
            sumf = _multi_fadd(sumf, _multi_d(dw, dv, ib), acc)
        end
        res = _multi_scale(sumf, scale)
        for c in 1:K
            out[t0+c-1] = res[c]
        end
    end
    return out
end

# ---------------------------------------------------------------- IQ2 operands

# IQ2_XXS: scale nibble in the high 4 bits of the high 32-bit word.
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
    return g, s, ls, ls
end

# IQ2_S: scale nibbles at b+74, grid high bits in b+66, sign bytes at b+34.
@inline function _iq2s_sub_ops(pg, ps, b::Ptr{UInt8}, ib32::Int)
    sc = unsafe_load(b + 74 + ib32)
    ls1 = Int32(2 * (sc & 0x0f) + 1)
    ls2 = Int32(2 * (sc >> 4) + 1)
    qh = Int(unsafe_load(b + 66 + ib32))
    qb = b + 2 + 4 * ib32
    sb = b + 34 + 4 * ib32
    i0 = Int(unsafe_load(qb)) | ((qh << 8) & 0x300)
    i1 = Int(unsafe_load(qb + 1)) | ((qh << 6) & 0x300)
    i2 = Int(unsafe_load(qb + 2)) | ((qh << 4) & 0x300)
    i3 = Int(unsafe_load(qb + 3)) | ((qh << 2) & 0x300)
    g = (
        VecElement(unsafe_load(pg, i0 + 1)),
        VecElement(unsafe_load(pg, i1 + 1)),
        VecElement(unsafe_load(pg, i2 + 1)),
        VecElement(unsafe_load(pg, i3 + 1)),
    )
    s = (
        VecElement(unsafe_load(ps, Int(unsafe_load(sb)) + 1)),
        VecElement(unsafe_load(ps, Int(unsafe_load(sb + 1)) + 1)),
        VecElement(unsafe_load(ps, Int(unsafe_load(sb + 2)) + 1)),
        VecElement(unsafe_load(ps, Int(unsafe_load(sb + 3)) + 1)),
    )
    return g, s, ls1, ls2
end

# IQ2_XS: 9-bit grid index and 7-bit sign index per group, scales at b+66.
@inline function _iq2xs_sub_ops(pg, ps, b::Ptr{UInt8}, ib32::Int)
    sc = unsafe_load(b + 66 + ib32)
    ls1 = Int32(2 * (sc & 0x0f) + 1)
    ls2 = Int32(2 * (sc >> 4) + 1)
    qw = ltoh(unsafe_load(Ptr{UInt64}(b + 2 + 8 * ib32)))
    g = (
        VecElement(unsafe_load(pg, Int(qw & 0x1ff) + 1)),
        VecElement(unsafe_load(pg, Int((qw >> 16) & 0x1ff) + 1)),
        VecElement(unsafe_load(pg, Int((qw >> 32) & 0x1ff) + 1)),
        VecElement(unsafe_load(pg, Int((qw >> 48) & 0x1ff) + 1)),
    )
    s = (
        VecElement(unsafe_load(ps, Int((qw >> 9) & 0x7f) + 1)),
        VecElement(unsafe_load(ps, Int((qw >> 25) & 0x7f) + 1)),
        VecElement(unsafe_load(ps, Int((qw >> 41) & 0x7f) + 1)),
        VecElement(unsafe_load(ps, Int((qw >> 57) & 0x7f) + 1)),
    )
    return g, s, ls1, ls2
end

@noinline _dot_iq2xxs_multi!(out, w, woff, xqs, t0, v::Val) = _iq2_multi_core!(
    out,
    w,
    woff,
    xqs,
    t0,
    v,
    66,
    IQ2XXS_GRID,
    _IQ_SIGN8_TAB,
    _iq2xxs_sub_ops,
    0.125f0,
)

@noinline _dot_iq2s_multi!(out, w, woff, xqs, t0, v::Val) = _iq2_multi_core!(
    out,
    w,
    woff,
    xqs,
    t0,
    v,
    82,
    IQ2S_GRID,
    _IQ_SIGN8_FULL_TAB,
    _iq2s_sub_ops,
    0.125f0,
)

@noinline _dot_iq2xs_multi!(out, w, woff, xqs, t0, v::Val) = _iq2_multi_core!(
    out,
    w,
    woff,
    xqs,
    t0,
    v,
    74,
    IQ2XS_GRID,
    _IQ_SIGN8_TAB,
    _iq2xs_sub_ops,
    0.125f0,
)

# ---------------------------------------------------------------- IQ3 operands

# IQ3_S: sub-block k = 2m + half, scale nibble half of the byte at b+106+m.
@inline function _iq3s_sub_ops(pg, ps, b::Ptr{UInt8}, k::Int)
    qh = Int(unsafe_load(b + 66 + k))
    qb = b + 2 + 8 * k
    sb = b + 74 + 4 * k
    scb = Int(unsafe_load(b + 106 + (k >> 1)))
    ls = Int32(2 * ((scb >> (4 * (k & 1))) & 0x0f) + 1)
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
    return g, s, ls
end

# IQ3_XXS: four 7-bit sign indices and the scale (top 4 bits) in the aux word.
@inline function _iq3xxs_sub_ops(pg, ps, b::Ptr{UInt8}, k::Int)
    aux = ltoh(unsafe_load(Ptr{UInt32}(b + 66 + 4 * k)))
    ls = Int32(2 * (aux >> 28) + 1)
    qi = ltoh(unsafe_load(Ptr{UInt64}(b + 2 + 8 * k)))
    g = (
        VecElement(unsafe_load(pg, Int(qi & 0xff) + 1)),
        VecElement(unsafe_load(pg, Int((qi >> 8) & 0xff) + 1)),
        VecElement(unsafe_load(pg, Int((qi >> 16) & 0xff) + 1)),
        VecElement(unsafe_load(pg, Int((qi >> 24) & 0xff) + 1)),
        VecElement(unsafe_load(pg, Int((qi >> 32) & 0xff) + 1)),
        VecElement(unsafe_load(pg, Int((qi >> 40) & 0xff) + 1)),
        VecElement(unsafe_load(pg, Int((qi >> 48) & 0xff) + 1)),
        VecElement(unsafe_load(pg, Int((qi >> 56) & 0xff) + 1)),
    )
    s = (
        VecElement(unsafe_load(ps, Int(aux & 0x7f) + 1)),
        VecElement(unsafe_load(ps, Int((aux >> 7) & 0x7f) + 1)),
        VecElement(unsafe_load(ps, Int((aux >> 14) & 0x7f) + 1)),
        VecElement(unsafe_load(ps, Int((aux >> 21) & 0x7f) + 1)),
    )
    return g, s, ls
end

@noinline _dot_iq3s_multi!(out, w, woff, xqs, t0, v::Val) = _iq3_multi_core!(
    out,
    w,
    woff,
    xqs,
    t0,
    v,
    110,
    IQ3S_GRID,
    _IQ_SIGN8_FULL_TAB,
    _iq3s_sub_ops,
    1.0f0,
)

@noinline _dot_iq3xxs_multi!(out, w, woff, xqs, t0, v::Val) = _iq3_multi_core!(
    out,
    w,
    woff,
    xqs,
    t0,
    v,
    98,
    IQ3XXS_GRID,
    _IQ_SIGN8_TAB,
    _iq3xxs_sub_ops,
    0.25f0,
)

# ---------------------------------------------------------------- IQ1 (16-lane)

# IQ1_S: one scale per block, each sub-block's sign rides on the quant sums.
@noinline function _dot_iq1s_multi!(
    out::AbstractVector{Float32},
    w::AbstractVector{UInt8},
    woff::Int,
    xqs,
    t0::Int,
    v::Val{K},
) where {K}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, 50 * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    GC.@preserve w xqs begin
        pw = pointer(w, woff + 1)
        pg = pointer(IQ1S_GRID)
        pq = ntuple(c -> pointer(xqs[t0+c-1].qs), Val(K))
        dv = ntuple(c -> xqs[t0+c-1].d, Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb-1)
            b = pw + 50 * ib
            dw = _iq_f16_at(b)
            acc = ntuple(_ -> _IQ1M_ACC_ZERO, Val(K))
            for s in 0:7
                qh = Int(ltoh(unsafe_load(Ptr{UInt16}(b + 34 + 2 * s))))
                ls = Int32(2 * ((qh >> 12) & 7) + 1)
                ld = (qh & 0x8000) != 0 ? -ls : ls
                qb = b + 2 + 4 * s
                i0 = Int(unsafe_load(qb)) | ((qh & 7) << 8)
                i1 = Int(unsafe_load(qb + 1)) | (((qh >> 3) & 7) << 8)
                i2 = Int(unsafe_load(qb + 2)) | (((qh >> 6) & 7) << 8)
                i3 = Int(unsafe_load(qb + 3)) | (((qh >> 9) & 7) << 8)
                g = (
                    VecElement(unsafe_load(pg, i0 + 1)),
                    VecElement(unsafe_load(pg, i1 + 1)),
                    VecElement(unsafe_load(pg, i2 + 1)),
                    VecElement(unsafe_load(pg, i3 + 1)),
                )
                wt = (
                    VecElement(ls),
                    VecElement(ls),
                    VecElement(ls),
                    VecElement(ls),
                    VecElement(ls),
                    VecElement(ls),
                    VecElement(ls),
                    VecElement(ls),
                    VecElement(ld),
                    VecElement(ld),
                    VecElement(ld),
                    VecElement(ld),
                    VecElement(ld),
                    VecElement(ld),
                    VecElement(ld),
                    VecElement(ld),
                )
                acc = _multi_iq1m_step(acc, g, pq, ib * QK_K + 32 * s, wt)
            end
            sumf = _multi_fadd_pair(sumf, _multi_d(dw, dv, ib), acc)
        end
        for c in 1:K
            out[t0+c-1] = sumf[c]
        end
    end
    return out
end

# IQ1_M: the block scale is four 3-bit sub-scales packed in the scale words.
@noinline function _dot_iq1m_multi!(
    out::AbstractVector{Float32},
    w::AbstractVector{UInt8},
    woff::Int,
    xqs,
    t0::Int,
    v::Val{K},
) where {K}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, 56 * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    GC.@preserve w xqs begin
        pw = pointer(w, woff + 1)
        pg = pointer(IQ1S_GRID)
        pq = ntuple(c -> pointer(xqs[t0+c-1].qs), Val(K))
        dv = ntuple(c -> xqs[t0+c-1].d, Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb-1)
            b = pw + 56 * ib
            s0 = ltoh(unsafe_load(Ptr{UInt16}(b + 48)))
            s1 = ltoh(unsafe_load(Ptr{UInt16}(b + 50)))
            s2 = ltoh(unsafe_load(Ptr{UInt16}(b + 52)))
            s3 = ltoh(unsafe_load(Ptr{UInt16}(b + 54)))
            dw = Float32(reinterpret(Float16, _iq1m_scale_bits(s0, s1, s2, s3)))
            acc = ntuple(_ -> _IQ1M_ACC_ZERO, Val(K))
            for s in 0:7
                scw = Int(ltoh(unsafe_load(Ptr{UInt16}(b + 48 + 2 * (s >> 1)))))
                ls1 = Int32(2 * ((scw >> (6 * (s & 1))) & 0x7) + 1)
                ls2 = Int32(2 * ((scw >> (6 * (s & 1) + 3)) & 0x7) + 1)
                qh0 = Int(unsafe_load(b + 32 + 2 * s))
                qh1 = Int(unsafe_load(b + 33 + 2 * s))
                qb = b + 4 * s
                i0 = Int(unsafe_load(qb)) | ((qh0 << 8) & 0x700)
                i1 = Int(unsafe_load(qb + 1)) | ((qh0 << 4) & 0x700)
                i2 = Int(unsafe_load(qb + 2)) | ((qh1 << 8) & 0x700)
                i3 = Int(unsafe_load(qb + 3)) | ((qh1 << 4) & 0x700)
                g = (
                    VecElement(unsafe_load(pg, i0 + 1)),
                    VecElement(unsafe_load(pg, i1 + 1)),
                    VecElement(unsafe_load(pg, i2 + 1)),
                    VecElement(unsafe_load(pg, i3 + 1)),
                )
                e0 = (qh0 & 0x08) != 0 ? -ls1 : ls1
                e1 = (qh0 & 0x80) != 0 ? -ls1 : ls1
                e2 = (qh1 & 0x08) != 0 ? -ls2 : ls2
                e3 = (qh1 & 0x80) != 0 ? -ls2 : ls2
                wt = (
                    VecElement(ls1),
                    VecElement(ls1),
                    VecElement(ls1),
                    VecElement(ls1),
                    VecElement(ls2),
                    VecElement(ls2),
                    VecElement(ls2),
                    VecElement(ls2),
                    VecElement(e0),
                    VecElement(e0),
                    VecElement(e1),
                    VecElement(e1),
                    VecElement(e2),
                    VecElement(e2),
                    VecElement(e3),
                    VecElement(e3),
                )
                acc = _multi_iq1m_step(acc, g, pq, ib * QK_K + 32 * s, wt)
            end
            sumf = _multi_fadd_pair(sumf, _multi_d(dw, dv, ib), acc)
        end
        for c in 1:K
            out[t0+c-1] = sumf[c]
        end
    end
    return out
end

# ---------------------------------------------------------------- IQ4_XS

@noinline function _dot_iq4_xs_multi!(
    out::AbstractVector{Float32},
    w::AbstractVector{UInt8},
    woff::Int,
    xqs,
    t0::Int,
    v::Val{K},
) where {K}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, 136 * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    GC.@preserve w xqs begin
        pq = ntuple(c -> pointer(xqs[t0+c-1].qs), Val(K))
        dv = ntuple(c -> xqs[t0+c-1].d, Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb-1)
            b = woff + ib * 136
            d4d8 = _multi_d(_ld_f16(w, b), dv, ib)
            h = _ld_u16(w, b + 2)
            ao = ib * QK_K
            for s in 0:7
                sl = _ld_u8(w, b + 4 + (s >> 1))
                ls =
                    (
                        Int32((sl >> (4 * (s & 1))) & 0x0f) |
                        (Int32((h >> (2 * s)) & 0x0003) << 4)
                    ) - Int32(32)
                qs = pointer(w, b + 9 + 16 * s)
                t = _multi_iq4_t(pq, qs, ao + 32 * s)
                sumf = _multi_iq4_acc(sumf, d4d8, ls, t)
            end
        end
        for c in 1:K
            out[t0+c-1] = sumf[c]
        end
    end
    return out
end

# ---------------------------------------------------------------- Q2_K
#
# The single kernel's float step reads only two integers per block: isum (the
# scaled sub-block dots) and summs (bsums against the mins). Integer addition
# is exact, so any grouping of the products that yields the same two totals
# keeps every column bit-equal. That frees the core from the lane layout of
# _q2k_ints_avx2: here each 32-value plane is one vpmaddubsw (weight 0..3 as the
# unsigned operand, so a pair sums to at most 2*3*128 and cannot saturate)
# followed by one vpmaddwd against the two sub-block scales. The planes, scales
# and mins are built once per block and shared by the K columns.

function _kq_ir_q2k_multi(K::Int)
    io = IOBuffer()
    qargs = join(("ptr %q$c" for c in 0:(K-1)), ", ")
    bargs = join(("ptr %b$c" for c in 0:(K-1)), ", ")
    print(
        io,
        "declare <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8>, <32 x i8>)\n",
        "declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>)\n",
        "define <$(2K) x i32> @entry(ptr %w, $qargs, $bargs) #0 {\n",
        "  %scb = load <16 x i8>, ptr %w, align 1\n",
        "  %scl = and <16 x i8> %scb, splat (i8 15)\n",
        "  %mnb = lshr <16 x i8> %scb, splat (i8 4)\n",
        "  %mn = zext <16 x i8> %mnb to <16 x i16>\n",
    )
    for k in 0:1
        _kq_ld(io, "qv$k", "w", 16 + 32 * k)
    end
    acc = ["zeroinitializer" for _ in 0:(K-1)]
    for k in 0:1, t in 0:3
        i = 8 * k + 2 * t
        mask = join(vcat(fill("i32 $i", 8), fill("i32 $(i + 1)", 8)), ", ")
        print(
            io,
            "  %pa$k$t = lshr <32 x i8> %qv$k, splat (i8 $(2t))\n",
            "  %p$k$t = and <32 x i8> %pa$k$t, splat (i8 3)\n",
            "  %s8$k$t = shufflevector <16 x i8> %scl, <16 x i8> poison, <16 x i32> <$mask>\n",
            "  %s$k$t = zext <16 x i8> %s8$k$t to <16 x i16>\n",
        )
        for c in 0:(K-1)
            _kq_ld(io, "x$c$k$t", "q$c", 128 * k + 32 * t)
            print(
                io,
                "  %m$c$k$t = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %p$k$t, <32 x i8> %x$c$k$t)\n",
                "  %d$c$k$t = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %m$c$k$t, <16 x i16> %s$k$t)\n",
                "  %a$c$k$t = add <8 x i32> $(acc[c + 1]), %d$c$k$t\n",
            )
            acc[c+1] = "%a$c$k$t"
        end
    end
    res = "poison"
    for c in 0:(K-1)
        print(
            io,
            "  %bs$c = load <16 x i16>, ptr %b$c, align 1\n",
            "  %mm$c = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %bs$c, <16 x i16> %mn)\n",
        )
        for (name, v) in (("i", acc[c+1]), ("n", "%mm$c"))
            for e in 0:7
                print(io, "  %h$name$c.$e = extractelement <8 x i32> $v, i32 $e\n")
            end
            prev = "%h$name$c.0"
            for e in 1:7
                print(io, "  %h$name$c.s$e = add i32 $prev, %h$name$c.$e\n")
                prev = "%h$name$c.s$e"
            end
            slot = name == "i" ? 2c : 2c + 1
            print(
                io,
                "  %r$name$c = insertelement <$(2K) x i32> $res, i32 $prev, i32 $slot\n",
            )
            res = "%r$name$c"
        end
    end
    print(io, "  ret <$(2K) x i32> $res\n}\n", _KQ_ATTRS)
    return String(take!(io))
end

# (isum, summs) of one block for each of the K columns, interleaved. One method
# per tile width, because llvmcall needs its IR and argument list as constants.
for K in 1:8
    qs = [:(pq[$c]) for c in 1:K]
    bs = [:(pb[$c]) for c in 1:K]
    argt = Tuple{Ptr{UInt8}, ntuple(_ -> Ptr{Int8}, K)..., ntuple(_ -> Ptr{Int16}, K)...}
    @eval @inline _q2k_multi_ints(
        pw::Ptr{UInt8},
        pq::NTuple{$K, Ptr{Int8}},
        pb::NTuple{$K, Ptr{Int16}},
    ) = Base.llvmcall(
        ($(_kq_ir_q2k_multi(K)), "entry"),
        NTuple{$(2K), VecElement{Int32}},
        $argt,
        pw,
        $(qs...),
        $(bs...),
    )
end

# sumf[c] += dall*isum - dmin*summs, with dall and dmin formed as the single
# kernel forms them.
@inline _multi_q2k_fadd(
    sumf::NTuple{K, Float32},
    xv::NTuple{K, Q8KVec},
    ib::Int,
    fd::Float32,
    fm::Float32,
    r::NTuple{N, VecElement{Int32}},
) where {K, N} = ntuple(
    c ->
        sumf[c] + (
            (xv[c].d[ib+1] * fd) * Float32(r[2c-1].value) -
            (xv[c].d[ib+1] * fm) * Float32(r[2c].value)
        ),
    Val(K),
)

@noinline function _dot_q2k_multi!(
    out::AbstractVector{Float32},
    w::AbstractVector{UInt8},
    woff::Int,
    xqs,
    t0::Int,
    ::Val{K},
) where {K}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, 84 * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    GC.@preserve w xqs begin
        xv = ntuple(c -> xqs[t0+c-1], Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb-1)
            b = woff + ib * 84
            fd = _kq_f16(w, b + 80)
            fm = _kq_f16(w, b + 82)
            r = _q2k_multi_ints(
                pointer(w, b + 1),
                ntuple(c -> pointer(xv[c].qs, ib * QK_K + 1), Val(K)),
                ntuple(c -> pointer(xv[c].bsums, 16 * ib + 1), Val(K)),
            )
            sumf = _multi_q2k_fadd(sumf, xv, ib, fd, fm, r)
        end
        for c in 1:K
            out[t0+c-1] = sumf[c]
        end
    end
    return out
end

# ---------------------------------------------------------------- Q6_K
#
# The single kernel's float step reads 8 integer lanes per block, and each lane
# keeps its own float sum, so here the lanes must be the same lanes: lane l sums
# values l and l + 8 of every 16-value sub-block, times its scale. Integer lane
# totals are exact, so any order of the integer work gives the same block lanes.
#
# Each 32-value chunk holds sub-blocks i (low 16) and j (high 16). vpermd and
# vpshufb reorder it to i_l, i_l+8, j_l, j_l+8 for l = 0..7, so vpmaddubsw pairs
# l with l + 8 inside one sub-block and vpmaddwd then pairs sub-block i with j
# under their two scales, leaving lane l in i32 slot l. The weights go in as
# unsigned 0..63 and 32 x the activation pair sums come off before vpmaddwd,
# which keeps every step exact for activations down to -128. The weight shuffle
# and scales are shared; each column shuffles its activations and runs the two
# vpmaddubsw, the subtract and the vpmaddwd.

const _Q6K_PERMD = "<8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 1, i32 3, i32 5, i32 7>"
const _Q6K_PSHUFB = let lane = (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15)
    "<32 x i8> <" * join(("i8 $(b)" for b in (lane..., lane...)), ", ") * ">"
end

# %dst = the chunk %src reordered for the vpmaddubsw / vpmaddwd pairing above.
function _q6k_pair_order(io::IOBuffer, dst::String, src::String)
    print(
        io,
        "  %$(dst)_d = bitcast <32 x i8> %$src to <8 x i32>\n",
        "  %$(dst)_p = call <8 x i32> @llvm.x86.avx2.permd(<8 x i32> %$(dst)_d, $_Q6K_PERMD)\n",
        "  %$(dst)_b = bitcast <8 x i32> %$(dst)_p to <32 x i8>\n",
        "  %$dst = call <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8> %$(dst)_b, $_Q6K_PSHUFB)\n",
    )
end

function _kq_ir_q6k_multi(K::Int)
    io = IOBuffer()
    qargs = join(("ptr %q$c" for c in 0:(K-1)), ", ")
    print(
        io,
        "declare <8 x i32> @llvm.x86.avx2.permd(<8 x i32>, <8 x i32>)\n",
        "declare <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8>, <32 x i8>)\n",
        "declare <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8>, <32 x i8>)\n",
        "declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>)\n",
        "define <$(8K) x float> @entry(ptr %w, $qargs, <$(8K) x float> %s, <$(8K) x float> %d) #0 {\n",
        "  %scbp = getelementptr i8, ptr %w, i64 192\n",
        "  %scb = load <16 x i8>, ptr %scbp, align 1\n",
        "  %scw = sext <16 x i8> %scb to <16 x i16>\n",
    )
    acc = ["zeroinitializer" for _ in 0:(K-1)]
    for h in 0:1
        _kq_ld(io, "qh$h", "w", 128 + 32 * h)
        for t in 0:3
            ht = "$(h)$t"
            i = 8 * h + 2 * t
            _kq_ld(io, "ql$ht", "w", 64 * h + 32 * (t & 1))
            if t < 2
                print(io, "  %nb$ht = and <32 x i8> %ql$ht, splat (i8 15)\n")
            else
                print(
                    io,
                    "  %nh$ht = lshr <32 x i8> %ql$ht, splat (i8 4)\n",
                    "  %nb$ht = and <32 x i8> %nh$ht, splat (i8 15)\n",
                )
            end
            print(
                io,
                "  %hs$ht = lshr <32 x i8> %qh$h, splat (i8 $(2t))\n",
                "  %hm$ht = and <32 x i8> %hs$ht, splat (i8 3)\n",
                "  %hb$ht = shl <32 x i8> %hm$ht, splat (i8 4)\n",
                "  %uv$ht = or <32 x i8> %nb$ht, %hb$ht\n",
            )
            _q6k_pair_order(io, "up$ht", "uv$ht")
            # scales i, j repeated as 16 words: [sc_i, sc_j] x 8
            print(
                io,
                "  %sw$ht = shufflevector <16 x i16> %scw, <16 x i16> poison, <16 x i32> <",
                join(("i32 $(isodd(k) ? i + 1 : i)" for k in 0:15), ", "),
                ">\n",
            )
            for c in 0:(K-1)
                x = "x$c.$ht"
                _kq_ld(io, x, "q$c", 128 * h + 32 * t)
                _q6k_pair_order(io, "xp$c.$ht", x)
                print(
                    io,
                    "  %mu$c.$ht = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %up$ht, <32 x i8> %xp$c.$ht)\n",
                    "  %mo$c.$ht = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> splat (i8 32), <32 x i8> %xp$c.$ht)\n",
                    "  %mf$c.$ht = sub <16 x i16> %mu$c.$ht, %mo$c.$ht\n",
                    "  %ms$c.$ht = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %mf$c.$ht, <16 x i16> %sw$ht)\n",
                    "  %A$c.$ht = add <8 x i32> $(acc[c + 1]), %ms$c.$ht\n",
                )
                acc[c+1] = "%A$c.$ht"
            end
        end
    end
    res = "poison"
    for c in 0:(K-1), e in 0:7
        print(
            io,
            "  %o$c.$e = extractelement <8 x i32> $(acc[c + 1]), i32 $e\n",
            "  %r$c.$e = insertelement <$(8K) x i32> $res, i32 %o$c.$e, i32 $(8c + e)\n",
        )
        res = "%r$c.$e"
    end
    print(
        io,
        "  %fa = sitofp <$(8K) x i32> $res to <$(8K) x float>\n",
        "  %fm = fmul <$(8K) x float> %d, %fa\n",
        "  %fs = fadd <$(8K) x float> %s, %fm\n",
        "  ret <$(8K) x float> %fs\n}\n",
        _KQ_ATTRS,
    )
    return String(take!(io))
end

# One block for all K columns: the 8 float lane sums of each column, column-major,
# with the block's integer lanes scaled and added in.
for K in 1:8
    qs = [:(pq[$c]) for c in 1:K]
    V = NTuple{8K, VecElement{Float32}}
    argt = Tuple{Ptr{UInt8}, ntuple(_ -> Ptr{Int8}, K)..., V, V}
    @eval @inline _q6k_multi_step(pw::Ptr{UInt8}, pq::NTuple{$K, Ptr{Int8}}, s::$V, d::$V) =
        Base.llvmcall(($(_kq_ir_q6k_multi(K)), "entry"), $V, $argt, pw, $(qs...), s, d)
end

# Each column's dw * d, repeated across its 8 lanes.
@inline _multi_q6k_scales(dd::NTuple{K, Float32}) where {K} =
    ntuple(i -> VecElement(dd[(i-1)>>3+1]), Val(8K))

@noinline function _dot_q6k_multi!(
    out::AbstractVector{Float32},
    w::AbstractVector{UInt8},
    woff::Int,
    xqs,
    t0::Int,
    ::Val{K},
    dt=nothing,
) where {K}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, 210 * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    i0 = _multi_d_base(dt, t0, K, nb)
    GC.@preserve w xqs begin
        xv = ntuple(c -> xqs[t0+c-1], Val(K))
        sums = ntuple(_ -> VecElement(0.0f0), Val(8K))
        @inbounds for ib in 0:(nb-1)
            b = woff + ib * 210
            dw = _kq_f16(w, b + 208)
            d = _multi_q6k_scales(_multi_tile_d(dt, xv, i0, ib, dw))
            sums = _q6k_multi_step(
                pointer(w, b + 1),
                ntuple(c -> pointer(xv[c].qs, ib * QK_K + 1), Val(K)),
                sums,
                d,
            )
        end
        for c in 1:K
            sumf = 0.0f0
            for l in 1:8
                sumf += sums[8*(c-1)+l].value
            end
            out[t0+c-1] = sumf
        end
    end
    return out
end

# The activation side of the Q6_K step depends only on the activation, so a
# prefill does it once per token instead of once per weight row: `xp` is qs with
# every 32-value chunk in pair order, and `mo` holds the chunk's 16 offset words
# 32 * (xp[2m-1] + xp[2m]), at the same byte offset as the chunk in `xp`. `d` is
# `q.d` and `q` the source, which the per-column fallback reads.
struct Q6KPrepVec
    d::Vector{Float32}
    q::Q8KVec
    xp::Vector{Int8}
    mo::Vector{Int16}
end

Base.length(x::Q6KPrepVec) = length(x.q)

# Byte k (0-based) of a pair-ordered chunk is source byte _Q6K_SRC[k + 1]: the
# vpermd of _Q6K_PERMD, then the vpshufb of the lane pattern within each half.
const _Q6K_SRC =
    let permd = (0, 2, 4, 6, 1, 3, 5, 7),
        pat = (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15)

        ntuple(
            k -> (b=16 * ((k - 1) >> 4) + pat[(k-1)&15+1]; 4 * permd[b>>2+1] + (b & 3)),
            32,
        )
    end

function _q6k_prep(x::Q8KVec)
    n = length(x.qs)
    xp = Vector{Int8}(undef, n)
    mo = Vector{Int16}(undef, n >> 1)
    @inbounds for o in 0:32:(n-1)
        for k in 1:32
            xp[o+k] = x.qs[o+_Q6K_SRC[k]+1]
        end
        for m in 1:16
            mo[o>>1+m] = Int16(32) * (Int16(xp[o+2m-1]) + Int16(xp[o+2m]))
        end
    end
    return Q6KPrepVec(x.d, x, xp, mo)
end

# The types whose multi kernel reads prepped columns: Q6_K uses xp and mo, Q4_K
# and Q5_K use xp.
_q6k_prep_type(t::Int32) = t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K

# The fallback for a prepped column is the plain Q8_K dot.
@inline _dot_q8k(
    ggml_type::Int32,
    w::AbstractVector{UInt8},
    woff::Int,
    x::Q6KPrepVec,
    simd::Bool,
) = _dot_q8k(ggml_type, w, woff, x.q, simd)

# _kq_ir_q6k_multi with prepped columns: per column pointers q (to xp) and m (to
# mo), so the chunk needs one vpmaddubsw, the subtract and the vpmaddwd.
function _kq_ir_q6k_prep_multi(K::Int)
    io = IOBuffer()
    qargs = join(("ptr %q$c, ptr %m$c" for c in 0:(K-1)), ", ")
    print(
        io,
        "declare <8 x i32> @llvm.x86.avx2.permd(<8 x i32>, <8 x i32>)\n",
        "declare <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8>, <32 x i8>)\n",
        "declare <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8>, <32 x i8>)\n",
        "declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>)\n",
        "define <$(8K) x float> @entry(ptr %w, $qargs, <$(8K) x float> %s, <$(8K) x float> %d) #0 {\n",
        "  %scbp = getelementptr i8, ptr %w, i64 192\n",
        "  %scb = load <16 x i8>, ptr %scbp, align 1\n",
        "  %scw = sext <16 x i8> %scb to <16 x i16>\n",
    )
    acc = ["zeroinitializer" for _ in 0:(K-1)]
    for h in 0:1
        _kq_ld(io, "qh$h", "w", 128 + 32 * h)
        for t in 0:3
            ht = "$(h)$t"
            i = 8 * h + 2 * t
            _kq_ld(io, "ql$ht", "w", 64 * h + 32 * (t & 1))
            if t < 2
                print(io, "  %nb$ht = and <32 x i8> %ql$ht, splat (i8 15)\n")
            else
                print(
                    io,
                    "  %nh$ht = lshr <32 x i8> %ql$ht, splat (i8 4)\n",
                    "  %nb$ht = and <32 x i8> %nh$ht, splat (i8 15)\n",
                )
            end
            print(
                io,
                "  %hs$ht = lshr <32 x i8> %qh$h, splat (i8 $(2t))\n",
                "  %hm$ht = and <32 x i8> %hs$ht, splat (i8 3)\n",
                "  %hb$ht = shl <32 x i8> %hm$ht, splat (i8 4)\n",
                "  %uv$ht = or <32 x i8> %nb$ht, %hb$ht\n",
            )
            _q6k_pair_order(io, "up$ht", "uv$ht")
            print(
                io,
                "  %sw$ht = shufflevector <16 x i16> %scw, <16 x i16> poison, <16 x i32> <",
                join(("i32 $(isodd(k) ? i + 1 : i)" for k in 0:15), ", "),
                ">\n",
            )
            off = 128 * h + 32 * t
            for c in 0:(K-1)
                _kq_ld(io, "xp$c.$ht", "q$c", off)
                print(
                    io,
                    "  %mp$c.$ht = getelementptr i8, ptr %m$c, i64 $off\n",
                    "  %mo$c.$ht = load <16 x i16>, ptr %mp$c.$ht, align 1\n",
                    "  %mu$c.$ht = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %up$ht, <32 x i8> %xp$c.$ht)\n",
                    "  %mf$c.$ht = sub <16 x i16> %mu$c.$ht, %mo$c.$ht\n",
                    "  %ms$c.$ht = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %mf$c.$ht, <16 x i16> %sw$ht)\n",
                    "  %A$c.$ht = add <8 x i32> $(acc[c + 1]), %ms$c.$ht\n",
                )
                acc[c+1] = "%A$c.$ht"
            end
        end
    end
    res = "poison"
    for c in 0:(K-1), e in 0:7
        print(
            io,
            "  %o$c.$e = extractelement <8 x i32> $(acc[c + 1]), i32 $e\n",
            "  %r$c.$e = insertelement <$(8K) x i32> $res, i32 %o$c.$e, i32 $(8c + e)\n",
        )
        res = "%r$c.$e"
    end
    print(
        io,
        "  %fa = sitofp <$(8K) x i32> $res to <$(8K) x float>\n",
        "  %fm = fmul <$(8K) x float> %d, %fa\n",
        "  %fs = fadd <$(8K) x float> %s, %fm\n",
        "  ret <$(8K) x float> %fs\n}\n",
        _KQ_ATTRS,
    )
    return String(take!(io))
end

for K in 1:8
    args = [a for c in 1:K for a in (:(pq[$c]), :(pm[$c]))]
    V = NTuple{8K, VecElement{Float32}}
    argt = Tuple{Ptr{UInt8}, ntuple(_ -> Ptr{Int8}, 2K)..., V, V}
    @eval @inline _q6k_prep_multi_step(
        pw::Ptr{UInt8},
        pq::NTuple{$K, Ptr{Int8}},
        pm::NTuple{$K, Ptr{Int8}},
        s::$V,
        d::$V,
    ) = Base.llvmcall(
        ($(_kq_ir_q6k_prep_multi(K)), "entry"),
        $V,
        $argt,
        pw,
        $(args...),
        s,
        d,
    )
end

@noinline function _dot_q6k_multi!(
    out::AbstractVector{Float32},
    w::AbstractVector{UInt8},
    woff::Int,
    xqs::AbstractVector{Q6KPrepVec},
    t0::Int,
    ::Val{K},
    dt=nothing,
) where {K}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, 210 * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    i0 = _multi_d_base(dt, t0, K, nb)
    GC.@preserve w xqs begin
        xv = ntuple(c -> xqs[t0+c-1], Val(K))
        sums = ntuple(_ -> VecElement(0.0f0), Val(8K))
        @inbounds for ib in 0:(nb-1)
            b = woff + ib * 210
            dw = _kq_f16(w, b + 208)
            d = _multi_q6k_scales(_multi_tile_d(dt, xv, i0, ib, dw))
            pq = ntuple(c -> pointer(xv[c].xp, ib * QK_K + 1), Val(K))
            pm = ntuple(c -> Ptr{Int8}(pointer(xv[c].mo, ib * (QK_K >> 1) + 1)), Val(K))
            sums = _q6k_prep_multi_step(pointer(w, b + 1), pq, pm, sums, d)
        end
        for c in 1:K
            sumf = 0.0f0
            for l in 1:8
                sumf += sums[8*(c-1)+l].value
            end
            out[t0+c-1] = sumf
        end
    end
    return out
end

# ---------------------------------------------------------------- Q4_K and Q5_K
#
# A 32-value chunk is one sub-block with one scale, so the Q6_K pair order
# carries over: after it, vpmaddubsw then vpmaddwd against the sub-block scale
# leave in i32 slot l the scaled sum of the values with index l mod 8, which is
# lane l of _q4k_ints_avx2 and _q5k_ints_avx2. The weights go in unsigned (0..31)
# so a pair sums to at most 2*31*128 and cannot saturate, and every later step is
# exact, so each lane equals the single kernel's and the float steps repeat it.
# The pair order is a byte permutation and the nibble and high-bit extraction
# are bytewise, so the four qs loads (and the qh load) are reordered once per
# block, before the split, and shared by the K columns. A prepped column
# (Q6KPrepVec) brings its activations already in pair order; a plain Q8KVec
# column reorders each chunk itself. The mins term is the single kernel's own
# _kq_mins_dot per column.

function _kq_ir_q45k_multi(K::Int, q5::Bool, prep::Bool)
    io = IOBuffer()
    qargs = join(("ptr %q$c" for c in 0:(K-1)), ", ")
    qs = q5 ? 48 : 16
    print(
        io,
        "declare <8 x i32> @llvm.x86.avx2.permd(<8 x i32>, <8 x i32>)\n",
        "declare <32 x i8> @llvm.x86.avx2.pshuf.b(<32 x i8>, <32 x i8>)\n",
        "declare <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8>, <32 x i8>)\n",
        "declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>)\n",
        "define <$(8K) x float> @entry(ptr %w, $qargs, <8 x i16> %sc, <$(8K) x float> %s, <$(8K) x float> %d) #0 {\n",
    )
    for j in 0:3
        _kq_ld(io, "qr$j", "w", qs + 32 * j)
        _q6k_pair_order(io, "qp$j", "qr$j")
    end
    if q5
        _kq_ld(io, "hr", "w", 16)
        _q6k_pair_order(io, "hp", "hr")
    end
    acc = ["zeroinitializer" for _ in 0:(K-1)]
    for s in 0:7
        j = s >> 1
        if iseven(s)
            print(io, "  %nb$s = and <32 x i8> %qp$j, splat (i8 15)\n")
        else
            print(io, "  %nb$s = lshr <32 x i8> %qp$j, splat (i8 4)\n")
        end
        u = "%nb$s"
        if q5
            print(
                io,
                "  %hs$s = lshr <32 x i8> %hp, splat (i8 $s)\n",
                "  %hm$s = and <32 x i8> %hs$s, splat (i8 1)\n",
                "  %hb$s = shl <32 x i8> %hm$s, splat (i8 4)\n",
                "  %uv$s = or <32 x i8> %nb$s, %hb$s\n",
            )
            u = "%uv$s"
        end
        print(
            io,
            "  %sw$s = shufflevector <8 x i16> %sc, <8 x i16> poison, <16 x i32> <",
            join(("i32 $s" for _ in 1:16), ", "),
            ">\n",
        )
        for c in 0:(K-1)
            if prep
                _kq_ld(io, "xp$c.$s", "q$c", 32 * s)
            else
                _kq_ld(io, "x$c.$s", "q$c", 32 * s)
                _q6k_pair_order(io, "xp$c.$s", "x$c.$s")
            end
            print(
                io,
                "  %mu$c.$s = call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> $u, <32 x i8> %xp$c.$s)\n",
                "  %ms$c.$s = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %mu$c.$s, <16 x i16> %sw$s)\n",
                "  %A$c.$s = add <8 x i32> $(acc[c + 1]), %ms$c.$s\n",
            )
            acc[c+1] = "%A$c.$s"
        end
    end
    res = "poison"
    for c in 0:(K-1), e in 0:7
        print(
            io,
            "  %o$c.$e = extractelement <8 x i32> $(acc[c + 1]), i32 $e\n",
            "  %r$c.$e = insertelement <$(8K) x i32> $res, i32 %o$c.$e, i32 $(8c + e)\n",
        )
        res = "%r$c.$e"
    end
    print(
        io,
        "  %fa = sitofp <$(8K) x i32> $res to <$(8K) x float>\n",
        "  %fm = fmul <$(8K) x float> %d, %fa\n",
        "  %fs = fadd <$(8K) x float> %s, %fm\n",
        "  ret <$(8K) x float> %fs\n}\n",
        _KQ_ATTRS,
    )
    return String(take!(io))
end

for K in 1:8, q5 in (false, true), prep in (false, true)
    qs = [:(pq[$c]) for c in 1:K]
    V = NTuple{8K, VecElement{Float32}}
    S = NTuple{8, VecElement{Int16}}
    argt = Tuple{Ptr{UInt8}, ntuple(_ -> Ptr{Int8}, K)..., S, V, V}
    @eval @inline _q45k_multi_step(
        pw::Ptr{UInt8},
        pq::NTuple{$K, Ptr{Int8}},
        sc::$S,
        s::$V,
        d::$V,
        ::Val{$q5},
        ::Val{$prep},
    ) = Base.llvmcall(
        ($(_kq_ir_q45k_multi(K, q5, prep)), "entry"),
        $V,
        $argt,
        pw,
        $(qs...),
        sc,
        s,
        d,
    )
end

# The activation chunk pointer and the Q8_K source of a column.
@inline _q45k_xptr(x::Q8KVec, ib::Int) = pointer(x.qs, ib * QK_K + 1)
@inline _q45k_xptr(x::Q6KPrepVec, ib::Int) = pointer(x.xp, ib * QK_K + 1)
@inline _q45k_src(x::Q8KVec) = x
@inline _q45k_src(x::Q6KPrepVec) = x.q

@noinline function _dot_q45k_multi!(
    out::AbstractVector{Float32},
    w::AbstractVector{UInt8},
    woff::Int,
    xqs::AbstractVector{T},
    t0::Int,
    ::Val{K},
    q5::Val{Q5},
    dt=nothing,
) where {K, Q5, T <: Union{Q8KVec, Q6KPrepVec}}
    bs = Q5 ? 176 : 144
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, bs * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    i0 = _multi_d_base(dt, t0, K, nb)
    prep = Val(T === Q6KPrepVec)
    GC.@preserve w xqs begin
        xv = ntuple(c -> xqs[t0+c-1], Val(K))
        sums = ntuple(_ -> VecElement(0.0f0), Val(8K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb-1)
            b = woff + ib * bs
            d = _multi_q6k_scales(_multi_tile_d(dt, xv, i0, ib, _kq_f16(w, b)))
            dmin = _multi_tile_d(dt, xv, i0, ib, _kq_f16(w, b + 2))
            sc, mn = _kq_scale_min_k4(w, b + 4)
            pq = ntuple(c -> _q45k_xptr(xv[c], ib), Val(K))
            scv = ntuple(i -> VecElement(Int16(sc[i])), Val(8))
            sums = _q45k_multi_step(pointer(w, b + 1), pq, scv, sums, d, q5, prep)
            sf = sumf
            sumf = ntuple(
                c -> sf[c] - dmin[c] * Float32(_kq_mins_dot(_q45k_src(xv[c]), ib, mn)),
                Val(K),
            )
        end
        for c in 1:K
            s = sumf[c]
            for l in 1:8
                s += sums[8*(c-1)+l].value
            end
            out[t0+c-1] = s
        end
    end
    return out
end

# ---------------------------------------------------------------- Q8_0
#
# The single kernel reads one integer per 32-value block, so only the total has
# to match. The sign-extended weights are shared; each column runs two vpmaddwd
# against its own activations. A phadd tree then transposes the K lane vectors
# into one vector of K block totals (integer adds, so exact in any order), and
# the float step Float32(sumi) * (dw * x.d) + sumf runs on all K columns at once,
# unfused, as the single kernel does it per column.
#
# With q4 the weights are Q4_0 nibbles and the float step is the Q4_0 one,
# (Float32(sumi) * dw) * x.d + sumf, with dw and x.d passed as two vectors.

function _kq_ir_q80_multi(K::Int, q4::Bool=false)
    io = IOBuffer()
    qargs = join(("ptr %q$c" for c in 0:(K-1)), ", ")
    dargs = q4 ? "<$K x float> %dw, <$K x float> %d" : "<$K x float> %d"
    print(
        io,
        "declare <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16>, <16 x i16>)\n",
        "declare <8 x i32> @llvm.x86.avx2.phadd.d(<8 x i32>, <8 x i32>)\n",
        "define <$K x float> @entry(ptr %w, $qargs, <$K x float> %s, $dargs) #0 {\n",
    )
    if q4
        _kq_q40_weights(io)
    else
        _kq_ld(io, "wv", "w", 2)
        _kq_half(io, "wl", "wv", 0)
        _kq_half(io, "wh", "wv", 1)
        print(
            io,
            "  %wl16 = sext <16 x i8> %wl to <16 x i16>\n",
            "  %wh16 = sext <16 x i8> %wh to <16 x i16>\n",
        )
    end
    for c in 0:(K-1)
        _kq_ld(io, "x$c", "q$c", 0)
        _kq_half(io, "xl$c", "x$c", 0)
        _kq_half(io, "xh$c", "x$c", 1)
        print(
            io,
            "  %xl16.$c = sext <16 x i8> %xl$c to <16 x i16>\n",
            "  %xh16.$c = sext <16 x i8> %xh$c to <16 x i16>\n",
            "  %pl$c = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %wl16, <16 x i16> %xl16.$c)\n",
            "  %ph$c = call <8 x i32> @llvm.x86.avx2.pmadd.wd(<16 x i16> %wh16, <16 x i16> %xh16.$c)\n",
            "  %p$c = add <8 x i32> %pl$c, %ph$c\n",
        )
    end
    p(c) = c < K ? "%p$c" : "zeroinitializer"
    hadd(dst, a, b) = print(
        io,
        "  %$dst = call <8 x i32> @llvm.x86.avx2.phadd.d(<8 x i32> $a, <8 x i32> $b)\n",
    )
    # h_j lanes: [p2j 01, p2j 23, p2j+1 01, p2j+1 23 | the same for 45, 67].
    nh = cld(K, 2)
    for j in 0:(nh-1)
        hadd("h$j", p(2j), p(2j + 1))
    end
    h(j) = j < nh ? "%h$j" : "zeroinitializer"
    # g lanes: [p0..p3 low-half totals | p0..p3 high-half totals], g1 for p4..p7.
    hadd("g0", h(0), h(1))
    K > 4 && hadd("g1", h(2), h(3))
    if K > 4
        print(
            io,
            "  %lo = shufflevector <8 x i32> %g0, <8 x i32> %g1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>\n",
            "  %hi = shufflevector <8 x i32> %g0, <8 x i32> %g1, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>\n",
            "  %t = add <8 x i32> %lo, %hi\n",
        )
        n = 8
    else
        print(
            io,
            "  %lo = shufflevector <8 x i32> %g0, <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>\n",
            "  %hi = shufflevector <8 x i32> %g0, <8 x i32> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>\n",
            "  %t = add <4 x i32> %lo, %hi\n",
        )
        n = 4
    end
    print(
        io,
        "  %tk = shufflevector <$n x i32> %t, <$n x i32> poison, <$K x i32> <",
        join(("i32 $c" for c in 0:(K-1)), ", "),
        ">\n",
        "  %fa = sitofp <$K x i32> %tk to <$K x float>\n",
        q4 ? "  %fw = fmul <$K x float> %fa, %dw\n  %fm = fmul <$K x float> %fw, %d\n" :
        "  %fm = fmul <$K x float> %fa, %d\n",
        "  %fs = fadd <$K x float> %s, %fm\n",
        "  ret <$K x float> %fs\n}\n",
        _KQ_ATTRS,
    )
    return String(take!(io))
end

# One block for all K columns: s + Float32(sumi_c) * d_c per column.
for K in 1:8
    qs = [:(pq[$c]) for c in 1:K]
    V = NTuple{K, VecElement{Float32}}
    argt = Tuple{Ptr{UInt8}, ntuple(_ -> Ptr{Int8}, K)..., V, V}
    @eval @inline _q80_multi_step(pw::Ptr{UInt8}, pq::NTuple{$K, Ptr{Int8}}, s::$V, d::$V) =
        Base.llvmcall(($(_kq_ir_q80_multi(K)), "entry"), $V, $argt, pw, $(qs...), s, d)
    argt4 = Tuple{Ptr{UInt8}, ntuple(_ -> Ptr{Int8}, K)..., V, V, V}
    @eval @inline _q40_multi_step(
        pw::Ptr{UInt8},
        pq::NTuple{$K, Ptr{Int8}},
        s::$V,
        dw::$V,
        d::$V,
    ) = Base.llvmcall(
        ($(_kq_ir_q80_multi(K, true)), "entry"),
        $V,
        $argt4,
        pw,
        $(qs...),
        s,
        dw,
        d,
    )
end

# Q4_0 tile: the d table holds each column's x.d, and 1.0f0 * x.d is exact, so
# _multi_tile_d with dw = 1 reads it back unchanged.
@noinline function _dot_q40_multi!(
    out::AbstractVector{Float32},
    w::AbstractVector{UInt8},
    woff::Int,
    xqs,
    t0::Int,
    ::Val{K},
    dt=nothing,
) where {K}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, 18 * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    i0 = _multi_d_base(dt, t0, K, nb)
    GC.@preserve w xqs begin
        xv = ntuple(c -> xqs[t0+c-1], Val(K))
        sumf = ntuple(_ -> VecElement(0.0f0), Val(K))
        @inbounds for ib in 0:(nb-1)
            b = woff + ib * 18
            dw = ntuple(_ -> VecElement(_kq_f16(w, b)), Val(K))
            d = map(VecElement, _multi_tile_d(dt, xv, i0, ib, 1.0f0))
            sumf = _q40_multi_step(
                pointer(w, b + 1),
                ntuple(c -> pointer(xv[c].qs, 32 * ib + 1), Val(K)),
                sumf,
                dw,
                d,
            )
        end
        for c in 1:K
            out[t0+c-1] = sumf[c].value
        end
    end
    return out
end

@noinline function _dot_q80_multi!(
    out::AbstractVector{Float32},
    w::AbstractVector{UInt8},
    woff::Int,
    xqs,
    t0::Int,
    ::Val{K},
    dt=nothing,
) where {K}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, 34 * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    i0 = _multi_d_base(dt, t0, K, nb)
    GC.@preserve w xqs begin
        xv = ntuple(c -> xqs[t0+c-1], Val(K))
        sumf = ntuple(_ -> VecElement(0.0f0), Val(K))
        @inbounds for ib in 0:(nb-1)
            b = woff + ib * 34
            dw = _kq_f16(w, b)
            d = map(VecElement, _multi_tile_d(dt, xv, i0, ib, dw))
            sumf = _q80_multi_step(
                pointer(w, b + 1),
                ntuple(c -> pointer(xv[c].qs, 32 * ib + 1), Val(K)),
                sumf,
                d,
            )
        end
        for c in 1:K
            out[t0+c-1] = sumf[c].value
        end
    end
    return out
end

# ---------------------------------------------------------------- dispatch

# True for the types with a multi-column kernel. Every other type takes the
# per-column loop in _dot_q8k_multi_tile!.
_q8k_multi_type(t::Int32) =
    t == GGML_TYPE_IQ2_XXS ||
    t == GGML_TYPE_IQ3_S ||
    t == GGML_TYPE_IQ2_S ||
    t == GGML_TYPE_IQ2_XS ||
    t == GGML_TYPE_IQ3_XXS ||
    t == GGML_TYPE_IQ1_S ||
    t == GGML_TYPE_IQ1_M ||
    t == GGML_TYPE_IQ4_XS ||
    t == GGML_TYPE_Q2_K ||
    t == GGML_TYPE_Q4_K ||
    t == GGML_TYPE_Q5_K ||
    t == GGML_TYPE_Q6_K ||
    t == GGML_TYPE_Q8_0 ||
    t == GGML_TYPE_Q4_0

function _dot_q8k_multi_kernel!(
    out::AbstractVector{Float32},
    ggml_type::Int32,
    w::AbstractVector{UInt8},
    woff::Int,
    xqs,
    t0::Int,
    v::Val,
    dt=nothing,
)
    ggml_type == GGML_TYPE_IQ2_XXS && return _dot_iq2xxs_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_IQ3_S && return _dot_iq3s_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_IQ2_S && return _dot_iq2s_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_IQ2_XS && return _dot_iq2xs_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_IQ3_XXS && return _dot_iq3xxs_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_IQ1_S && return _dot_iq1s_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_IQ1_M && return _dot_iq1m_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_Q2_K && return _dot_q2k_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_Q4_K && return _dot_q45k_multi!(out, w, woff, xqs, t0, v, Val(false), dt)
    ggml_type == GGML_TYPE_Q5_K && return _dot_q45k_multi!(out, w, woff, xqs, t0, v, Val(true), dt)
    ggml_type == GGML_TYPE_Q6_K && return _dot_q6k_multi!(out, w, woff, xqs, t0, v, dt)
    ggml_type == GGML_TYPE_Q8_0 && return _dot_q80_multi!(out, w, woff, xqs, t0, v, dt)
    ggml_type == GGML_TYPE_Q4_0 && return _dot_q40_multi!(out, w, woff, xqs, t0, v, dt)
    return _dot_iq4_xs_multi!(out, w, woff, xqs, t0, v)
end

# Writes out[t0 + c - 1] = _dot_q8k(ggml_type, w, woff, xqs[t0 + c - 1], simd)
# for the K columns of the tile starting at t0. The AVX2 multi kernels are used
# only when simd is true, and the per-column loop otherwise.
function _dot_q8k_multi_tile!(
    out::AbstractVector{Float32},
    ggml_type::Int32,
    w::AbstractVector{UInt8},
    woff::Int,
    xqs,
    t0::Int,
    v::Val{K},
    simd::Bool,
    dt=nothing,
) where {K}
    if simd && _q8k_multi_type(ggml_type)
        return _dot_q8k_multi_kernel!(out, ggml_type, w, woff, xqs, t0, v, dt)
    end
    for c in 1:K
        out[t0+c-1] = _dot_q8k(ggml_type, w, woff, xqs[t0+c-1], simd)
    end
    return out
end

"""
    _dot_q8k_multi!(out, ggml_type, w, woff, xqs, t0, ntile, simd, dt=nothing) -> out

Raw dots of the weight row at 0-based byte `woff` of `w` with the `ntile`
Q8_K activations `xqs[t0:t0+ntile-1]`, written to `out[t0:...]`. Column c is
bit-equal to `_dot_q8k(ggml_type, w, woff, xqs[t0+c-1], simd)`. `ntile` is
1..8 so that any remainder is a tile. `dt`, from `_multi_d_table(xqs)`, gives
the Q6_K and Q8_0 kernels each tile's d values as one load; with it, `t0` must
start one of the table's tiles and `ntile` be at most `_MULTI_TILE`.
"""
function _dot_q8k_multi!(
    out::AbstractVector{Float32},
    ggml_type::Int32,
    w::AbstractVector{UInt8},
    woff::Int,
    xqs,
    t0::Int,
    ntile::Int,
    simd::Bool,
    dt=nothing,
)
    1 <= ntile <= 8 || error("q8k multi: tile of $ntile columns (1..8)")
    t0 + ntile - 1 <= length(xqs) ||
        error("q8k multi: columns $t0:$(t0 + ntile - 1) exceed $(length(xqs))")
    ntile == 1 &&
        return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(1), simd, dt)
    ntile == 2 &&
        return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(2), simd, dt)
    ntile == 3 &&
        return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(3), simd, dt)
    ntile == 4 &&
        return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(4), simd, dt)
    ntile == 5 &&
        return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(5), simd, dt)
    ntile == 6 &&
        return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(6), simd, dt)
    ntile == 7 &&
        return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(7), simd, dt)
    return _dot_q8k_multi_tile!(out, ggml_type, w, woff, xqs, t0, Val(8), simd, dt)
end
