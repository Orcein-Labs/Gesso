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
# Tiles hold at most _MULTI_TILE columns; a remainder of 1..3 columns runs as a
# smaller tile, so any token count works.
#
# IQ4_XS is the exception to shared unpacking: its nibble-to-table lookup lives
# inside the llvmcall, so each column repeats that step. Its scale decode and
# the row pointers are still shared.

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

# Per-column block scale: the same product the single kernel forms, dw * x.d.
@inline _multi_d(dw::Float32, dv::NTuple{K,Vector{Float32}}, ib::Int) where {K} =
    ntuple(c -> dw * dv[c][ib + 1], Val(K))

# sumf[c] += d[c] * Float32(hsum(acc[c])), in block order as the single kernel.
@inline _multi_fadd(sumf::NTuple{K,Float32}, d::NTuple{K,Float32},
    acc::NTuple{K,NTuple{8,VecElement{Int32}}}) where {K} =
    ntuple(c -> sumf[c] + d[c] * Float32(_iq_hsum8(acc[c])), Val(K))

# The 16-lane IQ1 accumulators: lanes 1-8 and 9-16 are summed separately.
@inline function _iq1_pair(a::NTuple{16,VecElement{Int32}})
    s1 = a[1].value + a[2].value + a[3].value + a[4].value +
         a[5].value + a[6].value + a[7].value + a[8].value
    s2 = a[9].value + a[10].value + a[11].value + a[12].value +
         a[13].value + a[14].value + a[15].value + a[16].value
    return s1, s2
end

@inline _multi_fadd_pair(sumf::NTuple{K,Float32}, d::NTuple{K,Float32},
    acc::NTuple{K,NTuple{16,VecElement{Int32}}}) where {K} =
    ntuple(c -> begin
        s1, s2 = _iq1_pair(acc[c])
        sumf[c] + d[c] * (Float32(s1) + 0.125f0 * Float32(s2))
    end, Val(K))

@inline _multi_scale(sumf::NTuple{K,Float32}, s::Float32) where {K} =
    ntuple(c -> s * sumf[c], Val(K))

@inline _multi_iq2_step(acc::NTuple{K,NTuple{8,VecElement{Int32}}}, g, s,
    pq::NTuple{K,Ptr{Int8}}, off::Int, ls1::Int32, ls2::Int32) where {K} =
    ntuple(c -> _iq2_sub_acc(acc[c], g, s, pq[c] + off, ls1, ls2), Val(K))

@inline _multi_iq3_step(acc::NTuple{K,NTuple{8,VecElement{Int32}}}, g, s,
    pq::NTuple{K,Ptr{Int8}}, off::Int, ls::Int32) where {K} =
    ntuple(c -> _iq3_sub_acc(acc[c], g, s, pq[c] + off, ls), Val(K))

@inline _multi_iq1m_step(acc::NTuple{K,NTuple{16,VecElement{Int32}}}, g,
    pq::NTuple{K,Ptr{Int8}}, off::Int, wt::NTuple{16,VecElement{Int32}}) where {K} =
    ntuple(c -> _iq1m_sub_acc(acc[c], g, pq[c] + off, wt), Val(K))

@inline _multi_iq4_t(pq::NTuple{K,Ptr{Int8}}, qs::Ptr{UInt8}, off::Int) where {K} =
    ntuple(c -> _iq4_sub_avx2(_Q8K_KV_IQ4NL_X2, qs, pq[c] + off), Val(K))

@inline _multi_iq4_acc(sumf::NTuple{K,Float32}, d4d8::NTuple{K,Float32}, ls::Int32,
    t::NTuple{K,Int32}) where {K} =
    ntuple(c -> sumf[c] + (d4d8[c] * Float32(ls)) * Float32(t[c]), Val(K))

# Column count of a tile after checking that every column has the row's block
# count; a mismatch would read past the end of a shorter activation.
function _multi_check(xqs, t0::Int, K::Int, nb::Int)
    for c in 1:K
        length(xqs[t0 + c - 1].d) == nb ||
            error("q8k multi: activation $(t0 + c - 1) has $(length(xqs[t0 + c - 1].d)) blocks, row has $nb")
    end
    return nothing
end

function _multi_zero!(out::AbstractVector{Float32}, t0::Int, K::Int)
    for c in 1:K
        out[t0 + c - 1] = 0.0f0
    end
    return out
end

# ---------------------------------------------------------------- 8-lane families
#
# Shared drivers for the IQ2 and IQ3 layouts. `subops(pg, ps, b, k)` returns the
# operands of sub-block k (0..7) and its scale. Sub-blocks run in order 0..7 as
# in the single kernels, so each column's accumulator sees the same sequence.

# IQ2 sub-block operands: (g, s, ls1, ls2).
@noinline function _iq2_multi_core!(out::AbstractVector{Float32}, w::AbstractVector{UInt8},
    woff::Int, xqs, t0::Int, v::Val{K}, rb::Int, grid, signs, subops::F, scale::Float32) where {K,F}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, rb * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    GC.@preserve w xqs grid signs begin
        pw = pointer(w, woff + 1)
        pg = pointer(grid)
        ps = pointer(signs)
        pq = ntuple(c -> pointer(xqs[t0 + c - 1].qs), Val(K))
        dv = ntuple(c -> xqs[t0 + c - 1].d, Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb - 1)
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
            out[t0 + c - 1] = res[c]
        end
    end
    return out
end

# IQ3 sub-block operands: (g, s, ls). `scale` is the format's final factor
# (0.25 for IQ3_XXS, 1 for IQ3_S, which has none).
@noinline function _iq3_multi_core!(out::AbstractVector{Float32}, w::AbstractVector{UInt8},
    woff::Int, xqs, t0::Int, v::Val{K}, rb::Int, grid, signs, subops::F, scale::Float32) where {K,F}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, rb * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    GC.@preserve w xqs grid signs begin
        pw = pointer(w, woff + 1)
        pg = pointer(grid)
        ps = pointer(signs)
        pq = ntuple(c -> pointer(xqs[t0 + c - 1].qs), Val(K))
        dv = ntuple(c -> xqs[t0 + c - 1].d, Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb - 1)
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
            out[t0 + c - 1] = res[c]
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

@noinline _dot_iq2xxs_multi!(out, w, woff, xqs, t0, v::Val) =
    _iq2_multi_core!(out, w, woff, xqs, t0, v, 66, IQ2XXS_GRID, _IQ_SIGN8_TAB, _iq2xxs_sub_ops, 0.125f0)

@noinline _dot_iq2s_multi!(out, w, woff, xqs, t0, v::Val) =
    _iq2_multi_core!(out, w, woff, xqs, t0, v, 82, IQ2S_GRID, _IQ_SIGN8_FULL_TAB, _iq2s_sub_ops, 0.125f0)

@noinline _dot_iq2xs_multi!(out, w, woff, xqs, t0, v::Val) =
    _iq2_multi_core!(out, w, woff, xqs, t0, v, 74, IQ2XS_GRID, _IQ_SIGN8_TAB, _iq2xs_sub_ops, 0.125f0)

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

@noinline _dot_iq3s_multi!(out, w, woff, xqs, t0, v::Val) =
    _iq3_multi_core!(out, w, woff, xqs, t0, v, 110, IQ3S_GRID, _IQ_SIGN8_FULL_TAB, _iq3s_sub_ops, 1.0f0)

@noinline _dot_iq3xxs_multi!(out, w, woff, xqs, t0, v::Val) =
    _iq3_multi_core!(out, w, woff, xqs, t0, v, 98, IQ3XXS_GRID, _IQ_SIGN8_TAB, _iq3xxs_sub_ops, 0.25f0)

# ---------------------------------------------------------------- IQ1 (16-lane)

# IQ1_S: one scale per block, each sub-block's sign rides on the quant sums.
@noinline function _dot_iq1s_multi!(out::AbstractVector{Float32}, w::AbstractVector{UInt8},
    woff::Int, xqs, t0::Int, v::Val{K}) where {K}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, 50 * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    GC.@preserve w xqs begin
        pw = pointer(w, woff + 1)
        pg = pointer(IQ1S_GRID)
        pq = ntuple(c -> pointer(xqs[t0 + c - 1].qs), Val(K))
        dv = ntuple(c -> xqs[t0 + c - 1].d, Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb - 1)
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
                    VecElement(ls), VecElement(ls), VecElement(ls), VecElement(ls),
                    VecElement(ls), VecElement(ls), VecElement(ls), VecElement(ls),
                    VecElement(ld), VecElement(ld), VecElement(ld), VecElement(ld),
                    VecElement(ld), VecElement(ld), VecElement(ld), VecElement(ld),
                )
                acc = _multi_iq1m_step(acc, g, pq, ib * QK_K + 32 * s, wt)
            end
            sumf = _multi_fadd_pair(sumf, _multi_d(dw, dv, ib), acc)
        end
        for c in 1:K
            out[t0 + c - 1] = sumf[c]
        end
    end
    return out
end

# IQ1_M: the block scale is four 3-bit sub-scales packed in the scale words.
@noinline function _dot_iq1m_multi!(out::AbstractVector{Float32}, w::AbstractVector{UInt8},
    woff::Int, xqs, t0::Int, v::Val{K}) where {K}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, 56 * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    GC.@preserve w xqs begin
        pw = pointer(w, woff + 1)
        pg = pointer(IQ1S_GRID)
        pq = ntuple(c -> pointer(xqs[t0 + c - 1].qs), Val(K))
        dv = ntuple(c -> xqs[t0 + c - 1].d, Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb - 1)
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
                    VecElement(ls1), VecElement(ls1), VecElement(ls1), VecElement(ls1),
                    VecElement(ls2), VecElement(ls2), VecElement(ls2), VecElement(ls2),
                    VecElement(e0), VecElement(e0), VecElement(e1), VecElement(e1),
                    VecElement(e2), VecElement(e2), VecElement(e3), VecElement(e3),
                )
                acc = _multi_iq1m_step(acc, g, pq, ib * QK_K + 32 * s, wt)
            end
            sumf = _multi_fadd_pair(sumf, _multi_d(dw, dv, ib), acc)
        end
        for c in 1:K
            out[t0 + c - 1] = sumf[c]
        end
    end
    return out
end

# ---------------------------------------------------------------- IQ4_XS

@noinline function _dot_iq4_xs_multi!(out::AbstractVector{Float32}, w::AbstractVector{UInt8},
    woff::Int, xqs, t0::Int, v::Val{K}) where {K}
    nb = length(xqs[t0].d)
    _q8k_check_row(w, woff, 136 * nb)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    GC.@preserve w xqs begin
        pq = ntuple(c -> pointer(xqs[t0 + c - 1].qs), Val(K))
        dv = ntuple(c -> xqs[t0 + c - 1].d, Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb - 1)
            b = woff + ib * 136
            d4d8 = _multi_d(_ld_f16(w, b), dv, ib)
            h = _ld_u16(w, b + 2)
            ao = ib * QK_K
            for s in 0:7
                sl = _ld_u8(w, b + 4 + (s >> 1))
                ls = (Int32((sl >> (4 * (s & 1))) & 0x0f) | (Int32((h >> (2 * s)) & 0x0003) << 4)) -
                     Int32(32)
                qs = pointer(w, b + 9 + 16 * s)
                t = _multi_iq4_t(pq, qs, ao + 32 * s)
                sumf = _multi_iq4_acc(sumf, d4d8, ls, t)
            end
        end
        for c in 1:K
            out[t0 + c - 1] = sumf[c]
        end
    end
    return out
end

# ---------------------------------------------------------------- K-quants (Q4_K, Q2_K)
#
# The integer lane sums are exact, so their order does not matter. The float
# steps follow _dot_q4k_q8k and _dot_q2k_q8k for each column: Q4_K folds each
# block into eight lane sums and subtracts the mins term block by block, and
# Q2_K adds dall*isum - dmin*summs. Each packed nibble or 2-bit value is
# unpacked once per tile and used by all K columns. Closures take their state
# as arguments, so no captured variable is reassigned in a loop.

@inline _q4_acc(acc::NTuple{K,Int32}, scs::Int32, nib::NTuple{4,Int32},
    pq::NTuple{K,Ptr{Int8}}, ko::Int) where {K} =
    ntuple(c -> acc[c] + scs * (nib[1] * Int32(unsafe_load(pq[c], ko)) +
        nib[2] * Int32(unsafe_load(pq[c], ko + 8)) +
        nib[3] * Int32(unsafe_load(pq[c], ko + 16)) +
        nib[4] * Int32(unsafe_load(pq[c], ko + 24))), Val(K))

@inline _q4_nibs(qsb::Ptr{UInt8}, bo::Int, sh::Int) =
    ntuple(j -> Int32((unsafe_load(qsb, bo + 8 * (j - 1)) >> sh) & 0x0f), Val(4))

# Lane l (0-based) of one Q4_K block for every column: sum over sub-blocks s of
# sc[s] * sum_j nib(k) * q8_c[k], with k = 32s + l + 8j.
@inline function _q4_lane_cols(pq::NTuple{K,Ptr{Int8}}, qsb::Ptr{UInt8}, l::Int,
    sc::NTuple{8,Int32}) where {K}
    acc = ntuple(_ -> Int32(0), Val(K))
    for s in 0:7
        nib = _q4_nibs(qsb, 32 * (s >> 1) + l + 1, 4 * (s & 1))
        acc = _q4_acc(acc, sc[s + 1], nib, pq, 32 * s + l + 1)
    end
    return acc
end

# All eight lanes for every column: NTuple{8,NTuple{K,Int32}}.
@inline _q4_block_aux(pq::NTuple{K,Ptr{Int8}}, qsb::Ptr{UInt8}, ib::Int, sc::NTuple{8,Int32}) where {K} =
    ntuple(l -> _q4_lane_cols(ntuple(c -> pq[c] + ib * QK_K, Val(K)), qsb, l - 1, sc), Val(8))

@inline _q4_fold(sums::NTuple{K,NTuple{8,Float32}}, d::NTuple{K,Float32},
    aux::NTuple{8,NTuple{K,Int32}}) where {K} =
    ntuple(c -> _kq_fsum(sums[c], d[c], ntuple(l -> aux[l][c], Val(8))), Val(K))

@inline _q4_mins(xs::NTuple{K,Q8KVec}, ib::Int, mn::NTuple{8,Int32}) where {K} =
    ntuple(c -> _kq_mins_dot(xs[c], ib, mn), Val(K))

@inline _q4_scaled(w, off::Int, dv::NTuple{K,Vector{Float32}}, ib::Int) where {K} =
    ntuple(c -> _kq_f16(w, off) * dv[c][ib + 1], Val(K))

@inline _q4_sub_mins(sumf::NTuple{K,Float32}, dmin::NTuple{K,Float32}, mm::NTuple{K,Int32}) where {K} =
    ntuple(c -> sumf[c] - dmin[c] * Float32(mm[c]), Val(K))

@inline _q4_final(sumf::NTuple{K,Float32}, sums::NTuple{K,NTuple{8,Float32}}) where {K} =
    ntuple(c -> begin
        s = sumf[c]
        for l in 1:8
            s += sums[c][l]
        end
        s
    end, Val(K))

@noinline function _dot_q4k_multi!(out::AbstractVector{Float32}, w::AbstractVector{UInt8},
    woff::Int, xqs, t0::Int, v::Val{K}) where {K}
    nb = length(xqs[t0].d)
    _kq_check_row(w, woff, 144 * nb, nothing)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    xs = ntuple(c -> xqs[t0 + c - 1], Val(K))
    dv = ntuple(c -> xs[c].d, Val(K))
    GC.@preserve w xqs begin
        pw = pointer(w, woff + 1)
        pq = ntuple(c -> pointer(xs[c].qs), Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        sums = ntuple(_ -> ntuple(_ -> 0.0f0, Val(8)), Val(K))
        @inbounds for ib in 0:(nb - 1)
            b = woff + 144 * ib
            sc, mn = _kq_scale_min_k4(w, b + 4)
            aux = _q4_block_aux(pq, pw + 144 * ib + 16, ib, sc)
            d = _q4_scaled(w, b, dv, ib)
            dmin = _q4_scaled(w, b + 2, dv, ib)
            mm = _q4_mins(xs, ib, mn)
            sums = _q4_fold(sums, d, aux)
            sumf = _q4_sub_mins(sumf, dmin, mm)
        end
        res = _q4_final(sumf, sums)
        for c in 1:K
            out[t0 + c - 1] = res[c]
        end
    end
    return out
end

# Q2_K: the 2-bit value of k = 16j + i lives in byte 32h + l of qs, shift 2t,
# with h = k >> 7, t = (k & 127) >> 5, l = k & 31. Sub-block j is scaled by the
# low nibble of scale byte j.
@inline _q2_acc(sub::NTuple{K,Int32}, v::Int32, pq::NTuple{K,Ptr{Int8}}, kk::Int) where {K} =
    ntuple(c -> sub[c] + v * Int32(unsafe_load(pq[c], kk)), Val(K))

@inline _q2_add(total::NTuple{K,Int32}, sc::Int32, sub::NTuple{K,Int32}) where {K} =
    ntuple(c -> total[c] + sc * sub[c], Val(K))

@inline function _q2_sub_cols(pq::NTuple{K,Ptr{Int8}}, qsb::Ptr{UInt8}, j::Int) where {K}
    sub = ntuple(_ -> Int32(0), Val(K))
    for i in 0:15
        k = 16 * j + i
        h = k >> 7
        r = k & 127
        t = r >> 5
        l = r & 31
        v = Int32((unsafe_load(qsb, 32 * h + l + 1) >> (2 * t)) & 0x03)
        sub = _q2_acc(sub, v, pq, k + 1)
    end
    return sub
end

@inline function _q2_block_isum(pq::NTuple{K,Ptr{Int8}}, ib::Int, qsb::Ptr{UInt8}, sb::Ptr{UInt8}) where {K}
    pb = ntuple(c -> pq[c] + ib * QK_K, Val(K))
    total = ntuple(_ -> Int32(0), Val(K))
    for j in 0:15
        sc = Int32(unsafe_load(sb, j + 1) & 0x0f)
        total = _q2_add(total, sc, _q2_sub_cols(pb, qsb, j))
    end
    return total
end

@inline function _q2_summs(w, b::Int, bs::Vector{Int16}, ib::Int)::Int32
    s = Int32(0)
    @inbounds for j in 0:15
        s += Int32(bs[16 * ib + j + 1]) * Int32(_kq_u8(w, b + j) >> 4)
    end
    return s
end

@inline _q2_summs_cols(w, b::Int, xs::NTuple{K,Q8KVec}, ib::Int) where {K} =
    ntuple(c -> _q2_summs(w, b, xs[c].bsums, ib), Val(K))

@inline _q2_fold(sumf::NTuple{K,Float32}, dall::NTuple{K,Float32}, dmin::NTuple{K,Float32},
    isum::NTuple{K,Int32}, summs::NTuple{K,Int32}) where {K} =
    ntuple(c -> sumf[c] + (dall[c] * Float32(isum[c]) - dmin[c] * Float32(summs[c])), Val(K))

@noinline function _dot_q2k_multi!(out::AbstractVector{Float32}, w::AbstractVector{UInt8},
    woff::Int, xqs, t0::Int, v::Val{K}) where {K}
    nb = length(xqs[t0].d)
    _kq_check_row(w, woff, 84 * nb, nothing)
    _multi_check(xqs, t0, K, nb)
    nb == 0 && return _multi_zero!(out, t0, K)
    xs = ntuple(c -> xqs[t0 + c - 1], Val(K))
    dv = ntuple(c -> xs[c].d, Val(K))
    GC.@preserve w xqs begin
        pw = pointer(w, woff + 1)
        pq = ntuple(c -> pointer(xs[c].qs), Val(K))
        sumf = ntuple(_ -> 0.0f0, Val(K))
        @inbounds for ib in 0:(nb - 1)
            b = woff + 84 * ib
            dall = _q4_scaled(w, b + 80, dv, ib)
            dmin = _q4_scaled(w, b + 82, dv, ib)
            isum = _q2_block_isum(pq, ib, pw + 84 * ib + 16, pw + 84 * ib)
            summs = _q2_summs_cols(w, b, xs, ib)
            sumf = _q2_fold(sumf, dall, dmin, isum, summs)
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
_q8k_multi_type(t::Int32) =
    t == GGML_TYPE_IQ2_XXS || t == GGML_TYPE_IQ3_S || t == GGML_TYPE_IQ2_S ||
    t == GGML_TYPE_IQ2_XS || t == GGML_TYPE_IQ3_XXS || t == GGML_TYPE_IQ1_S ||
    t == GGML_TYPE_IQ1_M || t == GGML_TYPE_IQ4_XS || t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q4_K

function _dot_q8k_multi_kernel!(out::AbstractVector{Float32}, ggml_type::Int32,
    w::AbstractVector{UInt8}, woff::Int, xqs, t0::Int, v::Val)
    ggml_type == GGML_TYPE_Q4_K && return _dot_q4k_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_Q2_K && return _dot_q2k_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_IQ2_XXS && return _dot_iq2xxs_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_IQ3_S && return _dot_iq3s_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_IQ2_S && return _dot_iq2s_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_IQ2_XS && return _dot_iq2xs_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_IQ3_XXS && return _dot_iq3xxs_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_IQ1_S && return _dot_iq1s_multi!(out, w, woff, xqs, t0, v)
    ggml_type == GGML_TYPE_IQ1_M && return _dot_iq1m_multi!(out, w, woff, xqs, t0, v)
    return _dot_iq4_xs_multi!(out, w, woff, xqs, t0, v)
end

# Writes out[t0 + c - 1] = _dot_q8k(ggml_type, w, woff, xqs[t0 + c - 1], simd)
# for the K columns of the tile starting at t0. The AVX2 multi kernels are used
# only when simd is true, and the per-column loop otherwise.
function _dot_q8k_multi_tile!(out::AbstractVector{Float32}, ggml_type::Int32, w::AbstractVector{UInt8},
    woff::Int, xqs, t0::Int, v::Val{K}, simd::Bool) where {K}
    if simd && _q8k_multi_type(ggml_type)
        return _dot_q8k_multi_kernel!(out, ggml_type, w, woff, xqs, t0, v)
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
