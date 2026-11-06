# Eight-row attention tiles for Gemma 4 prefill. Each K and V element loaded
# from the cache serves eight query rows at once, one per vector lane, instead
# of one row. The arithmetic per lane is _g4_attend_head!'s, so the two paths
# agree bit for bit.

@inline _v8store(p::Ptr{Float32}, v::_V8) = Base.llvmcall(
    """
store <8 x float> %1, ptr %0, align 4
ret void""",
    Cvoid,
    Tuple{Ptr{Float32}, _V8},
    p,
    v,
)

# acc + a * b rounded twice, like the scalar loops (no fma contraction).
@inline _v8muladd(acc::_V8, a::_V8, b::_V8) = Base.llvmcall(
    """
%m = fmul <8 x float> %1, %2
%s = fadd <8 x float> %0, %m
ret <8 x float> %s""",
    _V8,
    Tuple{_V8, _V8, _V8},
    acc,
    a,
    b,
)

@inline _v8splat(x::Float32) = ntuple(_ -> VecElement(x), Val(8))

# Keys per score block: the block's 8-lane scores stay in L1 across the head dim.
const _G4_TILE_KEYS = 128

# Scores of keys A:B for the eight interleaved query rows at pQ, into the
# 8-float columns at pP + 32(u-1). Plain caches: one pass over each key block
# per head dim, the block's scores staying in L1.
@inline function _g4_tile8_scores!(pP, pQ, pk, mk, A::Int, B::Int, hk::Int, hd::Int)
    z = _v8splat(0.0f0)
    @inbounds for ub in A:_G4_TILE_KEYS:B
        ue = min(ub + _G4_TILE_KEYS - 1, B)
        for u in ub:ue
            _v8store(pP + 32 * (u - 1), z)
        end
        for j in 1:hd
            qv = _v8load(pQ + 32 * (j - 1))
            for u in ub:ue
                pu = pP + 32 * (u - 1)
                _v8store(pu, _v8muladd(_v8load(pu), qv, _v8splat(unsafe_load(pk, 1 + _g4_off(mk, u, hk, j)))))
            end
        end
    end
    return nothing
end

# Tiled keys: one 8-key tile is a contiguous 8 x hd run, so its eight score
# vectors stay in registers across the whole head dim. Same sums, same order.
@inline function _g4_tile8_scores!(pP, pQ, pk, mk::_G4TileMap, A::Int, B::Int, hk::Int, hd::Int)
    z = _v8splat(0.0f0)
    @inbounds for tb in ((A - 1) >> 3):((B - 1) >> 3)
        pt = pk + 4 * (((hk - 1) * mk.nt + tb) * mk.s3)
        Base.Cartesian.@nexprs 8 i -> s_i = z
        for j in 1:hd
            qv = _v8load(pQ + 32 * (j - 1))
            pj = pt + 32 * (j - 1)
            Base.Cartesian.@nexprs 8 i -> s_i = _v8muladd(s_i, qv, _v8splat(unsafe_load(pj + 4 * (i - 1))))
        end
        Base.Cartesian.@nexprs 8 i -> begin
            u_i = 8 * tb + i
            A <= u_i <= B && _v8store(pP + 32 * (u_i - 1), s_i)
        end
    end
    return nothing
end

# Query rows t0:t0+7 of head h (columns qo+1:qo+hd of Q and O) against the
# keys A:B that any of the eight rows reads. P holds one 8-float column per
# key, lane i for row t0+i-1, from key A on; Qp holds the eight query rows
# interleaved the same way.
function _g4_attend_tile8!(O, Q, kc, vc, P, Qp, t0::Int, s0::Int, koff::Int, win::Int, qo::Int, hk::Int, hd::Int)
    A = (win > 0 ? max(1, s0 + t0 - win) : 1) - koff
    B = s0 + t0 + 6 - koff
    z = _v8splat(0.0f0)
    GC.@preserve P Qp kc vc begin
        pP = pointer(P) - 32 * (A - 1)
        pQ = pointer(Qp)
        pk = _g4_ptr(kc)
        pv = _g4_ptr(vc)
        mk = _g4_map(kc)
        mv = _g4_map(vc)
        @inbounds for j in 1:hd, i in 1:8
            Qp[8 * (j - 1) + i] = Q[t0 + i - 1, qo + j]
        end
        # each score sums over j in order from +0, as the row loop does
        _g4_tile8_scores!(pP, pQ, pk, mk, A, B, hk, hd)
        @inbounds for i in 1:8
            pos = s0 + t0 + i - 2
            a = (win > 0 ? max(1, pos - win + 1) : 1) - koff
            b = pos - koff
            mx = P[8 * (a - A) + i]
            for u in (a + 1):b
                mx = max(mx, P[8 * (u - A) + i])
            end
            den = 0.0f0
            for u in a:b
                v = exp(P[8 * (u - A) + i] - mx)
                P[8 * (u - A) + i] = v
                den += v
            end
            inv_den = 1.0f0 / den
            for u in a:b
                P[8 * (u - A) + i] *= inv_den
            end
            # Keys outside the row's range get weight +0. Each output sum
            # starts at +0 and so never holds -0, which makes adding +0 * v exact.
            for u in A:(a - 1)
                P[8 * (u - A) + i] = 0.0f0
            end
            for u in (b + 1):B
                P[8 * (u - A) + i] = 0.0f0
            end
        end
        # eight output dims at a time, each summed over the keys in order
        jt = hd - hd % 8
        @inbounds for jb in 0:8:(jt - 8)
            Base.Cartesian.@nexprs 8 d -> acc_d = z
            for u in A:B
                pu = _v8load(pP + 32 * (u - 1))
                Base.Cartesian.@nexprs 8 d -> acc_d = _v8muladd(acc_d, pu, _v8splat(unsafe_load(pv, 1 + _g4_off(mv, u, hk, jb + d))))
            end
            Base.Cartesian.@nexprs 8 d -> for i in 1:8
                O[t0 + i - 1, qo + jb + d] = acc_d[i].value
            end
        end
        @inbounds for j in (jt + 1):hd
            acc = z
            for u in A:B
                acc = _v8muladd(acc, _v8load(pP + 32 * (u - 1)), _v8splat(unsafe_load(pv, 1 + _g4_off(mv, u, hk, j))))
            end
            for i in 1:8
                O[t0 + i - 1, qo + j] = acc[i].value
            end
        end
    end
    return O
end
