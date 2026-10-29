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

# Query rows t0:t0+7 of head h (columns qo+1:qo+hd of Q and O) against the
# keys A:B that any of the eight rows reads. P holds one 8-float column per
# key, lane i for row t0+i-1, from key A on; Qp holds the eight query rows
# interleaved the same way.
function _g4_attend_tile8!(O, Q, kc, vc, P, Qp, t0::Int, s0::Int, koff::Int, win::Int, qo::Int, hk::Int, hd::Int)
    A = (win > 0 ? max(1, s0 + t0 - win) : 1) - koff
    B = s0 + t0 + 6 - koff
    ks2, ks3 = stride(kc, 2), stride(kc, 3)
    vs2, vs3 = stride(vc, 2), stride(vc, 3)
    z = _v8splat(0.0f0)
    GC.@preserve P Qp kc vc begin
        pP = pointer(P) - 32 * (A - 1)
        pQ = pointer(Qp)
        pk = pointer(kc) + 4 * (hk - 1) * ks2
        pv = pointer(vc) + 4 * (hk - 1) * vs2
        @inbounds for j in 1:hd, i in 1:8
            Qp[8 * (j - 1) + i] = Q[t0 + i - 1, qo + j]
        end
        # each score sums over j in order from +0, as the row loop does
        @inbounds for ub in A:_G4_TILE_KEYS:B
            ue = min(ub + _G4_TILE_KEYS - 1, B)
            for u in ub:ue
                _v8store(pP + 32 * (u - 1), z)
            end
            for j in 1:hd
                qv = _v8load(pQ + 32 * (j - 1))
                kj = pk + 4 * ks3 * (j - 1)
                for u in ub:ue
                    pu = pP + 32 * (u - 1)
                    _v8store(pu, _v8muladd(_v8load(pu), qv, _v8splat(unsafe_load(kj, u))))
                end
            end
        end
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
            Base.Cartesian.@nexprs 8 d -> vp_d = pv + 4 * vs3 * (jb + d - 1)
            for u in A:B
                pu = _v8load(pP + 32 * (u - 1))
                Base.Cartesian.@nexprs 8 d -> acc_d = _v8muladd(acc_d, pu, _v8splat(unsafe_load(vp_d, u)))
            end
            Base.Cartesian.@nexprs 8 d -> for i in 1:8
                O[t0 + i - 1, qo + jb + d] = acc_d[i].value
            end
        end
        @inbounds for j in (jt + 1):hd
            acc = z
            vj = pv + 4 * vs3 * (j - 1)
            for u in A:B
                acc = _v8muladd(acc, _v8load(pP + 32 * (u - 1)), _v8splat(unsafe_load(vj, u)))
            end
            for i in 1:8
                O[t0 + i - 1, qo + j] = acc[i].value
            end
        end
    end
    return O
end
