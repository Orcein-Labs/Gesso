# Gemma 4 CPU KV cache layouts for decode. The plain (rows, n_kv, hd) cache
# puts one key's hd values rows * n_kv * 4 bytes apart (64 KiB at 2048 rows),
# so a decode score pass touches a new page per head dim and the prefetcher
# never locks on. These two layouts keep what one decode pass reads contiguous:
#
#   G4KeyTiles   data[i, j, t, h] = K[8(t-1)+i, h, j]: eight keys per head-dim
#                column, so a score pass reads one 8-key tile as a straight run.
#   G4ValBlocks  data[d, u, b, h] = V[u, h, 32(b-1)+d]: 32 dims per key, so the
#                output pass streams one (block, head) slab.
#
# Both present the (rows, n_kv, hd) array interface, so the generic save,
# restore and row-write paths keep working; the Gemma 4 kernels dispatch on the
# concrete type. Every kernel keeps the plain loop's order of operations (each
# score sums over the head dim in order, each output over the keys in order,
# mul then add with no fma), so results are bit-identical to _g4_attend_head!.

# false allocates the plain (rows, n_kv, hd) caches instead (A/B timing, tests)
const _G4_KV_TILED = Ref(true)

struct G4KeyTiles <: AbstractArray{Float32, 3}
    data::Array{Float32, 4}   # (8, hd, cld(rows, 8), n_kv)
    rows::Int
end

struct G4ValBlocks <: AbstractArray{Float32, 3}
    data::Array{Float32, 4}   # (32, rows, hd ÷ 32, n_kv)
end

G4KeyTiles(rows::Int, nkv::Int, hd::Int) = G4KeyTiles(zeros(Float32, 8, hd, cld(rows, 8), nkv), rows)
function G4ValBlocks(rows::Int, nkv::Int, hd::Int)
    hd % 32 == 0 || throw(ArgumentError("G4ValBlocks: head dim $hd is not a multiple of 32"))
    return G4ValBlocks(zeros(Float32, 32, rows, hd ÷ 32, nkv))
end

Base.size(x::G4KeyTiles) = (x.rows, size(x.data, 4), size(x.data, 2))
Base.size(x::G4ValBlocks) = (size(x.data, 2), size(x.data, 4), 32 * size(x.data, 3))
Base.IndexStyle(::Type{<:Union{G4KeyTiles, G4ValBlocks}}) = IndexCartesian()

# 0-based element offsets of (key u, kv head h, dim j). A kernel takes the map
# once with _g4_map and indexes through it, so the loop holds the layout in
# registers instead of reloading the array's size through its raw-pointer stores.
struct _G4TileMap
    nt::Int   # tiles per kv head
    s3::Int   # floats per tile
end
struct _G4BlockMap
    nb::Int   # dim blocks per kv head
    s3::Int   # floats per (block, head) slab
end
struct _G4PlainMap
    s2::Int
    s3::Int
end
@inline _g4_map(x::G4KeyTiles) = _G4TileMap(size(x.data, 3), stride(x.data, 3))
@inline _g4_map(x::G4ValBlocks) = _G4BlockMap(size(x.data, 3), stride(x.data, 3))
# a plain cache: _g4_tile_cache has checked that dim 1 is contiguous
@inline _g4_map(x::AbstractArray{Float32, 3}) = _G4PlainMap(stride(x, 2), stride(x, 3))

@inline _g4_off(m::_G4TileMap, u::Int, h::Int, j::Int) =
    ((h - 1) * m.nt + ((u - 1) >> 3)) * m.s3 + 8 * (j - 1) + ((u - 1) & 7)
@inline _g4_off(m::_G4BlockMap, u::Int, h::Int, j::Int) =
    ((h - 1) * m.nb + ((j - 1) >> 5)) * m.s3 + 32 * (u - 1) + ((j - 1) & 31)
@inline _g4_off(m::_G4PlainMap, u::Int, h::Int, j::Int) = (u - 1) + (h - 1) * m.s2 + (j - 1) * m.s3
@inline _g4_off(x::AbstractArray{Float32, 3}, u::Int, h::Int, j::Int) = _g4_off(_g4_map(x), u, h, j)

Base.@propagate_inbounds function Base.getindex(x::Union{G4KeyTiles, G4ValBlocks}, u::Int, h::Int, j::Int)
    @boundscheck checkbounds(x, u, h, j)
    return @inbounds x.data[1 + _g4_off(x, u, h, j)]
end
Base.@propagate_inbounds function Base.setindex!(x::Union{G4KeyTiles, G4ValBlocks}, v, u::Int, h::Int, j::Int)
    @boundscheck checkbounds(x, u, h, j)
    @inbounds x.data[1 + _g4_off(x, u, h, j)] = v
    return x
end

_g4_ptr(x::Union{G4KeyTiles, G4ValBlocks}) = pointer(x.data)
_g4_ptr(x::AbstractArray{Float32, 3}) = pointer(x)

_g4_tile_cache(x::Union{G4KeyTiles, G4ValBlocks}) = true

Base.copyto!(dst::G4KeyTiles, src::G4KeyTiles) =
    (size(dst) == size(src) || throw(DimensionMismatch("kv tiles $(size(src)) vs $(size(dst))")); copyto!(dst.data, src.data); dst)
Base.copyto!(dst::G4ValBlocks, src::G4ValBlocks) =
    (size(dst) == size(src) || throw(DimensionMismatch("kv blocks $(size(src)) vs $(size(dst))")); copyto!(dst.data, src.data); dst)

# Rows row0:row0+N-1 from K (N, n_kv * hd), head h at columns (h-1)*hd+1 : h*hd.
function _g4_kv_write!(dst::Union{G4KeyTiles, G4ValBlocks}, src::AbstractMatrix{Float32}, row0::Int, nkv::Int, hd::Int)
    N = size(src, 1)
    size(src, 2) == nkv * hd || throw(DimensionMismatch("kv rows $(size(src)) vs $nkv x $hd"))
    1 <= row0 && row0 + N - 1 <= size(dst, 1) && size(dst, 2) == nkv && size(dst, 3) == hd ||
        throw(BoundsError(dst, (row0 + N - 1, nkv, hd)))
    d = dst.data
    m = _g4_map(dst)
    @inbounds for h in 1:nkv, j in 1:hd, t in 1:N
        d[1 + _g4_off(m, row0 + t - 1, h, j)] = src[t, (h - 1) * hd + j]
    end
    return dst
end

function _kv_write_rows!(dst::Union{G4KeyTiles, G4ValBlocks}, src::AbstractArray, row0::Int)
    n, nkv, hd = size(src)
    (size(dst, 2), size(dst, 3)) == (nkv, hd) || throw(DimensionMismatch("kv rows $(size(src)) vs $(size(dst))"))
    d = dst.data
    m = _g4_map(dst)
    @inbounds for h in 1:nkv, j in 1:hd, t in 1:n
        d[1 + _g4_off(m, row0 + t - 1, h, j)] = Float32(src[t, h, j])
    end
    return nothing
end

# Score lanes of key tile tl for head h: only keys a:b are kept.
@inline function _g4_store_tile!(P, acc::_V8, tl::Int, h::Int, a::Int, b::Int)
    u0 = 8 * (tl - 1)
    @inbounds for i in 1:8
        u = u0 + i
        a <= u <= b && (P[u, h] = acc[i].value)
    end
    return nothing
end

# Decode softmax over P[a:b, h], the plain loop's arithmetic.
@inline function _g4_softmax_col!(P, h::Int, a::Int, b::Int)
    @inbounds begin
        mx = P[a, h]
        for u in (a + 1):b
            mx = max(mx, P[u, h])
        end
        den = 0.0f0
        for u in a:b
            v = exp(P[u, h] - mx)
            P[u, h] = v
            den += v
        end
        inv_den = 1.0f0 / den
        for u in a:b
            P[u, h] *= inv_den
        end
    end
    return nothing
end

# Keys below this go through one fused task per query head; at and above it the
# work splits into score, softmax and output passes over (kv head, block) tasks,
# which read each K and V element once for all the heads sharing it.
const _G4_SPLIT_MIN_KEYS = Ref(256)

# One decode row (Q is 1 x nh*hd) against keys a:b of the tiled caches.
function _g4_attend_decode!(O, Q, kc::G4KeyTiles, vc::G4ValBlocks, P, a::Int, b::Int, nh::Int, nkv::Int, hd::Int)
    size(Q, 1) == 1 && stride(Q, 2) == 1 && stride(O, 2) == 1 || throw(ArgumentError("decode attention takes one contiguous row"))
    if b - a + 1 < _G4_SPLIT_MIN_KEYS[]
        _g4_attend_decode_fused!(O, Q, kc, vc, P, a, b, nh, nkv, hd)
    else
        _g4_qk_tiles!(P, Q, kc, a, b, nh, nkv, hd)
        _pool_run_chunks!(nh) do h
            _g4_softmax_col!(P, h, a, b)
        end
        _g4_pv_blocks!(O, vc, P, a, b, nh, nkv, hd)
    end
    return O
end

function _g4_attend_decode_fused!(O, Q, kc::G4KeyTiles, vc::G4ValBlocks, P, a::Int, b::Int, nh::Int, nkv::Int, hd::Int)
    rep = nh ÷ nkv
    ta = (a - 1) >> 3 + 1
    tb = (b - 1) >> 3 + 1
    ndb = hd >> 5
    _pool_run_chunks!(nh) do h
        hk = (h - 1) ÷ rep + 1
        z = _v8splat(0.0f0)
        GC.@preserve kc vc Q O P begin
            pq = pointer(Q) + 4 * (h - 1) * hd
            stl = 4 * stride(kc.data, 3)
            pk = pointer(kc.data) + stl * (hk - 1) * size(kc.data, 3)
            @inbounds for tl in ta:tb
                pt = pk + stl * (tl - 1)
                x = z
                for j in 1:hd
                    x = _v8muladd(x, _v8splat(unsafe_load(pq, j)), _v8load(pt + 32 * (j - 1)))
                end
                _g4_store_tile!(P, x, tl, h, a, b)
            end
            _g4_softmax_col!(P, h, a, b)
            sb = 4 * stride(vc.data, 3)
            pv = pointer(vc.data) + sb * (hk - 1) * size(vc.data, 3)
            @inbounds for db in 1:ndb
                pb = pv + sb * (db - 1)
                Base.Cartesian.@nexprs 4 d -> x_d = z
                for u in a:b
                    pu = _v8splat(P[u, h])
                    pr = pb + 128 * (u - 1)
                    Base.Cartesian.@nexprs 4 d -> x_d = _v8muladd(x_d, pu, _v8load(pr + 32 * (d - 1)))
                end
                qo = (h - 1) * hd + 32 * (db - 1)
                Base.Cartesian.@nexprs 4 d -> for i in 1:8
                    O[1, qo + 8 * (d - 1) + i] = x_d[i].value
                end
            end
        end
    end
    return O
end

# Scores P[u, h] for keys a:b. A task takes up to TB key tiles of one kv head
# and every query head that shares it, two heads at a time, four tiles at a time.
function _g4_qk_tiles!(P, Q, kc::G4KeyTiles, a::Int, b::Int, nh::Int, nkv::Int, hd::Int)
    rep = nh ÷ nkv
    ta = (a - 1) >> 3 + 1
    tb = (b - 1) >> 3 + 1
    ntl = tb - ta + 1
    TB = clamp(cld(cld(ntl * nkv, 2 * Threads.nthreads()), 4) * 4, 4, 64)
    nub = cld(ntl, TB)
    _pool_run_chunks!(nkv * nub) do t
        hk = (t - 1) % nkv + 1
        t0 = ta + ((t - 1) ÷ nkv) * TB
        t1 = min(t0 + TB - 1, tb)
        h0 = (hk - 1) * rep
        z = _v8splat(0.0f0)
        GC.@preserve kc Q P begin
            stl = 4 * stride(kc.data, 3)
            base = pointer(kc.data) + stl * (hk - 1) * size(kc.data, 3)
            pq = pointer(Q)
            @inbounds for rp in 1:2:(rep - 1)
                ha = h0 + rp
                qa = pq + 4 * (ha - 1) * hd
                qb = qa + 4 * hd
                tl = t0
                while tl + 3 <= t1
                    pt = base + stl * (tl - 1)
                    Base.Cartesian.@nexprs 4 c -> (x_c = z; y_c = z)
                    for j in 1:hd
                        sa = _v8splat(unsafe_load(qa, j))
                        sb = _v8splat(unsafe_load(qb, j))
                        Base.Cartesian.@nexprs 4 c -> begin
                            kv_c = _v8load(pt + stl * (c - 1) + 32 * (j - 1))
                            x_c = _v8muladd(x_c, sa, kv_c)
                            y_c = _v8muladd(y_c, sb, kv_c)
                        end
                    end
                    Base.Cartesian.@nexprs 4 c -> begin
                        _g4_store_tile!(P, x_c, tl + c - 1, ha, a, b)
                        _g4_store_tile!(P, y_c, tl + c - 1, ha + 1, a, b)
                    end
                    tl += 4
                end
                while tl <= t1
                    pt = base + stl * (tl - 1)
                    x = z
                    y = z
                    for j in 1:hd
                        kv = _v8load(pt + 32 * (j - 1))
                        x = _v8muladd(x, _v8splat(unsafe_load(qa, j)), kv)
                        y = _v8muladd(y, _v8splat(unsafe_load(qb, j)), kv)
                    end
                    _g4_store_tile!(P, x, tl, ha, a, b)
                    _g4_store_tile!(P, y, tl, ha + 1, a, b)
                    tl += 1
                end
            end
            if isodd(rep)
                ha = h0 + rep
                qa = pq + 4 * (ha - 1) * hd
                @inbounds for tl in t0:t1
                    pt = base + stl * (tl - 1)
                    x = z
                    for j in 1:hd
                        x = _v8muladd(x, _v8splat(unsafe_load(qa, j)), _v8load(pt + 32 * (j - 1)))
                    end
                    _g4_store_tile!(P, x, tl, ha, a, b)
                end
            end
        end
    end
    return P
end

# Outputs O[1, :] from the probabilities P[a:b, :]. A task takes one 32-dim
# block of one kv head for every query head that shares it, two at a time.
function _g4_pv_blocks!(O, vc::G4ValBlocks, P, a::Int, b::Int, nh::Int, nkv::Int, hd::Int)
    rep = nh ÷ nkv
    ndb = hd >> 5
    _pool_run_chunks!(nkv * ndb) do t
        hk = (t - 1) % nkv + 1
        db = (t - 1) ÷ nkv + 1
        h0 = (hk - 1) * rep
        z = _v8splat(0.0f0)
        GC.@preserve vc O P begin
            sb = 4 * stride(vc.data, 3)
            base = pointer(vc.data) + sb * ((hk - 1) * size(vc.data, 3) + db - 1)
            @inbounds for rp in 1:2:(rep - 1)
                ha = h0 + rp
                Base.Cartesian.@nexprs 4 d -> (x_d = z; y_d = z)
                for u in a:b
                    pa = _v8splat(P[u, ha])
                    pb = _v8splat(P[u, ha + 1])
                    pr = base + 128 * (u - 1)
                    Base.Cartesian.@nexprs 4 d -> begin
                        vv_d = _v8load(pr + 32 * (d - 1))
                        x_d = _v8muladd(x_d, pa, vv_d)
                        y_d = _v8muladd(y_d, pb, vv_d)
                    end
                end
                qa = (ha - 1) * hd + 32 * (db - 1)
                qb = qa + hd
                Base.Cartesian.@nexprs 4 d -> for i in 1:8
                    O[1, qa + 8 * (d - 1) + i] = x_d[i].value
                    O[1, qb + 8 * (d - 1) + i] = y_d[i].value
                end
            end
            if isodd(rep)
                ha = h0 + rep
                Base.Cartesian.@nexprs 4 d -> x_d = z
                @inbounds for u in a:b
                    pa = _v8splat(P[u, ha])
                    pr = base + 128 * (u - 1)
                    Base.Cartesian.@nexprs 4 d -> x_d = _v8muladd(x_d, pa, _v8load(pr + 32 * (d - 1)))
                end
                qa = (ha - 1) * hd + 32 * (db - 1)
                Base.Cartesian.@nexprs 4 d -> for i in 1:8
                    @inbounds O[1, qa + 8 * (d - 1) + i] = x_d[i].value
                end
            end
        end
    end
    return O
end
