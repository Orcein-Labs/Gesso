# CPU-order Q8_0/Q4_K Vulkan kernels. Qualified standalone arithmetic;
# model/stress admission remains pending. Scratch is owned by the serial executor.

@kernel cpu=false function _c1_cpu_q80_pack!(Q, D, @Const(X), M::Int, K::Int, total::Int)
    lin=@index(Global, Linear)
    if lin<=total
        t=(lin-1)%M+1
        b=(lin-1)÷M
        amax=0.0f0
        @inbounds for j in 1:32
            amax=max(amax, abs(X[t, b*32+j]))
        end
        # Correctly rounded F32 operations; fixed F32 conversion points.
        d=Float32(Float64(amax)/127.0)
        inv=d!=0.0f0 ? Float32(1.0/Float64(d)) : 0.0f0
        @inbounds D[t, b+1]=Float16(d)
        @inbounds for j in 1:32
            scaled=Float32(Float64(X[t, b*32+j])*Float64(inv))
            a=abs(scaled)
            whole=unsafe_trunc(Int32, floor(a))
            rounded=whole+Int32(a-Float32(whole)>=0.5f0)
            Q[t, b*32+j]=scaled<0.0f0 ? -rounded : rounded
        end
    end
end

@kernel cpu=false function _c1_cpu_q80_dot!(
    Y,
    @Const(Q),
    @Const(D),
    @Const(WQ),
    @Const(WD),
    M::Int,
    K::Int,
    N::Int,
    total::Int,
)
    lin=@index(Global, Linear)
    if lin<=total
        t=(lin-1)%M+1
        r=(lin-1)÷M
        nb=K÷32
        acc=0.0f0
        @inbounds for b in 0:(nb-1)
            sumi=Int32(0)
            for j in 0:31
                e=b*32+j
                word=WQ[r*(K÷4)+(e÷4)+1]
                wi=unsafe_trunc(Int32, _lq8_i8(word, e%4))
                sumi+=wi*Q[t, e+1]
            end
            scale=Float32(Float64(WD[r*nb+b+1])*Float64(D[t, b+1]))
            term=Float32(Float64(Float32(sumi))*Float64(scale))
            acc=Float32(Float64(acc)+Float64(term))
        end
        @inbounds Y[t, r+1]=acc
    end
end

function _c1_cpu_q80_matmul!(Y, X, q::LavaQ8Quant)
    m, k=size(X)
    Q=_lq_scratch(Int32, :c1_cpu_q80_codes, (m, k))
    D=_lq_scratch(Float16, :c1_cpu_q80_scales, (m, k÷32))
    backend=Lava.KernelAbstractions.get_backend(X)
    pack=Lava.with_no_contraction(
        Lava.with_float_controls(_c1_cpu_q80_pack!(backend), (32, 64)),
    )
    dot=Lava.with_no_contraction(
        Lava.with_float_controls(_c1_cpu_q80_dot!(backend), (32, 64)),
    )
    pack(Q, D, X, m, k, m*(k÷32); ndrange=m*(k÷32))
    dot(Y, Q, D, q.qs, q.d, m, k, q.nrows, m*q.nrows; ndrange=m*q.nrows)
    return Y
end

# Separate physical F64/F32 buffers prevent LLVM from cancelling conversions
# across the two division round points. This is an external operator control;
# real emitted shader and native-pack comparisons still decide conformance.
@kernel cpu=false function _c1_cpu_q4k_iscale64!(I64, @Const(X), M::Int, total::Int)
    lin = @index(Global, Linear)
    if lin <= total
        t = (lin - 1) % M + 1
        b = (lin - 1) ÷ M
        amax = 0.0f0
        mx = 0.0f0
        @inbounds for j in 1:256
            x = X[t, b*256+j]
            a = abs(x)
            if a > amax
                amax = a
                mx = x
            end
        end
        @inbounds I64[t, b+1] = amax == 0.0f0 ? 0.0 : -127.0 / Float64(mx)
    end
end

@kernel cpu=false function _c1_cpu_q4k_iscale32!(I32, @Const(I64), total::Int)
    lin = @index(Global, Linear)
    if lin <= total
        @inbounds I32[lin] = Float32(I64[lin])
    end
end

@kernel cpu=false function _c1_cpu_q4k_recip64!(D64, @Const(I32), total::Int)
    lin = @index(Global, Linear)
    if lin <= total
        @inbounds v = I32[lin]
        @inbounds D64[lin] = v == 0.0f0 ? 0.0 : 1.0 / Float64(v)
    end
end

@kernel cpu=false function _c1_cpu_q4k_pack!(
    Q,
    B,
    D,
    @Const(I32),
    @Const(D64),
    @Const(X),
    M::Int,
    total::Int,
)
    lin = @index(Global, Linear)
    if lin <= total
        t = (lin - 1) % M + 1
        b = (lin - 1) ÷ M
        @inbounds inv = I32[t, b+1]
        @inbounds D[t, b+1] = Float32(D64[t, b+1])
        @inbounds for group in 0:15
            s = Int32(0)
            for j in 1:16
                e = b*256 + group*16 + j
                scaled = inv * X[t, e]
                magic = scaled + 12582912.0f0
                v = (reinterpret(Int32, magic) & Int32(0x007fffff)) - Int32(0x00400000)
                q = min(Int32(127), v)
                Q[t, e] = q
                s += q
            end
            B[t, b*16+group+1] = s
        end
    end
end

@inline _c1_cpu_q4k_fsum(s::NTuple{8, Float32}, d::Float32, a::NTuple{8, Int32}) =
    ntuple(l -> s[l]+d*Float32(a[l]), Val(8))

@kernel cpu=false function _c1_cpu_q4k_dot!(
    Y,
    @Const(Q),
    @Const(B),
    @Const(D),
    @Const(W),
    M::Int,
    K::Int,
    row_bytes::Int,
    total::Int,
)
    lin = @index(Global, Linear)
    if lin <= total
        t = (lin - 1) % M + 1
        r = (lin - 1) ÷ M
        sumf = 0.0f0
        sums = ntuple(_ -> 0.0f0, Val(8))
        @inbounds for ib in 0:(K÷256-1)
            b = r*row_bytes + ib*144
            d = _lq_f16(W, b) * D[t, ib+1]
            dmin = _lq_f16(W, b+2) * D[t, ib+1]
            aux = ntuple(Val(8)) do l
                lane = l-1
                acc = Int32(0)
                for group in 0:7
                    sc, _ = _lq_scale_min(W, b+4, group)
                    sub = Int32(0)
                    for offset in (lane, lane+8, lane+16, lane+24)
                        k = group*32+offset
                        byte = _lq_u8(W, b+16+32*(group>>1)+(k&31))
                        wi = iseven(group) ? byte & Int32(15) : byte >> 4
                        sub += wi * Q[t, ib*256+k+1]
                    end
                    acc += sc*sub
                end
                acc
            end
            sums = _c1_cpu_q4k_fsum(sums, d, aux)
            mins = Int32(0)
            for group in 0:15
                _, mn = _lq_scale_min(W, b+4, group>>1)
                mins += B[t, ib*16+group+1]*mn
            end
            sumf -= dmin*Float32(mins)
        end
        for l in 1:8
            sumf += sums[l]
        end
        @inbounds Y[t, r+1] = sumf
    end
end

function _c1_cpu_q4k_matmul!(Y, X, q::LavaBlockQuant)
    q.kind == _LQ_KIND_Q4_K || error("not Q4_K")
    m, k = size(X)
    k%256 == 0 && q.row_bytes == (k÷256)*144 || error("Q4_K layout")
    Q = _lq_scratch(Int32, :c1_cpu_q4k_codes, (m, k))
    B = _lq_scratch(Int32, :c1_cpu_q4k_bsums, (m, k÷16))
    D = _lq_scratch(Float32, :c1_cpu_q4k_scales, (m, k÷256))
    I32 = _lq_scratch(Float32, :c1_cpu_q4k_iscale32, (m, k÷256))
    I64 = _lq_scratch(Float64, :c1_cpu_q4k_iscale64, (m, k÷256))
    D64 = _lq_scratch(Float64, :c1_cpu_q4k_recip64, (m, k÷256))
    backend = Lava.KernelAbstractions.get_backend(X)
    total = m*(k÷256)
    control(f) = Lava.with_no_contraction(Lava.with_float_controls(f(backend), (32, 64)))
    control(_c1_cpu_q4k_iscale64!)(I64, X, m, total; ndrange=total)
    control(_c1_cpu_q4k_iscale32!)(I32, I64, total; ndrange=total)
    control(_c1_cpu_q4k_recip64!)(D64, I32, total; ndrange=total)
    control(_c1_cpu_q4k_pack!)(Q, B, D, I32, D64, X, m, total; ndrange=total)
    control(_c1_cpu_q4k_dot!)(
        Y,
        Q,
        B,
        D,
        q.data,
        m,
        k,
        q.row_bytes,
        m*q.nrows;
        ndrange=m*q.nrows,
    )
    return Y
end

# Workgroup-per-row CPU-order Q8_0/Q4_K dot kernels. The thread-per-(token, row)
# kernels walk a whole row in one thread, so a decode GEMV runs nrows threads
# with uncoalesced row reads. Here one workgroup owns a row: lanes form the
# per-block integer sums (exact in any order) and the block terms in parallel,
# then lane 1 folds the terms in block order with the same expressions as the
# thread-per-row kernels, so the result is bit-identical to them.
const _C1_WG = 64

@kernel cpu=false function _c1_cpu_q80_dot_wg!(
    Y,
    @Const(Q),
    @Const(D),
    @Const(WQ),
    @Const(WD),
    M::Int,
    K::Int,
)
    tl = @index(Local, Linear)
    r = @index(Group, Linear) - 1
    part = @localmem Float32 (_C1_WG,)
    nb = K ÷ 32
    for t in 1:M
        acc = 0.0f0
        b0 = 0
        while b0 < nb
            b = b0 + tl - 1
            term = 0.0f0
            if b < nb
                sumi = Int32(0)
                @inbounds for w in 0:7
                    word = WQ[r*(K÷4)+b*8+w+1]
                    for j in 0:3
                        e = b*32 + w*4 + j
                        sumi += unsafe_trunc(Int32, _lq8_i8(word, j)) * Q[t, e+1]
                    end
                end
                scale = @inbounds Float32(Float64(WD[r*nb+b+1])*Float64(D[t, b+1]))
                term = Float32(Float64(Float32(sumi))*Float64(scale))
            end
            part[tl] = term
            @synchronize
            if tl == 1
                n = min(_C1_WG, nb - b0)
                for i in 1:n
                    acc = Float32(Float64(acc)+Float64(part[i]))
                end
            end
            @synchronize
            b0 += _C1_WG
        end
        if tl == 1
            @inbounds Y[t, r+1] = acc
        end
    end
end

# Q4_K: eight lanes per superblock (one per aux lane of the CPU kernel), eight
# superblocks per pass; lane 0 of each superblock also forms the integer mins.
@kernel cpu=false function _c1_cpu_q4k_dot_wg!(
    Y,
    @Const(Q),
    @Const(B),
    @Const(D),
    @Const(W),
    M::Int,
    K::Int,
    row_bytes::Int,
)
    tl = @index(Local, Linear)
    r = @index(Group, Linear) - 1
    part = @localmem Int32 (_C1_WG,)
    pmin = @localmem Int32 (_C1_WG ÷ 8,)
    nsb = K ÷ 256
    lane = (tl - 1) % 8
    sbl = (tl - 1) ÷ 8
    for t in 1:M
        sumf = 0.0f0
        sums = ntuple(_ -> 0.0f0, Val(8))
        ib0 = 0
        while ib0 < nsb
            ib = ib0 + sbl
            acc = Int32(0)
            if ib < nsb
                b = r*row_bytes + ib*144
                @inbounds for group in 0:7
                    sc, _ = _lq_scale_min(W, b+4, group)
                    sub = Int32(0)
                    for offset in (lane, lane+8, lane+16, lane+24)
                        k = group*32+offset
                        byte = _lq_u8(W, b+16+32*(group>>1)+(k&31))
                        wi = iseven(group) ? byte & Int32(15) : byte >> 4
                        sub += wi * Q[t, ib*256+k+1]
                    end
                    acc += sc*sub
                end
                if lane == 0
                    mins = Int32(0)
                    @inbounds for group in 0:15
                        _, mn = _lq_scale_min(W, b+4, group>>1)
                        mins += B[t, ib*16+group+1]*mn
                    end
                    pmin[sbl+1] = mins
                end
            end
            part[tl] = acc
            @synchronize
            if tl == 1
                n = min(8, nsb - ib0)
                @inbounds for s in 0:(n-1)
                    ibs = ib0 + s
                    b = r*row_bytes + ibs*144
                    d = _lq_f16(W, b) * D[t, ibs+1]
                    dmin = _lq_f16(W, b+2) * D[t, ibs+1]
                    aux = ntuple(l -> part[s*8+l], Val(8))
                    sums = _c1_cpu_q4k_fsum(sums, d, aux)
                    sumf -= dmin*Float32(pmin[s+1])
                end
            end
            @synchronize
            ib0 += 8
        end
        if tl == 1
            for l in 1:8
                sumf += sums[l]
            end
            @inbounds Y[t, r+1] = sumf
        end
    end
end

# Largest token count that takes the workgroup kernels (receipts on the 5060: Q8_0
# wins through m=8, Q4_K through m=2; both lose at m=33). GESSO_TUNE=cpu_order_wg_max=N
# overrides both; 0 keeps the thread-per-row kernels.
function _c1_wg_max(default::Int)
    return env_tune(:cpu_order_wg_max, default)
end

function _c1_cpu_q80_matmul_wg!(Y, X, q::LavaQ8Quant)
    m, k=size(X)
    Q=_lq_scratch(Int32, :c1_cpu_q80_codes, (m, k))
    D=_lq_scratch(Float16, :c1_cpu_q80_scales, (m, k÷32))
    backend=Lava.KernelAbstractions.get_backend(X)
    control(f) = Lava.with_no_contraction(Lava.with_float_controls(f(backend), (32, 64)))
    control(_c1_cpu_q80_pack!)(Q, D, X, m, k, m*(k÷32); ndrange=m*(k÷32))
    control(_c1_cpu_q80_dot_wg!)(Y, Q, D, q.qs, q.d, m, k;
        ndrange=_C1_WG*q.nrows, workgroupsize=(_C1_WG,))
    return Y
end

function _c1_cpu_q4k_matmul_wg!(Y, X, q::LavaBlockQuant)
    q.kind == _LQ_KIND_Q4_K || error("not Q4_K")
    m, k = size(X)
    k%256 == 0 && q.row_bytes == (k÷256)*144 || error("Q4_K layout")
    Q = _lq_scratch(Int32, :c1_cpu_q4k_codes, (m, k))
    B = _lq_scratch(Int32, :c1_cpu_q4k_bsums, (m, k÷16))
    D = _lq_scratch(Float32, :c1_cpu_q4k_scales, (m, k÷256))
    I32 = _lq_scratch(Float32, :c1_cpu_q4k_iscale32, (m, k÷256))
    I64 = _lq_scratch(Float64, :c1_cpu_q4k_iscale64, (m, k÷256))
    D64 = _lq_scratch(Float64, :c1_cpu_q4k_recip64, (m, k÷256))
    backend = Lava.KernelAbstractions.get_backend(X)
    total = m*(k÷256)
    control(f) = Lava.with_no_contraction(Lava.with_float_controls(f(backend), (32, 64)))
    control(_c1_cpu_q4k_iscale64!)(I64, X, m, total; ndrange=total)
    control(_c1_cpu_q4k_iscale32!)(I32, I64, total; ndrange=total)
    control(_c1_cpu_q4k_recip64!)(D64, I32, total; ndrange=total)
    control(_c1_cpu_q4k_pack!)(Q, B, D, I32, D64, X, m, total; ndrange=total)
    control(_c1_cpu_q4k_dot_wg!)(Y, Q, B, D, q.data, m, k, q.row_bytes;
        ndrange=_C1_WG*q.nrows, workgroupsize=(_C1_WG,))
    return Y
end
