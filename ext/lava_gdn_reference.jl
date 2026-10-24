# CPU-order GDN recurrence on Vulkan. Keep Float64 q/k buffers and reductions,
# Float32 decay/state storage, and each reference rounding point explicit.
# Public per-kernel arithmetic contracts preserve unrelated kernel defaults.
@kernel cpu=false function _lava_gdn_qk_reference_kernel!(
    qn,
    kn,
    @Const(q),
    @Const(k),
    nk::Int,
    group::Int,
    tiled::Bool,
    l2norm::Bool,
    scale::Float64,
    eps::Float64,
    nv::Int,
    dk::Int,
    n::Int,
)
    i=@index(Global, Linear)
    if i<=n
        hh=(i-1)%nv+1
        t=(i-1)÷nv+1
        h=tiled ? (hh-1)%nk+1 : (hh-1)÷group+1
        sq=0.0
        sk=0.0
        @inbounds for d = 1:dk
            qd=Float64(q[t, h, d])
            kd=Float64(k[t, h, d])
            sq+=qd*qd
            sk+=kd*kd
        end
        iq=l2norm ? 1.0/sqrt(sq+eps) : 1.0
        ik=l2norm ? 1.0/sqrt(sk+eps) : 1.0
        @inbounds for d = 1:dk
            qnormalized=Float64(q[t, h, d])*iq
            qn[t, hh, d]=qnormalized*scale
            kn[t, hh, d]=Float64(k[t, h, d])*ik
        end
    end
end

@kernel cpu=false function _lava_gdn_decay_reference_kernel!(decay, @Const(g), n::Int)
    i=@index(Global, Linear)
    if i<=n
        @inbounds decay[i]=_lava_gdn_exp32(g[i])
    end
end
@kernel cpu=false function _lava_gdn_recur_reference_kernel!(
    out,
    @Const(qn),
    @Const(kn),
    @Const(v),
    @Const(decay),
    @Const(beta),
    state,
    nv::Int,
    dk::Int,
    dv::Int,
    seq::Int,
    n::Int,
)
    i=@index(Global, Linear)
    if i<=n
        hh=(i-1)%nv+1
        j=(i-1)÷nv+1
        @inbounds for t = 1:seq
            dt=decay[t, hh]
            for ii = 1:dk
                state[hh, ii, j]*=dt
            end
            kv=0.0
            for ii = 1:dk
                kv+=Float64(state[hh, ii, j])*kn[t, hh, ii]
            end
            delta=(Float64(v[t, hh, j])-kv)*Float64(beta[t, hh])
            for ii = 1:dk
                updated=Float64(state[hh, ii, j])+kn[t, hh, ii]*delta
                state[hh, ii, j]=Float32(updated)
            end
            acc=0.0
            for ii = 1:dk
                acc+=Float64(state[hh, ii, j])*qn[t, hh, ii]
            end
            out[t, hh, j]=Float32(acc)
        end
    end
end


_gdn_reference_kernel(f, backend) =
    Lava.with_no_contraction(Lava.with_float_controls(f(backend), (32, 64)))

function _lava_gdn_reference!(
    out,
    q,
    k,
    v,
    g,
    beta,
    state;
    l2norm::Bool = true,
    v_head_tiled::Bool = false,
)
    seq, nk, dk = size(q)
    nv, dv = size(v, 2), size(v, 3)
    kb = KA.get_backend(q)
    qn = Lava.LavaArray{Float64}(undef, seq, nv, dk)
    kn = Lava.LavaArray{Float64}(undef, seq, nv, dk)
    decay = Lava.LavaArray{Float32}(undef, seq, nv)
    n = seq * nv
    _gdn_reference_kernel(_lava_gdn_decay_reference_kernel!, kb)(decay, g, n; ndrange = n)
    _gdn_reference_kernel(_lava_gdn_qk_reference_kernel!, kb)(
        qn,
        kn,
        q,
        k,
        Int(nk),
        Int(nv ÷ nk),
        v_head_tiled,
        l2norm,
        1.0 / sqrt(Float64(dk)),
        1.0e-6,
        Int(nv),
        Int(dk),
        n;
        ndrange = n,
    )
    _gdn_reference_kernel(_lava_gdn_recur_reference_kernel!, kb)(
        out,
        qn,
        kn,
        v,
        decay,
        beta,
        state,
        Int(nv),
        Int(dk),
        Int(dv),
        Int(seq),
        nv * dv;
        ndrange = nv * dv,
    )
    return out
end
