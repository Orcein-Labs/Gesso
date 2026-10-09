# CPU single-token (seq = 1) decode path for the hybrid linear-attention block.
#
# The reference path in `_hybrid_linear_block!` runs the GDN mixer as separate
# Float64 operators with per-call allocation. At seq = 1 the state is small
# (nv x dk x dv floats), so the mixer is rewritten here as F32 fused loops over
# preallocated scratch. Projections still go through `matmul!`, so every
# weight format keeps its existing kernel. The multi-token path does not use
# this file.
#
# Layout: head index varies fastest in every (value-head, feature) buffer, so
# the delta-rule loops read the recurrent state with unit stride.

"""
    CPUDecodeScratch

Preallocated per-generate buffers for the CPU seq = 1 GDN decode path. The
delta rule is threaded over value-dim slabs; slab `c` owns column `c` of
`kvm`, so no buffer is indexed by thread id.
"""
struct CPUDecodeScratch
    nk::Int
    nv::Int
    dk::Int
    dv::Int
    mixed::Matrix{Float32}        # (1, conv_dim) qkv projection
    convout::Vector{Float32}      # (conv_dim,) causal conv + SiLU output
    z::Matrix{Float32}            # (1, value_dim) gate projection
    a::Matrix{Float32}            # (1, nv)
    b::Matrix{Float32}            # (1, nv)
    merged::Matrix{Float32}       # (1, value_dim) gated core, input to the output projection
    qn::Vector{Float32}           # (nv, dk) normalized, scaled query per value head
    kn::Vector{Float32}           # (nv, dk) normalized key per value head
    vn::Vector{Float32}           # (nv, dv) value per value head
    core::Vector{Float32}         # (nv, dv) delta-rule output
    decay::Vector{Float32}        # (nv,)
    beta::Vector{Float32}         # (nv,)
    delta::Matrix{Float32}        # (nv, nslab) per-slab delta / kv_mem buffer
    nslab::Int
end

function _cpu_decode_scratch(mx, dim::Integer)
    nk, nv = mx.num_key_heads, mx.num_value_heads
    dk, dv = mx.key_head_dim, mx.value_head_dim
    key_dim = nk * dk
    value_dim = nv * dv
    conv_dim = 2 * key_dim + value_dim
    nslab = max(1, min(Threads.nthreads(), dv))
    return CPUDecodeScratch(
        nk, nv, dk, dv,
        zeros(Float32, 1, conv_dim),
        zeros(Float32, conv_dim),
        zeros(Float32, 1, value_dim),
        zeros(Float32, 1, nv),
        zeros(Float32, 1, nv),
        zeros(Float32, 1, value_dim),
        zeros(Float32, nv * dk),
        zeros(Float32, nv * dk),
        zeros(Float32, nv * dv),
        zeros(Float32, nv * dv),
        zeros(Float32, nv),
        zeros(Float32, nv),
        zeros(Float32, nv, nslab),
        nslab,
    )
end

# The fast path needs host Float32 state, conv state, and norm/gate parameters.
# Projection weights of any format go through `matmul!` unchanged.
function _cpu_gdn_fast_ok(bt, conv_state, rec_state, h)
    return h.storage isa Matrix{Float32} &&
        conv_state isa Matrix{Float32} &&
        rec_state isa Array{Float32,3} &&
        bt.wgdn_conv.storage isa Array{Float32,3} &&
        bt.wgdn_A_log.storage isa AbstractVector{Float32} &&
        bt.wgdn_dt_bias.storage isa AbstractVector{Float32} &&
        bt.wgdn_norm.storage isa AbstractVector{Float32}
end

# Depthwise causal conv + SiLU for one token, then the conv-state shift.
# Matches `_cpu_causal_conv1d_silu!` at seq = 1 (taps read the old state).
function _gdn_conv_decode!(out, x, weight, conv_state, ksz::Int)
    channels = length(out)
    @inbounds for c in 1:channels
        acc = 0.0f0
        for p in 1:(ksz - 1)
            acc += weight[c, 1, p] * conv_state[c, p]
        end
        xc = x[c]
        acc += weight[c, 1, ksz] * xc
        out[c] = acc / (1.0f0 + exp(-acc))
        for p in 1:(ksz - 2)
            conv_state[c, p] = conv_state[c, p + 1]
        end
        conv_state[c, ksz - 1] = xc
    end
    return out
end

# Delta rule for the value-dim slab jlo:jhi. State layout (nv, dk, dv), so the
# slab `j` columns are contiguous in `rec`. Per value head this is the reference
# recurrence: decay, kv_mem, delta, rank-1 update, and output. Each j column is
# independent, so slabs need no synchronization.
function _gdn_delta_slab!(rec, core, qn, kn, v, decay, beta, delta, jlo::Int, jhi::Int, nv::Int, dk::Int)
    @inbounds for j in jlo:jhi
        base = nv * dk * (j - 1)
        vb = nv * (j - 1)
        for hh in 1:nv
            delta[hh] = 0.0f0
        end
        for i in 1:dk
            ib = nv * (i - 1)
            for hh in 1:nv
                s = rec[base + ib + hh] * decay[hh]
                delta[hh] += s * kn[ib + hh]
            end
        end
        for hh in 1:nv
            delta[hh] = (v[vb + hh] - delta[hh]) * beta[hh]
            core[vb + hh] = 0.0f0
        end
        for i in 1:dk
            ib = nv * (i - 1)
            for hh in 1:nv
                idx = base + ib + hh
                s = rec[idx] * decay[hh] + kn[ib + hh] * delta[hh]
                rec[idx] = s
                core[vb + hh] += s * qn[ib + hh]
            end
        end
    end
    return rec
end

function _gdn_decode_cpu!(cpu, h::Activation, normed::Activation, sub::Activation, bt, spec,
    wl, eps, conv_state, rec_state, sc::CPUDecodeScratch)
    mx = spec.mixer
    nk, nv, dk, dv = sc.nk, sc.nv, sc.dk, sc.dv
    key_dim = nk * dk
    value_dim = nv * dv
    conv_dim = 2 * key_dim + value_dim
    group = div(nv, nk)
    nv % nk == 0 || error("gdn decode: n_v_heads $nv not divisible by n_k_heads $nk")
    ksz = mx.conv_kernel_dim
    epsf = Float32(eps)

    @decode_sub :gdn_norm cpu rmsnorm!(cpu, normed, h, bt.attn_rms, wl; eps)
    mixed = Activation(; shape=(1, conv_dim), storage=sc.mixed)
    z = Activation(; shape=(1, value_dim), storage=sc.z)
    a = Activation(; shape=(1, nv), storage=sc.a)
    b = Activation(; shape=(1, nv), storage=sc.b)
    @decode_sub :gdn_proj cpu begin
        matmul!(cpu, mixed, normed, bt.wgdn_qkv, wl)
        matmul!(cpu, z, normed, bt.wgdn_z, wl)
        matmul!(cpu, a, normed, bt.wgdn_a, wl)
        matmul!(cpu, b, normed, bt.wgdn_b, wl)
    end

    @decode_sub :gdn_conv cpu begin
        _gdn_conv_decode!(sc.convout, vec(sc.mixed), bt.wgdn_conv.storage, conv_state, ksz)
    end
    return _gdn_decode_finish!(cpu, sub, bt, spec, wl, eps, conv_state, rec_state, sc, epsf)
end

# Gates, q/k normalization, delta rule, gated norm, and output projection,
# all reading `sc.mixed` (already holding the conv output) and `sc.z`.
function _gdn_decode_finish!(cpu, sub::Activation, bt, spec, wl, eps, conv_state, rec_state,
    sc::CPUDecodeScratch, epsf::Float32)
    nk, nv, dk, dv = sc.nk, sc.nv, sc.dk, sc.dv
    key_dim = nk * dk
    value_dim = nv * dv
    group = div(nv, nk)
    tiled = spec.mixer.v_head_tiled
    conv = sc.convout
    zv = vec(sc.z)
    av = vec(sc.a)
    bv = vec(sc.b)
    # FrozenParameter storage is `Any`; assert the concrete type once so the
    # per-head loops are not dynamically dispatched.
    A_log = bt.wgdn_A_log.storage::AbstractVector{Float32}
    dt_bias = bt.wgdn_dt_bias.storage::AbstractVector{Float32}
    scale_norm = bt.wgdn_norm.storage::AbstractVector{Float32}

    @decode_sub :gdn_gates cpu begin
        @inbounds for hh in 1:nv
            sc.beta[hh] = _cpu_sigmoid(bv[hh])
            sc.decay[hh] = exp(-exp(A_log[hh]) * _cpu_softplus(av[hh] + dt_bias[hh]))
        end
    end

    @decode_sub :gdn_qk cpu begin
        # value heads: head-major in conv output, (nv, dv) in scratch
        @inbounds for hh in 1:nv, j in 1:dv
            sc.vn[hh + nv * (j - 1)] = conv[2 * key_dim + (hh - 1) * dv + j]
        end
        # L2-normalize each key head once (q and k), then map key heads onto
        # value heads with the same pairing as the reference rule.
        @inbounds for hk in 1:nk
            qss = 0.0f0
            kss = 0.0f0
            for d in 1:dk
                qv = conv[(hk - 1) * dk + d]
                kv = conv[key_dim + (hk - 1) * dk + d]
                qss += qv * qv
                kss += kv * kv
            end
            qinv = 1.0f0 / sqrt(qss + 1.0f-6)
            kinv = 1.0f0 / sqrt(kss + 1.0f-6)
            scale = 1.0f0 / sqrt(Float32(dk))
            for hh in 1:nv
                kh = tiled ? mod(hh - 1, nk) + 1 : div(hh - 1, group) + 1
                kh == hk || continue
                for d in 1:dk
                    qv = conv[(hk - 1) * dk + d]
                    kv = conv[key_dim + (hk - 1) * dk + d]
                    sc.qn[hh + nv * (d - 1)] = (qv * qinv) * scale
                    sc.kn[hh + nv * (d - 1)] = kv * kinv
                end
            end
        end
    end

    @decode_sub :gdn_recur cpu begin
        nslab = sc.nslab
        Threads.@threads for c in 1:nslab
            jlo = div((c - 1) * dv, nslab) + 1
            jhi = div(c * dv, nslab)
            _gdn_delta_slab!(rec_state, sc.core, sc.qn, sc.kn, sc.vn, sc.decay, sc.beta,
                view(sc.delta, :, c), jlo, jhi, nv, dk)
        end
    end

    @decode_sub :gdn_gnorm cpu begin
        merged = vec(sc.merged)
        @inbounds for hh in 1:nv
            ss = 0.0f0
            for j in 1:dv
                cv = sc.core[hh + nv * (j - 1)]
                ss += cv * cv
            end
            r = sqrt(ss / dv + epsf)
            for j in 1:dv
                cv = sc.core[hh + nv * (j - 1)]
                zj = zv[(hh - 1) * dv + j]
                merged[(hh - 1) * dv + j] = (cv / r * scale_norm[j]) * (zj / (1.0f0 + exp(-zj)))
            end
        end
    end

    @decode_sub :gdn_outproj cpu begin
        merged_act = Activation(; shape=(1, value_dim), storage=sc.merged)
        matmul!(cpu, sub, merged_act, bt.wgdn_o, wl)
    end
    return sub
end
