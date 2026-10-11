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
    # Q8_K quantizations of the shared decode inputs, one per input width:
    # residual stream (dim), attention merged (qdim), GDN merged (value_dim),
    # FFN hidden (ff). Each is filled once per input and read by every
    # projection of that input.
    xq_dim::Q8KVec
    xq_qdim::Q8KVec
    xq_vdim::Q8KVec
    xq_ff::Q8KVec
    # Decode activations, reused by every token. The residual stream, the
    # normed inputs, the projection outputs and the attention heads all live
    # here, so a decode step allocates nothing for them.
    hp::Matrix{Float32}           # (1, dim) residual stream
    normed::Matrix{Float32}       # (1, dim)
    normed2::Matrix{Float32}      # (1, dim)
    sub::Matrix{Float32}          # (1, dim) sublayer output
    down::Matrix{Float32}         # (1, dim) FFN output
    q::Matrix{Float32}            # (1, qdim)
    gate_attn::Matrix{Float32}    # (1, qdim) attention output gate
    k::Matrix{Float32}            # (1, kdim)
    v::Matrix{Float32}            # (1, kdim)
    qh::Array{Float32,3}          # (1, n_heads, d_head)
    kh::Array{Float32,3}          # (1, n_kv_heads, d_head)
    vh::Array{Float32,3}          # (1, n_kv_heads, d_head)
    attn::Array{Float32,3}        # (1, n_heads, d_head)
    merged_attn::Matrix{Float32}  # (1, qdim)
    gate::Matrix{Float32}         # (1, ff)
    up::Matrix{Float32}           # (1, ff)
    act::Matrix{Float32}          # (1, ff)
    probs::Vector{Float32}        # (ctx,) one head's scores, then probabilities
    pos1::Vector{Int}             # (1,) RoPE position of the decode token
    logits::Matrix{Float32}       # (1, vocab)
    lastrow::Matrix{Float32}      # (1, dim) final hidden row before the head
    final_normed::Matrix{Float32} # (1, dim) final RMSNorm output
end

# Sizes come from the model. Attention and FFN widths default to 0 for a
# model without that layer; `ctx` is the KV capacity (prompt + new tokens).
function _cpu_decode_scratch(mx, dim::Integer; qdim::Integer=0, ff::Integer=0,
    kdim::Integer=0, n_heads::Integer=0, n_kv_heads::Integer=0, d_head::Integer=0,
    vocab::Integer=0, ctx::Integer=0)
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
        Q8KVec(div(Int(dim), QK_K)),
        Q8KVec(div(max(Int(qdim), 1), QK_K)),
        Q8KVec(div(value_dim, QK_K)),
        Q8KVec(div(max(Int(ff), 1), QK_K)),
        zeros(Float32, 1, dim),
        zeros(Float32, 1, dim),
        zeros(Float32, 1, dim),
        zeros(Float32, 1, dim),
        zeros(Float32, 1, dim),
        zeros(Float32, 1, qdim),
        zeros(Float32, 1, qdim),
        zeros(Float32, 1, kdim),
        zeros(Float32, 1, kdim),
        zeros(Float32, 1, n_heads, d_head),
        zeros(Float32, 1, n_kv_heads, d_head),
        zeros(Float32, 1, n_kv_heads, d_head),
        zeros(Float32, 1, n_heads, d_head),
        zeros(Float32, 1, qdim),
        zeros(Float32, 1, ff),
        zeros(Float32, 1, ff),
        zeros(Float32, 1, ff),
        zeros(Float32, ctx),
        [0],
        zeros(Float32, 1, vocab),
        zeros(Float32, 1, dim),
        zeros(Float32, 1, dim),
    )
end

# Storage for one decode activation: the scratch buffer when there is one,
# a fresh zero array otherwise. `Val` makes the field lookup static.
@inline _decode_buf(::Nothing, ::Val, zl, dims::Tuple) = zl(dims)
@inline _decode_buf(sc::CPUDecodeScratch, ::Val{N}, zl, dims::Tuple) where {N} = getfield(sc, N)

# RoPE position argument for one decode token: the scratch's one-element
# vector on the CPU path, a fresh vector otherwise.
_decode_pos(::Nothing, p::Int) = [p]
_decode_pos(sc::CPUDecodeScratch, p::Int) = (sc.pos1[1] = p; sc.pos1)

# Row `row` of `src` into the one-row matrix `dst`, without a copied slice.
function _copy_row!(dst::Matrix{Float32}, src::AbstractMatrix, row::Int)
    @inbounds for i in axes(dst, 2)
        dst[1, i] = src[row, i]
    end
    return dst
end

# Seq = 1 projections of one input `x` into `dsts`. When every weight has a
# Q8_K kernel, `x` is quantized once into `xq` and the weights run as one
# group (`_quant_matmul_group!`). Otherwise, and whenever `xq` is `nothing`
# (prefill, the reference path), the backend runs them as `_matmul_shared_input!`.
function _dec_matmul_group!(cpu, dsts::Tuple, xq, x::Activation, ws::Tuple, wl)
    if xq !== nothing && all(w -> w.storage isa QuantMatrix && _quant_q8k_kind(w.storage), ws)
        quantize_q8_K!(xq, x.storage)
        _quant_matmul_group!(map(d -> d.storage, dsts), xq, map(w -> w.storage, ws))
        return dsts
    end
    return _matmul_shared_input!(cpu, dsts, x, ws, wl)
end

# A device backend may quantize the shared input once for the whole group (the
# CUDA extension does); here each projection is an ordinary `matmul!`.
function _matmul_shared_input!(cpu, dsts::Tuple, x::Activation, ws::Tuple, wl)
    for i in eachindex(dsts)
        matmul!(cpu, dsts[i], x, ws[i], wl)
    end
    return dsts
end

# The fast path needs host Float32 state, conv state, and norm/gate parameters.
# Projection weights of any format go through `matmul!` unchanged.
function _cpu_gdn_fast_ok(bt, conv_state, rec_state, h)
    return h.storage isa Matrix{Float32} &&
        conv_state isa Matrix{Float32} &&
        rec_state isa Array{Float32,3} &&
        bt.wgdn_conv.storage isa Array{Float32,3} &&
        bt.wgdn_A_log.storage isa Vector{Float32} &&
        bt.wgdn_dt_bias.storage isa Vector{Float32} &&
        bt.wgdn_norm.storage isa Vector{Float32}
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

# One slab of the delta-rule recurrence. It runs on the shared CPU pool, not
# `Threads.@threads`: the pool workers are still spinning from the projection
# matvecs, and a scheduler fan-out here would compete with them for cores.
struct _GDNSlabChunk{R,S}
    rec::R
    sc::S
    dv::Int
    nv::Int
    dk::Int
end

function (g::_GDNSlabChunk)(c::Int)
    sc = g.sc
    nslab = sc.nslab
    jlo = div((c - 1) * g.dv, nslab) + 1
    jhi = div(c * g.dv, nslab)
    _gdn_delta_slab!(g.rec, sc.core, sc.qn, sc.kn, sc.vn, sc.decay, sc.beta,
        view(sc.delta, :, c), jlo, jhi, g.nv, g.dk)
    return nothing
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
        # wgdn_a and wgdn_b are dense Float32, so only qkv and z share the
        # quantized activation. a and b stay on plain matmul!.
        _dec_matmul_group!(cpu, (mixed, z), sc.xq_dim, normed,
            (bt.wgdn_qkv, bt.wgdn_z), wl)
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
    A_log = bt.wgdn_A_log.storage::Vector{Float32}
    dt_bias = bt.wgdn_dt_bias.storage::Vector{Float32}
    scale_norm = bt.wgdn_norm.storage::Vector{Float32}

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
        _pool_run_chunks!(_GDNSlabChunk(rec_state, sc, dv, nv, dk), sc.nslab)
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
        _dec_matmul_group!(cpu, (sub,), sc.xq_vdim, merged_act, (bt.wgdn_o,), wl)
    end
    return sub
end

# Single-query causal attention over the first `K` cache rows, all query heads,
# in one pass per head. Query head `hh` reads KV head `div(hh-1, group)+1` in
# place, so the GQA repeat of the whole cache is never built.
#
# The arithmetic is the reference `_attention_heads!` at L = 1, operation for
# operation: scores accumulate per key over the head dim in order, the softmax
# is the `_cpu_softmax_storage!` sequence (max, exp, Float64 sum, divide), and
# values accumulate per output over the keys in order. The outputs are
# therefore bit-identical to the reference, not just within tolerance.
function _attention_decode_cpu!(attn, q, kc, vc, probs::Vector{Float32},
    n_heads::Int, d_head::Int, group::Int, K::Int)
    fill!(attn, 0.0f0)
    sqd = sqrt(d_head)
    @inbounds for hh in 1:n_heads
        hk = div(hh - 1, group) + 1
        for u in 1:K
            probs[u] = 0.0f0
        end
        for j in 1:d_head
            qv = q[1, hh, j]
            for u in 1:K
                probs[u] += qv * kc[u, hk, j] / sqd
            end
        end
        row_max = probs[1]
        for u in 2:K
            row_max = max(row_max, probs[u])
        end
        isfinite(row_max) || throw(
            gesso_error(ERR_NUMERICAL_INSTABILITY, "softmax!: no finite unmasked maximum"),
        )
        acc = 0.0
        for u in 1:K
            probs[u] = exp(probs[u] - row_max)
            acc += probs[u]
        end
        for u in 1:K
            probs[u] /= acc
        end
        for j in 1:d_head
            for u in 1:K
                attn[1, hh, j] += probs[u] * vc[u, hk, j]
            end
        end
    end
    return attn
end
