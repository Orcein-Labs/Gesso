# Speculative verify window, CPU correctness path (docs/design/VERIFY_WINDOW.md).
#
# A window runs `[anchor, d_1 .. d_k]` through the model at consecutive cache rows
# in one pass, then the caller accepts a prefix of the drafts and commits. Every
# row is the sequential single-token decode for that row, so the logits, KV rows,
# GDN state, conv state and hidden row match `_decode_step!` bit for bit.
#
# Layer-major: each block runs for every row before the next block starts. The
# weight projections run once over all T rows, so each weight streams once per
# window. The steps that read earlier rows' state stay per row: the GDN conv and
# gates, the fused delta-rule kernel, and the attention core.

# Test-only red control. When set, a partial accept skips the replay from the
# pre-window state and commits the state after every window row. The state
# equality test must then fail, which shows the test can see a wrong commit.
const _VERIFY_WINDOW_SKIP_REPLAY = Ref(false)

# Per-token GDN records for one linear-attention block. Columns are tokens.
# `rec_out` is the out-of-place state the window writes; the pre-window state is
# the live `gdn_rec[bi]`, which the window never touches until commit.
struct _GDNWindowRec
    mixed::Matrix{Float32}     # (conv_dim, T) raw conv input, rebuilds the conv state
    z::Matrix{Float32}         # (value_dim, T) gate values for the gated norm
    qn::Matrix{Float32}        # (nv*dk, T)
    kn::Matrix{Float32}        # (nv*dk, T)
    vn::Matrix{Float32}        # (nv*dv, T)
    decay::Matrix{Float32}     # (nv, T)
    beta::Matrix{Float32}      # (nv, T)
    core::Matrix{Float32}      # (nv*dv, T) delta-rule output per token
    rec_out::Array{Float32,3}  # (nv, dk, dv) state after the window
    delta::Matrix{Float32}     # (nv, nslab) per-chunk scratch, as in CPUDecodeScratch
    nslab::Int
    nv::Int
    dk::Int
    dv::Int
end

# Same arithmetic as `_gdn_delta_slab!`, operation for operation. The state is
# read from `src` and written to `dst`; when they are the same array this is the
# in-place kernel. The first loop reads the prior state, the second writes the
# new one, so the out-of-place form has no read-after-write hazard.
function _gdn_delta_slab_oop!(dst, src, core, qn, kn, v, decay, beta, delta, jlo::Int, jhi::Int, nv::Int, dk::Int)
    @inbounds for j in jlo:jhi
        base = nv * dk * (j - 1)
        vb = nv * (j - 1)
        for hh in 1:nv
            delta[hh] = 0.0f0
        end
        for i in 1:dk
            ib = nv * (i - 1)
            for hh in 1:nv
                s = src[base + ib + hh] * decay[hh]
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
                s = src[idx] * decay[hh] + kn[ib + hh] * delta[hh]
                dst[idx] = s
                core[vb + hh] += s * qn[ib + hh]
            end
        end
    end
    return dst
end

# Applies `ntok` token updates to every slab. Chunk `c` owns the value columns of
# slab c and runs all its tokens in order, so the slabs need no synchronization.
# Token 1 reads the pre-window state; later tokens read the state the previous
# token wrote into `dst`.
struct _GDNWindowChunk
    src::Array{Float32,3}
    dst::Array{Float32,3}
    rec::_GDNWindowRec
    ntok::Int
end

function (g::_GDNWindowChunk)(c::Int)
    r = g.rec
    jlo = div((c - 1) * r.dv, r.nslab) + 1
    jhi = div(c * r.dv, r.nslab)
    dl = view(r.delta, :, c)
    for t in 1:g.ntok
        src = t == 1 ? g.src : g.dst
        _gdn_delta_slab_oop!(g.dst, src,
            view(r.core, :, t), view(r.qn, :, t), view(r.kn, :, t), view(r.vn, :, t),
            view(r.decay, :, t), view(r.beta, :, t), dl, jlo, jhi, r.nv, r.dk)
    end
    return nothing
end

# Verify-window result: the hidden rows after the whole stack, the logits and
# greedy id of every row, and the per-block GDN records (nothing for attention).
# `hwin` and `logits` take the element type of the state they run on (zeros_like),
# so a Float64 model keeps the bits the baseline `h` keeps. A prefill window runs
# without the head, so its `logits` is empty and `preds` is empty. `tapX` holds the
# residual rows after the requested block counts (empty without taps).
struct _VerifyWindow{H<:AbstractMatrix}
    T::Int
    hwin::H
    logits::AbstractMatrix
    preds::Vector{Int}
    recs::Vector{Any}
    tapX::Matrix{Float32}
end

# Copies the residual rows `hwin` into the column block of each tap that equals `b`.
function _window_tap_copy!(tapX::Matrix{Float32}, taps, b::Int, hwin::AbstractMatrix, dim::Int)
    for (k, tb) in enumerate(taps)
        tb == b || continue
        @views tapX[:, (k - 1) * dim + 1:k * dim] .= hwin
    end
    return tapX
end

# Row copies between the (T, n) window matrices and the one-row (1, n) buffers
# that the single-token decode kernels take. The arithmetic runs on the same
# one-row storage as decode, so per-row results are decode's bits.
_win_load!(dst::AbstractArray, src::AbstractMatrix, t::Int) = (copyto!(vec(dst), view(src, t, :)); dst)
_win_store!(dst::AbstractMatrix, t::Int, src::AbstractArray) = (copyto!(view(dst, t, :), vec(src)); dst)

# The quantized weight behind a projection, or `nothing` for a dense weight.
_window_quant(w) = hasfield(typeof(w), :storage) && getfield(w, :storage) isa QuantMatrix ?
    getfield(w, :storage) : nothing

# y = x * W' over all rows of `x`. A native quantized weight runs once over the
# whole window through the multi-column dot. Any other weight runs one (1, n)
# product per row, which is what decode runs, so the bits match by construction.
function _window_matmul!(cpu, dst::AbstractMatrix, x::AbstractMatrix, w, wl, zeros_like)
    q = _window_quant(w)
    if q !== nothing && _quant_native(q.ggml_type)
        _quant_matmul!(dst, x, q)
        return dst
    end
    xr = zeros_like((1, size(x, 2)))
    yr = zeros_like((1, size(dst, 2)))
    xa = Activation(; shape=(1, size(x, 2)), storage=xr)
    ya = Activation(; shape=(1, size(dst, 2)), storage=yr)
    for t in axes(x, 1)
        _win_load!(xr, x, t)
        matmul!(cpu, ya, xa, w, wl)
        _win_store!(dst, t, yr)
    end
    return dst
end

# Matrices the window's blocks write into, sized for T rows and allocated once
# per window.
function _window_buffers(zl, T::Int, dim::Int, kdim::Int, qdim::Int, conv_dim::Int,
    value_dim::Int, hidden::Int, vocab::Int, head::Bool)
    return (;
        normed=zl((T, dim)), mixed=zl((T, conv_dim)), zc=zl((T, value_dim)),
        gdn_merged=zl((T, value_dim)),
        q=zl((T, qdim)), k=zl((T, kdim)), v=zl((T, kdim)), gate=zl((T, qdim)),
        attn_merged=zl((T, qdim)),
        sub=zl((T, dim)), normed2=zl((T, dim)),
        ffn_gate=zl((T, hidden)), ffn_up=zl((T, hidden)), ffn_act=zl((T, hidden)),
        ffn_down=zl((T, dim)),
        hn=zl((T, dim)),
        logits=zl(head ? (T, vocab) : (0, vocab)),
    )
end

# One GDN block for the window. The norm and the qkv/z projections run over all
# rows at once. The conv and gates run per row, since row t reads the conv state
# that rows before t wrote. The fused kernel then runs over all rows, and the
# gated norm and ssm_out projection run over all rows. Returns the records and
# leaves the block's output in `W.sub`.
function _window_gdn_block!(bt, spec, hwin::AbstractMatrix, W, bufs, cpu, wl, zeros_like,
    conv_in, rec_in, sc, eps)
    T = size(hwin, 1)
    sc === nothing &&
        error("verify window: a GDN block needs the CPU decode scratch (the window runs on the CPU path)")
    _cpu_gdn_fast_ok(bt, conv_in, rec_in, bufs.hp) ||
        error("verify window: GDN block lacks the CPU fast path (F32 state and weights required)")
    nv, dk, dv = sc.nv, sc.dk, sc.dv
    conv_dim = size(sc.mixed, 2)
    value_dim = nv * dv
    epsf = Float32(eps)
    ksz = spec.mixer.conv_kernel_dim
    rec = _GDNWindowRec(
        zeros(Float32, conv_dim, T), zeros(Float32, value_dim, T),
        zeros(Float32, nv * dk, T), zeros(Float32, nv * dk, T), zeros(Float32, nv * dv, T),
        zeros(Float32, nv, T), zeros(Float32, nv, T), zeros(Float32, nv * dv, T),
        zeros(Float32, size(rec_in)), sc.delta, sc.nslab, nv, dk, dv,
    )
    # the conv state advances row by row here; the committed conv is rebuilt
    # from `rec.mixed` in `_commit_window!`, so this copy is discarded
    wconv = copy(conv_in)
    a_act = Activation(; shape=(1, nv), storage=sc.a)
    b_act = Activation(; shape=(1, nv), storage=sc.b)

    for t in 1:T
        _win_load!(bufs.hp.storage, hwin, t)
        rmsnorm!(cpu, bufs.normed, bufs.hp, bt.attn_rms, wl; eps)
        _win_store!(W.normed, t, bufs.normed.storage)
    end
    _window_matmul!(cpu, W.mixed, W.normed, bt.wgdn_qkv, wl, zeros_like)
    _window_matmul!(cpu, W.zc, W.normed, bt.wgdn_z, wl, zeros_like)

    for t in 1:T
        _win_load!(bufs.normed.storage, W.normed, t)
        _win_load!(sc.mixed, W.mixed, t)
        _win_load!(sc.z, W.zc, t)
        # wgdn_a and wgdn_b are dense Float32 and stay on the (1, n) product, as in decode
        matmul!(cpu, a_act, bufs.normed, bt.wgdn_a, wl)
        matmul!(cpu, b_act, bufs.normed, bt.wgdn_b, wl)
        _gdn_conv_decode!(sc.convout, vec(sc.mixed), bt.wgdn_conv.storage, wconv, ksz)
        _gdn_gates_qk!(cpu, bt, spec, sc)
        copyto!(view(rec.mixed, :, t), vec(sc.mixed))
        copyto!(view(rec.z, :, t), vec(sc.z))
        copyto!(view(rec.qn, :, t), sc.qn)
        copyto!(view(rec.kn, :, t), sc.kn)
        copyto!(view(rec.vn, :, t), sc.vn)
        copyto!(view(rec.decay, :, t), sc.decay)
        copyto!(view(rec.beta, :, t), sc.beta)
    end

    _pool_run_chunks!(_GDNWindowChunk(rec_in, rec.rec_out, rec, T), sc.nslab)

    scale_norm = bt.wgdn_norm.storage::Vector{Float32}
    for t in 1:T
        _gdn_gnorm_merge!(view(W.gdn_merged, t, :), nv, dv,
            view(rec.core, :, t), view(rec.z, :, t), scale_norm, epsf)
    end
    _window_matmul!(cpu, W.sub, W.gdn_merged, bt.wgdn_o, wl, zeros_like)
    return rec
end

# One attention block for the window. The norm and the q, k, v and gate
# projections run over all rows. The per-row core then appends each row's KV
# at its own cache row and attends over the rows up to it, so row t never reads
# a cache row after `base+t-1`. The output projection runs over all rows.
function _window_attn_block!(bt, bi::Int, kc, vc, hwin::AbstractMatrix, W, bufs, cpu, wl,
    tensors, zeros_like, cpu_scratch, base::Int;
    n_heads::Int, n_kv_heads::Int, d_head::Int, group::Int, eps, theta, inv_freq, rotary_dim::Int=0)
    T = size(hwin, 1)
    for t in 1:T
        _win_load!(bufs.hp.storage, hwin, t)
        rmsnorm!(cpu, bufs.normed, bufs.hp, bt.attn_rms, wl; eps)
        _win_store!(W.normed, t, bufs.normed.storage)
    end
    _window_matmul!(cpu, W.q, W.normed, bt.wq, wl, zeros_like)
    _window_matmul!(cpu, W.k, W.normed, bt.wk, wl, zeros_like)
    _window_matmul!(cpu, W.v, W.normed, bt.wv, wl, zeros_like)
    gated = haskey(bt, :wattn_gate)
    gated && _window_matmul!(cpu, W.gate, W.normed, bt.wattn_gate, wl, zeros_like)

    for t in 1:T
        _win_load!(bufs.q.storage, W.q, t)
        _win_load!(bufs.k.storage, W.k, t)
        _win_load!(bufs.v.storage, W.v, t)
        gated && _win_load!(bufs.gate_attn.storage, W.gate, t)
        if haskey(bt, :wattn_head_gate)
            _win_load!(bufs.normed.storage, W.normed, t)
            matmul!(cpu, bufs.gate_head, bufs.normed, bt.wattn_head_gate, wl)
        end
        _decode_attn_core!(bt, bi, kc, vc, bufs, cpu, wl, tensors, zeros_like, cpu_scratch, base + t - 1;
            n_heads, n_kv_heads, d_head, group, eps, theta, inv_freq, rotary_dim)
        _attn_merge_gate!(bt, bufs, cpu, n_heads, d_head)
        _win_store!(W.attn_merged, t, bufs.merged.storage)
    end
    _window_matmul!(cpu, W.sub, W.attn_merged, bt.wo, wl, zeros_like)
    return W.sub
end

# The FFN half of a block over all rows. `subT` is the mixer output for every
# row. Dense FFNs run gate, up and down over all rows. A MoE FFN routes each row
# on its own, as decode does, so its rows run through the decode block.
function _window_ffn_block!(bt, hwin::AbstractMatrix, subT::AbstractMatrix, W, bufs, cpu, wl,
    zeros_like; eps)
    T = size(hwin, 1)
    if haskey(bt, :wrouter)
        for t in 1:T
            _win_load!(bufs.hp.storage, hwin, t)
            _win_load!(bufs.sub.storage, subT, t)
            _decode_ffn_block!(bt, bufs, cpu, wl, zeros_like; eps)
            _win_store!(hwin, t, bufs.hp.storage)
        end
        return hwin
    end
    _add_storage!(hwin, subT)
    for t in 1:T
        _win_load!(bufs.hp.storage, hwin, t)
        rmsnorm!(cpu, bufs.normed2, bufs.hp, bt.ffn_rms, wl; eps)
        _win_store!(W.normed2, t, bufs.normed2.storage)
    end
    _window_matmul!(cpu, W.ffn_gate, W.normed2, bt.wgate, wl, zeros_like)
    _window_matmul!(cpu, W.ffn_up, W.normed2, bt.wup, wl, zeros_like)
    act = Activation(; shape=size(W.ffn_act), storage=W.ffn_act)
    gate = Activation(; shape=size(W.ffn_gate), storage=W.ffn_gate)
    up = Activation(; shape=size(W.ffn_up), storage=W.ffn_up)
    haskey(bt, :ffn_geglu) ? _geglu!(cpu, act, gate, up) : swiglu!(cpu, act, gate, up, wl)
    _window_matmul!(cpu, W.ffn_down, W.ffn_act, bt.wdown, wl, zeros_like)
    _add_storage!(hwin, W.ffn_down)
    return hwin
end

# The tied head over all rows: the final RMSNorm per row, then one lm_head product
# over the window. The argmax is taken per row by the caller.
function _window_head!(tensors, hwin::AbstractMatrix, W, bufs, cpu, wl, zeros_like; eps)
    final_rms = haskey(tensors, :final_rms) ? tensors.final_rms : nothing
    for t in axes(hwin, 1)
        _win_load!(bufs.hp.storage, hwin, t)
        if final_rms === nothing
            _win_store!(W.hn, t, bufs.hp.storage)
        else
            rmsnorm!(cpu, bufs.normed2, bufs.hp, final_rms, wl; eps)
            _win_store!(W.hn, t, bufs.normed2.storage)
        end
    end
    _window_matmul!(cpu, W.logits, W.hn, tensors.lm_head, wl, zeros_like)
    return W.logits
end

# Runs `toks` (the anchor, then the drafts) at 1-based cache rows base .. base+T-1.
# Row t writes its KV row at base+t-1 and attends over rows 1 .. base+t-1 only.
# Nothing in `gdn_conv` or `gdn_rec` changes: the window works on copies, and
# `_commit_window!` decides what survives. With `head=false` the window skips the
# logits, which is how a prompt prefill uses it.
function _decode_window!(model, tensors, kc, vc, toks::AbstractVector{Int}, base::Int, cpu, wl,
    gdn_conv, gdn_rec, cpu_scratch; vocab::Int, n_heads::Int, n_kv_heads::Int, d_head::Int,
    group::Int, eps, theta, inv_freq, rotary_dim::Int=0, head::Bool=true,
    taps::AbstractVector{Int}=Int[])
    T = length(toks)
    T >= 1 || error("verify window: empty window")
    base >= 1 || error("verify window: base row $base is before the first cache row")
    dim = model.embedding.dim
    kdim = n_kv_heads * d_head
    qdim = n_heads * d_head
    zeros_like = _decode_zeros_like(cpu, tensors)
    hidden_ffn = model.blocks[1].ffn.hidden
    bufs = _decode_row_buffers(cpu_scratch, zeros_like, dim, kdim, qdim, n_heads, n_kv_heads, d_head, hidden_ffn)
    hwin = zeros_like((T, dim))
    embedding_lookup!(cpu, Activation(; shape=(T, dim), storage=hwin), tensors.embedding, collect(toks), wl)
    conv_dim = cpu_scratch === nothing ? 0 : size(cpu_scratch.mixed, 2)
    value_dim = cpu_scratch === nothing ? 0 : cpu_scratch.nv * cpu_scratch.dv
    W = _window_buffers(zeros_like, T, dim, kdim, qdim, conv_dim, value_dim, hidden_ffn, vocab, head)
    tapX = zeros(Float32, isempty(taps) ? 0 : T, dim * length(taps))
    _window_tap_copy!(tapX, taps, 0, hwin, dim)

    recs = Vector{Any}(nothing, length(model.blocks))
    for (bi, blk) in enumerate(model.blocks)
        bt = tensors.blocks[bi]
        if _block_mixer(bt) === :linear_attention
            recs[bi] = _window_gdn_block!(bt, tensors.spec, hwin, W, bufs, cpu, wl, zeros_like,
                gdn_conv[bi], gdn_rec[bi], cpu_scratch, eps)
            subT = W.sub
        else
            subT = _window_attn_block!(bt, bi, kc, vc, hwin, W, bufs, cpu, wl, tensors, zeros_like,
                cpu_scratch, base; n_heads, n_kv_heads, d_head, group, eps, theta, inv_freq, rotary_dim)
        end
        _window_ffn_block!(bt, hwin, subT, W, bufs, cpu, wl, zeros_like; eps)
        _window_tap_copy!(tapX, taps, bi, hwin, dim)
    end

    if head
        logits = _window_head!(tensors, hwin, W, bufs, cpu, wl, zeros_like; eps)
        preds = Vector{Int}(undef, T)
        for t in 1:T
            preds[t] = _greedy_id(view(logits, t, :))
        end
        return _VerifyWindow(T, hwin, logits, preds, recs, tapX)
    end
    return _VerifyWindow(T, hwin, W.logits, Int[], recs, tapX)
end

# Rebuilds the conv state after the first `m` window inputs. The state holds the
# last ksz-1 inputs, newest in the last column; before the window's first input
# the old state supplies the columns.
function _gdn_conv_rebuild(conv_in::Matrix{Float32}, mixed::Matrix{Float32}, m::Int)
    channels, K = size(conv_in)
    out = similar(conv_in)
    @inbounds for p in 1:K, c in 1:channels
        idx = m - K + p
        out[c, p] = idx >= 1 ? mixed[c, idx] : conv_in[c, idx + K]
    end
    return out
end

# Commits the first `m` window rows (1 <= m <= T). Full accept takes the window's
# state directly. A partial accept first replays the fused kernel from the
# pre-window state over the m consumed tokens, because the window's state ran past
# the rejection point. The conv state is rebuilt from the consumed inputs, and `h`
# takes the hidden row of the last consumed token.
function _commit_window!(win::_VerifyWindow, m::Int, gdn_conv, gdn_rec, h::Activation)
    1 <= m <= win.T || error("verify window: commit of $m rows out of 1:$(win.T)")
    for bi in eachindex(win.recs)
        rec = win.recs[bi]
        rec === nothing && continue
        if m < win.T && !_VERIFY_WINDOW_SKIP_REPLAY[]
            _pool_run_chunks!(_GDNWindowChunk(gdn_rec[bi], rec.rec_out, rec, m), rec.nslab)
        end
        gdn_conv[bi] .= _gdn_conv_rebuild(gdn_conv[bi], rec.mixed, m)
        gdn_rec[bi] = rec.rec_out
    end
    @views h.storage[1, :] .= win.hwin[m, :]
    return h
end

# Hands the final-normed rows of the `m` committed window rows to `on_rows`.
_window_rows!(on_rows::Nothing, tensors, win, base::Int, m::Int, cpu, wl; kw...) = nothing
function _window_rows!(on_rows, tensors, win, base::Int, m::Int, cpu, wl; eps=1e-6, kw...)
    on_rows(base, _final_normed_rows(tensors, view(win.hwin, 1:m, :), cpu, wl; eps))
    return nothing
end

# Hands the raw tap rows of the `m` committed window rows to `on_taps`.
_window_taps!(on_taps::Nothing, win, base::Int, m::Int) = nothing
function _window_taps!(on_taps, win, base::Int, m::Int)
    on_taps(base, win.tapX[1:m, :])
    return nothing
end

# One window round, entered right after the anchor (`ids[end]`) was emitted and
# before it was consumed. `dids` are the non-empty drafts. Emits the accepted
# drafts with the baseline's emission and stop rules, commits the consumed rows,
# and leaves the correction for the caller's next top-of-loop argmax. Returns
# `(steps, accepted, stopped)`.
function _verify_window_round!(model, tensors, kc, vc, cpu, wl, h, gdn_conv, gdn_rec, cpu_scratch,
    ids::Vector{Int}, sids::Vector{Int}, dids::Vector{Int}, base::Int, steps::Int, cap::Int,
    on_token, stop_ids, eos_ids; on_rows=nothing, on_taps=nothing, taps=Int[], kw...)
    k = length(dids)
    toks = Vector{Int}(undef, k + 1)
    toks[1] = ids[end]
    toks[2:end] .= dids
    win = _decode_window!(model, tensors, kc, vc, toks, base, cpu, wl, gdn_conv, gdn_rec,
        cpu_scratch; taps, kw...)
    # row i predicts the token after toks[i], so it checks draft d_i
    n = 0
    while n < k && win.preds[n + 1] == dids[n + 1]
        n += 1
    end
    for i in 1:n
        tok = dids[i]
        push!(ids, tok)
        steps += 1
        stop = steps >= cap || tok in eos_ids || tok in stop_ids
        if on_token !== nothing && on_token(tok) === false
            stop = true
        end
        if stop
            # the stop token is output but not consumed, as in the baseline
            _commit_window!(win, i, gdn_conv, gdn_rec, h)
            _window_rows!(on_rows, tensors, win, base, i, cpu, wl; kw...)
            _window_taps!(on_taps, win, base, i)
            append!(sids, @view toks[1:i])
            return (steps, i, true)
        end
    end
    _commit_window!(win, n + 1, gdn_conv, gdn_rec, h)
    _window_rows!(on_rows, tensors, win, base, n + 1, cpu, wl; kw...)
    _window_taps!(on_taps, win, base, n + 1)
    append!(sids, @view toks[1:(n + 1)])
    return (steps, n, false)
end
