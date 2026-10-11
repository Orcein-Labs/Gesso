# Speculative verify window, CPU correctness path (docs/design/VERIFY_WINDOW.md).
#
# A window runs `[anchor, d_1 .. d_k]` through the model at consecutive cache rows
# in one pass, then the caller accepts a prefix of the drafts and commits. Every
# row is the sequential single-token decode for that row, so the logits, KV rows,
# GDN state, conv state and hidden row match `_decode_step!` bit for bit.
#
# Layer-major: each block runs for every row before the next block starts, so
# the GDN delta rule can run as one fused kernel over the window's tokens. The
# matvecs still run per row; P2 replaces them with multi-column products.

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
# so a Float64 model keeps the bits the baseline `h` keeps.
struct _VerifyWindow{H<:AbstractMatrix}
    T::Int
    hwin::H
    logits::AbstractMatrix
    preds::Vector{Int}
    recs::Vector{Any}
end

# One GDN block for the window. Phase 1 runs norm, projections, conv and gates per
# row and records the per-token values. Phase 2 runs the fused kernel over all
# rows. Phase 3 runs the gated norm and FFN per row. Each phase reads the hidden
# rows from `hwin`, so a row is never read after another row overwrote it.
function _window_gdn_block!(bt, spec, hwin::AbstractMatrix, bufs, cpu, wl, zeros_like,
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
    rec = _GDNWindowRec(
        zeros(Float32, conv_dim, T), zeros(Float32, value_dim, T),
        zeros(Float32, nv * dk, T), zeros(Float32, nv * dk, T), zeros(Float32, nv * dv, T),
        zeros(Float32, nv, T), zeros(Float32, nv, T), zeros(Float32, nv * dv, T),
        zeros(Float32, size(rec_in)), sc.delta, sc.nslab, nv, dk, dv,
    )
    # the conv state advances row by row here; the committed conv is rebuilt
    # from `rec.mixed` in `_commit_window!`, so this copy is discarded
    wconv = copy(conv_in)

    for t in 1:T
        @views bufs.hp.storage[1, :] .= hwin[t, :]
        _gdn_project_conv!(cpu, bufs.hp, bufs.normed, bt, spec, wl, eps, wconv, sc)
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

    for t in 1:T
        @views bufs.hp.storage[1, :] .= hwin[t, :]
        _gdn_gnorm_out!(cpu, bufs.sub, bt, wl, sc, view(rec.core, :, t), view(rec.z, :, t), epsf)
        _decode_ffn_block!(bt, bufs, cpu, wl, zeros_like; eps)
        @views hwin[t, :] .= bufs.hp.storage[1, :]
    end
    return rec
end

# Runs `toks` (the anchor, then the drafts) at 1-based cache rows base .. base+T-1.
# Row t writes its KV row at base+t-1 and attends over rows 1 .. base+t-1 only.
# Nothing in `gdn_conv` or `gdn_rec` changes: the window works on copies, and
# `_commit_window!` decides what survives.
function _decode_window!(model, tensors, kc, vc, toks::AbstractVector{Int}, base::Int, cpu, wl,
    gdn_conv, gdn_rec, cpu_scratch; vocab::Int, n_heads::Int, n_kv_heads::Int, d_head::Int,
    group::Int, eps, theta, inv_freq, rotary_dim::Int=0)
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

    recs = Vector{Any}(nothing, length(model.blocks))
    for (bi, blk) in enumerate(model.blocks)
        bt = tensors.blocks[bi]
        if _block_mixer(bt) === :linear_attention
            recs[bi] = _window_gdn_block!(bt, tensors.spec, hwin, bufs, cpu, wl, zeros_like,
                gdn_conv[bi], gdn_rec[bi], cpu_scratch, eps)
        else
            for t in 1:T
                @views bufs.hp.storage[1, :] .= hwin[t, :]
                _decode_attn_block!(bt, bi, kc, vc, bufs, cpu, wl, tensors, zeros_like, cpu_scratch,
                    base + t - 1; n_heads, n_kv_heads, d_head, group, eps, theta, inv_freq, rotary_dim)
                _decode_ffn_block!(bt, bufs, cpu, wl, zeros_like; eps)
                @views hwin[t, :] .= bufs.hp.storage[1, :]
            end
        end
    end

    logits = zeros_like((T, vocab))
    preds = Vector{Int}(undef, T)
    hact = Activation(; shape=(T, dim), storage=hwin)
    for t in 1:T
        # the head scratch is overwritten by the next row, so the argmax is taken here
        row = _last_logits(model, tensors, hact, cpu, wl, vocab, t; eps, buf=cpu_scratch)
        logits[t, :] .= row
        preds[t] = _greedy_id(row)
    end
    return _VerifyWindow(T, hwin, logits, preds, recs)
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

# One window round, entered right after the anchor (`ids[end]`) was emitted and
# before it was consumed. `dids` are the non-empty drafts. Emits the accepted
# drafts with the baseline's emission and stop rules, commits the consumed rows,
# and leaves the correction for the caller's next top-of-loop argmax. Returns
# `(steps, accepted, stopped)`.
function _verify_window_round!(model, tensors, kc, vc, cpu, wl, h, gdn_conv, gdn_rec, cpu_scratch,
    ids::Vector{Int}, sids::Vector{Int}, dids::Vector{Int}, base::Int, steps::Int, cap::Int,
    on_token, stop_ids, eos_ids; kw...)
    k = length(dids)
    toks = Vector{Int}(undef, k + 1)
    toks[1] = ids[end]
    toks[2:end] .= dids
    win = _decode_window!(model, tensors, kc, vc, toks, base, cpu, wl, gdn_conv, gdn_rec,
        cpu_scratch; kw...)
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
            append!(sids, @view toks[1:i])
            return (steps, i, true)
        end
    end
    _commit_window!(win, n + 1, gdn_conv, gdn_rec, h)
    append!(sids, @view toks[1:(n + 1)])
    return (steps, n, false)
end
