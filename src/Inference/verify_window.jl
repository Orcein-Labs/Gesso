# Speculative verify window, CPU correctness path.
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
    rec_out::Array{Float32, 3}  # (nv, dk, dv) state after the window
    nslab::Int
    nv::Int
    dk::Int
    dv::Int
end

# Applies `ntok` token updates to every slab. Chunk `c` owns the value columns of
# slab c and runs all its tokens in order, so the slabs need no synchronization.
# Token 1 reads the pre-window state; later tokens read the state the previous
# token wrote into `dst`.
struct _GDNWindowChunk
    src::Array{Float32, 3}
    dst::Array{Float32, 3}
    rec::_GDNWindowRec
    ntok::Int
end

function (g::_GDNWindowChunk)(c::Int)
    r = g.rec
    jlo = div((c - 1) * r.dv, r.nslab) + 1
    jhi = div(c * r.dv, r.nslab)
    for t in 1:g.ntok
        src = t == 1 ? g.src : g.dst
        _gdn_delta_slab!(
            g.dst,
            src,
            view(r.core, :, t),
            view(r.qn, :, t),
            view(r.kn, :, t),
            view(r.vn, :, t),
            view(r.decay, :, t),
            view(r.beta, :, t),
            jlo,
            jhi,
            r.nv,
            r.dk,
        )
    end
    return nothing
end

# Verify-window result: the hidden rows after the whole stack, the logits and
# greedy id of every row, and the per-block GDN records (nothing for attention).
# `hwin` and `logits` take the element type of the state they run on (zeros_like),
# so a Float64 model keeps the bits the baseline `h` keeps. A prefill window runs
# without the head, so its `logits` is empty and `preds` is empty. `tapX` holds the
# residual rows after the requested block counts (empty without taps).
struct _VerifyWindow{H <: AbstractMatrix}
    T::Int
    hwin::H
    logits::AbstractMatrix
    preds::Vector{Int}
    recs::Vector{Any}
    tapX::Matrix{Float32}
    cache::Union{Nothing, Dict{Any, Any}}
end

# Windows up to this many rows reuse their buffers through the decode scratch, so
# a steady draft length allocates nothing per window. Longer windows (prompt
# prefill chunks) allocate fresh, which keeps chunk-sized buffers from staying
# alive for the rest of the session.
const _WINDOW_REUSE_MAX_T = 32

_window_cache(cpu_scratch, T::Int) =
    cpu_scratch === nothing || T > _WINDOW_REUSE_MAX_T ? nothing : cpu_scratch.window

# Buffers from `make()`, kept in `cache[slot]` with `key`. A hit zeroes them, so
# a reused buffer starts as a fresh one would. A new key replaces the slot.
function _window_reuse!(make, cache, slot, key)
    cache === nothing && return make()
    hit = get(cache, slot, nothing)
    if hit !== nothing && hit[1] == key
        foreach(b -> fill!(b, zero(eltype(b))), hit[2])
        return hit[2]
    end
    bufs = make()
    cache[slot] = (key, bufs)
    return bufs
end

# Copies the residual rows `hwin` into the column block of each tap that equals `b`.
function _window_tap_copy!(
    tapX::Matrix{Float32},
    taps,
    b::Int,
    hwin::AbstractMatrix,
    dim::Int,
)
    for (k, tb) in enumerate(taps)
        tb == b || continue
        @views tapX[:, ((k-1)*dim+1):(k*dim)] .= hwin
    end
    return tapX
end

# Row copies between the (T, n) window matrices and the one-row (1, n) buffers
# that the single-token decode kernels take. The arithmetic runs on the same
# one-row storage as decode, so per-row results are decode's bits.
_win_load!(dst::AbstractArray, src::AbstractMatrix, t::Int) =
    (copyto!(vec(dst), view(src, t, :)); dst)
_win_store!(dst::AbstractMatrix, t::Int, src::AbstractArray) =
    (copyto!(view(dst, t, :), vec(src)); dst)

# The quantized weight behind a projection, or `nothing` for a dense weight.
_window_quant(w) =
    hasfield(typeof(w), :storage) && getfield(w, :storage) isa QuantMatrix ?
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

# Runs `f(rows)` over 1:T, split across threads when there is more than one row.
_window_split_rows!(f, T::Int) =
    Threads.nthreads() > 1 && T > 1 ? _cpu_split_rows!(f, T) : f(1:T)

# RMS norm of every window row in one call. The row kernel folds each row in the
# order the one-row decode call does, so the result is the per-row loop's bits.
# With mixed element types the rows still go through the one-row buffers.
function _window_rmsnorm!(cpu, dst::AbstractMatrix, src::AbstractMatrix, hp, out, w, wl; eps)
    if eltype(dst) == eltype(src) == eltype(hp.storage) == eltype(out.storage)
        rmsnorm!(
            cpu,
            Activation(; shape=size(dst), storage=dst),
            Activation(; shape=size(src), storage=src),
            w,
            wl;
            eps,
        )
        return dst
    end
    for t in axes(src, 1)
        _win_load!(hp.storage, src, t)
        rmsnorm!(cpu, out, hp, w, wl; eps)
        _win_store!(dst, t, out.storage)
    end
    return dst
end

# Depthwise causal conv + SiLU for window row t, read from the raw rows instead of
# a rolling state. Tap p of row t reads x_{t-ksz+p}: window row t-ksz+p when that
# is at least 1, otherwise column t-1+p of the pre-window state. The products and
# the sum run in `_gdn_conv_decode!`'s order, so the output is decode's bits.
function _gdn_conv_window_row!(out, mixed::Matrix{Float32}, conv_in, weight, t::Int, ksz::Int)
    @inbounds for c in eachindex(out)
        acc = 0.0f0
        for p in 1:(ksz-1)
            s = t - ksz + p
            xs = s >= 1 ? mixed[c, s] : conv_in[c, t-1+p]
            acc += weight[c, 1, p] * xs
        end
        acc += weight[c, 1, ksz] * mixed[c, t]
        out[c] = acc / (1.0f0 + exp(-acc))
    end
    return out
end

# The per-row GDN work between the projections and the delta rule: the a and b
# gate products, the causal conv, the gates and the q/k normalization. Row t's
# conv reads only raw rows up to t and the pre-window state, so after the raw
# rows are copied into `rec` the rows are independent and split across threads.
# Each row runs decode's kernels on its own one-row buffers. A quantized a or b
# product shares scratch, so it runs first over all rows through
# `_window_matmul!`, whose rows are decode's bits, and the rows read it from there.
function _window_gdn_rows!(rec, W, conv_in, bt, spec, sc, bufs, cpu, wl, zeros_like, ksz::Int)
    T = size(W.mixed, 1)
    _window_split_rows!(T) do rows
        for t in rows
            copyto!(view(rec.mixed, :, t), view(W.mixed, t, :))
            copyto!(view(rec.z, :, t), view(W.zc, t, :))
        end
    end
    A_log = bt.wgdn_A_log.storage::Vector{Float32}
    dt_bias = bt.wgdn_dt_bias.storage::Vector{Float32}
    weight = bt.wgdn_conv.storage
    nk, nv, dk, dv = sc.nk, sc.nv, sc.dk, sc.dv
    tiled = spec.mixer.v_head_tiled
    quant_ab = _is_quant_storage(bt.wgdn_a.storage) || _is_quant_storage(bt.wgdn_b.storage)
    A = similar(sc.a, quant_ab ? T : 0, size(sc.a, 2))
    B = similar(sc.b, quant_ab ? T : 0, size(sc.b, 2))
    if quant_ab
        _window_matmul!(cpu, A, W.normed, bt.wgdn_a, wl, zeros_like)
        _window_matmul!(cpu, B, W.normed, bt.wgdn_b, wl, zeros_like)
    end
    function run_rows(rows)
        xr = similar(bufs.normed.storage)
        a = similar(sc.a)
        b = similar(sc.b)
        xa = Activation(; shape=size(xr), storage=xr)
        aa = Activation(; shape=size(a), storage=a)
        ba = Activation(; shape=size(b), storage=b)
        convout = similar(sc.convout)
        for t in rows
            if quant_ab
                av, bv = view(A, t, :), view(B, t, :)
            else
                _win_load!(xr, W.normed, t)
                matmul!(cpu, aa, xa, bt.wgdn_a, wl)
                matmul!(cpu, ba, xa, bt.wgdn_b, wl)
                av, bv = vec(a), vec(b)
            end
            _gdn_conv_window_row!(convout, rec.mixed, conv_in, weight, t, ksz)
            _gdn_gates_row!(view(rec.beta, :, t), view(rec.decay, :, t), av, bv, A_log, dt_bias, nv)
            _gdn_qk_row!(
                view(rec.vn, :, t),
                view(rec.qn, :, t),
                view(rec.kn, :, t),
                convout,
                nk,
                nv,
                dk,
                dv,
                tiled,
            )
        end
        return nothing
    end
    _window_split_rows!(run_rows, T)
    return rec
end

# Matrices the window's blocks write into, sized for T rows and allocated once
# per window.
function _window_buffers(
    zl,
    T::Int,
    dim::Int,
    kdim::Int,
    qdim::Int,
    conv_dim::Int,
    value_dim::Int,
    hidden::Int,
    vocab::Int,
    head::Bool,
)
    return (;
        normed=zl((T, dim)),
        mixed=zl((T, conv_dim)),
        zc=zl((T, value_dim)),
        gdn_merged=zl((T, value_dim)),
        q=zl((T, qdim)),
        k=zl((T, kdim)),
        v=zl((T, kdim)),
        gate=zl((T, qdim)),
        attn_merged=zl((T, qdim)),
        sub=zl((T, dim)),
        normed2=zl((T, dim)),
        ffn_gate=zl((T, hidden)),
        ffn_up=zl((T, hidden)),
        ffn_act=zl((T, hidden)),
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
function _window_gdn_block!(
    bt,
    spec,
    hwin::AbstractMatrix,
    W,
    bufs,
    cpu,
    wl,
    zeros_like,
    conv_in,
    rec_in,
    sc,
    eps,
    cache=nothing,
    bi::Int=0,
)
    T = size(hwin, 1)
    sc === nothing && error(
        "verify window: a GDN block needs the CPU decode scratch (the window runs on the CPU path)",
    )
    _cpu_gdn_fast_ok(bt, conv_in, rec_in, bufs.hp) || error(
        "verify window: GDN block lacks the CPU fast path (F32 state and weights required)",
    )
    nv, dk, dv = sc.nv, sc.dk, sc.dv
    conv_dim = size(sc.mixed, 2)
    value_dim = nv * dv
    epsf = Float32(eps)
    ksz = spec.mixer.conv_kernel_dim
    cols = _window_reuse!(cache, (:gdn, bi), (conv_dim, value_dim, nv, dk, dv, T)) do
        (
            zeros(Float32, conv_dim, T),
            zeros(Float32, value_dim, T),
            zeros(Float32, nv * dk, T),
            zeros(Float32, nv * dk, T),
            zeros(Float32, nv * dv, T),
            zeros(Float32, nv, T),
            zeros(Float32, nv, T),
            zeros(Float32, nv * dv, T),
        )
    end
    # the state this block's last commit displaced, if it can take the new state
    spare = cache === nothing ? nothing : pop!(cache, (:spare, bi), nothing)
    rec_out =
        spare isa Array{Float32, 3} && size(spare) == size(rec_in) && spare !== rec_in ?
        fill!(spare, 0.0f0) : zeros(Float32, size(rec_in))
    rec = _GDNWindowRec(
        cols...,
        rec_out,
        sc.nslab,
        nv,
        dk,
        dv,
    )
    @decode_sub :gdn_norm cpu _window_rmsnorm!(cpu, W.normed, hwin, bufs.hp, bufs.normed, bt.attn_rms, wl; eps)
    @decode_sub :gdn_proj cpu begin
        _window_matmul!(cpu, W.mixed, W.normed, bt.wgdn_qkv, wl, zeros_like)
        _window_matmul!(cpu, W.zc, W.normed, bt.wgdn_z, wl, zeros_like)
    end

    @decode_sub :gdn_rows cpu _window_gdn_rows!(rec, W, conv_in, bt, spec, sc, bufs, cpu, wl, zeros_like, ksz)

    @decode_sub :gdn_recur cpu _pool_run_chunks!(
        _GDNWindowChunk(rec_in, rec.rec_out, rec, T),
        sc.nslab,
    )

    scale_norm = bt.wgdn_norm.storage::Vector{Float32}
    @decode_sub :gdn_gnorm cpu _window_split_rows!(T) do rows
        for t in rows
            _gdn_gnorm_merge!(
                view(W.gdn_merged, t, :),
                nv,
                dv,
                view(rec.core, :, t),
                view(rec.z, :, t),
                scale_norm,
                epsf,
            )
        end
    end
    @decode_sub :gdn_outproj cpu _window_matmul!(cpu, W.sub, W.gdn_merged, bt.wgdn_o, wl, zeros_like)
    return rec
end

# One attention block for the window. The norm and the q, k, v and gate
# projections run over all rows. The per-row core then appends each row's KV
# at its own cache row and attends over the rows up to it, so row t never reads
# a cache row after `base+t-1`. The output projection runs over all rows.
function _window_attn_block!(
    bt,
    bi::Int,
    kc,
    vc,
    hwin::AbstractMatrix,
    W,
    bufs,
    cpu,
    wl,
    tensors,
    zeros_like,
    cpu_scratch,
    base::Int;
    n_heads::Int,
    n_kv_heads::Int,
    d_head::Int,
    group::Int,
    eps,
    theta,
    inv_freq,
    rotary_dim::Int=0,
)
    T = size(hwin, 1)
    gated = haskey(bt, :wattn_gate)
    @decode_sub :attn_proj cpu begin
        _window_rmsnorm!(cpu, W.normed, hwin, bufs.hp, bufs.normed, bt.attn_rms, wl; eps)
        _window_matmul!(cpu, W.q, W.normed, bt.wq, wl, zeros_like)
        _window_matmul!(cpu, W.k, W.normed, bt.wk, wl, zeros_like)
        _window_matmul!(cpu, W.v, W.normed, bt.wv, wl, zeros_like)
        gated && _window_matmul!(cpu, W.gate, W.normed, bt.wattn_gate, wl, zeros_like)
    end

    if cpu_scratch !== nothing && T > 1 && _window_attn_batch_ok(bt, W, bufs)
        @decode_sub :attn_rows cpu _window_attn_rows_cpu!(
            bt,
            bi,
            kc,
            vc,
            W,
            bufs,
            cpu,
            wl,
            tensors,
            zeros_like,
            cpu_scratch,
            base;
            n_heads,
            n_kv_heads,
            d_head,
            group,
            eps,
            theta,
            inv_freq,
            rotary_dim,
        )
        @decode_sub :attn_outproj cpu _window_matmul!(cpu, W.sub, W.attn_merged, bt.wo, wl, zeros_like)
        return W.sub
    end
    for t in 1:T
        _win_load!(bufs.q.storage, W.q, t)
        _win_load!(bufs.k.storage, W.k, t)
        _win_load!(bufs.v.storage, W.v, t)
        gated && _win_load!(bufs.gate_attn.storage, W.gate, t)
        if haskey(bt, :wattn_head_gate)
            _win_load!(bufs.normed.storage, W.normed, t)
            _head_gate_proj!(cpu, bufs.gate_head, bufs.normed, bt.wattn_head_gate, wl)
        end
        _decode_attn_core!(
            bt,
            bi,
            kc,
            vc,
            bufs,
            cpu,
            wl,
            tensors,
            zeros_like,
            cpu_scratch,
            base + t - 1;
            n_heads,
            n_kv_heads,
            d_head,
            group,
            eps,
            theta,
            inv_freq,
            rotary_dim,
        )
        _attn_merge_gate!(bt, bufs, cpu, n_heads, d_head)
        _win_store!(W.attn_merged, t, bufs.merged.storage)
    end
    _window_matmul!(cpu, W.sub, W.attn_merged, bt.wo, wl, zeros_like)
    return W.sub
end

# The CPU attention core for all rows of the window, in three passes. The heads
# of every row are split, normed and rotated in one batched call each, and the
# rows' KV is written at cache rows base..base+T-1. Then every (row, head) pair
# attends across the pool, row t over cache rows up to base+t-1. Last, the heads
# are merged and gated over all rows. Each batched kernel runs decode's per-row
# arithmetic on every row (the norm folds each (row, head) in decode's order, the
# rope angle depends only on the row's position), so the bits are decode's.
function _window_attn_rows_cpu!(
    bt,
    bi::Int,
    kc,
    vc,
    W,
    bufs,
    cpu,
    wl,
    tensors,
    zeros_like,
    cpu_scratch,
    base::Int;
    n_heads::Int,
    n_kv_heads::Int,
    d_head::Int,
    group::Int,
    eps,
    theta,
    inv_freq,
    rotary_dim::Int=0,
)
    T = size(W.q, 1)
    gated = haskey(bt, :wattn_gate)
    head_gated = haskey(bt, :wattn_head_gate)
    Qw = similar(bufs.qh.storage, T, n_heads, d_head)
    Kw = similar(bufs.kh.storage, T, n_kv_heads, d_head)
    Vw = similar(bufs.vh.storage, T, n_kv_heads, d_head)
    Aw = similar(bufs.attn.storage, T, n_heads, d_head)
    if head_gated
        GH = similar(bufs.gate_head.storage, T, n_heads)
        if _head_gate_host(GH) && _head_gate_host(W.normed) &&
           _head_gate_host(bt.wattn_head_gate.storage)
            _head_gate_proj!(GH, W.normed, bt.wattn_head_gate.storage)
        else
            _window_matmul!(cpu, GH, W.normed, bt.wattn_head_gate, wl, zeros_like)
        end
    end
    @decode_sub :attn_headnorm_rope cpu begin
        _split_heads!(Qw, W.q, n_heads, d_head)
        _split_heads!(Kw, W.k, n_kv_heads, d_head)
        _split_heads!(Vw, W.v, n_kv_heads, d_head)
        Qa = Activation(; shape=size(Qw), storage=Qw)
        Ka = Activation(; shape=size(Kw), storage=Kw)
        if haskey(bt, :wq_norm)
            rmsnorm!(cpu, Qa, Qa, bt.wq_norm, wl; eps)
            rmsnorm!(cpu, Ka, Ka, bt.wk_norm, wl; eps)
        end
        br = _block_rope(bt, theta, inv_freq, rotary_dim)
        rope!(
            cpu,
            Qa,
            Ka,
            collect((base-1):(base+T-2)),
            wl;
            theta=br.theta,
            inv_freq=br.inv_freq,
            interleaved=_tensors_rope_interleaved(tensors),
            rotary_dim=br.rotary_dim,
        )  # 0-based positions
    end
    @decode_sub :attn_kv_append cpu begin
        _kv_write_rows!(kc[bi].storage, Kw, base)
        _kv_write_rows!(vc[bi].storage, Vw, base)
    end
    kv_rot = _kv_rotated(kc[bi].storage)
    kv_rot && _kv_rotate!(Qw, true)
    @decode_sub :attn_core cpu _attention_window_cpu!(
        Aw,
        Qw,
        kc[bi].storage,
        vc[bi].storage,
        n_heads,
        d_head,
        group,
        base,
        eltype(cpu_scratch.probs);
        window=_block_window(bt),
    )
    kv_rot && _kv_rotate!(Aw, false)
    M = W.attn_merged
    _merge_heads!(M, Aw, n_heads, d_head)
    gated && _sigmoid_mul!(cpu, M, M, W.gate)
    head_gated && _head_gate_mul!(cpu, M, GH, d_head)
    return M
end

# The batched rows run in the window's element type end to end. Decode moves
# each row through its one-row buffers, so a buffer of another element type
# would round at a point the batched rows skip; such a window keeps decode's
# per-row core.
function _window_attn_batch_ok(bt, W, bufs)
    E = eltype(W.q)
    xs = (W.k, W.v, W.normed, W.attn_merged, bufs.q.storage, bufs.k.storage, bufs.v.storage,
        bufs.qh.storage, bufs.kh.storage, bufs.vh.storage, bufs.attn.storage, bufs.merged.storage)
    all(x -> eltype(x) == E, xs) || return false
    haskey(bt, :wattn_gate) &&
        (eltype(W.gate) == eltype(bufs.gate_attn.storage) == E || return false)
    haskey(bt, :wattn_head_gate) &&
        (eltype(bufs.gate_head.storage) == eltype(bufs.normed.storage) == E || return false)
    return W.q isa Matrix && W.attn_merged isa Matrix
end

# The FFN half of a block over all rows. `subT` is the mixer output for every
# row. Dense FFNs run gate, up and down over all rows, and so does a softmax MoE
# on GGUF expert slabs (`_window_moe_ffn!`). Any other MoE runs its rows through
# the decode block.
function _window_ffn_block!(
    bt,
    hwin::AbstractMatrix,
    subT::AbstractMatrix,
    W,
    bufs,
    cpu,
    wl,
    zeros_like;
    eps,
)
    T = size(hwin, 1)
    if haskey(bt, :wrouter)
        if _window_moe_ok(bt, hwin, W, cpu)
            _window_moe_ffn!(bt, hwin, subT, W, bufs, cpu, wl, zeros_like; eps)
            return hwin
        end
        for t in 1:T
            _win_load!(bufs.hp.storage, hwin, t)
            _win_load!(bufs.sub.storage, subT, t)
            _decode_ffn_block!(bt, bufs, cpu, wl, zeros_like; eps)
            _win_store!(hwin, t, bufs.hp.storage)
        end
        return hwin
    end
    @decode_stage :dense_ffn cpu _window_dense_ffn!(bt, hwin, subT, W, bufs, cpu, wl, zeros_like; eps)
    return hwin
end

# The window MoE path needs host Float32 rows, softmax routing and GGUF expert
# slabs. Biased or sigmoid routers (LFM) and other banks keep the row loop.
function _window_moe_ok(bt, hwin, W, cpu)
    cpu isa CPUBackend || return false
    haskey(bt, :wrouter_bias) && return false
    haskey(bt, :router_gating) && bt.router_gating === :sigmoid && return false
    hwin isa Matrix{Float32} && W.normed2 isa Matrix{Float32} &&
        W.ffn_down isa Matrix{Float32} || return false
    bt.wrouter.storage isa AbstractMatrix || return false
    all(k -> haskey(bt, k), (:wexp_gate, :wexp_up, :wexp_down)) || return false
    return bt.wexp_gate.storage isa GgufQuantSlice &&
           bt.wexp_up.storage isa GgufQuantSlice &&
           bt.wexp_down.storage isa GgufQuantSlice
end

# `_moe_ffn!` over all rows. The routed experts take the seq > 1 host route,
# whose routing and expert sums match the one-token route op for op
# (test_moe_host_prefill.jl). The shared expert's products go through
# `_window_matmul!`, and its swiglu and sigmoid gate are per element and per row,
# so every row gets decode's bits.
function _window_moe_ffn!(bt, hwin, subT, W, bufs, cpu, wl, zeros_like; eps)
    T, h = size(hwin)
    _add_storage!(hwin, subT)
    _window_rmsnorm!(cpu, W.normed2, hwin, bufs.hp, bufs.normed2, bt.ffn_rms, wl; eps)
    x = W.normed2
    y = W.ffn_down
    n_used =
        haskey(bt, :n_expert_used) ? Int(bt.n_expert_used) : size(bt.wrouter.storage, 1)
    shared = haskey(bt, :wshared_gate)
    sd = shared ? zeros_like((T, h)) : nothing
    if shared
        @decode_stage :shared_expert cpu begin
            ff = size(bt.wshared_gate.storage, 1)
            sg = zeros_like((T, ff))
            su = zeros_like((T, ff))
            sa = zeros_like((T, ff))
            _window_matmul!(cpu, sg, x, bt.wshared_gate, wl, zeros_like)
            _window_matmul!(cpu, su, x, bt.wshared_up, wl, zeros_like)
            swiglu!(
                cpu,
                Activation(; shape=(T, ff), storage=sa),
                Activation(; shape=(T, ff), storage=sg),
                Activation(; shape=(T, ff), storage=su),
                wl,
            )
            _window_matmul!(cpu, sd, sa, bt.wshared_down, wl, zeros_like)
        end
    end
    _moe_topk_swiglu!(
        cpu,
        y,
        x,
        bt.wrouter.storage,
        bt.wexp_gate.storage,
        bt.wexp_up.storage,
        bt.wexp_down.storage,
        n_used,
    )
    if shared
        @decode_stage :shared_expert cpu begin
            if haskey(bt, :wshared_gate_inp)
                _shared_gate_add!(cpu, y, x, sd, bt.wshared_gate_inp.storage)
            else
                y .+= sd
            end
        end
    end
    _add_storage!(hwin, y)
    return hwin
end

# The dense half of `_window_ffn_block!`, timed as one :dense_ffn stage.
function _window_dense_ffn!(bt, hwin, subT, W, bufs, cpu, wl, zeros_like; eps)
    T = size(hwin, 1)
    _add_storage!(hwin, subT)
    _window_rmsnorm!(cpu, W.normed2, hwin, bufs.hp, bufs.normed2, bt.ffn_rms, wl; eps)
    @decode_sub :ffn_gateup cpu begin
        _window_matmul!(cpu, W.ffn_gate, W.normed2, bt.wgate, wl, zeros_like)
        _window_matmul!(cpu, W.ffn_up, W.normed2, bt.wup, wl, zeros_like)
    end
    act = Activation(; shape=size(W.ffn_act), storage=W.ffn_act)
    gate = Activation(; shape=size(W.ffn_gate), storage=W.ffn_gate)
    up = Activation(; shape=size(W.ffn_up), storage=W.ffn_up)
    @decode_sub :ffn_act cpu _dec_ffn_act!(cpu, act, gate, up, haskey(bt, :ffn_geglu), wl)
    @decode_sub :ffn_down cpu _window_matmul!(cpu, W.ffn_down, W.ffn_act, bt.wdown, wl, zeros_like)
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
    # the softcap the decode head applies, so each row's argmax is decode's
    if _spec_has_feature(tensors, :logit_softcap)
        _apply_logit_softcap!(cpu, (storage=W.logits,), tensors)
    end
    return W.logits
end

# Runs `toks` (the anchor, then the drafts) at 1-based cache rows base .. base+T-1.
# Row t writes its KV row at base+t-1 and attends over rows 1 .. base+t-1 only.
# Nothing in `gdn_conv` or `gdn_rec` changes: the window works on copies, and
# `_commit_window!` decides what survives. With `head=false` the window skips the
# logits, which is how a prompt prefill uses it.
function _decode_window!(
    model,
    tensors,
    kc,
    vc,
    toks::AbstractVector{Int},
    base::Int,
    cpu,
    wl,
    gdn_conv,
    gdn_rec,
    cpu_scratch;
    vocab::Int,
    n_heads::Int,
    n_kv_heads::Int,
    d_head::Int,
    group::Int,
    eps,
    theta,
    inv_freq,
    rotary_dim::Int=0,
    head::Bool=true,
    taps::AbstractVector{Int}=Int[],
)
    T = length(toks)
    T >= 1 || error("verify window: empty window")
    base >= 1 || error("verify window: base row $base is before the first cache row")
    dim = model.embedding.dim
    kdim = n_kv_heads * d_head
    qdim = n_heads * d_head
    zeros_like = _decode_zeros_like(cpu, tensors)
    hidden_ffn = model.blocks[1].ffn.hidden
    bufs = _decode_row_buffers(
        cpu_scratch,
        zeros_like,
        dim,
        kdim,
        qdim,
        n_heads,
        n_kv_heads,
        d_head,
        hidden_ffn,
    )
    hwin = zeros_like((T, dim))
    embedding_lookup!(
        cpu,
        Activation(; shape=(T, dim), storage=hwin),
        tensors.embedding,
        collect(toks),
        wl,
    )
    conv_dim = cpu_scratch === nothing ? 0 : size(cpu_scratch.mixed, 2)
    value_dim = cpu_scratch === nothing ? 0 : cpu_scratch.nv * cpu_scratch.dv
    cache = _window_cache(cpu_scratch, T)
    wkey = (typeof(hwin), T, dim, kdim, qdim, conv_dim, value_dim, hidden_ffn, vocab, head)
    W = _window_reuse!(cache, :W, wkey) do
        _window_buffers(
            zeros_like,
            T,
            dim,
            kdim,
            qdim,
            conv_dim,
            value_dim,
            hidden_ffn,
            vocab,
            head,
        )
    end
    tapX = zeros(Float32, isempty(taps) ? 0 : T, dim * length(taps))
    _window_tap_copy!(tapX, taps, 0, hwin, dim)

    recs = Vector{Any}(nothing, length(model.blocks))
    for (bi, blk) in enumerate(model.blocks)
        bt = tensors.blocks[bi]
        if _block_mixer(bt) === :linear_attention
            recs[bi] = @decode_stage :gdn_mixer cpu _window_gdn_block!(
                bt,
                tensors.spec,
                hwin,
                W,
                bufs,
                cpu,
                wl,
                zeros_like,
                gdn_conv[bi],
                gdn_rec[bi],
                cpu_scratch,
                eps,
                cache,
                bi,
            )
            subT = W.sub
        else
            subT = @decode_stage :attn_mixer cpu _window_attn_block!(
                bt,
                bi,
                kc,
                vc,
                hwin,
                W,
                bufs,
                cpu,
                wl,
                tensors,
                zeros_like,
                cpu_scratch,
                base;
                n_heads,
                n_kv_heads,
                d_head,
                group,
                eps,
                theta,
                inv_freq,
                rotary_dim,
            )
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
        return _VerifyWindow(T, hwin, logits, preds, recs, tapX, cache)
    end
    return _VerifyWindow(T, hwin, W.logits, Int[], recs, tapX, cache)
end

# Rebuilds the conv state after the first `m` window inputs. The state holds the
# last ksz-1 inputs, newest in the last column; before the window's first input
# the old state supplies the columns.
function _gdn_conv_rebuild(conv_in::Matrix{Float32}, mixed::Matrix{Float32}, m::Int)
    channels, K = size(conv_in)
    out = similar(conv_in)
    @inbounds for p in 1:K, c in 1:channels
        idx = m - K + p
        out[c, p] = idx >= 1 ? mixed[c, idx] : conv_in[c, idx+K]
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
        # the displaced state is dead (checkpoints copy), so the next window
        # writes its state there
        win.cache === nothing || (win.cache[(:spare, bi)] = gdn_rec[bi])
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
# and hands back the correction: the argmax of the last committed row, which is
# the bits the caller's top-of-loop head would give for it, so the caller skips
# that second head pass. Returns `(steps, accepted, stopped, correction)`, with
# `correction == -1` when the round stopped.
function _verify_window_round!(
    model,
    tensors,
    kc,
    vc,
    cpu,
    wl,
    h,
    gdn_conv,
    gdn_rec,
    cpu_scratch,
    ids::Vector{Int},
    sids::Vector{Int},
    dids::Vector{Int},
    base::Int,
    steps::Int,
    cap::Int,
    on_token,
    stop_ids,
    eos_ids;
    on_rows=nothing,
    on_taps=nothing,
    taps=Int[],
    kw...,
)
    k = length(dids)
    toks = Vector{Int}(undef, k + 1)
    toks[1] = ids[end]
    toks[2:end] .= dids
    win = _decode_window!(
        model,
        tensors,
        kc,
        vc,
        toks,
        base,
        cpu,
        wl,
        gdn_conv,
        gdn_rec,
        cpu_scratch;
        taps,
        kw...,
    )
    # row i predicts the token after toks[i], so it checks draft d_i
    n = 0
    while n < k && win.preds[n+1] == dids[n+1]
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
            return (steps, i, true, -1)
        end
    end
    _commit_window!(win, n + 1, gdn_conv, gdn_rec, h)
    _window_rows!(on_rows, tensors, win, base, n + 1, cpu, wl; kw...)
    _window_taps!(on_taps, win, base, n + 1)
    append!(sids, @view toks[1:(n+1)])
    return (steps, n, false, win.preds[n+1])
end
