# MTP (nextn) draft head for the Qwen3.5 family, after llama.cpp's
# `common_speculative_impl_draft_mtp` and the qwen35 nextn graph.
#
# The head is one full-attention block with its own KV cache. Its row r reads
# token ids[r] and the target's final-normed hidden row of ids[r-1] (zeros for
# r = 1): rmsnorm(emb) with `enorm` and rmsnorm(h) with `hnorm`, concatenated
# [e; h], then `eh_proj` back to the model width. The block runs as a verify
# window row, `shared_head_norm` (the head's final_rms) gives the row h_nextn,
# and the lm_head reads it. A draft chains: the next draft row reads the drafted
# id and the previous draft row's h_nextn.
#
# The target's rows arrive through `reference_generate(...; on_rows)`; see
# `mtp_rows!`. Draft rows past the last committed position are scratch and are
# rewritten with target rows on the next call.

mutable struct MTPDraft
    model
    tensors          # embedding, blocks (one), lm_head, final_rms = shared_head_norm, rope, spec
    eh_proj          # ProjectionWeight (dim, 2dim)
    enorm            # FrozenParameter (dim)
    hnorm            # FrozenParameter (dim)
    kc::Vector{Any}
    vc::Vector{Any}
    scratch
    zeros_like
    ctx::Int
    hn::Matrix{Float32}   # (ctx, dim): target final-normed row of ids[r] at row r
    have::Int             # hn rows 1..have are filled
    done::Int             # MTP cache rows 1..done hold rows paired with target rows
    eps::Float64
    theta
    inv_freq
    rotary_dim::Int
    d_head::Int
    n_heads::Int
    n_kv_heads::Int
    cpu
end

"""
    MTPDraft(model, tensors, eh_proj, enorm, hnorm; ctx, eps=1e-6)

An MTP draft head over a one-block model. `tensors.final_rms` is the head's
`shared_head_norm`. The KV cache and scratch hold `ctx` rows.
"""
function MTPDraft(model, tensors, eh_proj, enorm, hnorm; ctx::Int, eps::Real=1e-6,
    cpu=CPUBackend())
    length(tensors.blocks) == 1 || error("MTPDraft: the head is one block, got $(length(tensors.blocks))")
    _block_mixer(tensors.blocks[1]) === :linear_attention &&
        error("MTPDraft: the head block must be full attention")
    dim = model.embedding.dim
    n_heads = model.blocks[1].attention.n_heads
    n_kv_heads = model.blocks[1].attention.n_kv_heads
    d_head = _tensor_head_dim(model, tensors)
    theta = _resolved_rope_theta(tensors, nothing)
    vocab = size(tensors.embedding.storage, 1)
    zl = _decode_zeros_like(cpu, tensors)
    kc = Any[Activation(; shape=(ctx, n_kv_heads, d_head), storage=zl((ctx, n_kv_heads, d_head)))]
    vc = Any[Activation(; shape=(ctx, n_kv_heads, d_head), storage=zl((ctx, n_kv_heads, d_head)))]
    scratch = _cpu_decode_scratch(nothing, dim; qdim=n_heads * d_head, ff=_dense_ffn_hidden(model),
        kdim=n_kv_heads * d_head, n_heads, n_kv_heads, d_head, vocab, ctx,
        row=_decode_row_type(nothing, tensors))
    return MTPDraft(model, tensors, eh_proj, enorm, hnorm, kc, vc, scratch, zl, ctx,
        zeros(Float32, ctx, dim), 0, 0, Float64(eps), theta,
        tensors_rope_inv_freq(tensors, d_head; theta), _tensor_rotary_dim(tensors),
        d_head, n_heads, n_kv_heads, cpu)
end

"""
    load_mtp_draft(path; ctx, target=nothing, T=Float32) -> MTPDraft

Loads the nextn block of a qwen35 GGUF as a draft head. The file may be a full
model or a head-only extract; either way its last block holds the nextn
tensors. With `target` (a `load_gguf` tensors tuple) the head uses the
target's embedding and lm_head instead of the file's.
"""
function load_mtp_draft(path::AbstractString; ctx::Int, target=nothing, T::Type=Float32)
    hdr = gguf_header(path)
    arch = String(get(hdr.kv, "general.architecture", ""))
    arch == "qwen35" || error("load_mtp_draft: arch $(repr(arch)) is not qwen35")
    nextn = Int(get(hdr.kv, "$arch.nextn_predict_layers", 0))
    nextn == 1 || error("load_mtp_draft: nextn_predict_layers = $nextn, expected 1")
    src_blk = Int(hdr.kv["$arch.block_count"]) - 1
    # a one-layer full-attention spec: block 0 here is the file's nextn block
    kv = copy(hdr.kv)
    kv["$arch.block_count"] = 1
    kv["$arch.nextn_predict_layers"] = 0
    kv["$arch.full_attention_interval"] = 1
    info_by = Dict(info.name => info for info in hdr.tensors)
    spec = spec_from_gguf_kv(kv; tensor_names=Set(keys(info_by)))
    pmap = _gguf_param_map(spec)
    source(name) =
        name == "output_norm.weight" ? "blk.$src_blk.nextn.shared_head_norm.weight" :
        startswith(name, "blk.0.") ? "blk.$src_blk." * name[7:end] : name
    by = Dict{String,Any}()
    for name in _required_gguf_names(pmap, spec)
        s = source(name)
        haskey(info_by, s) || error("load_mtp_draft: missing tensor $s")
        by[name] = _materialize_gguf_tensor(hdr, info_by[s], spec, T)
    end
    bound = materialize_architecture(spec, pmap, by)
    model = spec_to_model(spec)
    emb, head = bound.embedding, bound.lm_head
    if target !== nothing
        size(target.embedding.storage) == size(emb.storage) ||
            error("load_mtp_draft: target embedding $(size(target.embedding.storage)) ≠ head $(size(emb.storage))")
        emb, head = target.embedding, target.lm_head
    end
    tensors = (embedding=emb, blocks=bound.blocks, lm_head=head, final_rms=bound.final_rms,
        rope=bound.rope, bindings=bound.bindings, spec=spec, gguf_path=String(path), mtp=nothing)
    nx(n) = (s = "blk.$src_blk.nextn.$n"; haskey(info_by, s) ? s : error("load_mtp_draft: missing tensor $s"))
    eh = _materialize_gguf_tensor(hdr, info_by[nx("eh_proj.weight")], spec, T)
    norm(n) = (a = _materialize_gguf_tensor(hdr, info_by[nx(n)], spec, T);
        FrozenParameter(; shape=size(a), storage=a))
    return MTPDraft(model, tensors, ProjectionWeight(; shape=size(eh), storage=eh),
        norm("enorm.weight"), norm("hnorm.weight"); ctx, eps=spec.rms_norm_eps)
end

"""
    mtp_rows!(m, pos, M)

Takes target rows for ids `pos .. pos+n-1` (the `on_rows` hook of
`reference_generate`). Head row r reads target row r-1, so head rows after
`pos` are recomputed on the next draft.
"""
function mtp_rows!(m::MTPDraft, pos::Int, M::AbstractMatrix)
    n = size(M, 1)
    pos + n - 1 <= m.ctx || error("mtp_rows!: rows to $(pos + n - 1) exceed the head's ctx $(m.ctx)")
    pos <= m.have + 1 || error("mtp_rows!: row $pos leaves a gap after row $(m.have)")
    m.hn[pos:(pos + n - 1), :] .= M
    m.have = pos + n - 1
    m.done = min(m.done, pos)
    return m
end

# Runs head rows base .. base+T-1 for tokens `toks` and the (T, dim) target or
# chained rows `H`. Returns (h_nextn rows, logits of the last row).
function _mtp_window!(m::MTPDraft, toks::AbstractVector{Int}, H::AbstractMatrix, base::Int)
    (; model, tensors, cpu, zeros_like) = m
    wl = DecodeWorkload()
    T = length(toks)
    dim = model.embedding.dim
    kdim = m.n_kv_heads * m.d_head
    qdim = m.n_heads * m.d_head
    hidden = model.blocks[1].ffn.hidden
    vocab = size(tensors.embedding.storage, 1)
    bufs = _decode_row_buffers(m.scratch, zeros_like, dim, kdim, qdim, m.n_heads, m.n_kv_heads,
        m.d_head, hidden)
    E = zeros_like((T, dim))
    embedding_lookup!(cpu, Activation(; shape=(T, dim), storage=E), tensors.embedding, collect(toks), wl)
    X = zeros_like((T, 2dim))
    for t in 1:T
        _win_load!(bufs.hp.storage, E, t)
        rmsnorm!(cpu, bufs.normed, bufs.hp, m.enorm, wl; eps=m.eps)
        @views X[t, 1:dim] .= bufs.normed.storage[1, :]
        _win_load!(bufs.hp.storage, H, t)
        rmsnorm!(cpu, bufs.normed, bufs.hp, m.hnorm, wl; eps=m.eps)
        @views X[t, (dim + 1):(2dim)] .= bufs.normed.storage[1, :]
    end
    hwin = zeros_like((T, dim))
    _window_matmul!(cpu, hwin, X, m.eh_proj, wl, zeros_like)
    W = _window_buffers(zeros_like, T, dim, kdim, qdim, 0, 0, hidden, vocab, false)
    bt = tensors.blocks[1]
    subT = _window_attn_block!(bt, 1, m.kc, m.vc, hwin, W, bufs, cpu, wl, tensors, zeros_like,
        m.scratch, base; n_heads=m.n_heads, n_kv_heads=m.n_kv_heads, d_head=m.d_head,
        group=div(m.n_heads, m.n_kv_heads), eps=m.eps, theta=m.theta, inv_freq=m.inv_freq,
        rotary_dim=m.rotary_dim)
    _window_ffn_block!(bt, hwin, subT, W, bufs, cpu, wl, zeros_like; eps=m.eps)
    for t in 1:T
        _win_load!(bufs.hp.storage, hwin, t)
        rmsnorm!(cpu, bufs.normed2, bufs.hp, tensors.final_rms, wl; eps=m.eps)
        _win_store!(W.hn, t, bufs.normed2.storage)
    end
    last = W.hn[T:T, :]
    logits = zeros_like((1, vocab))
    _window_matmul!(cpu, logits, last, tensors.lm_head, wl, zeros_like)
    return W.hn, logits
end

# Greedy id (0-based) and its probability under a softmax over the top `k`
# logits, which is what llama.cpp's top-k 10 draft sampler reports.
function _mtp_pick(logits::AbstractVector; k::Int=10)
    row = Array(logits)
    top = partialsortperm(row, 1:min(k, length(row)); rev=true)
    hi = Float64(row[top[1]])
    total = 0.0
    for i in top
        total += exp(Float64(row[i]) - hi)
    end
    return top[1] - 1, 1 / total
end

"""
    mtp_draft!(m, ids, k; p_min=0.0) -> Vector{Int}

Up to `k` draft ids after `ids[end]`, which the target has emitted but not
consumed, so target rows must cover ids `1 .. length(ids)-1`. The first head
pass catches the cache up over every row since the last call; each later draft
is one chained row. With `p_min > 0` drafting stops before the first id whose
top-10 probability is below it.
"""
function mtp_draft!(m::MTPDraft, ids::AbstractVector{Int}, k::Int; p_min::Real=0.0)
    L = length(ids)
    out = Int[]
    k <= 0 && return out
    m.have >= L - 1 || error("mtp_draft!: target rows end at $(m.have), need $(L - 1)")
    L + k - 1 <= m.ctx || (k = m.ctx - L + 1)
    k <= 0 && return out
    # row L always reruns: its h_nextn and logits start the chain
    r0 = min(m.done + 1, L)
    H = zeros(Float32, L - r0 + 1, size(m.hn, 2))
    for r in r0:L
        r >= 2 && (H[r - r0 + 1, :] .= @view m.hn[r - 1, :])
    end
    hrows, logits = _mtp_window!(m, view(ids, r0:L), H, r0)
    m.done = L
    hprev = hrows[end:end, :]
    for i in 1:k
        id, p = _mtp_pick(vec(logits))
        p_min > 0 && p < p_min && break
        push!(out, id)
        i == k && break
        hrows, logits = _mtp_window!(m, [id], Float32.(hprev), L + i)
        hprev = hrows[end:end, :]
    end
    return out
end

"""
    reset!(m::MTPDraft)

Forgets every target row, for a new conversation.
"""
function reset!(m::MTPDraft)
    m.have = 0
    m.done = 0
    return m
end
