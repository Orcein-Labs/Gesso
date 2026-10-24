# DFlash block drafter (llama.cpp's "dflash" arch) for speculative decoding of
# a Qwen3.8 target. The drafter reads target features, not token rows.
#
# Context: the target's hidden rows at the drafter's `target_layers` arrive as
# one (n, n_target*dim) matrix per id. `fc` projects them to the model width and
# `enc.output_norm` norms the result into g. Every draft layer keeps K and V
# caches of g: slot p holds rope(k_norm(wk·g)) and wv·g. `dflash_rows!` writes
# them as the target consumes ids. They are the drafter's only persistent state.
#
# A draft is one non-causal block [ids[L], mask, ..., mask]. Each layer runs an
# attention norm, a dynamic depthwise conv (side 0 before attention, side 1
# after), attention over the context plus the whole block, and a SwiGLU FFN with
# the same conv pair. The block's own K and V are temporaries, so the caches
# never hold block rows. The target's lm_head reads block rows 2..B, and a top-k
# selector walks a predecessor/successor chain over those candidates.

mutable struct DFlashDraft
    target              # target tensors: embedding (EmbeddingTable), lm_head (ProjectionWeight), blocks
    layers::Vector{Any} # per draft block: NamedTuple of norms, projections and conv bases
    fc                  # ProjectionWeight (dim, n_target*dim)
    enc_norm            # FrozenParameter (dim): enc.output_norm, applied to fc's output
    output_norm         # FrozenParameter (dim): applied to the block before the lm_head
    sel_hidden          # ProjectionWeight (rank, dim): selector gate
    sel_pred            # EmbeddingTable (vocab_d, rank): predecessor rows
    sel_succ            # EmbeddingTable (vocab_d, rank): successor rows
    kc::Vector{Array{Float32,3}}   # per layer (d_head, n_kv, ctx): a cached head vector is contiguous
    vc::Vector{Array{Float32,3}}
    ctx::Int
    have::Int           # context rows 1..have are injected
    block_size::Int
    mask_token::Int
    conv_kernel::Int
    conv_group::Int
    top_k::Int
    rank::Int
    window::Int
    eps::Float64
    theta::Float64
    n_heads::Int
    n_kv_heads::Int
    d_head::Int
    cpu
    zeros_like
    target_layers::Vector{Int}  # target block counts whose residual rows feed `fc`, in order
end

"""
    DFlashDraft(target, layers, fc, enc_norm, output_norm, sel_hidden, sel_pred, sel_succ; ...)

A drafter over built weights. `target` is the target's tensors (its embedding
and lm_head are used). `layers` is a vector of NamedTuples with the fields
`attn_norm, attn_conv_proj, attn_conv_base, wq, wk, wv, q_norm, k_norm, wo,
ffn_norm, ffn_conv_proj, ffn_conv_base, gate, up, down`. The KV caches hold
`ctx` rows.
"""
function DFlashDraft(target, layers, fc, enc_norm, output_norm, sel_hidden, sel_pred, sel_succ;
    ctx::Int, block_size::Int, mask_token::Int, conv_kernel::Int=2, conv_group::Int,
    top_k::Int, rank::Int, window::Int, eps::Real=1e-6, theta::Real=1e7,
    n_heads::Int, n_kv_heads::Int, d_head::Int, target_layers::AbstractVector{<:Integer}=Int[],
    cpu=CPUBackend())
    ctx >= 1 || error("DFlashDraft: ctx must be positive, got $ctx")
    block_size >= 2 || error("DFlashDraft: block_size must be at least 2, got $block_size")
    isempty(layers) && error("DFlashDraft: no draft layers")
    dim = fc.shape[1]
    dim % conv_group == 0 || error("DFlashDraft: dim $dim is not a multiple of conv_group $conv_group")
    n_heads % n_kv_heads == 0 || error("DFlashDraft: n_heads $n_heads is not a multiple of n_kv_heads $n_kv_heads")
    qdim = n_heads * d_head
    kdim = n_kv_heads * d_head
    ng = div(dim, conv_group)
    sel_hidden.shape == (rank, dim) ||
        error("DFlashDraft: selector gate $(sel_hidden.shape) ≠ ($rank, $dim)")
    for (l, lay) in enumerate(layers)
        ffn = lay.gate.shape[1]
        lay.attn_conv_proj.shape == (2 * conv_kernel * ng, dim) ||
            error("DFlashDraft: layer $l attn_conv_proj $(lay.attn_conv_proj.shape) ≠ ($(2 * conv_kernel * ng), $dim)")
        lay.ffn_conv_proj.shape == (2 * conv_kernel * ng, dim) ||
            error("DFlashDraft: layer $l ffn_conv_proj $(lay.ffn_conv_proj.shape) ≠ ($(2 * conv_kernel * ng), $dim)")
        size(lay.attn_conv_base) == (dim, conv_kernel, 2) ||
            error("DFlashDraft: layer $l attn_conv_base $(size(lay.attn_conv_base)) ≠ ($dim, $conv_kernel, 2)")
        size(lay.ffn_conv_base) == (dim, conv_kernel, 2) ||
            error("DFlashDraft: layer $l ffn_conv_base $(size(lay.ffn_conv_base)) ≠ ($dim, $conv_kernel, 2)")
        lay.wq.shape == (qdim, dim) || error("DFlashDraft: layer $l wq $(lay.wq.shape) ≠ ($qdim, $dim)")
        lay.wk.shape == (kdim, dim) || error("DFlashDraft: layer $l wk $(lay.wk.shape) ≠ ($kdim, $dim)")
        lay.wv.shape == (kdim, dim) || error("DFlashDraft: layer $l wv $(lay.wv.shape) ≠ ($kdim, $dim)")
        lay.wo.shape == (dim, qdim) || error("DFlashDraft: layer $l wo $(lay.wo.shape) ≠ ($dim, $qdim)")
        lay.gate.shape == (ffn, dim) || error("DFlashDraft: layer $l gate $(lay.gate.shape) ≠ ($ffn, $dim)")
        lay.up.shape == (ffn, dim) || error("DFlashDraft: layer $l up $(lay.up.shape) ≠ ($ffn, $dim)")
        lay.down.shape == (dim, ffn) || error("DFlashDraft: layer $l down $(lay.down.shape) ≠ ($dim, $ffn)")
    end
    # the activation prototype walks `blocks` when the embedding is quantized
    zl = _decode_zeros_like(cpu, hasproperty(target, :blocks) ? target : merge(target, (; blocks=())))
    kc = [zeros(Float32, d_head, n_kv_heads, ctx) for _ in layers]
    vc = [zeros(Float32, d_head, n_kv_heads, ctx) for _ in layers]
    return DFlashDraft(target, collect(Any, layers), fc, enc_norm, output_norm, sel_hidden,
        sel_pred, sel_succ, kc, vc, ctx, 0, block_size, mask_token, conv_kernel, conv_group,
        top_k, rank, window, Float64(eps), Float64(theta), n_heads, n_kv_heads, d_head, cpu, zl,
        collect(Int, target_layers))
end

# Metadata lookup: the first spelling present wins. Fails listing the keys the
# file does have.
function _dflash_meta(kv, names::Tuple{Vararg{String}})
    for n in names
        haskey(kv, n) && return kv[n]
    end
    error("load_dflash_draft: missing metadata $(join(names, " or ")); keys present: " *
          join(sort!(collect(keys(kv))), ", "))
end

function _dflash_tensor(info_by, names::Tuple{Vararg{String}})
    for n in names
        haskey(info_by, n) && return info_by[n]
    end
    error("load_dflash_draft: missing tensor $(join(names, " or "))")
end

_dflash_proj_load(hdr, info) = (w = _quant_resident(info) ? _quant_matrix(hdr, info) : _gguf_array(hdr, info);
    ProjectionWeight(; shape=size(w), storage=w))

_dflash_table_load(hdr, info) = (w = _quant_resident(info) ? _quant_matrix(hdr, info) : _gguf_array(hdr, info);
    EmbeddingTable(; shape=size(w), storage=w))

_dflash_norm_load(hdr, info) = (a = _gguf_array(hdr, info); FrozenParameter(; shape=size(a), storage=a))

# Conv bases are 3-D F32. `_gguf_array` reverses the axes, so read the raw
# ggml order: base[c, tap, side], 1-based.
_dflash_conv_load(hdr, info) = reshape(
    _dequant_f32(hdr.data, Int(hdr.data_start) + Int(info.offset), info.ggml_type, prod(info.dims)),
    info.dims...)

"""
    load_dflash_draft(path; ctx::Int, target) -> DFlashDraft

Loads a dflash drafter GGUF. The drafter has no embedding or lm_head of its
own, so `target` (a `load_gguf` tensors tuple) supplies both.
"""
function load_dflash_draft(path::AbstractString; ctx::Int, target)
    hdr = gguf_header(path)
    kv = hdr.kv
    arch = String(get(kv, "general.architecture", ""))
    arch == "dflash" || error("load_dflash_draft: arch $(repr(arch)) is not dflash")
    hasproperty(target, :embedding) && hasproperty(target, :lm_head) ||
        error("load_dflash_draft: target must carry embedding and lm_head")
    meta(names::String...) = _dflash_meta(kv, names)
    nblk = Int(meta("dflash.block_count"))
    dim = Int(meta("dflash.embedding_length"))
    n_heads = Int(meta("dflash.attention.head_count", "dflash.head_count"))
    n_kv = Int(meta("dflash.attention.head_count_kv", "dflash.head_count_kv"))
    d_head = Int(meta("dflash.attention.key_length", "dflash.head_dim"))
    theta = Float64(meta("dflash.rope.freq_base"))
    eps = Float64(meta("dflash.attention.layer_norm_rms_epsilon"))
    block = Int(meta("dflash.block_size"))
    conv_k = Int(meta("dflash.conv_kernel_size"))
    conv_g = Int(meta("dflash.conv_group_size"))
    rank = Int(meta("dflash.selector_rank"))
    top_k = Int(meta("dflash.selector_top_k"))
    tl = meta("dflash.target_layers")
    tl isa Integer && error("load_dflash_draft: dflash.target_layers is a count, not a layer list")
    target_layers = collect(Int, tl)
    n_target = length(target_layers)
    window = Int(meta("dflash.attention.sliding_window", "dflash.sliding_window"))
    mask = Int(meta("tokenizer.ggml.mask_token_id", "dflash.mask_token_id"))

    info_by = Dict(info.name => info for info in hdr.tensors)
    need(names::String...) = _dflash_tensor(info_by, names)
    fc = _dflash_proj_load(hdr, need("fc.weight"))
    enc_norm = _dflash_norm_load(hdr, need("enc.output_norm.weight"))
    output_norm = _dflash_norm_load(hdr, need("output_norm.weight"))
    sel_hidden = _dflash_proj_load(hdr, need("selector_hidden.weight"))
    sel_pred = _dflash_table_load(hdr, need("selector_predecessor.weight"))
    sel_succ = _dflash_table_load(hdr, need("selector_successor.weight"))
    layers = Any[]
    for i in 0:(nblk - 1)
        p = "blk.$i."
        push!(layers, (
            attn_norm=_dflash_norm_load(hdr, need(p * "attn_norm.weight")),
            attn_conv_proj=_dflash_proj_load(hdr, need(p * "attn_conv_proj.weight")),
            attn_conv_base=_dflash_conv_load(hdr, need(p * "attn_conv_base.weight", p * "attn_conv_base")),
            wq=_dflash_proj_load(hdr, need(p * "attn_q.weight")),
            wk=_dflash_proj_load(hdr, need(p * "attn_k.weight")),
            wv=_dflash_proj_load(hdr, need(p * "attn_v.weight")),
            q_norm=_dflash_norm_load(hdr, need(p * "attn_q_norm.weight")),
            k_norm=_dflash_norm_load(hdr, need(p * "attn_k_norm.weight")),
            wo=_dflash_proj_load(hdr, need(p * "attn_output.weight")),
            ffn_norm=_dflash_norm_load(hdr, need(p * "ffn_norm.weight")),
            ffn_conv_proj=_dflash_proj_load(hdr, need(p * "ffn_conv_proj.weight")),
            ffn_conv_base=_dflash_conv_load(hdr, need(p * "ffn_conv_base.weight", p * "ffn_conv_base")),
            gate=_dflash_proj_load(hdr, need(p * "ffn_gate.weight")),
            up=_dflash_proj_load(hdr, need(p * "ffn_up.weight")),
            down=_dflash_proj_load(hdr, need(p * "ffn_down.weight")),
        ))
    end
    length(layers) == nblk || error("load_dflash_draft: expected $nblk blocks")
    # the feature matrix is the target layers' rows side by side
    fc.shape[2] == n_target * dim ||
        error("load_dflash_draft: fc in-width $(fc.shape[2]) ≠ $n_target target layers × $dim")
    return DFlashDraft(target, layers, fc, enc_norm, output_norm, sel_hidden, sel_pred, sel_succ;
        ctx, block_size=block, mask_token=mask, conv_kernel=conv_k, conv_group=conv_g,
        top_k, rank, window, eps, theta, n_heads, n_kv_heads=n_kv, d_head, target_layers)
end

# --- per-vector and per-row helpers -----------------------------------------

# y = x / rms(x) * w. In place when y === x.
function _df_rmsnorm!(y::AbstractVector, x::AbstractVector, w::AbstractVector, eps::Real)
    scale = inv(sqrt(sum(abs2, x) / length(x) + eps))
    @inbounds for i in eachindex(x)
        y[i] = x[i] * scale * w[i]
    end
    return y
end

function _df_rmsnorm_rows(X::AbstractMatrix, w::FrozenParameter, eps::Real)
    Y = similar(X)
    for r in axes(X, 1)
        _df_rmsnorm!(view(Y, r, :), view(X, r, :), w.storage, eps)
    end
    return Y
end

# NeoX (half-split) rotary embedding of one head vector at position `pos`.
function _dflash_rope!(x::AbstractVector, pos::Integer, theta::Real)
    d = length(x)
    half = div(d, 2)
    for i in 1:half
        ang = pos * Float64(theta)^(-2 * (i - 1) / d)
        c = cos(ang)
        s = sin(ang)
        a = x[i]
        b = x[i + half]
        x[i] = a * c - b * s
        x[i + half] = a * s + b * c
    end
    return x
end

# RMS-norms every head of every row of X in place, then ropes it. Row r sits at
# position pos0 + r - 1.
function _dflash_heads!(X::AbstractMatrix, nrm::FrozenParameter, n_h::Int, d::Int, pos0::Int,
    eps::Real, theta::Real)
    for r in axes(X, 1), h in 1:n_h
        seg = view(X, r, ((h - 1) * d + 1):(h * d))
        _df_rmsnorm!(seg, seg, nrm.storage, eps)
        _dflash_rope!(seg, pos0 + r - 1, theta)
    end
    return X
end

# y = X * W' over all rows, with the drafter's workload and zeros.
function _dflash_proj(m::DFlashDraft, X::AbstractMatrix, w)
    dst = zeros(Float32, size(X, 1), w.shape[1])
    _window_matmul!(m.cpu, dst, X, w, DecodeWorkload(), m.zeros_like)
    return dst
end

# Rows of an embedding-shaped table for token ids (0-based).
function _dflash_table_rows(m::DFlashDraft, table::EmbeddingTable, toks::AbstractVector{Int})
    E = zeros(Float32, length(toks), table.shape[2])
    embedding_lookup!(m.cpu, Activation(; shape=size(E), storage=E), table, collect(Int, toks),
        DecodeWorkload())
    return E
end

# The dynamic depthwise conv of one block, `side` 0 or 1. Channel c of group
# grp = (c-1) ÷ G takes its taps from dyn[i, 1 + grp + n_groups*(a + K*side)]
# plus the static base. Row i reads rows i-a for a in 0:K-1, and rows before
# the block's first row are zero.
function _dflash_conv(hid::AbstractMatrix, dyn::AbstractMatrix, base::AbstractArray{<:Any,3},
    side::Int, G::Int)
    B, dim = size(hid)
    K = size(base, 2)
    ng = div(dim, G)
    out = zeros(Float32, B, dim)
    for i in 1:B, c in 1:dim
        grp = div(c - 1, G)
        acc = 0.0f0
        for a in 0:(min(K, B) - 1)
            i - a >= 1 || break
            w = dyn[i, 1 + grp + ng * (a + K * side)] + base[c, a + 1, side + 1]
            acc += w * hid[i - a, c]
        end
        out[i, c] = acc
    end
    return out
end

# Indices (1-based) of the k largest entries, highest first. Ties go to the
# lower index, so the 0-based ids come out in the same order as the logits.
function _dflash_top_k(lg::AbstractVector, k::Int)
    k = min(k, length(lg))
    k <= 0 && return Int[]
    top = Int[]
    for i in eachindex(lg)
        v = lg[i]
        if length(top) == k
            v > lg[top[end]] || continue
            pop!(top)
        end
        p = length(top) + 1
        while p > 1 && lg[top[p - 1]] < v
            p -= 1
        end
        insert!(top, p, i)
    end
    return top .- 1
end

# Selector scores for each draft row i: S[k, j] = dot(succ_k, pred_j .* gate_i)
# + unary_k. cand[i] are the row's candidate ids, unary[i] their logits, and
# pred_rows[i] the predecessor rows (one for row 1, the previous row's
# candidates after that).
function _dflash_select(cand, unary, gate, succ_rows, pred_rows)
    scores = Matrix{Float32}[]
    for i in eachindex(cand)
        K = length(cand[i])
        P = size(pred_rows[i], 1)
        S = zeros(Float32, K, P)
        for k in 1:K, j in 1:P
            s = 0.0
            for r in eachindex(gate[i])
                s += succ_rows[i][k, r] * pred_rows[i][j, r] * gate[i][r]
            end
            S[k, j] = s + unary[i][k]
        end
        push!(scores, S)
    end
    return scores
end

# Greedy walk: column j of each row's scores picks the candidate, and the pick
# becomes the next column. With p_min > 0 the walk stops before a pick whose
# softmax probability over the row falls below it. Fewer than n_min picks
# returns nothing.
function _dflash_walk(cands, scores; p_min::Real=0.0, n_min::Int=0)
    out = Int[]
    j = 1
    for i in eachindex(cands)
        s = view(scores[i], :, j)
        kbest = argmax(s)
        if p_min > 0
            hi = maximum(s)
            total = 0.0
            for x in s
                total += exp(Float64(x) - Float64(hi))
            end
            1 / total < p_min && break
        end
        push!(out, cands[i][kbest])
        j = kbest
    end
    return length(out) < n_min ? Int[] : out
end

# Block forward. Returns the normed block rows (B, dim) and the logits of block
# rows 2..B (n_draft, vocab). The persistent caches are read, never written.
function _dflash_block(m::DFlashDraft, ids::AbstractVector{Int}, n_draft::Int)
    (; target, cpu, eps) = m
    L = length(ids)
    B = n_draft + 1
    toks = vcat(ids[L], fill(m.mask_token, n_draft))
    inpL = _dflash_table_rows(m, target.embedding, toks)
    for (l, lay) in enumerate(m.layers)
        nn = _df_rmsnorm_rows(inpL, lay.attn_norm, eps)
        ad = _dflash_proj(m, nn, lay.attn_conv_proj)
        nn = _dflash_conv(nn, ad, lay.attn_conv_base, 0, m.conv_group)
        q = _dflash_proj(m, nn, lay.wq)
        kb = _dflash_proj(m, nn, lay.wk)
        vb = _dflash_proj(m, nn, lay.wv)
        _dflash_heads!(q, lay.q_norm, m.n_heads, m.d_head, L - 1, eps, m.theta)
        _dflash_heads!(kb, lay.k_norm, m.n_kv_heads, m.d_head, L - 1, eps, m.theta)
        att = _dflash_block_attn(m, l, q, kb, vb, L)
        o = _dflash_proj(m, att, lay.wo)
        o = _dflash_conv(o, ad, lay.attn_conv_base, 1, m.conv_group)
        ffn_inp = o .+ inpL
        x = _df_rmsnorm_rows(ffn_inp, lay.ffn_norm, eps)
        fd = _dflash_proj(m, x, lay.ffn_conv_proj)
        x = _dflash_conv(x, fd, lay.ffn_conv_base, 0, m.conv_group)
        gt = _dflash_proj(m, x, lay.gate)
        up = _dflash_proj(m, x, lay.up)
        act = gt .* (1 ./ (1 .+ exp.(-gt))) .* up
        y = _dflash_proj(m, act, lay.down)
        y = _dflash_conv(y, fd, lay.ffn_conv_base, 1, m.conv_group)
        inpL = y .+ ffn_inp
    end
    t = _df_rmsnorm_rows(inpL, m.output_norm, eps)
    logits = zeros(Float32, n_draft, target.lm_head.shape[1])
    _window_matmul!(cpu, logits, t[2:B, :], target.lm_head, DecodeWorkload(), m.zeros_like)
    return t, logits
end

# Attention for the block's B query rows over the context rows 1..L-1 and all B
# block rows. Block row i sits at rope position L+i-2, and a context row r at
# r-1. A context row is kept while its distance to the query is below the
# sliding window. The block is not causal. Each (row, head) pair is one job;
# large calls split the jobs across threads.
function _dflash_block_attn(m::DFlashDraft, l::Int, q::AbstractMatrix, kb::AbstractMatrix,
    vb::AbstractMatrix, L::Int; threaded::Union{Nothing,Bool}=nothing)
    B = size(q, 1)
    d = m.d_head
    nh = m.n_heads
    nkv = m.n_kv_heads
    group = div(nh, nkv)
    scale = 1 / sqrt(d)
    kc = m.kc[l]
    vc = m.vc[l]
    # head vectors as contiguous columns, (d_head, heads, B)
    qt = reshape(permutedims(Float32.(q)), d, nh, B)
    kt = reshape(permutedims(Float32.(kb)), d, nkv, B)
    vt = reshape(permutedims(Float32.(vb)), d, nkv, B)
    outt = zeros(Float32, d, nh, B)
    nmax = max(L - 1, 0) + B
    jobs = B * nh
    function run(js)
        sc = Vector{Float64}(undef, nmax)
        @inbounds for job in js
            i = div(job - 1, nh) + 1
            hd = mod(job - 1, nh) + 1
            kvh = div(hd - 1, group) + 1
            qp = L + i - 2
            r_lo = max(1, qp - m.window + 2)
            nk = 0
            for r in r_lo:(L - 1)
                nk += 1
                s = 0.0f0
                @simd for e in 1:d
                    s += qt[e, hd, i] * kc[e, kvh, r]
                end
                sc[nk] = s * scale
            end
            for j in 1:B
                nk += 1
                s = 0.0f0
                @simd for e in 1:d
                    s += qt[e, hd, i] * kt[e, kvh, j]
                end
                sc[nk] = s * scale
            end
            mx = maximum(view(sc, 1:nk))
            tot = 0.0
            for z in 1:nk
                sc[z] = exp(sc[z] - mx)
                tot += sc[z]
            end
            z = 0
            for r in r_lo:(L - 1)
                z += 1
                w = Float32(sc[z] / tot)
                @simd for e in 1:d
                    outt[e, hd, i] += w * vc[e, kvh, r]
                end
            end
            for j in 1:B
                z += 1
                w = Float32(sc[z] / tot)
                @simd for e in 1:d
                    outt[e, hd, i] += w * vt[e, kvh, j]
                end
            end
        end
        return nothing
    end
    if something(threaded, _attention_rows_threaded(jobs, nmax * d))
        _attention_rows!(run, jobs)
    else
        run(1:jobs)
    end
    return permutedims(reshape(outt, nh * d, B))
end

"""
    dflash_rows!(m, pos, X)

Takes target features for ids `pos .. pos+n-1`, with X an (n, n_target*dim)
matrix. Each row is normed into the context and written into every layer's K
and V cache. Same bookkeeping as `mtp_rows!`: gaps and overflow are errors.
"""
function dflash_rows!(m::DFlashDraft, pos::Int, X::AbstractMatrix)
    n = size(X, 1)
    pos >= 1 || error("dflash_rows!: position $pos is below 1")
    size(X, 2) == m.fc.shape[2] ||
        error("dflash_rows!: features are $(size(X, 2)) wide, the drafter takes $(m.fc.shape[2])")
    pos + n - 1 <= m.ctx || error("dflash_rows!: rows to $(pos + n - 1) exceed the drafter's ctx $(m.ctx)")
    pos <= m.have + 1 || error("dflash_rows!: row $pos leaves a gap after row $(m.have)")
    n == 0 && return m
    d = m.d_head
    g = _df_rmsnorm_rows(_dflash_proj(m, Float32.(X), m.fc), m.enc_norm, m.eps)
    for (l, lay) in enumerate(m.layers)
        K = _dflash_proj(m, g, lay.wk)
        V = _dflash_proj(m, g, lay.wv)
        _dflash_heads!(K, lay.k_norm, m.n_kv_heads, d, pos - 1, m.eps, m.theta)
        for r in 1:n
            p = pos + r - 1
            for h in 1:m.n_kv_heads
                seg = ((h - 1) * d + 1):(h * d)
                m.kc[l][:, h, p] .= view(K, r, seg)
                m.vc[l][:, h, p] .= view(V, r, seg)
            end
        end
    end
    m.have = pos + n - 1
    return m
end

"""
    dflash_draft!(m, ids, k; p_min=0.0, n_min=0) -> Vector{Int}

Up to `k` draft ids after `ids[end]`. The target has consumed `ids[1:end-1]`,
so context rows must cover those. The block is `[ids[end], mask, ...]`, sized
`min(k, block_size-1, ctx-L)`. The block's K and V are temporaries, and the
caches are left as they were. Returns an empty vector if fewer than `n_min`
ids pass the `p_min` gate.
"""
function dflash_draft!(m::DFlashDraft, ids::AbstractVector{Int}, k::Int; p_min::Real=0.0,
    n_min::Int=0)
    L = length(ids)
    L >= 1 || error("dflash_draft!: no ids")
    m.have >= L - 1 || error("dflash_draft!: context rows end at $(m.have), need $(L - 1)")
    n_draft = min(k, m.block_size - 1, m.ctx - L)
    n_draft <= 0 && return Int[]
    t, lg = _dflash_block(m, ids, n_draft)
    K = min(m.top_k, size(lg, 2))
    cands = [_dflash_top_k(view(lg, i, :), K) for i in 1:n_draft]
    unary = [Float32[lg[i, c + 1] for c in cands[i]] for i in 1:n_draft]
    G = _dflash_proj(m, t[2:(n_draft + 1), :], m.sel_hidden)
    gate = [collect(view(G, i, :)) for i in 1:n_draft]
    cflat = reduce(vcat, cands)
    succ_flat = _dflash_table_rows(m, m.sel_succ, cflat)
    pred_flat = _dflash_table_rows(m, m.sel_pred, cflat)
    anchor = _dflash_table_rows(m, m.sel_pred, [ids[L]])
    succ_rows = [succ_flat[((i - 1) * K + 1):(i * K), :] for i in 1:n_draft]
    pred_rows = [i == 1 ? anchor : pred_flat[((i - 2) * K + 1):((i - 1) * K), :] for i in 1:n_draft]
    scores = _dflash_select(cands, unary, gate, succ_rows, pred_rows)
    return _dflash_walk(cands, scores; p_min, n_min)
end

"""
    reset!(m::DFlashDraft)

Forgets every context row, for a new conversation.
"""
function reset!(m::DFlashDraft)
    m.have = 0
    return m
end
