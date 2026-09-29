# Inference — engine (§XXIX, §XXX; Phase 5) + Phase 2 reference interpreter.
#
# Phase 2 slice (§LXXV): the reference PREFILL interpreter. This is the CPU
# oracle that makes `toy2` executable and produces the known logits — it is
# NOT the serving engine (scheduler, batching, sessions are Phase 5).
#
# Laws:
#   * Uses CPUBackend + PrefillWorkload only (§LXXV).
#   * All math goes through the operator surface (rmsnorm!, matmul!, rope!,
#     softmax!, swiglu!, embedding_lookup!) — the interpreter composes; it
#     does not reimplement operator math.
#   * Residuals are interpreter-level `storage` addition (§LXXV: no `add!`).
#   * The tied output head reuses the embedding table's materialized bytes
#     (the definition of a tied head); no second table is created.
#   * Float64 end to end. Deterministic: same inputs ⇒ bit-identical outputs
#     in-process.

module Inference

using ..Harpe:
    CPUBackend,
    PrefillWorkload,
    Activation,
    EmbeddingTable,
    ProjectionWeight,
    FrozenParameter,
    TemporaryWorkspace,
    embedding_lookup!,
    rmsnorm!,
    rope!,
    matmul!,
    softmax!,
    swiglu!

"""
    reference_prefill(model, tensors, tokens) -> Matrix{Float64}

Run the toy reference model's forward pass over `tokens` (0-based ids) and
return logits with shape `(vocab_size, seq_len)` — column `t` holds the
next-token logits after consuming tokens `1..t`.

`model` is a `Harpe.Model`; `tensors` is the materialized tensor set
(weight walk per the fixture protocol — see test/fixtures/toy/README.md):

    (embedding, blocks, lm_head)

with `blocks` a vector of per-block named tuples
`(wq, wk, wv, wo, wgate, wup, wdown, attn_rms, ffn_rms)`.

Block recipe (pre-norm, §LXXV): rmsnorm → q/k/v projections → split heads
→ rope → scaled scores → causal softmax → attention @ v → merge heads →
output projection → residual; then rmsnorm → gate/up → swiglu → down →
residual. Tied embedding head: `logits = h * transpose(E)`.
"""
function reference_prefill(model, tensors, tokens::AbstractVector{Int})
    isempty(tokens) && error("reference_prefill: token sequence is empty")
    cpu = CPUBackend()
    wl = PrefillWorkload()
    seq = length(tokens)
    dim = model.embedding.dim
    n_heads = model.blocks[1].attention.n_heads
    d_head = div(dim, n_heads)
    dim == n_heads * d_head ||
        error("reference_prefill: dim $dim is not divisible by n_heads $n_heads")
    all(b -> b.attention.n_heads == n_heads, model.blocks) ||
        error("reference_prefill: uniform n_heads across blocks required (toy2 is uniform)")

    positions = collect(0:(seq-1))            # 0-based (§LXXV)

    # hidden states: (seq, dim)
    h = Activation(; shape=(seq, dim), storage=zeros(seq, dim))
    embedding_lookup!(cpu, h, tensors.embedding, tokens, wl)

    # scratch — allocated once per forward, written in place
    normed = Activation(; shape=(seq, dim), storage=zeros(seq, dim))
    q = Activation(; shape=(seq, dim), storage=zeros(seq, dim))
    k = Activation(; shape=(seq, dim), storage=zeros(seq, dim))
    v = Activation(; shape=(seq, dim), storage=zeros(seq, dim))
    qh = Activation(; shape=(seq, n_heads, d_head), storage=zeros(seq, n_heads, d_head))
    kh = Activation(; shape=(seq, n_heads, d_head), storage=zeros(seq, n_heads, d_head))
    vh = Activation(; shape=(seq, n_heads, d_head), storage=zeros(seq, n_heads, d_head))
    attn = Activation(; shape=(seq, n_heads, d_head), storage=zeros(seq, n_heads, d_head))
    merged = Activation(; shape=(seq, dim), storage=zeros(seq, dim))
    sub = Activation(; shape=(seq, dim), storage=zeros(seq, dim))
    normed2 = Activation(; shape=(seq, dim), storage=zeros(seq, dim))
    gate = Activation(;
        shape=(seq, model.blocks[1].ffn.hidden),
        storage=zeros(seq, model.blocks[1].ffn.hidden),
    )
    up = Activation(;
        shape=(seq, model.blocks[1].ffn.hidden),
        storage=zeros(seq, model.blocks[1].ffn.hidden),
    )
    act = Activation(;
        shape=(seq, model.blocks[1].ffn.hidden),
        storage=zeros(seq, model.blocks[1].ffn.hidden),
    )
    down = Activation(; shape=(seq, dim), storage=zeros(seq, dim))
    scores = TemporaryWorkspace(; shape=(seq, seq), storage=zeros(seq, seq))
    scores_out = TemporaryWorkspace(; shape=(seq, seq), storage=zeros(seq, seq))

    for (bi, blk) in enumerate(model.blocks)
        bt = tensors.blocks[bi]

        # --- attention sublayer (pre-norm) -----------------------------------
        rmsnorm!(cpu, normed, h, bt.attn_rms, wl)
        matmul!(cpu, q, normed, bt.wq, wl)
        matmul!(cpu, k, normed, bt.wk, wl)
        matmul!(cpu, v, normed, bt.wv, wl)

        # split heads: feature f (0-based) = head * d_head + j (head-major
        # packing, matching the packed Wk/Wv rows of the weight walk)
        _split_heads!(qh, q, n_heads, d_head)
        _split_heads!(kh, k, n_heads, d_head)
        _split_heads!(vh, v, n_heads, d_head)

        rope!(cpu, qh, kh, positions, wl)

        # scores per head: (seq, seq), scaled by √d_head
        fill!(scores.storage, 0.0)
        for t in 1:seq, u in 1:seq, hh in 1:n_heads, j in 1:d_head
            scores.storage[t, u] +=
                qh.storage[t, hh, j] * kh.storage[u, hh, j] / sqrt(d_head)
        end
        softmax!(cpu, scores_out, scores, wl)   # causal mask applied inside

        # attention @ v, per head
        fill!(attn.storage, 0.0)
        for t in 1:seq, hh in 1:n_heads, j in 1:d_head, u in 1:seq
            attn.storage[t, hh, j] += scores_out.storage[t, u] * vh.storage[u, hh, j]
        end

        # merge heads back to (seq, dim), then output projection + residual
        _merge_heads!(merged, attn, n_heads, d_head)
        matmul!(cpu, sub, merged, bt.wo, wl)
        h.storage .+= sub.storage               # residual (interpreter add)

        # --- ffn sublayer (pre-norm) -----------------------------------------
        rmsnorm!(cpu, normed2, h, bt.ffn_rms, wl)
        matmul!(cpu, gate, normed2, bt.wgate, wl)
        matmul!(cpu, up, normed2, bt.wup, wl)
        swiglu!(cpu, act, gate, up, wl)
        matmul!(cpu, down, act, bt.wdown, wl)
        h.storage .+= down.storage              # residual
    end

    # tied embedding head: logits(t, :) = h(t, :) * Eᵀ → (seq, vocab).
    # The head's weight IS the embedding table (tied) — same bytes, viewed
    # as the lm_head projection; no second table is materialized.
    lm_head = tensors.lm_head
    seqvocab = Activation(;
        shape=(seq, size(lm_head.storage, 1)),
        storage=zeros(seq, size(lm_head.storage, 1)),
    )
    matmul!(cpu, seqvocab, h, lm_head, wl)
    return permutedims(seqvocab.storage)        # (vocab, seq)
end

# head-major split/merge between (seq, dim) and (seq, n_heads, d_head)
function _split_heads!(dst, src, n_heads, d_head)
    for t in axes(src.storage, 1), hh in 1:n_heads, j in 1:d_head
        dst.storage[t, hh, j] = src.storage[t, (hh-1)*d_head+j]
    end
    return dst
end

function _merge_heads!(dst, src, n_heads, d_head)
    for t in axes(dst.storage, 1), hh in 1:n_heads, j in 1:d_head
        dst.storage[t, (hh-1)*d_head+j] = src.storage[t, hh, j]
    end
    return dst
end

export reference_prefill

end # module Inference
