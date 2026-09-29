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

using ..Gesso:
    CPUBackend,
    PrefillWorkload,
    DecodeWorkload,
    Activation,
    EmbeddingTable,
    ProjectionWeight,
    FrozenParameter,
    TemporaryWorkspace,
    KVCache,
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

`model` is a `Gesso.Model`; `tensors` is the materialized tensor set
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

# --- Phase 2 item C: greedy generation with a real KV append (§LXXV) --------

# EOS token id — the loader pins the tokenizer contract (PAD=0, BOS=1, EOS=2).
const TOY_EOS = 2

"""
    reference_generate(model, tensors, prompt; max_new_tokens=8, info=nothing) -> Vector{Int}

Greedy (argmax) generation on `CPUBackend`. Prefills `prompt` (0-based ids)
while filling the per-layer KV cache, then decodes one token at a time on
the `DecodeWorkload` path — each step appends exactly one K/V row per layer
and attends over the whole cache; the prompt is NEVER re-prefilled.

Returns the full 0-based id sequence (prompt + new tokens). Stops at EOS
(token 2) or `max_new_tokens`, whichever comes first.

`info`: pass a `Ref{NamedTuple}` to receive `(kv_len, steps)` — the
diagnostic channel the KV-length invariant is tested through. KV length
equals prefix length: `kv_len == length(prompt) + steps`.
"""
function reference_generate(
    model,
    tensors,
    prompt::AbstractVector{Int};
    max_new_tokens::Int=8,
    info=nothing,
)
    isempty(prompt) && error("reference_generate: prompt is empty")
    cpu = CPUBackend()
    wp, wd = PrefillWorkload(), DecodeWorkload()
    dim = model.embedding.dim
    n_heads = model.blocks[1].attention.n_heads
    d_head = div(dim, n_heads)
    dim == n_heads * d_head ||
        error("reference_generate: dim $dim is not divisible by n_heads $n_heads")
    P = length(prompt)
    cap = max(max_new_tokens, 0)
    vocab = size(tensors.embedding.storage, 1)

    # per-layer KV caches: preallocated (P + cap) rows, filled progressively
    # (the cache is runtime state, not a weight — no fixture-walk entries)
    kc = [
        Activation(;
            shape=(P + cap, n_heads, d_head),
            storage=zeros(P + cap, n_heads, d_head),
        ) for _ in 1:length(model.blocks)
    ]
    vc = [
        Activation(;
            shape=(P + cap, n_heads, d_head),
            storage=zeros(P + cap, n_heads, d_head),
        ) for _ in 1:length(model.blocks)
    ]

    # hidden states: (P + cap, dim) — prefill fills rows 1..P, each decode
    # step writes its single new row in place (Activation fields are
    # immutable, §CIX; storage CONTENT is what the interpreter owns)
    h = Activation(; shape=(P + cap, dim), storage=zeros(P + cap, dim))
    normed = Activation(; shape=(P, dim), storage=zeros(P, dim))
    q = Activation(; shape=(P, dim), storage=zeros(P, dim))
    k = Activation(; shape=(P, dim), storage=zeros(P, dim))
    v = Activation(; shape=(P, dim), storage=zeros(P, dim))
    qh = Activation(; shape=(P, n_heads, d_head), storage=zeros(P, n_heads, d_head))
    kh = Activation(; shape=(P, n_heads, d_head), storage=zeros(P, n_heads, d_head))
    vh = Activation(; shape=(P, n_heads, d_head), storage=zeros(P, n_heads, d_head))
    attn = Activation(; shape=(P, n_heads, d_head), storage=zeros(P, n_heads, d_head))
    merged = Activation(; shape=(P, dim), storage=zeros(P, dim))
    sub = Activation(; shape=(P, dim), storage=zeros(P, dim))
    normed2 = Activation(; shape=(P, dim), storage=zeros(P, dim))
    hidden_ffn = model.blocks[1].ffn.hidden
    gate = Activation(; shape=(P, hidden_ffn), storage=zeros(P, hidden_ffn))
    up = Activation(; shape=(P, hidden_ffn), storage=zeros(P, hidden_ffn))
    act = Activation(; shape=(P, hidden_ffn), storage=zeros(P, hidden_ffn))
    down = Activation(; shape=(P, dim), storage=zeros(P, dim))
    scores = TemporaryWorkspace(; shape=(P, P), storage=zeros(P, P))
    scores_out = TemporaryWorkspace(; shape=(P, P), storage=zeros(P, P))

    positions = collect(0:(P-1))
    # the prefill phase works on rows 1..P of h through a VIEW (Activation
    # storage is untyped by design; scratch stays exactly (P, …)-shaped so
    # softmax never sees padded zero columns — they would poison the
    # denominators). Residuals broadcast through the view, in place.
    hp = Activation(; shape=(P, dim), storage=@view h.storage[1:P, :])
    embedding_lookup!(cpu, hp, tensors.embedding, prompt, wp)
    for (bi, blk) in enumerate(model.blocks)
        bt = tensors.blocks[bi]
        rmsnorm!(cpu, normed, hp, bt.attn_rms, wp)
        matmul!(cpu, q, normed, bt.wq, wp)
        matmul!(cpu, k, normed, bt.wk, wp)
        matmul!(cpu, v, normed, bt.wv, wp)
        _split_heads!(qh, q, n_heads, d_head)
        _split_heads!(kh, k, n_heads, d_head)
        _split_heads!(vh, v, n_heads, d_head)
        rope!(cpu, qh, kh, positions, wp)
        # KV APPEND: the prefill's post-rope K/V become cache rows 1..P
        kc[bi].storage[1:P, :, :] .= kh.storage
        vc[bi].storage[1:P, :, :] .= vh.storage
        # attention over the cache (identical math/order to reference_prefill)
        fill!(scores.storage, 0.0)
        for t in 1:P, u in 1:P, hh in 1:n_heads, j in 1:d_head
            scores.storage[t, u] +=
                qh.storage[t, hh, j] * kc[bi].storage[u, hh, j] / sqrt(d_head)
        end
        softmax!(cpu, scores_out, scores, wp)
        fill!(attn.storage, 0.0)
        for t in 1:P, hh in 1:n_heads, j in 1:d_head, u in 1:P
            attn.storage[t, hh, j] += scores_out.storage[t, u] * vc[bi].storage[u, hh, j]
        end
        _merge_heads!(merged, attn, n_heads, d_head)
        matmul!(cpu, sub, merged, bt.wo, wp)
        hp.storage .+= sub.storage
        rmsnorm!(cpu, normed2, hp, bt.ffn_rms, wp)
        matmul!(cpu, gate, normed2, bt.wgate, wp)
        matmul!(cpu, up, normed2, bt.wup, wp)
        swiglu!(cpu, act, gate, up, wp)
        matmul!(cpu, down, act, bt.wdown, wp)
        hp.storage .+= down.storage
    end

    ids = collect(prompt)
    steps = 0
    while steps < cap
        # greedy argmax at the current last position (row P + steps)
        logits_row = _last_logits(model, tensors, h, cpu, wd, vocab, P + steps)
        next = argmax(logits_row) - 1                 # 0-based id
        push!(ids, next)
        steps += 1
        (next == TOY_EOS || steps >= cap) && break
        _decode_step!(
            model,
            tensors,
            kc,
            vc,
            Int(next),
            P + steps,
            cpu,
            wd,
            n_heads,
            d_head,
            h,
        )
    end

    info isa Ref{Any} && (info[] = (kv_len=P + steps, steps=steps))
    return ids
end

# tied head over hidden row `row`: returns the (vocab,) logits row
function _last_logits(model, tensors, h, cpu, wl, vocab, row::Int)
    lastrow = Activation(;
        shape=(1, size(h.storage, 2)),
        storage=reshape(h.storage[row, :], (1, :)),
    )
    out = Activation(; shape=(1, vocab), storage=zeros(1, vocab))
    matmul!(cpu, out, lastrow, tensors.lm_head, wl)
    return vec(out.storage)
end

# one DecodeWorkload step: embed `tok` at 0-based `pos`, append K/V per
# layer, attend over the WHOLE cache (no re-prefill), residual + FFN, and
# leave the new last-position hidden state in `h`'s single row slot
function _decode_step!(
    model,
    tensors,
    kc,
    vc,
    tok::Int,
    pos0::Int,
    cpu,
    wl,
    n_heads,
    d_head,
    h,
)
    dim = model.embedding.dim
    hp = Activation(; shape=(1, dim), storage=zeros(1, dim))
    embedding_lookup!(cpu, hp, tensors.embedding, [tok], wl)
    normed = Activation(; shape=(1, dim), storage=zeros(1, dim))
    q = Activation(; shape=(1, dim), storage=zeros(1, dim))
    k = Activation(; shape=(1, dim), storage=zeros(1, dim))
    v = Activation(; shape=(1, dim), storage=zeros(1, dim))
    qh = Activation(; shape=(1, n_heads, d_head), storage=zeros(1, n_heads, d_head))
    kh = Activation(; shape=(1, n_heads, d_head), storage=zeros(1, n_heads, d_head))
    vh = Activation(; shape=(1, n_heads, d_head), storage=zeros(1, n_heads, d_head))
    attn = Activation(; shape=(1, n_heads, d_head), storage=zeros(1, n_heads, d_head))
    merged = Activation(; shape=(1, dim), storage=zeros(1, dim))
    sub = Activation(; shape=(1, dim), storage=zeros(1, dim))
    normed2 = Activation(; shape=(1, dim), storage=zeros(1, dim))
    hidden_ffn = model.blocks[1].ffn.hidden
    gate = Activation(; shape=(1, hidden_ffn), storage=zeros(1, hidden_ffn))
    up = Activation(; shape=(1, hidden_ffn), storage=zeros(1, hidden_ffn))
    act = Activation(; shape=(1, hidden_ffn), storage=zeros(1, hidden_ffn))
    down = Activation(; shape=(1, dim), storage=zeros(1, dim))
    K = pos0                                    # cache rows now: P + steps
    scores = TemporaryWorkspace(; shape=(1, K), storage=zeros(1, K))
    scores_out = TemporaryWorkspace(; shape=(1, K), storage=zeros(1, K))

    for (bi, blk) in enumerate(model.blocks)
        bt = tensors.blocks[bi]
        rmsnorm!(cpu, normed, hp, bt.attn_rms, wl)
        matmul!(cpu, q, normed, bt.wq, wl)
        matmul!(cpu, k, normed, bt.wk, wl)
        matmul!(cpu, v, normed, bt.wv, wl)
        _split_heads!(qh, q, n_heads, d_head)
        _split_heads!(kh, k, n_heads, d_head)
        _split_heads!(vh, v, n_heads, d_head)
        rope!(cpu, qh, kh, [pos0 - 1], wl)      # 0-based position of this token
        # KV APPEND: exactly one row per layer per step
        kc[bi].storage[pos0, :, :] .= kh.storage[1, :, :]
        vc[bi].storage[pos0, :, :] .= vh.storage[1, :, :]
        # the query is the LAST position: it attends to the whole cache
        fill!(scores.storage, 0.0)
        for u in 1:K, hh in 1:n_heads, j in 1:d_head
            scores.storage[1, u] +=
                qh.storage[1, hh, j] * kc[bi].storage[u, hh, j] / sqrt(d_head)
        end
        softmax!(cpu, scores_out, scores, wl)   # offset mask: nothing masked
        fill!(attn.storage, 0.0)
        for hh in 1:n_heads, j in 1:d_head, u in 1:K
            attn.storage[1, hh, j] += scores_out.storage[1, u] * vc[bi].storage[u, hh, j]
        end
        _merge_heads!(merged, attn, n_heads, d_head)
        matmul!(cpu, sub, merged, bt.wo, wl)
        hp.storage .+= sub.storage              # residual
        rmsnorm!(cpu, normed2, hp, bt.ffn_rms, wl)
        matmul!(cpu, gate, normed2, bt.wgate, wl)
        matmul!(cpu, up, normed2, bt.wup, wl)
        swiglu!(cpu, act, gate, up, wl)
        matmul!(cpu, down, act, bt.wdown, wl)
        hp.storage .+= down.storage             # residual
    end
    # write the new hidden state into its row slot — storage CONTENT, not a
    # field reassignment (Activation is immutable, §CIX)
    h.storage[pos0, :] .= hp.storage[1, :]
    return h
end

export reference_generate

end # module Inference
