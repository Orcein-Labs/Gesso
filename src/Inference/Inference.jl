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
#
# Phase 3 slice (§LXXVI item A): GQA (repeat KV heads for the score/value
# contraction only; the cache stays at n_kv_heads), an optional `final_rms`
# in `tensors` (applied before the tied head; absent ⇒ toy2 path untouched),
# and `eps` / `theta` threaded as keyword defaults — never hardcoded config
# constants.
#
# Phase 4 slice (§LXXVII items B/C): the interpreter is backend-generic.
# `backend` defaults to CPUBackend so every CPU path is bit-identical; with
# `backend=CUDABackend()` the tensors MUST already be on device (to_device
# is the explicit transfer — the interpreter never copies, and host Array
# storage under a non-CPU backend is ERR_INVALID_PLAN). All buffers are
# allocated with `similar` off the tensors' storage, so device memory lands
# on the device. CPU keeps its scalar contraction loops bit-for-bit; other
# backends use the same math expressed as range broadcasts + CUBLAS.

module Inference

using ..Gesso:
    AbstractGessoBackend,
    CPUBackend,
    PrefillWorkload,
    DecodeWorkload,
    Activation,
    EmbeddingTable,
    ProjectionWeight,
    FrozenParameter,
    TemporaryWorkspace,
    KVCache,
    backend_name,
    gesso_error,
    LoweringNotImplemented,
    ERR_INVALID_PLAN,
    ERR_RESOURCE_LIMIT,
    ReceiptSink,
    default_receipt_sink,
    emit!,
    new_receipt,
    embedding_lookup!,
    rmsnorm!,
    rope!,
    matmul!,
    softmax!,
    swiglu!
# Phase 3 (§LXXVI item B): the importer composes ModelIR primitives — same
# vocabulary, a new composition (§VIII: no LlamaModel type)
using ..ModelIR: Embedding, RMSNorm, RoPE, Attention, SwiGLU, Block, Model

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

Phase 3 (§LXXVI item A): `eps` (default 1e-6) and `theta` (default
10000.0) thread into `rmsnorm!` / `rope!` — toy2's values, unchanged. If
`tensors` carries a non-`nothing` `final_rms::FrozenParameter`, a final
RMSNorm is applied after the last block, before the tied head (Llama's
`model.norm`; toy2 omits it).GQA supported (`n_kv_heads < n_heads`): K/V cached at `n_kv_heads` and
repeated per query head only inside the attention contraction.

Pass `backend=CUDABackend()` (Phase 4, §LXXVII) to run on an NVIDIA GPU —
`tensors` must already be on device (`to_device`); the interpreter never
copies host memory, and host Array storage under a non-CPU backend is
`ERR_INVALID_PLAN`.
"""
function reference_prefill(
    model,
    tensors,
    tokens::AbstractVector{Int};
    backend::AbstractGessoBackend=CPUBackend(),
    eps::Real=1e-6,
    theta::Real=10000.0,
)
    isempty(tokens) && error("reference_prefill: token sequence is empty")
    _infer_device_storage!(:reference_prefill, backend, tensors)
    cpu = backend
    wl = PrefillWorkload()
    seq = length(tokens)
    dim = model.embedding.dim
    n_heads = model.blocks[1].attention.n_heads
    n_kv_heads = model.blocks[1].attention.n_kv_heads
    d_head = div(dim, n_heads)
    dim == n_heads * d_head ||
        error("reference_prefill: dim $dim is not divisible by n_heads $n_heads")
    group = div(n_heads, n_kv_heads)
    n_heads == n_kv_heads * group ||
        error("reference_prefill: n_kv_heads $n_kv_heads does not divide n_heads $n_heads")
    all(
        b -> b.attention.n_heads == n_heads && b.attention.n_kv_heads == n_kv_heads,
        model.blocks,
    ) || error(
        "reference_prefill: uniform head counts across blocks required (toy2 is uniform)",
    )

    positions = collect(0:(seq-1))            # 0-based (§LXXV)
    # BREADTH-0 Pass D: the positional policy TRAVELS WITH the imported model.
    # `nothing` for every unscaled model ⇒ the oracle evaluates its original
    # expression ⇒ Llama is bit-identical (regression law §XIII).
    inv_freq = tensors_rope_inv_freq(tensors, d_head)
    on_cpu = backend_name(cpu) === :cpu
    T = typeof(tensors.embedding.storage)      # buffers live where the data lives
    zeros_like =
        (dims::Tuple{Vararg{Int}}) -> fill!(
            similar(tensors.embedding.storage, T <: Array ? Float64 : eltype(T), dims),
            zero(eltype(T)),
        )

    # hidden states: (seq, dim)
    h = Activation(; shape=(seq, dim), storage=zeros_like((seq, dim)))
    embedding_lookup!(cpu, h, tensors.embedding, tokens, wl)

    # scratch — allocated once per forward, written in place
    normed = Activation(; shape=(seq, dim), storage=zeros_like((seq, dim)))
    q = Activation(; shape=(seq, dim), storage=zeros_like((seq, dim)))
    kdim = n_kv_heads * d_head
    k = Activation(; shape=(seq, kdim), storage=zeros_like((seq, kdim)))
    v = Activation(; shape=(seq, kdim), storage=zeros_like((seq, kdim)))
    qh = Activation(;
        shape=(seq, n_heads, d_head),
        storage=zeros_like((seq, n_heads, d_head)),
    )
    kh = Activation(;
        shape=(seq, n_kv_heads, d_head),
        storage=zeros_like((seq, n_kv_heads, d_head)),
    )
    vh = Activation(;
        shape=(seq, n_kv_heads, d_head),
        storage=zeros_like((seq, n_kv_heads, d_head)),
    )
    attn = Activation(;
        shape=(seq, n_heads, d_head),
        storage=zeros_like((seq, n_heads, d_head)),
    )
    merged = Activation(; shape=(seq, dim), storage=zeros_like((seq, dim)))
    sub = Activation(; shape=(seq, dim), storage=zeros_like((seq, dim)))
    normed2 = Activation(; shape=(seq, dim), storage=zeros_like((seq, dim)))
    gate = Activation(;
        shape=(seq, model.blocks[1].ffn.hidden),
        storage=zeros_like((seq, model.blocks[1].ffn.hidden)),
    )
    up = Activation(;
        shape=(seq, model.blocks[1].ffn.hidden),
        storage=zeros_like((seq, model.blocks[1].ffn.hidden)),
    )
    act = Activation(;
        shape=(seq, model.blocks[1].ffn.hidden),
        storage=zeros_like((seq, model.blocks[1].ffn.hidden)),
    )
    down = Activation(; shape=(seq, dim), storage=zeros_like((seq, dim)))
    scores = TemporaryWorkspace(; shape=(seq, seq), storage=zeros_like((seq, seq)))
    scores_out = TemporaryWorkspace(; shape=(seq, seq), storage=zeros_like((seq, seq)))

    for (bi, blk) in enumerate(model.blocks)
        bt = tensors.blocks[bi]

        # --- attention sublayer (pre-norm) -----------------------------------
        rmsnorm!(cpu, normed, h, bt.attn_rms, wl; eps)
        matmul!(cpu, q, normed, bt.wq, wl)
        matmul!(cpu, k, normed, bt.wk, wl)
        matmul!(cpu, v, normed, bt.wv, wl)

        # split heads: feature f (0-based) = head * d_head + j (head-major
        # packing, matching the packed Wk/Wv rows of the weight walk)
        _split_heads!(qh, q, n_heads, d_head)
        _split_heads!(kh, k, n_kv_heads, d_head)
        _split_heads!(vh, v, n_kv_heads, d_head)

        rope!(cpu, qh, kh, positions, wl; theta, inv_freq)

        # repeat KV heads for the contraction only (identity when MHA)
        kx = _repeat_heads(kh, group)
        vx = _repeat_heads(vh, group)

        # scores per head: (seq, seq), scaled by √d_head. CPU keeps its
        # scalar loops (bit-identical, §LXXV); device storage gets the same
        # math as broadcasts (§LXXVII).
        fill!(scores.storage, zero(eltype(scores.storage)))
        if on_cpu
            for t in 1:seq, u in 1:seq, hh in 1:n_heads, j in 1:d_head
                scores.storage[t, u] +=
                    qh.storage[t, hh, j] * kx.storage[u, hh, j] / sqrt(d_head)
            end
        else
            for hh in 1:n_heads, j in 1:d_head
                @views scores.storage[:, :] .+=
                    qh.storage[:, hh, j] .* kx.storage[:, hh, j]' ./ sqrt(d_head)
            end
        end
        softmax!(cpu, scores_out, scores, wl)   # causal mask applied inside

        # attention @ v, per head
        fill!(attn.storage, zero(eltype(attn.storage)))
        if on_cpu
            for t in 1:seq, hh in 1:n_heads, j in 1:d_head, u in 1:seq
                attn.storage[t, hh, j] += scores_out.storage[t, u] * vx.storage[u, hh, j]
            end
        else
            for hh in 1:n_heads, j in 1:d_head
                @views attn.storage[:, hh, j] .= scores_out.storage * vx.storage[:, hh, j]
            end
        end

        # merge heads back to (seq, dim), then output projection + residual
        _merge_heads!(merged, attn, n_heads, d_head)
        matmul!(cpu, sub, merged, bt.wo, wl)
        h.storage .+= sub.storage               # residual (interpreter add)

        # --- ffn sublayer (pre-norm) -----------------------------------------
        rmsnorm!(cpu, normed2, h, bt.ffn_rms, wl; eps)
        matmul!(cpu, gate, normed2, bt.wgate, wl)
        matmul!(cpu, up, normed2, bt.wup, wl)
        swiglu!(cpu, act, gate, up, wl)
        matmul!(cpu, down, act, bt.wdown, wl)
        h.storage .+= down.storage              # residual
    end

    # Llama applies a final RMSNorm before the head (§LXXVI); toy2 has none
    # and skips this branch entirely.
    final_rms = haskey(tensors, :final_rms) ? tensors.final_rms : nothing
    hhead = h
    if final_rms !== nothing
        finaln = Activation(; shape=(seq, dim), storage=zeros_like((seq, dim)))
        rmsnorm!(cpu, finaln, h, final_rms, wl; eps)
        hhead = finaln
    end

    # tied embedding head: logits(t, :) = hhead(t, :) * Eᵀ → (seq, vocab).
    # The head's weight IS the embedding table (tied) — same bytes, viewed
    # as the lm_head projection; no second table is materialized.
    lm_head = tensors.lm_head
    seqvocab = Activation(;
        shape=(seq, size(lm_head.storage, 1)),
        storage=zeros_like((seq, size(lm_head.storage, 1))),
    )
    matmul!(cpu, seqvocab, hhead, lm_head, wl)
    return permutedims(seqvocab.storage)        # (vocab, seq)
end

# head-major split/merge between (seq, dim) and (seq, n_heads, d_head).
#
# Phase 10E: the bodies below take STORAGE ARRAYS and the Activation forms
# forward to them. `Activation.storage` is `::Any` by §CIX (packet P-1), so a
# hot-path loop that reads `dst.storage[i]` dispatches dynamically and BOXES
# every element (measured: 16 B per element on the decode loops). Specializing
# on the storage type as a FUNCTION ARGUMENT is ordinary Julia dispatch — it
# adds no type to the §CIX hierarchy, does not parameterize Session, and does
# not make `decode!` infer (P-1 stays packeted). Values are untouched: same
# slices, same copy, same order.
# Head-major layout maps head hh's feature block to a contiguous slice, so
# the split/merge is a plain elementwise copy per head — same values as the
# old scalar loops on ANY storage (identical on CPU, pinned by atol=0).
# Phase 10F: the three head-layout copies below are the ones the 10E receipt
# measured at 2.48 MB of HOST allocation per warmed SmolLM2 CUDA `decode!` —
# a `Broadcasted` wrapper per head, per layer, per token, to move 64 floats.
# The bodies below stay EXACTLY as they were (the CPU/Lava/oracle path, whose
# values the toy2 fingerprint pins at atol=0). A backend extension may add a
# MORE SPECIFIC method on its own storage type to replace the broadcast with a
# kernel — that is ordinary dispatch on the storage argument, the same form
# 10E already uses; it adds no type to the §CIX hierarchy (P-1 stays
# packeted).
function _split_heads!(
    dst_storage::AbstractArray,
    src_storage::AbstractArray,
    n_heads,
    d_head,
)
    for hh in 1:n_heads
        @views dst_storage[:, hh, :] .= src_storage[:, ((hh-1)*d_head+1):(hh*d_head)]
    end
    return dst_storage
end

_split_heads!(dst::Activation, src::Activation, n_heads, d_head) =
    _split_heads!(dst.storage, src.storage, n_heads, d_head)

function _merge_heads!(
    dst_storage::AbstractArray,
    src_storage::AbstractArray,
    n_heads,
    d_head,
)
    for hh in 1:n_heads
        @views dst_storage[:, ((hh-1)*d_head+1):(hh*d_head)] .= src_storage[:, hh, :]
    end
    return dst_storage
end

_merge_heads!(dst::Activation, src::Activation, n_heads, d_head) =
    _merge_heads!(dst.storage, src.storage, n_heads, d_head)

# --- backend / storage discipline (§LXXVII) -----------------------------------

# the interpreter never copies: a non-CPU backend REQUIRES device storage
# (to_device is the explicit transfer). Host Array storage under CUDA is
# ERR_INVALID_PLAN, not a silent transfer (§LXX).
function _infer_device_storage!(where::Symbol, backend::AbstractGessoBackend, tensors)
    backend_name(backend) === :cpu && return nothing
    _check_device_storage(where, backend, tensors.embedding)
    tensors.lm_head === tensors.embedding ||
        _check_device_storage(where, backend, tensors.lm_head)
    for bt in tensors.blocks
        for name in propertynames(bt)
            _check_device_storage(where, backend, getproperty(bt, name))
        end
    end
    fr = haskey(tensors, :final_rms) ? tensors.final_rms : nothing
    fr === nothing || _check_device_storage(where, backend, fr)
    return nothing
end

function _check_device_storage(where::Symbol, backend, t)
    s = t.storage
    s === nothing &&
        error("$where: tensor storage is unset — materialize before calling (§LXXV)")
    s isa Array || return nothing             # already device (or otherwise OK)
    throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "$where: backend :$(backend_name(backend)) received host Array " *
            "storage — call to_device(tensors, backend) first; the " *
            "interpreter does not copy host memory to the device (§LXXVII)";
            backend=backend_name(backend),
        ),
    )
end

# GQA (§LXXVI): query head h attends kv head div(h-1, group) + 1 — repeat
# each kv head `group` times for the score/value contraction ONLY. RoPE and
# the KV cache stay at n_kv_heads. `group == 1` (MHA) returns the input
# unchanged, so toy2's arithmetic and allocation profile are untouched.
function _repeat_heads(kv::Activation, group::Int)
    group == 1 && return kv
    seq, nk, d = size(kv.storage)
    out = Activation(;
        shape=(seq, nk * group, d),
        storage=similar(kv.storage, seq, nk * group, d),
    )
    # repeat = each kv head fills its group of CONSECUTIVE query-head slots
    # (q head h attends kv head div(h-1, group)+1, §LXXVI) — same values as
    # the old scalar loops on any storage
    for h in 1:nk
        @views out.storage[:, ((h-1)*group+1):(h*group), :] .= kv.storage[:, h:h, :]
    end
    return out
end

# Phase 10E item C: the IN-PLACE GQA repeat, engine decode only. The
# allocating `_repeat_heads` above stays — it is the oracle/interpreter
# contract and returns an Activation.
#
#   * `group == 1` (MHA): NO COPY. `dst` may be `src` (the engine passes the
#     gathered buffer itself), so toy2's allocation profile cannot grow here.
#   * `group > 1`: rows 1:K of the preallocated `(context_length,
#     n_heads, d_head)` destination are filled. Only the filled PREFIX is
#     touched — per-token work is O(K), never O(context_length).
#
# The copy itself is byte-for-byte the same layout as `_repeat_heads` (each kv
# head fills its group of consecutive query-head slots), so ids and logits do
# not move. Like the head split/merge above, the body takes STORAGE ARRAYS so
# the copy does not box element-by-element through `storage::Any` (P-1).
function _repeat_heads!(
    dst_storage::AbstractArray,
    src_storage::AbstractArray,
    group::Int,
    K::Int,
)
    group == 1 && return dst_storage
    nk = size(src_storage, 2)
    for h in 1:nk
        @views dst_storage[1:K, ((h-1)*group+1):(h*group), :] .= src_storage[1:K, h:h, :]
    end
    return dst_storage
end

_repeat_heads!(dst::Activation, src::Activation, group::Int, K::Int) =
    _repeat_heads!(dst.storage, src.storage, group, K)

# --- Phase 10F: storage-level residual add + score scale + row write ----------
#
# The interpreter adds the residual in STORAGE (§LXXV: no `add!` operator
# exists, and this does not introduce one), scales the attention scores by
# 1/sqrt(d_head), and writes the new hidden row. All three were inline
# `@views`/`./=` broadcasts at every call site; naming them lets a device
# backend specialize on its storage type. The generic bodies below ARE the
# previous expressions, unchanged, so CPU, Lava and the oracle keep the exact
# arithmetic they had (Phase 10F, item A: the CUDA path stops building a
# `Broadcasted` wrapper per head per layer per token; these are the seams
# that makes it possible).

_add_storage!(dst_storage::AbstractArray, src_storage::AbstractArray) =
    (@views dst_storage .+= src_storage)

_scale_storage!(dst_storage::AbstractArray, s) = (@views dst_storage ./= s)

# zero a TAIL of a buffer in place. `fill!` on a view is a single kernel and
# allocates 144 B where the `.=` broadcast allocated 2,112 B (measured on the
# RTX 5060, 2026-10-04), with identical results.
_zero_tail_storage!(dst_storage::AbstractArray, from::Int) = (
    from > size(dst_storage, 2) && return dst_storage;
    fill!(view(dst_storage, :, from:size(dst_storage, 2)), zero(eltype(dst_storage)))
)

# write ONE row of `src` into row `row` of `dst`. The engine's hidden row
# write is the same shape as the KV row copies in kv_manager.jl, so it gets
# the same named seam and the same chance to be a kernel on device.
_write_hidden_row_storage!(
    dst_storage::AbstractArray,
    src_storage::AbstractArray,
    row::Int,
) = (@views dst_storage[row, :] .= src_storage[1, :])

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

Phase 3 (§LXXVI item A): `eps` / `theta` thread through as in
`reference_prefill`; `tensors.final_rms` (when present) is applied before
the tied head on every argmax step; GQA models cache K/V at `n_kv_heads`
and repeat per query head only inside the attention contraction.

Pass `backend=CUDABackend()` for device execution (§LXXVII): `tensors` on
device via `to_device`, argmax computed per step (device-safe), ids on host.
"""
function reference_generate(
    model,
    tensors,
    prompt::AbstractVector{Int};
    backend::AbstractGessoBackend=CPUBackend(),
    max_new_tokens::Int=8,
    info=nothing,
    eps::Real=1e-6,
    theta::Real=10000.0,
)
    isempty(prompt) && error("reference_generate: prompt is empty")
    _infer_device_storage!(:reference_generate, backend, tensors)
    cpu = backend
    wp, wd = PrefillWorkload(), DecodeWorkload()
    dim = model.embedding.dim
    n_heads = model.blocks[1].attention.n_heads
    n_kv_heads = model.blocks[1].attention.n_kv_heads
    d_head = div(dim, n_heads)
    dim == n_heads * d_head ||
        error("reference_generate: dim $dim is not divisible by n_heads $n_heads")
    group = div(n_heads, n_kv_heads)
    n_heads == n_kv_heads * group ||
        error("reference_generate: n_kv_heads $n_kv_heads does not divide n_heads $n_heads")
    P = length(prompt)
    cap = max(max_new_tokens, 0)
    vocab = size(tensors.embedding.storage, 1)
    on_cpu = backend_name(cpu) === :cpu
    T = typeof(tensors.embedding.storage)
    zeros_like =
        (dims::Tuple{Vararg{Int}}) -> fill!(
            similar(tensors.embedding.storage, T <: Array ? Float64 : eltype(T), dims),
            zero(eltype(T)),
        )

    # per-layer KV caches: preallocated (P + cap) rows, filled progressively
    # (the cache is runtime state, not a weight — no fixture-walk entries)
    kc = [
        Activation(;
            shape=(P + cap, n_kv_heads, d_head),
            storage=zeros_like((P + cap, n_kv_heads, d_head)),
        ) for _ in 1:length(model.blocks)
    ]
    vc = [
        Activation(;
            shape=(P + cap, n_kv_heads, d_head),
            storage=zeros_like((P + cap, n_kv_heads, d_head)),
        ) for _ in 1:length(model.blocks)
    ]

    # hidden states: (P + cap, dim) — prefill fills rows 1..P, each decode
    # step writes its single new row in place (Activation fields are
    # immutable, §CIX; storage CONTENT is what the interpreter owns)
    h = Activation(; shape=(P + cap, dim), storage=zeros_like((P + cap, dim)))
    normed = Activation(; shape=(P, dim), storage=zeros_like((P, dim)))
    kdim = n_kv_heads * d_head
    q = Activation(; shape=(P, dim), storage=zeros_like((P, dim)))
    k = Activation(; shape=(P, kdim), storage=zeros_like((P, kdim)))
    v = Activation(; shape=(P, kdim), storage=zeros_like((P, kdim)))
    qh = Activation(; shape=(P, n_heads, d_head), storage=zeros_like((P, n_heads, d_head)))
    kh = Activation(;
        shape=(P, n_kv_heads, d_head),
        storage=zeros_like((P, n_kv_heads, d_head)),
    )
    vh = Activation(;
        shape=(P, n_kv_heads, d_head),
        storage=zeros_like((P, n_kv_heads, d_head)),
    )
    attn =
        Activation(; shape=(P, n_heads, d_head), storage=zeros_like((P, n_heads, d_head)))
    merged = Activation(; shape=(P, dim), storage=zeros_like((P, dim)))
    sub = Activation(; shape=(P, dim), storage=zeros_like((P, dim)))
    normed2 = Activation(; shape=(P, dim), storage=zeros_like((P, dim)))
    hidden_ffn = model.blocks[1].ffn.hidden
    gate = Activation(; shape=(P, hidden_ffn), storage=zeros_like((P, hidden_ffn)))
    up = Activation(; shape=(P, hidden_ffn), storage=zeros_like((P, hidden_ffn)))
    act = Activation(; shape=(P, hidden_ffn), storage=zeros_like((P, hidden_ffn)))
    down = Activation(; shape=(P, dim), storage=zeros_like((P, dim)))
    scores = TemporaryWorkspace(; shape=(P, P), storage=zeros_like((P, P)))
    scores_out = TemporaryWorkspace(; shape=(P, P), storage=zeros_like((P, P)))

    positions = collect(0:(P-1))
    # BREADTH-0 Pass D — see reference_prefill. `nothing` = unscaled model.
    inv_freq = tensors_rope_inv_freq(tensors, d_head)
    # the prefill phase works on rows 1..P of h through a VIEW (Activation
    # storage is untyped by design; scratch stays exactly (P, …)-shaped so
    # softmax never sees padded zero columns — they would poison the
    # denominators). Residuals broadcast through the view, in place.
    hp = Activation(; shape=(P, dim), storage=@view h.storage[1:P, :])
    embedding_lookup!(cpu, hp, tensors.embedding, prompt, wp)
    for (bi, blk) in enumerate(model.blocks)
        bt = tensors.blocks[bi]
        rmsnorm!(cpu, normed, hp, bt.attn_rms, wp; eps)
        matmul!(cpu, q, normed, bt.wq, wp)
        matmul!(cpu, k, normed, bt.wk, wp)
        matmul!(cpu, v, normed, bt.wv, wp)
        _split_heads!(qh, q, n_heads, d_head)
        _split_heads!(kh, k, n_kv_heads, d_head)
        _split_heads!(vh, v, n_kv_heads, d_head)
        rope!(cpu, qh, kh, positions, wp; theta, inv_freq)
        # KV APPEND: the prefill's post-rope K/V become cache rows 1..P,
        # stored at n_kv_heads (GQA caches the small side, §LXXVI)
        kc[bi].storage[1:P, :, :] .= kh.storage
        vc[bi].storage[1:P, :, :] .= vh.storage
        # repeat KV heads for the contraction only (identity when MHA)
        kx = _repeat_heads(kh, group)
        vx = _repeat_heads(vh, group)
        # attention over the cache (identical math/order to reference_prefill)
        fill!(scores.storage, zero(eltype(scores.storage)))
        if on_cpu
            for t in 1:P, u in 1:P, hh in 1:n_heads, j in 1:d_head
                scores.storage[t, u] +=
                    qh.storage[t, hh, j] * kx.storage[u, hh, j] / sqrt(d_head)
            end
        else
            for hh in 1:n_heads, j in 1:d_head
                @views scores.storage[:, :] .+=
                    qh.storage[:, hh, j] .* kx.storage[:, hh, j]' ./ sqrt(d_head)
            end
        end
        softmax!(cpu, scores_out, scores, wp)
        fill!(attn.storage, zero(eltype(attn.storage)))
        if on_cpu
            for t in 1:P, hh in 1:n_heads, j in 1:d_head, u in 1:P
                attn.storage[t, hh, j] += scores_out.storage[t, u] * vx.storage[u, hh, j]
            end
        else
            for hh in 1:n_heads, j in 1:d_head
                @views attn.storage[:, hh, j] .= scores_out.storage * vx.storage[:, hh, j]
            end
        end
        _merge_heads!(merged, attn, n_heads, d_head)
        matmul!(cpu, sub, merged, bt.wo, wp)
        hp.storage .+= sub.storage
        rmsnorm!(cpu, normed2, hp, bt.ffn_rms, wp; eps)
        matmul!(cpu, gate, normed2, bt.wgate, wp)
        matmul!(cpu, up, normed2, bt.wup, wp)
        swiglu!(cpu, act, gate, up, wp)
        matmul!(cpu, down, act, bt.wdown, wp)
        hp.storage .+= down.storage
    end

    ids = collect(prompt)
    steps = 0
    while steps < cap
        # greedy argmax at the current last position (row P + steps); the
        # argmax may run over device storage — only the scalar id crosses
        # back to the host
        logits_row = _last_logits(model, tensors, h, cpu, wd, vocab, P + steps; eps)
        next = Int(argmax(logits_row)) - 1            # 0-based id
        push!(ids, next)
        steps += 1
        (next == TOY_EOS || steps >= cap) && break
        _decode_step!(
            model,
            tensors,
            kc,
            vc,
            next,
            P + steps,
            cpu,
            wd,
            h;
            on_cpu,
            n_heads,
            n_kv_heads,
            d_head,
            group,
            eps,
            theta,
            inv_freq,
        )
    end

    info isa Ref{Any} && (info[] = (kv_len=P + steps, steps=steps))
    return ids
end

# tied head over hidden row `row`: returns the (vocab,) logits row. Applies
# the final RMSNorm first when `tensors` carries one (§LXXVI).
function _last_logits(model, tensors, h, cpu, wl, vocab, row::Int; eps::Real=1e-6)
    dim = size(h.storage, 2)
    T = typeof(h.storage)
    lastrow = Activation(; shape=(1, dim), storage=reshape(h.storage[row, :], (1, dim)))
    final_rms = haskey(tensors, :final_rms) ? tensors.final_rms : nothing
    if final_rms !== nothing
        normed = Activation(;
            shape=(1, dim),
            storage=fill!(
                similar(h.storage, T <: Array ? Float64 : eltype(T), (1, dim)),
                zero(eltype(h.storage)),
            ),
        )
        rmsnorm!(cpu, normed, lastrow, final_rms, wl; eps)
        lastrow = normed
    end
    out = Activation(;
        shape=(1, vocab),
        storage=fill!(
            similar(h.storage, T <: Array ? Float64 : eltype(T), (1, vocab)),
            zero(eltype(h.storage)),
        ),
    )
    matmul!(cpu, out, lastrow, tensors.lm_head, wl)
    return vec(out.storage)
end

# one DecodeWorkload step: embed `tok` at 0-based `pos0`, append K/V per
# layer at n_kv_heads, attend over the WHOLE cache with KV heads repeated
# per query head (no re-prefill), residual + FFN, and leave the new
# last-position hidden state in `h`'s single row slot
function _decode_step!(
    model,
    tensors,
    kc,
    vc,
    tok::Int,
    pos0::Int,
    cpu,
    wl,
    h;
    on_cpu,
    n_heads,
    n_kv_heads,
    d_head,
    group,
    eps,
    theta,
    inv_freq,
)
    dim = model.embedding.dim
    kdim = n_kv_heads * d_head
    T = typeof(tensors.embedding.storage)
    zeros_like =
        (dims::Tuple{Vararg{Int}}) -> fill!(
            similar(tensors.embedding.storage, T <: Array ? Float64 : eltype(T), dims),
            zero(eltype(T)),
        )
    hp = Activation(; shape=(1, dim), storage=zeros_like((1, dim)))
    embedding_lookup!(cpu, hp, tensors.embedding, [tok], wl)
    normed = Activation(; shape=(1, dim), storage=zeros_like((1, dim)))
    q = Activation(; shape=(1, dim), storage=zeros_like((1, dim)))
    k = Activation(; shape=(1, kdim), storage=zeros_like((1, kdim)))
    v = Activation(; shape=(1, kdim), storage=zeros_like((1, kdim)))
    qh = Activation(; shape=(1, n_heads, d_head), storage=zeros_like((1, n_heads, d_head)))
    kh = Activation(;
        shape=(1, n_kv_heads, d_head),
        storage=zeros_like((1, n_kv_heads, d_head)),
    )
    vh = Activation(;
        shape=(1, n_kv_heads, d_head),
        storage=zeros_like((1, n_kv_heads, d_head)),
    )
    attn =
        Activation(; shape=(1, n_heads, d_head), storage=zeros_like((1, n_heads, d_head)))
    merged = Activation(; shape=(1, dim), storage=zeros_like((1, dim)))
    sub = Activation(; shape=(1, dim), storage=zeros_like((1, dim)))
    normed2 = Activation(; shape=(1, dim), storage=zeros_like((1, dim)))
    hidden_ffn = model.blocks[1].ffn.hidden
    gate = Activation(; shape=(1, hidden_ffn), storage=zeros_like((1, hidden_ffn)))
    up = Activation(; shape=(1, hidden_ffn), storage=zeros_like((1, hidden_ffn)))
    act = Activation(; shape=(1, hidden_ffn), storage=zeros_like((1, hidden_ffn)))
    down = Activation(; shape=(1, dim), storage=zeros_like((1, dim)))
    K = pos0                                    # cache rows now: P + steps
    scores = TemporaryWorkspace(; shape=(1, K), storage=zeros_like((1, K)))
    scores_out = TemporaryWorkspace(; shape=(1, K), storage=zeros_like((1, K)))

    for (bi, blk) in enumerate(model.blocks)
        bt = tensors.blocks[bi]
        rmsnorm!(cpu, normed, hp, bt.attn_rms, wl; eps)
        matmul!(cpu, q, normed, bt.wq, wl)
        matmul!(cpu, k, normed, bt.wk, wl)
        matmul!(cpu, v, normed, bt.wv, wl)
        _split_heads!(qh, q, n_heads, d_head)
        _split_heads!(kh, k, n_kv_heads, d_head)
        _split_heads!(vh, v, n_kv_heads, d_head)
        rope!(cpu, qh, kh, [pos0 - 1], wl; theta, inv_freq)  # 0-based position
        # KV APPEND: exactly one row per layer per step, at n_kv_heads
        kc[bi].storage[pos0, :, :] .= kh.storage[1, :, :]
        vc[bi].storage[pos0, :, :] .= vh.storage[1, :, :]
        # the query is the LAST position: it attends to the whole cache —
        # repeat the CACHE rows per query head (kh above is the single
        # new row; it was consumed by the append)
        kx = _repeat_heads(kc[bi], group)
        vx = _repeat_heads(vc[bi], group)
        fill!(scores.storage, zero(eltype(scores.storage)))
        if on_cpu
            for u in 1:K, hh in 1:n_heads, j in 1:d_head
                scores.storage[1, u] +=
                    qh.storage[1, hh, j] * kx.storage[u, hh, j] / sqrt(d_head)
            end
        else
            # one-row contraction: per query head, q's (d_head,) row against
            # the (K, d_head) slice of k — CUBLAS matvec over device copies
            # (real gathers, no scalar indexing on device storage, §LXXVII)
            for hh in 1:n_heads
                q1 = vec(qh.storage[1, hh, :])          # (d_head,) device copy
                # rows 1..K ONLY: the cache buffer is (P + cap) rows and
                # decode attends to filled rows 1..K (the CPU loop is
                # `u in 1:K`); indexing (not view) drops the head dim → (K, d_head)
                Kmat = kx.storage[1:K, hh, :]
                @views scores.storage[1, :] .+= (Kmat * q1) ./ sqrt(d_head)
            end
        end
        softmax!(cpu, scores_out, scores, wl)   # offset mask: nothing masked
        fill!(attn.storage, zero(eltype(attn.storage)))
        if on_cpu
            for hh in 1:n_heads, j in 1:d_head, u in 1:K
                attn.storage[1, hh, j] += scores_out.storage[1, u] * vx.storage[u, hh, j]
            end
        else
            for hh in 1:n_heads
                V = vx.storage[1:K, hh, :]              # (K, d_head) device copy
                # (1,K)·(K,d) → (1,d): vec to the (d,) row shape the view wants
                @views attn.storage[1, hh, :] .= vec(scores_out.storage * V)
            end
        end
        _merge_heads!(merged, attn, n_heads, d_head)
        matmul!(cpu, sub, merged, bt.wo, wl)
        hp.storage .+= sub.storage              # residual
        rmsnorm!(cpu, normed2, hp, bt.ffn_rms, wl; eps)
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

# Phase 3 (§LXXVI item B): Llama-family checkpoint import — config,
# safetensors reader, name map. Transport, not architecture (§VIII).
include("llama_import.jl")

# Phase 5 (§LXXVIII item A): paged KV manager — Magenta §9.5 step 1. Pages
# are the cache; attention gathers into contiguous scratch and reuses the
# existing operators (the manager owns storage and append, not a kernel).
include("kv_manager.jl")

# Phase 5 (§LXXVIII items B/C): the Session engine — prefill!/decode!/generate
# over the paged manager. The oracle (reference_prefill/reference_generate)
# stays untouched; this path must match it (ids equal, CPU logits atol=0).
#
# Phase 7 (§LXXX item B): fork — DECLARED identity prefix share (Magenta
# §9.5 step 3). The declaration is the function call; sharing is never
# discovered by token match. CoW lives in kv_manager.jl (item A).
include("session.jl")

# Phase 3 (§LXXVI item C): GPT-2 byte-level BPE — the tokenizer path, in
# Gesso (no Tokenizers.jl; JSON is already the sanctioned dependency).
include("tokenizer_abstract.jl")  # Pass C: the protocol, BEFORE its implementors
include("gpt2_tokenizer.jl")

# --- BREADTH-0: the universal model doorway -------------------------------------
#
# Loaded AFTER llama_import.jl so the legacy strict Llama surface stays exactly
# where it was (regression law §XIII) and the generic doorway sits beside it.
# The generic path is what new families use; `load_llama` still works.
include("architecture_spec.jl")      # Pass A/F: canonical description + capabilities
include("semantic_params.jl")        # Pass B/E: canonical parameter identity
include("arch_adapters.jl")          # Pass A/B: the family boundary
include("materialize_architecture.jl") # Pass B/E: the generic binder
include("tokenizer_protocol.jl")   # Pass C: the tokenizer PROTOCOL
include("import_report.jl")      # Passes I/J: report + generated matrix

end # module Inference
