# Phase 5 (§LXXVIII items B/C): the Session — Gesso's first native engine
# surface (§XXIX, §LXVII). `reference_prefill` / `reference_generate` remain
# the ORACLE, untouched; this is a NEW path that must match them (token ids
# equal, CPU prefill logits atol=0).
#
# Shape of the engine:
#   * prefill! consumes the prompt ONCE (PrefillWorkload, §XXX) — the paged
#     manager receives the post-RoPE K/V rows 1..P; logits come back host-side.
#   * decode! is DecodeWorkload one token at a time: greedy argmax of the
#     last-position logits, then (unless the id is the session's eos) the token
#     is consumed — embedded, roped at its 0-based position, appended to every
#     (layer, kind) cache, hidden state written to its h row.
#   * attention GATHERS the pages into a contiguous scratch per step and runs
#     the SAME contraction loops as the oracle (§LXXVIII: the manager owns
#     storage and append, not a kernel). On CPU the gathered rows are
#     byte-identical to the oracle's contiguous cache, so results are
#     bit-identical; on CUDA the argmax gate is token-id equality (§LXXVII).
#   * no re-prefill per step, no "generic generate" (§XXX). Batch = 1.
#   * generate() RESETS session state (a fresh paged manager and hidden
#     buffer) so every call is independent and deterministic — two Sessions
#     and two generate() calls on one Session give the same ids.
#
# Laws honored here:
#   * the interpreter/engine never copies host memory (§LXXVII): a non-CPU
#     backend requires device tensors (to_device is the explicit transfer);
#     host Array storage under a non-CPU backend is ERR_INVALID_PLAN.
#   * greedy only: `argmax`, ties = first index, 0-based id, no Random (§LXXVIII).
#     The argmax lives in `_greedy_id` so it is not buried.

# Device fast paths (Phase 10 B/C) are gated on backend CAPABILITY (§XX),
# probed as `Gesso.supports(backend, :argmax / :attn_gemm)` — so a backend
# without the fast path (Lava) keeps its own contraction and the full-row
# host argmax. No silent fallback: the gate is declared, the branches are
# visible, and ids are asserted identical either way.
import ..Gesso                                # binds the parent module NAME for the capability probes
using LinearAlgebra: mul!                     # device GEMM fast paths (Phase 10 C)
#   * `eos_token_id` is a required Session field — the engine does NOT
#     hardcode the oracle's TOY_EOS. The oracle still does (its contract).
#   * Runtime stays contract-only (§XXXII): Session lives in Inference; there
#     is no scheduler, no batching, no queues this sprint.
#
# Phase 6 (§LXXIX item A): every generate / prefill! / decode! emits ONE
# receipt (§XLII fields filled, no new Receipt fields — no schema bump).
# emit! never throws and a telemetry failure never changes ids: the receipt
# is built AFTER the result (or the caught error) exists, and emit! itself
# swallows delivery failures (receipts.jl). Timing is wall-clock time_ns()
# around the phases (§XXXIII hygiene: tests that assert structure may include
# compile; benchmarks warm up first). kv_bytes is DERIVED from the page table
# (see kv_manager.jl) — a test reconstructs it from sizeof of the pages.

mutable struct Session
    model::Any
    tensors::Any
    backend::AbstractGessoBackend
    page_size::Int
    context_length::Int
    eos_token_id::Int
    tokenizer::Any
    eps::Float64
    theta::Float64
    sink::ReceiptSink
    mgr::PagedKVManager
    h::Any                                  # (context_length, dim) hidden rows
    seqlen::Int                        # consumed tokens == kv_len (lockstep)
    ready::Bool                        # prefill! has run
    n_heads::Int
    n_kv_heads::Int
    d_head::Int
    group::Int
    vocab::Int
end

"""
    Session(model, tensors; backend=CPUBackend(), page_size=16, context_length,
            eos_token_id, tokenizer=nothing, eps=1e-6, theta=10000.0,
            sink=default_receipt_sink())

The Phase 5 engine surface (§LXVII): `prefill!` / `decode!` / `generate`.

`backend=CUDABackend()` requires tensors already on device (`to_device`) —
the engine never copies host memory (§LXXVII). Pages allocate via `similar`
off the tensor storage (CPU `Array{Float64}`, CUDA `CuArray{Float32}`).
`eos_token_id` is REQUIRED: the engine never hardcodes a model's EOS
(toy2 uses 2; SmolLM2 uses 0). `eps`/`theta` thread to `rmsnorm!`/`rope!`
exactly as in the interpreter (§LXXVI).
"""
function Session(
    model,
    tensors;
    backend::AbstractGessoBackend=CPUBackend(),
    page_size::Int=16,
    context_length::Int,
    eos_token_id::Int,
    tokenizer=nothing,
    eps::Real=1e-6,
    theta::Real=10000.0,
    sink::ReceiptSink=default_receipt_sink(),
)
    context_length >= 1 || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "Session: context_length must be ≥ 1";
            context_length=context_length,
        ),
    )
    # the engine never copies: non-CPU backend REQUIRES device storage (§LXXVII)
    _infer_device_storage!(:Session, backend, tensors)

    dim = model.embedding.dim
    n_heads = model.blocks[1].attention.n_heads
    n_kv_heads = model.blocks[1].attention.n_kv_heads
    d_head = div(dim, n_heads)
    group = div(n_heads, n_kv_heads)
    vocab = size(tensors.embedding.storage, 1)

    T = typeof(tensors.embedding.storage)
    h = fill!(
        similar(
            tensors.embedding.storage,
            T <: Array ? Float64 : eltype(T),
            (context_length, dim),
        ),
        zero(eltype(T)),
    )

    mgr = PagedKVManager(
        tensors.embedding.storage;
        n_layers=length(model.blocks),
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        page_size=page_size,
        context_length=context_length,
    )
    return Session(
        model,
        tensors,
        backend,
        page_size,
        context_length,
        eos_token_id,
        tokenizer,
        Float64(eps),
        Float64(theta),
        sink,
        mgr,
        h,
        0,
        false,
        n_heads,
        n_kv_heads,
        d_head,
        group,
        vocab,
    )
end

# --- shared scratch plumbing ---------------------------------------------------

# zeroed scratch of the tensors' storage kind (CPU F64 buffers / device F32,
# §LXXVII) — the same pattern the oracle uses
_session_zeros_like(tensors, dims::Tuple{Vararg{Int}}) = fill!(
    similar(
        tensors.embedding.storage,
        typeof(tensors.embedding.storage) <: Array ? Float64 :
        eltype(tensors.embedding.storage),
        dims,
    ),
    zero(eltype(tensors.embedding.storage)),
)

# greedy id from a logits row: argmax, ties = first index, 0-based (§LXXVIII;
# no Random, no Sampler zoo)
_greedy_id(logits_row) = Int(argmax(logits_row)) - 1

# device greedy id over hidden row `row` (Phase 10 B): the `_last_logits`
# math (final RMSNorm when present, tied lm_head matmul) runs on the device
# storage, then `argmax` runs ON the device too — GPUArrays argmax resolves
# ties to the FIRST index deterministically (verified against host `argmax`
# on tie fixtures: mid-tie, all-equal, 50k-apart equal tail), matching the
# host reduction bit-for-bit on ids. Returns ONE 0-based Int: the (vocab,)
# logits row never crosses to the host on the decode hot path (§LXXVII:
# explicit transfers; the decode D2H is one Int per token).
function _device_greedy_id(model, tensors, h, cpu, wl, vocab, row::Int; eps::Real=1e-6)
    dim = size(h, 2)
    T = typeof(h)
    lastrow = Activation(; shape=(1, dim), storage=reshape(h[row, :], (1, dim)))
    final_rms = haskey(tensors, :final_rms) ? tensors.final_rms : nothing
    if final_rms !== nothing
        normed = Activation(;
            shape=(1, dim),
            storage=fill!(similar(h, eltype(T), (1, dim)), zero(eltype(h))),
        )
        rmsnorm!(cpu, normed, lastrow, final_rms, wl; eps)
        lastrow = normed
    end
    out = Activation(;
        shape=(1, vocab),
        storage=fill!(similar(h, eltype(T), (1, vocab)), zero(eltype(h))),
    )
    matmul!(cpu, out, lastrow, tensors.lm_head, wl)
    return Int(argmax(vec(out.storage))) - 1     # 0-based id, ties = first index (§LXXVIII)
end

function _session_reset!(s::Session)
    s.mgr = PagedKVManager(
        s.tensors.embedding.storage;
        n_layers=length(s.model.blocks),
        n_kv_heads=s.n_kv_heads,
        d_head=s.d_head,
        page_size=s.page_size,
        context_length=s.context_length,
    )
    fill!(s.h, zero(eltype(s.h)))
    s.seqlen = 0
    s.ready = false
    return s
end

# --- receipts (§LXXIX item A; §XLII fields, no new Receipt fields) ------------

# timing/tokens accumulator handed to the impl functions: they record their
# phase timings and token counts through it, so the wrapper can build the
# receipt from facts the call itself measured (no re-timing, no drift).
mutable struct _EngineSpan
    prefill_ns::UInt64
    decode_ns::UInt64
    prompt_tokens::Int
    new_tokens::Int
end
_EngineSpan() = _EngineSpan(UInt64(0), UInt64(0), 0, 0)

# one auditable record of an engine call. `failure` is the CONSTRUCTED
# GessoError when the call threw (the throw still propagates); timing is
# wall-clock time_ns() (§XXXIII: structure tests may include compile).
function _engine_receipt(
    s::Session,
    task::Symbol,
    t0_ns::UInt64,
    span::_EngineSpan,
    failure,
    max_new_tokens=nothing,
)
    kv_len = s.seqlen
    return new_receipt(
        task=task,
        model=nameof(typeof(s.model)),
        inference_request=(
            backend=backend_name(s.backend),
            page_size=s.page_size,
            max_new_tokens=max_new_tokens,
            eos_token_id=s.eos_token_id,
        ),
        timing=(
            prefill_ns=span.prefill_ns,
            decode_ns=span.decode_ns,
            total_ns=time_ns() - t0_ns,
            ttft_ns=span.prefill_ns + (span.new_tokens > 0 ? span.decode_ns : UInt64(0)),
        ),
        token_usage=(
            prompt_tokens=span.prompt_tokens,
            new_tokens=span.new_tokens,
            total_tokens=span.prompt_tokens + span.new_tokens,
        ),
        memory_usage=(
            kv_bytes=kv_bytes(s.mgr),
            page_count=page_count(s.mgr),
            kv_len=kv_len,
            context_length=s.context_length,
            context_remaining=s.context_length - kv_len,
        ),
        failure=failure,
        context=Dict{Symbol, Any}(:gap_class => :algorithm),   # §L label, not a detective
    )
end

# wrap one engine call: run f(span) → result, build the receipt from the
# span (on success) or from the caught error (on failure), emit exactly one
# receipt, then rethrow the ORIGINAL error. emit! never throws (receipts.jl
# law) — a telemetry failure cannot change ids or suppress the throw.
function _audited(f, s::Session, task::Symbol; max_new_tokens=nothing)
    t0 = time_ns()
    span = _EngineSpan()
    result = nothing
    try
        result = f(span)
    catch err
        # failure receipt carries the CONSTRUCTED error; the throw still
        # propagates (§LXXIX item A)
        receipt = _engine_receipt(s, task, t0, span, err, max_new_tokens)
        emit!(s.sink, receipt)
        rethrow()
    end
    receipt = _engine_receipt(s, task, t0, span, nothing, max_new_tokens)
    emit!(s.sink, receipt)
    return result
end

# --- prefill! (§XXX: prompt ingestion is PrefillWorkload, once) ----------------

"""
    prefill!(s, tokens) -> logits

Consume the prompt once. Returns host-visible logits `(vocab, P)` (column t
predicts token t+1). Requires a fresh session (multi-turn extension is not
this sprint). Throws `ERR_INVALID_PLAN` on an empty prompt or a reused
session; `ERR_RESOURCE_LIMIT` when the prompt exceeds `context_length`.
"""
function prefill!(s::Session, tokens::AbstractVector{Int})
    return _audited(s, :prefill) do span
        _prefill_impl!(s, tokens, span)
    end
end

function _prefill_impl!(s::Session, tokens::AbstractVector{Int}, span::_EngineSpan)
    isempty(tokens) &&
        throw(gesso_error(ERR_INVALID_PLAN, "prefill!: token sequence is empty"))
    (!s.ready && s.seqlen == 0) || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "prefill!: session already consumed $(s.seqlen) token(s) — " *
            "multi-turn extension is not supported this sprint; build a new Session",
        ),
    )
    P = length(tokens)
    P <= s.context_length || throw(
        gesso_error(
            ERR_RESOURCE_LIMIT,
            "prefill!: prompt length $P exceeds context_length $(s.context_length)";
            prompt_length=P,
            context_length=s.context_length,
        ),
    )
    t_prefill = time_ns()

    cpu = s.backend
    on_cpu = backend_name(cpu) === :cpu
    wl = PrefillWorkload()
    model, tensors = s.model, s.tensors
    dim = model.embedding.dim
    n_heads, n_kv_heads, d_head, group = s.n_heads, s.n_kv_heads, s.d_head, s.group
    kdim = n_kv_heads * d_head

    zeros_like = dims -> _session_zeros_like(tensors, dims)
    normed = Activation(; shape=(P, dim), storage=zeros_like((P, dim)))
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

    positions = collect(0:(P-1))            # 0-based (§LXXV)
    # rows 1..P of the session's h buffer, through a VIEW (same discipline as
    # the oracle: scratch stays exactly (P, …)-shaped so softmax never sees
    # padded zero columns)
    hp = Activation(; shape=(P, dim), storage=@view s.h[1:P, :])
    embedding_lookup!(cpu, hp, tensors.embedding, tokens, wl)

    for (bi, blk) in enumerate(model.blocks)
        bt = tensors.blocks[bi]
        rmsnorm!(cpu, normed, hp, bt.attn_rms, wl; eps=s.eps)
        matmul!(cpu, q, normed, bt.wq, wl)
        matmul!(cpu, k, normed, bt.wk, wl)
        matmul!(cpu, v, normed, bt.wv, wl)
        _split_heads!(qh, q, n_heads, d_head)
        _split_heads!(kh, k, n_kv_heads, d_head)
        _split_heads!(vh, v, n_kv_heads, d_head)
        rope!(cpu, qh, kh, positions, wl; theta=s.theta)
        # KV APPEND through the paged manager: post-RoPE rows 1..P, one page
        # row per token, K and V as a pair (§LXXVIII)
        for t in 1:P
            @views append_kv!(s.mgr, bi, kh.storage[t, :, :], vh.storage[t, :, :])
        end
        # gather pages into the contiguous scratch the contraction reads —
        # byte-identical to the oracle's cache rows on CPU. NOTE: _repeat_heads
        # RETURNS an Activation (oracle helper contract) — use it directly.
        kx = _repeat_heads(
            Activation(; shape=(P, n_kv_heads, d_head), storage=gather_kv(s.mgr, bi, :k)),
            group,
        )
        vx = _repeat_heads(
            Activation(; shape=(P, n_kv_heads, d_head), storage=gather_kv(s.mgr, bi, :v)),
            group,
        )
        # scores per head: (P, P), scaled by √d_head — SAME loop order as the
        # oracle prefill (bit-identical on CPU)
        fill!(scores.storage, zero(eltype(scores.storage)))
        if on_cpu
            for t in 1:P, u in 1:P, hh in 1:n_heads, j in 1:d_head
                scores.storage[t, u] +=
                    qh.storage[t, hh, j] * kx.storage[u, hh, j] / sqrt(d_head)
            end
        elseif Gesso.supports(cpu, :attn_gemm)
            # CUDA fast path (Phase 10 C): the same contraction, as one flat
            # GEMM over (P, n_heads·d_head) reshapes — a plain device `mul!`
            # (CUBLAS) on the SAME gathered scratch. Pages remain the cache;
            # gather remains legal; no page-table kernel (that is the packet).
            # reshape(CuArray) is zero-copy and stays a CuArray (probed); the
            # non-contiguous inner axis disappears in the 2-D flatten.
            Q2 = reshape(qh.storage, P, n_heads * d_head)
            K2 = reshape(kx.storage, P, n_heads * d_head)
            mul!(scores.storage, Q2, transpose(K2))
            scores.storage ./= sqrt(d_head)
        else
            # Device backends without the :attn_gemm cap (Lava): the original
            # per-head broadcasts, unchanged.
            for hh in 1:n_heads, j in 1:d_head
                @views scores.storage[:, :] .+=
                    qh.storage[:, hh, j] .* kx.storage[:, hh, j]' ./ sqrt(d_head)
            end
        end
        softmax!(cpu, scores_out, scores, wl)   # causal mask applied inside
        # attention @ v, per head
        fill!(attn.storage, zero(eltype(attn.storage)))
        if on_cpu
            for t in 1:P, hh in 1:n_heads, j in 1:d_head, u in 1:P
                attn.storage[t, hh, j] += scores_out.storage[t, u] * vx.storage[u, hh, j]
            end
        elseif Gesso.supports(cpu, :attn_gemm)
            # PV as one flat GEMM too: (P,P)·(P, H·d) → (P, H·d), reshaped back
            # in place over the SAME (P, n_heads, d_head) buffer.
            A2 = reshape(attn.storage, P, n_heads * d_head)
            V2 = reshape(vx.storage, P, n_heads * d_head)
            mul!(A2, scores_out.storage, V2)
        else
            for hh in 1:n_heads, j in 1:d_head
                @views attn.storage[:, hh, j] .= scores_out.storage * vx.storage[:, hh, j]
            end
        end
        _merge_heads!(merged, attn, n_heads, d_head)
        matmul!(cpu, sub, merged, bt.wo, wl)
        hp.storage .+= sub.storage              # residual
        rmsnorm!(cpu, normed2, hp, bt.ffn_rms, wl; eps=s.eps)
        matmul!(cpu, gate, normed2, bt.wgate, wl)
        matmul!(cpu, up, normed2, bt.wup, wl)
        swiglu!(cpu, act, gate, up, wl)
        matmul!(cpu, down, act, bt.wdown, wl)
        hp.storage .+= down.storage             # residual
    end

    if haskey(tensors, :final_rms) && tensors.final_rms !== nothing
        finaln = Activation(; shape=(P, dim), storage=zeros_like((P, dim)))
        rmsnorm!(cpu, finaln, hp, tensors.final_rms, wl; eps=s.eps)
        hp = finaln
    end

    seqvocab = Activation(; shape=(P, s.vocab), storage=zeros_like((P, s.vocab)))
    matmul!(cpu, seqvocab, hp, tensors.lm_head, wl)
    s.seqlen = P
    s.ready = true
    span.prefill_ns = UInt64(time_ns() - t_prefill)
    span.prompt_tokens = P
    logits = permutedims(seqvocab.storage)       # (vocab, P)
    return on_cpu ? logits : Array(logits)       # host-visible (§LXXVII)
end

# --- decode! (§XXX: one token at a time, DecodeWorkload) -----------------------

"""
    decode!(s) -> Int

Greedy 0-based id of the next token from the current state. Unless the id is
the session's `eos_token_id`, the token is CONSUMED (KV rows appended through
the paged manager, hidden row written) before it is returned; on EOS nothing
is appended (the engine does not step past a stop token).
"""
function decode!(s::Session)
    return _audited(s, :decode) do span
        _decode_impl!(s, span)
    end
end

function _decode_impl!(s::Session, span::_EngineSpan)
    s.ready || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "decode!: session has no prefill — call prefill! first",
        ),
    )
    cpu = s.backend
    on_cpu = backend_name(cpu) === :cpu
    wl = DecodeWorkload()    # _last_logits consumes an Activation (oracle helper contract) — wrap the
    # session's h buffer; storage CONTENT is shared, nothing is copied
    t_decode = time_ns()
    if on_cpu
        # CPU oracle path (bit-identical): full (vocab,) logits row, host argmax.
        h_act = Activation(; shape=(s.context_length, s.model.embedding.dim), storage=s.h)
        logits_row =
            _last_logits(s.model, s.tensors, h_act, cpu, wl, s.vocab, s.seqlen; eps=s.eps)
        next = _greedy_id(logits_row)
    elseif Gesso.supports(cpu, :argmax)
        # Device fast path (Phase 10 B): the (vocab,) logits row NEVER crosses
        # back to the host — argmax runs where the logits live and the host
        # receives ONE integer (§LXXVII explicit transfers; §LXX no silent
        # downgrades — same lm_head matmul, same reduction semantics, ties =
        # first index on both paths).
        next = _device_greedy_id(
            s.model,
            s.tensors,
            s.h,
            cpu,
            wl,
            s.vocab,
            s.seqlen;
            eps=s.eps,
        )
    else
        # Device backends without the :argmax cap (Lava): full (vocab,) row
        # crosses back, host argmax — §LXXVIII semantics unchanged.
        h_act = Activation(; shape=(s.context_length, s.model.embedding.dim), storage=s.h)
        logits_row =
            _last_logits(s.model, s.tensors, h_act, cpu, wl, s.vocab, s.seqlen; eps=s.eps)
        next = _greedy_id(logits_row)
    end
    if next != s.eos_token_id                    # EOS: returned, nothing appended
        _session_consume!(s, next)
    end
    span.decode_ns += UInt64(time_ns() - t_decode)
    span.new_tokens += 1
    return next
end

# consume one token: embed, rope at its 0-based position, append K/V rows
# through the manager, attend over gathered pages 1..K, write the h row
function _session_consume!(s::Session, tok::Int)
    cpu = s.backend
    on_cpu = backend_name(cpu) === :cpu
    wl = DecodeWorkload()
    model, tensors = s.model, s.tensors
    dim = model.embedding.dim
    n_heads, n_kv_heads, d_head, group = s.n_heads, s.n_kv_heads, s.d_head, s.group
    kdim = n_kv_heads * d_head

    pos0 = s.seqlen + 1                          # 1-based row this token occupies
    K = pos0                                     # cache rows after the append
    pos0 <= s.context_length || throw(
        gesso_error(
            ERR_RESOURCE_LIMIT,
            "decode!: context_length $(s.context_length) exhausted at token $pos0";
            context_length=s.context_length,
        ),
    )

    zeros_like = dims -> _session_zeros_like(tensors, dims)
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
    scores = TemporaryWorkspace(; shape=(1, K), storage=zeros_like((1, K)))
    scores_out = TemporaryWorkspace(; shape=(1, K), storage=zeros_like((1, K)))

    for (bi, blk) in enumerate(model.blocks)
        bt = tensors.blocks[bi]
        rmsnorm!(cpu, normed, hp, bt.attn_rms, wl; eps=s.eps)
        matmul!(cpu, q, normed, bt.wq, wl)
        matmul!(cpu, k, normed, bt.wk, wl)
        matmul!(cpu, v, normed, bt.wv, wl)
        _split_heads!(qh, q, n_heads, d_head)
        _split_heads!(kh, k, n_kv_heads, d_head)
        _split_heads!(vh, v, n_kv_heads, d_head)
        rope!(cpu, qh, kh, [pos0 - 1], wl; theta=s.theta)   # 0-based position
        # KV APPEND through the paged manager (K and V as a pair), THEN gather
        # rows 1..K — identical bytes and loop order to the oracle's decode.
        # _repeat_heads RETURNS an Activation — use it directly (oracle contract).
        @views append_kv!(s.mgr, bi, kh.storage[1, :, :], vh.storage[1, :, :])
        kx = _repeat_heads(
            Activation(; shape=(K, n_kv_heads, d_head), storage=gather_kv(s.mgr, bi, :k)),
            group,
        )
        vx = _repeat_heads(
            Activation(; shape=(K, n_kv_heads, d_head), storage=gather_kv(s.mgr, bi, :v)),
            group,
        )
        # scores over the gathered cache — SAME loop order as the oracle decode
        fill!(scores.storage, zero(eltype(scores.storage)))
        if on_cpu
            for u in 1:K, hh in 1:n_heads, j in 1:d_head
                scores.storage[1, u] +=
                    qh.storage[1, hh, j] * kx.storage[u, hh, j] / sqrt(d_head)
            end
        elseif Gesso.supports(cpu, :attn_gemm)
            # CUDA fast path (Phase 10 C): one (1, H·d)·(H·d, K) GEMM row over
            # the SAME gathered scratch — CUBLAS on reshape views, no per-head
            # Julia loop, no page-table kernel (that is the packet).
            Q1 = reshape(qh.storage, 1, n_heads * d_head)
            K2 = reshape(kx.storage, K, n_heads * d_head)
            mul!(scores.storage, Q1, transpose(K2))
            scores.storage ./= sqrt(d_head)
        else
            for hh in 1:n_heads
                q1 = vec(qh.storage[1, hh, :])           # (d_head,) device copy
                Kmat = kx.storage[1:K, hh, :]            # (K, d_head)
                @views scores.storage[1, :] .+= (Kmat * q1) ./ sqrt(d_head)
            end
        end
        softmax!(cpu, scores_out, scores, wl)   # offset mask: nothing masked
        fill!(attn.storage, zero(eltype(attn.storage)))
        if on_cpu
            for hh in 1:n_heads, j in 1:d_head, u in 1:K
                attn.storage[1, hh, j] += scores_out.storage[1, u] * vx.storage[u, hh, j]
            end
        elseif Gesso.supports(cpu, :attn_gemm)
            # PV: (1,K)·(K, H·d) → (1, H·d), reshaped back in place.
            A1 = reshape(attn.storage, 1, n_heads * d_head)
            V2 = reshape(vx.storage, K, n_heads * d_head)
            mul!(A1, scores_out.storage, V2)
        else
            for hh in 1:n_heads
                V = vx.storage[1:K, hh, :]               # (K, d_head)
                @views attn.storage[1, hh, :] .= vec(scores_out.storage * V)
            end
        end
        _merge_heads!(merged, attn, n_heads, d_head)
        matmul!(cpu, sub, merged, bt.wo, wl)
        hp.storage .+= sub.storage              # residual
        rmsnorm!(cpu, normed2, hp, bt.ffn_rms, wl; eps=s.eps)
        matmul!(cpu, gate, normed2, bt.wgate, wl)
        matmul!(cpu, up, normed2, bt.wup, wl)
        swiglu!(cpu, act, gate, up, wl)
        matmul!(cpu, down, act, bt.wdown, wl)
        hp.storage .+= down.storage             # residual
    end
    # write the new hidden row (storage CONTENT, not a field reassignment)
    @views s.h[pos0, :] .= hp.storage[1, :]
    s.seqlen = pos0
    return s
end

# --- generate (§LXVII: the simple thing) ----------------------------------------

"""
    generate(s, prompt::AbstractVector{Int}; max_new_tokens=8, on_token=nothing) -> Vector{Int}

Reset the session, prefill the prompt, then greedy-decode up to
`max_new_tokens` new ids. Returns prompt + new ids (same shape as
`reference_generate`). `on_token(id::Int)` fires once per NEW token,
including EOS if produced. Stops on `eos_token_id` (returned, never
stepped past).
"""
function generate(
    s::Session,
    prompt::AbstractVector{Int};
    max_new_tokens::Int=8,
    on_token=nothing,
)
    return _audited(s, :generate; max_new_tokens) do span
        _generate_impl!(s, prompt, max_new_tokens, on_token, span)
    end
end

function _generate_impl!(
    s::Session,
    prompt::AbstractVector{Int},
    max_new_tokens::Int,
    on_token,
    span::_EngineSpan,
)
    isempty(s.model.blocks) &&
        throw(gesso_error(ERR_INVALID_PLAN, "generate: model has no blocks"))
    isempty(prompt) && throw(gesso_error(ERR_INVALID_PLAN, "generate: prompt is empty"))
    _session_reset!(s)
    _prefill_impl!(s, prompt, span)
    ids = collect(prompt)
    cap = max(max_new_tokens, 0)
    steps = 0
    while steps < cap
        next = _decode_impl!(s, span)
        push!(ids, next)
        on_token === nothing || on_token(next)
        steps += 1
        (next == s.eos_token_id || steps >= cap) && break
    end
    return ids
end

"""
    generate(s, text::AbstractString; max_new_tokens=8, on_token=nothing) -> Vector{Int}

String prompt: `encode(s.tokenizer, text)`. Throws `ERR_INVALID_PLAN` when
the session has no tokenizer (§LXXVIII).
"""
function generate(s::Session, text::AbstractString; max_new_tokens::Int=8, on_token=nothing)
    s.tokenizer === nothing && throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "generate: session has no tokenizer — pass tokenizer=… to Session " *
            "or call generate with an integer prompt",
        ),
    )
    return generate(
        s,
        encode(s.tokenizer, text);
        max_new_tokens=max_new_tokens,
        on_token=on_token,
    )
end

# --- fork (§LXXX: declared identity prefix share, Magenta §9.5 step 3) --------

"""
    fork(s::Session; sink=default_receipt_sink()) -> Session

DECLARED identity prefix share (§LXXX; Magenta §9.5 step 3): the ONLY share
constructor in Gesso. Sharing is never discovered by token match — two
Sessions that prefill the same tokens independently do NOT share; they
share because `fork` was called.

Legal after `prefill!` (and after subsequent `decode!`s); on a Session that
is not `ready` it throws `ERR_INVALID_PLAN`. The child:

  * carries the same `model`, `tensors`, `backend`, `page_size`,
    `context_length`, `eos_token_id`, `tokenizer`, `eps`, `theta`,
  * owns a NEW `PagedKVManager` whose page lists are new vectors holding the
    SAME `KVPage` objects, each marked `shared=true` — copy-on-write in
    `append_kv!` copies only the page being written (kv_manager.jl),
  * has its own COPIED hidden buffer `h` plus copied `seqlen`/`ready` (no
    hidden-state CoW this sprint),
  * owns its own `sink` (inject one with `sink=`; tests do).

`fork` itself emits NO receipt: it is a declaration of sharing, not an
inference step (§LXXIX receipts record prefill!/decode!/generate). A forked
child that calls `generate` still RESETS — and so DROPS the share (§LXXVIII
law unchanged). The share path is `prefill!` / `fork` / `decode!`.
"""
function fork(s::Session; sink::ReceiptSink=default_receipt_sink())
    s.ready || throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "fork: session is not ready — identity prefix share is legal " *
            "after prefill! (and subsequent decode!s); this session has " *
            "consumed $(s.seqlen) token(s) (§LXXX)",
        ),
    )
    child = Session(
        s.model,
        s.tensors;
        backend=s.backend,
        page_size=s.page_size,
        context_length=s.context_length,
        eos_token_id=s.eos_token_id,
        tokenizer=s.tokenizer,
        eps=s.eps,
        theta=s.theta,
        sink=sink,
    )
    _alias_pages!(child.mgr, s.mgr)
    child.h = copy(s.h)                # hidden state is COPIED (§LXXX) —
    child.seqlen = s.seqlen            # device storage copies on device
    child.ready = true                 # (§LXXVII: no host round trip)
    return child
end

export Session, decode!, fork, generate, prefill!
