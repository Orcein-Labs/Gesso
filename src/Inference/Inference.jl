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
    Activation,
    EmbeddingTable,
    ProjectionWeight,
    FrozenParameter,
    TemporaryWorkspace,
    backend_name,
    _backend_storage_root,
    gesso_error,
    LoweringNotImplemented,
    ERR_INVALID_PLAN,
    ERR_RESOURCE_LIMIT,
    ERR_NUMERICAL_INSTABILITY,
    ReceiptSink,
    default_receipt_sink,
    emit!,
    new_receipt,
    embedding_lookup!,
    rmsnorm!,
    rope!,
    matmul!,
    softmax!,
    swiglu!,
    _cpu_gated_delta_rule!,
    _cpu_causal_conv1d_silu!,
    _cpu_sigmoid_mul!,
    _cpu_silu,
    _cpu_sigmoid,
    _cpu_softplus,
    _cpu_moe_topk_swiglu!
using ..Semantics: PrefillWorkload, DecodeWorkload
using ..Parameters: KVCache, SemanticTensor, ExpertWeight

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

_block_mixer(bt) = haskey(bt, :mixer) ? bt.mixer : :full_attention

include("decode_profile.jl")       # GESSO_PROFILE=1 decode stage profiler

const _T_GDN_NS = Ref(Int64(0))
const _T_ATTN_NS = Ref(Int64(0))
const _T_FFN_NS = Ref(Int64(0))

function _gdn_states(tensors, zeros_like)
    n = length(tensors.blocks)
    conv = Vector{Any}(undef, n)
    rec = Vector{Any}(undef, n)
    spec = haskey(tensors, :spec) ? tensors.spec : nothing
    mx = spec === nothing ? nothing : spec.mixer
    for bi in 1:n
        if mx !== nothing && _block_mixer(tensors.blocks[bi]) === :linear_attention
            key_dim = mx.num_key_heads * mx.key_head_dim
            value_dim = mx.num_value_heads * mx.value_head_dim
            conv_dim = 2 * key_dim + value_dim
            conv[bi] = zeros_like((conv_dim, mx.conv_kernel_dim - 1))
            rec[bi] = zeros_like((mx.num_value_heads, mx.key_head_dim, mx.value_head_dim))
        else
            conv[bi] = nothing
            rec[bi] = nothing
        end
    end
    return conv, rec
end

function _tensor_head_dim(model, tensors)
    if tensors isa NamedTuple && haskey(tensors, :spec) && tensors.spec !== nothing
        return tensors.spec.head_dim
    end
    n_heads = model.blocks[1].attention.n_heads
    dim = model.embedding.dim
    dim % n_heads == 0 ||
        error("dim $dim is not divisible by n_heads $n_heads")
    return div(dim, n_heads)
end

function _tensor_rotary_dim(tensors)
    policy = tensors isa NamedTuple && haskey(tensors, :rope) ? tensors.rope : nothing
    return policy === nothing ? 0 : Int(policy.rotary_dim)
end

function reference_prefill(
    model,
    tensors,
    tokens::AbstractVector{Int};
    backend::AbstractGessoBackend=CPUBackend(),
    eps::Real=1e-6,
    theta=nothing,
)
    isempty(tokens) && error("reference_prefill: token sequence is empty")
    theta=_resolved_rope_theta(tensors, theta)
    _infer_device_storage!(:reference_prefill, backend, tensors)
    cpu = backend
    wl = PrefillWorkload()
    seq = length(tokens)
    dim = model.embedding.dim
    n_heads = model.blocks[1].attention.n_heads
    n_kv_heads = model.blocks[1].attention.n_kv_heads
    d_head = _tensor_head_dim(model, tensors)
    qdim = n_heads * d_head
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
    inv_freq = tensors_rope_inv_freq(tensors, d_head; theta)
    rotary_dim = _tensor_rotary_dim(tensors)
    on_cpu = backend_name(cpu) === :cpu
    zeros_like = _zeros_like_factory(cpu, tensors)

    # hidden states: (seq, dim)
    h = Activation(; shape=(seq, dim), storage=zeros_like((seq, dim)))
    embedding_lookup!(cpu, h, tensors.embedding, tokens, wl)

    # scratch — allocated once per forward, written in place
    normed = Activation(; shape=(seq, dim), storage=zeros_like((seq, dim)))
    q = Activation(; shape=(seq, qdim), storage=zeros_like((seq, qdim)))
    gate_attn = Activation(; shape=(seq, qdim), storage=zeros_like((seq, qdim)))
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
    merged = Activation(; shape=(seq, qdim), storage=zeros_like((seq, qdim)))
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
        if _block_mixer(bt) === :linear_attention
            _hybrid_linear_block!(
                cpu,
                h,
                normed,
                sub,
                bt,
                tensors.spec,
                wl,
                eps,
                zeros_like,
            )
            h.storage .+= sub.storage
        else
        rmsnorm!(cpu, normed, h, bt.attn_rms, wl; eps)
        matmul!(cpu, q, normed, bt.wq, wl)
        matmul!(cpu, k, normed, bt.wk, wl)
        matmul!(cpu, v, normed, bt.wv, wl)

        # split heads: feature f (0-based) = head * d_head + j (head-major
        # packing, matching the packed Wk/Wv rows of the weight walk)
        _split_heads!(qh, q, n_heads, d_head)
        _split_heads!(kh, k, n_kv_heads, d_head)
        _split_heads!(vh, v, n_kv_heads, d_head)
        if haskey(bt, :wq_norm)
            rmsnorm!(cpu, qh, qh, bt.wq_norm, wl; eps)
            rmsnorm!(cpu, kh, kh, bt.wk_norm, wl; eps)
        end

        rope!(
            cpu,
            qh,
            kh,
            positions,
            wl;
            theta,
            inv_freq,
            interleaved=_tensors_rope_interleaved(tensors),
            rotary_dim,
        )

        # repeat KV heads for the contraction only (identity when MHA)
        kx = _repeat_heads(kh, group)
        vx = _repeat_heads(vh, group)

        _attention_heads!(
            cpu,
            attn.storage,
            scores,
            scores_out,
            qh.storage,
            kx.storage,
            vx.storage,
            n_heads,
            d_head,
            seq,
            seq,
            wl,
        )

        _merge_heads!(merged, attn, n_heads, d_head)
        if haskey(bt, :wattn_gate)
            matmul!(cpu, gate_attn, normed, bt.wattn_gate, wl)
            _sigmoid_mul!(cpu, merged.storage, merged.storage, gate_attn.storage)
        end
        matmul!(cpu, sub, merged, bt.wo, wl)
        h.storage .+= sub.storage               # residual (interpreter add)
        end

        # --- ffn sublayer (pre-norm) -----------------------------------------
        rmsnorm!(cpu, normed2, h, bt.ffn_rms, wl; eps)
        _ffn_block!(cpu, down, gate, up, act, normed2, bt, wl, zeros_like)
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

# Hybrid extras as Inference-internal dispatch (Phase 10F pattern). Default
# bodies are the CPU oracle; CUDA/Lava extensions add more-specific methods
# on their backend tag. These are not public `op!` stubs — planners still
# see :linear_attention via `supports`, and the engine composes.
function _gated_delta_rule!(
    backend,
    out,
    q,
    k,
    v,
    g,
    beta,
    state;
    l2norm::Bool=true,
    v_head_tiled::Bool=false,
)
    return _cpu_gated_delta_rule!(out, q, k, v, g, beta, state; l2norm, v_head_tiled)
end

function _causal_conv1d_silu!(backend, dst, x, weight, conv_state)
    return _cpu_causal_conv1d_silu!(dst, x, weight, conv_state)
end

function _sigmoid_mul!(backend, dst, src, gate)
    return _cpu_sigmoid_mul!(dst, src, gate)
end

function _silu_mul!(backend, dst, src, gate)
    @inbounds for i in eachindex(dst, src, gate)
        dst[i] = src[i] * _cpu_silu(gate[i])
    end
    return dst
end

function _gdn_gate_params!(backend, g, beta, a, b, A_log, dt_bias)
    seq, nv = size(beta)
    @inbounds for t in 1:seq, hh in 1:nv
        beta[t, hh] = _cpu_sigmoid(b[t, hh])
        g[t, hh] = -exp(A_log[hh]) * _cpu_softplus(a[t, hh] + dt_bias[hh])
    end
    return g, beta
end

function _hybrid_linear_block!(
    cpu,
    h::Activation,
    normed::Activation,
    sub::Activation,
    bt,
    spec,
    wl,
    eps,
    zeros_like;
    conv_state=nothing,
    rec_state=nothing,
)
    mx = spec.mixer
    seq = size(h.storage, 1)
    key_dim = mx.num_key_heads * mx.key_head_dim
    value_dim = mx.num_value_heads * mx.value_head_dim
    conv_dim = 2 * key_dim + value_dim
    nk, nv = mx.num_key_heads, mx.num_value_heads
    dk, dv = mx.key_head_dim, mx.value_head_dim
    ksz = mx.conv_kernel_dim

    rmsnorm!(cpu, normed, h, bt.attn_rms, wl; eps)
    mixed = Activation(; shape=(seq, conv_dim), storage=zeros_like((seq, conv_dim)))
    z = Activation(; shape=(seq, value_dim), storage=zeros_like((seq, value_dim)))
    a = Activation(; shape=(seq, nv), storage=zeros_like((seq, nv)))
    b = Activation(; shape=(seq, nv), storage=zeros_like((seq, nv)))
    matmul!(cpu, mixed, normed, bt.wgdn_qkv, wl)
    matmul!(cpu, z, normed, bt.wgdn_z, wl)
    matmul!(cpu, a, normed, bt.wgdn_a, wl)
    matmul!(cpu, b, normed, bt.wgdn_b, wl)

    conv_out = zeros_like((seq, conv_dim))
    if conv_state === nothing
        conv_state = zeros_like((conv_dim, ksz - 1))
    end
    _causal_conv1d_silu!(cpu, conv_out, mixed.storage, bt.wgdn_conv.storage, conv_state)

    q = zeros_like((seq, nk, dk))
    k = zeros_like((seq, nk, dk))
    v = zeros_like((seq, nv, dv))
    _split_heads!(q, view(conv_out, :, 1:key_dim), nk, dk)
    _split_heads!(k, view(conv_out, :, (key_dim+1):(2*key_dim)), nk, dk)
    _split_heads!(v, view(conv_out, :, (2*key_dim+1):conv_dim), nv, dv)

    g = zeros_like((seq, nv))
    beta = zeros_like((seq, nv))
    A_log = bt.wgdn_A_log.storage
    dt_bias = bt.wgdn_dt_bias.storage
    _gdn_gate_params!(cpu, g, beta, a.storage, b.storage, A_log, dt_bias)

    core = zeros_like((seq, nv, dv))
    if rec_state === nothing
        rec_state = zeros_like((nv, dk, dv))
    end
    _gated_delta_rule!(
        cpu, core, q, k, v, g, beta, rec_state;
        v_head_tiled=mx.v_head_tiled,
    )

    z_h = zeros_like((seq, nv, dv))
    _split_heads!(z_h, z.storage, nv, dv)
    nrm = zeros_like((seq, dv))
    xh = zeros_like((seq, dv))
    scale = bt.wgdn_norm
    for hh in 1:nv
        @views xh[:, :] .= core[:, hh, :]
        xh_act = Activation(; shape=(seq, dv), storage=xh)
        nrm_act = Activation(; shape=(seq, dv), storage=nrm)
        rmsnorm!(cpu, nrm_act, xh_act, scale, wl; eps)
        @views xh[:, :] .= z_h[:, hh, :]
        _silu_mul!(cpu, nrm, nrm, xh)
        @views core[:, hh, :] .= nrm
    end
    merged = zeros_like((seq, value_dim))
    _merge_heads!(merged, core, nv, dv)
    merged_act = Activation(; shape=(seq, value_dim), storage=merged)
    matmul!(cpu, sub, merged_act, bt.wgdn_o, wl)
    return sub
end

function _ffn_block!(cpu, down, gate, up, act, normed2, bt, wl, zeros_like)
    if haskey(bt, :wrouter)
        return _moe_ffn!(cpu, down, normed2, bt, wl, zeros_like)
    end
    @decode_stage :dense_ffn cpu begin
        matmul!(cpu, gate, normed2, bt.wgate, wl)
        matmul!(cpu, up, normed2, bt.wup, wl)
        swiglu!(cpu, act, gate, up, wl)
        matmul!(cpu, down, act, bt.wdown, wl)
    end
    return down
end

function _moe_ffn!(cpu, down, normed2, bt, wl, zeros_like)
    x = normed2.storage
    y = down.storage
    n_used = haskey(bt, :n_expert_used) ? Int(bt.n_expert_used) : size(bt.wrouter.storage, 1)
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
    if haskey(bt, :wshared_gate)
        @decode_stage :shared_expert cpu begin
            seq, h = size(x)
            ff = size(bt.wshared_gate.storage, 1)
            sg = Activation(; shape=(seq, ff), storage=zeros_like((seq, ff)))
            su = Activation(; shape=(seq, ff), storage=zeros_like((seq, ff)))
            sa = Activation(; shape=(seq, ff), storage=zeros_like((seq, ff)))
            sd = Activation(; shape=(seq, h), storage=zeros_like((seq, h)))
            matmul!(cpu, sg, normed2, bt.wshared_gate, wl)
            matmul!(cpu, su, normed2, bt.wshared_up, wl)
            swiglu!(cpu, sa, sg, su, wl)
            matmul!(cpu, sd, sa, bt.wshared_down, wl)
            if haskey(bt, :wshared_gate_inp)
                _shared_gate_add!(cpu, y, x, sd.storage, bt.wshared_gate_inp.storage)
            else
                y .+= sd.storage
            end
        end
    end
    return down
end

# HostOffload unwraps in gguf_import.jl; this generic keeps dense Array /
# CuArray / LavaArray banks pass-through for any MoE config.
_offload_host_array(s) = s

function _moe_topk_swiglu!(backend, y, x, router, gate, up, down, n_used)
    if gate isa GgufQuantSlice || gate isa GpuQuantSlice
        return _cpu_moe_topk_swiglu_gguf!(y, x, router, gate, up, down, n_used)
    end
    return _cpu_moe_topk_swiglu!(
        y,
        x,
        _offload_host_array(router),
        _offload_host_array(gate),
        _offload_host_array(up),
        _offload_host_array(down),
        n_used,
    )
end

function _shared_gate_add!(backend, y, x, sd, gvec)
    g = _offload_host_array(gvec)
    seq, h = size(x)
    @inbounds for t in 1:seq
        acc = zero(eltype(x))
        for i in 1:h
            acc += x[t, i] * g[i]
        end
        s = _cpu_sigmoid(acc)
        for i in 1:h
            y[t, i] += s * sd[t, i]
        end
    end
    return y
end

"""
Copy expert `e` (1-based) from a 3-D bank into contiguous `(out, in)` `W`.
Geometry comes from the bank: `(n_experts, out, in)` for any MoE config.
The CUDA method dequants a `GgufQuantSlice` on the host and H2Ds, or
gathers a device bank.
"""
function _fetch_expert!(backend, W::AbstractMatrix, bank, e::Int)
    if bank isa GgufQuantSlice
        raw = Vector{Float32}(undef, bank.ne0 * bank.ne1)
        return _gguf_dequant_expert_into!(W, raw, bank, e)
    end
    src = _offload_host_array(bank)
    @inbounds for b in axes(W, 2), a in axes(W, 1)
        W[a, b] = src[e, a, b]
    end
    return W
end

# Each query head owns its softmax distribution. The 2D score workspace
# is reused head by head; reducing across heads before softmax changes the
# model. GQA repetition has already mapped each query head to its KV head.
function _attention_heads!(
    backend,
    attn::AbstractArray,
    scores,
    scores_out,
    q::AbstractArray,
    k::AbstractArray,
    v::AbstractArray,
    n_heads::Int,
    d_head::Int,
    L::Int,
    K::Int,
    workload,
)
    sc, probs = scores.storage, scores_out.storage
    on_cpu = backend_name(backend) === :cpu
    on_cpu && fill!(attn, zero(eltype(attn)))
    for hh in 1:n_heads
        if on_cpu
            fill!(sc, zero(eltype(sc)))
            _attention_scores_cpu!(sc, q, k, hh, d_head, L, K)
        else
            # Ordinary per-head GEMM over explicit device slices. No host
            # transfer, no cross-head contraction, no padded keys.
            _attention_scores_device!(backend, sc, q, k, hh, d_head, K)
        end
        softmax!(backend, scores_out, scores, workload)
        if on_cpu
            _attention_values_cpu!(attn, probs, v, hh, d_head, L, K)
        else
            _attention_values_device!(backend, attn, probs, v, hh, K)
        end
    end
    return attn
end

# Backend-specific in-place realizations may replace these ordinary slice
# contractions without altering the shared attention/softmax semantics.
function _attention_scores_device!(
    backend,
    sc,
    q::AbstractArray,
    k::AbstractArray,
    hh,
    d_head,
    K,
)
    qh, kh = q[:, hh, :], k[1:K, hh, :]
    sc .= (qh * transpose(kh)) ./ sqrt(d_head)
    return sc
end
function _attention_values_device!(
    backend,
    attn::AbstractArray,
    probs,
    v::AbstractArray,
    hh,
    K,
)
    vh = v[1:K, hh, :]
    @views attn[:, hh, :] .= probs * vh
    return attn
end

function _attention_scores_cpu!(sc, q, k, hh, d_head, L, K)
    for t in 1:L, u in 1:K, j in 1:d_head
        sc[t, u] += q[t, hh, j] * k[u, hh, j] / sqrt(d_head)
    end
    return sc
end

function _attention_values_cpu!(attn, probs, v, hh, d_head, L, K)
    for t in 1:L, j in 1:d_head, u in 1:K
        attn[t, hh, j] += probs[t, u] * v[u, hh, j]
    end
    return attn
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

"""
    _gpu_budget_bytes() -> Int64

Device-resident weight budget for `to_device`. `GESSO_GPU_BUDGET_MIB`
(default 6144) is the 8 GB 5060 with headroom for activations, KV, and
the active-expert scratch. Float32 weights fill the budget first;
remaining bytes upload `GgufQuantSlice` expert slabs as `GpuQuantSlice`.
"""
function _gpu_budget_bytes()
    raw = get(ENV, "GESSO_GPU_BUDGET_MIB", "6144")
    mib = something(tryparse(Int, raw), 6144)
    return Int64(max(0, mib)) * Int64(1024 * 1024)
end

# Activations follow a live Array (CuArray / LavaArray / host Array), not a
# HostOffload embedding table. When every weight is offloaded, the backend
# extension supplies the device zeros factory.
function _first_array_storage(tensors)
    # F16 device weights are never the activation prototype: activations stay F32.
    hit(s) = s isa AbstractArray && eltype(s) !== Float16 ? s : nothing
    s = hit(tensors.embedding.storage)
    s !== nothing && return s
    if tensors.lm_head !== tensors.embedding
        s = hit(tensors.lm_head.storage)
        s !== nothing && return s
    end
    for b in tensors.blocks
        for v in values(b)
            hasproperty(v, :storage) || continue
            s = hit(v.storage)
            s !== nothing && return s
        end
    end
    fr = haskey(tensors, :final_rms) ? tensors.final_rms : nothing
    fr === nothing && return nothing
    return hit(fr.storage)
end

function _zeros_like_factory(backend, tensors)
    s = _first_array_storage(tensors)
    if s !== nothing
        T = eltype(s)
        return (dims::Tuple{Vararg{Int}}) -> fill!(similar(s, T, dims), zero(T))
    end
    return _fallback_zeros_factory(backend)
end

_fallback_zeros_factory(::CPUBackend) = (dims::Tuple{Vararg{Int}}) -> zeros(Float32, dims)
function _fallback_zeros_factory(backend)
    error(
        "$(backend_name(backend)): to_device left no device array to seed activations",
    )
end

# the interpreter never copies: a non-CPU backend REQUIRES device storage
# (to_device is the explicit transfer). Host Array storage under CUDA is
# ERR_INVALID_PLAN, not a silent transfer (§LXX). HostOffload and
# GgufQuantSlice are the explicit offload markers and are legal on device
# backends — a forgotten raw Array still fails.
function _infer_device_storage!(where::Symbol, backend::AbstractGessoBackend, tensors)
    _check_device_storage(where, backend, tensors.embedding)
    tensors.lm_head === tensors.embedding ||
        _check_device_storage(where, backend, tensors.lm_head)
    for bt in tensors.blocks
        for name in propertynames(bt)
            t = getproperty(bt, name)
            t isa SemanticTensor || continue
            _check_device_storage(where, backend, t)
        end
    end
    fr = haskey(tensors, :final_rms) ? tensors.final_rms : nothing
    fr === nothing || _check_device_storage(where, backend, fr)
    return nothing
end

# Inspect wrappers without copying: CPU views and another device are not
# interchangeable with the explicitly requested backend.
_storage_root(s) = _backend_storage_root(s)

_is_host_float_storage(s) =
    _storage_root(s) isa Array && eltype(s) in (Float32, Float64)

function _check_device_storage(where::Symbol, backend, t)
    throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "$where: unsupported backend storage";
            backend=backend_name(backend),
            storage_type=string(typeof(t.storage)),
        ),
    )
end
function _check_device_storage(where::Symbol, backend::CPUBackend, t)
    _is_host_float_storage(t.storage) && return nothing
    throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "$where: CPUBackend requires materialized host Array float storage";
            backend=:cpu,
            storage_type=string(typeof(t.storage)),
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
    theta=nothing,
)
    isempty(prompt) && error("reference_generate: prompt is empty")
    theta=_resolved_rope_theta(tensors, theta)
    _infer_device_storage!(:reference_generate, backend, tensors)
    cpu = backend
    wp, wd = PrefillWorkload(), DecodeWorkload()
    dim = model.embedding.dim
    n_heads = model.blocks[1].attention.n_heads
    n_kv_heads = model.blocks[1].attention.n_kv_heads
    d_head = _tensor_head_dim(model, tensors)
    rotary_dim = _tensor_rotary_dim(tensors)
    n_heads * d_head > 0 ||
        error("reference_generate: n_heads $n_heads × d_head $d_head is empty")
    group = div(n_heads, n_kv_heads)
    n_heads == n_kv_heads * group ||
        error("reference_generate: n_kv_heads $n_kv_heads does not divide n_heads $n_heads")
    P = length(prompt)
    cap = max(max_new_tokens, 0)
    vocab = size(tensors.embedding.storage, 1)
    on_cpu = backend_name(cpu) === :cpu
    zeros_like = _zeros_like_factory(cpu, tensors)

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
    qdim = n_heads * d_head
    q = Activation(; shape=(P, qdim), storage=zeros_like((P, qdim)))
    gate_attn = Activation(; shape=(P, qdim), storage=zeros_like((P, qdim)))
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
    merged = Activation(; shape=(P, qdim), storage=zeros_like((P, qdim)))
    sub = Activation(; shape=(P, dim), storage=zeros_like((P, dim)))
    normed2 = Activation(; shape=(P, dim), storage=zeros_like((P, dim)))
    hidden_ffn = model.blocks[1].ffn.hidden
    gate = Activation(; shape=(P, hidden_ffn), storage=zeros_like((P, hidden_ffn)))
    up = Activation(; shape=(P, hidden_ffn), storage=zeros_like((P, hidden_ffn)))
    act = Activation(; shape=(P, hidden_ffn), storage=zeros_like((P, hidden_ffn)))
    down = Activation(; shape=(P, dim), storage=zeros_like((P, dim)))
    scores = TemporaryWorkspace(; shape=(P, P), storage=zeros_like((P, P)))
    scores_out = TemporaryWorkspace(; shape=(P, P), storage=zeros_like((P, P)))
    gdn_conv, gdn_rec = _gdn_states(tensors, zeros_like)

    positions = collect(0:(P-1))
    # BREADTH-0 Pass D — see reference_prefill. `nothing` = unscaled model.
    inv_freq = tensors_rope_inv_freq(tensors, d_head; theta)
    t_prefill = time_ns()
    # the prefill phase works on rows 1..P of h through a VIEW (Activation
    # storage is untyped by design; scratch stays exactly (P, …)-shaped so
    # softmax never sees padded zero columns — they would poison the
    # denominators). Residuals broadcast through the view, in place.
    hp = Activation(; shape=(P, dim), storage=@view h.storage[1:P, :])
    embedding_lookup!(cpu, hp, tensors.embedding, prompt, wp)
    for (bi, blk) in enumerate(model.blocks)
        bt = tensors.blocks[bi]
        if _block_mixer(bt) === :linear_attention
            _hybrid_linear_block!(
                cpu,
                hp,
                normed,
                sub,
                bt,
                tensors.spec,
                wp,
                eps,
                zeros_like;
                conv_state=gdn_conv[bi],
                rec_state=gdn_rec[bi],
            )
            hp.storage .+= sub.storage
        else
        rmsnorm!(cpu, normed, hp, bt.attn_rms, wp; eps)
        matmul!(cpu, q, normed, bt.wq, wp)
        matmul!(cpu, k, normed, bt.wk, wp)
        matmul!(cpu, v, normed, bt.wv, wp)
        _split_heads!(qh, q, n_heads, d_head)
        _split_heads!(kh, k, n_kv_heads, d_head)
        _split_heads!(vh, v, n_kv_heads, d_head)
        if haskey(bt, :wq_norm)
            rmsnorm!(cpu, qh, qh, bt.wq_norm, wp; eps)
            rmsnorm!(cpu, kh, kh, bt.wk_norm, wp; eps)
        end
        rope!(
            cpu,
            qh,
            kh,
            positions,
            wp;
            theta,
            inv_freq,
            interleaved=_tensors_rope_interleaved(tensors),
            rotary_dim,
        )
        # KV APPEND: the prefill's post-rope K/V become cache rows 1..P,
        # stored at n_kv_heads (GQA caches the small side, §LXXVI)
        kc[bi].storage[1:P, :, :] .= kh.storage
        vc[bi].storage[1:P, :, :] .= vh.storage
        # repeat KV heads for the contraction only (identity when MHA)
        kx = _repeat_heads(kh, group)
        vx = _repeat_heads(vh, group)
        _attention_heads!(
            cpu,
            attn.storage,
            scores,
            scores_out,
            qh.storage,
            kx.storage,
            vx.storage,
            n_heads,
            d_head,
            P,
            P,
            wp,
        )

        _merge_heads!(merged, attn, n_heads, d_head)
        if haskey(bt, :wattn_gate)
            matmul!(cpu, gate_attn, normed, bt.wattn_gate, wp)
            _sigmoid_mul!(cpu, merged.storage, merged.storage, gate_attn.storage)
        end
        matmul!(cpu, sub, merged, bt.wo, wp)
        hp.storage .+= sub.storage
        end
        rmsnorm!(cpu, normed2, hp, bt.ffn_rms, wp; eps)
        _ffn_block!(cpu, down, gate, up, act, normed2, bt, wp, zeros_like)
        hp.storage .+= down.storage
    end

    ids = collect(prompt)
    steps = 0
    timing = get(ENV, "GESSO_TIMING", "") == "1"
    if timing
        println("gesso_timing prefill_ms=$(round((time_ns() - t_prefill) / 1e6; digits=1))")
        flush(stdout)
    end
    while steps < cap
        tok_t0 = _profile_token_begin()
        # greedy argmax at the current last position (row P + steps); the
        # argmax may run over device storage — only the scalar id crosses
        # back to the host
        t_head = time_ns()
        logits_row = @decode_stage :lm_head cpu _last_logits(
            model, tensors, h, cpu, wd, vocab, P + steps; eps,
        )
        next = @decode_stage :sample cpu _greedy_id(logits_row)            # 0-based id
        head_ms = (time_ns() - t_head) / 1e6
        push!(ids, next)
        steps += 1
        if timing
            println("gesso_timing token=$(steps) id=$(next) head_ms=$(round(head_ms; digits=1))")
            flush(stdout)
        end
        if next == TOY_EOS || steps >= cap
            _profile_token_end(tok_t0, false)
            break
        end
        t_dec = time_ns()
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
            rotary_dim,
            gdn_conv,
            gdn_rec,
        )
        _profile_token_end(tok_t0, true)
        if timing
            println(
                "gesso_timing token=$(steps) decode_ms=$(round((time_ns() - t_dec) / 1e6; digits=1)) gdn_ms=$(round(_T_GDN_NS[] / 1e6; digits=1)) attn_ms=$(round(_T_ATTN_NS[] / 1e6; digits=1)) ffn_ms=$(round(_T_FFN_NS[] / 1e6; digits=1))",
            )
            flush(stdout)
        end
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
                similar(h.storage, eltype(T), (1, dim)),
                zero(eltype(h.storage)),
            ),
        )
        rmsnorm!(cpu, normed, lastrow, final_rms, wl; eps)
        lastrow = normed
    end
    out = Activation(;
        shape=(1, vocab),
        storage=fill!(
            similar(h.storage, eltype(T), (1, vocab)),
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
    rotary_dim=0,
    gdn_conv=nothing,
    gdn_rec=nothing,
)
    dim = model.embedding.dim
    kdim = n_kv_heads * d_head
    qdim = n_heads * d_head
    zeros_like = _zeros_like_factory(cpu, tensors)
    hp = Activation(; shape=(1, dim), storage=zeros_like((1, dim)))
    @decode_stage :embed cpu embedding_lookup!(cpu, hp, tensors.embedding, [tok], wl)
    normed = Activation(; shape=(1, dim), storage=zeros_like((1, dim)))
    q = Activation(; shape=(1, qdim), storage=zeros_like((1, qdim)))
    gate_attn = Activation(; shape=(1, qdim), storage=zeros_like((1, qdim)))
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
    merged = Activation(; shape=(1, qdim), storage=zeros_like((1, qdim)))
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
    _T_GDN_NS[] = 0
    _T_ATTN_NS[] = 0
    _T_FFN_NS[] = 0

    for (bi, blk) in enumerate(model.blocks)
        bt = tensors.blocks[bi]
        if _block_mixer(bt) === :linear_attention
            t0 = time_ns()
            @decode_stage :gdn_mixer cpu _hybrid_linear_block!(
                cpu,
                hp,
                normed,
                sub,
                bt,
                tensors.spec,
                wl,
                eps,
                zeros_like;
                conv_state=gdn_conv === nothing ? nothing : gdn_conv[bi],
                rec_state=gdn_rec === nothing ? nothing : gdn_rec[bi],
            )
            hp.storage .+= sub.storage
            _T_GDN_NS[] += time_ns() - t0
        else
        t0 = time_ns()
        @decode_stage :norm cpu rmsnorm!(cpu, normed, hp, bt.attn_rms, wl; eps)
        @decode_stage :attn_mixer cpu begin
            matmul!(cpu, q, normed, bt.wq, wl)
            matmul!(cpu, k, normed, bt.wk, wl)
            matmul!(cpu, v, normed, bt.wv, wl)
            _split_heads!(qh, q, n_heads, d_head)
            _split_heads!(kh, k, n_kv_heads, d_head)
            _split_heads!(vh, v, n_kv_heads, d_head)
            if haskey(bt, :wq_norm)
                rmsnorm!(cpu, qh, qh, bt.wq_norm, wl; eps)
                rmsnorm!(cpu, kh, kh, bt.wk_norm, wl; eps)
            end
            rope!(
                cpu,
                qh,
                kh,
                [pos0 - 1],
                wl;
                theta,
                inv_freq,
                interleaved=_tensors_rope_interleaved(tensors),
                rotary_dim,
            )  # 0-based position
            # KV APPEND: exactly one row per layer per step, at n_kv_heads
            kc[bi].storage[pos0, :, :] .= kh.storage[1, :, :]
            vc[bi].storage[pos0, :, :] .= vh.storage[1, :, :]
            # the query is the LAST position: it attends to the whole cache —
            # repeat the CACHE rows per query head (kh above is the single
            # new row; it was consumed by the append)
            kx = _repeat_heads(kc[bi], group)
            vx = _repeat_heads(vc[bi], group)
            _attention_heads!(
                cpu,
                attn.storage,
                scores,
                scores_out,
                qh.storage,
                kx.storage,
                vx.storage,
                n_heads,
                d_head,
                1,
                K,
                wl,
            )

            _merge_heads!(merged, attn, n_heads, d_head)
            if haskey(bt, :wattn_gate)
                matmul!(cpu, gate_attn, normed, bt.wattn_gate, wl)
                _sigmoid_mul!(cpu, merged.storage, merged.storage, gate_attn.storage)
            end
            matmul!(cpu, sub, merged, bt.wo, wl)
            hp.storage .+= sub.storage              # residual
        end
        _T_ATTN_NS[] += time_ns() - t0
        end
        t1 = time_ns()
        @decode_stage :norm cpu rmsnorm!(cpu, normed2, hp, bt.ffn_rms, wl; eps)
        _ffn_block!(cpu, down, gate, up, act, normed2, bt, wl, zeros_like)
        hp.storage .+= down.storage             # residual
        _T_FFN_NS[] += time_ns() - t1
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
include("gguf_import.jl")          # qwen35moe GGUF mmap + lazy experts
include("gguf_quants.jl")          # IQ2_XS / IQ3_XXS / IQ4_XS fused expert GEMV
include("quant_q8k.jl")            # Q8_K activation quantization (ggml block_q8_K)
include("quant_cpu_simd.jl")       # native Q8_K x IQ dot products: scalar ref + AVX2
include("tokenizer_protocol.jl")   # Pass C: the tokenizer PROTOCOL
include("import_report.jl")      # Passes I/J: report + generated matrix

end # module Inference
