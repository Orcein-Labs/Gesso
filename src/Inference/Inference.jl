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
include("kv_format.jl")            # GESSO_KV_FORMAT cache storage: f32, f16, q8, q4h
include("generate_state.jl")       # KV and GDN state carried between generate calls

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

# Fused gated RMSNorm * silu(z) of every value head, written straight into the
# merged (seq, value_dim) layout. Returns false when the backend has none and
# the per-head loop must run instead.
_gdn_gnorm_merge!(backend, merged, core, z, scale, eps) = false

function _gdn_gate_params!(backend, g, beta, a, b, A_log, dt_bias)
    seq, nv = size(beta)
    @inbounds for t in 1:seq, hh in 1:nv
        beta[t, hh] = _cpu_sigmoid(b[t, hh])
        g[t, hh] = -exp(A_log[hh]) * _cpu_softplus(a[t, hh] + dt_bias[hh])
    end
    return g, beta
end

# Fused single-token GDN decode is a backend capability. Only a backend that
# defines `_gdn_decode_core!` (the device extension) turns it on; the
# unfused multi-token path stays the reference for every other case.
_gdn_fused_decode(::AbstractGessoBackend) = false

function _gdn_decode_core!(backend, scratch, bt, spec, eps, conv_state, rec_state)
    error("_gdn_decode_core!: no fused GDN decode kernel for $(backend_name(backend))")
end

# Preallocated per-token GDN temporaries (shape (1, n) each), created once per
# generation and reused by every linear-attention layer. Layers share shapes.
function _gdn_decode_scratch(zeros_like, tensors)
    spec = haskey(tensors, :spec) ? tensors.spec : nothing
    mx = spec === nothing ? nothing : spec.mixer
    mx === nothing && return nothing
    key_dim = mx.num_key_heads * mx.key_head_dim
    value_dim = mx.num_value_heads * mx.value_head_dim
    conv_dim = 2 * key_dim + value_dim
    nv = mx.num_value_heads
    return (
        mixed=zeros_like((1, conv_dim)),
        conv=zeros_like((1, conv_dim)),
        z=zeros_like((1, value_dim)),
        a=zeros_like((1, nv)),
        b=zeros_like((1, nv)),
        core=zeros_like((1, value_dim)),
        merged=zeros_like((1, value_dim)),
    )
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
    gdn_scratch=nothing,
    gdn_fused::Bool=true,
    cpu_scratch=nothing,
)
    mx = spec.mixer
    seq = size(h.storage, 1)
    key_dim = mx.num_key_heads * mx.key_head_dim
    value_dim = mx.num_value_heads * mx.value_head_dim
    conv_dim = 2 * key_dim + value_dim
    nk, nv = mx.num_key_heads, mx.num_value_heads
    dk, dv = mx.key_head_dim, mx.value_head_dim
    ksz = mx.conv_kernel_dim

    if seq == 1 && gdn_fused && gdn_scratch !== nothing && conv_state !== nothing &&
       rec_state !== nothing && _gdn_fused_decode(cpu)
        # single-token decode: same ops, preallocated buffers, fused GDN kernels
        @decode_sub :gdn_norm cpu rmsnorm!(cpu, normed, h, bt.attn_rms, wl; eps)
        s = gdn_scratch
        mixed = Activation(; shape=(1, conv_dim), storage=s.mixed)
        z = Activation(; shape=(1, value_dim), storage=s.z)
        a = Activation(; shape=(1, nv), storage=s.a)
        b = Activation(; shape=(1, nv), storage=s.b)
        @decode_sub :gdn_proj cpu _dec_matmul_group!(cpu, (mixed, z, a, b), nothing, normed,
            (bt.wgdn_qkv, bt.wgdn_z, bt.wgdn_a, bt.wgdn_b), wl)
        # conv+SiLU+state, gates, q/k L2norm, delta rule, gated norm: device kernels
        _gdn_decode_core!(cpu, s, bt, spec, eps, conv_state, rec_state)
        @decode_sub :gdn_outproj cpu begin
            merged_act = Activation(; shape=(1, value_dim), storage=s.merged)
            matmul!(cpu, sub, merged_act, bt.wgdn_o, wl)
        end
        return sub
    end

    if seq == 1 && cpu_scratch !== nothing && _cpu_gdn_fast_ok(bt, conv_state, rec_state, h)
        # CPU single-token decode: preallocated F32 mixer (cpu_decode.jl)
        return _gdn_decode_cpu!(cpu, h, normed, sub, bt, spec, wl, eps, conv_state, rec_state, cpu_scratch)
    end

    @decode_sub :gdn_norm cpu rmsnorm!(cpu, normed, h, bt.attn_rms, wl; eps)
    mixed = Activation(; shape=(seq, conv_dim), storage=zeros_like((seq, conv_dim)))
    z = Activation(; shape=(seq, value_dim), storage=zeros_like((seq, value_dim)))
    a = Activation(; shape=(seq, nv), storage=zeros_like((seq, nv)))
    b = Activation(; shape=(seq, nv), storage=zeros_like((seq, nv)))
    @decode_sub :gdn_proj cpu begin
        matmul!(cpu, mixed, normed, bt.wgdn_qkv, wl)
        matmul!(cpu, z, normed, bt.wgdn_z, wl)
        matmul!(cpu, a, normed, bt.wgdn_a, wl)
        matmul!(cpu, b, normed, bt.wgdn_b, wl)
    end

    conv_out = zeros_like((seq, conv_dim))
    if conv_state === nothing
        conv_state = zeros_like((conv_dim, ksz - 1))
    end
    @decode_sub :gdn_conv cpu _causal_conv1d_silu!(
        cpu, conv_out, mixed.storage, bt.wgdn_conv.storage, conv_state)

    q = zeros_like((seq, nk, dk))
    k = zeros_like((seq, nk, dk))
    v = zeros_like((seq, nv, dv))
    @decode_sub :gdn_split cpu begin
        # contiguous column views of a device array keep their own reference
        # to its memory; drop them so the block arena can free conv_out
        cols = (view(conv_out, :, 1:key_dim), view(conv_out, :, (key_dim+1):(2*key_dim)),
            view(conv_out, :, (2*key_dim+1):conv_dim))
        _split_heads!(q, cols[1], nk, dk)
        _split_heads!(k, cols[2], nk, dk)
        _split_heads!(v, cols[3], nv, dv)
        foreach(_release_storage!, cols)
    end

    g = zeros_like((seq, nv))
    beta = zeros_like((seq, nv))
    A_log = bt.wgdn_A_log.storage
    dt_bias = bt.wgdn_dt_bias.storage
    @decode_sub :gdn_gates cpu _gdn_gate_params!(
        cpu, g, beta, a.storage, b.storage, A_log, dt_bias)

    core = zeros_like((seq, nv, dv))
    if rec_state === nothing
        rec_state = zeros_like((nv, dk, dv))
    end
    # CPU: the whole call (q/k repeat + L2 norm + delta rule) is :gdn_recur;
    # :gdn_qk stays 0 here. The CUDA backend times its own split inside.
    @decode_sub :gdn_recur cpu _gated_delta_rule!(
        cpu, core, q, k, v, g, beta, rec_state;
        v_head_tiled=mx.v_head_tiled,
    )

    merged = zeros_like((seq, value_dim))
    @decode_sub :gdn_gnorm cpu if !_gdn_gnorm_merge!(cpu, merged, core, z.storage, bt.wgdn_norm, eps)
        z_h = zeros_like((seq, nv, dv))
        _split_heads!(z_h, z.storage, nv, dv)
        nrm = zeros_like((seq, dv))
        xh = zeros_like((seq, dv))
        scale = bt.wgdn_norm
        for hh in 1:nv
            xh .= view(core, :, hh, :)
            xh_act = Activation(; shape=(seq, dv), storage=xh)
            nrm_act = Activation(; shape=(seq, dv), storage=nrm)
            rmsnorm!(cpu, nrm_act, xh_act, scale, wl; eps)
            xh .= view(z_h, :, hh, :)
            _silu_mul!(cpu, nrm, nrm, xh)
            @views core[:, hh, :] .= nrm
        end
        _merge_heads!(merged, core, nv, dv)
    end
    @decode_sub :gdn_outproj cpu begin
        merged_act = Activation(; shape=(seq, value_dim), storage=merged)
        matmul!(cpu, sub, merged_act, bt.wgdn_o, wl)
    end
    return sub
end

# Hidden width of a dense FFN block, 0 for a model without one. Sizes the
# decode scratch's FFN-hidden quantization.
function _dense_ffn_hidden(model)
    ffn = model.blocks[1].ffn
    return hasproperty(ffn, :hidden) ? Int(ffn.hidden) : 0
end

# `xq_in` / `xq_ff` are the decode scratch's Q8_K buffers for the residual and
# FFN-hidden inputs. Prefill passes nothing and keeps the per-weight matmuls.
function _ffn_block!(cpu, down, gate, up, act, normed2, bt, wl, zeros_like;
    xq_in=nothing, xq_ff=nothing)
    if haskey(bt, :wrouter)
        return _moe_ffn!(cpu, down, normed2, bt, wl, zeros_like)
    end
    @decode_stage :dense_ffn cpu begin
        @decode_sub :ffn_gateup cpu begin
            _dec_matmul_group!(cpu, (gate, up), xq_in, normed2, (bt.wgate, bt.wup), wl)
        end
        @decode_sub :ffn_act cpu swiglu!(cpu, act, gate, up, wl)
        @decode_sub :ffn_down cpu _dec_matmul_group!(cpu, (down,), xq_ff, act, (bt.wdown,), wl)
    end
    return down
end

function _moe_ffn!(cpu, down, normed2, bt, wl, zeros_like)
    x = normed2.storage
    y = down.storage
    n_used = haskey(bt, :n_expert_used) ? Int(bt.n_expert_used) : size(bt.wrouter.storage, 1)
    shared = haskey(bt, :wshared_gate)
    sd = shared ? Activation(; shape=size(x), storage=zeros_like(size(x))) : nothing
    shared_ffn = shared ? _shared_expert_thunk(cpu, sd, normed2, bt, wl, zeros_like) : nothing
    _moe_topk_swiglu!(
        cpu,
        y,
        x,
        bt.wrouter.storage,
        bt.wexp_gate.storage,
        bt.wexp_up.storage,
        bt.wexp_down.storage,
        n_used;
        overlap=shared_ffn,
    )
    if shared
        @decode_stage :shared_expert cpu begin
            if haskey(bt, :wshared_gate_inp)
                _shared_gate_add!(cpu, y, x, sd.storage, bt.wshared_gate_inp.storage)
            else
                y .+= sd.storage
            end
        end
    end
    return down
end

# The shared expert reads only `normed2` and writes only `sd`, so it is handed
# to `_moe_topk_swiglu!` as `overlap` and queued wherever it hides best behind
# the routed experts.
function _shared_expert_thunk(cpu, sd, normed2, bt, wl, zeros_like)
    seq = size(normed2.storage, 1)
    ff = size(bt.wshared_gate.storage, 1)
    return () -> @decode_stage :shared_expert cpu begin
        sg = Activation(; shape=(seq, ff), storage=zeros_like((seq, ff)))
        su = Activation(; shape=(seq, ff), storage=zeros_like((seq, ff)))
        sa = Activation(; shape=(seq, ff), storage=zeros_like((seq, ff)))
        _dec_matmul_group!(cpu, (sg, su), nothing, normed2, (bt.wshared_gate, bt.wshared_up), wl)
        swiglu!(cpu, sa, sg, su, wl)
        matmul!(cpu, sd, sa, bt.wshared_down, wl)
        nothing
    end
end

# HostOffload unwraps in gguf_import.jl; this generic keeps dense Array /
# CuArray / LavaArray banks pass-through for any MoE config.
_offload_host_array(s) = s

# `overlap`, when given, is work that touches neither `y` nor the experts; it
# runs exactly once before this returns.
function _moe_topk_swiglu!(backend, y, x, router, gate, up, down, n_used; overlap=nothing)
    gate isa CachedExpertBank &&
        error("_moe_topk_swiglu!: CachedExpertBank needs a device backend, got $(typeof(backend))")
    overlap === nothing || overlap()
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

# Single-query decode attention over cache rows 1:K, read in place with the GQA
# mapping (query head h reads KV head div(h-1, group)+1). The generic method
# declines by returning false, and the caller runs the reference repeat +
# per-head path, so every other backend keeps its arithmetic unchanged. A
# backend extension returns true after it has written `attn`.
_attention_decode_device!(backend, attn, q, kc, vc, n_heads, d_head, group, K) = false

# Prefill attention of L query rows over K keys with the same in-place GQA
# mapping and its own scratch. The reference path needs the KV heads repeated
# (8*K*n_heads*d_head bytes, 1 GiB at 32k keys on a 16-head model) and two
# (L, K) score buffers; a backend that tiles the keys needs neither. Declining
# keeps the reference repeat + per-head path.
_attention_prefill_device!(backend, attn, q, k, v, n_heads, d_head, group, L, K) = false

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

# Activations for one decode step. A device backend may serve every step from
# the buffers of the previous one (the CUDA extension does), so nothing taken
# from this factory may outlive the step that asked for it.
_decode_zeros_like(backend, tensors) = _zeros_like_factory(backend, tensors)

function _zeros_like_factory(backend, tensors)
    s = _first_array_storage(tensors)
    if s !== nothing
        T = eltype(s)
        return (dims::Tuple{Vararg{Int}}) -> fill!(similar(s, T, dims), zero(T))
    end
    return _fallback_zeros_factory(backend)
end

# Returns `x`'s memory now instead of at its finalizer. Host arrays leave it to
# the GC; the device extension frees.
_release_storage!(x) = nothing

# Rows the next prefill pass takes: `left` prompt rows remain, `done` are cached
# already, so the pass attends done + n keys. The host has the memory for
# `chunk`; a device backend shrinks the pass to what it holds.
_prefill_pass_rows(backend, left::Int, done::Int, chunk::Int, qdim::Int, group::Int) =
    min(chunk, left)

# Called after a prompt's last pass, before decode: a backend hands back the
# buffers only prefill uses.
_prefill_end!(backend) = nothing

# Called before a generation allocates `bytes` of KV cache: a backend that
# fills spare device memory with caches of its own gives the room back first.
_device_make_room!(backend, bytes::Int) = nothing

# A zeros factory that remembers what it handed out so `_arena_release!` can
# free all of it at once. Prefill and generation state go through one: the
# host side of a generation allocates too little to trigger a Julia GC, so
# device garbage left to finalizers piles up until an allocation fails, and
# CUDA's retry then runs a full collection (hundreds of ms) mid-decode.
struct _ZerosArena{F}
    zeros_like::F
    held::Vector{Any}
end
_ZerosArena(zeros_like) = _ZerosArena(zeros_like, Any[])

function (a::_ZerosArena)(dims::Tuple{Vararg{Int}})
    x = a.zeros_like(dims)
    push!(a.held, x)
    return x
end

function _arena_release!(a::_ZerosArena)
    foreach(_release_storage!, a.held)
    empty!(a.held)
    return nothing
end

# Prompt tokens per prefill pass. A pass's activations grow with its length
# (~0.75 MiB a token on the 35B MoE), so an unbounded pass over a long prompt
# runs an 8 GB device out of memory next to the resident weights; passes carry
# the KV cache and the GDN states. GESSO_PREFILL_CHUNK=0 is one pass.
function _prefill_chunk(P::Int)
    raw = get(ENV, "GESSO_PREFILL_CHUNK", "1024")
    c = something(tryparse(Int, raw), 1024)
    return c <= 0 ? P : min(c, P)
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

# A residual add whose sum feeds straight into the next norm. A device
# backend can do the add in the norm's first pass and drop a launch; this
# body is the add followed by the norm.
function _add_rmsnorm!(backend, dst, h, src_storage, scale, wl; eps)
    _add_storage!(h.storage, src_storage)
    return rmsnorm!(backend, dst, h, scale, wl; eps)
end

# Decode appends one K row and one V row per attention layer. Both are in
# one seam so a device backend can write them with a single launch.
function _kv_append_storage!(kc::AbstractArray, vc::AbstractArray, kh::AbstractArray, vh::AbstractArray, row::Int)
    @views kc[row, :, :] .= kh[1, :, :]
    @views vc[row, :, :] .= vh[1, :, :]
    return nothing
end

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

`info`: pass a `Ref{NamedTuple}` to receive `(kv_len, steps, reused)` — the
diagnostic channel the KV-length invariant is tested through. KV length
equals prefix length: `kv_len == length(prompt) + steps`. `reused` is the
count of prompt tokens a `state` already held, which were not prefilled.

`state`: a `GenerateState` that keeps the KV cache and GDN states after the
call. Pass the whole conversation as `prompt` each turn; only the tokens past
what the state holds are prefilled. Without one, buffers live for the call.

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
    state::Union{Nothing,GenerateState}=nothing,
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
    kv_fmt = _kv_format(_kv_default_format(backend_name(cpu), d_head))
    st = state === nothing ? GenerateState(P + cap) : state
    key = (objectid(tensors.embedding.storage), backend_name(cpu), kv_fmt)
    st.key === nothing || st.key == key || throw(gesso_error(ERR_INVALID_PLAN,
        "reference_generate: state was built for another model, backend or KV format"))
    P + cap <= st.ctx || throw(gesso_error(ERR_INVALID_PLAN,
        "reference_generate: prompt $P + max_new_tokens $cap exceed the state's ctx $(st.ctx)"))
    fresh = st.arena === nothing
    fresh && (st.arena = _ZerosArena(_zeros_like_factory(cpu, tensors)))
    zeros_like = st.arena
    # a prefill block's temporaries, dead once the block returns
    block_tmp = _ZerosArena(zeros_like.zeros_like)
    # a pass's scratch, freed before the next pass and before decode
    pass_tmp = _ZerosArena(zeros_like.zeros_like)
    ok = false
    try

        # per-layer KV caches: preallocated ctx rows, filled progressively
        # (the cache is runtime state, not a weight — no fixture-walk entries)
        # (linear-attention blocks keep no cache: a zero-row placeholder)
        kv_rows(bt) = _block_mixer(bt) === :linear_attention ? 0 : st.ctx
        kv_like = fresh ? zeros_like((0, 0, 0)) : nothing
        function kv_cache(bt)
            rows = kv_rows(bt)
            (kv_fmt === :f32 || rows == 0) && return zeros_like((rows, n_kv_heads, d_head))
            # storage rows padded to a multiple of 4: the CUDA decode reads 4
            # consecutive rows per load and needs every column 4-row aligned.
            # Readers take K rows explicitly, so the tail rows stay unread.
            x = _kv_alloc(kv_fmt, kv_like, cld(rows, 4) * 4, n_kv_heads, d_head)
            push!(zeros_like.held, x)
            return x
        end
        if fresh
            st.key = key
            attn = count(bt -> kv_rows(bt) > 0, tensors.blocks)
            _device_make_room!(cpu, 2 * attn * cld(st.ctx, 4) * 4 * n_kv_heads * _kv_head_bytes(d_head, kv_fmt))
            st.kc = [
                Activation(; shape=(kv_rows(bt), n_kv_heads, d_head), storage=kv_cache(bt))
                for bt in tensors.blocks
            ]
            st.vc = [
                Activation(; shape=(kv_rows(bt), n_kv_heads, d_head), storage=kv_cache(bt))
                for bt in tensors.blocks
            ]
            # the newest hidden row: the head reads it and each decode step
            # overwrites it in place (Activation fields are immutable, §CIX;
            # storage CONTENT is what the interpreter owns). No row before it is
            # read again, and keeping them all was 8 KiB a token, 256 MiB at 32k
            # on the 35B.
            st.h = Activation(; shape=(1, dim), storage=zeros_like((1, dim)))
            st.gdn_conv, st.gdn_rec = _gdn_states(tensors, zeros_like)
            st.gdn_scratch = _gdn_decode_scratch(zeros_like, tensors)
            mx = haskey(tensors, :spec) && tensors.spec !== nothing ? tensors.spec.mixer : nothing
            st.cpu_scratch = on_cpu && mx !== nothing ? _cpu_decode_scratch(
                mx, dim; qdim=n_heads * d_head, ff=_dense_ffn_hidden(model),
                kdim=n_kv_heads * d_head, n_heads, n_kv_heads, d_head, vocab, ctx=st.ctx,
            ) : nothing
        end
        kc, vc, h = st.kc, st.vc, st.h
        gdn_conv, gdn_rec, gdn_scratch, cpu_scratch = st.gdn_conv, st.gdn_rec, st.gdn_scratch, st.cpu_scratch
        kdim = n_kv_heads * d_head
        qdim = n_heads * d_head
        hidden_ffn = model.blocks[1].ffn.hidden
        hybrid = any(bt -> _block_mixer(bt) === :linear_attention, tensors.blocks)
        n0 = _generate_resume!(st, prompt, hybrid)

        # BREADTH-0 Pass D — see reference_prefill. `nothing` = unscaled model.
        inv_freq = tensors_rope_inv_freq(tensors, d_head; theta)
        t_prefill = time_ns()
        chunk = _prefill_chunk(P)
        s1 = n0
        while s1 < P
            s0 = s1 + 1
            s1 += _prefill_pass_rows(cpu, P - s1, s1, chunk, qdim, group)
            n = s1 - s0 + 1
            K = s1                                   # keys: every cached row so far
            positions = collect((s0-1):(s1-1))
            tmp = pass_tmp
            normed = Activation(; shape=(n, dim), storage=tmp((n, dim)))
            q = Activation(; shape=(n, qdim), storage=tmp((n, qdim)))
            gate_attn = Activation(; shape=(n, qdim), storage=tmp((n, qdim)))
            k = Activation(; shape=(n, kdim), storage=tmp((n, kdim)))
            v = Activation(; shape=(n, kdim), storage=tmp((n, kdim)))
            qh = Activation(; shape=(n, n_heads, d_head), storage=tmp((n, n_heads, d_head)))
            kh = Activation(; shape=(n, n_kv_heads, d_head), storage=tmp((n, n_kv_heads, d_head)))
            vh = Activation(; shape=(n, n_kv_heads, d_head), storage=tmp((n, n_kv_heads, d_head)))
            attn = Activation(; shape=(n, n_heads, d_head), storage=tmp((n, n_heads, d_head)))
            merged = Activation(; shape=(n, qdim), storage=tmp((n, qdim)))
            sub = Activation(; shape=(n, dim), storage=tmp((n, dim)))
            normed2 = Activation(; shape=(n, dim), storage=tmp((n, dim)))
            gate = Activation(; shape=(n, hidden_ffn), storage=tmp((n, hidden_ffn)))
            up = Activation(; shape=(n, hidden_ffn), storage=tmp((n, hidden_ffn)))
            act = Activation(; shape=(n, hidden_ffn), storage=tmp((n, hidden_ffn)))
            down = Activation(; shape=(n, dim), storage=tmp((n, dim)))
            # (n, K) score buffers, made on the first layer that needs them:
            # each query row masks the keys after its own position, the
            # softmax's K - L offset
            scores = scores_out = nothing
            # the pass's hidden rows s0..s1; the last pass hands its final row to h
            hp = Activation(; shape=(n, dim), storage=tmp((n, dim)))
            embedding_lookup!(cpu, hp, tensors.embedding, prompt[s0:s1], wp)
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
                        block_tmp;
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
                    # KV APPEND: the pass's post-rope K/V become cache rows s0..s1,
                    # stored at n_kv_heads (GQA caches the small side, §LXXVI)
                    _kv_write_rows!(kc[bi].storage, kh.storage, s0)
                    _kv_write_rows!(vc[bi].storage, vh.storage, s0)
                    # a later pass attends the cached rows before it as well. A
                    # cache in another format is read back on the first pass too,
                    # so prefill attends the same keys decode will.
                    ks, vs = s0 == 1 && kv_fmt === :f32 ? (kh, vh) : (
                        Activation(; shape=(K, n_kv_heads, d_head), storage=@view kc[bi].storage[1:K, :, :]),
                        Activation(; shape=(K, n_kv_heads, d_head), storage=@view vc[bi].storage[1:K, :, :]))
                    kv_rot = _kv_rotated(kc[bi].storage)
                    kv_rot && _kv_rotate!(qh.storage, true)
                    if !_attention_prefill_device!(cpu, attn.storage, qh.storage,
                        ks.storage, vs.storage, n_heads, d_head, group, n, K)
                        if scores === nothing
                            scores = TemporaryWorkspace(; shape=(n, K), storage=tmp((n, K)))
                            scores_out = TemporaryWorkspace(; shape=(n, K), storage=tmp((n, K)))
                        end
                        # repeat KV heads for the contraction only (identity when MHA)
                        kx = _repeat_heads(ks, group)
                        vx = _repeat_heads(vs, group)
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
                            n,
                            K,
                            wp,
                        )
                        kx === ks || (_release_storage!(kx.storage); _release_storage!(vx.storage))
                    end
                    kv_rot && _kv_rotate!(attn.storage, false)

                    _merge_heads!(merged, attn, n_heads, d_head)
                    if haskey(bt, :wattn_gate)
                        matmul!(cpu, gate_attn, normed, bt.wattn_gate, wp)
                        _sigmoid_mul!(cpu, merged.storage, merged.storage, gate_attn.storage)
                    end
                    matmul!(cpu, sub, merged, bt.wo, wp)
                    hp.storage .+= sub.storage
                end
                rmsnorm!(cpu, normed2, hp, bt.ffn_rms, wp; eps)
                _ffn_block!(cpu, down, gate, up, act, normed2, bt, wp, block_tmp)
                hp.storage .+= down.storage
                _arena_release!(block_tmp)
            end
            s1 == P && (h.storage .= @view hp.storage[n:n, :])
            _arena_release!(pass_tmp)
        end
        _prefill_end!(cpu)
        append!(st.ids, @view prompt[(n0 + 1):P])
        hybrid && state !== nothing && _generate_snapshot!(st, P)

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
                model, tensors, h, cpu, wd, vocab, 1; eps,
                buf=cpu_scratch,
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
                gdn_scratch,
                cpu_scratch,
                hrow=1,
            )
            push!(st.ids, next)
            _profile_token_end(tok_t0, true)
            if timing
                println(
                    "gesso_timing token=$(steps) decode_ms=$(round((time_ns() - t_dec) / 1e6; digits=1)) gdn_ms=$(round(_T_GDN_NS[] / 1e6; digits=1)) attn_ms=$(round(_T_ATTN_NS[] / 1e6; digits=1)) ffn_ms=$(round(_T_FFN_NS[] / 1e6; digits=1))",
                )
                flush(stdout)
            end
        end

        info isa Ref{Any} && (info[] = (kv_len=P + steps, steps=steps, reused=n0))
        ok = true
        return ids
    finally
        # a generation that fails returns its device memory too; left to the GC,
        # the next long prompt runs out next to it
        _arena_release!(pass_tmp)
        _arena_release!(block_tmp)
        state === nothing ? close(st) : ok || _generate_invalidate!(st)
    end
end

# Head buffers for `_last_logits` when the caller passes none: `nothing` means
# allocate per call. A backend that returns buffers (fields `lastrow`,
# `final_normed`, `logits`, `xq_dim`) gets back a logits row that the next call
# overwrites.
_head_scratch(backend, dim::Int, vocab::Int) = nothing

# tied head over hidden row `row`: returns the (vocab,) logits row. Applies
# the final RMSNorm first when `tensors` carries one (§LXXVI).
function _last_logits(model, tensors, h, cpu, wl, vocab, row::Int; eps::Real=1e-6, buf=nothing)
    dim = size(h.storage, 2)
    T = typeof(h.storage)
    # CPU decode passes its scratch: the row, the normed row and the logits
    # are reused, so the head allocates nothing per token. A device backend may
    # keep its own (`_head_scratch`).
    buf === nothing && (buf = _head_scratch(cpu, dim, vocab))
    lastrow = if buf === nothing
        Activation(; shape=(1, dim), storage=reshape(h.storage[row, :], (1, dim)))
    else
        _copy_row!(buf.lastrow, h.storage, row)
        Activation(; shape=(1, dim), storage=buf.lastrow)
    end
    final_rms = haskey(tensors, :final_rms) ? tensors.final_rms : nothing
    if final_rms !== nothing
        normed = Activation(;
            shape=(1, dim),
            storage=buf === nothing ?
                fill!(similar(h.storage, eltype(T), (1, dim)), zero(eltype(h.storage))) :
                fill!(buf.final_normed, zero(eltype(h.storage))),
        )
        rmsnorm!(cpu, normed, lastrow, final_rms, wl; eps)
        lastrow = normed
    end
    out = Activation(;
        shape=(1, vocab),
        storage=buf === nothing ?
            fill!(similar(h.storage, eltype(T), (1, vocab)), zero(eltype(h.storage))) :
            fill!(buf.logits, zero(eltype(h.storage))),
    )
    _dec_matmul_group!(cpu, (out,), buf === nothing ? nothing : buf.xq_dim, lastrow, (tensors.lm_head,), wl)
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
    gdn_scratch=nothing,
    cpu_scratch=nothing,
    hrow::Int=pos0,
)
    dim = model.embedding.dim
    kdim = n_kv_heads * d_head
    qdim = n_heads * d_head
    zeros_like = _decode_zeros_like(cpu, tensors)
    xq_dim = cpu_scratch === nothing ? nothing : cpu_scratch.xq_dim
    xq_qdim = cpu_scratch === nothing ? nothing : cpu_scratch.xq_qdim
    xq_ff = cpu_scratch === nothing ? nothing : cpu_scratch.xq_ff
    # CPU decode reads every activation from the preallocated scratch; the
    # other backends allocate them per call, as before.
    hp = Activation(; shape=(1, dim), storage=_decode_buf(cpu_scratch, Val(:hp), zeros_like, (1, dim)))
    @decode_stage :embed cpu embedding_lookup!(cpu, hp, tensors.embedding, [tok], wl)
    normed = Activation(; shape=(1, dim), storage=_decode_buf(cpu_scratch, Val(:normed), zeros_like, (1, dim)))
    q = Activation(; shape=(1, qdim), storage=_decode_buf(cpu_scratch, Val(:q), zeros_like, (1, qdim)))
    gate_attn = Activation(; shape=(1, qdim), storage=_decode_buf(cpu_scratch, Val(:gate_attn), zeros_like, (1, qdim)))
    k = Activation(; shape=(1, kdim), storage=_decode_buf(cpu_scratch, Val(:k), zeros_like, (1, kdim)))
    v = Activation(; shape=(1, kdim), storage=_decode_buf(cpu_scratch, Val(:v), zeros_like, (1, kdim)))
    qh = Activation(; shape=(1, n_heads, d_head), storage=_decode_buf(cpu_scratch, Val(:qh), zeros_like, (1, n_heads, d_head)))
    kh = Activation(;
        shape=(1, n_kv_heads, d_head),
        storage=_decode_buf(cpu_scratch, Val(:kh), zeros_like, (1, n_kv_heads, d_head)),
    )
    vh = Activation(;
        shape=(1, n_kv_heads, d_head),
        storage=_decode_buf(cpu_scratch, Val(:vh), zeros_like, (1, n_kv_heads, d_head)),
    )
    attn = Activation(; shape=(1, n_heads, d_head), storage=_decode_buf(cpu_scratch, Val(:attn), zeros_like, (1, n_heads, d_head)))
    merged = Activation(; shape=(1, qdim), storage=_decode_buf(cpu_scratch, Val(:merged_attn), zeros_like, (1, qdim)))
    sub = Activation(; shape=(1, dim), storage=_decode_buf(cpu_scratch, Val(:sub), zeros_like, (1, dim)))
    normed2 = Activation(; shape=(1, dim), storage=_decode_buf(cpu_scratch, Val(:normed2), zeros_like, (1, dim)))
    hidden_ffn = model.blocks[1].ffn.hidden
    gate = Activation(; shape=(1, hidden_ffn), storage=_decode_buf(cpu_scratch, Val(:gate), zeros_like, (1, hidden_ffn)))
    up = Activation(; shape=(1, hidden_ffn), storage=_decode_buf(cpu_scratch, Val(:up), zeros_like, (1, hidden_ffn)))
    act = Activation(; shape=(1, hidden_ffn), storage=_decode_buf(cpu_scratch, Val(:act), zeros_like, (1, hidden_ffn)))
    down = Activation(; shape=(1, dim), storage=_decode_buf(cpu_scratch, Val(:down), zeros_like, (1, dim)))
    K = pos0                                    # cache rows now: P + steps
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
                gdn_scratch=gdn_scratch,
                cpu_scratch=cpu_scratch,
            )
            # the residual add of `sub` happens in the FFN norm below
            _T_GDN_NS[] += time_ns() - t0
        else
        t0 = time_ns()
        @decode_stage :norm cpu rmsnorm!(cpu, normed, hp, bt.attn_rms, wl; eps)
        @decode_stage :attn_mixer cpu begin
            @decode_sub :attn_proj cpu begin
                # q, k, v and the attention gate read the same normed row:
                # one quantization, one threaded pass over all their rows.
                if haskey(bt, :wattn_gate)
                    _dec_matmul_group!(cpu, (q, k, v, gate_attn), xq_dim, normed,
                        (bt.wq, bt.wk, bt.wv, bt.wattn_gate), wl)
                else
                    _dec_matmul_group!(cpu, (q, k, v), xq_dim, normed, (bt.wq, bt.wk, bt.wv), wl)
                end
            end
            @decode_sub :attn_headnorm_rope cpu begin
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
                    _decode_pos(cpu_scratch, pos0 - 1),
                    wl;
                    theta,
                    inv_freq,
                    interleaved=_tensors_rope_interleaved(tensors),
                    rotary_dim,
                )  # 0-based position
            end
            @decode_sub :attn_kv_append cpu begin
                # KV APPEND: exactly one row per layer per step, at n_kv_heads
                _kv_append_storage!(kc[bi].storage, vc[bi].storage, kh.storage, vh.storage, pos0)
            end
            kv_rot = _kv_rotated(kc[bi].storage)
            kv_rot && _kv_rotate!(qh.storage, true)
            if cpu_scratch === nothing
                # the query is the LAST position: it attends to the whole cache.
                # A device backend that fuses the decode attention reads the
                # cache rows in place; otherwise the CACHE rows are repeated per
                # query head (kh above is the single new row; it was consumed by
                # the append) and the reference per-head loop runs.
                fused = @decode_sub :attn_core cpu _attention_decode_device!(
                    cpu, attn.storage, qh.storage, kc[bi].storage, vc[bi].storage,
                    n_heads, d_head, group, K,
                )
                if !fused
                    # the reference attention needs (1, K) workspaces; the fused
                    # path allocates none
                    scores = TemporaryWorkspace(; shape=(1, K), storage=zeros_like((1, K)))
                    scores_out = TemporaryWorkspace(; shape=(1, K), storage=zeros_like((1, K)))
                    @decode_sub :attn_kv_append cpu begin
                        kx = _repeat_heads(kc[bi], group)
                        vx = _repeat_heads(vc[bi], group)
                    end
                    @decode_sub :attn_core cpu _attention_heads!(
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
                end
            else
                # fused: reads the cache rows 1:K in place, no repeat copy
                @decode_sub :attn_core cpu _attention_decode_cpu!(
                    attn.storage, qh.storage, kc[bi].storage, vc[bi].storage,
                    cpu_scratch.probs, n_heads, d_head, group, K,
                )
            end
            kv_rot && _kv_rotate!(attn.storage, false)
            @decode_sub :attn_outproj cpu begin
                _merge_heads!(merged, attn, n_heads, d_head)
                if haskey(bt, :wattn_gate)
                    _sigmoid_mul!(cpu, merged.storage, merged.storage, gate_attn.storage)
                end
                _dec_matmul_group!(cpu, (sub,), xq_qdim, merged, (bt.wo,), wl)
            end
        end
        _T_ATTN_NS[] += time_ns() - t0
        end
        t1 = time_ns()
        # both mixers leave their output in `sub`: residual add, then norm
        @decode_stage :norm cpu _add_rmsnorm!(cpu, normed2, hp, sub.storage, bt.ffn_rms, wl; eps)
        _ffn_block!(cpu, down, gate, up, act, normed2, bt, wl, zeros_like;
            xq_in=xq_dim, xq_ff=xq_ff)
        _add_storage!(hp.storage, down.storage)  # residual
        _T_FFN_NS[] += time_ns() - t1
    end
    # write the new hidden state into its row slot — storage CONTENT, not a
    # field reassignment (Activation is immutable, §CIX)
    _write_hidden_row_storage!(h.storage, hp.storage, hrow)
    return h
end

export reference_generate, GenerateState

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
include("context_budget.jl")       # KV bytes per token, ceilings, compaction policy (CPU only)
export kv_bytes_per_token, memory_ceiling_tokens, CostModel, prefill_seconds,
    decode_seconds, ContextBudget, latency_ceiling_tokens, CompactionPolicy,
    CompactionTracker, reset!, decide_compaction
include("semantic_params.jl")        # Pass B/E: canonical parameter identity
include("arch_adapters.jl")          # Pass A/B: the family boundary
include("materialize_architecture.jl") # Pass B/E: the generic binder
include("gguf_import.jl")          # qwen35moe GGUF mmap + lazy experts
include("gguf_quants.jl")          # IQ2_XS / IQ3_XXS / IQ4_XS fused expert GEMV
include("expert_residency.jl")     # per-expert device residency, profile, routing trace
include("quant_q8k.jl")            # Q8_K activation quantization (ggml block_q8_K)
include("quant_cpu_simd.jl")       # native Q8_K x IQ dot products: scalar ref + AVX2
include("quant_cpu_kquants.jl")    # native Q8_K / Q8_0 x K-quant dot products: scalar ref + AVX2
include("gguf_quant_matrix.jl")    # quantized dense 2-D weights (QuantMatrix) + CPU matmul
include("moe_host.jl")              # batched Q8_K top-k SwiGLU for host-resident expert slabs
include("cpu_decode.jl")           # CPU seq = 1 GDN decode: preallocated F32 mixer
include("tokenizer_protocol.jl")   # Pass C: the tokenizer PROTOCOL
include("import_report.jl")      # Passes I/J: report + generated matrix

end # module Inference
