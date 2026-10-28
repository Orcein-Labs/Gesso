# materialize_architecture — the generic binder (BREADTH-0 Passes B and E).
#
# `materialize_llama` bound nine hardcoded Llama tensor spellings to nine
# interpreter fields. This file binds CANONICAL SEMANTIC IDENTITIES to §XI
# parameter families, through a `FamilyParamMap`, for ANY family.
#
# Pass E is mechanical here: a bound parameter is
#
#     SemanticParamId  (meaning)          +     storage (materialization, today)
#
# and the two never appear in the same field of the same object — the id lives
# in the binding RECORD, the bytes live in the §XI parameter. Future
# materialization (dtype / quantization / residency / layout) attaches to the
# storage side without restructuring meaning.
#
# Laws carried over unchanged from the Llama importer (§LXX, §LXXVI):
#   * missing required tensor  → error naming the EXTERNAL key
#   * unknown tensor           → error naming it (no silent drop, ever)
#   * the only ignored leftover is reconstructed RoPE `*.rotary_emb.inv_freq`
#   * upcast already happened in `load_safetensors`

# --- expected shape law (meaning → invariant) -----------------------------------

"""
    expected_shape(spec, role) -> Tuple{Int,...}

The SHAPE INVARIANT of a canonical identity under a spec — meaning, not
spelling. A fused family's row slices satisfy it by construction; an
unexpected shape is an error naming the identity and both shapes.
"""
function expected_shape(spec::ArchitectureSpec, role::Symbol)
    h = spec.hidden_size
    qd = spec.n_heads * spec.head_dim
    kvd = spec.n_kv_heads * spec.head_dim
    mx = spec.mixer
    key_dim = mx isa LinearMixerSpec ? mx.num_key_heads * mx.key_head_dim : 0
    value_dim = mx isa LinearMixerSpec ? mx.num_value_heads * mx.value_head_dim : 0
    conv_dim = 2 * key_dim + value_dim
    return if role === ROLE_TOKEN_EMBEDDING || role === ROLE_LM_HEAD
        (spec.vocab_size, h)
    elseif role === ROLE_FINAL_NORM ||
           role === ROLE_NORM_PRE_ATTN ||
           role === ROLE_NORM_POST_ATTN
        (h,)
    elseif role === ROLE_Q || role === ROLE_ATTN_GATE
        (qd, h)
    elseif role === ROLE_ATTN_HEAD_GATE
        (spec.n_heads, h)
    elseif role === ROLE_O
        (h, qd)
    elseif role === ROLE_K || role === ROLE_V
        (kvd, h)
    elseif role === ROLE_Q_NORM || role === ROLE_K_NORM
        (spec.head_dim,)
    elseif role === ROLE_GATE || role === ROLE_UP
        (spec.intermediate_size, h)
    elseif role === ROLE_DOWN
        (h, spec.intermediate_size)
    elseif role === ROLE_GDN_QKV
        (conv_dim, h)
    elseif role === ROLE_GDN_Z
        (value_dim, h)
    elseif role === ROLE_GDN_A || role === ROLE_GDN_B
        (mx.num_value_heads, h)
    elseif role === ROLE_GDN_CONV
        (conv_dim, 1, mx.conv_kernel_dim)
    elseif role === ROLE_GDN_A_LOG || role === ROLE_GDN_DT_BIAS
        (mx.num_value_heads,)
    elseif role === ROLE_GDN_NORM
        (mx.value_head_dim,)
    elseif role === ROLE_GDN_O
        (h, value_dim)
    elseif role === ROLE_SHORT_IN
        (3h, h)
    elseif role === ROLE_SHORT_CONV
        mx isa ShortConvSpec || error("expected_shape: shortconv requires ShortConvSpec")
        (h, mx.conv_kernel_dim)
    elseif role === ROLE_SHORT_OUT
        (h, h)
    elseif role === ROLE_ROUTER_BIAS
        (spec.moe.n_experts,)
    elseif role === ROLE_ROUTER
        spec.moe === nothing && error("expected_shape: ffn.router requires a MoeSpec")
        (spec.moe.n_experts, h)
    elseif role === ROLE_EXPERT_GATE || role === ROLE_EXPERT_UP
        spec.moe === nothing && error("expected_shape: expert FFN requires a MoeSpec")
        (spec.moe.n_experts, spec.moe.expert_ff, h)
    elseif role === ROLE_EXPERT_DOWN
        spec.moe === nothing && error("expected_shape: expert FFN requires a MoeSpec")
        (spec.moe.n_experts, h, spec.moe.expert_ff)
    elseif role === ROLE_SHARED_GATE || role === ROLE_SHARED_UP
        spec.moe === nothing && error("expected_shape: shared expert requires a MoeSpec")
        (spec.moe.shared_ff, h)
    elseif role === ROLE_SHARED_DOWN
        spec.moe === nothing && error("expected_shape: shared expert requires a MoeSpec")
        (h, spec.moe.shared_ff)
    elseif role === ROLE_SHARED_GATE_INP
        spec.moe === nothing && error("expected_shape: shared gate requires a MoeSpec")
        (h,)
    else
        error("expected_shape: no shape law for canonical role $(repr(role))")
    end
end

# Qwen3.5 / Qwen3Next RMSNorm is Gemma-style: y = rms(x) * (1 + weight),
# with the checkpoint storing the residual to one. llama.cpp bakes +1 at
# GGUF conversion (every `*.norm.weight` except `linear_attn.norm`). Gesso
# keeps the operator as `rms(x) * scale` and converts at the binder.
const _RMS_UNIT_OFFSET_ROLES = Set((
    ROLE_NORM_PRE_ATTN,
    ROLE_NORM_POST_ATTN,
    ROLE_Q_NORM,
    ROLE_K_NORM,
    ROLE_FINAL_NORM,
))

function _apply_rms_unit_offset(spec::ArchitectureSpec, role::Symbol, arr)
    :rms_unit_offset in spec.features || return arr
    role in _RMS_UNIT_OFFSET_ROLES || return arr
    return arr .+ 1
end

# One layer's RoPE as the interpreter threads it to `rope!`. The inverse
# frequencies are always explicit, so every block's slot has the same type;
# `theta^(-2i / rd)` is the expression `rope!` evaluates when given none.
function _layer_rope_slot(policy::RoPEPolicy, d_head::Integer)
    rd = policy.rotary_dim == 0 ? Int(d_head) : Int(policy.rotary_dim)
    inv = rope_inv_freq(policy, d_head)
    inv === nothing && (inv = [policy.theta^(-2i / rd) for i in 0:(rd÷2-1)])
    return (
        theta=Float64(policy.theta),
        inv_freq=Vector{Float64}(inv),
        rotary_dim=Int(policy.rotary_dim),
    )
end

# --- layer origin detection (fixture protocol, once, at the boundary) -----------

"""
    detect_layer_origin(map, tensors_by_name, spec) -> UnitRange{Int}

HuggingFace checkpoints are 0-based (`model.layers.0`…); Gesso's own micro
fixtures were written 1-based. Before BREADTH-0 this was string surgery on one
probe key inside the Llama path. It is now a BOUNDARY concern for every family:
probe the map's own prefix/suffix, detect once, refuse a mix (§LXX).
"""
function detect_layer_origin(
    pmap::FamilyParamMap,
    tensors_by_name::AbstractDict,
    spec::ArchitectureSpec,
)
    probe = pmap.layer_prefix * "0." * pmap.layer_refs[1].external
    probe1 = pmap.layer_prefix * "1." * pmap.layer_refs[1].external
    has0 = haskey(tensors_by_name, probe)
    has1 = haskey(tensors_by_name, probe1)
    extra_prefix = pmap.layer_prefix * string(spec.num_layers) * "."
    has_extra = any(startswith(k, extra_prefix) for k in keys(tensors_by_name))
    if has0 && has1 && spec.num_layers > 1 && has_extra
        error(
            "materialize_architecture: checkpoint mixes 0-based $(probe) with 1-based $(probe1) layers",
        )
    end
    has0 && return 0:(spec.num_layers-1)
    has1 && return 1:spec.num_layers
    return error(
        "materialize_architecture: missing required tensor: $probe (0-based) or $probe1 (1-based)",
    )
end

# --- the binder ------------------------------------------------------------------

"""
    materialize_architecture(spec, map, tensors_by_name) -> NamedTuple

Bind a checkpoint's tensors to Gesso semantic parameters THROUGH the canonical
identity vocabulary, for any family.

Returns the tensor set the interpreter already consumes —
`(embedding, blocks, lm_head, final_rms, rope)` — plus a `bindings` record
carrying the meaning side (Pass E). The field names are the existing
interpreter slots, so `reference_prefill` / `Session` are UNCHANGED: that is
the meta-metric proof (§XVIII).

The `rope` field is new and OPTIONAL-by-construction for consumers: the engine
reads it when present and behaves exactly as before when it is not.
"""
function materialize_architecture(
    spec::ArchitectureSpec,
    pmap::FamilyParamMap,
    tensors_by_name::AbstractDict,
)
    consumed = Set{String}()
    bindings = Pair{SemanticParamId, String}[]

    function grab(external::AbstractString)
        haskey(tensors_by_name, external) ||
            error("materialize_architecture: missing required tensor: $external")
        push!(consumed, String(external))
        return tensors_by_name[external]
    end

    # --- model level --------------------------------------------------------
    emb_arr = grab(pmap.embedding_external)
    want = expected_shape(spec, ROLE_TOKEN_EMBEDDING)
    size(emb_arr) == want || error(
        "materialize_architecture: $(canonical_name(SemanticParamId(ROLE_TOKEN_EMBEDDING))) " *
        "expects shape $want but $(pmap.embedding_external) has $(size(emb_arr))",
    )
    emb = EmbeddingTable(; shape=(spec.vocab_size, spec.hidden_size), storage=emb_arr)
    push!(
        bindings,
        SemanticParamId(ROLE_TOKEN_EMBEDDING, :embedding) =>
            String(pmap.embedding_external),
    )

    final_rms = nothing
    if pmap.final_norm_external !== nothing &&
       haskey(tensors_by_name, pmap.final_norm_external)
        arr = grab(pmap.final_norm_external)
        w = expected_shape(spec, ROLE_FINAL_NORM)
        size(arr) == w || error(
            "materialize_architecture: final_norm expects shape $w but $(pmap.final_norm_external) has $(size(arr))",
        )
        arr = _apply_rms_unit_offset(spec, ROLE_FINAL_NORM, arr)
        final_rms = FrozenParameter(; shape=w, storage=arr)
        push!(
            bindings,
            SemanticParamId(ROLE_FINAL_NORM, :frozen) => String(pmap.final_norm_external),
        )
    end

    # lm_head: tied (the embedding IS the head) or a distinct tensor.
    lm_head = emb
    if pmap.lm_head_external !== nothing && haskey(tensors_by_name, pmap.lm_head_external)
        arr = grab(pmap.lm_head_external)
        if spec.tie_word_embeddings
            arr == emb_arr || error(
                "materialize_architecture: lm_head.weight differs from embed_tokens.weight " *
                "but tie_word_embeddings is true — untied heads are out of scope here (§LXX)",
            )
        else
            lm_head = ProjectionWeight(; shape=size(arr), storage=arr)
        end
        push!(
            bindings,
            SemanticParamId(ROLE_LM_HEAD, :projection) => String(pmap.lm_head_external),
        )
    elseif !spec.tie_word_embeddings
        error(
            "materialize_architecture: $(pmap.family) declares tie_word_embeddings=false but " *
            "no lm_head tensor was found — untied heads need an explicit projection",
        )
    end

    # --- layers -------------------------------------------------------------
    origin = detect_layer_origin(pmap, tensors_by_name, spec)
    blocks = map(origin) do i
        slots = Dict{Symbol, Any}()
        layer = i - first(origin)          # 0-based layer index, whatever the origin
        mixer = layer_mixer(spec, layer)
        slots[:mixer] = mixer
        if spec.moe !== nothing
            slots[:n_expert_used] = spec.moe.n_expert_used
            slots[:router_gating] = spec.moe.gating
        end
        # Per-layer attention geometry, added only for families that declare it,
        # so every other family keeps the block fields it had.
        if spec.sliding_window !== nothing
            slots[:attn_window] = layer_window(spec, layer)
        end
        if spec.rope_swa !== nothing
            slots[:rope_layer] = _layer_rope_slot(layer_rope(spec, layer), spec.head_dim)
        end
        if spec.activation_kind === :geglu
            slots[:ffn_geglu] = true
        end
        refs = _layer_param_refs(pmap, spec, layer)
        for ref in refs
            external = external_name(pmap, ref, i)
            arr = grab(external)
            slice = ref.rows === nothing ? arr : arr[ref.rows, :]
            slice = _apply_rms_unit_offset(spec, ref.id.role, slice)
            slot = slot_for(ref.id.role)
            slot === nothing && error(
                "materialize_architecture: canonical role $(repr(ref.id.role)) has no engine slot — " *
                "it is not an identity the interpreter consumes",
            )
            want = expected_shape(spec, ref.id.role)
            # Some converters write 1-D roles as a 2-D tensor with a unit axis
            # (GSQ Ornith stores ffn_gate_inp_shexp as 1x2048); the values are the same.
            if length(want) == 1 &&
               slice isa AbstractMatrix &&
               length(slice) == want[1] &&
               1 in size(slice)
                slice = vec(slice)
            end
            size(slice) == want || error(
                "materialize_architecture: $(canonical_name(SemanticParamId(ref.id.role, layer, ref.id.param_family))) " *
                "expects shape $want but $external" *
                (ref.rows === nothing ? "" : " rows $(ref.rows)") *
                " has $(size(slice))",
            )
            slots[slot] = if ref.id.param_family === :frozen
                FrozenParameter(; shape=want, storage=slice)
            elseif ref.id.param_family === :expert
                ExpertWeight(; shape=want, storage=slice)
            else
                ProjectionWeight(; shape=want, storage=slice)
            end
            push!(
                bindings,
                SemanticParamId(ref.id.role, layer, ref.id.param_family) =>
                    String(external),
            )
        end
        # deterministic block field order (sorted symbol names), so the
        # tensor set is reproducible across families and runs
        ordered = sort!(collect(keys(slots)))
        NamedTuple{Tuple(ordered)}(Tuple(slots[n] for n in ordered))
    end

    # --- nothing silently discarded (§LXX) ----------------------------------
    leftover = sort(collect(setdiff(Set(keys(tensors_by_name)), consumed)))
    unknown = filter(leftover) do k
        endswith(k, _IGNORED_ROPE_INV_FREQ_SUFFIX) && return false
        for p in pmap.ignored_prefixes
            startswith(k, p) && return false
        end
        return true
    end
    isempty(unknown) || error(
        "materialize_architecture: unknown tensor(s) in checkpoint: $(join(unknown, ", "))" *
        (
            isempty(leftover) || length(leftover) == length(unknown) ? "" :
            " (ignored reconstructed RoPE $(_IGNORED_ROPE_INV_FREQ_SUFFIX): $(join(filter(k -> endswith(k, _IGNORED_ROPE_INV_FREQ_SUFFIX), leftover), ", ")))"
        ),
    )

    return (
        embedding=emb,
        blocks=collect(blocks),
        lm_head=lm_head,
        final_rms=final_rms,
        rope=spec.rope,
        bindings=bindings,
        spec=spec,
    )
end

export materialize_architecture, expected_shape, detect_layer_origin
