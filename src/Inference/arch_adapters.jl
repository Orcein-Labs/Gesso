# Architecture adapters — the family boundary (BREADTH-0 Pass A/B).
#
# An adapter is the ONLY place that knows an external family's spelling. It
# answers two questions:
#
#   1. `parse_config`  — what does this family's config.json MEAN?
#                         (config.json → ArchitectureSpec)
#   2. `param_map`     — how does this family spell its parameters?
#                         (ArchitectureSpec → FamilyParamMap)
#
# Nothing downstream branches on `model_type`. The engine consumes the spec's
# MEANING and the map's CANONICAL IDENTITIES. Adding family #6 is a new
# adapter, not an engine edit — that is the meta-metric (§XVIII).
#
# Evidence discipline (Pass I philosophy): every variant below is a REAL
# convention observed in published configs of these families. Where Gesso does
# not implement the semantics, the spec still BUILDS and the capability check
# fails LATER at the named operation (Pass F) — that is the capability lattice
# (§II), not a model whitelist.

# --- shared config vocabulary ---------------------------------------------------

# The fields every family in this batch shares. Kept generic: these are common
# transformer semantics, not Llama semantics.
const _COMMON_CONFIG_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
)

function _require_keys(raw::AbstractDict, keys::Tuple, family::Symbol, who::AbstractString)
    missing_keys = [k for k in keys if !haskey(raw, k)]
    isempty(missing_keys) || error(
        "$who: $(family) config.json is missing required key(s): $(join(missing_keys, ", "))",
    )
    return nothing
end

# --- adapter protocol -----------------------------------------------------------

"""
    ArchitectureAdapter

Supertype of the per-family import adapters (BREADTH-0 Pass A).

    family_symbol(a)          the external family name (provenance only)
    parse_config(a, raw)      config.json (as a Dict) → ArchitectureSpec
    param_map(a, spec)        ArchitectureSpec → FamilyParamMap

`parse_config` VALIDATES and REFUSES loudly (§LXX): a config that cannot be
expressed as a spec is an error at import, never a partially-understood spec.
"""
abstract type ArchitectureAdapter end

"""
    family_symbol(adapter) -> Symbol

The external family name. PROVENANCE ONLY — no engine code may branch on it
(Pass F). Legal uses: the import report, receipts, the compatibility matrix.
"""
function family_symbol end

"""
    parse_config(adapter, raw::AbstractDict) -> ArchitectureSpec
"""
function parse_config end

"""
    param_map(adapter, spec::ArchitectureSpec) -> FamilyParamMap
"""
function param_map end

# --- RoPE policy parsing (Pass D) ----------------------------------------------

# Activation form → §VIII activation semantics. This is the Llama-3 rope
# spelling (`rope_type`, newer transformers) and the legacy Llama-2 spelling
# (`type`); both appear in the wild and both must parse.
function _rope_policy(raw::AbstractDict, family::Symbol; head_dim::Integer=0)
    theta = haskey(raw, "rope_theta") ? Float64(raw["rope_theta"]) : 10000.0
    scaling = get(raw, "rope_scaling", nothing)
    interleaved = Bool(get(raw, "rope_interleaved", false))
    rotary_dim = Int(get(raw, "rope_dim", 0))
    scaling === nothing && return RoPEPolicy(; theta, rotary_dim, interleaved)
    scaling isa AbstractDict || error(
        "RoPEPolicy: $family config rope_scaling must be an object or null (got $(repr(scaling)))",
    )
    kind_str = String(get(scaling, "rope_type", get(scaling, "type", "linear")))
    kind = if kind_str == "linear"
        :linear
    elseif kind_str == "llama3"
        :llama3
    elseif kind_str == "default" || kind_str == "none"
        :none
    else
        # NOT a refusal at import: the policy cannot be expressed by this
        # build, so say WHICH operation is missing (§II capability lattice).
        throw(LoweringNotImplemented(:rope_scaling, Symbol(family)))
    end
    factor = Float64(get(scaling, "factor", 1.0))
    return RoPEPolicy(;
        theta,
        kind,
        factor,
        original_max_position_embeddings=Int(
            get(scaling, "original_max_position_embeddings", 0),
        ),
        low_freq_factor=Float64(get(scaling, "low_freq_factor", 1.0)),
        high_freq_factor=Float64(get(scaling, "high_freq_factor", 1.0)),
        rotary_dim,
        interleaved,
    )
end

# Hidden-activation spelling → FFN semantics. These are the published names.
const _ACTIVATION_KINDS = Dict{String, Symbol}(
    "silu" => :swiglu,
    "swish" => :swiglu,
    "gelu" => :gelu,
    "gelu_pytorch_tanh" => :gelu,
    "gelu_new" => :gelu,
    "quick_gelu" => :gelu,
    "relu" => :relu,
)

function _activation_kind(raw::AbstractDict, family::Symbol)
    act = String(get(raw, "hidden_act", "silu"))
    haskey(_ACTIVATION_KINDS, act) ||
        throw(LoweringNotImplemented(Symbol("ffn_activation_", act), Symbol(family)))
    return _ACTIVATION_KINDS[act]
end

function _norm_kind(raw::AbstractDict, family::Symbol, act::Symbol)
    # Gemma/Mistral/Llama are RMSNorm families; Phi-2 uses LayerNorm. The
    # published discriminator is the presence of `norm_type`/`layer_norm_eps`
    # vs `rms_norm_eps` — read whichever the config declares.
    if haskey(raw, "norm_type")
        return String(raw["norm_type"]) == "layer_norm" ? :layer : :rms
    end
    if haskey(raw, "layer_norm_eps") && !haskey(raw, "rms_norm_eps")
        return :layer
    end
    return :rms
end

# --- FUSED-LAYOUT DECLARATION (Pass B: canonical identity ≠ source spelling) -----

"""
    FusedQKVLayout

DECLARED row layout of a fused `qkv_proj`-style tensor. Gesso does not guess a
fusion's packing from a name; the family adapter states it.

    :contiguous   [all Q rows][all K rows][all V rows]
    :grouped      per KV head-group: [that group's Q heads][K][V]

`:grouped` is the layout published for Phi-3; `:contiguous` is the simpler
block packing. Both produce the same three CANONICAL identities —
`layer[i].attention.q/.k/.v` — from one external tensor, which is the whole
point of Pass B: identity is meaning, not spelling.
"""
struct FusedQKVLayout
    mode::Symbol
end

"""
    fused_qkv_rows(layout, n_heads, n_kv_heads, head_dim) -> (q, k, v) ranges

Row ranges (1-based, inclusive) of the fused tensor that carry Q, K and V.
"""
function fused_qkv_rows(l::FusedQKVLayout, n_heads::Int, n_kv_heads::Int, head_dim::Int)
    if l.mode === :contiguous
        q = 1:(n_heads*head_dim)
        k = (n_heads*head_dim+1):((n_heads+n_kv_heads)*head_dim)
        v = ((n_heads+n_kv_heads)*head_dim+1):((n_heads+2*n_kv_heads)*head_dim)
        return (q=q, k=k, v=v)
    elseif l.mode === :grouped
        n_kv_heads > 0 && n_heads % n_kv_heads == 0 || error(
            "fused_qkv_rows: n_heads $n_heads not divisible by n_kv_heads $n_kv_heads",
        )
        per_group = div(n_heads, n_kv_heads)
        q = Int[]
        k = Int[]
        v = Int[]
        off = 0
        for _ in 1:n_kv_heads
            qblk = (off+1):(off+per_group*head_dim)
            kblk = (off+per_group*head_dim+1):(off+(per_group+1)*head_dim)
            vblk = (off+(per_group+1)*head_dim+1):(off+(per_group+2)*head_dim)
            append!(q, qblk)
            append!(k, kblk)
            append!(v, vblk)
            off += (per_group + 2) * head_dim
        end
        return (q=q, k=k, v=v)
    else
        error("fused_qkv_rows: unsupported declared layout $(repr(l.mode))")
    end
end

"""
    fused_qg_rows(n_heads, head_dim) -> (q, gate) row indices

Per-head interleaved packing of a fused Q|gate projection
(`q_proj` of shape `(2 * n_heads * head_dim, hidden)`):

    [H0_Q, H0_gate, H1_Q, H1_gate, …]

HuggingFace `Qwen3NextAttention` / `Qwen3_5Attention` split with
`view(..., n_heads, 2 * head_dim).chunk(2, dim=-1)`. A middle split
(all-Q then all-gate) changes the model.
"""
function fused_qg_rows(n_heads::Int, head_dim::Int)
    n_heads > 0 && head_dim > 0 ||
        error("fused_qg_rows: n_heads and head_dim must be positive")
    q = Vector{Int}(undef, n_heads * head_dim)
    gate = Vector{Int}(undef, n_heads * head_dim)
    qi = 1
    gi = 1
    for h in 0:(n_heads-1)
        base = h * 2 * head_dim
        for d in 1:head_dim
            q[qi] = base + d
            qi += 1
        end
        for d in 1:head_dim
            gate[gi] = base + head_dim + d
            gi += 1
        end
    end
    return (q=q, gate=gate)
end

# --- the family adapters ---------------------------------------------------------

"""
    LlamaAdapter

HuggingFace `llama` (Llama 2/3, SmolLM2, …): RMSNorm + SwiGLU, separate
q/k/v/o and gate/up/down projections, optional `rope_scaling`.

Names: `model.embed_tokens.weight`, `model.layers.N.self_attn.{q,k,v,o}_proj.weight`,
`model.layers.N.mlp.{gate,up,down}_proj.weight`,
`model.layers.N.input_layernorm.weight`,
`model.layers.N.post_attention_layernorm.weight`, `model.norm.weight`,
`lm_head.weight`.
"""
struct LlamaAdapter <: ArchitectureAdapter end
family_symbol(::LlamaAdapter) = :llama

function parse_config(::LlamaAdapter, raw::AbstractDict)
    _require_keys(raw, _COMMON_CONFIG_KEYS, :llama, "parse_config")
    hd = Int(raw["hidden_size"])
    nh = Int(raw["num_attention_heads"])
    return ArchitectureSpec(;
        family=:llama,
        hidden_size=hd,
        num_layers=Int(raw["num_hidden_layers"]),
        n_heads=nh,
        n_kv_heads=Int(raw["num_key_value_heads"]),
        vocab_size=Int(raw["vocab_size"]),
        intermediate_size=Int(raw["intermediate_size"]),
        norm_kind=_norm_kind(raw, :llama, :swiglu),
        activation_kind=_activation_kind(raw, :llama),
        rope=_rope_policy(raw, :llama; head_dim=div(hd, nh)),
        tie_word_embeddings=Bool(get(raw, "tie_word_embeddings", false)),
        attention_bias=Bool(get(raw, "attention_bias", false)),
        mlp_bias=Bool(get(raw, "mlp_bias", false)),
    )
end

function param_map(::LlamaAdapter, spec::ArchitectureSpec)
    refs = ParamRef[
        ParamRef(SemanticParamId(ROLE_NORM_PRE_ATTN, 0, :frozen), "input_layernorm.weight"),
        ParamRef(SemanticParamId(ROLE_Q, 0, :projection), "self_attn.q_proj.weight"),
        ParamRef(SemanticParamId(ROLE_K, 0, :projection), "self_attn.k_proj.weight"),
        ParamRef(SemanticParamId(ROLE_V, 0, :projection), "self_attn.v_proj.weight"),
        ParamRef(SemanticParamId(ROLE_O, 0, :projection), "self_attn.o_proj.weight"),
        ParamRef(
            SemanticParamId(ROLE_NORM_POST_ATTN, 0, :frozen),
            "post_attention_layernorm.weight",
        ),
        ParamRef(SemanticParamId(ROLE_GATE, 0, :projection), "mlp.gate_proj.weight"),
        ParamRef(SemanticParamId(ROLE_UP, 0, :projection), "mlp.up_proj.weight"),
        ParamRef(SemanticParamId(ROLE_DOWN, 0, :projection), "mlp.down_proj.weight"),
    ]
    return FamilyParamMap(
        :llama,
        "model.layers.",
        refs,
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    )
end

"""
    Qwen2Adapter

HuggingFace `qwen2`: RMSNorm + SwiGLU like Llama, but the q/k projections are
per-head normalized — `self_attn.q_norm.weight` / `k_norm.weight` of shape
`(head_dim,)` per layer. That is a genuine NEW SEMANTIC (the `qk_norm`
capability), not a naming difference.

Names: as Llama, plus the two q_norm/k_norm entries when present.
"""
struct Qwen2Adapter <: ArchitectureAdapter end
family_symbol(::Qwen2Adapter) = :qwen2

function parse_config(::Qwen2Adapter, raw::AbstractDict)
    _require_keys(raw, _COMMON_CONFIG_KEYS, :qwen2, "parse_config")
    hd = Int(raw["hidden_size"])
    nh = Int(raw["num_attention_heads"])
    feats = Set{Symbol}([:qk_norm])
    return ArchitectureSpec(;
        family=:qwen2,
        hidden_size=hd,
        num_layers=Int(raw["num_hidden_layers"]),
        n_heads=nh,
        n_kv_heads=Int(raw["num_key_value_heads"]),
        vocab_size=Int(raw["vocab_size"]),
        intermediate_size=Int(raw["intermediate_size"]),
        norm_kind=_norm_kind(raw, :qwen2, :swiglu),
        activation_kind=_activation_kind(raw, :qwen2),
        rope=_rope_policy(raw, :qwen2; head_dim=div(hd, nh)),
        tie_word_embeddings=Bool(get(raw, "tie_word_embeddings", false)),
        attention_bias=Bool(get(raw, "attention_bias", false)),
        features=feats,
    )
end

function param_map(::Qwen2Adapter, spec::ArchitectureSpec)
    refs = ParamRef[
        ParamRef(SemanticParamId(ROLE_NORM_PRE_ATTN, 0, :frozen), "input_layernorm.weight"),
        ParamRef(SemanticParamId(ROLE_Q, 0, :projection), "self_attn.q_proj.weight"),
        ParamRef(SemanticParamId(ROLE_Q_NORM, 0, :frozen), "self_attn.q_norm.weight"),
        ParamRef(SemanticParamId(ROLE_K, 0, :projection), "self_attn.k_proj.weight"),
        ParamRef(SemanticParamId(ROLE_K_NORM, 0, :frozen), "self_attn.k_norm.weight"),
        ParamRef(SemanticParamId(ROLE_V, 0, :projection), "self_attn.v_proj.weight"),
        ParamRef(SemanticParamId(ROLE_O, 0, :projection), "self_attn.o_proj.weight"),
        ParamRef(
            SemanticParamId(ROLE_NORM_POST_ATTN, 0, :frozen),
            "post_attention_layernorm.weight",
        ),
        ParamRef(SemanticParamId(ROLE_GATE, 0, :projection), "mlp.gate_proj.weight"),
        ParamRef(SemanticParamId(ROLE_UP, 0, :projection), "mlp.up_proj.weight"),
        ParamRef(SemanticParamId(ROLE_DOWN, 0, :projection), "mlp.down_proj.weight"),
    ]
    return FamilyParamMap(
        :qwen2,
        "model.layers.",
        refs,
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    )
end

"""
    Qwen35Adapter

HuggingFace `qwen3_5` (Qwen3.5, Ornith-1.5, …). Nested `text_config` is the
transformer; the root `model_type` is provenance. Hybrid mixer: most layers
are Gated DeltaNet (`linear_attention`), every `full_attention_interval`-th
layer is gated full attention. `head_dim` is a published field, not
`hidden_size ÷ n_heads`. Partial / interleaved mRoPE is declared on the
RoPEPolicy so execution fails at those named ops until they lower.
"""
struct Qwen35Adapter <: ArchitectureAdapter end
family_symbol(::Qwen35Adapter) = :qwen3_5

function _qwen35_text(raw::AbstractDict)
    if haskey(raw, "text_config")
        raw["text_config"] isa AbstractDict || error(
            "parse_config: qwen3_5 text_config must be an object (got $(typeof(raw["text_config"])))",
        )
        return raw["text_config"]
    end
    return raw
end

function _qwen35_layer_types(text::AbstractDict, n_layers::Int)
    lt = get(text, "layer_types", nothing)
    if lt === nothing
        interval = Int(get(text, "full_attention_interval", 4))
        interval > 0 ||
            error("parse_config: qwen3_5 full_attention_interval must be positive")
        return [
            i % interval == 0 ? "full_attention" : "linear_attention" for i in 1:n_layers
        ]
    end
    lt isa AbstractVector ||
        error("parse_config: qwen3_5 layer_types must be an array (got $(typeof(lt)))")
    length(lt) == n_layers || error(
        "parse_config: qwen3_5 layer_types length $(length(lt)) != num_hidden_layers $n_layers",
    )
    return String[String(x) for x in lt]
end

function _qwen35_rope(text::AbstractDict, head_dim::Int)
    rp = get(text, "rope_parameters", nothing)
    extra = Set{Symbol}()
    if rp === nothing
        return _rope_policy(text, :qwen3_5; head_dim), extra
    end
    rp isa AbstractDict ||
        error("parse_config: qwen3_5 rope_parameters must be an object (got $(typeof(rp)))")
    theta = Float64(get(rp, "rope_theta", get(text, "rope_theta", 10000.0)))
    kind_str = String(get(rp, "rope_type", "default"))
    kind = if kind_str == "default" || kind_str == "none"
        :none
    elseif kind_str == "linear"
        :linear
    elseif kind_str == "llama3"
        :llama3
    else
        throw(LoweringNotImplemented(:rope_scaling, :qwen3_5))
    end
    # mrope_interleaved is the T/H/W section packing, NOT GPT-J pair layout.
    # Text apply_rotary_pos_emb uses rotate_half (half-split), so
    # RoPEPolicy.interleaved stays false. Partial rotary is a real op.
    partial =
        Float64(get(rp, "partial_rotary_factor", get(text, "partial_rotary_factor", 1.0)))
    rotary_dim = partial == 1.0 ? 0 : Int(round(partial * head_dim))
    section = Int[]
    if haskey(rp, "mrope_section")
        push!(extra, :mrope)
        raw_sec = rp["mrope_section"]
        raw_sec isa AbstractVector || error(
            "parse_config: qwen3_5 mrope_section must be an array (got $(typeof(raw_sec)))",
        )
        section = Int[Int(x) for x in raw_sec]
        rotary_dim == 0 || sum(section) == rotary_dim ÷ 2 || error(
            "parse_config: qwen3_5 mrope_section $(section) sums to $(sum(section)), expected rotary_dim/2 = $(rotary_dim ÷ 2)",
        )
    end
    return RoPEPolicy(; theta, kind, rotary_dim, interleaved=false, mrope_section=section),
    extra
end

function parse_config(::Qwen35Adapter, raw::AbstractDict)
    text = _qwen35_text(raw)
    _require_keys(text, _COMMON_CONFIG_KEYS, :qwen3_5, "parse_config")
    hd = Int(text["hidden_size"])
    nh = Int(text["num_attention_heads"])
    n_layers = Int(text["num_hidden_layers"])
    published_head = haskey(text, "head_dim") ? Int(text["head_dim"]) : nothing
    head_dim = published_head === nothing ? div(hd, nh) : published_head
    layer_types = _qwen35_layer_types(text, n_layers)
    rope, rope_feats = _qwen35_rope(text, head_dim)
    feats = Set{Symbol}([:qk_norm, :rms_unit_offset])
    any(==("linear_attention"), layer_types) && push!(feats, :linear_attention)
    Bool(get(text, "attn_output_gate", false)) && push!(feats, :attn_output_gate)
    union!(feats, rope_feats)
    mixer = nothing
    if :linear_attention in feats
        mixer = LinearMixerSpec(;
            layer_types=Symbol.(layer_types),
            num_key_heads=Int(text["linear_num_key_heads"]),
            key_head_dim=Int(text["linear_key_head_dim"]),
            num_value_heads=Int(text["linear_num_value_heads"]),
            value_head_dim=Int(text["linear_value_head_dim"]),
            conv_kernel_dim=Int(get(text, "linear_conv_kernel_dim", 4)),
        )
    end
    return ArchitectureSpec(;
        family=:qwen3_5,
        hidden_size=hd,
        num_layers=n_layers,
        n_heads=nh,
        n_kv_heads=Int(text["num_key_value_heads"]),
        vocab_size=Int(text["vocab_size"]),
        intermediate_size=Int(text["intermediate_size"]),
        head_dim,
        norm_kind=_norm_kind(text, :qwen3_5, :swiglu),
        activation_kind=_activation_kind(text, :qwen3_5),
        rope,
        tie_word_embeddings=Bool(
            get(text, "tie_word_embeddings", get(raw, "tie_word_embeddings", false)),
        ),
        attention_bias=Bool(get(text, "attention_bias", false)),
        features=feats,
        mixer,
        rms_norm_eps=Float64(get(text, "rms_norm_eps", 1e-6)),
    )
end

function param_map(::Qwen35Adapter, spec::ArchitectureSpec)
    # Hybrid: full-attention layers share Qwen2 names (with a fused Q|gate on
    # q_proj). Linear-attention layers bind the Gated DeltaNet vocabulary.
    # Vision / MTP tensors are declared unused this campaign.
    # Q|gate packing is per-head interleaved (HF view+chunk, llama.cpp
    # ggml_view_3d stride 2*head_dim) — not concatenated all-Q then all-gate.
    (q_rows, gate_rows) = fused_qg_rows(spec.n_heads, spec.head_dim)
    shared = ParamRef[
        ParamRef(SemanticParamId(ROLE_NORM_PRE_ATTN, 0, :frozen), "input_layernorm.weight"),
        ParamRef(
            SemanticParamId(ROLE_NORM_POST_ATTN, 0, :frozen),
            "post_attention_layernorm.weight",
        ),
        ParamRef(SemanticParamId(ROLE_GATE, 0, :projection), "mlp.gate_proj.weight"),
        ParamRef(SemanticParamId(ROLE_UP, 0, :projection), "mlp.up_proj.weight"),
        ParamRef(SemanticParamId(ROLE_DOWN, 0, :projection), "mlp.down_proj.weight"),
    ]
    full = ParamRef[
        shared...,
        ParamRef(SemanticParamId(ROLE_Q, 0, :projection), "self_attn.q_proj.weight", q_rows),
        ParamRef(
            SemanticParamId(ROLE_ATTN_GATE, 0, :projection),
            "self_attn.q_proj.weight",
            gate_rows,
        ),
        ParamRef(SemanticParamId(ROLE_Q_NORM, 0, :frozen), "self_attn.q_norm.weight"),
        ParamRef(SemanticParamId(ROLE_K, 0, :projection), "self_attn.k_proj.weight"),
        ParamRef(SemanticParamId(ROLE_K_NORM, 0, :frozen), "self_attn.k_norm.weight"),
        ParamRef(SemanticParamId(ROLE_V, 0, :projection), "self_attn.v_proj.weight"),
        ParamRef(SemanticParamId(ROLE_O, 0, :projection), "self_attn.o_proj.weight"),
    ]
    linear = ParamRef[
        shared...,
        ParamRef(SemanticParamId(ROLE_GDN_QKV, 0, :projection), "linear_attn.in_proj_qkv.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_Z, 0, :projection), "linear_attn.in_proj_z.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_A, 0, :projection), "linear_attn.in_proj_a.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_B, 0, :projection), "linear_attn.in_proj_b.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_CONV, 0, :projection), "linear_attn.conv1d.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_A_LOG, 0, :frozen), "linear_attn.A_log"),
        ParamRef(SemanticParamId(ROLE_GDN_DT_BIAS, 0, :frozen), "linear_attn.dt_bias"),
        ParamRef(SemanticParamId(ROLE_GDN_NORM, 0, :frozen), "linear_attn.norm.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_O, 0, :projection), "linear_attn.out_proj.weight"),
    ]
    return FamilyParamMap(
        :qwen3_5,
        "model.language_model.layers.",
        full,
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight";
        ignored_prefixes=["model.visual.", "mtp."],
        linear_refs=linear,
    )
end

"""
    Qwen35MoeAdapter

HuggingFace `qwen3_5_moe` / GGUF `qwen35moe` (Ornith-1.5-35B-A3B). Same
hybrid mixer as `Qwen35Adapter`, plus softmax top-k routed experts and a
sigmoid-gated shared SwiGLU expert on every layer.
"""
struct Qwen35MoeAdapter <: ArchitectureAdapter end
family_symbol(::Qwen35MoeAdapter) = :qwen3_5_moe

function _qwen35_moe_spec(text::AbstractDict)
    n_experts = Int(text["num_experts"])
    n_used = Int(get(text, "num_experts_per_tok", text["num_experts"]))
    expert_ff = Int(
        get(text, "moe_intermediate_size", get(text, "intermediate_size", 0)),
    )
    shared_ff = Int(get(text, "shared_expert_intermediate_size", expert_ff))
    expert_ff > 0 ||
        error("parse_config: qwen3_5_moe needs moe_intermediate_size or intermediate_size")
    return MoeSpec(; n_experts, n_expert_used=n_used, expert_ff, shared_ff)
end

function parse_config(::Qwen35MoeAdapter, raw::AbstractDict)
    text = _qwen35_text(raw)
    haskey(text, "num_experts") ||
        error("parse_config: qwen3_5_moe config is missing num_experts")
    base = parse_config(Qwen35Adapter(), raw)
    moe = _qwen35_moe_spec(text)
    return ArchitectureSpec(;
        family=:qwen3_5_moe,
        hidden_size=base.hidden_size,
        num_layers=base.num_layers,
        n_heads=base.n_heads,
        n_kv_heads=base.n_kv_heads,
        vocab_size=base.vocab_size,
        intermediate_size=moe.expert_ff,
        head_dim=base.head_dim,
        norm_kind=base.norm_kind,
        activation_kind=base.activation_kind,
        rope=base.rope,
        tie_word_embeddings=base.tie_word_embeddings,
        attention_bias=base.attention_bias,
        mlp_bias=base.mlp_bias,
        sliding_window=base.sliding_window,
        features=base.features,
        mixer=base.mixer,
        rms_norm_eps=base.rms_norm_eps,
        moe,
    )
end

function _qwen35_moe_ffn_refs()
    return ParamRef[
        ParamRef(SemanticParamId(ROLE_ROUTER, 0, :projection), "mlp.gate.weight"),
        ParamRef(SemanticParamId(ROLE_EXPERT_GATE, 0, :expert), "mlp.experts.gate_proj.weight"),
        ParamRef(SemanticParamId(ROLE_EXPERT_UP, 0, :expert), "mlp.experts.up_proj.weight"),
        ParamRef(SemanticParamId(ROLE_EXPERT_DOWN, 0, :expert), "mlp.experts.down_proj.weight"),
        ParamRef(
            SemanticParamId(ROLE_SHARED_GATE, 0, :projection),
            "mlp.shared_expert.gate_proj.weight",
        ),
        ParamRef(
            SemanticParamId(ROLE_SHARED_UP, 0, :projection),
            "mlp.shared_expert.up_proj.weight",
        ),
        ParamRef(
            SemanticParamId(ROLE_SHARED_DOWN, 0, :projection),
            "mlp.shared_expert.down_proj.weight",
        ),
        ParamRef(
            SemanticParamId(ROLE_SHARED_GATE_INP, 0, :frozen),
            "mlp.shared_expert_gate.weight",
        ),
    ]
end

function param_map(::Qwen35MoeAdapter, spec::ArchitectureSpec)
    dense = param_map(Qwen35Adapter(), spec)
    ffn = _qwen35_moe_ffn_refs()
    drop = Set((ROLE_GATE, ROLE_UP, ROLE_DOWN))
    full = [r for r in dense.layer_refs if r.id.role ∉ drop]
    append!(full, ffn)
    linear = [r for r in dense.linear_refs if r.id.role ∉ drop]
    append!(linear, ffn)
    return FamilyParamMap(
        :qwen3_5_moe,
        dense.layer_prefix,
        full,
        dense.embedding_external,
        dense.final_norm_external,
        dense.lm_head_external;
        ignored_prefixes=dense.ignored_prefixes,
        linear_refs=linear,
    )
end

"""
    gguf_param_map(::Qwen35Adapter, spec)

llama.cpp `qwen35` (dense) tensor spellings: the same hybrid mixer names as
`qwen35moe`, with a dense SwiGLU FFN (`ffn_gate/up/down.weight`) in place of
the routed experts. The fused Q|gate rows follow `fused_qg_rows`. MTP blocks
are not part of `spec.num_layers`, so they are never bound here.
"""
function gguf_param_map(::Qwen35Adapter, spec::ArchitectureSpec)
    (q_rows, gate_rows) = fused_qg_rows(spec.n_heads, spec.head_dim)
    shared = ParamRef[
        ParamRef(SemanticParamId(ROLE_NORM_PRE_ATTN, 0, :frozen), "attn_norm.weight"),
        ParamRef(
            SemanticParamId(ROLE_NORM_POST_ATTN, 0, :frozen),
            "post_attention_norm.weight",
        ),
        ParamRef(SemanticParamId(ROLE_GATE, 0, :projection), "ffn_gate.weight"),
        ParamRef(SemanticParamId(ROLE_UP, 0, :projection), "ffn_up.weight"),
        ParamRef(SemanticParamId(ROLE_DOWN, 0, :projection), "ffn_down.weight"),
    ]
    full = ParamRef[
        shared...,
        ParamRef(SemanticParamId(ROLE_Q, 0, :projection), "attn_q.weight", q_rows),
        ParamRef(SemanticParamId(ROLE_ATTN_GATE, 0, :projection), "attn_q.weight", gate_rows),
        ParamRef(SemanticParamId(ROLE_Q_NORM, 0, :frozen), "attn_q_norm.weight"),
        ParamRef(SemanticParamId(ROLE_K, 0, :projection), "attn_k.weight"),
        ParamRef(SemanticParamId(ROLE_K_NORM, 0, :frozen), "attn_k_norm.weight"),
        ParamRef(SemanticParamId(ROLE_V, 0, :projection), "attn_v.weight"),
        ParamRef(SemanticParamId(ROLE_O, 0, :projection), "attn_output.weight"),
    ]
    linear = ParamRef[
        shared...,
        ParamRef(SemanticParamId(ROLE_GDN_QKV, 0, :projection), "attn_qkv.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_Z, 0, :projection), "attn_gate.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_A, 0, :projection), "ssm_alpha.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_B, 0, :projection), "ssm_beta.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_CONV, 0, :projection), "ssm_conv1d.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_A_LOG, 0, :frozen), "ssm_a"),
        ParamRef(SemanticParamId(ROLE_GDN_DT_BIAS, 0, :frozen), "ssm_dt.bias"),
        ParamRef(SemanticParamId(ROLE_GDN_NORM, 0, :frozen), "ssm_norm.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_O, 0, :projection), "ssm_out.weight"),
    ]
    return FamilyParamMap(
        :qwen3_5,
        "blk.",
        full,
        "token_embd.weight",
        "output_norm.weight",
        "output.weight";
        linear_refs=linear,
    )
end

"""
    gguf_param_map(::Qwen35MoeAdapter, spec)

llama.cpp `qwen35moe` tensor spellings. GGUF loader transposes weights into
Gesso `(out, in)` before this map runs.
"""
function gguf_param_map(::Qwen35MoeAdapter, spec::ArchitectureSpec)
    (q_rows, gate_rows) = fused_qg_rows(spec.n_heads, spec.head_dim)
    ffn = ParamRef[
        ParamRef(SemanticParamId(ROLE_NORM_PRE_ATTN, 0, :frozen), "attn_norm.weight"),
        ParamRef(
            SemanticParamId(ROLE_NORM_POST_ATTN, 0, :frozen),
            "post_attention_norm.weight",
        ),
        ParamRef(SemanticParamId(ROLE_ROUTER, 0, :projection), "ffn_gate_inp.weight"),
        ParamRef(SemanticParamId(ROLE_EXPERT_GATE, 0, :expert), "ffn_gate_exps.weight"),
        ParamRef(SemanticParamId(ROLE_EXPERT_UP, 0, :expert), "ffn_up_exps.weight"),
        ParamRef(SemanticParamId(ROLE_EXPERT_DOWN, 0, :expert), "ffn_down_exps.weight"),
        ParamRef(SemanticParamId(ROLE_SHARED_GATE, 0, :projection), "ffn_gate_shexp.weight"),
        ParamRef(SemanticParamId(ROLE_SHARED_UP, 0, :projection), "ffn_up_shexp.weight"),
        ParamRef(SemanticParamId(ROLE_SHARED_DOWN, 0, :projection), "ffn_down_shexp.weight"),
        ParamRef(
            SemanticParamId(ROLE_SHARED_GATE_INP, 0, :frozen),
            "ffn_gate_inp_shexp.weight",
        ),
    ]
    full = ParamRef[
        ffn...,
        ParamRef(SemanticParamId(ROLE_Q, 0, :projection), "attn_q.weight", q_rows),
        ParamRef(SemanticParamId(ROLE_ATTN_GATE, 0, :projection), "attn_q.weight", gate_rows),
        ParamRef(SemanticParamId(ROLE_Q_NORM, 0, :frozen), "attn_q_norm.weight"),
        ParamRef(SemanticParamId(ROLE_K, 0, :projection), "attn_k.weight"),
        ParamRef(SemanticParamId(ROLE_K_NORM, 0, :frozen), "attn_k_norm.weight"),
        ParamRef(SemanticParamId(ROLE_V, 0, :projection), "attn_v.weight"),
        ParamRef(SemanticParamId(ROLE_O, 0, :projection), "attn_output.weight"),
    ]
    linear = ParamRef[
        ffn...,
        ParamRef(SemanticParamId(ROLE_GDN_QKV, 0, :projection), "attn_qkv.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_Z, 0, :projection), "attn_gate.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_A, 0, :projection), "ssm_alpha.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_B, 0, :projection), "ssm_beta.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_CONV, 0, :projection), "ssm_conv1d.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_A_LOG, 0, :frozen), "ssm_a"),
        ParamRef(SemanticParamId(ROLE_GDN_DT_BIAS, 0, :frozen), "ssm_dt.bias"),
        ParamRef(SemanticParamId(ROLE_GDN_NORM, 0, :frozen), "ssm_norm.weight"),
        ParamRef(SemanticParamId(ROLE_GDN_O, 0, :projection), "ssm_out.weight"),
    ]
    return FamilyParamMap(
        :qwen3_5_moe,
        "blk.",
        full,
        "token_embd.weight",
        "output_norm.weight",
        "output.weight";
        linear_refs=linear,
    )
end

"""
    GemmaAdapter

HuggingFace `gemma`: RMSNorm, and the FFN is NOT SwiGLU — `hidden_act` is
`gelu_pytorch_tanh`, so `activation_kind == :gelu`. Embeddings are ALWAYS tied
and Gemma publishes no `lm_head.weight`.

Names: Llama's, with no `lm_head`.
"""
struct GemmaAdapter <: ArchitectureAdapter end
family_symbol(::GemmaAdapter) = :gemma

function parse_config(::GemmaAdapter, raw::AbstractDict)
    _require_keys(raw, _COMMON_CONFIG_KEYS, :gemma, "parse_config")
    hd = Int(raw["hidden_size"])
    nh = Int(raw["num_attention_heads"])
    return ArchitectureSpec(;
        family=:gemma,
        hidden_size=hd,
        num_layers=Int(raw["num_hidden_layers"]),
        n_heads=nh,
        n_kv_heads=Int(get(raw, "num_key_value_heads", raw["num_attention_heads"])),
        vocab_size=Int(raw["vocab_size"]),
        intermediate_size=Int(raw["intermediate_size"]),
        norm_kind=_norm_kind(raw, :gemma, :gelu),
        activation_kind=_activation_kind(raw, :gemma),
        rope=_rope_policy(raw, :gemma; head_dim=div(hd, nh)),
        tie_word_embeddings=true,          # Gemma always ties
        attention_bias=Bool(get(raw, "attention_bias", false)),
    )
end

function param_map(::GemmaAdapter, spec::ArchitectureSpec)
    refs = ParamRef[
        ParamRef(SemanticParamId(ROLE_NORM_PRE_ATTN, 0, :frozen), "input_layernorm.weight"),
        ParamRef(SemanticParamId(ROLE_Q, 0, :projection), "self_attn.q_proj.weight"),
        ParamRef(SemanticParamId(ROLE_K, 0, :projection), "self_attn.k_proj.weight"),
        ParamRef(SemanticParamId(ROLE_V, 0, :projection), "self_attn.v_proj.weight"),
        ParamRef(SemanticParamId(ROLE_O, 0, :projection), "self_attn.o_proj.weight"),
        ParamRef(
            SemanticParamId(ROLE_NORM_POST_ATTN, 0, :frozen),
            "post_attention_layernorm.weight",
        ),
        ParamRef(SemanticParamId(ROLE_GATE, 0, :projection), "mlp.gate_proj.weight"),
        ParamRef(SemanticParamId(ROLE_UP, 0, :projection), "mlp.up_proj.weight"),
        ParamRef(SemanticParamId(ROLE_DOWN, 0, :projection), "mlp.down_proj.weight"),
    ]
    return FamilyParamMap(
        :gemma,
        "model.layers.",
        refs,
        "model.embed_tokens.weight",
        "model.norm.weight",
        nothing,
    )
end

"""
    MistralAdapter

HuggingFace `mistral`: RMSNorm + SwiGLU, Llama-shaped names, and a
`sliding_window` — a genuine attention VARIANT, declared on the spec so the
capability check can fail at `:sliding_window_attention` rather than at the
doorway.
"""
struct MistralAdapter <: ArchitectureAdapter end
family_symbol(::MistralAdapter) = :mistral

function parse_config(::MistralAdapter, raw::AbstractDict)
    _require_keys(raw, _COMMON_CONFIG_KEYS, :mistral, "parse_config")
    hd = Int(raw["hidden_size"])
    nh = Int(raw["num_attention_heads"])
    sw = get(raw, "sliding_window", nothing)
    return ArchitectureSpec(;
        family=:mistral,
        hidden_size=hd,
        num_layers=Int(raw["num_hidden_layers"]),
        n_heads=nh,
        n_kv_heads=Int(raw["num_key_value_heads"]),
        vocab_size=Int(raw["vocab_size"]),
        intermediate_size=Int(raw["intermediate_size"]),
        norm_kind=_norm_kind(raw, :mistral, :swiglu),
        activation_kind=_activation_kind(raw, :mistral),
        rope=_rope_policy(raw, :mistral; head_dim=div(hd, nh)),
        tie_word_embeddings=Bool(get(raw, "tie_word_embeddings", false)),
        attention_bias=Bool(get(raw, "attention_bias", false)),
        sliding_window=sw === nothing ? nothing : Int(sw),
    )
end

function param_map(::MistralAdapter, spec::ArchitectureSpec)
    refs = ParamRef[
        ParamRef(SemanticParamId(ROLE_NORM_PRE_ATTN, 0, :frozen), "input_layernorm.weight"),
        ParamRef(SemanticParamId(ROLE_Q, 0, :projection), "self_attn.q_proj.weight"),
        ParamRef(SemanticParamId(ROLE_K, 0, :projection), "self_attn.k_proj.weight"),
        ParamRef(SemanticParamId(ROLE_V, 0, :projection), "self_attn.v_proj.weight"),
        ParamRef(SemanticParamId(ROLE_O, 0, :projection), "self_attn.o_proj.weight"),
        ParamRef(
            SemanticParamId(ROLE_NORM_POST_ATTN, 0, :frozen),
            "post_attention_layernorm.weight",
        ),
        ParamRef(SemanticParamId(ROLE_GATE, 0, :projection), "mlp.gate_proj.weight"),
        ParamRef(SemanticParamId(ROLE_UP, 0, :projection), "mlp.up_proj.weight"),
        ParamRef(SemanticParamId(ROLE_DOWN, 0, :projection), "mlp.down_proj.weight"),
    ]
    return FamilyParamMap(
        :mistral,
        "model.layers.",
        refs,
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    )
end

"""
    Phi3Adapter

HuggingFace `phi3`: RMSNorm + SwiGLU — the SAME primitive set Gesso already
has — but FUSED checkpoint tensors:

    self_attn.qkv_proj.weight    → q | k | v      (one external tensor, 3 identities)
    mlp.gate_up_proj.weight      → gate | up      (one external tensor, 2 identities)

The published packing is per KV head-group, so the layout is DECLARED as
`FusedQKVLayout(:grouped)`, not guessed from the name. This is the sharpest
test of Pass B: a second family reaches execution with a completely different
tensor layout and ZERO new operators and ZERO engine edits.

`PhiAdapter` (the older `phi` / Phi-2 convention) is recognized below as a
non-gated LayerNorm MLP family — it builds a spec and then fails at
`:dense_ffn`, which is the capability lattice working as designed.
"""
struct Phi3Adapter <: ArchitectureAdapter end
family_symbol(::Phi3Adapter) = :phi3

function parse_config(::Phi3Adapter, raw::AbstractDict)
    _require_keys(raw, _COMMON_CONFIG_KEYS, :phi3, "parse_config")
    hd = Int(raw["hidden_size"])
    nh = Int(raw["num_attention_heads"])
    nkv = Int(get(raw, "num_key_value_heads", nh))
    return ArchitectureSpec(;
        family=:phi3,
        hidden_size=hd,
        num_layers=Int(raw["num_hidden_layers"]),
        n_heads=nh,
        n_kv_heads=nkv,
        vocab_size=Int(raw["vocab_size"]),
        intermediate_size=Int(raw["intermediate_size"]),
        norm_kind=_norm_kind(raw, :phi3, :swiglu),
        activation_kind=_activation_kind(raw, :phi3),
        rope=_rope_policy(raw, :phi3; head_dim=div(hd, nh)),
        tie_word_embeddings=Bool(get(raw, "tie_word_embeddings", true)),
        features=Set{Symbol}([:fused_qkv, :fused_gate_up]),
    )
end

function param_map(::Phi3Adapter, spec::ArchitectureSpec)
    layout = FusedQKVLayout(:grouped)
    (qr, kr, vr) = fused_qkv_rows(layout, spec.n_heads, spec.n_kv_heads, spec.head_dim)
    hidden = spec.intermediate_size
    refs = ParamRef[
        ParamRef(SemanticParamId(ROLE_NORM_PRE_ATTN, 0, :frozen), "input_layernorm.weight"),
        ParamRef(SemanticParamId(ROLE_Q, 0, :projection), "self_attn.qkv_proj.weight", qr),
        ParamRef(SemanticParamId(ROLE_K, 0, :projection), "self_attn.qkv_proj.weight", kr),
        ParamRef(SemanticParamId(ROLE_V, 0, :projection), "self_attn.qkv_proj.weight", vr),
        ParamRef(SemanticParamId(ROLE_O, 0, :projection), "self_attn.o_proj.weight"),
        ParamRef(
            SemanticParamId(ROLE_NORM_POST_ATTN, 0, :frozen),
            "post_attention_layernorm.weight",
        ),
        ParamRef(
            SemanticParamId(ROLE_GATE, 0, :projection),
            "mlp.gate_up_proj.weight",
            1:hidden,
        ),
        ParamRef(
            SemanticParamId(ROLE_UP, 0, :projection),
            "mlp.gate_up_proj.weight",
            (hidden+1):(2*hidden),
        ),
        ParamRef(SemanticParamId(ROLE_DOWN, 0, :projection), "mlp.down_proj.weight"),
    ]
    return FamilyParamMap(
        :phi3,
        "model.layers.",
        refs,
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    )
end

"""
    PhiAdapter

The older `phi` / Phi-2 convention: LayerNorm + a NON-GATED dense MLP with
`fc1` / `fc2` and a GELU activation. It parses and builds a spec on purpose;
execution then fails at `:dense_ffn` (the named missing semantic operation)
instead of at a model-family whitelist (§II).
"""
struct PhiAdapter <: ArchitectureAdapter end
family_symbol(::PhiAdapter) = :phi

function parse_config(::PhiAdapter, raw::AbstractDict)
    _require_keys(raw, _COMMON_CONFIG_KEYS, :phi, "parse_config")
    hd = Int(raw["hidden_size"])
    nh = Int(raw["num_attention_heads"])
    return ArchitectureSpec(;
        family=:phi,
        hidden_size=hd,
        num_layers=Int(raw["num_hidden_layers"]),
        n_heads=nh,
        n_kv_heads=Int(get(raw, "num_key_value_heads", nh)),
        vocab_size=Int(raw["vocab_size"]),
        intermediate_size=Int(raw["intermediate_size"]),
        norm_kind=:layer,
        activation_kind=:gelu,
        rope=_rope_policy(raw, :phi; head_dim=div(hd, nh)),
        tie_word_embeddings=Bool(get(raw, "tie_word_embeddings", true)),
    )
end

function param_map(::PhiAdapter, spec::ArchitectureSpec)
    refs = ParamRef[
        ParamRef(SemanticParamId(ROLE_NORM_PRE_ATTN, 0, :frozen), "input_layernorm.weight"),
        ParamRef(SemanticParamId(ROLE_Q, 0, :projection), "self_attn.q_proj.weight"),
        ParamRef(SemanticParamId(ROLE_K, 0, :projection), "self_attn.k_proj.weight"),
        ParamRef(SemanticParamId(ROLE_V, 0, :projection), "self_attn.v_proj.weight"),
        ParamRef(SemanticParamId(ROLE_O, 0, :projection), "self_attn.dense.weight"),
        ParamRef(
            SemanticParamId(ROLE_NORM_POST_ATTN, 0, :frozen),
            "post_attention_layernorm.weight",
        ),
        ParamRef(SemanticParamId(ROLE_UP, 0, :projection), "mlp.fc1.weight"),
        ParamRef(SemanticParamId(ROLE_DOWN, 0, :projection), "mlp.fc2.weight"),
    ]
    return FamilyParamMap(
        :phi,
        "model.layers.",
        refs,
        "embed_tokens.weight",
        "final_layernorm.weight",
        "lm_head.weight",
    )
end

# --- the family registry ---------------------------------------------------------

# External `model_type` string → adapter. This registry is the ONLY place a
# family name is matched, and it lives entirely at the import boundary.
const _ADAPTERS = Dict{String, ArchitectureAdapter}(
    "llama" => LlamaAdapter(),
    "qwen2" => Qwen2Adapter(),
    "qwen3_5" => Qwen35Adapter(),
    "qwen3_5_moe" => Qwen35MoeAdapter(),
    "gemma" => GemmaAdapter(),
    "mistral" => MistralAdapter(),
    "phi3" => Phi3Adapter(),
    "phi" => PhiAdapter(),
)

"""
    adapter_for(model_type::AbstractString) -> ArchitectureAdapter

Resolve an external `model_type` to its adapter. An unknown family refuses
WITH THE NAME (§LXX) — this is a transport-level "I have no importer for this
spelling", which is different from "I cannot execute these semantics": the
latter happens later, at the named operation (Pass F).
"""
function adapter_for(model_type::AbstractString)
    haskey(_ADAPTERS, String(model_type)) || error(
        "adapter_for: no architecture adapter for model_type $(repr(model_type)) — " *
        "known: $(join(sort(collect(keys(_ADAPTERS))), ", "))",
    )
    return _ADAPTERS[String(model_type)]
end

"""
    known_families() -> Vector{String}

Every `model_type` this build can parse. The import report and the
compatibility matrix read THIS — documentation is generated from the registry,
so it cannot claim support the registry does not have (Pass J).
"""
known_families() = sort(collect(keys(_ADAPTERS)))

"""
    architecture_spec(path) -> ArchitectureSpec

`config.json` → `ArchitectureSpec` through the family's adapter. The new
canonical doorway.
"""
function architecture_spec(path::AbstractString)
    raw = _strict_jsonfile(path)
    mt = String(raw["model_type"])
    text = raw
    if haskey(raw, "text_config") && raw["text_config"] isa AbstractDict
        text = raw["text_config"]
    end
    if mt == "qwen3_5" && haskey(text, "num_experts")
        mt = "qwen3_5_moe"
    end
    return parse_config(adapter_for(mt), raw)
end

export ArchitectureAdapter,
    LlamaAdapter,
    Qwen2Adapter,
    Qwen35Adapter,
    Qwen35MoeAdapter,
    GemmaAdapter,
    MistralAdapter,
    Phi3Adapter,
    PhiAdapter,
    FusedQKVLayout,
    fused_qkv_rows,
    fused_qg_rows,
    adapter_for,
    known_families,
    architecture_spec,
    family_symbol,
    gguf_param_map
