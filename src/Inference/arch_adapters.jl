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
    return parse_config(adapter_for(String(raw["model_type"])), raw)
end

export ArchitectureAdapter,
    LlamaAdapter,
    Qwen2Adapter,
    GemmaAdapter,
    MistralAdapter,
    Phi3Adapter,
    PhiAdapter,
    FusedQKVLayout,
    fused_qkv_rows,
    adapter_for,
    known_families,
    architecture_spec,
    family_symbol
