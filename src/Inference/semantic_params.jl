# Semantic parameter identity + the family parameter-map boundary
# (BREADTH-0 Passes B and E).
#
# BEFORE: `llama_import.jl` carried `_LAYER_TENSOR_NAMES`, a hardcoded tuple
# of Llama tensor spellings bound to interpreter fields, plus string surgery on
# `model.layers.$i.` in three places. "Add a family" meant editing that tuple.
#
# AFTER: a CANONICAL SEMANTIC IDENTITY vocabulary
#
#     token_embedding · final_norm · lm_head
#     layer[i].attention.q / .k / .v / .o / .q_norm / .k_norm
#     layer[i].ffn.gate / .up / .down
#     layer[i].norm.pre_attention / .post_attention
#
# and a per-family map from EXTERNAL checkpoint spelling onto those
# identities. Canonical identity describes MEANING, not source spelling:
# `self_attn.q_proj.weight`, `attention.wq.weight` and a fused `qkv_proj` row
# block all mean the same thing and all bind to `layer[i].attention.q`.
#
# Pass E — the separation this file makes mechanically visible:
#
#     SemanticParam  ==  WHAT THE PARAMETER MEANS   (id, §XI family, shape law)
#     BoundParameter ==  semantic identity + HOW IT IS CURRENTLY STORED
#                       (bytes today; dtype/quant/residency/layout later)
#
# `BoundParameter` is where future materialization acts. Nothing in this batch
# fills those fields in — BREADTH-0 explicitly defers materialization — but
# the seam exists and is typed, so Phase MATERIALIZE-1 does not have to
# restructure the importer to reach it.

"""
    SemanticParamId

Canonical MEANING of one parameter (BREADTH-0 Pass B).

`role` is the canonical vocabulary; `layer` is the 0-based block index, or
`nothing` for a model-level parameter. `param_family` is the §XI semantic
tensor family the identity is realized as — NOT its storage.

These are MEANING. They say nothing about dtype, quantization, residency or
layout: that is `BoundParameter.storage` and its future materialization
fields, kept separate on purpose (Pass E).
"""
struct SemanticParamId
    role::Symbol
    layer::Union{Nothing, Int}
    param_family::Symbol
end

SemanticParamId(role::Symbol, param_family::Symbol) =
    SemanticParamId(role, nothing, param_family)

"""
    canonical_name(id) -> String

The canonical spelling of a semantic identity: `token_embedding`,
`final_norm`, `lm_head`, `layer[3].attention.q`, `layer[0].norm.pre_attention`.
Used by reports, receipts and the conformance tests — never to parse a
checkpoint.
"""
function canonical_name(id::SemanticParamId)
    id.layer === nothing && return String(id.role)
    return "layer[$(id.layer)].$(id.role)"
end

Base.:(==)(a::SemanticParamId, b::SemanticParamId) =
    a.role == b.role && a.layer == b.layer && a.param_family == b.param_family

Base.hash(s::SemanticParamId, h::UInt) =
    hash(s.param_family, hash(s.layer, hash(s.role, h)))

# --- the canonical role vocabulary ---------------------------------------------
#
# Exactly the identities the current interpreter consumes, named by MEANING.
# Inventing slots another framework happens to have is forbidden (Pass B):
# every role below is consumed today by `reference_prefill` /
# `reference_generate` / `Session` (they are the `bt.wq`, `bt.attn_rms`, …
# fields under a name that survives a different checkpoint spelling).

const ROLE_TOKEN_EMBEDDING = :token_embedding
const ROLE_FINAL_NORM = :final_norm
const ROLE_LM_HEAD = :lm_head
const ROLE_Q = Symbol("attention.q")
const ROLE_K = Symbol("attention.k")
const ROLE_V = Symbol("attention.v")
const ROLE_O = Symbol("attention.o")
const ROLE_Q_NORM = Symbol("attention.q_norm")
const ROLE_K_NORM = Symbol("attention.k_norm")
const ROLE_ATTN_GATE = Symbol("attention.gate")
const ROLE_ATTN_HEAD_GATE = Symbol("attention.head_gate")
const ROLE_GATE = Symbol("ffn.gate")
const ROLE_UP = Symbol("ffn.up")
const ROLE_DOWN = Symbol("ffn.down")
const ROLE_NORM_PRE_ATTN = Symbol("norm.pre_attention")
const ROLE_NORM_POST_ATTN = Symbol("norm.post_attention")
const ROLE_GDN_QKV = Symbol("linear.qkv")
const ROLE_GDN_Z = Symbol("linear.z")
const ROLE_GDN_A = Symbol("linear.a")
const ROLE_GDN_B = Symbol("linear.b")
const ROLE_GDN_CONV = Symbol("linear.conv")
const ROLE_GDN_A_LOG = Symbol("linear.A_log")
const ROLE_GDN_DT_BIAS = Symbol("linear.dt_bias")
const ROLE_GDN_NORM = Symbol("linear.norm")
const ROLE_GDN_O = Symbol("linear.o")
const ROLE_ROUTER = Symbol("ffn.router")
const ROLE_EXPERT_GATE = Symbol("ffn.experts.gate")
const ROLE_EXPERT_UP = Symbol("ffn.experts.up")
const ROLE_EXPERT_DOWN = Symbol("ffn.experts.down")
const ROLE_SHARED_GATE = Symbol("ffn.shared.gate")
const ROLE_SHARED_UP = Symbol("ffn.shared.up")
const ROLE_SHARED_DOWN = Symbol("ffn.shared.down")
const ROLE_SHARED_GATE_INP = Symbol("ffn.shared.gate_inp")

"""
    LAYER_ROLES

The per-layer canonical roles, in canonical order. This is the vocabulary a
family map must supply for a dense transformer block.
"""
const LAYER_ROLES = (
    ROLE_NORM_PRE_ATTN,
    ROLE_Q,
    ROLE_K,
    ROLE_V,
    ROLE_O,
    ROLE_NORM_POST_ATTN,
    ROLE_GATE,
    ROLE_UP,
    ROLE_DOWN,
)

# role → §XI parameter family. Norms are FrozenParameter (trained, frozen at
# serving); projections are ProjectionWeight. This is the Pass E meaning axis:
# a role is realized as a family regardless of how it is stored.
_role_family(::Val{Symbol("norm.pre_attention")}) = :frozen
_role_family(::Val{Symbol("norm.post_attention")}) = :frozen
_role_family(::Val{:final_norm}) = :frozen
_role_family(::Val{Symbol("attention.q_norm")}) = :frozen
_role_family(::Val{Symbol("attention.k_norm")}) = :frozen
_role_family(::Val{Symbol("linear.norm")}) = :frozen
_role_family(::Val{Symbol("linear.A_log")}) = :frozen
_role_family(::Val{Symbol("linear.dt_bias")}) = :frozen
_role_family(::Val{Symbol("ffn.shared.gate_inp")}) = :frozen
_role_family(::Val{Symbol("ffn.experts.gate")}) = :expert
_role_family(::Val{Symbol("ffn.experts.up")}) = :expert
_role_family(::Val{Symbol("ffn.experts.down")}) = :expert
_role_family(::Val) = :projection

param_family_for(role::Symbol) = _role_family(Val(role))

# --- interpreter slot: meaning → the engine's field name -----------------------
#
# The engine consumes `bt.wq`, `bt.attn_rms`, … — ALREADY a meaning-indexed
# vocabulary, not a Llama-indexed one (Inference.jl reads these fields for
# every model it runs). The map from a canonical role to that field name is
# the ONLY place the two vocabularies meet, which is why adding a family needs
# no engine edit: this table is family-agnostic.

const ROLE_TO_SLOT = Dict{Symbol, Symbol}(
    ROLE_NORM_PRE_ATTN => :attn_rms,
    ROLE_Q => :wq,
    ROLE_K => :wk,
    ROLE_V => :wv,
    ROLE_O => :wo,
    ROLE_NORM_POST_ATTN => :ffn_rms,
    ROLE_GATE => :wgate,
    ROLE_UP => :wup,
    ROLE_DOWN => :wdown,
    # qk_norm families (Qwen2) carry these two identities. The interpreter
    # does not READ them — that is exactly why importing them is not enough to
    # run the family: the `:qk_norm` capability check is what refuses. Binding
    # them here (rather than failing on the leftover tensor) is what lets the
    # family travel all the way to the capability gate (§II lattice).
    ROLE_Q_NORM => :wq_norm,
    ROLE_K_NORM => :wk_norm,
    ROLE_ATTN_GATE => :wattn_gate,
    # one sigmoid gate per head (Spark2.5 g_proj), broadcast over head_dim
    ROLE_ATTN_HEAD_GATE => :wattn_head_gate,
    ROLE_GDN_QKV => :wgdn_qkv,
    ROLE_GDN_Z => :wgdn_z,
    ROLE_GDN_A => :wgdn_a,
    ROLE_GDN_B => :wgdn_b,
    ROLE_GDN_CONV => :wgdn_conv,
    ROLE_GDN_A_LOG => :wgdn_A_log,
    ROLE_GDN_DT_BIAS => :wgdn_dt_bias,
    ROLE_GDN_NORM => :wgdn_norm,
    ROLE_GDN_O => :wgdn_o,
    ROLE_ROUTER => :wrouter,
    ROLE_EXPERT_GATE => :wexp_gate,
    ROLE_EXPERT_UP => :wexp_up,
    ROLE_EXPERT_DOWN => :wexp_down,
    ROLE_SHARED_GATE => :wshared_gate,
    ROLE_SHARED_UP => :wshared_up,
    ROLE_SHARED_DOWN => :wshared_down,
    ROLE_SHARED_GATE_INP => :wshared_gate_inp,
)

slot_for(role::Symbol) = get(ROLE_TO_SLOT, role, nothing)

"""
    role_slots() -> Vector{Symbol}

The interpreter field names for the roles a family map may bind. Family-
agnostic: this table is the ONLY meeting point between the canonical identity
vocabulary and the engine's field names, which is why a new family needs no
engine edit (§XVIII meta-metric).
"""
role_slots() = sort(unique(values(ROLE_TO_SLOT)))

# --- the family map --------------------------------------------------------------

"""
    ParamRef

How ONE canonical identity is obtained from a checkpoint: the external name
(the family-specific spelling) and, for FUSED layouts, the row block of that
external tensor the identity occupies.

`rows === nothing` means "the whole tensor". A fused `qkv_proj` yields three
ParamRefs sharing one external name with disjoint row blocks — which is why
canonical identity can never be derived by string manipulation.

The LAYOUT of a fusion is declared, not guessed: `_FusedQKVLayout` in
`arch_adapters.jl` states whether the family packs Q|K|V contiguously or per
head group, and the ranges below are built from that declaration.
"""
struct ParamRef
    id::SemanticParamId
    external::String
    rows::Union{Nothing, AbstractVector{Int}}
end

ParamRef(id::SemanticParamId, external::AbstractString) =
    ParamRef(id, String(external), nothing)

"""
    FamilyParamMap

The per-layer external→canonical mapping for one family, plus the
model-level parameters. Built by an `ArchitectureAdapter`; consumed by
`materialize_architecture`. It knows nothing about storage (Pass E) and
nothing about execution.
"""
struct FamilyParamMap
    family::Symbol
    layer_prefix::String                       # e.g. "model.layers."
    layer_refs::Vector{ParamRef}               # layer-suffixed external names
    embedding_external::String                 # e.g. "model.embed_tokens.weight"
    final_norm_external::Union{Nothing, String}
    lm_head_external::Union{Nothing, String}
    ignored_prefixes::Vector{String}
    linear_refs::Vector{ParamRef}
end

function FamilyParamMap(
    family::Symbol,
    layer_prefix::AbstractString,
    layer_refs::Vector{ParamRef},
    embedding_external::AbstractString,
    final_norm_external::Union{Nothing, AbstractString},
    lm_head_external::Union{Nothing, AbstractString};
    ignored_prefixes::Vector{String}=String[],
    linear_refs::Vector{ParamRef}=ParamRef[],
)
    return FamilyParamMap(
        family,
        String(layer_prefix),
        layer_refs,
        String(embedding_external),
        final_norm_external === nothing ? nothing : String(final_norm_external),
        lm_head_external === nothing ? nothing : String(lm_head_external),
        ignored_prefixes,
        linear_refs,
    )
end

"""
    external_name(map, ref_or_id) -> String

Resolve an external checkpoint name for a given layer. The layer origin is
0-based here ALWAYS: fixture 1-basedness is a fixture-protocol concern handled
once at the boundary (Pass B), never per-name.
"""
function external_name(map::FamilyParamMap, ref::ParamRef, layer::Int)
    return map.layer_prefix * string(layer) * "." * ref.external
end

export SemanticParamId,
    canonical_name,
    role_slots,
    param_family_for,
    slot_for,
    ParamRef,
    FamilyParamMap,
    external_name,
    LAYER_ROLES,
    ROLE_TOKEN_EMBEDDING,
    ROLE_FINAL_NORM,
    ROLE_LM_HEAD,
    ROLE_Q,
    ROLE_K,
    ROLE_V,
    ROLE_O,
    ROLE_Q_NORM,
    ROLE_K_NORM,
    ROLE_GATE,
    ROLE_UP,
    ROLE_DOWN,
    ROLE_NORM_PRE_ATTN,
    ROLE_NORM_POST_ATTN,
    ROLE_ATTN_GATE,
    ROLE_ATTN_HEAD_GATE,
    ROLE_GDN_QKV,
    ROLE_GDN_Z,
    ROLE_GDN_A,
    ROLE_GDN_B,
    ROLE_GDN_CONV,
    ROLE_GDN_A_LOG,
    ROLE_GDN_DT_BIAS,
    ROLE_GDN_NORM,
    ROLE_GDN_O,
    ROLE_ROUTER,
    ROLE_EXPERT_GATE,
    ROLE_EXPERT_UP,
    ROLE_EXPERT_DOWN,
    ROLE_SHARED_GATE,
    ROLE_SHARED_UP,
    ROLE_SHARED_DOWN,
    ROLE_SHARED_GATE_INP
