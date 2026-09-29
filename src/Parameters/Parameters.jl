# Parameters — semantic tensor families (§XI; Phase 1).
#
# Families distinguish STORAGE from MEANING: what a tensor IS to the model
# (a projection weight, an activation, decode state), not what bytes back it
# (Representation, Phase 10) and not how it executes (Operators).
#
# Discipline (§XI, §XIII, §CIX): semantic richness at parameter/tensor level;
# family is a TYPE, optimization properties are TRAITS, volatile facts are
# METADATA. Do NOT make every scalar symbolic. Do NOT encode volatile runtime
# facts as types. Gradient/OptimizerState are training-side and forbidden
# (§LVIII).
#
# §CIX encoding, implemented here:
#   TYPE      the family — stable structural identity dispatch may see
#   TRAIT     `frozen` — the ONLY trait this phase adds
#   METADATA  named fields (`shape`, `storage`) — never type parameters

module Parameters

# --- The family hierarchy (§CIX: TYPE = dispatch-visible identity) ----------

"""
    SemanticTensor

Supertype of the §XI semantic families. A value is a semantic fact ("this is
an embedding table"), never a byte layout: `storage` starts unset (`nothing`)
and is a Representation-phase concern (§XIV).
"""
abstract type SemanticTensor end

# Every family carries the same two metadata FIELDS (§CIX: metadata is
# fields or a small struct, never type parameters — a shape must not be a
# compile-time dispatch axis).
#
#   shape    the logical extents of the semantic object
#   storage  what physically holds it, if anything; `nothing` = unset
#
# Families are deliberately boring explicit structs. They differ in what they
# MEAN (and in the `frozen` trait), not in their field lists.

"""
    ProjectionWeight(; shape, storage)

A projection weight (attention/FFN linear maps). Frozen: it does not change
during inference.
"""
struct ProjectionWeight <: SemanticTensor
    shape::Tuple{Vararg{Int}}
    storage::Any
    ProjectionWeight(shape::Tuple{Vararg{Int}}, storage=nothing) = new(shape, storage)
end

"""
    KVCache(; shape, storage)

The KV cache of a session/layer. NOT frozen: decode appends. Its growth and
representation are runtime/lowering concerns (§XXXI; KV memory program) —
this type is only the semantic identity.
"""
struct KVCache <: SemanticTensor
    shape::Tuple{Vararg{Int}}
    storage::Any
    KVCache(shape::Tuple{Vararg{Int}}, storage=nothing) = new(shape, storage)
end

"""
    EmbeddingTable(; shape, storage)

The token-embedding table `(vocab, dim)`. Frozen.
"""
struct EmbeddingTable <: SemanticTensor
    shape::Tuple{Vararg{Int}}
    storage::Any
    EmbeddingTable(shape::Tuple{Vararg{Int}}, storage=nothing) = new(shape, storage)
end

"""
    ExpertWeight(; shape, storage)

A MoE expert's weights. Frozen (which experts RUN is RoutingState's business).
"""
struct ExpertWeight <: SemanticTensor
    shape::Tuple{Vararg{Int}}
    storage::Any
    ExpertWeight(shape::Tuple{Vararg{Int}}, storage=nothing) = new(shape, storage)
end

"""
    FrozenParameter(; shape, storage)

A parameter that is frozen by explicit declaration (role-level frozenness,
e.g. a frozen base model under adapters).
"""
struct FrozenParameter <: SemanticTensor
    shape::Tuple{Vararg{Int}}
    storage::Any
    FrozenParameter(shape::Tuple{Vararg{Int}}, storage=nothing) = new(shape, storage)
end

"""
    QuantizedParameter(; shape, storage)

A parameter whose representation is quantized. Its EXISTENCE as a family is
all Phase 1 provides: no quantization lattice, no representation stack
(Representation, Phase 10). Frozenness is orthogonal and not implied.
"""
struct QuantizedParameter <: SemanticTensor
    shape::Tuple{Vararg{Int}}
    storage::Any
    QuantizedParameter(shape::Tuple{Vararg{Int}}, storage=nothing) = new(shape, storage)
end

"""
    Activation(; shape, storage)

A live intermediate value flowing between operators. Never frozen.
"""
struct Activation <: SemanticTensor
    shape::Tuple{Vararg{Int}}
    storage::Any
    Activation(shape::Tuple{Vararg{Int}}, storage=nothing) = new(shape, storage)
end

"""
    TemporaryWorkspace(; shape, storage)

Scratch space an operator borrows. Ownership is scoped; content is meaningless
across uses.
"""
struct TemporaryWorkspace <: SemanticTensor
    shape::Tuple{Vararg{Int}}
    storage::Any
    TemporaryWorkspace(shape::Tuple{Vararg{Int}}, storage=nothing) = new(shape, storage)
end

"""
    RoutingState(; shape, storage)

MoE routing state — which experts a token consults. Runtime-volatile.
"""
struct RoutingState <: SemanticTensor
    shape::Tuple{Vararg{Int}}
    storage::Any
    RoutingState(shape::Tuple{Vararg{Int}}, storage=nothing) = new(shape, storage)
end

"""
    DecodeState(; shape, storage)

Per-request decode state (sampled token, position, session bookkeeping).
Runtime-volatile by definition.
"""
struct DecodeState <: SemanticTensor
    shape::Tuple{Vararg{Int}}
    storage::Any
    DecodeState(shape::Tuple{Vararg{Int}}, storage=nothing) = new(shape, storage)
end

"""
    AdapterDelta(; shape, storage)

A fine-tune delta applied over a frozen base (LoRA-class). The delta itself
is fixed at inference time — frozen as a weight-like family.
"""
struct AdapterDelta <: SemanticTensor
    shape::Tuple{Vararg{Int}}
    storage::Any
    AdapterDelta(shape::Tuple{Vararg{Int}}, storage=nothing) = new(shape, storage)
end

# Keyword construction (the documented interface): ProjectionWeight(shape=(2,3))
for F in (
    :ProjectionWeight,
    :KVCache,
    :EmbeddingTable,
    :ExpertWeight,
    :FrozenParameter,
    :QuantizedParameter,
    :Activation,
    :TemporaryWorkspace,
    :RoutingState,
    :DecodeState,
    :AdapterDelta,
)
    @eval $F(; shape, storage=nothing) = $F(shape, storage)
end

# --- The frozen trait (§CIX: the only trait Phase 1 adds) -------------------

abstract type FrozenTrait end
struct Frozen <: FrozenTrait end
struct NotFrozen <: FrozenTrait end

# default: not frozen. Weight-like families opt in below.
_frozen_trait(::Type) = NotFrozen()

for F in
    (:ProjectionWeight, :EmbeddingTable, :FrozenParameter, :ExpertWeight, :AdapterDelta)
    @eval _frozen_trait(::Type{<:$F}) = Frozen()
end

"""
    frozen(x) / frozen(::Type{T}) -> Bool

The `frozen` Holy-trait hook (§CIX): `true` for weight-like families —
`ProjectionWeight`, `EmbeddingTable`, `FrozenParameter`, `ExpertWeight`,
`AdapterDelta` — `false` for everything else. Dispatch may specialize on
this; adding a trait beyond `frozen` is a work item, not a drive-by.
"""
frozen(x) = frozen(typeof(x))
frozen(::Type{T}) where {T} = _frozen_trait(T) isa Frozen

export SemanticTensor,
    ProjectionWeight,
    KVCache,
    EmbeddingTable,
    ExpertWeight,
    FrozenParameter,
    QuantizedParameter,
    Activation,
    TemporaryWorkspace,
    RoutingState,
    DecodeState,
    AdapterDelta,
    frozen

end # module Parameters
