# ModelIR — the semantic composition graph (§VIII, §CIX; Phase 1).
#
# A ModelIR is WHAT the architecture is: an ordered composition of semantic
# primitives (an embedding, attention, FFN, norms) — NOT a tensor graph, NOT
# a runtime (no LlamaRuntime, §VIII), NOT a holder of KV contents or
# workspace buffers (those are Runtime/Inference concerns).
#
# §CIX encoding, implemented here:
#   * a node is an IMMUTABLE value; rewrites construct a new graph
#   * identity is STRUCTURAL: same primitive, same children, same logical
#     parameter bindings ⇒ same node. The block list is a Tuple (not a
#     Vector) so Julia's deep `===` over immutable fields gives structural
#     equality with no custom == overload — and no in-place mutation.
#   * a new architecture is a new composition, usually not a new node type.

module ModelIR

"""
    Embedding(; dim)

Token embedding primitive: `vocab_size × dim` table lookup. The table itself
is a `Parameters.EmbeddingTable` (logical parameter binding); the node only
records the semantic role.
"""
struct Embedding
    dim::Int
    Embedding(; dim) = dim > 0 ? new(dim) : throw(ArgumentError("dim must be positive"))
end

"""
    RMSNorm(; dim)

RMSNorm primitive over a `dim`-wide activation.
"""
struct RMSNorm
    dim::Int
    RMSNorm(; dim) = dim > 0 ? new(dim) : throw(ArgumentError("dim must be positive"))
end

"""
    RoPE()

Rotary position embedding primitive. Stateless; position handling is a
workload/lowering concern (§XXX).
"""
struct RoPE end

"""
    Attention(; n_heads, n_kv_heads = n_heads)

Attention primitive. `n_kv_heads < n_heads` is grouped-query attention;
the default makes MHA the ordinary case. Head layout is a realization
concern, not node state.
"""
struct Attention
    n_heads::Int
    n_kv_heads::Int
    Attention(; n_heads, n_kv_heads=n_heads) = begin
        n_heads > 0 || throw(ArgumentError("n_heads must be positive"))
        0 < n_kv_heads <= n_heads ||
            throw(ArgumentError("n_kv_heads must be in 1..n_heads (got $n_kv_heads)"))
        new(n_heads, n_kv_heads)
    end
end

"""
    SwiGLU(; hidden)

The dense FFN primitive with SwiGLU gating: `dim → hidden → dim`. The
fixture `kind = "mlp"` maps onto this primitive.
"""
struct SwiGLU
    hidden::Int
    SwiGLU(; hidden) =
        hidden > 0 ? new(hidden) : throw(ArgumentError("hidden must be positive"))
end

"""
    Block(attention, ffn)

One transformer block: attention sublayer then FFN sublayer, in that order.
A Block is a composition of primitives — not a runtime, not a type family
per architecture (§VIII).
"""
struct Block
    attention::Attention
    ffn::SwiGLU
end

"""
    Model(; vocab_size, embedding, blocks)

The whole architecture: an ordered composition — vocab size, the embedding
primitive, and an ordered tuple of Blocks. Identity is structural:
two independently built models with the same composition ARE the same model.

`blocks` is a Tuple on purpose (§CIX structural identity; Vector would be a
mutable identity). Rewrites construct a new Model.
"""
struct Model
    vocab_size::Int
    embedding::Embedding
    blocks::Tuple{Vararg{Block}}
    Model(; vocab_size, embedding, blocks) = begin
        vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
        embedding isa Embedding ||
            throw(ArgumentError("embedding must be an Embedding"))
        new(vocab_size, embedding, blocks)
    end
end

export Embedding, RMSNorm, RoPE, Attention, SwiGLU, Block, Model

end # module ModelIR
