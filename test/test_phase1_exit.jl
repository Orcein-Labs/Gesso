# Phase 1 exit criterion (§LXXIV, §CIX): the toy reference model `toy2` is
# expressible entirely through the semantic core.
#
# "Expressible" means the whole chain exists and type-checks:
#
#     fixture data → ModelIR → semantic tensors → named operators
#
# with storage unset everywhere and NO execution. This test does NOT run a
# forward pass and does NOT touch expected_logits.toml — the CPU oracle is
# Phase 2's. (Included after test_modelir.jl, which defines toy2_modelir().)

using .ToyFixtures: load_toy_fixture

@testset "Phase 1 exit: toy2 is expressible through the semantic core (§CIX)" begin
    # 1. load the toy fixture (data)
    fx = load_toy_fixture()

    # 2. build ModelIR from it (test-side builder maps fixture data onto
    #    primitives; the TOML schema is not a Harpe type)
    m = toy2_modelir()
    @test m isa Harpe.Model
    @test m.vocab_size == fx.vocab_size
    @test m.embedding.dim == fx.dim
    @test length(m.blocks) == 2

    # 3. construct the semantic tensors the model implies, with shapes
    #    DERIVED from the IR and storage unset (bytes are not the meaning).
    #    toy2 implies: an embedding table; per block, Q/K/V/O projections,
    #    FFN gate/up/down projections, RMSNorm scales; a (freshly empty) KV
    #    cache; activations and a score workspace. It implies NO
    #    ExpertWeight/RoutingState (no MoE), NO AdapterDelta (no adapters),
    #    NO QuantizedParameter (no quantization) — skipped, not invented.
    dim = m.embedding.dim
    tensors = Harpe.SemanticTensor[]

    push!(tensors, Harpe.EmbeddingTable(; shape=(m.vocab_size, dim)))
    for blk in m.blocks
        h, hd = blk.attention.n_heads, div(dim, blk.attention.n_heads)
        push!(tensors, Harpe.ProjectionWeight(; shape=(dim, dim)))        # Q
        push!(tensors, Harpe.ProjectionWeight(; shape=(hd, dim)))         # K (MHA: full)
        push!(tensors, Harpe.ProjectionWeight(; shape=(hd, dim)))         # V (MHA: full)
        push!(tensors, Harpe.ProjectionWeight(; shape=(dim, dim)))        # O
        push!(tensors, Harpe.ProjectionWeight(; shape=(blk.ffn.hidden, dim)))  # gate
        push!(tensors, Harpe.ProjectionWeight(; shape=(blk.ffn.hidden, dim)))  # up
        push!(tensors, Harpe.ProjectionWeight(; shape=(dim, blk.ffn.hidden)))  # down
        push!(tensors, Harpe.FrozenParameter(; shape=(dim,)))             # RMSNorm scale
    end
    n_layers = length(m.blocks)
    kv_heads = m.blocks[1].attention.n_kv_heads
    head_dim = div(dim, m.blocks[1].attention.n_heads)
    push!(tensors, Harpe.KVCache(; shape=(n_layers, kv_heads, head_dim, 0)))  # empty cache
    push!(tensors, Harpe.Activation(; shape=(0, dim)))             # hidden states
    push!(tensors, Harpe.TemporaryWorkspace(; shape=(m.blocks[1].attention.n_heads, 0, 0)))

    @test all(t -> t.storage === nothing, tensors)   # nothing materialized

    # frozen-ness falls out of the families the model implies
    weights = filter(
        t ->
            t isa
            Union{Harpe.ProjectionWeight, Harpe.EmbeddingTable, Harpe.FrozenParameter},
        tensors,
    )
    @test all(Harpe.frozen, weights)
    @test !Harpe.frozen(only(t for t in tensors if t isa Harpe.KVCache))

    # 4. name the operators the blocks would call (§CIX: operators are the
    #    existing functions; the dispatch surface from item C routes them)
    ops_called = Symbol[
        :embedding_lookup!,   # table lookup
        :rmsnorm!,            # pre-attention + pre-FFN norms
        :matmul!,             # Q/K/V/O and gate/up/down projections
        :rope!,               # positions
        :softmax!,            # attention scores
        :swiglu!,             # FFN gating
    ]
    vocabulary = (
        :rmsnorm!,
        :rope!,
        :softmax!,
        :swiglu!,
        :matmul!,
        :embedding_lookup!,
        :quantize!,
        :dequantize!,
    )
    @test all(op -> op in vocabulary, ops_called)
    # toy2 implies no quantization: quantize!/dequantize! are not called
    @test :quantize! ∉ ops_called && :dequantize! ∉ ops_called

    # and every named op HAS the semantic dispatch surface (item C): it
    # resolves a method on (backend, SemanticTensor, SemanticTensor, workload)
    for op in ops_called
        f = getglobal(Harpe, op)
        @test hasmethod(
            f,
            (
                Harpe.CPUBackend,
                Harpe.SemanticTensor,
                Harpe.SemanticTensor,
                Harpe.PrefillWorkload,
            ),
        )
        @test hasmethod(
            f,
            (
                Harpe.CPUBackend,
                Harpe.SemanticTensor,
                Harpe.SemanticTensor,
                Harpe.DecodeWorkload,
            ),
        )
    end

    # 5. this test computes nothing itself: no forward pass, no logits
    #    produced here (§CIX: execution is Phase 2). The expected-logits
    #    slot was filled by the Phase 2 oracle (item B) — that state is
    #    owned and tested there; expressibility does not depend on it.
end
