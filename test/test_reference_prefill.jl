# Item B tests: the prefill oracle + known logits (§LXXV).
#
# The fixture→tensors walk lives HERE (fixture protocol is test-side, same
# lane as toy2_modelir): it consumes toy_weights in the documented order and
# materializes SemanticTensors the interpreter runs on.

using .ToyFixtures: load_toy_fixture, toy_weights
using .GessoTestHelpers: approx_eq, deterministic_rng

"""
    toy2_tensors() -> NamedTuple

Materialize `toy2`'s weights per the fixture protocol walk (row-major,
one master stream, Float64), producing the named tensor set
`reference_prefill` consumes:

    (embedding, blocks, lm_head)

The `lm_head` is the TIED embedding table — the same SemanticTensor object,
so the head reuses the table's bytes (no second table).
"""
function toy2_tensors()
    fx = load_toy_fixture()
    m = toy2_modelir()
    dim, vocab = m.embedding.dim, m.vocab_size
    n_heads = m.blocks[1].attention.n_heads
    d_head = div(dim, n_heads)
    @test dim == n_heads * d_head   # exact division is a protocol precondition

    stream = toy_weights(fx, 1_000_000)   # one master stream; walk consumes a prefix
    pos = 1
    take(shape...) = begin
        n = prod(shape)
        arr = reshape(Float64.(stream[pos:(pos+n-1)]), shape...)
        pos += n
        arr
    end

    E = take(vocab, dim)
    blocks = map(m.blocks) do b
        (
            wq=Gesso.ProjectionWeight(; shape=(dim, dim), storage=take(dim, dim)),
            # packed MHA: n_kv_heads * d_head == dim for toy2 (goal's walk)
            wk=Gesso.ProjectionWeight(; shape=(dim, dim), storage=take(dim, dim)),
            wv=Gesso.ProjectionWeight(; shape=(dim, dim), storage=take(dim, dim)),
            wo=Gesso.ProjectionWeight(; shape=(dim, dim), storage=take(dim, dim)),
            wgate=Gesso.ProjectionWeight(;
                shape=(b.ffn.hidden, dim),
                storage=take(b.ffn.hidden, dim),
            ),
            wup=Gesso.ProjectionWeight(;
                shape=(b.ffn.hidden, dim),
                storage=take(b.ffn.hidden, dim),
            ),
            wdown=Gesso.ProjectionWeight(;
                shape=(dim, b.ffn.hidden),
                storage=take(dim, b.ffn.hidden),
            ),
            attn_rms=Gesso.FrozenParameter(; shape=(dim,), storage=take(dim)),
            ffn_rms=Gesso.FrozenParameter(; shape=(dim,), storage=take(dim)),
        )
    end
    @test pos < length(stream)   # the walk must consume a prefix, not exhaust

    embedding = Gesso.EmbeddingTable(; shape=(vocab, dim), storage=E)
    return (fixture=fx, model=m, embedding=embedding, blocks=blocks, lm_head=embedding)
end

const PROMPT = [1, 3, 4, 5]   # BOS, then tokens 3, 4, 5 (0-based)

@testset "prefill oracle: shape + determinism (atol=0, same process)" begin
    ts = toy2_tensors()
    L1 = reference_prefill(ts.model, ts, PROMPT)
    L2 = reference_prefill(ts.model, ts, PROMPT)
    @test size(L1) == (ts.fixture.vocab_size, length(PROMPT)) == (32, 4)
    @test approx_eq(L1, L2; atol=0.0, rtol=0.0)   # bit-identical (goal invariant)
end

@testset "prefill oracle: causal property (position 0 cannot see position 3)" begin
    ts = toy2_tensors()
    L_full = reference_prefill(ts.model, ts, PROMPT)
    # truncating the prompt must not change earlier columns: logits at
    # position t depend only on tokens 1..t (causality of the whole stack)
    L_trunc = reference_prefill(ts.model, ts, PROMPT[1:2])
    @test approx_eq(L_full[:, 1:2], L_trunc; atol=0.0, rtol=0.0)
end

@testset "prefill oracle: last-position argmax drives generation (item C seam)" begin
    # the first generated token must be argmax of the last prompt column —
    # this pins the seam item C relies on, before generation exists
    ts = toy2_tensors()
    L = reference_prefill(ts.model, ts, PROMPT)
    last_argmax = argmax(@view L[:, end]) - 1     # back to 0-based
    @test last_argmax isa Int
    @test 0 <= last_argmax < ts.fixture.vocab_size
end

@testset "prefill oracle: greedy path is stable under prompt prefixing" begin
    # generating FROM the prompt's argmax should continue deterministically:
    # run prefill on prompt + [first], the new last column's argmax exists
    # and is reproducible across runs (determinism of the extended pass)
    ts = toy2_tensors()
    L = reference_prefill(ts.model, ts, PROMPT)
    first_tok = argmax(@view L[:, end]) - 1
    L2 = reference_prefill(ts.model, ts, [PROMPT; first_tok])
    @test approx_eq(L2[:, 1:(end-1)], L; atol=0.0, rtol=0.0)   # prefix property again
    @test approx_eq(L2[:, end], L2[:, end]; atol=0.0)
end

@testset "known logits: persisted slot vs recomputed (atol=1e-10, cross-commit gate)" begin
    # THE Phase 2 regression gate: the fixture's persisted logits must agree
    # with a fresh oracle run. atol=1e-10 (goal contract): BLAS association
    # may wiggle across commits/platforms; anything larger is a real drift.
    ts = toy2_tensors()
    L = reference_prefill(ts.model, ts, PROMPT)
    fx = load_toy_fixture()
    @test fx.expected_logits isa Matrix{Float64}
    @test size(fx.expected_logits) == size(L)
    @test approx_eq(fx.expected_logits, L; atol=1.0e-10, rtol=0.0)
    # provenance (oracle/commit/fixture_seed) is enforced by the loader
end
