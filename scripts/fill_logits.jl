# scripts/fill_logits.jl — one-shot: run the CPU prefill oracle on toy2 with
# the canonical prompt and write test/fixtures/toy/expected_logits.toml.
#
# Run via: julia --project=test scripts/fill_logits.jl
# The provenance (commit, fixture seed) is filled automatically; the file is
# deterministic — re-running produces an identical file.

using Gesso

include(joinpath(@__DIR__, "..", "test", "testhelpers.jl"))
using .GessoTestHelpers

include(joinpath(@__DIR__, "..", "test", "toyfixtures.jl"))
import .ToyFixtures

# the toy2 ModelIR builder lives in test_modelir.jl among the testsets —
# rebuild it here without the test harness by re-including its builder part.
# Simpler and truthful: call the same mapping directly.
function toy2_modelir(fx)
    blocks = Gesso.Block[]
    i = 1
    while i <= length(fx.blocks)
        b = fx.blocks[i]
        if b.kind === :attention
            i + 1 <= length(fx.blocks) && fx.blocks[i+1].kind === :mlp ||
                error("toy2 builder: attention block not followed by mlp")
            push!(
                blocks,
                Gesso.Block(
                    Gesso.Attention(; n_heads=b.n_heads),
                    Gesso.SwiGLU(; hidden=fx.blocks[i+1].hidden),
                ),
            )
            i += 2
        else
            error("toy2 builder: unexpected leading block kind :$(b.kind)")
        end
    end
    return Gesso.Model(;
        vocab_size=fx.vocab_size,
        embedding=Gesso.Embedding(; dim=fx.dim),
        blocks=Tuple(blocks),
    )
end

fx = ToyFixtures.load_toy_fixture()
model = toy2_modelir(fx)

# materialize the walk (same order as test/test_reference_prefill.jl — the
# fixture protocol in test/fixtures/toy/README.md)
m = model
dim, vocab = m.embedding.dim, m.vocab_size
stream = ToyFixtures.toy_weights(fx, 1_000_000)
cursor = Ref(1)                              # boxed so `take` can advance it
take(shape...) = begin
    n = prod(shape)
    p = cursor[]
    arr = reshape(Float64.(stream[p:(p+n-1)]), shape...)
    cursor[] = p + n
    arr
end
E = take(vocab, dim)
blocks = map(m.blocks) do b
    (
        wq=Gesso.ProjectionWeight(; shape=(dim, dim), storage=take(dim, dim)),
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
embedding = Gesso.EmbeddingTable(; shape=(vocab, dim), storage=E)
tensors = (embedding=embedding, blocks=blocks, lm_head=embedding)

prompt = [1, 3, 4, 5]
L = Gesso.reference_prefill(model, tensors, prompt)
println("oracle logits: ", size(L), "   max_abs = ", maximum(abs, L))
println("last-position argmax (0-based): ", argmax(@view L[:, end]) - 1)

commit = strip(read(`git log -1 --format=%h`, String))

io = IOBuffer()
println(io, "# Expected logits for toy2, prompt [1, 3, 4, 5] (0-based, BOS-prefixed).")
println(
    io,
    "# Produced by the Phase 2 CPU oracle (scripts/fill_logits.jl) — NOT hand-computed.",
)
println(
    io,
    "# Column t = next-token logits after consuming tokens 1..t. Dense (vocab, seq).",
)
println(io, "schema = \"gesso-toy-expected-logits-v1\"")
println(io)
println(io, "[provenance]")
println(io, "oracle = \"cpu\"")
println(io, "commit = \"", commit, "\"")
println(io, "fixture_seed = \"", fx.seed, "\"")
println(io)
for c in 1:size(L, 2), r in 1:size(L, 1)
    println(io, "[[value]]")
    println(io, "row = ", r - 1)
    println(io, "col = ", c - 1)
    println(io, "v = ", repr(L[r, c]))
    println(io)
end
out = joinpath(@__DIR__, "..", "test", "fixtures", "toy", "expected_logits.toml")
write(out, String(take!(io)))
println("wrote ", out)
