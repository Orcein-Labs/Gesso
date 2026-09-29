# Item B tests: ModelIR composition (§VIII, §CIX).
#
# The fixture → primitives builder lives HERE in test/ (laboratory code):
# fixture `kind` strings stay fixture data; the builder maps them onto
# primitives. The TOML schema is not a Gesso type.

using .ToyFixtures: load_toy_fixture

"""
    toy2_modelir() -> Gesso.Model

Build the toy architecture's ModelIR from the fixture pack. The fixture's
flat block entries (attention, mlp, attention, mlp) are paired into
`Block(attention, ffn)` compositions — mapping data onto primitives is
exactly the importer-shaped work Phase 3 generalizes.
"""
function toy2_modelir()
    fx = load_toy_fixture()
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

@testset "ModelIR primitives (§CIX): construction + validation" begin
    @test Gesso.Embedding(; dim=16).dim == 16
    @test Gesso.RMSNorm(; dim=16).dim == 16
    @test Gesso.RoPE() isa Gesso.RoPE
    @test Gesso.SwiGLU(; hidden=64).hidden == 64

    a = Gesso.Attention(; n_heads=2)
    @test a.n_heads == 2
    @test a.n_kv_heads == 2              # MHA is the ordinary case
    @test Gesso.Attention(n_heads=4, n_kv_heads=2).n_kv_heads == 2  # GQA
    @test_throws ArgumentError Gesso.Attention(n_heads=2, n_kv_heads=4)
    @test_throws ArgumentError Gesso.Embedding(dim=0)

    blk = Gesso.Block(a, Gesso.SwiGLU(hidden=64))
    @test blk.attention === a
    @test blk isa Gesso.Block
end

@testset "ModelIR (§CIX): toy2 round-trip from the fixture" begin
    fx = load_toy_fixture()
    m = toy2_modelir()
    @test m isa Gesso.Model
    @test m.vocab_size == fx.vocab_size == 32
    @test m.embedding.dim == fx.dim == 16
    @test length(m.blocks) == 2                                  # 4 flat entries → 2 Blocks
    @test m.blocks[1].attention.n_heads == 2
    @test m.blocks[1].ffn.hidden == 64
    @test m.blocks[2].attention.n_heads == m.blocks[1].attention.n_heads
    @test m.blocks[2].ffn.hidden == m.blocks[1].ffn.hidden
end

@testset "ModelIR (§CIX): identity is structural" begin
    # two independently built copies ARE the same value (Tuple-of-immutable
    # composition; no custom == anywhere)
    @test toy2_modelir() === toy2_modelir()

    # changing n_heads yields a DIFFERENT model
    m = toy2_modelir()
    variant = Gesso.Model(;
        vocab_size=m.vocab_size,
        embedding=m.embedding,
        blocks=map(m.blocks) do b
            Gesso.Block(Gesso.Attention(n_heads=4, n_kv_heads=b.attention.n_kv_heads), b.ffn)
        end,
    )
    @test m !== variant

    # changing BLOCK ORDER yields a different model. toy2's two blocks are
    # identical, so reversing them is structurally the SAME model (§CIX says
    # that is correct!) — build a distinguishable pair instead: same shape,
    # different head counts (also exercises the GQA form).
    b1 = Gesso.Block(Gesso.Attention(n_heads=2), Gesso.SwiGLU(hidden=64))
    b2 = Gesso.Block(Gesso.Attention(n_heads=4, n_kv_heads=2), Gesso.SwiGLU(hidden=32))
    m_pair = Gesso.Model(vocab_size=32, embedding=Gesso.Embedding(dim=16), blocks=(b1, b2))
    @test Gesso.Model(;
        vocab_size=32,
        embedding=Gesso.Embedding(dim=16),
        blocks=(b2, b1),
    ) !== m_pair
    @test Gesso.Model(;
        vocab_size=32,
        embedding=Gesso.Embedding(dim=16),
        blocks=(b1, b2),
    ) === m_pair
    @test Gesso.Model(;
        vocab_size=32,
        embedding=Gesso.Embedding(dim=16),
        blocks=(b1, b1),
    ) !== m_pair

    # a different embedding dim is a different model
    @test Gesso.Model(; vocab_size=32, embedding=Gesso.Embedding(dim=8), blocks=()) !==
          Gesso.Model(; vocab_size=32, embedding=Gesso.Embedding(dim=16), blocks=())
end

@testset "ModelIR (§CIX): nodes are immutable values" begin
    m = toy2_modelir()
    err = try
        m.vocab_size = 1
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    err2 = try
        m.blocks[1] = m.blocks[2]
        nothing
    catch e
        e
    end
    # tuple setindex! fails with MethodError; a setfield!-style mutation
    # fails with ErrorException — either way, in-place rewrite is impossible
    @test err2 isa Exception

    # no per-family runtime types (§VIII/§CIX)
    @test !isdefined(Gesso.ModelIR, :LlamaModel)
    @test !isdefined(Gesso.ModelIR, :QwenModel)
    @test !isdefined(Gesso, :LlamaModel)
end
