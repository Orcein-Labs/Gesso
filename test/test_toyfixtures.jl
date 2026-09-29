# Toy fixture pack tests: the loader's contract enforcement IS the test
# surface. See test/fixtures/toy/README.md for the contract these pin.

using .ToyFixtures: load_toy_fixture, toy_weights

const FIXTURE_DIR = joinpath(@__DIR__, "fixtures", "toy")

@testset "toy fixtures: load + data shape" begin
    fx = load_toy_fixture()
    @test fx.name == "toy2"
    @test fx.vocab_size == 32
    @test fx.dim == 16
    @test length(fx.blocks) == 4
    @test [b.kind for b in fx.blocks] == [:attention, :mlp, :attention, :mlp]
    @test fx.blocks[1].n_heads == 2
    @test fx.blocks[2].hidden == 64
    @test fx.tokenizer.pad == 0 && fx.tokenizer.bos == 1 && fx.tokenizer.eos == 2
    @test length(fx.tokenizer.id_text) == 32
    # ids are dense 0..31: control tokens carry their names, the rest are
    # the table's placeholder strings
    @test fx.tokenizer.id_text[1] == "PAD"
    @test fx.tokenizer.id_text[2] == "BOS"
    @test fx.tokenizer.id_text[3] == "EOS"
    @test fx.tokenizer.id_text[4] == "tok3"
    @test fx.tokenizer.id_text[end] == "tok31"
    @test fx.seed isa UInt64                       # derived, run-stable
    @test fx.expected_logits === nothing           # slot open until Phase 2 oracle
end

@testset "toy fixtures: seeded weights are reproducible" begin
    fx = load_toy_fixture()
    w1 = toy_weights(fx, 128)
    w2 = toy_weights(fx, 128)
    @test w1 == w2                                  # same fixture ⇒ same stream
    @test length(w1) == 128
    @test eltype(w1) == Float64
    # prefix stability: the first 128 of a 160-draw equal the 128-draw —
    # the protocol is ONE master stream, so earlier draws never shift later ones
    @test toy_weights(fx, 160)[1:128] == w1
    # independent re-derivation from the documented seed matches
    rng = HarpeTestHelpers.deterministic_rng(fx.seed)
    @test [randn(rng) for _ in 1:128] == w1
end

@testset "toy fixtures: contract violations fail loudly" begin
    # each case mutates ONE contract point and expects the loader to refuse
    dir = mktempdir()

    # model: unknown block kind
    cp(joinpath(FIXTURE_DIR, "model.toml"), joinpath(dir, "model.toml"))
    cp(joinpath(FIXTURE_DIR, "tokenizer.toml"), joinpath(dir, "tokenizer.toml"))
    cp(joinpath(FIXTURE_DIR, "expected_logits.toml"), joinpath(dir, "expected_logits.toml"))
    write(
        joinpath(dir, "model.toml"),
        replace(
            read(joinpath(dir, "model.toml"), String),
            "kind = \"mlp\"" => "kind = \"transformer_block\"",
        ),
    )
    err = try
        load_toy_fixture(dir)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("unknown block kind", err.msg)

    # tokenizer: duplicate id
    dir2 = mktempdir()
    for f in ("model.toml", "tokenizer.toml", "expected_logits.toml")
        cp(joinpath(FIXTURE_DIR, f), joinpath(dir2, f))
    end
    t = read(joinpath(dir2, "tokenizer.toml"), String)
    write(joinpath(dir2, "tokenizer.toml"), t * "\n[[token]]\nid = 5\ntext = \"dupe\"\n")
    err2 = try
        load_toy_fixture(dir2)
        nothing
    catch e
        e
    end
    @test err2 isa ErrorException
    @test occursin("duplicate token id 5", err2.msg)

    # logits slot: filled before an oracle exists
    dir3 = mktempdir()
    for f in ("model.toml", "tokenizer.toml", "expected_logits.toml")
        cp(joinpath(FIXTURE_DIR, f), joinpath(dir3, f))
    end
    l = read(joinpath(dir3, "expected_logits.toml"), String)
    write(
        joinpath(dir3, "expected_logits.toml"),
        replace(l, "oracle = \"\"" => "oracle = \"cpu\""),
    )
    err3 = try
        load_toy_fixture(dir3)
        nothing
    catch e
        e
    end
    @test err3 isa ErrorException
    @test occursin("Phase 2 CPU oracle", err3.msg)
end
