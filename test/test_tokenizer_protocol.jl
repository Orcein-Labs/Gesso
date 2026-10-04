# Pass C: the tokenizer PROTOCOL. GPT-2 byte-level BPE is ONE implementation;
# `MetaspaceBPE` is a materially different second one, and both answer the same
# protocol so no caller can special-case.
#
# Fixture provenance (stated because the law requires it — §VII, citations):
#
#   test/fixtures/gpt2_tiny/          EXISTING checked-in GPT-2 fixture; the
#                                     golden ids in test_tokenizer_gpt2.jl were
#                                     traced from its own merge table.
#
#   test/fixtures/metaspace_tiny/     SYNTHETIC, authored for this batch. Its
#                                     vocabulary is the 256 `<0xNN>` byte
#                                     fallbacks (making the encoder TOTAL) plus
#                                     a handful of merged word tokens and the
#                                     merges that build them. The expected ids
#                                     below are DERIVED FROM THAT FIXTURE by
#                                     the published metaspace algorithm, NOT
#                                     copied from a production SentencePiece
#                                     model. Claiming otherwise would be a
#                                     citation we do not have. A conformance
#                                     fixture traced from a real published
#                                     tokenizer is listed as remaining work.

using JSON

const META_DIR = joinpath(@__DIR__, "fixtures", "metaspace_tiny")
const GPT2_DIR = joinpath(@__DIR__, "fixtures", "gpt2_tiny")

@testset "Pass C: both tokenizers satisfy ONE protocol" begin
    meta = Gesso.load_metaspace_tokenizer(META_DIR)
    gpt2 = Gesso.load_gpt2_tokenizer(GPT2_DIR)

    # different implementations, same protocol surface
    @test meta isa Gesso.MetaspaceBPE
    @test gpt2 isa Gesso.GPT2BPE
    @test !(meta isa typeof(gpt2))
    @test !(gpt2 isa typeof(meta))

    m1 = Gesso.tokenizer_metadata(meta)
    m2 = Gesso.tokenizer_metadata(gpt2)
    for m in (m1, m2)
        @test m.kind isa Symbol
        @test m.vocab_size > 0
        @test 0 <= m.bos_token_id < m.vocab_size
        @test 0 <= m.eos_token_id < m.vocab_size
    end
    # ...and they really are DIFFERENT implementations
    @test m1.kind !== m2.kind
    @test Gesso.vocab_size(meta) == m1.vocab_size
    @test Gesso.vocab_size(gpt2) == m2.vocab_size
end

@testset "Pass C: metaspace encode uses merges, byte fallback covers the rest" begin
    meta = Gesso.load_metaspace_tokenizer(META_DIR)
    vocab = meta.vocab

    # "hello world" → ▁hello, ▁world, both reached through the merge table
    ids = Gesso.encode(meta, "hello world")
    @test ids == [vocab["▁hello"], vocab["▁world"]]

    # an unseen word falls back to per-byte tokens — TOTAL, never an error.
    # 'z' is not in the fixture vocabulary, so it MUST take the byte path.
    ids_z = Gesso.encode(meta, "z")
    @test ids_z == [vocab["▁"], vocab["<0x7A>"]]

    # non-ASCII round-trips through multi-byte fallbacks
    ids_u = Gesso.encode(meta, "é")
    @test ids_u == [vocab["▁"], vocab["<0xC3>"], vocab["<0xA9>"]]
end

@testset "Pass C: decode is the exact inverse for both implementations" begin
    meta = Gesso.load_metaspace_tokenizer(META_DIR)
    for text in ("hello world", "hi", "a b", "hello hello")
        ids = Gesso.encode(meta, text)
        @test Gesso.decode(meta, ids) == text
    end

    # The GPT-2 tiny fixture's vocabulary is 13 tokens, so the round-trip is
    # asserted over exactly the strings its own golden test encodes.
    gpt2 = Gesso.load_gpt2_tokenizer(GPT2_DIR)
    for text in ("Hello", "ab", " Hello", "ab ", "  a")
        ids = Gesso.encode(gpt2, text)
        @test Gesso.decode(gpt2, ids) == text
    end
end

@testset "Pass C: GPT-2 remains reachable as ONE implementation (no regression)" begin
    gpt2 = Gesso.load_gpt2_tokenizer(GPT2_DIR)
    # the pre-existing golden behaviour is unchanged (ids pinned in
    # test_tokenizer_gpt2.jl; asserted here only to prove the protocol layer
    # did not disturb it)
    @test Gesso.encode(gpt2, "Hello") == [10]
    @test Gesso.encode(gpt2, "ab") == [11]
    @test Gesso.encode(gpt2, " Hello") == [6, 10]
    # and it is now ALSO a Tokenizer, reachable through the protocol
    @test Gesso.tokenizer_metadata(gpt2).kind === :gpt2_byte_level_bpe
end

@testset "Pass C: loader refusals are loud (§LXX)" begin
    dir = mktempdir()
    raw = JSON.parsefile(joinpath(META_DIR, "metaspace.json"))

    write_cfg(d) = open(joinpath(dir, "metaspace.json"), "w") do io
        JSON.print(io, d)
    end

    # empty vocabulary
    bad = deepcopy(raw)
    bad["vocab"] = Dict{String, Any}()
    write_cfg(bad)
    @test_throws ErrorException Gesso.load_metaspace_tokenizer(dir)

    # merge part absent from the vocabulary
    bad = deepcopy(raw)
    bad["merges"] = [["zzz", "yyy"]]
    write_cfg(bad)
    err = try
        Gesso.load_metaspace_tokenizer(dir)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("not in the vocab", sprint(showerror, err))

    # special id outside the vocabulary
    bad = deepcopy(raw)
    bad["eos_token_id"] = 10^9
    write_cfg(bad)
    @test_throws ErrorException Gesso.load_metaspace_tokenizer(dir)

    # missing file
    @test_throws ErrorException Gesso.load_metaspace_tokenizer(mktempdir())
end

@testset "Pass C: the engine must not special-case a tokenizer kind" begin
    # The protocol is the ONLY contract the engine needs: a caller that knows
    # only `encode`/`decode` works with either implementation. If execution
    # ever branched on `:gpt2_byte_level_bpe` it could not do this.
    for (tk, text) in (
        (Gesso.load_metaspace_tokenizer(META_DIR), "hello world"),
        (Gesso.load_gpt2_tokenizer(GPT2_DIR), "Hello"),
    )
        ids = Gesso.encode(tk, text)
        @test ids isa Vector{Int}
        @test all(0 .<= ids .< Gesso.vocab_size(tk))
        @test Gesso.decode(tk, ids) == text
    end
end
