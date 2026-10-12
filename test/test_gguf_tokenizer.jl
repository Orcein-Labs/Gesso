# Qwen3.5 GGUF byte-level BPE tokenizer (Ornith-35B): gate 1 (id parity with
# llama-server /tokenize), gate 2 (round trip and streaming equality), the
# 200 KB speed gate, and the negative controls. The vocabulary and merges come
# from the real GGUF, so every test here is gated on GESSO_ORNITH35_GGUF and
# skips cleanly without it. Expected ids are committed in
# test/fixtures/qwen35_tokenizer/corpus.json, captured from llama-server.

using Gesso.Inference: load_gguf_tokenizer, encode, decode, StreamDetokenizer, finish!, load_gguf_chat_template, chat_family
using JSON

const TOK_CORPUS_PATH = joinpath(@__DIR__, "fixtures", "qwen35_tokenizer", "corpus.json")
const ORNITH35_GGUF = get(ENV, "GESSO_ORNITH35_GGUF", "")

# Ids that differ from the corpus, as the texts that differ. An empty result
# means gate 1 passes.
function _gate1_mismatches(tk, corpus; parse_special_override = nothing)
    bad = String[]
    for c in corpus["cases"]
        ps = parse_special_override === nothing ? c["parse_special"] : parse_special_override
        got = encode(tk, c["text"]; parse_special = ps)
        got == Int.(c["ids"]) || push!(bad, c["text"])
    end
    return bad
end

if !isfile(ORNITH35_GGUF)
    @testset "qwen35 GGUF tokenizer" begin
        @test_skip "set GESSO_ORNITH35_GGUF to the Ornith 35B GGUF"
    end
else
    const _TK = load_gguf_tokenizer(ORNITH35_GGUF)
    const _CORPUS = JSON.parsefile(TOK_CORPUS_PATH)

    @testset "gate 1: tokenizer id parity vs llama-server /tokenize" begin
        cases = _CORPUS["cases"]
        @test length(cases) >= 300
        @test count(c -> c["parse_special"], cases) > 0
        @test count(c -> !c["parse_special"], cases) > 0
        bad = _gate1_mismatches(_TK, _CORPUS)
        @test isempty(bad)
        for t in first(bad, 5)
            @info "gate 1 mismatch" text = t
        end
    end

    @testset "gate 2: round trip and streaming equality" begin
        for c in _CORPUS["cases"]
            ids = encode(_TK, c["text"]; parse_special = c["parse_special"])
            dec = StreamDetokenizer(_TK)
            pieces = String[push!(dec, id) for id in ids]
            streamed = join(pieces) * finish!(dec)
            @test streamed == decode(_TK, ids)
        end
        texts = [c["text"] for c in _CORPUS["cases"] if !c["parse_special"]]
        @test all(s -> decode(_TK, encode(_TK, s; parse_special = false)) == s, texts)
    end

    @testset "speed: 200 KB single-threaded encode under 5 s after warmup" begin
        base = join((c["text"] for c in _CORPUS["cases"]), "\n")
        text = base
        while sizeof(text) < 200_000
            text *= "\n" * base
        end
        text = text[1:thisind(text, 200_000)]
        @test sizeof(text) >= 200_000 - 4
        encode(_TK, text)
        t = @elapsed ids = encode(_TK, text)
        @info "200 KB encode" seconds = t tokens = length(ids)
        @test t < 5.0
        @test decode(_TK, ids) == text
    end

    @testset "negative control: a mutated merge rank fails gate 1, then restores" begin
        orig = _TK.merge_target[1]
        _TK.merge_target[1] = _TK.merge_target[2]
        try
            @test !isempty(_gate1_mismatches(_TK, _CORPUS))
        finally
            _TK.merge_target[1] = orig
        end
        @test isempty(_gate1_mismatches(_TK, _CORPUS))
    end

    @testset "negative control: dropping the special pass fails gate 1, then restores" begin
        @test !isempty(_gate1_mismatches(_TK, _CORPUS; parse_special_override = false))
        @test isempty(_gate1_mismatches(_TK, _CORPUS))
    end

    @testset "GGUF chat template: read from tokenizer.chat_template and recognised" begin
        @test _TK.chat_template !== nothing
        @test chat_family(_TK.chat_template) === :qwen35_xml
        @test load_gguf_chat_template(ORNITH35_GGUF) == _TK.chat_template
    end
end
