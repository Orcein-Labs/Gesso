# Spark 2.5 GGUF byte-level BPE tokenizer (Spark-X2.6-4B, pre `spark2_5`): id
# parity with llama-server /tokenize, round trip and streaming equality, and a
# negative control. The vocabulary and merges come from the real GGUF, so every
# test is gated on GESSO_SPARK25_GGUF and skips cleanly without it. Expected
# ids are committed in test/fixtures/spark25_tokenizer/corpus.json, captured
# from llama-server.

using Gesso.Inference:
    load_gguf_tokenizer,
    encode,
    decode,
    StreamDetokenizer,
    finish!,
    tokenizer_metadata,
    GGUFBPE
using JSON

const SPARK_CORPUS_PATH = joinpath(@__DIR__, "fixtures", "spark25_tokenizer", "corpus.json")
const SPARK25_GGUF = get(ENV, "GESSO_SPARK25_GGUF", "")

function _spark_mismatches(tk, corpus)
    bad = String[]
    for c in corpus["cases"]
        encode(tk, c["text"]; parse_special=c["parse_special"]) == Int.(c["ids"]) ||
            push!(bad, c["text"])
    end
    return bad
end

if !isfile(SPARK25_GGUF)
    @testset "spark2_5 GGUF tokenizer" begin
        @test_skip "set GESSO_SPARK25_GGUF to the Spark-X2.6-4B GGUF"
    end
else
    const _SPARK_TK = load_gguf_tokenizer(SPARK25_GGUF)
    const _SPARK_CORPUS = JSON.parsefile(SPARK_CORPUS_PATH)

    @testset "spark2_5: pre-tokenizer recognised" begin
        @test _SPARK_TK.pre === :spark2_5
        @test tokenizer_metadata(_SPARK_TK).kind === :gguf_spark25_bpe
    end

    @testset "spark2_5 gate 1: id parity vs llama-server /tokenize" begin
        @test length(_SPARK_CORPUS["cases"]) >= 600
        bad = _spark_mismatches(_SPARK_TK, _SPARK_CORPUS)
        @test isempty(bad)
        for t in first(bad, 5)
            @info "spark2_5 gate 1 mismatch" text = t
        end
    end

    @testset "spark2_5 gate 2: round trip and streaming equality" begin
        for c in _SPARK_CORPUS["cases"]
            ids = encode(_SPARK_TK, c["text"]; parse_special=c["parse_special"])
            dec = StreamDetokenizer(_SPARK_TK)
            streamed = join(String[push!(dec, id) for id in ids]) * finish!(dec)
            @test streamed == decode(_SPARK_TK, ids)
        end
        texts = [c["text"] for c in _SPARK_CORPUS["cases"] if !c["parse_special"]]
        @test all(
            s -> decode(_SPARK_TK, encode(_SPARK_TK, s; parse_special=false)) == s,
            texts,
        )
    end

    @testset "spark2_5 negative control: the qwen35 split fails gate 1" begin
        f = [getfield(_SPARK_TK, n) for n in fieldnames(GGUFBPE)]
        tk = GGUFBPE(f[1:(end-1)]..., :qwen35)
        @test !isempty(_spark_mismatches(tk, _SPARK_CORPUS))
    end
end
