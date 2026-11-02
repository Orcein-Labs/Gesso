# Spark-X2.6 chat rendering against Jinja2. The template is the one the GGUF
# ships; the expected prompts were rendered from it (and from the Hugging Face
# chat_template.jinja, which gives the same bytes) with the HF tojson filter.

using Test
using Gesso
using JSON

const SPARK_CHAT_FIXDIR = joinpath(@__DIR__, "fixtures", "spark_chat")
const render_spark_chat = Gesso.Inference.render_spark_chat

@testset "Spark chat oracle: spark25" begin
    template_str = read(joinpath(SPARK_CHAT_FIXDIR, "spark25.jinja"), String)
    @test Gesso.Inference.chat_family(template_str) === :spark25

    corpus = JSON.parse(
        read(joinpath(SPARK_CHAT_FIXDIR, "spark25.json"), String);
        dicttype=Gesso.Inference._OrderedDict,
    )
    @test length(corpus["cases"]) >= 18
    for case in corpus["cases"]
        @testset "$(case["name"])" begin
            input = case["input"]
            rendered = render_spark_chat(
                input["messages"];
                tools=get(input, "tools", nothing),
                add_generation_prompt=get(input, "add_generation_prompt", true),
                enable_thinking=get(input, "enable_thinking", true),
                reasoning_effort=get(input, "reasoning_effort", "high"),
            )
            @test rendered == case["expected"]
        end
    end

    path = get(ENV, "GESSO_SPARK25_GGUF", "")
    if isempty(path) || !isfile(path)
        @test_skip "set GESSO_SPARK25_GGUF to the local model GGUF"
    else
        tokenizer = Gesso.Inference.load_gguf_tokenizer(path)
        @test tokenizer.chat_template == template_str
        for tag in ("<｜start▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<|User|>", "<|Bot|>",
                    "<|System|>", "<|Tool|>", "<think>", "</think>", "<tool_call>", "</tool_call>")
            @test length(Gesso.encode(tokenizer, tag; parse_special=true)) == 1
        end
    end
end

@testset "Spark chat: python float repr" begin
    f = Gesso.Inference._sp_py_float
    @test f(0.5) == "0.5"
    @test f(2.0) == "2.0"
    @test f(1e-7) == "1e-07"
    @test f(1e-4) == "0.0001"
    @test f(1e16) == "1e+16"
    @test f(1e15) == "1000000000000000.0"
    @test f(-2.5) == "-2.5"
    @test f(123.456) == "123.456"
    @test f(1.5e300) == "1.5e+300"
    @test f(-0.0) == "-0.0"
end

@testset "Spark chat: invalid input is an error" begin
    user = Dict{String, Any}("role" => "user", "content" => "hi")
    @test_throws ErrorException render_spark_chat(Any[])
    @test_throws ErrorException render_spark_chat(Any[user]; reasoning_effort="medium")
    @test render_spark_chat(Any[user]; enable_thinking=false, reasoning_effort="medium") isa String
    @test_throws ErrorException render_spark_chat(Any[Dict{String, Any}("role" => "critic", "content" => "x")])
    @test_throws ErrorException render_spark_chat(Any[Dict{String, Any}(
        "role" => "user", "content" => Any[Dict{String, Any}("type" => "image", "url" => "x")])])
    call = Dict{String, Any}("function" => Dict{String, Any}("name" => "f", "arguments" => "{\"a\": 1}"))
    @test_throws ErrorException render_spark_chat(Any[user,
        Dict{String, Any}("role" => "assistant", "content" => "", "tool_calls" => Any[call])])
    @test_throws ErrorException render_spark_chat(Any[user]; tools=Any[Dict{String, Any}("name" => "f")])
end
