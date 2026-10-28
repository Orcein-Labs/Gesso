# LFM chat rendering against llama.cpp's Jinja endpoint. The checked-in
# templates and per-case prompts/token IDs are captured from the actual GGUFs;
# local runs load only the named model paths supplied by the caller.

using Test
using Gesso
using JSON

const LFM_CHAT_FIXDIR = joinpath(@__DIR__, "fixtures", "lfm_chat")

@testset "LFM chat: real template delimiter fixtures" begin
    lfm2 = read(joinpath(LFM_CHAT_FIXDIR, "lfm2.jinja"), String)
    lfm25 = read(joinpath(LFM_CHAT_FIXDIR, "lfm25.jinja"), String)
    @test Gesso.Inference.chat_family(lfm2) === :lfm2
    @test Gesso.Inference.chat_family(lfm25) === :lfm25
    @test occursin("\"<|im_start|>system\n\"", lfm2)
    @test occursin("\"<|im_start|>system\\n\"", lfm25)

    user = Dict{String, Any}("role" => "user", "content" => "fixture")
    for template in (lfm2, lfm25)
        got = Gesso.Inference._render_lfm_chat(
            template,
            "<|begin_of_text|>",
            [user],
        )
        @test got == "<|begin_of_text|><|im_start|>user\nfixture<|im_end|>\n<|im_start|>assistant\n"
    end
end

for (env, stem, family) in (
    ("GESSO_LFM2_GGUF", "lfm2", :lfm2),
    ("GESSO_LFM25_GGUF", "lfm25", :lfm25),
)
    path = get(ENV, env, "")
    @testset "LFM chat oracle: $stem" begin
        template_path = joinpath(LFM_CHAT_FIXDIR, "$stem.jinja")
        template_str = read(template_path, String)
        @test Gesso.Inference.chat_family(template_str) === family

        corpus = JSON.parse(
            read(joinpath(LFM_CHAT_FIXDIR, "$stem.json"), String);
            dicttype=Gesso.Inference._OrderedDict,
        )
        @test length(corpus["cases"]) >= (stem == "lfm2" ? 4 : 7)
        for case in corpus["cases"]
            input = case["input"]
            rendered = Gesso.Inference._render_lfm_chat(
                template_str,
                "<|startoftext|>",
                input["messages"];
                tools=get(input, "tools", nothing),
                add_generation_prompt=get(input, "add_generation_prompt", true),
                preserve_thinking=get(input, "preserve_thinking", false),
            )
            @test rendered == case["expected"]
        end

        if isempty(path) || !isfile(path)
            @test_skip "set $env to the local model GGUF"
        else
            tokenizer = Gesso.Inference.load_gguf_tokenizer(path)
            @test tokenizer.chat_template == template_str
            @test Gesso.Inference.chat_family(tokenizer.chat_template) === family

            for case in corpus["cases"]
                input = case["input"]
                prompt = Gesso.Inference.render_chat(
                    tokenizer,
                    input["messages"];
                    tools=get(input, "tools", nothing),
                    add_generation_prompt=get(input, "add_generation_prompt", true),
                    preserve_thinking=get(input, "preserve_thinking", false),
                )
                @test prompt == case["expected"]
                ids = Gesso.encode(tokenizer, prompt; parse_special=true)
                @test ids == Int.(case["tokens"])
                @test !isempty(ids)
                @test first(ids) == tokenizer.bos_token_id
                @test count(==(tokenizer.bos_token_id), ids) == 1
            end
        end
    end
end
