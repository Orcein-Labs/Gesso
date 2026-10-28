# Chat server, engine side: the stop token is hidden from the client and not
# counted, the turn-end ids come from the tokenizer, and request objects keep the
# client's key order into the chat template. Helpers come from test_serve.jl.
# The real-model path needs GESSO_ORNITH35_GGUF and is one named skip without it.

using Gesso.Serve: _run_generation, _engine_eos_ids, _parse_body, _normalize_messages

# A GGUFBPE with only the fields the engine binding reads. Ids are as given.
function sv_bpe(;
    eos::Union{Nothing, Int}=nothing,
    vocab::Dict{String, Int}=Dict{String, Int}(),
    template::Union{Nothing, String}="{{ template }}",
)
    return Gesso.Inference.GGUFBPE(
        String[],
        Int32[],
        vocab,
        Int[],
        Dict{UInt64, Int}(),
        Int[],
        Int[],
        Int[],
        Gesso.Inference._GGUFSpecial[],
        nothing,
        eos,
        nothing,
        template,
    )
end

# An engine whose generate runs the same stop rule as reference_generate: the
# stop token is sampled, reported to on_token, and ends the loop.
function sv_engine(script::Vector{Int}; eos::Vector{Int}, cached::Integer=2)
    return (prompt_ids; max_new_tokens, on_token) ->
        _run_generation(prompt_ids, eos; on_token=on_token) do wrapped
            ids = copy(prompt_ids)
            for id in script
                length(ids) - length(prompt_ids) >= max_new_tokens && break
                push!(ids, id)
                keep = wrapped(id)
                (keep === false || id in eos) && break
            end
            return (ids, cached)
        end
end

@testset "engine: the stop token is not forwarded and not counted" begin
    seen = Int[]
    run = wrapped -> begin
        ids = Int[1, 2, 3]
        for id in (10, 11, 9, 12)
            push!(ids, id)
            wrapped(id) === false && break
            id == 9 && break
        end
        return (ids, 2)
    end
    out = _run_generation(run, [1, 2, 3], [9]; on_token=id -> (push!(seen, id); true))
    @test out.generated == [10, 11]
    @test out.cached == 2
    @test seen == [10, 11]
end

@testset "engine: a sampler that stops without the stop token keeps everything" begin
    run = wrapped -> begin
        ids = Int[5]
        for id in (10, 11, 12)
            push!(ids, id)
            wrapped(id) === false && break
        end
        return (ids, 0)
    end
    out = _run_generation(run, [5], [9]; on_token=id -> id != 11)
    @test out.generated == [10, 11]
    @test out.cached == 0
end

@testset "engine: a stop token that is the only output leaves no completion" begin
    run = wrapped -> (wrapped(9); ([1, 9], 1))
    out = _run_generation(run, [1], [9]; on_token=_ -> true)
    @test isempty(out.generated)
    @test out.cached == 1
end

@testset "engine: turn-end ids come from the tokenizer" begin
    tk = sv_bpe(eos=3, vocab=Dict("<|im_end|>" => 7, "<|endoftext|>" => 3))
    @test _engine_eos_ids(tk) == [3, 7]
    @test _engine_eos_ids(sv_bpe(eos=5)) == [5]
    @test _engine_eos_ids(sv_bpe(eos=nothing, vocab=Dict("<|im_end|>" => 4))) == [4]
    @test isempty(_engine_eos_ids(sv_bpe()))
end

@testset "engine: a tokenizer with no chat template is refused before any weights" begin
    tk = sv_bpe(eos=3, template=nothing)
    @test_throws ArgumentError gguf_chat_model(
        nothing,
        nothing,
        nothing,
        tk;
        backend=nothing,
        n_ctx=64,
        model_id="x",
    )
end

@testset "engine: the stop token never reaches the client over HTTP" begin
    eos = [300]
    text = "think\n</think>\n\nanswer"
    script = Int[Int(b) for b in codeunits(text)]
    push!(script, 300)
    engine = ChatModel(
        tokenize=s -> Int.(codeunits(s)),
        detokenize=ids -> String(UInt8.(filter(<(256), ids))),
        render=(msgs, tools, thinking) -> begin
            body = join((string(m["content"]) for m in msgs), "\n")
            thinking ? body * "\n<think>\n" : body
        end,
        generate=sv_engine(script; eos=eos, cached=2),
        n_ctx=4096,
        model_id="gesso-eos",
    )
    srv = serve(engine; port=0, block=false)
    try
        r = sv_request(srv.port, sv_post("/v1/chat/completions", sv_chat(SV_USER)))
        body = JSON.parse(r.body)
        @test r.status == 200
        @test body["choices"][1]["message"]["content"] == "answer"
        @test body["choices"][1]["finish_reason"] == "stop"
        @test body["usage"]["completion_tokens"] == ncodeunits(text)
        @test !occursin("<|", body["choices"][1]["message"]["content"])
        @test body["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
    finally
        close(srv)
    end
end

const SV_ORDERED_BODY = """
{"messages":[{"role":"user","content":"hi"},
  {"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function",
    "function":{"name":"f","arguments":"{\\"zeta\\":\\"1\\",\\"alpha\\":\\"2\\"}"}}]},
  {"role":"tool","tool_call_id":"c1","content":"ok"}],
 "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object",
   "properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"},"mid":{"type":"boolean"}}}}}]}
"""

@testset "requests keep client key order into the chat template" begin
    body = _parse_body(Vector{UInt8}(codeunits(SV_ORDERED_BODY)))
    props = body["tools"][1]["function"]["parameters"]["properties"]
    @test [k for (k, _) in props] == ["zeta", "alpha", "mid"]

    msgs = _normalize_messages(body["messages"])
    rendered = Gesso.Inference.render_chat(msgs; tools=collect(Any, body["tools"]))
    @test findfirst("zeta", rendered) !== nothing
    @test findfirst("zeta", rendered)[1] <
          findfirst("alpha", rendered)[1] <
          findfirst("mid", rendered)[1]
    @test findfirst("<parameter=zeta>", rendered)[1] <
          findfirst("<parameter=alpha>", rendered)[1]
end

@testset "gguf tokenizer: Ornith turn-end ids (GESSO_ORNITH35_GGUF)" begin
    path = get(ENV, "GESSO_ORNITH35_GGUF", "")
    if !isfile(path)
        @info "skip Ornith turn-end ids: set GESSO_ORNITH35_GGUF to the Ornith 35B GGUF"
    else
        tk = Gesso.Inference.load_gguf_tokenizer(path)
        @test Set(_engine_eos_ids(tk)) == Set([248046, 248044])
    end
end

# Byte-level tokens with no merges, plus the two Qwen turn markers as control
# tokens. Enough for the template's system framing, which is all prefix_len reads.
function sv_qwen_bpe()
    chars = [string(Gesso.Inference._BYTES_TO_UNICODE[UInt8(b)]) for b in 0:255]
    marks = ["<|im_start|>", "<|im_end|>"]
    tokens = vcat(chars, marks)
    types = vcat(fill(Int32(1), 256), fill(Int32(3), 2))
    vocab = Dict(t => i - 1 for (i, t) in enumerate(tokens))
    specials = [Gesso.Inference._GGUFSpecial(marks[k], 255 + k, true) for k in 1:2]
    return Gesso.Inference.GGUFBPE(
        tokens,
        types,
        vocab,
        collect(0:255),
        Dict{UInt64, Int}(),
        Int[],
        Int[],
        Int[],
        specials,
        nothing,
        257,
        nothing,
        read(joinpath(@__DIR__, "fixtures", "qwen35_chat", "ornith_chat_template.jinja"), String),
    )
end

@testset "engine: prefix_len is the lone system rendering when the prompt starts with it" begin
    tk = sv_qwen_bpe()
    cm = gguf_chat_model(
        nothing,
        nothing,
        (; rms_norm_eps=1e-6),
        tk;
        backend=nothing,
        n_ctx=4096,
        model_id="x",
    )
    sys = Dict{String, Any}("role" => "system", "content" => "be terse")
    usr = Dict{String, Any}("role" => "user", "content" => "hi")
    render(msgs) = Gesso.Inference.render_chat(
        msgs;
        tools=Any[],
        add_generation_prompt=true,
        enable_thinking=true,
    )
    ids = Gesso.Inference.encode(tk, render([sys, usr]); parse_special=true)
    head = Gesso.Inference.encode(
        tk,
        Gesso.Inference.render_chat(
            [sys];
            tools=Any[],
            add_generation_prompt=false,
            enable_thinking=true,
        );
        parse_special=true,
    )

    @test cm.prefix_len([sys, usr], Any[], true, ids) == length(head)
    @test length(head) > 0
    @test ids[1:length(head)] == head

    # a system prompt that differs from the shared one is not a boundary
    other = Dict{String, Any}("role" => "system", "content" => "be verbose")
    @test cm.prefix_len([other, usr], Any[], true, ids) == 0

    # no system message, or a prompt that is only the system message: no boundary
    usr_ids = Gesso.Inference.encode(tk, render([usr]); parse_special=true)
    @test cm.prefix_len([usr], Any[], true, usr_ids) == 0
    sys_only = Gesso.Inference.encode(
        tk,
        Gesso.Inference.render_chat(
            [sys];
            tools=Any[],
            add_generation_prompt=false,
            enable_thinking=true,
        );
        parse_special=true,
    )
    @test cm.prefix_len([sys], Any[], true, sys_only) == 0
end

# A GGUFBPE for the LFM2 renderer: BOS and the end-of-turn token are the only
# vocabulary entries, and the end-of-turn token is the EOS.
function sv_lfm_bpe(template::String)
    return Gesso.Inference.GGUFBPE(
        String["<|startoftext|>", "<|im_end|>"],
        Int32[3, 3],
        Dict("<|startoftext|>" => 0, "<|im_end|>" => 1),
        Int[],
        Dict{UInt64, Int}(),
        Int[],
        Int[],
        Int[],
        Gesso.Inference._GGUFSpecial[],
        0,
        1,
        nothing,
        template,
        false,
        :lfm2,
    )
end

@testset "engine: the chat family picks the render; LFM prompts have no think block" begin
    msgs = Any[Dict{String, Any}("role" => "user", "content" => "Say hello in one word.")]
    lfm_template = read(joinpath(@__DIR__, "fixtures", "lfm_chat", "lfm2.jinja"), String)
    lfm = gguf_chat_model(
        nothing, nothing, (; rms_norm_eps=1e-6), sv_lfm_bpe(lfm_template);
        backend=nothing, n_ctx=4096, model_id="lfm2",
    )
    prompt = lfm.render(msgs, Any[], true)
    @test prompt == Gesso.Inference.render_chat(
        sv_lfm_bpe(lfm_template), msgs; tools=Any[], add_generation_prompt=true,
    )
    @test startswith(prompt, "<|startoftext|>")
    @test endswith(prompt, "<|im_start|>assistant\n")
    @test !occursin("<think>", prompt)
    @test !lfm.accepts_tools

    qwen = gguf_chat_model(
        nothing, nothing, (; rms_norm_eps=1e-6), sv_qwen_bpe();
        backend=nothing, n_ctx=4096, model_id="x",
    )
    @test qwen.accepts_tools
    @test endswith(qwen.render(msgs, Any[], true), "<think>\n")
    @test endswith(qwen.render(msgs, Any[], false), "<think>\n\n</think>\n\n")
end

@testset "engine: an unsupported chat template is an ArgumentError" begin
    tk = sv_bpe(eos=3, template="{{ bananas }}")
    @test_throws ArgumentError gguf_chat_model(
        nothing, nothing, (; rms_norm_eps=1e-6), tk;
        backend=nothing, n_ctx=64, model_id="x",
    )
end

@testset "engine: turn markup without a thought channel is still refused" begin
    tk = sv_bpe(eos=3, template="<|turn>user\n{{ m }}<turn|>")
    @test_throws ArgumentError gguf_chat_model(
        nothing, nothing, (; rms_norm_eps=1e-6), tk;
        backend=nothing, n_ctx=64, model_id="x",
    )
end

@testset "engine: a made-up chat family is an ArgumentError" begin
    tk = sv_bpe(eos=3, template="<|madeup_start|>{{ m }}<|madeup_end|>")
    @test_throws ArgumentError gguf_chat_model(
        nothing, nothing, (; rms_norm_eps=1e-6), tk;
        backend=nothing, n_ctx=64, model_id="x",
    )
end

# A GGUFGemma4BPE with only the fields the engine binding reads.
function sv_gemma_bpe(; eos::Union{Nothing, Int}=nothing, vocab::Dict{String, Int}=Dict{String, Int}(), template::String)
    return Gesso.Inference.GGUFGemma4BPE(
        String[],
        Int32[],
        vocab,
        Int[],
        Dict{Tuple{String, String}, Int}(),
        Gesso.Inference._GGUFSpecial[],
        nothing,
        eos,
        nothing,
        template,
    )
end

@testset "engine: a gemma4 template is served with its own renderer and turn end" begin
    tmpl = "<|turn>system\n{{ s }}<turn|><|turn>user\n{{ m }}<turn|><|channel>thought\n"
    tk = sv_gemma_bpe(eos=1, vocab=Dict("<turn|>" => 106), template=tmpl)
    cm = gguf_chat_model(
        nothing, nothing, (; rms_norm_eps=1e-6), tk;
        backend=nothing, n_ctx=64, model_id="g4",
    )
    @test cm.family === :gemma4
    @test cm.accepts_tools
    @test 106 in _engine_eos_ids(tk)
    msgs = Dict{String, Any}[Dict{String, Any}("role" => "user", "content" => "hi")]
    on = cm.render(msgs, Any[], true)
    off = cm.render(msgs, Any[], false)
    @test startswith(on, "<bos>")
    @test endswith(on, "<|turn>model\n")
    @test endswith(off, "<channel|>")
    @test on == Gesso.Inference.render_gemma4_chat(msgs; add_generation_prompt=true, enable_thinking=true)
end
