# Chat server, engine side: the stop token is hidden from the client and not
# counted, the turn-end ids come from the tokenizer, and request objects keep the
# client's key order into the chat template. Helpers come from test_serve.jl.
# The real-model path needs GESSO_ORNITH35_GGUF and is one named skip without it.

using Gesso.Serve: _run_generation, _engine_eos_ids, _parse_body, _normalize_messages

# A GGUFBPE with only the fields the engine binding reads. Ids are as given.
function sv_bpe(; eos::Union{Nothing,Int}=nothing, vocab::Dict{String,Int}=Dict{String,Int}(),
        template::Union{Nothing,String}="{{ template }}")
    return Gesso.Inference.GGUFBPE(String[], Int32[], vocab, Int[], Dict{UInt64,Int}(),
        Int[], Int[], Int[], Gesso.Inference._GGUFSpecial[], nothing, eos, nothing, template)
end

# An engine whose generate runs the same stop rule as reference_generate: the
# stop token is sampled, reported to on_token, and ends the loop.
function sv_engine(script::Vector{Int}; eos::Vector{Int}, cached::Integer=2)
    return (prompt_ids; max_new_tokens, on_token) -> _run_generation(prompt_ids, eos;
            on_token=on_token) do wrapped
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
    @test_throws ArgumentError gguf_chat_model(nothing, nothing, nothing, tk;
        backend=nothing, n_ctx=64, model_id="x")
end

@testset "engine: the stop token never reaches the client over HTTP" begin
    eos = [300]
    text = "think\n</think>\n\nanswer"
    script = Int[Int(b) for b in codeunits(text)]
    push!(script, 300)
    engine = ChatModel(
        tokenize=s -> Int.(codeunits(s)),
        detokenize=ids -> String(UInt8.(filter(<(256), ids))),
        render=(msgs, tools, thinking) -> join((string(m["content"]) for m in msgs), "\n"),
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
    @test findfirst("zeta", rendered)[1] < findfirst("alpha", rendered)[1] < findfirst("mid", rendered)[1]
    @test findfirst("<parameter=zeta>", rendered)[1] < findfirst("<parameter=alpha>", rendered)[1]
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
