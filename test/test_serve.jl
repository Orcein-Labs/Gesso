# Chat server tests: gate 2 (the Qwen3.5 XML parser, every split point) and
# gate 3 (HTTP behaviour on an ephemeral port, with fake engines). Helpers carry
# an `sv_` prefix because every test file shares one module.

using Sockets
using JSON
using Random
using Gesso
using Gesso.Serve: QwenXMLParser, feed!, ReasoningDelta, ContentDelta, ToolCallDelta, opens_reasoning

# ---------------------------------------------------------------------------
# Gate 2: parser

const SV_WEATHER = Any[Dict{String, Any}(
    "type" => "function",
    "function" => Dict{String, Any}(
        "name" => "get_weather",
        "parameters" => Dict{String, Any}(
            "type" => "object",
            "properties" => Dict{String, Any}(
                "city" => Dict{String, Any}("type" => "string"),
                "days" => Dict{String, Any}("type" => "integer"),
                "metric" => Dict{String, Any}("type" => "boolean"),
                "ratio" => Dict{String, Any}("type" => "number"),
                "tags" => Dict{String, Any}("type" => "array"),
                "opts" => Dict{String, Any}("type" => "object"),
                "note" => Dict{String, Any}("type" => Any["string", "null"]),
            ),
        ),
    ),
)]

sv_think(r::AbstractString) = r * "\n</think>\n\n"
sv_r(s::AbstractString) = (:reasoning, String(s))
sv_c(s::AbstractString) = (:content, String(s))
sv_t(i::Int, name::AbstractString, args::AbstractDict) =
    (:call, i, String(name), Dict{String, Any}(args))

function sv_call(name::AbstractString, params::Vector{Pair{String, String}})
    s = "<tool_call>\n<function=$name>\n"
    for (k, v) in params
        s *= "<parameter=$k>\n$v\n</parameter>\n"
    end
    return s * "</function>\n</tool_call>"
end

# Each case: the raw output text, the request's tools, stop strings, whether
# thinking is on, and the events it must produce (merged per kind, in order).
function sv_parser_cases()
    cases = NamedTuple[]
    add!(text, expected; tools=Any[], stop=String[], thinking=true) = push!(
        cases,
        (text=text, tools=tools, stop=stop, thinking=thinking, expected=expected),
    )

    reasoning = [
        "Need the city.",
        "Check: 2 < 3 & 4 > 1.",
        "naïve café — ✓",
        "<b>not a tag</b>",
        "line one\nline two",
        "think about </thin then </thing",
        "a <tool_ and <tool_call is not yet",
        "x",
    ]
    contents = ["Paris.", "Sure — done ✓", "ok"]
    for r in reasoning, c in contents
        add!(sv_think(r) * c, [sv_r(r * "\n"), sv_c(c)])
    end

    W = SV_WEATHER
    add!(
        sv_think("plan") *
        "Let me check.\n\n" *
        sv_call("get_weather", ["city" => "Paris", "days" => "3"]) *
        "\nDone.",
        [
            sv_r("plan\n"),
            sv_c("Let me check.\n\n"),
            sv_t(0, "get_weather", Dict("city" => "Paris", "days" => 3)),
            sv_c("Done."),
        ];
        tools=W,
    )
    tool_params = [
        (["city" => "Oslo", "metric" => "true"], Dict("city" => "Oslo", "metric" => true)),
        (["ratio" => "0.25"], Dict("ratio" => 0.25)),
        (["ratio" => "7"], Dict("ratio" => 7)),
        (["tags" => "[\"a\", \"b\"]"], Dict("tags" => Any["a", "b"])),
        (["opts" => "{\"x\": 1}"], Dict("opts" => Dict("x" => 1))),
        (["note" => "line1\nline2"], Dict("note" => "line1\nline2")),
        (["days" => "abc"], Dict("days" => "abc")),
        (["extra" => "v"], Dict("extra" => "v")),
        (Pair{String, String}[], Dict{String, Any}()),
    ]
    for (params, args) in tool_params
        add!(
            sv_think("plan") * sv_call("get_weather", params),
            [sv_r("plan\n"), sv_t(0, "get_weather", args)];
            tools=W,
        )
    end
    add!(
        sv_think("plan") * sv_call("other", ["q" => "x"]),
        [sv_r("plan\n"), sv_t(0, "other", Dict("q" => "x"))];
        tools=W,
    )
    add!(
        sv_think("plan") *
        sv_call("get_weather", ["city" => "Paris"]) *
        "\n" *
        sv_call("get_weather", ["city" => "Rome"]),
        [
            sv_r("plan\n"),
            sv_t(0, "get_weather", Dict("city" => "Paris")),
            sv_t(1, "get_weather", Dict("city" => "Rome")),
        ];
        tools=W,
    )
    add!(
        sv_think("plan") * "café → " * sv_call("get_weather", ["city" => "Lima"]),
        [sv_r("plan\n"), sv_c("café → "), sv_t(0, "get_weather", Dict("city" => "Lima"))];
        tools=W,
    )

    add!("Hi there", [sv_c("Hi there")]; thinking=false)
    add!(
        "Sure.\n" * sv_call("get_weather", ["city" => "Bern"]),
        [sv_c("Sure.\n"), sv_t(0, "get_weather", Dict("city" => "Bern"))];
        tools=W,
        thinking=false,
    )

    S = ["<|im_end|>"]
    add!(
        sv_think("plan") * "Hello<|im_end|>tail ignored",
        [sv_r("plan\n"), sv_c("Hello")];
        stop=S,
    )
    add!("plan<|im_end|>x", [sv_r("plan")]; stop=S)
    add!(sv_think("plan") * "A <|im_end|>", [sv_r("plan\n"), sv_c("A ")]; stop=S)

    add!(
        sv_think("plan") * "x </thx y </tool_cal z <tool_",
        [sv_r("plan\n"), sv_c("x </thx y </tool_cal z <tool_")],
    )
    add!("plan</thin", [sv_r("plan</thin")])
    add!(
        sv_think("plan") *
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
        [
            sv_r("plan\n"),
            sv_c(
                "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
            ),
        ],
    )
    return cases
end

# Adjacent text of one kind is joined, so the sequence does not depend on how
# the bytes were chunked.
function sv_merge(events)
    items = Any[]
    for e in events
        if e isa ReasoningDelta || e isa ContentDelta
            kind = e isa ReasoningDelta ? :reasoning : :content
            if !isempty(items) && items[end][1] === kind
                items[end] = (kind, items[end][2] * e.text)
            else
                push!(items, (kind, e.text))
            end
        elseif e isa ToolCallDelta
            push!(items, (:call, e.index, e.name, JSON.parse(e.arguments)))
        else
            error("unexpected parser event $(typeof(e))")
        end
    end
    return items
end

function sv_run(case, chunks::Vector{Vector{UInt8}})
    p = QwenXMLParser(; tools=copy(case.tools), thinking=case.thinking, stop=case.stop)
    events = Any[]
    for c in chunks
        append!(events, feed!(p, c))
    end
    append!(events, Gesso.Serve.finish!(p))
    return sv_merge(events)
end

# Every two-way split, one byte per chunk, and seeded random partitions.
function sv_chunkings(bytes::Vector{UInt8}, rng::AbstractRNG)
    out = Vector{Vector{UInt8}}[]
    for i in 0:length(bytes)
        push!(out, [bytes[1:i], bytes[(i+1):end]])
    end
    push!(out, [UInt8[b] for b in bytes])
    for _ in 1:25
        chunks = Vector{UInt8}[]
        pos = 1
        while pos <= length(bytes)
            k = rand(rng, 1:9)
            push!(chunks, bytes[pos:min(pos+k-1, end)])
            pos += k
        end
        push!(out, chunks)
    end
    return out
end

@testset "parser: scripted outputs at every split point" begin
    cases = sv_parser_cases()
    @test length(cases) >= 30
    rng = MersenneTwister(20261011)
    for (i, case) in enumerate(cases)
        @testset "case $i" begin
            bytes = Vector{UInt8}(codeunits(case.text))
            @test sv_run(case, [bytes]) == case.expected
            chunkings = sv_chunkings(bytes, rng)
            two_way = chunkings[1:(end-26)]
            per_byte = chunkings[end-25]
            random_cuts = chunkings[(end-24):end]
            @test all(sv_run(case, ch) == case.expected for ch in two_way)
            @test sv_run(case, per_byte) == case.expected
            @test all(sv_run(case, ch) == case.expected for ch in random_cuts)
        end
    end
end

# ---------------------------------------------------------------------------
# Gate 3: HTTP on an ephemeral port, with fake engines

const SV_THINK = "think about it\n</think>\n\n" * repeat("word ", 40)

# Tokens are bytes, so detokenize is exact for these ASCII scripts. `log` gets
# the ids each generate call emitted, `peak` the most calls seen at once.
function sv_fake(;
    script::AbstractString=SV_THINK,
    accepts_tools::Bool=true,
    n_ctx::Integer=4096,
    delay::Real=0.0,
    cached::Integer=2,
    log=Vector{Vector{Int}}(),
    active=Ref(0),
    peak=Ref(0),
)
    return ChatModel(
        tokenize=s -> Int.(codeunits(s)),
        detokenize=ids -> String(UInt8.(ids)),
        accepts_tools=accepts_tools,
        render=(msgs, tools, thinking) -> begin
            body = join((string(m["content"]) for m in msgs), "\n")
            thinking ? body * "\n<think>\n" : body
        end,
        generate=(prompt_ids; max_new_tokens, on_token) -> begin
            active[] += 1
            peak[] = max(peak[], active[])
            try
                out = Int[]
                for b in codeunits(script)
                    length(out) >= max_new_tokens && break
                    push!(out, Int(b))
                    delay > 0 && sleep(delay)
                    on_token(Int(b)) === false && break
                end
                push!(log, out)
                return (generated=out, cached=cached)
            finally
                active[] -= 1
            end
        end,
        n_ctx=n_ctx,
        model_id="gesso-test",
    )
end

sv_post(path::AbstractString, body::AbstractString) =
    "POST $path HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n" *
    "Content-Length: $(ncodeunits(body))\r\n\r\n" *
    body
sv_get(path::AbstractString) = "GET $path HTTP/1.1\r\nHost: test\r\n\r\n"
sv_chat(messages; extra...) = JSON.json(
    merge(
        Dict{String, Any}("messages" => messages),
        Dict{String, Any}(String(k) => v for (k, v) in extra),
    ),
)
const SV_USER = [Dict("role" => "user", "content" => "hi")]

function sv_read_head(sock::TCPSocket)
    status = parse(Int, split(strip(readline(sock; keep=false)))[2])
    headers = Dict{String, String}()
    while true
        line = strip(readline(sock; keep=false))
        isempty(line) && break
        k, v = split(line, ':'; limit=2)
        headers[lowercase(strip(k))] = strip(v)
    end
    return status, headers
end

function sv_read_chunks(sock::TCPSocket)
    out = UInt8[]
    while true
        n = parse(Int, strip(readline(sock; keep=false)); base=16)
        if n == 0
            while !isempty(strip(readline(sock; keep=false)))
            end
            return out
        end
        append!(out, read(sock, n))
        readline(sock; keep=false)
    end
end

function sv_read_response(sock::TCPSocket)
    status, headers = sv_read_head(sock)
    body = if get(headers, "transfer-encoding", "") == "chunked"
        sv_read_chunks(sock)
    elseif haskey(headers, "content-length")
        read(sock, parse(Int, headers["content-length"]))
    else
        UInt8[]
    end
    return (status=status, headers=headers, body=String(body))
end

function sv_request(port::Integer, raw::AbstractString)
    sock = Sockets.connect(Sockets.localhost, port)
    try
        write(sock, raw)
        return sv_read_response(sock)
    finally
        close(sock)
    end
end

@testset "stream: SSE framing, usage chunk, [DONE]" begin
    srv = serve(sv_fake(); port=0, block=false)
    try
        body = sv_chat(SV_USER; stream=true, stream_options=Dict("include_usage" => true))
        resp = sv_request(srv.port, sv_post("/v1/chat/completions", body))
        @test resp.status == 200
        @test resp.headers["content-type"] == "text/event-stream"
        @test resp.headers["transfer-encoding"] == "chunked"
        blocks = split(resp.body, "\n\n"; keepempty=false)
        @test all(startswith(b, "data: ") for b in blocks)
        @test blocks[end] == "data: [DONE]"
        events = [JSON.parse(b[7:end]) for b in blocks[1:(end-1)]]
        deltas = [c["choices"][1]["delta"] for c in events if !isempty(c["choices"])]
        @test join(
            d["reasoning_content"] for d in deltas if haskey(d, "reasoning_content")
        ) == "think about it\n"
        @test join(d["content"] for d in deltas if haskey(d, "content")) ==
              repeat("word ", 40)
        finals =
            [c["choices"][1]["finish_reason"] for c in events if !isempty(c["choices"])]
        @test finals[end] == "stop"
        usage = events[end]
        @test isempty(usage["choices"])
        @test usage["usage"]["prompt_tokens"] == ncodeunits("hi\n<think>\n")
        @test usage["usage"]["completion_tokens"] == ncodeunits(SV_THINK)
        @test usage["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
    finally
        close(srv)
    end
end

@testset "non-stream: message, reasoning, usage" begin
    srv = serve(sv_fake(); port=0, block=false)
    try
        resp = sv_request(srv.port, sv_post("/v1/chat/completions", sv_chat(SV_USER)))
        @test resp.status == 200
        out = JSON.parse(resp.body)
        choice = out["choices"][1]
        @test choice["message"]["content"] == repeat("word ", 40)
        @test choice["message"]["reasoning_content"] == "think about it\n"
        @test choice["finish_reason"] == "stop"
        @test out["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
    finally
        close(srv)
    end
end

@testset "non-stream tool call: finish_reason and tool_calls" begin
    script = sv_think("plan") * sv_call("get_weather", ["city" => "Paris"])
    srv = serve(sv_fake(script=script); port=0, block=false)
    try
        resp = sv_request(
            srv.port,
            sv_post("/v1/chat/completions", sv_chat(SV_USER; tools=SV_WEATHER)),
        )
        choice = JSON.parse(resp.body)["choices"][1]
        @test choice["finish_reason"] == "tool_calls"
        @test choice["message"]["content"] === nothing
        call = choice["message"]["tool_calls"][1]
        @test call["function"]["name"] == "get_weather"
        @test JSON.parse(call["function"]["arguments"]) == Dict("city" => "Paris")
    finally
        close(srv)
    end
end

@testset "keep-alive: several requests on one socket" begin
    srv = serve(sv_fake(); port=0, block=false)
    try
        sock = Sockets.connect(Sockets.localhost, srv.port)
        try
            write(sock, sv_get("/health"))
            r1 = sv_read_response(sock)
            @test r1.status == 200
            @test JSON.parse(r1.body)["status"] == "ok"
            @test get(r1.headers, "connection", "") != "close"

            write(sock, sv_post("/v1/chat/completions", sv_chat(SV_USER)))
            r2 = sv_read_response(sock)
            @test r2.status == 200
            @test JSON.parse(r2.body)["choices"][1]["finish_reason"] == "stop"

            write(sock, sv_get("/props"))
            r3 = sv_read_response(sock)
            @test JSON.parse(r3.body)["default_generation_settings"]["n_ctx"] == 4096

            write(sock, sv_get("/v1/models"))
            r4 = sv_read_response(sock)
            @test JSON.parse(r4.body)["data"][1]["meta"]["n_ctx"] == 4096
        finally
            close(sock)
        end
    finally
        close(srv)
    end
end

@testset "chunked request body" begin
    srv = serve(sv_fake(); port=0, block=false)
    try
        raw_body = sv_chat(SV_USER)
        cut = 10
        pieces = [raw_body[1:cut], raw_body[(cut+1):end]]
        head = "POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\nTransfer-Encoding: chunked\r\n\r\n"
        payload =
            head *
            join(
                string(string(ncodeunits(p), base=16), "\r\n", p, "\r\n") for p in pieces
            ) *
            "0\r\n\r\n"
        resp = sv_request(srv.port, payload)
        @test resp.status == 200
        @test JSON.parse(resp.body)["choices"][1]["message"]["content"] ==
              repeat("word ", 40)
    finally
        close(srv)
    end
end

@testset "errors: 400 and 404 keep the connection usable" begin
    srv = serve(sv_fake(); port=0, block=false)
    try
        sock = Sockets.connect(Sockets.localhost, srv.port)
        try
            write(sock, sv_post("/v1/chat/completions", "{not json"))
            @test sv_read_response(sock).status == 400
            write(sock, sv_post("/v1/chat/completions", JSON.json(Dict("stream" => false))))
            @test sv_read_response(sock).status == 400
            write(sock, sv_post("/v1/chat/completions", sv_chat(SV_USER; stream="yes")))
            @test sv_read_response(sock).status == 400
            write(sock, sv_get("/nope"))
            @test sv_read_response(sock).status == 404
            write(sock, sv_get("/health"))
            @test sv_read_response(sock).status == 200
        finally
            close(sock)
        end
    finally
        close(srv)
    end
end

@testset "prompt longer than the context is a 400" begin
    srv = serve(sv_fake(n_ctx=8); port=0, block=false)
    try
        body = sv_chat([Dict("role" => "user", "content" => "hello world!!")])
        r = sv_request(srv.port, sv_post("/v1/chat/completions", body))
        @test r.status == 400
        err = JSON.parse(r.body)["error"]
        @test err["code"] == "context_length_exceeded"
        @test err["type"] == "invalid_request_error"
        @test err["param"] == "messages"
        m = match(
            r"maximum context length is 8 tokens\. However, your messages resulted in (\d+) tokens",
            err["message"],
        )
        @test m !== nothing && parse(Int, m[1]) >= 8
    finally
        close(srv)
    end
end

@testset "other 400s keep the numeric code and no param" begin
    srv = serve(sv_fake(); port=0, block=false)
    try
        r = sv_request(srv.port, sv_post("/v1/chat/completions", sv_chat(Any[])))
        @test r.status == 400
        err = JSON.parse(r.body)["error"]
        @test err["code"] == 400
        @test !haskey(err, "param")
    finally
        close(srv)
    end
end

@testset "client disconnect cancels generation; the next request succeeds" begin
    log = Vector{Vector{Int}}()
    srv = serve(sv_fake(delay=0.01, log=log); port=0, block=false)
    try
        sock = Sockets.connect(Sockets.localhost, srv.port)
        write(sock, sv_post("/v1/chat/completions", sv_chat(SV_USER; stream=true)))
        status, _ = sv_read_head(sock)
        @test status == 200
        readline(sock; keep=false)     # size line of the role chunk: the stream is open
        close(sock)

        r = sv_request(
            srv.port,
            sv_post("/v1/chat/completions", sv_chat(SV_USER; max_tokens=5)),
        )
        @test r.status == 200
        @test JSON.parse(r.body)["usage"]["completion_tokens"] == 5
        @test length(log) == 2
        @test length(log[1]) < ncodeunits(SV_THINK)
    finally
        close(srv)
    end
end

@testset "two concurrent clients are served in turn" begin
    active = Ref(0)
    peak = Ref(0)
    reply = repeat("ok ", 20)
    srv = serve(
        sv_fake(
            script="think\n</think>\n\n" * reply,
            delay=0.002,
            active=active,
            peak=peak,
        );
        port=0,
        block=false,
    )
    try
        raw = sv_post("/v1/chat/completions", sv_chat(SV_USER))
        tasks = [@async sv_request(srv.port, raw) for _ in 1:2]
        results = fetch.(tasks)
        @test all(r.status == 200 for r in results)
        @test all(
            JSON.parse(r.body)["choices"][1]["message"]["content"] == reply for r in results
        )
        @test peak[] == 1
    finally
        close(srv)
    end
end

# An engine that computes without yielding, like a long CPU prefill, until the
# probe has its /health answer. /health must still answer while it runs, which
# needs generation off thread 1. The spin waits on the probe rather than a fixed
# time so a slow start under a loaded suite cannot end it before the probe.
@testset "health answers while a generation computes" begin
    if Gesso.Serve._engine_tid() == 1
        @test_skip "needs a default thread other than this one"
    else
        busy = Threads.Atomic{Bool}(false)
        probed = Threads.Atomic{Bool}(false)
        base = sv_fake(script="ok")
        spin = ChatModel(
            tokenize=base.tokenize,
            detokenize=base.detokenize,
            accepts_tools=base.accepts_tools,
            render=base.render,
            generate=(prompt_ids; max_new_tokens, on_token) -> begin
                busy[] = true
                t0 = time_ns()
                while !probed[] && time_ns() - t0 < 60_000_000_000
                    GC.safepoint()
                end
                busy[] = false
                return base.generate(prompt_ids; max_new_tokens, on_token)
            end,
            n_ctx=base.n_ctx,
            model_id=base.model_id,
        )
        srv = on_engine_thread(() -> serve(spin; port=0, block=false))
        try
            chat = Threads.@spawn :default sv_request(
                srv.port,
                sv_post("/v1/chat/completions", sv_chat(SV_USER)),
            )
            probe = Threads.@spawn :default begin
                t0 = time()
                while !busy[] && time() - t0 < 60
                    sleep(0.01)
                end
                started = busy[]
                t1 = time()
                still_busy = false
                r = try
                    sv_request(srv.port, sv_get("/health"))
                finally
                    still_busy = busy[]
                    probed[] = true
                end
                (started, r.status, time() - t1, still_busy)
            end
            started, status, waited, still_busy = fetch(probe)
            @test started
            @test status == 200
            @test waited < 1.0
            @test still_busy
            @test fetch(chat).status == 200
        finally
            close(srv)
        end
    end
end

@testset "engine contract: a ChatModel without prefix_len asks for no boundary" begin
    cm = sv_fake()
    @test cm.prefix_len(
        [Dict{String, Any}("role" => "system", "content" => "s")],
        Any[],
        false,
        [1, 2, 3],
    ) == 0
    @test cm.prefix_len(Dict{String, Any}[], Any[], false, Int[]) == 0
end

# A decoder that, like GPT-2 BPE, ends an unfinished UTF-8 character with
# U+FFFD, and counts every id it is asked to decode.
function sv_counting_detok(counter::Ref{Int})
    return ids -> begin
        counter[] += length(ids)
        bytes = UInt8.(ids)
        s = String(copy(bytes))
        isvalid(s) && return s
        j = length(bytes)
        while j > 0 && !isvalid(String(bytes[1:j]))
            j -= 1
        end
        return String(bytes[1:j]) * "�"
    end
end

@testset "decode cost stays linear in the reply, multi-byte text intact" begin
    reply = repeat("mot é 日本 ", 300)
    script = "think\n</think>\n\n" * reply
    counter = Ref(0)
    base = sv_fake(; script=script, n_ctx=1 << 16)
    cm = ChatModel(;
        tokenize=base.tokenize,
        detokenize=sv_counting_detok(counter),
        render=base.render,
        generate=base.generate,
        n_ctx=base.n_ctx,
        model_id=base.model_id,
    )
    srv = serve(cm; port=0, block=false)
    try
        body = sv_chat(SV_USER; max_tokens=1 << 15)
        resp = sv_request(srv.port, sv_post("/v1/chat/completions", body))
        @test resp.status == 200
        @test JSON.parse(resp.body)["choices"][1]["message"]["content"] == reply
        n = ncodeunits(script)
        # re-decoding every id per token would be about n^2 / 2 ids
        @test counter[] <= 8n
    finally
        close(srv)
    end
end

# Waits up to `limit` seconds for `cond()`, polling. Returns the final value.
function sv_wait_until(cond; limit::Real=10.0)
    t0 = time()
    while !cond() && time() - t0 < limit
        sleep(0.02)
    end
    return cond()
end

@testset "read deadline closes idle and trickling connections" begin
    srv = serve(sv_fake(); port=0, block=false, read_timeout=0.5)
    try
        # A connection that never sends a byte is closed by the server.
        idle = Sockets.connect(Sockets.localhost, srv.port)
        @test sv_wait_until(() -> eof(idle))
        close(idle)

        # A request head sent a few bytes at a time, never finished, is cut off
        # at the deadline instead of holding the connection open.
        slow = Sockets.connect(Sockets.localhost, srv.port)
        t0 = time()
        for b in codeunits("GET /v1/models HTTP/1.1\r\nHost: t")
            isopen(slow) || break
            try
                write(slow, b)
            catch
                break
            end
            sleep(0.05)
        end
        @test sv_wait_until(() -> eof(slow))
        @test time() - t0 < 5
        close(slow)

        # A request delivered in time still works, and keep-alive still works
        # for a second request sent before the deadline.
        sock = Sockets.connect(Sockets.localhost, srv.port)
        try
            write(sock, sv_get("/v1/models"))
            @test sv_read_response(sock).status == 200
            write(sock, sv_get("/v1/models"))
            @test sv_read_response(sock).status == 200
            # Then the idle keep-alive connection is closed by the deadline.
            @test sv_wait_until(() -> eof(sock))
        finally
            close(sock)
        end
        @test sv_wait_until(() -> srv.open_connections[] == 0)
    finally
        close(srv)
    end
end

@testset "connections past the cap get 503" begin
    srv = serve(sv_fake(); port=0, block=false, max_connections=1)
    try
        first = Sockets.connect(Sockets.localhost, srv.port)
        @test sv_wait_until(() -> srv.open_connections[] == 1)
        over = Sockets.connect(Sockets.localhost, srv.port)
        resp = sv_read_response(over)
        @test resp.status == 503
        @test resp.headers["connection"] == "close"
        close(over)
        # The held connection is still served.
        write(first, sv_get("/v1/models"))
        @test sv_read_response(first).status == 200
        close(first)
        @test sv_wait_until(() -> srv.open_connections[] == 0)
        # With the slot free, a new connection is served again.
        @test sv_request(srv.port, sv_get("/v1/models")).status == 200
    finally
        close(srv)
    end
    @test_throws ArgumentError serve(sv_fake(); port=0, block=false, read_timeout=0)
    @test_throws ArgumentError serve(sv_fake(); port=0, block=false, max_connections=0)
end

const SV_LONG = "think\n</think>\n\n" * repeat("ok ", 200)

@testset "generation queue: a full queue answers 503 with Retry-After" begin
    active = Ref(0)
    srv = serve(
        sv_fake(script=SV_LONG, delay=0.002, active=active);
        port=0,
        block=false,
        max_queue=0,
    )
    try
        busy = Sockets.connect(Sockets.localhost, srv.port)
        write(busy, sv_post("/v1/chat/completions", sv_chat(SV_USER; stream=true)))
        @test sv_wait_until(() -> active[] == 1)
        r = sv_request(srv.port, sv_post("/v1/chat/completions", sv_chat(SV_USER)))
        @test r.status == 503
        @test r.headers["retry-after"] == "1"
        @test JSON.parse(r.body)["error"]["type"] == "server_error"
        close(busy)
        @test sv_wait_until(() -> !srv.gate.busy)
    finally
        close(srv)
    end
end

@testset "generation queue: a request that waits too long gets 503 and a closed connection" begin
    @test_throws ArgumentError serve(sv_fake(); port=0, block=false, max_queue=-1)
    @test_throws ArgumentError serve(sv_fake(); port=0, block=false, queue_timeout=0)
    active = Ref(0)
    srv = serve(
        sv_fake(script=SV_LONG, delay=0.002, active=active);
        port=0,
        block=false,
        max_queue=4,
        queue_timeout=0.3,
    )
    try
        busy = Sockets.connect(Sockets.localhost, srv.port)
        write(busy, sv_post("/v1/chat/completions", sv_chat(SV_USER; stream=true)))
        @test sv_wait_until(() -> active[] == 1)
        waiter = Sockets.connect(Sockets.localhost, srv.port)
        t0 = time()
        write(waiter, sv_post("/v1/chat/completions", sv_chat(SV_USER)))
        resp = sv_read_response(waiter)
        @test time() - t0 < 5
        @test resp.status == 503
        @test resp.headers["retry-after"] == "1"
        @test resp.headers["connection"] == "close"
        @test sv_wait_until(() -> eof(waiter))
        close(waiter)
        @test srv.gate.waiting == 0
        # the generation that holds the engine is not disturbed by the timeout
        @test active[] == 1
        close(busy)
        @test sv_wait_until(() -> !srv.gate.busy)
    finally
        close(srv)
    end
end

@testset "generation queue: a client that leaves while queued frees its place" begin
    active = Ref(0)
    srv = serve(
        sv_fake(script=SV_LONG, delay=0.002, active=active);
        port=0,
        block=false,
        max_queue=1,
        queue_timeout=30,
    )
    try
        busy = Sockets.connect(Sockets.localhost, srv.port)
        write(busy, sv_post("/v1/chat/completions", sv_chat(SV_USER; stream=true)))
        @test sv_wait_until(() -> active[] == 1)

        queued = Sockets.connect(Sockets.localhost, srv.port)
        write(queued, sv_post("/v1/chat/completions", sv_chat(SV_USER)))
        @test sv_wait_until(() -> srv.gate.waiting == 1)
        # the one place is taken, so a further request is refused
        full = sv_request(srv.port, sv_post("/v1/chat/completions", sv_chat(SV_USER)))
        @test full.status == 503

        close(queued)
        @test sv_wait_until(() -> srv.gate.waiting == 0)

        # the place that the departed client held is free for the next request
        next = Sockets.connect(Sockets.localhost, srv.port)
        write(next, sv_post("/v1/chat/completions", sv_chat(SV_USER)))
        @test sv_wait_until(() -> srv.gate.waiting == 1)
        close(busy)
        resp = sv_read_response(next)
        close(next)
        @test resp.status == 200
        @test JSON.parse(resp.body)["usage"]["completion_tokens"] > 0
    finally
        close(srv)
    end
end

@testset "parser: the start state comes from the prompt" begin
    @test opens_reasoning("<|im_start|>assistant\n<think>\n")
    @test !opens_reasoning("<|im_start|>assistant\n")
    @test !opens_reasoning("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    @test !opens_reasoning("<|im_start|>assistant\n")
    function sv_parse(p, text)
        events = feed!(p, Vector{UInt8}(codeunits(text)))
        append!(events, Gesso.Serve.finish!(p))
        return sv_merge(events)
    end
    @test sv_parse(QwenXMLParser(; start_in_reasoning=true), "plan\n</think>\nanswer") ==
          [(:reasoning, "plan\n"), (:content, "answer")]
    @test sv_parse(QwenXMLParser(; start_in_reasoning=false), "answer") ==
          [(:content, "answer")]
    @test sv_parse(QwenXMLParser(; thinking=false), "answer") == [(:content, "answer")]
    @test sv_parse(QwenXMLParser(; thinking=true), "plan\n</think>\nanswer") ==
          [(:reasoning, "plan\n"), (:content, "answer")]
end

@testset "a model that opens its own think block is parsed as reasoning" begin
    function sv_bytewise(text)
        p = QwenXMLParser(; start_in_reasoning=false)
        events = Any[]
        for b in codeunits(text)
            append!(events, feed!(p, [b]))
        end
        append!(events, Gesso.Serve.finish!(p))
        return sv_merge(events)
    end
    @test sv_bytewise("<think>\nplan\n</think>\n\nParis") ==
          [(:reasoning, "plan\n"), (:content, "Paris")]
    @test sv_bytewise("\n<think>plan</think>Paris") ==
          [(:reasoning, "plan"), (:content, "Paris")]
    @test sv_bytewise("<thinking is fun") == [(:content, "<thinking is fun")]
    @test sv_bytewise("Paris <think>") == [(:content, "Paris <think>")]
    @test sv_bytewise("<thi") == [(:content, "<thi")]
end

@testset "tools on a model with no tool-call parser are a 400" begin
    srv = serve(sv_fake(; accepts_tools=false); port=0, block=false)
    try
        r = sv_request(srv.port, sv_post("/v1/chat/completions", sv_chat(SV_USER; tools=SV_WEATHER)))
        @test r.status == 400
        @test JSON.parse(r.body)["error"]["code"] == "tools_unsupported"
        r = sv_request(srv.port, sv_post("/v1/chat/completions", sv_chat(SV_USER; tools=Any[])))
        @test r.status == 200
    finally
        close(srv)
    end
end

# ---------------------------------------------------------------------------
# Gemma 4 dialect: the streaming parser and the tool-call round trip

using Gesso.Serve: Gemma4Parser, gemma4_opens_reasoning

function sv_g4_parse(text::AbstractString; start_in_reasoning::Bool=false, stop=String[], chunks=nothing)
    p = Gemma4Parser(; start_in_reasoning=start_in_reasoning, stop=stop)
    bytes = Vector{UInt8}(codeunits(text))
    events = Any[]
    for c in (chunks === nothing ? [bytes] : chunks)
        append!(events, feed!(p, c))
    end
    append!(events, Gesso.Serve.finish!(p))
    return sv_merge(events)
end

@testset "gemma4: the start state comes from the prompt" begin
    @test gemma4_opens_reasoning("<|turn>model\n<|channel>thought\n")
    @test !gemma4_opens_reasoning("<|turn>model\n<|channel>thought\n<channel|>")
    @test !gemma4_opens_reasoning("<|turn>model\n")
    @test sv_g4_parse("plan<channel|>Paris"; start_in_reasoning=true) ==
          [(:reasoning, "plan"), (:content, "Paris")]
    @test sv_g4_parse("Paris") == [(:content, "Paris")]
end

@testset "gemma4: every chunking gives the whole-text events" begin
    cases = [
        ("<|channel>thought\nplan\n<channel|>Paris", false,
         [(:reasoning, "plan\n"), (:content, "Paris")]),
        ("<|channel>thought\nplan<channel|>\n\n  Paris", false,
         [(:reasoning, "plan"), (:content, "Paris")]),
        ("<|channel>thought\n<channel|>Paris", false, [(:content, "Paris")]),
        ("Paris <|channel>x", false, [(:content, "Paris <|channel>x")]),
        ("Paris <|chan", false, [(:content, "Paris <|chan")]),
        ("café <|channel>thought\nnaïve<channel|>ok", false,
         [(:content, "café "), (:reasoning, "naïve"), (:content, "ok")]),
        ("Let me check.<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>\n\nDone.", false,
         [(:content, "Let me check."), (:call, 0, "get_weather", Dict("city" => "Paris")), (:content, "Done.")]),
        ("<|tool_call>call:ping{}<tool_call|>", false, [(:call, 0, "ping", Dict{String, Any}())]),
        ("<|tool_call>call:f{a:<|\"|>x}<tool_call|>", false,
         [(:content, "<|tool_call>call:f{a:<|\"|>x}<tool_call|>")]),
        ("<|tool_call>call:f{a:b:c}<tool_call|>", false,
         [(:content, "<|tool_call>call:f{a:b:c}<tool_call|>")]),
        ("<|tool_call>call:f{a:1}", false, [(:content, "<|tool_call>call:f{a:1}")]),
    ]
    rng = MersenneTwister(20261028)
    for (text, start, expected) in cases
        @testset "$(repr(text))" begin
            bytes = Vector{UInt8}(codeunits(text))
            @test sv_g4_parse(text; start_in_reasoning=start) == expected
            for ch in sv_chunkings(bytes, rng)
                @test sv_g4_parse(text; start_in_reasoning=start, chunks=ch) == expected
            end
        end
    end
end

@testset "gemma4: a stop string ends the stream" begin
    @test sv_g4_parse("Paris STOP more"; stop=["STOP"]) == [(:content, "Paris ")]
    bytes = Vector{UInt8}(codeunits("Paris STOP more"))
    chunks = [UInt8[b] for b in bytes]
    @test sv_g4_parse("Paris STOP more"; stop=["STOP"], chunks=chunks) == [(:content, "Paris ")]
end

@testset "gemma4: tool-call arguments round-trip through the renderer" begin
    args = Dict{String, Any}(
        "quote" => "say \"hi\" and 'bye'",
        "braces" => "{a: [1, 2]} }",
        "commas" => "a,b,c",
        "newline" => "line one\nline two\n",
        "unicode" => "naïve ✓",
        "int" => 42,
        "negative" => -7,
        "float" => 2.5,
        "flag" => true,
        "off" => false,
        "nothing" => nothing,
        "nested" => Dict{String, Any}(
            "inner" => Dict{String, Any}("deep" => "x,y}"),
            "list" => Any[1, "two", Dict{String, Any}("k" => false)],
        ),
        "empty_list" => Any[],
        "empty_obj" => Dict{String, Any}(),
    )
    text = "<|tool_call>call:lookup" *
           Gesso.Inference._g4_format_argument(args; escape_keys=false) *
           "<tool_call|>"
    expected = [(:call, 0, "lookup", JSON.parse(JSON.json(args)))]
    @test sv_g4_parse(text) == expected
    rng = MersenneTwister(7)
    bytes = Vector{UInt8}(codeunits(text))
    @test sv_g4_parse(text; chunks=[UInt8[b] for b in bytes]) == expected
    for ch in sv_chunkings(bytes, rng)
        @test sv_g4_parse(text; chunks=ch) == expected
    end
end

@testset "gemma4: a tool call finished by its turn end leaves no tool-response text" begin
    p = Gemma4Parser(; start_in_reasoning=false, stop=String[])
    text = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
    events = feed!(p, Vector{UInt8}(codeunits(text)))
    append!(events, Gesso.Serve.finish!(p))
    @test p.n_calls == 1
    @test !occursin("tool_response", repr(events))
end
