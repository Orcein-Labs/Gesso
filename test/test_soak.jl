# The soak client (scripts/soak.jl) against a fake engine for about 20 seconds,
# so the client keeps working as the server changes. A second, deliberately
# unrepeatable engine is the negative control: a greedy check that cannot fail
# proves nothing.

using Sockets
using Gesso

include(joinpath(@__DIR__, "..", "scripts", "soak.jl"))
include(joinpath(@__DIR__, "..", "scripts", "soak_margin.jl"))

# Output depends only on the prompt, so greedy repeats agree. `flaky` adds a
# counter, so no two replies to the same prompt agree.
#
# It also behaves like the parts of a real server the client has to cope with:
#  - a one-slot prompt cache keeping the longest common prefix with the last
#    prompt and its reply, reported as `cached` (never the whole prompt);
#  - reasoning: with thinking on the prompt ends in an open <think>, so the whole
#    reply is reasoning and `content` is empty;
#  - a get_weather call in the qwen35_xml format when tools are on offer and the
#    last message is the user's, except for Lima, where it answers in text.
function soak_engine(; flaky::Bool=false, accepts_tools::Bool=true)
    calls = Threads.Atomic{Int}(0)
    slot = Int[]
    slot_lock = ReentrantLock()
    lcp(a, b) = something(findfirst(i -> a[i] != b[i], 1:min(length(a), length(b))), min(length(a), length(b)) + 1) - 1
    generate = (prompt_ids; max_new_tokens, on_token, sampling=nothing, kwargs...) -> begin
        flaky && Threads.atomic_add!(calls, 1)
        cached = lock(slot_lock) do
            min(lcp(prompt_ids, slot), length(prompt_ids) - 1)
        end
        text = String(UInt8.(prompt_ids))
        out = Int[]
        if endswith(text, "[call]")
            city = something(match(r"weather in (\w+)\?", text), (; captures=["Paris"])).captures[1]
            reply = city == "Lima" ? "Cloudy." :
                    "<tool_call>\n<function=get_weather>\n<parameter=city>\n$city\n</parameter>\n</function>\n</tool_call>"
            for b in first(codeunits(reply), max_new_tokens)
                push!(out, Int(b))
                on_token(Int(b)) === false && break
            end
        else
            h = sum(prompt_ids) + calls[]
            for i in 1:min(max_new_tokens, 30)
                id = Int(UInt8('a') + (h + 7i) % 26)
                push!(out, id)
                on_token(id) === false && break
                sleep(0.002)
            end
        end
        lock(slot_lock) do
            slot = vcat(prompt_ids, out)
        end
        return (generated=out, cached=cached)
    end
    render = (msgs, tools, thinking) -> begin
        text = join((string(something(get(m, "content", nothing), "")) for m in msgs), "\n")
        !isempty(tools) && last(msgs)["role"] == "user" && (text *= "[call]")
        thinking && (text *= "\n<think>\n")
        return text
    end
    return Gesso.ChatModel(
        tokenize=s -> Int.(codeunits(s)),
        detokenize=ids -> String(UInt8.(ids)),
        render=render,
        generate=generate,
        n_ctx=512,
        model_id="soak-toy",
        accepts_tools=accepts_tools,
    )
end

@testset "soak: the client passes against a repeatable server" begin
    srv = Gesso.serve(soak_engine(); port=0, block=false)
    try
        cfg = GessoSoak.Config(
            port=srv.port,
            duration=20,
            clients=3,
            pid=getpid(),
            rss_interval=2,
            rss_growth_mb=Inf,      # the test process's own GC says nothing about the server
            long_words=60,
            replay_frac=0.5,
            metrics_interval=2,
            warmup=0,
        )
        io = IOBuffer()
        res = GessoSoak.soak(cfg; log=io)
        for f in res.failures
            @info "soak failure" f.kind f.reason f.request
        end
        @test res.ok
        @test isempty(res.failures)
        for (kind, _) in GessoSoak.MIX
            @test get(res.counts, kind, 0) > 0
        end
        @test get(res.counts, "health", 0) > 5
        @test res.rss.peak_kb > 0
        @test res.replay.n > 0
        @test res.replay.diverged == 0      # the toy engine is deterministic
        # the evicting request makes replays fresh, and the one-slot cache lets some
        # originals run on a cached prefix (see the single-client test)
        @test res.replay.fresh == res.replay.n
        @test res.replay.eligible_diverged == 0
        @test get(res.counts, "evict", 0) >= res.replay.n
        # tool turns: the toy calls the tool except for Lima, which it answers in text
        @test res.tools.enabled && res.tools.calls > 0 && res.tools.no_call > 0
        @test res.tools.turns == res.tools.calls + res.tools.no_call
        @test res.metrics.windows >= 5
        @test res.metrics.mean_tok_s > 0
        @test res.p99.judged && !res.p99.grew
        # every event is one line of JSON
        lines = split(String(take!(io)), '\n'; keepempty=false)
        @test length(lines) > 100
        events = [GessoSoak.JSON.parse(l) for l in lines]
        @test all(e -> e isa AbstractDict, events)
        kinds(name) = [e for e in events if e["event"] == name]
        @test length(kinds("metrics")) >= 5
        @test all(e -> haskey(e, "requests") && haskey(e, "errors") && haskey(e, "ttft_p99_ms"), kinds("metrics"))
        replays = kinds("replay")
        @test !isempty(replays) && all(e -> e["ok"] === true, replays)
        @test all(e -> haskey(e, "original_cached") && haskey(e, "replay_cached"), replays)
        @test all(e -> e["fresh"] === true && e["replay_cached"] <= GessoSoak.FRESH_CACHED_MAX, replays)
        @test all(e -> e["thinking"] === false, replays)
        # the toy context is small, so conversations end on the size estimate, never on a 400
        @test any(e -> e["ended"] == "context", kinds("conversation"))
        @test !any(f -> f.kind in ("conversation", "conversation_stream"), res.failures)
        # the toy thinks unless told not to: conversations only get past turn 1
        # because every turn carries enable_thinking=false
        @test any(e -> e["turns"] > 1, kinds("conversation"))
        @test !any(e -> e["ended"] == "empty reply", kinds("conversation"))
    finally
        close(srv)
    end
end

@testset "soak: a server that is not repeatable fails the greedy check" begin
    srv = Gesso.serve(soak_engine(flaky=true); port=0, block=false)
    try
        cfg = GessoSoak.Config(port=srv.port, duration=4, clients=2)
        res = GessoSoak.soak(cfg)
        @test !res.ok
        @test any(f -> startswith(f.kind, "greedy") && occursin("differs", f.reason), res.failures)
    finally
        close(srv)
    end
end

@testset "soak: --replay-strict turns a replay divergence into a failure" begin
    srv = Gesso.serve(soak_engine(flaky=true); port=0, block=false)
    try
        io = IOBuffer()
        cfg = GessoSoak.Config(port=srv.port, duration=4, clients=2, replay_frac=1.0, replay_strict=true, warmup=0)
        res = GessoSoak.soak(cfg; log=io)
        @test !res.ok
        @test res.replay.diverged > 0
        @test any(f -> f.kind == "replay" && occursin("replay diverged", f.reason), res.failures)
        diverged = [GessoSoak.JSON.parse(l) for l in split(String(take!(io)), '\n'; keepempty=false)]
        diverged = [e for e in diverged if e["event"] == "replay" && e["ok"] === false]
        @test !isempty(diverged) && haskey(first(diverged), "messages")
        # without --replay-strict the same divergence is only counted
        cfg = GessoSoak.Config(port=srv.port, duration=4, clients=2, replay_frac=1.0, warmup=0)
        res = GessoSoak.soak(cfg)
        @test res.replay.diverged > 0
        @test !any(f -> f.kind == "replay", res.failures)
    finally
        close(srv)
    end
end

@testset "soak: trend verdicts" begin
    cfg = GessoSoak.Config(warmup=300)
    series(ys; step=60.0, t0=0.0) = Tuple{Float64, Float64}[(t0 + (i - 1) * step, Float64(y)) for (i, y) in enumerate(ys)]
    p99(points) = GessoSoak.trend_verdict(points, cfg.warmup; growth_abs=cfg.p99_growth_ms, growth_frac=cfg.p99_growth_frac)

    # p99 time to first token, 60 s windows; the first 5 are warm-up
    flat = series(fill(800, 20))
    rising = series(vcat(fill(800, 5), 800 .+ 600 .* (0:14)))
    short = series(vcat(fill(800, 5), 800, 4000, 8000, 12000))     # 4 windows after warm-up
    @test p99(flat).judged && !p99(flat).grew
    @test p99(rising).judged && p99(rising).grew
    @test !p99(short).judged && !p99(short).grew
    # a slow start that then settles is warm-up, not a trend
    settling = series(vcat(9000, 6000, 3000, 1500, 900, fill(800, 15)))
    @test !p99(settling).grew
    # rising but under the absolute threshold, then under the fractional one
    @test !p99(series(vcat(fill(800, 5), 800 .+ 20 .* (0:14)))).grew
    @test !p99(series(vcat(fill(10000, 5), 10000 .+ 300 .* (0:14)))).grew

    # RSS and VRAM share one verdict; samples are (seconds, kB), every 60 s
    mb(x) = round(Int, x * 1024)
    rss(mbs; step=60.0) = Tuple{Float64, Int}[((i - 1) * step, mb(m)) for (i, m) in enumerate(mbs)]
    @test GessoSoak.rss_verdict(rss(fill(500, 30)), cfg).judged
    @test !GessoSoak.rss_verdict(rss(fill(500, 30)), cfg).grew
    @test GessoSoak.rss_verdict(rss(500 .+ 10 .* (0:29)), cfg).grew          # +290 MB over 30 min
    @test !GessoSoak.rss_verdict(rss(500 .+ 10 .* (0:3)), cfg).judged        # too few samples
    @test !GessoSoak.rss_verdict(rss(500 .+ 10 .* (0:3)), cfg).grew
    # growth that happens before the warm-up ends is not judged
    early = rss(vcat(500 .+ 100 .* (0:4), fill(1000, 20)))
    @test !GessoSoak.rss_verdict(early, cfg).grew
    # the VRAM threshold is its own
    vram = rss(vcat(fill(1000, 16), fill(1200, 6)))     # +200 MB, +20%
    @test !GessoSoak.rss_verdict(vram, cfg; growth_mb=cfg.vram_growth_mb).grew
    @test GessoSoak.rss_verdict(vram, cfg; growth_mb=50).grew
end

@testset "soak: the mix cancels about a tenth of streamed requests" begin
    @test 0.08 <= GessoSoak.cancel_share(GessoSoak.Config()) <= 0.12
end

@testset "soak: a dead port fails instead of passing" begin
    srv = Gesso.serve(soak_engine(); port=0, block=false)
    port = srv.port
    close(srv)
    res = GessoSoak.soak(GessoSoak.Config(port=port, duration=1, clients=1))
    @test !res.ok
end

@testset "soak: the evicting request makes the replay a fresh session" begin
    # without it the one-slot cache serves the replay whole
    srv = Gesso.serve(soak_engine(); port=0, block=false)
    try
        ctx = GessoSoak.Ctx(
            GessoSoak.Config(port=srv.port), 512, devnull, ReentrantLock(), time(),
            Dict{String, Vector{Float64}}(), Dict{String, Int}(), Dict{String, Int}(),
            GessoSoak.Failure[], Dict{String, NamedTuple}(), Tuple{Float64, Int}[], 0, 0.0, Ref(false),
            0, NTuple{3, Float64}[], NamedTuple[], Tuple{Float64, Int}[], 0, Int[], (0, 0, 0),
            0, 0, 0, true, "", 0, 0, 0,
        )
        rng = GessoSoak.MersenneTwister(3)
        body = GessoSoak.chat_json(GessoSoak.user("one two three four five six seven eight nine ten");
            max_tokens=4, temperature=0, enable_thinking=false)
        raw = GessoSoak.post_chat(body)
        first_try = GessoSoak.chat_exchange(ctx, raw; stream=false, max_tokens=4)
        again = GessoSoak.chat_exchange(ctx, raw; stream=false, max_tokens=4)
        @test GessoSoak.usage_of(first_try[1], false).cached == 0
        @test GessoSoak.usage_of(again[1], false).cached > GessoSoak.FRESH_CACHED_MAX
        @test GessoSoak.evict!(ctx, rng)
        third = GessoSoak.chat_exchange(ctx, raw; stream=false, max_tokens=4)
        @test GessoSoak.usage_of(third[1], false).cached <= GessoSoak.FRESH_CACHED_MAX
        @test get(ctx.counts, "evict", 0) == 1 && isempty(ctx.failures)
    finally
        close(srv)
    end
end

@testset "soak: conversation requests carry enable_thinking=false" begin
    srv = Gesso.serve(soak_engine(); port=0, block=false)
    try
        # with thinking on, the toy's reply is all reasoning and the conversation dies at turn 1
        ctx_cfg = GessoSoak.Config(port=srv.port)
        raw = GessoSoak.post_chat(GessoSoak.chat_json(GessoSoak.user("hello there"); max_tokens=8, temperature=0))
        r = GessoSoak.exchange(ctx_cfg, raw)
        _, answer = GessoSoak.check_completion(r, 8)
        @test isempty(answer.content) && !isempty(answer.reasoning)
        raw = GessoSoak.post_chat(GessoSoak.chat_json(GessoSoak.user("hello there"); max_tokens=8, temperature=0, enable_thinking=false))
        _, answer = GessoSoak.check_completion(GessoSoak.exchange(ctx_cfg, raw), 8)
        @test !isempty(answer.content) && isempty(answer.reasoning)
    finally
        close(srv)
    end
end

@testset "soak: a model without tool support switches the tool kinds off" begin
    srv = Gesso.serve(soak_engine(accepts_tools=false); port=0, block=false)
    try
        io = IOBuffer()
        res = GessoSoak.soak(GessoSoak.Config(port=srv.port, duration=4, clients=2, warmup=0, long_words=60); log=io)
        @test res.ok
        @test isempty(res.failures)
        @test !res.tools.enabled && occursin("tool", res.tools.note)
        @test get(res.counts, "tool_call", 0) == 0 && get(res.counts, "tool_call_stream", 0) == 0
        events = [GessoSoak.JSON.parse(l) for l in split(String(take!(io)), '\n'; keepempty=false)]
        @test any(e -> e["event"] == "note" && occursin("tool", e["message"]), events)
        out = IOBuffer()
        GessoSoak.print_summary(out, res)
        @test occursin("tools: off", String(take!(out)))
    finally
        close(srv)
    end
end

@testset "soak: tool call validation" begin
    ok = Dict("id" => "call_1", "type" => "function",
        "function" => Dict("name" => "get_weather", "arguments" => "{\"city\": \"Paris\"}"))
    withfn(k, v) = merge(ok, Dict("function" => merge(ok["function"], Dict(k => v))))
    @test GessoSoak.check_tool_call(ok) === nothing
    @test GessoSoak.check_tool_call(withfn("name", "get_time")) !== nothing
    @test GessoSoak.check_tool_call(withfn("arguments", "{\"city\": ")) !== nothing
    @test GessoSoak.check_tool_call(withfn("arguments", "[1]")) !== nothing
    @test GessoSoak.check_tool_call(withfn("arguments", "{\"city\": 3}")) !== nothing
    @test GessoSoak.check_tool_call(merge(ok, Dict("id" => ""))) !== nothing
    @test GessoSoak.check_tool_call(merge(ok, Dict("type" => "tool"))) !== nothing
    answer(finish, calls) = (reasoning="", content="", finish=finish, tool_calls=Any[calls...])
    @test GessoSoak.check_tool_answer(answer("tool_calls", [ok])) === nothing
    @test GessoSoak.check_tool_answer(answer("stop", [])) === nothing
    @test GessoSoak.check_tool_answer(answer("tool_calls", [])) !== nothing
    @test GessoSoak.check_tool_answer(answer("stop", [ok])) !== nothing
end

@testset "soak: one client leaves originals on a cached prefix, so replays are drift-eligible" begin
    # with several clients the one slot is rarely still holding the conversation
    srv = Gesso.serve(soak_engine(); port=0, block=false)
    try
        cfg = GessoSoak.Config(port=srv.port, duration=8, clients=1, warmup=0, long_words=60,
            replay_frac=1.0, metrics_interval=2, rss_growth_mb=Inf)
        io = IOBuffer()
        res = GessoSoak.soak(cfg; log=io)
        @test res.replay.n > 0 && res.replay.fresh == res.replay.n
        @test res.replay.eligible > 0
        @test res.replay.eligible_diverged == 0
        events = [GessoSoak.JSON.parse(l) for l in split(String(take!(io)), '\n'; keepempty=false)]
        @test any(e -> e["event"] == "replay" && e["drift_eligible"] === true &&
                       e["original_cached"] > GessoSoak.FRESH_CACHED_MAX, events)
    finally
        close(srv)
    end
end

@testset "soak margin: pure parts" begin
    M = SoakMargin
    @test M.first_diff_index([1, 2, 3], [1, 2, 4]) == 3
    @test M.first_diff_index([5, 2], [1, 2]) == 1
    @test M.first_diff_index([1, 2], [1, 2, 3]) === nothing
    @test M.first_diff_index(Int[], Int[]) === nothing
    # a tie closer than the bar, with one candidate on top
    @test M.verdict(7, 9, 3.10, 3.00, 7, 0.25) == "near-tie"
    @test M.verdict(7, 9, 3.00, 3.10, 9, 0.25) == "near-tie"
    # the bar is strict: a gap equal to it is a defect
    @test M.verdict(7, 9, 3.25, 3.00, 7, 0.25) == "defect"
    @test M.verdict(7, 9, 5.0, 1.0, 7, 0.25) == "defect"
    # neither candidate on top is a defect however close the two are
    @test M.verdict(7, 9, 3.00, 3.01, 4, 0.25) == "defect"
    @test M.verdict(7, 9, nothing, 3.0, 7, 0.25) == "undetermined"
    # where the replies part; a reply that stops early has the end-of-turn token as its candidate
    @test M.divergence_point([1, 2, 3], [1, 2, 4], "length", "length") == (3, :none)
    @test M.divergence_point([1, 2], [1, 2, 3], "stop", "length") == (3, :orig)
    @test M.divergence_point([1, 2, 3, 4], [1, 2], "length", "stop") == (3, :replay)
    @test M.divergence_point([1, 2], [1, 2, 3], "length", "stop") isa String
    @test M.divergence_point([1, 2], [1, 2, 3], nothing, nothing) isa String
    @test M.divergence_point([1, 2], [1, 2], "stop", "stop") isa String
    # an undetermined divergence fails the run like a defect
    tie, und = (verdict="near-tie",), (verdict="undetermined",)
    @test M.passes([tie, tie])
    @test !M.passes([tie, und])
    @test !M.passes([(verdict="defect",)])
    # logits come back through the sampler hook, argmax 0-based
    p = M.LogitProbe()
    @test Gesso.Inference._sample_id(p, Float32[0.1, 2.0, 0.5]) == 1
    @test p.row == Float64.(Float32[0.1, 2.0, 0.5])
    @test M.parse_args(["--gguf", "m", "--events", "e"])["--bar"] == "0.25"
    @test_throws ArgumentError M.parse_args(["--gguf", "m"])
    @test_throws ArgumentError M.parse_args(["--bogus", "1"])
end
