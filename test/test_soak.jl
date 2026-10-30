# The soak client (scripts/soak.jl) against a fake engine for about 20 seconds,
# so the client keeps working as the server changes. A second, deliberately
# unrepeatable engine is the negative control: a greedy check that cannot fail
# proves nothing.

using Sockets
using Gesso

include(joinpath(@__DIR__, "..", "scripts", "soak.jl"))

# Output depends only on the prompt, so greedy repeats agree. `flaky` adds a
# counter, so no two replies to the same prompt agree.
function soak_engine(; flaky::Bool=false)
    calls = Threads.Atomic{Int}(0)
    generate = (prompt_ids; max_new_tokens, on_token, sampling=nothing, kwargs...) -> begin
        flaky && Threads.atomic_add!(calls, 1)
        h = sum(prompt_ids) + calls[]
        out = Int[]
        for i in 1:min(max_new_tokens, 30)
            id = Int(UInt8('a') + (h + 7i) % 26)
            push!(out, id)
            on_token(id) === false && break
            sleep(0.002)
        end
        return (generated=out, cached=0)
    end
    return Gesso.ChatModel(
        tokenize=s -> Int.(codeunits(s)),
        detokenize=ids -> String(UInt8.(ids)),
        render=(msgs, tools, thinking) -> join((string(m["content"]) for m in msgs), "\n"),
        generate=generate,
        n_ctx=512,
        model_id="soak-toy",
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
        # every event is one line of JSON
        lines = split(String(take!(io)), '\n'; keepempty=false)
        @test length(lines) > 100
        @test all(l -> JSON.parse(l) isa AbstractDict, lines)
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

@testset "soak: a dead port fails instead of passing" begin
    srv = Gesso.serve(soak_engine(); port=0, block=false)
    port = srv.port
    close(srv)
    res = GessoSoak.soak(GessoSoak.Config(port=port, duration=1, clients=1))
    @test !res.ok
end
