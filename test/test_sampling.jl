# G6: opt-in temperature/top-p sampling. Greedy stays the default and must be
# bit-identical; a seed must reproduce the same ids run to run.

using Gesso.Inference: Sampler, _sample_id

@testset "sampling: no sampler is greedy and unchanged" begin
    ts = toy2_tensors()
    g = reference_generate(ts.model, ts, PROMPT; max_new_tokens=8)
    @test reference_generate(ts.model, ts, PROMPT; max_new_tokens=8, sampler=nothing) == g
end

@testset "sampling: the same seed gives the same ids run to run" begin
    ts = toy2_tensors()
    run(seed) = reference_generate(
        ts.model,
        ts,
        PROMPT;
        max_new_tokens=8,
        sampler=Sampler(temperature=1.0, top_p=1.0, seed=seed),
    )
    @test run(7) == run(7)
    @test run(7) == run(7)
    @test 4 < length(run(7)) <= 12
end

@testset "sampling: different seeds sample different ids" begin
    ts = toy2_tensors()
    seqs = [
        reference_generate(
            ts.model,
            ts,
            PROMPT;
            max_new_tokens=8,
            sampler=Sampler(temperature=1.0, top_p=1.0, seed=s),
        ) for s in 1:8
    ]
    @test length(unique(seqs)) > 1
end

@testset "sampling: a sampler with a draft is refused" begin
    ts = toy2_tensors()
    @test_throws ArgumentError reference_generate(
        ts.model,
        ts,
        PROMPT;
        max_new_tokens=4,
        sampler=Sampler(temperature=1.0, seed=1),
        draft=:unused,
    )
end

@testset "sampling: sampler arguments are validated" begin
    @test_throws ArgumentError Sampler(temperature=0.0, seed=1)
    @test_throws ArgumentError Sampler(temperature=1.0, top_p=0.0, seed=1)
    @test_throws ArgumentError Sampler(temperature=1.0, top_p=1.5, seed=1)
    @test_throws ArgumentError Sampler(temperature=1.0, seed=-1)
end

@testset "sampling: a low temperature picks the argmax" begin
    s = Sampler(temperature=0.01, top_p=1.0, seed=5)
    @test all(_sample_id(s, [1.0, 2.0, 3.0]) == 2 for _ in 1:200)
end

@testset "sampling: top-p keeps only the nucleus" begin
    s = Sampler(temperature=1.0, top_p=0.5, seed=9)
    draws = [_sample_id(s, [0.0, 0.0, 0.0, 0.0]) for _ in 1:400]
    @test length(unique(draws)) == 2
    @test all(0 <= d <= 3 for d in draws)
end

@testset "sampling: top-p at 1 reaches every id of a flat row" begin
    s = Sampler(temperature=1.0, top_p=1.0, seed=9)
    draws = [_sample_id(s, [0.0, 0.0, 0.0, 0.0]) for _ in 1:400]
    @test sort(unique(draws)) == [0, 1, 2, 3]
end

# The draw as it was with a full stable sortperm; the nucleus selection must
# give the same id for every seed and row.
function sortperm_sample_id(s::Sampler, logits)
    hi = maximum(Float64, logits)
    w = [exp((Float64(x) - hi) / s.temperature) for x in logits]
    total = sum(w)
    order = sortperm(w; rev=true)
    target = s.top_p * total
    acc = 0.0
    keep = length(w)
    for k in eachindex(order)
        acc += w[order[k]]
        if acc >= target
            keep = k
            break
        end
    end
    cut = 0.0
    for k in 1:keep
        cut += w[order[k]]
    end
    r = Gesso.Inference._sampler_unit!(s) * cut
    acc = 0.0
    for k in 1:keep
        acc += w[order[k]]
        r < acc && return order[k] - 1
    end
    return order[keep] - 1
end

@testset "sampling: top-p draws the same ids as a full sort" begin
    rows = Dict{String, Vector{Float32}}()
    x = UInt64(12345)
    unit() = (x=x * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(x >> 11) * 0x1p-53)
    # A flat row puts most of the vocab in the sorted band; the tied rows only
    # agree if ties still break on the lower id.
    rows["flat"] = Float32[unit() for _ in 1:5000]
    rows["peaked"] = Float32[unit() < 0.01 ? 12 + unit() : unit() for _ in 1:5000]
    rows["ties"] = Float32[floor(4 * unit()) for _ in 1:3000]
    rows["all equal"] = zeros(Float32, 2049)
    rows["underflow"] = Float32[i <= 3 ? 0 : -2000 for i in 1:3000]
    for (name, row) in rows, top_p in (0.05, 0.5, 0.9, 0.999), temperature in (0.3, 1.0)
        a = Sampler(temperature=temperature, top_p=top_p, seed=31)
        b = Sampler(temperature=temperature, top_p=top_p, seed=31)
        same = all(_sample_id(a, row) == sortperm_sample_id(b, row) for _ in 1:40)
        same || @info "top-p ids differ from the full sort" name top_p temperature
        @test same
    end
end

@testset "sampling: serve sends a sampler only when a temperature is set" begin
    seen = Any[]
    engine = ChatModel(
        tokenize=s -> Int.(codeunits(s)),
        detokenize=ids -> String(UInt8.(filter(<(256), ids))),
        render=(msgs, tools, thinking) ->
            join((string(m["content"]) for m in msgs), "\n"),
        generate=(prompt_ids; max_new_tokens, on_token, sampling=nothing) -> begin
            push!(seen, sampling)
            (generated=Int[], cached=0)
        end,
        n_ctx=4096,
        model_id="gesso-sampling",
    )
    srv = serve(engine; port=0, block=false)
    try
        post(; fields...) = sv_request(
            srv.port,
            sv_post("/v1/chat/completions", sv_chat(SV_USER; fields...)),
        )

        @test post().status == 200
        @test seen[end] === nothing

        @test post(temperature=0).status == 200
        @test seen[end] === nothing

        r = post(temperature=0.8, top_p=0.9, seed=42)
        @test r.status == 200
        @test seen[end] isa Sampler
        @test seen[end].temperature == 0.8
        @test seen[end].top_p == 0.9

        @test post(temperature=3).status == 400
        @test post(temperature=0.5, top_p=0).status == 400
        @test post(temperature=0.5, seed=-1).status == 400
        @test post(temperature=true).status == 400
    finally
        close(srv)
    end
end
