# Generation hooks: `on_token` and `stop_ids` on reference_generate.
#
# A hook stop takes the EOS path: the stopping id is returned, its K/V rows are
# never written, and the state holds every id before it. So a state reused
# after an early stop must answer exactly like a fresh run, and the reuse
# prefills only the one id the stop left out.
#
# Attention-only only. The GDN snapshot path needs a hybrid model, and the only
# hybrid fixtures are the real Qwen3.5 0.8B checkpoint (env-gated).

const HOOK_PROMPT = [1, 3, 4, 5]

@testset "hooks: defaults leave output unchanged" begin
    ts = toy2_tensors()
    base = reference_generate(ts.model, ts, HOOK_PROMPT; max_new_tokens=6)
    @test length(base) > length(HOOK_PROMPT) + 2
    @test reference_generate(
        ts.model,
        ts,
        HOOK_PROMPT;
        max_new_tokens=6,
        on_token=nothing,
        stop_ids=(),
    ) == base
    @test reference_generate(
        ts.model,
        ts,
        HOOK_PROMPT;
        max_new_tokens=6,
        on_token=_ -> nothing,
    ) == base
    @test reference_generate(
        ts.model,
        ts,
        HOOK_PROMPT;
        max_new_tokens=6,
        on_token=_ -> true,
    ) == base
end

@testset "hooks: on_token sees exactly the returned generated tokens" begin
    ts = toy2_tensors()
    seen = Int[]
    g = reference_generate(
        ts.model,
        ts,
        HOOK_PROMPT;
        max_new_tokens=6,
        on_token=t -> (push!(seen, t); nothing),
    )
    @test seen == g[(length(HOOK_PROMPT)+1):end]
    @test length(seen) >= 3
end

@testset "hooks: on_token false stops after the token it was given" begin
    ts = toy2_tensors()
    P = length(HOOK_PROMPT)
    full = reference_generate(ts.model, ts, HOOK_PROMPT; max_new_tokens=6)
    for k in 1:3
        n = Ref(0)
        info = Ref{Any}(nothing)
        g = reference_generate(
            ts.model,
            ts,
            HOOK_PROMPT;
            max_new_tokens=6,
            info=info,
            on_token=_ -> (n[] += 1; n[] == k ? false : nothing),
        )
        @test g == full[1:(P+k)]
        @test info[].steps == k
    end
    # false on the very first token: the prompt plus that one token
    @test reference_generate(
        ts.model,
        ts,
        HOOK_PROMPT;
        max_new_tokens=6,
        on_token=_ -> false,
    ) == full[1:(P+1)]
end

@testset "hooks: stop_ids stops at the first emitted match" begin
    ts = toy2_tensors()
    P = length(HOOK_PROMPT)
    full = reference_generate(ts.model, ts, HOOK_PROMPT; max_new_tokens=6)
    gen = full[(P+1):end]
    # a token that has not appeared earlier, so the stop lands exactly there
    k = findfirst(i -> !(gen[i] in gen[1:(i-1)]), eachindex(gen))
    @test k !== nothing
    info = Ref{Any}(nothing)
    g = reference_generate(
        ts.model,
        ts,
        HOOK_PROMPT;
        max_new_tokens=6,
        info=info,
        stop_ids=(-1, gen[k]),
    )
    @test g == full[1:(P+k)]
    @test g[end] == gen[k]
    @test info[].steps == k
    @test reference_generate(ts.model, ts, HOOK_PROMPT; max_new_tokens=6, stop_ids=(-1,)) ==
          full
end

# The default EOS is the toy contract's id 2. A real vocabulary passes its own,
# because id 2 is an ordinary token there (`#` in GPT-2 byte-level vocabs).
@testset "hooks: eos_ids replaces the toy EOS" begin
    ts = toy2_tensors()
    P = length(HOOK_PROMPT)
    full = reference_generate(ts.model, ts, HOOK_PROMPT; max_new_tokens=6)
    gen = full[(P+1):end]
    @test !(2 in gen)
    k = findfirst(i -> !(gen[i] in gen[1:(i-1)]), eachindex(gen))
    @test reference_generate(
        ts.model,
        ts,
        HOOK_PROMPT;
        max_new_tokens=6,
        eos_ids=(gen[k],),
    ) == full[1:(P+k)]
    @test reference_generate(ts.model, ts, HOOK_PROMPT; max_new_tokens=6, eos_ids=()) ==
          full
end

@testset "hooks: a state reused after an early stop equals a fresh run" begin
    ts = toy2_tensors()
    P = length(HOOK_PROMPT)
    st = GenerateState(64)
    info = Ref{Any}(nothing)
    n = Ref(0)
    a = reference_generate(
        ts.model,
        ts,
        HOOK_PROMPT;
        max_new_tokens=8,
        state=st,
        info=info,
        on_token=_ -> (n[] += 1; n[] < 3 ? nothing : false),
    )
    @test length(a) == P + 3
    @test length(st.ids) == length(a) - 1
    @test info[].steps == 3

    # the next turn: the stopped reply plus a new user message
    p2 = vcat(a, [7, 9, 11])
    b = reference_generate(ts.model, ts, p2; max_new_tokens=5, state=st, info=info)
    @test info[].reused == length(a) - 1
    @test b == reference_generate(ts.model, ts, p2; max_new_tokens=5)

    # a regenerate of the stopped turn itself
    c = reference_generate(ts.model, ts, a; max_new_tokens=5, state=st, info=info)
    @test info[].reused == length(a) - 1
    @test c == reference_generate(ts.model, ts, a; max_new_tokens=5)
    close(st)
end

@testset "hooks: a stop_ids stop leaves the state as an EOS stop does" begin
    ts = toy2_tensors()
    P = length(HOOK_PROMPT)
    full = reference_generate(ts.model, ts, HOOK_PROMPT; max_new_tokens=6)
    gen = full[(P+1):end]
    k = findfirst(i -> !(gen[i] in gen[1:(i-1)]), eachindex(gen))
    st = GenerateState(64)
    info = Ref{Any}(nothing)
    a = reference_generate(
        ts.model,
        ts,
        HOOK_PROMPT;
        max_new_tokens=6,
        state=st,
        info=info,
        stop_ids=(gen[k],),
    )
    @test a == full[1:(P+k)]
    p2 = vcat(a, [8, 2, 5])
    b = reference_generate(ts.model, ts, p2; max_new_tokens=4, state=st, info=info)
    @test info[].reused == length(a) - 1
    @test b == reference_generate(ts.model, ts, p2; max_new_tokens=4)
    close(st)
end

@testset "hooks: an on_token error invalidates the state, never a half-written one" begin
    ts = toy2_tensors()
    st = GenerateState(64)
    info = Ref{Any}(nothing)
    n = Ref(0)
    @test_throws ErrorException reference_generate(
        ts.model,
        ts,
        HOOK_PROMPT;
        max_new_tokens=6,
        state=st,
        info=info,
        on_token=_ -> (n[] += 1; n[] < 2 ? nothing : error("client gone")),
    )
    @test isempty(st.ids)
    g = reference_generate(ts.model, ts, HOOK_PROMPT; max_new_tokens=5, state=st, info=info)
    @test info[].reused == 0
    @test g == reference_generate(ts.model, ts, HOOK_PROMPT; max_new_tokens=5)
    close(st)
end
