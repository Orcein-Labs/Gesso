# Phase 5 item B tests (§LXXVIII): the Session engine vs the oracle.
#
# The gate: `generate(session, PROMPT)` token ids EQUAL `reference_generate`,
# and session `prefill!` logits equal `reference_prefill` at atol=0 on CPU.
# page_size=4 (prompt of 4 + decode crosses page boundaries) is mandatory.
# The oracle is called on the same fixture in-process — if these pass, the
# oracle contract is intact (it is not modified by this file).

using .GessoTestHelpers: approx_eq

const PROMPT = [1, 3, 4, 5]

_session(model, ts; kw...) =
    Gesso.Session(model, ts; context_length=128, eos_token_id=2, kw...)

@testset "Session engine vs oracle (§LXXVIII item B)" begin
    ts = toy2_tensors()
    model = toy2_modelir()

    @testset "generate matches reference_generate exactly" begin
        s = _session(model, ts)
        g_session = Gesso.generate(s, PROMPT; max_new_tokens=8)
        g_oracle = Gesso.reference_generate(model, ts, PROMPT; max_new_tokens=8)
        @test g_session == g_oracle
        @test g_session[1:4] == PROMPT
    end

    @testset "two sessions, same prompt ⇒ same ids (deterministic)" begin
        g1 = Gesso.generate(_session(model, ts), PROMPT; max_new_tokens=8)
        g2 = Gesso.generate(_session(model, ts), PROMPT; max_new_tokens=8)
        @test g1 == g2
        # and the same session reused is deterministic too (generate resets)
        s = _session(model, ts)
        @test Gesso.generate(s, PROMPT; max_new_tokens=8) ==
              Gesso.generate(s, PROMPT; max_new_tokens=8)
    end

    @testset "max_new_tokens=0 returns exactly the prompt" begin
        g = Gesso.generate(_session(model, ts), PROMPT; max_new_tokens=0)
        @test g == PROMPT
    end

    @testset "first generated token == argmax of reference_prefill last column" begin
        L = Gesso.reference_prefill(model, ts, PROMPT)
        expected_first = argmax(@view L[:, end]) - 1
        g = Gesso.generate(_session(model, ts), PROMPT; max_new_tokens=1)
        @test length(g) == 5
        @test g[end] == expected_first
    end

    @testset "prefill! logits == reference_prefill (atol=0, CPU)" begin
        s = _session(model, ts)
        logits = Gesso.prefill!(s, PROMPT)
        @test size(logits) == (32, 4)
        @test approx_eq(logits, Gesso.reference_prefill(model, ts, PROMPT); atol=0.0)
        # the paged manager holds exactly the prompt's K/V rows
        @test Gesso.Inference.kv_len(s.mgr) == 4
        @test length(Gesso.Inference.kv_cache(s.mgr, 1, :k).storage) == 1  # one page
    end

    @testset "decode-via-session matches re-prefill argmax (cache correctness)" begin
        g = Gesso.generate(_session(model, ts), PROMPT; max_new_tokens=3)
        @test length(g) == 7
        L_full = Gesso.reference_prefill(model, ts, g)
        for t in 5:7
            @test argmax(@view L_full[:, t-1]) - 1 == g[t]
        end
    end

    @testset "page_size=4 still matches the oracle (page boundary is not optional)" begin
        s = _session(model, ts; page_size=4)
        g = Gesso.generate(s, PROMPT; max_new_tokens=8)
        @test g == Gesso.reference_generate(model, ts, PROMPT; max_new_tokens=8)
        # 4 prompt + up to 8 decode steps at page_size=4 ⇒ ≥2 pages per cache
        @test length(Gesso.Inference.kv_cache(s.mgr, 1, :k).storage) >= 2
        # and prefill! is STILL bit-identical at page_size=4
        s2 = _session(model, ts; page_size=4)
        @test approx_eq(
            Gesso.prefill!(s2, PROMPT),
            Gesso.reference_prefill(model, ts, PROMPT);
            atol=0.0,
        )
    end

    @testset "EOS: eos_token_id=2 stops the loop, EOS is last" begin
        g = Gesso.generate(_session(model, ts), PROMPT; max_new_tokens=8)
        eos_idx = findfirst(==(2), g)
        if eos_idx !== nothing
            @test eos_idx == lastindex(g)     # nothing generated after EOS
        end
        @test length(g) <= 12
    end

    @testset "empty prompt throws (ERR_INVALID_PLAN)" begin
        err = try
            Gesso.generate(_session(model, ts), Int[]; max_new_tokens=2)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
        err2 = try
            Gesso.prefill!(_session(model, ts), Int[])
            nothing
        catch e
            e
        end
        @test err2 isa Gesso.GessoError && err2.code == Gesso.ERR_INVALID_PLAN
    end

    @testset "context exhaustion is typed ERR_RESOURCE_LIMIT" begin
        s = Gesso.Session(model, ts; context_length=6, eos_token_id=2)
        err = try
            Gesso.generate(s, PROMPT; max_new_tokens=8)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError
        @test err.code == Gesso.ERR_RESOURCE_LIMIT
    end

    @testset "oracle regression: reference_* still bit-deterministic in-process" begin
        @test approx_eq(
            Gesso.reference_prefill(model, ts, PROMPT),
            Gesso.reference_prefill(model, ts, PROMPT);
            atol=0.0,
        )
        @test Gesso.reference_generate(model, ts, PROMPT; max_new_tokens=8) ==
              Gesso.reference_generate(model, ts, PROMPT; max_new_tokens=8)
    end

    @testset "prefill! refuses a consumed session (multi-turn is not this sprint)" begin
        s = _session(model, ts)
        Gesso.prefill!(s, PROMPT)
        err = try
            Gesso.prefill!(s, PROMPT)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
    end
end
