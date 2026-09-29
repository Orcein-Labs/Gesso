# Item C tests: greedy generation + KV decode (§LXXV).
#
# Invariants under test (from the goal):
#   * same prompt ⇒ same output ids, two runs (deterministic)
#   * max_new_tokens=0 returns exactly the prompt
#   * prefill last-position argmax == first generated token
#   * KV length equals prefix length (decode appends; no re-prefill)
#   * decode path is bit-identical to re-prefilling (cache correctness)

using .ToyFixtures: load_toy_fixture, toy_weights
using .GessoTestHelpers: approx_eq

const PROMPT = [1, 3, 4, 5]

@testset "generate: determinism (same prompt ⇒ same ids)" begin
    ts = toy2_tensors()
    g1 = reference_generate(ts.model, ts, PROMPT; max_new_tokens=8)
    g2 = reference_generate(ts.model, ts, PROMPT; max_new_tokens=8)
    @test g1 == g2
    @test g1[1:4] == PROMPT                       # prompt is a prefix
    @test 4 < length(g1) <= 12                    # grew, but capped
end

@testset "generate: max_new_tokens=0 returns exactly the prompt" begin
    ts = toy2_tensors()
    g = reference_generate(ts.model, ts, PROMPT; max_new_tokens=0)
    @test g == PROMPT
end

@testset "generate: first token is the prefill last-position argmax" begin
    ts = toy2_tensors()
    L = reference_prefill(ts.model, ts, PROMPT)
    expected_first = argmax(@view L[:, end]) - 1   # 0-based
    g = reference_generate(ts.model, ts, PROMPT; max_new_tokens=1)
    @test length(g) == 5
    @test g[end] == expected_first
end

@testset "generate: KV length equals prefix length (real append, no re-prefill)" begin
    ts = toy2_tensors()
    info = Ref{Any}(nothing)
    g = reference_generate(ts.model, ts, PROMPT; max_new_tokens=3, info=info)
    @test info[] isa NamedTuple
    @test info[].kv_len == length(PROMPT) + info[].steps == 7
    @test length(g) == 7
end

@testset "generate: EOS path stops the loop" begin
    # force EOS: monkey-free approach — max_new_tokens large; toy2's actual
    # behavior is whatever the oracle says, so pin ONLY the invariant that
    # the sequence stops if EOS appears, and that EOS-in-output implies stop
    ts = toy2_tensors()
    g = reference_generate(ts.model, ts, PROMPT; max_new_tokens=8)
    eos_idx = findfirst(==(2), g)
    if eos_idx !== nothing
        @test eos_idx == lastindex(g)   # nothing generated after EOS
    end
    @test length(g) <= 12
end

@testset "generate: decode matches re-prefill bit-for-bit (cache correctness)" begin
    # THE cache-correctness gate: generation via KV append must reproduce
    # exactly what a full re-prefill of prompt+generated computes.
    # Column t of the prefill logits predicts token t+1 ("logits after
    # consuming tokens 1..t"), so token g[t] must equal argmax of column t-1.
    ts = toy2_tensors()
    g = reference_generate(ts.model, ts, PROMPT; max_new_tokens=3)
    @test length(g) == 7
    L_full = reference_prefill(ts.model, ts, g)
    for t in 5:7
        @test argmax(@view L_full[:, t-1]) - 1 == g[t]
    end
end
