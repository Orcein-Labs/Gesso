# test_type_stability.jl — Phase 10D item C: the hot path is idiomatic Julia.
#
# The oracle gates (10D RPDO law) only say the engine is CORRECT; this file
# says it is also TYPE-STABLE on the fixtures. Three required gates
# (10D item C), CPU, toy2 + llama_micro, always-on:
#
#   @inferred reference_prefill(toy2, ts, prompt)   → @test_broken (PACKET)
#   @inferred decode!(session)                      → @test_broken (PACKET)
#   @inferred unique_kv_bytes(mgr)                  → GREEN (Win 1)
#
# Why the first two are @test_broken and NOT fixed here: the inference leak
# is `storage::Any` on the §XI family structs (Parameters.jl, pinned by the
# §CIX encoding law "metadata is named fields, never type parameters") and
# `Session.{model,tensors,h}::Any`. Fixing it means re-encoding the §CIX
# object model — a new type hierarchy — which 10D forbids ("do not invent
# the hierarchy here"). PACKETED, not silently dropped (10D receipt law).
# The byte counters ARE locally fixable: the manager pins `page_bytes::Int`
# (runtime metadata, §XIII) so Profiling byte math infers concretely.
#
# A future packet that re-encodes `storage` MUST flip the two @test_broken
# to @inferred — that is the receipt's exit condition.

using Test

const STAB_PROMPT = [1, 3, 4, 5]   # toy2: BOS + 3, 4, 5 (0-based), own const — no coupling

@testset "10D item C: byte counters are type-stable (Win 1)" begin
    ts = toy2_tensors()
    m = ts.model

    s = Gesso.Session(m, ts; context_length=128, eos_token_id=2)
    Gesso.prefill!(s, STAB_PROMPT)
    n = @inferred Gesso.Profiling.unique_kv_bytes(s.mgr)
    fp = @inferred Gesso.Profiling.kv_footprint(s.mgr)
    @test n isa Int && fp isa Int
    @test n == fp                       # single manager: distinct == total
end

@testset "10D item C: oracle/engine stability is PACKETED on §CIX storage::Any" begin
    ts = toy2_tensors()
    m = ts.model

    # REQUIRED GATE 1 (10D item C): currently leaks Any through
    # `permutedims(seqvocab.storage::Any)`. Flip to @inferred when the
    # §CIX storage re-encoding packet lands.
    @test_broken (@inferred Gesso.reference_prefill(m, ts, STAB_PROMPT)) isa Matrix{Float64}

    # behavioral sanity alongside the broken inference gate: the value is
    # still the right shape and dtype — only INFERENCE is blocked, not
    # correctness
    logits = Gesso.reference_prefill(m, ts, STAB_PROMPT)
    @test logits isa Matrix{Float64}

    s = Gesso.Session(m, ts; context_length=128, eos_token_id=2)
    Gesso.prefill!(s, STAB_PROMPT)

    # REQUIRED GATE 2 (10D item C): `decode!` reads `s.h[row, :]::Any` and
    # `Session.model::Any`, so the return infers Any. Flip to @inferred with
    # the same packet.
    @test_broken (@inferred Gesso.decode!(s)) isa Int
    id = Gesso.decode!(s)
    @test id isa Int
    @test id == s.eos_token_id || id in 0:(m.vocab_size-1)
end

@testset "10D item C: llama_micro mirrors toy2 (both tiers)" begin
    dir = mktempdir()
    make_micro_checkpoint(dir)
    lm, lts, lcfg = Gesso.load_llama(dir)
    lm_logits = Gesso.reference_prefill(lm, lts, [0, 1, 2])
    @test lm_logits isa Matrix{Float64}

    ls = Gesso.Session(
        lm,
        lts;
        context_length=32,
        eos_token_id=0,
        eps=lcfg.rms_norm_eps,
        theta=lcfg.rope_theta,
    )
    Gesso.prefill!(ls, [0, 1, 2])
    @test (@inferred Gesso.Profiling.kv_footprint(ls.mgr)) isa Int
    # value path intact (inference leak packeted above, toy2 block)
    @test Gesso.decode!(ls) isa Int
    @test_broken (@inferred Gesso.decode!(ls)) isa Int    # packeted, same leak
end
