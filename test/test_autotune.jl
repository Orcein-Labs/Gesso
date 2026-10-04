# Phase 9 item A tests (§LXXXII): the Autotune loop — register, search with
# two pure-Julia candidates, correctness gate, cache hit, invalidate, and the
# typed all-fail throw. Core machinery: ALWAYS runs, no device, no backends
# (Autotune.jl imports neither CUDA nor Lava).

using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

@testset "autotune: AUTOTUNE_CACHE_VERSION exported at v0.1.0 (§LXIX)" begin
    @test Gesso.AUTOTUNE_CACHE_VERSION == v"0.1.0"
    @test :AUTOTUNE_CACHE_VERSION in names(Gesso)
end

@testset "autotune: search! picks a winner among two passing candidates" begin
    Gesso.Autotune.invalidate_all!()
    count1 = Ref(0)
    count2 = Ref(0)
    impl1(b) = (b[] += 1; b)
    impl2(b) = (b[] += 1; b)
    c1 = Gesso.Autotune.Candidate(
        :fast_add,
        b -> (count1[] += 1; impl1(b)),
        b -> (s=Ref(0); impl1(s); s[] == 1),
    )
    c2 = Gesso.Autotune.Candidate(
        :slow_add,
        b -> (count2[] += 1; impl2(b)),
        b -> (s=Ref(0); impl2(s); s[] == 1),
    )
    Gesso.Autotune.register!(:demo, :cpu, c1)
    Gesso.Autotune.register!(:demo, :cpu, c2)

    buf = Ref(0)
    result = Gesso.Autotune.search!(:demo, :cpu, :r1, "testdev", buf)
    @test result isa Gesso.Autotune.TuneResult
    @test result.winner in (:fast_add, :slow_add)
    @test result.cache_hit == false
    @test haskey(result.medians, :fast_add) && haskey(result.medians, :slow_add)
    @test isempty(result.rejected)
    @test result.op === :demo && result.backend === :cpu && result.regime === :r1
    @test buf[] > 0                                  # candidates ran on the live args
    # NOTE: the registration-order tie-break only fires on an EXACT median
    # tie; two identically-costed candidates under real timing noise select
    # either way — and that is the contract ("the point is that selection
    # happened", §LXXXII), not a specific winner.
end

@testset "autotune: a gate-failing candidate is rejected, never the winner" begin
    Gesso.Autotune.invalidate_all!()
    good = Gesso.Autotune.Candidate(
        :good_add,
        b -> (b[] += 1; b),
        b -> (s=Ref(0); s[] += 1; s[] == 1),
    )
    bad = Gesso.Autotune.Candidate(
        :bad_add,
        b -> (b[] += 100; b),                        # runs, but the gate rejects it
        b -> false,
    )
    Gesso.Autotune.register!(:demo2, :cpu, good)
    Gesso.Autotune.register!(:demo2, :cpu, bad)

    buf = Ref(0)
    result = Gesso.Autotune.search!(:demo2, :cpu, :r1, "testdev", buf)
    @test result.winner === :good_add
    @test length(result.rejected) == 1
    @test result.rejected[1][1] === :bad_add
    @test !haskey(result.medians, :bad_add)          # medians only carry passing candidates
    @test haskey(result.medians, :good_add)
end

@testset "autotune: all candidates failing the gate throws typed ERR_VERIFY_MISMATCH (§LXX)" begin
    Gesso.Autotune.invalidate_all!()
    bad1 = Gesso.Autotune.Candidate(:nope1, b -> b, b -> false)
    bad2 = Gesso.Autotune.Candidate(:nope2, b -> error("impl exploded"), b -> true)
    Gesso.Autotune.register!(:demo3, :cpu, bad1)
    Gesso.Autotune.register!(:demo3, :cpu, bad2)

    err = try
        Gesso.Autotune.search!(:demo3, :cpu, :r1, "testdev", Ref(0))
        nothing
    catch e
        e
    end
    @test err isa Gesso.GessoError
    @test err.code == Gesso.ERR_VERIFY_MISMATCH
    @test occursin("no legal lowering", sprint(showerror, err))
end

@testset "autotune: select caches (same result object, no re-bench), invalidate! forces re-search" begin
    Gesso.Autotune.invalidate_all!()
    count = Ref(0)
    c = Gesso.Autotune.Candidate(
        :counted_add,
        b -> (count[] += 1; b[] += 1; b),
        b -> (s=Ref(0); s[] += 1; s[] == 1),
    )
    Gesso.Autotune.register!(:demo4, :cpu, c)

    sink = Gesso.InMemorySink()
    buf = Ref(0)
    first_result = Gesso.Autotune.select(:demo4, :cpu, :r1, "testdev", buf; sink)
    after_search = count[]

    second_result = Gesso.Autotune.select(:demo4, :cpu, :r1, "testdev", buf; sink)
    @test count[] == after_search                   # no re-bench on a hit

    # 10G: a hit is not a decision — ONE receipt (the miss), not two.
    receipts = sink.buf                              # InMemorySink exposes the live buffer
    @test length(receipts) == 1
    @test receipts[1].context[:cache_hit] == false
    @test receipts[1].context[:winner] === :counted_add
    @test receipts[1].task === :autotune_select

    # `cache_hit` stays the consult-site signal, on the RETURN VALUE. The two
    # results are not `===` (TuneResult is non-isbits, so `===` is field-wise
    # and the flipped field differs) — but every decision-bearing field is the
    # SAME OBJECT, not a copy.
    @test first_result.cache_hit == false
    @test second_result.cache_hit == true
    @test second_result !== first_result
    @test second_result.winner === first_result.winner
    @test second_result.medians === first_result.medians
    @test second_result.rejected === first_result.rejected
    @test second_result.key === first_result.key
    # the STORED entry is the search record and keeps cache_hit = false
    @test Gesso.Autotune.cached_result(:demo4, :cpu, :r1; device="testdev").cache_hit ==
          false

    dropped = Gesso.Autotune.invalidate!(:demo4, :cpu, :r1)
    @test dropped == 1
    third_result = Gesso.Autotune.select(:demo4, :cpu, :r1, "testdev", buf; sink)
    @test third_result !== first_result             # re-searched
    @test count[] > after_search                    # candidates ran again
    @test third_result.cache_hit == false           # invalidate forces a MISS
    @test length(sink.buf) == 2                     # ... and a miss DOES emit
    @test Gesso.Autotune.cached_result(:demo4, :cpu, :r1) === third_result

    @test Gesso.Autotune.invalidate_all!() === nothing
    @test Gesso.Autotune.cached_result(:demo4, :cpu, :r1) === nothing
end

@testset "autotune: search! with no registered candidates throws typed (§LXX)" begin
    Gesso.Autotune.invalidate_all!()
    err = try
        Gesso.Autotune.search!(:never_registered, :cpu, :r1, "testdev", Ref(0))
        nothing
    catch e
        e
    end
    @test err isa Gesso.GessoError
    @test err.code == Gesso.ERR_VERIFY_MISMATCH
    @test occursin("no registered candidates", sprint(showerror, err))
end
