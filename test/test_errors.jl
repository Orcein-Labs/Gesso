# Failure taxonomy tests (§LXX; North Star §22): every code has stable
# persisted identity, is constructible through HarpeError, and displays
# sanely. Intended producers are documented in src/errors.jl (the taxonomy
# file itself). No speculative codes — the count below is deliberate.
#
# Moved from test_foundation.jl per the per-area harness convention.

@testset "errors (§LXX): code identity is stable" begin
    # Persisted identity: failure records classify (and may persist) by code
    # value (§LXIX). Renumbering, reordering, or inserting codes silently is
    # a compatibility break — this list pins them.
    codes = [
        (Harpe.ERR_INTERNAL, 0),
        (Harpe.ERR_INVALID_PLAN, 1),
        (Harpe.ERR_CONSTRAINT_REJECTED, 2),
        (Harpe.ERR_COMPILE, 3),
        (Harpe.ERR_RESOURCE_LIMIT, 4),
        (Harpe.ERR_ALLOCATION, 5),
        (Harpe.ERR_LAUNCH, 6),
        (Harpe.ERR_RUNTIME, 7),
        (Harpe.ERR_TIMEOUT, 8),
        (Harpe.ERR_VERIFY_MISMATCH, 9),
        (Harpe.ERR_NUMERICAL_INSTABILITY, 10),
        (Harpe.ERR_BENCHMARK, 11),
        (Harpe.ERR_CACHE, 12),
        (Harpe.ERR_APPROXIMATION_BUDGET_EXCEEDED, 13),
    ]
    for (code, i) in codes
        @test Int(code) == i
    end
    # exhaustive: every instance is pinned above; nothing untested
    @test length(instances(Harpe.ErrorCode)) == length(codes)

    # identity doubles as the display name — logs and receipts render it
    # verbatim, so the spelling is API
    @test string(Harpe.ERR_CACHE) == "ERR_CACHE"
    @test string(Harpe.ERR_APPROXIMATION_BUDGET_EXCEEDED) ==
          "ERR_APPROXIMATION_BUDGET_EXCEEDED"
end

@testset "errors (§LXX): every code constructs and classifies" begin
    for code in instances(Harpe.ErrorCode)
        e = Harpe.harpe_error(code, "diagnostic"; probe=:x)
        @test e isa Harpe.HarpeException
        @test e isa Exception
        @test e.code === code
        @test e.message == "diagnostic"
        @test e.detail[:probe] === :x
        # classification survives into display (receipt-friendly)
        @test occursin(string(code), sprint(showerror, e))
    end
end

@testset "errors (§LXX): display" begin
    e = Harpe.harpe_error(
        Harpe.ERR_VERIFY_MISMATCH,
        "oracle tier-2 mismatch";
        oracle=:logit,
        epsilon=1.0e-6,
    )
    s = sprint(showerror, e)
    @test occursin("HarpeError(ERR_VERIFY_MISMATCH)", s)
    @test occursin("oracle tier-2 mismatch", s)
    @test occursin(":oracle => :logit", s)   # structured detail survives

    # without detail: no dangling "detail =" suffix
    e2 = Harpe.harpe_error(Harpe.ERR_TIMEOUT, "budget exhausted")
    s2 = sprint(showerror, e2)
    @test occursin("HarpeError(ERR_TIMEOUT)", s2)
    @test occursin("budget exhausted", s2)
    @test !occursin("detail", s2)
end

@testset "errors (§LXX): the approximation-budget entry exists (KV program §6)" begin
    @test Harpe.ERR_APPROXIMATION_BUDGET_EXCEEDED isa Harpe.ErrorCode
    e = Harpe.harpe_error(
        Harpe.ERR_APPROXIMATION_BUDGET_EXCEEDED,
        "exceeded declared ε";
        metric=:logit_kl,
        epsilon=1.0e-6,
        measured=1.0e-4,
    )
    @test e isa Harpe.HarpeException
    @test e.detail[:measured] == 1.0e-4
end
