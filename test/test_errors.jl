# Failure taxonomy tests (§LXX; North Star §22): every code has stable
# persisted identity, is constructible through GessoError, and displays
# sanely. Intended producers are documented in src/errors.jl (the taxonomy
# file itself). No speculative codes — the count below is deliberate.
#
# Moved from test_foundation.jl per the per-area harness convention.

@testset "errors (§LXX): code identity is stable" begin
    # Persisted identity: failure records classify (and may persist) by code
    # value (§LXIX). Renumbering, reordering, or inserting codes silently is
    # a compatibility break — this list pins them.
    codes = [
        (Gesso.ERR_INTERNAL, 0),
        (Gesso.ERR_INVALID_PLAN, 1),
        (Gesso.ERR_CONSTRAINT_REJECTED, 2),
        (Gesso.ERR_COMPILE, 3),
        (Gesso.ERR_RESOURCE_LIMIT, 4),
        (Gesso.ERR_ALLOCATION, 5),
        (Gesso.ERR_LAUNCH, 6),
        (Gesso.ERR_RUNTIME, 7),
        (Gesso.ERR_TIMEOUT, 8),
        (Gesso.ERR_VERIFY_MISMATCH, 9),
        (Gesso.ERR_NUMERICAL_INSTABILITY, 10),
        (Gesso.ERR_BENCHMARK, 11),
        (Gesso.ERR_CACHE, 12),
        (Gesso.ERR_APPROXIMATION_BUDGET_EXCEEDED, 13),
    ]
    for (code, i) in codes
        @test Int(code) == i
    end
    # exhaustive: every instance is pinned above; nothing untested
    @test length(instances(Gesso.ErrorCode)) == length(codes)

    # identity doubles as the display name — logs and receipts render it
    # verbatim, so the spelling is API
    @test string(Gesso.ERR_CACHE) == "ERR_CACHE"
    @test string(Gesso.ERR_APPROXIMATION_BUDGET_EXCEEDED) ==
          "ERR_APPROXIMATION_BUDGET_EXCEEDED"
end

@testset "errors (§LXX): every code constructs and classifies" begin
    for code in instances(Gesso.ErrorCode)
        e = Gesso.gesso_error(code, "diagnostic"; probe=:x)
        @test e isa Gesso.GessoException
        @test e isa Exception
        @test e.code === code
        @test e.message == "diagnostic"
        @test e.detail[:probe] === :x
        # classification survives into display (receipt-friendly)
        @test occursin(string(code), sprint(showerror, e))
    end
end

@testset "errors (§LXX): display" begin
    e = Gesso.gesso_error(
        Gesso.ERR_VERIFY_MISMATCH,
        "oracle tier-2 mismatch";
        oracle=:logit,
        epsilon=1.0e-6,
    )
    s = sprint(showerror, e)
    @test occursin("GessoError(ERR_VERIFY_MISMATCH)", s)
    @test occursin("oracle tier-2 mismatch", s)
    @test occursin(":oracle => :logit", s)   # structured detail survives

    # without detail: no dangling "detail =" suffix
    e2 = Gesso.gesso_error(Gesso.ERR_TIMEOUT, "budget exhausted")
    s2 = sprint(showerror, e2)
    @test occursin("GessoError(ERR_TIMEOUT)", s2)
    @test occursin("budget exhausted", s2)
    @test !occursin("detail", s2)
end

@testset "errors (§LXX): the approximation-budget entry exists (KV program §6)" begin
    @test Gesso.ERR_APPROXIMATION_BUDGET_EXCEEDED isa Gesso.ErrorCode
    e = Gesso.gesso_error(
        Gesso.ERR_APPROXIMATION_BUDGET_EXCEEDED,
        "exceeded declared ε";
        metric=:logit_kl,
        epsilon=1.0e-6,
        measured=1.0e-4,
    )
    @test e isa Gesso.GessoException
    @test e.detail[:measured] == 1.0e-4
end
