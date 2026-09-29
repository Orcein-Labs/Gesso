# Self-tests for the test helpers (goal §H): the laboratory equipment must
# itself be tested — a tolerance helper with a broken comparison would
# silently certify wrong results.

using .HarpeTestHelpers: deterministic_rng, approx_eq, parity_report, assert_parity

@testset "test helpers: deterministic_rng" begin
    r1 = deterministic_rng()
    r2 = deterministic_rng()
    @test rand(r1) == rand(r2)                    # same default seed ⇒ same stream
    @test rand(deterministic_rng(42)) != rand(deterministic_rng(43))
    @test rand(deterministic_rng(42)) == rand(deterministic_rng(42))
end

@testset "test helpers: approx_eq" begin
    x = [1.0, 2.0, 3.0]

    # default is EXACT equality — one ulp apart must fail
    # (nextfloat, not 3.0 + eps(): that sum is exactly halfway between
    # doubles at 3.0 and rounds back to 3.0)
    @test approx_eq(x, copy(x))
    @test !approx_eq(x, [1.0, 2.0, nextfloat(3.0)])

    # tolerance semantics: |a-b| ≤ atol + rtol·max(|a|,|b|)
    y = [1.0 + 1.0e-12, 2.0, 3.0]
    @test approx_eq(x, y; rtol=1.0e-9)
    @test !approx_eq(x, y; rtol=1.0e-13, atol=0)
    @test approx_eq(x, y; rtol=1.0e-13, atol=1.0e-12)  # atol floor catches it
    @test !approx_eq(x, y; atol=1.0e-13)               # atol alone is too tight

    # NaN/Inf semantics: isequal short-circuit
    n = [NaN, Inf, -0.0]
    @test approx_eq(n, [NaN, Inf, 0.0])            # NaN==NaN, -0.0 passes 0.0
    @test !approx_eq(n, [NaN, -Inf, -0.0])         # Inf vs -Inf fails
    @test !approx_eq(n, [1.0, Inf, -0.0])          # NaN vs finite fails
end

@testset "test helpers: type strictness" begin
    # no silent float-width conversion: Float32 vs Float64 is a decision a
    # test must make explicitly
    @test_throws MethodError approx_eq([1.0f0], [1.0])
end

@testset "test helpers: parity_report + assert_parity" begin
    a = [1.0, 2.0, 3.0]
    @test parity_report(a, copy(a)).kind == :exact
    @test parity_report(a, copy(a)).agree

    b = [1.0, 2.0 + 1.0e-12, 3.0]
    rep = parity_report(a, b; rtol=1.0e-9)
    @test rep.kind == :tolerance    # agrees, not exact
    @test rep.agree
    @test rep.n_mismatch == 0
    @test rep.n_exact == 2

    c = [1.0, 2.0 + 0.5, 3.0]
    rep = parity_report(a, c; rtol=1.0e-9, name=:probe)
    @test rep.kind == :mismatch
    @test !rep.agree
    @test rep.n_mismatch == 1
    @test rep.first_mismatch == 2
    @test rep.max_abs_diff == 0.5
    @test rep.rel_at_max ≈ 0.5 / 2.5
    @test rep.name == :probe

    # shape mismatch is reported, not thrown
    rep = parity_report(a, [1.0, 2.0])
    @test rep.kind == :shape_mismatch
    @test !rep.agree

    # assert_parity returns the report on agreement, throws with context on failure
    @test assert_parity(parity_report(a, copy(a))) isa NamedTuple
    err = try
        assert_parity(parity_report(a, c; name=:oracle_t1); context=(case=3, backend=:cpu))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("oracle_t1", err.msg)
    @test occursin("context        = (case = 3, backend = :cpu)", err.msg)
end
