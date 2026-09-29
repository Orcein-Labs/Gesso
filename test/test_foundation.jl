# BONES sprint tests: foundation modules (versions, errors, receipts).

@testset "versions (§LXIX)" begin
    @test Harpe.HARPE_SCHEMA_VERSION isa VersionNumber
    @test Harpe.RECEIPT_SCHEMA_VERSION isa VersionNumber
    @test Harpe.BENCH_RESULT_SCHEMA_VERSION isa VersionNumber
    # schema versions track the package version while we are pre-1.0
    @test Harpe.HARPE_SCHEMA_VERSION == v"0.1.0"
end

# Errors tests moved to test_errors.jl (per-area harness convention).
# The LoweringNotImplemented coverage lives in the backend-interface
# testset in runtests.jl.

# Receipt tests moved to test_receipts.jl (per-area harness convention).
# Nothing else tested here yet beyond runtests.jl's package-level laws.
