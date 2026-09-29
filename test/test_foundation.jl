# BONES sprint tests: foundation modules (versions, errors, receipts).

@testset "versions (§LXIX)" begin
    @test Harpe.HARPE_SCHEMA_VERSION isa VersionNumber
    @test Harpe.RECEIPT_SCHEMA_VERSION isa VersionNumber
    @test Harpe.BENCH_RESULT_SCHEMA_VERSION isa VersionNumber
    # schema versions track the package version while we are pre-1.0
    @test Harpe.HARPE_SCHEMA_VERSION == v"0.1.0"
end

@testset "errors (§LXX taxonomy)" begin
    e = Harpe.harpe_error(Harpe.ERR_RESOURCE_LIMIT, "budget exhausted"; what=:search)
    @test e isa Harpe.HarpeException
    @test e.code === Harpe.ERR_RESOURCE_LIMIT
    @test e.detail[:what] === :search
    @test occursin("HarpeError", sprint(showerror, e))

    # §LXX: lowering stubs still fail explicitly after the errors.jl move
    cpu = Harpe.CPUBackend()
    @test_throws Harpe.LoweringNotImplemented Harpe.rmsnorm!(cpu, nothing)
    @test_throws Harpe.HarpeException Harpe.matmul!(cpu, nothing)

    # the approximation-budget entry exists (KV program §6)
    @test Harpe.ERR_APPROXIMATION_BUDGET_EXCEEDED isa Harpe.ErrorCode
end

# Receipt tests moved to test_receipts.jl (per-area harness convention).
# Nothing else tested here yet beyond runtests.jl's package-level laws.
