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

@testset "receipts (§XLII)" begin
    sink = Harpe.InMemorySink(4)
    ids = UInt64[]
    for i in 1:6
        r = Harpe.new_receipt(task="t$i", output_digest="digest$i")
        push!(ids, r.id)
        Harpe.emit!(sink, r)
    end
    # monotonic ids
    @test all(diff(ids) .> 0)
    # bounded ring keeps the newest, drops the oldest
    @test length(sink.buf) == 4
    @test sink.buf[1].task == "t3"
    @test sink.buf[end].task == "t6"
    @test sink.dropped == UInt64(2)
    # schema stamping
    @test sink.buf[end].schema_version == Harpe.RECEIPT_SCHEMA_VERSION
    # §XLII: receipts carry structured context, never prose-only
    r = Harpe.new_receipt(; context=Dict(:op => :rmsnorm, :backend => :cpu))
    @test r.context[:op] === :rmsnorm
end
