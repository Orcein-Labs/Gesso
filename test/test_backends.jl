# Operator-stub inventory (Phase 1 ready-room): the lowering vocabulary is
# documented as a LIST, not a type lattice (§CIX: an Operator is a function;
# adding one means adding a function — and acknowledging it here).
#
# The inventory is parsed live from src/backends.jl (same style as the
# dependency-law test parsing Project.toml), so the test cannot drift from
# the source: adding a stub without updating EXPECTED_VOCABULARY fails, and
# so does removing one. Do not add ops "because canon might want them" —
# the vocabulary grows when a phase's work item requires it.

@testset "lowering-stub inventory: vocabulary is pinned" begin
    src = read(joinpath(pkgdir(Gesso), "src", "backends.jl"), String)
    # the stub loop: `for op in ( :name!, :other!, ... )` — a bare symbol
    # ending in `!` inside that block is a vocabulary entry
    # `s` flag: dotall — the formatter lays the tuple out across lines
    block = match(r"for op in \((.*?)\)"s, src, 1)   # offset is positional
    block === nothing && error("could not locate the lowering-stub loop in src/backends.jl")
    parsed = sort!(unique!(Symbol.(m[1] for m in eachmatch(r":(\w+!)", block[1]))))

    EXPECTED_VOCABULARY = sort!([
        :rmsnorm!,
        :rope!,
        :softmax!,
        :swiglu!,
        :matmul!,
        :embedding_lookup!,
        :quantize!,
        :dequantize!,
    ])
    @test parsed == EXPECTED_VOCABULARY ||
          "lowering vocabulary changed: got $parsed, inventory says " *
          "$EXPECTED_VOCABULARY — update this inventory WITH the change and " *
          "its work item, never as a drive-by" == ""
end

@testset "lowering-stub inventory: every stub present and explicitly failing" begin
    cpu = Gesso.CPUBackend()
    for op in (
        :rmsnorm!,
        :rope!,
        :softmax!,
        :swiglu!,
        :matmul!,
        :embedding_lookup!,
        :quantize!,
        :dequantize!,
    )
        @test isdefined(Gesso, op)
        err = try
            getglobal(Gesso, op)(cpu, nothing, nothing)
            nothing
        catch e
            e
        end
        # §LXX: the ONLY legal way to decline work
        @test err isa Gesso.LoweringNotImplemented
        @test err.op === op          # identifies the operation
        @test err.backend === :cpu   # identifies the backend
    end
end

@testset "lowering-stub inventory: supports reflects real CPU coverage" begin
    # Phase 2 (§LXXV): true exactly where a CPU method computes. This fence
    # keeps supports honest in both directions — no silent capability
    # invention, and no false denial of implemented coverage.
    cpu = Gesso.CPUBackend()
    for cap in (
        :rmsnorm,
        :attention,
        :softmax,
        :swiglu_ffn,
        :matmul,
        :embedding_lookup,
        :rope_none,
    )
        @test Gesso.supports(cpu, cap) == true
    end
    @test Gesso.supports(cpu, :quantize) == false
    @test Gesso.supports(cpu, :dequantize) == false
    @test Gesso.supports(cpu, :anything_at_all) == false   # unknown: still false, never throws
end
