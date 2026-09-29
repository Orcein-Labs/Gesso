# §CIX encoding fence (formerly the empty-core fence).
#
# History: this file began as the ready-room fence (the four §CIX modules
# must stay empty until their items land). Items A, B, C of Phase 1 landed
# exactly the §CIX vocabulary, so the fence now does the opposite job with
# the same spirit: it pins WHAT each module contains — no more, no less —
# so nothing can be added to the semantic core outside a work item.
#
# It also pins two deeper fences:
#   * the §LVIII training boundary: no AD/training machinery identifiers
#     anywhere in src/;
#   * the §CIX decisions stay law: no ExecutionPhase / WorkloadKind
#     mega-enum, receipt ids still process-local UInt64.

const CORE_MODULES = (:Semantics, :ModelIR, :Parameters, :Operators)

# Exports each module is allowed now that items A, B, C have landed.
const PHASE1_EXPORTS = Dict(
    :Semantics => [:PrefillWorkload, :DecodeWorkload],
    :Parameters => [
        :SemanticTensor,
        :ProjectionWeight,
        :KVCache,
        :EmbeddingTable,
        :ExpertWeight,
        :FrozenParameter,
        :QuantizedParameter,
        :Activation,
        :TemporaryWorkspace,
        :RoutingState,
        :DecodeState,
        :AdapterDelta,
        :frozen,
    ],
    :ModelIR => [:Embedding, :RMSNorm, :RoPE, :Attention, :SwiGLU, :Block, :Model],
    :Operators => Symbol[],   # operators are functions owned by backends.jl (§CIX)
)

@testset "§CIX fence: modules export exactly the Phase 1 vocabulary" begin
    for (name, allowed) in PHASE1_EXPORTS
        m = getfield(Gesso, name)
        @test m isa Module
        exported = setdiff(names(m), [name])
        @test Set(exported) == Set(allowed) ||
              "module $name exports $exported — " *
              "the §CIX fence pins exactly $allowed; extending the core is a " *
              "work item, not a drive-by" == ""
    end
end

@testset "§CIX fence: no speculative additions inside the core modules" begin
    # known non-goals (§CIX "WHAT PHASE 1 DOES NOT IMPLEMENT") must not appear
    @test !isdefined(Gesso, :ExecutionPhase)
    @test !isdefined(Gesso, :WorkloadKind)
    @test !isdefined(Gesso, :LlamaModel)
    @test !isdefined(Gesso.ModelIR, :LlamaModel)
    @test !isdefined(Gesso, :Quantized)          # no representation lattice
    @test !isdefined(Gesso.Parameters, :Quantized)
    @test !isdefined(Gesso, :Gradient)
    @test !isdefined(Gesso, :OptimizerState)
    @test !isdefined(Gesso.Parameters, :Gradient)
    @test !isdefined(Gesso.Parameters, :OptimizerState)
end

@testset "§CIX fence: training boundary in src/ (§LVIII)" begin
    # Training is out of scope permanently (§LVIII): no AD machinery in the
    # package source. Scanned live so the fence cannot go stale.
    forbidden = [
        r"\bGradient\b",
        r"\bOptimizerState\b",
        r"\bbackward\b",
        r"\bautodiff\b",
        r"\bZygote\b",
        r"\bEnzyme\b",
    ]
    src_root = joinpath(pkgdir(Gesso), "src")
    violations = Tuple{String, String}[]
    for (dir, _, files) in walkdir(src_root)
        for f in files
            endswith(f, ".jl") || continue
            path = joinpath(dir, f)
            for (i, line) in enumerate(eachline(path))
                # full-line comments are exempt: contract headers that NAME
                # the forbidden identifiers in order to document the
                # prohibition are the law book citing the law, not code
                # (conservative: inline comments after code still trip it)
                startswith(strip(line), "#") && continue
                for pat in forbidden
                    if occursin(pat, line)
                        push!(violations, (relpath(path, src_root) * ":$i", line))
                    end
                end
            end
        end
    end
    @test isempty(violations) ||
          "src/ contains AD/training identifiers " *
          "(§LVIII: training is NOT our problem): $violations" == ""
end

@testset "§CIX fence: packet resolutions stay law in code" begin
    @test isfile(joinpath(pkgdir(Gesso), "docs", "DECISION_PACKETS.md"))
    md = read(joinpath(pkgdir(Gesso), "docs", "DECISION_PACKETS.md"), String)
    @test occursin("## Status", md)
    @test occursin("RESOLVED INTO CANON", md)

    # Packet 1 (§CIX): the two-level law. The dispatch types exist; no
    # mega-enum does, and the singleton cuts are distinct.
    @test !isdefined(Gesso, :ExecutionPhase)
    @test !isdefined(Gesso, :WorkloadKind)
    @test isdefined(Gesso, :PrefillWorkload)
    @test isdefined(Gesso, :DecodeWorkload)
    @test typeof(Gesso.PrefillWorkload()) !== typeof(Gesso.DecodeWorkload())

    # Packet 2 (§CIX / §XLII): ids stay process-local UInt64 until the
    # persistence/swarm schema bump.
    @test Gesso.next_receipt_id() isa UInt64
end
