# Empty-core fence (Phase 1).
#
# Semantics, ModelIR, Parameters, Operators are §CIX modules: the fence
# tracks what each module may contain as Phase 1 items land. After item A,
# Semantics/Parameters carry exactly the §CIX vocabulary; ModelIR/Operators
# stay empty until items B and C. This file is what the next eager agent
# hits when they "just add a struct": the fence fails, and the failure
# message says where the decision actually belongs.
#
# It also pins two deeper fences:
#   * the §LVIII training boundary: no AD/training machinery identifiers
#     anywhere in src/;
#   * architecture packets 1–2 are resolved in canon (§CIX) and stay
#     unimplemented beyond what a landed Phase 1 item added — no
#     ExecutionPhase / WorkloadKind mega-enum, no global receipt ids.

const CORE_MODULES = (:Semantics, :ModelIR, :Parameters, :Operators)

# Exports each filled module is allowed after Phase 1 item A.
const ITEM_A_EXPORTS = Dict(
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
)
const STILL_EMPTY_MODULES = (:ModelIR, :Operators)

@testset "empty-core fence: filled modules export exactly the §CIX item-A vocabulary" begin
    for (name, allowed) in ITEM_A_EXPORTS
        m = getfield(Harpe, name)
        @test m isa Module
        exported = setdiff(names(m), [name])
        @test Set(exported) == Set(allowed) ||
              "module $name exports $exported — " *
              "item A pins exactly $allowed (§CIX); extending it is a work item" == ""
    end
end

@testset "empty-core fence: ModelIR/Operators stay empty until items B/C" begin
    for name in STILL_EMPTY_MODULES
        m = getfield(Harpe, name)
        exported = setdiff(names(m), [name])
        @test isempty(exported) ||
              "module $name exports $exported — it stays " *
              "empty until its Phase 1 item (B: ModelIR, C: Operators) lands" == ""
    end
end

@testset "empty-core fence: training boundary in src/ (§LVIII)" begin
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
    src_root = joinpath(pkgdir(Harpe), "src")
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

@testset "empty-core fence: packets resolved in canon, unimplemented in code" begin
    @test isfile(joinpath(pkgdir(Harpe), "docs", "DECISION_PACKETS.md"))
    md = read(joinpath(pkgdir(Harpe), "docs", "DECISION_PACKETS.md"), String)
    @test occursin("## Status", md)
    @test occursin("RESOLVED INTO CANON", md)

    # Packet 1 (§CIX): the two-level law. The dispatch types exist (item A
    # landed them); no mega-enum does, and the singleton cuts are distinct.
    @test !isdefined(Harpe, :ExecutionPhase)
    @test !isdefined(Harpe, :WorkloadKind)
    @test isdefined(Harpe, :PrefillWorkload)
    @test isdefined(Harpe, :DecodeWorkload)
    @test typeof(Harpe.PrefillWorkload()) !== typeof(Harpe.DecodeWorkload())

    # Packet 2 (§CIX / §XLII): ids stay process-local UInt64 until the
    # persistence/swarm schema bump.
    @test Harpe.next_receipt_id() isa UInt64
end
