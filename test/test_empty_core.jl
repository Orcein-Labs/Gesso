# Empty-core fence (Phase 1 ready-room law).
#
# Semantics, ModelIR, Parameters, Operators are CONTRACT-ONLY modules: they
# export nothing and define nothing until their owning phase fills them
# (docs/ARCHITECTURE.md; §LXXIII exit rule — no speculative implementation).
# This file is the fence the next eager agent hits when they "just add a
# struct": the fence fails, and the failure message says where the decision
# actually belongs.
#
# It also pins two deeper fences:
#   * the §LVIII training boundary: no AD/training machinery identifiers
#     anywhere in src/;
#   * architecture packets 1–2 are resolved in canon (§CIX) and stay
#     unimplemented in code until a Phase 1 work item fills them.

const CORE_MODULES = (:Semantics, :ModelIR, :Parameters, :Operators)

@testset "empty-core fence: contract-only modules stay empty" begin
    for name in CORE_MODULES
        m = getfield(Harpe, name)
        @test m isa Module
        exported = setdiff(names(m), [name])
        @test isempty(exported) ||
              "module $name exports $exported — it is " *
              "contract-only until its phase fills it (docs/ARCHITECTURE.md); " *
              "if a real phase fills it, update this fence WITH that phase" == ""

        # nothing type-like or callable is defined inside, not even unexported
        additions = Symbol[]
        for n in names(m; all=true)
            n in (:eval, :include, name) && continue
            isdefined(m, n) || continue
            v = getfield(m, n)
            (v isa Module) && continue
            push!(additions, n)
        end
        @test isempty(additions) ||
              "module $name defines $additions — the " *
              "semantic object model is an open architecture decision " *
              "(docs/DECISION_PACKETS.md); do not answer it with a struct" == ""
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

    # Packet 1 (§CIX): PrefillWorkload / DecodeWorkload are the types; they
    # land with a Phase 1 work item. No mega-enum in the meantime.
    @test !isdefined(Harpe, :ExecutionPhase)
    @test !isdefined(Harpe, :WorkloadKind)
    @test !isdefined(Harpe, :PrefillWorkload)
    @test !isdefined(Harpe, :DecodeWorkload)

    # Packet 2 (§CIX / §XLII): ids stay process-local UInt64 until the
    # persistence/swarm schema bump.
    @test Harpe.next_receipt_id() isa UInt64
end
