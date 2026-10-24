# Phase 20 (§XCIII; docs/design/SPECULATION_SEAM.md): the host-side draft
# seam and its sources. Ported from the GessoSymbolicDrafts package at commit
# 59553aa (v0.2.0); the assertions are unchanged except where noted below.
#
# Wrapped in a module so the helpers below (bytes, piece, context, replay)
# do not become globals shared with the other per-area test files.
module SpeculationTests

using Test, Random
using Gesso

bytes(text) = Int.(codeunits(text))
piece(id) = String(UInt8[id])
context(text) = DraftContext(bytes(text), piece, 256)

# Simulates the runtime's agreement rule against a FIXED trace. It never
# evaluates a model or claims to implement the multi-token verify kernel.
function replay(source, prompt, target; width=4)
    ids = copy(prompt)
    emitted = Int[]
    i = 1
    windows = 0
    while i <= length(target)
        draft = propose(source, DraftContext(ids, piece, 256); max_tokens=width)
        accepted = 0
        while accepted < min(length(draft.ids), length(target)-i+1) &&
              draft.ids[accepted+1] == target[i+accepted]
            accepted += 1
        end
        count = min(accepted+1, length(target)-i+1)
        committed = target[i:(i+count-1)]
        append!(ids, committed)
        append!(emitted, committed)
        observe!(source, draft, accepted)
        i += count
        windows += 1
    end
    emitted, windows
end

@testset "Token-source seam and ownership" begin
    @test fieldnames(DraftContext) == (:ids, :detok, :vocab_size)
    @test fieldnames(Draft) == (:ids, :origin)
    suffix = SuffixSource()
    @test propose(suffix, DraftContext(Int[], piece, 256); max_tokens=8).ids == Int[]
    @test propose(suffix, DraftContext([0, 255, 0], piece, 256); max_tokens=2).ids ==
          [255, 0]
    @test_throws ArgumentError propose(suffix, context("a"); max_tokens=-1)
    @test_throws ArgumentError propose(
        suffix,
        DraftContext([256], piece, 256);
        max_tokens=1,
    )
    @test_throws ArgumentError propose(suffix, DraftContext([-1], piece, 256); max_tokens=1)
    @test_throws ArgumentError propose(suffix, DraftContext([0], piece, 0); max_tokens=1)
    @test_throws ArgumentError SuffixSource(max_match=0)
    ids = [1, 2, 3, 1, 2]
    d = propose(suffix, DraftContext(ids, piece, 256); max_tokens=3)
    @test d.ids == [3, 1, 2]
    ids[3] = 9
    @test d.ids == [3, 1, 2]
    d.ids[1] = 99
    @test ids[3] == 9
    @test propose(suffix, context("abcdef"); max_tokens=8).ids == Int[]
    @test propose(SuffixSource(max_history=2), context("abcabc"); max_tokens=8).ids == Int[]
    @test propose(suffix, context("abXabYab"); max_tokens=1).ids == bytes("Y")
    # min_match: a one-token match no longer drafts, a long enough one still does
    @test propose(SuffixSource(min_match=2), context("aXba"); max_tokens=2).ids == Int[]
    @test propose(SuffixSource(min_match=2), context("abXab"); max_tokens=1).ids ==
          bytes("X")
    @test_throws ArgumentError SuffixSource(min_match=0)
    @test_throws ArgumentError SuffixSource(min_match=5, max_match=4)
end

@testset "Templates: partial re-entry and specificity" begin
    trigger, continuation = bytes("status: "), bytes("ready\n")
    rule = TemplateRule(trigger, continuation)
    src = TemplateSource([rule])
    trigger[1] = 0
    continuation[1] = 0
    @test propose(src, context("status: "); max_tokens=2).ids == bytes("re")
    @test propose(src, context("status: re"); max_tokens=20).ids == bytes("ady\n")
    @test propose(src, context("status: ready\n"); max_tokens=20).ids == Int[]
    @test propose(src, context("status: wrong"); max_tokens=20).ids == Int[]
    @test propose(src, context("status:"); max_tokens=20).ids == Int[]
    @test_throws ArgumentError TemplateRule(Int[], [1])
    @test_throws ArgumentError TemplateRule([1], [-1])
    @test_throws ArgumentError TemplateSource([rule, rule])
    @test_throws ArgumentError TemplateSource(TemplateRule[])
    bad = TemplateSource([TemplateRule([1], [256])])
    @test_throws ArgumentError propose(bad, DraftContext([1], piece, 256); max_tokens=1)
    specific = TemplateSource([TemplateRule([1], [10]), TemplateRule([2, 1], [20])])
    @test propose(specific, DraftContext([2, 1], piece, 256); max_tokens=8).ids == [20]
    @test replay(src, bytes("status: "), bytes("ready\n"))[1] == bytes("ready\n")
end

@testset "Bounded exact integer procedure" begin
    proc = IntegerArithmetic()
    for (expr, value) in [
        ("1+2*3", 7),
        ("(1+2)*3", 9),
        ("-(-5)+2", 7),
        ("4-9", -5),
        ("0*999", 0),
        ("7 + 8 * 2", 23),
    ]
        @test complete_text(proc, "calc: $expr = ") == string(value)
    end
    @test complete_text(proc, "calc: 100+23 = 1") == "23"
    @test complete_text(proc, "calc: 100+23 = 123") == ""
    @test complete_text(proc, "calc: 100+23 = 9") === nothing
    @test complete_text(proc, "note🙂\ncalc: 2+3 = ") == "5"
    for expr in ["evil()", "1/2", "2^3", "x+1", "1;exit()", "1+", "(1+2", "1 2", "1.0+2"]
        @test complete_text(proc, "calc: $expr = ") === nothing
    end
    @test complete_text(proc, "The answer is ") === nothing
    @test complete_text(IntegerArithmetic(max_bits=8), "calc: 255+1 = ") === nothing
    @test complete_text(IntegerArithmetic(max_depth=2), "calc: (((1))) = ") === nothing
    @test complete_text(IntegerArithmetic(max_nodes=2), "calc: 1+2+3 = ") === nothing
    @test complete_text(IntegerArithmetic(max_expression_bytes=3), "calc: 1234 = ") ===
          nothing
    @test_throws ArgumentError IntegerArithmetic(max_bits=0)
    rng = MersenneTwister(1028)
    for _ in 1:250
        a, b, c, d = rand(rng, -100000:100000, 4)
        expected = (BigInt(a)+b)*(BigInt(c)-d)
        @test complete_text(proc, "calc: ($a+$b)*($c-$d) = ") == string(expected)
    end
end

@testset "Symbolic encoding and fail-closed boundaries" begin
    src = SymbolicSource(IntegerArithmetic(), bytes)
    @test propose(src, context("calc: 12*12 = "); max_tokens=2).ids == bytes("14")
    @test propose(src, context("calc: 12*12 = 14"); max_tokens=8).ids == bytes("4")
    @test propose(src, context("calc: 12*12 = 144"); max_tokens=8).ids == Int[]
    @test propose(src, context("calc: 12*12 = 9"); max_tokens=8).ids == Int[]
    @test propose(src, context("calc: evil() = "); max_tokens=8).ids == Int[]
    @test propose(src, context("calc: 2+3 = "); max_tokens=0).ids == Int[]
    @test propose(
        SymbolicSource(IntegerArithmetic(), bytes; max_context_tokens=3),
        context("calc: 1+1 = ");
        max_tokens=8,
    ).ids == Int[]
    @test propose(
        SymbolicSource(IntegerArithmetic(), bytes; max_text_bytes=4),
        context("calc: 1+1 = ");
        max_tokens=8,
    ).ids == Int[]
    @test_throws ArgumentError SymbolicSource(IntegerArithmetic(), bytes; max_draft=0)
    # A tokenizer that merges the final prefix space with the first answer
    # digit forces a rewrite of an already committed token: decline.
    merge_encode =
        text -> endswith(text, " 2") ? vcat(bytes(text[1:(end-2)]), 256) : bytes(text)
    merged = SymbolicSource(IntegerArithmetic(), merge_encode)
    ctx = DraftContext(bytes("calc: 1+1 = "), piece, 257)
    @test propose(merged, ctx; max_tokens=8).ids == Int[]
    # Special/noncanonical prefix text cannot silently lose its ID identity.
    special_ids = vcat(256, bytes(": 1+1 = "))
    special_detok = id -> id == 256 ? "calc" : piece(id)
    @test propose(src, DraftContext(special_ids, special_detok, 257); max_tokens=8).ids ==
          Int[]
    bad_encode =
        text -> endswith(text, " 2") ? vcat(bytes(text[1:(end-1)]), 999) : bytes(text)
    @test_throws ArgumentError propose(
        SymbolicSource(IntegerArithmetic(), bad_encode),
        context("calc: 1+1 = ");
        max_tokens=8,
    )
    bad_detok = id -> 42
    @test_throws ArgumentError propose(src, DraftContext([1], bad_detok, 256); max_tokens=8)
    original = bytes("calc: 1+1 = ")
    result = propose(src, DraftContext(original, piece, 256); max_tokens=8)
    original[end] = 0
    @test result.ids == bytes("2")
end

@testset "Agreement is separate from mathematical truth" begin
    source = SymbolicSource(IntegerArithmetic(), bytes)
    # The target fixture intentionally disagrees with the correct arithmetic.
    # Runtime acceptance still follows that fixed target, never our answer.
    emitted, windows = replay(source, bytes("calc: 2+2 = "), bytes("5!"))
    @test emitted == bytes("5!")
    @test windows == 2
    @test feedback_counts(source).accepted == 0
    source = SymbolicSource(IntegerArithmetic(), bytes)
    emitted, windows = replay(source, bytes("calc: 123*456 = "), bytes("56088"))
    @test emitted == bytes("56088")
    @test windows == 1
    @test feedback_counts(source).accepted == 4
end

@testset "Feedback: bounded, deterministic and concurrent counts" begin
    src = SuffixSource()
    d = Draft([1, 2], :suffix)
    @test observe!(src, d, 1) === nothing
    @test feedback_counts(src) == (windows=1, drafted=2, accepted=1)
    @test_throws ArgumentError observe!(src, d, -1)
    @test_throws ArgumentError observe!(src, d, 3)
    @test_throws ArgumentError observe!(src, Draft([1], :template), 1)
    @test feedback_counts(src) == (windows=1, drafted=2, accepted=1)
    @sync for _ in 1:4
        Threads.@spawn for _ in 1:25
            observe!(src, d, 2)
        end
    end
    @test feedback_counts(src) == (windows=101, drafted=202, accepted=201)
    for factory in [
        () -> SuffixSource(),
        () -> TemplateSource([TemplateRule(bytes("calc: 1+1 = "), bytes("2"))]),
        () -> SymbolicSource(IntegerArithmetic(), bytes),
    ]
        a, b = factory(), factory()
        for prefix in ["abcabc", "calc: 1+1 = ", "calc: 1+1 = 2", "nonsense"]
            da = propose(a, context(prefix); max_tokens=8)
            db = propose(b, context(prefix); max_tokens=8)
            @test da.ids == db.ids
            observe!(a, da, 0)
            observe!(b, db, 0)
        end
        @test feedback_counts(a) == feedback_counts(b)
    end
end

# Independent declaration of Claude's ABI: source include must extend THESE
# types, not import the lab's type identities.
module RuntimePort
abstract type DraftSource end
struct DraftContext
    ids::AbstractVector{Int}
    detok::Any
    vocab_size::Int
end
struct Draft
    ids::Vector{Int}
    origin::Symbol
end
function propose end
function observe! end
include(joinpath(@__DIR__, "../src/Speculation/HostSources.jl"))
end
@testset "Include-based runtime port, no lab ABI leakage" begin
    @test RuntimePort.SuffixSource <: RuntimePort.DraftSource
    @test !(RuntimePort.SuffixSource <: Gesso.Speculation.DraftSource)
    @test SuffixSource <: DraftSource
    src = RuntimePort.SymbolicSource(RuntimePort.IntegerArithmetic(), bytes)
    ctx = RuntimePort.DraftContext(bytes("calc: 9*9 = "), piece, 256)
    d = RuntimePort.propose(src, ctx; max_tokens=8)
    @test d isa RuntimePort.Draft
    @test d.ids == bytes("81")
    @test RuntimePort.observe!(src, d, 1) === nothing
end

@testset "Explicit bounded last-line context" begin
    src =
        SymbolicSource(LastLineProcedure(IntegerArithmetic()), bytes; max_context_tokens=32)
    full = SymbolicSource(IntegerArithmetic(), bytes; max_context_tokens=32)
    prefix=repeat("old context ", 10000)*"\ncalc: 12*12 = "
    @test propose(src, context(prefix); max_tokens=2).ids == bytes("14")
    @test isempty(propose(full, context(prefix); max_tokens=2).ids)
    @test isempty(
        propose(src, context(repeat("x", 100)*"calc: 12*12 = "); max_tokens=2).ids,
    )
    @test propose(src, context("calc: 12*12 = 14"); max_tokens=2).ids == bytes("4")
    @test isempty(propose(src, context("old\ncalc: 12*12 = 9"); max_tokens=2).ids)
    special = vcat(256, bytes("\ncalc: 1+1 = "))
    special_detok = id -> id==256 ? "<bos>" : piece(id)
    @test propose(src, DraftContext(special, special_detok, 257); max_tokens=2).ids ==
          bytes("2")
    @test isempty(
        propose(full, DraftContext(special, special_detok, 257); max_tokens=2).ids,
    )
    # A newline inside a token selects the whole token, preserving its ID.
    dictionary=["old\ncalc: ", "1+1", " = ", "2"]
    encoder =
        text ->
            text=="old\ncalc: 1+1 = " ? [0, 1, 2] :
            text=="old\ncalc: 1+1 = 2" ? [0, 1, 2, 3] : Int[]
    inside=SymbolicSource(LastLineProcedure(IntegerArithmetic()), encoder)
    @test propose(inside, DraftContext([0, 1, 2], id->dictionary[id+1], 4); max_tokens=1).ids==[
        3,
    ]
    # A noncanonical selected span still refuses; no rewrite permission added.
    @test isempty(
        propose(
            src,
            DraftContext(
                vcat(bytes("\n"), 256, bytes(": 1+1 = ")),
                id->id==256 ? "calc" : piece(id),
                257,
            );
            max_tokens=2,
        ).ids,
    )
    tiny=SymbolicSource(LastLineProcedure(IntegerArithmetic()), bytes; max_text_bytes=4)
    @test isempty(propose(tiny, context("old\ncalc: 1+1 = "); max_tokens=2).ids)
    bad=id->nothing
    @test_throws ArgumentError propose(src, DraftContext([1], bad, 256); max_tokens=2)
    @test_throws ArgumentError propose(src, DraftContext([256], piece, 256); max_tokens=2)
    @test isempty(propose(src, context(""); max_tokens=2).ids)
    # Full and last-line paths concatenate byte fragments before validation.
    unicode="π\ncalc: 1+1 = "
    @test propose(
        SymbolicSource(IntegerArithmetic(), bytes),
        context(unicode);
        max_tokens=2,
    ).ids==bytes("2")
    @test propose(src, context(unicode); max_tokens=2).ids==bytes("2")
    ids=bytes("old\ncalc: 1+1 = ")
    saved=copy(ids)
    draft=propose(src, DraftContext(ids, piece, 256); max_tokens=2)
    @test ids==saved
    ids[end]=0
    @test draft.ids==bytes("2")
    # The scan never touches earlier special-token callbacks or distant history.
    seen=Int[]
    detok=id->begin
        push!(seen, id)
        piece(id)
    end
    @test propose(src, DraftContext(special, detok, 257); max_tokens=2).ids==bytes("2")
    @test 256 ∉ seen
end

@testset "Bounded sequence hypotheses" begin
    proc=IntegerSequence()
    for (series, answer) in (
        ("1,4,9,16", "25"),
        ("2,4,6,8", "10"),
        ("7,7,7,7", "7"),
        ("1,1,2,3,5", "8"),
        ("8,5,2,-1", "-4"),
        ("1,-1,0,-1,-1", "-2"),
    )
        @test complete_text(proc, "seq: "*series*" -> ")==answer
    end
    @test complete_text(proc, "seq: 1,4,9,16 -> 2")=="5"
    @test complete_text(proc, "seq: 1,4,9,16 -> 25")==""
    @test complete_text(proc, "seq: 1,4,9,16 -> 3")===nothing
    for series in ("1,2,3", "1,4,9,15", "1,2,nope,4", "1,2,3,4;evil()")
        @test complete_text(proc, "seq: "*series*" -> ")===nothing
    end
    @test complete_text(IntegerSequence(max_bits=4), "seq: 4,8,12,16 -> ")===nothing
    @test complete_text(IntegerSequence(max_bits=4), "seq: 3,6,9,12 -> ")=="15"
    @test complete_text(IntegerSequence(max_bits=4), "seq: 4,7,10,13 -> ")===nothing
    @test complete_text(IntegerSequence(max_terms=4), "seq: 1,1,2,3,5 -> ")===nothing
    @test complete_text(IntegerSequence(max_input_bytes=4), "seq: 1,2,3,4 -> ")===nothing
    @test_throws ArgumentError IntegerSequence(min_terms=3)
    rng=MersenneTwister(1702)
    for _ in 1:100
        a, b, c=rand(rng, -20:20, 3)
        values=BigInt[a*n*n+b*n+c for n in 1:6]
        prefix="seq: "*join(values, ",")*" -> "
        @test complete_text(proc, prefix)==string(BigInt(a)*49+BigInt(b)*7+c)
    end
    for _ in 1:50
        values=BigInt.(rand(rng, -20:20, 2))
        for _ in 1:5
            push!(values, values[end]+values[end-1])
        end
        @test complete_text(proc, "seq: "*join(values, ",")*" -> ")==string(
            values[end]+values[end-1],
        )
    end
    src=SymbolicSource(LastLineProcedure(proc), bytes; origin=:sequence)
    @test propose(src, context("old\nseq: 1,4,9,16 -> "); max_tokens=1).ids==bytes("2")
    output, windows=replay(src, bytes("seq: 1,4,9,16 -> "), bytes("banana"))
    @test output==bytes("banana")
    @test feedback_counts(src).accepted==0
    @test RuntimePort.propose(
        RuntimePort.SymbolicSource(
            RuntimePort.LastLineProcedure(RuntimePort.IntegerSequence()),
            bytes,
        ),
        RuntimePort.DraftContext(bytes("seq: 1,4,9,16 -> "), piece, 256);
        max_tokens=2,
    ).ids==bytes("25")
end

end # module SpeculationTests
