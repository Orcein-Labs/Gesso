# Speculative verify window: committed state must equal sequential decode.
#
# A window runs the anchor and its drafts in one pass. Whatever prefix the
# window commits, the GDN state, conv state, KV rows and hidden row must be the
# bits sequential decode leaves at the same position, and the next plain decode
# step must continue exactly as a cold run does.

const _VW_PROMPT = [1, 3, 4, 5]
const _VW_CTX = 64
# the oracle's continuation is read from a run this long, so drafts can run past
# a short cap in the cap tests
const _VW_LONG = 12

# A scripted draft source. The target is the baseline continuation, so the
# oracle proposes exactly what greedy decode will emit. `wrong` replaces one
# draft token (1-based depth), `allwrong` replaces every one, `overrun` ignores
# the budget the caller passed and proposes `width` tokens regardless.
mutable struct _VWDraft <: Gesso.Speculation.DraftSource
    target::Vector{Int}
    plen::Int
    width::Int
    wrong::Int
    allwrong::Bool
    empty::Bool
    overrun::Bool
    vocab::Int
    log::Vector{Tuple{Int,Int}}   # (drafted, accepted) per window, from observe!
end

function _vw_draft(target, plen, vocab; width=8, wrong=0, allwrong=false, empty=false, overrun=false)
    return _VWDraft(target, plen, width, wrong, allwrong, empty, overrun, vocab, Tuple{Int,Int}[])
end

function Gesso.Speculation.propose(src::_VWDraft, ctx::Gesso.Speculation.DraftContext; max_tokens::Int)
    src.empty && return Gesso.Speculation.Draft(Int[], :vw_empty)
    steps = length(ctx.ids) - src.plen
    n = src.overrun ? src.width : max_tokens
    hi = min(length(src.target), steps + n)
    d = src.target[(steps + 1):hi]
    if src.allwrong
        d = [mod(t + 1, src.vocab) for t in d]
    elseif 1 <= src.wrong <= length(d)
        d[src.wrong] = mod(d[src.wrong] + 1, src.vocab)
    end
    return Gesso.Speculation.Draft(d, :vw_oracle)
end

function Gesso.Speculation.observe!(src::_VWDraft, d::Gesso.Speculation.Draft, accepted::Int)
    push!(src.log, (length(d.ids), accepted))
    return nothing
end

# The GDN fast path, which the window requires, runs on F32 state and weights, as
# the CPU GGUF models do. The hybrid fixture is F64 end to end, so its tensors are
# converted to F32 for the window cases. Structure and values are kept.
_vw_f32(a::AbstractArray{Float64}) = Float32.(a)
_vw_f32(w::Gesso.ProjectionWeight) = Gesso.ProjectionWeight(; shape=w.shape, storage=_vw_f32(w.storage))
_vw_f32(w::Gesso.FrozenParameter) = Gesso.FrozenParameter(; shape=w.shape, storage=_vw_f32(w.storage))
_vw_f32(w::Gesso.EmbeddingTable) = Gesso.EmbeddingTable(; shape=w.shape, storage=_vw_f32(w.storage))
_vw_f32(x::NamedTuple) = map(_vw_f32, x)
_vw_f32(x::Tuple) = map(_vw_f32, x)
_vw_f32(x) = x

_vw_hybrid_f32() = _vw_f32(_hybrid_toy_tensors(toy2_tensors()))

# Every field the next decode reads, compared with `==` (bitwise for these
# Float32 arrays). Attention blocks have no GDN state and GDN blocks have no KV
# rows, so the absent side is skipped by its shape.
function _vw_state_diff(A::GenerateState, B::GenerateState)
    diffs = String[]
    A.ids == B.ids || push!(diffs, "ids")
    A.h.storage == B.h.storage || push!(diffs, "h")
    for bi in eachindex(A.gdn_rec)
        A.gdn_rec[bi] == B.gdn_rec[bi] || push!(diffs, "gdn_rec[$bi]")
        A.gdn_conv[bi] == B.gdn_conv[bi] || push!(diffs, "gdn_conv[$bi]")
    end
    n = length(A.ids)
    for bi in eachindex(A.kc)
        size(A.kc[bi].storage, 1) >= n || continue
        A.kc[bi].storage[1:n, :, :] == B.kc[bi].storage[1:n, :, :] || push!(diffs, "kc[$bi]")
        A.vc[bi].storage[1:n, :, :] == B.vc[bi].storage[1:n, :, :] || push!(diffs, "vc[$bi]")
    end
    return diffs
end

# Runs the baseline and the window path on fresh states, then the next plain
# decode on the window state against a cold run from the same prompt.
function _vw_check(ts; max_new::Int, eos=(), max_draft::Int=8, draft=nothing)
    plen = length(_VW_PROMPT)
    vocab = size(ts.embedding.storage, 1)
    base = GenerateState(_VW_CTX)
    win = GenerateState(_VW_CTX)
    try
        G = reference_generate(ts.model, ts, _VW_PROMPT; max_new_tokens=max_new, eos_ids=eos, state=base)
        src = nothing
        if draft !== nothing
            long = reference_generate(ts.model, ts, _VW_PROMPT; max_new_tokens=_VW_LONG, eos_ids=eos)
            src = _vw_draft(long[(plen + 1):end], plen, vocab; draft...)
        end
        info = Ref{Any}(nothing)
        W = reference_generate(ts.model, ts, _VW_PROMPT; max_new_tokens=max_new, eos_ids=eos,
            state=win, draft=src, max_draft, info)
        diffs = _vw_state_diff(base, win)
        cold = reference_generate(ts.model, ts, W; max_new_tokens=2, eos_ids=())
        nextW = reference_generate(ts.model, ts, W; max_new_tokens=2, eos_ids=(), state=win)
        return (; G, W, diffs, cold, nextW, info=info[], src)
    finally
        close(base)
        close(win)
    end
end

function _vw_assert(r)
    @test r.W == r.G
    @test isempty(r.diffs)
    @test r.nextW == r.cold
    s = r.info.speculation
    @test s.windows == length(r.src.log)
    @test s.drafted == sum(t -> t[1], r.src.log; init=0)
    @test s.accepted == sum(t -> t[2], r.src.log; init=0)
    return s
end

function _vw_suite(ts)
    @testset "oracle drafts are accepted in full, k=$k" for k in (1, 2, 3, 7)
        r = _vw_check(ts; max_new=12, max_draft=k, draft=NamedTuple())
        s = _vw_assert(r)
        @test s.windows >= 1
        @test s.drafted == s.accepted
    end

    @testset "wrong token at draft depth $j of 7" for j in 1:7
        r = _vw_check(ts; max_new=12, max_draft=7, draft=(; wrong=j))
        s = _vw_assert(r)
        @test s.windows >= 1
        @test s.accepted < s.drafted
    end

    @testset "all-wrong draft, k=$k" for k in (3, 7)
        r = _vw_check(ts; max_new=12, max_draft=k, draft=(; allwrong=true))
        s = _vw_assert(r)
        @test s.windows >= 1
        @test s.accepted == 0
    end

    @testset "empty draft" begin
        r = _vw_check(ts; max_new=12, draft=(; empty=true))
        s = _vw_assert(r)
        @test s.windows == 0
    end

    @testset "EOS inside a window" begin
        plen = length(_VW_PROMPT)
        G0 = reference_generate(ts.model, ts, _VW_PROMPT; max_new_tokens=_VW_LONG, eos_ids=())
        eos = G0[plen + 5]
        @test findfirst(==(eos), G0[(plen + 1):end]) == 5
        r = _vw_check(ts; max_new=_VW_LONG, eos=(eos,), max_draft=7, draft=NamedTuple())
        _vw_assert(r)
        @test r.G[end] == eos
        @test length(r.G) == plen + 5
    end

    @testset "output cap inside a window" begin
        r = _vw_check(ts; max_new=5, max_draft=7, draft=(; overrun=true, width=7))
        _vw_assert(r)
        @test length(r.G) == length(_VW_PROMPT) + 5
    end

    @testset "output cap at the end of a window" begin
        r = _vw_check(ts; max_new=5, max_draft=7, draft=NamedTuple())
        _vw_assert(r)
        @test length(r.G) == length(_VW_PROMPT) + 5
    end
end

@testset "verify window: dense" begin
    _vw_suite(toy2_tensors())
end

@testset "verify window: hybrid (GDN block 1)" begin
    _vw_suite(_vw_hybrid_f32())
end

# The replay is what makes a partial accept leave the state at the consumed
# prefix. Skipping it must make the state comparison fail, and the failing
# fields are printed so the report can quote them.
@testset "verify window: red control, skipped replay" begin
    ts = _vw_hybrid_f32()
    r = try
        Gesso.Inference._VERIFY_WINDOW_SKIP_REPLAY[] = true
        _vw_check(ts; max_new=12, max_draft=3, draft=(; wrong=2))
    finally
        Gesso.Inference._VERIFY_WINDOW_SKIP_REPLAY[] = false
    end
    println("red control: ids equal = ", r.W == r.G, "; state fields differing = ", r.diffs)
    @test r.W == r.G
    @test !isempty(r.diffs)
end
