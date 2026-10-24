# MTP draft head (mtp_draft.jl) and the on_rows hook it reads.
#
# on_rows must hand out every consumed position once, in order, with the row
# the head reads (on_hidden's row), in plain decode and in verify windows. The
# head itself is checked three ways on the dense toy: a catch-up window equals
# one row per call, a head wired to read only the token is the target's greedy
# decode of a one-block model, and drafting through it never changes the ids.

const _MTP_PROMPT = [1, 3, 4, 5]
const _MTP_CTX = 64

_mtp_unit_rows(E) = E ./ sqrt.(sum(abs2, E; dims=2) ./ size(E, 2))

# A one-block head over the toy's first block. `eh` picks the eh_proj; the
# default mixes both halves.
function _mtp_toy(ts; eh=nothing, ctx=_MTP_CTX)
    dim = ts.model.embedding.dim
    scale = 1.0 .+ 0.25 .* sin.(1.0:dim)
    ht = merge(
        ts,
        (;
            blocks=ts.blocks[1:1],
            final_rms=Gesso.FrozenParameter(; shape=(dim,), storage=scale),
        ),
    )
    W = eh === nothing ? [0.3 * sin(0.7i + 1.3j) for i in 1:dim, j in 1:(2dim)] : eh
    en = Gesso.FrozenParameter(; shape=(dim,), storage=1.0 .+ 0.1 .* cos.(1.0:dim))
    hn = Gesso.FrozenParameter(; shape=(dim,), storage=1.0 .+ 0.1 .* sin.(2.0:(dim+1)))
    return Gesso.Inference.MTPDraft(
        ts.model,
        ht,
        Gesso.ProjectionWeight(; shape=size(W), storage=W),
        en,
        hn;
        ctx,
    )
end

# Draft source over an MTP head, as scripts/saluki_mtp_source.jl does it.
mutable struct _MTPSource <: Gesso.Speculation.DraftSource
    m::Any
    k::Int
    log::Vector{Tuple{Int, Int}}
end
function Gesso.Speculation.propose(
    src::_MTPSource,
    ctx::Gesso.Speculation.DraftContext;
    max_tokens::Int,
)
    d = Gesso.Inference.mtp_draft!(src.m, collect(ctx.ids), min(max_tokens, src.k))
    return Gesso.Speculation.Draft(d, :mtp)
end
Gesso.Speculation.observe!(src::_MTPSource, d::Gesso.Speculation.Draft, a::Integer) =
    (push!(src.log, (length(d.ids), a)); nothing)

function _mtp_rows_log()
    calls = Tuple{Int, Matrix{Float32}}[]
    return calls, (p, M) -> push!(calls, (p, copy(M)))
end

# rows by position from an on_rows log; checks they tile 1..n with no gap or repeat
function _mtp_tile(calls)
    next = 1
    rows = Vector{Vector{Float32}}()
    for (p, M) in calls
        @test p == next
        for t in axes(M, 1)
            push!(rows, M[t, :])
        end
        next = p + size(M, 1)
    end
    return rows
end

@testset "on_rows: plain decode tiles every consumed position" begin
    for ts in (toy2_tensors(), _oh_with_final_norm(toy2_tensors()))
        plain = reference_generate(ts.model, ts, _MTP_PROMPT; max_new_tokens=6, eos_ids=())
        calls, cb = _mtp_rows_log()
        hid = Tuple{Int, Vector{Float32}}[]
        ids = reference_generate(
            ts.model,
            ts,
            _MTP_PROMPT;
            max_new_tokens=6,
            eos_ids=(),
            on_rows=cb,
        )
        @test ids == plain
        rows = _mtp_tile(calls)
        @test length(rows) == length(ids) - 1
        reference_generate(
            ts.model,
            ts,
            _MTP_PROMPT;
            max_new_tokens=6,
            eos_ids=(),
            on_hidden=(p, v) -> push!(hid, (p, v)),
        )
        for (p, v) in hid
            @test rows[p] == v
        end
        # prompt rows: the prefill oracle's final-normed hidden state
        ref = reference_prefill(ts.model, ts, _MTP_PROMPT; hidden=true)
        for p in 1:(length(_MTP_PROMPT)-1)
            @test isapprox(rows[p], ref[:, p]; rtol=1e-4, atol=1e-6)
        end
    end
end

@testset "on_rows: verify windows tile the same rows" begin
    ts = toy2_tensors()
    plen = length(_MTP_PROMPT)
    vocab = size(ts.embedding.storage, 1)
    pc, pcb = _mtp_rows_log()
    plain = reference_generate(
        ts.model,
        ts,
        _MTP_PROMPT;
        max_new_tokens=12,
        eos_ids=(),
        on_rows=pcb,
    )
    prow = _mtp_tile(pc)
    for draft in (NamedTuple(), (; wrong=2), (; allwrong=true))
        src = _vw_draft(plain[(plen+1):end], plen, vocab; draft...)
        st = GenerateState(_MTP_CTX)
        try
            wc, wcb = _mtp_rows_log()
            ids = reference_generate(
                ts.model,
                ts,
                _MTP_PROMPT;
                max_new_tokens=12,
                eos_ids=(),
                state=st,
                draft=src,
                max_draft=3,
                on_rows=wcb,
            )
            @test ids == plain
            wrow = _mtp_tile(wc)
            @test length(wrow) == length(ids) - 1
            @test all(
                isapprox(wrow[p], prow[p]; rtol=1e-5, atol=1e-6) for p in eachindex(wrow)
            )
        finally
            close(st)
        end
    end
end

@testset "MTP head: a catch-up window equals one row per call" begin
    ts = toy2_tensors()
    calls, cb = _mtp_rows_log()
    ids = reference_generate(
        ts.model,
        ts,
        _MTP_PROMPT;
        max_new_tokens=6,
        eos_ids=(),
        on_rows=cb,
    )
    a = _mtp_toy(ts)
    b = _mtp_toy(ts)
    for (p, M) in calls
        Gesso.Inference.mtp_rows!(a, p, M)
        Gesso.Inference.mtp_rows!(b, p, M)
    end
    L = length(ids)
    # b runs one row per call: the draft after each prefix
    for n in 1:(L-1)
        Gesso.Inference.mtp_draft!(b, ids[1:n], 1)
    end
    da = Gesso.Inference.mtp_draft!(a, ids, 3)
    db = Gesso.Inference.mtp_draft!(b, ids, 3)
    @test da == db
    @test a.done == b.done == L
    @test isapprox(
        a.kc[1].storage[1:L, :, :],
        b.kc[1].storage[1:L, :, :];
        rtol=1e-10,
        atol=1e-12,
    )
    @test isapprox(
        a.vc[1].storage[1:L, :, :],
        b.vc[1].storage[1:L, :, :];
        rtol=1e-10,
        atol=1e-12,
    )
    # a repeat call on the same ids reruns only row L and drafts the same
    kv = copy(a.kc[1].storage[1:L, :, :])
    @test Gesso.Inference.mtp_draft!(a, ids, 3) == da
    @test a.done == L
    @test a.kc[1].storage[1:L, :, :] == kv
end

@testset "MTP head: reading only the token, it decodes the one-block model" begin
    # unit-RMS embedding rows make rmsnorm(e) == e (up to eps), so with enorm = 1
    # and eh_proj = [I 0] the head's block input is the token embedding
    base = toy2_tensors()
    dim = base.model.embedding.dim
    E = Gesso.EmbeddingTable(;
        shape=size(base.embedding.storage),
        storage=_mtp_unit_rows(base.embedding.storage),
    )
    ts = merge(base, (; embedding=E, lm_head=E))
    m1 = Gesso.Model(;
        vocab_size=ts.model.vocab_size,
        embedding=ts.model.embedding,
        blocks=(ts.model.blocks[1],),
    )
    head = _mtp_toy(ts; eh=[Float64(i == j) for i in 1:dim, j in 1:(2dim)])
    head.enorm = Gesso.FrozenParameter(; shape=(dim,), storage=ones(dim))
    ts1 = merge(head.tensors, (; model=m1))
    calls, cb = _mtp_rows_log()
    ids = reference_generate(
        ts.model,
        ts,
        _MTP_PROMPT;
        max_new_tokens=3,
        eos_ids=(),
        on_rows=cb,
    )
    for (p, M) in calls
        Gesso.Inference.mtp_rows!(head, p, M)
    end
    d = Gesso.Inference.mtp_draft!(head, ids, 4)
    ref = reference_generate(m1, ts1, ids; max_new_tokens=4, eos_ids=())
    @test d == ref[(length(ids)+1):end]
end

@testset "MTP head: drafting never changes the ids" begin
    ts = toy2_tensors()
    plain = reference_generate(ts.model, ts, _MTP_PROMPT; max_new_tokens=12, eos_ids=())
    for k in (1, 2, 3)
        src = _MTPSource(_mtp_toy(ts), k, Tuple{Int, Int}[])
        st = GenerateState(_MTP_CTX)
        try
            info = Ref{Any}(nothing)
            ids = reference_generate(
                ts.model,
                ts,
                _MTP_PROMPT;
                max_new_tokens=12,
                eos_ids=(),
                state=st,
                draft=src,
                max_draft=k,
                info,
                on_rows=(p, M) -> Gesso.Inference.mtp_rows!(src.m, p, M),
            )
            @test ids == plain
            @test info[].speculation.windows == length(src.log) >= 1
            @test all(t -> t[1] <= k, src.log)
        finally
            close(st)
        end
    end
end
