# on_taps on reference_generate (DFlash drafter inputs): the residual stream after
# chosen block counts, for every consumed position. Plain decode, the window
# prefill and verify windows must hand out the same rows, and the ids never change.
# Built on the hybrid toy, which takes the window prefill path.

const _DF_PROMPT = [1, 3, 4, 5]
const _DF_CTX = 64

# taps ride the window prefill, which only the CPU hybrid F32 path takes
_df_toy(ts=toy2_tensors()) = _vw_f32(_hybrid_toy_tensors(ts))

# (p, X) log; the callback copies, so the log keeps each delivery
function _df_log()
    calls = Tuple{Int, Matrix{Float32}}[]
    return calls, (p, X) -> push!(calls, (p, copy(X)))
end

# the rows of a log stacked by position; checks they tile 1..n with no gap or repeat
function _df_tile(calls)
    next = 1
    out = Matrix{Float32}[]
    for (p, X) in calls
        @test p == next
        push!(out, X)
        next = p + size(X, 1)
    end
    return vcat(out...)
end

@testset "taps: plain decode tiles every consumed position" begin
    ts = _df_toy()
    nb = length(ts.model.blocks)
    dim = ts.model.embedding.dim
    plain = reference_generate(ts.model, ts, _DF_PROMPT; max_new_tokens=6, eos_ids=())
    calls, cb = _df_log()
    ids = reference_generate(
        ts.model,
        ts,
        _DF_PROMPT;
        max_new_tokens=6,
        eos_ids=(),
        taps=[0, 1, nb],
        on_taps=cb,
    )
    @test ids == plain
    @test all(size(X, 2) == 3dim for (_, X) in calls)
    rows = _df_tile(calls)
    @test size(rows, 1) == length(ids) - 1
end

@testset "taps: rows equal the residual after b blocks" begin
    for ts in (_df_toy(), _df_toy(_oh_with_final_norm(toy2_tensors())))
        nb = length(ts.model.blocks)
        dim = ts.model.embedding.dim
        tlog, tcb = _df_log()
        rlog, rcb = _df_log()
        ids = reference_generate(
            ts.model,
            ts,
            _DF_PROMPT;
            max_new_tokens=6,
            eos_ids=(),
            taps=[0, 1, nb],
            on_taps=tcb,
            on_rows=rcb,
        )
        X = _df_tile(tlog)
        R = _df_tile(rlog)
        L = length(ids) - 1
        @test size(X, 1) == size(R, 1) == L
        # the last tap is the residual after every block: final-normed, it is on_rows' row
        fin = Gesso.Inference._final_normed_rows(
            ts,
            X[:, (2dim+1):(3dim)],
            Gesso.CPUBackend(),
            Gesso.DecodeWorkload(),
        )
        @test isapprox(fin, R; rtol=1e-5, atol=1e-6)
        # tap 0 is the embedding output of the consumed id
        for p in 1:L
            @test isapprox(
                X[p, 1:dim],
                ts.embedding.storage[ids[p]+1, :];
                rtol=1e-5,
                atol=1e-6,
            )
        end
    end
end

@testset "taps: verify windows tile the same rows" begin
    ts = _df_toy()
    nb = length(ts.model.blocks)
    plen = length(_DF_PROMPT)
    vocab = size(ts.embedding.storage, 1)
    pc, pcb = _df_log()
    plain = reference_generate(
        ts.model,
        ts,
        _DF_PROMPT;
        max_new_tokens=12,
        eos_ids=(),
        taps=[0, 1, nb],
        on_taps=pcb,
    )
    prow = _df_tile(pc)
    for draft in (NamedTuple(), (; wrong=2), (; allwrong=true))
        src = _vw_draft(plain[(plen+1):end], plen, vocab; draft...)
        st = GenerateState(_DF_CTX)
        try
            wc, wcb = _df_log()
            info = Ref{Any}(nothing)
            ids = reference_generate(
                ts.model,
                ts,
                _DF_PROMPT;
                max_new_tokens=12,
                eos_ids=(),
                state=st,
                draft=src,
                max_draft=3,
                info,
                taps=[0, 1, nb],
                on_taps=wcb,
            )
            @test ids == plain
            @test info[].speculation.windows >= 1
            wrow = _df_tile(wc)
            @test size(wrow) == size(prow)
            @test isapprox(wrow, prow; rtol=1e-5, atol=1e-6)
        finally
            close(st)
        end
    end
end

@testset "taps: argument checks" begin
    ts = _df_toy()
    nb = length(ts.model.blocks)
    cb = (p, X) -> nothing
    _df_run(; kw...) =
        reference_generate(ts.model, ts, _DF_PROMPT; max_new_tokens=2, eos_ids=(), kw...)
    @test_throws ArgumentError _df_run(taps=[0])
    @test_throws ArgumentError _df_run(on_taps=cb)
    @test_throws ArgumentError _df_run(taps=[nb + 1], on_taps=cb)
    @test_throws ArgumentError _df_run(taps=[-1], on_taps=cb)
    # a dense model never takes the window prefill, so it refuses taps
    dense = toy2_tensors()
    @test_throws ArgumentError reference_generate(
        dense.model,
        dense,
        _DF_PROMPT;
        max_new_tokens=2,
        eos_ids=(),
        taps=[0],
        on_taps=cb,
    )
end
