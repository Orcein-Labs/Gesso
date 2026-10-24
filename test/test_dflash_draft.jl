# DFlash block drafter (dflash_draft.jl) on a toy drafter with hand-checked pieces.
#
# The dynamic conv, the top-k and the selector walk are checked against values
# worked out by hand. The drafter as a whole is checked for chunking
# invariance of its context caches, for a fixed-size draft that leaves the
# caches alone, for non-causal attention across block rows, and for the
# context bookkeeping errors.

_dft_w(seed, shape...) = reshape(Float32.(0.3 .* sin.(0.37 .* (1:prod(shape)) .+ 1.3 * seed)), shape)
_dft_norm(seed, n) = Float32.(1 .+ 0.1 .* sin.(0.9 .* (1:n) .+ seed))
_dft_x(n) = reshape(Float32.(0.5 .* cos.(0.61 .* (1:(n * 16)) .+ 0.2)), n, 16)

# One block of the toy: 2 heads over 1 kv head of width 4, FFN 12, and a conv
# kernel of 2 with groups of 2 channels. dim is 8 unless a target sets it.
function _dft_layer(s; dim=8)
    K, ng, qd, kd, ffn = 2, div(dim, 2), 8, 4, 12
    return (
        attn_norm=Gesso.FrozenParameter(; shape=(dim,), storage=_dft_norm(s, dim)),
        attn_conv_proj=Gesso.ProjectionWeight(; shape=(2 * K * ng, dim), storage=_dft_w(s + 1, 2 * K * ng, dim)),
        attn_conv_base=_dft_w(s + 2, dim, K, 2),
        wq=Gesso.ProjectionWeight(; shape=(qd, dim), storage=_dft_w(s + 3, qd, dim)),
        wk=Gesso.ProjectionWeight(; shape=(kd, dim), storage=_dft_w(s + 4, kd, dim)),
        wv=Gesso.ProjectionWeight(; shape=(kd, dim), storage=_dft_w(s + 5, kd, dim)),
        q_norm=Gesso.FrozenParameter(; shape=(4,), storage=_dft_norm(s + 6, 4)),
        k_norm=Gesso.FrozenParameter(; shape=(4,), storage=_dft_norm(s + 7, 4)),
        wo=Gesso.ProjectionWeight(; shape=(dim, qd), storage=_dft_w(s + 8, dim, qd)),
        ffn_norm=Gesso.FrozenParameter(; shape=(dim,), storage=_dft_norm(s + 9, dim)),
        ffn_conv_proj=Gesso.ProjectionWeight(; shape=(2 * K * ng, dim), storage=_dft_w(s + 10, 2 * K * ng, dim)),
        ffn_conv_base=_dft_w(s + 11, dim, K, 2),
        gate=Gesso.ProjectionWeight(; shape=(ffn, dim), storage=_dft_w(s + 12, ffn, dim)),
        up=Gesso.ProjectionWeight(; shape=(ffn, dim), storage=_dft_w(s + 13, ffn, dim)),
        down=Gesso.ProjectionWeight(; shape=(dim, ffn), storage=_dft_w(s + 14, dim, ffn)),
    )
end

function _dft_toy(; ctx=32, target=nothing, n_in=16, window=6)
    dim = target === nothing ? 8 : target.model.embedding.dim
    vocab = target === nothing ? 20 : size(target.embedding.storage, 1)
    rank = 3
    tgt = target === nothing ? (
        embedding=Gesso.EmbeddingTable(; shape=(vocab, dim), storage=_dft_w(1, vocab, dim)),
        lm_head=Gesso.ProjectionWeight(; shape=(vocab, dim), storage=_dft_w(2, vocab, dim)),
    ) : target
    return Gesso.Inference.DFlashDraft(tgt, [_dft_layer(10; dim), _dft_layer(40; dim)],
        Gesso.ProjectionWeight(; shape=(dim, n_in), storage=_dft_w(3, dim, n_in)),
        Gesso.FrozenParameter(; shape=(dim,), storage=_dft_norm(4, dim)),
        Gesso.FrozenParameter(; shape=(dim,), storage=_dft_norm(5, dim)),
        Gesso.ProjectionWeight(; shape=(rank, dim), storage=_dft_w(6, rank, dim)),
        Gesso.EmbeddingTable(; shape=(vocab, rank), storage=_dft_w(7, vocab, rank)),
        Gesso.EmbeddingTable(; shape=(vocab, rank), storage=_dft_w(8, vocab, rank));
        ctx, block_size=4, mask_token=vocab - 1, conv_kernel=2, conv_group=2, top_k=4, rank,
        window, theta=1e4, n_heads=2, n_kv_heads=1, d_head=4)
end

@testset "DFlash conv: taps, zero left pad, causal per row" begin
    dim, G, K, ng = 8, 2, 2, 4
    base = _dft_w(11, dim, K, 2)
    for B in (1, 3, 8), side in (0, 1)
        hid = _dft_x(B)[:, 1:dim]
        dyn = _dft_w(12, B, 2 * K * ng)
        out = Gesso.Inference._dflash_conv(hid, dyn, base, side, G)
        ref = zeros(Float32, B, dim)
        for c in 1:dim, i in 1:B
            grp = div(c - 1, G)
            for a in 0:(K - 1)
                src = i - a
                src >= 1 || continue
                ref[i, c] += (dyn[i, 1 + grp + ng * (a + K * side)] + base[c, a + 1, side + 1]) * hid[src, c]
            end
        end
        @test isapprox(out, ref; rtol=1e-6, atol=1e-7)
        # row i never reads a later row
        for i in 1:B
            one = Gesso.Inference._dflash_conv(hid[1:i, :], dyn[1:i, :], base, side, G)
            @test isapprox(one[i, :], out[i, :]; rtol=1e-6, atol=1e-7)
        end
    end
end

@testset "DFlash top-k: descending, ties to the lower id" begin
    top = Gesso.Inference._dflash_top_k
    @test top(Float32[0.1, 0.5, 0.5, 0.3], 2) == [1, 2]
    @test top(Float32[0.1, 0.5, 0.5, 0.3], 0) == Int[]
    # k past the vocabulary returns every id, highest logit first
    @test top(Float32[0.1, 0.5, 0.5, 0.3], 9) == [1, 2, 3, 0]
end

@testset "DFlash selector: gate-weighted chain score by hand" begin
    cand = [[1, 2], [3, 4]]
    unary = [Float32[0.1, 0.2], Float32[0.0, 0.5]]
    gate = [[2.0], [1.0]]
    succ = [reshape(Float32[1.0, 3.0], 2, 1), reshape(Float32[0.0, 2.0], 2, 1)]
    pred = [reshape(Float32[0.5], 1, 1), reshape(Float32[0.5, 1.0], 2, 1)]
    S = Gesso.Inference._dflash_select(cand, unary, gate, succ, pred)
    # row 1: 1*0.5*2 + 0.1 and 3*0.5*2 + 0.2
    @test isapprox(S[1], reshape(Float32[1.1, 3.2], 2, 1); atol=1e-6)
    # row 2: succ 0 and 2 against pred 0.5 and 1.0, gate 1
    @test isapprox(S[2], Float32[0.0 0.0; 1.5 2.5]; atol=1e-6)
end

@testset "DFlash walk: greedy chain, p_min gate, n_min floor" begin
    cands = [[5, 7], [2, 9], [1, 3], [4, 6]]
    scores = [
        reshape(Float32[0.1, 0.9], 2, 1),
        Float32[0.0 0.0; 0.0 3.0],
        Float32[0.0 0.0; 0.0 0.0],
        Float32[0.0 0.0; 2.0 0.0],
    ]
    W = Gesso.Inference._dflash_walk
    # row 1 picks 7; row 2 column 2 picks 9; row 3 is flat, so it picks 1
    # (probability 0.5); row 4 column 1 picks 6
    @test W(cands, scores) == [7, 9, 1, 6]
    @test W(cands, scores; p_min=0.6) == [7, 9]
    @test W(cands, scores; p_min=0.6, n_min=3) == Int[]
    @test W(cands, scores; p_min=0.6, n_min=2) == [7, 9]
end

@testset "DFlash rope and rmsnorm: identity at position 0, norm kept" begin
    x = _dft_w(21, 4)
    y = copy(x)
    Gesso.Inference._dflash_rope!(y, 0, 1e4)
    @test isapprox(y, x; atol=1e-7)
    z = Gesso.Inference._dflash_rope!(copy(x), 7, 1e4)
    @test isapprox(sqrt(sum(abs2, z)), sqrt(sum(abs2, x)); rtol=1e-6)
    out = similar(x)
    Gesso.Inference._df_rmsnorm!(out, x, ones(Float32, 4), 1e-6)
    @test isapprox(sqrt(sum(abs2, out) / 4), 1; atol=1e-4)
end

@testset "DFlash context: chunked rows equal one call" begin
    X = _dft_x(6)
    a = _dft_toy()
    b = _dft_toy()
    Gesso.Inference.dflash_rows!(a, 1, X)
    Gesso.Inference.dflash_rows!(b, 1, X[1:2, :])
    Gesso.Inference.dflash_rows!(b, 3, X[3:6, :])
    @test a.have == b.have == 6
    for l in eachindex(a.kc)
        @test isapprox(a.kc[l][:, :, 1:6], b.kc[l][:, :, 1:6]; rtol=1e-5, atol=1e-6)
        @test isapprox(a.vc[l][:, :, 1:6], b.vc[l][:, :, 1:6]; rtol=1e-5, atol=1e-6)
    end
end

@testset "DFlash draft: caches stay put, draft is deterministic and bounded" begin
    m = _dft_toy()
    ids = [3, 5, 8, 2, 11, 7]
    Gesso.Inference.dflash_rows!(m, 1, _dft_x(5))
    kc0 = copy.(m.kc)
    vc0 = copy.(m.vc)
    d1 = Gesso.Inference.dflash_draft!(m, ids, 3)
    d2 = Gesso.Inference.dflash_draft!(m, ids, 3)
    @test d1 == d2
    @test length(d1) <= 3
    @test all(t -> 0 <= t < 20, d1)
    @test m.kc == kc0
    @test m.vc == vc0
    @test m.have == 5
end

@testset "DFlash block: attention is not causal across block rows" begin
    m = _dft_toy()
    ids = [3, 5, 8, 2, 11, 7]
    Gesso.Inference.dflash_rows!(m, 1, _dft_x(5))
    _, lg1 = Gesso.Inference._dflash_block(m, ids, 1)
    _, lg3 = Gesso.Inference._dflash_block(m, ids, 3)
    @test size(lg1) == (1, 20)
    @test size(lg3) == (3, 20)
    # the first draft row sees the later block rows, so its logits move
    @test !isapprox(lg1[1, :], lg3[1, :]; atol=1e-6)
end

@testset "DFlash context: gaps, overflow and short context are errors" begin
    m = _dft_toy(; ctx=8)
    @test_throws ErrorException Gesso.Inference.dflash_rows!(m, 3, _dft_x(2))
    @test_throws ErrorException Gesso.Inference.dflash_rows!(m, 1, _dft_x(9))
    Gesso.Inference.dflash_rows!(m, 1, _dft_x(3))
    @test_throws ErrorException Gesso.Inference.dflash_draft!(m, [3, 5, 8, 2, 11], 2)
    Gesso.Inference.reset!(m)
    @test m.have == 0
end

@testset "DFlash attention: window-masked softmax over context and block" begin
    # a context row r sits at position r-1 and is masked once qp - (r-1) >= window;
    # every block row is kept
    X = _dft_x(10)
    L, B, d = 11, 4, 4
    q = _dft_w(31, B, 8)
    kb = _dft_w(32, B, 4)
    vb = _dft_w(33, B, 4)
    outs = Matrix{Float32}[]
    for window in (6, 100)
        m = _dft_toy(; window)
        Gesso.Inference.dflash_rows!(m, 1, X)
        out = Gesso.Inference._dflash_block_attn(m, 1, q, kb, vb, L)
        # the threaded split runs the same per-job loops
        @test Gesso.Inference._dflash_block_attn(m, 1, q, kb, vb, L; threaded=true) == out
        ref = zeros(Float64, B, 8)
        for i in 1:B, hd in 1:2
            qp = L + i - 2
            qv = q[i, ((hd - 1) * d + 1):(hd * d)]
            keys = Vector{Float64}[]
            vals = Vector{Float64}[]
            for r in 1:(L - 1)
                qp - (r - 1) >= window && continue
                push!(keys, m.kc[1][:, 1, r])
                push!(vals, m.vc[1][:, 1, r])
            end
            for j in 1:B
                push!(keys, kb[j, :])
                push!(vals, vb[j, :])
            end
            s = [sum(qv .* k) / sqrt(d) for k in keys]
            w = exp.(s .- maximum(s))
            w ./= sum(w)
            ref[i, ((hd - 1) * d + 1):(hd * d)] = sum(w[z] .* vals[z] for z in eachindex(w))
        end
        @test isapprox(out, ref; rtol=1e-5, atol=1e-6)
        push!(outs, out)
    end
    # the small window cuts context rows, so the two differ
    @test !isapprox(outs[1], outs[2]; atol=1e-6)
end

# Draft source over a DFlash drafter, as scripts/saluki_dflash_source.jl does it.
mutable struct _DFTSource <: Gesso.Speculation.DraftSource
    m
    k::Int
    log::Vector{Tuple{Int,Int}}
end
function Gesso.Speculation.propose(src::_DFTSource, ctx::Gesso.Speculation.DraftContext; max_tokens::Int)
    d = Gesso.Inference.dflash_draft!(src.m, collect(ctx.ids), min(max_tokens, src.k))
    return Gesso.Speculation.Draft(d, :dflash)
end
Gesso.Speculation.observe!(src::_DFTSource, d::Gesso.Speculation.Draft, a::Integer) =
    (push!(src.log, (length(d.ids), a)); nothing)

@testset "DFlash draft: drafting over the hybrid toy never changes the ids" begin
    ts = _vw_hybrid_f32()   # taps need the window prefill (CPU hybrid F32)
    dim = ts.model.embedding.dim
    prompt = [1, 3, 4, 5]
    plain = reference_generate(ts.model, ts, prompt; max_new_tokens=12, eos_ids=())
    for k in (1, 3)
        src = _DFTSource(_dft_toy(; ctx=64, target=ts, n_in=2dim), k, Tuple{Int,Int}[])
        st = GenerateState(64)
        try
            info = Ref{Any}(nothing)
            ids = reference_generate(ts.model, ts, prompt; max_new_tokens=12, eos_ids=(),
                state=st, draft=src, max_draft=k, info, taps=[0, 1],
                on_taps=(p, X) -> Gesso.Inference.dflash_rows!(src.m, p, X))
            @test ids == plain
            @test info[].speculation.windows == length(src.log) >= 1
            @test all(t -> t[1] <= k, src.log)
            @test src.m.have == length(ids) - 1
        finally
            close(st)
        end
    end
end
