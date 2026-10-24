# Token-batched host MoE prefill (_moe_host_prefill! in src/Inference/moe_host.jl)
# against the token loop over _moe_host_experts_auto! it replaces. Reuses the
# slab helpers from test_moe_kquant_experts.jl.
#   * every row must match the token loop bit for bit: both run the same
#     _dot_q8k (or its bit-equal multi-column form) on the same Q8_K activations
#     and sum each token's experts in k order; checked with the multi-column
#     kernel on and off
#   * the chunk splitter must hand every (expert, row) to exactly one chunk for
#     any chunk count, including more chunks than rows
#   * dead tokens (zero router mass) get a zero row
#   * scratch regrowth: a short batch after a long one reuses the larger buffers
#   * red control: one token's weights permuted must change that row and only it

using Random: Xoshiro, randperm

function _mhp_route(rng, seq, nexp, nused)
    E = zeros(Int, nused, seq)
    W = zeros(Float32, nused, seq)
    for t in 1:seq
        E[:, t] .= randperm(rng, nexp)[1:nused]
        w = rand(rng, Float32, nused) .+ 0.1f0
        W[:, t] .= w ./ sum(w)
    end
    return E, W
end

function _mhp_token_loop(X, E, W, live, gate_s, up_s, down_s)
    seq, h = size(X)
    nused = size(E, 1)
    Y = zeros(Float32, seq, h)
    yrow = zeros(Float32, h)
    for t in 1:seq
        live[t] || continue
        _MKE._moe_host_experts_auto!(yrow, X[t, :], E[:, t], W[:, t], nused, gate_s, up_s, down_s)
        Y[t, :] .= yrow
    end
    return Y
end

@testset "token-batched host MoE prefill" begin
    @testset "chunk splitter covers each row once" begin
        rng = Xoshiro(7)
        s = _MKE.MoePrefillScratch()
        for trial in 1:200
            na = rand(rng, 1:9)
            s.active = sort(randperm(rng, 12)[1:na])
            s.cnt = zeros(Int, 12)
            for e in s.active
                s.cnt[e] = rand(rng, 1:40)
            end
            rows = rand(rng, 1:7)
            tot = _MKE._moe_prefill_cost!(s, rows)
            nc = rand(rng, 1:min(tot, 64))
            seen = Dict{Tuple{Int,Int},Int}()
            for c in 1:nc
                _MKE._moe_prefill_pieces(s, rows, nc, c) do e, r0, r1
                    for r in r0:r1
                        seen[(e, r)] = get(seen, (e, r), 0) + 1
                    end
                end
            end
            @test length(seen) == na * rows && all(==(1), values(seen))
        end
    end

    rng = Xoshiro(44)
    h, ff, nexp, nused = 512, 256, 6, 3
    cases = (
        ("Q2_K gate/up, Q3_K down", _MKE.GGML_TYPE_Q2_K, _MKE.GGML_TYPE_Q2_K, _MKE.GGML_TYPE_Q3_K),
        ("Q4_K gate/up, Q6_K down", _MKE.GGML_TYPE_Q4_K, _MKE.GGML_TYPE_Q4_K, _MKE.GGML_TYPE_Q6_K),
        ("IQ2_XS gate/up, IQ3_XXS down (35B)", _MKE.GGML_TYPE_IQ2_XS, _MKE.GGML_TYPE_IQ2_XS, _MKE.GGML_TYPE_IQ3_XXS),
    )
    for (label, tg, tu, td) in cases
        gate_s = _mke_slice(_mke_random_bytes(rng, tg, nexp, ff, h), tg, nexp, h, ff)
        up_s = _mke_slice(_mke_random_bytes(rng, tu, nexp, ff, h), tu, nexp, h, ff)
        down_s = _mke_slice(_mke_random_bytes(rng, td, nexp, h, ff), td, nexp, ff, h)
        @test _MKE._moe_host_batchable(gate_s, up_s, down_s, h, ff)
        for seq in (37, 1, 5)
            X = randn(rng, Float32, seq, h)
            E, W = _mhp_route(rng, seq, nexp, nused)
            live = trues(seq) |> collect
            seq >= 5 && (live[3] = false)
            Yref = _mhp_token_loop(X, E, W, live, gate_s, up_s, down_s)
            saved = _MKE._PREFILL_MULTI[]
            try
                for multi in (Int8(1), Int8(0))
                    _MKE._PREFILL_MULTI[] = multi
                    Y = fill(NaN32, seq, h)
                    _MKE._moe_host_prefill!(Y, X, E, W, live, nexp, gate_s, up_s, down_s)
                    println("moe prefill ", label, " seq=", seq, " multi=", multi, " bitwise=", Y == Yref,
                        " max_abs=", maximum(abs.(Y .- Yref)))
                    @test Y == Yref
                    seq >= 5 && @test all(iszero, Y[3, :])
                end
            finally
                _MKE._PREFILL_MULTI[] = saved
            end
        end

        seq = 9
        X = randn(rng, Float32, seq, h)
        E, W = _mhp_route(rng, seq, nexp, nused)
        live = collect(trues(seq))
        Y = zeros(Float32, seq, h)
        _MKE._moe_host_prefill!(Y, X, E, W, live, nexp, gate_s, up_s, down_s)
        Wbad = copy(W)
        Wbad[:, 4] .= W[[3, 1, 2], 4]
        Ybad = zeros(Float32, seq, h)
        _MKE._moe_host_prefill!(Ybad, X, E, Wbad, live, nexp, gate_s, up_s, down_s)
        ctrl = _mhb_relrms(Ybad[4, :], Y[4, :])
        println("moe prefill ", label, " red_control_permuted_weights_rel_rms=", ctrl)
        @test ctrl > 1.0e-2
        @test Ybad[[1:3; 5:seq], :] == Y[[1:3; 5:seq], :]

        # A hot layer splits the experts between the host and the GPU. Each
        # side's partial sum skips the other's experts, so the two partials add
        # to the full result up to the order of the additions.
        cpu = [isodd(e) for e in 1:nexp]
        Ya = fill(NaN32, seq, h)
        Yb = fill(NaN32, seq, h)
        _MKE._moe_host_prefill!(Ya, X, E, W, live, nexp, gate_s, up_s, down_s; cpu=cpu)
        _MKE._moe_host_prefill!(Yb, X, E, W, live, nexp, gate_s, up_s, down_s; cpu=[!c for c in cpu])
        split = _mhb_relrms(Ya .+ Yb, Y)
        println("moe prefill ", label, " cpu_mask_split_rel_rms=", split)
        @test split < 1.0e-5
        @test _mhb_relrms(Ya, Y) > 1.0e-2
        Ynone = fill(NaN32, seq, h)
        _MKE._moe_host_prefill!(Ynone, X, E, W, live, nexp, gate_s, up_s, down_s; cpu=fill(false, nexp))
        @test all(iszero, Ynone)
    end
end
