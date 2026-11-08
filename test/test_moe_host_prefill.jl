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
        _MKE._moe_host_experts_auto!(
            yrow,
            X[t, :],
            E[:, t],
            W[:, t],
            nused,
            gate_s,
            up_s,
            down_s,
        )
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
            seen = Dict{Tuple{Int, Int}, Int}()
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
        (
            "Q2_K gate/up, Q3_K down",
            _MKE.GGML_TYPE_Q2_K,
            _MKE.GGML_TYPE_Q2_K,
            _MKE.GGML_TYPE_Q3_K,
        ),
        (
            "Q4_K gate/up, Q6_K down",
            _MKE.GGML_TYPE_Q4_K,
            _MKE.GGML_TYPE_Q4_K,
            _MKE.GGML_TYPE_Q6_K,
        ),
        (
            "IQ2_XS gate/up, IQ3_XXS down (35B)",
            _MKE.GGML_TYPE_IQ2_XS,
            _MKE.GGML_TYPE_IQ2_XS,
            _MKE.GGML_TYPE_IQ3_XXS,
        ),
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
                    println(
                        "moe prefill ",
                        label,
                        " seq=",
                        seq,
                        " multi=",
                        multi,
                        " bitwise=",
                        Y == Yref,
                        " max_abs=",
                        maximum(abs.(Y .- Yref)),
                    )
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
        _MKE._moe_host_prefill!(
            Yb,
            X,
            E,
            W,
            live,
            nexp,
            gate_s,
            up_s,
            down_s;
            cpu=[!c for c in cpu],
        )
        split = _mhb_relrms(Ya .+ Yb, Y)
        println("moe prefill ", label, " cpu_mask_split_rel_rms=", split)
        @test split < 1.0e-5
        @test _mhb_relrms(Ya, Y) > 1.0e-2
        Ynone = fill(NaN32, seq, h)
        _MKE._moe_host_prefill!(
            Ynone,
            X,
            E,
            W,
            live,
            nexp,
            gate_s,
            up_s,
            down_s;
            cpu=fill(false, nexp),
        )
        @test all(iszero, Ynone)
    end
end

# Slab bytes sized by the weight block (Q2_0 packs 64 values in 18 bytes, which
# `_mke_qblock` does not know), with finite f16 scales at d and dmin.
function _mhp_random_bytes(rng, ty::Int32, nexp::Int, nrows::Int, ne0::Int)
    bsz = _MKE._iq_block_bytes(ty)
    nb = div(ne0, _MKE._moe_weight_block_values(ty))
    w = rand(rng, UInt8, nexp * nrows * nb * bsz)
    for blk in 0:(nexp*nrows*nb-1)
        for off in (_MKE._q8k_d_offset(ty), _MKE._q8k_dmin_offset(ty))
            off < 0 && continue
            bits = reinterpret(UInt16, Float16(0.005f0 + 0.05f0 * rand(rng, Float32)))
            w[blk*bsz+off+1] = UInt8(bits & 0xff)
            w[blk*bsz+off+2] = UInt8(bits >> 8)
        end
    end
    return w
end

_mhp_slice(w, ty::Int32, nexp::Int, ne0::Int, ne1::Int) = Gesso.GgufQuantSlice(
    w, 0, ty, nexp, ne0, ne1, ne1 * _MKE._qrow_bytes(ty, ne0), Int64(_MKE._moe_weight_block_values(ty)))

# Flash-Next's expert slabs: IQ2_XXS gate/up over h, Q2_0 down over ff = 640.
# The Q2_0 down takes a Q8_0 activation and ff is not a multiple of 256, so the
# decode kernels refuse it and prefill used to fall to the token-block pieces
# path. The grouped prefill must make the same dots in the same order.
#   * seq > 1 through the public entry equals the pieces path bit for bit, with
#     the multi-column kernels on and off
#   * seq == 1 still takes the pieces path (the decode scratch is Q8_K only)
#   * red control: the down slab replaced must change the output
@testset "grouped MoE prefill: IQ2_XXS gate/up, Q2_0 down, ff=640" begin
    rng = Xoshiro(45)
    h, ff, nexp, nused = 512, 640, 6, 3
    tg, td = _MKE.GGML_TYPE_IQ2_XXS, _MKE.GGML_TYPE_Q2_0
    gate_s = _mhp_slice(_mhp_random_bytes(rng, tg, nexp, ff, h), tg, nexp, h, ff)
    up_s = _mhp_slice(_mhp_random_bytes(rng, tg, nexp, ff, h), tg, nexp, h, ff)
    wd = _mhp_random_bytes(rng, td, nexp, h, ff)
    down_s = _mhp_slice(wd, td, nexp, ff, h)
    @test !_MKE._moe_host_batchable(gate_s, up_s, down_s, h, ff)
    @test _MKE._moe_host_prefill_ok(gate_s, up_s, down_s, h, ff)
    router = randn(rng, Float32, nexp, h)
    pieces(x, dn) = _MKE._cpu_moe_gguf_pieces!(zeros(Float32, size(x)), x, router, gate_s, up_s, dn, nused, h, ff)
    entry(x, dn) = _MKE._cpu_moe_topk_swiglu_gguf!(zeros(Float32, size(x)), x, router, gate_s, up_s, dn, nused)
    saved = _MKE._PREFILL_MULTI[]
    try
        for seq in (37, 16, 17, 2, 1), multi in (Int8(1), Int8(0))
            _MKE._PREFILL_MULTI[] = multi
            X = randn(rng, Float32, seq, h)
            Yref = pieces(X, down_s)
            Y = entry(X, down_s)
            println("moe prefill IQ2_XXS/Q2_0 ff=640 seq=", seq, " multi=", multi,
                " bitwise=", Y == Yref, " max_abs=", maximum(abs.(Y .- Yref)))
            @test all(isfinite, Y)
            @test Y == Yref
        end
    finally
        _MKE._PREFILL_MULTI[] = saved
    end

    X = randn(rng, Float32, 37, h)
    Y = entry(X, down_s)
    wbad = copy(wd)
    bpe = down_s.bytes_per_expert
    wbad[1:nexp*bpe] .= _mhp_random_bytes(Xoshiro(9), td, nexp, h, ff)
    Ybad = entry(X, _mhp_slice(wbad, td, nexp, ff, h))
    ctrl = _mhb_relrms(Ybad, Y)
    println("moe prefill IQ2_XXS/Q2_0 ff=640 red_control_replaced_down_rel_rms=", ctrl)
    @test ctrl > 1.0e-2
end

# Flash-Next's other layer mixes: IQ2_XS gate/up over Q2_0 down, and IQ2_XXS
# gate/up over IQ4_NL down. `_qgemv_bytes!` dots IQ2_XS and IQ4_NL against float
# x, so the grouped prefill must refuse these slabs; a Q8 activation for them
# moved Flash-Next's logits by rel_rms 0.06. The entry must equal the pieces
# path bit for bit at every seq.
@testset "grouped MoE prefill float-activation types match pieces" begin
    rng = Xoshiro(46)
    h, ff, nexp, nused = 512, 640, 6, 3
    for (tg, td) in ((_MKE.GGML_TYPE_IQ2_XS, _MKE.GGML_TYPE_Q2_0),
                     (_MKE.GGML_TYPE_IQ2_XXS, _MKE.GGML_TYPE_IQ4_NL),
                     (_MKE.GGML_TYPE_IQ2_XS, _MKE.GGML_TYPE_IQ4_NL))
        gate_s = _mhp_slice(_mhp_random_bytes(rng, tg, nexp, ff, h), tg, nexp, h, ff)
        up_s = _mhp_slice(_mhp_random_bytes(rng, tg, nexp, ff, h), tg, nexp, h, ff)
        down_s = _mhp_slice(_mhp_random_bytes(rng, td, nexp, h, ff), td, nexp, ff, h)
        @test _MKE._moe_host_prefill_ok(gate_s, up_s, down_s, h, ff)
        router = randn(rng, Float32, nexp, h)
        for seq in (37, 2, 1)
            X = randn(rng, Float32, seq, h)
            Yref = _MKE._cpu_moe_gguf_pieces!(zeros(Float32, seq, h), X, router, gate_s, up_s, down_s, nused, h, ff)
            Y = _MKE._cpu_moe_topk_swiglu_gguf!(zeros(Float32, seq, h), X, router, gate_s, up_s, down_s, nused)
            println("moe prefill float-act gt=", tg, " dt=", td, " seq=", seq, " bitwise=", Y == Yref)
            @test all(isfinite, Y)
            @test Y == Yref
        end
    end
    # Gate and up must take the same activation kind.
    gate_s = _mhp_slice(_mhp_random_bytes(rng, _MKE.GGML_TYPE_IQ2_XS, nexp, ff, h), _MKE.GGML_TYPE_IQ2_XS, nexp, h, ff)
    up_s = _mhp_slice(_mhp_random_bytes(rng, _MKE.GGML_TYPE_IQ2_XXS, nexp, ff, h), _MKE.GGML_TYPE_IQ2_XXS, nexp, h, ff)
    down_s = _mhp_slice(_mhp_random_bytes(rng, _MKE.GGML_TYPE_Q2_0, nexp, h, ff), _MKE.GGML_TYPE_Q2_0, nexp, ff, h)
    @test !_MKE._moe_host_prefill_ok(gate_s, up_s, down_s, h, ff)
end
