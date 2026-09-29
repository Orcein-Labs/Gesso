# Item A tests: CPU reference operator methods (§LXXV math law).
#
# Each testset computes the goal's formula TEST-SIDE on a tiny array and
# compares. Tolerances are declared per op: pure copying/copy-with-scale is
# bit-identical (atol=0); anything touching exp/cos/BLAS gets atol=1e-12
# (BLAS association vs naive loops is not bit-identical in general).

using .HarpeTestHelpers: approx_eq, deterministic_rng

# helper: a materialized tensor of family F
mat(F, arr) = F(; shape=size(arr), storage=arr)

@testset "cpu embedding_lookup! (§LXXV)" begin
    table = mat(
        Harpe.EmbeddingTable,
        [10.0 11.0 12.0; 20.0 21.0 22.0; 30.0 31.0 32.0; 40.0 41.0 42.0],
    )
    dst = mat(Harpe.Activation, zeros(2, 3))
    Harpe.embedding_lookup!(Harpe.CPUBackend(), dst, table, [2, 0], Harpe.PrefillWorkload())
    # 0-based ids → rows 3 and 1
    @test approx_eq(dst.storage, [30.0 31.0 32.0; 10.0 11.0 12.0]; atol=0.0)
end

@testset "cpu rmsnorm! (§LXXV)" begin
    x = mat(Harpe.Activation, [1.0 2.0; 3.0 4.0])
    scale = mat(Harpe.FrozenParameter, [10.0, 20.0])
    dst = mat(Harpe.Activation, zeros(2, 2))
    Harpe.rmsnorm!(Harpe.CPUBackend(), dst, x, scale, Harpe.PrefillWorkload())
    ε = 1e-6
    expected = similar(x.storage)
    for i in 1:2
        rms = sqrt((x.storage[i, 1]^2 + x.storage[i, 2]^2) / 2 + ε)
        expected[i, :] .= (x.storage[i, :] ./ rms) .* scale.storage
    end
    @test approx_eq(dst.storage, expected; atol=1e-12)

    # last dim is the feature dim: a (seq, heads, features) cube normalizes
    # per (seq, head) row
    x3 = mat(Harpe.Activation, reshape(collect(1.0:12.0), (2, 2, 3)))
    s3 = mat(Harpe.FrozenParameter, [1.0, 1.0, 1.0])
    dst3 = mat(Harpe.Activation, zeros(2, 2, 3))
    Harpe.rmsnorm!(Harpe.CPUBackend(), dst3, x3, s3, Harpe.PrefillWorkload())
    for t in 1:2, h in 1:2
        row = x3.storage[t, h, :]
        rms = sqrt(sum(abs2, row) / 3 + ε)
        @test approx_eq(dst3.storage[t, h, :], row ./ rms; atol=1e-12)
    end
end

@testset "cpu rope! (§LXXV): LLaMA pairwise rotate" begin
    # d_head = 4, two heads, two positions. Position 0 must be the IDENTITY
    # (θ = 0 ⇒ cos = 1, sin = 0).
    q0 = reshape(Float64[i for i in 1:16], (2, 2, 4))
    k0 = reshape(Float64[i for i in 17:32], (2, 2, 4))
    # COPIES: rope! rotates in place, and mat() wraps without copying —
    # handing it q0 directly would mutate the expected-source mid-test
    q, k = mat(Harpe.Activation, copy(q0)), mat(Harpe.Activation, copy(k0))
    Harpe.rope!(Harpe.CPUBackend(), q, k, [0, 1], Harpe.PrefillWorkload())
    # position 0 rows unchanged bit-for-bit (θ=0 ⇒ cos=1, sin=0); position 1
    # rows are rotated and checked against the formula below
    @test approx_eq(q.storage[1, :, :], q0[1, :, :]; atol=0.0)
    @test approx_eq(k.storage[1, :, :], k0[1, :, :]; atol=0.0)

    # position 1: θ_i = 10000^{-2i/4}, pairwise rotate per pair
    expected_q = copy(q0)
    for h in 1:2
        for i in 0:1
            θ = 10000.0^(-2i / 4)
            c, s = cos(θ), sin(θ)
            x1, x2 = q0[2, h, 2i+1], q0[2, h, 2i+2]
            expected_q[2, h, 2i+1] = x1 * c - x2 * s
            expected_q[2, h, 2i+2] = x1 * s + x2 * c
        end
    end
    @test approx_eq(q.storage[2, :, :], expected_q[2, :, :]; atol=1e-12)
end

@testset "cpu matmul! (§LXXV): W is (out, in)" begin
    x = mat(Harpe.Activation, [1.0 2.0 3.0; 4.0 5.0 6.0])       # (seq=2, in=3)
    w = mat(Harpe.ProjectionWeight, [1.0 0.0 1.0; 0.0 1.0 0.0]) # (out=2, in=3)
    dst = mat(Harpe.Activation, zeros(2, 2))
    Harpe.matmul!(Harpe.CPUBackend(), dst, x, w, Harpe.PrefillWorkload())
    expected = x.storage * transpose(w.storage)
    @test approx_eq(dst.storage, expected; atol=1e-12)
end

@testset "cpu softmax! (§LXXV): causal mask before softmax" begin
    scores = mat(Harpe.TemporaryWorkspace, [3.0 1.0 0.5; 2.0 4.0 1.0; 0.1 0.2 0.3])
    dst = mat(Harpe.TemporaryWorkspace, zeros(3, 3))
    Harpe.softmax!(Harpe.CPUBackend(), dst, scores, Harpe.PrefillWorkload())

    # causal: position i attends only to j ≤ i — upper triangle is exactly 0
    for i in 1:3, j in 1:3
        if j > i
            @test dst.storage[i, j] == 0.0
        end
    end
    # rows are probability distributions over the allowed keys
    @test approx_eq(sum(dst.storage; dims=2), ones(3, 1); atol=1e-12)
    # row 0 sees only itself ⇒ 1.0
    @test dst.storage[1, 1] == 1.0
    # row 1: softmax over [2.0, 4.0]
    e = exp(2.0) / (exp(2.0) + exp(4.0))
    @test approx_eq(dst.storage[2, 1:2], [e, 1 - e]; atol=1e-12)
    # row 2: softmax over all three
    r = exp.([0.1, 0.2, 0.3])
    r ./= sum(r)
    @test approx_eq(dst.storage[3, :], r; atol=1e-12)
    # stability: huge scores must not overflow
    big = mat(Harpe.TemporaryWorkspace, [1e6 1e6 1e6; 1e6 1e6 1e6; 1e6 1e6 1e6])
    dst2 = mat(Harpe.TemporaryWorkspace, zeros(3, 3))
    Harpe.softmax!(Harpe.CPUBackend(), dst2, big, Harpe.PrefillWorkload())
    @test all(isfinite, dst2.storage)
end

@testset "cpu softmax! (§LXXV): decode row attends to the whole cache" begin
    # decode step: one query row (seq_q=1) against 3 cached keys — the query
    # is the LAST position, so nothing is masked
    scores = mat(Harpe.TemporaryWorkspace, reshape([1.0, 2.0, 3.0], (1, 3)))
    dst = mat(Harpe.TemporaryWorkspace, zeros(1, 3))
    Harpe.softmax!(Harpe.CPUBackend(), dst, scores, Harpe.DecodeWorkload())
    r = exp.([1.0, 2.0, 3.0])
    r ./= sum(r)
    @test approx_eq(dst.storage, reshape(r, (1, 3)); atol=1e-12)
end

@testset "cpu swiglu! (§LXXV)" begin
    g = mat(Harpe.Activation, [-1.0, 0.5, 2.0])
    u = mat(Harpe.Activation, [3.0, -0.25, 1.0])
    dst = mat(Harpe.Activation, zeros(3))
    Harpe.swiglu!(Harpe.CPUBackend(), dst, g, u, Harpe.PrefillWorkload())
    silu = g.storage ./ (1.0 .+ exp.(.-g.storage))
    @test approx_eq(dst.storage, silu .* u.storage; atol=1e-12)
end

@testset "cpu: decode workload shares prefill math (goal: identical elementwise/matmul)" begin
    rng = deterministic_rng(0xfeed)
    x_arr = randn(rng, 2, 4)
    s_arr = randn(rng, 4)
    w_arr = randn(rng, 3, 4)
    g_arr = randn(rng, 2, 3)
    u_arr = randn(rng, 2, 3)

    # rmsnorm
    dp = mat(Harpe.Activation, zeros(2, 4))
    Harpe.rmsnorm!(
        Harpe.CPUBackend(),
        dp,
        mat(Harpe.Activation, copy(x_arr)),
        mat(Harpe.FrozenParameter, copy(s_arr)),
        Harpe.PrefillWorkload(),
    )
    dd = mat(Harpe.Activation, zeros(2, 4))
    Harpe.rmsnorm!(
        Harpe.CPUBackend(),
        dd,
        mat(Harpe.Activation, copy(x_arr)),
        mat(Harpe.FrozenParameter, copy(s_arr)),
        Harpe.DecodeWorkload(),
    )
    @test approx_eq(dp.storage, dd.storage; atol=0.0)

    # matmul
    mp_ = mat(Harpe.Activation, zeros(2, 3))
    Harpe.matmul!(
        Harpe.CPUBackend(),
        mp_,
        mat(Harpe.Activation, copy(x_arr)),
        mat(Harpe.ProjectionWeight, copy(w_arr)),
        Harpe.PrefillWorkload(),
    )
    md = mat(Harpe.Activation, zeros(2, 3))
    Harpe.matmul!(
        Harpe.CPUBackend(),
        md,
        mat(Harpe.Activation, copy(x_arr)),
        mat(Harpe.ProjectionWeight, copy(w_arr)),
        Harpe.DecodeWorkload(),
    )
    @test approx_eq(mp_.storage, md.storage; atol=0.0)

    # swiglu
    sp = mat(Harpe.Activation, zeros(2, 3))
    Harpe.swiglu!(
        Harpe.CPUBackend(),
        sp,
        mat(Harpe.Activation, copy(g_arr)),
        mat(Harpe.Activation, copy(u_arr)),
        Harpe.PrefillWorkload(),
    )
    sd = mat(Harpe.Activation, zeros(2, 3))
    Harpe.swiglu!(
        Harpe.CPUBackend(),
        sd,
        mat(Harpe.Activation, copy(g_arr)),
        mat(Harpe.Activation, copy(u_arr)),
        Harpe.DecodeWorkload(),
    )
    @test approx_eq(sp.storage, sd.storage; atol=0.0)
end

@testset "cpu: stub invariants survive (§LXX, §LXXV)" begin
    cpu = Harpe.CPUBackend()
    # un-typed 3-arg calls still hit the stub and still decline
    @test_throws Harpe.LoweringNotImplemented Harpe.rmsnorm!(cpu, nothing, nothing)
    @test_throws Harpe.LoweringNotImplemented Harpe.matmul!(cpu, nothing, nothing, nothing)

    # quantize!/dequantize! decline even with fully typed CPU arguments —
    # their math is a Representation-phase concern, not this sprint's
    a = mat(Harpe.Activation, [1.0 2.0])
    p = mat(Harpe.ProjectionWeight, [1.0 2.0])
    @test_throws Harpe.LoweringNotImplemented Harpe.quantize!(
        cpu,
        p,
        a,
        Harpe.PrefillWorkload(),
    )
    @test_throws Harpe.LoweringNotImplemented Harpe.dequantize!(
        cpu,
        a,
        p,
        Harpe.PrefillWorkload(),
    )

    # unset storage is a LOUD error, not a silent no-op
    @test_throws ErrorException Harpe.rmsnorm!(
        cpu,
        mat(Harpe.Activation, zeros(2, 2)),
        mat(Harpe.Activation, [1.0 2.0]),
        Harpe.FrozenParameter(; shape=(2,)),
        Harpe.PrefillWorkload(),
    )

    # vocabulary unchanged: still exactly the eight ops (inventory owns this,
    # repeated here so a drift is caught even if inventory parsing changes)
    @test length(methods(Harpe.rmsnorm!)) == 5   # stub + generic prefill + generic decode + cpu prefill + cpu decode
end
