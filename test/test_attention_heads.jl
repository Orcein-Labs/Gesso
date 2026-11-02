using Test
using Random: Xoshiro

@testset "Attention heads own independent probabilities (Regime I arm 1)" begin
    q = zeros(1, 2, 2)
    q[1, :, 1] .= 1
    k = zeros(2, 2, 2)
    k[:, 1, 1] .= [1, -1]
    k[:, 2, 1] .= [-1, 1]
    v = zeros(2, 2, 2)
    v[:, 1, 1] .= [1, 0]
    v[:, 2, 1] .= [10, 20]
    attn = zeros(1, 2, 2)
    scores = Gesso.TemporaryWorkspace(; shape=(1, 2), storage=zeros(1, 2))
    probs = Gesso.TemporaryWorkspace(; shape=(1, 2), storage=zeros(1, 2))
    run_attention() = Gesso.Inference._attention_heads!(
        Gesso.CPUBackend(),
        attn,
        scores,
        probs,
        q,
        k,
        v,
        2,
        2,
        1,
        2,
        Gesso.DecodeWorkload(),
    )
    run_attention()
    a, b = exp(1 / sqrt(2)), exp(-1 / sqrt(2))
    @test attn[1, 1, 1] ≈ a / (a + b)
    @test attn[1, 2, 1] ≈ (10b + 20a) / (a + b)
    @test attn[1, :, 2] == [0, 0]
    other_head = copy(attn[:, 2, :])
    q[1, 1, 1] = 100
    run_attention()
    @test attn[:, 2, :] == other_head
end

# The CPU score and value loops are reordered for memory and split across
# threads; each element must still take the plain triple-loop order, which
# the decode path is pinned to bit-for-bit.
@testset "CPU attention loops keep the reference per-element order" begin
    rng = Xoshiro(20261022)
    L, K, H = 64, 600, 3
    # d = 64 scales by an exact power of two (sqrt 8), d = 48 by a true divide.
    for d in (64, 48)
        q = randn(rng, Float32, L, H, d)
        k = randn(rng, Float32, K, H, d)
        v = randn(rng, Float32, K, H, d)
        probs = rand(rng, Float32, L, K)
        for hh in 1:H
            ref = zeros(Float32, L, K)
            for t in 1:L, u in 1:K, j in 1:d
                ref[t, u] += q[t, hh, j] * k[u, hh, j] / sqrt(d)
            end
            sc = zeros(Float32, L, K)
            Gesso.Inference._attention_scores_cpu!(sc, q, k, hh, d, L, K)
            @test reinterpret(UInt32, sc) == reinterpret(UInt32, ref)
        end
        ref = zeros(Float32, L, H, d)
        attn = zeros(Float32, L, H, d)
        for hh in 1:H
            for t in 1:L, j in 1:d, u in 1:K
                ref[t, hh, j] += probs[t, u] * v[u, hh, j]
            end
            Gesso.Inference._attention_values_cpu!(attn, probs, v, hh, d, L, K)
        end
        @test reinterpret(UInt32, attn) == reinterpret(UInt32, ref)
    end
end

# _attention_heads! reads only the keys the causal and window masks keep, and
# on threaded CPU runs scores, softmax and outputs in one pass over the rows;
# the result must equal scoring every key, masking, and summing every key, head
# by head. L < K covers a prompt chunk after cached keys; d = 48 a true divide.
@testset "CPU prefill attention skips masked keys bit for bit" begin
    rng = Xoshiro(0x5a1e)
    H = 2
    # The small shapes take the one-pass rows form too, since every multi-row
    # CPU call does; d = 128 and 80 are true divides by the corrected reciprocal.
    for (L, K, d) in ((300, 300, 64), (200, 450, 48), (3, 3, 128), (16, 40, 80))
        q = randn(rng, Float32, L, H, d)
        k = randn(rng, Float32, K, H, d)
        v = randn(rng, Float32, K, H, d)
        for window in (0, 100)
            ref = zeros(Float32, L, H, d)
            for hh in 1:H
                sc = Gesso.Inference._attention_scores_cpu!(
                    zeros(Float32, L, K),
                    q,
                    k,
                    hh,
                    d,
                    L,
                    K,
                )
                window > 0 && Gesso.Inference._window_mask!(sc, L, K, window)
                probs = Gesso._cpu_softmax_storage!(similar(sc), sc)
                Gesso.Inference._attention_values_cpu!(ref, probs, v, hh, d, L, K)
            end
            ws() = Gesso.TemporaryWorkspace(; shape=(L, K), storage=zeros(Float32, L, K))
            attn = zeros(Float32, L, H, d)
            Gesso.Inference._attention_heads!(
                Gesso.CPUBackend(),
                attn,
                ws(),
                ws(),
                q,
                k,
                v,
                H,
                d,
                L,
                K,
                Gesso.PrefillWorkload();
                window,
            )
            @test reinterpret(UInt32, attn) == reinterpret(UInt32, ref)
        end
    end
end

# A row with no finite score fails the softmax inside a threaded task; it must
# surface as the same GessoError the per-head path throws.
@testset "threaded CPU prefill attention rethrows the softmax error" begin
    rng = Xoshiro(0x0bad)
    L, H, d = 300, 2, 64
    q = randn(rng, Float32, L, H, d)
    k = randn(rng, Float32, L, H, d)
    v = randn(rng, Float32, L, H, d)
    q[150, 2, 3] = NaN32
    ws() = Gesso.TemporaryWorkspace(; shape=(L, L), storage=zeros(Float32, L, L))
    err = try
        Gesso.Inference._attention_heads!(
            Gesso.CPUBackend(),
            zeros(Float32, L, H, d),
            ws(),
            ws(),
            q,
            k,
            v,
            H,
            d,
            L,
            L,
            Gesso.PrefillWorkload(),
        )
        nothing
    catch e
        e
    end
    @test err isa Gesso.GessoError && err.code == Gesso.ERR_NUMERICAL_INSTABILITY
end

# The score scale for a non-power-of-two sqrt(d_head) takes the reciprocal with
# an fma correction. It must give the divide's Float64 bits for every Float32;
# the full 2^32 sweep ran offline for every d_head it is used on, so this
# samples edge values and random bit patterns.
@testset "corrected reciprocal score scale equals the Float64 divide" begin
    I = Gesso.Inference
    rng = Xoshiro(0x5ca1e)
    edges = Float32[0, -0.0, 1, -1, floatmin(Float32), -floatmin(Float32),
        nextfloat(0.0f0), -nextfloat(0.0f0), floatmax(Float32), -floatmax(Float32),
        Inf32, -Inf32, prevfloat(Inf32), 3.0f-39, 1.0f20]
    for d in (48, 80, 96, 128, 160, 192, 512, 1000, I._SCORE_RECIP_MAX_D - 1)
        o = I._attention_score_scale(d)
        @test o isa I._ScoreRecip
        s = sqrt(d)
        xs = vcat(edges, reinterpret(Float32, rand(rng, UInt32, 200_000)))
        @test all(x -> isnan(x) || o(x) === Float64(x) / s, xs)
        @test isnan(o(NaN32))
    end
    @test I._attention_score_scale(64) isa I._ScoreMul
    @test I._attention_score_scale(I._SCORE_RECIP_MAX_D + 1) isa I._ScoreDiv
end
