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
    L, K, H, d = 64, 600, 3, 64
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
