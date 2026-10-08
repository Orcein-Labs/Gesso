using Test

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
