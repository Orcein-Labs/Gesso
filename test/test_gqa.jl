# Phase 3 item A tests (§LXXVI): GQA, final RMSNorm, eps/theta knobs.
#
# The GQA micro test computes the WHOLE forward pass test-side with an
# independent formula (repeat KV heads FIRST, then scores) and compares
# against `reference_prefill` at atol=1e-12 (BLAS association vs the
# test's naive loops is not bit-identical in general). toy2's own gates
# (persisted logits atol=1e-10, in-process determinism atol=0) stay in
# test_reference_prefill.jl / test_reference_generate.jl and must keep
# passing unchanged — the defaults here are exactly the Phase 2 constants.

using .GessoTestHelpers: approx_eq, deterministic_rng

# local helper (test_cpu_ops.jl owns `mat`; this file stays self-contained)
gmat(F, arr) = F(; shape=size(arr), storage=arr)

# build the micro GQA model + deterministic tensors:
# vocab 16, dim 16, n_heads 4, n_kv_heads 2 (group 3? no: 4/2 = 2), d_head 4,
# hidden 8, ONE block. Weight shapes follow the §LXXVI walk (wk/wv carry
# n_kv_heads * d_head rows).
function gqa_micro(; n_heads=4, n_kv_heads=2)
    dim, hidden, vocab = 16, 8, 16
    d_head = div(dim, n_heads)
    embedding = gmat(Gesso.EmbeddingTable, randn(deterministic_rng(0x6a), vocab, dim))
    wk_rows = n_kv_heads * d_head
    wq = gmat(Gesso.ProjectionWeight, randn(deterministic_rng(0x6b), dim, dim))
    wk = gmat(Gesso.ProjectionWeight, randn(deterministic_rng(0x6c), wk_rows, dim))
    wv = gmat(Gesso.ProjectionWeight, randn(deterministic_rng(0x6d), wk_rows, dim))
    wo = gmat(Gesso.ProjectionWeight, randn(deterministic_rng(0x6e), dim, dim))
    wgate = gmat(Gesso.ProjectionWeight, randn(deterministic_rng(0x6f), hidden, dim))
    wup = gmat(Gesso.ProjectionWeight, randn(deterministic_rng(0x70), hidden, dim))
    wdown = gmat(Gesso.ProjectionWeight, randn(deterministic_rng(0x71), dim, hidden))
    attn_rms = gmat(Gesso.FrozenParameter, randn(deterministic_rng(0x72), dim))
    ffn_rms = gmat(Gesso.FrozenParameter, randn(deterministic_rng(0x73), dim))
    final_rms = gmat(Gesso.FrozenParameter, randn(deterministic_rng(0x74), dim))
    model = Gesso.Model(;
        vocab_size=vocab,
        embedding=Gesso.Embedding(dim=dim),
        blocks=(
            Gesso.Block(
                Gesso.Attention(n_heads=n_heads, n_kv_heads=n_kv_heads),
                Gesso.SwiGLU(hidden=hidden),
            ),
        ),
    )
    tensors = (
        embedding=embedding,
        blocks=[(
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            wgate=wgate,
            wup=wup,
            wdown=wdown,
            attn_rms=attn_rms,
            ffn_rms=ffn_rms,
        )],
        lm_head=embedding,                      # tied head: same bytes
    )
    return model, tensors, final_rms
end

# independent test-side forward formula (§LXXV recipe + §LXXVI deltas):
# repeat KV heads FIRST, then per-head scores — deliberately a different
# code shape from the implementation's contraction-time repeat.
function gqa_reference_formula(
    model,
    tensors,
    tokens;
    eps=1e-6,
    theta=10000.0,
    final=nothing,
)
    dim = model.embedding.dim
    seq = length(tokens)
    blk = model.blocks[1]
    nh, nkv = blk.attention.n_heads, blk.attention.n_kv_heads
    dh = div(dim, nh)
    g = div(nh, nkv)
    E = tensors.embedding.storage
    bt = tensors.blocks[1]

    h = zeros(seq, dim)
    for (t, tok) in enumerate(tokens)
        h[t, :] .= E[tok+1, :]                  # 0-based ids
    end

    # attention sublayer (pre-norm)
    normed = similar(h)
    for t in 1:seq
        rms = sqrt(sum(abs2, h[t, :]) / dim + eps)
        normed[t, :] .= (h[t, :] ./ rms) .* bt.attn_rms.storage
    end
    q = normed * transpose(bt.wq.storage)
    k = normed * transpose(bt.wk.storage)
    v = normed * transpose(bt.wv.storage)

    qh = zeros(seq, nh, dh)
    kh = zeros(seq, nkv, dh)
    vh = zeros(seq, nkv, dh)
    for t in 1:seq, hh in 1:nh, j in 1:dh
        qh[t, hh, j] = q[t, (hh-1)*dh+j]
    end
    for t in 1:seq, hk in 1:nkv, j in 1:dh
        kh[t, hk, j] = k[t, (hk-1)*dh+j]
        vh[t, hk, j] = v[t, (hk-1)*dh+j]
    end

    # rope: pairwise rotate, position = t-1 (0-based), Q at nh heads, K at nkv
    for t in 1:seq
        m = Float64(t - 1)
        for hh in 1:nh, i in 0:(dh÷2-1)
            θv = m * theta^(-2i / dh)
            c, s = cos(θv), sin(θv)
            x1, x2 = qh[t, hh, 2i+1], qh[t, hh, 2i+2]
            qh[t, hh, 2i+1] = x1 * c - x2 * s
            qh[t, hh, 2i+2] = x1 * s + x2 * c
        end
        for hk in 1:nkv, i in 0:(dh÷2-1)
            θv = m * theta^(-2i / dh)
            c, s = cos(θv), sin(θv)
            x1, x2 = kh[t, hk, 2i+1], kh[t, hk, 2i+2]
            kh[t, hk, 2i+1] = x1 * c - x2 * s
            kh[t, hk, 2i+2] = x1 * s + x2 * c
        end
    end

    # GQA: REPEAT each kv head g times (for the contraction only)
    kr = zeros(seq, nh, dh)
    vr = zeros(seq, nh, dh)
    for u in 1:seq, hh in 1:nh, j in 1:dh
        kr[u, hh, j] = kh[u, div(hh-1, g)+1, j]
        vr[u, hh, j] = vh[u, div(hh-1, g)+1, j]
    end

    # Independent per-head probabilities, never aggregate heads before softmax.
    attn = zeros(seq, nh, dh)
    for hh in 1:nh
        scores = qh[:, hh, :] * transpose(kr[:, hh, :]) / sqrt(dh)
        for t in 1:seq, u in 1:seq
            u > t && (scores[t, u] = -Inf)
        end
        for t in 1:seq
            weights = exp.(scores[t, :] .- maximum(scores[t, :]))
            weights ./= sum(weights)
            for j in 1:dh
                attn[t, hh, j] = sum(weights .* vr[:, hh, j])
            end
        end
    end
    merged = zeros(seq, dim)
    for t in 1:seq, hh in 1:nh, j in 1:dh
        merged[t, (hh-1)*dh+j] = attn[t, hh, j]
    end
    h .+= merged * transpose(bt.wo.storage)

    # ffn sublayer (pre-norm)
    for t in 1:seq
        rms = sqrt(sum(abs2, h[t, :]) / dim + eps)
        normed[t, :] .= (h[t, :] ./ rms) .* bt.ffn_rms.storage
    end
    gate = normed * transpose(bt.wgate.storage)
    up = normed * transpose(bt.wup.storage)
    act = gate ./ (1.0 .+ exp.(-gate)) .* up
    h .+= act * transpose(bt.wdown.storage)

    # final RMSNorm when the composition carries one (§LXXVI)
    if final !== nothing
        hn = similar(h)
        for t in 1:seq
            rms = sqrt(sum(abs2, h[t, :]) / dim + eps)
            hn[t, :] .= (h[t, :] ./ rms) .* final.storage
        end
        h = hn
    end

    return permutedims(h * transpose(E))        # (vocab, seq)
end

@testset "gqa micro: reference_prefill vs independent formula (§LXXVI)" begin
    model, tensors, final_rms = gqa_micro()
    tokens = [0, 1, 2]

    # without final norm
    got = Gesso.reference_prefill(model, tensors, tokens)
    want = gqa_reference_formula(model, tensors, tokens)
    @test size(got) == (16, 3)
    @test approx_eq(got, want; atol=1e-12)

    # with final norm: the §LXXVI placement (after blocks, before tied head)
    got_f = Gesso.reference_prefill(model, merge(tensors, (final_rms=final_rms,)), tokens)
    want_f = gqa_reference_formula(model, tensors, tokens; final=final_rms)
    @test approx_eq(got_f, want_f; atol=1e-12)
    @test got_f != got                          # the norm actually did something

    # in-process determinism at the toy tolerance
    @test approx_eq(Gesso.reference_prefill(model, tensors, tokens), got; atol=0.0)
end

@testset "gqa: repeat-for-contraction equals materialized MHA repeat (atol=0)" begin
    model, tensors, _ = gqa_micro()             # 4 q heads, 2 kv heads
    dim = model.embedding.dim
    dh = div(dim, 4)
    g = 2
    # materialize the repeat into the weights: MHA Wk rows per q head =
    # the gqa kv head's rows it maps to
    rep_rows(W) =
        vcat([W.storage[((div(h-1, g))*dh+1):((div(h-1, g)+1)*dh), :] for h in 1:4]...)
    mha_wk = gmat(Gesso.ProjectionWeight, rep_rows(tensors.blocks[1].wk))
    mha_wv = gmat(Gesso.ProjectionWeight, rep_rows(tensors.blocks[1].wv))
    mha_model = Gesso.Model(;
        vocab_size=model.vocab_size,
        embedding=model.embedding,
        blocks=(Gesso.Block(
            Gesso.Attention(n_heads=4),     # n_kv_heads defaults to n_heads
            model.blocks[1].ffn,
        ),),
    )
    mha_tensors = (
        embedding=tensors.embedding,
        blocks=[merge(tensors.blocks[1], (wk=mha_wk, wv=mha_wv))],
        lm_head=tensors.lm_head,
    )
    tokens = [0, 1, 2]
    @test approx_eq(
        Gesso.reference_prefill(model, tensors, tokens),
        Gesso.reference_prefill(mha_model, mha_tensors, tokens);
        atol=0.0,
    )
end

@testset "gqa: n_kv_heads that does not divide n_heads errors loudly" begin
    model, tensors, _ = gqa_micro(n_heads=4, n_kv_heads=3)
    @test_throws ErrorException Gesso.reference_prefill(model, tensors, [0, 1])
    @test_throws ErrorException Gesso.reference_generate(model, tensors, [0, 1])
end

@testset "rmsnorm eps keyword (§LXXVI): default is Phase 2, knob differs" begin
    cpu = Gesso.CPUBackend()
    x = [1.0 2.0; 3.0 4.0]
    s = [1.0, 1.0]
    d_def = gmat(Gesso.Activation, zeros(2, 2))
    Gesso.rmsnorm!(
        cpu,
        d_def,
        gmat(Gesso.Activation, copy(x)),
        gmat(Gesso.FrozenParameter, s),
        Gesso.PrefillWorkload(),
    )
    d_exp = gmat(Gesso.Activation, zeros(2, 2))
    Gesso.rmsnorm!(
        cpu,
        d_exp,
        gmat(Gesso.Activation, copy(x)),
        gmat(Gesso.FrozenParameter, s),
        Gesso.PrefillWorkload();
        eps=1e-6,
    )
    @test approx_eq(d_def.storage, d_exp.storage; atol=0.0)   # default == Phase 2 constant

    d_5 = gmat(Gesso.Activation, zeros(2, 2))
    Gesso.rmsnorm!(
        cpu,
        d_5,
        gmat(Gesso.Activation, copy(x)),
        gmat(Gesso.FrozenParameter, s),
        Gesso.PrefillWorkload();
        eps=1e-5,
    )
    @test d_def.storage[1, 1] != d_5.storage[1, 1]            # the knob is real
end

@testset "rope theta keyword (§LXXVI): default is Phase 2, knob differs" begin
    cpu = Gesso.CPUBackend()
    q0 = reshape(Float64[i for i in 1:16], (2, 2, 4))
    k0 = reshape(Float64[i for i in 17:32], (2, 2, 4))
    q1, k1 = gmat(Gesso.Activation, copy(q0)), gmat(Gesso.Activation, copy(k0))
    Gesso.rope!(cpu, q1, k1, [0, 1], Gesso.PrefillWorkload())
    q2, k2 = gmat(Gesso.Activation, copy(q0)), gmat(Gesso.Activation, copy(k0))
    Gesso.rope!(cpu, q2, k2, [0, 1], Gesso.PrefillWorkload(); theta=10000.0)
    @test approx_eq(q1.storage, q2.storage; atol=0.0)          # default == Phase 2 constant
    @test approx_eq(k1.storage, k2.storage; atol=0.0)

    q3, k3 = gmat(Gesso.Activation, copy(q0)), gmat(Gesso.Activation, copy(k0))
    Gesso.rope!(cpu, q3, k3, [0, 1], Gesso.PrefillWorkload(); theta=100000.0)
    @test approx_eq(q1.storage[1, :, :], q3.storage[1, :, :]; atol=0.0)
    # i=0 pair (components 1,2) has θ = m·θ⁰ = 1 rad for ANY theta — only the
    # i=1 pair (components 3,4) distinguishes the base
    @test q1.storage[2, 1, 1] == q3.storage[2, 1, 1]
    @test q1.storage[2, 1, 3] != q3.storage[2, 1, 3]
    @test q1.storage[2, 1, 4] != q3.storage[2, 1, 4]
end

@testset "final_rms present vs absent (§LXXVI)" begin
    model, tensors, final_rms = gqa_micro()
    tokens = [0, 1, 2]
    without = Gesso.reference_prefill(model, tensors, tokens)
    with = Gesso.reference_prefill(model, merge(tensors, (final_rms=final_rms,)), tokens)
    @test without != with
    # determinism on both paths
    @test approx_eq(Gesso.reference_prefill(model, tensors, tokens), without; atol=0.0)
    @test approx_eq(
        Gesso.reference_prefill(model, merge(tensors, (final_rms=final_rms,)), tokens),
        with;
        atol=0.0,
    )
end

@testset "gqa decode: cache stays at n_kv_heads, generate stays deterministic" begin
    model, tensors, _ = gqa_micro()
    info = Ref{Any}(nothing)
    ids = Gesso.reference_generate(model, tensors, [0, 1, 2]; max_new_tokens=3, info=info)
    @test length(ids) == 6
    kv = info[].kv_len
    @test kv == 6                               # prefix length: 3 + 3 steps
    @test Gesso.reference_generate(model, tensors, [0, 1, 2]; max_new_tokens=3) == ids
end

@testset "cpu: stub invariants survive Phase 3 (§LXX, §LXXVI)" begin
    cpu = Gesso.CPUBackend()
    @test_throws Gesso.LoweringNotImplemented Gesso.rmsnorm!(cpu, nothing, nothing)
    @test_throws Gesso.LoweringNotImplemented Gesso.rope!(cpu, nothing, nothing, nothing)
end
