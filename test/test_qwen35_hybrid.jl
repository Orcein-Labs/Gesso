# Qwen3.5 / Ornith hybrid: Gated DeltaNet CPU bodies + 0.8B load.

using Test
using Gesso

@testset "Gated DeltaNet V-head pairing: grouped vs tiled" begin
    # Two K heads, four V heads. Distinct Q heads so the repeat layout is
    # visible in the output: grouped [K0,K0,K1,K1], tiled [K0,K1,K0,K1].
    seq, nk, nv, dk, dv = 1, 2, 4, 2, 2
    q = zeros(seq, nk, dk)
    k = ones(seq, nk, dk)
    v = ones(seq, nv, dv)
    g = fill(-20.0, seq, nv)   # decay ≈ 0 so the first write dominates
    beta = fill(1.0, seq, nv)
    q[1, 1, 1] = 1.0
    q[1, 2, 1] = 3.0
    function _run(; tiled)
        out = zeros(seq, nv, dv)
        state = zeros(nv, dk, dv)
        Gesso._cpu_gated_delta_rule!(out, q, k, v, g, beta, state; l2norm=false, v_head_tiled=tiled)
        return [out[1, hh, 1] for hh in 1:nv]
    end
    grouped = _run(; tiled=false)
    tiled = _run(; tiled=true)
    @test grouped[1] == grouped[2]
    @test grouped[3] == grouped[4]
    @test grouped[1] != grouped[3]
    @test tiled[1] == tiled[3]
    @test tiled[2] == tiled[4]
    @test tiled[1] != tiled[2]
    @test grouped != tiled
end

@testset "Gated DeltaNet recurrent: zero state, one step" begin
    rng = GessoTestHelpers.deterministic_rng(11)
    seq, nv, dk, dv = 3, 2, 4, 4
    q = randn(rng, seq, nv, dk)
    k = randn(rng, seq, nv, dk)
    v = randn(rng, seq, nv, dv)
    g = fill(-0.5, seq, nv)
    beta = fill(0.25, seq, nv)
    out = zeros(seq, nv, dv)
    state = zeros(nv, dk, dv)
    Gesso._cpu_gated_delta_rule!(out, q, k, v, g, beta, state; l2norm=false)
    @test all(isfinite, out)
    @test all(isfinite, state)
    # a second call continuing the state must change it
    state2 = copy(state)
    out2 = zeros(seq, nv, dv)
    Gesso._cpu_gated_delta_rule!(out2, q, k, v, g, beta, state2; l2norm=false)
    @test state2 != state
end

@testset "Qwen3.5-0.8B local snapshot materializes" begin
    dir = get(ENV, "GESSO_QWEN35_08B_DIR", joinpath(@__DIR__, "..", "snapshots", "Qwen3.5-0.8B"))
    cfg = joinpath(dir, "config.json")
    if !isfile(cfg)
        @info "skip 0.8B snapshot: $dir missing"
        return
    end
    spec = Gesso.architecture_spec(cfg)
    @test spec.family === :qwen3_5
    @test spec.head_dim == 256
    @test spec.mixer !== nothing
    @test spec.mixer.v_head_tiled === false
    @test Gesso.layer_mixer(spec, 0) === :linear_attention
    @test Gesso.layer_mixer(spec, 3) === :full_attention
    @test :linear_attention in Gesso.required_semantics(spec)
    miss = Gesso.import_report(spec)
    @test occursin("Failure boundary: none", miss)

    model, tensors, loaded = Gesso.load_architecture(dir)
    @test loaded.num_layers == 24
    @test length(tensors.blocks) == 24
    @test tensors.blocks[1].mixer === :linear_attention
    @test tensors.blocks[4].mixer === :full_attention
    @test haskey(tensors.blocks[1], :wgdn_qkv)
    @test haskey(tensors.blocks[4], :wq)
    @test haskey(tensors.blocks[4], :wattn_gate)
    @test size(tensors.blocks[4].wq.storage) == (8 * 256, 1024)
    @test size(tensors.blocks[4].wo.storage) == (1024, 8 * 256)
    @test :rms_unit_offset in spec.features

    # ID-gate vs llama.cpp third-column (Qwen3.5-0.8B-BF16.gguf, Hello).
    # Tokenizer id 9419; greedy and top-5 ranks must match.
    logits = Gesso.reference_prefill(model, tensors, [9419]; eps=spec.rms_norm_eps)
    @test size(logits) == (spec.vocab_size, 1)
    @test all(isfinite, logits)
    col = logits[:, 1]
    greedy = argmax(col) - 1
    @test greedy == 11
    top5 = [i - 1 for i in sortperm(col; rev=true)[1:5]]
    @test top5 == [11, 13, 0, 198, 4858]

    # Hybrid generate must keep GDN conv/recurrent state across tokens.
    # llama.cpp Hello continuation (prompt id 9419): 11,271,40,1044,3133,440,264,12654
    ids = Gesso.reference_generate(
        model,
        tensors,
        [9419];
        max_new_tokens=8,
        eps=spec.rms_norm_eps,
    )
    @test ids[1] == 9419
    @test ids[2] == 11
    @test ids[2:9] == [11, 271, 40, 1044, 3133, 440, 264, 12654]

    # Prefill in passes carries KV and GDN state: 300 tokens as 128+128+44
    # continue exactly like one pass.
    prompt = [(i * 7919) % 50000 + 100 for i in 1:300]
    gen(c) = withenv("GESSO_PREFILL_CHUNK" => c) do
        Gesso.reference_generate(model, tensors, prompt; max_new_tokens=6, eps=spec.rms_norm_eps)
    end
    @test gen("128") == gen("0")
end
