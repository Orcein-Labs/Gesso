# CPU MoE routing (softmax top-k + SwiGLU experts + shared expert) and the
# qwen3_5_moe doorway.

using Test
using Random
using Gesso

@testset "HostOffload is a host-resident weight wrapper" begin
    a = Float32[1 2; 3 4]
    s = Gesso.HostOffload(a)
    @test size(s) == (2, 2)
    @test size(s, 1) == 2
    @test ndims(s) == 2
    @test eltype(s) === Float32
    @test length(s) == 4
    @test Gesso.Inference._offload_host_array(s) === a
    @test Gesso.Inference._is_host_float_storage(s)
    @test Gesso.Inference._check_device_storage(
        :probe,
        Gesso.CPUBackend(),
        Gesso.Activation(; shape=(2, 2), storage=s),
    ) === nothing
end

@testset "GPU budget reads GESSO_GPU_BUDGET_MIB" begin
    withenv("GESSO_GPU_BUDGET_MIB" => "6144") do
        @test Gesso.Inference._gpu_budget_bytes() == Int64(6144) * 1024 * 1024
    end
    withenv("GESSO_GPU_BUDGET_MIB" => "0") do
        @test Gesso.Inference._gpu_budget_bytes() == 0
    end
    withenv("GESSO_GPU_BUDGET_MIB" => "1") do
        @test Gesso.Inference._gpu_budget_bytes() == 1024 * 1024
    end
end

@testset "GPU budget from free memory and the reserve" begin
    G = Gesso.Inference
    mib = Int64(1) << 20
    r = G._gpu_reserve(; kv_per_token=10_000, ctx=1000, prefill=300 * mib, floor=1024 * mib, blas=64 * mib)
    @test r.kv == 10_000_000
    @test r.total == r.kv + 300mib + 1024mib + 64mib + 128mib
    withenv("GESSO_GPU_BUDGET_MIB" => nothing) do
        # normal probe: free minus reserve
        @test G._gpu_budget_decide(8000mib, r.total) == (8000mib - r.total, :probe)
        # a bigger card gets a bigger budget
        @test G._gpu_budget_decide(24000mib, r.total)[1] > G._gpu_budget_decide(8000mib, r.total)[1]
        # probe smaller than the reserve: nothing goes on the device, no throw
        @test G._gpu_budget_decide(r.total - 1, r.total) == (0, :probe)
        @test G._gpu_budget_decide(0, r.total) == (0, :probe)
        # no probe: the fixed 4800 MiB
        @test G._gpu_budget_decide(nothing, r.total) == (4800mib, :fallback)
    end
    # the env wins over any probe
    withenv("GESSO_GPU_BUDGET_MIB" => "123") do
        @test G._gpu_budget_decide(8000mib, r.total) == (123mib, :env)
        @test G._gpu_budget_decide(nothing, r.total) == (123mib, :env)
        @test G._gpu_budget_decide(1, r.total) == (123mib, :env)
    end
    # context: server hint, then env, then the default
    withenv("GESSO_GPU_BUDGET_CTX" => nothing) do
        @test G._gpu_budget_ctx() == 65536
        G.gpu_budget_context!(4096)
        @test G._gpu_budget_ctx() == 4096
        G.gpu_budget_context!(0)
        withenv("GESSO_GPU_BUDGET_CTX" => "2048") do
            @test G._gpu_budget_ctx() == 2048
        end
    end
    @test G._gpu_budget_pass_rows(300) == 300
    @test G._gpu_budget_pass_rows(65536) == 1024
    # a hand-built toy has no spec: no KV is reserved
    geo = G._gpu_geometry((; embedding=nothing))
    @test G._gpu_kv_per_token(geo, :cuda) == 0
end

@testset "MoE top-k SwiGLU: unit router" begin
    seq, h, n_exp, ff, k = 1, 4, 4, 4, 2
    x = ones(seq, h)
    router = zeros(n_exp, h)
    router[1, :] .= 10.0
    router[2, :] .= 5.0
    gate3 = zeros(n_exp, ff, h)
    up3 = zeros(n_exp, ff, h)
    down3 = zeros(n_exp, h, ff)
    # expert 1: identity-ish SwiGLU contribution
    for j in 1:ff, i in 1:h
        gate3[1, j, i] = i == j ? 1.0 : 0.0
        up3[1, j, i] = i == j ? 1.0 : 0.0
        down3[1, i, j] = i == j ? 1.0 : 0.0
        gate3[2, j, i] = i == j ? 0.5 : 0.0
        up3[2, j, i] = i == j ? 0.5 : 0.0
        down3[2, i, j] = i == j ? 0.5 : 0.0
    end
    y = zeros(seq, h)
    Gesso._cpu_moe_topk_swiglu!(y, x, router, gate3, up3, down3, k)
    @test all(isfinite, y)
    @test any(!iszero, y)
    # experts 3 and 4 have zero weights after top-2, so y is finite and
    # driven by experts 1 and 2 only
    y_only12 = copy(y)
    router[3, :] .= 100.0
    y2 = zeros(seq, h)
    Gesso._cpu_moe_topk_swiglu!(y2, x, router, gate3, up3, down3, k)
    @test y2 != y_only12
end

@testset "qwen3_5_moe adapter doorway" begin
    text = Dict{String, Any}(
        "hidden_size" => 16,
        "num_hidden_layers" => 4,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "intermediate_size" => 8,
        "vocab_size" => 32,
        "head_dim" => 8,
        "full_attention_interval" => 4,
        "linear_num_key_heads" => 2,
        "linear_key_head_dim" => 4,
        "linear_num_value_heads" => 2,
        "linear_value_head_dim" => 4,
        "linear_conv_kernel_dim" => 4,
        "attn_output_gate" => true,
        "num_experts" => 8,
        "num_experts_per_tok" => 2,
        "moe_intermediate_size" => 8,
        "shared_expert_intermediate_size" => 8,
        "rms_norm_eps" => 1e-6,
        "rope_parameters" => Dict{String, Any}(
            "rope_theta" => 10000.0,
            "rope_type" => "default",
            "partial_rotary_factor" => 0.5,
            "mrope_section" => [1, 1],
        ),
    )
    spec = Gesso.parse_config(
        Gesso.Qwen35MoeAdapter(),
        Dict{String, Any}("model_type" => "qwen3_5_moe", "text_config" => text),
    )
    @test spec.family === :qwen3_5_moe
    @test spec.moe !== nothing
    @test spec.moe.n_experts == 8
    @test spec.moe.n_expert_used == 2
    @test spec.moe.expert_ff == 8
    @test :moe in spec.features
    @test :moe_routing in Gesso.required_semantics(spec)
    report = Gesso.import_report(spec)
    @test occursin("Failure boundary: none", report)

    pmap = Gesso.param_map(Gesso.Qwen35MoeAdapter(), spec)
    @test pmap.family === :qwen3_5_moe
    externals = Set(r.external for r in pmap.layer_refs)
    @test "mlp.gate.weight" in externals
    @test "mlp.experts.gate_proj.weight" in externals
    @test "mlp.gate_proj.weight" ∉ externals

    gmap = Gesso.gguf_param_map(Gesso.Qwen35MoeAdapter(), spec)
    @test gmap.layer_prefix == "blk."
    @test gmap.embedding_external == "token_embd.weight"
    @test any(r -> r.external == "ffn_gate_exps.weight", gmap.layer_refs)
    @test any(r -> r.external == "attn_qkv.weight", gmap.linear_refs)
end

@testset "synthetic MoE block prefills" begin
    h, nh, nkv, hd, vocab = 8, 2, 2, 4, 16
    n_exp, k, ff = 4, 2, 8
    spec = Gesso.ArchitectureSpec(;
        family=:probe,
        hidden_size=h,
        num_layers=1,
        n_heads=nh,
        n_kv_heads=nkv,
        vocab_size=vocab,
        intermediate_size=ff,
        head_dim=hd,
        moe=Gesso.MoeSpec(; n_experts=n_exp, n_expert_used=k, expert_ff=ff, shared_ff=ff),
    )
    model = Gesso.spec_to_model(spec)
    rng = GessoTestHelpers.deterministic_rng(3)
    emb = randn(rng, vocab, h)
    function pw(sh)
        return Gesso.ProjectionWeight(; shape=sh, storage=randn(rng, sh...))
    end
    function fr(sh)
        return Gesso.FrozenParameter(; shape=sh, storage=ones(sh...))
    end
    bt = (
        mixer=:full_attention,
        n_expert_used=k,
        attn_rms=fr((h,)),
        wq=pw((nh * hd, h)),
        wk=pw((nkv * hd, h)),
        wv=pw((nkv * hd, h)),
        wo=pw((h, nh * hd)),
        ffn_rms=fr((h,)),
        wrouter=pw((n_exp, h)),
        wexp_gate=Gesso.ExpertWeight(;
            shape=(n_exp, ff, h),
            storage=randn(rng, n_exp, ff, h),
        ),
        wexp_up=Gesso.ExpertWeight(;
            shape=(n_exp, ff, h),
            storage=randn(rng, n_exp, ff, h),
        ),
        wexp_down=Gesso.ExpertWeight(;
            shape=(n_exp, h, ff),
            storage=randn(rng, n_exp, h, ff),
        ),
        wshared_gate=pw((ff, h)),
        wshared_up=pw((ff, h)),
        wshared_down=pw((h, ff)),
        wshared_gate_inp=fr((h,)),
    )
    tensors = (
        embedding=Gesso.EmbeddingTable(; shape=(vocab, h), storage=emb),
        blocks=[bt],
        lm_head=Gesso.EmbeddingTable(; shape=(vocab, h), storage=emb),
        final_rms=fr((h,)),
        rope=spec.rope,
        spec=spec,
    )
    logits = Gesso.reference_prefill(model, tensors, [1, 2]; eps=spec.rms_norm_eps)
    @test size(logits) == (vocab, 2)
    @test all(isfinite, logits)
    ids = Gesso.reference_generate(
        model,
        tensors,
        [1];
        max_new_tokens=3,
        eps=spec.rms_norm_eps,
    )
    @test length(ids) == 4
    @test ids[1] == 1
    @test all(0 .<= ids .< vocab)
end

@testset "host qgemv: row-chunked threads are bit-identical to serial" begin
    Inf_ = Gesso.Inference
    rng = Random.Xoshiro(20261008)
    ne0, ne1 = 512, 37
    x = randn(rng, Float32, ne0)
    cases = (
        (Inf_.GGML_TYPE_IQ2_XS, 74),
        (Inf_.GGML_TYPE_IQ3_XXS, 98),
        (Inf_.GGML_TYPE_IQ4_XS, 136),
    )
    for (ty, bsz) in cases
        q = rand(rng, UInt8, ne1 * (ne0 ÷ 256) * bsz)
        y1 = zeros(Float32, ne1)
        Inf_._qgemv_bytes!(y1, x, q, ty, ne0, ne1, 0.5f0, 0.0f0)
        for nc in (1, 2, 3, 5, 8, 37, 64)
            y2 = zeros(Float32, ne1)
            Inf_._qgemv_bytes_threaded!(y2, x, q, ty, ne0, ne1, 0.5f0, 0.0f0; nchunks=nc)
            @test reinterpret(UInt32, y2) == reinterpret(UInt32, y1)
        end
        y3 = zeros(Float32, ne1)
        Inf_._qgemv_bytes_threaded!(y3, x, q, ty, ne0, ne1, 0.5f0, 0.0f0)
        @test reinterpret(UInt32, y3) == reinterpret(UInt32, y1)
    end
    qf = reinterpret(UInt8, randn(rng, Float32, ne0 * ne1))
    y1 = zeros(Float32, ne1)
    Inf_._qgemv_bytes!(y1, x, qf, Inf_.GGML_TYPE_F32, ne0, ne1, 1.0f0, 0.0f0)
    for nc in (1, 4, 37)
        y2 = zeros(Float32, ne1)
        Inf_._qgemv_bytes_threaded!(
            y2,
            x,
            qf,
            Inf_.GGML_TYPE_F32,
            ne0,
            ne1,
            1.0f0,
            0.0f0;
            nchunks=nc,
        )
        @test reinterpret(UInt32, y2) == reinterpret(UInt32, y1)
    end
end
