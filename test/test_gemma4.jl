using Test, Random
using Gesso

include(joinpath(@__DIR__, "gemma4_toy.jl"))


mutable struct _Gemma4Capture
    rows::Vector{Vector{Float32}}
end
function Gesso.Inference._sample_id(s::_Gemma4Capture, logits)
    row = Float32.(vec(logits))
    push!(s.rows, copy(row))
    return argmax(row) - 1
end

struct _MockNonCPUBackend <: Gesso.AbstractGessoBackend
    name::Symbol
end
Gesso.backend_name(b::_MockNonCPUBackend) = b.name
Gesso.supports(::_MockNonCPUBackend, ::Symbol) = false

# -----------------------------------------------------------------------------
# Testset 1: Metadata and capability boundary
# -----------------------------------------------------------------------------
@testset "Gemma4: metadata and capability boundary" begin
    kv = _gemma4_toy_kv()
    spec = _gemma4_inf.spec_from_gguf_kv(kv)
    @test spec.family === :gemma4
    @test spec.num_layers == 2
    @test spec.hidden_size == 32
    @test spec.n_heads == 4
    @test spec.sliding_window == 4
    @test spec.embedding_scale ≈ sqrt(32.0)
    @test spec.final_logit_softcap == 30.0

    # Per-layer geometry accessors
    @test _gemma4_inf.layer_window(spec, 0) == 4
    @test _gemma4_inf.layer_window(spec, 1) == 0
    @test _gemma4_inf.layer_head_dim(spec, 0) == 8
    @test _gemma4_inf.layer_head_dim(spec, 1) == 16
    @test _gemma4_inf.layer_n_kv_heads(spec, 0) == 2
    @test _gemma4_inf.layer_n_kv_heads(spec, 1) == 1

    # Semantic capabilities
    reqs = Gesso.required_semantics(spec)
    for cap in (
        :embedding_scale,
        :logit_softcap,
        :v_norm_weightless,
        :k_only_global_attention,
        :gemma4_parallel_moe,
        :layer_output_scale,
        :rope_freq_factors,
    )
        @test cap in reqs
        @test Gesso.supports(Gesso.CPUBackend(), cap)
    end

    # Import report
    report = Gesso.import_report(spec)
    @test occursin("gemma4", report)
    @test occursin("Failure boundary: none", report)

    # Explicit typed refusals for unsupported Gemma 4 features & missing metadata
    for (mut!, expected_reason) in (
        (d -> (d["gemma4.embedding_length_per_layer_input"] = 256), :unsupported_architecture),
        (d -> (d["gemma4.attention.shared_kv_layers"] = 4), :unsupported_architecture),
        (d -> (d["gemma4.expert_count"] = 0), :unsupported_architecture),
        (d -> delete!(d, "gemma4.expert_count"), :unsupported_architecture),
        (d -> delete!(d, "gemma4.block_count"), :missing_metadata),
        (d -> delete!(d, "gemma4.vocab_size"), :missing_metadata),
    )
        bad = copy(kv)
        mut!(bad)
        err = try
            _gemma4_inf.spec_from_gguf_kv(bad)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError
        @test err.detail[:reason] === expected_reason
    end
end

# -----------------------------------------------------------------------------
# Testset 2: LoweringNotImplemented refusals on non-CPU backends
# -----------------------------------------------------------------------------
@testset "Gemma4: LoweringNotImplemented on non-CPU backends" begin
    backends = Gesso.AbstractGessoBackend[
        _MockNonCPUBackend(:cuda),
        _MockNonCPUBackend(:lava),
    ]
    if isdefined(Gesso, :CUDABackend)
        push!(backends, Gesso.CUDABackend())
    end
    # LavaBackend is not listed: it lowers these hooks (test_lava_gemma4.jl).

    spec = _gemma4_inf.spec_from_gguf_kv(_gemma4_toy_kv())
    by = _gemma4_toy(spec)
    model, tensors = _gemma4_bind(spec, by)
    bt = tensors.blocks[1]
    h_dummy = Gesso.Activation(; shape=(1, spec.hidden_size), storage=zeros(Float32, 1, spec.hidden_size))
    logits_dummy = Gesso.Activation(; shape=(1, spec.vocab_size), storage=zeros(Float32, 1, spec.vocab_size))

    for be in backends
        # supports query must report false for non-CPU backends
        for cap in (
            :embedding_scale,
            :logit_softcap,
            :v_norm_weightless,
            :k_only_global_attention,
            :gemma4_parallel_moe,
            :layer_output_scale,
            :rope_freq_factors,
        )
            @test !Gesso.supports(be, cap)
        end

        # Direct lowering calls must refuse explicitly with LoweringNotImplemented
        @test_throws Gesso.LoweringNotImplemented _gemma4_inf._apply_embedding_scale!(be, h_dummy, tensors)
        @test_throws Gesso.LoweringNotImplemented _gemma4_inf._apply_logit_softcap!(be, logits_dummy, tensors)
        @test_throws Gesso.LoweringNotImplemented _gemma4_inf._gemma4_v_norm!(be, zeros(Float32, 1, 16), 1e-6; nheads=2, hd=8)
        @test_throws Gesso.LoweringNotImplemented _gemma4_inf._gemma4_k_only_v!(be, zeros(Float32, 1, 16))
        @test_throws Gesso.LoweringNotImplemented _gemma4_inf._gemma4_parallel_moe_ffn!(be, h_dummy, bt, 1e-6)
        @test_throws Gesso.LoweringNotImplemented _gemma4_inf._gemma4_layer_output_scale!(be, zeros(Float32, 1, 32), Float32[1.0])
        @test_throws Gesso.LoweringNotImplemented _gemma4_inf._apply_rope_neox!(be, zeros(Float32, 1, 16), 0:0, 2, 8, 8, [1.0, 0.5, 0.25, 0.125])
    end
end

# -----------------------------------------------------------------------------
# Testset 3: Toy prefill, decode step equivalence, and negative controls
# -----------------------------------------------------------------------------
@testset "Gemma4: toy prefill, decode equivalence, and negative controls" begin
    spec = _gemma4_inf.spec_from_gguf_kv(_gemma4_toy_kv())
    by = _gemma4_toy(spec)
    model, tensors = _gemma4_bind(spec, by)
    eps = spec.rms_norm_eps

    short_prompt = [3, 7, 1]                     # length 3 <= sliding_window 4
    long_prompt = [3, 7, 1, 12, 5, 9, 2]         # length 7 > sliding_window 4

    for prompt in (short_prompt, long_prompt)
        cap = _Gemma4Capture(Vector{Float32}[])
        ids = Gesso.reference_generate(
            model, tensors, prompt;
            max_new_tokens=4,
            sampler=cap,
            eos_ids=(),
            eps=eps,
        )
        @test length(cap.rows) == 4
        @test length(ids) == length(prompt) + 4

        # Verify step-by-step decode logits match reference_prefill on extended prefix to atol=2e-5
        for k in 1:4
            prefix = ids[1:(length(prompt) + k - 1)]
            ref_logits = Gesso.reference_prefill(model, tensors, prefix; eps=eps)[:, end]
            @test isapprox(cap.rows[k], ref_logits; atol=2e-5, rtol=2e-5)
            @test ids[length(prompt) + k] == argmax(ref_logits) - 1
        end
    end

    # Concurrency check: Two concurrent tasks must not race on module-global scratch
    serial_short = Gesso.reference_generate(model, tensors, short_prompt; max_new_tokens=6, eos_ids=(), eps=eps)
    serial_long = Gesso.reference_generate(model, tensors, long_prompt; max_new_tokens=6, eos_ids=(), eps=eps)

    task_short = Threads.@spawn Gesso.reference_generate(model, tensors, short_prompt; max_new_tokens=6, eos_ids=(), eps=eps)
    task_long = Threads.@spawn Gesso.reference_generate(model, tensors, long_prompt; max_new_tokens=6, eos_ids=(), eps=eps)

    conc_short = fetch(task_short)
    conc_long = fetch(task_long)

    @test conc_short == serial_short
    @test conc_long == serial_long

    # Negative Control A: Disable sliding-window mask (_GEMMA4_WINDOW_MASK_ENABLED)
    baseline_short = Gesso.reference_prefill(model, tensors, short_prompt; eps=eps)[:, end]
    baseline_long = Gesso.reference_prefill(model, tensors, long_prompt; eps=eps)[:, end]

    _gemma4_inf._GEMMA4_WINDOW_MASK_ENABLED[] = false
    try
        nomask_short = Gesso.reference_prefill(model, tensors, short_prompt; eps=eps)[:, end]
        nomask_long = Gesso.reference_prefill(model, tensors, long_prompt; eps=eps)[:, end]

        # Short prompt <= window must remain bit-identical
        @test maximum(abs.(nomask_short .- baseline_short)) == 0.0f0

        # Long prompt > window must diverge
        @test maximum(abs.(nomask_long .- baseline_long)) > 1e-3
    finally
        _gemma4_inf._GEMMA4_WINDOW_MASK_ENABLED[] = true
    end

    # Negative Control B: Disable final logit softcap (_GEMMA4_SOFTCAP_ENABLED)
    _gemma4_inf._GEMMA4_SOFTCAP_ENABLED[] = false
    try
        nosoftcap_long = Gesso.reference_prefill(model, tensors, long_prompt; eps=eps)[:, end]
        @test maximum(abs.(nosoftcap_long .- baseline_long)) > 1e-2
    finally
        _gemma4_inf._GEMMA4_SOFTCAP_ENABLED[] = true
    end

    # Negative Control C: Optional layer_output_scale (omitted vs 1.0 vs 0.5)
    by_noscale = Dict(k => v for (k, v) in by if !endswith(k, ".layer_output_scale.weight"))
    spec_noscale = _gemma4_inf.spec_from_gguf_kv(_gemma4_toy_kv(); tensor_names=Set(keys(by_noscale)))
    @test !(:layer_output_scale in spec_noscale.features)
    model_noscale, tensors_noscale = _gemma4_bind(spec_noscale, by_noscale)
    logits_noscale = Gesso.reference_prefill(model_noscale, tensors_noscale, short_prompt; eps=eps)[:, end]
    @test logits_noscale == baseline_short

    by_halfscale = copy(by)
    by_halfscale["blk.0.layer_output_scale.weight"] = Float32[0.5]
    model_halfscale, tensors_halfscale = _gemma4_bind(spec, by_halfscale)
    logits_halfscale = Gesso.reference_prefill(model_halfscale, tensors_halfscale, short_prompt; eps=eps)[:, end]
    @test maximum(abs.(logits_halfscale .- baseline_short)) > 1e-2
end

@testset "Gemma4: q8 KV cache decodes on CPU" begin
    spec = _gemma4_inf.spec_from_gguf_kv(_gemma4_toy_kv_wide())
    model, tensors = _gemma4_bind(spec, _gemma4_toy(spec))
    gen(fmt) = withenv("GESSO_KV_FORMAT" => fmt) do
        rows = Matrix{Float32}[]
        ids = Gesso.reference_generate(model, tensors, [3, 7, 1, 12, 5, 9, 2];
            max_new_tokens=6, eos_ids=(), eps=spec.rms_norm_eps,
            on_rows=(pos, M) -> push!(rows, Float32.(Array(M))))
        return ids, rows
    end
    ids_f, rows_f = gen("f32")
    ids_q, rows_q = gen("q8")
    @test ids_q == ids_f
    # The rows move, so the cache really is quantized.
    @test any(rows_q[i] != rows_f[i] for i in eachindex(rows_f, rows_q))
end

# -----------------------------------------------------------------------------
# Testset 4: Local GGUF metadata and geometry (env-gated)
# -----------------------------------------------------------------------------
@testset "Gemma4: local GGUF metadata (env-gated)" begin
    path = get(ENV, "GESSO_GEMMA4_GGUF", "")
    if isempty(path)
        @test_skip !isempty(path)
    else
        hdr = _gemma4_inf.gguf_header(path)
        spec = _gemma4_inf.spec_from_gguf_kv(hdr.kv; tensor_names=Set(t.name for t in hdr.tensors))
        @test spec.family === :gemma4
        @test spec.num_layers == 30
        @test spec.hidden_size == 2816
        @test spec.moe !== nothing
        @test spec.moe.n_experts == 128
        @test spec.moe.n_expert_used == 8
        @test spec.sliding_window == 1024
        @test count(identity, spec.swa_layers) == 25
        @test count(!, spec.swa_layers) == 5
    end
end
