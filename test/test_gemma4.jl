using Test, Random, LinearAlgebra
using Gesso

const _gemma4_inf = Gesso.Inference

function _gemma4_toy_kv()
    return Dict{String, Any}(
        "general.architecture" => "gemma4",
        "gemma4.block_count" => 2,
        "gemma4.embedding_length" => 32,
        "gemma4.attention.head_count" => 4,
        "gemma4.attention.head_count_kv" => [2, 1],
        "gemma4.attention.sliding_window_pattern" => [true, false],
        "gemma4.attention.sliding_window" => 4,
        "gemma4.attention.key_length" => 16,
        "gemma4.attention.key_length_swa" => 8,
        "gemma4.attention.value_length" => 16,
        "gemma4.attention.value_length_swa" => 8,
        "gemma4.feed_forward_length" => 24,
        "gemma4.expert_feed_forward_length" => 12,
        "gemma4.expert_count" => 4,
        "gemma4.expert_used_count" => 2,
        "gemma4.rope.dimension_count" => 16,
        "gemma4.rope.dimension_count_swa" => 8,
        "gemma4.rope.freq_base" => 1e6,
        "gemma4.rope.freq_base_swa" => 1e4,
        "gemma4.final_logit_softcapping" => 30.0f0,
        "gemma4.attention.layer_norm_rms_epsilon" => 1.0f-6,
        "gemma4.vocab_size" => 40,
    )
end

function _gemma4_toy(spec)
    rng = MersenneTwister(20261028)
    hidden = spec.hidden_size
    vocab = spec.vocab_size
    by = Dict{String, Any}()
    by["token_embd.weight"] = randn(rng, Float32, vocab, hidden)
    by["output_norm.weight"] = ones(Float32, hidden) .+ 0.05f0 .* randn(rng, Float32, hidden)
    if :rope_freq_factors in spec.features
        by["rope_freqs.weight"] = ones(Float32, spec.head_dim ÷ 2)
    end

    map = _gemma4_inf._gguf_gemma4_param_map(spec)
    for l in 0:(spec.num_layers - 1)
        for ref in _gemma4_inf._layer_param_refs(map, spec, l)
            shape = _gemma4_inf.expected_shape(spec, ref.id.role, l)
            name = _gemma4_inf.external_name(map, ref, l)
            by[name] = if length(shape) == 1
                (ref.id.role === _gemma4_inf.ROLE_LAYER_OUTPUT_SCALE ? Float32[1.0] :
                 ref.id.role === _gemma4_inf.ROLE_EXPERT_DOWN_SCALE ? ones(Float32, shape...) :
                 ones(Float32, shape...) .+ 0.05f0 .* randn(rng, Float32, shape...))
            elseif length(shape) == 2
                randn(rng, Float32, shape...) ./ sqrt(Float32(shape[end]))
            elseif length(shape) == 3
                randn(rng, Float32, shape...) ./ sqrt(Float32(shape[end]))
            else
                error("unexpected shape $shape for role $(ref.id.role)")
            end
        end
    end
    return by
end

function _gemma4_bind(spec, by)
    b = _gemma4_inf.materialize_architecture(spec, _gemma4_inf._gguf_gemma4_param_map(spec), by)
    return _gemma4_inf.spec_to_model(spec), (
        embedding=b.embedding,
        blocks=b.blocks,
        lm_head=b.lm_head,
        final_rms=b.final_rms,
        rope=b.rope,
        spec=spec,
    )
end

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

    # Explicit refusals for unsupported Gemma 4 features
    bad = copy(kv)
    bad["gemma4.embedding_length_per_layer_input"] = 256
    @test_throws ArgumentError _gemma4_inf.spec_from_gguf_kv(bad)

    bad = copy(kv)
    bad["gemma4.attention.shared_kv_layers"] = 4
    @test_throws ArgumentError _gemma4_inf.spec_from_gguf_kv(bad)
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
    if isdefined(Gesso, :LavaBackend)
        push!(backends, Gesso.LavaBackend())
    end

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
        @test_throws Gesso.LoweringNotImplemented _gemma4_inf._gemma4_v_norm!(be, zeros(Float32, 1, 2, 8), 1e-6)
        @test_throws Gesso.LoweringNotImplemented _gemma4_inf._gemma4_k_only_v!(be, zeros(Float32, 1, 16))
        @test_throws Gesso.LoweringNotImplemented _gemma4_inf._gemma4_parallel_moe_ffn!(be, h_dummy, bt, 1e-6)
        @test_throws Gesso.LoweringNotImplemented _gemma4_inf._gemma4_layer_output_scale!(be, zeros(Float32, 1, 32), Float32[1.0])
        @test_throws Gesso.LoweringNotImplemented _gemma4_inf._gemma4_rope_with_factors!(be, zeros(Float32, 1, 32), 8, 8, 0, 1e4, nothing)
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
