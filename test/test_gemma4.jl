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

# -----------------------------------------------------------------------------
# Testset 5: batched decode experts over quantized slabs
# -----------------------------------------------------------------------------
# Random quant bytes for an expert bank, with a small finite fp16 scale at the
# head of every block so the weights stay O(0.1).
function _g4_quant_bank(rng, t::Int32, n_exp::Int, ne0::Int, ne1::Int)
    blk_vals, blk_bytes = Dict(_gemma4_inf.GGML_TYPE_Q8_0 => (32, 34),
                               _gemma4_inf.GGML_TYPE_IQ4_NL => (32, 18),
                               _gemma4_inf.GGML_TYPE_IQ2_XS => (256, 74),
                               _gemma4_inf.GGML_TYPE_IQ3_XXS => (256, 98))[t]
    nblk = n_exp * ne1 * div(ne0, blk_vals)
    bpe = div(nblk, n_exp) * blk_bytes
    data = rand(rng, UInt8, nblk * blk_bytes)
    for b in 0:(nblk-1)
        data[(b*blk_bytes+1):(b*blk_bytes+2)] .= reinterpret(UInt8, [Float16(0.002 + 0.01 * rand(rng))])
    end
    return _gemma4_inf.GgufQuantSlice(data, 0, t, n_exp, ne0, ne1, bpe, Int64(blk_vals))
end

function _g4_quant_experts(bt, rng, gu_t, dn_t, E, ff)
    n_exp = size(bt.wexp_gate_up.storage, 1)
    gu = _g4_quant_bank(rng, gu_t, n_exp, E, 2ff)
    dn = _g4_quant_bank(rng, dn_t, n_exp, ff, E)
    return merge(bt, (; wexp_gate_up=Gesso.ExpertWeight((n_exp, 2ff, E), gu),
                        wexp_down=Gesso.ExpertWeight((n_exp, E, ff), dn)))
end

function _g4_moe_out(bt, X; batched::Bool)
    prev = _gemma4_inf._GEMMA4_BATCHED_EXPERTS[]
    _gemma4_inf._GEMMA4_BATCHED_EXPERTS[] = batched
    try
        s = _gemma4_inf._Gemma4Scratch()
        y = copy(_gemma4_inf._gemma4_parallel_moe_ffn!(Gesso.CPUBackend(), X, bt, 1e-6; scratch=s))
        return y, s.experts
    finally
        _gemma4_inf._GEMMA4_BATCHED_EXPERTS[] = prev
    end
end

@testset "Gemma4: batched quantized experts match the per-expert path bit for bit" begin
    # Q8_K input (IQ3_XXS gate_up needs E a multiple of 256) and Q8_0 input
    # (Q8_0 / IQ4_NL) both run; the 26B ships IQ2_XS/IQ3_XXS gate_up, IQ4_NL down.
    for (E, gu_t, dn_t) in ((256, _gemma4_inf.GGML_TYPE_IQ3_XXS, _gemma4_inf.GGML_TYPE_IQ4_NL),
                            (256, _gemma4_inf.GGML_TYPE_IQ2_XS, _gemma4_inf.GGML_TYPE_IQ4_NL),
                            (32, _gemma4_inf.GGML_TYPE_Q8_0, _gemma4_inf.GGML_TYPE_Q8_0))
        kv = _gemma4_toy_kv()
        kv["gemma4.embedding_length"] = E
        spec = _gemma4_inf.spec_from_gguf_kv(kv)
        _, tensors = _gemma4_bind(spec, _gemma4_toy(spec))
        rng = MersenneTwister(20261028 + E)
        bt = _g4_quant_experts(tensors.blocks[1], rng, gu_t, dn_t, E, 32)
        X = randn(MersenneTwister(7), Float32, 3, E)

        y_ref, s_ref = _g4_moe_out(bt, X; batched=false)
        y_bat, s_bat = _g4_moe_out(bt, X; batched=true)
        @test s_ref === nothing
        @test s_bat isa _gemma4_inf._G4MoeScratch
        @test all(isfinite, y_ref) && any(!iszero, y_ref)
        @test y_bat == y_ref

        # Prefill groups tokens by expert, in batches of _G4_PREFILL_TOKENS; the
        # decode path runs the same routing one token at a time. Crossing a
        # batch boundary must not change a bit. (The whole FFN is not compared:
        # its dense matmuls sum in a different order at N=1.)
        gu, dn = bt.wexp_gate_up.storage, bt.wexp_down.storage
        n_exp, n_used, ff = gu.n_exp, 2, dn.ne0
        N = _gemma4_inf._G4_PREFILL_TOKENS + 5
        r = MersenneTwister(8)
        x2 = randn(r, Float32, N, E)
        experts = reduce(hcat, (randperm(r, n_exp)[1:n_used] for _ in 1:N))
        weights = rand(r, Float32, n_used, N)
        y_grp = _gemma4_inf._g4_moe_prefill!(zeros(Float32, N, E), x2, experts, weights, gu, dn)
        y_tok = zeros(Float32, N, E)
        ms = _gemma4_inf._G4MoeScratch(E, ff, n_used, gu.ggml_type, dn.ggml_type)
        probs = zeros(Float32, n_exp)
        for t in 1:N
            probs[experts[:, t]] .= weights[:, t]
            _gemma4_inf._g4_moe_token_batched!(y_tok, t, x2, experts[:, t], probs, 1.0f0,
                                               ones(Float32, n_exp), n_used, gu, dn, ms)
        end
        @test any(!iszero, y_grp)
        @test y_grp == y_tok

        # Negative control: the comparison sees the expert bytes. Changing one
        # block scale of the down bank in every expert must move the output.
        dn = bt.wexp_down.storage
        data = copy(dn.data)
        for e in 1:dn.n_exp
            o = (e - 1) * dn.bytes_per_expert
            data[(o+1):(o+2)] .= reinterpret(UInt8, [Float16(0.5)])
        end
        dn2 = _gemma4_inf.GgufQuantSlice(data, 0, dn.ggml_type, dn.n_exp, dn.ne0, dn.ne1,
                                         dn.bytes_per_expert, dn.blck_size)
        bt2 = merge(bt, (; wexp_down=Gesso.ExpertWeight(bt.wexp_down.shape, dn2)))
        y_neg, _ = _g4_moe_out(bt2, X; batched=true)
        @test y_neg != y_ref
    end
end

# -----------------------------------------------------------------------------
# Testset 6: row-parallel prefill helpers
# -----------------------------------------------------------------------------
@testset "Gemma4: row-chunked norms, RoPE and router match one row at a time" begin
    # 64 rows split into chunks whenever the suite runs with more than one
    # thread; each row must come out exactly as it does on its own.
    N, nh, hd, n_exp, n_used = 64, 4, 16, 8, 2
    rng = MersenneTwister(29)
    X = randn(rng, Float32, N, nh * hd)
    w = 1.0f0 .+ 0.1f0 .* randn(rng, Float32, nh * hd)
    wh = 1.0f0 .+ 0.1f0 .* randn(rng, Float32, hd)
    inv_freq = [1.0e4^(-2(i - 1) / hd) for i in 1:(hd ÷ 2)]
    positions = collect(0:(N - 1)) .+ 100
    eps = 1.0f-6
    by_row(f) = reduce(vcat, (f(t) for t in 1:N))

    Y = _gemma4_inf._g4_rmsnorm_rows!(similar(X), X, w, eps)
    @test Y == by_row(t -> _gemma4_inf._g4_rmsnorm_rows!(zeros(Float32, 1, nh * hd), X[t:t, :], w, eps))

    H = _gemma4_inf._g4_head_norm!(copy(X), nh, hd, wh, eps)
    @test H == by_row(t -> _gemma4_inf._g4_head_norm!(X[t:t, :], nh, hd, wh, eps))
    @test H != Y

    R = _gemma4_inf._apply_rope_neox!(Gesso.CPUBackend(), copy(X), positions, nh, hd, hd, inv_freq)
    @test R == by_row(t -> _gemma4_inf._apply_rope_neox!(Gesso.CPUBackend(), X[t:t, :], positions[t:t], nh, hd, hd, inv_freq))
    @test R != X

    r = randn(rng, Float32, n_exp, nh * hd)
    scale = 0.5f0 .+ rand(rng, Float32, n_exp)
    experts = zeros(Int, n_used, N)
    weights = zeros(Float32, n_used, N)
    _gemma4_inf._g4_route_rows!(experts, weights, X, r, n_used, scale)
    for t in 1:N
        e1 = zeros(Int, n_used, 1)
        w1 = zeros(Float32, n_used, 1)
        _gemma4_inf._g4_route_rows!(e1, w1, X[t:t, :], r, n_used, scale)
        @test experts[:, t] == e1[:, 1]
        @test weights[:, t] == w1[:, 1]
    end
    @test length(unique(experts)) > n_used
end
