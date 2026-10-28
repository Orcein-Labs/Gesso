using Test, Random

isdefined(@__MODULE__, :_lfm_toy) || include(joinpath(@__DIR__, "lfm25_toy.jl"))

mutable struct _LFMCapture
    rows::Vector{Vector{Float32}}
end
function Gesso.Inference._sample_id(s::_LFMCapture, logits)
    row = Float32.(vec(logits))
    push!(s.rows, copy(row))
    return argmax(row) - 1
end

@testset "LFM: metadata and capability boundary" begin
    for moe in (false, true)
        kv = _lfm_toy_kv(moe)
        spec = _lfm_inf.spec_from_gguf_kv(kv)
        @test spec.family === (moe ? :lfm2moe : :lfm2)
        @test spec.mixer isa Gesso.ShortConvSpec
        @test spec.mixer.layer_types == [:short_convolution, :full_attention, :short_convolution, :full_attention]
        @test !spec.rope.interleaved
        @test :short_convolution in Gesso.required_semantics(spec)
        @test isempty(_lfm_inf._missing_capabilities(Gesso.required_semantics(spec), Gesso.CPUBackend()))
        report = Gesso.import_report(spec)
        @test occursin("short_convolution", report)
        @test occursin("Failure boundary: none", report)
        if moe
            @test :sigmoid_moe_routing in Gesso.required_semantics(spec)
            @test :moe_selection_bias in Gesso.required_semantics(spec)
            @test spec.moe.dense_lead == 1
        end
        bad = copy(kv)
        bad["$(spec.family).attention.head_count_kv"] = [0, 2]
        @test_throws ErrorException _lfm_inf.spec_from_gguf_kv(bad)
        bad = copy(kv)
        bad["$(spec.family).shortconv.l_cache"] = 1
        @test_throws ErrorException _lfm_inf.spec_from_gguf_kv(bad)
        bad = copy(kv)
        bad["$(spec.family).attention.sliding_window"] = 16
        @test_throws ErrorException _lfm_inf.spec_from_gguf_kv(bad)
        bad = copy(kv)
        bad["$(spec.family).rope.dimension_count"] = 32
        @test_throws ErrorException _lfm_inf.spec_from_gguf_kv(bad)
        @test_throws ErrorException _lfm_inf.spec_from_gguf_kv(kv; tensor_names=["dense_2_out.weight"])
        bad = copy(kv)
        bad["$(spec.family).feed_forward_length"] = fill(48, 4)
        @test_throws ErrorException _lfm_inf.spec_from_gguf_kv(bad)
        if moe
            bad = copy(kv)
            bad["$(spec.family).expert_group_count"] = 2
            @test_throws ErrorException _lfm_inf.spec_from_gguf_kv(bad)
            bad["$(spec.family).expert_group_count"] = 1
            @test _lfm_inf.spec_from_gguf_kv(bad).moe.n_experts == 4
            bad = copy(kv)
            bad["$(spec.family).expert_used_count"] = fill(2, 4)
            @test_throws ErrorException _lfm_inf.spec_from_gguf_kv(bad)
            bad = copy(kv)
            bad["$(spec.family).expert_gating_func"] = 1
            soft = _lfm_inf.spec_from_gguf_kv(bad)
            @test soft.moe.gating === :softmax
            @test :sigmoid_moe_routing ∉ Gesso.required_semantics(soft)
            @test :moe_selection_bias in Gesso.required_semantics(soft)
            bad["$(spec.family).expert_gating_func"] = 3
            @test_throws ErrorException _lfm_inf.spec_from_gguf_kv(bad)
        end
        bad = copy(kv)
        delete!(bad, "$(spec.family).vocab_size")
        bad["tokenizer.ggml.tokens"] = 40
        @test _lfm_inf.spec_from_gguf_kv(bad).vocab_size == 40
        bad["$(spec.family).vocab_size"] = 39
        @test_throws ErrorException _lfm_inf.spec_from_gguf_kv(bad)
        bad = copy(kv)
        delete!(bad, "$(spec.family).vocab_size")
        bad["tokenizer.ggml.tokens"] = ["token-$i" for i in 1:40]
        @test _lfm_inf.spec_from_gguf_kv(bad).vocab_size == 40
    end
end

@testset "LFM: convolution tap order and unbiased mixture weights" begin
    act(x) = Gesso.Activation(; shape=size(x), storage=x)
    frozen(x) = Gesso.FrozenParameter(; shape=size(x), storage=x)
    weight(x) = Gesso.ProjectionWeight(; shape=size(x), storage=x)
    expert(x) = Gesso.ExpertWeight(; shape=size(x), storage=x)
    cpu, wl = Gesso.CPUBackend(), Gesso.PrefillWorkload()
    bt = (attn_rms=frozen(ones(1)), wshort_in=weight(ones(3, 1)),
        wshort_conv=frozen(reshape([2.0, -1.0, 3.0], 1, 3)), wshort_out=weight(ones(1, 1)))
    out = act(zeros(4, 1))
    _lfm_inf._short_convolution_block!(cpu, act(ones(4, 1)), act(zeros(4, 1)), out,
        bt, wl, 1e-5, dims -> zeros(dims))
    @test vec(out.storage) ≈ [3, 2, 4, 4] ./ (1 + 1e-5)^1.5

    bt = (wrouter=weight(reshape([0.0, log(3.0)], 2, 1)),
        wrouter_bias=frozen([100.0, -100.0]), wexp_gate=expert(ones(2, 1, 1)),
        wexp_up=expert(ones(2, 1, 1)), wexp_down=expert(reshape([1.0, 3.0], 2, 1, 1)),
        n_expert_used=2, router_gating=:sigmoid)
    y = act(zeros(1, 1))
    _lfm_inf._biased_moe_ffn!(cpu, y, act(ones(1, 1)), bt)
    @test y.storage[1] ≈ 2.2 / (1 + exp(-1))
    _lfm_inf._biased_moe_ffn!(cpu, y, act(ones(1, 1)), merge(bt, (n_expert_used=1,)))
    @test y.storage[1] ≈ 1 / (1 + exp(-1))

    # Softmax over logits [0, log 3] gives [1/4, 3/4]; sigmoid would give 2.2 above.
    soft = merge(bt, (router_gating=:softmax,))
    _lfm_inf._biased_moe_ffn!(cpu, y, act(ones(1, 1)), soft)
    @test y.storage[1] ≈ 2.5 / (1 + exp(-1))
    _lfm_inf._biased_moe_ffn!(cpu, y, act(ones(1, 1)), merge(soft, (n_expert_used=1,)))
    @test y.storage[1] ≈ 1 / (1 + exp(-1))
    _lfm_inf._biased_moe_ffn!(cpu, y, act(ones(1, 1)),
        merge(soft, (n_expert_used=1, wrouter_bias=frozen([0.0, 0.0]))))
    @test y.storage[1] ≈ 3 / (1 + exp(-1))
end

# The 1.2B GGUF sets tokenizer.ggml.add_bos_token; the 8B omits it and relies on
# llama.cpp's lfm2 pre-tokenizer default, so both must come out true.
@testset "LFM: raw-text prompts honor add_bos_token ($key)" for key in ("GESSO_LFM2_GGUF", "GESSO_LFM25_GGUF")
    path = get(ENV, key, "")
    if isempty(path)
        @test_skip !isempty(path)
    else
        tk = _lfm_inf.load_gguf_tokenizer(path)
        text = "It is a truth universally acknowledged"
        ids = Gesso.encode(tk, text; parse_special=false)
        @test tk.add_bos
        @test first(ids) != tk.bos_token_id
        @test _lfm_inf._prompt_ids(tk, text) == [tk.bos_token_id; ids]
        # Same tokenizer with the flag cleared: the prompt is the bare encoding.
        fields = [getfield(tk, f) for f in fieldnames(typeof(tk))]
        fields[findfirst(==(:add_bos), fieldnames(typeof(tk)))] = false
        @test _lfm_inf._prompt_ids(typeof(tk)(fields...), text) == ids
        # The pre-add_bos positional form still builds, with BOS off.
        legacy = typeof(tk)(fields[1:(end-2)]..., tk.pre)
        @test !legacy.add_bos && legacy.pre === :lfm2
        @test _lfm_inf._prompt_ids(legacy, text) == ids
    end
end

@testset "LFM: declared chat forms" begin
    dense = "<|tool_list_start|> <|tool_response_start|> \"<|im_start|>system\\n\""
    modern = "CONTINUE_FINAL_MESSAGE_TAG render_tool_calls \"<|im_start|>system\\n\""
    messages = [Dict("role" => "system", "content" => "rules"),
        Dict("role" => "user", "content" => "question"),
        Dict("role" => "tool", "content" => "answer")]
    @test _lfm_inf._render_lfm_chat(dense, "<bos>", messages) ==
        "<bos><|im_start|>system\nrules<|im_end|>\n<|im_start|>user\nquestion<|im_end|>\n" *
        "<|im_start|>tool\n<|tool_response_start|>answer<|tool_response_end|><|im_end|>\n<|im_start|>assistant\n"
    @test _lfm_inf._render_lfm_chat(modern, "<bos>", messages) ==
        "<bos><|im_start|>system\nrules<|im_end|>\n<|im_start|>user\nquestion<|im_end|>\n" *
        "<|im_start|>tool\nanswer<|im_end|>\n<|im_start|>assistant\n"
    history = [Dict("role" => "assistant", "content" => "<think>old</think>  done  ", "thinking" => "old"),
        Dict("role" => "user", "content" => "next"),
        Dict("role" => "assistant", "content" => "final", "thinking" => "new")]
    got = _lfm_inf._render_lfm_chat(modern, "B", history; add_generation_prompt=false)
    @test !occursin("old", got)
    @test occursin("assistant\ndone<|im_end|>", got)
    @test occursin("assistant\n<think>new</think>final", got)
    @test occursin("old", _lfm_inf._render_lfm_chat(modern, "B", history; preserve_thinking=true))
    literal = replace(modern, "system\\n" => "system\\\\n")
    @test _lfm_inf._render_lfm_chat(literal, "<bos>", messages) ==
        replace(_lfm_inf._render_lfm_chat(modern, "<bos>", messages), "\n" => "\\n")
    args = _lfm_inf._chat_json_parse("{\"x\": \"O'Reilly\", \"n\": 3, \"ok\": true}")
    call = Dict("function" => Dict("name" => "f", "arguments" => args))
    msg = [Dict("role" => "assistant", "content" => "beforeCONTINUE_FINAL_MESSAGE_TAG ", "tool_calls" => [call])]
    got = _lfm_inf._render_lfm_chat(modern, "B", msg; add_generation_prompt=false)
    @test occursin("before<|tool_call_start|>[f(x='O'Reilly', n=3, ok=True)]<|tool_call_end|>CONTINUE_FINAL_MESSAGE_TAG ", got)

    real_template = read(joinpath(@__DIR__, "fixtures", "lfm_chat", "lfm25.jinja"), String)
    @test _lfm_inf.chat_family(real_template) === :lfm25
    got_real = _lfm_inf._render_lfm_chat(real_template, "<|startoftext|>", messages)
    @test got_real == "<|startoftext|><|im_start|>system\nrules<|im_end|>\n<|im_start|>user\nquestion<|im_end|>\n<|im_start|>tool\nanswer<|im_end|>\n<|im_start|>assistant\n"
end

@testset "LFM: real-map prefill and recurrent greedy decode" begin
    for moe in (false, true)
        spec = _lfm_inf.spec_from_gguf_kv(_lfm_toy_kv(moe))
        by = _lfm_toy(spec)
        model, tensors = _lfm_bind(spec, by)
        prompt = [3, 7, 1, 12, 5]
        cap = _LFMCapture(Vector{Float32}[])
        ids = Gesso.reference_generate(model, tensors, prompt; max_new_tokens=6,
            sampler=cap, eos_ids=(), eps=spec.rms_norm_eps)
        @test length(cap.rows) == 6
        @test length(ids) == length(prompt) + 6
        for k in 1:6
            ref = Gesso.reference_prefill(model, tensors, ids[1:(length(prompt)+k-1)]; eps=spec.rms_norm_eps)[:, end]
            @test isapprox(cap.rows[k], ref; atol=2e-5, rtol=2e-5)
            @test ids[length(prompt)+k] == argmax(ref) - 1
        end
        ref = Gesso.reference_prefill(model, tensors, prompt; eps=spec.rms_norm_eps)
        reversed = deepcopy(by)
        for (name, weight) in reversed
            endswith(name, "shortconv.conv.weight") && (reversed[name] = weight[:, end:-1:1])
        end
        m, t = _lfm_bind(spec, reversed)
        @test maximum(abs, Gesso.reference_prefill(m, t, prompt; eps=spec.rms_norm_eps) .- ref) > 1e-3
        if moe
            permuted = deepcopy(by)
            for (name, weight) in permuted
                endswith(name, "ffn_gate_inp.weight") && (permuted[name] = weight[end:-1:1, :])
            end
            m, t = _lfm_bind(spec, permuted)
            @test maximum(abs, Gesso.reference_prefill(m, t, prompt; eps=spec.rms_norm_eps) .- ref) > 1e-3
            @test haskey(tensors.blocks[1], :wgate) && !haskey(tensors.blocks[1], :wrouter)
            @test haskey(tensors.blocks[2], :wrouter) && !haskey(tensors.blocks[2], :wgate)
        end

        extended = vcat(prompt, 9)
        want = Gesso.reference_prefill(model, tensors, extended; eps=spec.rms_norm_eps)[:, end]
        lagged = Gesso.GenerateState(32)
        lagged_conv = nothing
        try
            Gesso.reference_generate(model, tensors, prompt[1:end-1]; state=lagged,
                max_new_tokens=0, eps=spec.rms_norm_eps)
            lagged_conv = deepcopy(lagged.gdn_conv)
        finally
            close(lagged)
        end
        for corrupt in (false, true)
            state = Gesso.GenerateState(32)
            try
                Gesso.reference_generate(model, tensors, prompt; state, max_new_tokens=0, eps=spec.rms_norm_eps)
                if corrupt
                    for i in eachindex(state.gdn_conv)
                        hist = state.gdn_conv[i]
                        hist === nothing || copyto!(hist, lagged_conv[i])
                    end
                end
                got = _LFMCapture(Vector{Float32}[])
                Gesso.reference_generate(model, tensors, extended; state, sampler=got,
                    max_new_tokens=1, eos_ids=(), eps=spec.rms_norm_eps)
                delta = maximum(abs, only(got.rows) .- want)
                @test corrupt ? delta > 1e-3 : delta < 2e-4
            finally
                close(state)
            end
        end
    end
end

@testset "LFM: local GGUF metadata and tokenizer" begin
    for key in ("GESSO_LFM2_GGUF", "GESSO_LFM25_GGUF")
        path = get(ENV, key, "")
        @testset "$key" begin
            if isempty(path)
                @test_skip !isempty(path)
            else
                hdr = _lfm_inf.gguf_header(path)
                spec = _lfm_inf.spec_from_gguf_kv(hdr.kv; tensor_names=Set(t.name for t in hdr.tensors))
                @test spec.family in (:lfm2, :lfm2moe)
                tk = _lfm_inf.load_gguf_tokenizer(path)
                text = "Hello, we're testing 1234567. Καλημέρα!\n日本語 e\u0301\t"
                ids = Gesso.encode(tk, text; parse_special=false)
                @test Gesso.decode(tk, ids) == text
                @test tk.chat_template !== nothing
            end
        end
    end
end
