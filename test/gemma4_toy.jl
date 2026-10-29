# Gemma 4 toy model shared by the CPU and Lava test files.
using Random: MersenneTwister

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

# The same toy with 32-wide heads, the narrowest a q8 KV cache accepts.
function _gemma4_toy_kv_wide()
    kv = _gemma4_toy_kv()
    for key in ("gemma4.attention.key_length", "gemma4.attention.key_length_swa",
                "gemma4.attention.value_length", "gemma4.attention.value_length_swa",
                "gemma4.rope.dimension_count", "gemma4.rope.dimension_count_swa")
        kv[key] = 32
    end
    return kv
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
