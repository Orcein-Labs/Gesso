# Shared LFM2 toy fixture: the metadata header (dense, or MoE with a leading
# dense block and sigmoid routing) and random weights for every tensor the
# param map binds. Included by test_lfm25.jl and test_lava_lfm25.jl.

const _lfm_inf = Gesso.Inference

# `width` is the embedding length; at 128 the four heads are 32 wide, which a
# q8 KV cache needs.
function _lfm_toy_kv(moe=false; width=32)
    arch = moe ? "lfm2moe" : "lfm2"
    d = Dict{String,Any}("general.architecture" => arch)
    for (k, v) in ("block_count" => 4, "embedding_length" => width,
        "attention.head_count" => 4, "attention.head_count_kv" => [0, 2, 0, 2],
        "attention.layer_norm_rms_epsilon" => 1f-5, "feed_forward_length" => 48,
        "shortconv.l_cache" => 3, "vocab_size" => 40, "rope.freq_base" => 1f6)
        d["$arch.$k"] = v
    end
    if moe
        for (k, v) in ("expert_count" => 4, "expert_used_count" => 2,
            "expert_feed_forward_length" => 24, "leading_dense_block_count" => 1,
            "expert_gating_func" => 2)
            d["$arch.$k"] = v
        end
    end
    return d
end

function _lfm_toy(spec)
    rng = MersenneTwister(20261027)
    by = Dict{String,Any}("token_embd.weight" => randn(rng, Float32, spec.vocab_size, spec.hidden_size),
        "token_embd_norm.weight" => ones(Float32, spec.hidden_size))
    map = _lfm_inf._gguf_lfm_param_map(spec)
    for l in 0:(spec.num_layers-1), ref in _lfm_inf._layer_param_refs(map, spec, l)
        shape = _lfm_inf.expected_shape(spec, ref.id.role)
        by[_lfm_inf.external_name(map, ref, l)] = length(shape) == 1 ?
            (ref.id.role === _lfm_inf.ROLE_ROUTER_BIAS ? 0.1f0 .* randn(rng, Float32, shape...) :
                1f0 .+ 0.1f0 .* randn(rng, Float32, shape...)) :
            randn(rng, Float32, shape...) ./ sqrt(Float32(shape[end]))
    end
    return by
end

function _lfm_bind(spec, by)
    b = _lfm_inf.materialize_architecture(spec, _lfm_inf._gguf_lfm_param_map(spec), by)
    return _lfm_inf.spec_to_model(spec), (embedding=b.embedding, blocks=b.blocks,
        lm_head=b.lm_head, final_rms=b.final_rms, rope=b.rope, spec=spec)
end

