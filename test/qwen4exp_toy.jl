# Shared qwen4exp toy fixture: the metadata header and a random model with every
# tensor the forward reads (GDN and attention layers, PLE on layer 1, MoE with a
# shared expert). Included by test_qwen4exp.jl and test_lava_qwen4exp.jl.

function _q4x_toy_kv()
    voc = [7, 11, 13, 17]
    return Dict{String, Any}(
        "general.architecture" => "qwen4exp",
        "tokenizer.ggml.tokens" => 40,
        "qwen4exp.block_count" => 4,
        "qwen4exp.embedding_length" => 16,
        "qwen4exp.hyper_connection.count" => 2,
        "qwen4exp.attention.head_count" => 4,
        "qwen4exp.attention.head_count_kv" => 2,
        "qwen4exp.attention.key_length" => 8,
        "qwen4exp.attention.value_length" => 8,
        "qwen4exp.rope.dimension_count" => 4,
        "qwen4exp.rope.dimension_sections" => [1, 1, 0, 0],
        "qwen4exp.rope.freq_base" => 10000.0,
        "qwen4exp.attention.layer_norm_rms_epsilon" => 1e-6,
        "qwen4exp.full_attention_interval" => 4,
        "qwen4exp.attention.compress_ratios" => [0, 0, 0, 4],
        "qwen4exp.attention.indexer.top_k" => 16,
        "qwen4exp.ssm.conv_kernel" => 4,
        "qwen4exp.ssm.group_count" => 2,
        "qwen4exp.ssm.state_size" => 4,
        "qwen4exp.ssm.time_step_rank" => 4,
        "qwen4exp.ssm.inner_size" => 16,
        "qwen4exp.expert_count" => 4,
        "qwen4exp.expert_used_count" => 2,
        "qwen4exp.expert_feed_forward_length" => 6,
        "qwen4exp.expert_shared_feed_forward_length" => 6,
        "qwen4exp.ple.layers" => [1],
        "qwen4exp.ple.ngram_size" => 3,
        "qwen4exp.ple.heads_per_ngram" => 2,
        "qwen4exp.ple.layer_multipliers" => UInt64[23703573157769, 20109073645365, 8052911324071],
        "qwen4exp.ple.head_offsets" => cumsum([0; voc[1:end-1]]),
        "qwen4exp.ple.head_vocab_sizes" => voc,
        "qwen4exp.ple.eos_token_id" => 39,
        "qwen4exp.ple.conv_kernel" => 4,
    )
end

function _q4x_toy_model(cfg; seed=7)
    rng = MersenneTwister(seed)
    NE, HC, R = cfg.n_embd, cfg.n_hc, 3
    W(o, i) = 0.3f0 .* randn(rng, Float32, o, i)
    N(n) = 1f0 .+ 0.1f0 .* randn(rng, Float32, n)
    t = Dict{String, Any}()
    t["token_embd.weight"] = W(cfg.vocab, NE)
    t["output.weight"] = W(cfg.vocab, NE)
    nh = (cfg.ple_ngram - 1) * cfg.ple_hpn
    t["per_layer_token_embd.weight"] = W(Int(sum(cfg.ple_voc)), NE ÷ nh)
    hc(pre; inject=true) = begin
        t[pre*"_norm.weight"] = N(HC * NE)
        t[pre*"_down.weight"] = W(R, HC * NE)
        t[pre*"_up.weight"] = W(HC * NE, R)
        inject && (t[pre*"_inject.weight"] = W(HC, HC * NE))
    end
    hc("output_hc"; inject=false)
    nq, nvd = cfg.ssm_nk * cfg.ssm_dk, cfg.ssm_nv * cfg.ssm_dv
    NH, NKV, HD = cfg.n_head, cfg.n_head_kv, cfg.head_dim
    ff = 6
    for il in 0:(cfg.n_layer-1)
        p = "blk.$il."
        hc(p * "hc_attn")
        hc(p * "hc_ffn")
        if il in cfg.ple_layers
            t[p*"ple_key.weight"] = W(HC * NE, NE)
            t[p*"ple_value.weight"] = W(NE, NE)
            t[p*"ple_norm_key.weight"] = N(NE)
            t[p*"ple_norm_query.weight"] = N(NE)
            t[p*"ple_norm_conv.weight"] = N(NE)
            t[p*"ple_conv1d.weight"] = vec(W(cfg.ple_kernel, HC * NE))
        end
        if cfg.attn_layer[il+1]
            t[p*"attn_q.weight"] = W(2NH * HD, NE)
            t[p*"attn_q_norm.weight"] = N(HD)
            t[p*"attn_k.weight"] = W(NKV * HD, NE)
            t[p*"attn_k_norm.weight"] = N(HD)
            t[p*"attn_v.weight"] = W(NKV * HD, NE)
            t[p*"attn_output.weight"] = W(NE, NH * HD)
        else
            t[p*"attn_qkv.weight"] = W(2nq + nvd, NE)
            t[p*"attn_gate.weight"] = W(nvd, NE)
            t[p*"ssm_beta.weight"] = W(cfg.ssm_nv, NE)
            t[p*"ssm_alpha.weight"] = W(cfg.ssm_nv, NE)
            t[p*"ssm_dt.bias"] = 0.1f0 .* randn(rng, Float32, cfg.ssm_nv)
            t[p*"ssm_a"] = -abs.(randn(rng, Float32, cfg.ssm_nv))
            t[p*"ssm_conv1d.weight"] = vec(W(cfg.conv_kernel, 2nq + nvd))
            t[p*"ssm_norm.weight"] = N(cfg.ssm_dv)
            t[p*"ssm_out.weight"] = W(NE, nvd)
        end
        t[p*"ffn_gate_inp.weight"] = W(cfg.n_exp, NE)
        t[p*"ffn_gate_exps.weight"] = 0.3f0 .* randn(rng, Float32, cfg.n_exp, ff, NE)
        t[p*"ffn_up_exps.weight"] = 0.3f0 .* randn(rng, Float32, cfg.n_exp, ff, NE)
        t[p*"ffn_down_exps.weight"] = 0.3f0 .* randn(rng, Float32, cfg.n_exp, NE, ff)
        t[p*"ffn_gate_shexp.weight"] = W(ff, NE)
        t[p*"ffn_up_shexp.weight"] = W(ff, NE)
        t[p*"ffn_down_shexp.weight"] = W(NE, ff)
        t[p*"ffn_gate_inp_shexp.weight"] = 0.3f0 .* randn(rng, Float32, NE)
    end
    return Qwen4expModel(cfg, t)
end
