# qwen4exp (Flash-Next) forward: the metadata reader on a synthetic header, a
# small random model fed in chunks against one call (GDN conv and delta state,
# KV cache, PLE token and conv history all carry across), the QSA context
# refusal, and (gated on GESSO_QWEN4EXP_GGUF, the first shard) a 4-layer run
# against sums and corners from the llama.cpp eval-callback dump, and with
# GESSO_QWEN4EXP_FULL=1 all 48 layers against the CPU llama.cpp top tokens.

using Test
using Gesso
using Random

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

@testset "qwen4exp: config and spec from metadata" begin
    kv = _q4x_toy_kv()
    cfg = qwen4exp_config(kv)
    @test cfg.attn_layer == [false, false, false, true]
    @test cfg.ssm_dv == 4
    @test cfg.ple_layers == [1]
    spec = Gesso.Inference.spec_from_gguf_kv(kv; tensor_names=["output.weight"])
    @test spec.family === :qwen4exp
    @test :hyper_connections in Gesso.Inference.required_semantics(spec)
    @test :ple in Gesso.Inference.required_semantics(spec)
    bad = copy(kv)
    bad["qwen4exp.rope.dimension_sections"] = [1, 0, 0, 0]
    @test_throws ErrorException qwen4exp_config(bad)
end

@testset "qwen4exp: chunked prefill equals one call" begin
    cfg = qwen4exp_config(_q4x_toy_kv())
    m = _q4x_toy_model(cfg)
    toks = [3, 17, 5, 39, 22, 8, 8, 30, 1, 12, 26]   # 39 is the PLE EOS
    st1 = Qwen4expState(cfg)
    full = qwen4exp_forward!(m, st1, toks)
    l1 = qwen4exp_logits(m, full)
    st2 = Qwen4expState(cfg)
    rows = Matrix{Float32}[]
    local last
    for r in (1:3, 4:4, 5:9, 10:11)
        last = qwen4exp_forward!(m, st2, toks[r])
        push!(rows, last)
    end
    @test st2.pos == length(toks)
    @test isapprox(vcat(rows...), full; rtol=1e-4)
    @test isapprox(qwen4exp_logits(m, last), l1; rtol=1e-4)
    # The state matters: the same tail without it gives different rows.
    cold = qwen4exp_forward!(m, Qwen4expState(cfg), toks[10:11])
    @test maximum(abs.(cold .- full[10:11, :])) > 1e-3
end

@testset "qwen4exp: QSA layers refuse contexts past top_k" begin
    cfg = qwen4exp_config(_q4x_toy_kv())
    m = _q4x_toy_model(cfg)
    st = Qwen4expState(cfg)
    qwen4exp_forward!(m, st, collect(1:16))
    @test_throws ErrorException qwen4exp_forward!(m, st, [2])
    # Truncated before the QSA layer, no refusal.
    @test size(qwen4exp_forward!(m, Qwen4expState(cfg), collect(0:20); n_layers=3), 1) == 21
end

# Printed corners of the llama.cpp eval-callback dump of this GGUF (12 tokens):
# name => (ggml shape, sum, corner values).
function _q4x_oracle()
    path = joinpath(@__DIR__, "fixtures", "qwen4exp", "oracle-corners.txt")
    lines = filter(l -> !startswith(l, "#"), readlines(path))
    out = Dict{String, Tuple{NTuple{4, Int}, Float64, Vector{Float64}}}()
    for i in 1:2:length(lines)
        name, shp, s = split(lines[i])
        ne = Tuple(parse.(Int, split(shp, ",")))
        out[name] = (ne, parse(Float64, s), parse.(Float64, split(lines[i+1])))
    end
    return out
end

_q4x_pick(n) = n > 6 ? [1, 2, 3, n - 2, n - 1, n] : collect(1:n)

# The corners llama.cpp prints, from A in ggml order A[i0, i1, i2, i3].
function _q4x_corners(A)
    return [A[i0, i1, i2, i3] for i3 in _q4x_pick(size(A, 4)) for i2 in _q4x_pick(size(A, 3))
            for i1 in _q4x_pick(size(A, 2)) for i0 in _q4x_pick(size(A, 1))]
end

# Corner error as a fraction of the node's largest printed value.
_q4x_cerr(c, vals) = maximum(abs.(c .- vals)) / maximum(abs.(vals))

@testset "qwen4exp: real GGUF against the llama.cpp dump" begin
    path = get(ENV, "GESSO_QWEN4EXP_GGUF", "")
    if isempty(path)
        @test_skip "GESSO_QWEN4EXP_GGUF not set"
    else
        m = load_qwen4exp(path)
        @test m.cfg.n_layer == 48
        @test m.cfg.attn_layer[4] && !m.cfg.attn_layer[3]
        toks = [760, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 6124, 369]
        taps = Dict{String, Any}()
        res = qwen4exp_forward!(m, Qwen4expState(m.cfg), toks; n_layers=4, taps)
        oracle = _q4x_oracle()
        ggml(name) = reshape(permutedims(Float64.(taps[name])), oracle[name][1]...)
        for (name, (_, osum, vals)) in oracle
            A = ggml(name)
            @test abs(sum(A) - osum) <= 0.01 * abs(osum)
            @test _q4x_cerr(_q4x_corners(A), vals) < 0.05
        end
        # The corner bound tells layers apart: layer 0 fails against layer 3.
        @test _q4x_cerr(_q4x_corners(ggml("l_last-0")), oracle["l_last-3"][3]) > 0.05
        st = Qwen4expState(m.cfg)
        a = qwen4exp_forward!(m, st, toks[1:5]; n_layers=4)
        b = qwen4exp_forward!(m, st, toks[6:end]; n_layers=4)
        @test isapprox(vcat(a, b), res; rtol=1e-3)
        if get(ENV, "GESSO_QWEN4EXP_FULL", "") == "1"
            # All 48 layers: llama.cpp (CPU) puts " Tokyo" first and these ten on top.
            lg = qwen4exp_logits(m, qwen4exp_forward!(m, Qwen4expState(m.cfg), toks))
            @test argmax(lg) - 1 == 25358
            top = Set(partialsortperm(lg, 1:10; rev=true) .- 1)
            ref = Set([25358, 271, 524, 198, 1092, 239784, 13, 78229, 1076, 30])
            @test length(intersect(top, ref)) >= 8
        end
    end
end
