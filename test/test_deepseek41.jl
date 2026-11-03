# deepseek41 (DeepSeek-V4.1-Flash) text forward: metadata reader and spec on a
# synthetic header, bad metadata refused, and (later) the forward checks on a
# small random model. The real-model test is gated on GESSO_DSV41_GGUF.

using Test
using Gesso
using Random

include("deepseek41_toy.jl")

@testset "deepseek41: config and spec from metadata" begin
    kv = _ds41_toy_kv()
    cfg = Gesso.Inference.deepseek41_config(kv)
    @test cfg isa Gesso.Inference.Deepseek41Config
    @test cfg.n_layer == 6
    @test cfg.compress == [0, 0, 1, 1, 2, 2]
    @test cfg.kv_src == [-1, -1, 2, 2, 4, 4]
    @test cfg.idx_src == [-1, -1, 2, 2, 4, 4]
    @test cfg.cand_src == 4
    @test cfg.ple_layers == [1, 4]
    @test cfg.hc == 2
    @test cfg.n_exp == 4 && cfg.n_used == 2 && cfg.n_shared == 1
    @test cfg.vocab == 40
    @test cfg.rope_base == 10000.0
    @test cfg.compress_rope_base == 160000.0

    spec = Gesso.Inference._spec_from_gguf_deepseek41(kv, nothing)
    @test spec.family == :deepseek41
    @test spec.num_layers == 6
    @test spec.hidden_size == 16
    @test spec.moe === nothing
    @test :sqrtsoftplus_moe in spec.features
    @test :ple in spec.features
end

@testset "deepseek41: bad metadata is refused" begin
    kv = _ds41_toy_kv()
    bad(f) = (k = copy(kv); f(k); k)
    # a missing required key names the key
    k = bad(k -> delete!(k, "deepseek41.attention.indexer.top_k"))
    @test_throws ErrorException Gesso.Inference.deepseek41_config(k)
    # compress ratios outside 0-2
    k = bad(k -> k["deepseek41.attention.compress_ratios"] = [0, 0, 3, 1, 2, 2])
    @test_throws ErrorException Gesso.Inference.deepseek41_config(k)
    # a compressed layer without a matching KV source
    k = bad(k -> k["deepseek41.attention.kv_source_layers"] = [2])
    @test_throws ErrorException Gesso.Inference.deepseek41_config(k)
    # gating must be sqrtsoftplus
    k = bad(k -> k["deepseek41.expert_gating_func"] = 2)
    @test_throws ErrorException Gesso.Inference.deepseek41_config(k)
    # multi-query only
    k = bad(k -> k["deepseek41.attention.head_count_kv"] = 2)
    @test_throws ErrorException Gesso.Inference.deepseek41_config(k)
    # wrong architecture
    k = bad(k -> k["general.architecture"] = "qwen4exp")
    @test_throws ErrorException Gesso.Inference.deepseek41_config(k)
end
