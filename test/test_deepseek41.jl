# deepseek41 (DeepSeek-V4.1-Flash) text forward: metadata reader and spec on a
# synthetic header, bad metadata refused, the loader on the toy GGUF, and the
# forward checked per block against a llama.cpp reference on the same toy.

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
    @test cfg.hc == 4
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
    # compressed layers always run YaRN, so the scaling type must be yarn
    k = bad(k -> k["deepseek41.rope.scaling.type"] = "linear")
    @test_throws ErrorException Gesso.Inference.deepseek41_config(k)
    k = bad(k -> delete!(k, "deepseek41.rope.scaling.type"))
    @test_throws ErrorException Gesso.Inference.deepseek41_config(k)
end

# The fixture and its llama reference (tools/make-deepseek41-toy-gguf.jl and
# tools/dump-deepseek41-ref.cpp). The reference ran llama with an f32 KV cache.
const DS41_FIXTURE = joinpath(@__DIR__, "fixtures", "deepseek41_toy", "deepseek41-toy.gguf")
const DS41_REFERENCE = joinpath(@__DIR__, "fixtures", "deepseek41_toy", "reference.txt")
# Max abs diff over the largest reference magnitude of the tap. Measured worst case
# across every checked tap is 2.2e-6 (logits 1.05e-6); this leaves about 4x headroom.
const DS41_REL_TOL = 1e-5

function _ds41_read_reference(path)
    toks = Int[]
    argmax = Int[]
    taps = Dict{String, Vector{Float64}}()
    for line in eachline(path)
        f = split(line)
        isempty(f) && continue
        if f[1] == "tokens"
            toks = parse.(Int, f[3:end])
        elseif f[1] == "argmax"
            push!(argmax, parse(Int, f[3]))
        elseif f[1] == "tap"
            taps[f[2]] = parse.(Float64, f[4:end])
        end
    end
    return toks, argmax, taps
end

# llama stores a (cols, T) tensor with cols fastest; this gives the Julia (T, cols) layout.
_ds41_ref_rows(v, cols, T) = permutedims(reshape(v, cols, T))

@testset "deepseek41: loader reads the fixture" begin
    m = Gesso.Inference.load_deepseek41(DS41_FIXTURE)
    @test m isa Gesso.Inference.Deepseek41Model
    @test m.cfg.n_layer == 6 && m.cfg.vocab == 40 && m.cfg.hc == 4
    @test length(m.index) == 165
    @test haskey(m.index, "blk.2.indexer.attn_q_b.weight")
    # Engram tables sit on the PLE layers only
    @test haskey(m.index, "blk.1.engram_kv.weight")
    @test !haskey(m.index, "blk.0.engram_kv.weight")
    @test_throws ErrorException Gesso.Inference.load_deepseek41(joinpath(@__DIR__, "no-such.gguf"))
end

@testset "deepseek41: forward matches the llama reference" begin
    toks, ref_argmax, ref = _ds41_read_reference(DS41_REFERENCE)
    m = Gesso.Inference.load_deepseek41(DS41_FIXTURE)
    cfg = m.cfg
    st = Gesso.Inference.Deepseek41State(cfg)
    taps = Dict{String, Any}()
    res = Gesso.Inference.deepseek41_forward!(m, st, toks; taps = taps)
    logits = Float64.(Gesso.Inference.deepseek41_logits_all(m, res; taps = taps))
    T = length(toks)
    NE, V = cfg.n_embd, cfg.vocab

    # max abs diff over the largest reference magnitude
    rel(a, b) = maximum(abs.(a .- b)) / max(maximum(abs, b), 1e-12)

    @test size(logits) == (T, V)
    @test rel(logits, _ds41_ref_rows(ref["result_output"], V, T)) <= DS41_REL_TOL
    @test [argmax(view(logits, t, :)) - 1 for t in 1:T] == ref_argmax
    for il in 0:(cfg.n_layer - 1), (base, cols) in (("attn_out", NE), ("ffn_out", NE), ("l_out", NE * cfg.hc))
        name = "$base-$il"
        jm = Float64.(taps[name])
        @test size(jm) == (T, cols)
        @test rel(jm, _ds41_ref_rows(ref[name], cols, T)) <= DS41_REL_TOL
    end
end
