# qwen4exp (Flash-Next) forward: the metadata reader on a synthetic header, a
# small random model fed in chunks against one call (GDN conv and delta state,
# KV cache, PLE token and conv history all carry across), the QSA context
# refusal, and (gated on GESSO_QWEN4EXP_GGUF, the first shard) a 4-layer run
# against sums and corners from the llama.cpp eval-callback dump, and with
# GESSO_QWEN4EXP_FULL=1 all 48 layers against the CPU llama.cpp top tokens.

using Test
using Gesso
using Random

include("qwen4exp_toy.jl")

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
