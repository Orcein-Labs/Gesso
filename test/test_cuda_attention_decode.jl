# CUDA fused single-query decode attention (ext/cuda_attention.jl) against a
# host F64 oracle with the GQA mapping written out explicitly.
#
# Tolerance. Inputs are N(0,1), so scores are O(1) and softmax weights are
# O(1/K). The kernel accumulates in F32 with an online-softmax rescale per
# tile; the error is a few ulps of the output scale per tile, so atol 1e-4
# on outputs of magnitude <= ~3 is a 30x margin over the expected error.
#
# Cache rows beyond K hold 1e3, so any read past row K shows up as a failure.

using Random: MersenneTwister

using Test: AbstractTestSet, Broken, get_testset, record
if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end


# The generic hook declines so the reference path runs on every other backend.
# This needs no device, so it is outside the skip.
@testset "decode attention hook: generic backend declines" begin
    z = zeros(Float32, 1, 2, 4)
    kc = zeros(Float32, 3, 1, 4)
    @test Gesso.Inference._attention_decode_device!(
        Gesso.CPUBackend(),
        z,
        z,
        kc,
        kc,
        2,
        4,
        2,
        3,
    ) === false
end

if !CUDA_LOADED || !CUDA.functional()
    _skip("no NVIDIA device — CUDA fused decode attention tests skipped (§LXXVII skip law)")
else
    _attn_cuda = Gesso.CUDABackend()

    # host F64 oracle: query head h reads KV head div(h-1, group)+1
    function _decode_attn_oracle(q, kc, vc, n_heads, group, K, d)
        out = zeros(Float64, 1, n_heads, d)
        for h in 1:n_heads
            hk = div(h - 1, group) + 1
            s = [
                sum(Float64(q[1, h, j]) * Float64(kc[u, hk, j]) for j in 1:d) / sqrt(d)
                for u in 1:K
            ]
            p = exp.(s .- maximum(s))
            p ./= sum(p)
            for j in 1:d, u in 1:K
                out[1, h, j] += p[u] * Float64(vc[u, hk, j])
            end
        end
        return out
    end

    function _run_decode_attn_case(n_heads, n_kv, d, K, extra; seed)
        rng = MersenneTwister(seed)
        R = K + extra
        q = randn(rng, Float32, 1, n_heads, d)
        kc = randn(rng, Float32, R, n_kv, d)
        vc = randn(rng, Float32, R, n_kv, d)
        kc[(K+1):end, :, :] .= 1.0f3
        vc[(K+1):end, :, :] .= 1.0f3
        group = div(n_heads, n_kv)
        attn = CUDA.zeros(Float32, 1, n_heads, d)
        ok = Gesso.Inference._attention_decode_device!(
            _attn_cuda,
            attn,
            CuArray(q),
            CuArray(kc),
            CuArray(vc),
            n_heads,
            d,
            group,
            K,
        )
        ok === true || return Inf
        want = _decode_attn_oracle(q, kc, vc, n_heads, group, K, d)
        return maximum(abs.(Float64.(Array(attn)) .- want))
    end

    @testset "cuda fused decode attention vs host F64 oracle" begin
        cases = [
            # (n_heads, n_kv, d_head, K, extra rows past K)
            (4, 4, 8, 1, 0),        # MHA, one key
            (4, 4, 8, 7, 2),        # MHA, K below the block size
            (4, 2, 8, 5, 3),        # GQA group 2
            (8, 2, 16, 33, 1),      # GQA group 4, K not a multiple of 32
            (6, 3, 32, 300, 5),     # GQA group 2, K not a multiple of the block
            (4, 1, 8, 1025, 0),     # one key past the first score tile
            (2, 1, 16, 2600, 7),    # three score tiles, last partial
            (16, 2, 256, 1, 0),     # Ornith-35B shape (group 8), one key
            (8, 1, 64, 700, 2),     # group 8, last key chunk partial
            (16, 2, 256, 4100, 3),  # Ornith-35B shape at a long context
            (6, 1, 16, 90, 1),      # group 6 takes the per-head kernel
        ]
        for (i, (nh, nk, d, K, extra)) in enumerate(cases)
            err = _run_decode_attn_case(nh, nk, d, K, extra; seed=0x5a0 + i)
            @test err <= 1e-4
        end
    end

    @testset "cuda fused decode attention rejects an empty cache" begin
        attn = CUDA.zeros(Float32, 1, 2, 4)
        kc = CUDA.zeros(Float32, 1, 1, 4)
        @test_throws ArgumentError Gesso.Inference._attention_decode_device!(
            _attn_cuda,
            attn,
            CUDA.zeros(Float32, 1, 2, 4),
            kc,
            kc,
            2,
            4,
            2,
            0,
        )
    end
end
