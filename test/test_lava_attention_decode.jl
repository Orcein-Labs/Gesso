# Lava fused single-query decode attention (ext/lava_attention.jl) against a
# host F64 oracle with the GQA mapping written out explicitly. Same cases and
# tolerance as the CUDA counterpart, so the two device paths are judged by one
# yardstick.
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

if !LAVA_LOADED || !VULKAN_OK
    _skip("no usable Vulkan device — Lava fused decode attention tests skipped")
else
    _attn_lava = Gesso.LavaBackend()

    # host F64 oracle: query head h reads KV head div(h-1, group)+1
    function _lava_decode_attn_oracle(q, kc, vc, n_heads, group, K, d)
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

    function _run_lava_decode_attn_case(n_heads, n_kv, d, K, extra; seed)
        rng = MersenneTwister(seed)
        R = K + extra
        q = randn(rng, Float32, 1, n_heads, d)
        kc = randn(rng, Float32, R, n_kv, d)
        vc = randn(rng, Float32, R, n_kv, d)
        kc[(K+1):end, :, :] .= 1.0f3
        vc[(K+1):end, :, :] .= 1.0f3
        group = div(n_heads, n_kv)
        attn = Lava.LavaArray{Float32}(zeros(Float32, 1, n_heads, d))
        ok = Gesso.Inference._attention_decode_device!(
            _attn_lava,
            attn,
            Lava.LavaArray{Float32}(q),
            Lava.LavaArray{Float32}(kc),
            Lava.LavaArray{Float32}(vc),
            n_heads,
            d,
            group,
            K,
        )
        ok === true || return Inf
        want = _lava_decode_attn_oracle(q, kc, vc, n_heads, group, K, d)
        return maximum(abs.(Float64.(Array(attn)) .- want))
    end

    @testset "lava fused decode attention vs host F64 oracle" begin
        cases = [
            # (n_heads, n_kv, d_head, K, extra rows past K)
            (4, 4, 8, 1, 0),        # MHA, one key
            (4, 4, 8, 7, 2),        # MHA, K below the tile
            (4, 2, 8, 5, 3),        # GQA group 2
            (8, 2, 16, 33, 1),      # GQA group 4, K not a multiple of 64
            (6, 3, 32, 300, 5),     # GQA group 2, K not a multiple of the tile
            (4, 1, 8, 1025, 0),     # one key past the second score tile
            (2, 1, 16, 2600, 7),    # several score tiles, last partial
            (16, 2, 256, 2048, 0),  # the benchmark shape (group 8, d_head 256)
        ]
        for (i, (nh, nk, d, K, extra)) in enumerate(cases)
            err = _run_lava_decode_attn_case(nh, nk, d, K, extra; seed=0x7a0 + i)
            @test err <= 1e-4
        end
    end

    @testset "lava fused decode attention rejects an empty cache" begin
        attn = Lava.LavaArray{Float32}(zeros(Float32, 1, 2, 4))
        kc = Lava.LavaArray{Float32}(zeros(Float32, 1, 1, 4))
        @test_throws ArgumentError Gesso.Inference._attention_decode_device!(
            _attn_lava,
            attn,
            attn,
            kc,
            kc,
            2,
            4,
            2,
            0,
        )
    end
end
