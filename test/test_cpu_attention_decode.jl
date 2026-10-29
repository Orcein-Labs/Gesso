# Fused single-query attention (CPU decode) against the reference attention.
#
#   julia --project=test -t 8 test/test_cpu_attention_decode.jl
#
# The reference is `_attention_heads!` over the GQA-repeated cache, the path
# the decode step used before. The fused kernel must reproduce its outputs
# bit for bit, for MHA, GQA and the one-key case.

using Gesso
using Test
using Random: Xoshiro

const _GI = Gesso.Inference

_adec_bits(a::AbstractArray{Float32}) = reinterpret(UInt32, vec(a))

@testset "fused single-query attention equals the reference (bit-exact)" begin
    rng = Xoshiro(20261009)
    backend = _GI.CPUBackend()
    wl = Gesso.Semantics.DecodeWorkload()
    # (n_heads, n_kv_heads, d_head, K keys in use, cache capacity)
    cases = [(8, 2, 16, 5, 12), (6, 6, 8, 1, 4), (12, 4, 32, 17, 40), (24, 4, 64, 300, 320)]
    for (n_heads, n_kv, d, K, cap) in cases
        group = div(n_heads, n_kv)
        q = randn(rng, Float32, 1, n_heads, d)
        kc = randn(rng, Float32, cap, n_kv, d)
        vc = randn(rng, Float32, cap, n_kv, d)

        kx = _GI._repeat_heads(_GI.Activation(; shape=(cap, n_kv, d), storage=kc), group)
        vx = _GI._repeat_heads(_GI.Activation(; shape=(cap, n_kv, d), storage=vc), group)
        scores = _GI.TemporaryWorkspace(; shape=(1, K), storage=zeros(Float32, 1, K))
        probs = _GI.TemporaryWorkspace(; shape=(1, K), storage=zeros(Float32, 1, K))
        ref = zeros(Float32, 1, n_heads, d)
        _GI._attention_heads!(
            backend,
            ref,
            scores,
            probs,
            q,
            kx.storage,
            vx.storage,
            n_heads,
            d,
            1,
            K,
            wl,
        )

        fused = fill(NaN32, 1, n_heads, d)
        buf = zeros(Float32, cap)
        _GI._attention_decode_cpu!(fused, q, kc, vc, buf, n_heads, d, group, K)
        @test _adec_bits(fused) == _adec_bits(ref)
    end
end

@testset "pooled heads equal the reference and the one-slot loop (bit-exact)" begin
    rng = Xoshiro(20261029)
    backend = _GI.CPUBackend()
    wl = Gesso.Semantics.DecodeWorkload()
    # d_head 64 scales by a multiply, 128 by a divide; both are big enough to
    # take the pool when the scratch holds a slot per head
    for (n_heads, n_kv, d, K) in ((8, 2, 64, 400), (16, 4, 128, 600))
        group = div(n_heads, n_kv)
        q = randn(rng, Float32, 1, n_heads, d)
        kc = randn(rng, Float32, K, n_kv, d)
        vc = randn(rng, Float32, K, n_kv, d)
        kx = _GI._repeat_heads(_GI.Activation(; shape=(K, n_kv, d), storage=kc), group)
        vx = _GI._repeat_heads(_GI.Activation(; shape=(K, n_kv, d), storage=vc), group)
        scores = _GI.TemporaryWorkspace(; shape=(1, K), storage=zeros(Float32, 1, K))
        probs = _GI.TemporaryWorkspace(; shape=(1, K), storage=zeros(Float32, 1, K))
        ref = zeros(Float32, 1, n_heads, d)
        _GI._attention_heads!(backend, ref, scores, probs, q, kx.storage, vx.storage,
            n_heads, d, 1, K, wl)
        for window in (0, 37)
            one = fill(NaN32, 1, n_heads, d)
            _GI._attention_decode_cpu!(one, q, kc, vc, zeros(Float32, K), n_heads, d, group, K;
                window)
            pooled = fill(NaN32, 1, n_heads, d)
            _GI._attention_decode_cpu!(pooled, q, kc, vc, zeros(Float32, K * n_heads), n_heads,
                d, group, K; window)
            @test _adec_bits(pooled) == _adec_bits(one)
            window == 0 && @test _adec_bits(pooled) == _adec_bits(ref)
        end
    end
end

@testset "window rows across the pool equal decode one row at a time (bit-exact)" begin
    rng = Xoshiro(20261030)
    # base 1 is a window from the first cache row; base 90 starts after a
    # committed prefix. Both are big enough to take the pool.
    for (n_heads, n_kv, d, base, T) in ((8, 2, 64, 1, 120), (16, 4, 128, 90, 70))
        group = div(n_heads, n_kv)
        Kmax = base + T - 1
        kc = randn(rng, Float32, Kmax, n_kv, d)
        vc = randn(rng, Float32, Kmax, n_kv, d)
        q = randn(rng, Float32, T, n_heads, d)
        for window in (0, 37)
            rows = fill(NaN32, T, n_heads, d)
            _GI._attention_window_cpu!(rows, q, kc, vc, n_heads, d, group, base, Float32;
                window)
            for t in 1:T
                one = fill(NaN32, 1, n_heads, d)
                _GI._attention_decode_cpu!(one, q[t:t, :, :], kc, vc, zeros(Float32, Kmax),
                    n_heads, d, group, base + t - 1; window)
                @test _adec_bits(rows[t:t, :, :]) == _adec_bits(one)
            end
        end
    end
end
