# KV cache formats (src/Inference/kv_format.jl, ext/cuda_kv_format.jl) and the
# tiled prefill attention that reads them (ext/cuda_attention.jl).
#
# Tolerances. q8 rounds to the nearest of 255 steps of amax/127, so a value is
# off by at most half a step plus the Float16 rounding of the scale (2^-11
# relative): amax/127 per element is a 2x margin. q4h on Gaussian rows is
# Lloyd-Max at 16 levels, distortion 0.0095 of the variance when the scale is
# the true sigma; a per-32 RMS estimate adds a little, so 0.02 is the bound.

using Random: MersenneTwister

using Test: AbstractTestSet, Broken, get_testset, record
if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const _KVI = Gesso.Inference

_kv_dense(x) = Float32[x[u, h, j] for u in axes(x, 1), h in axes(x, 2), j in axes(x, 3)]

@testset "kv format: GESSO_KV_FORMAT parsing" begin
    for (raw, want) in (("f32", :f32), (" Q4H ", :q4h), ("q8", :q8), ("f16", :f16))
        @test withenv(() -> _KVI._kv_format(), "GESSO_KV_FORMAT" => raw) === want
    end
    @test withenv(() -> _KVI._kv_format(), "GESSO_KV_FORMAT" => nothing) === :f32
    @test_throws Gesso.GessoError withenv(
        () -> _KVI._kv_format(),
        "GESSO_KV_FORMAT" => "q3",
    )
    # unset takes the backend default; a set variable overrides it either way
    @test withenv(() -> _KVI._kv_format(:q8), "GESSO_KV_FORMAT" => nothing) === :q8
    @test withenv(() -> _KVI._kv_format(:q8), "GESSO_KV_FORMAT" => "f32") === :f32
    @test _KVI._kv_default_format(:cuda, 256) === :q8
    @test _KVI._kv_default_format(:cuda, 16) === :f32
    @test _KVI._kv_default_format(:cpu, 256) === :f32
end

@testset "kv format: allocation sizes and shape checks" begin
    like = zeros(Float32, 0, 0, 0)
    q8 = _KVI._kv_alloc(:q8, like, 10, 2, 256)
    q4 = _KVI._kv_alloc(:q4h, like, 10, 2, 256)
    @test size(q8) == size(q4) == (10, 2, 256)
    f32 = 4 * 10 * 2 * 256
    @test (sizeof(q8.codes) + sizeof(q8.scales)) / f32 ≈ 272 / 1024
    @test (sizeof(q4.codes) + sizeof(q4.scales)) / f32 ≈ 144 / 1024
    @test eltype(_KVI._kv_alloc(:f16, like, 3, 1, 64)) === Float16
    @test_throws Gesso.GessoError _KVI._kv_alloc(:q8, like, 3, 1, 40)
    @test_throws Gesso.GessoError _KVI._kv_alloc(:q4h, like, 3, 1, 96)
end

@testset "kv format: rotation is orthonormal" begin
    rng = MersenneTwister(0x4b1)
    x = randn(rng, Float32, 5, 3, 64)
    y = _KVI._kv_rotate!(copy(x), true)
    @test maximum(abs.(_KVI._kv_rotate!(copy(y), false) .- x)) <= 1e-5
    for r in 1:5, h in 1:3
        @test sum(abs2, y[r, h, :]) ≈ sum(abs2, x[r, h, :]) rtol = 1e-5
    end
    # (Rq)'(Rk) = q'k: the score a rotated query takes against a rotated key
    @test sum(y[1, 1, :] .* y[2, 2, :]) ≈ sum(x[1, 1, :] .* x[2, 2, :]) atol = 1e-4
end

@testset "kv format: q8 and q4h round trips" begin
    rng = MersenneTwister(0x4b2)
    rows, n_kv, d = 64, 2, 128
    x = randn(rng, Float32, rows, n_kv, d)
    x[:, 1, 7] .*= 20  # an outlier channel, which is what q4h's rotation is for
    like = zeros(Float32, 0, 0, 0)

    q8 = _KVI._kv_alloc(:q8, like, rows, n_kv, d)
    _KVI._kv_write_rows!(q8, x, 1)
    err = abs.(_kv_dense(q8) .- x)
    for u in 1:rows, h in 1:n_kv, b in 1:(d÷32)
        js = ((b-1)*32+1):(b*32)
        @test maximum(err[u, h, js]) <= maximum(abs.(x[u, h, js])) / 127
    end

    q4 = _KVI._kv_alloc(:q4h, like, rows, n_kv, d)
    _KVI._kv_write_rows!(q4, x, 1)
    back = _KVI._kv_rotate!(_kv_dense(q4), false)
    for h in 1:n_kv
        @test sum(abs2, back[:, h, :] .- x[:, h, :]) / sum(abs2, x[:, h, :]) <= 0.02
    end

    # rows land where asked and leave the rest alone
    part = _KVI._kv_alloc(:q8, like, rows, n_kv, d)
    _KVI._kv_write_rows!(part, x[3:4, :, :], 10)
    @test _kv_dense(part)[10:11, :, :] == _kv_dense(q8)[3:4, :, :]
    @test all(iszero, _kv_dense(part)[1:9, :, :])

    a = _KVI._kv_alloc(:q4h, like, 4, n_kv, d)
    b = _KVI._kv_alloc(:q4h, like, 4, n_kv, d)
    _KVI._kv_append_storage!(a, b, x[5:5, :, :], x[6:6, :, :], 3)
    @test _kv_dense(a)[3, :, :] == _kv_dense(q4)[5, :, :]
    @test _kv_dense(b)[3, :, :] == _kv_dense(q4)[6, :, :]
end

@testset "kv format: CPU decode reads a quantized cache as its values" begin
    rng = MersenneTwister(0x4b3)
    n_heads, n_kv, d, K = 4, 2, 64, 9
    kx = randn(rng, Float32, K, n_kv, d)
    vx = randn(rng, Float32, K, n_kv, d)
    q = randn(rng, Float32, 1, n_heads, d)
    like = zeros(Float32, 0, 0, 0)
    for fmt in (:q8, :q4h)
        kc = _KVI._kv_alloc(fmt, like, K, n_kv, d)
        vc = _KVI._kv_alloc(fmt, like, K, n_kv, d)
        _KVI._kv_write_rows!(kc, kx, 1)
        _KVI._kv_write_rows!(vc, vx, 1)
        a = zeros(Float32, 1, n_heads, d)
        b = zeros(Float32, 1, n_heads, d)
        _KVI._attention_decode_cpu!(a, q, kc, vc, zeros(Float32, K), n_heads, d, 2, K)
        _KVI._attention_decode_cpu!(
            b,
            q,
            _kv_dense(kc),
            _kv_dense(vc),
            zeros(Float32, K),
            n_heads,
            d,
            2,
            K,
        )
        @test a == b
    end
end

if !CUDA_LOADED || !CUDA.functional()
    _skip(
        "no NVIDIA device — CUDA KV format and tiled prefill tests skipped (§LXXVII skip law)",
    )
else
    _kv_cuda = Gesso.CUDABackend()
    _kv_like = CUDA.zeros(Float32, 0, 0, 0)
    # a device QuantKV adapts to a host one; Array() would index its codes one by one
    _kv_host(x) = CUDA.CUDACore.Adapt.adapt(Array, x)

    @testset "cuda kv format: device writes match the host" begin
        rng = MersenneTwister(0x4b4)
        rows, n_kv, d = 40, 2, 256
        x = randn(rng, Float32, rows, n_kv, d)
        x[:, 2, 100] .*= 30
        for fmt in (:q8, :q4h)
            h = _KVI._kv_alloc(fmt, zeros(Float32, 0, 0, 0), rows + 3, n_kv, d)
            g = _KVI._kv_alloc(fmt, _kv_like, rows + 3, n_kv, d)
            _KVI._kv_write_rows!(h, x, 3)
            _KVI._kv_write_rows!(g, CuArray(x), 3)
            if fmt === :q8
                # absmax is order-free, so codes and scales are exact
                @test Array(g.codes) == h.codes
                @test Array(g.scales) == h.scales
            else
                # the RMS sums in a different order on device; a last-bit
                # difference in it may move a code sitting on a threshold
                @test count(Array(g.codes) .!= h.codes) <= length(h.codes) ÷ 1000
                @test maximum(abs.(Float32.(Array(g.scales)) .- Float32.(h.scales))) <=
                      1e-3 * maximum(Float32.(h.scales))
            end
        end
    end

    @testset "cuda kv format: device rotation is the host's bit for bit" begin
        rng = MersenneTwister(0x4b5)
        x = randn(rng, Float32, 7, 16, 256)
        for fwd in (true, false)
            @test Array(_KVI._kv_rotate!(CuArray(x), fwd)) == _KVI._kv_rotate!(copy(x), fwd)
        end
    end

    @testset "cuda kv format: fused decode reads a quantized cache as its values" begin
        rng = MersenneTwister(0x4b6)
        n_heads, n_kv, d = 16, 2, 256
        kx = randn(rng, Float32, 1504, n_kv, d)
        vx = randn(rng, Float32, 1504, n_kv, d)
        q = CuArray(randn(rng, Float32, 1, n_heads, d))
        # rows % 4 == 0 takes the 4-row loads, with the last chunk ending on
        # and off a row group; 1503 rows take the per-element loads
        for (rows, K) in ((1504, 1500), (1504, 1501), (1503, 1499)),
            fmt in (:f16, :q8, :q4h)

            kc = _KVI._kv_alloc(fmt, _kv_like, rows, n_kv, d)
            vc = _KVI._kv_alloc(fmt, _kv_like, rows, n_kv, d)
            _KVI._kv_write_rows!(kc, CuArray(kx[1:rows, :, :]), 1)
            _KVI._kv_write_rows!(vc, CuArray(vx[1:rows, :, :]), 1)
            kd, vd = CuArray(_kv_dense(_kv_host(kc))), CuArray(_kv_dense(_kv_host(vc)))
            a = CUDA.zeros(Float32, 1, n_heads, d)
            b = CUDA.zeros(Float32, 1, n_heads, d)
            @test _KVI._attention_decode_device!(_kv_cuda, a, q, kc, vc, n_heads, d, 8, K)
            @test _KVI._attention_decode_device!(_kv_cuda, b, q, kd, vd, n_heads, d, 8, K)
            @test maximum(abs.(Array(a) .- Array(b))) <= 1e-6
        end
    end

    # Query row i attends keys 1 .. K - L + i.
    function _prefill_attn_oracle(q, k, v, n_heads, group, L, K, d)
        out = zeros(Float64, L, n_heads, d)
        for h in 1:n_heads, i in 1:L
            hk = div(h - 1, group) + 1
            vis = K - L + i
            s = [
                sum(Float64(q[i, h, j]) * Float64(k[u, hk, j]) for j in 1:d) / sqrt(d)
                for u in 1:vis
            ]
            p = exp.(s .- maximum(s))
            p ./= sum(p)
            for j in 1:d
                out[i, h, j] = sum(p[u] * Float64(v[u, hk, j]) for u in 1:vis)
            end
        end
        return out
    end

    @testset "cuda tiled prefill attention vs host F64 oracle" begin
        HE = Base.get_extension(Gesso, :GessoCUDAExt)
        cases = [
            # (n_heads, n_kv, d_head, L, K)
            (4, 2, 16, 5, 5),          # one pass, one tile
            (4, 1, 32, 8, 300),        # a later pass: 292 cached keys before it
            (2, 1, 16, 2048, 4548),    # three key tiles, the last partial
        ]
        for (i, (nh, nk, d, L, K)) in enumerate(cases)
            rng = MersenneTwister(0x4c0 + i)
            q = randn(rng, Float32, L, nh, d)
            k = randn(rng, Float32, K + 3, nk, d)
            v = randn(rng, Float32, K + 3, nk, d)
            k[(K+1):end, :, :] .= 1.0f3
            v[(K+1):end, :, :] .= 1.0f3
            want = _prefill_attn_oracle(q, k, v, nh, div(nh, nk), L, K, d)
            attn = CUDA.zeros(Float32, L, nh, d)
            kv = (view(CuArray(k), 1:K, :, :), view(CuArray(v), 1:K, :, :))
            @test _KVI._attention_prefill_device!(
                _kv_cuda,
                attn,
                CuArray(q),
                kv...,
                nh,
                d,
                div(nh, nk),
                L,
                K,
            )
            @test maximum(abs.(Float64.(Array(attn)) .- want)) <= 1e-4
            i == 3 && @test cld(K, HE._attn_prefill_tile(L, K)) == 3
        end
    end

    @testset "cuda tiled prefill reads a converted cache through its tiles" begin
        rng = MersenneTwister(0x4c8)
        nh, nk, d, L, K = 16, 2, 256, 64, 900
        q = CuArray(randn(rng, Float32, L, nh, d))
        kx = CuArray(randn(rng, Float32, K, nk, d))
        vx = CuArray(randn(rng, Float32, K, nk, d))
        for fmt in (:f16, :q8, :q4h)
            kc = _KVI._kv_alloc(fmt, _kv_like, K + 5, nk, d)
            vc = _KVI._kv_alloc(fmt, _kv_like, K + 5, nk, d)
            _KVI._kv_write_rows!(kc, kx, 1)
            _KVI._kv_write_rows!(vc, vx, 1)
            kd, vd = CuArray(_kv_dense(_kv_host(kc))), CuArray(_kv_dense(_kv_host(vc)))
            a = CUDA.zeros(Float32, L, nh, d)
            b = CUDA.zeros(Float32, L, nh, d)
            @test _KVI._attention_prefill_device!(
                _kv_cuda,
                a,
                q,
                view(kc, 1:K, :, :),
                view(vc, 1:K, :, :),
                nh,
                d,
                8,
                L,
                K,
            )
            @test _KVI._attention_prefill_device!(
                _kv_cuda,
                b,
                q,
                view(kd, 1:K, :, :),
                view(vd, 1:K, :, :),
                nh,
                d,
                8,
                L,
                K,
            )
            @test maximum(abs.(Array(a) .- Array(b))) <= 1e-5
        end
    end
end
