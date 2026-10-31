# AMDGPU attention on KADeviceBackend. Parity runs on KernelAbstractions' CPU
# backend against an F64 oracle, with the same 1e-4 tolerance as the Lava and
# CUDA decode tests. Compile checks lower each kernel for the AMD targets with
# no device present. GPU-untested, like the rest of the AMDGPU seam.

using Random: MersenneTwister

function _amd_attn_heads_oracle(q, k, v, n_heads, d, L, K, window)
    out = zeros(Float64, L, n_heads, d)
    for h in 1:n_heads, t in 1:L
        hi = K - L + t
        lo = window > 0 ? max(1, hi - window + 1) : 1
        s = [sum(Float64(q[t, h, c]) * Float64(k[j, h, c]) for c in 1:d) / sqrt(d) for j in lo:hi]
        w = exp.(s .- maximum(s))
        w ./= sum(w)
        for (a, j) in enumerate(lo:hi), c in 1:d
            out[t, h, c] += w[a] * Float64(v[j, h, c])
        end
    end
    return out
end

function _amd_attn_decode_oracle(q, kc, vc, n_heads, group, K, d)
    out = zeros(Float64, 1, n_heads, d)
    for h in 1:n_heads
        hk = div(h - 1, group) + 1
        s = [sum(Float64(q[1, h, c]) * Float64(kc[j, hk, c]) for c in 1:d) / sqrt(d) for j in 1:K]
        w = exp.(s .- maximum(s))
        w ./= sum(w)
        for j in 1:K, c in 1:d
            out[1, h, c] += w[j] * Float64(vc[j, hk, c])
        end
    end
    return out
end

function _amd_attn_catch(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

const _AMD_ATTN_HEAD_CASES = (
    (2, 8, 1, 1, 0),
    (3, 8, 1, 5, 0),
    (4, 16, 5, 7, 0),
    (2, 8, 4, 9, 3),
    (1, 8, 3, 3, 2),
    (4, 16, 7, 33, 0),
    (2, 32, 3, 70, 5),
)

const _AMD_ATTN_DECODE_CASES = (
    (4, 4, 8, 1, 0),
    (4, 4, 8, 7, 2),
    (4, 2, 8, 5, 3),
    (8, 2, 16, 33, 1),
    (6, 3, 32, 300, 5),
    (4, 1, 8, 1025, 0),
    (2, 1, 16, 2600, 7),
    (16, 2, 256, 2048, 0),
)

if !AMDGPU_LOADED || AMDGPU_EXT === nothing
    @test _skip("AMDGPU.jl or KernelAbstractions not loadable — AMDGPU attention tests skipped")
else
    const _AMD_ATTN_KA = _AMDKA.CPU()

    @testset "amdgpu attention: per-head parity vs F64 oracle (odd sizes, causal, window, K=1)" begin
        rng = MersenneTwister(20261031)
        for (H, d, L, K, window) in _AMD_ATTN_HEAD_CASES, wg in (16, 64)
            q = randn(rng, Float32, L, H, d)
            k = randn(rng, Float32, K, H, d)
            v = randn(rng, Float32, K, H, d)
            attn = zeros(Float32, L, H, d)
            sc = zeros(Float32, L, K)
            pr = zeros(Float32, L, K)
            _AMDX._ka_attention_heads!(_AMD_ATTN_KA, attn, sc, pr, q, k, v, H, d, L, K; window, wg)
            ref = _amd_attn_heads_oracle(q, k, v, H, d, L, K, window)
            err = maximum(abs.(Float64.(attn) .- ref))
            println("  heads H=$H d=$d L=$L K=$K window=$window wg=$wg max_err=$(err)")
            @test err <= 1e-4
        end
    end

    @testset "amdgpu attention: masked keys never reach the sum, even when non-finite" begin
        rng = MersenneTwister(7)
        H, d, L, K = 1, 4, 2, 5
        q = randn(rng, Float32, L, H, d)
        k = randn(rng, Float32, K, H, d)
        v = randn(rng, Float32, K, H, d)
        v[K, :, :] .= NaN32
        attn = zeros(Float32, L, H, d)
        sc = zeros(Float32, L, K)
        pr = zeros(Float32, L, K)
        _AMDX._ka_attention_heads!(_AMD_ATTN_KA, attn, sc, pr, q, k, v, H, d, L, K; wg=16)
        # row 1 sees keys 1:4 only (causal), so key 5 is masked and must not poison it
        @test all(isfinite, attn[1, :, :])
        ref = _amd_attn_heads_oracle(q, k, v, H, d, L, K, 0)
        @test maximum(abs.(Float64.(attn[1, :, :]) .- ref[1, :, :])) <= 1e-4
    end

    @testset "amdgpu attention: split-K decode parity vs F64 oracle (GQA, 1e-4)" begin
        rng = MersenneTwister(20261101)
        for (H, nkv, d, K, extra) in _AMD_ATTN_DECODE_CASES
            group = div(H, nkv)
            q = randn(rng, Float32, 1, H, d)
            kc = randn(rng, Float32, K + extra, nkv, d)
            vc = randn(rng, Float32, K + extra, nkv, d)
            kc[(K + 1):end, :, :] .= 1.0f3
            vc[(K + 1):end, :, :] .= 1.0f3
            attn = zeros(Float32, 1, H, d)
            ok = _AMDX._ka_attention_decode!(_AMD_ATTN_KA, attn, q, kc, vc, H, d, group, K; wg=64)
            @test ok === true
            ref = _amd_attn_decode_oracle(q, kc, vc, H, group, K, d)
            err = maximum(abs.(Float64.(attn) .- ref))
            println("  decode H=$H nkv=$nkv d=$d K=$K max_err=$(err)")
            @test err <= 1e-4
        end
        # a second workgroup size on one multi-tile case
        H, nkv, d, K = 4, 4, 8, 1025
        q = randn(rng, Float32, 1, H, d)
        kc = randn(rng, Float32, K, nkv, d)
        vc = randn(rng, Float32, K, nkv, d)
        attn = zeros(Float32, 1, H, d)
        _AMDX._ka_attention_decode!(_AMD_ATTN_KA, attn, q, kc, vc, H, d, 1, K; wg=256)
        ref = _amd_attn_decode_oracle(q, kc, vc, H, 1, K, d)
        err = maximum(abs.(Float64.(attn) .- ref))
        println("  decode wg=256 K=$K max_err=$(err)")
        @test err <= 1e-4
    end

    @testset "amdgpu attention: invalid workgroup throws ERR_INVALID_PLAN" begin
        H, d, L, K = 2, 8, 1, 4
        q = zeros(Float32, L, H, d)
        k = zeros(Float32, K, H, d)
        v = zeros(Float32, K, H, d)
        attn = zeros(Float32, L, H, d)
        sc = zeros(Float32, L, K)
        pr = zeros(Float32, L, K)
        qd = zeros(Float32, 1, H, d)
        kcd = zeros(Float32, K, H, d)
        for wg in (0, 48, 512)
            e1 = _amd_attn_catch(() -> _AMDX._ka_attention_heads!(_AMD_ATTN_KA, attn, sc, pr, q, k, v, H, d, L, K; wg))
            @test e1 isa Gesso.GessoError
            @test e1 isa Gesso.GessoError && e1.code === Gesso.ERR_INVALID_PLAN
            e2 = _amd_attn_catch(() -> _AMDX._ka_attention_decode!(_AMD_ATTN_KA, zeros(Float32, 1, H, d), qd, kcd, kcd, H, d, 1, K; wg))
            @test e2 isa Gesso.GessoError
            @test e2 isa Gesso.GessoError && e2.code === Gesso.ERR_INVALID_PLAN
        end
    end

    @testset "amdgpu attention: empty cache, wide head, quantized cache" begin
        H, d = 2, 8
        attn = zeros(Float32, 1, H, d)
        q = zeros(Float32, 1, H, d)
        kc = zeros(Float32, 4, H, d)
        @test_throws ArgumentError _AMDX._ka_attention_decode!(_AMD_ATTN_KA, attn, q, kc, kc, H, d, 1, 0)
        @test _AMDX._ka_attention_decode!(
            _AMD_ATTN_KA,
            zeros(Float32, 1, 1, 520),
            zeros(Float32, 1, 1, 520),
            zeros(Float32, 4, 1, 520),
            zeros(Float32, 4, 1, 520),
            1,
            520,
            1,
            4,
        ) === false
        qk = Gesso.Inference.QuantKV{:q4h}(zeros(UInt8, 4, H, d ÷ 2), zeros(Float16, 4, H, 1), d)
        e = _amd_attn_catch(() -> _AMDX._ka_attention_decode!(_AMD_ATTN_KA, attn, q, qk, kc, H, d, 1, 4))
        @test e isa Gesso.GessoError
        @test e isa Gesso.GessoError && e.code === Gesso.ERR_INVALID_PLAN
    end

    @testset "amdgpu attention: compile and link for $(join(first.(AMD_TARGETS), ", "))" begin
        G = amd_global
        K = _AMDX
        for wg in (64, 256)
            res = amd_compile(K._ka_attn_scores_kernel!, wg, (wg,),
                (G(Float32, 2), G(Float32, 3), G(Float32, 3), Int, Float32, Int, Int, Int))
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)

            res = amd_compile(K._ka_attn_softmax_values_kernel!, wg, (wg * 2,),
                (G(Float32, 3), G(Float32, 2), G(Float32, 2), G(Float32, 3), Int, Int, Int, Int, Int))
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)

            res = amd_compile(K._ka_attn_split_kernel!, wg, (wg * 2,),
                (G(Float32, 3), G(Float32, 3), G(Float32, 3), G(Float32, 3), G(Float32, 3), G(Float32, 3),
                    Bool, Bool, Int, Int, Float32, Int, Int))
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)

            res = amd_compile(K._ka_attn_combine_kernel!, wg, (wg * 2,),
                (G(Float32, 3), G(Float32, 3), Int, Int))
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)
        end
    end
end
