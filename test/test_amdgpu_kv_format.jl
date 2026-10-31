# AMDGPU q8 KV cache: the KA write kernel matches the host quantizer bit for
# bit (including exact ties), the dense copy reads back what getindex gives, the
# fused split decode reads codes in place within the 1e-4 bound and matches the
# dense decode exactly, and each kernel compiles for the AMD targets with no
# device. Runs on KA's CPU backend. GPU-UNTESTED: nothing here runs on ROCm.
#
# The production path (ROCArray storage through _kv_write_rows! and _kv_dense) is
# not run here: it needs a device. It differs from these tests only in the KA
# backend it passes, and the backend-explicit forms are what these tests call.

using Random: MersenneTwister
using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

# F64 softmax(q k^T / sqrt(d)) v for one decode row, GQA heads sharing a KV head.
function _amdkv_decode_oracle(q, kc, vc, n_heads, group, K, d)
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

if !AMDGPU_LOADED || AMDGPU_EXT === nothing
    @test _skip("AMDGPU.jl or KernelAbstractions not loadable — AMDGPU KV format tests skipped")
else
    const _AMDKV_KA = _AMDKA.CPU()

    _amdkv_host(R::Int, n_kv::Int, d::Int) = Gesso.Inference.QuantKV{:q8}(
        zeros(Int8, R, n_kv, d), zeros(Float16, R, n_kv, d ÷ 32), d,
    )

    # Each q8 row block runs through the device kernel on KA's CPU backend.
    _amdkv_ka(R::Int, n_kv::Int, d::Int) = _amdkv_host(R, n_kv, d)

    # Rows whose 32-value blocks land exactly on half steps, so round-half-even
    # decides the code (the block scale is 127/127 = 1).
    _amdkv_tie_block() = Float32[127, 0.5, 1.5, 2.5, -2.5, -0.5, 3.5, -1.5]

    @testset "amdgpu KV q8: write kernel matches the host quantizer bit for bit" begin
        rng = MersenneTwister(20261034)
        for (R, n, n_kv, d, row0) in ((8, 5, 3, 64, 2), (6, 1, 1, 32, 1), (10, 7, 2, 128, 4))
            src = randn(rng, Float32, n, n_kv, d)
            for h in 1:n_kv
                src[1, h, 1:32] .= repeat(_amdkv_tie_block(), 4)[1:32]
            end
            host = _amdkv_host(R, n_kv, d)
            dev = _amdkv_ka(R, n_kv, d)
            Gesso.Inference._kv_write_rows!(host, src, row0)
            K = _AMDX._ka_kv_write_rows!(_AMDKV_KA, dev, src, row0)
            @test K === nothing
            @test dev.codes == host.codes
            @test dev.scales == host.scales
        end
        # a block of zeros keeps scale 0 and codes 0
        src = zeros(Float32, 2, 1, 32)
        host = _amdkv_host(2, 1, 32)
        dev = _amdkv_ka(2, 1, 32)
        Gesso.Inference._kv_write_rows!(host, src, 1)
        _AMDX._ka_kv_write_rows!(_AMDKV_KA, dev, src, 1)
        @test dev.codes == host.codes
        @test dev.scales == host.scales
    end

    @testset "amdgpu KV q8: dense copy reads back the cache elements" begin
        rng = MersenneTwister(20261035)
        R, n_kv, d, K = 40, 2, 96, 37
        dev = _amdkv_ka(R, n_kv, d)
        src = randn(rng, Float32, K, n_kv, d)
        _AMDX._ka_kv_write_rows!(_AMDKV_KA, dev, src, 1)
        dense = _AMDX._ka_kv_dense(_AMDKV_KA, dev, K)
        @test size(dense) == (K, n_kv, d)
        ref = [dev[u, h, j] for u in 1:K, h in 1:n_kv, j in 1:d]
        @test dense == ref
        # the copy is quantized, not the source: each block sits within half a step
        err = maximum(abs.(dense .- src))
        @test err <= maximum(abs.(src)) / 127
    end

    @testset "amdgpu KV q8: fused split decode vs F64 oracle (GQA, several chunks, 1e-4)" begin
        rng = MersenneTwister(20261036)
        for (H, nkv, d, K) in ((4, 2, 64, 1100), (8, 2, 128, 513), (2, 1, 32, 40))
            group = div(H, nkv)
            R = K + 3
            kq = _amdkv_ka(R, nkv, d)
            vq = _amdkv_ka(R, nkv, d)
            _AMDX._ka_kv_write_rows!(_AMDKV_KA, kq, randn(rng, Float32, K, nkv, d), 1)
            _AMDX._ka_kv_write_rows!(_AMDKV_KA, vq, randn(rng, Float32, K, nkv, d), 1)
            q = randn(rng, Float32, 1, H, d)
            attn = zeros(Float32, 1, H, d)
            ok = _AMDX._ka_attention_decode!(_AMDKV_KA, attn, q, kq, vq, H, d, group, K; wg=64)
            @test ok === true
            kd = _AMDX._ka_kv_dense(_AMDKV_KA, kq, K)
            vd = _AMDX._ka_kv_dense(_AMDKV_KA, vq, K)
            ref = _amdkv_decode_oracle(q, kd, vd, H, group, K, d)
            err = maximum(abs.(Float64.(attn) .- ref))
            tol = 1.0e-4 * max(1.0, maximum(abs.(ref)))
            println("  kv q8 decode H=$H nkv=$nkv d=$d K=$K max_err=$(err) tol=$(tol)")
            @test err <= tol

            # the in-place read is the dense read, element for element
            attn_dense = zeros(Float32, 1, H, d)
            _AMDX._ka_attention_decode!(_AMDKV_KA, attn_dense, q, kd, vd, H, d, group, K; wg=64)
            @test attn == attn_dense
        end
    end

    @testset "amdgpu KV q8: prefill and per-head paths refuse a quantized cache" begin
        H, d, L, K = 2, 32, 1, 4
        q = zeros(Float32, L, H, d)
        attn = zeros(Float32, L, H, d)
        sc = zeros(Float32, L, K)
        pr = zeros(Float32, L, K)
        kq = _amdkv_ka(K, H, d)
        vd = zeros(Float32, K, H, d)
        e = try
            _AMDX._ka_attention_heads!(_AMDKV_KA, attn, sc, pr, q, kq, vd, H, d, L, K)
            nothing
        catch err
            err
        end
        @test e isa Gesso.GessoError && e.code === Gesso.ERR_INVALID_PLAN
    end

    @testset "amdgpu KV q8: compile and link for $(join(first.(AMD_TARGETS), ", "))" begin
        G = amd_global
        K = _AMDX
        for wg in (64, 256)
            res = amd_compile(K._ka_kv_q8_write_kernel!, wg, (wg * 2,),
                (G(Int8, 3), G(Float16, 3), G(Float32, 3), Int, Int, Int))
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)

            res = amd_compile(K._ka_kv_q8_dense_kernel!, wg, (wg * 2,),
                (G(Float32, 3), G(Int8, 3), G(Float16, 3), Int, Int))
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)

            res = amd_compile(K._ka_attn_split_kernel!, wg, (wg * 2,),
                (G(Float32, 3), G(Float32, 3), G(Int8, 3), G(Float16, 3), G(Int8, 3), G(Float16, 3),
                    Bool, Bool, Int, Int, Float32, Int, Int))
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)
        end
    end
end
