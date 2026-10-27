# Lava storage of the quantized KV cache (ext/lava_kv_format.jl). The device q8
# write must reproduce the host quantizer bit for bit, the dense read must
# reproduce the host dequantizer, and the fused decode over a device q8 cache
# must agree with an F64 oracle that reads the host dequantized values.
#
# Negative controls corrupt the device cache in the two ways a plausible bug
# would: scales shifted by one block, and codes read as unsigned. Each must
# exceed the tolerance, so a pass here means the comparison can fail.

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
    _skip("no usable Vulkan device — Lava q8 KV cache tests skipped")
else
    _kvl = Gesso.Inference
    _lava_kv_like = Lava.LavaArray{Float32}(zeros(Float32, 0, 0, 0))
    _lava_kv_tol(ref) = 1e-4 * max(1.0, maximum(abs, ref))

    # the host F64 oracle of decode attention over dense K/V; query head h reads
    # KV head div(h-1, group)+1
    function _lava_kv_decode_oracle(q, kc, vc, n_heads, group, K, d)
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

    # rows of a block that the host quantizer must round to even: each
    # x / scale lands exactly on a half-integer, so half-away-from-zero differs
    _lava_kv_tie_block() = Float32[127, 0.5, 1.5, 2.5, -2.5, -0.5, 3.5, -1.5]

    function _lava_kv_src(rng, rows, n_kv, d)
        src = randn(rng, Float32, rows, n_kv, d)
        blk = repeat(_lava_kv_tie_block(), 4)
        for h in 1:n_kv
            src[1, h, 1:32] .= blk
        end
        return src
    end

    @testset "q8 default is on for Lava when d_head is a whole number of blocks" begin
        @test _kvl._kv_default_format(:lava, 64) === :q8
        @test _kvl._kv_default_format(:lava, 256) === :q8
        @test _kvl._kv_default_format(:lava, 16) === :f32
        @test _kvl._kv_default_format(:cpu, 64) === :f32
    end

    @testset "q4h on Lava is refused with ERR_INVALID_PLAN" begin
        err = try
            _kvl._kv_alloc(:q4h, _lava_kv_like, 8, 2, 64)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError
        @test err.code == Gesso.ERR_INVALID_PLAN
        @test occursin("q4h", sprint(showerror, err))
    end

    @testset "device q8 write equals the host write bit for bit" begin
        rng = MersenneTwister(0x6b76)
        rows, n_kv, d, row0 = 8, 2, 64, 3
        src = _lava_kv_src(rng, 5, n_kv, d)
        host = _kvl._kv_alloc(:q8, zeros(Float32, 0, 0, 0), rows, n_kv, d)
        dev = _kvl._kv_alloc(:q8, _lava_kv_like, rows, n_kv, d)
        @test dev isa _kvl.QuantKV{:q8,<:Lava.LavaArray,<:Lava.LavaArray}
        _kvl._kv_write_rows!(host, src, row0)
        _kvl._kv_write_rows!(dev, Lava.LavaArray{Float32}(src), row0)
        println("lava q8 write: codes that differ from the host = ", count(Array(dev.codes) .!= host.codes))
        @test Array(dev.codes) == host.codes
        @test Array(dev.scales) == host.scales
        # the tie block: round to even, with 127 giving scale 1.0 exactly
        @test host.codes[row0, 1, 1:8] == Int8[127, 0, 2, 2, -2, 0, 4, -2]
        @test Array(dev.codes)[row0, 1, 1:8] == Int8[127, 0, 2, 2, -2, 0, 4, -2]
    end

    @testset "device dense read equals the host dequantizer" begin
        rng = MersenneTwister(0x6d72)
        rows, n_kv, d, K = 12, 2, 64, 9
        src = _lava_kv_src(rng, K, n_kv, d)
        host = _kvl._kv_alloc(:q8, zeros(Float32, 0, 0, 0), rows, n_kv, d)
        dev = _kvl._kv_alloc(:q8, _lava_kv_like, rows, n_kv, d)
        _kvl._kv_write_rows!(host, src, 1)
        _kvl._kv_write_rows!(dev, Lava.LavaArray{Float32}(src), 1)
        dense = Array(_kvl._kv_dense(dev, K))
        ref = Float32[host[u, h, j] for u in 1:K, h in 1:n_kv, j in 1:d]
        @test size(dense) == (K, n_kv, d)
        @test dense == ref
    end

    @testset "fused decode over a device q8 cache matches the F64 oracle" begin
        rng = MersenneTwister(0x6b64)
        n_heads, n_kv, d, K, extra = 4, 2, 64, 600, 8
        group = div(n_heads, n_kv)
        R = K + extra
        host_k = _kvl._kv_alloc(:q8, zeros(Float32, 0, 0, 0), R, n_kv, d)
        host_v = _kvl._kv_alloc(:q8, zeros(Float32, 0, 0, 0), R, n_kv, d)
        dev_k = _kvl._kv_alloc(:q8, _lava_kv_like, R, n_kv, d)
        dev_v = _kvl._kv_alloc(:q8, _lava_kv_like, R, n_kv, d)
        ksrc = _lava_kv_src(rng, K, n_kv, d)
        vsrc = _lava_kv_src(rng, K, n_kv, d)
        garbage = fill(1.0f3, extra, n_kv, d)
        for (host, dev, src) in ((host_k, dev_k, ksrc), (host_v, dev_v, vsrc))
            _kvl._kv_write_rows!(host, src, 1)
            _kvl._kv_write_rows!(host, garbage, K + 1)
            _kvl._kv_write_rows!(dev, Lava.LavaArray{Float32}(src), 1)
            _kvl._kv_write_rows!(dev, Lava.LavaArray{Float32}(garbage), K + 1)
        end
        q = randn(rng, Float32, 1, n_heads, d)
        kref = Float32[host_k[u, h, j] for u in 1:K, h in 1:n_kv, j in 1:d]
        vref = Float32[host_v[u, h, j] for u in 1:K, h in 1:n_kv, j in 1:d]
        ref = _lava_kv_decode_oracle(q, kref, vref, n_heads, group, K, d)
        attn = Lava.LavaArray{Float32}(zeros(Float32, 1, n_heads, d))
        ok = _kvl._attention_decode_device!(
            Gesso.LavaBackend(),
            attn,
            Lava.LavaArray{Float32}(q),
            dev_k,
            dev_v,
            n_heads,
            d,
            group,
            K,
        )
        @test ok
        got = Float64.(Array(attn))
        err = maximum(abs, got .- ref)
        @test err <= _lava_kv_tol(ref)
        println("lava q8 decode max|err| vs F64 oracle = ", err)
        # the same kernel over the dense dequantized cache reads the same
        # values in the same order, so the two must agree bit for bit
        dense_attn = Lava.LavaArray{Float32}(zeros(Float32, 1, n_heads, d))
        @test _kvl._attention_decode_device!(
            Gesso.LavaBackend(),
            dense_attn,
            Lava.LavaArray{Float32}(q),
            _kvl._kv_dense(dev_k, R),
            _kvl._kv_dense(dev_v, R),
            n_heads,
            d,
            group,
            K,
        )
        @test Array(attn) == Array(dense_attn)
        println(
            "lava f32 decode over the dequantized cache max|err| vs F64 oracle = ",
            maximum(abs, Float64.(Array(dense_attn)) .- ref),
        )
    end

    @testset "negative control: scales shifted by one block exceed tolerance" begin
        rng = MersenneTwister(0x6e63)
        rows, n_kv, d, K = 12, 2, 64, 9
        src = _lava_kv_src(rng, K, n_kv, d)
        host = _kvl._kv_alloc(:q8, zeros(Float32, 0, 0, 0), rows, n_kv, d)
        dev = _kvl._kv_alloc(:q8, _lava_kv_like, rows, n_kv, d)
        _kvl._kv_write_rows!(host, src, 1)
        _kvl._kv_write_rows!(dev, Lava.LavaArray{Float32}(src), 1)
        ref = Float32[host[u, h, j] for u in 1:K, h in 1:n_kv, j in 1:d]
        shifted = Lava.LavaArray{Float16}(circshift(Array(dev.scales), (0, 0, 1)))
        bad = _kvl.QuantKV{:q8}(dev.codes, shifted, d)
        err = maximum(abs, Array(_kvl._kv_dense(bad, K)) .- ref)
        @test err > _lava_kv_tol(ref)
    end

    @testset "negative control: codes read as UInt8 exceed tolerance" begin
        rng = MersenneTwister(0x6e75)
        rows, n_kv, d, K = 12, 2, 64, 9
        src = _lava_kv_src(rng, K, n_kv, d)
        host = _kvl._kv_alloc(:q8, zeros(Float32, 0, 0, 0), rows, n_kv, d)
        dev = _kvl._kv_alloc(:q8, _lava_kv_like, rows, n_kv, d)
        _kvl._kv_write_rows!(host, src, 1)
        _kvl._kv_write_rows!(dev, Lava.LavaArray{Float32}(src), 1)
        ref = Float32[host[u, h, j] for u in 1:K, h in 1:n_kv, j in 1:d]
        unsigned = Lava.LavaArray{UInt8}(collect(reinterpret(UInt8, Array(dev.codes))))
        bad = _kvl.QuantKV{:q8}(unsigned, dev.scales, d)
        err = maximum(abs, Array(_kvl._kv_dense(bad, K)) .- ref)
        @test err > _lava_kv_tol(ref)
    end
    @testset "negative control: the fused decode with shifted scales exceeds tolerance" begin
        rng = MersenneTwister(0x6e66)
        n_heads, n_kv, d, K = 4, 2, 64, 40
        group = div(n_heads, n_kv)
        host_k = _kvl._kv_alloc(:q8, zeros(Float32, 0, 0, 0), K, n_kv, d)
        host_v = _kvl._kv_alloc(:q8, zeros(Float32, 0, 0, 0), K, n_kv, d)
        dev_k = _kvl._kv_alloc(:q8, _lava_kv_like, K, n_kv, d)
        dev_v = _kvl._kv_alloc(:q8, _lava_kv_like, K, n_kv, d)
        for (host, dev) in ((host_k, dev_k), (host_v, dev_v))
            src = _lava_kv_src(rng, K, n_kv, d)
            _kvl._kv_write_rows!(host, src, 1)
            _kvl._kv_write_rows!(dev, Lava.LavaArray{Float32}(src), 1)
        end
        q = randn(rng, Float32, 1, n_heads, d)
        kref = Float32[host_k[u, h, j] for u in 1:K, h in 1:n_kv, j in 1:d]
        vref = Float32[host_v[u, h, j] for u in 1:K, h in 1:n_kv, j in 1:d]
        ref = _lava_kv_decode_oracle(q, kref, vref, n_heads, group, K, d)
        shift(x) = Lava.LavaArray{Float16}(circshift(Array(x.scales), (0, 0, 1)))
        bad_v = _kvl.QuantKV{:q8}(dev_v.codes, shift(dev_v), d)
        attn = Lava.LavaArray{Float32}(zeros(Float32, 1, n_heads, d))
        @test _kvl._attention_decode_device!(
            Gesso.LavaBackend(),
            attn,
            Lava.LavaArray{Float32}(q),
            dev_k,
            bad_v,
            n_heads,
            d,
            group,
            K,
        )
        err = maximum(abs, Float64.(Array(attn)) .- ref)
        @test err > _lava_kv_tol(ref)
        println("lava q8 decode, shifted V scales: max|err| = ", err)
    end

end
