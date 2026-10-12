# Row-range partials on the device (ext/cuda_attention.jl) merged with host
# partials (src/Inference/kv_tier.jl) by log-sum-exp. A split at any row must
# give the full attention. Uses the helpers of test_kv_tier.jl, which is
# included earlier in runtests.jl.
#
# Tolerance is 1e-5 relerr, as the host tier test. The negative control moves
# the device range one row past its split, skipping a row; that comparison
# must exceed the tolerance, or the check cannot see a wrong range.

using Test: AbstractTestSet, Broken, get_testset, record
if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

if !CUDA_LOADED || !CUDA.functional()
    _skip("no NVIDIA device — CUDA tier partial tests skipped (§LXXVII skip law)")
else
    const _kvtg_be = Gesso.CUDABackend()
    const _kvtg_cfgs = ((16, 2, 256, 5000), (8, 2, 256, 1000), (4, 2, 256, 300), (2, 2, 256, 1000))
    const _kvtg_tol = 1e-5

    _kvtg_dev(x) = CUDA.CUDACore.Adapt.adapt(CuArray, x)

    # Device partial over rows, downloaded to a host KVPartial.
    function _kvtg_gpu_partial(q, kc, vc, rows, n_heads, d, group)
        pm = CUDA.zeros(Float32, n_heads)
        pl = CUDA.zeros(Float32, n_heads)
        pacc = CUDA.zeros(Float32, n_heads, d)
        _KVT._attention_partial_device!(_kvtg_be, pm, pl, pacc, CuArray(q), kc, vc, rows, n_heads, d, group)
        return _KVT._partial_download!(_KVT._kv_partial(n_heads, d), pm, pl, pacc)
    end

    function _kvtg_cpu_partial(q, kc, vc, rows, n_heads, d, group)
        part = _KVT._kv_partial(n_heads, d)
        return _KVT._attention_partial_cpu!(part, q, kc, vc, rows, n_heads, d, group)
    end

    function _kvtg_finish(parts, n_heads, d)
        total = _KVT._kv_partial(n_heads, d)
        for p in parts
            _KVT._partial_merge!(total, p)
        end
        return _KVT._partial_finish!(zeros(Float32, 1, n_heads, d), total)
    end

    # Device decode over all K rows, the reference for the device partials.
    function _kvtg_decode(qq, kd_k, vd_v, n_heads, d, group, K)
        attn = CUDA.zeros(Float32, 1, n_heads, d)
        ok = _KVT._attention_decode_device!(_kvtg_be, attn, CuArray(qq), kd_k, vd_v, n_heads, d, group, K)
        ok === true || error("decode hook declined")
        return Array(attn)
    end

    @testset "cuda tier partial: full range matches the device decode" begin
        worst = 0.0
        for (n_heads, n_kv, d, K) in _kvtg_cfgs, (fi, fmt) in enumerate(_KVT_FORMATS)
            group = div(n_heads, n_kv)
            rng = MersenneTwister(0x6d00 + 17 * fi + K)
            q = randn(rng, Float32, 1, n_heads, d)
            kh = _kvt_cache(fmt, randn(rng, Float32, K, n_kv, d))
            vh = _kvt_cache(fmt, randn(rng, Float32, K, n_kv, d))
            qq = _kvt_query(fmt, q)
            kd, vd = _kvtg_dev(kh), _kvtg_dev(vh)
            dec = _kvtg_decode(qq, kd, vd, n_heads, d, group, K)
            out = _kvtg_finish([_kvtg_gpu_partial(qq, kd, vd, 1:K, n_heads, d, group)], n_heads, d)
            e = _kvt_relerr(out, dec)
            worst = max(worst, e)
            @test e <= _kvtg_tol
        end
        @info "cuda tier partial full range" worst_relerr = worst
    end

    @testset "cuda tier partial: device head and host tail merge to the full attention" begin
        worst = 0.0
        worst_rev = 0.0
        for (n_heads, n_kv, d, K) in _kvtg_cfgs, (fi, fmt) in enumerate(_KVT_FORMATS)
            group = div(n_heads, n_kv)
            rng = MersenneTwister(0x6d80 + 17 * fi + K)
            q = randn(rng, Float32, 1, n_heads, d)
            kh = _kvt_cache(fmt, randn(rng, Float32, K, n_kv, d))
            vh = _kvt_cache(fmt, randn(rng, Float32, K, n_kv, d))
            qq = _kvt_query(fmt, q)
            kd, vd = _kvtg_dev(kh), _kvtg_dev(vh)
            dec = _kvtg_decode(qq, kd, vd, n_heads, d, group, K)
            splits = unique([0, 1, 3, 4, 5, 64, K ÷ 3, K ÷ 2 + 1, K - 1, K])
            for a in splits
                0 <= a <= K || continue
                gpu_head = _kvtg_gpu_partial(qq, kd, vd, 1:a, n_heads, d, group)
                cpu_tail = _kvtg_cpu_partial(qq, kh, vh, (a + 1):K, n_heads, d, group)
                e = _kvt_relerr(_kvtg_finish([gpu_head, cpu_tail], n_heads, d), dec)
                worst = max(worst, e)
                @test e <= _kvtg_tol
                # the other orientation: host head, device tail
                cpu_head = _kvtg_cpu_partial(qq, kh, vh, 1:a, n_heads, d, group)
                gpu_tail = _kvtg_gpu_partial(qq, kd, vd, (a + 1):K, n_heads, d, group)
                e2 = _kvt_relerr(_kvtg_finish([cpu_head, gpu_tail], n_heads, d), dec)
                worst_rev = max(worst_rev, e2)
                @test e2 <= _kvtg_tol
            end
        end
        @info "cuda tier partial head/tail" worst_relerr = worst worst_relerr_host_head = worst_rev
    end

    @testset "cuda tier partial: q8 agrees across aligned and unaligned range starts" begin
        n_heads, n_kv, d, K = 16, 2, 256, 1000
        group = div(n_heads, n_kv)
        rng = MersenneTwister(0x6dc0)
        q = randn(rng, Float32, 1, n_heads, d)
        kh = _kvt_cache(:q8, randn(rng, Float32, K, n_kv, d))
        vh = _kvt_cache(:q8, randn(rng, Float32, K, n_kv, d))
        kd, vd = _kvtg_dev(kh), _kvtg_dev(vh)
        dec = _kvtg_decode(q, kd, vd, n_heads, d, group, K)
        worst_al = 0.0
        worst_un = 0.0
        for r0 in (1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 64, 65)
            # the device range starts at r0 + 1, so its tiles start at r0
            # (vector loads when r0 % 4 == 0)
            gpu_tail = _kvtg_gpu_partial(q, kd, vd, (r0 + 1):K, n_heads, d, group)
            cpu_head = _kvtg_cpu_partial(q, kh, vh, 1:r0, n_heads, d, group)
            e = _kvt_relerr(_kvtg_finish([cpu_head, gpu_tail], n_heads, d), dec)
            if r0 % 4 == 0
                worst_al = max(worst_al, e)
            else
                worst_un = max(worst_un, e)
            end
            @test e <= _kvtg_tol
        end
        # the device partial over all rows against the host partial over the same rows
        full_gpu = _kvtg_gpu_partial(q, kd, vd, 1:K, n_heads, d, group)
        full_cpu = _kvtg_cpu_partial(q, kh, vh, 1:K, n_heads, d, group)
        e_host = _kvt_relerr(_kvtg_finish([full_gpu], n_heads, d), _kvtg_finish([full_cpu], n_heads, d))
        @test e_host <= _kvtg_tol
        @info "cuda tier q8" worst_aligned = worst_al worst_unaligned = worst_un device_vs_host = e_host
    end

    @testset "cuda tier partial: negative control, a skipped row must fail the comparison" begin
        smallest = Inf
        for (n_heads, n_kv, d, K) in _kvtg_cfgs, fmt in _KVT_FORMATS
            group = div(n_heads, n_kv)
            rng = MersenneTwister(0x6de0 + K)
            q = randn(rng, Float32, 1, n_heads, d)
            kh = _kvt_cache(fmt, randn(rng, Float32, K, n_kv, d))
            vh = _kvt_cache(fmt, randn(rng, Float32, K, n_kv, d))
            qq = _kvt_query(fmt, q)
            kd, vd = _kvtg_dev(kh), _kvtg_dev(vh)
            dec = _kvtg_decode(qq, kd, vd, n_heads, d, group, K)
            a = K ÷ 2
            # the tail starts at a + 2, so row a + 1 is in neither range
            gpu_head = _kvtg_gpu_partial(qq, kd, vd, 1:a, n_heads, d, group)
            bad_tail = _kvtg_cpu_partial(qq, kh, vh, (a + 2):K, n_heads, d, group)
            e = _kvt_relerr(_kvtg_finish([gpu_head, bad_tail], n_heads, d), dec)
            smallest = min(smallest, e)
            @test e > _kvtg_tol
        end
        @info "cuda tier negative control" smallest_wrong_range_relerr = smallest
    end
end
