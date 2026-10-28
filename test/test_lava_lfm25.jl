# LFM2 / LFM2.5 on Lava: the short-convolution kernel against the CPU loop
# with history carried across calls, then the dense and MoE toys through
# prefill and greedy decode on Lava and on the CPU. Named skip without a usable
# Vulkan device.
#
# Tolerance. The kernel runs the CPU loop's operations in the same order in
# F32, so it gets 1e-5 * max(1, |ref|). The forward bound is 1e-4 * max(1,
# max|ref|), as in test_lava_spark25.jl. The negative controls (reversed
# convolution taps, reversed router rows) move the logits far outside it.

using Test: AbstractTestSet, Broken, get_testset, record
using Random: MersenneTwister

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

isdefined(@__MODULE__, :LAVA_LOADED) || include(joinpath(@__DIR__, "test_lava_seam.jl"))
isdefined(@__MODULE__, :_lfm_toy) || include(joinpath(@__DIR__, "lfm25_toy.jl"))

if !LAVA_LOADED || !VULKAN_OK
    _skip("no usable Vulkan device — Lava LFM2 tests skipped (§LXXXI skip law)")
else
    _llf_lava = Gesso.LavaBackend()
    _llf_cpu = Gesso.CPUBackend()

    _llf_err(a, b) = maximum(abs, Float64.(Array(a)) .- Float64.(Array(b)); init=0.0)
    _llf_tol(ref) = 1.0e-4 * max(1.0, maximum(abs, Float64.(Array(ref)); init=0.0))
    _llf_dev(a) = Lava.LavaArray{Float32}(a)

    _LLF_PROMPT = [3, 7, 1, 12, 5, 9, 2]
    _LLF_NEW = 6

    @testset "lava LFM2: short convolution matches the CPU loop" begin
        rng = MersenneTwister(2028)
        dim, width = 70, 3
        w = randn(rng, Float32, dim, width)
        hist_c = zeros(Float32, dim, width - 1)
        hist_d = _llf_dev(hist_c)
        # A prompt, then single tokens, then a chunk: the history carries over.
        for seq in (5, 1, 1, 4)
            bcx = randn(rng, Float32, seq, 3dim)
            ref = zeros(Float32, seq, dim)
            _lfm_inf._short_conv_mix!(_llf_cpu, ref, bcx, w, hist_c)
            got = _llf_dev(zeros(Float32, seq, dim))
            _lfm_inf._short_conv_mix!(_llf_lava, got, _llf_dev(bcx), _llf_dev(w), hist_d)
            @test _llf_err(got, ref) <= 1.0e-5 * max(1.0, maximum(abs, ref))
            @test _llf_err(hist_d, hist_c) <= 1.0e-5 * max(1.0, maximum(abs, hist_c))
        end
        # Host storage is refused, not copied.
        @test_throws Exception _lfm_inf._short_conv_mix!(_llf_lava, zeros(Float32, 1, dim),
            _llf_dev(zeros(Float32, 1, 3dim)), _llf_dev(w), hist_d)
    end

    for moe in (false, true)
        name = moe ? "MoE" : "dense"
        spec = _lfm_inf.spec_from_gguf_kv(_lfm_toy_kv(moe; width=128))
        eps = spec.rms_norm_eps
        by = _lfm_toy(spec)
        model, t = _lfm_bind(spec, by)
        t_dev = Gesso.to_device(_llf_lava, t)
        prefill(m, tt, backend) = Array(Gesso.reference_prefill(m, tt, _LLF_PROMPT; backend, eps))
        function generate(tt, backend)
            rows = Matrix{Float32}[]
            ids = Gesso.reference_generate(model, tt, _LLF_PROMPT; backend, eps,
                max_new_tokens=_LLF_NEW, eos_ids=(), on_rows=(pos, M) -> push!(rows, Float32.(Array(M))))
            return ids, rows
        end
        logits = prefill(model, t, _llf_cpu)

        @testset "lava LFM2 $name: import report finds no missing capability" begin
            @test isempty(_lfm_inf._missing_capabilities(Gesso.required_semantics(spec), _llf_lava))
        end

        @testset "lava LFM2 $name: toy prefill" begin
            @test _llf_err(prefill(model, t_dev, _llf_lava), logits) <= _llf_tol(logits)
        end

        # The CPU cache is f32, so the tight bound holds with Lava's pinned to
        # f32 too; the q8 arm gets 200x, as in test_lava_spark25.jl.
        @testset "lava LFM2 $name: toy greedy decode" begin
            ids_c, rows_c = generate(t, _llf_cpu)
            for (fmt, k) in (("f32", 1), ("q8", 200))
                ids_d, rows_d = withenv(() -> generate(t_dev, _llf_lava), "GESSO_KV_FORMAT" => fmt)
                @test ids_d == ids_c
                @test length(rows_d) == length(rows_c) == _LLF_NEW
                for i in eachindex(rows_c, rows_d)
                    @test _llf_err(rows_d[i], rows_c[i]) <= k * _llf_tol(rows_c[i])
                end
            end
        end

        @testset "lava LFM2 $name: negative controls exceed the bound" begin
            tol = _llf_tol(logits)
            reversed = deepcopy(by)
            for (n, w) in reversed
                endswith(n, "shortconv.conv.weight") && (reversed[n] = w[:, end:-1:1])
            end
            m, tt = _lfm_bind(spec, reversed)
            @test _llf_err(prefill(m, Gesso.to_device(_llf_lava, tt), _llf_lava), logits) > tol
            if moe
                permuted = deepcopy(by)
                for (n, w) in permuted
                    endswith(n, "ffn_gate_inp.weight") && (permuted[n] = w[end:-1:1, :])
                end
                m, tt = _lfm_bind(spec, permuted)
                @test _llf_err(prefill(m, Gesso.to_device(_llf_lava, tt), _llf_lava), logits) > tol
            end
        end
    end
end
