# Phase 8 item C tests (§LXXXI): full prefill + greedy generate on the
# Vulkan path vs the CPU F64 oracle. The correctness gate is the declared
# atol on logits (F32 device vs F64 host) and EXACT token-id identity for
# greedy argmax — a divergent argmax is a real bug, not a tolerance problem.
#
# One named skip without a usable Vulkan device. The toy2 fingerprint file
# on disk stays the CPU one; nothing here rewrites expected_logits.toml.
# LAVA_LOADED / VULKAN_OK come from test_lava_seam.jl (include order is
# load-bearing, same as the CUDA seam/ops/inference chain).

using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

using .GessoTestHelpers: approx_eq

if !LAVA_LOADED || !VULKAN_OK
    _skip(
        "no usable Vulkan device (Lava.vk_context() failed) — Lava inference tests skipped (§LXXXI skip law)",
    )
else
    lava = Gesso.LavaBackend()

    # ---- Phase 10B item B: Lava does NOT have the CUDA fast-path caps ------
    # The device fast paths (:argmax = argmax on device; :attn_gemm = flat
    # device GEMM over the gathered scratch) are CUDA-only capabilities this
    # sprint — declared via supports, not silently shared. Lava keeps the
    # full-row host argmax and the per-head contraction. Type-level probes:
    # legal without a device (no construction, §XX probing is always safe).
    @testset "Phase 10B: Lava lacks the CUDA fast-path caps" begin
        @test Gesso.supports(Gesso.LavaBackend, :argmax) == false
        @test Gesso.supports(Gesso.LavaBackend, :attn_gemm) == false
        # the six real Lava capabilities are untouched by the two new caps
        @test Gesso.supports(Gesso.LavaBackend, :matmul) == true
        @test Gesso.supports(Gesso.LavaBackend, :quantize) == false
    end

    # ---- toy2 --------------------------------------------------------------

    ts = toy2_tensors()
    cpu_logits = Gesso.reference_prefill(toy2_modelir(), ts, PROMPT)   # F64 host

    @testset "toy2 Lava prefill vs CPU oracle (atol=1e-3)" begin
        gpu_ts = Gesso.to_device(lava, ts)
        gpu_logits = Gesso.reference_prefill(toy2_modelir(), gpu_ts, PROMPT; backend=lava)
        @test size(gpu_logits) == size(cpu_logits)
        # the host copy back is explicit: Array(...) on the returned matrix;
        # approx_eq is same-eltype by design, so widen F32 → F64 for compare
        got = Float64.(Array(gpu_logits))
        @test approx_eq(got, cpu_logits; atol=1e-3)
        # numerical-delta fingerprint for the receipt (F32-vs-F64 spread)
        println("toy2 lava-vs-cpu max|Δlogit| = ", maximum(abs, got .- cpu_logits))
    end

    @testset "toy2 Lava greedy generate: token ids match CPU" begin
        gpu_ts = Gesso.to_device(lava, ts)
        cpu_ids = Gesso.reference_generate(toy2_modelir(), ts, PROMPT; max_new_tokens=8)
        gpu_ids = Gesso.reference_generate(
            toy2_modelir(),
            gpu_ts,
            PROMPT;
            backend=lava,
            max_new_tokens=8,
        )
        @test gpu_ids == cpu_ids                        # argmax identity gate
    end

    @testset "toy2 CPUBackend default is still bit-identical (regression)" begin
        @test approx_eq(
            Gesso.reference_prefill(toy2_modelir(), ts, PROMPT),
            cpu_logits;
            atol=0.0,
        )
    end

    # ---- llama_micro (GQA on device) ----------------------------------------

    @testset "llama_micro Lava prefill vs CPU (atol=1e-3), GQA end-to-end" begin
        dir = mktempdir()
        make_micro_checkpoint(dir)
        model, cpu_tensors, cfg = Gesso.load_llama(dir)
        tokens = [0, 1, 2]
        cpu_out = Gesso.reference_prefill(
            model,
            cpu_tensors,
            tokens;
            eps=cfg.rms_norm_eps,
            theta=cfg.rope_theta,
        )
        gpu_tensors = Gesso.to_device(lava, cpu_tensors)
        gpu_out = Float64.(
            Array(
                Gesso.reference_prefill(
                    model,
                    gpu_tensors,
                    tokens;
                    backend=lava,
                    eps=cfg.rms_norm_eps,
                    theta=cfg.rope_theta,
                ),
            ),
        )
        @test size(gpu_out) == size(cpu_out) == (32, 3)
        @test approx_eq(gpu_out, cpu_out; atol=1e-3)

        # greedy generation matches token-for-token on the GQA micro model
        cpu_gen = Gesso.reference_generate(
            model,
            cpu_tensors,
            tokens;
            max_new_tokens=3,
            eps=cfg.rms_norm_eps,
            theta=cfg.rope_theta,
        )
        gpu_gen = Gesso.reference_generate(
            model,
            gpu_tensors,
            tokens;
            backend=lava,
            max_new_tokens=3,
            eps=cfg.rms_norm_eps,
            theta=cfg.rope_theta,
        )
        @test gpu_gen == cpu_gen
        println("llama_micro lava-vs-cpu max|Δlogit| = ", maximum(abs, gpu_out .- cpu_out))
    end

    # ---- the interpreter does not copy (§LXX) --------------------------------

    @testset "Lava backend + host Array storage throws ERR_INVALID_PLAN" begin
        for fn in (
            () -> Gesso.reference_prefill(toy2_modelir(), ts, PROMPT; backend=lava),
            () -> Gesso.reference_generate(toy2_modelir(), ts, PROMPT; backend=lava),
        )
            err = try
                fn()
                nothing
            catch e
                e
            end
            @test err isa Gesso.GessoError
            @test err.code == Gesso.ERR_INVALID_PLAN
            @test occursin("to_device", sprint(showerror, err))
        end
        # and the CPU tensors were not silently mutated by the attempt
        @test ts.embedding.storage isa Array{Float64}
    end
end
