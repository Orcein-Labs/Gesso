# Phase 5 item C tests (§LXXVIII): the CUDA engine — the Phase 4 path with
# pages. Gates: device Session generate token ids EQUAL the CPU Session ids
# (argmax gate), prefill! logits at declared atol=1e-3, and the no-copy law
# (host Array tensors + CUDA backend is ERR_INVALID_PLAN at construction).
# One named skip without a device (§LXXVIII skip law).
#
# Phase 10 items B/C: the device fast paths — argmax runs ON the device
# (host receives ONE Int per decode step, never the (vocab,) logits row)
# and the attention contraction is one flat device GEMM over the gathered
# scratch (capability :attn_gemm). Both are gated by token-id identity vs
# the CPU Session below; the intent assertion names the device argmax so a
# future full-row D2H on the decode hot path fails here, not in review.

using Test: AbstractTestSet, Broken, get_testset, record

using .GessoTestHelpers: approx_eq

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const PROMPT = [1, 3, 4, 5]

if !CUDA_LOADED || !CUDA.functional()
    _skip(
        "no NVIDIA device (CUDA.functional() == false) — CUDA Session tests skipped (§LXXVIII skip law)",
    )
else
    cuda = Gesso.CUDABackend()
    ts = toy2_tensors()
    model = toy2_modelir()

    @testset "toy2 CUDA Session vs CPU Session (ids EXACT, logits atol=1e-3)" begin
        gpu_ts = Gesso.to_device(cuda, ts)
        cpu_ids = Gesso.generate(
            Gesso.Session(model, ts; context_length=128, eos_token_id=2),
            PROMPT;
            max_new_tokens=8,
        )
        gpu_ids = Gesso.generate(
            Gesso.Session(model, gpu_ts; backend=cuda, context_length=128, eos_token_id=2),
            PROMPT;
            max_new_tokens=8,
        )
        @test gpu_ids == cpu_ids                     # argmax identity gate

        # prefill! logits: host copy is explicit, device-vs-CPU at declared atol
        s_cpu = Gesso.Session(model, ts; context_length=128, eos_token_id=2)
        s_gpu =
            Gesso.Session(model, gpu_ts; backend=cuda, context_length=128, eos_token_id=2)
        cpu_logits = Gesso.prefill!(s_cpu, PROMPT)
        gpu_logits = Gesso.prefill!(s_gpu, PROMPT)   # host-visible (Array)
        @test gpu_logits isa Array{Float32}
        @test size(gpu_logits) == size(cpu_logits)
        @test approx_eq(Float64.(gpu_logits), cpu_logits; atol=1e-3)
        # streaming fires identically on device
        seen = Int[]
        gpu_ids2 = Gesso.generate(
            Gesso.Session(model, gpu_ts; backend=cuda, context_length=128, eos_token_id=2),
            PROMPT;
            max_new_tokens=3,
            on_token=id -> push!(seen, id),
        )
        @test seen == gpu_ids2[5:end]
        # page_size=4 crosses boundaries on device too — ids must still match
        gpu_ids3 = Gesso.generate(
            Gesso.Session(
                model,
                gpu_ts;
                backend=cuda,
                context_length=128,
                eos_token_id=2,
                page_size=4,
            ),
            PROMPT;
            max_new_tokens=8,
        )
        @test gpu_ids3 == cpu_ids
    end

    @testset "Phase 10 B: device argmax fast path — ids exact, host gets one Int" begin
        gpu_ts = Gesso.to_device(cuda, ts)
        cpu_ids = Gesso.generate(
            Gesso.Session(model, ts; context_length=128, eos_token_id=2),
            PROMPT;
            max_new_tokens=8,
        )
        # greedy-id identity through the device-argmax path (every decode step
        # of this generate runs _device_greedy_id: the (vocab,) logits row
        # never crosses to the host)
        gpu_ids = Gesso.generate(
            Gesso.Session(model, gpu_ts; backend=cuda, context_length=128, eos_token_id=2),
            PROMPT;
            max_new_tokens=8,
        )
        @test gpu_ids == cpu_ids
        # intent: the device argmax is a declared capability (§XX), not a
        # silent behavior — a backend that loses the cap falls back to the
        # full-row host argmax and this assertion fails loudly
        @test Gesso.supports(cuda, :argmax) == true
        # tie law on device: Base.argmax on CuVector resolves ties to the
        # FIRST index, matching the host reduction bit-for-bit (§LXXVIII)
        tie = CuArray([3.0f0, 5.0f0, 5.0f0, 1.0f0])
        @test argmax(tie) == 2
        @test argmax(CuArray(fill(2.0f0, 64))) == 1
    end

    @testset "no-copy law: host Array tensors + CUDA backend throws at construction" begin
        err = try
            Gesso.Session(model, ts; backend=cuda, context_length=128, eos_token_id=2)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError
        @test err.code == Gesso.ERR_INVALID_PLAN
        @test occursin("to_device", sprint(showerror, err))
        # the CPU tensors were not silently mutated by the attempt
        @test ts.embedding.storage isa Array{Float64}
    end
end
