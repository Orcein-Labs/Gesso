# Phase 9 item B tests (§LXXXII): the two CUDA `matmul!` candidates —
# :cublas_mul and :generic_mul — both gated against the CPU F64 oracle at the
# EXISTING Phase 4 CUDA op atol (1e-2; no third atol invented), on toy2-sized
# and llama_micro-sized pairs. A deliberately-wrong candidate is rejected
# (proven under its own op key so the real pair's registration order stays
# untouched). quantize!/dequantize! still decline; the CPU matmul! path is
# untouched by Autotune (no consult).
#
# Phase 9 item C (§LXXXII): the CUDA `matmul!` op methods CONSULT Autotune —
# first call searches (cache MISS), later calls hit the cache — and dispatch
# to the winning candidate by name. Engine-level gates here: toy2 and
# llama_micro greedy ids still match CPU exactly, logits stay inside the
# declared atol, host Array + CUDABackend is still ERR_INVALID_PLAN, and the
# receipts name the winner.
#
# Named skip without a CUDA device (§LXXXII skip law). Registration happens
# in GessoCUDAExt.__init__ — loading CUDA (test_cuda_seam.jl) registered the
# two real candidates.

using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

using .GessoTestHelpers: approx_eq

const _AT_NAMES = (:cublas_mul, :generic_mul)

# regime shape tables — the (K, N) pairs each fixture's matmuls actually hit
# (toy2: dim=16, mlp hidden=64, vocab=32; micro: hidden=32, intermediate=64,
# kv-head dim=16, vocab=32 — see the fixtures' config data)
const _REGIME_PAIRS = Dict(
    :toy2 => [(16, 16), (16, 64), (64, 16), (16, 32)],
    :llama_micro => [(32, 32), (32, 16), (32, 64), (64, 32)],
)

if !CUDA_LOADED || !CUDA.functional()
    _skip(
        "no NVIDIA device (CUDA.functional() == false) — Autotune CUDA candidate tests skipped (§LXXXII skip law)",
    )
else
    cuda = Gesso.CUDABackend()
    wl = Gesso.PrefillWorkload()
    A = Gesso.Autotune

    @testset "autotune: candidates registered for (:matmul!, :cuda) in order" begin
        cands = A.candidates(:matmul!, :cuda)
        @test length(cands) == 2
        @test cands[1].name === :cublas_mul          # registration order
        @test cands[2].name === :generic_mul
    end

    for (regime, pairs) in _REGIME_PAIRS
        @testset "autotune: both candidates gated at :$regime (atol=1e-2)" begin
            for (K, N) in pairs
                M = 3
                x = randn(deterministic_rng(42), Float64, M, K)
                w = randn(deterministic_rng(43), Float64, N, K)
                ref = x * w'
                dst = mat(Gesso.Activation, CUDA.zeros(Float32, M, N))
                xt = mat(Gesso.Activation, CuArray{Float32}(x))
                wt = mat(Gesso.ProjectionWeight, CuArray{Float32}(w))
                for name in _AT_NAMES
                    c = only(filter(c -> c.name === name, A.candidates(:matmul!, :cuda)))
                    @test c.validate(dst, xt, wt) == true
                    fill!(dst.storage, 0)
                    c.run!(dst, xt, wt)
                    @test approx_eq(Array(dst.storage), Float32.(ref); atol=1e-2)
                end
            end
        end
    end

    @testset "autotune: search! selects a legal winner for :matmul!:cuda" begin
        M, K, N = 3, 16, 64
        x = randn(deterministic_rng(44), Float64, M, K)
        w = randn(deterministic_rng(45), Float64, N, K)
        dst = mat(Gesso.Activation, CUDA.zeros(Float32, M, N))
        xt = mat(Gesso.Activation, CuArray{Float32}(x))
        wt = mat(Gesso.ProjectionWeight, CuArray{Float32}(w))
        sink = Gesso.InMemorySink()
        result =
            A.search!(:matmul!, :cuda, :toy2, "cuda-device", dst, xt, wt; sink, samples=8)
        @test result.winner in _AT_NAMES
        @test isempty(result.rejected)
        @test haskey(result.medians, result.winner)
        r = only(sink.buf)
        @test r.context[:winner] === result.winner
        @test r.context[:op] === :matmul!
        @test r.context[:backend] === :cuda
    end

    @testset "autotune: a gate-failing candidate is rejected, never the winner" begin
        # registered under its OWN op key so the real pair's registration
        # order stays untouched; mixed with one honest candidate so the
        # rejection path AND the winner path are both exercised
        wrong = A.Candidate(
            :wrong_mul,
            (dst, x, w) -> (fill!(dst.storage, 1.0f9); dst),   # runs, wrong values
            (dst, x, w) -> false,                              # gate: disqualified
        )
        honest = A.Candidate(
            :honest_mul,
            (dst, x, w) -> (dst.storage.=x.storage * transpose(w.storage); dst),
            (dst, x, w) -> true,
        )
        A.register!(:wrong_op, :cuda, wrong)
        A.register!(:wrong_op, :cuda, honest)
        M, K, N = 3, 16, 16
        dst = mat(Gesso.Activation, CUDA.zeros(Float32, M, N))
        xt = mat(
            Gesso.Activation,
            CuArray{Float32}(randn(deterministic_rng(46), Float64, M, K)),
        )
        wt = mat(
            Gesso.ProjectionWeight,
            CuArray{Float32}(randn(deterministic_rng(47), Float64, N, K)),
        )
        result = A.search!(:wrong_op, :cuda, :toy2, "cuda-device", dst, xt, wt; samples=4)
        @test result.winner === :honest_mul                # the gated-out name can never win
        @test any(p -> p[1] === :wrong_mul, result.rejected)
        @test !haskey(result.medians, :wrong_mul)
    end

    @testset "autotune: all candidates gated out throws typed ERR_VERIFY_MISMATCH" begin
        bad = A.Candidate(:bad_mul, (dst, x, w) -> dst, (dst, x, w) -> false)
        A.register!(:all_bad_op, :cuda, bad)
        dst = mat(Gesso.Activation, CUDA.zeros(Float32, 2, 2))
        xt = mat(Gesso.Activation, CuArray{Float32}([1.0 2.0]))
        wt = mat(Gesso.ProjectionWeight, CuArray{Float32}([1.0 1.0]))
        err = try
            A.search!(:all_bad_op, :cuda, :toy2, "cuda-device", dst, xt, wt; samples=2)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError
        @test err.code == Gesso.ERR_VERIFY_MISMATCH
        @test occursin("failed the correctness gate", sprint(showerror, err))
    end

    @testset "quantize!/dequantize! still decline; CPU path unchanged" begin
        a = mat(Gesso.Activation, CuArray{Float32}([1.0 2.0]))
        p = mat(Gesso.ProjectionWeight, CuArray{Float32}([1.0 2.0]))
        @test_throws Gesso.LoweringNotImplemented Gesso.quantize!(cuda, p, a, wl)
        @test_throws Gesso.LoweringNotImplemented Gesso.dequantize!(cuda, a, p, wl)
        # CPU matmul! does not consult Autotune: same result with the
        # registry populated (x * transpose(w) = [4 5; 10 11])
        cpu_d = mat(Gesso.Activation, zeros(2, 2))
        Gesso.matmul!(
            Gesso.CPUBackend(),
            cpu_d,
            mat(Gesso.Activation, [1.0 2.0 3.0; 4.0 5.0 6.0]),
            mat(Gesso.ProjectionWeight, [1.0 0.0 1.0; 0.0 1.0 1.0]),
            wl,
        )
        @test cpu_d.storage == [4.0 5.0; 10.0 11.0]
    end

    # ---- Phase 9 item C: the op consults Autotune (§LXXXII exit) ------------

    psink = Gesso.default_receipt_sink()   # process sink: op consults emit here
    _last_at(regime) = findlast(
        r ->
            r.task === :autotune_select &&
            (regime === nothing || r.context[:regime] === regime),
        psink.buf,
    )

    @testset "op consults Autotune: cache entry on first matmul!, hit on second" begin
        K, N, M = 16, 64, 3                    # toy2 mlp-up shape → :toy2 regime
        dst = mat(Gesso.Activation, CUDA.zeros(Float32, M, N))
        xt = mat(
            Gesso.Activation,
            CuArray{Float32}(randn(deterministic_rng(48), Float64, M, K)),
        )
        wt = mat(
            Gesso.ProjectionWeight,
            CuArray{Float32}(randn(deterministic_rng(49), Float64, N, K)),
        )
        A.invalidate_all!()                    # only THIS call's miss in the table
        n_before = _last_at(nothing)
        Gesso.matmul!(cuda, dst, xt, wt, wl)
        i1 = _last_at(nothing)
        @test i1 !== nothing && i1 != n_before          # a select receipt appeared
        @test psink.buf[i1].context[:cache_hit] == false # first call = MISS
        @test psink.buf[i1].context[:regime] === :toy2   # (K,N)=(16,64) → toy2
        entry = A.cached_result(:matmul!, :cuda, :toy2; device=CUDA.name(CUDA.device()))
        @test entry !== nothing
        @test entry.winner in _AT_NAMES                 # winner is a registered name
        # second call with the same key: cache HIT — same winner, no re-search,
        # and NO second receipt (10G: a hit is not a decision; the miss receipt
        # already named the winner). The last :autotune_select receipt is
        # therefore still the MISS.
        Gesso.matmul!(cuda, dst, xt, wt, wl)
        i2 = _last_at(nothing)
        @test i2 == i1
        @test psink.buf[i1].context[:winner] === entry.winner
        # and the winner actually ran: dst holds the right product
        ref = Float64.(Array(xt.storage)) * Float64.(Array(wt.storage))'
        @test approx_eq(Array(dst.storage), Float32.(ref); atol=1e-2)
    end

    @testset "regimes: every declared (K, N) pair resolves and searches legally" begin
        for (regime, pairs) in _REGIME_PAIRS
            for (K, N) in pairs
                M = 2
                dst = mat(Gesso.Activation, CUDA.zeros(Float32, M, N))
                xt = mat(
                    Gesso.Activation,
                    CuArray{Float32}(randn(deterministic_rng(50), Float64, M, K)),
                )
                wt = mat(
                    Gesso.ProjectionWeight,
                    CuArray{Float32}(randn(deterministic_rng(51), Float64, N, K)),
                )
                r = A.search!(
                    :matmul!,
                    :cuda,
                    regime,
                    "cuda-device",
                    dst,
                    xt,
                    wt;
                    samples=4,
                )
                @test r.winner in _AT_NAMES
                @test isempty(r.rejected)
            end
        end
    end

    @testset "regression: host Array + CUDABackend still ERR_INVALID_PLAN (§LXX)" begin
        err = try
            Gesso.matmul!(
                cuda,
                mat(Gesso.Activation, zeros(2, 2)),
                mat(Gesso.Activation, [1.0 2.0]),
                mat(Gesso.ProjectionWeight, [1.0 1.0]),
                wl,
            )
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError
        @test err.code == Gesso.ERR_INVALID_PLAN
        @test occursin("to_device", sprint(showerror, err))
    end

    @testset "end-to-end: toy2 + llama_micro ids match CPU with the consult live" begin
        # toy2: greedy ids EXACT (argmax is the gate), logits inside the atol
        ts = toy2_tensors()
        cpu_ids = Gesso.reference_generate(toy2_modelir(), ts, PROMPT; max_new_tokens=8)
        gpu_ts = Gesso.to_device(cuda, ts)
        gpu_logits = Gesso.reference_prefill(toy2_modelir(), gpu_ts, PROMPT; backend=cuda)
        cpu_logits = Gesso.reference_prefill(toy2_modelir(), ts, PROMPT)
        got = Float64.(Array(gpu_logits))
        @test approx_eq(got, cpu_logits; atol=1e-3)
        println(
            "toy2 cuda-vs-cpu max|Δlogit| (autotuned) = ",
            maximum(abs, got .- cpu_logits),
        )
        gpu_ids = Gesso.reference_generate(
            toy2_modelir(),
            gpu_ts,
            PROMPT;
            backend=cuda,
            max_new_tokens=8,
        )
        @test gpu_ids == cpu_ids
        i_toy = _last_at(:toy2)
        @test i_toy !== nothing
        @test psink.buf[i_toy].context[:winner] in _AT_NAMES

        # llama_micro (GQA on device): same two gates, :llama_micro regime used
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
        gpu_tensors = Gesso.to_device(cuda, cpu_tensors)
        i_pre = _last_at(:llama_micro)                       # last micro receipt BEFORE this prefill
        gpu_out = Float64.(
            Array(
                Gesso.reference_prefill(
                    model,
                    gpu_tensors,
                    tokens;
                    backend=cuda,
                    eps=cfg.rms_norm_eps,
                    theta=cfg.rope_theta,
                ),
            ),
        )
        # FIRST micro receipt of this prefill = the (32,32) search → a MISS
        i_first_micro = findfirst(
            i ->
                i > (i_pre === nothing ? 0 : i_pre) &&
                psink.buf[i].task === :autotune_select &&
                psink.buf[i].context[:regime] === :llama_micro,
            1:length(psink.buf),
        )
        @test i_first_micro !== nothing
        @test psink.buf[i_first_micro].context[:cache_hit] == false  # first micro matmul searched
        @test approx_eq(gpu_out, cpu_out; atol=1e-3)
        println(
            "llama_micro cuda-vs-cpu max|Δlogit| (autotuned) = ",
            maximum(abs, gpu_out .- cpu_out),
        )
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
            backend=cuda,
            max_new_tokens=3,
            eps=cfg.rms_norm_eps,
            theta=cfg.rope_theta,
        )
        @test gpu_gen == cpu_gen
        i_micro = _last_at(:llama_micro)
        # 10G: only the MISS emits, so the last llama_micro receipt is still
        # the search — there is no per-consult hit receipt left to read.
        @test i_micro == i_first_micro
        @test psink.buf[i_micro].context[:cache_hit] == false
        @test A.cached_result(
            :matmul!,
            :cuda,
            :llama_micro;
            device=CUDA.name(CUDA.device()),
        ) !== nothing
    end
end
