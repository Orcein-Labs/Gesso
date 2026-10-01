# Phase 9 item B tests (§LXXXII): the two CUDA `matmul!` candidates —
# :cublas_mul and :generic_mul — both gated against the CPU F64 oracle at the
# EXISTING Phase 4 CUDA op atol (1e-2; no third atol invented), on toy2-sized
# and llama_micro-sized pairs. A deliberately-wrong candidate is rejected
# (proven under its own op key so the real pair's registration order stays
# untouched). quantize!/dequantize! still decline; the CPU matmul! path is
# untouched by Autotune (no consult).
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
end
