# AMDGPU quant kernels on a real ROCm device: every quant GEMV agrees with the
# same kernel on KA's CPU backend, and the prefill GEMM agrees with the F64
# reference, at the row widths of the real models (Gemma4 2816/704, Qwen
# 5120-class) as well as small ones. The other quant test files run on KA CPU
# only, so without this file device numerics were checked only by model parity.
# Skips without a functional device. The weight generators come from
# test_amdgpu_quant.jl, test_amdgpu_iquant.jl and test_amdgpu_kquant.jl.

using Random
using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

if !AMDGPU_LOADED || AMDGPU_EXT === nothing || !Main.AMDGPU.functional()
    @test _skip("no functional ROCm device — AMDGPU device quant tests skipped")
else
    @testset "amdgpu quant kernels on device vs KA CPU" begin
        X_ = AMDGPU_EXT
        dev(a) = Main.AMDGPU.ROCArray(a)
        cases = Any[
            ("q8_0", (r, n, k) -> _amdq_random_q8(r, n, k), 32),
            ("q4_k", (r, n, k) -> _amdq_random_q4k(r, n, k), 256),
        ]
        for t in sort!(collect(keys(_AMDKQ_LAYOUT)))
            push!(cases, ("ggml_type_$t", (r, n, k) -> _amdkq_random(r, t, n, k), 256))
        end
        for (t, bv, bb) in _AMDIQ_LAYOUTS
            push!(cases, ("ggml_type_$t", (r, n, k) -> _amdiq_random(r, t, bv, bb, n, k), bv))
        end
        rng = MersenneTwister(20261116)
        for (name, gen, bv) in cases, k in (256, 704, 2816, 5376), m in (1, 3, 40)
            k % bv == 0 || continue
            qm = gen(rng, 19, k)
            X = randn(rng, Float32, m, k)
            host = X_.AmdDenseQuant(UInt8.(qm.data), qm.ggml_type, qm.ne0, qm.nrows, qm.row_bytes)
            dq = X_.AmdDenseQuant(dev(UInt8.(qm.data)), qm.ggml_type, qm.ne0, qm.nrows, qm.row_bytes)
            Yc = zeros(Float32, m, 19)
            X_._amdq_gemv!(Yc, X, host)
            Yd = dev(zeros(Float32, m, 19))
            X_._amdq_gemv!(Yd, dev(X), dq)
            Yd = Array(Yd)
            Wf = Gesso.Inference._quant_host_f32(qm)
            A = abs.(X) * transpose(abs.(Wf))
            rel = maximum(abs.(Yd .- Yc) ./ (A .+ 1.0f-9))
            rel <= 1.0e-5 || @info "amdgpu device GEMV mismatch" name k m rel
            @test rel <= 1.0e-5
            if m == 40 && X_._lq_kind(qm.ggml_type) !== nothing
                Gd = dev(zeros(Float32, m, 19))
                X_._amdq_gemm_prefill!(Gd, dev(X), dq)
                ref = Float64.(X) * transpose(Float64.(Wf))
                relg = maximum(abs.(Float64.(Array(Gd)) .- ref) ./ (Float64.(A) .+ 1.0e-9))
                relg <= 1.0e-5 || @info "amdgpu device GEMM mismatch" name k relg
                @test relg <= 1.0e-5
            end
        end
    end
end
