# AMDGPU quant prefill GEMM: the shared element decoder matches the host
# dequant for every device quant type, the dequant + BLAS GEMM matches the F64
# reference with the unquantized activation, a scratch cap that forces row
# chunks gives the same result, and matmul! takes the GEMM from dq_gemm_min
# tokens on. Runs on KA's CPU backend; the weight generators come from
# test_amdgpu_quant.jl, test_amdgpu_iquant.jl and test_amdgpu_kquant.jl, which
# runtests.jl includes first.
# The ROCm path is exercised by the device parity run, not by this file.

using Random
using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

if !AMDGPU_LOADED || AMDGPU_EXT === nothing
    @test _skip("AMDGPU.jl or KernelAbstractions not loadable — AMDGPU quant GEMM tests skipped")
else
    const _AMDG_B = _AMDX.KADeviceBackend(_AMDKA.CPU())

    # (name, generator) for every type the device quant path places.
    function _amdg_cases()
        cases = Any[
            ("q8_0", (rng, n, k) -> _amdq_random_q8(rng, n, k), 32),
            ("q4_k", (rng, n, k) -> _amdq_random_q4k(rng, n, k), 256),
        ]
        for t in sort!(collect(keys(_AMDKQ_LAYOUT)))
            push!(cases, ("ggml_type_$t", (rng, n, k) -> _amdkq_random(rng, t, n, k), 256))
        end
        for (t, bv, bb) in _AMDIQ_LAYOUTS
            push!(cases, ("ggml_type_$t", (rng, n, k) -> _amdiq_random(rng, t, bv, bb, n, k), bv))
        end
        return cases
    end

    _amdg_dev(qm) = _AMDX.AmdDenseQuant(UInt8.(qm.data), qm.ggml_type, qm.ne0, qm.nrows, qm.row_bytes)

    function _amdg_check_gemm(qm, X::Matrix{Float32}, Y::Matrix{Float32}; label::AbstractString)
        W = Float64.(Gesso.Inference._quant_host_f32(qm))
        X64 = Float64.(X)
        ref = X64 * transpose(W)
        tol = 5.0e-5 .* (abs.(X64) * transpose(abs.(W))) .+ 1.0e-9
        err = abs.(Float64.(Y) .- ref)
        @info "amdgpu quant GEMM" label maxerr = maximum(err) worst_err_over_tol = maximum(err ./ tol)
        @test all(err .<= tol)
    end

    @testset "amdgpu quant GEMM: element decode matches the host dequant" begin
        rng = Random.MersenneTwister(20261116)
        for (name, gen, bv) in _amdg_cases()
            nrows, k = 5, 2 * max(bv, 256)
            qm = gen(rng, nrows, k)
            wtf = zeros(Float32, k * nrows)
            _AMDX._amdq_dequant_rows!(wtf, _amdg_dev(qm), 0, nrows)
            W = Gesso.Inference._quant_host_f32(qm)
            got = transpose(reshape(wtf, k, nrows))
            # Same formulas, possibly a different multiply order: a few ulp.
            ok = all(abs.(got .- W) .<= 4 .* eps.(abs.(W)) .+ 1.0f-30)
            ok || @info "amdgpu quant GEMM dequant mismatch" name maxerr = maximum(abs.(got .- W))
            @test ok
            # A row offset decodes the same rows.
            wt2 = zeros(Float32, k * 2)
            _AMDX._amdq_dequant_rows!(wt2, _amdg_dev(qm), 3, 2)
            @test wt2 == wtf[(3k+1):(5k)]
        end
    end

    @testset "amdgpu quant GEMM: dequant + BLAS vs F64 oracle (KA CPU backend)" begin
        rng = Random.MersenneTwister(20261117)
        for (name, gen, bv) in _amdg_cases()
            qm = gen(rng, 37, 512)
            X = randn(rng, Float32, 40, 512)
            Y = zeros(Float32, 40, 37)
            _AMDX._amdq_gemm_prefill!(Y, X, _amdg_dev(qm))
            _amdg_check_gemm(qm, X, Y; label="$name 37x512 ntok=40")
        end
    end

    @testset "amdgpu quant GEMM: row chunks match one pass" begin
        rng = Random.MersenneTwister(20261118)
        # 1 MiB of F32 scratch holds 10 rows of k=25600, so 37 rows take 4 chunks.
        qm = _amdq_random_q4k(rng, 37, 25600)
        X = randn(rng, Float32, 33, 25600)
        one_pass = zeros(Float32, 33, 37)
        _AMDX._amdq_gemm_prefill!(one_pass, X, _amdg_dev(qm))
        chunked = zeros(Float32, 33, 37)
        withenv("GESSO_TUNE" => "dq_gemm_scratch_mib=1") do
            _AMDX._amdq_gemm_prefill!(chunked, X, _amdg_dev(qm))
        end
        # BLAS rounding depends on the chunk width, so the two agree to the
        # oracle bound, not bit for bit.
        _amdg_check_gemm(qm, X, one_pass; label="q4_k 37x25600 one pass")
        _amdg_check_gemm(qm, X, chunked; label="q4_k 37x25600 chunked")
        @test isapprox(chunked, one_pass; rtol=1.0f-5)
    end

    @testset "amdgpu quant GEMM: matmul! dispatch by token count" begin
        rng = Random.MersenneTwister(20261119)
        qm = _amdq_random_q4k(rng, 37, 512)
        pw = Gesso.ProjectionWeight(size(qm), qm)
        emb = Gesso.EmbeddingTable((4, 2), ones(Float32, 4, 2))
        w = Gesso.to_device(_AMDG_B, (; embedding=emb, blocks=[(; w=pw)], lm_head=emb)).blocks[1].w
        for ntok in (31, 32)
            X = randn(rng, Float32, ntok, 512)
            dst = Gesso.Activation((ntok, 37), zeros(Float32, ntok, 37))
            Gesso.matmul!(_AMDG_B, dst, Gesso.Activation(size(X), X), w, Gesso.PrefillWorkload())
            gemm = zeros(Float32, ntok, 37)
            _AMDX._amdq_gemm_prefill!(gemm, X, w.storage)
            gemv = zeros(Float32, ntok, 37)
            _AMDX._amdq_gemv!(gemv, X, w.storage)
            @test dst.storage == (ntok >= 32 ? gemm : gemv)
        end
    end
end
