# AMDGPU dense quant GEMV: a supported GGUF projection stays quantized through
# to_device, the GEMV matches the CPU F64 oracle on KA's CPU backend, invalid
# workgroup sizes are typed failures, and each kernel compiles and links for
# the AMD targets with no device. GPU-UNTESTED: nothing here runs on ROCm.

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
    @test _skip("AMDGPU.jl or KernelAbstractions not loadable — AMDGPU quant tests skipped")
else
    const _AMDQ_B = _AMDX.KADeviceBackend(_AMDKA.CPU())
    const _AMDQ_TOL_RTOL = 5.0e-5

    # Random Q8_0 weight: f16 block scales in +-[0.002, 0.042], random int8 codes.
    function _amdq_random_q8(rng, nrows::Int, k::Int)
        nblk = k ÷ 32
        rb = nblk * 34
        data = rand(rng, UInt8, nrows * rb)
        for r in 1:nrows, j in 1:nblk
            base = (r - 1) * rb + (j - 1) * 34
            v = Float16((rand(rng) * 0.04 + 0.002) * (rand(rng) < 0.5 ? -1 : 1))
            data[(base+1):(base+2)] .= reinterpret(UInt8, [v])
        end
        return Gesso.Inference.QuantMatrix(
            data, 0, Gesso.Inference.GGML_TYPE_Q8_0, k, nrows, rb, Int64(32),
        )
    end

    # The activation as the Q8_0 kernel sees it: each 32-block quantized to
    # int8 with its scale, dequantized in F64.
    function _amdq_ref_x(X::AbstractMatrix{Float32})
        X64 = Float64.(X)
        for r in axes(X, 1)
            y = Gesso.Inference.quantize_q8_0!(Gesso.Inference.Q80Vec(size(X, 2) ÷ 32), X[r, :])
            X64[r, :] .= Float64.(repeat(y.d; inner=32)) .* y.qs
        end
        return X64
    end

    # Random Q4_K weight: f16 d and dmin in a moderate range, random scale, min
    # and nibble bytes (every 6-bit field is a valid value).
    function _amdq_random_q4k(rng, nrows::Int, k::Int)
        nsb = k ÷ 256
        rb = nsb * 144
        data = rand(rng, UInt8, nrows * rb)
        for r in 1:nrows, s in 1:nsb
            base = (r - 1) * rb + (s - 1) * 144
            d = Float16((rand(rng) * 0.04 + 0.002) * (rand(rng) < 0.5 ? -1 : 1))
            dm = Float16((rand(rng) * 0.04 + 0.002) * (rand(rng) < 0.5 ? -1 : 1))
            data[(base+1):(base+2)] .= reinterpret(UInt8, [d])
            data[(base+3):(base+4)] .= reinterpret(UInt8, [dm])
        end
        return Gesso.Inference.QuantMatrix(
            data, 0, Gesso.Inference.GGML_TYPE_Q4_K, k, nrows, rb, Int64(256),
        )
    end

    # The activation as the Q4_K kernel sees it: each 256-block quantized to Q8_K
    # by the host reference, dequantized in F64.
    function _amdq_ref_xk(X::AbstractMatrix{Float32})
        X64 = Float64.(X)
        for r in axes(X, 1)
            y = Gesso.Inference.quantize_q8_K!(Gesso.Inference.Q8KVec(size(X, 2) ÷ 256), X[r, :])
            X64[r, :] .= Float64.(repeat(y.d; inner=256)) .* Float64.(y.qs)
        end
        return X64
    end

    # Device path: to_device places the weight, matmul! runs the GEMV.
    function _amdq_device_matmul(qm, X::Matrix{Float32})
        pw = Gesso.ProjectionWeight(size(qm), qm)
        emb = Gesso.EmbeddingTable((4, 2), ones(Float32, 4, 2))
        dev = Gesso.to_device(_AMDQ_B, (; embedding=emb, blocks=[(; w=pw)], lm_head=emb))
        w = dev.blocks[1].w
        @test w.storage isa _AMDX.AmdDenseQuant
        n, nrows = size(X, 1), size(qm, 1)
        dst = Gesso.Activation((n, nrows), zeros(Float32, n, nrows))
        x = Gesso.Activation(size(X), X)
        Gesso.matmul!(_AMDQ_B, dst, x, w, Gesso.DecodeWorkload())
        return dst.storage
    end

    # Per-element error against the F32-accumulation bound, F64 reference.
    function _amdq_check(qm, X::Matrix{Float32}; label::AbstractString, refx=_amdq_ref_x)
        dev = _amdq_device_matmul(qm, X)
        W = Float64.(Gesso.Inference._quant_host_f32(qm))
        X64 = refx(X)
        ref = X64 * transpose(W)
        A = abs.(X64) * transpose(abs.(W))
        tol = _AMDQ_TOL_RTOL .* A .+ 1.0e-9
        err = abs.(Float64.(dev) .- ref)
        @info "amdgpu dense quant GEMV" label maxerr = maximum(err) worst_err_over_tol = maximum(err ./ tol)
        @test all(err .<= tol)
        return nothing
    end

    @testset "amdgpu quant Q8_0: device GEMV vs CPU F64 oracle (KA CPU backend)" begin
        rng = Random.MersenneTwister(20261031)
        for (nrows, k, ntok) in ((1, 32, 1), (37, 32, 1), (37, 32, 3), (37, 96, 1), (37, 96, 3), (37, 512, 1))
            qm = _amdq_random_q8(rng, nrows, k)
            X = randn(rng, Float32, ntok, k)
            _amdq_check(qm, X; label="q8_0 $(nrows)x$k ntok=$ntok")
        end
        qm = _amdq_random_q8(rng, 37, 512)
        X = randn(rng, Float32, 40, 512)
        _amdq_check(qm, X; label="q8_0 37x512 ntok=40 (not a multiple of wg)")
    end

    @testset "amdgpu quant Q4_K: device GEMV vs CPU F64 oracle (KA CPU backend)" begin
        rng = Random.MersenneTwister(20261032)
        for (nrows, k, ntok) in ((1, 256, 1), (37, 256, 1), (37, 256, 3), (37, 512, 1), (37, 512, 3))
            qm = _amdq_random_q4k(rng, nrows, k)
            X = randn(rng, Float32, ntok, k)
            _amdq_check(qm, X; label="q4_k $(nrows)x$k ntok=$ntok", refx=_amdq_ref_xk)
        end
        qm = _amdq_random_q4k(rng, 37, 512)
        X = randn(rng, Float32, 40, 512)
        _amdq_check(qm, X; label="q4_k 37x512 ntok=40 (not a multiple of wg)", refx=_amdq_ref_xk)
    end

    @testset "amdgpu quant: invalid workgroup sizes are ERR_INVALID_PLAN" begin
        rng = Random.MersenneTwister(7)
        qm = _amdq_random_q8(rng, 4, 64)
        X = randn(rng, Float32, 2, 64)
        Y = zeros(Float32, 2, 4)
        for wg in (0, 48, 512)
            err = try
                _AMDX._amdq_q8_matmul!(Y, X, _AMDX.AmdDenseQuant(
                    UInt8.(qm.data), qm.ggml_type, qm.ne0, qm.nrows, qm.row_bytes,
                ); wg)
                nothing
            catch e
                e
            end
            @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
        end
    end

    @testset "amdgpu quant Q4_K: invalid workgroup sizes are ERR_INVALID_PLAN" begin
        rng = Random.MersenneTwister(8)
        qm = _amdq_random_q4k(rng, 4, 256)
        X = randn(rng, Float32, 2, 256)
        Y = zeros(Float32, 2, 4)
        for wg in (0, 48, 512)
            err = try
                _AMDX._amdq_q4k_matmul!(Y, X, _AMDX.AmdDenseQuant(
                    qm.data, qm.ggml_type, qm.ne0, qm.nrows, qm.row_bytes,
                ); wg)
                nothing
            catch e
                e
            end
            @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
        end
    end

    @testset "amdgpu quant: to_device keeps supported formats quantized" begin
        rng = Random.MersenneTwister(11)
        qm = _amdq_random_q8(rng, 4, 64)
        q4 = _amdq_random_q4k(rng, 4, 256)
        emb = Gesso.EmbeddingTable((4, 2), ones(Float32, 4, 2))
        q2 = Gesso.Inference.QuantMatrix(
            rand(rng, UInt8, 2 * 36), 0, Gesso.Inference.GGML_TYPE_Q4_0, 64, 2, 36, Int64(32),
        )
        blk = (
            wq=Gesso.ProjectionWeight(size(qm), qm),
            wv=Gesso.ProjectionWeight(size(q4), q4),
            wk=Gesso.ProjectionWeight(size(q2), q2),
        )
        dev = Gesso.to_device(_AMDQ_B, (; embedding=emb, blocks=[blk], lm_head=emb))
        @test dev.blocks[1].wq.storage isa _AMDX.AmdDenseQuant
        @test Array(dev.blocks[1].wq.storage.data) == qm.data
        @test dev.blocks[1].wq.storage.ggml_type == Gesso.Inference.GGML_TYPE_Q8_0
        @test dev.blocks[1].wv.storage isa _AMDX.AmdDenseQuant
        @test Array(dev.blocks[1].wv.storage.data) == q4.data
        @test dev.blocks[1].wv.storage.ggml_type == Gesso.Inference.GGML_TYPE_Q4_K
        @test dev.blocks[1].wk.storage isa Array{Float32}
        @test dev.blocks[1].wk.storage == Gesso.Inference._quant_host_f32(q2)
        @test dev.embedding.storage isa Array{Float32}
    end

    @testset "amdgpu quant: compile and link for $(join(first.(AMD_TARGETS), ", "))" begin
        G = amd_global
        for wg in (64, 256)
            res = amd_compile(
                _AMDX._amdq_q8_act_pack_kernel!,
                wg,
                (wg * 3,),
                (G(Int32, 2), G(Float32, 2), G(Float32, 2), Int, Int, Int),
            )
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)

            res = amd_compile(
                _AMDX._amdq_q8_dot_kernel!,
                wg,
                (wg * 3,),
                (G(Float32, 2), G(Int32, 2), G(Float32, 2), G(UInt8, 1), Int, Int, Int, Int),
            )
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)

            res = amd_compile(
                _AMDX._amdq_q8k_act_pack_kernel!,
                wg,
                (wg * 3,),
                (G(Int32, 2), G(Float32, 2), G(Float32, 2), Int, Int, Int),
            )
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)

            res = amd_compile(
                _AMDX._amdq_q4k_dot_kernel!,
                wg,
                (wg * 3,),
                (G(Float32, 2), G(Int32, 2), G(Float32, 2), G(UInt8, 1), Int, Int, Int, Int),
            )
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)
        end
    end
end
