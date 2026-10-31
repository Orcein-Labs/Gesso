# AMDGPU dense K-quant GEMV (Q2_K, Q3_K, Q5_K, Q6_K): device weights stay
# quantized through to_device, the GEMV matches the CPU F64 oracle on KA's CPU
# backend with a Q8_K activation, invalid workgroup sizes are typed failures,
# and each dot kernel compiles and links for the AMD targets with no device.
# GPU-UNTESTED: nothing here runs on ROCm.

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
    @test _skip("AMDGPU.jl or KernelAbstractions not loadable — AMDGPU K-quant tests skipped")
else
    const _AMDKQ_B = _AMDX.KADeviceBackend(_AMDKA.CPU())
    const _AMDKQ_TOL_RTOL = 5.0e-5

    # The activation as the K-quant kernels see it: each 256-block quantized to
    # Q8_K by the host reference, dequantized in F64.
    function _amdkq_ref_x(X::AbstractMatrix{Float32})
        X64 = Float64.(X)
        for r in axes(X, 1)
            y = Gesso.Inference.quantize_q8_K!(Gesso.Inference.Q8KVec(size(X, 2) ÷ 256), X[r, :])
            X64[r, :] .= Float64.(repeat(y.d; inner=256)) .* Float64.(y.qs)
        end
        return X64
    end

    # Device path: to_device places the weight, matmul! runs the GEMV.
    function _amdkq_device_matmul(qm, X::Matrix{Float32})
        pw = Gesso.ProjectionWeight(size(qm), qm)
        emb = Gesso.EmbeddingTable((4, 2), ones(Float32, 4, 2))
        dev = Gesso.to_device(_AMDKQ_B, (; embedding=emb, blocks=[(; w=pw)], lm_head=emb))
        w = dev.blocks[1].w
        @test w.storage isa _AMDX.AmdDenseQuant
        n, nrows = size(X, 1), size(qm, 1)
        dst = Gesso.Activation((n, nrows), zeros(Float32, n, nrows))
        x = Gesso.Activation(size(X), X)
        Gesso.matmul!(_AMDKQ_B, dst, x, w, Gesso.DecodeWorkload())
        return dst.storage
    end

    # Per-element error against the F32-accumulation bound, F64 reference.
    function _amdkq_check(qm, X::Matrix{Float32}; label::AbstractString)
        dev = _amdkq_device_matmul(qm, X)
        W = Float64.(Gesso.Inference._quant_host_f32(qm))
        X64 = _amdkq_ref_x(X)
        ref = X64 * transpose(W)
        A = abs.(X64) * transpose(abs.(W))
        tol = _AMDKQ_TOL_RTOL .* A .+ 1.0e-9
        err = abs.(Float64.(dev) .- ref)
        @info "amdgpu K-quant GEMV" label maxerr = maximum(err) worst_err_over_tol = maximum(err ./ tol)
        @test all(err .<= tol)
        return nothing
    end

    # Superblock bytes and the offsets of the f16 scale fields inside one block.
    const _AMDKQ_LAYOUT = Dict(
        Gesso.Inference.GGML_TYPE_Q2_K => (84, (80, 82)),
        Gesso.Inference.GGML_TYPE_Q3_K => (110, (108,)),
        Gesso.Inference.GGML_TYPE_Q5_K => (176, (0, 2)),
        Gesso.Inference.GGML_TYPE_Q6_K => (210, (208,)),
    )

    # Random superblocks: every packed field is a valid bit pattern, the f16
    # scale fields are moderate and signed.
    function _amdkq_random(rng, t::Int32, nrows::Int, k::Int)
        sz, fp = _AMDKQ_LAYOUT[t]
        nsb = k ÷ 256
        rb = nsb * sz
        data = rand(rng, UInt8, nrows * rb)
        for r in 1:nrows, s in 1:nsb, off in fp
            base = (r - 1) * rb + (s - 1) * sz + off
            v = Float16((rand(rng) * 0.04 + 0.002) * (rand(rng) < 0.5 ? -1 : 1))
            data[(base+1):(base+2)] .= reinterpret(UInt8, [v])
        end
        return Gesso.Inference.QuantMatrix(data, 0, t, k, nrows, rb, Int64(256))
    end

    const _AMDKQ_MATMUL = Dict(
        Gesso.Inference.GGML_TYPE_Q2_K => _AMDX._amdq_q2k_matmul!,
        Gesso.Inference.GGML_TYPE_Q3_K => _AMDX._amdq_q3k_matmul!,
        Gesso.Inference.GGML_TYPE_Q5_K => _AMDX._amdq_q5k_matmul!,
        Gesso.Inference.GGML_TYPE_Q6_K => _AMDX._amdq_q6k_matmul!,
    )

    const _AMDKQ_NAME = Dict(
        Gesso.Inference.GGML_TYPE_Q2_K => "q2_k",
        Gesso.Inference.GGML_TYPE_Q3_K => "q3_k",
        Gesso.Inference.GGML_TYPE_Q5_K => "q5_k",
        Gesso.Inference.GGML_TYPE_Q6_K => "q6_k",
    )

    @testset "amdgpu quant K-quants: device GEMV vs CPU F64 oracle (KA CPU backend)" begin
        rng = Random.MersenneTwister(20261033)
        for t in keys(_AMDKQ_LAYOUT)
            name = _AMDKQ_NAME[t]
            for (nrows, k, ntok) in ((1, 256, 1), (37, 256, 1), (37, 256, 3), (37, 512, 1), (37, 768, 3))
                qm = _amdkq_random(rng, t, nrows, k)
                X = randn(rng, Float32, ntok, k)
                _amdkq_check(qm, X; label="$name $(nrows)x$k ntok=$ntok")
            end
            qm = _amdkq_random(rng, t, 37, 512)
            X = randn(rng, Float32, 40, 512)
            _amdkq_check(qm, X; label="$name 37x512 ntok=40 (not a multiple of wg)")
        end
    end

    @testset "amdgpu quant K-quants: invalid workgroup sizes are ERR_INVALID_PLAN" begin
        rng = Random.MersenneTwister(9)
        for t in keys(_AMDKQ_LAYOUT)
            qm = _amdkq_random(rng, t, 4, 256)
            X = randn(rng, Float32, 2, 256)
            Y = zeros(Float32, 2, 4)
            dev = _AMDX.AmdDenseQuant(qm.data, qm.ggml_type, qm.ne0, qm.nrows, qm.row_bytes)
            for wg in (0, 48, 512)
                err = try
                    _AMDKQ_MATMUL[t](Y, X, dev; wg)
                    nothing
                catch e
                    e
                end
                @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
            end
        end
    end

    @testset "amdgpu quant K-quants: unsupported ggml type is ERR_INVALID_PLAN" begin
        dev = _AMDX.AmdDenseQuant(zeros(UInt8, 36), Gesso.Inference.GGML_TYPE_Q4_0, 64, 2, 36)
        X = zeros(Float32, 1, 64)
        Y = zeros(Float32, 1, 2)
        err = try
            _AMDX._amdq_gemv!(Y, X, dev)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
    end

    @testset "amdgpu quant K-quants: to_device keeps Q2_K..Q6_K quantized" begin
        rng = Random.MersenneTwister(12)
        emb = Gesso.EmbeddingTable((4, 2), ones(Float32, 4, 2))
        for t in keys(_AMDKQ_LAYOUT)
            qm = _amdkq_random(rng, t, 4, 256)
            pw = Gesso.ProjectionWeight(size(qm), qm)
            dev = Gesso.to_device(_AMDKQ_B, (; embedding=emb, blocks=[(; w=pw)], lm_head=emb))
            @test dev.blocks[1].w.storage isa _AMDX.AmdDenseQuant
            @test Array(dev.blocks[1].w.storage.data) == qm.data
            @test dev.blocks[1].w.storage.ggml_type == t
        end
    end

    @testset "amdgpu quant K-quants: compile and link for $(join(first.(AMD_TARGETS), ", "))" begin
        G = amd_global
        dots = (
            _AMDX._amdq_q2k_dot_kernel!,
            _AMDX._amdq_q3k_dot_kernel!,
            _AMDX._amdq_q5k_dot_kernel!,
            _AMDX._amdq_q6k_dot_kernel!,
        )
        for wg in (64, 256), kern in dots
            res = amd_compile(
                kern,
                wg,
                (wg * 3,),
                (G(Float32, 2), G(Int32, 2), G(Float32, 2), G(UInt8, 1), Int, Int, Int, Int),
            )
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)
        end
    end
end
