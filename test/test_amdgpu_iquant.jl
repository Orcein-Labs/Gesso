# AMDGPU dense IQ GEMV (IQ4_NL, IQ4_XS, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
# IQ3_S): device parity against the F64 oracle on KA's CPU backend with an
# unquantized Float32 activation, invalid workgroup sizes and unsupported types
# are typed failures, and each kernel compiles and links for the AMD targets
# with no device. GPU-UNTESTED: nothing here runs on ROCm.

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
    @test _skip("AMDGPU.jl or KernelAbstractions not loadable — AMDGPU IQ tests skipped")
else
    const _AMDIQ_B = _AMDX.KADeviceBackend(_AMDKA.CPU())
    const _AMDIQ_TOL_RTOL = 5.0e-5

    # Block layouts: (ggml type, values per block, bytes per block). Every byte
    # pattern is a valid IQ block because the grid indices cover their tables.
    const _AMDIQ_LAYOUTS = (
        (Gesso.Inference.GGML_TYPE_IQ4_NL, 32, 18),
        (Gesso.Inference.GGML_TYPE_IQ4_XS, 256, 136),
        (Gesso.Inference.GGML_TYPE_IQ2_XXS, 256, 66),
        (Gesso.Inference.GGML_TYPE_IQ2_XS, 256, 74),
        (Gesso.Inference.GGML_TYPE_IQ2_S, 256, 82),
        (Gesso.Inference.GGML_TYPE_IQ3_XXS, 256, 98),
        (Gesso.Inference.GGML_TYPE_IQ3_S, 256, 110),
    )

    # Random IQ weight: random block bytes with an f16 block scale at offset 0 in
    # +-[0.002, 0.042], the same scale range the Q8_0 and Q4_K tests use.
    function _amdiq_random(rng, t::Int32, bv::Int, bb::Int, nrows::Int, k::Int)
        nblk = k ÷ bv
        rb = nblk * bb
        data = rand(rng, UInt8, nrows * rb)
        for r in 1:nrows, j in 1:nblk
            base = (r - 1) * rb + (j - 1) * bb
            v = Float16((rand(rng) * 0.04 + 0.002) * (rand(rng) < 0.5 ? -1 : 1))
            data[(base+1):(base+2)] .= reinterpret(UInt8, [v])
        end
        return Gesso.Inference.QuantMatrix(data, 0, t, k, nrows, rb, Int64(bv))
    end

    # Device path: to_device places the weight, matmul! runs the GEMV.
    function _amdiq_device_matmul(qm, X::Matrix{Float32})
        pw = Gesso.ProjectionWeight(size(qm), qm)
        emb = Gesso.EmbeddingTable((4, 2), ones(Float32, 4, 2))
        dev = Gesso.to_device(_AMDIQ_B, (; embedding=emb, blocks=[(; w=pw)], lm_head=emb))
        w = dev.blocks[1].w
        @test w.storage isa _AMDX.AmdDenseQuant
        n, nrows = size(X, 1), size(qm, 1)
        dst = Gesso.Activation((n, nrows), zeros(Float32, n, nrows))
        x = Gesso.Activation(size(X), X)
        Gesso.matmul!(_AMDIQ_B, dst, x, w, Gesso.DecodeWorkload())
        return dst.storage
    end

    # Per-element error against the F64 reference with the unquantized activation,
    # which is the oracle the Lava IQ test uses for these formats.
    function _amdiq_check(qm, X::Matrix{Float32}; label::AbstractString)
        dev = _amdiq_device_matmul(qm, X)
        W = Float64.(Gesso.Inference._quant_host_f32(qm))
        X64 = Float64.(X)
        ref = X64 * transpose(W)
        A = abs.(X64) * transpose(abs.(W))
        tol = _AMDIQ_TOL_RTOL .* A .+ 1.0e-9
        err = abs.(Float64.(dev) .- ref)
        @info "amdgpu IQ GEMV" label maxerr = maximum(err) worst_err_over_tol = maximum(err ./ tol)
        @test all(err .<= tol)
        return nothing
    end

    @testset "amdgpu IQ: device GEMV vs CPU F64 oracle (KA CPU backend)" begin
        rng = Random.MersenneTwister(20261101)
        for (t, bv, bb) in _AMDIQ_LAYOUTS
            name = "ggml_type_$t"
            cases = bv == 32 ?
                ((1, 32, 1), (37, 32, 1), (37, 96, 3), (37, 512, 1)) :
                ((1, 256, 1), (37, 256, 1), (37, 256, 3), (37, 512, 1), (37, 768, 3))
            for (nrows, k, ntok) in cases
                qm = _amdiq_random(rng, t, bv, bb, nrows, k)
                X = randn(rng, Float32, ntok, k)
                _amdiq_check(qm, X; label="$name $(nrows)x$k ntok=$ntok")
            end
            qm = _amdiq_random(rng, t, bv, bb, 37, 512)
            X = randn(rng, Float32, 40, 512)
            _amdiq_check(qm, X; label="$name 37x512 ntok=40 (not a multiple of wg)")
        end
    end

    @testset "amdgpu IQ: invalid workgroup sizes are ERR_INVALID_PLAN" begin
        rng = Random.MersenneTwister(9)
        for (launcher, t, bv, bb) in (
            (_AMDX._amdq_iq4nl_matmul!, Gesso.Inference.GGML_TYPE_IQ4_NL, 32, 18),
            (_AMDX._amdq_iq_matmul!, Gesso.Inference.GGML_TYPE_IQ4_XS, 256, 136),
        )
            qm = _amdiq_random(rng, t, bv, bb, 4, 256)
            X = randn(rng, Float32, 2, 256)
            Y = zeros(Float32, 2, 4)
            for wg in (0, 48, 512)
                err = try
                    launcher(Y, X, _AMDX.AmdDenseQuant(
                        qm.data, qm.ggml_type, qm.ne0, qm.nrows, qm.row_bytes,
                    ); wg)
                    nothing
                catch e
                    e
                end
                @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
            end
        end
    end

    @testset "amdgpu IQ: codebooks upload once per backend" begin
        ka = _AMDKA.CPU()
        a = _AMDX._amdq_iq_device_tables(ka)
        b = _AMDX._amdq_iq_device_tables(ka)
        @test a.g2 === b.g2 && a.g3 === b.g3 && a.ks === b.ks && a.kv === b.kv
    end

    @testset "amdgpu IQ: a type with no IQ kernel is ERR_INVALID_PLAN" begin
        rng = Random.MersenneTwister(10)
        f16 = Gesso.Inference.QuantMatrix(
            rand(rng, UInt8, 4 * 512), 0, Gesso.Inference.GGML_TYPE_F16, 256, 4, 512, Int64(1),
        )
        X = randn(rng, Float32, 2, 256)
        Y = zeros(Float32, 2, 4)
        err = try
            _AMDX._amdq_gemv!(Y, X, _AMDX.AmdDenseQuant(
                f16.data, f16.ggml_type, f16.ne0, f16.nrows, f16.row_bytes,
            ))
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
    end

    @testset "amdgpu IQ: compile and link for $(join(first.(AMD_TARGETS), ", "))" begin
        G = amd_global
        for wg in (64, 256)
            res = amd_compile(
                _AMDX._amdq_iq4nl_dot_kernel!,
                wg,
                (wg * 3,),
                (G(Float32, 2), G(Float32, 2), G(UInt8, 1), Int, Int, Int, Int, G(Int32, 1)),
            )
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)

            res = amd_compile(
                _AMDX._amdq_iq_dot_kernel!,
                wg,
                (wg * 3,),
                (
                    G(Float32, 2), G(Float32, 2), G(UInt8, 1),
                    Int, Int, Int, Int, Int,
                    G(Int32, 1), G(UInt32, 1), G(UInt32, 1), G(UInt8, 1),
                ),
            )
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)
        end
    end
end
