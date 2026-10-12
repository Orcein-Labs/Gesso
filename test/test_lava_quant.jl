# Lava dense quant GEMV (LavaDenseQuant): device placement and the in-kernel
# dequant matmul against the host dequant reference. Named skip without a
# usable Vulkan device. Mirrors test_cuda_quant_dense.jl; the CUDA side also
# quantizes the activation to Q8_1, so its bound has an extra term. The Lava
# port keeps the activation in F32, so that term is absent here.
#
# Tolerance. The weight side is exact: the kernel and `_quant_host_f32` decode
# the same bytes. With F32 activations the only error left is F32 accumulation,
# bounded by 5e-5 times the absolute sum sum_i |W[r,i] x[t,i]| for k <= 2048.
# A value passes when |dev - ref| <= 5e-5 * A + 1e-9 against the F64 reference.

using Test: AbstractTestSet, Broken, get_testset, record
using Random: MersenneTwister

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

if !LAVA_LOADED || !VULKAN_OK
    _skip("no usable Vulkan device — Lava dense quant GEMV tests skipped (§LXXXI skip law)")
else
    _lq_ext = Base.get_extension(Gesso, :GessoLavaExt)
    _lq_lava = Gesso.LavaBackend()
    _lq_inf = Gesso.Inference
    _lq_wl = Gesso.PrefillWorkload()

    # GGUF block layouts: (block bytes, values per block, byte offsets of the
    # f16 scale fields inside a block).
    const _LQ_LAYOUT = Dict(
        :q4_k => (144, 256, (0, 2)),
        :q5_k => (176, 256, (0, 2)),
        :q6_k => (210, 256, (208,)),
        :iq4_xs => (136, 256, (0,)),
        :q8_0 => (34, 32, (0,)),
    )
    const _LQ_TYPE = Dict(
        :q4_k => _lq_inf.GGML_TYPE_Q4_K,
        :q5_k => _lq_inf.GGML_TYPE_Q5_K,
        :q6_k => _lq_inf.GGML_TYPE_Q6_K,
        :iq4_xs => _lq_inf.GGML_TYPE_IQ4_XS,
        :q8_0 => _lq_inf.GGML_TYPE_Q8_0,
    )

    # Random block bytes with sane f16 scale fields (random bytes elsewhere).
    function _lq_random_matrix(rng, kind::Symbol, nrows::Int, k::Int)
        bbytes, bvals, scale_offs = _LQ_LAYOUT[kind]
        nblk = div(k, bvals)
        rb = nblk * bbytes
        data = rand(rng, UInt8, nrows * rb)
        for r in 1:nrows, j in 1:nblk
            base = (r - 1) * rb + (j - 1) * bbytes
            for off in scale_offs
                v = Float16((rand(rng) * 0.04 + 0.002) * (rand(rng) < 0.5 ? -1 : 1))
                data[base + off + 1:base + off + 2] .= reinterpret(UInt8, [v])
            end
        end
        return _lq_inf.QuantMatrix(data, 0, _LQ_TYPE[kind], k, nrows, rb, C_NULL, Int64(bvals))
    end

    # Device path: to_device places the weight; matmul! runs the GEMV.
    function _lq_device_matmul(qm, X)
        pw = Gesso.ProjectionWeight(; shape=size(qm), storage=qm)
        emb = Gesso.EmbeddingTable(; shape=(4, 2), storage=ones(Float32, 4, 2))
        dev = Gesso.to_device(_lq_lava, (; embedding=emb, blocks=[(; w=pw)], lm_head=emb))
        w = dev.blocks[1].w
        @test w.storage isa _lq_ext.LavaDenseQuant
        n, nrows = size(X, 1), size(qm, 1)
        dst = Gesso.Activation(; shape=(n, nrows), storage=fill!(Lava.LavaArray{Float32}(undef, n, nrows), 0.0f0))
        x = Gesso.Activation(; shape=size(X), storage=Lava.LavaArray{Float32}(X))
        Gesso.matmul!(_lq_lava, dst, x, w, _lq_wl)
        return Array(dst.storage)
    end

    # Per-element error against the F32-activation bound, host F64 accumulation.
    function _lq_check(qm, X, dev; label::AbstractString)
        if !isfile(joinpath(get(ENV, "GESSO_GGML_DIR", ""), "libggml-base.so"))
            _skip("set GESSO_GGML_DIR to a directory with libggml-base.so — $label host reference skipped")
            return nothing
        end
        W = Float64.(_lq_inf._quant_host_f32(qm))
        X64 = Float64.(X)
        ref = X64 * transpose(W)
        A = abs.(X64) * transpose(abs.(W))
        tol = 5.0e-5 .* A .+ 1.0e-9
        err = abs.(Float64.(dev) .- ref)
        ratio = maximum(err ./ tol)
        @info "lava dense quant GEMV" label maxerr = maximum(err) max_ref = maximum(abs, ref) worst_err_over_tol = ratio
        @test all(err .<= tol)
        return nothing
    end

    @testset "LavaDenseQuant: random blocks, device GEMV vs host dequant" begin
        rng = MersenneTwister(20261011)
        for kind in (:q4_k, :q5_k, :q6_k, :iq4_xs, :q8_0)
            @testset "$kind" begin
                k = 512
                nrows = 37
                ntok = 3
                qm = _lq_random_matrix(rng, kind, nrows, k)
                X = randn(rng, Float32, ntok, k)
                _lq_check(qm, X, _lq_device_matmul(qm, X); label=string(kind))
            end
        end
    end

    @testset "LavaDenseQuant: unsupported and misaligned weights stay F32" begin
        rng = MersenneTwister(7)
        qm = _lq_random_matrix(rng, :q4_k, 8, 256)
        @test _lq_ext._lq_eligible(qm)
        bad = _lq_inf.QuantMatrix(qm.data, 0, _lq_inf.GGML_TYPE_Q4_K, 100, 8, qm.row_bytes, C_NULL, Int64(256))
        @test !_lq_ext._lq_eligible(bad)
    end
end
