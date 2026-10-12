# CUDA dense quant GEMV (GpuDenseQuant): device placement and the decode
# matmul against the host dequant reference. Named skip without a device.
#
# Tolerance (derived, not fitted). The weight side is exact: the kernel and
# `_quant_host_f32` decode the same bytes. The only approximation is the Q8_1
# activation quantization: each activation element moves by at most d/2 with
# d = amax/127 over its 32-value block, so |err| <= sum_b (amax_b/254) *
# sum_{i in b} |W[r,i]|. The F32 accumulation adds a slack of 5e-5 times the
# absolute sum sum_i |W[r,i] x[t,i]|, which bounds the summation error for
# k <= 2048. A value passes when |dev - ref| <= bound + slack.

using Test: AbstractTestSet, Broken, get_testset, record
using Random: MersenneTwister

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

if !CUDA_LOADED || !CUDA.functional()
    _skip("no NVIDIA device — CUDA dense quant GEMV tests skipped (§LXXVII skip law)")
elseif !isfile(joinpath(get(ENV, "GESSO_GGML_DIR", ""), "libggml-base.so"))
    # Every set here needs ggml's to_float: the host reference directly, and the
    # placement fallback through the F32 dequant in to_device.
    _skip("set GESSO_GGML_DIR to a directory with libggml-base.so — CUDA dense quant GEMV tests skipped")
else
    _dq_ext = Base.get_extension(Gesso, :GessoCUDAExt)
    cuda = Gesso.CUDABackend()
    Inf_ = Gesso.Inference
    _mat(F, arr) = F(; shape=size(arr), storage=arr)

    # GGUF block layouts: (block bytes, values per block, byte offsets of the
    # f16 scale fields inside a block).
    const _DQ_LAYOUT = Dict(
        :q4_k => (144, 256, (0, 2)),
        :q5_k => (176, 256, (0, 2)),
        :q6_k => (210, 256, (208,)),
        :iq4_xs => (136, 256, (0,)),
        :iq2_xs => (74, 256, (0,)),
        :iq3_xxs => (98, 256, (0,)),
        :q8_0 => (34, 32, (0,)),
    )
    const _DQ_TYPE = Dict(
        :q4_k => Inf_.GGML_TYPE_Q4_K,
        :q5_k => Inf_.GGML_TYPE_Q5_K,
        :q6_k => Inf_.GGML_TYPE_Q6_K,
        :iq4_xs => Inf_.GGML_TYPE_IQ4_XS,
        :iq2_xs => Inf_.GGML_TYPE_IQ2_XS,
        :iq3_xxs => Inf_.GGML_TYPE_IQ3_XXS,
        :q8_0 => Inf_.GGML_TYPE_Q8_0,
    )

    # Random block bytes with sane f16 scale fields (random bytes elsewhere).
    function _random_quant_matrix(rng, kind::Symbol, nrows::Int, k::Int)
        bbytes, bvals, scale_offs = _DQ_LAYOUT[kind]
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
        return Inf_.QuantMatrix(data, 0, _DQ_TYPE[kind], k, nrows, rb, C_NULL, Int64(bvals))
    end

    # Device path: to_device places the weight; matmul! runs the GEMV.
    function _device_matmul(qm, X, workload)
        pw = Gesso.ProjectionWeight(; shape=size(qm), storage=qm)
        emb = Gesso.EmbeddingTable(; shape=(4, 2), storage=ones(Float32, 4, 2))
        gpu = Gesso.to_device(cuda, (; embedding=emb, blocks=[(; w=pw)], lm_head=emb))
        w = gpu.blocks[1].w
        @test w.storage isa _dq_ext.GpuDenseQuant
        n, nrows = size(X, 1), size(qm, 1)
        dst = _mat(Gesso.Activation, CUDA.zeros(Float32, n, nrows))
        x = _mat(Gesso.Activation, CuArray{Float32}(X))
        Gesso.matmul!(cuda, dst, x, w, workload)
        return Array(dst.storage)
    end

    # Per-element error against the derived bound, host F64 accumulation.
    function _check_against_host(qm, X, dev; label::AbstractString)
        W = Float64.(Inf_._quant_host_f32(qm))
        X64 = Float64.(X)
        ref = X64 * transpose(W)
        n, k = size(X64)
        nrows = size(W, 1)
        nb = div(k, 32)
        amax = zeros(Float64, n, nb)
        for t in 1:n, b in 1:nb
            amax[t, b] = maximum(abs, view(X64, t, (32 * (b - 1) + 1):(32 * b)))
        end
        Sb = zeros(Float64, nrows, nb)
        for b in 1:nb
            Sb[:, b] = sum(abs, view(W, :, (32 * (b - 1) + 1):(32 * b)); dims=2)
        end
        bound = (amax ./ 254.0) * transpose(Sb)
        A = abs.(X64) * transpose(abs.(W))
        tol = bound .+ 5.0e-5 .* A .+ 1.0e-9
        err = abs.(Float64.(dev) .- ref)
        ratio = maximum(err ./ tol)
        rel = maximum(err) / maximum(abs, ref)
        @info "dense quant GEMV" label maxerr = maximum(err) max_ref = maximum(abs, ref) rel_to_max_ref = rel worst_err_over_tol = ratio
        @test all(err .<= tol)
        return nothing
    end

    @testset "GpuDenseQuant: random blocks, device GEMV vs host dequant" begin
        rng = MersenneTwister(20261008)
        for kind in (:q5_k, :q6_k, :iq4_xs, :q4_k, :q8_0)
            qm = _random_quant_matrix(rng, kind, 37, 512)
            X = randn(rng, Float32, 3, 512)
            _check_against_host(qm, X, _device_matmul(qm, X, Gesso.DecodeWorkload()); label="$kind decode n=3")
            _check_against_host(qm, X, _device_matmul(qm, X, Gesso.PrefillWorkload()); label="$kind prefill n=3")
        end
    end

    @testset "GpuDenseQuant: single token row (n = 1) at a wider k" begin
        rng = MersenneTwister(7)
        for kind in (:q5_k, :q6_k, :iq4_xs, :q4_k, :q8_0)
            qm = _random_quant_matrix(rng, kind, 130, 2048)
            X = randn(rng, Float32, 1, 2048)
            _check_against_host(qm, X, _device_matmul(qm, X, Gesso.DecodeWorkload()); label="$kind n=1 k=2048")
        end
    end

    # From GESSO_DQ_GEMM_MIN tokens the prefill dequantizes W to F16 and runs
    # cuBLAS. Scratch 0 MiB forces one row per chunk, so the chunk offsets into
    # W and into dst are exercised as well.
    @testset "GpuDenseQuant: prefill GEMM vs host dequant" begin
        rng = MersenneTwister(20261009)
        for kind in (:q5_k, :q6_k, :iq4_xs, :q4_k, :q8_0), mib in ("32", "0")
            qm = _random_quant_matrix(rng, kind, 37, 512)
            X = randn(rng, Float32, 40, 512)
            dev = withenv("GESSO_DQ_GEMM_MIN" => "32", "GESSO_DQ_GEMM_SCRATCH_MIB" => mib) do
                _device_matmul(qm, X, Gesso.PrefillWorkload())
            end
            _check_against_host(qm, X, dev; label="$kind GEMM n=40 scratch=$mib")
        end
        # Q8_0 rows need not fill whole 256-value groups; the dequant warp
        # stops at the row's last block.
        qm = _random_quant_matrix(rng, :q8_0, 37, 96)
        X = randn(rng, Float32, 40, 96)
        dev = withenv(() -> _device_matmul(qm, X, Gesso.PrefillWorkload()), "GESSO_DQ_GEMM_MIN" => "32")
        _check_against_host(qm, X, dev; label="q8_0 GEMM n=40 k=96")
    end

    @testset "GpuDenseQuant: placement switch and budget fallback" begin
        rng = MersenneTwister(3)
        qm = _random_quant_matrix(rng, :q5_k, 1024, 1024)
        pw = Gesso.ProjectionWeight(; shape=size(qm), storage=qm)
        emb = Gesso.EmbeddingTable(; shape=(4, 2), storage=ones(Float32, 4, 2))
        ts = (; embedding=emb, blocks=[(; w=pw)], lm_head=emb)
        withenv("GESSO_GPU_DENSE_QUANT" => "0") do
            gpu = Gesso.to_device(cuda, ts)
            @test gpu.blocks[1].w.storage isa CUDA.CuArray{Float16}
        end
        # budget 0: the native bytes do not fit, so the weight takes the existing
        # path (dequant to F32, then HostOffload when even F32 does not fit)
        withenv("GESSO_GPU_DENSE_QUANT" => "1", "GESSO_GPU_BUDGET_MIB" => "0") do
            gpu = Gesso.to_device(cuda, ts)
            @test gpu.blocks[1].w.storage isa Gesso.Inference.HostOffload
        end
    end

    # Expert path: one launch of the vector entry point on a single-column
    # activation, against the same host reference and bound as the dense path.
    function _device_expert_matmul(qm, X1)
        k = size(X1, 2)
        nrows = size(qm, 1)
        q = CuArray{UInt8}(Inf_._quant_bytes(qm))
        x = CuArray{Float32}(vec(X1))
        y = CUDA.zeros(Float32, nrows)
        _dq_ext._cuda_launch_qgemv!(y, x, q, qm.ggml_type, 0, k, nrows, 1.0f0, 0.0f0)
        return reshape(Array(y), 1, nrows)
    end

    @testset "CUDA expert MMVQ: IQ2_XS, IQ3_XXS, IQ4_XS vs host dequant" begin
        rng = MersenneTwister(41)
        for kind in (:iq2_xs, :iq3_xxs, :iq4_xs)
            qm = _random_quant_matrix(rng, kind, 37, 512)
            X = randn(rng, Float32, 1, 512)
            _check_against_host(qm, X, _device_expert_matmul(qm, X); label="$kind expert n=1")
        end
    end

    @testset "GpuDenseQuant: Ornith 35B tensors (Q5_K attn_qkv, Q6_K ssm_out)" begin
        path = get(ENV, "GESSO_ORNITH35_GGUF", "")
        if isempty(path) || !isfile(path)
            @info "skip Ornith dense-quant GEMV: set GESSO_ORNITH35_GGUF to the Ornith 35B GGUF"
            return
        end
        hdr = Inf_.gguf_header(path)
        for (name, want) in (("blk.0.attn_qkv.weight", Inf_.GGML_TYPE_Q5_K), ("blk.0.ssm_out.weight", Inf_.GGML_TYPE_Q6_K))
            info = only(filter(i -> i.name == name, hdr.tensors))
            qm = Inf_._quant_matrix(hdr, info)
            @test qm.ggml_type == want
            X = randn(MersenneTwister(11), Float32, 2, qm.ne0)
            _check_against_host(qm, X, _device_matmul(qm, X, Gesso.DecodeWorkload()); label=name)
            X = randn(MersenneTwister(12), Float32, 64, qm.ne0)
            _check_against_host(qm, X, _device_matmul(qm, X, Gesso.PrefillWorkload()); label="$name GEMM n=64")
        end
    end
end
