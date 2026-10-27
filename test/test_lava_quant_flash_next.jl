# Flash-Next quant types on the Lava (Vulkan) backend: IQ4_NL and Q2_0, dense and
# as expert banks. Every device GEMV is checked against the host F64 dequant times x,
# with the bound of
# test_lava_quant.jl: |dev - ref| <= 5e-5 * sum|W x| + 1e-9. The negative control
# doubles every block scale of the weight on the device and asserts that the same
# bound then fails. Named skip without a Vulkan device or without GESSO_GGML_DIR
# (the host reference). The real-tensor check reads one expert slice per type from
# GESSO_QWEN4EXP_GGUF and is skipped when that is unset.

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
    _skip("no usable Vulkan device — Flash-Next Lava quant tests skipped")
else
    const _FNQ_EXT = Base.get_extension(Gesso, :GessoLavaExt)
    const _FNQ_INF = Gesso.Inference
    const _FNQ_LAVA = Gesso.LavaBackend()
    const _FNQ_WL = Gesso.PrefillWorkload()

    # (block bytes, values per block). The f16 block scale sits at byte 0 of every
    # block for both types.
    const _FNQ_LAYOUT = Dict(
        :iq4_nl => (18, 32),
        :q2_0 => (18, 64),
    )
    const _FNQ_TYPE = Dict(
        :iq4_nl => _FNQ_INF.GGML_TYPE_IQ4_NL,
        :q2_0 => _FNQ_INF.GGML_TYPE_Q2_0,
    )

    _fnq_ggml_ready() = isfile(joinpath(get(ENV, "GESSO_GGML_DIR", ""), "libggml-base.so"))

    # Random block bytes with sane f16 block scales; every other byte is random,
    # so all grid indices and sign bytes are exercised.
    function _fnq_random(rng, kind::Symbol, nrows::Int, k::Int)
        bbytes, bvals = _FNQ_LAYOUT[kind]
        nblk = div(k, bvals)
        rb = nblk * bbytes
        data = rand(rng, UInt8, nrows * rb)
        for r in 1:nrows, j in 1:nblk
            o = (r - 1) * rb + (j - 1) * bbytes
            v = Float16((rand(rng) * 0.04 + 0.002) * (rand(rng) < 0.5 ? -1 : 1))
            data[(o+1):(o+2)] .= reinterpret(UInt8, [v])
        end
        return _FNQ_INF.QuantMatrix(
            data,
            0,
            _FNQ_TYPE[kind],
            k,
            nrows,
            rb,
            C_NULL,
            Int64(bvals),
        )
    end

    # The same bytes with every block scale doubled.
    function _fnq_double_scales(qm, kind::Symbol)
        bbytes, _ = _FNQ_LAYOUT[kind]
        data = copy(qm.data)
        nblk = div(qm.ne0, _FNQ_LAYOUT[kind][2]) * qm.nrows
        for b in 0:(nblk-1)
            o = b * bbytes
            v = reinterpret(Float16, data[(o+1):(o+2)])[1]
            data[(o+1):(o+2)] .= reinterpret(UInt8, [Float16(2 * v)])
        end
        return _FNQ_INF.QuantMatrix(
            data,
            qm.offset0,
            qm.ggml_type,
            qm.ne0,
            qm.nrows,
            qm.row_bytes,
            C_NULL,
            qm.blck_size,
        )
    end

    function _fnq_expert_bank(qms)
        data = reduce(vcat, [collect(_FNQ_INF._quant_bytes(q)) for q in qms])
        q1 = qms[1]
        return _FNQ_INF.GpuQuantSlice(
            Lava.LavaArray{UInt8}(data),
            q1.ggml_type,
            length(qms),
            q1.ne0,
            q1.nrows,
            q1.nrows * q1.row_bytes,
            q1.blck_size,
        )
    end

    function _fnq_expert_gemv(bank, e::Int, xv::Vector{Float32})
        x = Lava.LavaArray{Float32}(xv)
        y = Lava.LavaArray{Float32}(zeros(Float32, bank.ne1))
        _FNQ_INF._qgemv_expert!(y, x, bank, e)
        return Array(y)
    end

    # Dense path: to_device places the weight; matmul! runs the GEMV (or the
    # dequant-to-F16 prefill step from ntok >= GESSO_DQ_GEMM_MIN).
    function _fnq_dense_matmul(qm, X)
        pw = Gesso.ProjectionWeight(; shape=size(qm), storage=qm)
        emb = Gesso.EmbeddingTable(; shape=(4, 2), storage=ones(Float32, 4, 2))
        dev = Gesso.to_device(_FNQ_LAVA, (; embedding=emb, blocks=[(; w=pw)], lm_head=emb))
        w = dev.blocks[1].w
        @test w.storage isa _FNQ_EXT.LavaBlockQuant
        n, nrows = size(X, 1), size(qm, 1)
        dst = Gesso.Activation(;
            shape=(n, nrows),
            storage=fill!(Lava.LavaArray{Float32}(undef, n, nrows), 0.0f0),
        )
        x = Gesso.Activation(; shape=size(X), storage=Lava.LavaArray{Float32}(X))
        Gesso.matmul!(_FNQ_LAVA, dst, x, w, _FNQ_WL)
        return Array(dst.storage)
    end

    # Host F64 reference and the per-element bound. Returns the pass flag, the
    # largest absolute error and the largest error relative to its bound.
    function _fnq_bound(qm, X, dev; rtol::Float64=5.0e-5)
        W = Float64.(_FNQ_INF._quant_host_f32(qm))
        X64 = Float64.(X)
        ref = X64 * transpose(W)
        A = abs.(X64) * transpose(abs.(W))
        tol = rtol .* A .+ 1.0e-9
        err = abs.(Float64.(dev) .- ref)
        return all(err .<= tol), maximum(err), maximum(err ./ tol)
    end

    function _fnq_check(qm, X, dev; label::AbstractString, rtol::Float64=5.0e-5)
        if !_fnq_ggml_ready()
            _skip("set GESSO_GGML_DIR to a directory with libggml-base.so — $label host reference skipped")
            return nothing
        end
        ok, maxerr, ratio = _fnq_bound(qm, X, dev; rtol)
        @info "flash-next lava quant" label maxerr worst_err_over_tol = ratio
        @test ok
        return nothing
    end

    # The bound must fail on the doubled-scale device result.
    function _fnq_negative(qm_ref, X, dev_bad; label::AbstractString)
        if !_fnq_ggml_ready()
            _skip("set GESSO_GGML_DIR to a directory with libggml-base.so — $label negative control skipped")
            return nothing
        end
        ok, maxerr, _ = _fnq_bound(qm_ref, X, dev_bad)
        @info "flash-next lava negative control" label maxerr
        @test !ok
        return nothing
    end

    @testset "Flash-Next block sizes: a row is a whole number of blocks" begin
        rng = MersenneTwister(20261025)
        @test _FNQ_EXT._lq_eligible(_fnq_random(rng, :iq4_nl, 4, 96))
        @test !_FNQ_EXT._lq_eligible(_fnq_random(rng, :iq4_nl, 4, 32 * 4 + 16))
        q2_96 = _FNQ_INF.QuantMatrix(
            zeros(UInt8, 4 * 18),
            0,
            _FNQ_INF.GGML_TYPE_Q2_0,
            96,
            4,
            18,
            C_NULL,
            Int64(64),
        )
        @test !_FNQ_EXT._lq_eligible(q2_96)
        @test _FNQ_EXT._lq_eligible(_fnq_random(rng, :q2_0, 4, 128))
    end

    @testset "Flash-Next dense: IQ4_NL and Q2_0, device GEMV vs host dequant" begin
        rng = MersenneTwister(20261026)
        for kind in (:iq4_nl, :q2_0)
            for ntok in (1, 3)
                @testset "$kind ntok=$ntok" begin
                    qm = _fnq_random(rng, kind, 37, 512)
                    X = randn(rng, Float32, ntok, 512)
                    _fnq_check(qm, X, _fnq_dense_matmul(qm, X); label="dense $kind ntok=$ntok")
                    bad = _fnq_double_scales(qm, kind)
                    _fnq_negative(
                        qm,
                        X,
                        _fnq_dense_matmul(bad, X);
                        label="dense $kind ntok=$ntok",
                    )
                end
            end
        end
    end

    @testset "Flash-Next dense prefill: IQ4_NL and Q2_0 through the F16 dequant" begin
        rng = MersenneTwister(20261027)
        for kind in (:iq4_nl, :q2_0)
            @testset "$kind ntok=40" begin
                qm = _fnq_random(rng, kind, 37, 512)
                X = randn(rng, Float32, 40, 512)
                _fnq_check(
                    qm,
                    X,
                    _fnq_dense_matmul(qm, X);
                    label="prefill $kind 37x512 n40",
                    rtol=2.0e-3,
                )
            end
        end
    end

    @testset "Flash-Next expert GEMV: device bank vs host dequant" begin
        rng = MersenneTwister(20261028)
        for kind in (:iq4_nl, :q2_0)
            @testset "$kind" begin
                k = 512
                nrows = 37
                qms = [_fnq_random(rng, kind, nrows, k) for _ in 1:3]
                bank = _fnq_expert_bank(qms)
                xv = randn(rng, Float32, k)
                for e in (1, 3)
                    y = _fnq_expert_gemv(bank, e, xv)
                    _fnq_check(qms[e], reshape(xv, 1, k), reshape(y, 1, nrows); label="$kind expert $e")
                end
                e = 2
                bad_bank = _fnq_expert_bank([
                    qms[1],
                    _fnq_double_scales(qms[e], kind),
                    qms[3],
                ])
                y_bad = _fnq_expert_gemv(bad_bank, e, xv)
                _fnq_negative(
                    qms[e],
                    reshape(xv, 1, k),
                    reshape(y_bad, 1, nrows);
                    label="$kind expert $e",
                )
            end
        end
    end

    # One expert slice per type from a real GGUF. Expert tensors are 3-D; the
    # first type found in the file supplies the slice, and expert 2 is used so the
    # byte offset of a non-first expert is covered.
    @testset "Flash-Next real tensors: one expert slice per type" begin
        path = get(ENV, "GESSO_QWEN4EXP_GGUF", "")
        if isempty(path)
            _skip("GESSO_QWEN4EXP_GGUF not set — Flash-Next real-tensor check skipped")
        else
            hdr = _FNQ_INF.gguf_header(path)
            rng = MersenneTwister(20261029)
            for (kind, t) in (
                (:iq4_nl, _FNQ_INF.GGML_TYPE_IQ4_NL),
                (:q2_0, _FNQ_INF.GGML_TYPE_Q2_0),
            )
                @testset "$kind" begin
                    i = findfirst(
                        info ->
                            info.ggml_type == t &&
                                length(info.dims) == 3 &&
                                _FNQ_INF._is_expert_tensor(info.name),
                        hdr.tensors,
                    )
                    if i === nothing
                        _skip("no $kind expert tensor in $path")
                    else
                        info = hdr.tensors[i]
                        ne0, ne1, n_exp = info.dims
                        tr = _FNQ_INF._ggml_traits(t)
                        blck = Int(tr.blck_size)
                        rb = div(ne0, blck) * Int(tr.type_size)
                        bpe = rb * ne1
                        n_exp >= 2 || error("$kind: need at least two experts")
                        off = Int(hdr.data_start) + Int(info.offset) + bpe
                        qm = _FNQ_INF.QuantMatrix(
                            hdr.data,
                            off,
                            t,
                            ne0,
                            ne1,
                            rb,
                            tr.to_float,
                            Int64(blck),
                        )
                        xv = randn(rng, Float32, ne0)
                        bank = _fnq_expert_bank([qm])
                        y = _fnq_expert_gemv(bank, 1, xv)
                        _fnq_check(qm, reshape(xv, 1, ne0), reshape(y, 1, ne1); label="real $kind $(info.name)")
                    end
                end
            end
        end
    end
end
