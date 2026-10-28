# Lava dense quant for the half types (BF16, F16) and the IQ types that the
# Flash-Next and Qwen3.5 GGUFs carry as dense matrices (IQ2_XS, IQ2_XXS, IQ2_S,
# IQ3_XXS, IQ3_S). The bytes stay in GGUF layout on the device, and the in-kernel
# decode is checked against the host dequant reference. Named skip without a
# usable Vulkan device.
#
# Bounds. The GEMV (one token) decodes the same bytes on both sides, so the only
# error is F32 accumulation: |dev - ref| <= 1e-4 * max(1, max|ref|). The prefill
# path dequantizes W to F16 before its GEMM, so each product carries an F16
# rounding of W and of X. That is the same 2e-3 * absolute-sum bound the Q5_K
# and Q6_K prefill checks use. BF16 values have 8 significant bits and the F16
# staging keeps 11, so BF16 is exact in F16 whenever the exponent is in range;
# its prefill error sits at the F16 bound, not above it.

using Test: AbstractTestSet, Broken, get_testset, record
using Random: MersenneTwister

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

isdefined(@__MODULE__, :_q4x_toy_model) || include(joinpath(@__DIR__, "qwen4exp_toy.jl"))

if !LAVA_LOADED || !VULKAN_OK
    _skip("no usable Vulkan device — Lava half and IQ dense quant tests skipped (§LXXXI skip law)")
else
    _lh_ext = Base.get_extension(Gesso, :GessoLavaExt)
    _lh_inf = Gesso.Inference
    _lh_lava = Gesso.LavaBackend()
    _lh_wl = Gesso.PrefillWorkload()

    # Block bytes and values per block. The IQ scale d is the first f16 field of
    # every block, and the rest of the block is decoded from arbitrary bytes, so
    # random payloads are valid input for these types.
    const _LH_BLOCK = Dict(
        :bf16 => (2, 1),
        :f16 => (2, 1),
        :iq2_xs => (74, 256),
        :iq2_xxs => (66, 256),
        :iq2_s => (82, 256),
        :iq3_xxs => (98, 256),
        :iq3_s => (110, 256),
    )
    const _LH_TYPE = Dict(
        :bf16 => _lh_inf.GGML_TYPE_BF16,
        :f16 => _lh_inf.GGML_TYPE_F16,
        :iq2_xs => _lh_inf.GGML_TYPE_IQ2_XS,
        :iq2_xxs => _lh_inf.GGML_TYPE_IQ2_XXS,
        :iq2_s => _lh_inf.GGML_TYPE_IQ2_S,
        :iq3_xxs => _lh_inf.GGML_TYPE_IQ3_XXS,
        :iq3_s => _lh_inf.GGML_TYPE_IQ3_S,
    )
    # Worst error per kind, and each negative control's error, for the report.
    const _LH_WORST = Dict{String,Float64}()
    const _LH_NC = Dict{String,Float64}()

    _lh_note!(d::Dict{String,Float64}, k::String, v::Float64) =
        d[k] = haskey(d, k) ? max(d[k], v) : v

    # Host reference in F64: the decode goes through the host dequant, which is
    # the same byte layout the kernel reads.
    function _lh_ref(qm, X)
        W = Float64.(_lh_inf._quant_host_f32(qm))
        X64 = Float64.(X)
        return X64 * transpose(W), abs.(X64) * transpose(abs.(W))
    end

    function _lh_random_matrix(rng, kind::Symbol, nrows::Int, ne0::Int)
        bbytes, bvals = _LH_BLOCK[kind]
        rb = div(ne0, bvals) * bbytes
        if kind == :bf16 || kind == :f16
            # Stored row-major (ne0 innermost), one value per element.
            vals = vec(permutedims(0.5f0 .* randn(rng, Float32, nrows, ne0)))
            data = kind == :bf16 ?
                collect(reinterpret(UInt8, UInt16.(reinterpret(UInt32, vals) .>> 16))) :
                collect(reinterpret(UInt8, Float16.(vals)))
        else
            data = rand(rng, UInt8, nrows * rb)
            for r in 1:nrows, j in 1:div(ne0, bvals)
                base = (r - 1) * rb + (j - 1) * bbytes
                d = Float16((rand(rng) * 0.04 + 0.002) * (rand(rng) < 0.5 ? -1 : 1))
                data[(base+1):(base+2)] .= reinterpret(UInt8, [d])
            end
        end
        return _lh_inf.QuantMatrix(data, 0, _LH_TYPE[kind], ne0, nrows, rb, Int64(bvals))
    end

    # BF16 matrix from dense F32 (out, in): the top 16 bits of each value, so the
    # host decode and the device decode see the same numbers.
    function _lh_bf16_matrix(W::Matrix{Float32})
        nrows, ne0 = size(W)
        u = UInt16.(reinterpret(UInt32, vec(permutedims(W))) .>> 16)
        return _lh_inf.QuantMatrix(
            collect(reinterpret(UInt8, u)), 0, _lh_inf.GGML_TYPE_BF16,
            ne0, nrows, 2 * ne0, Int64(1),
        )
    end

    # Device path through to_device and matmul!, the production placement.
    function _lh_device_matmul(qm, X)
        pw = Gesso.ProjectionWeight(; shape=size(qm), storage=qm)
        emb = Gesso.EmbeddingTable(; shape=(4, 2), storage=ones(Float32, 4, 2))
        dev = Gesso.to_device(_lh_lava, (; embedding=emb, blocks=[(; w=pw)], lm_head=emb))
        w = dev.blocks[1].w
        @test w.storage isa _lh_ext.LavaBlockQuant
        n, nrows = size(X, 1), size(qm, 1)
        dst = Gesso.Activation(;
            shape=(n, nrows),
            storage=fill!(Lava.LavaArray{Float32}(undef, n, nrows), 0.0f0),
        )
        x = Gesso.Activation(; shape=size(X), storage=Lava.LavaArray{Float32}(X))
        Gesso.matmul!(_lh_lava, dst, x, w, _lh_wl)
        return Array(dst.storage)
    end

    @testset "Lava half and IQ dense quant: GEMV and prefill vs host dequant" begin
        rng = MersenneTwister(20261026)
        cases = (
            (:bf16, 40), (:bf16, 512),
            (:f16, 40), (:f16, 512),
            (:iq2_xs, 512), (:iq2_xxs, 512), (:iq2_s, 512),
            (:iq3_xxs, 512), (:iq3_s, 512),
        )
        for (kind, ne0) in cases
            @testset "$kind ne0=$ne0" begin
                nrows = 37
                qm = _lh_random_matrix(rng, kind, nrows, ne0)
                @test _lh_ext._lq_eligible(qm)
                up = _lh_ext._lq_upload(qm)
                @test up isa _lh_ext.LavaBlockQuant
                @test up.kind == _lh_ext._lq_kind(qm.ggml_type)
                label = string(kind)

                X1 = randn(rng, Float32, 1, ne0)
                ref1, _ = _lh_ref(qm, X1)
                dev1 = _lh_device_matmul(qm, X1)
                err1 = maximum(abs, Float64.(dev1) .- ref1)
                bound1 = 1.0e-4 * max(1.0, maximum(abs, ref1))
                _lh_note!(_LH_WORST, "$label gemv", err1)
                @info "lava half/IQ GEMV" label ne0 maxerr = err1 bound = bound1
                @test err1 <= bound1

                ntok = max(40, _lh_ext._lq_gemm_min())
                Xp = randn(rng, Float32, ntok, ne0)
                refp, A = _lh_ref(qm, Xp)
                devp = Float64.(_lh_device_matmul(qm, Xp))
                errp = abs.(devp .- refp)
                tolp = 2.0e-3 .* A .+ 1.0e-9
                _lh_note!(_LH_WORST, "$label prefill", maximum(errp))
                @info "lava half/IQ prefill" label ne0 ntok maxerr = maximum(errp) worst_over_tol = maximum(errp ./ tolp)
                @test all(errp .<= tolp)
            end
        end
    end

    @testset "half kinds: device bytes are the GGUF bytes" begin
        rng = MersenneTwister(40)
        for kind in (:bf16, :f16), (nrows, ne0) in ((37, 40), (64, 512))
            qm = _lh_random_matrix(rng, kind, nrows, ne0)
            @test _lh_ext._lq_device_bytes(qm) == 2 * ne0 * nrows
            @test _lh_ext._lq_block_values(_lh_ext._lq_kind(qm.ggml_type)) == 1
        end
    end

    # Each control must fall outside the GEMV bound. A bound that a wrong
    # decode also passes would not catch a wrong decode.
    @testset "negative controls must exceed the bound" begin
        rng = MersenneTwister(0x1f16)
        ne0, nrows = 512, 37
        qm = _lh_random_matrix(rng, :bf16, nrows, ne0)
        up = _lh_ext._lq_upload(qm)
        X = randn(rng, Float32, 1, ne0)
        Xd = Lava.LavaArray{Float32}(X)
        function run_gemv(lq)
            Y = fill!(Lava.LavaArray{Float32}(undef, 1, nrows), 0.0f0)
            _lh_ext._lq_matmul!(Y, Xd, lq)
            return Float64.(Array(Y))
        end
        ref, _ = _lh_ref(qm, X)
        bound = 1.0e-4 * max(1.0, maximum(abs, ref))
        ncs = (
            "bf16 bytes read as f16" => _lh_ext.LavaBlockQuant(
                up.data, _lh_inf.GGML_TYPE_F16, _lh_ext._LQ_KIND_F16, ne0, nrows, 2 * ne0,
            ),
            "bf16 row stride minus 2" => _lh_ext.LavaBlockQuant(
                up.data, up.ggml_type, up.kind, ne0, nrows, 2 * ne0 - 2,
            ),
        )
        for (name, lq) in ncs
            err = maximum(abs, run_gemv(lq) .- ref)
            _lh_note!(_LH_NC, name, err)
            @info "negative control" name err bound
            @test !(err <= bound)
        end
    end

    # BF16 toy model: the eligible dense matrices of the qwen4exp toy are
    # replaced by BF16 QuantMatrix values (the same numbers, truncated to 16
    # bits), so the CPU half path and the device GGUF-bytes path decode the same
    # weights. Embeddings, norms, conv taps, expert banks and the router stay as
    # they are, since they are not dense matmul weights on the device.
    @testset "BF16 qwen4exp toy forward on Lava vs CPU" begin
        cfg = _lh_inf.qwen4exp_config(_q4x_toy_kv())
        m = _q4x_toy_model(cfg)
        converted = String[]
        for name in collect(keys(m.cache))
            v = m.cache[name]
            v isa Matrix{Float32} || continue
            (name == "token_embd.weight" || name == "per_layer_token_embd.weight" ||
             occursin("norm", name) || occursin("conv1d", name) ||
             occursin("ffn_gate_inp", name) || occursin("_exps", name)) && continue
            m.cache[name] = _lh_bf16_matrix(v)
            push!(converted, name)
        end
        @test !isempty(converted)
        dm = _lh_inf.qwen4exp_device(_lh_lava, m)
        chunks = ([5, 17, 2, 33, 8], [1])
        cst = _lh_inf.Qwen4expState(cfg)
        dst = _lh_inf.qwen4exp_device_state(dm)
        function check_rel(name::String, dev, ref)
            host = Float64.(Array(dev))
            r = Float64.(ref)
            @test size(host) == size(r)
            size(host) == size(r) || return
            err = maximum(abs, host .- r; init=0.0)
            scale = max(1.0, maximum(abs, r; init=0.0))
            _lh_note!(_LH_WORST, "toy bf16 $name", err / scale)
            @test err <= 1.0e-4 * scale
        end
        for toks in chunks
            tc = Dict{String,Any}()
            td = Dict{String,Any}()
            rc = _lh_inf.qwen4exp_forward!(m, cst, toks; taps=tc)
            rd = _lh_inf.qwen4exp_forward!(dm, dst, toks; taps=td)
            for k in keys(tc)
                haskey(td, k) && check_rel(k, td[k], tc[k])
            end
            check_rel("residual", rd, rc)
            lc = _lh_inf.qwen4exp_logits(m, rc; taps=nothing)
            ld = _lh_inf.qwen4exp_logits(dm, rd; taps=nothing)
            check_rel("logits", reshape(ld, 1, :), reshape(lc, 1, :))
        end
        println("bf16 toy converted matrices: ", length(converted))
    end

    println("lava half/IQ worst errors (absolute, GEMV bound 1e-4 * max(1, |ref|)):")
    for (k, v) in sort(collect(_LH_WORST); by=first)
        println("  ", rpad(k, 30), " worst ", v)
    end
    println("negative controls (must exceed the GEMV bound):")
    for (k, v) in sort(collect(_LH_NC); by=first)
        println("  ", rpad(k, 30), " error ", v)
    end
end
