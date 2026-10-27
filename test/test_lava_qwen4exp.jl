# qwen4exp forward on Lava: the device seams checked one op at a time against
# CPU references, the toy model checked tap by tap over a prefill, decode steps
# and a multi-token chunk, negative controls that must exceed the tolerance,
# and a real-model block gated on GESSO_QWEN4EXP_GGUF. The CPU oracle is
# src/Inference/qwen4exp.jl. Named skip without a usable Vulkan device.
#
# Tolerance. The device runs in F32 and the CPU forward in F32; the per-op
# references are F64. Each output is a short inner product or a row-wise
# reduction of length at most n_embd (16 here), so the F32 error is a small
# multiple of 16 * eps_F32 (about 1e-6 relative). The forward chains a few
# dozen such ops per layer. The bound is 1e-4 * max(1, max|ref|), the same
# bound test_lava_hybrid.jl uses; the measured worst on the toy forward is about
# 4e-6, so the margin is about 25x. The negative controls produce errors of
# 0.8 to 2.5 relative, so the bound separates a correct seam from a broken one.

using Test: AbstractTestSet, Broken, get_testset, record
using Random: MersenneTwister

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

isdefined(@__MODULE__, :LAVA_LOADED) || include(joinpath(@__DIR__, "test_lava_seam.jl"))
isdefined(@__MODULE__, :_q4x_toy_model) || include(joinpath(@__DIR__, "qwen4exp_toy.jl"))

if !LAVA_LOADED || !VULKAN_OK
    _skip("no usable Vulkan device — Lava qwen4exp tests skipped (§LXXXI skip law)")
else
    _lq4_lava = Gesso.LavaBackend()
    _lq4_inf = Gesso.Inference
    _lq4_ext = Base.get_extension(Gesso, :GessoLavaExt)
    _LQ4_TOL = 1.0e-4

    _lq4_scale(ref) = max(1.0, maximum(abs, ref; init=0.0))
    _lq4_tol(ref) = _LQ4_TOL * _lq4_scale(ref)
    _lq4_dev(a::AbstractArray) = Lava.LavaArray{Float32}(Float32.(a))

    # A routed expert bank read exactly. The host MoE quantizes its activation
    # to Q8_K for a GgufQuantSlice, and that rounding turns a 1e-6 input
    # difference into ~2e-5 at the MoE output, so a chained comparison through
    # it measures the rounding, not the device. Wrapping the slice sends the
    # host MoE down its per-expert loop with each used expert dequantized to
    # F32 once. Both forwards read the same wrapper.
    struct _Lq4ExactBank
        s::_lq4_inf.GgufQuantSlice
        w::Dict{Int,Matrix{Float32}}
        lock::ReentrantLock
    end
    _Lq4ExactBank(s) = _Lq4ExactBank(s, Dict{Int,Matrix{Float32}}(), ReentrantLock())
    Base.size(b::_Lq4ExactBank) = size(getfield(b, :s))
    Base.getproperty(b::_Lq4ExactBank, f::Symbol) =
        f in (:s, :w, :lock) ? getfield(b, f) : getproperty(getfield(b, :s), f)
    _lq4_inf._q4x_moe!(y, x, router, g::_Lq4ExactBank, u, d, k) =
        _lq4_inf._cpu_moe_topk_swiglu_gguf!(y, x, router, g, u, d, k)
    function _lq4_inf._qgemv_expert!(y::AbstractVector{Float32}, x::AbstractVector{Float32},
            b::_Lq4ExactBank, e::Int; alpha::Float32=1.0f0, beta::Float32=0.0f0)
        W = lock(b.lock) do
            get!(() -> _lq4_inf._gguf_dequant_expert_uncached(b.s, e), b.w, e)
        end
        p = W * x
        if iszero(beta)
            y .= alpha .* p
        else
            y .= alpha .* p .+ beta .* y
        end
        return y
    end

    # Max abs error of a device result against a CPU reference, and the same
    # error relative to max(1, max|ref|), which is what the bound is stated in.
    function _lq4_err(dev, ref)
        host = Float64.(Array(dev))
        r = Float64.(ref)
        size(host) == size(r) || return (Inf, Inf)
        err = maximum(abs, host .- r; init=0.0)
        return err, err / _lq4_scale(r)
    end

    # Records the relative error in `worst` under `name` and asserts the bound.
    function _lq4_check(worst::Dict{String,Float64}, name::String, dev, ref)
        err, rel = _lq4_err(dev, ref)
        worst[name] = max(get(worst, name, 0.0), rel)
        @test err <= _lq4_tol(Float64.(ref))
        return err
    end

    _lq4_print(worst::Dict{String,Float64}) = for (k, v) in sort(collect(worst); by=last, rev=true)
        println("  ", rpad(k, 28), " worst rel err ", v, " (bound ", _LQ4_TOL, ")")
    end

    @testset "qwen4exp Lava per-op seams vs CPU" begin
        cfg = _lq4_inf.qwen4exp_config(_q4x_toy_kv())
        m = _q4x_toy_model(cfg)
        dm = _lq4_inf.qwen4exp_device(_lq4_lava, m)
        rng = MersenneTwister(0x4e4)
        NE, HC = cfg.n_embd, cfg.n_hc
        T = 5
        worst = Dict{String,Float64}()

        @testset "group RMS, shared and per-column weight" begin
            x = randn(rng, Float32, T, 2NE)
            for w in (1f0 .+ 0.1f0 .* randn(rng, Float32, NE),
                      1f0 .+ 0.1f0 .* randn(rng, Float32, 2NE))
                ref = _lq4_inf._q4x_group_rms(x, NE, w, cfg.eps)
                dev = _lq4_ext._q4x_dev_group_rms(_lq4_dev(x), NE, _lq4_dev(w), cfg.eps)
                _lq4_check(worst, "group_rms w=$(length(w))", dev, ref)
            end
        end

        @testset "HC mix with and without inject" begin
            x = randn(rng, Float32, T, NE * HC)
            mixed_ref, inj_ref = _lq4_inf._q4x_hc_mix(m, x, "blk.0.hc_attn")
            mixed, inj = _lq4_ext._q4x_dev_hc_mix(dm, _lq4_dev(x), "blk.0.hc_attn")
            _lq4_check(worst, "hc_mix mixed", mixed, mixed_ref)
            _lq4_check(worst, "hc_mix inject", inj, inj_ref)
        end

        @testset "HC combine" begin
            res = randn(rng, Float32, T, NE * HC)
            out = randn(rng, Float32, T, NE)
            inj = randn(rng, Float32, T, HC)
            ref = _lq4_inf._q4x_hc_combine(cfg, res, out, inj)
            dev = _lq4_ext._q4x_dev_hc_combine(cfg, _lq4_dev(res), _lq4_dev(out), _lq4_dev(inj))
            _lq4_check(worst, "hc_combine", dev, ref)
        end

        @testset "causal conv, dilation 1 and 2, T below and above history, two chunks" begin
            k, C = 4, 5
            for dil in (1, 2)
                H = (k - 1) * dil
                w = randn(rng, Float32, k * C)
                hist = zeros(Float32, H, C)
                hist_d = _lq4_dev(hist)
                for Tc in (2, 9)    # Tc < H shifts the history, Tc > H replaces it
                    x = randn(rng, Float32, Tc, C)
                    y_ref = _lq4_inf._q4x_causal_conv!(hist, x, w, k, dil)
                    y_dev, newh = _lq4_ext._q4x_dev_conv(hist_d, _lq4_dev(x), _lq4_dev(w), k, dil)
                    _lq4_check(worst, "conv y dil=$dil", y_dev, y_ref)
                    _lq4_check(worst, "conv hist dil=$dil", newh, hist)
                    hist_d = newh
                end
            end
        end

        @testset "PLE gate and gated value" begin
            key = randn(rng, Float32, T, NE * HC)
            qry = randn(rng, Float32, T, NE * HC)
            val = randn(rng, Float32, T, NE)
            gate_ref = zeros(Float64, T, HC)
            gated_ref = zeros(Float64, T, NE * HC)
            for t in 1:T, s in 0:(HC-1)
                acc = 0.0
                for i in 1:NE
                    acc += Float64(key[t, i+NE*s]) * qry[t, i+NE*s]
                end
                sc = acc / sqrt(NE)
                gate_ref[t, s+1] = 1 / (1 + exp(-sign(sc) * sqrt(max(abs(sc), 1e-6))))
                for i in 1:NE
                    gated_ref[t, i+NE*s] = val[t, i] * gate_ref[t, s+1]
                end
            end
            gate, gated = _lq4_ext._q4x_dev_ple_gate(_lq4_dev(key), _lq4_dev(qry), _lq4_dev(val), NE, HC)
            _lq4_check(worst, "ple gate", gate, gate_ref)
            _lq4_check(worst, "ple gated", gated, gated_ref)
        end

        @testset "GDN gate, softplus with the x > 20 branch" begin
            nv = cfg.ssm_nv
            alpha = 3f0 .* randn(rng, Float32, T, nv)
            alpha[1, 1] = 25f0
            dt = randn(rng, Float32, nv)
            A = -(0.5f0 .+ rand(rng, Float32, nv))
            sp_ref = zeros(Float64, T, nv)
            g_ref = zeros(Float64, T, nv)
            for t in 1:T, h in 1:nv
                x = Float64(alpha[t, h]) + Float64(dt[h])
                sp_ref[t, h] = x > 20 ? x : log1p(exp(x))
                g_ref[t, h] = sp_ref[t, h] * Float64(A[h])
            end
            sp, g = _lq4_ext._q4x_dev_gdn_gate(_lq4_dev(alpha), _lq4_dev(dt), _lq4_dev(A))
            _lq4_check(worst, "gdn sp", sp, sp_ref)
            _lq4_check(worst, "gdn g", g, g_ref)
        end

        @testset "causal GQA attention core, prefix p0 > 0" begin
            NH, NKV, HD = cfg.n_head, cfg.n_head_kv, cfg.head_dim
            p0 = 3
            L = p0 + T
            grp = NH ÷ NKV
            q = randn(rng, Float32, T, NH * HD)
            K = randn(rng, Float32, L, NKV * HD)
            V = randn(rng, Float32, L, NKV * HD)
            ref = zeros(Float64, T, NH * HD)
            for h in 0:(NH-1), t in 1:T
                kb = (h ÷ grp) * HD
                lim = p0 + t
                sc = [sum(Float64(q[t, h*HD+d]) * K[u, kb+d] for d in 1:HD) / sqrt(HD) for u in 1:lim]
                wts = exp.(sc .- maximum(sc))
                wts ./= sum(wts)
                for d in 1:HD
                    ref[t, h*HD+d] = sum(wts[u] * V[u, kb+d] for u in 1:lim)
                end
            end
            dev = _lq4_ext._q4x_dev_attn_core(_lq4_dev(q), _lq4_dev(K), _lq4_dev(V), p0, NH, NKV, HD)
            _lq4_check(worst, "attn core", dev, ref)
        end

        println("per-op worst relative errors:")
        _lq4_print(worst)
    end

    # Toy model on Lava against the CPU oracle: one prefill, decode steps and a
    # multi-token chunk, with state carried across all of them. Every tap the CPU
    # forward records must exist on the device, and each must match.
    @testset "qwen4exp toy forward on Lava vs CPU: all taps, residual, logits" begin
        cfg = _lq4_inf.qwen4exp_config(_q4x_toy_kv())
        m = _q4x_toy_model(cfg)
        dm = _lq4_inf.qwen4exp_device(_lq4_lava, m)
        chunks = ([5, 17, 2, 33, 8], [1], [4], [30, 2, 8, 21])
        cst = _lq4_inf.Qwen4expState(cfg)
        dst = _lq4_inf.qwen4exp_device_state(dm)
        worst = Dict{String,Float64}()
        for (i, toks) in enumerate(chunks)
            tc = Dict{String,Any}()
            td = Dict{String,Any}()
            rc = _lq4_inf.qwen4exp_forward!(m, cst, toks; taps=tc)
            rd = _lq4_inf.qwen4exp_forward!(dm, dst, toks; taps=td)
            missing_taps = setdiff(collect(keys(tc)), collect(keys(td)))
            @test isempty(missing_taps)
            isempty(missing_taps) || println("  chunk $i missing device taps: ", missing_taps)
            for k in keys(tc)
                haskey(td, k) && _lq4_check(worst, k, td[k], tc[k])
            end
            _lq4_check(worst, "residual", rd, rc)
            lc = _lq4_inf.qwen4exp_logits(m, rc; taps=nothing)
            ld = _lq4_inf.qwen4exp_logits(dm, rd; taps=nothing)
            _lq4_check(worst, "logits", reshape(ld, 1, :), reshape(lc, 1, :))
        end
        println("toy forward worst relative errors over ", length(chunks), " chunks:")
        _lq4_print(worst)
    end

    # Each control injects one wrong seam or drops the carried state. Its
    # residual must fall outside the bound, or the bound cannot catch that seam.
    @testset "negative controls must exceed tolerance" begin
        cfg = _lq4_inf.qwen4exp_config(_q4x_toy_kv())
        m = _q4x_toy_model(cfg)
        dm = _lq4_inf.qwen4exp_device(_lq4_lava, m)
        chunks = ([5, 17, 2, 33, 8], [1], [4], [30, 2, 8, 21])
        cst = _lq4_inf.Qwen4expState(cfg)
        refs = [Matrix{Float32}(_lq4_inf.qwen4exp_forward!(m, cst, toks)) for toks in chunks]

        function residual_worst(fault::Symbol; fresh::Bool=false)
            dst = _lq4_inf.qwen4exp_device_state(dm)
            worst = 0.0
            for (i, toks) in enumerate(chunks)
                fresh && (dst = _lq4_inf.qwen4exp_device_state(dm))
                rd = _lq4_inf.qwen4exp_forward!(dm, dst, toks; fault=fault)
                _, rel = _lq4_err(rd, refs[i])
                worst = max(worst, rel)
            end
            return worst
        end

        clean = residual_worst(:none)
        println("  control none: worst rel err ", clean)
        @test clean <= _LQ4_TOL
        for f in (:gdn_silu, :ple_dil1, :hc_swap, :hc_no_scale)
            w = residual_worst(f)
            println("  control ", f, ": worst rel err ", w)
            @test w > _LQ4_TOL
        end
        w = residual_worst(:none; fresh=true)
        println("  control fresh_state: worst rel err ", w)
        @test w > _LQ4_TOL
    end

    # Real model: gated on the environment variable, otherwise a named skip.
    @testset "real qwen4exp GGUF, 4 layers, all taps" begin
        path = get(ENV, "GESSO_QWEN4EXP_GGUF", "")
        if isempty(path) || !isfile(path)
            _skip("GESSO_QWEN4EXP_GGUF not set — real qwen4exp Lava block skipped")
        else
            nl = 4
            m = _lq4_inf.load_qwen4exp(path)
            cfg = m.cfg
            println("  dense-weight bytes before upload (", nl, " layers): ",
                _lq4_ext._q4x_dense_bytes(m, nl))
            # Q4_K and Q8_0 take the cpu-order kernels on Lava, which quantize
            # the activation to Q8_K / Q8_0 the way the CPU does. That rounding
            # turns a 6e-5 input difference into ~3e-3 at the output (measured
            # on blk.2.ssm_out), so a chained forward through them cannot be
            # held to 1e-4 by either side. The kernels are pinned against the
            # CPU on the real tensors instead, with no tolerance to speak of.
            cpu_order = (_lq4_inf.GGML_TYPE_Q4_K, _lq4_inf.GGML_TYPE_Q8_0)
            dense = String[]
            for name in keys(m.index)
                startswith(name, "blk.") || continue
                parse(Int, split(name, '.')[2]) < nl || continue
                occursin("_exps", name) && continue
                push!(dense, name)
            end
            ncpu = 0
            for name in sort(dense)
                h = _lq4_inf._q4x_w(m, name)
                (h isa _lq4_inf.QuantMatrix && h.ggml_type in cpu_order) || continue
                lq = _lq4_ext._lq_upload(h)
                for ntok in (1, 8)
                    x = randn(MersenneTwister(ntok), Float32, ntok, h.ne0)
                    yc = zeros(Float32, ntok, h.nrows)
                    _lq4_inf._quant_matmul!(yc, x, h)
                    yd = Lava.LavaArray{Float32}(zeros(Float32, ntok, h.nrows))
                    _lq4_ext._lq_matmul!(yd, _lq4_dev(x), lq)
                    err = maximum(abs.(Array(yd) .- yc)) / max(1.0f0, maximum(abs, yc))
                    @test err <= 1.0e-6
                    err > 1.0e-6 && println("  cpu-order mismatch ", name, " ntok=", ntok, ": ", err)
                end
                ncpu += 1
            end
            println("  cpu-order kernels checked against the CPU on ", ncpu, " real tensors")
            @test ncpu > 0
            # The chained forward runs exact against exact. The reference is
            # the CPU forward with every dense quantized projection dequantized
            # to F32. The device model dequantizes only the cpu-order kinds and
            # keeps the rest in GGUF bytes, whose Lava kernels take F32
            # activations and so compute the same exact product. Routed expert
            # banks run on the host in both forwards; both read one exact bank.
            # Half-precision matrices are left as loaded: the CPU widens them
            # to F32 exactly, and Lava decodes them in the kernel against F32
            # activations, so both sides compute the exact product.
            ref = _lq4_inf.Qwen4expModel(cfg, m.index, Dict{String,Any}())
            dev = _lq4_inf.Qwen4expModel(cfg, m.index, Dict{String,Any}())
            for name in dense
                h = _lq4_inf._q4x_w(m, name)
                (h isa _lq4_inf.QuantMatrix && !_lq4_inf._quant_half(h.ggml_type)) || continue
                w = _lq4_inf._quant_host_f32(h)
                ref.cache[name] = w
                h.ggml_type in cpu_order && (dev.cache[name] = w)
            end
            nbank = 0
            for name in keys(m.index)
                startswith(name, "blk.") && occursin("_exps", name) || continue
                parse(Int, split(name, '.')[2]) < nl || continue
                h = _lq4_inf._q4x_w(m, name)
                h isa _lq4_inf.GgufQuantSlice || continue
                ref.cache[name] = dev.cache[name] = _Lq4ExactBank(h)
                nbank += 1
            end
            @test nbank == 3nl
            dm = _lq4_inf.qwen4exp_device(_lq4_lava, dev)
            toks = [1 + (7i) % cfg.vocab for i in 0:7]
            cst = _lq4_inf.Qwen4expState(cfg)
            dst = _lq4_inf.qwen4exp_device_state(dm)
            tc = Dict{String,Any}()
            td = Dict{String,Any}()
            rc = _lq4_inf.qwen4exp_forward!(ref, cst, toks; n_layers=nl, taps=tc)
            rq = _lq4_inf.qwen4exp_forward!(m, _lq4_inf.Qwen4expState(cfg), toks; n_layers=nl)
            println("  CPU Q8_K-activation forward vs exact reference, residual rel err: ",
                _lq4_err(rq, rc)[2])
            rd = _lq4_inf.qwen4exp_forward!(dm, dst, toks; n_layers=nl, taps=td)
            worst = Dict{String,Float64}()
            @test isempty(setdiff(collect(keys(tc)), collect(keys(td))))
            for k in keys(tc)
                haskey(td, k) && _lq4_check(worst, k, td[k], tc[k])
            end
            _lq4_check(worst, "residual", rd, rc)
            # One decode step on the carried state: GDN, conv, KV and PLE history.
            for tok in ([toks[end] % cfg.vocab + 1], [3])
                rc = _lq4_inf.qwen4exp_forward!(ref, cst, tok; n_layers=nl)
                rd = _lq4_inf.qwen4exp_forward!(dm, dst, tok; n_layers=nl)
                _lq4_check(worst, "decode residual", rd, rc)
            end
            println("real model worst relative errors:")
            _lq4_print(worst)
        end
    end
end
