# Gemma 4 on Lava: the toy model through prefill and greedy decode on Lava
# against the CPU oracle, with f32 and q8 KV, and a negative control with the
# sliding window disabled. Named skip without a usable Vulkan device.
#
# Tolerance. The forward bound is 1e-4 * max(1, max|ref|), as in
# test_lava_lfm25.jl, against a CPU oracle running the same KV cache format.
# The prompt is longer than the sliding window, so the window masking is
# exercised in layer 0. The q8 arm runs on a toy with 32-wide heads, because
# q8 KV needs d_head divisible by 32; the f32 arm runs on the shared toy.

using Gesso
using Test
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
isdefined(@__MODULE__, :_gemma4_toy) || include(joinpath(@__DIR__, "gemma4_toy.jl"))

if !LAVA_LOADED || !VULKAN_OK
    _skip("no usable Vulkan device — Lava Gemma4 tests skipped (§LXXXI skip law)")
else
    _lg4_lava = Gesso.LavaBackend()
    _lg4_cpu = Gesso.CPUBackend()
    _lg4_err(a, b) = maximum(abs, Float64.(Array(a)) .- Float64.(Array(b)); init=0.0)
    _lg4_tol(ref) = 1.0e-4 * max(1.0, maximum(abs, Float64.(Array(ref)); init=0.0))

    function _lg4_case(kv)
        spec = Gesso.Inference.spec_from_gguf_kv(kv)
        model, t = _gemma4_bind(spec, _gemma4_toy(spec))
        return (; spec, eps=spec.rms_norm_eps, model, t, t_dev=Gesso.to_device(_lg4_lava, t))
    end

    _LG4_PROMPT = [3, 7, 1, 12, 5, 9, 2]
    _LG4_NEW = 6

    _lg4_prefill(c, tt, backend) =
        Array(Gesso.reference_prefill(c.model, tt, _LG4_PROMPT; backend, eps=c.eps))

    function _lg4_generate(c, tt, backend)
        rows = Matrix{Float32}[]
        ids = Gesso.reference_generate(c.model, tt, _LG4_PROMPT; backend, eps=c.eps,
            max_new_tokens=_LG4_NEW, eos_ids=(),
            on_rows=(pos, M) -> push!(rows, Float32.(Array(M))))
        return ids, rows
    end

    _lg4_f = _lg4_case(_gemma4_toy_kv())
    _lg4_q = _lg4_case(_gemma4_toy_kv_wide())
    logits_cpu = _lg4_prefill(_lg4_f, _lg4_f.t, _lg4_cpu)

    @testset "lava Gemma4: import report finds no missing capability" begin
        @test isempty(Gesso.Inference._missing_capabilities(
            Gesso.required_semantics(_lg4_f.spec), _lg4_lava))
    end

    @testset "lava Gemma4: toy prefill matches the CPU oracle" begin
        @test _lg4_err(_lg4_prefill(_lg4_f, _lg4_f.t_dev, _lg4_lava), logits_cpu) <= _lg4_tol(logits_cpu)
    end

    @testset "lava Gemma4: toy greedy decode matches the CPU oracle" begin
        # The oracle runs the same cache format. Against an f32 CPU cache, q8
        # moves one row of the 32-wide toy by 0.51 on the CPU itself, which is
        # the format and not the port.
        for (c, fmt) in ((_lg4_f, "f32"), (_lg4_q, "q8"))
            ids_c, rows_c = withenv(() -> _lg4_generate(c, c.t, _lg4_cpu), "GESSO_KV_FORMAT" => fmt)
            ids_d, rows_d = withenv(() -> _lg4_generate(c, c.t_dev, _lg4_lava), "GESSO_KV_FORMAT" => fmt)
            @test ids_d == ids_c
            @test length(rows_d) == length(rows_c) == _LG4_NEW
            for i in eachindex(rows_c, rows_d)
                @test _lg4_err(rows_d[i], rows_c[i]) <= _lg4_tol(rows_c[i])
            end
        end
    end

    # The real model ships the fused gate_up bank as GGUF bytes. Lava keeps it on
    # the host and the MoE routine reads it in place, as the CPU path does.
    function _lg4_q8_gate_up(t)
        rng = MersenneTwister(20261029)
        blocks = map(t.blocks) do bt
            w = bt.wexp_gate_up
            n_exp, ne1, ne0 = size(w.storage)
            nblk = div(ne0, 32)
            bpe = ne1 * nblk * 34
            data = rand(rng, UInt8, n_exp * bpe)
            for b in 0:(n_exp*ne1*nblk-1)
                data[(b*34+1):(b*34+2)] .= reinterpret(UInt8, [Float16(0.002 + 0.01 * rand(rng))])
            end
            s = Gesso.Inference.GgufQuantSlice(
                data, 0, Gesso.Inference.GGML_TYPE_Q8_0, n_exp, ne0, ne1, bpe, Int64(32))
            merge(bt, (; wexp_gate_up=Gesso.ExpertWeight(w.shape, s)))
        end
        return merge(t, (; blocks))
    end

    @testset "lava Gemma4: GGUF expert bank stays on the host and matches the CPU oracle" begin
        tq = _lg4_q8_gate_up(_lg4_f.t)
        tq_dev = Gesso.to_device(_lg4_lava, tq)
        @test all(bt -> bt.wexp_gate_up.storage isa Gesso.Inference.GgufQuantSlice, tq_dev.blocks)
        ref = _lg4_prefill(_lg4_f, tq, _lg4_cpu)
        @test _lg4_err(_lg4_prefill(_lg4_f, tq_dev, _lg4_lava), ref) <= _lg4_tol(ref)
    end

    # The attention passes against a host loop, with a cache offset, GQA and a
    # window, untiled and with one query row per tile.
    function _lg4_attn_ref(Q, Kd, Vd, nh, nkv, hd, qoff, win)
        Nq = size(Q, 1)
        O = zeros(Float64, Nq, nh * hd)
        for t in 1:Nq, h in 1:nh
            kvh = (h - 1) ÷ (nh ÷ nkv) + 1
            r = qoff + t - 1
            ks = (win > 0 ? max(1, r - win + 1) : 1):r
            s = [sum(Float64(Q[t, (h-1)*hd+d]) * Kd[k, kvh, d] for d in 1:hd) for k in ks]
            p = exp.(s .- maximum(s))
            p ./= sum(p)
            for d in 1:hd
                O[t, (h-1)*hd+d] = sum(p[i] * Vd[k, kvh, d] for (i, k) in enumerate(ks))
            end
        end
        return O
    end

    @testset "lava Gemma4: attention passes match a host loop, tiled and untiled" begin
        ext = Base.get_extension(Gesso, :GessoLavaExt)
        rng = MersenneTwister(20261030)
        nh, nkv, hd = 4, 2, 16
        for (Nq, qoff, win) in ((9, 1, 0), (9, 1, 4), (5, 8, 0), (5, 8, 3), (1, 12, 5))
            K = qoff + Nq - 1
            Q = randn(rng, Float32, Nq, nh * hd)
            Kd = randn(rng, Float32, K, nkv, hd)
            Vd = randn(rng, Float32, K, nkv, hd)
            ref = _lg4_attn_ref(Q, Kd, Vd, nh, nkv, hd, qoff, win)
            run() = Array(ext._g4l_attention(ext._g4l_up(Q), ext._g4l_up(Kd), ext._g4l_up(Vd), nh, nkv, hd, qoff, win))
            @test _lg4_err(run(), ref) <= _lg4_tol(ref)
            prev = ext._G4L_SCORE_BYTES[]
            ext._G4L_SCORE_BYTES[] = 4 * nh
            try
                @test _lg4_err(run(), ref) <= _lg4_tol(ref)
            finally
                ext._G4L_SCORE_BYTES[] = prev
            end
        end
    end

    @testset "lava Gemma4: negative control, sliding window off exceeds the bound" begin
        prev = Gesso.Inference._GEMMA4_WINDOW_MASK_ENABLED[]
        Gesso.Inference._GEMMA4_WINDOW_MASK_ENABLED[] = false
        try
            off = _lg4_prefill(_lg4_f, _lg4_f.t_dev, _lg4_lava)
            @test _lg4_err(off, logits_cpu) > _lg4_tol(logits_cpu)
        finally
            Gesso.Inference._GEMMA4_WINDOW_MASK_ENABLED[] = prev
        end
    end
end
