# spark2_5 on Lava: the device GEGLU against the Float64 CPU GELU on a grid,
# then a toy three-layer model (two windowed layers, one full, partial RoPE on
# the full layer, per-head attention gate, GEGLU FFN, tied head) run through
# prefill and greedy decode on Lava and on the CPU. The prompt is longer than
# the window and decode runs past it, so the masked prefill and the K > W decode
# path are both exercised. Named skip without a usable Vulkan device.
#
# Tolerance. The device runs in F32 and the CPU GEGLU in F64. The device GELU
# is a Float32 Remez fit within 4e-7 absolute of the F64 one, and its output
# rounds to F32, so the grid bound is 2e-6 * max(1, |ref|). The forward bound
# is 1e-4 * max(1, max|ref|), as in test_lava_qwen4exp.jl; the measured worst
# on this toy is about 5e-6. The negative controls (window W-1, RoPE bases
# swapped between layer types, ffn gate and up swapped) move the logits by
# 0.5 or more, far outside the bound.

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

if !LAVA_LOADED || !VULKAN_OK
    _skip("no usable Vulkan device — Lava spark2_5 tests skipped (§LXXXI skip law)")
else
    _lsp_lava = Gesso.LavaBackend()
    _lsp_cpu = Gesso.CPUBackend()
    _lsp_inf = Gesso.Inference

    _lsp_err(a, b) = maximum(abs, Float64.(Array(a)) .- Float64.(Array(b)); init=0.0)
    _lsp_tol(ref) = 1.0e-4 * max(1.0, maximum(abs, Float64.(Array(ref)); init=0.0))
    _lsp_act(a) = Gesso.Activation(; shape=size(a), storage=a)

    @testset "lava spark2_5: GEGLU matches the F64 GELU" begin
        # Band edges of the device erf, in x: the fit switches at |x|/√2 =
        # 0.5, 1.25, 2.5 and 5.
        edges = [s * b * sqrt(2.0) for b in (0.5, 1.25, 2.5, 5.0) for s in (-1, 1)]
        xs = vcat(collect(-12.0:0.005:12.0), edges, prevfloat.(Float32.(edges)), [0.0, 40.0, -40.0])
        gate = Float32.(xs)
        up = Float32.(1 .+ 0.5 .* randn(MersenneTwister(2510), length(xs)))
        d = Lava.LavaArray{Float32}(zeros(Float32, length(xs)))
        _lsp_inf._geglu!(_lsp_lava, _lsp_act(d),
            _lsp_act(Lava.LavaArray{Float32}(gate)), _lsp_act(Lava.LavaArray{Float32}(up)))
        got = Float64.(Array(d))
        ref = [Gesso._cpu_gelu_erf(Float64(g)) * u for (g, u) in zip(gate, up)]
        worst = maximum(abs(got[i] - ref[i]) / (2.0e-6 * max(1.0, abs(ref[i]))) for i in eachindex(ref))
        @test worst <= 1.0
        @test got[end-2] == 0.0
        @test isfinite(got[end-1]) && isfinite(got[end])
    end

    _LSP_H, _LSP_NH, _LSP_HD, _LSP_NKV, _LSP_INTER, _LSP_V, _LSP_NL, _LSP_WIN = 64, 2, 32, 1, 96, 40, 3, 4
    _LSP_SWA = [true, true, false]
    _LSP_PROMPT = [3, 7, 1, 12, 5, 9]
    _LSP_NEW = 6
    _LSP_ROPE_FULL = Gesso.RoPEPolicy(; theta=5.0e6, rotary_dim=8, interleaved=false)
    _LSP_ROPE_SWA = Gesso.RoPEPolicy(; theta=1.0e4, rotary_dim=0, interleaved=false)

    function _lsp_spec(; layers=_LSP_NL, window=_LSP_WIN, rope=_LSP_ROPE_FULL, rope_swa=_LSP_ROPE_SWA)
        return Gesso.ArchitectureSpec(;
            family=:spark2_5,
            hidden_size=_LSP_H,
            num_layers=layers,
            n_heads=_LSP_NH,
            n_kv_heads=_LSP_NKV,
            vocab_size=_LSP_V,
            intermediate_size=_LSP_INTER,
            head_dim=_LSP_HD,
            activation_kind=:geglu,
            rope=rope,
            rope_swa=rope_swa,
            sliding_window=window,
            swa_layers=_LSP_SWA[1:layers],
            tie_word_embeddings=true,
            features=Set([:fused_qkv, :attn_head_gate]),
            rms_norm_eps=1e-6,
        )
    end

    # Names follow _gguf_spark_param_map; shapes follow expected_shape.
    function _lsp_tensors(rng)
        qd = _LSP_NH * _LSP_HD
        kvd = _LSP_NKV * _LSP_HD
        w(rows, cols) = Float32.(randn(rng, rows, cols) ./ sqrt(cols))
        nrm(n) = Float32.(1 .+ 0.1 .* randn(rng, n))
        by = Dict{String,Any}()
        by["token_embd.weight"] = Float32.(0.5 .* randn(rng, _LSP_V, _LSP_H))
        by["output_norm.weight"] = nrm(_LSP_H)
        for i in 0:(_LSP_NL-1)
            p = "blk.$i."
            by[p*"attn_norm.weight"] = nrm(_LSP_H)
            by[p*"attn_qkv.weight"] = w(qd + 2kvd, _LSP_H)
            by[p*"attn_gate.weight"] = w(_LSP_NH, _LSP_H)
            by[p*"attn_output.weight"] = w(_LSP_H, qd)
            by[p*"ffn_norm.weight"] = nrm(_LSP_H)
            by[p*"ffn_gate.weight"] = w(_LSP_INTER, _LSP_H)
            by[p*"ffn_up.weight"] = w(_LSP_INTER, _LSP_H)
            by[p*"ffn_down.weight"] = w(_LSP_H, _LSP_INTER)
        end
        return by
    end

    # Bind a checkpoint through a spec the way load_gguf does; the first
    # spec.num_layers blocks are kept.
    function _lsp_bind(spec, by)
        keep = filter(by) do (k, _)
            !startswith(k, "blk.") || parse(Int, split(k, ".")[2]) < spec.num_layers
        end
        bound = _lsp_inf.materialize_architecture(spec, _lsp_inf._gguf_spark_param_map(spec), keep)
        t = (embedding=bound.embedding, blocks=bound.blocks, lm_head=bound.lm_head,
            final_rms=bound.final_rms, rope=bound.rope)
        return _lsp_inf.spec_to_model(spec), t
    end

    _lsp_prefill(model, t, backend; hidden=false) =
        Array(Gesso.reference_prefill(model, t, _LSP_PROMPT; backend=backend, hidden=hidden))

    function _lsp_generate(model, t, backend)
        rows = Matrix{Float32}[]
        ids = Gesso.reference_generate(model, t, _LSP_PROMPT; backend=backend,
            max_new_tokens=_LSP_NEW, eos_ids=(), on_rows=(pos, M) -> push!(rows, Float32.(Array(M))))
        return ids, rows
    end

    _lsp_by = _lsp_tensors(MersenneTwister(20261027))
    _lsp_model, _lsp_t = _lsp_bind(_lsp_spec(), _lsp_by)
    _lsp_t_dev = Gesso.to_device(_lsp_lava, _lsp_t)
    _lsp_logits = _lsp_prefill(_lsp_model, _lsp_t, _lsp_cpu)

    @testset "lava spark2_5: import report finds no missing capability" begin
        @test isempty(_lsp_inf._missing_capabilities(Gesso.required_semantics(_lsp_spec()), _lsp_lava))
    end

    @testset "lava spark2_5: toy prefill past the window" begin
        @test _lsp_err(_lsp_prefill(_lsp_model, _lsp_t_dev, _lsp_lava), _lsp_logits) <= _lsp_tol(_lsp_logits)
        hid = _lsp_prefill(_lsp_model, _lsp_t, _lsp_cpu; hidden=true)
        @test _lsp_err(_lsp_prefill(_lsp_model, _lsp_t_dev, _lsp_lava; hidden=true), hid) <= _lsp_tol(hid)
    end

    # Lava's default KV cache is q8 and the CPU's is f32, so the tight bound
    # holds only with the cache pinned to f32 on both. Under q8 the toy's rows
    # drift by about 1% of max|ref| (worst measured 0.029 at max|ref| 2.8), so
    # the q8 arm gets 2e-2 relative; the window control still moves the
    # logits by 0.44.
    @testset "lava spark2_5: toy greedy decode, K from 6 to 12 with W = 4" begin
        ids_c, rows_c = _lsp_generate(_lsp_model, _lsp_t, _lsp_cpu)
        ids_d, rows_d = withenv(() -> _lsp_generate(_lsp_model, _lsp_t_dev, _lsp_lava), "GESSO_KV_FORMAT" => "f32")
        @test ids_d == ids_c
        @test length(rows_d) == length(rows_c)
        for i in eachindex(rows_c, rows_d)
            @test _lsp_err(rows_d[i], rows_c[i]) <= _lsp_tol(rows_c[i])
        end
        ids_q, rows_q = withenv(() -> _lsp_generate(_lsp_model, _lsp_t_dev, _lsp_lava), "GESSO_KV_FORMAT" => "q8")
        @test ids_q == ids_c
        @test length(rows_q) == length(rows_c)
        for i in eachindex(rows_c, rows_q)
            @test _lsp_err(rows_q[i], rows_c[i]) <= 200 * _lsp_tol(rows_c[i])
        end
    end

    # Residual taps need the CPU window prefill path, so per-layer agreement is
    # checked on the same weights truncated to their first k layers.
    @testset "lava spark2_5: toy truncated to k layers" begin
        for k in 1:_LSP_NL
            m, t = _lsp_bind(_lsp_spec(; layers=k), _lsp_by)
            ref = _lsp_prefill(m, t, _lsp_cpu; hidden=true)
            @test _lsp_err(_lsp_prefill(m, Gesso.to_device(_lsp_lava, t), _lsp_lava; hidden=true), ref) <= _lsp_tol(ref)
        end
    end

    @testset "lava spark2_5: negative controls exceed the bound" begin
        tol = _lsp_tol(_lsp_logits)
        m, t = _lsp_bind(_lsp_spec(; window=_LSP_WIN - 1), _lsp_by)
        @test _lsp_err(_lsp_prefill(m, Gesso.to_device(_lsp_lava, t), _lsp_lava), _lsp_logits) > tol

        m, t = _lsp_bind(_lsp_spec(; rope=_LSP_ROPE_SWA, rope_swa=_LSP_ROPE_FULL), _lsp_by)
        @test _lsp_err(_lsp_prefill(m, Gesso.to_device(_lsp_lava, t), _lsp_lava), _lsp_logits) > tol

        swapped = copy(_lsp_by)
        for i in 0:(_LSP_NL-1)
            swapped["blk.$i.ffn_gate.weight"] = _lsp_by["blk.$i.ffn_up.weight"]
            swapped["blk.$i.ffn_up.weight"] = _lsp_by["blk.$i.ffn_gate.weight"]
        end
        m, t = _lsp_bind(_lsp_spec(), swapped)
        @test _lsp_err(_lsp_prefill(m, Gesso.to_device(_lsp_lava, t), _lsp_lava), _lsp_logits) > tol
    end
end
