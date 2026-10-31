# AMDGPU Gemma 4 ops: segment RMS norm, neox rope, cache put, embedding scale,
# logit softcap, and the attention passes, each against an F64 host oracle on
# KA's CPU backend; then the toy model through prefill and greedy decode
# against the CPU oracle with f32 KV; q8 KV is a typed refusal. Invalid
# workgroup sizes are typed failures and each kernel compiles for the AMD
# targets with no device.
# GPU-UNTESTED: nothing here runs on ROCm.
#
# Tolerance. Every forward bound is 1e-4 * max(1, max|ref|), as in
# test_amdgpu_lfm2.jl. The sliding-window negative control must exceed the
# forward bound.

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
    @test _skip("AMDGPU.jl or KernelAbstractions not loadable — AMDGPU Gemma4 tests skipped")
else
    isdefined(@__MODULE__, :_gemma4_toy) || include(joinpath(@__DIR__, "gemma4_toy.jl"))

    const _AMDG_B = _AMDX.KADeviceBackend(_AMDKA.CPU())
    const _AMDG_CPU = Gesso.CPUBackend()
    const _AMDG_PROMPT = [3, 7, 1, 12, 5, 9, 2]
    const _AMDG_NEW = 6

    _amdg_err(a, b) = maximum(abs, Float64.(Array(a)) .- Float64.(Array(b)); init=0.0)
    _amdg_tol(ref) = 1.0e-4 * max(1.0, maximum(abs, Float64.(Array(ref)); init=0.0))

    @testset "amdgpu Gemma4: segment RMS norm matches the F64 oracle" begin
        rng = MersenneTwister(4101)
        for (N, nheads, hd) in ((5, 3, 8), (4, 2, 7), (1, 1, 1), (3, 4, 32))
            src = randn(rng, Float32, N, nheads * hd)
            w = randn(rng, Float32, hd)
            eps = 1.0e-6
            for use_w in (true, false)
                ref = zeros(Float64, N, nheads * hd)
                for t in 1:N, h in 1:nheads
                    o = (h - 1) * hd
                    x = Float64.(src[t, o+1:o+hd])
                    r = 1 / sqrt(sum(abs2, x) / hd + eps)
                    ref[t, o+1:o+hd] .= x .* r .* (use_w ? Float64.(w) : 1.0)
                end
                dst = zeros(Float32, N, nheads * hd)
                _AMDX._ka_g4_seg_norm!(_AMDG_B.ka, dst, src, w, eps, hd, N, nheads, use_w)
                @test _amdg_err(dst, ref) <= 1.0e-4 * max(1.0, maximum(abs, ref))
            end
        end
    end

    @testset "amdgpu Gemma4: segment RMS norm propagates NaN to its row only" begin
        src = ones(Float32, 2, 6)
        src[2, 3] = NaN32
        dst = zeros(Float32, 2, 6)
        _AMDX._ka_g4_seg_norm!(_AMDG_B.ka, dst, src, nothing, 1.0e-6, 3, 2, 2, false)
        @test isnan(dst[2, 1]) && isnan(dst[2, 2]) && isnan(dst[2, 3])
        @test all(isfinite, dst[1, :])
        @test all(isfinite, dst[2, 4:6])
    end

    @testset "amdgpu Gemma4: neox rope matches the CPU rope, full and partial" begin
        rng = MersenneTwister(4102)
        inv_freq = [1.0 / 10000.0^(2i / 8) for i in 0:3]
        for (n_rot, hd) in ((8, 8), (4, 8))
            nheads, N = 3, 5
            positions = collect(0:N-1) .+ 7
            A = randn(rng, Float32, N, nheads * hd)
            ref = copy(A)
            Gesso.Inference._apply_rope_neox!(_AMDG_CPU, ref, positions, nheads, hd, n_rot, inv_freq[1:n_rot ÷ 2])
            dev = copy(A)
            Gesso.Inference._apply_rope_neox!(_AMDG_B, dev, positions, nheads, hd, n_rot, inv_freq[1:n_rot ÷ 2])
            @test _amdg_err(dev, ref) <= 1.0e-5 * max(1.0, maximum(abs, ref))
        end
    end

    @testset "amdgpu Gemma4: cache put splits heads into the cache layout" begin
        rng = MersenneTwister(4103)
        N, nheads, hd, row0 = 3, 2, 4, 5
        src = randn(rng, Float32, N, nheads * hd)
        dst = zeros(Float32, 10, nheads, hd)
        _AMDX._ka_g4_kv_put!(_AMDG_B.ka, dst, src, row0, nheads, hd)
        for t in 1:N, h in 1:nheads, d in 1:hd
            @test dst[row0 + t - 1, h, d] == src[t, (h - 1) * hd + d]
        end
    end

    @testset "amdgpu Gemma4: embedding scale and logit softcap match the F64 oracle" begin
        rng = MersenneTwister(4104)
        spec = (; embedding_scale=sqrt(16.0), final_logit_softcap=30.0)
        tensors = (; spec)
        x = randn(rng, Float32, 4, 16)
        h = (; storage=copy(x))
        Gesso.Inference._apply_embedding_scale!(_AMDG_B, h, tensors)
        @test _amdg_err(h.storage, Float64.(x) .* 4.0) <= 1.0e-5 * max(1.0, maximum(abs, x))
        z = randn(rng, Float32, 4, 16) .* 20
        l = (; storage=copy(z))
        prev = Gesso.Inference._GEMMA4_SOFTCAP_ENABLED[]
        Gesso.Inference._GEMMA4_SOFTCAP_ENABLED[] = true
        try
            Gesso.Inference._apply_logit_softcap!(_AMDG_B, l, tensors)
        finally
            Gesso.Inference._GEMMA4_SOFTCAP_ENABLED[] = prev
        end
        ref = 30.0 .* tanh.(Float64.(z) ./ 30.0)
        @test _amdg_err(l.storage, ref) <= 1.0e-5 * max(1.0, maximum(abs, ref))
    end

    # The attention passes against a host loop, with a cache offset, GQA and a
    # window, untiled and with one query row per tile.
    function _amdg_attn_ref(Q, Kd, Vd, nh, nkv, hd, qoff, win)
        Nq = size(Q, 1)
        O = zeros(Float64, Nq, nh * hd)
        for t in 1:Nq, h in 1:nh
            kvh = (h - 1) ÷ (nh ÷ nkv) + 1
            r = qoff + t - 1
            ks = (win > 0 ? max(1, r - win + 1) : 1):r
            s = [sum(Float64(Q[t, (h-1)*hd+d]) * Float64(Kd[k, kvh, d]) for d in 1:hd) for k in ks]
            p = exp.(s .- maximum(s))
            p ./= sum(p)
            for d in 1:hd
                O[t, (h-1)*hd+d] = sum(p[i] * Float64(Vd[k, kvh, d]) for (i, k) in enumerate(ks))
            end
        end
        return O
    end

    @testset "amdgpu Gemma4: attention passes match a host loop, tiled and untiled" begin
        rng = MersenneTwister(4105)
        nh, nkv, hd = 4, 2, 16
        for (Nq, qoff, win) in ((9, 1, 0), (9, 1, 4), (5, 8, 0), (5, 8, 3), (1, 12, 5))
            K = qoff + Nq - 1
            Q = randn(rng, Float32, Nq, nh * hd)
            Kd = randn(rng, Float32, K, nkv, hd)
            Vd = randn(rng, Float32, K, nkv, hd)
            ref = _amdg_attn_ref(Q, Kd, Vd, nh, nkv, hd, qoff, win)
            run() = _AMDX._ka_g4_attention(_AMDG_B.ka, Q, Kd, Vd, nh, nkv, hd, qoff, win)
            @test _amdg_err(run(), ref) <= _amdg_tol(ref)
            prev = _AMDX._G4A_SCORE_BYTES[]
            _AMDX._G4A_SCORE_BYTES[] = 4 * nh
            try
                @test _amdg_err(run(), ref) <= _amdg_tol(ref)
            finally
                _AMDX._G4A_SCORE_BYTES[] = prev
            end
        end
    end

    @testset "amdgpu Gemma4: invalid workgroup sizes are typed failures" begin
        Q = zeros(Float32, 2, 8)
        Kd = zeros(Float32, 2, 1, 4)
        dst = zeros(Float32, 2, 8)
        for wg in (0, 48, 512)
            e1 = try
                _AMDX._ka_g4_seg_norm!(_AMDG_B.ka, dst, Q, nothing, 1.0e-6, 4, 2, 2, false; wg)
                nothing
            catch e
                e
            end
            e2 = try
                _AMDX._ka_g4_attention(_AMDG_B.ka, Q, Kd, Kd, 2, 1, 4, 1, 0; wg)
                nothing
            catch e
                e
            end
            @test e1 isa Gesso.GessoError && e1.code === Gesso.ERR_INVALID_PLAN
            @test e2 isa Gesso.GessoError && e2.code === Gesso.ERR_INVALID_PLAN
        end
    end

    @testset "amdgpu Gemma4: compile and link for $(join(first.(AMD_TARGETS), ", "))" begin
        G = amd_global
        for wg in (64, 256)
            cases = (
                (_AMDX._ka_g4_seg_norm_kernel!, (G(Float32, 2), G(Float32, 2), G(Float32, 1), Float32, Int, Int, Bool)),
                (_AMDX._ka_g4_rope_kernel!, (G(Float32, 2), G(Float32, 2), G(Float32, 2), Int, Int, Int, Int, Int)),
                (_AMDX._ka_g4_kv_put_kernel!, (G(Float32, 3), G(Float32, 2), Int, Int, Int, Int, Int)),
                (_AMDX._ka_g4_scores_kernel!, (G(Float32, 2), G(Float32, 2), G(Float32, 3), Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)),
                (_AMDX._ka_g4_softmax_cols_kernel!, (G(Float32, 2), Int)),
                (_AMDX._ka_g4_pv_kernel!, (G(Float32, 2), G(Float32, 2), G(Float32, 3), Int, Int, Int, Int, Int, Int, Int, Int)),
                (_AMDX._ka_g4_add_kernel!, (G(Float32, 2), G(Float32, 2), G(Float32, 2), Int)),
                (_AMDX._ka_g4_copy_kernel!, (G(Float32, 2), G(Float32, 2), Int)),
                (_AMDX._ka_g4_scale_kernel!, (G(Float32, 2), Float32, Int)),
                (_AMDX._ka_g4_softcap_kernel!, (G(Float32, 2), Float32, Int)),
            )
            for (kernel, argtypes) in cases
                res = amd_compile(kernel, wg, (wg * 3,), argtypes)
                @test length(res) == length(AMD_TARGETS)
                @test all(r -> r.bytes > 0, res)
            end
        end
    end

    function _amdg_case(kv)
        spec = Gesso.Inference.spec_from_gguf_kv(kv)
        model, t = _gemma4_bind(spec, _gemma4_toy(spec))
        return (; spec, eps=spec.rms_norm_eps, model, t, t_dev=Gesso.to_device(_AMDG_B, t))
    end

    _amdg_prefill(c, tt, backend) =
        Array(Gesso.reference_prefill(c.model, tt, _AMDG_PROMPT; backend, eps=c.eps))

    function _amdg_generate(c, tt, backend)
        rows = Matrix{Float32}[]
        ids = Gesso.reference_generate(c.model, tt, _AMDG_PROMPT; backend, eps=c.eps,
            max_new_tokens=_AMDG_NEW, eos_ids=(),
            on_rows=(pos, M) -> push!(rows, Float32.(Array(M))))
        return ids, rows
    end

    _amdg_f = _amdg_case(_gemma4_toy_kv())
    _amdg_q = _amdg_case(_gemma4_toy_kv_wide())
    logits_cpu = _amdg_prefill(_amdg_f, _amdg_f.t, _AMDG_CPU)

    @testset "amdgpu Gemma4: import report finds no missing capability" begin
        @test isempty(Gesso.Inference._missing_capabilities(
            Gesso.required_semantics(_amdg_f.spec), _AMDG_B))
    end

    @testset "amdgpu Gemma4: toy prefill matches the CPU oracle" begin
        @test _amdg_err(_amdg_prefill(_amdg_f, _amdg_f.t_dev, _AMDG_B), logits_cpu) <= _amdg_tol(logits_cpu)
    end

    @testset "amdgpu Gemma4: toy greedy decode matches the CPU oracle" begin
        c = _amdg_f
        ids_c, rows_c = withenv(() -> _amdg_generate(c, c.t, _AMDG_CPU), "GESSO_KV_FORMAT" => "f32")
        ids_d, rows_d = withenv(() -> _amdg_generate(c, c.t_dev, _AMDG_B), "GESSO_KV_FORMAT" => "f32")
        @test ids_d == ids_c
        @test length(rows_d) == length(rows_c) == _AMDG_NEW
        for i in eachindex(rows_c, rows_d)
            @test _amdg_err(rows_d[i], rows_c[i]) <= _amdg_tol(rows_c[i])
        end
        # The KA backend refuses a quantized KV cache with a typed error.
        q = _amdg_q
        err = try
            withenv(() -> _amdg_generate(q, q.t_dev, _AMDG_B), "GESSO_KV_FORMAT" => "q8")
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code === Gesso.ERR_INVALID_PLAN
    end

    @testset "amdgpu Gemma4: negative control, sliding window off exceeds the bound" begin
        prev = Gesso.Inference._GEMMA4_WINDOW_MASK_ENABLED[]
        Gesso.Inference._GEMMA4_WINDOW_MASK_ENABLED[] = false
        try
            off = _amdg_prefill(_amdg_f, _amdg_f.t_dev, _AMDG_B)
            @test _amdg_err(off, logits_cpu) > _amdg_tol(logits_cpu)
        finally
            Gesso.Inference._GEMMA4_WINDOW_MASK_ENABLED[] = prev
        end
    end
end
