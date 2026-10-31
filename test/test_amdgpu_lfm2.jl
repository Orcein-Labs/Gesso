# AMDGPU LFM2 / LFM2.5 ops: the gated short convolution against the F64 loop
# with history carried across calls, then the dense and MoE toys through
# prefill and greedy decode on KA's CPU backend against the CPU oracle. Invalid
# workgroup sizes are typed failures and each kernel compiles for the AMD
# targets with no device. GPU-UNTESTED: nothing here runs on ROCm.
#
# Tolerance. The convolution bound is 1e-5 * max(1, |ref|); the forward bound is
# 1e-4 * max(1, max|ref|), the same bounds as test_lava_lfm25.jl. The q8 KV
# arm gets 200x, as there; q4h is a typed refusal. The negative controls
# (reversed taps, reversed router rows) must exceed the forward bound.

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
    @test _skip("AMDGPU.jl or KernelAbstractions not loadable — AMDGPU LFM2 tests skipped")
else
    isdefined(@__MODULE__, :_lfm_toy) || include(joinpath(@__DIR__, "lfm25_toy.jl"))

    const _AMDL_B = _AMDX.KADeviceBackend(_AMDKA.CPU())
    const _AMDL_CPU = Gesso.CPUBackend()
    const _AMDL_PROMPT = [3, 7, 1, 12, 5, 9, 2]
    const _AMDL_NEW = 6

    _amdl_err(a, b) = maximum(abs, Float64.(Array(a)) .- Float64.(Array(b)); init=0.0)
    _amdl_tol(ref) = 1.0e-4 * max(1.0, maximum(abs, Float64.(Array(ref)); init=0.0))

    # F64 oracle for one short-convolution call; updates the history in place.
    function _amdl_short_conv_oracle(bcx, w, hist)
        seq = size(bcx, 1)
        dim = size(w, 1)
        width = size(w, 2)
        out = zeros(Float64, seq, dim)
        h = Float64.(hist)
        # Sequential in time: each token sees the history the previous ones left.
        for t in 1:seq, c in 1:dim
            bx = Float64(bcx[t, c]) * Float64(bcx[t, 2dim + c])
            y = 0.0
            for p in 1:(width - 1)
                y += h[c, p] * Float64(w[c, p])
            end
            y += bx * Float64(w[c, width])
            out[t, c] = Float64(bcx[t, dim + c]) * y
            for p in 1:(width - 2)
                h[c, p] = h[c, p + 1]
            end
            h[c, width - 1] = bx
        end
        return out, h
    end

    @testset "amdgpu LFM2: short convolution matches the F64 loop" begin
        rng = MersenneTwister(2028)
        dim, width = 70, 3
        w = randn(rng, Float32, dim, width)
        hist_d = zeros(Float32, dim, width - 1)
        hist_ref = zeros(Float64, dim, width - 1)
        for seq in (5, 1, 1, 4)
            bcx = randn(rng, Float32, seq, 3dim)
            ref, hist_ref = _amdl_short_conv_oracle(bcx, w, hist_ref)
            got = zeros(Float32, seq, dim)
            _AMDX._ka_short_conv!(_AMDL_B.ka, got, bcx, w, hist_d)
            @test _amdl_err(got, ref) <= 1.0e-5 * max(1.0, maximum(abs, ref))
            @test _amdl_err(hist_d, hist_ref) <= 1.0e-5 * max(1.0, maximum(abs, hist_ref))
        end
    end

    @testset "amdgpu LFM2: short convolution edge sizes and NaN" begin
        rng = MersenneTwister(2029)
        # Size one in every dimension that can be one.
        w1 = randn(rng, Float32, 1, 2)
        h1 = zeros(Float32, 1, 1)
        b1 = randn(rng, Float32, 1, 3)
        ref, _ = _amdl_short_conv_oracle(b1, w1, h1)
        got = zeros(Float32, 1, 1)
        _AMDX._ka_short_conv!(_AMDL_B.ka, got, b1, w1, h1)
        @test _amdl_err(got, ref) <= 1.0e-5 * max(1.0, maximum(abs, ref))
        # Odd channel count with a workgroup larger than the channel count.
        dim = 13
        w = randn(rng, Float32, dim, 4)
        h = zeros(Float32, dim, 3)
        bcx = randn(rng, Float32, 3, 3dim)
        ref, _ = _amdl_short_conv_oracle(bcx, w, zeros(Float64, dim, 3))
        got = zeros(Float32, 3, dim)
        _AMDX._ka_short_conv!(_AMDL_B.ka, got, bcx, w, h; wg=64)
        @test _amdl_err(got, ref) <= 1.0e-5 * max(1.0, maximum(abs, ref))
        # A NaN in one channel stays in that channel and does not leak.
        bn = copy(bcx)
        bn[2, 5] = NaN32
        got = zeros(Float32, 3, dim)
        _AMDX._ka_short_conv!(_AMDL_B.ka, got, bn, w, zeros(Float32, dim, 3))
        @test isnan(got[2, 5]) && isnan(got[3, 5])
        @test all(isfinite, got[:, [1:4; 6:dim]])
    end

    @testset "amdgpu LFM2: invalid workgroup sizes are typed failures" begin
        w = ones(Float32, 4, 3)
        h = zeros(Float32, 4, 2)
        b = zeros(Float32, 2, 12)
        m = zeros(Float32, 2, 4)
        for wg in (0, 48, 512)
            err = try
                _AMDX._ka_short_conv!(_AMDL_B.ka, m, b, w, h; wg)
                nothing
            catch e
                e
            end
            @test err isa Gesso.GessoError && err.code === Gesso.ERR_INVALID_PLAN
        end
    end

    @testset "amdgpu LFM2: compile and link for $(join(first.(AMD_TARGETS), ", "))" begin
        G = amd_global
        for wg in (64, 256)
            res = amd_compile(
                _AMDX._ka_short_conv_kernel!,
                wg,
                (wg * 3,),
                (G(Float32, 2), G(Float32, 2), G(Float32, 2), G(Float32, 2), Int, Int, Int),
            )
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)
        end
    end

    for moe in (false, true)
        name = moe ? "MoE" : "dense"
        spec = _lfm_inf.spec_from_gguf_kv(_lfm_toy_kv(moe; width=128))
        eps = spec.rms_norm_eps
        by = _lfm_toy(spec)
        model, t = _lfm_bind(spec, by)
        t_dev = Gesso.to_device(_AMDL_B, t)
        prefill(m, tt, backend) = Array(Gesso.reference_prefill(m, tt, _AMDL_PROMPT; backend, eps))
        function generate(tt, backend)
            rows = Matrix{Float32}[]
            ids = Gesso.reference_generate(model, tt, _AMDL_PROMPT; backend, eps,
                max_new_tokens=_AMDL_NEW, eos_ids=(),
                on_rows=(pos, M) -> push!(rows, Float32.(Array(M))))
            return ids, rows
        end
        logits = prefill(model, t, _AMDL_CPU)

        @testset "amdgpu LFM2 $name: import report finds no missing capability" begin
            @test isempty(_lfm_inf._missing_capabilities(Gesso.required_semantics(spec), _AMDL_B))
        end

        @testset "amdgpu LFM2 $name: toy prefill" begin
            @test _amdl_err(prefill(model, t_dev, _AMDL_B), logits) <= _amdl_tol(logits)
        end

        @testset "amdgpu LFM2 $name: toy greedy decode" begin
            ids_c, rows_c = generate(t, _AMDL_CPU)
            for (fmt, k) in (("f32", 1), ("q8", 200))
                ids_d, rows_d = withenv(() -> generate(t_dev, _AMDL_B), "GESSO_KV_FORMAT" => fmt)
                @test ids_d == ids_c
                @test length(rows_d) == length(rows_c) == _AMDL_NEW
                for i in eachindex(rows_c, rows_d)
                    @test _amdl_err(rows_d[i], rows_c[i]) <= k * _amdl_tol(rows_c[i])
                end
            end
            # q4h has no AMDGPU reader and is a typed refusal.
            err = try
                withenv(() -> generate(t_dev, _AMDL_B), "GESSO_KV_FORMAT" => "q4h")
                nothing
            catch e
                e
            end
            @test err isa Gesso.GessoError && err.code === Gesso.ERR_INVALID_PLAN
        end

        @testset "amdgpu LFM2 $name: negative controls exceed the bound" begin
            tol = _amdl_tol(logits)
            reversed = deepcopy(by)
            for (n, w) in reversed
                endswith(n, "shortconv.conv.weight") && (reversed[n] = w[:, end:-1:1])
            end
            m, tt = _lfm_bind(spec, reversed)
            @test _amdl_err(prefill(m, Gesso.to_device(_AMDL_B, tt), _AMDL_B), logits) > tol
            if moe
                permuted = deepcopy(by)
                for (n, w) in permuted
                    endswith(n, "ffn_gate_inp.weight") && (permuted[n] = w[end:-1:1, :])
                end
                m, tt = _lfm_bind(spec, permuted)
                @test _amdl_err(prefill(m, Gesso.to_device(_AMDL_B, tt), _AMDL_B), logits) > tol
            end
        end
    end
end
