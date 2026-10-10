# CUDA GDN / hybrid extras vs the CPU F64 oracle (atol=1e-3) and the 0.8B
# Hello ID-gate. Named skip without a functional NVIDIA device.

using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

using .GessoTestHelpers: approx_eq, deterministic_rng

if !CUDA_LOADED || !CUDA.functional()
    _skip(
        "no NVIDIA device — CUDA hybrid / GDN tests skipped (§LXXVII skip law)",
    )
else
    cuda = Gesso.CUDABackend()
    cpu = Gesso.CPUBackend()
    wl = Gesso.PrefillWorkload()
    Inf_ = Gesso.Inference

    @testset "cuda to_device keeps spec and rope metadata" begin
        emb = Gesso.EmbeddingTable(; shape=(4, 2), storage=ones(4, 2))
        blk = (;
            mixer=:linear_attention,
            attn_rms=Gesso.FrozenParameter(; shape=(2,), storage=ones(2)),
        )
        policy = Gesso.RoPEPolicy(; rotary_dim=4, theta=1.0e7)
        ts = (;
            embedding=emb,
            blocks=[blk],
            lm_head=emb,
            spec=(; family=:probe),
            rope=policy,
        )
        gpu = Gesso.to_device(cuda, ts)
        @test gpu.spec === ts.spec
        @test gpu.rope === ts.rope
        @test gpu.blocks[1].mixer === :linear_attention
        @test gpu.embedding.storage isa CUDA.CuArray{Float32}
    end

    @testset "cuda F16 projection weights: placement and matmul" begin
        # placement: a >=1M-element projection is F16 on device by default,
        # F32 under GESSO_GPU_WEIGHT_DTYPE=f32, and the host copy stays F32
        big = Gesso.ProjectionWeight(; shape=(1024, 1024), storage=ones(Float32, 1024, 1024))
        ts = (; embedding=big, blocks=[(; w=big)], lm_head=big)
        withenv("GESSO_GPU_WEIGHT_DTYPE" => "f16") do
            gpu = Gesso.to_device(cuda, ts)
            @test gpu.lm_head.storage isa CUDA.CuArray{Float16}
            @test big.storage isa Array{Float32}
        end
        withenv("GESSO_GPU_WEIGHT_DTYPE" => "f32") do
            gpu = Gesso.to_device(cuda, ts)
            @test gpu.lm_head.storage isa CUDA.CuArray{Float32}
        end

        # matmul: F16 device weight vs a host reference computed from the same
        # F16-rounded weight, F32 accumulate; only summation order differs
        m, k, n = 37, 45, 3
        W0 = reshape(Float32.(cos.(1:m*k)) ./ 4, m, k)
        x0 = reshape(Float32.(sin.(1:n*k)), n, k)
        W16 = Float64.(Float16.(W0))
        ref = Float64.(x0) * transpose(W16)
        wF = Gesso.ProjectionWeight(; shape=(m, k), storage=Float16.(CUDA.CuArray{Float32}(W0)))
        dst = mat(Gesso.Activation, CUDA.zeros(Float32, n, m))
        x = mat(Gesso.Activation, CuArray{Float32}(x0))
        Gesso.matmul!(cuda, dst, x, wF, Gesso.DecodeWorkload())
        @test approx_eq(Float64.(Array(dst.storage)), ref; atol=1e-4)
    end

    @testset "cuda F16 matmul: decode row and k/m off the warp grid" begin
        # n=1 is the decode shape; m=70 spans three 32-row blocks with a
        # partial last block, and k=300 spans the 8 k-slices unevenly
        for (m, k, n) in ((70, 300, 1), (513, 2051, 1), (33, 8, 2))
            W0 = reshape(Float32.(cos.(1:m*k)) ./ 4, m, k)
            x0 = reshape(Float32.(sin.(1:n*k)), n, k)
            ref = Float64.(x0) * transpose(Float64.(Float16.(W0)))
            wF = Gesso.ProjectionWeight(; shape=(m, k), storage=Float16.(CUDA.CuArray{Float32}(W0)))
            dst = mat(Gesso.Activation, CUDA.zeros(Float32, n, m))
            x = mat(Gesso.Activation, CuArray{Float32}(x0))
            Gesso.matmul!(cuda, dst, x, wF, Gesso.DecodeWorkload())
            @test approx_eq(Float64.(Array(dst.storage)), ref; atol=1e-4)
        end
    end

    @testset "cuda rmsnorm! decode row wider than one thread" begin
        rng = deterministic_rng(11)
        x0 = randn(rng, 1, 300)
        s0 = randn(rng, 300)
        cpu_d = mat(Gesso.Activation, zeros(1, 300))
        Gesso.rmsnorm!(
            cpu, cpu_d, mat(Gesso.Activation, x0), mat(Gesso.FrozenParameter, s0), wl,
        )
        gpu_d = mat(Gesso.Activation, CUDA.zeros(Float32, 1, 300))
        Gesso.rmsnorm!(
            cuda,
            gpu_d,
            mat(Gesso.Activation, CuArray{Float32}(x0)),
            mat(Gesso.FrozenParameter, CuArray{Float32}(s0)),
            Gesso.DecodeWorkload(),
        )
        @test approx_eq(Array(gpu_d.storage), Float32.(cpu_d.storage); atol=1e-4)
    end

    @testset "cuda rope!: inv_freq + rotary_dim match CPU" begin
        q0 = reshape(Float64[i for i in 1:32], (2, 2, 8))
        k0 = reshape(Float64[i for i in 33:64], (2, 2, 8))
        inv = Gesso.rope_inv_freq(Gesso.RoPEPolicy(; rotary_dim=4, theta=10000.0), 8)
        cq, ck = mat(Gesso.Activation, copy(q0)), mat(Gesso.Activation, copy(k0))
        Gesso.rope!(
            cpu, cq, ck, [0, 1], wl;
            theta=10000.0, inv_freq=inv, interleaved=false, rotary_dim=4,
        )
        q = mat(Gesso.Activation, CuArray{Float32}(q0))
        k = mat(Gesso.Activation, CuArray{Float32}(k0))
        Gesso.rope!(
            cuda, q, k, [0, 1], wl;
            theta=10000.0, inv_freq=inv, interleaved=false, rotary_dim=4,
        )
        @test approx_eq(Array(q.storage), Float32.(cq.storage); atol=1e-3)
        @test approx_eq(Array(k.storage), Float32.(ck.storage); atol=1e-3)
        # unrotated tail of the head stays put
        @test Array(q.storage)[1, 1, 5:8] == Float32.(q0[1, 1, 5:8])
    end

    @testset "cuda qk_norm: 3-D rmsnorm matches CPU" begin
        rng = deterministic_rng(3)
        x0 = randn(rng, 2, 2, 4)
        s0 = randn(rng, 4)
        cpu_d = mat(Gesso.Activation, zeros(size(x0)))
        Gesso.rmsnorm!(
            cpu, cpu_d, mat(Gesso.Activation, x0), mat(Gesso.FrozenParameter, s0), wl,
        )
        gpu_d = mat(Gesso.Activation, CUDA.zeros(Float32, size(x0)))
        Gesso.rmsnorm!(
            cuda,
            gpu_d,
            mat(Gesso.Activation, CuArray{Float32}(x0)),
            mat(Gesso.FrozenParameter, CuArray{Float32}(s0)),
            wl,
        )
        @test approx_eq(Array(gpu_d.storage), Float32.(cpu_d.storage); atol=1e-3)
    end

    @testset "cuda sigmoid_mul / silu_mul / gdn gates match CPU" begin
        rng = deterministic_rng(5)
        src = randn(rng, 8)
        gate = randn(rng, 8)
        dst_c = zeros(8)
        Inf_._sigmoid_mul!(cpu, dst_c, src, gate)
        dst_g = CUDA.zeros(Float32, 8)
        Inf_._sigmoid_mul!(cuda, dst_g, CuArray{Float32}(src), CuArray{Float32}(gate))
        @test approx_eq(Array(dst_g), Float32.(dst_c); atol=1e-3)

        dst_c .= 0
        Inf_._silu_mul!(cpu, dst_c, src, gate)
        CUDA.fill!(dst_g, 0)
        Inf_._silu_mul!(cuda, dst_g, CuArray{Float32}(src), CuArray{Float32}(gate))
        @test approx_eq(Array(dst_g), Float32.(dst_c); atol=1e-3)

        seq, nv = 3, 4
        a = randn(rng, seq, nv)
        b = randn(rng, seq, nv)
        A_log = randn(rng, nv)
        dt = randn(rng, nv)
        g_c = zeros(seq, nv)
        beta_c = zeros(seq, nv)
        Inf_._gdn_gate_params!(cpu, g_c, beta_c, a, b, A_log, dt)
        g_g = CUDA.zeros(Float32, seq, nv)
        beta_g = CUDA.zeros(Float32, seq, nv)
        Inf_._gdn_gate_params!(
            cuda, g_g, beta_g, CuArray{Float32}(a), CuArray{Float32}(b),
            CuArray{Float32}(A_log), CuArray{Float32}(dt),
        )
        @test approx_eq(Array(g_g), Float32.(g_c); atol=1e-3)
        @test approx_eq(Array(beta_g), Float32.(beta_c); atol=1e-3)
    end

    @testset "cuda prefill gated norm writes merged heads" begin
        rng = deterministic_rng(11)
        seq, nv, dv = 5, 3, 8
        core = randn(rng, Float32, seq, nv, dv)
        z = randn(rng, Float32, seq, nv * dv)
        w = randn(rng, Float32, dv)
        eps = 1f-6
        ref = zeros(Float32, seq, nv * dv)
        for t in 1:seq, hh in 1:nv
            inv = 1 / sqrt(sum(abs2, core[t, hh, :]) / dv + eps)
            for j in 1:dv
                zv = z[t, (hh - 1) * dv + j]
                ref[t, (hh - 1) * dv + j] = core[t, hh, j] * inv * w[j] * zv / (1 + exp(-zv))
            end
        end
        @test !Inf_._gdn_gnorm_merge!(cpu, zeros(Float32, seq, nv * dv), core, z, (; storage=w), eps)
        merged = CUDA.zeros(Float32, seq, nv * dv)
        @test Inf_._gdn_gnorm_merge!(cuda, merged, CuArray(core), CuArray(z), (; storage=CuArray(w)), eps)
        @test approx_eq(Array(merged), ref; atol=1e-5)
    end

    @testset "cuda causal_conv1d_silu matches CPU and keeps decode state" begin
        rng = deterministic_rng(7)
        seq, channels, k = 4, 6, 4
        x = randn(rng, seq, channels)
        w = randn(rng, channels, 1, k)
        st = randn(rng, channels, k - 1)
        dst_c = zeros(seq, channels)
        st_c = copy(st)
        Inf_._causal_conv1d_silu!(cpu, dst_c, x, w, st_c)
        dst_g = CUDA.zeros(Float32, seq, channels)
        st_g = CuArray{Float32}(st)
        Inf_._causal_conv1d_silu!(
            cuda, dst_g, CuArray{Float32}(x), CuArray{Float32}(w), st_g,
        )
        @test approx_eq(Array(dst_g), Float32.(dst_c); atol=1e-3)
        @test approx_eq(Array(st_g), Float32.(st_c); atol=1e-3)

        x1 = randn(rng, 1, channels)
        dst1_c = zeros(1, channels)
        Inf_._causal_conv1d_silu!(cpu, dst1_c, x1, w, st_c)
        dst1_g = CUDA.zeros(Float32, 1, channels)
        Inf_._causal_conv1d_silu!(
            cuda, dst1_g, CuArray{Float32}(x1), CuArray{Float32}(w), st_g,
        )
        @test approx_eq(Array(dst1_g), Float32.(dst1_c); atol=1e-3)
        @test approx_eq(Array(st_g), Float32.(st_c); atol=1e-3)
    end

    @testset "cuda gated_delta_rule grouped and tiled match CPU" begin
        rng = deterministic_rng(11)
        seq, nk, nv, dk, dv = 3, 2, 4, 8, 8
        q = randn(rng, seq, nk, dk)
        k = randn(rng, seq, nk, dk)
        v = randn(rng, seq, nv, dv)
        g = -rand(rng, seq, nv)
        beta = rand(rng, seq, nv)
        for tiled in (false, true)
            out_c = zeros(seq, nv, dv)
            st_c = zeros(nv, dk, dv)
            Inf_._gated_delta_rule!(
                cpu, out_c, q, k, v, g, beta, st_c; l2norm=true, v_head_tiled=tiled,
            )
            out_g = CUDA.zeros(Float32, seq, nv, dv)
            st_g = CUDA.zeros(Float32, nv, dk, dv)
            Inf_._gated_delta_rule!(
                cuda,
                out_g,
                CuArray{Float32}(q),
                CuArray{Float32}(k),
                CuArray{Float32}(v),
                CuArray{Float32}(g),
                CuArray{Float32}(beta),
                st_g;
                l2norm=true,
                v_head_tiled=tiled,
            )
            @test approx_eq(Array(out_g), Float32.(out_c); atol=1e-3)
            @test approx_eq(Array(st_g), Float32.(st_c); atol=2e-3)
        end
    end

    @testset "Qwen3.5-0.8B CUDA Hello ID-gate" begin
        dir = get(ENV, "GESSO_QWEN35_08B_DIR", "")
        cfg = joinpath(dir, "config.json")
        if isempty(dir) || !isfile(cfg)
            @info "skip 0.8B CUDA ID-gate: set GESSO_QWEN35_08B_DIR to a Qwen3.5-0.8B snapshot"
            return
        end
        spec = Gesso.architecture_spec(cfg)
        @test :linear_attention in Gesso.required_semantics(spec)
        @test Gesso.supports(cuda, :linear_attention)
        miss = Gesso.import_report(spec; backend=cuda)
        @test occursin("Failure boundary: none", miss)

        model, tensors, _ = Gesso.load_architecture(dir)
        gpu = Gesso.to_device(cuda, tensors)
        @test gpu.spec === tensors.spec
        ids = Gesso.reference_generate(
            model,
            gpu,
            [9419];
            max_new_tokens=8,
            eps=spec.rms_norm_eps,
            backend=cuda,
        )
        @test ids[1] == 9419
        @test ids[2:9] == [11, 271, 40, 1044, 3133, 440, 264, 12654]
    end
end
