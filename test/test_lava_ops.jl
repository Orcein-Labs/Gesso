# Phase 8 item B tests (§LXXXI): Lava operator methods on LavaArray{Float32}
# vs the CPU F64 oracle at the declared atol=1e-3 (F32 vs F64 is not
# bit-identity — the goal fixes the tolerance, per-op exactness would lie).
#
# Mirrors test_cuda_ops.jl op-for-op (same families, same oracles, same
# atol) so the two device paths are compared by the same yardstick. The
# whole file is one named skip when there is no usable Vulkan device; the
# seam tests in test_lava_seam.jl still run without one. LAVA_LOADED and
# VULKAN_OK come from test_lava_seam.jl (include order is load-bearing,
# same as the CUDA seam/ops pair).

using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

using .GessoTestHelpers: approx_eq

# `mat` (family constructor + storage wrap) comes from test_cpu_ops.jl —
# same helper, same include order, no redefinition warning

# device-side zero buffer (Base.zeros would build host memory)
lz(dims...) = fill!(Lava.LavaArray{Float32}(undef, dims...), 0.0f0)

if !LAVA_LOADED || !VULKAN_OK
    _skip(
        "no usable Vulkan device (Lava.vk_context() failed) — Lava op tests skipped (§LXXXI skip law)",
    )
else
    lava = Gesso.LavaBackend()
    wl = Gesso.PrefillWorkload()

    @testset "lava embedding_lookup!: rows land, 0-based ids" begin
        tab = [10.0 11.0 12.0; 20.0 21.0 22.0; 30.0 31.0 32.0; 40.0 41.0 42.0]
        dst = mat(Gesso.Activation, lz(2, 3))
        Gesso.embedding_lookup!(
            lava,
            dst,
            mat(Gesso.EmbeddingTable, Lava.LavaArray{Float32}(tab)),
            [2, 0],
            wl,
        )
        @test approx_eq(
            Array(dst.storage),
            Float32[30.0 31.0 32.0; 10.0 11.0 12.0];
            atol=0.0,
        )
    end

    @testset "lava rmsnorm!: matches CPU F64 within atol=1e-3" begin
        x = [1.0 2.0; 3.0 4.0; -1.5 0.25]
        s = [0.9, 1.1]
        cpu_d = mat(Gesso.Activation, zeros(3, 2))
        Gesso.rmsnorm!(
            Gesso.CPUBackend(),
            cpu_d,
            mat(Gesso.Activation, x),
            mat(Gesso.FrozenParameter, s),
            wl,
        )
        gpu_d = mat(Gesso.Activation, lz(3, 2))
        Gesso.rmsnorm!(
            lava,
            gpu_d,
            mat(Gesso.Activation, Lava.LavaArray{Float32}(x)),
            mat(Gesso.FrozenParameter, Lava.LavaArray{Float32}(s)),
            wl,
        )
        @test approx_eq(Array(gpu_d.storage), Float32.(cpu_d.storage); atol=1e-3)

        # eps knob threads through on device too (§LXXVI semantics, §LXXXI ops)
        gpu_d5 = mat(Gesso.Activation, lz(3, 2))
        Gesso.rmsnorm!(
            lava,
            gpu_d5,
            mat(Gesso.Activation, Lava.LavaArray{Float32}(x)),
            mat(Gesso.FrozenParameter, Lava.LavaArray{Float32}(s)),
            wl;
            eps=1e-5,
        )
        @test maximum(abs, Array(gpu_d5.storage) .- Array(gpu_d.storage)) > 0
    end

    @testset "lava rope!: pairwise rotate on device (position 0 = identity)" begin
        q0 = reshape(Float64[i for i in 1:16], (2, 2, 4))
        k0 = reshape(Float64[i for i in 17:32], (2, 2, 4))
        q = mat(Gesso.Activation, Lava.LavaArray{Float32}(q0))
        k = mat(Gesso.Activation, Lava.LavaArray{Float32}(k0))
        Gesso.rope!(lava, q, k, [0, 1], wl)
        # position 0 unchanged (θ=0 ⇒ identity)
        @test Array(q.storage)[1, :, :] == Float32.(q0[1, :, :])
        # position 1 vs CPU oracle within atol
        cq, ck = mat(Gesso.Activation, copy(q0)), mat(Gesso.Activation, copy(k0))
        Gesso.rope!(Gesso.CPUBackend(), cq, ck, [0, 1], wl)
        @test approx_eq(Array(q.storage), Float32.(cq.storage); atol=1e-3)
        @test approx_eq(Array(k.storage), Float32.(ck.storage); atol=1e-3)

        # GQA: K rotates over ITS OWN head count (fewer heads than Q)
        qg = mat(Gesso.Activation, Lava.LavaArray{Float32}(q0))                 # (2, 2, 4): 2 q-heads
        kg = mat(Gesso.Activation, Lava.LavaArray{Float32}(k0[:, 1:1, :]))     # (2, 1, 4): 1 kv-head
        Gesso.rope!(lava, qg, kg, [0, 1], wl)
        @test Array(kg.storage)[2, 1, :] != Float32.(k0[2, 1, :])              # the single K head rotated at position 1
    end

    @testset "lava softmax!: causal mask + stability on device" begin
        sc = [3.0 1.0 0.5; 2.0 4.0 1.0; 0.1 0.2 0.3]
        scores = mat(Gesso.TemporaryWorkspace, Lava.LavaArray{Float32}(sc))
        out = mat(Gesso.TemporaryWorkspace, lz(3, 3))
        Gesso.softmax!(lava, out, scores, wl)
        got = Array(out.storage)
        @test got[1, 2] == 0.0 && got[1, 3] == 0.0 && got[2, 3] == 0.0          # causal mask
        @test got[1, 1] ≈ 1.0 atol = 1e-6
        @test approx_eq(sum(got; dims=2), Float32.(ones(3, 1)); atol=1e-5)  # rows are distributions
        # decode row (1 query, whole cache) attends to everything
        sd = mat(
            Gesso.TemporaryWorkspace,
            Lava.LavaArray{Float32}(reshape([1.0, 2.0, 3.0], (1, 3))),
        )
        od = mat(Gesso.TemporaryWorkspace, lz(1, 3))
        Gesso.softmax!(lava, od, sd, Gesso.DecodeWorkload())
        @test all(Array(od.storage) .> 0)                                       # nothing masked
        # stability: huge scores do not overflow
        big = mat(Gesso.TemporaryWorkspace, Lava.LavaArray{Float32}(fill(1.0f6, 3, 3)))
        ob = mat(Gesso.TemporaryWorkspace, lz(3, 3))
        Gesso.softmax!(lava, ob, big, wl)
        @test all(isfinite, Array(ob.storage))
    end

    @testset "lava swiglu! + matmul!: match CPU within atol=1e-3" begin
        g = [-1.0, 0.5, 2.0]
        u = [3.0, -0.25, 1.0]
        gpu_d = mat(Gesso.Activation, lz(3))
        Gesso.swiglu!(
            lava,
            gpu_d,
            mat(Gesso.Activation, Lava.LavaArray{Float32}(g)),
            mat(Gesso.Activation, Lava.LavaArray{Float32}(u)),
            wl,
        )
        silu = g ./ (1.0 .+ exp.(-g))
        @test approx_eq(Array(gpu_d.storage), Float32.(silu .* u); atol=1e-6)

        x = [1.0 2.0 3.0; 4.0 5.0 6.0]
        w = [1.0 0.0 1.0; 0.0 1.0 0.0]
        cpu_m = mat(Gesso.Activation, zeros(2, 2))
        Gesso.matmul!(
            Gesso.CPUBackend(),
            cpu_m,
            mat(Gesso.Activation, x),
            mat(Gesso.ProjectionWeight, w),
            wl,
        )
        gpu_m = mat(Gesso.Activation, lz(2, 2))
        Gesso.matmul!(
            lava,
            gpu_m,
            mat(Gesso.Activation, Lava.LavaArray{Float32}(x)),
            mat(Gesso.ProjectionWeight, Lava.LavaArray{Float32}(w)),
            wl,
        )
        @test approx_eq(Array(gpu_m.storage), Float32.(cpu_m.storage); atol=1e-6)
    end

    @testset "lava: host storage through a Lava op is ERR_INVALID_PLAN (§LXX)" begin
        xbad = mat(Gesso.Activation, [1.0 2.0; 3.0 4.0])
        err = try
            Gesso.rmsnorm!(
                lava,
                mat(Gesso.Activation, lz(2, 2)),
                xbad,
                mat(Gesso.FrozenParameter, [1.0, 1.0]),
                wl,
            )
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError
        @test err.code == Gesso.ERR_INVALID_PLAN
        @test occursin("to_device", sprint(showerror, err))
    end

    @testset "lava: quantize!/dequantize! still decline (§LXXXI)" begin
        a = mat(Gesso.Activation, Lava.LavaArray{Float32}([1.0 2.0]))
        p = mat(Gesso.ProjectionWeight, Lava.LavaArray{Float32}([1.0 2.0]))
        @test_throws Gesso.LoweringNotImplemented Gesso.quantize!(lava, p, a, wl)
        @test_throws Gesso.LoweringNotImplemented Gesso.dequantize!(lava, a, p, wl)
    end

    @testset "to_device: F64→F32 copy, source untouched, tied head stays tied" begin
        emb_arr = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        tensors = (
            embedding=mat(Gesso.EmbeddingTable, emb_arr),
            blocks=[(
                wq=mat(Gesso.ProjectionWeight, [1.0 2.0; 3.0 4.0]),
                wk=mat(Gesso.ProjectionWeight, [1.0 2.0; 3.0 4.0]),
                wv=mat(Gesso.ProjectionWeight, [1.0 2.0; 3.0 4.0]),
                wo=mat(Gesso.ProjectionWeight, [1.0 2.0; 3.0 4.0]),
                wgate=mat(Gesso.ProjectionWeight, [1.0 2.0; 3.0 4.0]),
                wup=mat(Gesso.ProjectionWeight, [1.0 2.0; 3.0 4.0]),
                wdown=mat(Gesso.ProjectionWeight, [1.0 2.0; 3.0 4.0]),
                attn_rms=mat(Gesso.FrozenParameter, [1.0, 1.0]),
                ffn_rms=mat(Gesso.FrozenParameter, [1.0, 1.0]),
            )],
            lm_head=nothing,      # filled below as the same object
        )
        tensors = merge(tensors, (lm_head=tensors.embedding,))
        gpu = Gesso.to_device(lava, tensors)
        @test gpu.embedding.storage isa Lava.LavaArray{Float32}
        @test gpu.lm_head === gpu.embedding                     # tied ON DEVICE
        @test Array(gpu.embedding.storage) == Float32.(emb_arr) # values survive F64→F32
        @test tensors.embedding.storage isa Array{Float64}      # CPU source untouched
        @test gpu.blocks[1].wq.storage isa Lava.LavaArray{Float32}
        # final_rms absent → nothing; present → transferred
        @test !haskey(gpu, :final_rms) || gpu.final_rms === nothing
        fr = mat(Gesso.FrozenParameter, [0.5, 0.5])
        gpu2 = Gesso.to_device(lava, merge(tensors, (final_rms=fr,)))
        @test gpu2.final_rms.storage isa Lava.LavaArray{Float32}
    end
end
