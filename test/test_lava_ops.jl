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

    @testset "lava rmsnorm!: Float32 squares, Float64 row accumulation" begin
        max_ulp_distance = function (a, b)
            size(a) == size(b) || return typemax(Int64)
            all(isfinite, a) && all(isfinite, b) || return typemax(Int64)
            largest = Int64(0)
            for i in eachindex(a, b)
                a_bits = reinterpret(UInt32, a[i])
                b_bits = reinterpret(UInt32, b[i])
                a_key = (a_bits & 0x80000000) == 0 ?
                    (a_bits | 0x80000000) : ~a_bits
                b_key = (b_bits & 0x80000000) == 0 ?
                    (b_bits | 0x80000000) : ~b_bits
                largest = max(largest, abs(Int64(a_key) - Int64(b_key)))
            end
            return largest
        end

        cpu_reference = function (xhost, scalehost; eps = 1e-6)
            xcpu = copy(xhost)
            scpu = copy(scalehost)
            dst_cpu = mat(Gesso.Activation, zeros(Float32, size(xhost)))
            Gesso.rmsnorm!(Gesso.CPUBackend(), dst_cpu,
                mat(Gesso.Activation, xcpu), mat(Gesso.FrozenParameter, scpu), wl;
                eps)
            @test isequal(xcpu, xhost)
            @test isequal(scpu, scalehost)
            return copy(dst_cpu.storage)
        end

        lava_run = function (xhost, scalehost; eps = 1e-6)
            xdev = Lava.LavaArray{Float32}(xhost)
            sdev = Lava.LavaArray{Float32}(scalehost)
            dst = mat(Gesso.Activation, lz(size(xhost)...))
            Gesso.rmsnorm!(lava, dst, mat(Gesso.Activation, xdev),
                mat(Gesso.FrozenParameter, sdev), wl; eps)
            return Array(dst.storage), xdev, sdev, dst
        end

        make_input = function (dims)
            n = prod(dims)
            vals = Float32.(mod.(collect(1:n), 37) .- 18) ./ 13.0f0
            return reshape(vals, dims)
        end
        make_scale = d -> collect(range(0.5f0, 1.5f0; length = d))

        # Widths below 64, ragged widths above 64, more than 64 leading rows,
        # and flattened token/head axes all take the same one-workgroup-per-row
        # path. CPU is run with Float32 storage so its Float64 sum is the oracle.
        cases = (
            (make_input((1, 17)), make_scale(17), 1e-6),
            (make_input((3, 17)), make_scale(17), 1e-5),
            (make_input((65, 67)), make_scale(67), 1e-6),
            (make_input((1, 3, 17)), make_scale(17), 1e-5),
            (make_input((2, 3, 70)), make_scale(70), 1e-6),
        )
        first_head_ulp = typemax(Int64)
        for (xhost, scalehost, eps) in cases
            expected = cpu_reference(xhost, scalehost; eps)
            got, xdev, sdev, _ = lava_run(xhost, scalehost; eps)
            @test size(got) == size(xhost)
            @test all(isfinite, got)
            @test max_ulp_distance(got, expected) <= 1
            @test isequal(Array(xdev), xhost)
            @test isequal(Array(sdev), scalehost)
            if size(xhost) == (1, 3, 17)
                first_head_ulp = max_ulp_distance(got[1, 1, :], expected[1, 1, :])
            end
        end
        @test first_head_ulp <= 1

        # Repeated first rows remain bit-identical as the leading extent crosses
        # one workgroup and then 64 workgroups.
        base = make_input((1, 67))
        base_scale = make_scale(67)
        base_out, _, _, _ = lava_run(base, base_scale)
        for rows in (2, 65)
            repeated = repeat(base, rows, 1)
            repeated_out, _, _, _ = lava_run(repeated, base_scale)
            @test isequal(vec(repeated_out[1, :]), vec(base_out[1, :]))
        end

        # The four large squares are finite Float32 values, while their
        # Float32 sum overflows. Float64 accumulation keeps the row valid.
        wide = zeros(Float32, 3, 65)
        high = ldexp(1.0f0, 63)
        low = ldexp(1.0f0, -60)
        wide[1, 1:4] .= Float32[high, -high, low, -low]
        wide[2, 1:4] .= high
        wide[3, 1:5] .= Float32[low, -ldexp(1.0f0, -30), 1.0f0,
            -ldexp(1.0f0, 30), low]
        @test isfinite(abs2(high))
        f32_sum = Float32(0)
        for value in wide[2, 1:4]
            f32_sum += abs2(value)
        end
        @test !isfinite(f32_sum)
        wide_scale = ones(Float32, 65)
        expected_wide = cpu_reference(wide, wide_scale)
        wide_out, _, _, _ = lava_run(wide, wide_scale)
        @test all(isfinite, expected_wide)
        @test all(isfinite, wide_out)
        @test max_ulp_distance(wide_out, expected_wide) <= 1

        # A normal F32 input produces a subnormal F32 square. The row's unit
        # component keeps its denominator well-conditioned; this checks the
        # declared square precision without assuming arbitrary denormal output
        # behavior across Vulkan devices.
        subnormal_square_input = ldexp(1.0f0, -70)
        subnormal_square = abs2(subnormal_square_input)
        @test subnormal_square > 0.0f0 && subnormal_square < floatmin(Float32)
        subnormal_x = zeros(Float32, 2, 65)
        subnormal_x[:, 1] .= 1.0f0
        subnormal_x[:, 2] .= subnormal_square_input
        subnormal_scale = ones(Float32, 65)
        subnormal_expected = cpu_reference(subnormal_x, subnormal_scale)
        subnormal_out, _, _, _ = lava_run(subnormal_x, subnormal_scale)
        @test max_ulp_distance(subnormal_out, subnormal_expected) <= 1

        # Public in-place operation aliases the destination safely and leaves
        # the scale intact; its values equal the out-of-place CPU reference.
        alias_input = make_input((2, 67))
        alias_scale_host = make_scale(67)
        alias_expected = cpu_reference(alias_input, alias_scale_host)
        out_of_place, _, _, _ = lava_run(alias_input, alias_scale_host)
        alias_storage = Lava.LavaArray{Float32}(alias_input)
        alias_scale = Lava.LavaArray{Float32}(alias_scale_host)
        alias = mat(Gesso.Activation, alias_storage)
        returned = Gesso.rmsnorm!(lava, alias, alias,
            mat(Gesso.FrozenParameter, alias_scale), wl)
        @test returned === alias
        @test isequal(Array(alias.storage), out_of_place)
        @test max_ulp_distance(Array(alias.storage), alias_expected) <= 1
        @test isequal(Array(alias_scale), alias_scale_host)
    end

    @testset "lava rmsnorm!: invalid dimensions and prefill errors stay typed" begin
        capture_error = f -> try
            f()
            nothing
        catch err
            err
        end

        xhost = reshape(Float32.(1:34) ./ 11.0f0, 2, 17)
        scalehost = ones(Float32, 17)
        for eps in (0.0, -1.0, NaN, Inf, 1e-100)
            xdev = Lava.LavaArray{Float32}(xhost)
            dst = mat(Gesso.Activation, lz(2, 17))
            err = capture_error(() -> Gesso.rmsnorm!(lava, dst,
                mat(Gesso.Activation, xdev),
                mat(Gesso.FrozenParameter, Lava.LavaArray{Float32}(scalehost)), wl;
                eps))
            @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
        end

        wrong_scale = Lava.LavaArray{Float32}(ones(Float32, 16))
        scale_dst = mat(Gesso.Activation, lz(2, 17))
        scale_err = capture_error(() -> Gesso.rmsnorm!(lava, scale_dst,
            mat(Gesso.Activation, Lava.LavaArray{Float32}(xhost)),
            mat(Gesso.FrozenParameter, wrong_scale), wl))
        @test scale_err isa Gesso.GessoError && scale_err.code == Gesso.ERR_INVALID_PLAN

        mismatched_dst = mat(Gesso.Activation, Lava.LavaArray{Float32}(undef, 2, 16))
        shape_err = capture_error(() -> Gesso.rmsnorm!(lava, mismatched_dst,
            mat(Gesso.Activation, Lava.LavaArray{Float32}(xhost)),
            mat(Gesso.FrozenParameter, Lava.LavaArray{Float32}(scalehost)), wl))
        @test shape_err isa Gesso.GessoError && shape_err.code == Gesso.ERR_INVALID_PLAN

        for dims in ((0, 17), (2, 0))
            xdev = Lava.LavaArray{Float32}(undef, dims...)
            dst = mat(Gesso.Activation, Lava.LavaArray{Float32}(undef, dims...))
            sdev = Lava.LavaArray{Float32}(undef, dims[end])
            err = capture_error(() -> Gesso.rmsnorm!(lava, dst,
                mat(Gesso.Activation, xdev), mat(Gesso.FrozenParameter, sdev), wl))
            @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
        end

        for bad_value in (Float32(NaN), Float32(Inf), 1.0f20)
            bad = fill(bad_value, 2, 65)
            dst = mat(Gesso.Activation, lz(2, 65))
            err = capture_error(() -> Gesso.rmsnorm!(lava, dst,
                mat(Gesso.Activation, Lava.LavaArray{Float32}(bad)),
                mat(Gesso.FrozenParameter, Lava.LavaArray{Float32}(ones(Float32, 65))), wl))
            @test err isa Gesso.GessoError && err.code == Gesso.ERR_NUMERICAL_INSTABILITY
        end

        bad_scale = ones(Float32, 65)
        bad_scale[1] = Float32(Inf)
        scale_value_err = capture_error(() -> Gesso.rmsnorm!(lava,
            mat(Gesso.Activation, lz(2, 65)),
            mat(Gesso.Activation, Lava.LavaArray{Float32}(ones(Float32, 2, 65))),
            mat(Gesso.FrozenParameter, Lava.LavaArray{Float32}(bad_scale)), wl))
        @test scale_value_err isa Gesso.GessoError &&
            scale_value_err.code == Gesso.ERR_NUMERICAL_INSTABILITY

        # Decode deliberately retains its existing unchecked asynchronous
        # policy: the kernel leaves a NaN marker and the public call returns.
        decode_dst = mat(Gesso.Activation, lz(2, 65))
        decode_result = Gesso.rmsnorm!(lava, decode_dst,
            mat(Gesso.Activation, Lava.LavaArray{Float32}(fill(Float32(Inf), 2, 65))),
            mat(Gesso.FrozenParameter, Lava.LavaArray{Float32}(ones(Float32, 65))),
            Gesso.DecodeWorkload())
        @test decode_result === decode_dst
        @test all(isnan, Array(decode_dst.storage))

        # Float64 device storage remains outside Lava's public arithmetic route.
        f64_dst = mat(Gesso.Activation, Lava.LavaArray{Float64}(undef, 2, 17))
        f64_err = capture_error(() -> Gesso.rmsnorm!(lava, f64_dst,
            mat(Gesso.Activation, Lava.LavaArray{Float64}(ones(2, 17))),
            mat(Gesso.FrozenParameter, Lava.LavaArray{Float64}(ones(17))), wl))
        @test f64_err isa Gesso.GessoError && f64_err.code == Gesso.ERR_INVALID_PLAN
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

    @testset "lava scalar attention sums: repeatable across subgroup boundaries" begin
        ext = Base.get_extension(Gesso, :GessoLavaExt)
        for n in (63, 64, 65, 127, 128, 129)
            x = zeros(Float32, 1, n)
            x[1] = 1.0f0
            x[33] = ldexp(1.0f0, -24)
            n >= 65 && (x[65] = ldexp(1.0f0, -24))
            device_x = Lava.LavaArray(x)
            # The fixed tree rounds each half-ULP addition to one. An atomic
            # sum can add the two small subgroup partials first instead.
            for _ in 1:64
                got = ext._lava_dimreduce(identity, +, device_x, 2, 0.0f0)
                @test only(Array(got)) === 1.0f0
            end
        end

        for n in (65, 127, 128)
            sc = fill(-50.0f0, 1, n)
            sc[1] = 0.0f0
            sc[33] = sc[65] = log(ldexp(1.0f0, -24))
            scores = mat(Gesso.TemporaryWorkspace, Lava.LavaArray(sc))
            out = mat(Gesso.TemporaryWorkspace, lz(1, n))
            reference = exp.(Float64.(sc))
            reference ./= sum(reference)
            Gesso.softmax!(lava, out, scores, wl)
            first_result = Array(out.storage)
            @test approx_eq(Float64.(first_result), reference; atol=1e-6)
            for _ in 1:32
                Gesso.softmax!(lava, out, scores, wl)
                @test isequal(Array(out.storage), first_result)
            end
        end

        bad = mat(Gesso.TemporaryWorkspace, Lava.LavaArray(fill(-Inf32, 1, 65)))
        out = mat(Gesso.TemporaryWorkspace, lz(1, 65))
        err = try
            Gesso.softmax!(lava, out, bad, wl)
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_NUMERICAL_INSTABILITY
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

    # The decode arena lays step n-1's requests out as views of one slab. Step
    # C is the first with a matching layout and no loose buffers, so its views
    # must come back zeroed even though step B left them dirty.
    @testset "lava decode arena: views reused from the next step, zeroed each step" begin
        e = (storage=lz(2, 2),)
        tens = (embedding=e, lm_head=e, blocks=())
        z1 = Gesso.Inference._decode_zeros_like(lava, tens)
        a1 = z1((3, 2))
        b1 = z1((4,))
        @test a1 isa Lava.LavaArray{Float32, 2}
        @test all(==(0.0f0), Array(a1))
        @test all(==(0.0f0), Array(b1))
        fill!(a1, 7.0f0)
        fill!(b1, 9.0f0)

        z2 = Gesso.Inference._decode_zeros_like(lava, tens)
        a2 = z2((3, 2))
        b2 = z2((4,))
        fill!(a2, 5.0f0)
        fill!(b2, 6.0f0)

        z3 = Gesso.Inference._decode_zeros_like(lava, tens)
        a3 = z3((3, 2))
        b3 = z3((4,))
        @test a3 isa Lava.LavaArray{Float32, 2}
        @test size(a3) == (3, 2)
        @test all(==(0.0f0), Array(a3))
        @test all(==(0.0f0), Array(b3))
        # disjoint views: writing one must not show in the other
        fill!(a3, 1.0f0)
        @test all(==(0.0f0), Array(b3))
    end

    # A request the slab cannot hold (rank 5, or empty) is served loose, and
    # must not drop the layout for the requests around it.
    @testset "lava decode arena: a rank-5 or empty request keeps the slab layout" begin
        ext = Base.get_extension(Gesso, :GessoLavaExt)
        arena = ext._LAVA_DECODE_ARENA
        e = (storage=lz(2, 2),)
        tens = (embedding=e, lm_head=e, blocks=())
        for mid in ((1, 1, 1, 1, 2), (0, 3))
            z1 = Gesso.Inference._decode_zeros_like(lava, tens)
            a1 = z1((3, 2))
            m1 = z1(mid)
            b1 = z1((4,))
            fill!(a1, 7.0f0)
            fill!(m1, 7.0f0)
            fill!(b1, 7.0f0)

            z2 = Gesso.Inference._decode_zeros_like(lava, tens)
            slab2 = arena.slab
            a2 = z2((3, 2))
            m2 = z2(mid)
            b2 = z2((4,))
            fill!(a2, 5.0f0)
            fill!(m2, 5.0f0)
            fill!(b2, 6.0f0)

            z3 = Gesso.Inference._decode_zeros_like(lava, tens)
            a3 = z3((3, 2))
            m3 = z3(mid)
            b3 = z3((4,))
            @test arena.slab === slab2
            @test arena.views[1] === a3
            @test arena.views[2] === nothing
            @test arena.views[3] === b3
            @test size(m3) == mid
            @test all(==(0.0f0), Array(a3))
            @test all(==(0.0f0), Array(m3))
            @test all(==(0.0f0), Array(b3))
        end
    end

    # Single-token decode with a scaled inv_freq reads the cached cos/sin table.
    # Position 5000 exercises table growth past the first 256 rows.
    @testset "lava rope! seq=1 inv_freq: cached table matches CPU at positions 0, 1, 5000" begin
        d = 8
        q0 = reshape(collect(range(-1.0, 1.0; length=16)), (1, 2, d))
        k0 = reshape(collect(range(1.0, -1.0; length=16)), (1, 2, d))
        for pos in (0, 1, 5000), rd in (4, d), interleaved in (true, false)
            inv = Gesso.rope_inv_freq(Gesso.RoPEPolicy(; rotary_dim=rd, theta=10000.0), d)
            cq, ck = mat(Gesso.Activation, copy(q0)), mat(Gesso.Activation, copy(k0))
            Gesso.rope!(Gesso.CPUBackend(), cq, ck, [pos], wl;
                theta=10000.0, inv_freq=inv, interleaved, rotary_dim=rd)
            q = mat(Gesso.Activation, Lava.LavaArray{Float32}(q0))
            k = mat(Gesso.Activation, Lava.LavaArray{Float32}(k0))
            Gesso.rope!(lava, q, k, [pos], wl;
                theta=10000.0, inv_freq=inv, interleaved, rotary_dim=rd)
            @test approx_eq(Array(q.storage), Float32.(cq.storage); atol=1e-3)
            @test approx_eq(Array(k.storage), Float32.(ck.storage); atol=1e-3)
        end
    end

    # QK-norm in decode: one workgroup per head, checked against an F64 per-head reference.
    @testset "lava rmsnorm! (1, heads, d_head): per-head normalize matches F64 reference" begin
        d, H = 8, 3
        x = reshape(collect(range(-2.0, 3.0; length=H * d)), (1, H, d))
        s = collect(range(0.5, 1.5; length=d))
        ref = similar(x)
        for h in 1:H
            v = x[1, h, :]
            ref[1, h, :] = v ./ sqrt(sum(abs2, v) / d + 1e-6) .* s
        end
        dst = mat(Gesso.Activation, lz(1, H, d))
        Gesso.rmsnorm!(lava, dst, mat(Gesso.Activation, Lava.LavaArray{Float32}(x)),
            mat(Gesso.FrozenParameter, Lava.LavaArray{Float32}(Float32.(s))), wl)
        @test approx_eq(Array(dst.storage), Float32.(ref); atol=1e-3)
    end
end
