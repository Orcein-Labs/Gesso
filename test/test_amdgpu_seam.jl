# The AMDGPU (ROCm) backend seam — extension, not identity. GPU-UNTESTED: no
# machine that runs this suite has an AMD GPU. What runs here is everything a
# device-less box can check:
#   1. BEFORE the import: manifest law + "core does not know AMDGPUBackend".
#   2. Probe-import AMDGPU + KernelAbstractions at top level (world age).
#   3. AFTER: the binding, traits, the typed no-device failure, the HIP error
#      mapping, the storage guard and to_device on KA's CPU backend, and the
#      reference kernel against the CPU F64 oracle on KA's CPU backend and
#      compiled and linked for four AMD targets.

using TOML
using Random
using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

@testset "amdgpu seam: extension, not identity — BEFORE import" begin
    project = TOML.parsefile(joinpath(pkgdir(Gesso), "Project.toml"))
    weak = get(project, "weakdeps", Dict{String, Any}())
    @test weak["AMDGPU"] == "21141c5a-9bdb-4563-92ae-f87d6854732e"
    @test weak["KernelAbstractions"] == "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    deps = get(project, "deps", Dict{String, Any}())
    @test !haskey(deps, "AMDGPU")
    @test !haskey(deps, "KernelAbstractions")
    exts = get(project, "extensions", Dict{String, Any}())
    @test exts["GessoAMDGPUExt"] == ["AMDGPU", "KernelAbstractions"]

    @test !isdefined(Gesso, :AMDGPUBackend)
end

const AMDGPU_LOADED = let
    ok = true
    try
        @eval Main using AMDGPU, KernelAbstractions
    catch
        ok = false
    end
    ok
end

const AMDGPU_EXT = AMDGPU_LOADED ? Base.get_extension(Gesso, :GessoAMDGPUExt) : nothing

if !AMDGPU_LOADED || AMDGPU_EXT === nothing
    @test _skip("AMDGPU.jl or KernelAbstractions not loadable — AMDGPU seam tests skipped")
else
    const _AMDX = AMDGPU_EXT
    const _AMDKA = Main.KernelAbstractions
    const _AMDHIP = Main.AMDGPU.HIP
    include("amdgpu_compile.jl")

    @testset "amdgpu seam: binding, traits, explicit failure" begin
        @test isdefined(Gesso, :AMDGPUBackend)
        @test Gesso.AMDGPUBackend <: Gesso.AbstractGessoBackend
        @test Gesso.AMDGPUBackend === _AMDX.KADeviceBackend{Main.AMDGPU.ROCBackend}

        @test Gesso.backend_name(Gesso.AMDGPUBackend) == :amdgpu
        @test Gesso.execution_tier(Gesso.AMDGPUBackend) == 1
        # nothing is claimed until an op lands with both tests (caps set in GessoAMDGPUExt)
        @test Gesso.supports(Gesso.AMDGPUBackend, :matmul) == true
        @test Gesso.supports(Gesso.AMDGPUBackend, :rmsnorm) == true

        if Main.AMDGPU.functional()
            @test _skip("a functional ROCm device is present — the no-device failure test does not apply")
        else
            err = try
                Gesso.AMDGPUBackend()
                nothing
            catch e
                e
            end
            @test err isa Gesso.GessoError
            @test err.code == Gesso.ERR_RESOURCE_LIMIT
            @test occursin("cpubackend", lowercase(sprint(showerror, err)))
        end

        host = _AMDX.KADeviceBackend(_AMDKA.CPU())
        @test Gesso.backend_name(host) == :ka
        @test Gesso.execution_tier(host) == 1
    end

    @testset "amdgpu seam: HIP errors map to typed failures" begin
        oom = Gesso.Inference._engine_failure(_AMDHIP.HIPError(_AMDHIP.hipErrorOutOfMemory))
        @test oom.code == Gesso.ERR_ALLOCATION
        @test oom.detail[:backend] == :amdgpu
        alloc = Gesso.Inference._engine_failure(_AMDHIP.HIPError(_AMDHIP.hipErrorMemoryAllocation))
        @test alloc.code == Gesso.ERR_ALLOCATION
        other = Gesso.Inference._engine_failure(_AMDHIP.HIPError(_AMDHIP.hipErrorInvalidValue))
        @test other.code == Gesso.ERR_RUNTIME
    end

    @testset "amdgpu seam: storage guard and to_device (KA CPU backend)" begin
        b = _AMDX.KADeviceBackend(_AMDKA.CPU())
        P = Gesso.ProjectionWeight
        @test Gesso.Inference._check_device_storage(:x, b, P((2, 2), ones(Float32, 2, 2))) === nothing
        @test Gesso.Inference._check_device_storage(:x, b, P((2, 2), view(ones(Float32, 4, 2), 1:2, :))) ===
              nothing
        for bad in (ones(Float64, 2, 2), nothing, :not_storage)
            err = try
                Gesso.Inference._check_device_storage(:x, b, P((2, 2), bad))
                nothing
            catch e
                e
            end
            @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
        end

        bank = Gesso.Inference.GgufQuantSlice(zeros(UInt8, 16), 0, Int32(0), 1, 2, 2, 16, Int64(1))
        expert = Gesso.ExpertWeight((1, 2, 2), bank)
        @test Gesso.Inference._check_device_storage(:moe, b, expert) === nothing

        E = Gesso.EmbeddingTable((5, 3), reshape(collect(1.0:15.0), 5, 3))
        blk = (
            wq=P((3, 3), [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]),
            attn_rms=Gesso.FrozenParameter((3,), [0.5, 1.0, 2.0]),
            wexp_gate=expert,
        )
        fr = Gesso.FrozenParameter((3,), [1.0, 1.0, 1.0])
        tensors = (; embedding=E, blocks=[blk], lm_head=E, final_rms=fr, spec=:meta)
        before = deepcopy(tensors.embedding.storage)

        dev = Gesso.to_device(b, tensors)
        @test dev.lm_head === dev.embedding
        @test dev.embedding.storage isa Array{Float32}
        @test dev.embedding.storage == Float32.(before)
        @test tensors.embedding.storage == before && eltype(tensors.embedding.storage) === Float64
        @test dev.blocks[1].wq.storage == Float32.(blk.wq.storage)
        @test dev.blocks[1].attn_rms isa Gesso.FrozenParameter
        @test dev.blocks[1].wexp_gate === expert
        @test dev.final_rms.storage == Float32[1, 1, 1]
        @test dev.spec === :meta
        @test keys(dev) == (:embedding, :blocks, :lm_head, :final_rms, :spec)

        untied = merge(tensors, (; lm_head=Gesso.ProjectionWeight((5, 3), copy(E.storage))))
        dev2 = Gesso.to_device(b, untied)
        @test dev2.lm_head !== dev2.embedding
        @test dev2.lm_head.storage == Float32.(E.storage)

        err = try
            Gesso.to_device(b, merge(tensors, (; final_rms=Gesso.FrozenParameter((3,), :not_storage))))
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
    end

    # CPU F64 oracle for the row RMS-norm contract (Float32 squares, Float64 sum)
    function _amd_rms_oracle(x::AbstractArray{Float32}, w, eps)
        d = size(x, ndims(x))
        lead = div(length(x), d)
        xm = reshape(x, lead, d)
        y = similar(xm)
        for r in 1:lead
            s = sum(Float64(abs2(xm[r, j])) for j in 1:d)
            rms = sqrt(s / d + eps)
            for j in 1:d
                y[r, j] = Float32(Float64(xm[r, j]) / rms * Float64(w[j]))
            end
        end
        return reshape(y, size(x))
    end

    @testset "amdgpu kernels: row rmsnorm vs CPU F64 oracle (KA CPU backend)" begin
        ka = _AMDKA.CPU()
        rng = Random.MersenneTwister(0x5eed)
        for (shape, wg) in (((1, 1000), 64), ((7, 33), 32), ((2, 3, 40), 64), ((4, 300), 256), ((3, 5), 1))
            x = randn(rng, Float32, shape...)
            w = randn(rng, Float32, shape[end])
            y = similar(x)
            _AMDX._ka_rmsnorm_rows!(ka, y, x, w, 1e-6; wg)
            _AMDKA.synchronize(ka)
            @test maximum(abs, y .- _amd_rms_oracle(x, w, 1e-6)) <= 1f-5
        end

        x = randn(rng, Float32, 3, 16)
        x[2, 5] = Inf32
        y = similar(x)
        _AMDX._ka_rmsnorm_rows!(ka, y, x, ones(Float32, 16), 1e-6; wg=8)
        @test all(isnan, y[2, :])
        @test all(isfinite, y[[1, 3], :])

        for wg in (0, 48, 512)
            err = try
                _AMDX._ka_rmsnorm_rows!(ka, similar(x), x, ones(Float32, 16), 1e-6; wg)
                nothing
            catch e
                e
            end
            @test err isa Gesso.GessoError && err.code == Gesso.ERR_INVALID_PLAN
        end
    end

    @testset "amdgpu kernels: compile and link for $(join(first.(AMD_TARGETS), ", "))" begin
        G = amd_global
        rows = 7
        for wg in (64, 256)
            res = amd_compile(
                _AMDX._ka_rmsnorm_rows_kernel!,
                wg,
                (wg * rows,),
                (G(Float32, 2), G(Float32, 2), G(Float32, 1), Float64, Int, Int),
            )
            @test length(res) == length(AMD_TARGETS)
            @test all(r -> r.bytes > 0, res)
        end

        # negative control: a heap allocation cannot lower for the device, so
        # the harness must refuse it rather than report bytes
        @test_throws Main.AMDGPU.GPUCompiler.InvalidIRError amd_compile(_amd_heap_alloc_kernel!, 64, (64,), (G(Float32, 1),))

        # that throw leaves its hostcalls behind in AMDGPU; the next fresh
        # compile must not report them (wg 128 is not in the compile cache)
        clean = amd_compile(_AMDX._ka_rmsnorm_rows_kernel!, 128, (128 * rows,), (G(Float32, 2), G(Float32, 2), G(Float32, 1), Float64, Int, Int))
        @test all(r -> isempty(r.hostcalls), clean)

        # negative control: a kernel that calls the host allocator compiles,
        # and the harness reports and refuses it. AMDGPU.jl cannot lower device
        # malloc for gfx1030 at all, so the control runs on the other targets.
        hc_targets = filter(t -> t[1] != "gfx1030", AMD_TARGETS)
        @test_throws ErrorException amd_compile(_amd_hostcall_kernel!, 64, (64,), (G(Float32, 1),); targets=hc_targets)
        hc = amd_compile(_amd_hostcall_kernel!, 64, (64,), (G(Float32, 1),); allow_hostcalls=true, targets=hc_targets)
        @test length(hc) == length(hc_targets)
        @test all(r -> :malloc_hostcall in r.hostcalls, hc)
    end
end
