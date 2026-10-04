# Phase 8 item A tests (§LXXXI): the Lava seam — extension, not identity.
#
# Structure mirrors test_cuda_seam.jl and is load-order-sensitive on purpose:
#   1. BEFORE any Lava import: manifest law + "core does not know LavaBackend".
#   2. Probe-import Lava (and thereby GessoLavaExt) if it is in the test env —
#      top-level, never inside a running testset (world age).
#   3. AFTER the import: the binding exists in Gesso's namespace, the Gesso
#      vs Lava name clash is kept distinct, traits hold, and a missing Vulkan
#      device throws a typed error instead of a CPU/CUDA fallback.
# Every non-runnable case is a NAMED skip (§LXXXI skip law). The
# extension-loads tests below the probe never skip: they are exactly what a
# device-less box must still verify.

using TOML
using Test: AbstractTestSet, Broken, get_testset, record

# test_smollm2.jl owns _skip; this file only defines it if that ran first
if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

@testset "lava seam: extension, not identity — BEFORE import (§LXXXI)" begin
    # manifest law: weakdep + extension mapping, and NEVER a core dep (§VII)
    project = joinpath(pkgdir(Gesso), "Project.toml")
    manifest = TOML.parsefile(project)
    weak = get(manifest, "weakdeps", Dict{String, Any}())
    @test haskey(weak, "Lava")
    @test weak["Lava"] == "3a680b1f-cb25-4bee-9cf7-bc880b76dc8c"
    @test !haskey(get(manifest, "deps", Dict{String, Any}()), "Lava")
    exts = get(manifest, "extensions", Dict{String, Any}())
    @test exts["GessoLavaExt"] == "Lava"

    # core namespace purity: without Lava loaded, the ext has not triggered
    @test !isdefined(Gesso, :LavaBackend)
    @test !(:LavaBackend in string.(names(Gesso)))
end

# top-level probe-import: Lava is in the test env (declared in
# test/Project.toml with a [sources] URL — it is not in the General
# registry); if it is not loadable, everything below skips by name.
# NOTE: upstream Lava (GitHub master) currently fails to PRECOMPILE under
# Julia 1.12 (it overwrites a KernelAbstractions adapt_structure method
# during module precompilation, which 1.12 forbids). Julia then loads it
# from source with a warning — slow (~1 min) but functional. That warning
# is upstream's, not Gesso's; this file's assertions do not depend on it.
const LAVA_LOADED = let
    ok = true
    try
        @eval Main using Lava
    catch
        ok = false
    end
    ok
end

# device probe, once, at top level (never inside a running testset). Lava
# has no `functional()` twin — the context constructor IS the probe.
const VULKAN_OK = LAVA_LOADED && let
    ok = true
    try
        Lava.vk_context()
    catch
        ok = false
    end
    ok
end

@testset "lava seam: extension binding + explicit failure (§LXXXI)" begin
    if !LAVA_LOADED
        @test _skip(
            "Lava.jl not loadable in the test env — binding/traits tests skipped; core stays Lava-free either way (§VII)",
        )
    else
        # importing Lava triggered GessoLavaExt: the backend now exists in
        # Gesso's namespace — extension, not identity (§VII, §LXXXI)
        @test isdefined(Gesso, :LavaBackend)
        @test Gesso.LavaBackend <: Gesso.AbstractGessoBackend

        # the load-bearing name clash (§LXXXI): Lava's KA type and Gesso's
        # tag are DISTINCT types. Gesso's binding is the AbstractGessoBackend
        # from this extension (asserted above) — had Gesso accidentally bound
        # Lava's type instead, that subtyping test would fail.
        @test Lava.LavaBackend !== Gesso.LavaBackend

        if !VULKAN_OK
            # no device: construction throws a TYPED error — never a
            # CPUBackend, never a CUDABackend, never a silent fallback (§LXX)
            err = try
                Gesso.LavaBackend()
                nothing
            catch e
                e
            end
            @test err isa Gesso.GessoError
            @test err.code == Gesso.ERR_RESOURCE_LIMIT
            @test occursin("vulkan", lowercase(sprint(showerror, err)))
            @test occursin("cpubackend", lowercase(sprint(showerror, err)))  # names the alternative, no silent switch
            # the constructor is the ONLY thing the missing device blocks:
            # type-level traits stay probe-able without a device
            @test Gesso.backend_name(Gesso.LavaBackend) == :lava
            @test Gesso.execution_tier(Gesso.LavaBackend) == 1
            @test Gesso.supports(Gesso.LavaBackend, :matmul) == true
            @test Gesso.supports(Gesso.LavaBackend, :quantize) == false
        else
            b = Gesso.LavaBackend()
            @test b isa Gesso.AbstractGessoBackend
            @test Gesso.backend_name(b) == :lava
            @test Gesso.execution_tier(b) == 1     # OPTIMIZED_GENERIC (§XXI)
            # same capability coverage as the CPU reference; quantize stays false
            for cap in (
                :rmsnorm,
                :attention,
                :softmax,
                :swiglu_ffn,
                :matmul,
                :embedding_lookup,
                :rope_none,
            )
                @test Gesso.supports(b, cap) == true
            end
            @test Gesso.supports(b, :quantize) == false
            @test Gesso.supports(b, :dequantize) == false
            @test Gesso.supports(b, :definitely_unknown_capability) == false
        end
    end
end
