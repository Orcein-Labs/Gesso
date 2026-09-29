# Phase 4 item A tests (§LXXVII): the backend seam — extension, not identity.
#
# Structure is load-order-sensitive and deliberate:
#   1. BEFORE any CUDA import: manifest law + "core does not know CUDABackend".
#   2. Probe-import CUDA (and thereby GessoCUDAExt) if the package is in the
#      test env — top-level, never inside a running testset (world age).
#   3. AFTER the import: the binding exists in Gesso's namespace, traits hold,
#      and a missing device throws a typed error instead of CPU-fallback.
# Every non-runnable case is a NAMED skip (§LXXVII skip law).

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

@testset "cuda seam: extension, not identity — BEFORE import (§LXXVII)" begin
    # manifest law: weakdep + extension mapping, and NEVER a core dep (§VII)
    project = joinpath(pkgdir(Gesso), "Project.toml")
    manifest = TOML.parsefile(project)
    weak = get(manifest, "weakdeps", Dict{String, Any}())
    @test haskey(weak, "CUDA")
    @test !haskey(get(manifest, "deps", Dict{String, Any}()), "CUDA")
    exts = get(manifest, "extensions", Dict{String, Any}())
    @test exts["GessoCUDAExt"] == "CUDA"

    # core namespace purity: without CUDA loaded, the ext has not triggered
    @test !isdefined(Gesso, :CUDABackend)
    @test !(:CUDABackend in string.(names(Gesso)))
end

# top-level probe-import: CUDA is in the test env (declared in
# test/Project.toml); if it is not loadable, everything below skips by name.
const CUDA_LOADED = let
    ok = true
    try
        @eval Main using CUDA
    catch
        ok = false
    end
    ok
end

@testset "cuda seam: extension binding + explicit failure (§LXXVII)" begin
    if !CUDA_LOADED
        @test _skip(
            "CUDA.jl not loadable in the test env — binding/traits tests skipped; core stays CUDA-free either way (§VII)",
        )
    else
        # importing CUDA triggered GessoCUDAExt: the backend now exists in
        # Gesso's namespace — extension, not identity (§VII, §CIX)
        @test isdefined(Gesso, :CUDABackend)
        @test Gesso.CUDABackend <: Gesso.AbstractGessoBackend

        if !CUDA.functional()
            # no device: construction throws a TYPED error — never a
            # CPUBackend, never a silent fallback (§LXX)
            err = try
                Gesso.CUDABackend()
                nothing
            catch e
                e
            end
            @test err isa Gesso.GessoError
            @test err.code == Gesso.ERR_RESOURCE_LIMIT
            @test occursin("cuda", lowercase(sprint(showerror, err)))
            @test occursin("cpubackend", lowercase(sprint(showerror, err)))  # names the alternative, no silent switch
            # the constructor is the ONLY thing the missing device blocks:
            # type-level traits stay probe-able without a device
            @test Gesso.backend_name(Gesso.CUDABackend) == :cuda
            @test Gesso.execution_tier(Gesso.CUDABackend) == 1
            @test Gesso.supports(Gesso.CUDABackend, :matmul) == true
            @test Gesso.supports(Gesso.CUDABackend, :quantize) == false
        else
            b = Gesso.CUDABackend()
            @test b isa Gesso.AbstractGessoBackend
            @test Gesso.backend_name(b) == :cuda
            @test Gesso.execution_tier(b) == 1     # OPTIMIZED_GENERIC (§XXI)
            # same capability coverage as the CPU reference; quantize stays false
            for cap in (:rmsnorm, :rope, :softmax, :swiglu, :matmul, :embedding_lookup)
                @test Gesso.supports(b, cap) == true
            end
            @test Gesso.supports(b, :quantize) == false
            @test Gesso.supports(b, :dequantize) == false
            @test Gesso.supports(b, :definitely_unknown_capability) == false
        end
    end
end
