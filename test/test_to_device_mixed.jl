# Phase 9 item A tests (§LXXXII): mixed-backend `to_device` survives EITHER
# extension load order. Phase 8's receipts documented the wart: Lava's ext
# attaches by name, but GessoCUDAExt.__init__ used to rebind unconditionally,
# clobbering Lava's methods when CUDA loaded second. Item A mirrors the
# attach-or-own pattern on the CUDA side; this file pins BOTH orders.
#
# The in-suite order (CUDA seam → Lava seam) is testable directly — by the
# time this file runs, both extensions are loaded and the binding must carry
# both backends' methods. The reverse order (Lava first, CUDA second) needs a
# CHILD PROCESS: load order is a process property. The child asserts method
# EXISTENCE only — no device, no vk_context — so it runs on device-less CI.
#
# Skips by name when either extension is not in the test env (env-level, not
# device-level — the assertions here need no device).

using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const _MIXED_EXTS_OK =
    @isdefined(CUDA_LOADED) && @isdefined(LAVA_LOADED) && CUDA_LOADED && LAVA_LOADED

@testset "to_device mixed: CUDA-then-Lava keeps both backends (in-suite order)" begin
    if !_MIXED_EXTS_OK
        @test _skip(
            "CUDA or Lava extension not loaded in the test env — mixed to_device tests skipped (§LXXXII)",
        )
    else
        # both seams ran earlier in this suite: CUDA loaded first (test_cuda_seam.jl),
        # Lava second (test_lava_seam.jl). The single function object bound at
        # Gesso.to_device must carry BOTH backends' methods — the Lava attach
        # went through the name, and the CUDA attach-or-own must not have
        # clobbered it.
        f = getfield(Gesso, :to_device)
        @test hasmethod(f, Tuple{Gesso.CUDABackend, Any})
        @test hasmethod(f, Tuple{Gesso.LavaBackend, Any})
        # and the name clash stays distinct (§LXXXI)
        @test Lava.LavaBackend !== Gesso.LavaBackend
    end
end

@testset "to_device mixed: Lava-then-CUDA keeps both backends (child process)" begin
    if !_MIXED_EXTS_OK
        @test _skip(
            "CUDA or Lava extension not loaded in the test env — mixed to_device tests skipped (§LXXXII)",
        )
    else
        script = """
        using Gesso
        @eval Main using Lava
        isdefined(Gesso, :LavaBackend) || error("Lava ext did not trigger")
        f = getfield(Gesso, :to_device)
        hasmethod(f, Tuple{Gesso.LavaBackend, Any}) || error("Lava method missing after Lava-first bind")
        @eval Main using CUDA
        isdefined(Gesso, :CUDABackend) || error("CUDA ext did not trigger")
        g = getfield(Gesso, :to_device)
        hasmethod(g, Tuple{Gesso.LavaBackend, Any}) || error("Lava to_device method CLOBBERED by CUDA load")
        hasmethod(g, Tuple{Gesso.CUDABackend, Any}) || error("CUDA to_device method missing after attach")
        println("MIXED_LAVA_FIRST_OK")
        """
        proj = joinpath(pkgdir(Gesso), "test")
        out = try
            read(`$(Base.julia_cmd()) --project=$proj --startup-file=no -e $script`, String)
        catch e
            # the child's stderr flows to our stderr (diagnosable in the log);
            # record the failure shape for the assertion message
            "CHILD_FAILED: " * sprint(showerror, e)
        end
        @test occursin("MIXED_LAVA_FIRST_OK", out)
    end
end
