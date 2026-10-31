# Device-free AMD compile of KernelAbstractions kernels: build and link a kernel
# for each AMD target with no GPU and no ROCm install (AMDGPU ships its own LLVM
# backend and device libraries as JLLs). Works on KA 0.9 (AMDGPU ≤ 2.8) and
# KA 0.10 (AMDGPU ≥ 2.9). Needs AMDGPU and KernelAbstractions loaded in Main.

if !isdefined(@__MODULE__, :amd_compile)

    # (isa, wave64): RDNA 2/3 run wave32, CDNA 2/3 wave64
    const AMD_TARGETS = (("gfx1030", false), ("gfx1100", false), ("gfx90a", true), ("gfx942", true))

    const _AMD_KA = Main.KernelAbstractions
    const _AMD_GPUC = Main.AMDGPU.GPUCompiler

    function _amd_context(obj, ndr)
        KA = _AMD_KA
        if pkgversion(KA) >= v"0.10"
            ndrange, wgs, iterspace = KA.launch_setup(KA.sizes_only(obj), ndr, nothing)
            groups = size(KA.blocks(iterspace))
            items = size(KA.workitems(iterspace))
            launch = KA.select_launch(
                map(KA.mul_extent, groups, items),
                items,
                1024,
                (1024, 1024, 1024),
                (Int(typemax(Int32)), 65535, 65535),
                KA.narrow_index_type(Main.AMDGPU.ROCBackend()),
            )
            return KA.mkcontext(obj, ndrange, iterspace, launch; subgroups=KA.kernel_subgroups(obj))
        end
        ndrange, wgs, iterspace, dynamic = KA.launch_config(obj, ndr, nothing)
        return KA.mkcontext(obj, ndrange, iterspace)
    end

    function _amd_params(w64::Bool)
        P = Main.AMDGPU.Compiler.HIPCompilerParams
        return fieldcount(P) >= 3 ? P(w64, true, true) : P(w64, true)
    end

    "Device-side global array type, for `amd_compile` argument tuples."
    amd_global(T, N) = Main.AMDGPU.Device.ROCDeviceArray{T, N, Main.AMDGPU.Device.AS.Global}

    """
        amd_compile(kernel, wg, ndr, argtypes) -> [(; isa, bytes)]

    Compile and link `kernel(ROCBackend(), wg)` over `ndr` for every AMD target.
    `argtypes` are the device-side argument types, without the KA context.
    """
    function amd_compile(kernel, wg, ndr, argtypes::Tuple)
        GPUC = _AMD_GPUC
        obj = kernel(Main.AMDGPU.ROCBackend(), wg)
        ctx = _amd_context(obj, ndr)
        tt = Tuple{typeof(ctx), argtypes...}
        return map(AMD_TARGETS) do (isa, w64)
            feat = w64 ? "" : "+wavefrontsize32,-wavefrontsize64"
            target = GPUC.GCNCompilerTarget(; dev_isa=isa, features=feat)
            cfg = GPUC.CompilerConfig(
                target,
                _amd_params(w64);
                kernel=true,
                name=nothing,
                always_inline=true,
            )
            job = GPUC.CompilerJob(GPUC.methodinstance(typeof(obj.f), tt), cfg)
            r = Main.AMDGPU.Compiler.compile_or_lookup(job)
            (; isa, bytes=r.obj === nothing ? 0 : length(r.obj))
        end
    end

    # Negative control for amd_compile: a heap allocation cannot lower for the
    # device, so compiling this must throw.
    @Main.KernelAbstractions.kernel function _amd_heap_alloc_kernel!(dst)
        i = @Main.KernelAbstractions.index(Global, Linear)
        dst[i] = Float32(sum(Vector{Float32}(undef, i)))
    end

end
