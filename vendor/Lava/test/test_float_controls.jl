# Per-kernel SignedZeroInfNanPreserve: host identity, SPIR-V emission,
# capability rejection, cache isolation, and public KA dispatch.
#
# The host/parser/compiler checks still need a Vulkan context, but do not depend
# on float-controls support from that device. Positive dispatches are gated by
# per-device Vulkan properties; C1_STRICT_FLOAT_CONTROLS=1 turns missing
# Float32/Float64 support into failures and disallows feature skips.

using Test, Lava, Serialization
using Lava.KernelAbstractions: @kernel, @index, @Const
const KA = Lava.KernelAbstractions
const GPUCompiler = Lava.GPUCompiler
const _FC_SHADER_EXPORT_SEQUENCE = Threads.Atomic{Int}(0)

function _fc_export_shader_artifact(label, bytes::Vector{UInt8})
    output_dir = get(ENV, "C1_FC_SHADER_OUT", "")
    isempty(output_dir) && return nothing
    mkpath(output_dir)
    sequence = Threads.atomic_add!(_FC_SHADER_EXPORT_SEQUENCE, 1) + 1
    safe_label = replace(string(label), r"[^A-Za-z0-9_-]" => "_")
    filename = string(getpid(), "_", time_ns(), "_", sequence, "_", safe_label, ".spv")
    path = joinpath(output_dir, filename)
    open(path, "w") do io
        write(io, bytes)
    end
    return path
end

@kernel function _fc_arithmetic!(out, @Const(a), @Const(b), scale, bias)
    i = @index(Global, Linear)
    @inbounds out[i] = (a[i] + b[i]) * scale + bias
end

# Capturing both a device buffer and a scalar exercises Adapt on the wrapped
# closure. The launch-time scale exercises the regular argument pack.
function _fc_capture_kernel(captured, captured_bias::Float32)
    @kernel function _fc_capture!(out, @Const(src), scale::Float32)
        i = @index(Global, Linear)
        @inbounds out[i] = src[i] * scale + captured_bias + captured[i]
    end
    return _fc_capture!
end

function _fc_emit_probe(out::Lava.LavaDeviceArray{T,1},
                        a::Lava.LavaDeviceArray{T,1},
                        b::Lava.LavaDeviceArray{T,1}, scale::T, bias::T) where {T<:AbstractFloat}
    i = Lava.lava_global_invocation_id_x()
    @inbounds out[i] = (a[i] + b[i]) * scale + bias
    return nothing
end

function _fc_string_words(s::AbstractString)
    bytes = vcat(collect(codeunits(s)), UInt8[0])
    append!(bytes, zeros(UInt8, mod(-length(bytes), 4)))
    words = UInt32[]
    for i in 1:4:length(bytes)
        word = UInt32(0)
        for j in 0:3
            word |= UInt32(bytes[i + j]) << (8 * j)
        end
        push!(words, word)
    end
    return words
end

# Header-only parser fixture. Multiple entry points deliberately test that the
# parsed module contract is the union of the modes declared by its entries.
function _fc_header_spirv(; version::UInt32 = UInt32(0x00010400),
                           capability::Bool = true,
                           extension::Bool = false,
                           entries::Vector{UInt32} = UInt32[1],
                           modes::Vector{Tuple{UInt32,UInt32}} = Tuple{UInt32,UInt32}[])
    words = UInt32[Lava.SPIRV_MAGIC, version, 0, 16, 0]
    if capability
        push!(words, (UInt32(2) << 16) | UInt32(Lava.Op.OpCapability))
        push!(words, Lava.Cap.SignedZeroInfNanPreserve)
    end
    if extension
        strwords = _fc_string_words("SPV_KHR_float_controls")
        push!(words, (UInt32(1 + length(strwords)) << 16) | UInt32(Lava.Op.OpExtension))
        append!(words, strwords)
    end
    for id in entries
        strwords = _fc_string_words("main")
        push!(words, (UInt32(3 + length(strwords)) << 16) | UInt32(Lava.Op.OpEntryPoint))
        push!(words, UInt32(5)) # GLCompute
        push!(words, id)
        append!(words, strwords)
    end
    for (entry, width) in modes
        push!(words, (UInt32(4) << 16) | UInt32(Lava.Op.OpExecutionMode))
        push!(words, entry)
        push!(words, Lava.ExecMode.SignedZeroInfNanPreserve)
        push!(words, width)
    end
    return collect(reinterpret(UInt8, words))
end

function _fc_find_plan(ctx, kernel, expected_mask::UInt8)
    ftype = typeof(kernel.f)
    found = Any[]
    for (key, plans) in ctx.caches.launchplans
        fieldcount(key) > 0 && fieldtype(key, 1) === ftype || continue
        for plan in plans
            Lava.spirv_float_controls_mask(plan.compiled.spirv_bytes) == expected_mask || continue
            push!(found, (key = key, plan = plan))
        end
    end
    isempty(found) && error("no launch plan for $(ftype) with float-controls mask $(expected_mask)")
    return only(found)
end

function _fc_compiler_signature(kernel, ::Type{T}; ndrange::Int = 4) where {T}
    ndr_canon, _, iterspace, _ = KA.launch_config(kernel, ndrange, nothing)
    ka_ctx = KA.mkcontext(kernel, ndr_canon, iterspace)
    array_type = Lava.LavaDeviceArray{T,1}
    tt = Tuple{typeof(ka_ctx), array_type, array_type, array_type, T, T}
    return tt, (64, 1, 1)
end

function _fc_assert_positive_cache_roundtrips(ctx, kernel, ::Type{T}, plan, mask::UInt8;
                                               label::AbstractString) where {T}
    tt, wg = _fc_compiler_signature(kernel, T)
    source = GPUCompiler.methodinstance(typeof(kernel.f), tt)

    Lava.lava_disk_cache_store(source, wg, plan.compiled, mask)
    disk_hit = Lava.lava_disk_cache_load(source, wg, mask)
    @test disk_hit !== nothing
    if disk_hit !== nothing
        @test Lava.spirv_float_controls_mask(disk_hit.spirv_bytes) == mask
        @test disk_hit.spirv_bytes == plan.compiled.spirv_bytes
    end

    old_version = Lava.FROZEN_VERSION[]
    old_recording = Lava.FROZEN_RECORDING[]
    old_supported = ctx.float_controls_supported
    try
        Lava.FROZEN_VERSION[] = "float-controls-positive-$(label)-$(time_ns())"
        Lava.FROZEN_RECORDING[] = true
        Lava.frozen_store(ctx, kernel.f, tt, wg, plan.compiled)
        memkey = (typeof(kernel.f), tt, wg)
        haskey(ctx.caches.frozen_mem, memkey) && delete!(ctx.caches.frozen_mem, memkey)
        frozen_hit = Lava.frozen_load(ctx, kernel.f, tt, wg)
        @test frozen_hit isa Lava.LavaLinkedKernel
        if frozen_hit isa Lava.LavaLinkedKernel
            @test Lava.spirv_float_controls_mask(frozen_hit.compiled.spirv_bytes) == mask
            @test frozen_hit.compiled.spirv_bytes == plan.compiled.spirv_bytes
        end
    finally
        ctx.float_controls_supported = old_supported
        Lava.FROZEN_VERSION[] = old_version
        Lava.FROZEN_RECORDING[] = old_recording
    end
    return nothing
end

function _fc_copy_launchplans(cache)
    saved = copy(cache)
    for key in keys(saved)
        saved[key] = copy(saved[key])
    end
    return saved
end

function _fc_restore_dict!(destination, saved)
    empty!(destination)
    merge!(destination, saved)
    return nothing
end

function _fc_restore_env!(key, value)
    value === nothing ? pop!(ENV, key, nothing) : (ENV[key] = value)
    return nothing
end

# Give disk/frozen/debug artifacts a private lifetime and restore every global
# ref and device cache touched by the exercise, even when a test fails.
function _fc_with_isolated_caches(f, ctx)
    disk_dir = mktempdir(; prefix = "lava-float-controls-disk-")
    frozen_dir = mktempdir(; prefix = "lava-float-controls-frozen-")
    debug_dir = joinpath(disk_dir, "debug")
    mkpath(debug_dir)
    env_refs = (debug = get(ENV, "LAVA_DEBUG_DIR", nothing),
                spirv_dump = get(ENV, "LAVA_SPIRV_DUMP_DIR", nothing))
    old_refs = (
        disk_dir = Lava.LAVA_DISK_CACHE_DIR[],
        frozen_dir = Lava.FROZEN_CACHE_DIR[],
        frozen_version = Lava.FROZEN_VERSION[],
        frozen_recording = Lava.FROZEN_RECORDING[],
        frozen_hits = Lava.FROZEN_HITS[],
        frozen_stores = Lava.FROZEN_STORES[],
        frozen_misses = Lava.FROZEN_MISSES[],
        compiler_configs = copy(Lava.COMPILER_CONFIGS),
        float_controls_supported = ctx.float_controls_supported,
        frozen_last_pcache = ctx.caches.frozen_last_pcache,
        launchplans = _fc_copy_launchplans(ctx.caches.launchplans),
        linked = copy(ctx.caches.linked),
        pipelines = copy(ctx.caches.pipelines),
        pipeline_order = copy(ctx.caches.pipeline_order),
        iterplans = copy(ctx.caches.iterplans),
        frozen_mem = copy(ctx.caches.frozen_mem),
    )
    try
        Lava.LAVA_DISK_CACHE_DIR[] = disk_dir
        Lava.FROZEN_CACHE_DIR[] = frozen_dir
        ENV["LAVA_DEBUG_DIR"] = debug_dir
        ENV["LAVA_SPIRV_DUMP_DIR"] = debug_dir
        ctx.caches.frozen_last_pcache = nothing
        return f(disk_dir, frozen_dir)
    finally
        Lava.LAVA_DISK_CACHE_DIR[] = old_refs.disk_dir
        Lava.FROZEN_CACHE_DIR[] = old_refs.frozen_dir
        Lava.FROZEN_VERSION[] = old_refs.frozen_version
        Lava.FROZEN_RECORDING[] = old_refs.frozen_recording
        Lava.FROZEN_HITS[] = old_refs.frozen_hits
        Lava.FROZEN_STORES[] = old_refs.frozen_stores
        Lava.FROZEN_MISSES[] = old_refs.frozen_misses
        empty!(Lava.COMPILER_CONFIGS)
        merge!(Lava.COMPILER_CONFIGS, old_refs.compiler_configs)
        ctx.float_controls_supported = old_refs.float_controls_supported
        ctx.caches.frozen_last_pcache = old_refs.frozen_last_pcache
        _fc_restore_dict!(ctx.caches.launchplans, old_refs.launchplans)
        _fc_restore_dict!(ctx.caches.linked, old_refs.linked)
        _fc_restore_dict!(ctx.caches.pipelines, old_refs.pipelines)
        empty!(ctx.caches.pipeline_order)
        append!(ctx.caches.pipeline_order, old_refs.pipeline_order)
        _fc_restore_dict!(ctx.caches.iterplans, old_refs.iterplans)
        _fc_restore_dict!(ctx.caches.frozen_mem, old_refs.frozen_mem)
        _fc_restore_env!("LAVA_DEBUG_DIR", env_refs.debug)
        _fc_restore_env!("LAVA_SPIRV_DUMP_DIR", env_refs.spirv_dump)
        rm(disk_dir; recursive = true, force = true)
        rm(frozen_dir; recursive = true, force = true)
    end
end

function _fc_run_arithmetic!(kernel, ::Type{T}; special::Bool) where {T<:AbstractFloat}
    if special
        a_host = T[one(T), T(Inf), T(Inf), -zero(T), T(NaN)]
        b_host = T[2, one(T), -T(Inf), -zero(T), zero(T)]
        bias = -zero(T)
    else
        a_host = T[1, 2, 3, 4]
        b_host = T[2, 3, 4, 5]
        bias = one(T)
    end
    a = Lava.LavaArray(a_host)
    b = Lava.LavaArray(b_host)
    out = Lava.LavaArray(fill(zero(T), length(a_host)))
    kernel(out, a, b, one(T), bias; ndrange = length(a_host))
    KA.synchronize(KA.get_backend(out))
    return Array(out), Array(a), Array(b), a_host, b_host
end

function _fc_launch_capture!(ctx, backend, out, source_host, captured_host)
    old_keys = Set(keys(ctx.caches.launchplans))
    source = Lava.LavaArray(copy(source_host))
    captured = Lava.LavaArray(copy(captured_host))
    kernel = _fc_capture_kernel(captured, 2.5f0)(backend, 64)
    controlled = Lava.with_float_controls(kernel, (32,))
    controlled(out, source, 3.0f0; ndrange = length(source_host))
    new_keys = [key for key in keys(ctx.caches.launchplans)
                if !(key in old_keys) && fieldcount(key) > 0 &&
                   Lava.required_float_controls(fieldtype(key, 1)) == Lava.float_controls_mask((32,))]
    return only(new_keys)
end

@testset "per-kernel SignedZeroInfNanPreserve contract" begin
    backend = Lava.LavaBackend()
    ctx = Lava.vk_context(backend)
    req16 = Lava.float_controls_mask((16,))
    req32 = Lava.float_controls_mask((32,))
    req64 = Lava.float_controls_mask((64,))
    req3264 = Lava.float_controls_mask((32, 64))
    strict = get(ENV, "C1_STRICT_FLOAT_CONTROLS", "0") == "1"

    _fc_with_isolated_caches(ctx) do disk_dir, frozen_dir
        default = _fc_arithmetic!(backend, 64)
        none = Lava.with_float_controls(default, ())
        c32 = Lava.with_float_controls(default, (32,))
        c64 = Lava.with_float_controls(default, (64,))
        c3264 = Lava.with_float_controls(default, (32, 64))

        @testset "host API identity and SPIR-V parser" begin
            @test req16 == 0x01
            @test req32 == 0x02
            @test req64 == 0x04
            @test req3264 == 0x06

            @test none === default
            @test Lava.required_float_controls(typeof(default.f)) == 0x00
            @test KA.backend(c3264) === KA.backend(default)
            @test KA.workgroupsize(c3264) == KA.workgroupsize(default)
            @test KA.ndrange(c3264) == KA.ndrange(default)
            @test Lava.required_float_controls(typeof(c32.f)) == req32
            @test Lava.required_float_controls(typeof(c64.f)) == req64
            @test Lava.required_float_controls(typeof(c3264.f)) == req3264
            @test typeof(c32.f) != typeof(default.f)
            @test typeof(c32.f) != typeof(c64.f)
            @test typeof(c3264.f) != typeof(c32.f)

            origin = Lava.kernel_origin_type(typeof(c3264.f))
            @test origin === typeof(default.f)
            @test parentmodule(origin) === parentmodule(typeof(default.f))

            @test Lava.lava_compiler_config().params.float_controls == 0x00
            @test Lava.lava_compiler_config(; float_controls = req32).params.float_controls == req32
            @test Lava.lava_compiler_config(; float_controls = req64).params.float_controls == req64
            @test Lava.lava_compiler_config(; float_controls = req3264).params.float_controls == req3264

            @test_throws ArgumentError Lava.with_float_controls(default, (32, 32))
            @test_throws ArgumentError Lava.with_float_controls(default, (128,))
            @test_throws ArgumentError Lava.with_float_controls(default, (32.0,))
            @test_throws ArgumentError Lava.with_float_controls(c32, (64,))
            @test_throws ArgumentError Lava.with_float_controls(c32, ())
            @test_throws ArgumentError Lava.validate_float_controls_mask(UInt8(0x80))

            @test Lava.spirv_float_controls_mask(_fc_header_spirv()) == 0x00
            @test Lava.spirv_float_controls_mask(_fc_header_spirv(capability = false)) == 0x00
            @test Lava.spirv_float_controls_mask(_fc_header_spirv(
                entries = UInt32[1, 2],
                modes = [(UInt32(1), UInt32(32)), (UInt32(2), UInt32(64))])) == req3264
            @test Lava.spirv_float_controls_mask(_fc_header_spirv(
                version = UInt32(0x00010300), extension = true,
                modes = [(UInt32(1), UInt32(32))])) == req32
            @test_throws ArgumentError Lava.spirv_float_controls_mask(_fc_header_spirv(
                capability = false, modes = [(UInt32(1), UInt32(32))]))
            @test_throws ArgumentError Lava.spirv_float_controls_mask(_fc_header_spirv(
                version = UInt32(0x00010300), modes = [(UInt32(1), UInt32(32))]))
            @test_throws ArgumentError Lava.spirv_float_controls_mask(_fc_header_spirv(
                modes = [(UInt32(1), UInt32(32)), (UInt32(1), UInt32(32))]))
            @test_throws ArgumentError Lava.spirv_float_controls_mask(_fc_header_spirv(
                entries = UInt32[1], modes = [(UInt32(2), UInt32(32))]))
            @test_throws ArgumentError Lava.spirv_float_controls_mask(_fc_header_spirv(
                modes = [(UInt32(1), UInt32(128))]))
            malformed = _fc_header_spirv()
            resize!(malformed, 24) # instruction header has no operand word
            @test_throws ArgumentError Lava.spirv_float_controls_mask(malformed)
        end

        @testset "mode emission precedes external SPIR-V validation" begin
            array_type32 = Lava.LavaDeviceArray{Float32,1}
            tuple32 = Tuple{array_type32, array_type32, array_type32, Float32, Float32}
            default_raw = Lava.lava_compile_full(_fc_emit_probe, tuple32;
                workgroup_size = (64, 1, 1), validate = false)
            _fc_export_shader_artifact("raw_default_f32", default_raw.spirv_bytes)
            @test Lava.spirv_float_controls_mask(default_raw.spirv_bytes) == 0x00
            @test !occursin("OpCapability SignedZeroInfNanPreserve", default_raw.spirv_disasm)
            @test Lava.validate_spirv(default_raw.spirv_bytes, default_raw.post_pass_ir,
                                      default_raw.source_map) === nothing

            for (mask, T) in ((req32, Float32), (req64, Float64), (req3264, Float32))
                array_type = Lava.LavaDeviceArray{T,1}
                tt = Tuple{array_type, array_type, array_type, T, T}
                f = Lava.FloatControlledKernel{typeof(_fc_emit_probe),mask}(_fc_emit_probe)
                raw = Lava.lava_compile_full(f, tt;
                    workgroup_size = (64, 1, 1), validate = false)
                _fc_export_shader_artifact("raw_mask_$(mask)_f$(sizeof(T) * 8)", raw.spirv_bytes)
                # Inspect the emitted bytes before invoking spirv-val ourselves.
                @test Lava.spirv_float_controls_mask(raw.spirv_bytes) == mask
                @test occursin("OpCapability SignedZeroInfNanPreserve", raw.spirv_disasm)
                @test Lava.validate_spirv(raw.spirv_bytes, raw.post_pass_ir,
                                          raw.source_map) === nothing
            end
        end

        @testset "device capability rejection and default launch" begin
            @test Lava.validate_float_controls_mask(ctx.float_controls_supported) ==
                  ctx.float_controls_supported
            if strict
                @test ctx.float_controls_supported & req3264 == req3264
            end
            a = Lava.LavaArray(Float32[1, 2, 3, 4])
            b = Lava.LavaArray(Float32[2, 3, 4, 5])
            out = Lava.LavaArray(zeros(Float32, 4))
            supported_before = ctx.float_controls_supported
            iterplans_before = length(ctx.caches.iterplans)
            launchplans_before = length(ctx.caches.launchplans)
            linked_before = length(ctx.caches.linked)
            pipelines_before = length(ctx.caches.pipelines)
            try
                ctx.float_controls_supported = supported_before & ~req32
                @test_throws ArgumentError c32(out, a, b, 1.0f0, 0.0f0; ndrange = 4)
                @test length(ctx.caches.iterplans) == iterplans_before
                @test length(ctx.caches.launchplans) == launchplans_before
                @test length(ctx.caches.linked) == linked_before
                @test length(ctx.caches.pipelines) == pipelines_before

                missing_mode = _fc_header_spirv(modes = [(UInt32(1), UInt32(32))])
                @test_throws ArgumentError Lava.get_compute_pipeline(ctx, missing_mode, "main")
                @test length(ctx.caches.pipelines) == pipelines_before

                # A default kernel remains usable while the optional property is
                # absent from this context's simulated per-device capability.
                default(out, a, b, 1.0f0, 0.0f0; ndrange = 4)
                KA.synchronize(backend)
                @test Array(out) == Float32[3, 5, 7, 9]
            finally
                ctx.float_controls_supported = supported_before
            end
        end

        @testset "cache identity and stale-artifact rejection" begin
            default_plan = _fc_find_plan(ctx, default, 0x00).plan
            _fc_export_shader_artifact("plan_default_f32", default_plan.compiled.spirv_bytes)
            tt0, wg0 = _fc_compiler_signature(default, Float32)
            source0 = GPUCompiler.methodinstance(typeof(default.f), tt0)
            tt32, wg = _fc_compiler_signature(c32, Float32)
            source = GPUCompiler.methodinstance(typeof(c32.f), tt32)
            @test Lava.lava_disk_cache_key(source, wg, 0x00) !=
                  Lava.lava_disk_cache_key(source, wg, req32)
            @test Lava.frozen_key(default.f, tt32, wg) != Lava.frozen_key(c32.f, tt32, wg)

            # Exercise a real serialized disk-cache round trip inside this
            # test's temporary cache root before planting a stale entry.
            Lava.lava_disk_cache_store(source0, wg0, default_plan.compiled, 0x00)
            disk_roundtrip = Lava.lava_disk_cache_load(source0, wg0, 0x00)
            @test disk_roundtrip !== nothing
            @test Lava.spirv_float_controls_mask(disk_roundtrip.spirv_bytes) == 0x00

            stale = Lava.LavaLinkedKernel(default_plan.compiled, default_plan.pipeline,
                                          default_plan.offsets, default_plan.byval_sizes)
            memkey = (typeof(c32.f), tt32, wg)
            haskey(ctx.caches.frozen_mem, memkey) && delete!(ctx.caches.frozen_mem, memkey)
            old_supported = ctx.float_controls_supported
            try
                # No Vulkan pipeline is built on either mismatch path. Temporarily
                # pass the preflight so the cache's SPIR-V contract is exercised.
                ctx.float_controls_supported = old_supported | req32
                Lava.FROZEN_VERSION[] = "float-controls-" * string(hash(time()); base = 16)

                # A matching frozen artifact deserializes and links through the
                # normal path; the pipeline already exists from the default
                # dispatch, so this remains a cache-path check.
                default_frozen_path = Lava.frozen_path(
                    Lava.frozen_key(default.f, tt0, wg0))
                open(default_frozen_path, "w") do io
                    Serialization.serialize(io, default_plan.compiled)
                end
                default_memkey = (typeof(default.f), tt0, wg0)
                haskey(ctx.caches.frozen_mem, default_memkey) &&
                    delete!(ctx.caches.frozen_mem, default_memkey)
                frozen_roundtrip = Lava.frozen_load(ctx, default.f, tt0, wg0)
                @test frozen_roundtrip isa Lava.LavaLinkedKernel
                @test Lava.spirv_float_controls_mask(
                    frozen_roundtrip.compiled.spirv_bytes) == 0x00

                ctx.caches.frozen_mem[memkey] = stale
                @test Lava.frozen_load(ctx, c32.f, tt32, wg) === nothing
                @test !haskey(ctx.caches.frozen_mem, memkey)

                frozen_key = Lava.frozen_key(c32.f, tt32, wg)
                frozen_path = Lava.frozen_path(frozen_key)
                open(frozen_path, "w") do io
                    Serialization.serialize(io, default_plan.compiled)
                end
                @test Lava.frozen_load(ctx, c32.f, tt32, wg) === nothing
                @test !haskey(ctx.caches.frozen_mem, memkey)

                loose_entry = (spec_types = source.specTypes, workgroup_size = wg,
                               kernel = default_plan.compiled)
                loose_path = joinpath(disk_dir, Lava.lava_disk_cache_key(source, wg, 0x00))
                controlled_path = joinpath(disk_dir, Lava.lava_disk_cache_key(source, wg, req32))
                open(loose_path, "w") do io
                    Serialization.serialize(io, loose_entry)
                end
                open(controlled_path, "w") do io
                    Serialization.serialize(io, loose_entry)
                end
                @test Lava.spirv_float_controls_mask(
                    Lava.lava_disk_cache_load(source, wg, 0x00).spirv_bytes) == 0x00
                @test Lava.lava_disk_cache_load(source, wg, req32) === nothing
            finally
                ctx.float_controls_supported = old_supported
                Lava.FROZEN_VERSION[] = ""
            end
        end

        @testset "Float32 and Float64 arithmetic on supported widths" begin
            supported = ctx.float_controls_supported
            if supported & req32 == req32
                c32_result, a_after, b_after, a_before, b_before =
                    _fc_run_arithmetic!(c32, Float32; special = false)
                @test c32_result == Float32[4, 6, 8, 10]
                @test a_after == a_before && b_after == b_before
                c32_special, _, _, special_a, special_b =
                    _fc_run_arithmetic!(c32, Float32; special = true)
                @test c32_special[1] == 3.0f0
                @test isinf(c32_special[2]) && c32_special[2] > 0
                @test isnan(c32_special[3])
                @test iszero(c32_special[4]) && signbit(c32_special[4])
                @test isnan(c32_special[5])
                @test isequal(special_a, Float32[1, Inf, Inf, -0.0f0, NaN])
                @test isequal(special_b, Float32[2, 1, -Inf, -0.0f0, 0.0f0])
                f32_plan = _fc_find_plan(ctx, c32, req32)
                default_found = _fc_find_plan(ctx, default, 0x00)
                @test f32_plan.key != default_found.key
                @test Lava.required_float_controls(fieldtype(f32_plan.key, 1)) == req32
                _fc_export_shader_artifact("plan_controlled_f32", f32_plan.plan.compiled.spirv_bytes)
                _fc_assert_positive_cache_roundtrips(ctx, c32, Float32,
                                                     f32_plan.plan, req32; label = "f32")

                # This exact signature already has a launch plan and linked
                # pipeline. Removing the capability must still fail before the
                # cached plan can be reused.
                cached_plans = get(ctx.caches.launchplans, f32_plan.key, nothing)
                @test cached_plans !== nothing && any(p -> p === f32_plan.plan, cached_plans)
                cached_tt, cached_wg = _fc_compiler_signature(c32, Float32)
                cached_frozen_key = (typeof(c32.f), cached_tt, cached_wg)
                @test haskey(ctx.caches.frozen_mem, cached_frozen_key)
                cached_a = Lava.LavaArray(Float32[1, 2, 3, 4])
                cached_b = Lava.LavaArray(Float32[2, 3, 4, 5])
                cached_out = Lava.LavaArray(zeros(Float32, 4))
                caps_before = ctx.float_controls_supported
                iterplans_before = length(ctx.caches.iterplans)
                launchplans_before = length(ctx.caches.launchplans)
                linked_before = length(ctx.caches.linked)
                pipelines_before = length(ctx.caches.pipelines)
                frozen_mem_before = length(ctx.caches.frozen_mem)
                try
                    ctx.float_controls_supported = caps_before & ~req32
                    @test_throws ArgumentError c32(cached_out, cached_a, cached_b,
                                                   1.0f0, 1.0f0; ndrange = 4)
                    @test length(ctx.caches.iterplans) == iterplans_before
                    @test length(ctx.caches.launchplans) == launchplans_before
                    @test length(ctx.caches.linked) == linked_before
                    @test length(ctx.caches.pipelines) == pipelines_before
                    @test length(ctx.caches.frozen_mem) == frozen_mem_before
                    @test haskey(ctx.caches.frozen_mem, cached_frozen_key)
                finally
                    ctx.float_controls_supported = caps_before
                    KA.synchronize(backend)
                end
                @test Array(cached_out) == zeros(Float32, 4)
            elseif !strict
                @info "Float32 SignedZeroInfNanPreserve is unsupported; skipping positive Float32 dispatches"
                @test_skip supported & req32 == req32
            end

            if supported & req64 == req64
                c64_result, a_after, b_after, a_before, b_before =
                    _fc_run_arithmetic!(c64, Float64; special = false)
                @test c64_result == Float64[4, 6, 8, 10]
                @test a_after == a_before && b_after == b_before
                c64_special, _, _, special_a, special_b =
                    _fc_run_arithmetic!(c64, Float64; special = true)
                @test c64_special[1] == 3.0
                @test isinf(c64_special[2]) && c64_special[2] > 0
                @test isnan(c64_special[3])
                @test iszero(c64_special[4]) && signbit(c64_special[4])
                @test isnan(c64_special[5])
                @test isequal(special_a, Float64[1, Inf, Inf, -0.0, NaN])
                @test isequal(special_b, Float64[2, 1, -Inf, -0.0, 0.0])
                f64_plan = _fc_find_plan(ctx, c64, req64)
                @test Lava.spirv_float_controls_mask(f64_plan.plan.compiled.spirv_bytes) == req64
                _fc_export_shader_artifact("plan_controlled_f64", f64_plan.plan.compiled.spirv_bytes)
                _fc_assert_positive_cache_roundtrips(ctx, c64, Float64,
                                                     f64_plan.plan, req64; label = "f64")
            elseif !strict
                @info "Float64 SignedZeroInfNanPreserve is unsupported; skipping positive Float64 dispatches"
                @test_skip supported & req64 == req64
            end

            if supported & req32 == req32
                captured_host = Float32[10, 20, 30, 40]
                source_host = Float32[1, 2, 3, 4]
                out = Lava.LavaArray(zeros(Float32, 4))
                capture_key = _fc_launch_capture!(ctx, backend, out, source_host, captured_host)
                GC.gc(true)
                KA.synchronize(backend)
                @test Array(out) == source_host .* 3.0f0 .+ 2.5f0 .+ captured_host
                capture_plan = only(ctx.caches.launchplans[capture_key])
                @test Lava.spirv_float_controls_mask(capture_plan.compiled.spirv_bytes) == req32
                _fc_export_shader_artifact("plan_capture_controlled_f32", capture_plan.compiled.spirv_bytes)
            elseif !strict
                @info "Float32 capture dispatch requires Float32 preservation support; skipping"
                @test_skip supported & req32 == req32
            end
        end
    end
end
