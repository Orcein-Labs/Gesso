# Maintained-test candidate for Lava's opt-in per-kernel NoContraction contract.
# Include from test/runtests.jl in the qualified Lava checkout. This file owns
# its names in a module and uses no campaign fixtures or absolute source paths.

module LavaNoContractionTests

using Test, Lava, Random, Serialization
using Lava.KernelAbstractions: @kernel, @index, @Const

const KA = Lava.KernelAbstractions
const GPUCompiler = Lava.GPUCompiler
const F32 = Float32
const NC_DECORATION = UInt32(42) # SPIR-V DecorationNoContraction.
const NC_CORE_OPS = Set(
    UInt16[
        UInt16(Lava.Op.OpFAdd),
        UInt16(Lava.Op.OpFSub),
        UInt16(Lava.Op.OpFMul),
        UInt16(Lava.Op.OpFDiv),
    ],
)

@kernel cpu=false function separated!(out, @Const(a), @Const(b), @Const(c), n::Int)
    i = @index(Global, Linear)
    if i <= n
        @inbounds begin
            av = a[i]
            bv = b[i]
            cv = c[i]
            out[i] = av * bv + cv
        end
    end
end

@kernel cpu=false function explicit_fma!(out, @Const(a), @Const(b), @Const(c), n::Int)
    i = @index(Global, Linear)
    if i <= n
        @inbounds out[i] = muladd(a[i], b[i], c[i])
    end
end

@kernel cpu=false function integer_only!(out, @Const(a), n::Int)
    i = @index(Global, Linear)
    if i <= n
        @inbounds out[i] = a[i] + Int32(1)
    end
end

function _bytes(a::Array)
    return reinterpret(UInt8, vec(a))
end

_same_bits(a::Array, b::Array) = size(a) == size(b) && _bytes(a) == _bytes(b)

function _bigfloat_oracles(a::Vector{F32}, b::Vector{F32}, c::Vector{F32})
    length(a) == length(b) == length(c) || error("oracle input length mismatch")
    separated = Vector{F32}(undef, length(a))
    fused = similar(separated)
    setprecision(BigFloat, 256) do
        setrounding(BigFloat, RoundNearest) do
            @inbounds for i in eachindex(a, b, c)
                exact_product = BigFloat(a[i]) * BigFloat(b[i])
                stored_product = F32(exact_product)
                separated[i] = F32(BigFloat(stored_product) + BigFloat(c[i]))
                fused[i] = F32(exact_product + BigFloat(c[i]))
            end
        end
    end
    return separated, fused
end

function _signature(kernel, n::Int)
    ndrange, workgroup_size, iterspace, _ = KA.launch_config(kernel, n, nothing)
    ka_context = KA.mkcontext(kernel, ndrange, iterspace)
    array_type = Lava.LavaDeviceArray{F32,1}
    tt = Tuple{typeof(ka_context),array_type,array_type,array_type,array_type,Int}
    return tt, (length(KA.workitems(iterspace)), 1, 1)
end

function _integer_signature(kernel, n::Int)
    ndrange, workgroup_size, iterspace, _ = KA.launch_config(kernel, n, nothing)
    ka_context = KA.mkcontext(kernel, ndrange, iterspace)
    array_type = Lava.LavaDeviceArray{Int32,1}
    tt = Tuple{typeof(ka_context),array_type,array_type,Int}
    return tt, (length(KA.workitems(iterspace)), 1, 1)
end

function _header_only_spirv()
    words = UInt32[Lava.SPIRV_MAGIC, UInt32(0x00010400), 0, 1, 0]
    return collect(reinterpret(UInt8, words))
end

function _lower_spirv_id_bound(bytes::Vector{UInt8})
    words = collect(reinterpret(UInt32, bytes))
    words[4] = UInt32(1)
    return collect(reinterpret(UInt8, words))
end

function _zero_instruction_extent(bytes::Vector{UInt8})
    words = collect(reinterpret(UInt32, bytes))
    words[6] &= UInt32(0xffff)
    return collect(reinterpret(UInt8, words))
end

function _spirv_instructions(bytes::Vector{UInt8})
    length(bytes) >= 20 && length(bytes) % 4 == 0 || error("invalid SPIR-V byte extent")
    words = collect(reinterpret(UInt32, bytes))
    words[1] == Lava.SPIRV_MAGIC || error("invalid SPIR-V magic")
    instructions = Vector{Vector{UInt32}}()
    i = 6
    while i <= length(words)
        count = Int(words[i] >> 16)
        count > 0 && i + count - 1 <= length(words) || error("invalid SPIR-V instruction extent")
        push!(instructions, words[i:(i+count-1)])
        i += count
    end
    return words, instructions
end

function _coverage(bytes::Vector{UInt8})
    _, instructions = _spirv_instructions(bytes)
    required = Set{UInt32}()
    decorated = Set{UInt32}()
    for instruction in instructions
        opcode = UInt16(instruction[1] & 0xffff)
        if opcode in NC_CORE_OPS
            length(instruction) == 5 || error("unexpected covered arithmetic instruction shape")
            push!(required, instruction[3])
        elseif opcode == Lava.Op.OpDecorate &&
               length(instruction) == 3 &&
               instruction[3] == NC_DECORATION
            push!(decorated, instruction[2])
        end
    end
    return required, decorated
end

function _remove_one_decoration(bytes::Vector{UInt8})
    words, instructions = _spirv_instructions(bytes)
    required, decorated = _coverage(bytes)
    isempty(required) && error("separated shader has no covered core arithmetic")
    issubset(required, decorated) || error("source shader lacks required NoContraction coverage")
    cursor = 6
    for instruction in instructions
        opcode = UInt16(instruction[1] & 0xffff)
        if opcode == Lava.Op.OpDecorate &&
           length(instruction) == 3 &&
           instruction[3] == NC_DECORATION &&
           instruction[2] in required
            target = instruction[2]
            deleteat!(words, cursor:(cursor+length(instruction)-1))
            return collect(reinterpret(UInt8, words)), target
        end
        cursor += length(instruction)
    end
    error("no actual NoContraction decoration targets a covered arithmetic result")
end

function _has_explicit_fma(bytes::Vector{UInt8})
    disassembly = Lava.disassemble_spirv(bytes)
    return occursin(r"OpExtInst .*\bFma\b", disassembly) || occursin(r"\bOpFma\b", disassembly)
end

function _find_plan(ctx, kernel)
    found = Any[]
    ftype = typeof(kernel.f)
    for (key, plans) in ctx.caches.launchplans
        fieldcount(key) > 0 && fieldtype(key, 1) === ftype || continue
        for plan in plans
            push!(found, (key = key, plan = plan))
        end
    end
    length(found) == 1 ||
        error("expected one public launch plan for $(ftype), found $(length(found))")
    return only(found)
end

function _copy_launchplans(cache)
    saved = copy(cache)
    for key in keys(saved)
        saved[key] = copy(saved[key])
    end
    return saved
end

function _restore_dict!(destination, saved)
    empty!(destination)
    merge!(destination, saved)
    return nothing
end

function _restore_env!(key, value)
    value === nothing ? pop!(ENV, key, nothing) : (ENV[key] = value)
    return nothing
end

function _with_isolated_state(f, ctx, backend)
    disk_dir = mktempdir(; prefix = "lava-no-contraction-disk-")
    frozen_dir = mktempdir(; prefix = "lava-no-contraction-frozen-")
    debug_dir = joinpath(disk_dir, "debug")
    mkpath(debug_dir)
    env_refs = (
        debug = get(ENV, "LAVA_DEBUG_DIR", nothing),
        spirv_dump = get(ENV, "LAVA_SPIRV_DUMP_DIR", nothing),
        load_disk = get(ENV, "LAVA_LOAD_SPIRV_DISK_CACHE", nothing),
    )
    # `lava_compile_gpu_from_job` currently writes compiler diagnostics under a
    # package-relative tmp_kernels directory shared by sibling worktrees. Leave
    # those compiler-owned artifacts untouched; only the isolated dirs below
    # are owned and cleaned up by this test.
    old_refs = (
        disk_dir = Lava.LAVA_DISK_CACHE_DIR[],
        frozen_dir = Lava.FROZEN_CACHE_DIR[],
        frozen_version = Lava.FROZEN_VERSION[],
        frozen_recording = Lava.FROZEN_RECORDING[],
        frozen_hits = Lava.FROZEN_HITS[],
        frozen_stores = Lava.FROZEN_STORES[],
        frozen_misses = Lava.FROZEN_MISSES[],
        compiler_configs = copy(Lava.COMPILER_CONFIGS),
        spirv_opt = Lava.SPIRV_OPT_ENABLED[],
        frozen_last_pcache = ctx.caches.frozen_last_pcache,
        launchplans = _copy_launchplans(ctx.caches.launchplans),
        linked = copy(ctx.caches.linked),
        pipelines = copy(ctx.caches.pipelines),
        pipeline_order = copy(ctx.caches.pipeline_order),
        iterplans = copy(ctx.caches.iterplans),
        frozen_mem = copy(ctx.caches.frozen_mem),
        diag_dump_dir = ctx.diag.spirv_dump_dir,
        diag_dump_counter = ctx.diag.spirv_dump_counter,
        diag_dispatch_logging = ctx.diag.dispatch_logging,
        diag_dispatch_log_file = ctx.diag.dispatch_log_file,
        diag_dispatch_timing = ctx.diag.dispatch_timing,
        diag_dispatch_log = copy(ctx.diag.dispatch_log),
    )
    try
        Lava.LAVA_DISK_CACHE_DIR[] = disk_dir
        Lava.FROZEN_CACHE_DIR[] = frozen_dir
        ENV["LAVA_DEBUG_DIR"] = debug_dir
        ENV["LAVA_SPIRV_DUMP_DIR"] = debug_dir
        ENV["LAVA_LOAD_SPIRV_DISK_CACHE"] = "0"
        ctx.diag.spirv_dump_dir = debug_dir
        ctx.diag.spirv_dump_counter = 0
        ctx.diag.dispatch_logging = false
        ctx.diag.dispatch_log_file = nothing
        ctx.diag.dispatch_timing = false
        ctx.caches.frozen_last_pcache = nothing
        Lava.FROZEN_VERSION[] = ""
        Lava.FROZEN_RECORDING[] = false
        Lava.SPIRV_OPT_ENABLED[] = true
        return f(disk_dir, frozen_dir)
    finally
        try
            KA.synchronize(backend)
        catch
            # Preserve the original test exception; the package test runner
            # already owns the context lifecycle.
        end
        Lava.LAVA_DISK_CACHE_DIR[] = old_refs.disk_dir
        Lava.FROZEN_CACHE_DIR[] = old_refs.frozen_dir
        Lava.FROZEN_VERSION[] = old_refs.frozen_version
        Lava.FROZEN_RECORDING[] = old_refs.frozen_recording
        Lava.FROZEN_HITS[] = old_refs.frozen_hits
        Lava.FROZEN_STORES[] = old_refs.frozen_stores
        Lava.FROZEN_MISSES[] = old_refs.frozen_misses
        empty!(Lava.COMPILER_CONFIGS)
        merge!(Lava.COMPILER_CONFIGS, old_refs.compiler_configs)
        Lava.SPIRV_OPT_ENABLED[] = old_refs.spirv_opt
        ctx.caches.frozen_last_pcache = old_refs.frozen_last_pcache
        _restore_dict!(ctx.caches.launchplans, old_refs.launchplans)
        _restore_dict!(ctx.caches.linked, old_refs.linked)
        _restore_dict!(ctx.caches.pipelines, old_refs.pipelines)
        empty!(ctx.caches.pipeline_order)
        append!(ctx.caches.pipeline_order, old_refs.pipeline_order)
        _restore_dict!(ctx.caches.iterplans, old_refs.iterplans)
        _restore_dict!(ctx.caches.frozen_mem, old_refs.frozen_mem)
        ctx.diag.spirv_dump_dir = old_refs.diag_dump_dir
        ctx.diag.spirv_dump_counter = old_refs.diag_dump_counter
        ctx.diag.dispatch_logging = old_refs.diag_dispatch_logging
        ctx.diag.dispatch_log_file = old_refs.diag_dispatch_log_file
        ctx.diag.dispatch_timing = old_refs.diag_dispatch_timing
        empty!(ctx.diag.dispatch_log)
        append!(ctx.diag.dispatch_log, old_refs.diag_dispatch_log)
        _restore_env!("LAVA_DEBUG_DIR", env_refs.debug)
        _restore_env!("LAVA_SPIRV_DUMP_DIR", env_refs.spirv_dump)
        _restore_env!("LAVA_LOAD_SPIRV_DISK_CACHE", env_refs.load_disk)
        rm(disk_dir; recursive = true, force = true)
        rm(frozen_dir; recursive = true, force = true)
    end
end

function _core_pipeline_key(compiled)
    return hash((
        Lava.spirv_content_hash(compiled.spirv_bytes),
        length(compiled.spirv_bytes),
        compiled.entry_name,
        compiled.push_info.push_size,
        compiled.enable_ray_query,
    ))
end

function _test_compiler_and_cache(ctx, stock_sep, separated, fma, n, disk_dir)
    tt_stock, wg_stock = _signature(stock_sep, n)
    tt_sep, wg = _signature(separated, n)
    tt_fma, wg_fma = _signature(fma, n)
    @test wg_stock == wg
    @test wg == wg_fma

    header_only = _header_only_spirv()
    # The pure coverage scan is arithmetic-only and therefore vacuous here;
    # structural validation must reject the incomplete module.
    @test Lava.spirv_no_contraction_satisfied(header_only)
    @test !Lava.spirv_no_contraction_conformant(header_only)
    @test_throws ArgumentError Lava.require_spirv_no_contraction(header_only, true)

    integer_kernel = Lava.with_no_contraction(integer_only!(stock_sep.backend, 64))
    tt_integer, wg_integer = _integer_signature(integer_kernel, n)
    integer_compiled =
        Lava.lava_compile_full(integer_kernel.f, tt_integer; workgroup_size = wg_integer)
    integer_required, _ = _coverage(integer_compiled.spirv_bytes)
    @test isempty(integer_required)
    @test Lava.spirv_no_contraction_satisfied(integer_compiled.spirv_bytes)
    @test Lava.spirv_no_contraction_conformant(integer_compiled.spirv_bytes)
    @test Lava.require_spirv_no_contraction(integer_compiled.spirv_bytes, true) === nothing

    opt_before = Lava.SPIRV_OPT_ENABLED[]
    raw_stock = raw_sep = raw_fma = opt_sep = opt_fma = nothing
    try
        Lava.SPIRV_OPT_ENABLED[] = false
        raw_stock = Lava.lava_compile_full(stock_sep.f, tt_stock; workgroup_size = wg_stock)
        raw_sep = Lava.lava_compile_full(separated.f, tt_sep; workgroup_size = wg)
        raw_fma = Lava.lava_compile_full(fma.f, tt_fma; workgroup_size = wg_fma)
        Lava.SPIRV_OPT_ENABLED[] = true
        opt_sep = Lava.optimize_spirv(raw_sep.spirv_bytes)
        opt_fma = Lava.optimize_spirv(raw_fma.spirv_bytes)
    finally
        Lava.SPIRV_OPT_ENABLED[] = opt_before
    end

    stock_required, stock_decorated = _coverage(raw_stock.spirv_bytes)
    @test !isempty(stock_required)
    @test isempty(intersect(stock_required, stock_decorated))

    for (label, bytes) in
        (("separated pre-opt", raw_sep.spirv_bytes), ("separated post-opt", opt_sep))
        required, decorated = _coverage(bytes)
        @test !isempty(required)
        @test issubset(required, decorated)
        @test Lava.spirv_no_contraction_satisfied(bytes)
    end
    for (label, bytes) in
        (("explicit FMA pre-opt", raw_fma.spirv_bytes), ("explicit FMA post-opt", opt_fma))
        @test Lava.spirv_no_contraction_satisfied(bytes)
        @test _has_explicit_fma(bytes)
    end

    # Also verify the actual public-dispatch artifacts, which have passed the
    # regular compiler/optimizer/validation path rather than the direct probes.
    sep_plan = _find_plan(ctx, separated).plan
    fma_plan = _find_plan(ctx, fma).plan
    @test Lava.spirv_no_contraction_satisfied(sep_plan.compiled.spirv_bytes)
    @test Lava.spirv_no_contraction_satisfied(fma_plan.compiled.spirv_bytes)
    @test _has_explicit_fma(fma_plan.compiled.spirv_bytes)

    # Derive the exact public KA signature for the real frozen and serialized
    # disk-cache APIs.
    tt = tt_sep
    source = GPUCompiler.methodinstance(typeof(separated.f), tt)
    @test Lava.lava_disk_cache_key(source, wg, UInt8(0), false) !=
          Lava.lava_disk_cache_key(source, wg, UInt8(0), true)

    old_version, old_recording = Lava.FROZEN_VERSION[], Lava.FROZEN_RECORDING[]
    version = "no-contraction-maintained-" * string(time_ns())
    try
        Lava.FROZEN_VERSION[] = version
        Lava.FROZEN_RECORDING[] = true
        @test Lava.frozen_key(stock_sep.f, tt, wg) != Lava.frozen_key(separated.f, tt, wg)
    finally
        Lava.FROZEN_VERSION[] = old_version
        Lava.FROZEN_RECORDING[] = old_recording
    end

    # The ordinary disk cache is also contract checked. Plant a structurally
    # valid shader made by removing one real decoration from this compiled
    # kernel; the altered artifact must miss before it can be linked.
    good = sep_plan.compiled
    mutated_bytes, target_id = _remove_one_decoration(good.spirv_bytes)
    @test target_id != 0
    @test !Lava.spirv_no_contraction_satisfied(mutated_bytes)
    @test_throws ArgumentError Lava.require_spirv_no_contraction(mutated_bytes, true)
    Lava.validate_spirv(mutated_bytes)
    bad = Lava.LavaGPUKernel(
        mutated_bytes,
        good.entry_name,
        good.workgroup_size,
        good.push_info,
        good.ir,
        good.enable_ray_query,
    )

    disk_path = joinpath(disk_dir, Lava.lava_disk_cache_key(source, wg, UInt8(0), true))
    Lava.lava_disk_cache_store(source, wg, good, UInt8(0), true)
    good_disk_bytes = read(disk_path)
    bad_disk_entry =
        (spec_types = source.specTypes, workgroup_size = wg, no_contraction = true, kernel = bad)
    open(disk_path, "w") do io
        Serialization.serialize(io, bad_disk_entry)
    end
    @test Lava.lava_disk_cache_load(source, wg, UInt8(0), true) === nothing
    bad_disk_bytes = read(disk_path)
    @test_throws ArgumentError Lava.lava_disk_cache_store(source, wg, bad, UInt8(0), true)
    @test read(disk_path) == bad_disk_bytes
    write(disk_path, good_disk_bytes)
    disk_hit = Lava.lava_disk_cache_load(source, wg, UInt8(0), true)
    @test disk_hit isa Lava.LavaGPUKernel
    if disk_hit isa Lava.LavaGPUKernel
        @test disk_hit.spirv_bytes == good.spirv_bytes
        @test Lava.spirv_no_contraction_satisfied(disk_hit.spirv_bytes)
    end

    # A parseable but invalid ID bound keeps the coverage-only predicate true;
    # strict disk-cache admission must reject it as a cache miss.
    bad_bound_bytes = _lower_spirv_id_bound(good.spirv_bytes)
    @test Lava.spirv_no_contraction_satisfied(bad_bound_bytes)
    @test !Lava.spirv_no_contraction_conformant(bad_bound_bytes)
    bad_bound = Lava.LavaGPUKernel(
        bad_bound_bytes,
        good.entry_name,
        good.workgroup_size,
        good.push_info,
        good.ir,
        good.enable_ray_query,
    )
    bad_bound_entry = (
        spec_types = source.specTypes,
        workgroup_size = wg,
        no_contraction = true,
        kernel = bad_bound,
    )
    open(disk_path, "w") do io
        Serialization.serialize(io, bad_bound_entry)
    end
    @test Lava.lava_disk_cache_load(source, wg, UInt8(0), true) === nothing
    write(disk_path, good_disk_bytes)

    for invalid_bytes in (_header_only_spirv(), _zero_instruction_extent(good.spirv_bytes))
        invalid = Lava.LavaGPUKernel(
            invalid_bytes,
            good.entry_name,
            good.workgroup_size,
            good.push_info,
            good.ir,
            good.enable_ray_query,
        )
        invalid_entry = (
            spec_types = source.specTypes,
            workgroup_size = wg,
            no_contraction = true,
            kernel = invalid,
        )
        open(disk_path, "w") do io
            Serialization.serialize(io, invalid_entry)
        end
        invalid_disk_bytes = read(disk_path)
        @test Lava.lava_disk_cache_load(source, wg, UInt8(0), true) === nothing
        @test_throws ArgumentError Lava.lava_disk_cache_store(source, wg, invalid, UInt8(0), true)
        @test read(disk_path) == invalid_disk_bytes
        write(disk_path, good_disk_bytes)
    end

    # Frozen-cache memory and disk entries use the same live callable/signature
    # as the successful public KA dispatch. The bad linked record reuses the
    # genuine good pipeline/argument metadata but is never executed.
    memkey = (typeof(separated.f), tt, wg)
    had_mem = haskey(ctx.caches.frozen_mem, memkey)
    old_mem = get(ctx.caches.frozen_mem, memkey, nothing)
    frozen_path = Lava.with_frozen_recording(version) do
        Lava.frozen_path(Lava.frozen_key(separated.f, tt, wg))
    end
    frozen_version = Lava.FROZEN_VERSION[]
    frozen_recording = Lava.FROZEN_RECORDING[]
    try
        Lava.with_frozen_recording(version) do
            Lava.frozen_store(ctx, separated.f, tt, wg, good)
        end
        good_frozen_bytes = read(frozen_path)
        bad_link = Lava.LavaLinkedKernel(
            bad,
            sep_plan.pipeline,
            copy(sep_plan.offsets),
            copy(sep_plan.byval_sizes),
        )
        bad_pipeline_key = _core_pipeline_key(bad)
        @test !haskey(ctx.caches.pipelines, bad_pipeline_key)

        Lava.use_frozen_kernels(version)
        ctx.caches.frozen_mem[memkey] = bad_link
        bad_serialized = IOBuffer()
        Serialization.serialize(bad_serialized, bad)
        bad_frozen_bytes = take!(bad_serialized)
        write(frozen_path, bad_frozen_bytes)
        before_pipeline_keys = Set(keys(ctx.caches.pipelines))
        before_dispatches = ctx.diag.total_dispatches[]
        @test Lava.frozen_load(ctx, separated.f, tt, wg) === nothing
        @test !haskey(ctx.caches.frozen_mem, memkey)
        @test Set(keys(ctx.caches.pipelines)) == before_pipeline_keys
        @test !haskey(ctx.caches.pipelines, bad_pipeline_key)
        @test ctx.diag.total_dispatches[] == before_dispatches
        # A second read covers the disk-only corruption path after the invalid
        # memory value has been evicted.
        @test Lava.frozen_load(ctx, separated.f, tt, wg) === nothing
        @test !haskey(ctx.caches.pipelines, bad_pipeline_key)
        @test Set(keys(ctx.caches.pipelines)) == before_pipeline_keys
        @test ctx.diag.total_dispatches[] == before_dispatches

        @test_throws ArgumentError Lava.with_frozen_recording(version) do
            Lava.frozen_store(ctx, separated.f, tt, wg, bad)
        end
        @test read(frozen_path) == bad_frozen_bytes

        # Preserve the operation/decorations but invalidate the SPIR-V ID bound.
        # Memory and disk entries must both miss without linking or dispatching.
        bad_bound_bytes = _lower_spirv_id_bound(good.spirv_bytes)
        @test Lava.spirv_no_contraction_satisfied(bad_bound_bytes)
        @test !Lava.spirv_no_contraction_conformant(bad_bound_bytes)
        bad_bound = Lava.LavaGPUKernel(
            bad_bound_bytes,
            good.entry_name,
            good.workgroup_size,
            good.push_info,
            good.ir,
            good.enable_ray_query,
        )
        bad_bound_link = Lava.LavaLinkedKernel(
            bad_bound,
            sep_plan.pipeline,
            copy(sep_plan.offsets),
            copy(sep_plan.byval_sizes),
        )
        bad_bound_io = IOBuffer()
        Serialization.serialize(bad_bound_io, bad_bound)
        bad_bound_frozen_bytes = take!(bad_bound_io)
        write(frozen_path, bad_bound_frozen_bytes)
        ctx.caches.frozen_mem[memkey] = bad_bound_link
        before_bad_bound_pipelines = Set(keys(ctx.caches.pipelines))
        before_bad_bound_dispatches = ctx.diag.total_dispatches[]
        @test Lava.frozen_load(ctx, separated.f, tt, wg) === nothing
        @test !haskey(ctx.caches.frozen_mem, memkey)
        @test Set(keys(ctx.caches.pipelines)) == before_bad_bound_pipelines
        @test ctx.diag.total_dispatches[] == before_bad_bound_dispatches
        @test Lava.frozen_load(ctx, separated.f, tt, wg) === nothing
        @test Set(keys(ctx.caches.pipelines)) == before_bad_bound_pipelines
        @test ctx.diag.total_dispatches[] == before_bad_bound_dispatches
        @test_throws ArgumentError Lava.with_frozen_recording(version) do
            Lava.frozen_store(ctx, separated.f, tt, wg, bad_bound)
        end

        for invalid_bytes in (_header_only_spirv(), _zero_instruction_extent(good.spirv_bytes))
            invalid = Lava.LavaGPUKernel(
                invalid_bytes,
                good.entry_name,
                good.workgroup_size,
                good.push_info,
                good.ir,
                good.enable_ray_query,
            )
            invalid_link = Lava.LavaLinkedKernel(
                invalid,
                sep_plan.pipeline,
                copy(sep_plan.offsets),
                copy(sep_plan.byval_sizes),
            )
            invalid_io = IOBuffer()
            Serialization.serialize(invalid_io, invalid)
            invalid_file = take!(invalid_io)
            write(frozen_path, invalid_file)
            ctx.caches.frozen_mem[memkey] = invalid_link
            @test Lava.frozen_load(ctx, separated.f, tt, wg) === nothing
            @test !haskey(ctx.caches.frozen_mem, memkey)
            @test Lava.frozen_load(ctx, separated.f, tt, wg) === nothing
            @test Set(keys(ctx.caches.pipelines)) == before_bad_bound_pipelines
            @test ctx.diag.total_dispatches[] == before_bad_bound_dispatches
            @test_throws ArgumentError Lava.with_frozen_recording(version) do
                Lava.frozen_store(ctx, separated.f, tt, wg, invalid)
            end
            @test read(frozen_path) == invalid_file
        end

        # Restore the real cache entry. A bad memory value is evicted, the
        # valid disk shader is linked through its existing genuine pipeline,
        # and the next call is a positive memory hit.
        write(frozen_path, good_frozen_bytes)
        ctx.caches.frozen_mem[memkey] = bad_link
        restored = Lava.frozen_load(ctx, separated.f, tt, wg)
        @test restored isa Lava.LavaLinkedKernel
        if restored isa Lava.LavaLinkedKernel
            @test restored.compiled.spirv_bytes == good.spirv_bytes
            @test Lava.spirv_no_contraction_satisfied(restored.compiled.spirv_bytes)
            @test ctx.caches.frozen_mem[memkey] === restored
            @test Lava.frozen_load(ctx, separated.f, tt, wg) === restored
        end
    finally
        Lava.FROZEN_VERSION[] = frozen_version
        Lava.FROZEN_RECORDING[] = frozen_recording
        had_mem ? (ctx.caches.frozen_mem[memkey] = old_mem) : delete!(ctx.caches.frozen_mem, memkey)
    end
    return nothing
end

function _test_public_dispatch(backend, ctx, n)
    rng = MersenneTwister(0xC1_2026_1018)
    a = 4.0f0 .* rand(rng, F32, n) .- 2.0f0
    b = 4.0f0 .* rand(rng, F32, n) .- 2.0f0
    c = 4.0f0 .* rand(rng, F32, n) .- 2.0f0
    a[1] = nextfloat(1.0f0)
    b[1] = 1.0f0 - ldexp(1.0f0, -23)
    c[1] = -1.0f0
    separated_ref, fused_ref = _bigfloat_oracles(a, b, c)
    @test separated_ref[1] == 0.0f0 && !signbit(separated_ref[1])
    @test fused_ref[1] == -ldexp(1.0f0, -46)

    ad = Lava.LavaArray(copy(a))
    bd = Lava.LavaArray(copy(b))
    cd = Lava.LavaArray(copy(c))
    outd = Lava.LavaArray(fill(F32(NaN), n))
    Lava.vk_context(backend) === ctx || error("test arrays are not on the expected Vulkan context")

    stock_sep = separated!(backend, 64)
    stock_fma = explicit_fma!(backend, 64)
    no_contract_sep = Lava.with_no_contraction(stock_sep)
    no_contract_fma = Lava.with_no_contraction(stock_fma)

    @testset "host API composition, adaptation and cache identity" begin
        no_contract = no_contract_sep
        controlled = Lava.with_float_controls(stock_sep, (32,))
        both_a = Lava.with_float_controls(no_contract, (32,))
        both_b = Lava.with_no_contraction(controlled)
        @test no_contract === Lava.with_no_contraction(no_contract)
        @test !Lava.requires_no_contraction(typeof(stock_sep.f))
        @test Lava.required_float_controls(typeof(stock_sep.f)) == UInt8(0)
        @test typeof(both_a.f) === typeof(both_b.f)
        @test Lava.required_float_controls(typeof(both_a.f)) == Lava.float_controls_mask((32,))
        @test Lava.requires_no_contraction(typeof(both_a.f))
        @test Lava.kernel_origin_type(typeof(no_contract.f)) === typeof(stock_sep.f)
        @test KA.backend(no_contract) === KA.backend(stock_sep)
        @test KA.workgroupsize(no_contract) == KA.workgroupsize(stock_sep)
        @test KA.ndrange(no_contract) == KA.ndrange(stock_sep)
        adapted = Lava.Adapt.adapt(Array, no_contract.f)
        @test typeof(adapted) === typeof(no_contract.f)
        @test Lava.requires_no_contraction(typeof(adapted))
        adapted_both = Lava.Adapt.adapt(Array, both_a.f)
        @test typeof(adapted_both) === typeof(both_a.f)
        @test Lava.required_float_controls(typeof(adapted_both)) == Lava.float_controls_mask((32,))
        @test Lava.requires_no_contraction(typeof(adapted_both))
        @test Lava.lava_compiler_config(; no_contraction = false).params.no_contraction == false
        @test Lava.lava_compiler_config(; no_contraction = true).params.no_contraction == true
        @test Lava.lava_compiler_config(; no_contraction = false) !==
              Lava.lava_compiler_config(; no_contraction = true)
        combined_config = Lava.lava_compiler_config(;
            float_controls = Lava.float_controls_mask((32,)),
            no_contraction = true,
        )
        @test combined_config.params.float_controls == Lava.float_controls_mask((32,))
        @test combined_config.params.no_contraction
    end

    @testset "public dispatch, poisoned repeats, BigFloat witness" begin
        original_inputs = (copy(a), copy(b), copy(c))
        previous = Dict{String,Array{F32,1}}()
        for (label, kernel, expected) in (
            ("separated", no_contract_sep, separated_ref),
            ("explicit_fma", no_contract_fma, fused_ref),
        )
            for rep = 1:3
                copyto!(outd, fill(F32(NaN), n))
                KA.synchronize(backend)
                kernel(outd, ad, bd, cd, n; ndrange = n)
                KA.synchronize(backend)
                got = Array(outd)
                @test all(isfinite, got)
                @test _same_bits(got, expected)
                @test _same_bits(Array(ad), original_inputs[1])
                @test _same_bits(Array(bd), original_inputs[2])
                @test _same_bits(Array(cd), original_inputs[3])
                if haskey(previous, label)
                    @test _same_bits(got, previous[label])
                else
                    previous[label] = copy(got)
                end
            end
        end
    end

    return stock_sep, no_contract_sep, no_contract_fma
end

function run()
    @testset "per-kernel NoContraction contract" begin
        backend = Lava.LavaBackend()
        ctx = Lava.vk_context(backend)
        n = 4096
        _with_isolated_state(ctx, backend) do disk_dir, frozen_dir
            stock_sep, no_contract_sep, no_contract_fma = _test_public_dispatch(backend, ctx, n)
            _test_compiler_and_cache(ctx, stock_sep, no_contract_sep, no_contract_fma, n, disk_dir)
        end
    end
    return nothing
end

end # module LavaNoContractionTests

LavaNoContractionTests.run()
