# Kernel launches for the decode path. `@cuda` pays about 5 us of host time on
# every call before the driver sees the launch: `cufunction` takes a global
# lock, rebuilds the method instance and compiler config (two device-attribute
# queries) and looks the job up, and `with_managed` asks the driver whether the
# stream is capturing, then locks every argument buffer. A 35B token makes
# about 1100 launches, so that is 5 to 6 ms per token of pure bookkeeping.
#
# `@fastcuda` keeps the semantics and skips the repeated work. The compiled
# kernel is cached per (kernel, converted argument types, context). Buffers
# already owned by the launching stream are only marked dirty, which is all
# `take_ownership!` does for them; a buffer owned by another stream, or held in
# unified or host memory, sends the whole launch through `with_managed`.
#
# Not safe under stream capture: nothing here checks for it. A CUDA graph
# segment must launch with `@cuda`, or this path must learn to ask.

const _CC = CUDA.CUDACore

# `CuVector{T}` leaves the memory parameter free, so a struct field of that
# type is abstract and every launch that reads it dispatches at run time.
# Decode buffers are device memory; fields and caches use this alias.
const _CuVec{T} = CuArray{T,1,_CC.DeviceMemory}

const _FAST_KERNELS = IdDict{Any,Any}()
const _FAST_KERNELS_LOCK = ReentrantLock()

# CUDACore 6.3.x does not export `argument_types`; this is its 6.4 definition,
# kept local so the kernel-type key matches what CUDACore compiles. 6.3 stays
# reachable: the pinned Lava caps GPUCompiler below what 6.4 requires, so a test
# env that loads both extensions resolves 6.3.
@inline @generated function _arg_types(args::Tuple)
    :(Tuple{$((:(Core.Typeof(args[$i])) for i in 1:fieldcount(args))...)})
end

@inline function _fast_kernel(f::F, ::Type{TT}, ctx) where {F,TT}
    key = Tuple{F,TT}
    k = get(_FAST_KERNELS, key, nothing)
    if k === nothing || (k::_CC.HostKernel{F,TT}).fun.mod.ctx !== ctx
        k = Base.@lock _FAST_KERNELS_LOCK begin
            _FAST_KERNELS[key] = _CC.cufunction(f, TT)
        end
    end
    return k::_CC.HostKernel{F,TT}
end

# One scratch list of buffers, reused while no other launch holds it; a
# concurrent launch from another task gets its own.
const _FAST_MANAGED = _CC.Managed[]
const _FAST_MANAGED_BUSY = Threads.Atomic{Bool}(false)

@inline _fast_owned(m::_CC.Managed{_CC.DeviceMemory}, s) = m.stream == s
@inline _fast_owned(::_CC.Managed, s) = false

# Kernel parameters for `_raw_launch`. cuLaunchKernel copies the values before
# it returns, so one buffer serves every launch made while holding
# `_FAST_MANAGED_BUSY`. CUDACore instead boxes each argument on the heap, about
# a dozen allocations per launch, which was most of a token's 15k.
const _RAW_VALS_BYTES = 4096
const _RAW_VALS = Vector{UInt8}(undef, _RAW_VALS_BYTES)
const _RAW_PTRS = Vector{Ptr{Cvoid}}(undef, 256)

# The argument tuple `launch_converted` would hand to cudacall: the kernel
# state first, then every argument that is neither a ghost nor a constant type.
@generated function _raw_call_args(kernel::_CC.HostKernel{F,TT}, args::A) where {F,TT,A<:Tuple}
    sig = (F, TT.parameters...)
    exprs = Any[:(kernel.f); Any[:(args[$i]) for i in 1:length(A.parameters)]]
    keep = Any[e for (e, t) in zip(exprs, sig) if !(_CC.isghosttype(t) || Core.Compiler.isconstType(t))]
    return :((_CC.KernelState(kernel.state.exception_info, _CC.make_seed(kernel)), $(keep...)))
end

# Lays the values out at their natural alignment; offsets are fixed per type.
@generated function _raw_store!(vp::Ptr{UInt8}, pp::Ptr{Ptr{Cvoid}}, args::A) where {A<:Tuple}
    ex = Expr(:block)
    off = 0
    for (i, T) in enumerate(A.parameters)
        al = Base.datatype_alignment(T)
        off = cld(off, al) * al
        push!(ex.args, :(unsafe_store!(Ptr{$T}(vp + $off), args[$i])))
        push!(ex.args, :(unsafe_store!(pp, Ptr{Cvoid}(vp + $off), $i)))
        off += sizeof(T)
    end
    push!(ex.args, nothing)
    return ex
end

@generated function _raw_packable(::Type{A}) where {A<:Tuple}
    off = 0
    for T in A.parameters
        (isbitstype(T) && Base.datatype_alignment(T) <= 16) || return false
        al = Base.datatype_alignment(T)
        off = cld(off, al) * al + sizeof(T)
    end
    return off <= _RAW_VALS_BYTES && length(A.parameters) <= 256
end

@noinline function _raw_launch_failed(f, res, bd, td, shmem)
    try
        throw(_CC.CuError(res))
    catch err
        _CC.diagnose_launch_failure(f, err; blockdim = bd, threaddim = td,
                                    clusterdim = _CC.CuDim3(1), shmem)
    end
end

@inline function _raw_launch(kernel::_CC.HostKernel, cargs::Tuple, s::CuStream, threads, blocks, shmem)
    bd = _CC.CuDim3(blocks)
    td = _CC.CuDim3(threads)
    res = GC.@preserve _RAW_VALS _RAW_PTRS cargs s begin
        pp = pointer(_RAW_PTRS)
        _raw_store!(pointer(_RAW_VALS), pp, cargs)
        _CC.unchecked_cuLaunchKernel(kernel.fun, bd.x, bd.y, bd.z, td.x, td.y, td.z,
                                     shmem, s, pp, C_NULL)
    end
    res == _CC.SUCCESS || _raw_launch_failed(kernel.fun, res, bd, td, shmem)
    return nothing
end

_fast_launch(f::F, args::Tuple; threads = 1, blocks = 1, shmem = 0) where {F} =
    _fast_launch(f, args, threads, blocks, shmem)

function _fast_launch(f::F, args::Tuple, threads, blocks, shmem::Integer) where {F}
    state = _CC.active_state()
    s = state.stream
    own = !Threads.atomic_xchg!(_FAST_MANAGED_BUSY, true)
    managed = own ? _FAST_MANAGED : _CC.Managed[]
    try
        GC.@preserve args begin
            ad = _CC.KernelAdaptor(managed)
            conv = map(a -> _CC.adapt(ad, a), args)
            kernel = _fast_kernel(f, _arg_types(conv), state.context)
            fast = true
            @inbounds for m in managed
                fast &= _fast_owned(m, s)
            end
            if fast
                @inbounds for m in managed
                    m.dirty = true
                end
                cargs = _raw_call_args(kernel, conv)
                if own && _raw_packable(typeof(cargs))
                    _raw_launch(kernel, cargs, s, threads, blocks, shmem)
                else
                    _CC.launch_converted(kernel, conv; stream = s, threads, blocks, shmem)
                end
            else
                _CC.with_managed(managed; stream = s) do
                    _CC.launch_converted(kernel, conv; stream = s, threads, blocks, shmem)
                end
            end
        end
    finally
        empty!(managed)
        own && (_FAST_MANAGED_BUSY[] = false)
    end
    return nothing
end

"""
    @fastcuda threads=... blocks=... [shmem=...] kernel(args...)

`@cuda` for host-side launches on the decode path: the same launch, without
the per-call compile lookup and buffer locking. Takes only launch keywords.
"""
macro fastcuda(ex...)
    call = ex[end]
    Meta.isexpr(call, :call) || error("@fastcuda: the last argument must be a kernel call")
    kws = Dict{Symbol,Any}(:threads => 1, :blocks => 1, :shmem => 0)
    for kw in ex[1:end-1]
        Meta.isexpr(kw, :(=)) || error("@fastcuda: expected keyword=value, got $kw")
        haskey(kws, kw.args[1]) || error("@fastcuda: unsupported keyword $(kw.args[1])")
        kws[kw.args[1]] = esc(kw.args[2])
    end
    f = esc(call.args[1])
    args = map(esc, call.args[2:end])
    return :(_fast_launch($f, ($(args...),), $(kws[:threads]), $(kws[:blocks]), $(kws[:shmem])))
end
