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

const _FAST_KERNELS = IdDict{Any,Any}()
const _FAST_KERNELS_LOCK = ReentrantLock()

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

function _fast_launch(f::F, args::Tuple; kwargs...) where {F}
    state = _CC.active_state()
    s = state.stream
    own = !Threads.atomic_xchg!(_FAST_MANAGED_BUSY, true)
    managed = own ? _FAST_MANAGED : _CC.Managed[]
    try
        GC.@preserve args begin
            ad = _CC.KernelAdaptor(managed)
            conv = map(a -> _CC.adapt(ad, a), args)
            kernel = _fast_kernel(f, _CC.argument_types(conv), state.context)
            fast = true
            @inbounds for m in managed
                fast &= _fast_owned(m, s)
            end
            if fast
                @inbounds for m in managed
                    m.dirty = true
                end
                _CC.launch_converted(kernel, conv; stream = s, kwargs...)
            else
                _CC.with_managed(managed; stream = s) do
                    _CC.launch_converted(kernel, conv; stream = s, kwargs...)
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
    kws = map(ex[1:end-1]) do kw
        Meta.isexpr(kw, :(=)) || error("@fastcuda: expected keyword=value, got $kw")
        kw.args[1] in (:threads, :blocks, :shmem) ||
            error("@fastcuda: unsupported keyword $(kw.args[1])")
        Expr(:kw, kw.args[1], esc(kw.args[2]))
    end
    f = esc(call.args[1])
    args = map(esc, call.args[2:end])
    return :(_fast_launch($f, ($(args...),); $(kws...)))
end
