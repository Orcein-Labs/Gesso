# Persistent CPU worker pool for chunked, data-parallel kernels (stdlib only).
#
#   pool_run!(f, nchunks)
#
# Calls f(c) for every c in 1:nchunks and returns once all of them have
# finished. The caller claims chunks too, so the pool never waits on a worker
# that has not started. Chunk ids are claimed from one atomic word, never
# indexed by threadid(), so a chunk's result does not depend on which thread
# ran it. A kernel whose chunks write disjoint outputs is therefore bit-exact
# against its serial loop.
#
# State passing. f is stored in the pool while the call runs, so it must not
# be reassigned during the call. Pass state in one of two ways:
#   * a callable struct built once and reused (no allocation per call), or
#   * a closure over locals that are never reassigned.
# A closure that captures locals allocates once per call, because the pool
# keeps a reference to it. A closure over a local that is reassigned captures
# a Core.Box, which is the boxing the pool exists to avoid.
#
# Workers. The pool holds participants-1 worker tasks, started on first use.
# A worker spins on the generation word for GESSO_CPU_POOL_SPIN_US (default
# 200) after each job, then parks on a Condition. A parked pool costs no CPU.
# The spin keeps back-to-back calls off the wake path. The spin is also the
# most CPU a burst can leave behind: at most workers x spin per burst. The
# wake cost after a park was measured separately (about the same burst time
# after 50 ms idle as back-to-back), so the spin is a CPU trade, not a latency
# fix. The gap between calls in a real decode step is not measured yet.
#
# Concurrency. One top-level call owns the pool at a time. A call made while
# the pool is owned (nested inside a chunk, or from another task) runs its
# chunks inline, serially, on the calling task. It never waits on the pool,
# so it cannot deadlock.
#
# Exceptions. The first exception in a job is kept, the remaining chunks of
# that job are counted but not run, and the exception is rethrown to the
# caller after every chunk has finished or been skipped. The pool stays
# usable afterwards.

using Base.Threads: Atomic, atomic_add!, atomic_cas!, atomic_xchg!

const CPU_POOL_MAX_PARTICIPANTS = 18
const CPU_POOL_THREADS_ENV = "GESSO_CPU_POOL_THREADS"
const CPU_POOL_SPIN_ENV = "GESSO_CPU_POOL_SPIN_US"
const CPU_POOL_DEFAULT_SPIN_US = 200
const _CPU_POOL_SYSFS = "/sys/devices/system/cpu"
const _CPU_POOL_IDX_MASK = UInt64(0xffff_ffff)

# Ticket word: high 32 bits hold the job generation, low 32 bits the number of
# chunks claimed so far. A claim is a compare-and-swap on the whole word, so a
# stale claim against an earlier generation always fails.
mutable struct CPUPool
    const participants::Int
    const spin_ns::UInt64
    const cond::Base.Threads.Condition
    const ticket::Atomic{UInt64}
    const completed::Atomic{Int}
    const failed::Atomic{Int}
    const busy::Atomic{Int}
    const stop::Atomic{Int}
    const workers::Vector{Task}
    const warmed::Base.IdSet{Any}
    nchunks::Int
    job::Any
    err::Any

    function CPUPool(participants::Int, spin_ns::UInt64)
        return new(
            participants,
            spin_ns,
            Base.Threads.Condition(),
            Atomic{UInt64}(0),
            Atomic{Int}(0),
            Atomic{Int}(0),
            Atomic{Int}(0),
            Atomic{Int}(0),
            Task[],
            Base.IdSet{Any}(),
            0,
            nothing,
            nothing,
        )
    end
end

mutable struct _CPUPoolSlot
    @atomic pool::Union{Nothing,CPUPool}
end

const _CPU_POOL_SLOT = _CPUPoolSlot(nothing)
const _CPU_POOL_LOCK = ReentrantLock()
const _CPU_POOL_ATEXIT = Ref(false)

# ------------------------------------------------------------ sizing

# Distinct (package, core) pairs in a Linux sysfs topology tree. Offline CPUs
# with no topology directory are skipped. Returns nothing when the tree is
# missing or unreadable, so the caller can fall back.
function _cpu_pool_sysfs_cores(base::AbstractString)::Union{Nothing,Int}
    isdir(base) || return nothing
    cores = Set{Tuple{Int,Int}}()
    try
        for name in readdir(base)
            occursin(r"^cpu[0-9]+$", name) || continue
            topo = joinpath(base, name, "topology")
            isdir(topo) || continue
            pkg = tryparse(Int, strip(read(joinpath(topo, "physical_package_id"), String)))
            core = tryparse(Int, strip(read(joinpath(topo, "core_id"), String)))
            (pkg === nothing || core === nothing) && continue
            push!(cores, (pkg, core))
        end
    catch e
        e isa Union{SystemError,Base.IOError} || rethrow()
        return nothing
    end
    return isempty(cores) ? nothing : length(cores)
end

# Physical core count. Without a Linux topology tree the logical count is
# halved, assuming 2-way SMT. That is conservative: it can under-count (Apple
# silicon has no SMT) but cannot over-subscribe a hyperthreaded machine.
function _cpu_pool_physical_cores(base::AbstractString=_CPU_POOL_SYSFS)::Int
    n = _cpu_pool_sysfs_cores(base)
    n === nothing || return n
    return max(1, Sys.CPU_THREADS ÷ 2)
end

_cpu_pool_available_threads()::Int =
    max(1, min(Base.Threads.nthreads(), Base.Threads.threadpoolsize(:default)))

function _cpu_pool_env_int(name::AbstractString; minimum::Int=1)::Union{Nothing,Int}
    raw = get(ENV, name, nothing)
    raw === nothing && return nothing
    v = tryparse(Int, strip(raw))
    if v === nothing || v < minimum
        @warn "ignoring invalid environment value" name value = raw
        return nothing
    end
    return v
end

# Caller plus workers. Default: physical cores, the available Julia threads,
# and the cap of 18. On the 18-core, 36-thread machine measured for this pool,
# 36 threads was markedly slower than 16 to 18 under spin-based waiting.
# GESSO_CPU_POOL_THREADS overrides the first two and is clamped to the
# available threads, so it may exceed 18.
function _cpu_pool_participants(;
    cores::Integer=_cpu_pool_physical_cores(),
    threads::Integer=_cpu_pool_available_threads(),
    override::Union{Nothing,Integer}=_cpu_pool_env_int(CPU_POOL_THREADS_ENV),
)::Int
    threads = max(1, Int(threads))
    if override !== nothing
        return clamp(Int(override), 1, threads)
    end
    return clamp(min(Int(cores), threads, CPU_POOL_MAX_PARTICIPANTS), 1, threads)
end

# ------------------------------------------------------------ lifecycle

function _cpu_pool_start(participants::Int, spin_us::Int)::CPUPool
    p = CPUPool(participants, UInt64(spin_us) * UInt64(1000))
    for _ in 2:participants
        push!(p.workers, Base.Threads.@spawn _cpu_pool_worker!(p))
    end
    # A freshly spawned task can sit in the spawning thread's queue until that
    # thread yields. Yield here until every worker has run once, so the first
    # job is not served by the caller alone.
    while !all(istaskstarted, p.workers)
        yield()
    end
    return p
end

function _cpu_pool_stop!(p::CPUPool)
    p.stop[] = 1
    lock(p.cond) do
        notify(p.cond)
    end
    return nothing
end

function _cpu_pool_atexit()
    p = @atomic :acquire _CPU_POOL_SLOT.pool
    p === nothing || _cpu_pool_stop!(p)
    return nothing
end

# First use starts the pool. Double-checked under a lock, so two tasks racing
# to start it end up with one pool.
function _cpu_pool()::CPUPool
    p = @atomic :acquire _CPU_POOL_SLOT.pool
    p === nothing || return p
    return lock(_CPU_POOL_LOCK) do
        q = _CPU_POOL_SLOT.pool
        if q === nothing
            spin_us = something(
                _cpu_pool_env_int(CPU_POOL_SPIN_ENV; minimum=0),
                CPU_POOL_DEFAULT_SPIN_US,
            )
            q = _cpu_pool_start(_cpu_pool_participants(), spin_us)
            @atomic :release _CPU_POOL_SLOT.pool = q
            if !_CPU_POOL_ATEXIT[]
                _CPU_POOL_ATEXIT[] = true
                atexit(_cpu_pool_atexit)
            end
        end
        q
    end
end

# Internal hook for tests and the benchmark: replace the global pool with one of
# `participants` threads (caller included). Stops the old workers first. Must not
# be called while a pool_run! is in flight.
function _cpu_pool_configure!(participants::Integer; spin_us::Integer=CPU_POOL_DEFAULT_SPIN_US)
    n = clamp(Int(participants), 1, _cpu_pool_available_threads())
    return lock(_CPU_POOL_LOCK) do
        old = _CPU_POOL_SLOT.pool
        if old !== nothing
            _cpu_pool_stop!(old)
            for t in old.workers
                try
                    wait(t)
                catch
                end
            end
        end
        q = _cpu_pool_start(n, Int(spin_us))
        @atomic :release _CPU_POOL_SLOT.pool = q
        q
    end
end

# ------------------------------------------------------------ workers

# Blocks until the generation differs from `seen`. Spins for spin_ns first, then
# parks. Returns false when the pool is stopping.
function _cpu_pool_wait!(p::CPUPool, seen::UInt64)::Bool
    t0 = time_ns()
    while (p.ticket[] >> 32) == seen
        p.stop[] == 0 || return false
        time_ns() - t0 >= p.spin_ns && break
        GC.safepoint()
    end
    (p.ticket[] >> 32) == seen || return p.stop[] == 0
    lock(p.cond)
    try
        while (p.ticket[] >> 32) == seen && p.stop[] == 0
            wait(p.cond)
        end
    finally
        unlock(p.cond)
    end
    return p.stop[] == 0
end

function _cpu_pool_worker!(p::CPUPool)
    seen = p.ticket[] >> 32
    while _cpu_pool_wait!(p, seen)
        seen = p.ticket[] >> 32
        Base.invokelatest(_cpu_pool_help!, p)
    end
    return nothing
end

# A worker reads the job only after it has read a claimable ticket. A
# successful claim then proves the job it read is the one that ticket belongs
# to: the caller cannot replace the job until every chunk of it is claimed and
# finished.
function _cpu_pool_help!(p::CPUPool)
    while true
        t = p.ticket[]
        idx = Int(t & _CPU_POOL_IDX_MASK)
        idx < p.nchunks || return nothing
        f = p.job
        atomic_cas!(p.ticket, t, t + 1) == t || continue
        _cpu_pool_run!(p, f, idx + 1)
    end
end

# The caller's own claim loop. Same protocol as _cpu_pool_help!, with f typed.
function _cpu_pool_claim!(p::CPUPool, f::F) where {F}
    while true
        t = p.ticket[]
        idx = Int(t & _CPU_POOL_IDX_MASK)
        idx < p.nchunks || return nothing
        atomic_cas!(p.ticket, t, t + 1) == t || continue
        _cpu_pool_run!(p, f, idx + 1)
    end
end

@inline function _cpu_pool_run!(p::CPUPool, f, c::Int)
    if p.failed[] == 0
        try
            f(c)
        catch e
            _cpu_pool_fail!(p, e)
        end
    end
    atomic_add!(p.completed, 1)
    return nothing
end

function _cpu_pool_fail!(p::CPUPool, e)
    atomic_cas!(p.failed, 0, 1) == 0 && (p.err = e)
    return nothing
end

# ------------------------------------------------------------ calls

# Closes the ticket so stale claims fail, writes the job, then opens the next
# generation and wakes parked workers. The lock makes the wake race-free: a
# worker checks the generation and waits while holding the same lock.
function _cpu_pool_publish!(p::CPUPool, f::F, n::Int) where {F}
    # Workers run the chunk body inside invokelatest; compile it here, once per
    # closure type, so no worker compiles it. The owner holds p.busy, so the
    # set needs no lock.
    if !(F in p.warmed)
        precompile(_cpu_pool_run!, (CPUPool, F, Int))
        push!(p.warmed, F)
    end
    gen = p.ticket[] >> 32
    p.ticket[] = (gen << 32) | _CPU_POOL_IDX_MASK
    p.completed[] = 0
    p.failed[] = 0
    p.nchunks = n
    p.job = f
    lock(p.cond)
    try
        p.ticket[] = (gen + 1) << 32
        notify(p.cond)
    finally
        unlock(p.cond)
    end
    return nothing
end

function _cpu_pool_await!(p::CPUPool, n::Int)
    while p.completed[] < n
        yield()
    end
    p.job = nothing
    return nothing
end

"""
    pool_run!(f, nchunks) -> nothing

Call `f(c)` for each `c in 1:nchunks` across the CPU pool, the calling thread
included, and return when all have finished. See the file header for how to
pass state and for the nesting and exception rules.
"""
function pool_run!(f::F, nchunks::Integer) where {F}
    n = Int(nchunks)
    n >= 0 || throw(ArgumentError("pool_run!: nchunks must be nonnegative, got $n"))
    n == 0 && return nothing
    n < typemax(UInt32) || throw(ArgumentError("pool_run!: nchunks $n exceeds the chunk-count limit"))
    p = _cpu_pool()
    if n == 1 || p.participants == 1 || atomic_cas!(p.busy, 0, 1) != 0
        for c in 1:n
            f(c)
        end
        return nothing
    end
    try
        _cpu_pool_publish!(p, f, n)
        _cpu_pool_claim!(p, f)
        _cpu_pool_await!(p, n)
        if p.failed[] != 0
            e = p.err
            p.err = nothing
            throw(e)
        end
    finally
        atomic_xchg!(p.busy, 0)
    end
    return nothing
end
