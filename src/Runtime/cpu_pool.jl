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
# most CPU a burst can leave behind: at most workers x spin per burst.
#
# Holds. A caller that knows more work follows shortly can call
# pool_hold!(ns), and workers keep spinning until that deadline instead of
# parking. Measured inside a 35B CUDA decode, where host-expert calls are
# separated by about 1 to 2 ms of GPU work: parked workers woke 115 us late and
# then ran each chunk about twice as slowly as back-to-back (the core had left
# its clock and C-state), so a host MoE call took 1701 us. Held through the
# gap, the same call took 913 us. A synthetic idle loop does not show this
# (after 50 ms idle the burst time matched back-to-back), so measure in situ.
# A held worker yields every 50 us, so other tasks on its thread are delayed
# by at most that much.
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
const _CPU_POOL_HOLD_YIELD_NS = UInt64(50_000)
const _CPU_POOL_IDX_MASK = UInt64(0xffff_ffff)

# Node lanes (pinned pools on more than one NUMA node). Chunks 1:n are split
# into contiguous ranges, one per lane, in proportion to the lane's worker
# count. Each lane has its own ticket. A lane's workers claim from their lane
# first and take chunks from other lanes only when their own is empty, so every
# chunk is still claimed by someone and nothing can be stranded. A chunk's
# lane is fixed by its index, so a copy of a chunk's rows made by its lane's
# worker lands on that worker's node.
struct _CPULanes
    weights::Vector{Int}          # workers per lane
    tickets::Vector{Atomic{UInt64}}
    bases::Vector{Int}            # chunk index before the lane's first chunk
    counts::Vector{Int}           # chunks in the lane for the current job
end

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
    const hot_until::Atomic{UInt64}
    const workers::Vector{Task}
    const warmed::Base.IdSet{Any}
    nchunks::Int
    job::Any
    err::Any
    lanes::Union{Nothing, _CPULanes}

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
            Atomic{UInt64}(0),
            Task[],
            Base.IdSet{Any}(),
            0,
            nothing,
            nothing,
            nothing,
        )
    end
end

mutable struct _CPUPoolSlot
    @atomic pool::Union{Nothing, CPUPool}
end

const _CPU_POOL_SLOT = _CPUPoolSlot(nothing)
const _CPU_POOL_LOCK = ReentrantLock()
const _CPU_POOL_ATEXIT = Ref(false)

# ------------------------------------------------------------ sizing

_cpu_pool_available_threads()::Int =
    max(1, min(Base.Threads.nthreads(), Base.Threads.threadpoolsize(:default)))

function _cpu_pool_env_int(name::AbstractString; minimum::Int=1)::Union{Nothing, Int}
    raw = get(ENV, name, nothing)
    raw === nothing && return nothing
    v = tryparse(Int, strip(raw))
    if v === nothing || v < minimum
        @warn "ignoring invalid environment value" name value = raw
        return nothing
    end
    return v
end

# Caller plus workers. Default: the available Julia threads, capped at 18.
# SMT siblings count. On an 8-core, 16-thread EPYC Rome VM, 16 participants
# ran decode about 35% faster and a 512-token prefill about 50% faster than 8
# (one per physical core), with the same output bits. On the 18-core,
# 36-thread machine measured for this pool, 36 threads was markedly slower
# than 16 to 18 under spin-based waiting, hence the cap.
# GESSO_CPU_POOL_THREADS overrides the default and is clamped to the
# available threads, so it may exceed 18.
function _cpu_pool_participants(;
    threads::Integer=_cpu_pool_available_threads(),
    override::Union{Nothing, Integer}=_cpu_pool_env_int(CPU_POOL_THREADS_ENV),
)::Int
    threads = max(1, Int(threads))
    if override !== nothing
        return clamp(Int(override), 1, threads)
    end
    return min(threads, CPU_POOL_MAX_PARTICIPANTS)
end

# ------------------------------------------------------------ lifecycle

function _cpu_pool_start(participants::Int, spin_us::Int)::CPUPool
    p = CPUPool(participants, UInt64(spin_us) * UInt64(1000))
    assign = _cpu_pool_pin_assignments(participants)
    assign === nothing || _pin_reset!()
    lane_of, weights = _cpu_lane_plan(assign)
    if weights !== nothing
        L = length(weights)
        p.lanes = _CPULanes(
            weights,
            [Atomic{UInt64}(0) for _ in 1:L],
            zeros(Int, L),
            zeros(Int, L),
        )
    end
    for k in 2:participants
        if assign === nothing
            push!(p.workers, Base.Threads.@spawn _cpu_pool_worker!(p))
        else
            _, cpu = assign[k - 1]
            lane = lane_of === nothing ? 0 : lane_of[k - 1]
            push!(p.workers, Base.Threads.@spawn _cpu_pool_pinned_worker!(p, cpu, lane))
        end
    end
    # A freshly spawned task can sit in the spawning thread's queue until that
    # thread yields. Yield here until every worker has run once, so the first
    # job is not served by the caller alone.
    while !all(istaskstarted, p.workers)
        yield()
    end
    # Pinned workers record their outcome when they start; wait for all of them.
    if assign !== nothing
        while _pin_logged_count() < participants - 1
            yield()
        end
    end
    return p
end

# One (node_index, cpu) per worker. The caller thread is never pinned.
# nothing means pinning is off and the pool starts exactly as before.
function _cpu_pool_pin_assignments(participants::Int)::Union{Nothing, Vector{Tuple{Int, Int}}}
    participants > 1 || return nothing
    topo = numa_topology()
    pin_threads_enabled(topo) || return nothing
    return pin_assignments(topo, participants - 1)
end

# Lanes exist only when the pinned workers span at least two nodes. Returns the
# lane of each worker and the workers per lane, or (nothing, nothing).
function _cpu_lane_plan(assign)
    assign === nothing && return nothing, nothing
    nodes = unique(first.(assign))
    length(nodes) >= 2 || return nothing, nothing
    lane_of_node = Dict(n => i for (i, n) in enumerate(nodes))
    lane_of = [lane_of_node[node] for (node, _) in assign]
    weights = [count(==(n), first.(assign)) for n in nodes]
    return lane_of, weights
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
function _cpu_pool_configure!(
    participants::Integer;
    spin_us::Integer=CPU_POOL_DEFAULT_SPIN_US,
)
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

# Blocks until the generation differs from `seen`. Spins for spin_ns or until
# the hold deadline, whichever is later, then parks. Returns false when the
# pool is stopping.
function _cpu_pool_wait!(p::CPUPool, seen::UInt64)::Bool
    t0 = time_ns()
    last_yield = t0
    while (p.ticket[] >> 32) == seen
        p.stop[] == 0 || return false
        now = time_ns()
        if now - t0 >= p.spin_ns
            now >= p.hot_until[] && break
            if now - last_yield >= _CPU_POOL_HOLD_YIELD_NS
                yield()
                last_yield = time_ns()
            end
        end
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

# Pins the worker's OS thread to `cpu` (when this worker is the first on its
# thread), runs the worker loop, then restores the thread's affinity. lane 0
# means the pool has no lanes.
function _cpu_pool_pinned_worker!(p::CPUPool, cpu::Int, lane::Int)
    claim = _pin_claim!(cpu)
    try
        return lane == 0 ? _cpu_pool_worker!(p) : _cpu_pool_lane_worker!(p, lane)
    finally
        claim === nothing || _pin_release!(claim)
    end
end

function _cpu_pool_lane_worker!(p::CPUPool, lane::Int)
    seen = p.ticket[] >> 32
    while _cpu_pool_wait!(p, seen)
        seen = p.ticket[] >> 32
        Base.invokelatest(_cpu_pool_lane_help!, p, lane)
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

# Lane claim loop, used by a lane worker (own = its lane) and by the caller
# (own = 1). Scans from `own` and takes one chunk from the first lane that has
# one, so a worker leaves its own lane for another only when its own is empty.
# Read the job before the claim, as _cpu_pool_help! does: a successful claim
# proves the job read belongs to the ticket. Returns when no lane has a chunk
# left to claim.
function _cpu_pool_lane_help!(p::CPUPool, own::Int)
    lanes = p.lanes::_CPULanes
    nl = length(lanes.tickets)
    while true
        pending = false
        for k in 0:(nl - 1)
            li = mod1(own + k, nl)
            t = lanes.tickets[li][]
            idx = Int(t & _CPU_POOL_IDX_MASK)
            idx < lanes.counts[li] || continue
            pending = true
            f = p.job
            atomic_cas!(lanes.tickets[li], t, t + 1) == t || continue
            _cpu_pool_run!(p, f, lanes.bases[li] + idx + 1)
            break
        end
        pending || return nothing
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
    lanes = p.lanes
    gen = p.ticket[] >> 32
    p.ticket[] = (gen << 32) | _CPU_POOL_IDX_MASK
    if lanes !== nothing
        for t in lanes.tickets
            t[] = (gen << 32) | _CPU_POOL_IDX_MASK
        end
    end
    p.completed[] = 0
    p.failed[] = 0
    p.nchunks = n
    p.job = f
    lanes === nothing || _cpu_lanes_layout!(lanes, n)
    lock(p.cond)
    try
        p.ticket[] = (gen + 1) << 32
        if lanes !== nothing
            for t in lanes.tickets
                t[] = (gen + 1) << 32
            end
        end
        notify(p.cond)
    finally
        unlock(p.cond)
    end
    return nothing
end

# Sets each lane's chunk range for a job of n chunks. Called while every lane
# ticket is closed, so no claim can read a half-written range.
function _cpu_lanes_layout!(lanes::_CPULanes, n::Int)
    for (li, r) in enumerate(proportional_ranges(n, lanes.weights))
        lanes.bases[li] = first(r) - 1
        lanes.counts[li] = length(r)
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
    pool_hold!(ns) -> nothing

Keep the pool's workers spinning for at least `ns` nanoseconds from now
instead of parking, for a caller that will submit more work after a short gap.
A later call extends the hold and never shortens it. Starts the pool if it is
not running.
"""
function pool_hold!(ns::Integer)
    ns > 0 || return nothing
    p = _cpu_pool()
    p.participants == 1 && return nothing
    until = time_ns() + UInt64(ns)
    while true
        cur = p.hot_until[]
        cur >= until && return nothing
        atomic_cas!(p.hot_until, cur, until) == cur && return nothing
    end
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
    n < typemax(UInt32) ||
        throw(ArgumentError("pool_run!: nchunks $n exceeds the chunk-count limit"))
    p = _cpu_pool()
    if n == 1 || p.participants == 1 || atomic_cas!(p.busy, 0, 1) != 0
        for c in 1:n
            f(c)
        end
        return nothing
    end
    try
        _cpu_pool_publish!(p, f, n)
        if p.lanes === nothing
            _cpu_pool_claim!(p, f)
        else
            _cpu_pool_lane_help!(p, 1)
        end
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
