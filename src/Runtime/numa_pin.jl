# CPU affinity for pool workers (Linux, via libc sched_setaffinity).
#
# Pinning is per OS thread. A Julia task stays on the thread that started it,
# but two long-lived worker tasks can share one thread, so the first worker to
# start on a thread pins it and later ones on that thread are recorded as
# :shared and left alone. A pinned worker restores its thread's previous
# affinity when it exits, so pinning never outlives its pool.
#
# Pinning is off unless the topology has more than one node or
# GESSO_PIN_THREADS=1 is set. GESSO_PIN_THREADS=0 turns it off.

const PIN_ENV = "GESSO_PIN_THREADS"
const _AFFINITY_WORDS = 16   # 1024-bit cpu_set_t, the glibc default

const PinRecord = NamedTuple{(:tid, :cpu, :status), Tuple{Int, Int, Symbol}}

const _PIN_LOCK = ReentrantLock()
const _PIN_OWNERS = Dict{Int, Task}()   # threadid -> task that pinned it
const _PIN_LOG = PinRecord[]

function _affinity_mask(cpus::AbstractVector{<:Integer})::Union{Nothing, Vector{UInt64}}
    mask = zeros(UInt64, _AFFINITY_WORDS)
    for c in cpus
        (0 <= c < 64 * _AFFINITY_WORDS) || return nothing
        mask[fld(c, 64) + 1] |= UInt64(1) << rem(c, 64)
    end
    return mask
end

"""
    current_thread_cpus() -> Union{Nothing, Vector{Int}}

CPUs the calling OS thread may run on, or nothing when the call is unavailable.
"""
function current_thread_cpus()::Union{Nothing, Vector{Int}}
    Sys.islinux() || return nothing
    mask = zeros(UInt64, _AFFINITY_WORDS)
    rc = ccall(:sched_getaffinity, Cint, (Cint, Csize_t, Ptr{UInt64}), 0, sizeof(mask), mask)
    rc == 0 || return nothing
    cpus = Int[]
    for w in 1:_AFFINITY_WORDS, b in 0:63
        (mask[w] >> b) & 1 == 1 && push!(cpus, (w - 1) * 64 + b)
    end
    return cpus
end

"""
    set_current_thread_affinity!(cpus) -> Bool

Restricts the calling OS thread to `cpus`. Returns false if the call is
unavailable or the kernel rejects the mask (for example, CPUs not on this box).
"""
function set_current_thread_affinity!(cpus::AbstractVector{<:Integer})::Bool
    Sys.islinux() || return false
    isempty(cpus) && return false
    mask = _affinity_mask(cpus)
    mask === nothing && return false
    # pid 0 is the calling thread.
    return ccall(:sched_setaffinity, Cint, (Cint, Csize_t, Ptr{UInt64}), 0, sizeof(mask), mask) == 0
end

"""
    pin_assignments(topo, workers) -> Vector{Tuple{Int, Int}}

One `(node_index, cpu)` per worker. Workers are split across nodes in
proportion to each node's CPU count (largest remainder, ties to the lower node
index), so a dual-socket box gets workers on both sockets. A node's workers take
its CPUs in order and wrap when they outnumber them. `node_index` is the
position in `topo.nodes`.
"""
function pin_assignments(topo::NUMATopology, workers::Integer)::Vector{Tuple{Int, Int}}
    w = Int(workers)
    out = Tuple{Int, Int}[]
    w <= 0 && return out
    nodes = [i for i in eachindex(topo.nodes) if !isempty(topo.nodes[i].cpus)]
    isempty(nodes) && return out
    sizes = [length(topo.nodes[i].cpus) for i in nodes]
    total = sum(sizes)
    counts = [div(w * s, total) for s in sizes]
    left = w - sum(counts)
    order = sort(eachindex(nodes); by=k -> (-mod(w * sizes[k], total), k))
    for k in order[1:left]
        counts[k] += 1
    end
    for (k, i) in enumerate(nodes)
        cpus = topo.nodes[i].cpus
        for j in 1:counts[k]
            push!(out, (i, cpus[mod1(j, length(cpus))]))
        end
    end
    return out
end

"""
    pin_plan(topo, workers) -> Vector{Int}

The CPU of each `pin_assignments` entry, in the same order.
"""
pin_plan(topo::NUMATopology, workers::Integer)::Vector{Int} =
    [cpu for (_, cpu) in pin_assignments(topo, workers)]

"""
    pin_threads_enabled(topo; env) -> Bool

GESSO_PIN_THREADS=1 forces pinning on, 0 forces it off. Unset means on only for
a multi-node topology. Any other value is an error.
"""
function pin_threads_enabled(topo::NUMATopology; env::AbstractDict=ENV)::Bool
    raw = strip(get(env, PIN_ENV, ""))
    isempty(raw) && return numa_node_count(topo) > 1
    raw == "1" && return true
    raw == "0" && return false
    throw(gesso_error(ERR_INVALID_PLAN, "$PIN_ENV must be 0 or 1, got $(repr(raw))"))
end

function _pin_reset!()
    lock(_PIN_LOCK) do
        empty!(_PIN_OWNERS)
        empty!(_PIN_LOG)
    end
    return nothing
end

function _pin_logged_count()::Int
    return lock(_PIN_LOCK) do
        length(_PIN_LOG)
    end
end

"""
    pool_pin_report() -> Vector{NamedTuple}

Per pool worker since the last pool start: `(tid, cpu, status)` with status
:pinned, :shared (its thread was already pinned) or :failed.
"""
pool_pin_report() = lock(() -> copy(_PIN_LOG), _PIN_LOCK)

# Pins the calling worker's OS thread to `cpu`. Returns a claim to release, or
# nothing when the thread was not pinned by this call.
function _pin_claim!(cpu::Int)
    tid = Base.Threads.threadid()
    me = current_task()
    return lock(_PIN_LOCK) do
        if haskey(_PIN_OWNERS, tid)
            push!(_PIN_LOG, (tid=tid, cpu=cpu, status=:shared))
            return nothing
        end
        prev = current_thread_cpus()
        if prev === nothing || !set_current_thread_affinity!([cpu])
            push!(_PIN_LOG, (tid=tid, cpu=cpu, status=:failed))
            return nothing
        end
        _PIN_OWNERS[tid] = me
        push!(_PIN_LOG, (tid=tid, cpu=cpu, status=:pinned))
        return (tid=tid, task=me, prev=prev)
    end
end

# Restores the thread's prior affinity. Runs on the same OS thread as the claim.
function _pin_release!(claim)
    lock(_PIN_LOCK) do
        get(_PIN_OWNERS, claim.tid, nothing) === claim.task && delete!(_PIN_OWNERS, claim.tid)
        set_current_thread_affinity!(claim.prev)
    end
    return nothing
end
