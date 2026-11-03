# NUMA topology for the CPU backend (stdlib and Linux sysfs only).
#
#   numa_topology() -> NUMATopology
#
# Nodes come from <sysfs>/node/node<N>/cpulist, filtered to the CPUs listed in
# <sysfs>/cpu/online. A node left with no online CPU (memory-only) is dropped.
# Without a usable node tree the box is one node holding every online CPU.
#
# GESSO_NUMA_NODES="0-8;9-17" replaces the sysfs view with fake nodes, one per
# ';'-separated cpulist. It lets the multi-node paths run on a single-socket
# box. A malformed value is rejected, never ignored. An empty value is unset.

const NUMA_ENV = "GESSO_NUMA_NODES"
const NUMA_SYSFS_ROOT = "/sys/devices/system"

struct NUMANode
    id::Int
    cpus::Vector{Int}
end

struct NUMATopology
    nodes::Vector{NUMANode}
    source::Symbol   # :sysfs, :env or :single
end

numa_node_count(t::NUMATopology) = length(t.nodes)

# Splits 1:n into len(weights) contiguous blocks whose sizes follow `weights`
# (each block gets its share, rounded down; the last block ends at n). Returns
# the cumulative bounds: block i is (bounds[i-1]+1):bounds[i], bounds[0] = 0.
# Pure integer arithmetic, so every index lands in exactly one block and the
# same n and weights always give the same split.
function proportional_bounds(n::Integer, weights::AbstractVector{<:Integer})::Vector{Int}
    n >= 0 || throw(ArgumentError("proportional_bounds: n must be nonnegative"))
    all(w -> w >= 0, weights) || throw(ArgumentError("proportional_bounds: weights must be nonnegative"))
    total = sum(weights; init=0)
    total > 0 || throw(ArgumentError("proportional_bounds: weights sum to zero"))
    out = Vector{Int}(undef, length(weights))
    cum = 0
    for (i, w) in enumerate(weights)
        cum += Int(w)
        out[i] = div(Int(n) * cum, total)
    end
    return out
end

"""
    proportional_ranges(n, weights) -> Vector{UnitRange{Int}}

Contiguous ranges that cover 1:n exactly once, sized in proportion to `weights`.
"""
function proportional_ranges(n::Integer, weights::AbstractVector{<:Integer})::Vector{UnitRange{Int}}
    b = proportional_bounds(n, weights)
    out = Vector{UnitRange{Int}}(undef, length(b))
    lo = 0
    for (i, hi) in enumerate(b)
        out[i] = (lo + 1):hi
        lo = hi
    end
    return out
end

# Whether node-local weight copies are made. GESSO_NUMA=1 forces them on and
# 0 forces them off; unset means on for a real multi-node sysfs topology only.
# Copies need more than one node, so a single node is always off.
const NUMA_WEIGHTS_ENV = "GESSO_NUMA"

function numa_weights_enabled(topo::NUMATopology; env::AbstractDict=ENV)::Bool
    numa_node_count(topo) > 1 || return false
    raw = strip(get(env, NUMA_WEIGHTS_ENV, ""))
    isempty(raw) && return topo.source === :sysfs
    raw == "1" && return true
    raw == "0" && return false
    throw(gesso_error(ERR_INVALID_PLAN, "$NUMA_WEIGHTS_ENV must be 0 or 1, got $(repr(raw))"))
end

# Parses a sysfs cpulist such as "0-15,32-47". Empty input is an empty list
# (memory-only nodes). Malformed entries and reversed ranges are rejected.
function parse_cpulist(s::AbstractString)::Vector{Int}
    t = strip(s)
    isempty(t) && return Int[]
    cpus = Int[]
    for raw in split(t, ',')
        m = match(r"^(\d{1,6})(?:-(\d{1,6}))?$", strip(raw))
        m === nothing && throw(
            gesso_error(ERR_INVALID_PLAN, "NUMA cpulist: malformed entry $(repr(raw)) in $(repr(t))"),
        )
        lo = parse(Int, m.captures[1])
        hi = m.captures[2] === nothing ? lo : parse(Int, m.captures[2])
        hi < lo && throw(
            gesso_error(ERR_INVALID_PLAN, "NUMA cpulist: reversed range $(repr(raw)) in $(repr(t))"),
        )
        append!(cpus, lo:hi)
    end
    return sort!(unique!(cpus))
end

function _check_disjoint(nodes::Vector{NUMANode})
    owner = Dict{Int, Int}()
    for n in nodes, c in n.cpus
        prev = get(owner, c, nothing)
        prev === nothing || throw(
            gesso_error(ERR_INVALID_PLAN, "NUMA: cpu $c listed in both node $prev and node $(n.id)"),
        )
        owner[c] = n.id
    end
    return nodes
end

function _read_optional(path::AbstractString)::Union{Nothing, String}
    isfile(path) || return nothing
    return read(path, String)
end

function _online_cpus(sysfs::AbstractString)::Union{Nothing, Set{Int}}
    raw = _read_optional(joinpath(sysfs, "cpu", "online"))
    raw === nothing && return nothing
    return Set(parse_cpulist(raw))
end

function _topology_from_sysfs(sysfs::AbstractString)::Union{Nothing, NUMATopology}
    nodedir = joinpath(sysfs, "node")
    isdir(nodedir) || return nothing
    online = _online_cpus(sysfs)
    found = Tuple{Int, String}[]
    for name in readdir(nodedir)
        m = match(r"^node(\d+)$", name)
        m === nothing && continue
        raw = _read_optional(joinpath(nodedir, name, "cpulist"))
        raw === nothing && continue
        push!(found, (parse(Int, m.captures[1]), raw))
    end
    sort!(found; by=first)
    nodes = NUMANode[]
    for (id, raw) in found
        cpus = parse_cpulist(raw)
        online === nothing || filter!(in(online), cpus)
        isempty(cpus) && continue
        push!(nodes, NUMANode(id, cpus))
    end
    isempty(nodes) && return nothing
    _check_disjoint(nodes)
    return NUMATopology(nodes, :sysfs)
end

function _single_node_topology(sysfs::AbstractString)::NUMATopology
    online = _online_cpus(sysfs)
    cpus = online === nothing ? collect(0:(max(Sys.CPU_THREADS, 1) - 1)) : sort!(collect(online))
    return NUMATopology([NUMANode(0, cpus)], :single)
end

# Parses a GESSO_NUMA_NODES value: one cpulist per ';'-separated fake node.
function parse_numa_override(spec::AbstractString)::NUMATopology
    t = strip(spec)
    isempty(t) && throw(gesso_error(ERR_INVALID_PLAN, "$NUMA_ENV is empty"))
    nodes = NUMANode[]
    for (k, seg) in enumerate(split(t, ';'))
        cpus = parse_cpulist(seg)
        isempty(cpus) && throw(
            gesso_error(ERR_INVALID_PLAN, "$NUMA_ENV: fake node $(k - 1) has no CPUs in $(repr(t))"),
        )
        push!(nodes, NUMANode(k - 1, cpus))
    end
    _check_disjoint(nodes)
    return NUMATopology(nodes, :env)
end

# The real machine's sysfs topology, read once. A load asks once per large
# tensor, and each sysfs read costs ~170 us.
const _HOST_TOPOLOGY = Ref{Union{Nothing, NUMATopology}}(nothing)

function _sysfs_topology(sysfs::AbstractString)::NUMATopology
    topo = _topology_from_sysfs(sysfs)
    return topo === nothing ? _single_node_topology(sysfs) : topo
end

# Topology for this process. The env override wins over sysfs; sysfs wins over
# the single-node fallback. `sysfs` and `env` are injectable for tests; only
# the default sysfs root is cached.
function numa_topology(; sysfs::AbstractString=NUMA_SYSFS_ROOT, env::AbstractDict=ENV)::NUMATopology
    spec = strip(get(env, NUMA_ENV, ""))
    isempty(spec) || return parse_numa_override(spec)
    sysfs == NUMA_SYSFS_ROOT || return _sysfs_topology(sysfs)
    topo = _HOST_TOPOLOGY[]
    topo === nothing || return topo
    return _HOST_TOPOLOGY[] = _sysfs_topology(sysfs)
end
