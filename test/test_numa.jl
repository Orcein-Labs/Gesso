# test_numa.jl — NUMA topology, thread pinning and node-local row ownership
# (src/Runtime/numa.jl and the CPU pool). Sysfs trees are fixtures written to a
# temporary directory, so these cases run on any box, single-socket included:
#   julia --project=test -t 8 test/test_numa.jl

using Gesso
using Test

const _RN = Gesso.Runtime
const _RP = Gesso.Runtime
const _NUMA_ENV = "GESSO_NUMA_NODES"

# Builds a fake sysfs tree and runs f(root) on it. `nodes` maps a node id to
# its cpulist; `online` is the cpu/online text, or nothing to omit that file.
function _with_sysfs(f; nodes::Vector{Pair{Int, String}}=Pair{Int, String}[], online=nothing)
    mktempdir() do root
        for (id, cpulist) in nodes
            d = joinpath(root, "node", "node$id")
            mkpath(d)
            write(joinpath(d, "cpulist"), cpulist * "\n")
        end
        if online !== nothing
            mkpath(joinpath(root, "cpu"))
            write(joinpath(root, "cpu", "online"), online * "\n")
        end
        return f(root)
    end
end

_cpus_of(t) = [n.cpus for n in t.nodes]
_ids_of(t) = [n.id for n in t.nodes]

@testset "NUMA: cpulist parsing" begin
    P = _RN.parse_cpulist
    @test P("0-15,32-47") == vcat(0:15, 32:47)
    @test P("0") == [0]
    @test P("3,1-2\n") == [1, 2, 3]
    @test P(" 4-5 \n") == [4, 5]
    @test P("2,2-3") == [2, 3]
    @test P("") == Int[]
    @test P("\n") == Int[]
    for bad in ("0-", "-1", "a", "5-3", "1,,2", "1,", "0-1-2", "9999999", "1 2", "0x3")
        @test_throws Gesso.GessoException P(bad)
    end
end

@testset "NUMA: topology from a sysfs fixture" begin
    # One socket, all CPUs online.
    _with_sysfs(; nodes=[0 => "0-35"], online="0-35") do root
        t = _RN.numa_topology(; sysfs=root, env=Dict{String, String}())
        @test t.source === :sysfs
        @test _RN.numa_node_count(t) == 1
        @test _cpus_of(t) == [collect(0:35)]
    end

    # Two sockets, interleaved ranges as on real EPYC/Xeon layouts.
    _with_sysfs(; nodes=[0 => "0-15,32-47", 1 => "16-31,48-63"], online="0-63") do root
        t = _RN.numa_topology(; sysfs=root, env=Dict{String, String}())
        @test t.source === :sysfs
        @test _ids_of(t) == [0, 1]
        @test _cpus_of(t) == [vcat(0:15, 32:47), vcat(16:31, 48:63)]
    end

    # Offline CPUs are removed from their node's list.
    _with_sysfs(; nodes=[0 => "0-15,32-47", 1 => "16-31,48-63"], online="0-7,16-23,32-35") do root
        t = _RN.numa_topology(; sysfs=root, env=Dict{String, String}())
        @test _cpus_of(t) == [vcat(0:7, 32:35), collect(16:23)]
    end

    # Node ids may have gaps; order follows the id.
    _with_sysfs(; nodes=[2 => "16-31", 0 => "0-15"], online="0-31") do root
        t = _RN.numa_topology(; sysfs=root, env=Dict{String, String}())
        @test _ids_of(t) == [0, 2]
        @test _cpus_of(t) == [collect(0:15), collect(16:31)]
    end

    # A memory-only node (empty cpulist) is dropped.
    _with_sysfs(; nodes=[0 => "0-7", 1 => ""], online="0-7") do root
        t = _RN.numa_topology(; sysfs=root, env=Dict{String, String}())
        @test _ids_of(t) == [0]
    end

    # No cpu/online file: the cpulists are taken as they are.
    _with_sysfs(; nodes=[0 => "0-3", 1 => "4-7"]) do root
        t = _RN.numa_topology(; sysfs=root, env=Dict{String, String}())
        @test _cpus_of(t) == [collect(0:3), collect(4:7)]
    end

    # Two nodes listing the same CPU is a malformed tree.
    _with_sysfs(; nodes=[0 => "0-7", 1 => "4-11"], online="0-11") do root
        @test_throws Gesso.GessoException _RN.numa_topology(; sysfs=root, env=Dict{String, String}())
    end

    # No node directory: one node holding the online CPUs.
    _with_sysfs(; online="0-5") do root
        t = _RN.numa_topology(; sysfs=root, env=Dict{String, String}())
        @test t.source === :single
        @test _cpus_of(t) == [collect(0:5)]
    end

    # No sysfs at all: one node with every CPU the process can see.
    _with_sysfs() do root
        t = _RN.numa_topology(; sysfs=root, env=Dict{String, String}())
        @test t.source === :single
        @test _RN.numa_node_count(t) == 1
        @test _cpus_of(t) == [collect(0:(max(Sys.CPU_THREADS, 1) - 1))]
    end
    t = _RN.numa_topology(; sysfs=joinpath(tempdir(), "gesso-numa-no-such-root"), env=Dict{String, String}())
    @test t.source === :single
end

@testset "NUMA: GESSO_NUMA_NODES override" begin
    fixture2 = Pair{Int, String}[0 => "0-15,32-47", 1 => "16-31,48-63"]
    _with_sysfs(; nodes=fixture2, online="0-63") do root
        env = Dict(_NUMA_ENV => "0-8;9-17")
        t = _RN.numa_topology(; sysfs=root, env=env)
        @test t.source === :env
        @test _ids_of(t) == [0, 1]
        @test _cpus_of(t) == [collect(0:8), collect(9:17)]

        t1 = _RN.numa_topology(; sysfs=root, env=Dict(_NUMA_ENV => "0-7"))
        @test t1.source === :env
        @test _cpus_of(t1) == [collect(0:7)]

        # Unset or blank means the sysfs view.
        @test _RN.numa_topology(; sysfs=root, env=Dict{String, String}()).source === :sysfs
        @test _RN.numa_topology(; sysfs=root, env=Dict(_NUMA_ENV => "  ")).source === :sysfs
    end

    for bad in ("0-8;", ";9-17", "0-8;;9-17", "0-8;8-17", "0-8;x", "9-3;0-1",
                "0-8|9-17", "0-8 9-17", "0-8;9-17;-1")
        @test_throws Gesso.GessoException _RN.parse_numa_override(bad)
        @test_throws Gesso.GessoException _RN.numa_topology(; sysfs="/nonexistent", env=Dict(_NUMA_ENV => bad))
    end
end

@testset "NUMA: real host topology" begin
    t = _RN.numa_topology(; env=Dict{String, String}())
    @test _RN.numa_node_count(t) >= 1
    @test all(!isempty(n.cpus) for n in t.nodes)
    @test t.source in (:sysfs, :single)
end

@testset "NUMA: pin plan and gate" begin
    topo = _RN.parse_numa_override("0-8;9-17")
    # Workers fill node 0 before node 1.
    @test _RN.pin_plan(topo, 12) == vcat(0:8, 9:11)
    @test length(_RN.pin_plan(topo, 20)) == 20
    @test isempty(_RN.pin_plan(topo, 0))

    one = _RN.parse_numa_override("0-7")
    @test !_RN.pin_threads_enabled(one; env=Dict{String, String}())
    @test _RN.pin_threads_enabled(topo; env=Dict{String, String}())
    @test _RN.pin_threads_enabled(one; env=Dict(_RN.PIN_ENV => "1"))
    @test !_RN.pin_threads_enabled(topo; env=Dict(_RN.PIN_ENV => "0"))
    @test_throws Gesso.GessoException _RN.pin_threads_enabled(one; env=Dict(_RN.PIN_ENV => "yes"))
end

@testset "NUMA: thread affinity round trip" begin
    if !Sys.islinux()
        @test !_RN.set_current_thread_affinity!([0])
        @test _RN.current_thread_cpus() === nothing
    else
        saved = _RN.current_thread_cpus()
        @test saved !== nothing && !isempty(saved)
        try
            @test _RN.set_current_thread_affinity!([saved[1]])
            @test _RN.current_thread_cpus() == [saved[1]]
            # A CPU this kernel does not have is refused and leaves the mask alone.
            @test !_RN.set_current_thread_affinity!([1000])
            @test _RN.current_thread_cpus() == [saved[1]]
        finally
            _RN.set_current_thread_affinity!(saved)
        end
        @test _RN.current_thread_cpus() == saved
    end
end

@testset "NUMA: pool workers pin one CPU per thread" begin
    host = _RN.current_thread_cpus()
    old_n = _RP._cpu_pool().participants
    n = min(4, _RP._cpu_pool_available_threads())
    if !Sys.islinux() || host === nothing || n < 2
        @info "pool pinning check skipped" linux=Sys.islinux() threads=n
    else
        # One fake node made of host CPUs, so the planned CPUs exist here.
        spec = join(host[1:min(length(host), 4)], ",")
        try
            withenv(_NUMA_ENV => spec, _RN.PIN_ENV => "1") do
                _RP._cpu_pool_configure!(n)
                report = _RN.pool_pin_report()
                @test length(report) == n - 1
                @test all(r.status in (:pinned, :shared) for r in report)
                pinned = [r for r in report if r.status === :pinned]
                @test !isempty(pinned)
                @test all(r.cpu in host for r in pinned)

                # Chunks running on a pinned worker thread see exactly its CPU.
                seen = Dict{Int, Vector{Int}}()
                lk = ReentrantLock()
                _RP.pool_run!(64) do c
                    sleep(0.001)
                    tid = Threads.threadid()
                    aff = _RN.current_thread_cpus()
                    lock(() -> (seen[tid] = aff), lk)
                end
                @test any(haskey(seen, r.tid) for r in pinned)
                for r in pinned
                    haskey(seen, r.tid) && @test seen[r.tid] == [r.cpu]
                end
            end
        finally
            _RP._cpu_pool_configure!(old_n)
        end
        # The workers restored their threads, so pinning does not leak into the next pool.
        restored = Dict{Int, Vector{Int}}()
        lk = ReentrantLock()
        _RP.pool_run!(64) do c
            sleep(0.001)
            lock(() -> (restored[Threads.threadid()] = _RN.current_thread_cpus()), lk)
        end
        @test all(v == host for v in values(restored))
    end
end
