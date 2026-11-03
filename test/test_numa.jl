# test_numa.jl — NUMA topology, thread pinning and node-local row ownership
# (src/Runtime/numa.jl and the CPU pool). Sysfs trees are fixtures written to a
# temporary directory, so these cases run on any box, single-socket included:
#   julia --project=test -t 8 test/test_numa.jl

using Gesso
using Test

const _RN = Gesso.Runtime
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
