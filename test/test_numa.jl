# test_numa.jl — NUMA topology, thread pinning and node-local row ownership
# (src/Runtime/numa.jl and the CPU pool). Sysfs trees are fixtures written to a
# temporary directory, so these cases run on any box, single-socket included:
#   julia --project=test -t 8 test/test_numa.jl

using Gesso
using Test
using Mmap
using Random

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
    # Workers split across nodes in proportion to CPU count, not node 0 first.
    @test _RN.pin_plan(topo, 12) == vcat(0:5, 9:14)
    @test _RN.pin_assignments(topo, 12) == vcat([(1, c) for c in 0:5], [(2, c) for c in 9:14])
    @test length(_RN.pin_plan(topo, 20)) == 20
    @test isempty(_RN.pin_plan(topo, 0))
    # 17 workers over 9+9 CPUs: the extra worker goes to the lower node index.
    @test count(a -> a[1] == 1, _RN.pin_assignments(topo, 17)) == 9
    @test count(a -> a[1] == 2, _RN.pin_assignments(topo, 17)) == 8
    # Uneven nodes: 4 workers over 3+1 CPUs gives 3 and 1.
    lop = _RN.parse_numa_override("0-2;3")
    @test _RN.pin_plan(lop, 4) == [0, 1, 2, 3]
    @test count(a -> a[1] == 1, _RN.pin_assignments(lop, 4)) == 3
    # 2 workers over 3+1 CPUs: quotas are 1 and 0, the spare goes to the tie at node index 1.
    @test count(a -> a[1] == 1, _RN.pin_assignments(lop, 2)) == 2

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

@testset "NUMA: proportional row ranges" begin
    @test _RN.proportional_ranges(10, [1, 1]) == [1:5, 6:10]
    @test _RN.proportional_ranges(0, [1, 2]) == [1:0, 1:0]
    for (n, w) in ((37, [9, 9]), (100, [3, 1, 2]), (7, [1, 0, 1]), (5, [2, 9]))
        rs = _RN.proportional_ranges(n, w)
        @test vcat(collect.(rs)...) == collect(1:n)
        tot = sum(w)
        for (k, rr) in enumerate(rs)
            @test abs(length(rr) - n * w[k] / tot) < 1
        end
    end
    @test_throws ArgumentError _RN.proportional_ranges(4, [0, 0])
end

@testset "NUMA: lanes visit every chunk exactly once" begin
    host = _RN.current_thread_cpus()
    old_n = _RP._cpu_pool().participants
    n = min(4, _RP._cpu_pool_available_threads())
    if !Sys.islinux() || host === nothing || length(host) < 4 || n < 3
        @info "lane check skipped" linux=Sys.islinux() threads=n
    else
        # Two fake nodes made of host CPUs, so the lane pool has real CPUs to pin to.
        spec = string(join(host[1:2], ","), ";", join(host[3:4], ","))
        try
            withenv(_NUMA_ENV => spec, _RN.PIN_ENV => "1") do
                _RP._cpu_pool_configure!(n)
                @test _RP._cpu_pool().lanes !== nothing
                for nc in (1, 5, 64, 257)
                    hits = [Base.Threads.Atomic{Int}(0) for _ in 1:nc]
                    _RP.pool_run!(nc) do c
                        Base.Threads.atomic_add!(hits[c], 1)
                    end
                    @test all(h -> h[] == 1, hits)
                end
            end
        finally
            _RP._cpu_pool_configure!(old_n)
        end
    end
end

# Random quant bytes for one type, with the fp16 scale fields set to finite values.
function _numa_test_quant(rng, ty, ne0::Int, ne1::Int)::Vector{UInt8}
    Inf_ = Gesso.Inference
    if ty == Inf_.GGML_TYPE_F32
        return collect(reinterpret(UInt8, randn(rng, Float32, ne0 * ne1)))
    end
    bsz = ty == Inf_.GGML_TYPE_Q8_0 ? 32 : 256
    tsz = ty == Inf_.GGML_TYPE_Q8_0 ? 34 : (ty == Inf_.GGML_TYPE_Q4_K ? 144 : 210)
    nblk = (ne0 ÷ bsz) * ne1
    q = rand(rng, UInt8, nblk * tsz)
    # 1-based byte positions of the fp16 scale fields in each block.
    fields = ty == Inf_.GGML_TYPE_Q8_0 ? (1,) : (ty == Inf_.GGML_TYPE_Q4_K ? (1, 3) : (209,))
    for k in 0:(nblk - 1), p in fields
        at = k * tsz + p
        q[at:(at + 1)] .= reinterpret(UInt8, [Float16(0.0625 * (1 + k % 3))])
    end
    return q
end

@testset "NUMA: row-split GEMV is bit-identical to serial" begin
    Inf_ = Gesso.Inference
    rng = Random.Xoshiro(20261102)
    ne0, ne1 = 512, 96
    x = randn(rng, Float32, ne0)
    types = (Inf_.GGML_TYPE_Q8_0, Inf_.GGML_TYPE_Q4_K, Inf_.GGML_TYPE_Q6_K, Inf_.GGML_TYPE_F32)
    fixtures = [(ty, _numa_test_quant(rng, ty, ne0, ne1)) for ty in types]
    refs = Vector{Vector{Float32}}()
    for (ty, q) in fixtures
        y = zeros(Float32, ne1)
        Inf_._qgemv_bytes!(y, x, q, ty, ne0, ne1, 1.0f0, 0.0f0)
        push!(refs, y)
    end
    check_all = function ()
        for ((ty, q), ref) in zip(fixtures, refs)
            y = zeros(Float32, ne1)
            Inf_._qgemv_bytes_threaded!(y, x, q, ty, ne0, ne1, 1.0f0, 0.0f0)
            @test reinterpret(UInt32, y) == reinterpret(UInt32, ref)
        end
    end
    # Default pool.
    check_all()
    # Two fake nodes: the lane pool hands row ranges to each node's workers.
    host = _RN.current_thread_cpus()
    n = min(4, _RP._cpu_pool_available_threads())
    if !Sys.islinux() || host === nothing || length(host) < 4 || n < 3
        @info "split GEMV check skipped" linux=Sys.islinux() threads=n
    else
        old_n = _RP._cpu_pool().participants
        spec = string(join(host[1:2], ","), ";", join(host[3:4], ","))
        try
            withenv(_NUMA_ENV => spec, _RN.PIN_ENV => "1") do
                _RP._cpu_pool_configure!(n)
                @test _RP._cpu_pool().lanes !== nothing
                check_all()
            end
        finally
            _RP._cpu_pool_configure!(old_n)
        end
    end
end

@testset "NUMA: localized copy and file-range release" begin
    Inf_ = Gesso.Inference
    rng = Random.Xoshiro(20261103)
    rb, nrows, off = 34 * 8, 75, 1000
    src = rand(rng, UInt8, off + nrows * rb + 500)
    out = Inf_._numa_localize_rows(src, off, nrows, rb)
    @test length(out) == nrows * rb
    @test out == src[(off + 1):(off + nrows * rb)]
    mktempdir() do dir
        path = joinpath(dir, "blob.bin")
        payload = rand(rng, UInt8, 3 * 1024 * 1024)
        write(path, payload)
        open(path, "r") do io
            data = Mmap.mmap(io, Vector{UInt8}; grow=false, shared=false)
            @test Inf_._release_file_range!(path, data, 100_000, 1024 * 1024 + 3 * 4096) == Sys.islinux()
            # The release is advice only: the mapping still reads the file's bytes.
            @test data == payload
        end
    end
end
