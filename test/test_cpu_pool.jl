# test_cpu_pool.jl — persistent CPU worker pool (src/Runtime/cpu_pool.jl).
#
# The pool is serial on one thread, so the multi-thread cases need Julia threads:
#   julia --project=test -t 8 test/test_cpu_pool.jl
# Sizing, sysfs and shutdown cases run at any thread count.

using Gesso
using Test
using Random: Xoshiro

const _RP = Gesso.Runtime
const _QK = Gesso.Inference

_pool_nthreads() = Threads.threadpoolsize(:default)

# Pool sizes to exercise: serial, a small pool, and the largest the process allows up to 8.
_pool_sizes() = unique(filter(k -> k <= _pool_nthreads(), [1, 2, 3, 8]))

# Runs f() in a task and fails instead of hanging if it does not finish. A
# failure inside f is rethrown as itself, not wrapped in TaskFailedException.
function _within(f; seconds::Real=60)
    t = Threads.@spawn f()
    status = timedwait(() -> istaskdone(t), seconds; pollint=0.005)
    status === :ok || error("pool case did not finish within $seconds s")
    try
        return fetch(t)
    catch e
        e isa TaskFailedException && throw(e.task.result)
        rethrow()
    end
end

struct _Hits
    hits::Vector{Int}
end
(f::_Hits)(c::Int) = (f.hits[c] += 1; nothing)

struct _Scale
    out::Vector{Int}
    k::Int
end
(f::_Scale)(c::Int) = (f.out[c]=c * f.k; nothing)

struct _Noop end
(::_Noop)(::Int) = nothing

struct _Inner
    out::Matrix{Int}
    c::Int
end
(f::_Inner)(j::Int) = (f.out[f.c, j]=10 * f.c + j; nothing)

struct _Outer
    out::Matrix{Int}
    inner::Int
end
(f::_Outer)(c::Int) = _RP.pool_run!(_Inner(f.out, c), f.inner)

struct _Thrower
    started::Vector{Int}
    bad::Int
end
function (f::_Thrower)(c::Int)
    c == f.bad && error("chunk $c failed")
    f.started[c] += 1
    return nothing
end

# GEMV chunk body, split exactly as _qgemv_q8k_threaded! splits rows.
struct _GemvChunks{X}
    y::Vector{Float32}
    xq::X
    W::Vector{UInt8}
    ty::Int32
    ne0::Int
    ne1::Int
    nc::Int
    simd::Bool
end
function (g::_GemvChunks)(c::Int)
    j0 = div((c - 1) * g.ne1, g.nc) + 1
    j1 = div(c * g.ne1, g.nc)
    j0 <= j1 &&
        _QK._qgemv_q8k_rows!(g.y, g.xq, g.W, g.ty, g.ne0, j0, j1, 1.0f0, 0.0f0, g.simd)
    return nothing
end

# Allocation is measured inside a function: a global-scope loop allocates for
# its own boxing and would measure the harness, not pool_run!.
function _pool_bytes_per_call(f, n::Int, reps::Int)
    a = @allocated for _ in 1:reps
        _RP.pool_run!(f, n)
    end
    return a ÷ reps
end

# Random valid IQ rows with finite f16 super-scales, as in test_quant_cpu_simd.jl.
function _random_iq_rows(rng, ty::Int32, nrows::Int, ne0::Int)
    bsz = _QK._iq_block_bytes(ty)
    nb = div(ne0, _QK._q8k_block_values(ty))
    doff = _QK._q8k_d_offset(ty)
    w = rand(rng, UInt8, nrows * nb * bsz)
    for blk in 0:(nrows*nb-1)
        bits = reinterpret(UInt16, Float16(0.005f0 + 0.05f0 * rand(rng, Float32)))
        w[blk*bsz+doff+1] = UInt8(bits & 0xff)
        w[blk*bsz+doff+2] = UInt8(bits >> 8)
    end
    return w
end

@testset "CPU pool" begin
    @testset "sizing" begin
        P = _RP._cpu_pool_participants
        @test P(; cores=18, threads=36, override=nothing) == 18
        @test P(; cores=36, threads=36, override=nothing) == 18
        @test P(; cores=4, threads=36, override=nothing) == 4
        @test P(; cores=18, threads=8, override=nothing) == 8
        @test P(; cores=0, threads=8, override=nothing) == 1
        @test P(; cores=18, threads=36, override=3) == 3
        @test P(; cores=18, threads=36, override=40) == 36
        @test P(; cores=18, threads=36, override=1) == 1
    end

    @testset "physical core count from sysfs" begin
        mktempdir() do root
            # Three cores over two packages; cpu4 is offline (no topology dir);
            # cpu6 has an unreadable core_id and is skipped.
            for (cpu, pkg, core) in ((0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1), (5, 1, 0))
                d = joinpath(root, "cpu$cpu", "topology")
                mkpath(d)
                write(joinpath(d, "physical_package_id"), "$pkg\n")
                write(joinpath(d, "core_id"), "$core\n")
            end
            mkpath(joinpath(root, "cpu4"))
            mkpath(joinpath(root, "cpu6", "topology"))
            write(joinpath(root, "cpu6", "topology", "physical_package_id"), "0\n")
            write(joinpath(root, "cpu6", "topology", "core_id"), "junk\n")
            mkpath(joinpath(root, "cpufreq"))
            @test _RP._cpu_pool_sysfs_cores(root) == 3
            @test _RP._cpu_pool_physical_cores(root) == 3
        end
        missing_root = joinpath(tempdir(), "gesso-no-such-sysfs-$(getpid())")
        @test _RP._cpu_pool_sysfs_cores(missing_root) === nothing
        @test _RP._cpu_pool_physical_cores(missing_root) == max(1, Sys.CPU_THREADS ÷ 2)
        @test _RP._cpu_pool_physical_cores() >= 1
    end

    @testset "bad nchunks" begin
        @test_throws ArgumentError _RP.pool_run!(_Noop(), -1)
        @test _RP.pool_run!(_Noop(), 0) === nothing
    end

    @testset "every chunk runs exactly once" begin
        for k in _pool_sizes()
            _RP._cpu_pool_configure!(k)
            for n in (1, 2, 3, 7, 16, 17, 64, 257, 1000)
                hits = zeros(Int, n)
                _RP.pool_run!(_Hits(hits), n)
                @test all(==(1), hits)
            end
            hits = zeros(Int, 5)
            _RP.pool_run!(_Hits(hits), 0)
            @test all(==(0), hits)
        end
    end

    @testset "nested calls run inline and finish" begin
        for k in _pool_sizes()
            _RP._cpu_pool_configure!(k)
            out = zeros(Int, 6, 5)
            _within(() -> _RP.pool_run!(_Outer(out, 5), 6))
            @test out == [10 * c + j for c in 1:6, j in 1:5]
        end
    end

    @testset "concurrent callers" begin
        if _pool_nthreads() >= 2
            _RP._cpu_pool_configure!(min(_pool_nthreads(), 8))
        end
        function caller(i)
            ok = true
            for rep in 1:20
                n = 37 + i
                out = zeros(Int, n)
                _RP.pool_run!(_Scale(out, i * rep), n)
                ok &= out == [c * i * rep for c in 1:n]
            end
            return ok
        end
        tasks = [Threads.@spawn caller(i) for i in 1:8]
        status = timedwait(() -> all(istaskdone, tasks), 120.0; pollint=0.005)
        @test status === :ok
        @test all(fetch, tasks)
    end

    @testset "exception propagates and the pool stays usable" begin
        for k in _pool_sizes()
            _RP._cpu_pool_configure!(k)
            started = zeros(Int, 200)
            f = _Thrower(started, 3)
            @test_throws ErrorException _within(() -> _RP.pool_run!(f, 200))
            # Nothing runs after pool_run! returned: the count is stable.
            snap = copy(started)
            sleep(0.02)
            @test started == snap
            @test started[3] == 0
            # Reuse after a failure, twice: the failure state is cleared.
            out = zeros(Int, 50)
            _RP.pool_run!(_Scale(out, 1), 50)
            @test out == collect(1:50)
            @test_throws ErrorException _RP.pool_run!(f, 200)
            out2 = zeros(Int, 50)
            _RP.pool_run!(_Scale(out2, 2), 50)
            @test out2 == [2c for c in 1:50]
        end
        # A single chunk runs inline and propagates the same way.
        @test_throws ErrorException _RP.pool_run!(_Thrower(zeros(Int, 1), 1), 1)
    end

    @testset "chunked GEMV is bit-exact with _qgemv_q8k_threaded!" begin
        rng = Xoshiro(20261008)
        simd = _QK.q8k_simd_enabled()
        for ty in (_QK.GGML_TYPE_IQ4_XS, _QK.GGML_TYPE_IQ2_XS, _QK.GGML_TYPE_IQ3_XXS)
            for (ne0, ne1) in ((512, 257), (2048, 512), (512, 2048))
                W = _random_iq_rows(rng, ty, ne1, ne0)
                x = randn(rng, Float32, ne0)
                xq = _QK._q8k_quantize_act(ty, x)
                yref = zeros(Float32, ne1)
                _QK._qgemv_q8k!(yref, xq, W, ty, ne0, ne1)
                for k in unique(clamp.((1, 2, 3, 8, _pool_nthreads()), 1, ne1))
                    _RP._cpu_pool_configure!(k)
                    yt = zeros(Float32, ne1)
                    _QK._qgemv_q8k_threaded!(yt, xq, W, ty, ne0, ne1; nchunks=k)
                    yp = zeros(Float32, ne1)
                    _RP.pool_run!(_GemvChunks(yp, xq, W, ty, ne0, ne1, k, simd), k)
                    @test yp == yt
                    @test yp == yref
                end
            end
        end
    end

    @testset "steady-state pool_run! does not allocate" begin
        _RP._cpu_pool_configure!(min(_pool_nthreads(), 8))
        f = _Noop()
        _RP.pool_run!(f, 16)
        _RP.pool_run!(f, 16)
        @test _pool_bytes_per_call(f, 16, 200) == 0
    end

    @testset "stop joins the workers" begin
        k = min(_pool_nthreads(), 4)
        p = _RP._cpu_pool_configure!(k)
        workers = copy(p.workers)
        _RP._cpu_pool_stop!(p)
        status = timedwait(() -> all(istaskdone, workers), 10.0; pollint=0.005)
        @test status === :ok
    end

    @testset "holds extend, run jobs, and do not delay stop" begin
        k = min(_pool_nthreads(), 4)
        p = _RP._cpu_pool_configure!(k; spin_us=0)
        _RP.pool_hold!(0)
        @test p.hot_until[] == 0
        _RP.pool_hold!(10_000_000_000)
        far = p.hot_until[]
        # A one-participant pool has no workers to keep hot, so the hold is a no-op.
        @test k >= 2 ? far > time_ns() : far == 0
        _RP.pool_hold!(1_000)
        @test p.hot_until[] == far
        # Held workers are spinning rather than parked; jobs still run once each.
        sleep(0.01)
        hits = zeros(Int, 64)
        _RP.pool_run!(c -> (hits[c] += 1; nothing), 64)
        @test all(==(1), hits)
        # A 10 s hold must not keep stop from joining the workers.
        workers = copy(p.workers)
        _RP._cpu_pool_stop!(p)
        status = timedwait(() -> all(istaskdone, workers), 5.0; pollint=0.005)
        @test status === :ok
    end

    # Restore the default pool for later test files.
    _RP._cpu_pool_configure!(_RP._cpu_pool_participants())
end
