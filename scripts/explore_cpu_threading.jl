# Exploration: fork-join cost, persistent worker pool, thread count, pinning and
# activation-quantization share for the CPU IQ4_XS GEMV path. Stdlib only.
#
#   flock /tmp/gesso-cpu-bench.lock nice -n 5 julia --project=. -t 16 scripts/explore_cpu_threading.jl
#
# Environment:
#   EXP_MODE     forkjoin (default): Threads.@threads / @spawn costs, the Gesso
#                GEMV via @threads, quantization share, memory ceiling.
#                pool: the same GEMVs through a persistent pool of workers built
#                from Threads.@spawn/@threads + atomics + spin-then-sleep.
#   EXP_PIN      1 pins each Julia thread to CPU (threadid-1) with
#                pthread_setaffinity_np before any timing.
#   EXP_SAMPLES  timed samples per figure (default 9; the median and spread are reported).
#
# Modes are separate processes on purpose: a spinning pool occupies its threads,
# so any Threads.@threads run in the same process would queue behind it.
#
# GB/s is weight bytes streamed per second (decimal GB). Weight copies rotate so
# each timed working set is at least 256 MiB, which keeps the matrix out of the
# last-level cache.

using Gesso
using Base.Threads

const I = Gesso.Inference
const MODE = get(ENV, "EXP_MODE", "forkjoin")
const PIN = get(ENV, "EXP_PIN", "0") == "1"
const SAMPLES = parse(Int, get(ENV, "EXP_SAMPLES", "9"))
const TARGET_S = 0.25
const WS_BYTES = 256 * 2^20
const SPIN_NS = 2_000_000
const FMT = I.GGML_TYPE_IQ4_XS
const SHAPES = (
    ("expert 2048x512", 2048, 512),
    ("expert 512x2048", 512, 2048),
    ("dense 5120x13824", 5120, 13824),
)

# ------------------------------------------------------------------ timing

function time_samples(f)
    f()
    t1 = @elapsed f()
    reps = clamp(ceil(Int, TARGET_S / max(t1, 1.0e-9)), 1, 10_000_000)
    out = Float64[]
    for _ in 1:SAMPLES
        t = @elapsed for _ in 1:reps
            f()
        end
        push!(out, t / reps)
    end
    return out
end

function summarize(v::Vector{Float64})
    s = sort(v)
    n = length(s)
    med = isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2
    m = sum(v) / n
    sd = n > 1 ? sqrt(sum((v .- m) .^ 2) / (n - 1)) : 0.0
    return (; med, mean=m, sd, min=s[1], max=s[end])
end

function bench_weights(ty::Int32, ne0::Int, ne1::Int)
    bsz = I._iq_block_bytes(ty)
    nb = div(ne0, I._q8k_block_values(ty))
    doff = I._q8k_d_offset(ty)
    w = rand(UInt8, ne1 * nb * bsz)
    for blk in 0:(ne1*nb-1)
        bits = reinterpret(UInt16, Float16(0.01f0 + 0.02f0 * rand(Float32)))
        w[blk*bsz+doff+1] = UInt8(bits & 0xff)
        w[blk*bsz+doff+2] = UInt8(bits >> 8)
    end
    return w
end

function rotator(Ws::Vector)
    k = Ref(0)
    n = length(Ws)
    return () -> begin
        k[] = mod(k[], n) + 1
        return Ws[k[]]
    end
end

# ------------------------------------------------------------------ pinning

function pin_self!(cpu::Integer)::Bool
    mask = zeros(UInt64, 16)
    mask[(cpu÷64)+1] = UInt64(1) << (cpu % 64)
    ret = ccall(
        :pthread_setaffinity_np,
        Cint,
        (Culong, Csize_t, Ptr{UInt64}),
        ccall(:pthread_self, Culong, ()),
        128,
        mask,
    )
    return ret == 0
end

function affinity_cpus()::Vector{Int}
    mask = zeros(UInt64, 16)
    ccall(
        :pthread_getaffinity_np,
        Cint,
        (Culong, Csize_t, Ptr{UInt64}),
        ccall(:pthread_self, Culong, ()),
        128,
        mask,
    )
    return [c for c in 0:1023 if ((mask[(c÷64)+1] >> (c % 64)) & 1) == 1]
end

# Pins every Julia thread to CPU threadid-1. Returns (threadid, allowed-cpu list).
function pin_julia_threads!()
    res = Vector{Tuple{Int, Vector{Int}}}(undef, nthreads())
    @threads :static for i in 1:nthreads()
        tid = threadid()
        pin_self!(tid - 1)
        res[i] = (tid, affinity_cpus())
    end
    return res
end

# ------------------------------------------------------------------ persistent pool

mutable struct ExpPool
    nw::Int
    dynamic::Bool
    nchunks::Int
    gen::Atomic{Int}
    done::Atomic{Int}
    next::Atomic{Int}
    stop::Atomic{Int}
    started::Atomic{Int}
    job::Any
    tids::Vector{Int}
    launcher::Any
end

# Non-inlined atomic load, so the spin loop cannot have the load hoisted.
@noinline load_gen(p::ExpPool)::Int = p.gen[]

# Worker loop: spin on the generation counter for SPIN_NS, then sleep-poll in
# 20 us steps. The OS thread is dedicated, so usleep does not yield to Julia.
function pool_worker!(p::ExpPool, id::Int)
    seen = 0
    while true
        t0 = time_ns()
        while load_gen(p) == seen
            if time_ns() - t0 > SPIN_NS
                ccall(:usleep, Cint, (Cuint,), 20)
            else
                GC.safepoint()
            end
        end
        seen = load_gen(p)
        p.stop[] != 0 && break
        run_job!(p, id)
        atomic_add!(p.done, 1)
    end
    return nothing
end

# invokelatest: workers were started before each job closure existed, and a
# long-running worker frame keeps its world age, so a direct call cannot see
# the closure's method ("method too new"). The closure is compiled on the main
# thread first (pool_run!), so the worker only dispatches.
function run_job!(p::ExpPool, id::Int)
    if p.dynamic
        while true
            c = atomic_add!(p.next, 1) + 1
            c > p.nchunks && break
            Base.invokelatest(p.job, c)
        end
    else
        Base.invokelatest(p.job, id)
    end
    return nothing
end

function start_pool!(nw::Int; dynamic::Bool=false, pin::Bool=false)
    nch = dynamic ? nw * 4 : nw
    p = ExpPool(
        nw,
        dynamic,
        nch,
        Atomic{Int}(0),
        Atomic{Int}(0),
        Atomic{Int}(0),
        Atomic{Int}(0),
        Atomic{Int}(0),
        nothing,
        zeros(Int, nw),
        nothing,
    )
    # Compile the worker-side code on the main thread before any worker runs it.
    precompile(pool_worker!, (ExpPool, Int))
    precompile(run_job!, (ExpPool, Int))
    precompile(load_gen, (ExpPool,))
    if pin
        pin_self!(threadid() - 1)
    end
    p.tids[1] = threadid()
    p.launcher = @spawn begin
        @threads :static for id in 2:nw
            tid = threadid()
            pin && pin_self!(tid - 1)
            p.tids[id] = tid
            atomic_add!(p.started, 1)
            pool_worker!(p, id)
        end
    end
    flush(stdout)
    t0 = time_ns()
    while p.started[] < nw - 1
        if time_ns() - t0 > 10_000_000_000
            error(
                "pool: workers did not start within 10 s (started=$(p.started[]) of $(nw - 1)); threads=$(p.tids)",
            )
        end
        GC.safepoint()
    end
    return p
end

function pool_run!(p::ExpPool, f)
    # Compile the job on this thread before publishing it: a first call from a
    # worker thread crashed LLVM during codegen in testing.
    precompile(f, (Int,))
    p.job = f
    atomic_xchg!(p.done, 0)
    atomic_xchg!(p.next, 0)
    atomic_add!(p.gen, 1)
    run_job!(p, 1)
    while p.done[] < p.nw - 1
        GC.safepoint()
    end
    return nothing
end

function stop_pool!(p::ExpPool)
    atomic_xchg!(p.stop, 1)
    atomic_add!(p.gen, 1)
    wait(p.launcher)
    return nothing
end

# GEMV job: static chunk c = rows of an even split over nw participants (same
# split as Gesso's _qgemv_q8k_threaded!); dynamic chunk c = 16-row slice.
function gemv_job(p::ExpPool, y, xq, W, ty, ne0, ne1)
    if p.dynamic
        p.nchunks = cld(ne1, 16)
        return c -> begin
            j0 = (c - 1) * 16 + 1
            j1 = min(c * 16, ne1)
            j0 <= j1 &&
                I._qgemv_q8k_rows!(y, xq, W, ty, ne0, j0, j1, 1.0f0, 0.0f0, true)
            nothing
        end
    end
    nw = p.nw
    return c -> begin
        j0 = div((c - 1) * ne1, nw) + 1
        j1 = div(c * ne1, nw)
        j0 <= j1 && I._qgemv_q8k_rows!(y, xq, W, ty, ne0, j0, j1, 1.0f0, 0.0f0, true)
        nothing
    end
end

# ------------------------------------------------------------------ experiments

function fmt_row(io, label, s, extra="")
    println(
        rpad(label, 46),
        " median ",
        lpad(round(s.med; digits=3), 9),
        "  mean ",
        lpad(round(s.mean; digits=3), 9),
        "  sd ",
        lpad(round(s.sd; digits=3), 7),
        "  min ",
        lpad(round(s.min; digits=3), 8),
        "  max ",
        lpad(round(s.max; digits=3), 8),
        extra,
    )
end

function rotated_weights(ne0::Int, ne1::Int)
    nbytes = ne1 * I._qrow_bytes(FMT, ne0)
    return [bench_weights(FMT, ne0, ne1) for _ in 1:max(1, cld(WS_BYTES, nbytes))]
end

function fj_gemv_row(name::String, ne0::Int, ne1::Int)
    nbytes = ne1 * I._qrow_bytes(FMT, ne0)
    nextw = rotator(rotated_weights(ne0, ne1))
    x = rand(Float32, ne0) .- 0.5f0
    xq = I._q8k_quantize_act(FMT, x)
    y = zeros(Float32, ne1)
    t = time_samples(() -> I._qgemv_q8k_threaded!(y, xq, nextw(), FMT, ne0, ne1))
    s = summarize(nbytes ./ t ./ 1.0e9)
    fmt_row(
        stdout,
        "IQ4_XS $name @threads GB/s",
        s,
        "  call_us_med " * string(round(summarize(t).med * 1.0e6; digits=1)),
    )
    return nothing
end

function fj_quant_row(name::String, ne0::Int, ne1::Int)
    nextw = rotator(rotated_weights(ne0, ne1))
    x = rand(Float32, ne0) .- 0.5f0
    xq = I._q8k_quantize_act(FMT, x)
    y = zeros(Float32, ne1)
    tq = summarize(time_samples(() -> I._q8k_quantize_act(FMT, x)))
    tm =
        summarize(time_samples(() -> I._qgemv_q8k_threaded!(y, xq, nextw(), FMT, ne0, ne1)))
    tf = summarize(
        time_samples(() -> I._qgemv_q8k_f32!(y, x, nextw(), FMT, ne0, ne1; threaded=true)),
    )
    println(
        "quant share  ",
        rpad(name, 18),
        " quantize_us ",
        round(tq.med * 1.0e6; digits=2),
        "  gemv_us ",
        round(tm.med * 1.0e6; digits=2),
        "  full_call_us ",
        round(tf.med * 1.0e6; digits=2),
        "  share_of_full ",
        round(100 * tq.med / tf.med; digits=2),
        " %",
    )
    return nothing
end

# Three matrices reading one x: quantize per matrix versus once.
function fj_shared_x(ne0::Int, ne1::Int)
    nextw = rotator(rotated_weights(ne0, ne1))
    x = rand(Float32, ne0) .- 0.5f0
    y = zeros(Float32, ne1)
    xq = I._q8k_quantize_act(FMT, x)
    t_each = summarize(
        time_samples(
            () -> begin
                I._qgemv_q8k_f32!(y, x, nextw(), FMT, ne0, ne1; threaded=true)
                I._qgemv_q8k_f32!(y, x, nextw(), FMT, ne0, ne1; threaded=true)
                I._qgemv_q8k_f32!(y, x, nextw(), FMT, ne0, ne1; threaded=true)
            end,
        ),
    )
    t_once = summarize(
        time_samples(() -> begin
            q = I._q8k_quantize_act(FMT, x)
            I._qgemv_q8k_threaded!(y, q, nextw(), FMT, ne0, ne1)
            I._qgemv_q8k_threaded!(y, q, nextw(), FMT, ne0, ne1)
            I._qgemv_q8k_threaded!(y, q, nextw(), FMT, ne0, ne1)
        end),
    )
    println(
        "3 matrices, one x, dense ",
        ne1,
        "x",
        ne0,
        ": quantize each time ",
        round(t_each.med * 1.0e3; digits=3),
        " ms; quantize once ",
        round(t_once.med * 1.0e3; digits=3),
        " ms; gain ",
        round(100 * (t_each.med - t_once.med) / t_each.med; digits=2),
        " %",
    )
    return nothing
end

function pool_gemv_row(p::ExpPool, name::String, ne0::Int, ne1::Int)
    nbytes = ne1 * I._qrow_bytes(FMT, ne0)
    Ws = rotated_weights(ne0, ne1)
    nextw = rotator(Ws)
    x = rand(Float32, ne0) .- 0.5f0
    xq = I._q8k_quantize_act(FMT, x)
    y = zeros(Float32, ne1)
    yref = zeros(Float32, ne1)
    I._qgemv_q8k!(yref, xq, Ws[1], FMT, ne0, ne1)
    pool_run!(p, gemv_job(p, y, xq, Ws[1], FMT, ne0, ne1))
    ok = y == yref
    t = time_samples(() -> pool_run!(p, gemv_job(p, y, xq, nextw(), FMT, ne0, ne1)))
    s = summarize(nbytes ./ t ./ 1.0e9)
    fmt_row(
        stdout,
        "IQ4_XS $name pool dyn=$(p.dynamic) GB/s",
        s,
        "  call_us_med " *
        string(round(summarize(t).med * 1.0e6; digits=1)) *
        "  bitexact " *
        string(ok),
    )
    return nothing
end

function run_forkjoin()
    nt = nthreads()
    println("mode=forkjoin threads=", nt, " pinned=", PIN, " samples=", SAMPLES)
    if PIN
        res = pin_julia_threads!()
        bad = [r for r in res if length(r[2]) != 1 || r[2][1] != r[1] - 1]
        println(
            "pinning: ",
            isempty(bad) ? "all $(nt) threads pinned to threadid-1" : "MISMATCH $(bad)",
        )
    end

    # Fork-join micro-costs (per call, microseconds).
    sink = zeros(Int, nt)
    t = time_samples(() -> begin
        @threads for c in 1:nt
            sink[c] += 1
        end
    end)
    println(
        "fork-join @threads (empty body, $(nt) iters) us: ",
        round.(summarize(t).med .* 1.0e6; digits=2),
        " sd ",
        round(summarize(t).sd * 1.0e6; digits=2),
    )
    t = time_samples(() -> begin
        ts = [@spawn(sink[c] += 1) for c in 1:nt]
        foreach(wait, ts)
    end)
    println(
        "fork-join @spawn+wait (empty body, $(nt) tasks) us: ",
        round(summarize(t).med * 1.0e6; digits=2),
        " sd ",
        round(summarize(t).sd * 1.0e6; digits=2),
    )

    # Gesso GEMV through @threads (the production path).
    for (name, ne0, ne1) in SHAPES
        fj_gemv_row(name, ne0, ne1)
    end

    # Quantization share and reuse across matrices that read the same x.
    println()
    for (name, ne0, ne1) in SHAPES
        fj_quant_row(name, ne0, ne1)
    end

    fj_shared_x(5120, 13824)

    # Host memory-read ceiling, 2 GiB, summed across nt tasks.
    A = Vector{UInt64}(undef, 2^28)
    fill!(A, UInt64(0x0123456789abcdef))
    out = zeros(UInt64, nt)
    n = length(A)
    f = () -> begin
        @threads for c in 1:nt
            j0 = div((c - 1) * n, nt) + 1
            j1 = div(c * n, nt)
            out[c] = sum(view(A, j0:j1))
        end
        return sum(out)
    end
    t = summarize(time_samples(f))
    println(
        "host read ceiling ",
        nt,
        " tasks GB/s median ",
        round((8 * n) / t.med / 1.0e9; digits=2),
        " min ",
        round((8 * n) / t.max / 1.0e9; digits=2),
        " max ",
        round((8 * n) / t.min / 1.0e9; digits=2),
    )
    return nothing
end

function pool_config(nw::Int, dyn::Bool)
    p = start_pool!(nw; dynamic=dyn, pin=PIN)
    println(
        "pool dynamic=",
        dyn,
        " thread ids of participants: ",
        p.tids,
        "  (main=",
        threadid(),
        ")",
    )
    PIN && println("  main cpus: ", affinity_cpus())
    println("  workers all distinct: ", length(unique(p.tids)) == nw)
    t = summarize(time_samples(() -> pool_run!(p, c -> nothing)))
    println(
        "  pool empty fork-join us median ",
        round(t.med * 1.0e6; digits=2),
        " sd ",
        round(t.sd * 1.0e6; digits=2),
    )
    for (name, ne0, ne1) in SHAPES
        pool_gemv_row(p, name, ne0, ne1)
    end
    stop_pool!(p)
    return nothing
end

function run_pool()
    nw = nthreads()
    println("mode=pool workers=", nw, " pinned=", PIN, " samples=", SAMPLES)
    flush(stdout)
    pool_config(nw, false)
    pool_config(nw, true)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    if MODE == "pool"
        run_pool()
    elseif MODE == "forkjoin"
        run_forkjoin()
    else
        error("EXP_MODE must be forkjoin or pool, got $MODE")
    end
end
