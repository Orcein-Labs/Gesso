# Fork-join cost of Threads.@threads at the thread count Gesso uses (R&D, 2026-10-08).
#
# Measures the cost of one parallel region with a near-empty body, i.e. the
# pure scheduling overhead, for the patterns Gesso's CPU path uses
# (`Threads.@threads for c in 1:nc`, nc = min(nthreads, rows / min_rows)).
#
#   julia -t 16 scripts/julia_runtime/threads_forkjoin.jl
#
# Prints one line per variant: median and p90 ns per region over many calls.
# Pair with the decode profile (GESSO_PROFILE=1) to convert to ms per token.

using Statistics

const NT = Threads.nthreads()
const CHUNKS = NT                 # Gesso sizes nc = min(nthreads, ...)
const REPS = 20_000

sink = Ref(0.0)

@noinline function region_threads!(out::Vector{Float64})
    Threads.@threads for c in 1:CHUNKS
        out[c] += 1.0
    end
    return nothing
end

@noinline function region_threads_static!(out::Vector{Float64})
    Threads.@threads :static for c in 1:CHUNKS
        out[c] += 1.0
    end
    return nothing
end

@noinline function region_spawn!(out::Vector{Float64})
    tasks = Vector{Task}(undef, CHUNKS)
    for c in 1:CHUNKS
        tasks[c] = Threads.@spawn (out[c] += 1.0)
    end
    foreach(wait, tasks)
    return nothing
end

@noinline function region_serial!(out::Vector{Float64})
    for c in 1:CHUNKS
        out[c] += 1.0
    end
    return nothing
end

function bench(f, out, reps)
    for _ in 1:200
        f(out)
    end
    ts = Vector{Float64}(undef, reps)
    for i in 1:reps
        t0 = time_ns()
        f(out)
        ts[i] = (time_ns() - t0) / 1e3
    end
    sort!(ts)
    return (median(ts), ts[cld(9 * reps, 10)], mean(ts))
end

out = zeros(Float64, CHUNKS)
println("nthreads=$NT chunks=$CHUNKS reps=$REPS")
for (name, f) in (
    ("serial_loop", region_serial!),
    ("threads_dynamic", region_threads!),
    ("threads_static", region_threads_static!),
    ("spawn_wait", region_spawn!),
)
    med, p90, mu = bench(f, out, REPS)
    println(
        rpad(name, 18),
        " median_us=",
        round(med; digits=2),
        " p90_us=",
        round(p90; digits=2),
        " mean_us=",
        round(mu; digits=2),
    )
end
sink[] = sum(out)
