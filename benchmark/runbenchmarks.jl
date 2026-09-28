# Harpe benchmark harness (Harpe_Stack.md §LXXIII; §XXXIII integrity rules).
#
# Conventions:
#
#   §XXXIII  No timing is published from a run that included compilation
#            without saying so. Suites warm up before measuring and report
#            samples, not single runs.
#
#   §XLIX    Harpe must explain performance. Results print structured facts —
#            name, samples, median time — and persist them.
#
#   North Star §24: results persist to benchmark/results/ so a regression
#   corpus accrues from the very first sprint. You cannot retroactively
#   baseline.
#
# Run with:
#   julia --project=benchmark benchmark/runbenchmarks.jl
#   (or: make bench / scripts/bench.jl)

using Harpe
using BenchmarkTools
using Dates

const BENCH_RESULT_SCHEMA = "bench-result-v1"

const RESULTS = []

function record(name; samples, time_ns)
    push!(RESULTS, (name=name, samples=samples, time_ns=time_ns))
    println(
        rpad(name, 34),
        lpad(string(round(time_ns / 1_000; digits=2)), 12),
        " µs   (",
        samples,
        " samples)",
    )
end

suite = Dict{String, Function}()

# Constructing backend tags must be free — planners probe constantly (§XX).
suite["backend_tag_construction"] = function ()
    Harpe.CPUBackend()
end

suite["capability_probe"] = function ()
    Harpe.supports(Harpe.CPUBackend(), :some_capability)
end

# Structured event emission (§XLII) must be cheap enough that code never
# avoids logging to save time.
suite["structured_log_event"] = function ()
    Harpe.hlog(devnull, Harpe.Log.LOG_INFO, :bench_event; op=:noop, backend=:cpu)
end

# Receipt emission (§XLII) is on the hot path of every audited action; it
# must stay cheap and must never throw.
suite["receipt_emit_inmemory"] = function ()
    sink = Harpe.InMemorySink(64)
    Harpe.emit!(sink, Harpe.new_receipt(task=:bench))
end

# §LXX: the explicit-failure path must be cheap enough to be used everywhere.
suite["lowering_not_implemented_throw"] = function ()
    try
        Harpe.rmsnorm!(Harpe.CPUBackend(), nothing)
        error("expected LoweringNotImplemented")
    catch e
        e isa Harpe.LoweringNotImplemented || rethrow()
    end
end

println("Harpe benchmark suite (foundation cost)")
println("=" ^ 62)
println(
    "date: ",
    Dates.format(now(UTC), dateformat"yyyy-mm-dd\ THH:MM:SS\Z"),
    "   schema: ",
    BENCH_RESULT_SCHEMA,
    "   julia: ",
    VERSION,
)

for name in sort(collect(keys(suite)))
    f = suite[name]
    b = @benchmark $f()
    record(name; samples=length(b.times), time_ns=median(b.times))
end

println("=" ^ 62)
println(length(RESULTS), " benchmarks recorded.")

# --- persistence (North Star §24: accrue a regression corpus) ---------------
results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)
tsv = joinpath(results_dir, Dates.format(now(UTC), dateformat"yyyy-mm-dd") * ".tsv")
header_needed = !isfile(tsv)
open(tsv, "a") do io
    header_needed && println(
        io,
        join(
            ["schema", "date_utc", "julia_version", "benchmark", "samples", "median_ns"],
            "\t",
        ),
    )
    for r in RESULTS
        println(
            io,
            join(
                [
                    BENCH_RESULT_SCHEMA,
                    Dates.format(now(UTC), dateformat"yyyy-mm-dd\ THH:MM:SS\Z"),
                    string(VERSION),
                    r.name,
                    r.samples,
                    r.time_ns,
                ],
                "\t",
            ),
        )
    end
end
println("results appended to ", relpath(tsv, dirname(@__DIR__)))
println("Law of this suite: foundation hot paths stay sub-microsecond,")
println("so later phases can use them without cost-benefit debates.")
