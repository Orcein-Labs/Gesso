# Phase 0 benchmark harness (Harpe_Stack.md §LXXIII).
#
# Phase 0 has exactly one thing worth timing: that the foundation itself costs
# nothing. These conventions keep honesty cheap later:
#
#   §XXXIII  Benchmark integrity: no timing is ever published from a run that
#            included compilation without saying so. Suites warm up before
#            measuring and report samples, not single runs.
#
#   §XLIX    Harpe must explain performance. Results print structured facts —
#            name, samples, median time — the vocabulary Phase 6 telemetry
#            will build on.
#
# Run with:
#   julia --project=benchmark benchmark/runbenchmarks.jl

using Harpe
using BenchmarkTools

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

# §LXX: the explicit-failure path must be cheap enough to be used everywhere.
suite["lowering_not_implemented_throw"] = function ()
    try
        Harpe.rmsnorm!(Harpe.CPUBackend(), nothing)
        error("expected LoweringNotImplemented")
    catch e
        e isa Harpe.LoweringNotImplemented || rethrow()
    end
end

println("Harpe Phase 0 benchmark suite (foundation cost)")
println("=" ^ 62)

for name in sort(collect(keys(suite)))
    f = suite[name]
    b = @benchmark $f()
    record(name; samples=length(b.times), time_ns=median(b.times))
end

println("=" ^ 62)
println(length(RESULTS), " benchmarks recorded.")
println("Law of this suite: foundation hot paths stay sub-microsecond,")
println("so later phases can use them without cost-benefit debates.")
