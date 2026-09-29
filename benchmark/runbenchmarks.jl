# Gesso benchmark harness (Gesso_Stack.md §LXXIII; §XXXIII integrity rules).
#
# Conventions:
#
#   §XXXIII  No timing is published from a run that included compilation
#            without saying so. Suites warm up before measuring and report
#            samples, not single runs. This harness compiles each entry
#            explicitly before the BenchmarkTools run (which additionally
#            tunes), and never gates on absolute numbers — CI only checks
#            the harness runs and persists.
#
#   §XLIX    Gesso must explain performance. Results print structured facts —
#            name, samples, median, allocations — and persist them with
#            environment metadata (host, arch, threads, commit, dirty state).
#
#   North Star §24: results persist to benchmark/results/ so a regression
#   corpus accrues from the very first sprint. You cannot retroactively
#   baseline. Result files are append-only and may mix row schemas — group
#   by the `schema` column; never silently reinterpret old rows (§LXIX).
#
#   §LXXII   No performance claims beyond what a run in this repo produced.
#            The harness measures; it does not editorialize. The only
#            standing claims are recorded per-benchmark as `bench_note`s.
#
# This is infrastructure, not a performance claim: today it establishes that
# the measurement machinery works and accrues a corpus for future phases.
#
# Run with:
#   make bench   (or: julia --project=benchmark benchmark/runbenchmarks.jl)

using Gesso
using BenchmarkTools
using Dates
using Sockets

# The schema tag is IMPORTED, not hardcoded: the persisted format and the
# schema constant in src/versions.jl cannot drift apart silently (§LXIX).
const BENCH_RESULT_SCHEMA = string(Gesso.BENCH_RESULT_SCHEMA_VERSION)

const RESULTS = []

function record(name; samples, median_ns, mean_ns, min_ns, allocs, bytes, note)
    push!(
        RESULTS,
        (
            name=name,
            samples=samples,
            median_ns=median_ns,
            mean_ns=mean_ns,
            min_ns=min_ns,
            allocs=allocs,
            bytes=bytes,
            note=note,
        ),
    )
    println(
        rpad(name, 34),
        lpad(string(round(median_ns / 1_000; digits=2)), 10),
        " µs",
        lpad(string(allocs), 6),
        " allocs   (",
        samples,
        " samples)",
        note == "" ? "" : "   # " * note,
    )
end

function git_meta()
    try
        head = strip(read(`git log -1 --format=%h`, String))
        dirty = !isempty(strip(read(`git status --porcelain`, String)))
        return head, dirty ? "true" : "false"
    catch
        return "unknown", "unknown"   # e.g. bench from a tarball; say so
    end
end

suite = Dict{String, Tuple{Function, String}}()   # name => (f, bench_note)

# Constructing backend tags must be free — planners probe constantly (§XX).
suite["backend_tag_construction"] =
    (() -> Gesso.CPUBackend(), "§XX: tag construction must stay free")

suite["capability_probe"] = (
    () -> Gesso.supports(Gesso.CPUBackend(), :some_capability),
    "§XX: capability probing is always safe, never throws",
)

# Structured event emission (§XLII) must be cheap enough that code never
# avoids logging to save time.
suite["structured_log_event"] = (
    () -> Gesso.glog(devnull, Gesso.Log.LOG_INFO, :bench_event; op=:noop, backend=:cpu),
    "§XLII: logging must never be the reason to skip logging",
)

# Receipt emission (§XLII) is on the hot path of every audited action; it
# must stay cheap and must never throw.
suite["receipt_emit_inmemory"] = (
    () -> begin
        sink = Gesso.InMemorySink(64)
        Gesso.emit!(sink, Gesso.new_receipt(task=:bench))
    end,
    "§XLII: audited actions pay for receipts; keep it negligible",
)

# §LXX: the explicit-failure path must be cheap enough to be used everywhere.
suite["lowering_not_implemented_throw"] = (
    () -> begin
        try
            Gesso.rmsnorm!(Gesso.CPUBackend(), nothing)
            error("expected LoweringNotImplemented")
        catch e
            e isa Gesso.LoweringNotImplemented || rethrow()
        end
    end,
    "§LXX: explicit failure must be affordable",
)

# --- environment metadata (recorded once; persisted per row) ----------------
host = try
    gethostname()
catch
    "unknown"
end
commit, dirty = git_meta()

println("Gesso benchmark suite (foundation cost)")
println("=" ^ 72)
println(
    "date: ",
    Dates.format(now(UTC), dateformat"yyyy-mm-dd\ THH:MM:SS\Z"),
    "   schema: ",
    BENCH_RESULT_SCHEMA,
    "   julia: ",
    VERSION,
)
println(
    "host: ",
    host,
    "   arch: ",
    Sys.ARCH,
    "   threads: ",
    Threads.nthreads(),
    "   commit: ",
    commit,
    "   dirty: ",
    dirty,
)

for name in sort(collect(keys(suite)))
    f, note = suite[name]
    # §XXXIII: explicit compile before measurement (BenchmarkTools then
    # tunes and samples); allocation numbers are per single evaluation.
    f()
    b = @benchmark $f()
    record(
        name;
        samples=length(b.times),
        median_ns=median(b.times),
        mean_ns=mean(b.times),
        min_ns=minimum(b.times),
        allocs=b.allocs,
        bytes=b.memory,
        note=note,
    )
end

println("=" ^ 72)
println(length(RESULTS), " benchmarks recorded.")

# --- persistence (North Star §24: accrue a regression corpus) ---------------
results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)
tsv = joinpath(results_dir, Dates.format(now(UTC), dateformat"yyyy-mm-dd") * ".tsv")
header_needed = !isfile(tsv)
stamp = Dates.format(now(UTC), dateformat"yyyy-mm-dd\ THH:MM:SS\Z")
open(tsv, "a") do io
    if header_needed
        println(
            io,
            join(
                [
                    "schema",
                    "date_utc",
                    "julia_version",
                    "host",
                    "arch",
                    "nthreads",
                    "commit",
                    "dirty",
                    "benchmark",
                    "samples",
                    "median_ns",
                    "mean_ns",
                    "min_ns",
                    "allocs",
                    "bytes",
                    "bench_note",
                ],
                "\t",
            ),
        )
    end
    for r in RESULTS
        println(
            io,
            join(
                [
                    BENCH_RESULT_SCHEMA,
                    stamp,
                    string(VERSION),
                    string(host),
                    string(Sys.ARCH),
                    Threads.nthreads(),
                    commit,
                    dirty,
                    r.name,
                    r.samples,
                    r.median_ns,
                    r.mean_ns,
                    r.min_ns,
                    r.allocs,
                    r.bytes,
                    r.note,
                ],
                "\t",
            ),
        )
    end
end
println(
    "results appended to ",
    relpath(tsv, dirname(@__DIR__)),
    " (schema ",
    BENCH_RESULT_SCHEMA,
    ")",
)
