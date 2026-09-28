# scripts/bench.jl — run the benchmark suite (results persist to
# benchmark/results/). Entry: `make bench`.
import Pkg
Pkg.activate("benchmark"; io=devnull)
Pkg.instantiate(; io=devnull)
include(joinpath(@__DIR__, "..", "benchmark", "runbenchmarks.jl"))
