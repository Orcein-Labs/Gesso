# scripts/test.jl — run the test suite. Entry: `make test`.
#
# The repo is a Pkg workspace (root Project.toml + test/ + benchmark/ members).
# Pkg.test() must run with the workspace ROOT active: it then discovers the
# package's test project (test/Project.toml) automatically. Activating the
# bare test env and calling Pkg.test() fails on Julia 1.12 with
# "The Project.toml of the package being tested must have a name and a UUID
# entry" — the test env itself is not a testable package.
import Pkg
root = dirname(@__DIR__)
Pkg.activate(root; io=devnull)
Pkg.instantiate(; io=devnull)
# Pkg.test prints normally: `make test` must show the test summary, not
# swallow it into devnull (a silent green run is indistinguishable from
# nothing happening).
#
# -t 2: concurrency surfaces (receipt sink) must be exercised with more than
# one thread; CI runs tests directly and still covers them single-threaded.
Pkg.test(; julia_args=`-t 2`)
