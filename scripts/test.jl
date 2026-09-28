# scripts/test.jl — run the test suite. Entry: `make test`.
import Pkg
Pkg.activate("test"; io=devnull)
Pkg.instantiate(; io=devnull)
Pkg.test(; io=devnull)
