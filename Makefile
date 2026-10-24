# Gesso dev commands (wrappers over scripts/)

.PHONY: test bench format format-check env-vars

test:
	julia scripts/test.jl

bench:
	julia scripts/bench.jl

format:
	julia scripts/format.jl

format-check:
	julia scripts/format.jl --check

env-vars:
	julia scripts/env-vars.jl
