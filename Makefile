# Gesso dev commands (wrappers over scripts/ — see AGENTS.md §8)

.PHONY: test bench format format-check freeze clean

test:
	julia scripts/test.jl

bench:
	julia scripts/bench.jl

format:
	julia scripts/format.jl

format-check:
	julia scripts/format.jl --check

freeze:
	julia scripts/freeze.jl

clean:
	rm -f Gesso_*_Freeze_*.zip
