# scripts/format.jl — format the repo (or check with `--check`). Entry:
# `make format` / `make format-check`.
run_check = "--check" in ARGS
Pkg = Base.require(Main, :Pkg)
# JuliaFormatter is kept out of the project (dependency law) in its own shared
# environment, `@gesso-format`. The version is pinned, and CI runs this same
# script, because formatter releases change their output and an unpinned
# formatter makes the check drift without any code change.
const JF_VERSION = v"2.14.0"
Pkg.activate("gesso-format"; shared=true, io=devnull)
if get(Pkg.dependencies(), Base.UUID("98e50ef6-434e-11e9-1051-2b60c6c9e899"), nothing) |>
   (info -> info === nothing || info.version != JF_VERSION)
    Pkg.add(; name="JuliaFormatter", version=string(JF_VERSION), io=devnull)
end
JF = Base.require(Main, :JuliaFormatter)
pkgversion(JF) == JF_VERSION ||
    error("loaded JuliaFormatter $(pkgversion(JF)), need $JF_VERSION")
# Never walk libs/ (local Lava checkout, not the package). Format only the
# package surfaces agents are allowed to touch.
let
    root = dirname(@__DIR__)
    targets = ("src", "ext", "test", "benchmark", "scripts")
    ok = true
    for t in targets
        ok &= JF.format(joinpath(root, t); verbose=(!run_check))
    end
    if run_check
        ok || error("formatting check FAILED — run `make format` and commit")
        println("formatting OK")
    end
end
