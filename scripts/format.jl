# scripts/format.jl — format the repo (or check with `--check`). Entry:
# `make format` / `make format-check`.
run_check = "--check" in ARGS
Pkg = Base.require(Main, :Pkg)
# JuliaFormatter is added ad hoc (not a project dep) to respect the
# dependency law; cache it in the default depot.
try
    Base.require(Main, :JuliaFormatter)
catch
    Pkg.add("JuliaFormatter"; io=devnull)
    Base.require(Main, :JuliaFormatter)
end
JF = Base.require(Main, :JuliaFormatter)
ok = JF.format(dirname(@__DIR__); verbose=(!run_check))
if run_check
    ok || error("formatting check FAILED — run `make format` and commit")
    println("formatting OK")
end
