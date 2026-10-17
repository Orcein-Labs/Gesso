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
# The test env loads Lava, which imports GLFW, and GLFW's __init__ throws
# without a display ("X11: The DISPLAY environment variable is missing").
# Pkg.test then fails on the precompile of a direct dependency. On a headless
# Linux box, re-run under xvfb-run when it is available, else say why.
if Sys.islinux() && !haskey(ENV, "DISPLAY") && !haskey(ENV, "WAYLAND_DISPLAY")
    xvfb = Sys.which("xvfb-run")
    if xvfb !== nothing && get(ENV, "GESSO_TEST_XVFB", "") != "1"
        cmd = `$xvfb -a $(Base.julia_cmd()) $(@__FILE__)`
        exit(success(run(addenv(ignorestatus(cmd), "GESSO_TEST_XVFB" => "1"))) ? 0 : 1)
    elseif xvfb === nothing
        @warn "No display and no xvfb-run: Lava (GLFW) will fail to precompile " *
              "and Pkg.test will stop. Install xvfb, or run under a display."
    end
end
Pkg.activate(root; io=devnull)
Pkg.instantiate(; io=devnull)
# Pkg.test prints normally: `make test` must show the test summary, not
# swallow it into devnull (a silent green run is indistinguishable from
# nothing happening).
#
# -t 2: concurrency surfaces (receipt sink) must be exercised with more than
# one thread; CI runs tests directly and still covers them single-threaded.
Pkg.test(; julia_args=`-t 2`)
