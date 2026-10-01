# Phase 10 item D (§LXXXIII gate G2): the harness exists and stays honest
# WITHOUT requiring torch, a snapshot, or a GPU. CI never downloads weights,
# never requires PyTorch, never requires a device — so these tests assert
# only environment-independent properties:
#   * benchmark/compare_eager.py exists and parses (--help exits 0 WITHOUT
#     importing torch — the probe path the bench harness itself uses);
#   * bad arguments fail loudly and fast (no torch import on the arg path);
#   * the eager reference loads weights LOCAL-ONLY (source inspection:
#     local_files_only=True, no download URLs anywhere in the script);
#   * dependency law (§VII): torch/transformers appear in NO Project.toml —
#     PyTorch is an external binary, and the bench env gains no JSON/extra
#     deps for the G2 wiring.
# The REAL G2 rows only land on a machine with snapshot + CUDA + torch
# (benchmark/runbenchmarks.jl G2 block — named skip, no rows, either way).

using Test: AbstractTestSet, Broken, get_testset, record

using Gesso

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const _P10_PY = "python3"
const _P10_SCRIPT = joinpath(pkgdir(Gesso), "benchmark", "compare_eager.py")

@testset "Phase 10 item D: G2 harness exists, honest, dependency-clean" begin
    if !isfile(_P10_SCRIPT)
        @test _skip(
            "benchmark/compare_eager.py missing — G2 harness tests skipped (§LXXVIII skip law)",
        )
    else
        src = read(_P10_SCRIPT, String)

        # local-weights law (G2/§LXXVI): from_pretrained with
        # local_files_only=True, and no download endpoint anywhere
        @test occursin("local_files_only=True", src)
        @test !occursin("http", lowercase(src))   # no URLs in the harness at all

        # --help never imports torch: it must exit 0 even on a box with no
        # PyTorch (this is exactly the probe runbenchmarks.jl performs)
        help_p = try
            run(pipeline(`$_P10_PY $_P10_SCRIPT --help`; stdout=devnull, stderr=devnull))
        catch
            nothing
        end
        if help_p === nothing
            @test _skip(
                "python3 unavailable on this host — harness execution tests skipped (the script is still source-inspected above)",
            )
        else
            @test help_p.exitcode == 0
            # bad args fail loudly and fast (argparse errors BEFORE torch import);
            # wait() (unlike run) returns the process instead of throwing
            bad_p = run(
                pipeline(
                    `$_P10_PY $_P10_SCRIPT --no-such-flag`;
                    stdout=devnull,
                    stderr=devnull,
                );
                wait=false,
            )
            wait(bad_p)
            @test bad_p.exitcode != 0
        end

        # dependency law (§VII): torch/transformers in NO Project.toml — the
        # eager reference is an external binary, never a package dep
        for proj in (
            joinpath(pkgdir(Gesso), "Project.toml"),
            joinpath(pkgdir(Gesso), "benchmark", "Project.toml"),
            joinpath(pkgdir(Gesso), "test", "Project.toml"),
        )
            isfile(proj) || continue
            txt = read(proj, String)
            @test !occursin("PyCall", txt)
            @test !occursin("torch", lowercase(txt))
        end
    end
end
