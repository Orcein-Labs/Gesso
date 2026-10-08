# Prepare the pinned compiler prerequisites without precompiling Lava itself.
# Run before --compiled-modules=existing primary-device probes on a fresh depot.
import Pkg
root=dirname(@__DIR__)
Pkg.activate(root; io=devnull)
Pkg.instantiate(; allow_autoprecomp=false)
manifest=read(joinpath(root, "Manifest.toml"), String)
try
    mktempdir() do environment
        write(
            joinpath(environment, "Project.toml"),
            """
[deps]
LLVM = "929cbde3-209d-540e-8aea-75f648917ca0"
GPUCompiler = "61eb1bfa-7361-4325-ad38-22787b887f55"
""",
        )
        # The dependencies are already instantiated from this exact lock.
        # Precompilation cannot resolve or upgrade that dependency set.
        write(joinpath(environment, "Manifest.toml"), manifest)
        Pkg.activate(environment; io=devnull)
        Pkg.precompile(; strict=true, already_instantiated=true)
    end
finally
    Pkg.activate(root; io=devnull)
end
