# juliac --trim stage 1 (R&D, 2026-10-08): the CPU backend and nothing else.
# Reachability is static, so this file must not reach load_gguf or any
# decode path. The stage-2 entry (juliac_min_entry.jl) adds those.
#
#   julia --project=<worktree> juliac.jl --output-exe gesso_stage1 \
#       --experimental --trim=safe scripts/julia_runtime/juliac_stage1_entry.jl

using Gesso

function main(args::Vector{String})::Cint
    backend = Gesso.CPUBackend()
    return backend isa Gesso.CPUBackend ? Cint(0) : Cint(1)
end

@main
