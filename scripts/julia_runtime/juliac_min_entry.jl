# Minimal juliac (Julia 1.12 --trim) entry point for Gesso (R&D, 2026-10-08).
#
# Stage 1 (this file, GESSO_TRIM_STAGE=1): construct the CPU backend only.
# Stage 2 (GESSO_TRIM_STAGE=2): load a GGUF and run the greedy reference
# decode for N tokens, the same call path as scripts/bench_qwen38_27b.jl.
#
#   julia --project=<worktree> juliac.jl --output-exe gesso_min \
#       --experimental --trim=safe scripts/julia_runtime/juliac_min_entry.jl
#
# Output is informational only; trim verification errors are the result.

using Gesso

function main(args::Vector{String})::Cint
    stage = get(ENV, "GESSO_TRIM_STAGE", "1")
    backend = Gesso.CPUBackend()
    if stage == "1"
        println(stdout, "gesso trim stage1 ok backend=", typeof(backend))
        return 0
    end
    gguf = get(ENV, "GESSO_QWEN38_27B_GGUF", "")
    isfile(gguf) || (println(stderr, "set GESSO_QWEN38_27B_GGUF"); return 2)
    model, tensors, spec = Gesso.load_gguf(gguf)
    ids = Gesso.reference_generate(
        model,
        tensors,
        [9419];
        max_new_tokens=8,
        eps=spec.rms_norm_eps,
        backend=backend,
    )
    println(stdout, "gesso trim stage2 ids=", ids)
    return 0
end

@main
