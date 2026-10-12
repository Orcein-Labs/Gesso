# Time-to-first-token and decode GC probe (R&D, 2026-10-08).
#
# Splits one cold process into phases, each measured with @timed so that JIT
# time (compile_time, recompile_time) and GC (gctime, gcstats.pause) are
# reported next to wall time:
#
#   load      load_gguf (and to_device for cuda)
#   ttft1     first gen(1): first-call JIT + prefill + one token
#   ttft2     second gen(1): warm prefill + one token
#   genN      gen(N): warm prefill + N tokens
#
# Decode per token = (genN - ttft2) / (N - 1). Prefill cancels in that
# difference, as in scripts/bench_qwen38_27b.jl.
#
#   GESSO_BACKEND=lava|cuda julia --project=<env> -t 16 scripts/julia_runtime/ttft_probe.jl
#   GESSO_BACKEND=cpu      julia --project=<env> -t 16 scripts/julia_runtime/ttft_probe.jl
#
# GESSO_BACKEND defaults to lava. lava and cuda run the 35B; cpu runs the 27B.
# A GPU package is loaded only when its backend is asked for.
#
# Env: GESSO_QWEN38_27B_GGUF, GESSO_ORNITH35_GGUF, GESSO_BENCH_N (default 8).
# Process startup and package load are measured by the shell wrapper.

which = get(ENV, "GESSO_BACKEND", "lava")
t_pkg0 = time()
using Gesso
const PKG_LOAD_S = time() - t_pkg0

const N = parse(Int, get(ENV, "GESSO_BENCH_N", "8"))
const PROMPT = [9419]

if which == "lava"
    const GGUF = get(ENV, "GESSO_ORNITH35_GGUF", "")
    isfile(GGUF) || error("set GESSO_ORNITH35_GGUF")
    t_gpu0 = time()
    using Lava
    const CUDA_PKG_LOAD_S = time() - t_gpu0
    backend = Gesso.LavaBackend()
elseif which == "cuda"
    const GGUF = get(ENV, "GESSO_ORNITH35_GGUF", "")
    isfile(GGUF) || error("set GESSO_ORNITH35_GGUF")
    t_cuda0 = time()
    using CUDA
    const CUDA_PKG_LOAD_S = time() - t_cuda0
    backend = Gesso.CUDABackend()
elseif which == "cpu"
    const GGUF = get(ENV, "GESSO_QWEN38_27B_GGUF", "")
    isfile(GGUF) || error("set GESSO_QWEN38_27B_GGUF")
    backend = Gesso.CPUBackend()
    const CUDA_PKG_LOAD_S = 0.0
else
    error("ttft_probe: GESSO_BACKEND must be lava, cuda or cpu, got '$which'")
end

function row(name, r)
    g = r.gcstats
    println(
        rpad(name, 8),
        " wall_s=", round(r.time; digits=3),
        " compile_s=", round(r.compile_time; digits=3),
        " recompile_s=", round(r.recompile_time; digits=3),
        " gc_s=", round(r.gctime; digits=3),
        " gc_pauses=", g.pause,
        " alloc_MiB=", round(r.bytes / 2^20; digits=1),
    )
end

println("pkg_load_s=", round(PKG_LOAD_S; digits=3), " cuda_pkg_load_s=", round(CUDA_PKG_LOAD_S; digits=3))

rl = @timed Gesso.load_gguf(GGUF)
model, tensors, spec = rl.value
dev = which == "cpu" ? tensors : Gesso.to_device(backend, tensors)
row("load", rl)

gen(n) = Gesso.reference_generate(
    model, dev, PROMPT; max_new_tokens=n, eps=spec.rms_norm_eps, backend=backend,
)

r1 = @timed gen(1)
row("ttft1", r1)
r2 = @timed gen(1)
row("ttft2", r2)
rN = @timed gen(N)
row("genN", rN)

per_tok_s = (rN.time - r2.time) / (N - 1)
per_tok_mib = (rN.bytes - r2.bytes) / (N - 1) / 2^20
per_tok_gc_ms = (rN.gctime - r2.gctime) / (N - 1) * 1e3
per_tok_pauses = (rN.gcstats.pause - r2.gcstats.pause) / (N - 1)
println(
    "RECEIPT ttft_probe which=$which threads=$(Threads.nthreads()) N=$N ",
    "decode_ms_per_tok=$(round(1e3 * per_tok_s; digits=2)) ",
    "tok_s=$(round(1 / per_tok_s; digits=3)) ",
    "alloc_MiB_per_tok=$(round(per_tok_mib; digits=3)) ",
    "gc_ms_per_tok=$(round(per_tok_gc_ms; digits=3)) ",
    "gc_pauses_per_tok=$(round(per_tok_pauses; digits=2)) ",
    "first_token_extra_s=$(round(r1.time - r2.time; digits=3)) ",
    "first_token_compile_s=$(round(r1.compile_time; digits=3))",
)
