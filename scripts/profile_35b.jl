# Ornith-1.5-35B-A3B decode stage profile (GESSO_PROFILE stage profiler).
#
#   GESSO_ORNITH35_GGUF=/path/to/Unsloth-Ornith-1.5-35B-A3B-UD-Q2_K_XL.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   julia --project=benchmark -t 16 scripts/profile_35b.jl [cuda|cpu]
#
# Optional env: GESSO_PROFILE_TOKENS (decode tokens to measure, default 8).
# The engine switches are read as usual: GESSO_GPU_WEIGHT_DTYPE (f16 default),
# GESSO_MOE_HOST_EXPERTS (0 default).
#
# Three phases, one process:
#   1. JIT warm: two tokens, profiler off (compiles the decode path).
#   2. Cold token log: N+1 tokens profiled right after the JIT warm, so the
#      first touches of the mmap'd expert pages happen inside the log.
#   3. Warm table: profiled 1-token run and profiled (N+1)-token run; the
#      difference is exactly N decode tokens (prefill and the head-only
#      token cancel). Prefill is never inside a token window anyway.
#
# Prints a RECEIPT line, the cold token log, and the warm per-token table.
# Exit 1 if the greedy ids change under profiling.

using Gesso

const GGUF = get(ENV, "GESSO_ORNITH35_GGUF", "")
isfile(GGUF) || error("profile_35b: set GESSO_ORNITH35_GGUF to the Ornith 35B GGUF file")
const GGML_DIR = get(ENV, "GESSO_GGML_DIR", "")
isdir(GGML_DIR) || error("profile_35b: set GESSO_GGML_DIR to the llama.cpp build/bin directory")
const EXPECTED = [9419, 11, 271, 40, 2688, 4313, 310, 958, 279]
const N = parse(Int, get(ENV, "GESSO_PROFILE_TOKENS", "8"))
N >= 1 || error("profile_35b: GESSO_PROFILE_TOKENS must be >= 1")

which = isempty(ARGS) ? "cuda" : ARGS[1]
if which == "cuda"
    using CUDA
    backend = Gesso.CUDABackend()
else
    backend = Gesso.CPUBackend()
end

const INF = Gesso.Inference

t_load = @elapsed begin
    model, tensors, spec = Gesso.load_gguf(GGUF)
    dev = which == "cuda" ? Gesso.to_device(backend, tensors) : tensors
end

gen(n) = Gesso.reference_generate(
    model, dev, [9419]; max_new_tokens=n, eps=spec.rms_norm_eps, backend=backend,
)

dtype = get(ENV, "GESSO_GPU_WEIGHT_DTYPE", "f16")
moe_host = get(ENV, "GESSO_MOE_HOST_EXPERTS", "0")

INF.profile_disable!()
gen(2)

INF.profile_enable!()
ids_cold = gen(N + 1)
cold = INF.profile_snapshot()

INF.profile_enable!()
gen(1)
a = INF.profile_snapshot()
INF.profile_reset!()
ids_warm = gen(N + 1)
b = INF.profile_snapshot()
INF.profile_disable!()

ids_warm == ids_cold || error("profile_35b: ids differ between the cold and warm profiled runs")
if N == length(EXPECTED) - 1
    ok = ids_cold[1:length(EXPECTED)] == EXPECTED
else
    ok = true
end

d = INF.profile_delta(b, a)
d.tokens == N || error("profile_35b: measured $(d.tokens) decode tokens, expected $N (EOS reached?)")

println(
    "RECEIPT profile_35b backend=$which threads=$(Threads.nthreads()) ",
    "dtype=$dtype moe_host_experts=$moe_host decode_tokens=$N ",
    "ids_ok=$ok load_s=$(round(t_load; digits=1))",
)
println()
println("== cold token log (JIT warm, then first profiled run in this process) ==")
INF.print_token_log(stdout, cold)
println()
println("== warm per-token table (difference of profiled runs, $N decode tokens) ==")
INF.print_decode_profile(stdout, d; label="backend=$which dtype=$dtype moe_host=$moe_host")
ok || (println("ids got  $ids_cold\nids want $EXPECTED"); exit(1))
