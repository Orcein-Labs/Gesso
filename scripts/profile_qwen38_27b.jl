# Qwen3.8-27B (dense qwen35) decode stage profile on the CPU backend.
#
#   GESSO_QWEN38_27B_GGUF=/path/to/model.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   julia --project=<env with Gesso> -t 18 scripts/profile_qwen38_27b.jl
#
# Optional env: GESSO_PROFILE_TOKENS (decode tokens to measure, default 8).
#
# Phases, one process (same contract as profile_35b.jl, without the
# Ornith-specific id gate):
#   1. JIT warm: two tokens, profiler off.
#   2. Cold token log: N+1 tokens profiled.
#   3. Warm table: profiled 1-token run and profiled (N+1)-token run; the
#      difference is exactly N decode tokens.
#
# Prints a RECEIPT line, the cold token log and the warm per-token table.
# The profile is opt-in; timing numbers belong to bench_qwen38_27b.jl, which
# runs without the profiler.

using Gesso

const GGUF = get(ENV, "GESSO_QWEN38_27B_GGUF", "")
isfile(GGUF) || error("profile_qwen38_27b: set GESSO_QWEN38_27B_GGUF to the Qwen3.8-27B GGUF file")
const GGML_DIR = get(ENV, "GESSO_GGML_DIR", "")
isdir(GGML_DIR) || error("profile_qwen38_27b: set GESSO_GGML_DIR to the llama.cpp build/bin directory")
const N = parse(Int, get(ENV, "GESSO_PROFILE_TOKENS", "8"))
N >= 1 || error("profile_qwen38_27b: GESSO_PROFILE_TOKENS must be >= 1")
const PROMPT = [9419]

backend = Gesso.CPUBackend()
const INF = Gesso.Inference

t_load = @elapsed begin
    model, tensors, spec = Gesso.load_gguf(GGUF)
end

gen(n) = Gesso.reference_generate(
    model, tensors, PROMPT; max_new_tokens=n, eps=spec.rms_norm_eps, backend=backend,
)

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

ids_warm == ids_cold || error("profile_qwen38_27b: ids differ between the cold and warm profiled runs")

d = INF.profile_delta(b, a)
d.tokens == N || error("profile_qwen38_27b: measured $(d.tokens) decode tokens, expected $N (EOS reached?)")

println(
    "RECEIPT profile_qwen38_27b backend=cpu threads=$(Threads.nthreads()) ",
    "decode_tokens=$N load_s=$(round(t_load; digits=1))",
)
println("IDS ", ids_cold)
println()
println("== cold token log (JIT warm, then first profiled run in this process) ==")
INF.print_token_log(stdout, cold)
println()
println("== warm per-token table (difference of profiled runs, $N decode tokens) ==")
INF.print_decode_profile(stdout, d; label="backend=cpu model=qwen38_27b")
