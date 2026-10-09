# Qwen3.8-27B (dense qwen35) warmed CPU decode benchmark.
#
#   GESSO_QWEN38_27B_GGUF=/path/to/model.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   julia --project=<env with Gesso> -t 18 scripts/bench_qwen38_27b.jl
#
# Optional env:
#   GESSO_27B_PROMPT   comma-separated prompt ids (default 9419, "Hello")
#   GESSO_BENCH_N      tokens for the decode window (default 8)
#   GESSO_BENCH_EXPECT comma-separated expected ids for the first run (the
#                      prompt followed by the greedy continuation). Exit 1 on
#                      mismatch.
#
# Method (same as gate_35b.jl): one compile + correctness pass, one warm
# single-token pass, then timed t(1 token) and t(N tokens) on the warmed
# path. Decode tok/s = (N - 1) / (t(N) - t(1)), so prefill and the first
# head-only token cancel.
#
# Prints one RECEIPT line. Load time is wall time of load_gguf. Peak RSS is
# VmHWM from /proc/self/status at the end of the run. The MTP block is not
# executed. The CPU backend is the only backend; no device is touched.

using Gesso

const GGUF = get(ENV, "GESSO_QWEN38_27B_GGUF", "")
isfile(GGUF) || error("bench_qwen38_27b: set GESSO_QWEN38_27B_GGUF to the Qwen3.8-27B GGUF file")
const PROMPT = [parse(Int, s) for s in split(get(ENV, "GESSO_27B_PROMPT", "9419"), ',')]
const N = parse(Int, get(ENV, "GESSO_BENCH_N", "8"))
N >= 2 || error("bench_qwen38_27b: GESSO_BENCH_N must be >= 2")
const EXPECT = let s = get(ENV, "GESSO_BENCH_EXPECT", "")
    isempty(s) ? nothing : [parse(Int, x) for x in split(s, ',')]
end

function _proc_status_mib(key)
    for line in eachline("/proc/self/status")
        startswith(line, key) || continue
        return round(parse(Int, split(line)[2]) / 1024; digits=1)
    end
    return missing
end

backend = Gesso.CPUBackend()

t_load = @elapsed begin
    model, tensors, spec = Gesso.load_gguf(GGUF)
end
rss_after_load = _proc_status_mib("VmRSS:")

gen(n) = Gesso.reference_generate(
    model, tensors, PROMPT; max_new_tokens=n, eps=spec.rms_norm_eps, backend=backend,
)

ids = gen(N)            # compile + correctness
gen(1)                  # warm the single-token path
t1 = @elapsed gen(1)
tN = @elapsed gen(N)
tok_s = (N - 1) / (tN - t1)

peak = _proc_status_mib("VmHWM:")
ok = EXPECT === nothing ? true : ids[1:length(EXPECT)] == EXPECT

println(
    "RECEIPT bench_qwen38_27b backend=cpu threads=$(Threads.nthreads()) ",
    "n=$N load_s=$(round(t_load; digits=1)) rss_after_load_mib=$rss_after_load ",
    "prefill_1tok_s=$(round(t1; digits=3)) decode_tok_s=$(round(tok_s; digits=3)) ",
    "peak_rss_mib=$peak ids_ok=$ok",
)
println("IDS ", ids)
ok || (println("ids want $EXPECT"); exit(1))
