# Ornith-1.5-35B-A3B decode and prefill by context length, one backend.
#
#   GESSO_ORNITH35_GGUF=/path/to/model.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   GESSO_BACKEND=lava|cuda|cpu \
#   [GESSO_BENCH_CONTEXTS=512,8192,16384,32768] [GESSO_BENCH_NEW=65] \
#   [GESSO_BENCH_TEXT=docs/Gesso_Stack.md] [GESSO_BENCH_IDS=out.tsv] [GESSO_BENCH_REPS=2] \
#   julia --project=benchmark -t 16 scripts/bench_35b.jl
#
# Each context is one greedy generation on a fresh state, timed from the call
# to every token as `on_token` sees it. TTFT therefore includes prefill, the
# first argmax and the state's allocation, which is what a client waits for.
# Inter-token gaps exclude the first token. The prompt is real text, repeated
# until it is long enough, because hot-expert hit rates depend on routing and
# random ids route unlike any conversation.
#
# A context runs GESSO_BENCH_REPS times and the last run is reported, so a
# kernel first compiled at that length is not billed as prefill. The first
# run's TTFT is printed as ttft_cold_s.
#
# Prints one RECEIPT line per context. GESSO_BENCH_IDS, when set, gets the
# prompt and generated ids of every context so another engine can be fed the
# same ids and compared token for token.

using Gesso
using Printf
using Statistics: mean, median, quantile

const GGUF = get(ENV, "GESSO_ORNITH35_GGUF", "")
isfile(GGUF) || error("bench_35b: set GESSO_ORNITH35_GGUF to the Ornith 35B GGUF file")
const CONTEXTS =
    parse.(Int, split(get(ENV, "GESSO_BENCH_CONTEXTS", "512,8192,16384,32768"), ','))
const NEW = parse(Int, get(ENV, "GESSO_BENCH_NEW", "65"))
const TEXT =
    get(ENV, "GESSO_BENCH_TEXT", joinpath(@__DIR__, "..", "docs", "Gesso_Stack.md"))
const IDS_OUT = get(ENV, "GESSO_BENCH_IDS", "")
const REPS = parse(Int, get(ENV, "GESSO_BENCH_REPS", "2"))
NEW >= 3 || error("bench_35b: GESSO_BENCH_NEW must be at least 3 to have inter-token gaps")

which = get(ENV, "GESSO_BACKEND", "lava")
if which == "lava"
    using Lava
    backend = Gesso.LavaBackend()
elseif which == "cuda"
    using CUDA
    backend = Gesso.CUDABackend()
elseif which == "cpu"
    backend = Gesso.CPUBackend()
else
    error("bench_35b: GESSO_BACKEND must be lava, cuda or cpu, got '$which'")
end

t_load = @elapsed begin
    model, tensors, spec = Gesso.load_gguf(GGUF)
    dev = which == "cpu" ? tensors : Gesso.to_device(backend, tensors)
end
tk = Gesso.Inference.load_gguf_tokenizer(GGUF)
corpus = Gesso.Inference.encode(tk, read(TEXT, String); parse_special=false)
prompt_of(P) = [corpus[mod1(i, length(corpus))] for i in 1:P]

function timed_generate(prompt, n)
    stamps = Float64[]
    t0 = time()
    ids = Gesso.reference_generate(
        model,
        dev,
        prompt;
        max_new_tokens=n,
        eps=spec.rms_norm_eps,
        backend=backend,
        eos_ids=(),
        on_token=_ -> (push!(stamps, time()); true),
    )
    return ids, stamps .- t0
end

# Compile the prefill and decode paths before anything is timed.
timed_generate(prompt_of(64), 4)

ids_io = isempty(IDS_OUT) ? nothing : open(IDS_OUT, "w")
println(
    "RECEIPT bench_35b load backend=$which threads=$(Threads.nthreads()) load_s=$(round(t_load; digits=1)) ",
    "kv=$(get(ENV, "GESSO_KV_FORMAT", "default")) hot=$(get(ENV, "GESSO_GPU_HOT_EXPERTS", "1")) ",
    "hot_mib=$(get(ENV, "GESSO_GPU_HOT_EXPERTS_MIB", "auto")) model=$(basename(GGUF))",
)
for P in CONTEXTS
    prompt = prompt_of(P)
    ids, t = timed_generate(prompt, NEW)
    cold = t[1]
    for _ in 2:REPS
        GC.gc()
        ids, t = timed_generate(prompt, NEW)
    end
    gaps = diff(t) .* 1e3
    @printf(
        "RECEIPT bench_35b ctx=%d new=%d ttft_cold_s=%.3f ttft_s=%.3f prefill_tok_s=%.1f decode_tok_s=%.2f itl_mean_ms=%.2f itl_p50_ms=%.2f itl_p90_ms=%.2f itl_p99_ms=%.2f itl_max_ms=%.2f maxrss_gib=%.2f\n",
        P,
        length(t),
        cold,
        t[1],
        P / t[1],
        (length(t) - 1) / (t[end] - t[1]),
        mean(gaps),
        median(gaps),
        quantile(gaps, 0.9),
        quantile(gaps, 0.99),
        maximum(gaps),
        Sys.maxrss() / 2^30
    )
    flush(stdout)
    if ids_io !== nothing
        println(
            ids_io,
            P,
            '\t',
            join(prompt, ','),
            '\t',
            join(ids[(length(prompt)+1):end], ','),
        )
        flush(ids_io)
    end
    GC.gc()
end
ids_io === nothing || close(ids_io)
