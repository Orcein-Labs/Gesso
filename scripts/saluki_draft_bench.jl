# Saluki 27B end-to-end draft bench on the CPU backend: plain greedy decode
# against the same decode with a real draft source in the verify window.
#
#   GESSO_QWEN38_27B_GGUF=/path/to/Underdog-Saluki-27B-1.0-IQ2-mix.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   julia --project=benchmark -t 16 scripts/saluki_draft_bench.jl
#
# Optional env: GESSO_BENCH_NEW (new tokens per prompt, default 96),
# GESSO_BENCH_DRAFT (max drafted ids per window, default "3,7"),
# GESSO_BENCH_MIN (minimum suffix match lengths, default "1").
#
# The source is SuffixSource (prompt lookup: the longest suffix of the context
# that occurred earlier proposes what followed it). It runs on the host and
# costs microseconds, so the measured time is the whole cost. Every drafted run
# must reproduce the plain ids, since the window verifies greedily.

using Gesso
using Printf

const GGUF = get(ENV, "GESSO_QWEN38_27B_GGUF", "")
isfile(GGUF) || error("saluki_draft_bench: set GESSO_QWEN38_27B_GGUF")
const NEW = parse(Int, get(ENV, "GESSO_BENCH_NEW", "96"))
const KS = [parse(Int, s) for s in split(get(ENV, "GESSO_BENCH_DRAFT", "3,7"), ',')]
const MINS = [parse(Int, s) for s in split(get(ENV, "GESSO_BENCH_MIN", "1"), ',')]
const IM_END = 248046

include(joinpath(@__DIR__, "saluki_prompts.jl"))

backend = Gesso.CPUBackend()
model, tensors, spec = Gesso.load_gguf(GGUF)
tk = Gesso.Inference.load_gguf_tokenizer(GGUF)
gen(prompt, n; kw...) = Gesso.reference_generate(
    model,
    tensors,
    prompt;
    max_new_tokens=n,
    eps=spec.rms_norm_eps,
    backend,
    eos_ids=(IM_END,),
    kw...,
)

let p = Gesso.Inference.encode(tk, chat("Hi"))
    gen(p, 4)
    for k in KS
        gen(p, 4; draft=Gesso.Speculation.SuffixSource(; max_draft=k), max_draft=k)
    end
end

for (tag, msg) in PROMPTS
    prompt = Gesso.Inference.encode(tk, chat(msg))
    P = length(prompt)
    # Prefill and the first token are shared by every run: time them once and
    # subtract, so the rest is decode.
    t1 = @elapsed gen(prompt, 1)
    tN = @elapsed ids = gen(prompt, NEW)
    n_dec = length(ids) - P - 1
    plain_ms = 1000 * (tN - t1)
    @printf(
        "PLAIN %s prompt=%d new=%d step_ms=%.1f\n",
        tag,
        P,
        length(ids) - P,
        plain_ms / n_dec
    )
    for mm in MINS, k in KS
        src = Gesso.Speculation.SuffixSource(; min_match=mm, max_draft=k)
        info = Ref{Any}(nothing)
        tw = @elapsed w = gen(prompt, NEW; draft=src, max_draft=k, info)
        w == ids || error("suffix min=$mm k=$k changed the ids on $tag")
        sp = info[].speculation
        @printf(
            "SUFFIX %s min=%d k=%d windows=%d drafted=%d accepted=%d accept_rate=%.2f speedup=%.2f\n",
            tag,
            mm,
            k,
            sp.windows,
            sp.drafted,
            sp.accepted,
            sp.drafted == 0 ? 0.0 : sp.accepted / sp.drafted,
            plain_ms / (1000 * (tw - t1))
        )
    end
end
