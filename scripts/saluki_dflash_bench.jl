# Saluki 27B end-to-end bench with the published DFlash2 drafter in the verify
# window, against plain greedy decode. The drafter reads the target's residual
# stream after its target layers (Gesso.Inference.DFlashDraft) and drafts a
# block of ids in one pass.
#
#   GESSO_QWEN38_27B_GGUF=/path/to/Underdog-Saluki-27B-1.0-IQ2-mix.gguf \
#   GESSO_QWEN38_DFLASH_GGUF=/path/to/dflash-Qwen3.8-27B-Q4_0.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   julia --project=benchmark -t 16 scripts/saluki_dflash_bench.jl
#
# Optional env: GESSO_BENCH_NEW (new tokens per prompt, default 96),
# GESSO_BENCH_DRAFT (max drafted ids per window, default "7"),
# GESSO_DFLASH_PMIN (comma list of walk confidence gates, default "0,0.7"),
# GESSO_BENCH_ONLY (space-separated prompt tags).
# Every drafted run must give the plain run's ids; a mismatch is an error.

using Gesso
using Printf

const GGUF = get(ENV, "GESSO_QWEN38_27B_GGUF", "")
isfile(GGUF) || error("saluki_dflash_bench: set GESSO_QWEN38_27B_GGUF")
const DFLASH = get(ENV, "GESSO_QWEN38_DFLASH_GGUF", "")
isfile(DFLASH) || error("saluki_dflash_bench: set GESSO_QWEN38_DFLASH_GGUF")
const NEW = parse(Int, get(ENV, "GESSO_BENCH_NEW", "96"))
const KS = [parse(Int, s) for s in split(get(ENV, "GESSO_BENCH_DRAFT", "7"), ',')]
const PMINS = [parse(Float64, strip(s)) for s in split(get(ENV, "GESSO_DFLASH_PMIN", "0,0.7"), ',')]
const ONLY = split(get(ENV, "GESSO_BENCH_ONLY", ""))
const IM_END = 248046
const CTX = 4096

include(joinpath(@__DIR__, "saluki_prompts.jl"))
include(joinpath(@__DIR__, "saluki_dflash_source.jl"))

backend = Gesso.CPUBackend()
model, tensors, spec = Gesso.load_gguf(GGUF)
tk = Gesso.Inference.load_gguf_tokenizer(GGUF)
gen(prompt, n; kw...) = Gesso.reference_generate(model, tensors, prompt; max_new_tokens=n,
    eps=spec.rms_norm_eps, backend, eos_ids=(IM_END,), kw...)
drafter = Gesso.Inference.load_dflash_draft(DFLASH; ctx=CTX, target=tensors)

# JIT warm for the plain step, the drafter and every window size.
let p = Gesso.Inference.encode(tk, chat("Hi"))
    gen(p, 4)
    for k in KS
        Gesso.Inference.reset!(drafter)
        src = DFlashSource(drafter, k)
        gen(p, 4; draft=src, max_draft=k, taps=dflash_taps(src), on_taps=dflash_hook(src))
    end
end

for (tag, msg) in PROMPTS
    isempty(ONLY) || tag in ONLY || continue
    prompt = Gesso.Inference.encode(tk, chat(msg))
    P = length(prompt)
    # Prefill and the first token are shared by every run: time them once and
    # subtract, so the rest is decode.
    t1 = @elapsed gen(prompt, 1)
    tN = @elapsed ids = gen(prompt, NEW)
    n_dec = length(ids) - P - 1
    plain_ms = 1000 * (tN - t1)
    @printf("PLAIN %s prompt=%d new=%d step_ms=%.1f\n", tag, P, length(ids) - P, plain_ms / n_dec)
    flush(stdout)
    for k in KS, p_min in PMINS
        info = Ref{Any}(nothing)
        Gesso.Inference.reset!(drafter)
        src = DFlashSource(drafter, k; p_min)
        tw = @elapsed w = gen(prompt, NEW; draft=src, max_draft=k, taps=dflash_taps(src),
            on_taps=dflash_hook(src), info)
        w == ids || error("dflash k=$k p_min=$p_min changed the ids on $tag")
        sp = info[].speculation
        dec_s = tw - t1
        @printf("DFLASH %s k=%d p_min=%.2f gated=%d windows=%d drafted=%d accepted=%d accept_rate=%.2f draft_ms_per_window=%.1f draft_share=%.2f speedup=%.2f\n",
            tag, k, p_min, src.gated, sp.windows, sp.drafted, sp.accepted,
            sp.drafted == 0 ? 0.0 : sp.accepted / sp.drafted,
            1e-6 * src.ns / max(src.calls, 1), (src.ns / 1e9) / dec_s,
            plain_ms / (1000 * dec_s))
        flush(stdout)
    end
end
