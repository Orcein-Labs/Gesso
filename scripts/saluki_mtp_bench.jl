# Saluki 27B end-to-end bench with the published MTP head drafting in the verify
# window, against plain greedy decode. The head is one qwen35 nextn block that
# reads the target's final-normed hidden rows (Gesso.Inference.MTPDraft).
#
#   GESSO_QWEN38_27B_GGUF=/path/to/Underdog-Saluki-27B-1.0-IQ2-mix.gguf \
#   GESSO_QWEN38_MTP_GGUF=/path/to/mtp-Qwen3.8-27B-Q4_0.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   julia --project=benchmark -t 16 scripts/saluki_mtp_bench.jl
#
# Optional env: GESSO_BENCH_NEW (new tokens per prompt, default 96),
# GESSO_BENCH_DRAFT (max drafted ids per window, default "2,3"),
# GESSO_MTP_PMIN (comma list of top-10 confidence gates, default "0,0.5"),
# GESSO_MTP_HEAD ("own" uses the file's embedding and lm_head, "target" uses
# Saluki's; default "own"), GESSO_BENCH_ONLY (space-separated prompt tags).
# Every drafted run must give the plain run's ids; a mismatch is an error.

using Gesso
using Printf

const GGUF = get(ENV, "GESSO_QWEN38_27B_GGUF", "")
isfile(GGUF) || error("saluki_mtp_bench: set GESSO_QWEN38_27B_GGUF")
const MTP = get(ENV, "GESSO_QWEN38_MTP_GGUF", "")
isfile(MTP) || error("saluki_mtp_bench: set GESSO_QWEN38_MTP_GGUF")
const NEW = parse(Int, get(ENV, "GESSO_BENCH_NEW", "96"))
const KS = [parse(Int, s) for s in split(get(ENV, "GESSO_BENCH_DRAFT", "2,3"), ',')]
const PMINS =
    [parse(Float64, strip(s)) for s in split(get(ENV, "GESSO_MTP_PMIN", "0,0.5"), ',')]
const HEAD = get(ENV, "GESSO_MTP_HEAD", "own")
HEAD in ("own", "target") || error("saluki_mtp_bench: GESSO_MTP_HEAD is own or target")
const ONLY = split(get(ENV, "GESSO_BENCH_ONLY", ""))
const IM_END = 248046
const CTX = 4096

include(joinpath(@__DIR__, "saluki_prompts.jl"))
include(joinpath(@__DIR__, "saluki_mtp_source.jl"))

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
head = Gesso.Inference.load_mtp_draft(
    MTP;
    ctx=CTX,
    target=HEAD == "target" ? tensors : nothing,
)

# JIT warm for the plain step, the head and every window size.
let p = Gesso.Inference.encode(tk, chat("Hi"))
    gen(p, 4)
    for k in KS
        Gesso.Inference.reset!(head)
        src = MTPSource(head, k)
        gen(p, 4; draft=src, max_draft=k, on_rows=mtp_hook(src))
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
    @printf(
        "PLAIN %s prompt=%d new=%d step_ms=%.1f\n",
        tag,
        P,
        length(ids) - P,
        plain_ms / n_dec
    )
    flush(stdout)
    for k in KS, p_min in PMINS
        info = Ref{Any}(nothing)
        Gesso.Inference.reset!(head)
        src = MTPSource(head, k; p_min)
        tw = @elapsed w =
            gen(prompt, NEW; draft=src, max_draft=k, on_rows=mtp_hook(src), info)
        w == ids || error("mtp k=$k p_min=$p_min changed the ids on $tag")
        sp = info[].speculation
        dec_s = tw - t1
        @printf(
            "MTP %s head=%s k=%d p_min=%.2f gated=%d windows=%d drafted=%d accepted=%d accept_rate=%.2f mtp_ms_per_window=%.1f mtp_share=%.2f speedup=%.2f\n",
            tag,
            HEAD,
            k,
            p_min,
            src.gated,
            sp.windows,
            sp.drafted,
            sp.accepted,
            sp.drafted == 0 ? 0.0 : sp.accepted / sp.drafted,
            1e-6 * src.ns / max(src.calls, 1),
            (src.ns / 1e9) / dec_s,
            plain_ms / (1000 * dec_s)
        )
        flush(stdout)
    end
end
