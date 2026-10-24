# Saluki 27B end-to-end bench with a real bean draft source: plain greedy decode
# against the same decode with Qwen3.5-0.8B (the bean) drafting in the verify window.
#
#   GESSO_QWEN38_27B_GGUF=/path/to/Underdog-Saluki-27B-1.0-IQ2-mix.gguf \
#   GESSO_BEAN_GGUF=/path/to/Qwen3.5-0.8B-Q8_0.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   julia --project=benchmark -t 16 scripts/saluki_bean_bench.jl
#
# Optional env: GESSO_BENCH_NEW (new tokens per prompt, default 96),
# GESSO_BENCH_DRAFT (max drafted ids per window, default "2,3,4"),
# GESSO_BEAN_TAU (comma list of confidence gates, default "0"). With tau > 0 the
# bean stops before the first token whose top softmax probability is below tau,
# and that token is not drafted. tau = 0 means no gate.
#
# GESSO_BEAN_SELFTEST=1 loads only the bean and uses it as both target and
# drafter. Under greedy decode a model drafting for itself must be accepted
# 100%, and the ids must match plain decode. Any mismatch is an error.
#
# The source is BeanSource. It keeps one GenerateState across calls: on a hybrid
# model reference_generate snapshots the GDN state where each prompt ends, so
# after the target rejects part of a draft the next call resumes from that
# snapshot and prefills only the accepted ids plus the correction.

using Gesso
using Printf

const SELFTEST = get(ENV, "GESSO_BEAN_SELFTEST", "") == "1"
const GGUF = get(ENV, "GESSO_QWEN38_27B_GGUF", "")
SELFTEST || isfile(GGUF) || error("saluki_bean_bench: set GESSO_QWEN38_27B_GGUF")
const BEAN = get(ENV, "GESSO_BEAN_GGUF", "")
isfile(BEAN) || error("saluki_bean_bench: set GESSO_BEAN_GGUF")
const NEW = SELFTEST ? 32 : parse(Int, get(ENV, "GESSO_BENCH_NEW", "96"))
const KS =
    SELFTEST ? [2, 3, 4] :
    [parse(Int, s) for s in split(get(ENV, "GESSO_BENCH_DRAFT", "2,3,4"), ',')]
const TAUS =
    SELFTEST ? [0.0, 0.6] :
    [parse(Float64, strip(s)) for s in split(get(ENV, "GESSO_BEAN_TAU", "0"), ',')]
const IM_END = 248046

include(joinpath(@__DIR__, "saluki_prompts.jl"))
include(joinpath(@__DIR__, "saluki_bean_source.jl"))

backend = Gesso.CPUBackend()
bean, btensors, bspec = Gesso.load_gguf(BEAN)
bgen(prompt, n; kw...) = Gesso.reference_generate(
    bean,
    btensors,
    prompt;
    max_new_tokens=n,
    eps=bspec.rms_norm_eps,
    backend,
    eos_ids=(IM_END,),
    kw...,
)
btk = Gesso.Inference.load_gguf_tokenizer(BEAN)

if SELFTEST
    for (tag, msg) in PROMPTS
        prompt = Gesso.Inference.encode(btk, chat(msg))
        ids = bgen(prompt, NEW)
        for k in KS, tau in TAUS
            src = BeanSource(
                bean,
                btensors,
                Float64(bspec.rms_norm_eps),
                backend,
                Gesso.GenerateState(4096),
                k,
                tau,
                0,
                0,
                0,
            )
            info = Ref{Any}(nothing)
            w = bgen(prompt, NEW; draft=src, max_draft=k, info)
            w == ids || error("selftest k=$k tau=$tau changed the ids on $tag")
            sp = info[].speculation
            sp.accepted == sp.drafted || error(
                "selftest k=$k tau=$tau rejected a self-draft on $tag: accepted=$(sp.accepted) drafted=$(sp.drafted)",
            )
            @printf(
                "SELFTEST %s k=%d tau=%.2f gated=%d windows=%d drafted=%d accepted=%d\n",
                tag,
                k,
                tau,
                src.gated,
                sp.windows,
                sp.drafted,
                sp.accepted
            )
        end
    end
    exit(0)
end

model, tensors, spec = Gesso.load_gguf(GGUF)
tk = Gesso.Inference.load_gguf_tokenizer(GGUF)
size(tensors.embedding.storage, 1) == size(btensors.embedding.storage, 1) ||
    error("saluki_bean_bench: bean and Saluki vocabularies differ")
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

# JIT warm for the plain step and every window size.
let p = Gesso.Inference.encode(tk, chat("Hi"))
    gen(p, 4)
    for k in KS, tau in TAUS
        src = BeanSource(
            bean,
            btensors,
            Float64(bspec.rms_norm_eps),
            backend,
            Gesso.GenerateState(4096),
            k,
            tau,
            0,
            0,
            0,
        )
        gen(p, 4; draft=src, max_draft=k)
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
    flush(stdout)
    for k in KS, tau in TAUS
        # fresh source and state per run, so runs are independent
        src = BeanSource(
            bean,
            btensors,
            Float64(bspec.rms_norm_eps),
            backend,
            Gesso.GenerateState(4096),
            k,
            tau,
            0,
            0,
            0,
        )
        # The bean's prompt prefill belongs to time to first token, like the
        # target's: prime the state outside the timed decode, report it apart.
        bp_ms = 1000 * @elapsed bgen(prompt, 1; state=src.state)
        info = Ref{Any}(nothing)
        tw = @elapsed w = gen(prompt, NEW; draft=src, max_draft=k, info)
        w == ids || error("bean k=$k tau=$tau changed the ids on $tag")
        sp = info[].speculation
        drafted_s = tw - t1
        @printf(
            "BEAN %s k=%d tau=%.2f gated=%d windows=%d drafted=%d accepted=%d accept_rate=%.2f bean_ms_per_window=%.1f bean_share=%.2f bean_prefill_ms=%.0f speedup=%.2f\n",
            tag,
            k,
            tau,
            src.gated,
            sp.windows,
            sp.drafted,
            sp.accepted,
            sp.drafted == 0 ? 0.0 : sp.accepted / sp.drafted,
            1e-6 * src.ns / max(sp.windows, 1),
            (src.ns / 1e9) / drafted_s,
            bp_ms,
            plain_ms / (1000 * drafted_s)
        )
        flush(stdout)
    end
end
