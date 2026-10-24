# Saluki 27B end-to-end bench: the n-gram suffix drafter first, the bean (small
# model) as fallback when the suffix finds nothing. Compares against plain greedy
# decode and the bean alone (saluki_bean_bench.jl), with an optional back-off that
# stops asking the bean for a while after repeated low-confidence proposals.
#
#   GESSO_QWEN38_27B_GGUF=/path/to/Underdog-Saluki-27B-1.0-IQ2-mix.gguf \
#   GESSO_BEAN_GGUF=/path/to/Qwen3.5-0.8B-Q8_0.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   julia --project=benchmark -t 16 scripts/saluki_combo_bench.jl
#
# Optional env: GESSO_BENCH_NEW (new tokens per prompt, default 96),
# GESSO_BENCH_DRAFT (max drafted ids per window, default "3,4"),
# GESSO_BEAN_TAU (comma list of bean confidence gates, default "0.5"; see
# saluki_bean_source.jl), GESSO_SUFFIX_MIN (suffix min_match, default 3),
# GESSO_BEAN_BACKOFF (comma list of after:skip pairs, default "0:0,2:4,3:8"). After
# `after` consecutive bean proposals that drafted nothing, the next `skip` proposals
# are answered empty, so those steps run plain. after = 0 never backs off.
# GESSO_BENCH_ONLY (comma list of prompt tags, default all) reruns a subset.

using Gesso
using Printf

const GGUF = get(ENV, "GESSO_QWEN38_27B_GGUF", "")
isfile(GGUF) || error("saluki_combo_bench: set GESSO_QWEN38_27B_GGUF")
const BEAN = get(ENV, "GESSO_BEAN_GGUF", "")
isfile(BEAN) || error("saluki_combo_bench: set GESSO_BEAN_GGUF")
const NEW = parse(Int, get(ENV, "GESSO_BENCH_NEW", "96"))
const KS = [parse(Int, s) for s in split(get(ENV, "GESSO_BENCH_DRAFT", "3,4"), ',')]
const TAUS = [parse(Float64, strip(s)) for s in split(get(ENV, "GESSO_BEAN_TAU", "0.5"), ',')]
const SUFFIX_MIN = parse(Int, get(ENV, "GESSO_SUFFIX_MIN", "3"))
const BACKOFFS = [Tuple(parse.(Int, split(strip(s), ':'))) for s in split(get(ENV, "GESSO_BEAN_BACKOFF", "0:0,2:4,3:8"), ',')]
const ONLY = Set(filter(!isempty, strip.(split(get(ENV, "GESSO_BENCH_ONLY", ""), ','))))
const IM_END = 248046

include(joinpath(@__DIR__, "saluki_prompts.jl"))
include(joinpath(@__DIR__, "saluki_bean_source.jl"))

# Suffix first, bean as fallback. The back-off state is per source, so every run
# starts fresh.
mutable struct ComboSource <: Gesso.Speculation.DraftSource
    suffix::Gesso.Speculation.SuffixSource
    bean::BeanSource
    backoff_after::Int   # this many consecutive gated bean proposals start a back-off; 0 = never
    backoff_skip::Int    # propose calls answered empty (plain step, no bean) during a back-off
    consec_gated::Int
    skip_left::Int
    suffix_windows::Int  # proposals answered by the n-gram
    bean_calls::Int      # proposals that ran the bean
    skipped::Int         # proposals answered empty because of back-off
end
function Gesso.Speculation.propose(src::ComboSource, ctx::Gesso.Speculation.DraftContext; max_tokens::Int)
    k = min(max_tokens, src.bean.max_draft)
    k <= 0 && return Gesso.Speculation.Draft(Int[], :bean)
    d = Gesso.Speculation.propose(src.suffix, ctx; max_tokens=k)
    if !isempty(d.ids)
        src.suffix_windows += 1
        return d
    end
    # While the bean is skipped its GenerateState falls behind the context. The next
    # bean call prefills the gap from its snapshot (same mechanism as after a rejection).
    if src.skip_left > 0
        src.skip_left -= 1
        src.skipped += 1
        return Gesso.Speculation.Draft(Int[], :bean)
    end
    src.bean_calls += 1
    db = Gesso.Speculation.propose(src.bean, ctx; max_tokens=k)
    if isempty(db.ids)
        src.consec_gated += 1
        if src.backoff_after > 0 && src.consec_gated >= src.backoff_after
            src.skip_left = src.backoff_skip
            src.consec_gated = 0
        end
    else
        src.consec_gated = 0
    end
    db
end
function Gesso.Speculation.observe!(src::ComboSource, d::Gesso.Speculation.Draft, accepted::Int)
    d.origin == :suffix && Gesso.Speculation.observe!(src.suffix, d, accepted)
    nothing
end

backend = Gesso.CPUBackend()
bean, btensors, bspec = Gesso.load_gguf(BEAN)
bgen(prompt, n; kw...) = Gesso.reference_generate(bean, btensors, prompt; max_new_tokens=n,
    eps=bspec.rms_norm_eps, backend, eos_ids=(IM_END,), kw...)
btk = Gesso.Inference.load_gguf_tokenizer(BEAN)

model, tensors, spec = Gesso.load_gguf(GGUF)
tk = Gesso.Inference.load_gguf_tokenizer(GGUF)
size(tensors.embedding.storage, 1) == size(btensors.embedding.storage, 1) ||
    error("saluki_combo_bench: bean and Saluki vocabularies differ")
gen(prompt, n; kw...) = Gesso.reference_generate(model, tensors, prompt; max_new_tokens=n,
    eps=spec.rms_norm_eps, backend, eos_ids=(IM_END,), kw...)

# JIT warm for the plain step, the bean-only path and every window size.
let p = Gesso.Inference.encode(tk, chat("Hi"))
    gen(p, 4)
    for k in KS, tau in TAUS
        src = BeanSource(bean, btensors, Float64(bspec.rms_norm_eps), backend,
            Gesso.GenerateState(4096), k, tau, 0, 0, 0)
        gen(p, 4; draft=src, max_draft=k)
        combo = ComboSource(Gesso.Speculation.SuffixSource(; min_match=SUFFIX_MIN, max_draft=k), src,
            BACKOFFS[1][1], BACKOFFS[1][2], 0, 0, 0, 0, 0)
        gen(p, 4; draft=combo, max_draft=k)
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
    for k in KS, tau in TAUS
        # bean alone, the reference the combo runs are judged against
        src = BeanSource(bean, btensors, Float64(bspec.rms_norm_eps), backend,
            Gesso.GenerateState(4096), k, tau, 0, 0, 0)
        bp_ms = 1000 * @elapsed bgen(prompt, 1; state=src.state)
        info = Ref{Any}(nothing)
        tw = @elapsed w = gen(prompt, NEW; draft=src, max_draft=k, info)
        w == ids || error("bean k=$k tau=$tau changed the ids on $tag")
        sp = info[].speculation
        drafted_s = tw - t1
        @printf("BEAN %s k=%d tau=%.2f gated=%d windows=%d drafted=%d accepted=%d accept_rate=%.2f bean_ms_per_window=%.1f bean_share=%.2f bean_prefill_ms=%.0f speedup=%.2f\n",
            tag, k, tau, src.gated, sp.windows, sp.drafted, sp.accepted,
            sp.drafted == 0 ? 0.0 : sp.accepted / sp.drafted,
            1e-6 * src.ns / max(sp.windows, 1),
            (src.ns / 1e9) / drafted_s, bp_ms,
            plain_ms / (1000 * drafted_s))
        flush(stdout)
        for (after, skip) in BACKOFFS
            # fresh bean and suffix state per run, so runs are independent
            bsrc = BeanSource(bean, btensors, Float64(bspec.rms_norm_eps), backend,
                Gesso.GenerateState(4096), k, tau, 0, 0, 0)
            bgen(prompt, 1; state=bsrc.state)
            combo = ComboSource(Gesso.Speculation.SuffixSource(; min_match=SUFFIX_MIN, max_draft=k), bsrc,
                after, skip, 0, 0, 0, 0, 0)
            info = Ref{Any}(nothing)
            tw = @elapsed w = gen(prompt, NEW; draft=combo, max_draft=k, info)
            w == ids || error("combo k=$k tau=$tau backoff=$after:$skip changed the ids on $tag")
            sp = info[].speculation
            drafted_s = tw - t1
            @printf("COMBO %s k=%d tau=%.2f min=%d backoff=%d:%d windows=%d drafted=%d accepted=%d accept_rate=%.2f suffix_windows=%d bean_calls=%d skipped=%d gated=%d bean_share=%.2f speedup=%.2f\n",
                tag, k, tau, SUFFIX_MIN, after, skip, sp.windows, sp.drafted, sp.accepted,
                sp.drafted == 0 ? 0.0 : sp.accepted / sp.drafted,
                combo.suffix_windows, combo.bean_calls, combo.skipped, bsrc.gated,
                (bsrc.ns / 1e9) / drafted_s, plain_ms / (1000 * drafted_s))
            flush(stdout)
        end
    end
end
