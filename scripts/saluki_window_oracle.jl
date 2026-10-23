# Saluki 27B verify-window oracle on the CPU backend (docs/design/VERIFY_WINDOW.md,
# "Acceptance oracle" and "Cost model for this box").
#
#   GESSO_QWEN38_27B_GGUF=/path/to/Underdog-Saluki-27B-1.0-IQ2-mix.gguf \
#   GESSO_BEAN_GGUF=/path/to/Qwen3.5-0.8B-BF16.gguf \
#   GESSO_GGML_DIR=/path/to/llama.cpp/build/bin \
#   julia --project=benchmark -t 16 scripts/saluki_window_oracle.jl
#
# Optional env: GESSO_ORACLE_NEW (new tokens per prompt, default 96),
# GESSO_ORACLE_T (window sizes, default "2,4,8").
#
# Per prompt (chat template, thinking off):
#   1. Plain greedy decode: the transcript and the T = 1 time.
#   2. For each T, the same decode with a source that always proposes the
#      transcript's continuation. The ids must equal the transcript. Time per
#      round over time per plain step is W(T) at full acceptance.
#   3. One teacher-forced bean prefill over the transcript. Its greedy id at
#      each position gives the exact accepted length of a chained bean draft,
#      so the tokens per round for each T follow without running the window.
# The bean decode step time is measured once, so the predicted speedup counts
# the drafter's cost.

using Gesso
using Printf

const GGUF = get(ENV, "GESSO_QWEN38_27B_GGUF", "")
isfile(GGUF) || error("saluki_window_oracle: set GESSO_QWEN38_27B_GGUF")
const BEAN = get(ENV, "GESSO_BEAN_GGUF", "")
isfile(BEAN) || error("saluki_window_oracle: set GESSO_BEAN_GGUF")
const NEW = parse(Int, get(ENV, "GESSO_ORACLE_NEW", "96"))
const TS = [parse(Int, s) for s in split(get(ENV, "GESSO_ORACLE_T", "2,4,8"), ',')]
const IM_END = 248046

const PROMPTS = [
    "json" => "Return only a JSON object for a tool call to get_weather with arguments city set to Toronto, units set to metric and days set to 3. No prose.",
    "code" => "Rewrite this function so it skips empty lines, and print the whole file back:\n\n```python\ndef count_words(path):\n    total = 0\n    with open(path) as f:\n        for line in f:\n            total += len(line.split())\n    return total\n\n\ndef main():\n    import sys\n    print(count_words(sys.argv[1]))\n\n\nif __name__ == \"__main__\":\n    main()\n```",
    "shell" => "Summarize this command output in two sentences:\n\n```\n\$ df -h\nFilesystem      Size  Used Avail Use% Mounted on\n/dev/nvme0n1p2  476G  431G   41G  92% /\n/dev/sda1       3.6T  2.9T  700G  81% /mnt/d\ntmpfs            16G  1.2G   15G   8% /tmp\n```",
    "math" => "Compute 17 * 23 + 144 / 12 - 9 step by step, then give the final number.",
    "prose" => "Write a short paragraph about a lighthouse keeper who finds a letter in a bottle.",
]

chat(msg) = "<|im_start|>user\n" * msg * "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Proposes the next `max_tokens` ids of a known greedy transcript.
struct TranscriptSource <: Gesso.Speculation.DraftSource
    ids::Vector{Int}
end
function Gesso.Speculation.propose(s::TranscriptSource, ctx::Gesso.Speculation.DraftContext; max_tokens::Int)
    i = length(ctx.ids)
    stop = min(length(s.ids), i + max_tokens)
    return Gesso.Speculation.Draft(s.ids[i+1:stop], :transcript)
end
Gesso.Speculation.observe!(::TranscriptSource, ::Gesso.Speculation.Draft, ::Integer) = nothing

backend = Gesso.CPUBackend()
model, tensors, spec = Gesso.load_gguf(GGUF)
tk = Gesso.Inference.load_gguf_tokenizer(GGUF)
gen(prompt, n; kw...) = Gesso.reference_generate(model, tensors, prompt; max_new_tokens=n,
    eps=spec.rms_norm_eps, backend, eos_ids=(IM_END,), kw...)

# JIT warm for the plain step and every window size.
let p = Gesso.Inference.encode(tk, chat("Hi"))
    base = gen(p, 4)
    for T in TS
        gen(p, 4; draft=TranscriptSource(base), max_draft=T - 1)
    end
end

transcripts = Dict{String,Tuple{Int,Vector{Int}}}()
step_ms = Dict{String,Float64}()
round_ms = Dict{Tuple{String,Int},Float64}()
for (tag, msg) in PROMPTS
    prompt = Gesso.Inference.encode(tk, chat(msg))
    P = length(prompt)
    # Prefill and the first token are shared by every run, so time them once
    # and subtract: the rest is decode.
    t1 = @elapsed gen(prompt, 1)
    tN = @elapsed ids = gen(prompt, NEW)
    n_dec = length(ids) - P - 1
    step_ms[tag] = 1000 * (tN - t1) / n_dec
    transcripts[tag] = (P, ids)
    @printf("PLAIN %s prompt=%d new=%d step_ms=%.1f\n", tag, P, length(ids) - P, step_ms[tag])
    println("TEXT ", tag, " ", repr(Gesso.Inference.decode(tk, ids[P+1:end])))
    for T in TS
        info = Ref{Any}(nothing)
        tw = @elapsed w = gen(prompt, NEW; draft=TranscriptSource(ids), max_draft=T - 1, info)
        w == ids || error("window T=$T changed the ids on $tag")
        sp = info[].speculation
        # every decode step after the first token is either a window round or a
        # plain step (a window that would pass the cap falls back to plain). A
        # window that emits the stop token has decoded its row too, which the
        # plain run never does, so the count can come out one low: floor it.
        plain = max(0, n_dec - sp.windows - sp.accepted)
        ms = (1000 * (tw - t1) - plain * step_ms[tag]) / sp.windows
        round_ms[(tag, T)] = ms
        @printf("WINDOW %s T=%d windows=%d accepted=%d plain=%d round_ms=%.1f W=%.3f ids_ok=true\n",
            tag, T, sp.windows, sp.accepted, plain, ms, ms / step_ms[tag])
    end
end

# Free the 27B before the bean loads.
model = tensors = nothing
GC.gc()

bean, btensors, bspec = Gesso.load_gguf(BEAN)
bgen(prompt, n) = Gesso.reference_generate(bean, btensors, prompt; max_new_tokens=n,
    eps=bspec.rms_norm_eps, backend, eos_ids=(IM_END,))
let p = transcripts["prose"][2][1:transcripts["prose"][1]]
    bgen(p, 4)
    t1 = @elapsed bgen(p, 1)
    tN = @elapsed o = bgen(p, 33)
    global bean_ms = 1000 * (tN - t1) / (length(o) - length(p) - 1)
end
@printf("BEAN step_ms=%.1f\n", bean_ms)

println("== bean acceptance (exact, from teacher-forced greedy) and predicted speed ==")
for (tag, _) in PROMPTS
    P, ids = transcripts[tag]
    logits = Gesso.reference_prefill(bean, btensors, ids; eps=bspec.rms_norm_eps, backend)
    # m[j]: the bean's greedy id after ids[1:j] equals ids[j+1].
    m = [argmax(view(logits, :, j)) - 1 == ids[j+1] for j in 1:length(ids)-1]
    top1 = count(m[P+1:end]) / length(m[P+1:end])
    for T in TS
        # A round starts after the anchor ids[L] is known: the bean drafts up to
        # T - 1 ids, the window accepts the matching prefix n, and the round
        # yields n + 1 ids (the next anchor is the correction).
        L, rounds, drafted = P + 1, 0, 0
        while L < length(ids)
            k = min(T - 1, length(ids) - L)
            n = 0
            while n < k && m[L+n]
                n += 1
            end
            # the bean cannot tell where it goes wrong, so it drafts all k ids
            drafted += k
            rounds += 1
            L += n + 1
        end
        gen_ids = length(ids) - P - 1
        per_round = gen_ids / rounds
        cost = rounds * round_ms[(tag, T)] + drafted * bean_ms
        speedup = gen_ids * step_ms[tag] / cost
        @printf("ORACLE %s T=%d top1=%.3f tok_per_round=%.2f bean_ms_per_round=%.1f speedup=%.2f\n",
            tag, T, top1, per_round, drafted * bean_ms / rounds, speedup)
    end
end
