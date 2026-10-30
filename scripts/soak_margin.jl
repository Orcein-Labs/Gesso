# Offline margin check for the replay divergences that scripts/soak.jl records.
#
#   julia -t auto --project scripts/soak_margin.jl --gguf <path> --events soak-events.jsonl
#         [--bar 0.25] [--n-ctx 2048]
#
# A divergence between a reply and its replay is acceptable only when the model
# genuinely could not tell the two tokens apart. For each replay event with
# ok=false this rebuilds the prompt the server rendered, runs one fresh CPU
# forward pass over prompt + the part of the original reply the two replies
# share, and reads the logits at the first token where they differ:
#
#   near-tie   one of the two tokens is the pass's top-1 and the two logits are
#              closer than the bar
#   defect     neither token is top-1, or the gap is at least the bar
#   undetermined  the events cannot place the divergence (see `divergence_point`)
#
# When one reply is a prefix of the other, the shorter one stopped on an
# end-of-turn token where the longer one went on; that end-of-turn token is its
# candidate. The bar is fixed before any run and is a command-line default, not
# a knob to tune against results. Exit 0 only when every divergence is a
# near-tie: an undetermined one has not been shown to be a near-tie, so it fails
# the run like a defect.

module SoakMargin

using Gesso
using JSON

const BAR = 0.25

"""First index where `a` and `b` differ, or `nothing` when one is a prefix of the other."""
function first_diff_index(a::AbstractVector, b::AbstractVector)
    for i in 1:min(length(a), length(b))
        a[i] == b[i] || return i
    end
    return nothing
end

"""
Verdict for one divergence from the logits of the two candidate tokens and the
pass's top-1 id. Any missing input is undetermined.
"""
function verdict(id_orig, id_replay, logit_orig, logit_replay, top1, bar::Real)
    any(isnothing, (id_orig, id_replay, logit_orig, logit_replay, top1)) && return "undetermined"
    (top1 == id_orig || top1 == id_replay) || return "defect"
    return abs(logit_orig - logit_replay) < bar ? "near-tie" : "defect"
end

"""
Where two token lists part: `(k, stopped)` with `stopped` = :none, :orig or
:replay naming the reply whose token `k` is an end-of-turn token, or a string
saying why the events cannot place it. A reply that ends early counts as having
stopped only when its finish reason says so.
"""
function divergence_point(orig_ids, replay_ids, orig_finish, replay_finish)
    k = first_diff_index(orig_ids, replay_ids)
    k === nothing || return (k, :none)
    length(orig_ids) == length(replay_ids) && return "the two replies differ as text but tokenize to the same ids"
    shorter, finish = length(orig_ids) < length(replay_ids) ? (:orig, orig_finish) : (:replay, replay_finish)
    finish == "stop" || return "the shorter reply ended with finish reason $(repr(finish)), not on an end-of-turn token"
    return (min(length(orig_ids), length(replay_ids)) + 1, shorter)
end

# Captures the final-position logits of the pass; the sampler hook of
# reference_generate is the one place those rows are visible.
mutable struct LogitProbe
    row::Vector{Float64}
end
LogitProbe() = LogitProbe(Float64[])

function Gesso.Inference._sample_id(p::LogitProbe, row)
    p.row = Float64.(Array(row))
    return argmax(p.row) - 1
end

function final_logits(model, tensors, spec, eos, ids; backend)
    probe = LogitProbe()
    Gesso.Inference.reference_generate(model, tensors, ids; backend=backend, max_new_tokens=1,
        sampler=probe, eos_ids=eos, eps=spec.rms_norm_eps)
    isempty(probe.row) && error("the forward pass produced no logits")
    return probe.row
end

function parse_args(args)
    opts = Dict{String, String}("--bar" => string(BAR), "--n-ctx" => "2048")
    i = 1
    while i <= length(args)
        a = args[i]
        (a in ("--gguf", "--events", "--bar", "--n-ctx") && i < length(args)) ||
            throw(ArgumentError("bad argument $a; usage: --gguf PATH --events FILE [--bar 0.25] [--n-ctx N]"))
        opts[a] = args[i+1]
        i += 2
    end
    for req in ("--gguf", "--events")
        haskey(opts, req) || throw(ArgumentError("$req is required"))
    end
    return opts
end

function diverged_events(path)
    out = Any[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        e = JSON.parse(line)
        e isa AbstractDict && get(e, "event", nothing) == "replay" && get(e, "ok", true) === false &&
            push!(out, e)
    end
    return out
end

piece(chat, id) = repr(chat.detokenize([id]))

function judge(e, chat, model, tensors, spec, eos, bar, backend)
    turn = get(e, "turn", "?")
    haskey(e, "messages") || return (turn=turn, verdict="undetermined", note="event has no messages")
    msgs = Any[Dict{String, Any}(string(k) => v for (k, v) in m) for m in e["messages"]]
    thinking = get(e, "thinking", false) === true
    prompt = chat.tokenize(chat.render(msgs, Any[], thinking))
    oids, rids = chat.tokenize(string(e["original"])), chat.tokenize(string(e["replay"]))
    point = divergence_point(oids, rids, get(e, "original_finish", nothing), get(e, "replay_finish", nothing))
    point isa AbstractString && return (turn=turn, verdict="undetermined", note=point)
    k, stopped = point
    stopped === :none || !isempty(eos) || return (turn=turn, k=k, verdict="undetermined", note="the tokenizer names no end-of-turn token")
    row = final_logits(model, tensors, spec, eos, vcat(prompt, oids[1:k-1]); backend=backend)
    # the stopping reply's candidate is whichever end-of-turn id the pass ranks highest
    stop_id = eos[argmax([id < length(row) ? row[id+1] : -Inf for id in eos])]
    io = stopped === :orig ? stop_id : oids[k]
    ir = stopped === :replay ? stop_id : rids[k]
    (io < length(row) && ir < length(row)) || return (turn=turn, k=k, verdict="undetermined", note="token id outside the logit row")
    top1 = argmax(row) - 1
    lo, lr = row[io+1], row[ir+1]
    return (turn=turn, k=k, orig=(io, piece(chat, io)), replay=(ir, piece(chat, ir)),
        logits=(lo, lr), gap=abs(lo - lr), top1=top1,
        verdict=verdict(io, ir, lo, lr, top1, bar), note="")
end

"""The run passes only when every divergence is a near-tie."""
passes(results) = all(r -> r.verdict == "near-tie", results)

function report_line(r)
    r.verdict == "undetermined" && return "turn $(r.turn)  k=$(get(r, :k, nothing))  undetermined: $(r.note)"
    return "turn $(r.turn)  k=$(r.k)  original id=$(r.orig[1]) $(r.orig[2]) logit=$(round(r.logits[1]; digits=4))" *
           "  replay id=$(r.replay[1]) $(r.replay[2]) logit=$(round(r.logits[2]; digits=4))" *
           "  gap=$(round(r.gap; digits=4))  top1=$(r.top1)  $(r.verdict)"
end

function main(args)
    opts = try
        parse_args(args)
    catch e
        e isa ArgumentError || rethrow()
        println(stderr, "soak_margin: ", e.msg)
        return 2
    end
    bar = parse(Float64, opts["--bar"])
    events = diverged_events(opts["--events"])
    if isempty(events)
        println("no diverged replay events; nothing to judge")
        println("PASS")
        return 0
    end
    path = opts["--gguf"]
    isfile(path) || (println(stderr, "soak_margin: GGUF not found: $path"); return 2)
    backend = Gesso.CPUBackend()
    results = Gesso.on_engine_thread() do
        model, tensors, spec = Gesso.load_gguf(path)
        tk = Gesso.Inference.load_gguf_tokenizer(path)
        chat = gguf_chat_model(model, tensors, spec, tk; backend=backend,
            n_ctx=parse(Int, opts["--n-ctx"]), model_id="soak-margin")
        eos = Gesso.Serve._engine_eos_ids(tk)
        [judge(e, chat, model, tensors, spec, eos, bar, backend) for e in events]
    end
    foreach(r -> println(report_line(r)), results)
    defects = count(r -> r.verdict == "defect", results)
    ties = count(r -> r.verdict == "near-tie", results)
    und = count(r -> r.verdict == "undetermined", results)
    println("$(length(results)) divergences: $ties near-tie, $defects defect, $und undetermined (bar $bar)")
    ok = passes(results)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    exit(SoakMargin.main(ARGS))
end
