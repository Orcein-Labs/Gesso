# Bean draft source for the Saluki scripts. The includer must `using Gesso` and define IM_END.

# Greedy id that also keeps the top softmax probability in s.p_top, for the gate.
mutable struct ConfSampler
    p_top::Float64
end
function Gesso.Inference._sample_id(s::ConfSampler, logits_row)
    row = Gesso.Inference._require_finite_logits(Array(logits_row))
    hi, i = findmax(row)
    total = 0.0
    @inbounds for x in row
        total += exp(Float64(x) - Float64(hi))
    end
    s.p_top = 1 / total
    return i - 1   # 0-based id, same as Gesso's _greedy_id
end

# Drafts with the bean. The state persists across calls (see the header).
mutable struct BeanSource <: Gesso.Speculation.DraftSource
    model; tensors; eps::Float64; backend
    state                 # a GenerateState owned by this source, kept across calls
    max_draft::Int
    tau::Float64          # confidence gate; 0 = off
    gated::Int            # propose calls that drafted nothing because token 1 was below tau
    ns::Int               # total nanoseconds spent inside propose
    calls::Int
end
function Gesso.Speculation.propose(src::BeanSource, ctx::Gesso.Speculation.DraftContext; max_tokens::Int)
    k = min(max_tokens, src.max_draft)
    k <= 0 && return Gesso.Speculation.Draft(Int[], :bean)
    t0 = time_ns()
    if src.tau > 0
        # stop at the first token below tau and drop it from the draft
        cs = ConfSampler(1.0)
        low = Ref(false)
        ids = Gesso.reference_generate(src.model, src.tensors, ctx.ids; max_new_tokens=k,
            state=src.state, eps=src.eps, backend=src.backend, eos_ids=(IM_END,),
            sampler=cs, on_token=_ -> (low[] = cs.p_top < src.tau; !low[]))
        drop = low[] ? 1 : 0
        out = ids[length(ctx.ids)+1:end-drop]
        isempty(out) && (src.gated += 1)
    else
        ids = Gesso.reference_generate(src.model, src.tensors, ctx.ids; max_new_tokens=k,
            state=src.state, eps=src.eps, backend=src.backend, eos_ids=(IM_END,))
        out = ids[length(ctx.ids)+1:end]
    end
    src.ns += time_ns() - t0
    src.calls += 1
    Gesso.Speculation.Draft(out, :bean)
end
Gesso.Speculation.observe!(::BeanSource, ::Gesso.Speculation.Draft, ::Integer) = nothing
