# MTP draft source for the Saluki scripts. The includer must `using Gesso`.
#
# The head (Gesso.Inference.MTPDraft) reads the target's final-normed rows, so
# the generate call that drafts with it must also pass
#   on_rows=(p, M) -> Gesso.Inference.mtp_rows!(src.m, p, M)
# One source serves one conversation; build a fresh one per run.

mutable struct MTPSource <: Gesso.Speculation.DraftSource
    m::Gesso.Inference.MTPDraft
    max_draft::Int
    p_min::Float64        # top-10 confidence gate; 0 = off
    gated::Int            # propose calls that drafted nothing because token 1 was below p_min
    ns::Int               # total nanoseconds spent inside propose
    calls::Int
end
MTPSource(m, max_draft; p_min=0.0) = MTPSource(m, max_draft, Float64(p_min), 0, 0, 0)

function Gesso.Speculation.propose(src::MTPSource, ctx::Gesso.Speculation.DraftContext; max_tokens::Int)
    k = min(max_tokens, src.max_draft)
    k <= 0 && return Gesso.Speculation.Draft(Int[], :mtp)
    t0 = time_ns()
    out = Gesso.Inference.mtp_draft!(src.m, collect(ctx.ids), k; p_min=src.p_min)
    isempty(out) && (src.gated += 1)
    src.ns += time_ns() - t0
    src.calls += 1
    return Gesso.Speculation.Draft(out, :mtp)
end
Gesso.Speculation.observe!(::MTPSource, ::Gesso.Speculation.Draft, ::Integer) = nothing

mtp_hook(src::MTPSource) = (p, M) -> Gesso.Inference.mtp_rows!(src.m, p, M)
