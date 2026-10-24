# DFlash2 draft source for the Saluki scripts. The includer must `using Gesso`.
#
# The drafter (Gesso.Inference.DFlashDraft) reads the target's residual stream
# after a few blocks, so the generate call that drafts with it must also pass
#   taps=dflash_taps(src), on_taps=dflash_hook(src)
# One source serves one conversation; build a fresh one per run.

mutable struct DFlashSource <: Gesso.Speculation.DraftSource
    m::Gesso.Inference.DFlashDraft
    max_draft::Int
    p_min::Float64        # lattice-walk confidence gate; 0 = off
    gated::Int            # propose calls that drafted nothing
    ns::Int               # total nanoseconds spent inside propose
    calls::Int
end
DFlashSource(m, max_draft; p_min=0.0) = DFlashSource(m, max_draft, Float64(p_min), 0, 0, 0)

function Gesso.Speculation.propose(
    src::DFlashSource,
    ctx::Gesso.Speculation.DraftContext;
    max_tokens::Int,
)
    k = min(max_tokens, src.max_draft)
    k <= 0 && return Gesso.Speculation.Draft(Int[], :dflash)
    t0 = time_ns()
    out = Gesso.Inference.dflash_draft!(src.m, collect(ctx.ids), k; p_min=src.p_min)
    isempty(out) && (src.gated += 1)
    src.ns += time_ns() - t0
    src.calls += 1
    return Gesso.Speculation.Draft(out, :dflash)
end
Gesso.Speculation.observe!(::DFlashSource, ::Gesso.Speculation.Draft, ::Integer) = nothing

dflash_taps(src::DFlashSource) = src.m.target_layers
dflash_hook(src::DFlashSource) = (p, X) -> Gesso.Inference.dflash_rows!(src.m, p, X)
