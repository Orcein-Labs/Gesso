# Speculation — the host-side draft seam for speculative decoding
# (Phase 20).
#
# Owns: the DraftSource interface (the source, the context a source may read,
# the proposal it returns, and the post-verify feedback hook) and the
# host-only draft sources built against it. Sources never call a model and
# never touch a device.
#
# Not owned here: the verify window, state commit and rollback, and the
# decode loop, which live in Inference. Nothing in this module wires a source
# into decoding.

module Speculation

export DraftSource, DraftContext, Draft, propose, observe!

abstract type DraftSource end

# What a source may read. Host-only and immutable for the call.
struct DraftContext
    ids::AbstractVector{Int}   # every token so far: prompt + accepted output
    detok::Any                      # id -> text, for sources that reason in text
    vocab_size::Int
end

struct Draft
    ids::Vector{Int}           # proposed continuation (may be empty)
    origin::Symbol             # e.g. :template, :mse, :suffix (for receipts)
end

# Return at most `max_tokens` ids. An empty Draft means "no opinion".
function propose end

# Feedback after verification: how many of `d.ids` were accepted.
function observe! end

include("HostSources.jl")

end # module Speculation
