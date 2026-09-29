# Receipts (Gesso_Stack.md §XLII; engineering receipts §LXXII).
#
# Law: significant actions are auditable. Every receipt answers what happened,
# in what context, at what cost, and with what result — well enough for
# debugging, deterministic replay where possible, benchmarking, safety,
# performance analysis, and swarm coordination.
#
# Phase 0/BONES scope: the record TYPE and the sink INTERFACE only. Field
# vocabulary below is transcribed verbatim from §XLII. Serialization (JSON3
# etc.) arrives with the phase that first persists receipts; until then the
# in-memory sink suffices. No dependencies added — the dependency law holds.

export Receipt, ReceiptSink, InMemorySink, emit!, next_receipt_id, new_receipt
# RECEIPT_SCHEMA_VERSION is owned and exported by versions.jl — re-exporting
# it here too would make `names(Gesso)` ambiguous about ownership.

using Dates

"""
    Receipt

One auditable significant action (§XLII fields, verbatim vocabulary):

    agent · task · model · materialization · inference request ·
    tool request · tool result · parent dependency · timing ·
    token usage · memory usage · failure · retry · cancellation ·
    output digest

Fields are intentionally permissive (`Any`) at this stage: they are filled
by the phases that own the corresponding concepts. `nothing` means "not
applicable to this action." What is NOT here yet: serialization, signing,
retention policy — those arrive with their owning phases and bump
RECEIPT_SCHEMA_VERSION.
"""
Base.@kwdef struct Receipt
    id::UInt64
    timestamp::DateTime
    schema_version::VersionNumber = RECEIPT_SCHEMA_VERSION
    # --- §XLII field vocabulary -------------------------------------------
    agent::Any = nothing
    task::Any = nothing
    model::Any = nothing
    materialization::Any = nothing
    inference_request::Any = nothing
    tool_request::Any = nothing
    tool_result::Any = nothing
    parent_dependency::Any = nothing          # receipt id of the parent action
    timing::Any = nothing                     # ns-precision timing record
    token_usage::Any = nothing
    memory_usage::Any = nothing
    failure::Any = nothing                    # GessoError or taxonomy record
    retry::Any = nothing
    cancellation::Any = nothing
    output_digest::Any = nothing
    # --- open context (structured, never prose-only) -----------------------
    context::Dict{Symbol, Any} = Dict{Symbol, Any}()
end

"""
    ReceiptSink

Abstract sink for receipts. `emit!` must never throw on receipt delivery —
a telemetry failure must not fail the action it is auditing; it logs instead.
"""
abstract type ReceiptSink end

"""
    InMemorySink([capacity])

Bounded ring of receipts kept in memory. `emit!` drops the oldest receipt
past capacity (and logs that it did — a dropped receipt is itself notable).

Thread-safe: `emit!` holds a lock for the whole push/drop/log sequence, so
concurrent emitters cannot interleave ring updates or the `dropped` counter.
(receipt ids are atomic and independent of this lock; ids stay strictly
monotonic across threads.)
"""
mutable struct InMemorySink <: ReceiptSink
    const buf::Vector{Receipt}
    const capacity::Int
    const lock::ReentrantLock
    dropped::UInt64
    function InMemorySink(capacity::Int=10_000)
        capacity > 0 || throw(ArgumentError("capacity must be positive"))
        new(Vector{Receipt}(undef, 0), capacity, ReentrantLock(), UInt64(0))
    end
end

const _RECEIPT_COUNTER = Threads.Atomic{UInt64}(0)

"Monotonic receipt id counter (process-local; global ids are a later concern)."
next_receipt_id() = Threads.atomic_add!(_RECEIPT_COUNTER, UInt64(1))

"""
    emit!(sink, receipt) -> Receipt

Deliver a receipt to a sink. Returns the receipt. Never throws — a telemetry
failure must not fail the action it is auditing; problems are logged instead
(and if logging itself fails, that is swallowed too).
"""
function emit!(sink::InMemorySink, r::Receipt)
    lock(sink.lock) do
        try
            # Order matters: the incoming receipt is pushed FIRST. Overflow
            # bookkeeping (dropping the oldest) happens after it is safely
            # stored, and logging happens last — a log failure must never cost
            # us the receipt we are currently auditing.
            push!(sink.buf, r)
            while length(sink.buf) > sink.capacity
                deleteat!(sink.buf, 1)
                sink.dropped += UInt64(1)
            end
            sink.dropped == 1 &&
                glog(Log.LOG_WARN, :receipt_sink_overflow; capacity=sink.capacity)
        catch err
            try
                glog(Log.LOG_ERROR, :receipt_emit_failed; error=repr(err))
            catch
                # nothing more we can do; auditing must not take the host down
            end
        end
    end
    return r
end

"""
    new_receipt(; kw...) -> Receipt

Convenience constructor: a fresh receipt stamped with a fresh id and current
UTC time. Keyword arguments fill the §XLII fields.
"""
new_receipt(; kw...) = Receipt(id=next_receipt_id(), timestamp=now(UTC); kw...)
