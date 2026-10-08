# Ordinary fixed round-robin scheduling of independently owned Sessions.
module Runtime
using Base.Threads: Atomic
using ..Inference:
    _safe_emit!,
    _output_digest,
    _engine_failure,
    Session,
    prefill!,
    decode!,
    _session_reset!,
    kv_bytes,
    page_count
using ..Gesso:
    GessoException,
    gesso_error,
    ERR_INVALID_PLAN,
    ERR_RUNTIME,
    new_receipt,
    emit!,
    backend_name
export BatchRequest, BatchResult, run_batch, cancel!

struct BatchRequest
    session::Session
    prompt::Vector{Int}
    max_new_tokens::Int
    cancelled::Atomic{Bool}
end
function BatchRequest(session::Session, prompt::AbstractVector{Int}; max_new_tokens::Int=8)
    max_new_tokens>=0 || throw(
        gesso_error(ERR_INVALID_PLAN, "BatchRequest: token budget must be nonnegative"),
    )
    return BatchRequest(session, collect(prompt), max_new_tokens, Atomic{Bool}(false))
end
cancel!(r::BatchRequest) = (r.cancelled[]=true; r)

struct BatchResult
    ids::Vector{Int}
    status::Symbol
    error::Union{Nothing, Exception}
end

function _terminal!(request, ids, status, error, prefilled)
    s=request.session
    status===:failed && (s.ready=false)
    n=prefilled ? length(request.prompt) : 0
    _safe_emit!(
        s.sink,
        new_receipt(;
            task=:batch_request,
            inference_request=(;
                backend=backend_name(s.backend),
                max_new_tokens=request.max_new_tokens,
            ),
            token_usage=(;
                prompt_tokens=n,
                new_tokens=length(ids)-n,
                total_tokens=length(ids),
            ),
            memory_usage=(;
                kv_bytes=kv_bytes(s.mgr),
                page_count=page_count(s.mgr),
                kv_len=s.seqlen,
            ),
            output_digest=_output_digest(ids),
            cancellation=status===:cancelled ?
                         (; observed=true, committed_tokens=length(ids)) : nothing,
            failure=error,
            context=Dict{Symbol, Any}(:status=>status),
        ),
    )
    return BatchResult(copy(ids), status, error)
end
_batch_error(e) = _engine_failure(e)

"""
    run_batch(requests; on_token=nothing) -> Vector{BatchResult}

Own each Session exclusively until return. Prefill once, then advance one
new token per active request per round, in input order. Cancellation is
observed before prefill and between tokens. `on_token(index,id)` may cancel
requests; it must not mutate their Sessions. Failure is terminal for that
request and other requests continue. Results include committed prompt IDs
and new IDs; cancellation before prefill returns an empty sequence.
Independent batches may run on independent Sessions. No tensor fusion or
continuous/adaptive batching is implemented by this ordinary mechanism.
"""
function run_batch(requests::AbstractVector{BatchRequest}; on_token=nothing)
    seen=IdDict{Session, Nothing}()
    for r in requests
        haskey(seen, r.session) &&
            throw(gesso_error(ERR_INVALID_PLAN, "run_batch: duplicate Session ownership"))
        seen[r.session]=nothing
    end
    leased=Session[]
    try
        for r in requests
            trylock(r.session.run_lock) || throw(
                gesso_error(
                    ERR_INVALID_PLAN,
                    "run_batch: Session is owned by another task",
                ),
            )
            push!(leased, r.session)
            r.session.busy && throw(
                gesso_error(
                    ERR_INVALID_PLAN,
                    "run_batch: Session operation already active",
                ),
            )
        end
        return _run_owned(requests, on_token)
    finally
        for s in reverse(leased)
            unlock(s.run_lock)
        end
    end
end

function _run_owned(requests, on_token)
    n=length(requests)
    ids=[Int[] for _ in 1:n]
    results=Vector{BatchResult}(undef, n)
    active=falses(n)
    prefilled=falses(n)
    for (i, r) in enumerate(requests)
        try
            _session_reset!(r.session)
            if r.cancelled[]
                results[i]=_terminal!(r, ids[i], :cancelled, nothing, false)
            else
                prefill!(r.session, r.prompt)
                append!(ids[i], r.prompt)
                prefilled[i]=true
                if r.max_new_tokens==0
                    results[i]=_terminal!(r, ids[i], :complete, nothing, true)
                else
                    active[i]=true
                end
            end
        catch e
            results[i]=_terminal!(r, ids[i], :failed, _batch_error(e), prefilled[i])
        end
    end
    while any(active)
        for (i, r) in enumerate(requests)
            active[i] || continue
            try
                if r.cancelled[]
                    results[i]=_terminal!(r, ids[i], :cancelled, nothing, prefilled[i])
                    active[i]=false
                    continue
                end
                id=decode!(r.session)
                push!(ids[i], id)
                on_token===nothing || on_token(i, id)
                status=r.cancelled[] ? :cancelled :
                       id==r.session.eos_token_id ? :eos :
                       length(ids[i])-length(r.prompt)>=r.max_new_tokens ? :complete :
                       nothing
                if status!==nothing
                    results[i]=_terminal!(r, ids[i], status, nothing, prefilled[i])
                    active[i]=false
                end
            catch e
                results[i]=_terminal!(r, ids[i], :failed, _batch_error(e), prefilled[i])
                active[i]=false
            end
        end
    end
    return results
end
end
