# Routes and the chat completion handler.

function route(srv::Server, req::HTTPRequest, io::IO, keepalive::Bool)::Bool
    path = String(first(split(req.target, '?'; limit=2)))
    if path == "/health"
        _require_method(req, "GET")
        write_json(io, 200, Dict("status" => "ok"); keepalive=keepalive)
        return keepalive
    elseif path == "/props"
        _require_method(req, "GET")
        write_json(
            io,
            200,
            Dict(
                "default_generation_settings" => Dict("n_ctx" => srv.model.n_ctx),
                "total_slots" => 1,
                "model_alias" => srv.model.model_id,
            );
            keepalive=keepalive,
        )
        return keepalive
    elseif path == "/v1/models"
        _require_method(req, "GET")
        write_json(
            io,
            200,
            Dict(
                "object" => "list",
                "data" => [
                    Dict(
                        "id" => srv.model.model_id,
                        "object" => "model",
                        "created" => 0,
                        "owned_by" => "gesso",
                        "meta" => Dict("n_ctx" => srv.model.n_ctx),
                    ),
                ],
            );
            keepalive=keepalive,
        )
        return keepalive
    elseif path == "/v1/chat/completions"
        _require_method(req, "POST")
        return handle_chat(srv, req, io, keepalive)
    end
    throw(HTTPError(404, "no route for $path"))
end

function _require_method(req::HTTPRequest, method::String)
    req.method == method ||
        throw(HTTPError(405, "$(req.method) not allowed here, use $method"))
    return nothing
end

# The output parser of the model's chat family.
const OutputParser = Union{QwenXMLParser, Gemma4Parser}

# One parser per request, chosen by the family the engine renders.
function _new_parser(model::ChatModel, prompt::AbstractString, tools, stop)::OutputParser
    if model.family === :gemma4
        return Gemma4Parser(;
            start_in_reasoning=gemma4_opens_reasoning(prompt),
            stop=stop,
        )
    end
    return QwenXMLParser(;
        tools=tools,
        start_in_reasoning=opens_reasoning(prompt),
        stop=stop,
    )
end

# State of one chat request. `started` is set once a stream's headers are out:
# after that an error can no longer become a 500, so the connection is closed.
mutable struct Turn
    model::ChatModel
    parser::OutputParser
    io::IO
    stream::Bool
    include_usage::Bool
    keepalive::Bool
    cancelled::Ref{Bool}
    started::Ref{Bool}
    ids::Vector{Int}
    prefix_at::Int                  # ids before the decode window (see _sync_text!)
    read_at::Int                    # ids whose text the parser has already had
    id::String
    created::Int
    reasoning::Vector{String}
    content::Vector{String}
    calls::Vector{ToolCallDelta}
end

function handle_chat(srv::Server, req::HTTPRequest, io::IO, keepalive::Bool)::Bool
    body = _parse_body(req.body)
    model = srv.model

    messages = get(body, "messages", nothing)
    (messages isa AbstractVector && !isempty(messages)) ||
        throw(HTTPError(400, "messages must be a non-empty array"))
    tools = get(body, "tools", nothing)
    tools = tools === nothing ? Any[] : tools
    tools isa AbstractVector || throw(HTTPError(400, "tools must be an array"))
    (isempty(tools) || model.accepts_tools) || throw(
        HTTPError(
            400,
            "tools are not supported by this model: its chat format has no tool-call parser";
            code="tools_unsupported",
            param="tools",
        ),
    )
    stream = get(body, "stream", false)
    stream isa Bool || throw(HTTPError(400, "stream must be a boolean"))
    include_usage = stream && _flag(get(body, "stream_options", nothing), "include_usage")
    stop = _stop_list(get(body, "stop", nothing))
    thinking = _thinking(body, model.thinking_default)
    requested = _max_tokens(body)
    sampling = _sampling(body)

    msgs = _normalize_messages(messages)
    prompt = model.render(msgs, collect(Any, tools), thinking)
    prompt_ids = model.tokenize(prompt)
    boundary = model.prefix_len(msgs, collect(Any, tools), thinking, prompt_ids)
    n = length(prompt_ids)
    n >= model.n_ctx && throw(
        HTTPError(
            400,
            "This model's maximum context length is $(model.n_ctx) tokens. However, your messages resulted in $n tokens. Please reduce the length of the messages.";
            code="context_length_exceeded",
            param="messages",
        ),
    )
    room = model.n_ctx - n
    max_new = requested === nothing ? room : min(requested, room)

    # A client that goes away is seen as EOF on its socket. The watcher reads
    # no bytes, so a pipelined request is still there for the next read.
    cancelled = Ref(false)
    @async try
        if eof(io)
            cancelled[] = true
            _wake_waiters(srv.gate)
        end
    catch
    end

    turn = Turn(
        model,
        _new_parser(model, prompt, collect(Any, tools), stop),
        io,
        stream,
        include_usage,
        keepalive,
        cancelled,
        Ref(false),
        Int[],
        0,
        0,
        _completion_id(),
        round(Int, time()),
        String[],
        String[],
        ToolCallDelta[],
    )

    slot = _acquire_slot!(srv.gate, cancelled)
    slot === :full && return _refuse_busy(io, keepalive, "the generation queue is full")
    slot === :timeout &&
        return _refuse_busy(io, false, "waited too long for the generation queue")
    slot === :cancelled && return false

    gen = try
        _run_turn!(turn, prompt_ids, max_new, sampling, boundary)
    catch e
        if turn.started[]
            @error "generation failed after the stream started" exception=(e, catch_backtrace())
            return false
        end
        rethrow()
    finally
        _release_slot!(srv.gate)
    end
    (gen === nothing || turn.cancelled[]) && return false

    _sync_text!(turn; final=true)
    _deliver!(turn, finish!(turn.parser))
    generated = length(gen.generated)
    reason = _finish_reason(turn, generated, max_new)
    usage = _usage(n, generated, gen.cached)

    if turn.stream
        _write_event!(turn, _chunk(turn, Dict{String, Any}(), reason))
        if include_usage
            _write_event!(
                turn,
                Dict(
                    "id" => turn.id,
                    "object" => "chat.completion.chunk",
                    "created" => turn.created,
                    "model" => model.model_id,
                    "choices" => Any[],
                    "usage" => usage,
                ),
            )
        end
        turn.cancelled[] && return false
        try
            write_chunk(io, "data: [DONE]\n\n")
            end_event_stream(io)
        catch e
            e isa Base.IOError || e isa EOFError || rethrow()
            return false
        end
        return keepalive
    end

    content = join(turn.content)
    message = Dict{String, Any}("role" => "assistant")
    message["content"] = (isempty(content) && !isempty(turn.calls)) ? nothing : content
    isempty(turn.reasoning) || (message["reasoning_content"] = join(turn.reasoning))
    isempty(turn.calls) || (message["tool_calls"] = [_call_object(c) for c in turn.calls])
    write_json(
        io,
        200,
        Dict(
            "id" => turn.id,
            "object" => "chat.completion",
            "created" => turn.created,
            "model" => model.model_id,
            "choices" =>
                [Dict("index" => 0, "message" => message, "finish_reason" => reason)],
            "usage" => usage,
        );
        keepalive=keepalive,
    )
    return keepalive
end

# A 503 for a chat request that could not get the engine. Retry-After tells the
# client to come back once the running generation has had time to finish.
function _refuse_busy(io::IO, keepalive::Bool, reason::AbstractString)::Bool
    write_error(
        io,
        503,
        "$reason, retry later";
        keepalive=keepalive,
        headers=["Retry-After" => "1"],
    )
    return keepalive
end

function _run_turn!(
    turn::Turn,
    prompt_ids::Vector{Int},
    max_new::Int,
    sampling,
    boundary::Int,
)
    turn.cancelled[] && return nothing
    if turn.stream
        start_event_stream(turn.io; keepalive=turn.keepalive)
        turn.started[] = true
        _write_event!(
            turn,
            _chunk(turn, Dict("role" => "assistant", "content" => ""), nothing),
        )
    end
    extra = sampling === nothing ? (;) : (sampling=sampling,)
    # the boundary is asked for only when the prompt holds one, so an engine
    # without the keyword is called as before
    boundary > 0 && (extra = merge(extra, (checkpoint_at=boundary,)))
    return turn.model.generate(
        prompt_ids;
        max_new_tokens=max_new,
        on_token=id -> _on_token(turn, id),
        extra...,
    )
end

function _on_token(turn::Turn, id::Integer)
    yield()     # lets the disconnect watcher run between tokens
    push!(turn.ids, Int(id))
    turn.cancelled[] && return false
    _sync_text!(turn; final=false)
    return !turn.parser.stopped && !turn.cancelled[]
end

# Feed the parser the text of the ids it has not seen yet. Only a short window
# is decoded, so each token costs the window, not the whole reply (decoding all
# ids per token made a long reply quadratic). The window starts at `prefix_at`,
# a step before `read_at`, so a tokenizer that treats the first id of a decode
# specially (a stripped leading space, bytes merged across ids) treats both
# strings the same way; the new text is the difference. A trailing U+FFFD
# means the last id ended mid-character, so it waits for more ids.
function _sync_text!(turn::Turn; final::Bool)
    n = length(turn.ids)
    n <= turn.read_at && return nothing
    detok = turn.model.detokenize
    prefix = if turn.prefix_at < turn.read_at
        detok(turn.ids[(turn.prefix_at+1):turn.read_at])
    else
        ""
    end
    text = detok(turn.ids[(turn.prefix_at+1):n])
    !final && endswith(text, '\ufffd') && return nothing
    bytes = codeunits(text)
    skip = ncodeunits(prefix)
    # an id that adds no text yet stays in the window until one does
    length(bytes) <= skip && !final && return nothing
    turn.prefix_at = turn.read_at
    turn.read_at = n
    length(bytes) <= skip && return nothing
    new = Vector{UInt8}(view(bytes, (skip+1):length(bytes)))
    _deliver!(turn, feed!(turn.parser, new))
    return nothing
end

function _deliver!(turn::Turn, events)
    for e in events
        if e isa ReasoningDelta
            push!(turn.reasoning, e.text)
            turn.stream && _write_event!(
                turn,
                _chunk(turn, Dict("reasoning_content" => e.text), nothing),
            )
        elseif e isa ContentDelta
            push!(turn.content, e.text)
            turn.stream &&
                _write_event!(turn, _chunk(turn, Dict("content" => e.text), nothing))
        elseif e isa ToolCallDelta
            push!(turn.calls, e)
            turn.stream && _write_event!(
                turn,
                _chunk(
                    turn,
                    Dict(
                        "tool_calls" => [
                            Dict(
                                "index" => e.index,
                                "id" => e.id,
                                "type" => "function",
                                "function" => Dict(
                                    "name" => e.name,
                                    "arguments" => e.arguments,
                                ),
                            ),
                        ],
                    ),
                    nothing,
                ),
            )
        end
    end
    return nothing
end

function _write_event!(turn::Turn, obj)
    turn.cancelled[] && return nothing
    try
        write_sse(turn.io, JSON.json(obj))
    catch e
        e isa Base.IOError || e isa EOFError || rethrow()
        turn.cancelled[] = true
    end
    return nothing
end

function _chunk(turn::Turn, delta, finish)
    return Dict(
        "id" => turn.id,
        "object" => "chat.completion.chunk",
        "created" => turn.created,
        "model" => turn.model.model_id,
        "choices" => [Dict("index" => 0, "delta" => delta, "finish_reason" => finish)],
    )
end

function _finish_reason(turn::Turn, generated::Int, max_new::Int)
    turn.parser.n_calls > 0 && return "tool_calls"
    (!turn.parser.stopped && generated >= max_new) && return "length"
    return "stop"
end

_call_object(c::ToolCallDelta) = Dict(
    "id" => c.id,
    "type" => "function",
    "function" => Dict("name" => c.name, "arguments" => c.arguments),
)

_usage(prompt::Int, completion::Int, cached::Int) = Dict(
    "prompt_tokens" => prompt,
    "completion_tokens" => completion,
    "total_tokens" => prompt + completion,
    "prompt_tokens_details" => Dict("cached_tokens" => cached),
)

_completion_id() = "chatcmpl-" * string(rand(UInt64), base=16, pad=16)

function _parse_body(raw::Vector{UInt8})
    body = try
        _chat_json_parse(String(copy(raw)))
    catch
        throw(HTTPError(400, "request body is not valid JSON"))
    end
    body isa AbstractDict || throw(HTTPError(400, "request body must be a JSON object"))
    return body
end

function _normalize_messages(messages::AbstractVector)
    out = Dict{String, Any}[]
    for m in messages
        m isa AbstractDict || throw(HTTPError(400, "each message must be an object"))
        haskey(m, "role") || throw(HTTPError(400, "each message needs a role"))
        msg = Dict{String, Any}(String(k) => v for (k, v) in m)
        calls = get(m, "tool_calls", nothing)
        if calls isa AbstractVector
            msg["tool_calls"] = Any[_normalize_call(c) for c in calls]
        end
        push!(out, msg)
    end
    return out
end

# The chat template reads tool-call arguments as a mapping, but the OpenAI wire
# format sends them as a JSON string. Parse them here, once.
function _normalize_call(c)
    c isa AbstractDict || throw(HTTPError(400, "tool_calls entries must be objects"))
    f = get(c, "function", nothing)
    f isa AbstractDict || throw(HTTPError(400, "a tool call needs a function object"))
    args = get(f, "arguments", "{}")
    parsed = if args isa AbstractString
        try
            _chat_json_parse(String(args))
        catch
            throw(HTTPError(400, "tool call arguments are not valid JSON"))
        end
    else
        args
    end
    parsed isa AbstractDict ||
        throw(HTTPError(400, "tool call arguments must be a JSON object"))
    return Dict{String, Any}(
        "id" => get(c, "id", nothing),
        "type" => "function",
        "function" =>
            Dict{String, Any}("name" => String(get(f, "name", "")), "arguments" => parsed),
    )
end

# `enable_thinking` may come at the top level or inside chat_template_kwargs.
# Unset, the model's template default applies.
function _thinking(body::AbstractDict, default::Bool)
    kw = get(body, "chat_template_kwargs", nothing)
    if kw isa AbstractDict && haskey(kw, "enable_thinking")
        return kw["enable_thinking"] !== false
    end
    haskey(body, "enable_thinking") && return body["enable_thinking"] !== false
    return default
end

# Greedy unless the request sets a temperature above 0. top_p and seed only
# matter once sampling is on; a seed that is absent is taken from the clock.
function _sampling(body::AbstractDict)
    t = get(body, "temperature", nothing)
    p = get(body, "top_p", 1.0)
    seed = get(body, "seed", nothing)
    t === nothing && return nothing
    (t isa Real && !(t isa Bool) && isfinite(t) && 0 <= t <= 2) ||
        throw(HTTPError(400, "temperature must be a number from 0 to 2"))
    (p isa Real && !(p isa Bool) && 0 < p <= 1) ||
        throw(HTTPError(400, "top_p must be a number in (0, 1]"))
    t == 0 && return nothing
    if seed === nothing
        seed = Int(time_ns() >> 2)
    end
    (seed isa Integer && !(seed isa Bool) && 0 <= seed <= typemax(Int64)) ||
        throw(HTTPError(400, "seed must be a non-negative integer"))
    return Sampler(temperature=t, top_p=p, seed=seed)
end

function _max_tokens(body::AbstractDict)
    v =
        haskey(body, "max_completion_tokens") ? body["max_completion_tokens"] :
        get(body, "max_tokens", nothing)
    v === nothing && return nothing
    (v isa Integer && v > 0) ||
        throw(HTTPError(400, "max_tokens must be a positive integer"))
    return Int(v)
end

function _stop_list(v)
    v === nothing && return String[]
    v isa AbstractString && return String[v]
    if v isa AbstractVector && all(x -> x isa AbstractString, v)
        return String[String(x) for x in v]
    end
    throw(HTTPError(400, "stop must be a string or an array of strings"))
end

_flag(d, key::String) = d isa AbstractDict && get(d, key, false) === true
