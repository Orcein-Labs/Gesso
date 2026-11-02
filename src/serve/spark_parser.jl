# Streaming parser for the Spark-X2.6 chat dialect.
#
# The model writes reasoning up to `</think>` (the prompt opens `<think>`), then
# content, and tool calls as
#
#     <tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>
#
# This is the inverse of `_sp_tool_calls!` in Inference/spark_chat.jl: a string
# argument is written raw and anything else as JSON. So a value is a string when
# the tool schema says so, coerced like QwenXMLParser when the schema gives
# another type, and read as JSON when it parses as JSON and the schema is silent.
#
# The byte interface matches QwenXMLParser (feed!, finish!, stopped, n_calls),
# so the server picks a parser by chat family and nothing else changes.

const _SP_THINK_OPEN = Vector{UInt8}("<think>")
const _SP_THINK_CLOSE = Vector{UInt8}("</think>")
const _SP_TOOL_OPEN = Vector{UInt8}("<tool_call>")
const _SP_TOOL_CLOSE = Vector{UInt8}("</tool_call>")

mutable struct SparkParser
    state::Symbol               # :reasoning, :content or :tool
    buf::Vector{UInt8}          # bytes not yet emitted; the tool body while in :tool
    lead_pending::Bool          # drop whitespace before the first visible content
    tools::Vector{Any}
    stops::Vector{Vector{UInt8}}
    n_calls::Int
    stopped::Bool
end

"""
    SparkParser(; tools=[], start_in_reasoning=false, stop=String[])

`start_in_reasoning` comes from the rendered prompt (see `opens_reasoning`).
`tools` is the request's tool list, used to type call arguments. A stop string
ends generation as it does for QwenXMLParser.
"""
function SparkParser(; tools=Any[], start_in_reasoning::Bool=false, stop=String[])
    stops = Vector{UInt8}[Vector{UInt8}(s) for s in stop if !isempty(s)]
    return SparkParser(
        start_in_reasoning ? :reasoning : :content,
        UInt8[],
        false,
        Any[t for t in tools],
        stops,
        0,
        false,
    )
end

function feed!(p::SparkParser, bytes::AbstractVector{UInt8})
    out = Any[]
    p.stopped && return out
    append!(p.buf, bytes)
    _advance!(p, out)
    return out
end

feed!(p::SparkParser, s::AbstractString) = feed!(p, codeunits(s))

function finish!(p::SparkParser)
    out = Any[]
    p.stopped && return out
    if p.state === :tool
        raw = String(copy(_SP_TOOL_OPEN)) * String(copy(p.buf))
        p.buf = UInt8[]
        p.state = :content
        push!(out, ContentDelta(raw))
        return out
    end
    if !isempty(p.buf)
        _sp_emit!(p, out, copy(p.buf))
        p.buf = UInt8[]
    end
    return out
end

function _advance!(p::SparkParser, out::Vector{Any})
    while true
        if p.state === :tool
            i = _find(p.buf, _SP_TOOL_CLOSE)
            i === nothing && return
            body = String(p.buf[1:(i-1)])
            p.buf = p.buf[(i+length(_SP_TOOL_CLOSE)):end]
            _close_tool!(p, out, body)
            p.state = :content
            p.lead_pending = true
            continue
        end

        i, kind, len = _sp_next_match(p)
        if i !== nothing
            _sp_emit!(p, out, p.buf[1:(i-1)])
            p.buf = p.buf[(i+len):end]
            if kind === :think_close
                p.state = :content
                p.lead_pending = true
            elseif kind === :think_open
                p.state = :reasoning
            elseif kind === :tool_open
                p.state = :tool
            else
                p.stopped = true
                p.buf = UInt8[]
                return
            end
            continue
        end

        hold = max(_sp_partial_suffix(p), _utf8_incomplete_tail(p.buf))
        n = length(p.buf) - hold
        if n > 0
            _sp_emit!(p, out, p.buf[1:n])
            p.buf = p.buf[(n+1):end]
        end
        return
    end
end

function _sp_tags(p::SparkParser)
    tags = p.state === :reasoning ? [(_SP_THINK_CLOSE, :think_close)] :
        [(_SP_THINK_OPEN, :think_open), (_SP_TOOL_OPEN, :tool_open)]
    for s in p.stops
        push!(tags, (s, :stop))
    end
    return tags
end

function _sp_next_match(p::SparkParser)
    best, best_kind, best_len = nothing, :none, 0
    for (tag, kind) in _sp_tags(p)
        i = _find(p.buf, tag)
        if i !== nothing && (best === nothing || i < best)
            best, best_kind, best_len = i, kind, length(tag)
        end
    end
    return best, best_kind, best_len
end

# Longest buffer suffix that is a proper prefix of a tag that could still complete.
function _sp_partial_suffix(p::SparkParser)
    best = 0
    for (tag, _) in _sp_tags(p)
        for k in min(length(tag)-1, length(p.buf)):-1:1
            if view(p.buf, (length(p.buf)-k+1):length(p.buf)) == view(tag, 1:k)
                best = max(best, k)
                break
            end
        end
    end
    return best
end

function _sp_emit!(p::SparkParser, out::Vector{Any}, bytes::AbstractVector{UInt8})
    isempty(bytes) && return
    if p.state === :reasoning
        push!(out, ReasoningDelta(String(copy(bytes))))
        return
    end
    start = 1
    if p.lead_pending
        while start <= length(bytes) && bytes[start] in (0x20, 0x09, 0x0a, 0x0d)
            start += 1
        end
        start > length(bytes) && return
        p.lead_pending = false
    end
    push!(out, ContentDelta(String(copy(bytes[start:end]))))
    return
end

function _close_tool!(p::SparkParser, out::Vector{Any}, body::String)
    parsed = _sp_parse_call(body)
    if parsed === nothing
        push!(out, ContentDelta("<tool_call>" * body * "</tool_call>"))
        return
    end
    name, params = parsed
    args = String[]
    for (key, raw) in params
        push!(args, JSON.json(key) * ":" * _sp_arg_json(raw, _param_type(p.tools, name, key)))
    end
    p.n_calls += 1
    push!(out, ToolCallDelta(p.n_calls - 1, _call_id(), name, "{" * join(args, ",") * "}"))
    return
end

# A JSON fragment for one argument. Object, array and schema-less values that
# already parse as JSON are passed through as written, so key order survives.
function _sp_arg_json(raw::String, typ)::String
    typ == "string" && return JSON.json(raw)
    s = String(strip(raw))
    if typ === nothing || typ == "object" || typ == "array"
        v = try
            _chat_json_parse(s)
        catch
            return JSON.json(raw)
        end
        typ == "object" && !(v isa AbstractDict) && return JSON.json(raw)
        typ == "array" && !(v isa AbstractVector) && return JSON.json(raw)
        return s
    end
    return JSON.json(_coerce(raw, typ))
end

# Returns (name, [(key, raw_value)]), or nothing when the body is not a call.
# Whitespace is allowed around the name and between the tags.
function _sp_parse_call(body::AbstractString)
    k = findfirst("<arg_key>", body)
    name = strip(k === nothing ? body : body[1:prevind(body, first(k))])
    (isempty(name) || any(isspace, name) || occursin('<', name)) && return nothing
    params = Tuple{String, String}[]
    i = k === nothing ? lastindex(body) + 1 : first(k)
    while true
        while i <= lastindex(body) && isspace(body[i])
            i = nextind(body, i)
        end
        i > lastindex(body) && return String(name), params
        startswith(SubString(body, i), "<arg_key>") || return nothing
        kc = findnext("</arg_key>", body, i)
        kc === nothing && return nothing
        key = body[(i+length("<arg_key>")):prevind(body, first(kc))]
        i = last(kc) + 1
        while i <= lastindex(body) && isspace(body[i])
            i = nextind(body, i)
        end
        startswith(SubString(body, i), "<arg_value>") || return nothing
        vc = findnext("</arg_value>", body, i)
        vc === nothing && return nothing
        value = body[(i+length("<arg_value>")):prevind(body, first(vc))]
        any(p -> p[1] == key, params) && return nothing
        push!(params, (String(key), String(value)))
        i = last(vc) + 1
    end
end
