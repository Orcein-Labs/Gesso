# Streaming parser for the Qwen3.5 chat dialect.
#
# The model writes reasoning inside `<think>...</think>`, then content, and tool
# calls as
#
#     <tool_call>
#     <function=NAME>
#     <parameter=KEY>
#     VALUE
#     </parameter>
#     </function>
#     </tool_call>
#
# The parser consumes raw UTF-8 bytes, so a token boundary or a byte boundary can
# fall anywhere. Text that could still turn into a tag is held back until the
# next bytes decide it. A tool call is emitted whole, when its close tag arrives,
# because the arguments have to be valid JSON to be useful to a client.

struct ReasoningDelta
    text::String
end

struct ContentDelta
    text::String
end

struct ToolCallDelta
    index::Int
    id::String
    name::String
    arguments::String
end

const _THINK_CLOSE = Vector{UInt8}("</think>")
const _TOOL_OPEN = Vector{UInt8}("<tool_call>")
const _TOOL_CLOSE = Vector{UInt8}("</tool_call>")

mutable struct QwenXMLParser
    tools::Vector{Any}
    state::Symbol               # :reasoning, :content or :tool
    buf::Vector{UInt8}          # bytes not yet emitted; the tool body while in :tool
    lead_pending::Bool          # drop whitespace before the first visible content
    stops::Vector{Vector{UInt8}}
    n_calls::Int
    stopped::Bool
end

"""
    opens_reasoning(prompt) -> Bool

True when the rendered prompt ends inside an open `<think>` block, so the model's
first bytes are reasoning. The last `<think>` has to come after the last
`</think>`, and trailing whitespace is ignored.
"""
function opens_reasoning(prompt::AbstractString)::Bool
    t = rstrip(prompt)
    last_open = findlast("<think>", t)
    last_open === nothing && return false
    last_close = findlast("</think>", t)
    return last_close === nothing || first(last_open) > first(last_close)
end

"""
    QwenXMLParser(; tools=[], thinking=true, start_in_reasoning=thinking, stop=String[])

`start_in_reasoning` says whether the prompt left a `<think>` open. The server
takes it from the rendered prompt (see `opens_reasoning`). It defaults to
`thinking`, which is the Qwen3.5 rule. `tools` is the request's tool list, used to
coerce parameter values by their declared JSON type. A stop string ends
generation: the text before it is emitted and the rest is discarded. `p.stopped`
tells the caller to stop.
"""
function QwenXMLParser(;
    tools=Any[],
    thinking::Bool=true,
    start_in_reasoning::Bool=thinking,
    stop=String[],
)
    stops = Vector{UInt8}[Vector{UInt8}(s) for s in stop if !isempty(s)]
    return QwenXMLParser(
        collect(Any, tools),
        start_in_reasoning ? :reasoning : :content,
        UInt8[],
        false,
        stops,
        0,
        false,
    )
end

"""
    feed!(p, bytes) -> Vector{Any}

Consume a chunk of output bytes and return the events they complete. A string is
accepted too. Events are ReasoningDelta, ContentDelta and ToolCallDelta.
"""
function feed!(p::QwenXMLParser, bytes::AbstractVector{UInt8})
    out = Any[]
    p.stopped && return out
    append!(p.buf, bytes)
    _advance!(p, out)
    return out
end

feed!(p::QwenXMLParser, s::AbstractString) = feed!(p, codeunits(s))

"""
    finish!(p) -> Vector{Any}

Flush held-back bytes at the end of generation. An unclosed tool call is surfaced
as raw content rather than dropped.
"""
function finish!(p::QwenXMLParser)
    out = Any[]
    p.stopped && return out
    if p.state === :tool
        raw = "<tool_call>" * String(copy(p.buf))
        p.buf = UInt8[]
        p.state = :content
        push!(out, ContentDelta(raw))
        return out
    end
    if !isempty(p.buf)
        _emit!(p, out, copy(p.buf))
        p.buf = UInt8[]
    end
    return out
end

function _advance!(p::QwenXMLParser, out::Vector{Any})
    while true
        if p.state === :tool
            i = _find(p.buf, _TOOL_CLOSE)
            i === nothing && return
            body = String(p.buf[1:(i-1)])
            p.buf = p.buf[(i+length(_TOOL_CLOSE)):end]
            _close_tool!(p, out, body)
            p.state = :content
            p.lead_pending = true
            continue
        end

        i, kind, len = _next_match(p)
        if i !== nothing
            _emit!(p, out, p.buf[1:(i-1)])
            p.buf = p.buf[(i+len):end]
            if kind === :think_close
                p.state = :content
                p.lead_pending = true
            elseif kind === :tool_open
                p.state = :tool
            else
                p.stopped = true
                p.buf = UInt8[]
                return
            end
            continue
        end

        hold = max(_partial_suffix(p), _utf8_incomplete_tail(p.buf))
        n = length(p.buf) - hold
        if n > 0
            _emit!(p, out, p.buf[1:n])
            p.buf = p.buf[(n+1):end]
        end
        return
    end
end

function _next_match(p::QwenXMLParser)
    tag, kind =
        p.state === :reasoning ? (_THINK_CLOSE, :think_close) : (_TOOL_OPEN, :tool_open)
    best = _find(p.buf, tag)
    best_kind, best_len = kind, length(tag)
    for s in p.stops
        i = _find(p.buf, s)
        if i !== nothing && (best === nothing || i < best)
            best = i
            best_kind = :stop
            best_len = length(s)
        end
    end
    return best, best_kind, best_len
end

function _tags(p::QwenXMLParser)
    tags = p.state === :reasoning ? [_THINK_CLOSE] : [_TOOL_OPEN]
    append!(tags, p.stops)
    return tags
end

# Longest suffix of the buffer that is a proper prefix of a tag that could still
# complete. Only the tags for the current state count.
function _partial_suffix(p::QwenXMLParser)
    best = 0
    for tag in _tags(p)
        for k in min(length(tag)-1, length(p.buf)):-1:1
            if view(p.buf, (length(p.buf)-k+1):length(p.buf)) == view(tag, 1:k)
                best = max(best, k)
                break
            end
        end
    end
    return best
end

# Bytes at the end that start a multibyte UTF-8 character not yet complete.
function _utf8_incomplete_tail(buf::Vector{UInt8})
    n = length(buf)
    for j in 1:min(4, n)
        b = buf[n-j+1]
        b < 0x80 && return 0
        if b >= 0xC0
            need = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : 2
            return need > j ? j : 0
        end
    end
    return 0
end

function _emit!(p::QwenXMLParser, out::Vector{Any}, bytes::AbstractVector{UInt8})
    isempty(bytes) && return
    if p.state === :reasoning
        push!(out, ReasoningDelta(String(copy(bytes))))
        return
    end
    # Content: a leading run of whitespace after </think> or </tool_call> is
    # formatting, not reply, so it is dropped until visible text arrives.
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

function _find(buf::Vector{UInt8}, pat::Vector{UInt8})
    n, m = length(buf), length(pat)
    m == 0 && return nothing
    for i in 1:(n-m+1)
        @inbounds if buf[i] == pat[1] && view(buf, i:(i+m-1)) == pat
            return i
        end
    end
    return nothing
end

function _close_tool!(p::QwenXMLParser, out::Vector{Any}, body::String)
    parsed = _parse_call_body(body)
    if parsed === nothing
        push!(out, ContentDelta("<tool_call>" * body * "</tool_call>"))
        return
    end
    name, params = parsed
    args = String[]
    for (key, raw) in params
        typ = _param_type(p.tools, name, key)
        push!(args, JSON.json(key) * ":" * JSON.json(_coerce(raw, typ)))
    end
    p.n_calls += 1
    push!(out, ToolCallDelta(p.n_calls - 1, _call_id(), name, "{" * join(args, ",") * "}"))
    return
end

# Returns (name, [(key, raw_value)]) or nothing when there is no function tag.
function _parse_call_body(body::AbstractString)
    f = findfirst("<function=", body)
    f === nothing && return nothing
    start = last(f) + 1
    gt = findnext('>', body, start)
    gt === nothing && return nothing
    name = _unquote(body[start:(gt-1)])
    fend = findnext("</function>", body, gt + 1)
    fbody = fend === nothing ? body[(gt+1):end] : body[(gt+1):(first(fend)-1)]

    params = Tuple{String, String}[]
    pos = 1
    while true
        ps = findnext("<parameter=", fbody, pos)
        ps === nothing && break
        ks = last(ps) + 1
        kgt = findnext('>', fbody, ks)
        kgt === nothing && break
        pe = findnext("</parameter>", fbody, kgt + 1)
        pe === nothing && break
        key = _unquote(fbody[ks:(kgt-1)])
        push!(params, (key, _trim_value(fbody[(kgt+1):(first(pe)-1)])))
        pos = last(pe) + 1
    end
    return (name, params)
end

# The template writes one newline after the tag and one before the close tag.
# Strip exactly one of each. Everything else in the value is the value.
function _trim_value(v::AbstractString)
    startswith(v, '\n') && (v = v[2:end])
    endswith(v, '\n') && (v = v[1:(end-1)])
    return String(v)
end

function _unquote(s::AbstractString)
    t = strip(s)
    if length(t) >= 2 &&
       ((t[1] == '"' && t[end] == '"') || (t[1] == '\'' && t[end] == '\''))
        return String(t[2:(end-1)])
    end
    return String(t)
end

function _param_type(tools::Vector{Any}, name::AbstractString, key::AbstractString)
    for t in tools
        t isa AbstractDict || continue
        f = get(t, "function", t)
        f isa AbstractDict || continue
        get(f, "name", nothing) == name || continue
        params = get(f, "parameters", nothing)
        params isa AbstractDict || return nothing
        props = get(params, "properties", nothing)
        props isa AbstractDict || return nothing
        spec = get(props, key, nothing)
        spec isa AbstractDict || return nothing
        typ = get(spec, "type", nothing)
        if typ isa AbstractVector
            i = findfirst(x -> x != "null", typ)
            typ = i === nothing ? nothing : typ[i]
        end
        return typ isa AbstractString ? String(typ) : nothing
    end
    return nothing
end

# Coerce a raw value to its declared JSON type. A value that does not parse as
# its declared type stays a string: the model's output is not an error here.
function _coerce(raw::AbstractString, typ)
    typ === nothing && return String(raw)
    s = strip(raw)
    if typ == "integer"
        n = tryparse(Int, s)
        return n === nothing ? String(raw) : n
    elseif typ == "number"
        n = tryparse(Int, s)
        n !== nothing && return n
        x = tryparse(Float64, s)
        return x === nothing ? String(raw) : x
    elseif typ == "boolean"
        s == "true" && return true
        s == "false" && return false
        return String(raw)
    elseif typ == "null"
        return s == "null" ? nothing : String(raw)
    elseif typ == "object" || typ == "array"
        v = try
            JSON.parse(String(s))
        catch
            return String(raw)
        end
        return (typ == "object" ? v isa AbstractDict : v isa AbstractVector) ? v :
               String(raw)
    end
    return String(raw)
end

_call_id() = "call_" * string(rand(UInt64), base=16, pad=16)
