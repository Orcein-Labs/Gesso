# Streaming parser for the Gemma 4 chat dialect.
#
# The model writes reasoning inside `<|channel>thought\n ... <channel|>`, then
# content, and tool calls as
#
#     <|tool_call>call:NAME{KEY:VALUE,...}<tool_call|>
#
# where a string value is wrapped in `<|"|>` on both sides, and an object or
# array nests the same way. This is the inverse of `_g4_format_argument` in
# Inference/gemma4_chat.jl: a value the renderer wrote parses back to the same
# value. Keys are bare, so a key holding one of `:,{}[]` cannot be written in
# this format, and the parser refuses such a call rather than guess.
#
# The byte interface matches QwenXMLParser (feed!, finish!, stopped, n_calls),
# so the server picks a parser by chat family and nothing else changes.

const _G4_QUOTE = "<|\"|>"
const _G4_THOUGHT_OPEN = Vector{UInt8}("<|channel>thought\n")
const _G4_THOUGHT_CLOSE = Vector{UInt8}("<channel|>")
const _G4_TOOL_OPEN = Vector{UInt8}("<|tool_call>")
const _G4_TOOL_CLOSE = Vector{UInt8}("<tool_call|>")

mutable struct Gemma4Parser
    state::Symbol               # :reasoning, :content or :tool
    buf::Vector{UInt8}          # bytes not yet emitted; the tool body while in :tool
    lead_pending::Bool          # drop whitespace before the first visible content
    stops::Vector{Vector{UInt8}}
    n_calls::Int
    stopped::Bool
end

"""
    gemma4_opens_reasoning(prompt) -> Bool

True when the rendered prompt ends inside an open thought channel, so the model's
first bytes are reasoning. With thinking off the template closes the channel
itself, and the model's first bytes are content.
"""
gemma4_opens_reasoning(prompt::AbstractString)::Bool = endswith(prompt, "<|channel>thought\n")

"""
    Gemma4Parser(; start_in_reasoning=false, stop=String[])

`start_in_reasoning` comes from the rendered prompt (see `gemma4_opens_reasoning`).
Without it the parser still routes a model-opened thought channel to reasoning.
A stop string ends generation as it does for QwenXMLParser.
"""
function Gemma4Parser(; start_in_reasoning::Bool=false, stop=String[])
    stops = Vector{UInt8}[Vector{UInt8}(s) for s in stop if !isempty(s)]
    return Gemma4Parser(
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

Consume a chunk of output bytes and return the events they complete. Events are
ReasoningDelta, ContentDelta and ToolCallDelta.
"""
function feed!(p::Gemma4Parser, bytes::AbstractVector{UInt8})
    out = Any[]
    p.stopped && return out
    append!(p.buf, bytes)
    _advance!(p, out)
    return out
end

feed!(p::Gemma4Parser, s::AbstractString) = feed!(p, codeunits(s))

"""
    finish!(p) -> Vector{Any}

Flush held-back bytes at the end of generation. An unclosed tool call is surfaced
as raw content rather than dropped.
"""
function finish!(p::Gemma4Parser)
    out = Any[]
    p.stopped && return out
    if p.state === :tool
        raw = String(copy(_G4_TOOL_OPEN)) * String(copy(p.buf))
        p.buf = UInt8[]
        p.state = :content
        push!(out, ContentDelta(raw))
        return out
    end
    if !isempty(p.buf)
        _g4_emit!(p, out, copy(p.buf))
        p.buf = UInt8[]
    end
    return out
end

function _advance!(p::Gemma4Parser, out::Vector{Any})
    while true
        if p.state === :tool
            i = _find(p.buf, _G4_TOOL_CLOSE)
            i === nothing && return
            body = String(p.buf[1:(i-1)])
            p.buf = p.buf[(i+length(_G4_TOOL_CLOSE)):end]
            _close_tool!(p, out, body)
            p.state = :content
            p.lead_pending = true
            continue
        end

        i, kind, len = _g4_next_match(p)
        if i !== nothing
            _g4_emit!(p, out, p.buf[1:(i-1)])
            p.buf = p.buf[(i+len):end]
            if kind === :thought_close
                p.state = :content
                p.lead_pending = true
            elseif kind === :thought_open
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

        hold = max(_g4_partial_suffix(p), _utf8_incomplete_tail(p.buf))
        n = length(p.buf) - hold
        if n > 0
            _g4_emit!(p, out, p.buf[1:n])
            p.buf = p.buf[(n+1):end]
        end
        return
    end
end

function _g4_tags(p::Gemma4Parser)
    tags = p.state === :reasoning ? [(_G4_THOUGHT_CLOSE, :thought_close)] :
        [(_G4_THOUGHT_OPEN, :thought_open), (_G4_TOOL_OPEN, :tool_open)]
    for s in p.stops
        push!(tags, (s, :stop))
    end
    return tags
end

function _g4_next_match(p::Gemma4Parser)
    best, best_kind, best_len = nothing, :none, 0
    for (tag, kind) in _g4_tags(p)
        i = _find(p.buf, tag)
        if i !== nothing && (best === nothing || i < best)
            best, best_kind, best_len = i, kind, length(tag)
        end
    end
    return best, best_kind, best_len
end

# Longest buffer suffix that is a proper prefix of a tag that could still complete.
function _g4_partial_suffix(p::Gemma4Parser)
    best = 0
    for (tag, _) in _g4_tags(p)
        for k in min(length(tag)-1, length(p.buf)):-1:1
            if view(p.buf, (length(p.buf)-k+1):length(p.buf)) == view(tag, 1:k)
                best = max(best, k)
                break
            end
        end
    end
    return best
end

function _g4_emit!(p::Gemma4Parser, out::Vector{Any}, bytes::AbstractVector{UInt8})
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

function _close_tool!(p::Gemma4Parser, out::Vector{Any}, body::AbstractString)
    parsed = _g4_parse_call(body)
    if parsed === nothing
        push!(out, ContentDelta(String(copy(_G4_TOOL_OPEN)) * body * String(copy(_G4_TOOL_CLOSE))))
        return
    end
    name, args = parsed
    p.n_calls += 1
    push!(out, ToolCallDelta(p.n_calls - 1, _call_id(), name, _g4_json(args)))
    return
end

# ---------------------------------------------------------------------------
# The call body: `call:NAME{ARGS}`. Each value is one of
#   <|"|>text<|"|>      a string, raw between the delimiters
#   {k:v,...}           an object; a key is bare or <|"|>quoted<|"|>
#   [v,...]             an array
#   true, false, null   literals
#   digits, with optional fraction and exponent   a number
# The grammar has no escapes, which is exactly what the renderer writes.

struct _G4Malformed <: Exception end

# An object keeps its keys in the order they were written.
struct _G4Object
    pairs::Vector{Pair{String, Any}}
end

# Returns (name, args::_G4Object), or nothing when the body is not a call.
function _g4_parse_call(body::AbstractString)
    cs = collect(body)
    _g4_at(cs, 1, "call:") || return nothing
    brace = findfirst(==('{'), cs)
    brace === nothing && return nothing
    name = String(cs[6:(brace-1)])
    (isempty(name) || any(c -> isspace(c) || c in ":,{}[]", name)) && return nothing
    try
        value, next = _g4_value(cs, brace)
        next == length(cs) + 1 || return nothing
        value isa _G4Object || return nothing
        return name, value
    catch e
        e isa _G4Malformed || rethrow()
        return nothing
    end
end

_g4_at(cs::Vector{Char}, i::Int, lit::AbstractString) =
    i + length(lit) - 1 <= length(cs) && all(k -> cs[i+k-1] == lit[k], eachindex(lit))

# `i` is the index of the first character of the value. Returns (value, next index).
function _g4_value(cs::Vector{Char}, i::Int)
    i <= length(cs) || throw(_G4Malformed())
    _g4_at(cs, i, _G4_QUOTE) && return _g4_quoted(cs, i + length(_G4_QUOTE))
    cs[i] == '{' && return _g4_object(cs, i + 1)
    cs[i] == '[' && return _g4_array(cs, i + 1)
    j = i
    while j <= length(cs) && !(cs[j] in ",}]")
        j += 1
    end
    return _g4_scalar(String(cs[i:(j-1)])), j
end

function _g4_quoted(cs::Vector{Char}, i::Int)
    k = i
    while k <= length(cs) && !_g4_at(cs, k, _G4_QUOTE)
        k += 1
    end
    k > length(cs) && throw(_G4Malformed())
    return String(cs[i:(k-1)]), k + length(_G4_QUOTE)
end

function _g4_object(cs::Vector{Char}, i::Int)
    pairs = Pair{String, Any}[]
    if i <= length(cs) && cs[i] == '}'
        return _G4Object(pairs), i + 1
    end
    while true
        key, i = _g4_key(cs, i)
        any(p -> p.first == key, pairs) && throw(_G4Malformed())
        (i <= length(cs) && cs[i] == ':') || throw(_G4Malformed())
        value, i = _g4_value(cs, i + 1)
        push!(pairs, key => value)
        i <= length(cs) || throw(_G4Malformed())
        cs[i] == '}' && return _G4Object(pairs), i + 1
        cs[i] == ',' || throw(_G4Malformed())
        i += 1
    end
end

function _g4_key(cs::Vector{Char}, i::Int)
    _g4_at(cs, i, _G4_QUOTE) && return _g4_quoted(cs, i + length(_G4_QUOTE))
    j = i
    while j <= length(cs) && !(cs[j] in ":,{}[]")
        j += 1
    end
    j == i && throw(_G4Malformed())
    return String(cs[i:(j-1)]), j
end

function _g4_array(cs::Vector{Char}, i::Int)
    items = Any[]
    if i <= length(cs) && cs[i] == ']'
        return items, i + 1
    end
    while true
        value, i = _g4_value(cs, i)
        push!(items, value)
        i <= length(cs) || throw(_G4Malformed())
        cs[i] == ']' && return items, i + 1
        cs[i] == ',' || throw(_G4Malformed())
        i += 1
    end
end

function _g4_scalar(tok::AbstractString)
    tok == "true" && return true
    tok == "false" && return false
    tok == "null" && return nothing
    occursin(r"^-?\d+$", tok) && return _g4_int(tok)
    occursin(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$", tok) && return parse(Float64, tok)
    throw(_G4Malformed())
end

# An integer keeps every digit, so a large one is a BigInt rather than rounded.
function _g4_int(tok::AbstractString)
    n = parse(BigInt, tok)
    return n <= typemax(Int) && n >= typemin(Int) ? Int(n) : n
end

_g4_json(o::_G4Object) =
    "{" * join((JSON.json(k) * ":" * _g4_json(v) for (k, v) in o.pairs), ",") * "}"
_g4_json(a::Vector{Any}) = "[" * join((_g4_json(x) for x in a), ",") * "]"
_g4_json(v) = JSON.json(v)
