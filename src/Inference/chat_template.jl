# Chat template: a hand-ported renderer for the Qwen3.5 XML chat template
# family. No Jinja. The reference is the GGUF's `tokenizer.chat_template` as
# llama-server renders it (`POST /apply-template`). Every branch below is
# checked byte-for-byte against that oracle by test/test_chat_template.jl.
#
# The server does not render a trailing assistant message as a normal turn.
# It renders it as a PREFILL: no generation prompt, no end-of-turn, no tool
# calls, no `</think>` split, and no trimming. `_render_prefill_assistant!`
# encodes that rule, measured against the oracle.
#
# Text only. Image and video parts raise an error.

using JSON

# Insertion-ordered JSON object. `tojson` emits keys in client order, and the
# parser keeps that order when called with `dicttype = _OrderedDict`. A plain
# `Dict` has no client order, so the server must parse request bodies with
# `_chat_json_parse` when key order matters.
struct _OrderedDict <: AbstractDict{String, Any}
    keys::Vector{String}
    data::Dict{String, Any}
end
_OrderedDict() = _OrderedDict(String[], Dict{String, Any}())
Base.length(d::_OrderedDict) = length(d.keys)
Base.iterate(d::_OrderedDict, i::Int=1) =
    i > length(d.keys) ? nothing : (d.keys[i] => d.data[d.keys[i]], i + 1)
Base.getindex(d::_OrderedDict, k) = d.data[k]
Base.haskey(d::_OrderedDict, k) = haskey(d.data, k)
Base.get(d::_OrderedDict, k, default) = get(d.data, k, default)
function Base.setindex!(d::_OrderedDict, v, k)
    key = String(k)
    haskey(d.data, key) || push!(d.keys, key)
    d.data[key] = v
    return d
end

_chat_json_parse(text::AbstractString) = JSON.parse(String(text); dicttype=_OrderedDict)

# ---------------------------------------------------------------------------
# ASCII-only trim (minja): space, \t, \n, \v, \f, \r. NBSP and NEL are kept.
const _CT_WS = (' ', '\t', '\n', '\v', '\f', '\r')

function _ct_trim(s::AbstractString)::String
    a = firstindex(s)
    b = lastindex(s)
    while a <= b && s[a] in _CT_WS
        a = nextind(s, a)
    end
    while b >= a && s[b] in _CT_WS
        b = prevind(s, b)
    end
    return a > b ? "" : String(s[a:b])
end

# ---------------------------------------------------------------------------
# tojson: minja byte format. Floats use C++ ostream precision 6 (%g style,
# exact rounding via Rational so the digits match printf, not a shortest form).

function _pow10(e::Integer)::Rational{BigInt}
    e >= 0 ? Rational{BigInt}(BigInt(10)^e) : 1 // BigInt(10)^(-e)
end

function _fmt_g6(x::Float64)::String
    isnan(x) && return "nan"
    isinf(x) && return x > 0 ? "inf" : "-inf"
    x == 0 && return signbit(x) ? "-0" : "0"
    neg = x < 0
    r = Rational{BigInt}(abs(x))
    e = floor(Int, log10(abs(x)))
    while r < _pow10(e)
        e -= 1
    end
    while r >= _pow10(e + 1)
        e += 1
    end
    n = round(BigInt, r / _pow10(e - 5))
    if n == BigInt(10)^6
        n = BigInt(10)^5
        e += 1
    end
    d = string(n)
    sign = neg ? "-" : ""
    if e < -4 || e >= 6
        frac = rstrip(d[2:6], '0')
        mant = isempty(frac) ? d[1:1] : d[1:1] * "." * frac
        return sign * mant * "e" * (e < 0 ? "-" : "+") * lpad(string(abs(e)), 2, '0')
    end
    if e >= 0
        ip = d[1:(e+1)]
        frac = rstrip(d[(e+2):6], '0')
    else
        ip = "0"
        frac = rstrip("0"^(-e - 1) * d, '0')
    end
    return sign * ip * (isempty(frac) ? "" : "." * frac)
end

function _json_escape!(io::IO, s::AbstractString)
    write(io, '"')
    for c in s
        if c == '"'
            write(io, "\\\"")
        elseif c == '\\'
            write(io, "\\\\")
        elseif c == '\b'
            write(io, "\\b")
        elseif c == '\f'
            write(io, "\\f")
        elseif c == '\n'
            write(io, "\\n")
        elseif c == '\r'
            write(io, "\\r")
        elseif c == '\t'
            write(io, "\\t")
        elseif c < ' '
            write(io, "\\u", lpad(string(UInt32(c), base=16), 4, '0'))
        else
            write(io, c)
        end
    end
    write(io, '"')
end

function _tojson!(io::IO, v)
    if v === nothing
        write(io, "null")
    elseif v isa Bool
        write(io, v ? "true" : "false")
    elseif v isa Integer
        write(io, string(v))
    elseif v isa AbstractFloat
        write(io, _fmt_g6(Float64(v)))
    elseif v isa AbstractString
        _json_escape!(io, v)
    elseif v isa AbstractVector
        write(io, "[")
        for (i, x) in enumerate(v)
            i > 1 && write(io, ", ")
            _tojson!(io, x)
        end
        write(io, "]")
    elseif v isa AbstractDict
        write(io, "{")
        first = true
        for (k, x) in v
            first || write(io, ", ")
            first = false
            _json_escape!(io, String(k))
            write(io, ": ")
            _tojson!(io, x)
        end
        write(io, "}")
    else
        error("chat template: value of type $(typeof(v)) cannot be rendered")
    end
    return io
end

# `| string` on a scalar. Python-style spellings: True, False, None.
function _py_str(v)::String
    v === nothing && return "None"
    v isa Bool && return v ? "True" : "False"
    v isa Integer && return string(v)
    v isa AbstractFloat && return _fmt_g6(Float64(v))
    v isa AbstractString && return String(v)
    error("chat template: value of type $(typeof(v)) cannot be converted to string")
end

# ---------------------------------------------------------------------------
# Content. Text only: a string, or an array of parts with a `text` key.
function _render_content(content)::String
    content === nothing && return ""
    content isa AbstractString && return String(content)
    if content isa AbstractVector
        # The server joins text parts with "\n" before the template sees them.
        parts = String[]
        for item in content
            item isa AbstractDict || error("chat template: unexpected content part")
            if haskey(item, "image") ||
               haskey(item, "image_url") ||
               get(item, "type", nothing) == "image"
                error(
                    "chat template: image content is not supported by the text-only renderer",
                )
            elseif haskey(item, "video") || get(item, "type", nothing) == "video"
                error(
                    "chat template: video content is not supported by the text-only renderer",
                )
            elseif haskey(item, "text")
                t = item["text"]
                t isa AbstractString || error("chat template: text part must be a string")
                push!(parts, String(t))
            else
                error("chat template: unexpected content part (no text key)")
            end
        end
        return join(parts, "\n")
    end
    error("chat template: 'content' must be a string, an array of parts, or null")
end

_role(m) = String(get(m, "role", ""))

# Tools are rebuilt as {type, function:{name, description, parameters}} in that
# fixed order. Unknown keys are dropped, as the server does.
function _render_tool!(io::IO, tool)
    tool isa AbstractDict || error("chat template: tool must be an object")
    haskey(tool, "type") || error("chat template: tool is missing 'type'")
    fn = get(tool, "function", nothing)
    fn isa AbstractDict || error("chat template: tool is missing 'function'")
    haskey(fn, "name") || error("chat template: tool function is missing 'name'")
    desc = get(fn, "description", nothing)
    params = get(fn, "parameters", nothing)
    desc === nothing && (desc = "")
    params === nothing && (params = _OrderedDict())
    write(io, "{\"type\": ")
    _tojson!(io, tool["type"])
    write(io, ", \"function\": {\"name\": ")
    _tojson!(io, fn["name"])
    write(io, ", \"description\": ")
    _tojson!(io, desc)
    write(io, ", \"parameters\": ")
    _tojson!(io, params)
    write(io, "}}")
end

const _TOOLS_PREAMBLE = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
const _TOOLS_INSTRUCTIONS =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n" *
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" *
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\n" *
    "multiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" *
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" *
    "- Required parameters MUST be specified\n" *
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" *
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" *
    "</IMPORTANT>"

# ---------------------------------------------------------------------------
# Tool-call arguments. A JSON string is parsed first. Only an object renders
# parameters. Arrays and absent arguments render none.
function _render_arguments!(io::IO, args)
    if args isa AbstractString
        args = try
            _chat_json_parse(args)
        catch
            error("chat template: tool call arguments are not valid JSON")
        end
    end
    if args isa AbstractDict
        for (k, v) in args
            write(io, "<parameter=", String(k), ">\n")
            if v isa AbstractDict || (v isa AbstractVector)
                _tojson!(io, v)
            else
                write(io, _py_str(v))
            end
            write(io, "\n</parameter>\n")
        end
    elseif args === nothing || args isa AbstractVector
        nothing
    else
        error(
            "chat template: tool call arguments must be an object or a JSON object string",
        )
    end
    return io
end

function _tool_call_function(tc)
    tc isa AbstractDict || error("chat template: tool call must be an object")
    haskey(tc, "type") || error("chat template: tool call is missing 'type'")
    fn = haskey(tc, "function") ? tc["function"] : tc
    fn isa AbstractDict || error("chat template: tool call 'function' must be an object")
    haskey(fn, "name") && fn["name"] isa AbstractString ||
        error("chat template: tool call is missing a string 'name'")
    return fn
end

# ---------------------------------------------------------------------------
# Splitting helpers mirror Python's str.split(sep)[0] and [-1].
_before_first(s::AbstractString, sep::AbstractString) =
    String(first(split(s, sep; limit=2)))
_after_last(s::AbstractString, sep::AbstractString) = String(last(split(s, sep)))

# Mid-conversation assistant turn, a closed message.
function _render_assistant!(io::IO, m)
    content = _ct_trim(_render_content(get(m, "content", nothing)))
    rc = get(m, "reasoning_content", nothing)
    reasoning = ""
    if rc isa AbstractString
        reasoning = String(rc)
    else
        rc === nothing || error("chat template: reasoning_content must be a string")
        if occursin("</think>", content)
            reasoning = lstrip(
                _after_last(rstrip(_before_first(content, "</think>"), '\n'), "<think>"),
                '\n',
            )
            content = lstrip(_after_last(content, "</think>"), '\n')
        end
    end
    reasoning = _ct_trim(reasoning)
    write(io, "<|im_start|>assistant\n<think>\n", reasoning, "\n</think>\n\n", content)
    tool_calls = get(m, "tool_calls", nothing)
    if tool_calls isa AbstractVector && !isempty(tool_calls)
        for (k, tc) in enumerate(tool_calls)
            fn = _tool_call_function(tc)
            if k == 1
                write(
                    io,
                    _ct_trim(content) != "" ? "\n\n<tool_call>\n<function=" :
                    "<tool_call>\n<function=",
                )
            else
                write(io, "\n<tool_call>\n<function=")
            end
            write(io, fn["name"], ">\n")
            _render_arguments!(io, get(fn, "arguments", nothing))
            write(io, "</function>\n</tool_call>")
        end
    end
    write(io, "<|im_end|>\n")
    return io
end

# Trailing assistant message: the prefill the server emits. Tool calls are
# never rendered here. An empty message with no reasoning and no tool calls is
# dropped entirely.
function _render_prefill_assistant!(io::IO, m)
    content = _render_content(get(m, "content", nothing))
    rc = get(m, "reasoning_content", nothing)
    rc === nothing ||
        rc isa AbstractString ||
        error("chat template: reasoning_content must be a string")
    reasoning = rc === nothing ? "" : String(rc)
    tool_calls = get(m, "tool_calls", nothing)
    has_tools = tool_calls isa AbstractVector && !isempty(tool_calls)
    isempty(content) && isempty(reasoning) && !has_tools && return io
    write(io, "<|im_start|>assistant\n<think>\n")
    if !isempty(content)
        write(io, reasoning, "\n</think>\n\n", content)
    elseif !isempty(reasoning)
        write(io, reasoning)
    else
        write(io, "\n</think>\n\n")
    end
    return io
end

"""
    render_chat(messages; tools=nothing, add_generation_prompt=true, enable_thinking=true) -> String

Render an OpenAI-style message list with the Qwen3.5 XML template. Matches
llama-server's `/apply-template` byte-for-byte on the covered fixtures.
A trailing assistant message is rendered as a prefill, and `add_generation_prompt`
does not apply to it.
"""
function render_chat(
    messages::AbstractVector;
    tools=nothing,
    add_generation_prompt::Bool=true,
    enable_thinking::Bool=true,
)::String
    isempty(messages) && error("chat template: no messages provided")
    io = IOBuffer()
    n = length(messages)

    num_sys = 0
    merged = ""
    if _role(messages[1]) in ("system", "developer")
        first = _ct_trim(_render_content(get(messages[1], "content", nothing)))
        if n > 1 && _role(messages[2]) in ("system", "developer")
            second = _ct_trim(_render_content(get(messages[2], "content", nothing)))
            merged = first * "\n" * second
            num_sys = 2
        else
            merged = first
            num_sys = 1
        end
    end

    if tools isa AbstractVector && !isempty(tools)
        write(io, "<|im_start|>system\n", _TOOLS_PREAMBLE)
        for tool in tools
            write(io, "\n")
            _render_tool!(io, tool)
        end
        write(io, _TOOLS_INSTRUCTIONS)
        isempty(merged) || write(io, "\n\n", merged)
        write(io, "<|im_end|>\n")
    elseif !isempty(merged)
        write(io, "<|im_start|>system\n", merged, "<|im_end|>\n")
    end

    prefill = _role(messages[n]) == "assistant"
    for i in (num_sys+1):n
        m = messages[i]
        role = _role(m)
        if role == "user"
            write(
                io,
                "<|im_start|>user\n",
                _ct_trim(_render_content(get(m, "content", nothing))),
                "<|im_end|>\n",
            )
        elseif role == "assistant"
            if prefill && i == n
                _render_prefill_assistant!(io, m)
            else
                _render_assistant!(io, m)
            end
        elseif role == "tool"
            if i > 1 && _role(messages[i-1]) != "tool"
                write(io, "<|im_start|>user")
            end
            write(
                io,
                "\n<tool_response>\n",
                _ct_trim(_render_content(get(m, "content", nothing))),
                "\n</tool_response>",
            )
            if i < n && _role(messages[i+1]) != "tool"
                write(io, "<|im_end|>\n")
            elseif i == n
                write(io, "<|im_end|>\n")
            end
        end
    end

    if add_generation_prompt && !prefill
        write(io, "<|im_start|>assistant\n")
        write(io, enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n")
    end
    return String(take!(io))
end

"""
    chat_family(template::AbstractString) -> Symbol

Identify a chat template by its markers. Returns `:qwen35_xml` for the
Qwen3.5 XML family (`<function=`, `<tool_response>`, `<think>`). Any other
template is an error.
"""
function chat_family(template::AbstractString)::Symbol
    if occursin("<function=", template) &&
       occursin("<tool_response>", template) &&
       occursin("<think>", template)
        return :qwen35_xml
    end
    error(
        "chat_family: template is not a supported chat family (expected Qwen3.5 XML markers)",
    )
end

"""
    load_gguf_chat_template(path) -> String

Read `tokenizer.chat_template` from a GGUF. This goes through
`load_gguf_tokenizer`, so it also reads the vocabulary.
"""
function load_gguf_chat_template(path::AbstractString)::String
    tpl = load_gguf_tokenizer(path).chat_template
    tpl === nothing &&
        error("load_gguf_chat_template: $(basename(path)) has no tokenizer.chat_template")
    return tpl
end

export render_chat, chat_family, load_gguf_chat_template
