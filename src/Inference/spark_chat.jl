# Spark-X2.6 chat template: a hand-ported renderer for the DeepSeek-style
# template the Spark GGUFs ship (`<｜start▁of▁sentence｜><|User|>...`). No Jinja.
# The reference is the GGUF's `tokenizer.chat_template` rendered by Jinja2 with
# the Hugging Face `tojson` filter (json.dumps, ensure_ascii off), the way the
# model's tokenizer applies it. test/test_spark_chat.jl checks every branch
# byte-for-byte against fixtures captured from that oracle.
#
# Text and tool calls only. Tool-call arguments must already be a mapping; the
# server parses the OpenAI JSON string before rendering.

const _SP_BOS = "<｜start▁of▁sentence｜>"
const _SP_EOS = "<｜end▁of▁sentence｜>"
const _SP_EFFORT_NOTE =
    "\n`thinking_effort` guides on how much to think in your thinking channel " *
    "(not including the response channel), supported values include `low` and `high`.\n" *
    "Now the system is invoked with `thinking_effort="

# Python's float repr: the shortest round-trip digits, fixed notation for
# decimal exponents in [-4, 16), otherwise d.ddde±XX.
function _sp_py_float(x::Float64)::String
    isnan(x) && return "NaN"
    isinf(x) && return x > 0 ? "Infinity" : "-Infinity"
    x == 0 && return signbit(x) ? "-0.0" : "0.0"
    sign = x < 0 ? "-" : ""
    s = string(abs(x))
    m, e = occursin('e', s) ? split(s, 'e') : (s, "0")
    ip, fp = occursin('.', m) ? split(m, '.') : (m, "")
    digits = ip * fp
    point = length(ip) + parse(Int, e)       # digits before the decimal point
    lead = findfirst(!=('0'), digits)
    point -= lead - 1
    digits = rstrip(digits[lead:end], '0')
    exp10 = point - 1                        # exponent of the first digit
    if -4 <= exp10 < 16
        if point <= 0
            return sign * "0." * "0"^(-point) * digits
        elseif point >= length(digits)
            return sign * digits * "0"^(point - length(digits)) * ".0"
        end
        return sign * digits[1:point] * "." * digits[(point+1):end]
    end
    mant = length(digits) == 1 ? digits : digits[1:1] * "." * digits[2:end]
    return sign * mant * "e" * (exp10 < 0 ? "-" : "+") * lpad(string(abs(exp10)), 2, '0')
end

# json.dumps(v, ensure_ascii=False, separators=(item, key)).
function _sp_tojson!(io::IO, v, item::String, key::String)
    if v === nothing
        write(io, "null")
    elseif v isa Bool
        write(io, v ? "true" : "false")
    elseif v isa Integer
        write(io, string(v))
    elseif v isa AbstractFloat
        write(io, _sp_py_float(Float64(v)))
    elseif v isa AbstractString
        _json_escape!(io, v)
    elseif v isa AbstractVector
        write(io, "[")
        for (i, x) in enumerate(v)
            i > 1 && write(io, item)
            _sp_tojson!(io, x, item, key)
        end
        write(io, "]")
    elseif v isa AbstractDict
        write(io, "{")
        first = true
        for (k, x) in v
            first || write(io, item)
            first = false
            _json_escape!(io, String(k))
            write(io, key)
            _sp_tojson!(io, x, item, key)
        end
        write(io, "}")
    else
        error("chat template: value of type $(typeof(v)) cannot be rendered")
    end
    return io
end

# The template's render_content: a string, or a list of {type: text} blocks
# joined with nothing between them.
function _sp_content(content, context::String)::String
    content === nothing && return ""
    content isa AbstractString && return String(content)
    if content isa AbstractVector
        io = IOBuffer()
        for block in content
            block isa AbstractDict || error("chat template: unexpected $context content block")
            typ = get(block, "type", nothing)
            typ == "text" ||
                error("chat template: unsupported $context content block type: $(_py_str(typ))")
            text = get(block, "text", "")
            text isa AbstractString || error("chat template: text block must be a string")
            write(io, text)
        end
        return String(take!(io))
    end
    error("chat template: $context content must be a string or a list of text blocks")
end

function _sp_tool_calls!(io::IO, calls)
    calls isa AbstractVector || return io
    for tc in calls
        tc isa AbstractDict || error("chat template: tool call must be an object")
        fn = get(tc, "function", nothing)
        fn isa AbstractDict || error("chat template: tool call is missing 'function'")
        name = get(fn, "name", nothing)
        name isa AbstractString || error("chat template: tool call is missing a string 'name'")
        args = get(fn, "arguments", nothing)
        args isa AbstractDict || error(
            "chat template: tool_call.function.arguments must be a dictionary",
        )
        write(io, "<tool_call>", name)
        for (k, v) in args
            write(io, "<arg_key>", String(k), "</arg_key><arg_value>")
            v isa AbstractString ? write(io, v) : _sp_tojson!(io, v, ", ", ": ")
            write(io, "</arg_value>")
        end
        write(io, "</tool_call>")
    end
    return io
end

"""
    render_spark_chat(messages; tools=nothing, add_generation_prompt=true,
                      enable_thinking=true, reasoning_effort="high") -> String

Render an OpenAI-style message list with the Spark-X2.6 template. The system
block always carries the template's default prompt; a leading system message is
appended to it. `reasoning_effort` is "high" or "low" and only applies with
thinking on.
"""
function render_spark_chat(
    messages::AbstractVector;
    tools=nothing,
    add_generation_prompt::Bool=true,
    enable_thinking::Bool=true,
    reasoning_effort::AbstractString="high",
)::String
    isempty(messages) && error("chat template: no messages provided")
    note = ""
    if enable_thinking
        reasoning_effort in ("high", "low") || error(
            "chat template: unexpected reasoning effort $reasoning_effort; supported types are high (default) and low",
        )
        note = _SP_EFFORT_NOTE * reasoning_effort * "`."
    end
    io = IOBuffer()
    write(io, _SP_BOS, "<|System|>\nyou are a helpful assistant.", note)
    if tools isa AbstractVector && !isempty(tools)
        write(io, "\n\n## Tools\nYou have access to the following functions:\n<tools>")
        for tool in tools
            fn = tool isa AbstractDict ? get(tool, "function", nothing) : nothing
            fn === nothing && error("chat template: tool is missing 'function'")
            write(io, "\n")
            _sp_tojson!(io, fn, ",", ":")
        end
        write(io, "\n</tools>")
    end
    if _role(messages[1]) == "system"
        initial = _sp_content(get(messages[1], "content", nothing), "system")
        isempty(initial) || write(io, "\n\n", initial)
    end
    write(io, _SP_EOS)

    n = length(messages)
    for i in 1:n
        m = messages[i]
        role = _role(m)
        if role == "system"
            i == 1 && continue
            write(io, _SP_BOS, "<|System|>\n", _sp_content(get(m, "content", nothing), "system"), _SP_EOS)
        elseif role == "user"
            write(io, _SP_BOS, "<|User|>", _sp_content(get(m, "content", nothing), "user"), _SP_EOS)
        elseif role == "assistant"
            content = _sp_content(get(m, "content", nothing), "assistant")
            rc = get(m, "reasoning_content", nothing)
            write(io, _SP_BOS, "<|Bot|>")
            if rc isa AbstractString && !isempty(rc)
                write(io, "<think>", rc, "</think>")
            else
                write(io, "</think>")
            end
            write(io, content)
            _sp_tool_calls!(io, get(m, "tool_calls", nothing))
            write(io, _SP_EOS)
        elseif role == "tool"
            (i == 1 || _role(messages[i-1]) != "tool") && write(io, _SP_BOS, "<|Tool|>")
            write(io, "<tool_response>", _sp_content(get(m, "content", nothing), "tool"), "</tool_response>")
            (i == n || _role(messages[i+1]) != "tool") && write(io, _SP_EOS)
        else
            error("chat template: unsupported message role: $role")
        end
    end

    if add_generation_prompt
        write(io, _SP_BOS, "<|Bot|>", enable_thinking ? "<think>" : "</think>")
    end
    return String(take!(io))
end

export render_spark_chat
