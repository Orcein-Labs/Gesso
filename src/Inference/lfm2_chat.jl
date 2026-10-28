_lfm_json(x) = sprint(io -> _tojson!(io, x))

function _lfm_content(x, modern::Bool)
    x isa AbstractString && return String(x)
    if !modern
        if x isa AbstractVector && all(item -> item isa AbstractDict && get(item, "type", "") == "text" && haskey(item, "text"), x)
            return join((String(item["text"]) for item in x), "\n")
        end
        return _lfm_json(x)
    end
    x isa AbstractVector || error("LFM chat: structured content must be an array")
    return join(map(x) do item
        item isa AbstractDict || error("LFM chat: expected a content object")
        kind = get(item, "type", "")
        kind == "image" ? "<image>" : kind == "text" ? String(item["text"]) : _lfm_json(item)
    end)
end

function _lfm_repr(x)
    if x isa AbstractString
        return occursin('\'', x) ? _lfm_json(x) : "'" * x * "'"
    elseif x isa AbstractVector
        return "[" * join(_lfm_repr.(x), ", ") * "]"
    elseif x isa AbstractDict
        return "{" * join([_lfm_repr(k) * ": " * _lfm_repr(v) for (k, v) in x], ", ") * "}"
    end
    return _py_str(x)
end

function _lfm_tool_calls(calls)
    rendered = String[]
    for call in calls
        fn = call["function"]
        args = fn["arguments"]
        args isa AbstractDict || error("LFM chat: tool arguments must be an object")
        parts = String[]
        for (k, v) in args
            val = v isa AbstractString ? "'" * v * "'" : v isa AbstractDict ? _lfm_json(v) : _lfm_repr(v)
            push!(parts, String(k) * "=" * val)
        end
        push!(rendered, String(fn["name"]) * "(" * join(parts, ", ") * ")")
    end
    return "<|tool_call_start|>[" * join(rendered, ", ") * "]<|tool_call_end|>"
end

function _render_lfm_chat(template, bos, messages; tools=nothing,
    add_generation_prompt::Bool=true, preserve_thinking::Bool=false)
    isempty(messages) && error("LFM chat: no messages provided")
    family = chat_family(template)
    family in (:lfm2, :lfm25) || error("LFM chat: expected an LFM template")
    modern = family === :lfm25
    # GGUF Jinja source stores either an actual line feed or an escaped `\n`;
    # a doubled escape renders the two literal characters backslash and n.
    raw_newline = "\"<|im_start|>system\n\""
    jinja_escape = "\"<|im_start|>system\\n\""
    literal_escape = "\"<|im_start|>system\\\\n\""
    sep = if occursin(raw_newline, template) || occursin(jinja_escape, template)
        "\n"
    elseif occursin(literal_escape, template)
        "\\n"
    else
        error("LFM chat: unrecognized system delimiter")
    end
    first_message = 1
    system = ""
    if _role(first(messages)) == "system"
        content = get(first(messages), "content", "")
        if content !== nothing && !isempty(content)
            if modern
                system = _lfm_content(content, true)
            else
                content isa AbstractString || error("LFM2 chat: system content must be a string")
                system = String(content)
            end
        end
        first_message = 2
    end
    if tools !== nothing && !isempty(tools)
        ts = [t isa AbstractString ? String(t) : _lfm_json(t) for t in tools]
        system *= (isempty(system) ? "" : sep) * "List of tools: " *
            (modern ? "[" : "<|tool_list_start|>[") * join(ts, ", ") *
            (modern ? "]" : "]<|tool_list_end|>")
    end
    io = IOBuffer()
    write(io, bos)
    isempty(system) || write(io, "<|im_start|>system", sep, system, "<|im_end|>", sep)
    last_user = findlast(m -> _role(m) == "user", messages)
    last_user === nothing && (last_user = 0)
    for i in first_message:length(messages)
        message = messages[i]
        role = _role(message)
        write(io, "<|im_start|>", role, sep)
        if modern && role == "assistant"
            keep = preserve_thinking || i > last_user
            if haskey(message, "thinking") && keep
                write(io, "<think>", String(message["thinking"]), "</think>")
            end
            content = haskey(message, "content") ? _lfm_content(message["content"], true) : ""
            !keep && occursin("</think>", content) && (content = _ct_trim(last(split(content, "</think>"))))
            tag = "CONTINUE_FINAL_MESSAGE_TAG "
            cfm = haskey(message, "tool_calls") && endswith(content, tag)
            write(io, cfm ? chop(content; tail=length(tag)) : content)
            haskey(message, "tool_calls") && write(io, _lfm_tool_calls(message["tool_calls"]))
            cfm && write(io, tag)
        elseif modern
            content = get(message, "content", nothing)
            content === nothing || isempty(content) || write(io, _lfm_content(content, true))
        else
            content = _lfm_content(message["content"], false)
            role == "tool" && (content = "<|tool_response_start|>" * content * "<|tool_response_end|>")
            write(io, content)
        end
        write(io, "<|im_end|>", sep)
    end
    add_generation_prompt && write(io, "<|im_start|>assistant", sep)
    return String(take!(io))
end

"""
    render_chat(tokenizer::GGUFBPE, messages; tools, add_generation_prompt, preserve_thinking)

Render either declared LFM template with the tokenizer's actual BOS token.
`preserve_thinking` retains historical LFM2.5 reasoning; it does not request new reasoning.
"""
function render_chat(tk::GGUFBPE, messages; kwargs...)
    tk.pre === :lfm2 || error("tokenizer-aware render_chat currently supports LFM tokenizers")
    tk.chat_template === nothing && error("LFM chat: tokenizer has no declared template")
    tk.bos_token_id === nothing && error("LFM chat: tokenizer has no BOS token")
    return _render_lfm_chat(tk.chat_template, decode(tk, [tk.bos_token_id]), messages; kwargs...)
end
