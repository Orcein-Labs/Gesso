# Gemma 4 Chat Template Renderer
# Pinned line-for-line to gemma4_chat_template.jinja.
# Text and tool calls only.

const _G4_WS = (' ', '\t', '\n', '\v', '\f', '\r')
_g4_trim(s::AbstractString) = strip(s, _G4_WS)

_g4_key(d::AbstractDict, str::String, sym::Symbol, default=nothing) =
    haskey(d, str) ? d[str] : haskey(d, sym) ? d[sym] : default

_g4_haskey(d::AbstractDict, str::String, sym::Symbol) =
    haskey(d, str) || haskey(d, sym)

function _g4_format_argument(arg; escape_keys::Bool=true)::String
    if arg === nothing
        return "null"
    elseif arg isa AbstractString
        return "<|\"|>" * String(arg) * "<|\"|>"
    elseif arg isa Bool
        return arg ? "true" : "false"
    elseif arg isa AbstractDict
        ks = sort!(collect(keys(arg)))
        pairs = String[]
        for k in ks
            k_str = escape_keys ? ("<|\"|>" * string(k) * "<|\"|>") : string(k)
            v_str = _g4_format_argument(arg[k]; escape_keys=escape_keys)
            push!(pairs, k_str * ":" * v_str)
        end
        return "{" * join(pairs, ",") * "}"
    elseif arg isa AbstractVector || arg isa Tuple
        items = [_g4_format_argument(x; escape_keys=escape_keys) for x in arg]
        return "[" * join(items, ",") * "]"
    elseif arg isa Integer
        return string(arg)
    elseif arg isa AbstractFloat
        if @isdefined(_fmt_g6)
            return _fmt_g6(Float64(arg))
        else
            return isinteger(arg) ? string(Int(arg)) : string(arg)
        end
    else
        return string(arg)
    end
end

const _G4_STANDARD_KEYS = ("description", "type", "properties", "required", "nullable")

function _g4_format_parameters(properties::AbstractDict, required=nothing; filter_keys::Bool=false)::String
    io = IOBuffer()
    found_first = false
    for key in sort!(collect(keys(properties)))
        k_str = string(key)
        if filter_keys && (k_str in _G4_STANDARD_KEYS)
            continue
        end
        value = properties[key]
        value isa AbstractDict || continue
        if found_first
            write(io, ",")
        end
        found_first = true
        write(io, k_str, ":{")
        add_comma = false

        # description
        desc = _g4_key(value, "description", :description)
        if desc !== nothing && !isempty(desc)
            write(io, "description:<|\"|>", string(desc), "<|\"|>")
            add_comma = true
        end

        # type
        raw_type = _g4_key(value, "type", :type, "")
        vtype = uppercase(string(raw_type))
        if vtype == "STRING"
            enum_val = _g4_key(value, "enum", :enum)
            if enum_val !== nothing && !isempty(enum_val)
                if add_comma
                    write(io, ",")
                else
                    add_comma = true
                end
                write(io, "enum:", _g4_format_argument(enum_val))
            end
        elseif vtype == "ARRAY"
            items = _g4_key(value, "items", :items)
            if items isa AbstractDict && !isempty(items)
                if add_comma
                    write(io, ",")
                else
                    add_comma = true
                end
                write(io, "items:{")
                ns_items_found_first = false
                for item_key in sort!(collect(keys(items)))
                    ik_str = string(item_key)
                    item_value = items[item_key]
                    if item_value !== nothing
                        if ns_items_found_first
                            write(io, ",")
                        end
                        ns_items_found_first = true
                        if ik_str == "properties"
                            write(io, "properties:{")
                            if item_value isa AbstractDict
                                req = _g4_key(items, "required", :required, [])
                                req === nothing && (req = [])
                                write(io, _g4_format_parameters(item_value, req))
                            end
                            write(io, "}")
                        elseif ik_str == "required"
                            write(io, "required:[")
                            if item_value isa AbstractVector
                                for (ri, req_item) in enumerate(item_value)
                                    ri > 1 && write(io, ",")
                                    write(io, "<|\"|>", string(req_item), "<|\"|>")
                                end
                            end
                            write(io, "]")
                        elseif ik_str == "type"
                            if item_value isa AbstractString
                                write(io, "type:", _g4_format_argument(uppercase(item_value)))
                            elseif item_value isa AbstractVector
                                write(io, "type:", _g4_format_argument([uppercase(string(x)) for x in item_value]))
                            else
                                write(io, "type:", _g4_format_argument(item_value))
                            end
                        else
                            write(io, ik_str, ":", _g4_format_argument(item_value))
                        end
                    end
                end
                write(io, "}")
            end
        end

        # nullable
        if _g4_key(value, "nullable", :nullable, false) === true
            if add_comma
                write(io, ",")
            else
                add_comma = true
            end
            write(io, "nullable:true")
        end

        # OBJECT
        if vtype == "OBJECT"
            props = _g4_key(value, "properties", :properties)
            if props isa AbstractDict
                if add_comma
                    write(io, ",")
                else
                    add_comma = true
                end
                write(io, "properties:{")
                req = _g4_key(value, "required", :required, [])
                req === nothing && (req = [])
                write(io, _g4_format_parameters(props, req))
                write(io, "}")
            elseif value isa AbstractDict
                if add_comma
                    write(io, ",")
                else
                    add_comma = true
                end
                write(io, "properties:{")
                req = _g4_key(value, "required", :required, [])
                req === nothing && (req = [])
                write(io, _g4_format_parameters(value, req; filter_keys=true))
                write(io, "}")
            end
            req_obj = _g4_key(value, "required", :required)
            if req_obj isa AbstractVector && !isempty(req_obj)
                if add_comma
                    write(io, ",")
                else
                    add_comma = true
                end
                write(io, "required:[")
                for (ri, req_item) in enumerate(req_obj)
                    ri > 1 && write(io, ",")
                    write(io, "<|\"|>", string(req_item), "<|\"|>")
                end
                write(io, "]")
            end
        end

        if add_comma
            write(io, ",")
        else
            add_comma = true
        end
        write(io, "type:<|\"|>", vtype, "<|\"|>}")
    end
    return String(take!(io))
end

function _g4_format_function_declaration(tool_data::AbstractDict)::String
    fn = _g4_key(tool_data, "function", :function, tool_data)
    name = string(_g4_key(fn, "name", :name, ""))
    desc = string(_g4_key(fn, "description", :description, ""))
    io = IOBuffer()
    write(io, "declaration:", name, "{description:<|\"|>", desc, "<|\"|>")

    params = _g4_key(fn, "parameters", :parameters)
    if params isa AbstractDict && !isempty(params)
        write(io, ",parameters:{")
        props = _g4_key(params, "properties", :properties)
        if props isa AbstractDict && !isempty(props)
            req = _g4_key(params, "required", :required, [])
            req === nothing && (req = [])
            write(io, "properties:{", _g4_format_parameters(props, req), "},")
        end
        req_list = _g4_key(params, "required", :required)
        if req_list isa AbstractVector && !isempty(req_list)
            write(io, "required:[")
            for (i, item) in enumerate(req_list)
                i > 1 && write(io, ",")
                write(io, "<|\"|>", string(item), "<|\"|>")
            end
            write(io, "],")
        end
        ptype = _g4_key(params, "type", :type)
        if ptype !== nothing
            write(io, "type:<|\"|>", uppercase(string(ptype)), "<|\"|>}")
        end
    end

    resp = _g4_key(fn, "response", :response)
    if resp isa AbstractDict
        write(io, ",response:{")
        rdesc = _g4_key(resp, "description", :description)
        if rdesc !== nothing && !isempty(rdesc)
            write(io, "description:<|\"|>", string(rdesc), "<|\"|>,")
        end
        rtype = uppercase(string(_g4_key(resp, "type", :type, "")))
        if rtype == "OBJECT"
            write(io, "type:<|\"|>", rtype, "<|\"|>}")
        end
    end
    write(io, "}")
    return String(take!(io))
end

function _g4_strip_thinking(text::AbstractString)::String
    parts = split(text, "<channel|>")
    res = IOBuffer()
    for part in parts
        if occursin("<|channel>", part)
            write(res, split(part, "<|channel>")[1])
        else
            write(res, part)
        end
    end
    return _g4_trim(String(take!(res)))
end

function _g4_format_tool_response_block(tool_name::AbstractString, response)::String
    io = IOBuffer()
    write(io, "<|tool_response>")
    if response isa AbstractDict
        write(io, "response:", tool_name, "{")
        first = true
        for k in sort!(collect(keys(response)))
            first || write(io, ",")
            first = false
            write(io, string(k), ":", _g4_format_argument(response[k]; escape_keys=false))
        end
        write(io, "}")
    else
        write(io, "response:", tool_name, "{value:", _g4_format_argument(response; escape_keys=false), "}")
    end
    write(io, "<tool_response|>")
    return String(take!(io))
end

"""
    render_gemma4_chat(messages; tools=nothing, add_generation_prompt=true, enable_thinking=false, preserve_thinking=false, bos_token="<bos>") -> String

Render a conversation according to the Gemma 4 Jinja template line-for-line.
"""
function render_gemma4_chat(
    messages::AbstractVector;
    tools=nothing,
    add_generation_prompt::Bool=true,
    enable_thinking::Bool=false,
    preserve_thinking::Bool=false,
    bos_token::AbstractString="<bos>",
)::String
    io = IOBuffer()
    prev_message_type::Union{Nothing, String} = nothing
    prev_non_tool_role::Union{Nothing, String} = nothing

    write(io, bos_token)

    first_role = !isempty(messages) ? string(_g4_key(messages[1], "role", :role, "")) : ""
    has_initial_sys = !isempty(messages) && (first_role in ("system", "developer"))
    has_tools = tools !== nothing && !isempty(tools)

    loop_messages = messages
    if enable_thinking || has_tools || has_initial_sys
        write(io, "<|turn>system\n")
        if enable_thinking
            write(io, "<|think|>\n")
            prev_message_type = "think"
        end
        if has_initial_sys
            c = _g4_key(messages[1], "content", :content, "")
            if c isa AbstractString
                write(io, _g4_trim(c))
            elseif c isa AbstractVector
                for item in c
                    if item isa AbstractDict
                        txt = _g4_key(item, "text", :text)
                        if txt !== nothing
                            write(io, _g4_trim(string(txt)), " ")
                        end
                    end
                end
            end
            loop_messages = messages[2:end]
        end
        if has_tools
            for tool in tools
                write(io, "<|tool>", _g4_trim(_g4_format_function_declaration(tool)), "<tool|>")
            end
            prev_message_type = "tool"
        end
        write(io, "<turn|>\n")
    end

    # Pre-scan: find last user message index (0-based in loop_messages)
    last_user_idx = -1
    for (idx0, msg) in enumerate(loop_messages)
        if string(_g4_key(msg, "role", :role, "")) == "user"
            last_user_idx = idx0 - 1
        end
    end

    # Loop through messages
    n_loop = length(loop_messages)
    for (idx0, message) in enumerate(loop_messages)
        i0 = idx0 - 1
        m_role = string(_g4_key(message, "role", :role, ""))
        if m_role == "tool"
            continue
        end
        prev_message_type = nothing
        role = (m_role == "assistant") ? "model" : m_role

        continue_same_model_turn = (role == "model" && prev_non_tool_role == "assistant")
        if !continue_same_model_turn
            write(io, "<|turn>", role, "\n")
        end

        # Render reasoning/reasoning_content as thinking channel
        reasoning = _g4_key(message, "reasoning", :reasoning)
        if reasoning === nothing || isempty(reasoning)
            reasoning = _g4_key(message, "reasoning_content", :reasoning_content)
        end
        thinking_text = (reasoning !== nothing && !isempty(reasoning)) ? string(reasoning) : nothing

        t_calls = _g4_key(message, "tool_calls", :tool_calls)
        has_tc = t_calls !== nothing && !isempty(t_calls)
        thinking_gate = (i0 > last_user_idx) || (preserve_thinking && has_tc)
        if thinking_text !== nothing && thinking_gate
            write(io, "<|channel>thought\n", thinking_text, "\n<channel|>")
        end

        # Render tool_calls
        if has_tc
            for tool_call in t_calls
                fn = _g4_key(tool_call, "function", :function, tool_call)
                fn_name = string(_g4_key(fn, "name", :name, ""))
                write(io, "<|tool_call>call:", fn_name, "{")
                args = _g4_key(fn, "arguments", :arguments)
                if args isa AbstractDict
                    first_arg = true
                    for k in sort!(collect(keys(args)))
                        first_arg || write(io, ",")
                        first_arg = false
                        write(io, string(k), ":", _g4_format_argument(args[k]; escape_keys=false))
                    end
                elseif args === nothing
                    # empty
                elseif args isa AbstractString
                    argstr = _g4_trim(args)
                    if startswith(argstr, "{") && endswith(argstr, "}") && ncodeunits(argstr) >= 2
                        write(io, argstr[2:end-1])
                    else
                        write(io, args)
                    end
                end
                write(io, "}<tool_call|>")
            end
            prev_message_type = "tool_call"
        end

        ns_tr_out_flag = false
        t_resp = _g4_key(message, "tool_responses", :tool_responses)
        if t_resp !== nothing && !isempty(t_resp)
            for tr in t_resp
                tname = _g4_key(tr, "name", :name, "unknown")
                (tname === nothing || isempty(tname)) && (tname = "unknown")
                resp_val = _g4_key(tr, "response", :response)
                write(io, _g4_format_tool_response_block(string(tname), resp_val))
                ns_tr_out_flag = true
                prev_message_type = "tool_response"
            end
        elseif has_tc
            # Forward-scan consecutive role:tool messages
            k = idx0 + 1
            while k <= n_loop && string(_g4_key(loop_messages[k], "role", :role, "")) == "tool"
                follow = loop_messages[k]
                tname = _g4_key(follow, "name", :name)
                (tname === nothing || isempty(tname)) && (tname = "unknown")
                call_id = _g4_key(follow, "tool_call_id", :tool_call_id)
                if call_id !== nothing
                    for tc in t_calls
                        tc_id = _g4_key(tc, "id", :id)
                        if tc_id == call_id
                            tc_fn = _g4_key(tc, "function", :function, tc)
                            tname = _g4_key(tc_fn, "name", :name, tname)
                            break
                        end
                    end
                end

                tool_body = _g4_key(follow, "content", :content)
                if tool_body isa AbstractString
                    write(io, _g4_format_tool_response_block(string(tname), tool_body))
                elseif tool_body isa AbstractVector
                    s = IOBuffer()
                    for part in tool_body
                        if part isa AbstractDict
                            ptype = _g4_key(part, "type", :type, "")
                            if ptype == "text"
                                write(s, string(_g4_key(part, "text", :text, "")))
                            end
                        end
                    end
                    write(io, _g4_format_tool_response_block(string(tname), String(take!(s))))
                    for part in tool_body
                        if part isa AbstractDict
                            ptype = _g4_key(part, "type", :type, "")
                            if ptype in ("image", "image_url")
                                write(io, "<|image|>")
                            elseif ptype in ("audio", "input_audio")
                                write(io, "<|audio|>")
                            elseif ptype == "video"
                                write(io, "<|video|>")
                            end
                        end
                    end
                else
                    write(io, _g4_format_tool_response_block(string(tname), tool_body))
                end
                ns_tr_out_flag = true
                prev_message_type = "tool_response"
                k += 1
            end
        end

        # captured_content
        cap_io = IOBuffer()
        raw_content = _g4_key(message, "content", :content)
        if raw_content isa AbstractString
            if role == "model"
                write(cap_io, _g4_strip_thinking(raw_content))
            else
                write(cap_io, _g4_trim(raw_content))
            end
        elseif raw_content isa AbstractVector
            for item in raw_content
                if item isa AbstractDict
                    ptype = _g4_key(item, "type", :type, "")
                    if ptype == "text"
                        txt = string(_g4_key(item, "text", :text, ""))
                        if role == "model"
                            write(cap_io, _g4_strip_thinking(txt))
                        else
                            write(cap_io, _g4_trim(txt))
                        end
                    elseif ptype in ("image", "image_url")
                        write(cap_io, "<|image|>")
                    elseif ptype in ("audio", "input_audio")
                        write(cap_io, "<|audio|>")
                    elseif ptype == "video"
                        write(cap_io, "<|video|>")
                    end
                end
            end
        end
        captured_content = String(take!(cap_io))
        write(io, captured_content)
        has_content = !isempty(_g4_trim(captured_content))

        # Forward-scan: find next non-tool message role
        next_nt_role = nothing
        next_nt_found = false
        for j in (idx0 + 1):n_loop
            nr = string(_g4_key(loop_messages[j], "role", :role, ""))
            if nr != "tool"
                next_nt_role = nr
                next_nt_found = true
                break
            end
        end

        continues_into_next = (
            role == "model" &&
            next_nt_role == "assistant" &&
            (!has_tc || ns_tr_out_flag)
        )

        if prev_message_type == "tool_call" && !ns_tr_out_flag
            write(io, "<|tool_response>")
        elseif continues_into_next
            # do not close turn
        elseif !(ns_tr_out_flag && !has_content && !next_nt_found)
            write(io, "<turn|>\n")
        end

        prev_non_tool_role = m_role
    end

    # Generation prompt
    if add_generation_prompt
        if prev_message_type != "tool_response" && prev_message_type != "tool_call"
            write(io, "<|turn>model\n")
            if !enable_thinking
                write(io, "<|channel>thought\n<channel|>")
            end
        elseif prev_message_type == "tool_response" && enable_thinking
            write(io, "<|channel>thought\n")
        end
    end

    return String(take!(io))
end

export render_gemma4_chat
