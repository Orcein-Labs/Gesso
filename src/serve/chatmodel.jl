# What the server needs from an engine. Nothing here knows about weights or
# GPUs. The caller supplies four pieces:
#
#   tokenize(text::String) -> Vector{Int}
#   detokenize(ids::Vector{Int}) -> String
#       the whole generated sequence, so byte-level pieces can be joined
#   render(messages::Vector{Dict{String,Any}}, tools::Vector{Any}, thinking::Bool) -> String
#       the chat template, with tool-call arguments already parsed to mappings
#   generate(prompt_ids::Vector{Int}; max_new_tokens::Int, on_token, sampling=nothing) -> (generated, cached)
#       `on_token(id)` returns false to stop; `cached` is the number of prompt
#       tokens whose K/V was reused from the previous call. `sampling` is a
#       Sampler when the request set a temperature, and nothing for greedy.
#       It is passed only when set, so engines that ignore it keep working.
#   prefix_len(messages, tools, thinking, ids) -> Int
#       how many leading prompt ids are the system message alone, so the engine
#       can checkpoint there. 0 when there is no such prefix. Optional: the
#       default returns 0, and the engine is then asked for no boundary.
#
# `n_ctx` bounds prompt plus completion, and `model_id` is the name clients send.
# `accepts_tools` is false when the engine's chat format has no tool-call parser;
# the server then refuses a request that names tools rather than misreading them.

"""
    ChatModel(; tokenize, detokenize, render, generate, n_ctx, model_id, prefix_len, accepts_tools)

The engine as seen by the server. See the comment block in chatmodel.jl for the
contract of each function. `prefix_len` defaults to a function that finds no
boundary, so an engine without one is unchanged. `accepts_tools` defaults to true.
`family` names the chat format; the handlers use it to pick the output parser.
`thinking_default` is what a request that does not set `enable_thinking` gets:
the template's own default (Qwen3.5 thinks, Gemma 4 does not).
"""
struct ChatModel
    tokenize::Function
    detokenize::Function
    render::Function
    generate::Function
    n_ctx::Int
    model_id::String
    prefix_len::Function
    accepts_tools::Bool
    family::Symbol
    thinking_default::Bool

    function ChatModel(;
        tokenize::Function,
        detokenize::Function,
        render::Function,
        generate::Function,
        n_ctx::Integer,
        model_id::AbstractString,
        prefix_len::Function=(msgs, tools, thinking, ids) -> 0,
        accepts_tools::Bool=true,
        family::Symbol=:qwen35_xml,
        thinking_default::Bool=true,
    )
        n_ctx > 0 || throw(ArgumentError("n_ctx must be positive, got $n_ctx"))
        isempty(model_id) && throw(ArgumentError("model_id must not be empty"))
        return new(
            tokenize,
            detokenize,
            render,
            generate,
            Int(n_ctx),
            String(model_id),
            prefix_len,
            accepts_tools,
            family,
            thinking_default,
        )
    end
end
