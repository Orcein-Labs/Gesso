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
#
# `n_ctx` bounds prompt plus completion, and `model_id` is the name clients send.

"""
    ChatModel(; tokenize, detokenize, render, generate, n_ctx, model_id)

The engine as seen by the server. See the comment block in chatmodel.jl for the
contract of each function.
"""
struct ChatModel
    tokenize::Function
    detokenize::Function
    render::Function
    generate::Function
    n_ctx::Int
    model_id::String

    function ChatModel(; tokenize::Function, detokenize::Function, render::Function,
            generate::Function, n_ctx::Integer, model_id::AbstractString)
        n_ctx > 0 || throw(ArgumentError("n_ctx must be positive, got $n_ctx"))
        isempty(model_id) && throw(ArgumentError("model_id must not be empty"))
        return new(tokenize, detokenize, render, generate, Int(n_ctx), String(model_id))
    end
end
