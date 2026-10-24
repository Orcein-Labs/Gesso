# The Gesso engine behind a ChatModel: a GGUF tokenizer, its chat template and
# reference_generate over one GenerateState that lives as long as the server.
# Weights and the backend are passed in by the caller, so this file names no
# GPU type and the package stays free of CUDA.

using ..Inference: GGUFBPE, render_chat, reference_generate, GenerateState, Sampler
import ..Inference: encode, decode

"""
    gguf_chat_model(model, tensors, spec, tk::GGUFBPE; backend, n_ctx, model_id) -> ChatModel

Wrap a loaded GGUF model as a ChatModel. `model`, `tensors` and `spec` are what
`load_gguf` returns, with `tensors` already placed on `backend`. Requests are
serialized by the server, so the one GenerateState is never used by two calls at
once.
"""
function gguf_chat_model(
    model,
    tensors,
    spec,
    tk::GGUFBPE;
    backend,
    n_ctx::Integer,
    model_id::AbstractString,
)
    n_ctx > 0 || throw(ArgumentError("n_ctx must be positive, got $n_ctx"))
    tk.chat_template === nothing && throw(
        ArgumentError(
            "tokenizer has no chat template; the server cannot render chat requests",
        ),
    )
    eos = _engine_eos_ids(tk)
    isempty(eos) && throw(
        ArgumentError("tokenizer names no end-of-turn token; generation would not stop"),
    )
    eps = spec.rms_norm_eps
    state = GenerateState(n_ctx)

    function generate(
        prompt_ids;
        max_new_tokens,
        on_token,
        sampling=nothing,
        checkpoint_at::Integer=0,
    )
        run =
            wrapped -> begin
                info = Ref{Any}(nothing)
                ids = reference_generate(
                    model,
                    tensors,
                    prompt_ids;
                    backend=backend,
                    max_new_tokens=max_new_tokens,
                    info=info,
                    eps=eps,
                    state=state,
                    on_token=wrapped,
                    eos_ids=eos,
                    sampler=sampling,
                    checkpoint_at=checkpoint_at,
                )
                return (ids, info[].reused)
            end
        return _run_generation(run, prompt_ids, eos; on_token=on_token)
    end

    return ChatModel(
        tokenize=s -> encode(tk, s; parse_special=true),
        detokenize=ids -> decode(tk, ids),
        render=(msgs, tools, thinking) -> render_chat(
            msgs;
            tools=tools,
            add_generation_prompt=true,
            enable_thinking=thinking,
        ),
        generate=generate,
        n_ctx=n_ctx,
        model_id=model_id,
        prefix_len=(msgs, tools, thinking, ids) ->
            _system_prefix_len(tk, msgs, tools, thinking, ids),
    )
end

# Length of the prompt's leading system message, rendered alone, when the prompt
# starts with exactly those ids. A new session that shares the system prompt
# then resumes from a checkpoint at that boundary. A conversation whose system
# messages are merged by the template does not begin with the lone rendering,
# so it gets 0 and no boundary.
function _system_prefix_len(
    tk::GGUFBPE,
    msgs,
    tools,
    thinking::Bool,
    ids::AbstractVector{Int},
)
    (isempty(msgs) || get(msgs[1], "role", nothing) != "system") && return 0
    head = encode(
        tk,
        render_chat(
            [msgs[1]];
            tools=tools,
            add_generation_prompt=false,
            enable_thinking=thinking,
        );
        parse_special=true,
    )
    L = length(head)
    return 0 < L < length(ids) && view(ids, 1:L) == head ? L : 0
end

# The end-of-turn ids the sampler must stop on. The GGUF's eos key is not always
# the token the chat template ends a turn with, so the two named turn-end tokens
# are added whenever the vocabulary has them.
function _engine_eos_ids(tk::GGUFBPE)::Vector{Int}
    ids = Int[]
    tk.eos_token_id === nothing || push!(ids, tk.eos_token_id)
    for name in ("<|im_end|>", "<|endoftext|>")
        id = get(tk.vocab, name, nothing)
        id === nothing || push!(ids, id)
    end
    return unique!(ids)
end

# `run(wrapped)` performs one generation and returns (ids, reused): ids is the
# prompt followed by what was sampled, reused is how many prompt tokens came from
# the cache. The stop token is sampled, so it is in ids, but the caller must not
# see it: the server would print it as text, and it would count as a completion
# token.
function _run_generation(
    run::Function,
    prompt_ids::AbstractVector{Int},
    eos::AbstractVector{Int};
    on_token,
)
    wrapped = id -> id in eos ? true : on_token(id)
    ids, reused = run(wrapped)
    generated = Int.(ids[(length(prompt_ids)+1):end])
    if !isempty(generated) && last(generated) in eos
        pop!(generated)
    end
    return (generated=generated, cached=Int(reused))
end
