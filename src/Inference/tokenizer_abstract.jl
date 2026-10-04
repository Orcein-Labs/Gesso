# The tokenizer ABSTRACTION (BREADTH-0 Pass C).
#
# Split from tokenizer_protocol.jl because `GPT2BPE` is declared in
# gpt2_tokenizer.jl, which is included earlier — a subtype declaration needs
# its supertype to already exist. This file therefore loads BEFORE
# gpt2_tokenizer.jl and holds only the protocol; the implementations and the
# materially-different second implementation follow.

"""
    Tokenizer

Inference-sufficient tokenizer protocol (BREADTH-0 Pass C). Implementations
provide:

    encode(tk, text)     text → Vector{Int}      (0-based ids, §LXXV)
    decode(tk, ids)      ids → text              (round-trip)
    vocab_size(tk)       Int
    tokenizer_metadata(tk)  NamedTuple with bos/eos/pad + kind
    tokenizer_kind(tk)   Symbol                  (provenance / reporting only)

The vocabulary METADATA is part of the protocol because a Session needs
`eos_token_id` and BOS/PAD semantics; `chat_template` metadata rides along
for transport without becoming execution semantics.
"""
abstract type Tokenizer end

function tokenizer_metadata end

tokenizer_kind(tk::Tokenizer) = tokenizer_metadata(tk).kind
vocab_size(tk::Tokenizer) = tokenizer_metadata(tk).vocab_size

export Tokenizer, tokenizer_metadata, vocab_size
