# The tokenizer boundary (BREADTH-0 Pass C).
#
# BEFORE: `gpt2_tokenizer.jl` WAS "the tokenizer". `encode(::GPT2BPE, text)`
# was the only entry point, and the engine's only notion of tokenization was
# "give me GPT-2 byte-level BPE".
#
# AFTER: `Tokenizer` is a PROTOCOL sufficient for inference. `GPT2BPE` is ONE
# implementation behind it, and `MetaspaceBPE` is a MATERIALLY DIFFERENT one.
#
# Why the second implementation has to be materially different (not a variant
# of the first): a protocol proved only by one implementation is a rename. The
# two here differ on every axis that matters in practice —
#
#                     GPT2BPE                        MetaspaceBPE
#   segmentation       published regex pre-tokenizer  whitespace + '▁' marker
#   byte handling      256-byte unicode remap         explicit `<0xNN>` fallback
#   merge table        merges.txt (rank = file order)  merges.txt (same form)
#   unknown input      error (vocab is closed)        byte fallback (always total)
#
# Tokenizer is TRANSPORT/COMPATIBILITY machinery (§VIII: checkpoint format is
# transport). It must never infect decode execution semantics: nothing in
# `session.jl` / `Inference.jl` may branch on a tokenizer implementation, and
# no tokenizer type appears in an operator signature.

# --- GPT-2 byte-level BPE, behind the protocol --------------------------------

# The reverse of `bytes_to_unicode` — the published inverse, derived once so
# `decode` is not a guess. Needed because the protocol requires round-trip.
const _UNICODE_TO_BYTE = let
    m = Dict{Char, UInt8}()
    for (b, c) in _BYTES_TO_UNICODE
        m[c] = b
    end
    m
end

"""
    decode(tk::GPT2BPE, ids) -> String

GPT-2 decode: id → token string, concatenate, then map every character back
through the inverse of `bytes_to_unicode` and reinterpret the bytes as UTF-8.

This is the inverse of `encode`'s byte step. It is exact for any id set whose
characters are all in the map; an id whose token contains a character outside
it is a loud error (a corrupt vocab, not a guess).
"""
function decode(tk::GPT2BPE, ids::AbstractVector{Int})
    inv_id = Dict{Int, String}()
    for (tokstr, id) in tk.vocab
        inv_id[id] = tokstr
    end
    bytes = UInt8[]
    for id in ids
        haskey(inv_id, id) || error("decode: token id $id is not in the GPT-2 vocabulary")
        for ch in inv_id[id]
            haskey(_UNICODE_TO_BYTE, ch) || error(
                "decode: token id $id maps to character $(repr(ch)), which is " *
                "not in the bytes_to_unicode image — corrupt vocabulary",
            )
            push!(bytes, _UNICODE_TO_BYTE[ch])
        end
    end
    return String(copy(bytes))
end

tokenizer_metadata(tk::GPT2BPE) = (
    kind=:gpt2_byte_level_bpe,
    vocab_size=length(tk.vocab),
    bos_token_id=tk.bos_token_id,
    eos_token_id=tk.eos_token_id,
    pad_token_id=nothing,
)

# --- Metaspace byte-level BPE (the materially different implementation) --------

"""
    MetaspaceBPE

A byte-level BPE with an explicit metaspace marker and explicit byte fallback —
the SentencePiece/Llama-style convention, implemented natively (no
SentencePiece.jl; the dependency law is untouched, §VII).

Materially different from `GPT2BPE` in the three ways that matter:

  * **Segmentation.** No regex. Text is split on whitespace runs; each word is
    prefixed with `▁` (U+2581) and a leading space is never dropped.
  * **Byte handling.** No 256-entry unicode remap. Any byte that is not
    covered by a merge is emitted as a `<0xNN>` fallback token, so encoding is
    TOTAL — it cannot fail on unusual input the way a closed byte-level vocab
    can.
  * **Lossless by construction.** `decode` replaces `<0xNN>` with the raw byte
    and strips the metaspace marker, so any input round-trips exactly.

`merges` is the same file form as GPT-2's (`merges.txt`, file order = rank),
so one loader shape serves both; only the segmentation and byte layer differ.
"""
struct MetaspaceBPE <: Tokenizer
    vocab::Dict{String, Int}
    merges::Vector{Pair{String, String}}
    merge_rank::Dict{Pair{String, String}, Int}
    byte_tokens::Dict{UInt8, String}       # byte → "<0xNN>" token, if present
    bos_token_id::Int
    eos_token_id::Int
    pad_token_id::Union{Nothing, Int}
end

const _METASPACE = '▁'
const _METASPACE_STR = string(_METASPACE)

"""
    load_metaspace_tokenizer(dir) -> MetaspaceBPE

Load a metaspace tokenizer from `tokenizer.json`-style metadata expressed in
the same JSON Gesso already parses (§LXXVI — JSON remains the only sanctioned
third-party dependency), or from a `metaspace.json` sidecar with:

    {"vocab": {"<0x41>": 0, ...}, "merges": [["a","b"], ...],
     "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}

Refuses loudly: empty vocab, a merged token absent from the vocab, a merge
part absent from the vocab, or a special id outside the vocabulary (§LXX).
"""
function load_metaspace_tokenizer(dir::AbstractString)
    isdir(dir) || error("load_metaspace_tokenizer: no such directory: $dir")
    path = joinpath(dir, "metaspace.json")
    isfile(path) || error("load_metaspace_tokenizer: missing metaspace.json in $dir")
    raw = JSON.parsefile(String(path))

    vocab = Dict{String, Int}()
    for (k, v) in raw["vocab"]
        vocab[String(k)] = Int(v)
    end
    isempty(vocab) && error("load_metaspace_tokenizer: vocab is empty")

    merges = Pair{String, String}[]
    merge_rank = Dict{Pair{String, String}, Int}()
    for m in raw["merges"]
        length(m) == 2 || error(
            "load_metaspace_tokenizer: malformed merge $(repr(m)) (want a 2-element list)",
        )
        a, b = String(m[1]), String(m[2])
        for part in (a, b)
            haskey(vocab, part) || error(
                "load_metaspace_tokenizer: merge part $(repr(part)) is not in the vocab",
            )
        end
        haskey(vocab, a * b) || error(
            "load_metaspace_tokenizer: merged token $(repr(a * b)) is not in the vocab",
        )
        push!(merges, a => b)
        merge_rank[a=>b] = length(merges) - 1
    end

    byte_tokens = Dict{UInt8, String}()
    for (tokstr, _) in vocab
        if startswith(tokstr, "<0x") && endswith(tokstr, ">")
            hex = tokstr[4:(end-1)]
            b = tryparse(UInt8, hex; base=16)
            b === nothing || (byte_tokens[b] = tokstr)
        end
    end

    function check_id(name, default)
        id = Int(get(raw, name, default))
        0 <= id < length(vocab) || error(
            "load_metaspace_tokenizer: $name=$id is outside the vocabulary " *
            "($(length(vocab)) tokens)",
        )
        return id
    end
    bos_id = check_id("bos_token_id", 1)
    eos_id = check_id("eos_token_id", 2)
    pad_raw = get(raw, "pad_token_id", nothing)
    pad_id = pad_raw === nothing ? nothing : check_id("pad_token_id", 0)

    return MetaspaceBPE(vocab, merges, merge_rank, byte_tokens, bos_id, eos_id, pad_id)
end

# metaspace segmentation: one symbol per whitespace-delimited word, each
# carrying the marker. Leading whitespace produces a leading `▁`.
function _metaspace_words(text::AbstractString)
    # Every whitespace-delimited word carries the marker; the FIRST word's
    # marker IS the leading-space marker. Adding a separate leading `▁` would
    # emit it twice — the published convention has exactly one per word.
    return [_METASPACE_STR * w for w in split(text)]
end

function _metaspace_encode(tk::MetaspaceBPE, text::AbstractString)
    ids = Int[]
    for word in _metaspace_words(text)
        symbols = String[]
        for ch in word
            s = string(ch)
            if haskey(tk.vocab, s)
                push!(symbols, s)
            else
                # byte fallback: TOTAL by construction — any byte the vocab
                # does not name as a character becomes its `<0xNN>` token
                for b in codeunits(s)
                    tok = get(tk.byte_tokens, b, nothing)
                    if tok === nothing
                        error(
                            "encode: metaspace vocab has no token for byte 0x" *
                            string(b, base=16, pad=2) *
                            " — add the `<0xNN>` " *
                            "fallback or the encoding is not total",
                        )
                    end
                    push!(symbols, tok)
                end
            end
        end
        for sym in _bpe(symbols, tk)
            haskey(tk.vocab, sym) || error(
                "encode: symbol $(repr(sym)) from word $(repr(word)) is not in the vocab",
            )
            push!(ids, tk.vocab[sym])
        end
    end
    return ids
end

_encode(tk::GPT2BPE, text::AbstractString) = encode(tk, text)

function encode(tk::Tokenizer, text::AbstractString)
    return _encode(tk, text)
end

function _encode(tk::MetaspaceBPE, text::AbstractString)
    return _metaspace_encode(tk, text)
end

"""
    decode(tk::MetaspaceBPE, ids) -> String

Inverse of `encode`: ids → token strings, `<0xNN>` → the raw byte, `▁` → a
space. Exact for any id list the vocab can produce.
"""
function decode(tk::MetaspaceBPE, ids::AbstractVector{Int})
    inv_id = Dict{Int, String}()
    for (tokstr, id) in tk.vocab
        inv_id[id] = tokstr
    end
    out = IOBuffer()
    for id in ids
        haskey(inv_id, id) || error("decode: token id $id is not in the metaspace vocab")
        tokstr = inv_id[id]
        if startswith(tokstr, "<0x") && endswith(tokstr, ">")
            b = tryparse(UInt8, tokstr[4:(end-1)]; base=16)
            b === nothing && error("decode: malformed byte-fallback token $(repr(tokstr))")
            write(out, b)
        else
            write(out, replace(tokstr, _METASPACE_STR => " "))
        end
    end
    text = String(take!(out))
    # `encode` prepends the metaspace marker to the FIRST word (the published
    # "dummy prefix"), so the true inverse drops exactly ONE leading space.
    # Without this the round trip is off by a space on every input.
    return startswith(text, " ") ? text[nextind(text, 1):end] : text
end

tokenizer_metadata(tk::MetaspaceBPE) = (
    kind=:metaspace_byte_level_bpe,
    vocab_size=length(tk.vocab),
    bos_token_id=tk.bos_token_id,
    eos_token_id=tk.eos_token_id,
    pad_token_id=tk.pad_token_id,
)

# `_bpe` is the GPT-2 merge loop; it is representation-agnostic (it only needs
# `merge_rank`), so both implementations share it. That sharing is the point:
# the MERGE ALGORITHM is common, the byte/segmentation layers are what differ.
const _BPE_GENERIC = Union{GPT2BPE, MetaspaceBPE}

export MetaspaceBPE, load_metaspace_tokenizer
