# GPT-2 byte-level BPE tokenizer (§LXXVI item C).
#
# The published GPT-2 algorithm, nothing more:
#   1. the pre-tokenizer regex splits RAW text (contractions, letters,
#      numbers, punctuation runs, whitespace) — before byte mapping, because
#      the mapped space char 'Ġ' (U+0120) is a LETTER category and would
#      corrupt \p{L} chunking if mapping ran first;
#   2. each pre-token's UTF-8 bytes go through bytes_to_unicode (the
#      published 256-byte map: printable ASCII and some latin-1 map to
#      themselves; the rest map to U+0100..U+0143);
#   3. greedy BPE merge by rank from merges.txt (file order IS rank order);
#   4. vocab.json maps the resulting string → id (0-based ids out).
#
# Laws:
#   * Refusals are loud: empty vocab, a merge pair whose parts are not in
#     the vocab, an unknown special-token id in the config (§LXX).
#   * No Tokenizers.jl. No decode this sprint (nice-to-have, not the gate).
#   * JSON is already the sanctioned dependency (§LXXVI) — vocab.json is JSON.

using JSON

# --- bytes_to_unicode: the published GPT-2 256-byte map -----------------------
#
# bs: the byte values that map to themselves — printable ASCII except
# whitespace control chars, plus ¡..¬ and ®..ÿ. Everything else maps to
# U+0100 + next free slot, in byte order.

const _BYTES_TO_UNICODE = let
    bs = UInt8[]
    for b in UInt8('!'):UInt8('~')   # 33..126
        push!(bs, b)
    end
    for b in UInt8(0xa1):UInt8(0xac)
        push!(bs, b)
    end
    for b in UInt8(0xae):UInt8(0xff)
        push!(bs, b)
    end
    cs = copy(bs)
    n = 0
    m = Dict{UInt8, Char}()
    for b in bs
        m[b] = Char(b)
    end
    for b in UInt8(0):UInt8(0xff)
        if b ∉ bs
            m[b] = Char(0x100 + n)
            n += 1
        end
    end
    m
end

_byte_encoder() = _BYTES_TO_UNICODE

# --- pre-tokenizer: the GPT-2 regex -------------------------------------------
#
# PCRE translation of GPT-2's `regex.py` pattern
# ('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+).
# Julia's PCRE supports \p{L}/\p{N} natively; look-ahead (?!\S) too.

const _GPT2_SPLIT_RE =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

# --- tokenizer object ----------------------------------------------------------

"""
    GPT2BPE

A loaded byte-level BPE tokenizer: the bytes_to_unicode map, the ordered
merge list (file order = rank order), the token-string → id vocabulary, and
the special ids from the tokenizer config. Ids are 0-based, as everywhere
in Gesso (§LXXV convention).
"""
struct GPT2BPE <: Tokenizer
    byte_encoder::Dict{UInt8, Char}
    merges::Vector{Pair{String, String}}          # rank order
    merge_rank::Dict{Pair{String, String}, Int}
    vocab::Dict{String, Int}                      # token string → 0-based id
    bos_token_id::Int
    eos_token_id::Int
end

"""
    load_gpt2_tokenizer(dir) -> GPT2BPE

Load `vocab.json` + `merges.txt` (+ optional `tokenizer_config.json` for
`bos_token_id` / `eos_token_id`) from `dir`. Refuses an empty vocab, a
merge pair whose parts are not in the vocab, and special ids outside the
vocab — each with the offending name (§LXX).
"""
function _load_gpt2_tokenizer_impl(dir::AbstractString)
    isdir(dir) || error("load_gpt2_tokenizer: no such directory: $dir")
    vocab_path = joinpath(dir, "vocab.json")
    merges_path = joinpath(dir, "merges.txt")
    isfile(vocab_path) || error("load_gpt2_tokenizer: missing vocab.json in $dir")
    isfile(merges_path) || error("load_gpt2_tokenizer: missing merges.txt in $dir")

    vocab_raw = _strict_jsonfile(vocab_path)
    vocab = Dict{String, Int}()
    for (k, v) in vocab_raw
        vocab[String(k)] = _checkpoint_int(v, "tokenizer id for $(repr(k))")
    end
    isempty(vocab) && error("load_gpt2_tokenizer: vocab.json is empty")

    sort(collect(values(vocab))) == collect(0:(length(vocab)-1)) ||
        error("load_gpt2_tokenizer: vocab IDs must be unique and contiguous from zero")
    merges = Pair{String, String}[]
    merge_rank = Dict{Pair{String, String}, Int}()
    for line in eachline(merges_path)
        line = String(line)
        (isempty(line) || startswith(line, "#version")) && continue
        parts = split(line, ' '; keepempty=false)
        length(parts) == 2 ||
            error("load_gpt2_tokenizer: malformed merge line $(repr(line)) in merges.txt")
        a, b = String(parts[1]), String(parts[2])
        haskey(vocab, a) ||
            error("load_gpt2_tokenizer: merge part $(repr(a)) is not in vocab.json")
        haskey(vocab, b) ||
            error("load_gpt2_tokenizer: merge part $(repr(b)) is not in vocab.json")
        haskey(vocab, a * b) ||
            error("load_gpt2_tokenizer: merged token $(repr(a * b)) is not in vocab.json")
        haskey(merge_rank, a => b) &&
            error("load_gpt2_tokenizer: duplicate merge $(repr(line))")
        rank = length(merges)
        push!(merges, a => b)
        merge_rank[a=>b] = rank
    end

    # special ids: from tokenizer_config.json when present, else 0 (GPT-2
    # has none; the micro configs pin bos/eos at 0 like the toy pack)
    bos_id, eos_id = 0, 0
    cfg_path = joinpath(dir, "tokenizer_config.json")
    if isfile(cfg_path)
        cfg = _strict_jsonfile(cfg_path)
        if haskey(cfg, "bos_token_id")
            bos_id = Int(cfg["bos_token_id"])
            0 <= bos_id < length(vocab) || error(
                "load_gpt2_tokenizer: unknown special-token id bos_token_id=$bos_id (vocab has $(length(vocab)) tokens)",
            )
        end
        if haskey(cfg, "eos_token_id")
            eos_id = Int(cfg["eos_token_id"])
            0 <= eos_id < length(vocab) || error(
                "load_gpt2_tokenizer: unknown special-token id eos_token_id=$eos_id (vocab has $(length(vocab)) tokens)",
            )
        end
    end

    return GPT2BPE(_byte_encoder(), merges, merge_rank, vocab, bos_id, eos_id)
end

# --- BPE core ------------------------------------------------------------------

# greedy lowest-rank merge over one pre-token's symbol list (the published
# algorithm: find the best pair present, merge ALL its occurrences, repeat)
function _bpe(toks::Vector{String}, tk::Tokenizer)
    length(toks) <= 1 && return toks
    work = copy(toks)
    while true
        best = nothing
        best_rank = typemax(Int)
        for i in 1:(length(work)-1)
            r = get(tk.merge_rank, work[i] => work[i+1], nothing)
            if r !== nothing && r < best_rank
                best_rank = r
                best = work[i] => work[i+1]
            end
        end
        best === nothing && break
        a, b = best
        merged = String[]
        i = 1
        n = length(work)
        while i <= n
            if i < n && work[i] == a && work[i+1] == b
                push!(merged, a * b)
                i += 2
            else
                push!(merged, work[i])
                i += 1
            end
        end
        work = merged
        length(work) == 1 && break
    end
    return work
end

# --- encode ---------------------------------------------------------------------

"""
    encode(tokenizer, text) -> Vector{Int}

Encode `text` to 0-based token ids: pre-tokenize raw text with the GPT-2
regex, map each pre-token's UTF-8 bytes through bytes_to_unicode, greedily
merge by rank, then look the resulting strings up in vocab.json. Any
out-of-vocab symbol is a loud error (a correct vocab + merge list always
covers its own tokens — byte fallbacks are the single-byte symbols).
"""
function encode(tk::GPT2BPE, text::AbstractString)
    ids = Int[]
    enc = tk.byte_encoder
    for m in eachmatch(_GPT2_SPLIT_RE, String(text))
        pretok = m.match
        symbols = String[string(enc[byte]) for byte in codeunits(pretok)]
        for sym in _bpe(symbols, tk)
            haskey(tk.vocab, sym) || error(
                "encode: symbol $(repr(sym)) from pre-token $(repr(pretok)) is not in vocab.json",
            )
            push!(ids, tk.vocab[sym])
        end
    end
    return ids
end

export GPT2BPE, load_gpt2_tokenizer, encode

load_gpt2_tokenizer(dir::AbstractString) =
    _load_boundary(() -> _load_gpt2_tokenizer_impl(dir), :load_gpt2_tokenizer)

@doc (@doc _load_gpt2_tokenizer_impl) load_gpt2_tokenizer
