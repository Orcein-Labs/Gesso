# GGUF byte-level BPE tokenizer for the qwen35 pre-tokenizer (Ornith-35B).
#
# The tokenizer section of a GGUF header (tokens, token types, merges, special
# ids, chat template) is read by its own scan. gguf_header and load_gguf are
# untouched, so loading weights costs nothing extra.
#
# The behaviour is pinned to llama.cpp, not to the published regex:
#   * pre-tokenizer: unicode_regex_split_custom_qwen35 (src/unicode.cpp). The
#     PCRE string in llama-vocab.cpp is only its documentation; the custom
#     splitter is what runs. Its \s is the explicit unicode_set_whitespace set,
#     because Julia's \s is ASCII-only.
#   * specials: the CONTROL, USER_DEFINED and UNKNOWN tokens, longest text
#     first. Each one splits every raw fragment left by the ones before it
#     (tokenizer_st_partition). parse_special=false leaves CONTROL and UNKNOWN
#     as plain text; USER_DEFINED is still split.
#   * merges: the lowest-rank adjacent pair merges first. Every occurrence of
#     that pair merges left to right without overlap, then the next rank.
#   * decode: NORMAL tokens map back through bytes_to_unicode. CONTROL,
#     USER_DEFINED and UNKNOWN tokens emit their text verbatim. UNUSED tokens
#     are an error.

const _Q35_OUT = typemax(UInt32)
const _Q35_N = UInt8(0x01)    # \p{N}
const _Q35_L = UInt8(0x02)    # \p{L}
const _Q35_M = UInt8(0x04)    # \p{M}
const _Q35_WS = UInt8(0x08)   # \s, unicode_set_whitespace
# Set on every in-range codepoint. llama.cpp flags are never all-zero for a
# table entry (an unassigned one still carries UNDEFINED), so its
# `flags.as_uint()` test only fails outside the fragment.
const _Q35_ANY = UInt8(0x10)

const _GGUF_TOK_NORMAL = Int32(1)
const _GGUF_TOK_UNKNOWN = Int32(2)
const _GGUF_TOK_CONTROL = Int32(3)
const _GGUF_TOK_USER = Int32(4)
const _GGUF_TOK_UNUSED = Int32(5)

struct _GGUFSpecial
    text::String
    id::Int
    control::Bool        # skipped when parse_special=false
end

"""
    GGUFBPE

Byte-level BPE built from a GGUF tokenizer section. Ids are 0-based. The
merge table is keyed by the packed pair of token ids, so encoding never builds
a string per symbol. `specials` holds every CONTROL, USER_DEFINED and UNKNOWN
token with non-empty text, longest first.
"""
struct GGUFBPE <: Tokenizer
    tokens::Vector{String}                  # id + 1 → token text as stored in GGUF
    token_type::Vector{Int32}
    vocab::Dict{String, Int}                # token text → 0-based id
    byte_id::Vector{Int}                    # byte b (index b + 1) → id of its single-char token
    merge_rank::Dict{UInt64, Int}           # packed (left, right) → rank (1-based)
    merge_left::Vector{Int}                 # rank → left id
    merge_right::Vector{Int}                # rank → right id
    merge_target::Vector{Int}               # rank → merged id
    specials::Vector{_GGUFSpecial}
    bos_token_id::Union{Nothing, Int}
    eos_token_id::Union{Nothing, Int}
    pad_token_id::Union{Nothing, Int}
    chat_template::Union{Nothing, String}
end

# --- the GGUF tokenizer section ------------------------------------------------

function _load_gguf_tokenizer_impl(path::AbstractString)
    isfile(path) || error("load_gguf_tokenizer: no such file: $path")
    data = open(path, "r") do io
        Mmap.mmap(io, Vector{UInt8}; grow=false, shared=false)
    end
    c = _Cur(data, 1)
    magic = (_u8(c), _u8(c), _u8(c), _u8(c))
    magic == _GGUF_MAGIC || error("load_gguf_tokenizer: not a GGUF file: $path")
    version = _u32(c)
    version >= 2 || error("load_gguf_tokenizer: GGUF version $version is too old")
    _u64(c)
    n_kv = Int(_u64(c))
    kv = Dict{String, Any}()
    for _ in 1:n_kv
        k = _str(c)
        t = _u32(c)
        if !startswith(k, "tokenizer.")
            _skip_value!(c, t)
        elseif k == "tokenizer.ggml.tokens" || k == "tokenizer.ggml.merges"
            t == UInt32(9) || error("load_gguf_tokenizer: $k is not an array")
            at = _u32(c)
            at == UInt32(8) || error("load_gguf_tokenizer: $k elements are not strings")
            n = Int(_u64(c))
            arr = Vector{String}(undef, n)
            for i in 1:n
                arr[i] = _str(c)
            end
            kv[k] = arr
        elseif k == "tokenizer.ggml.token_type"
            t == UInt32(9) || error("load_gguf_tokenizer: $k is not an array")
            at = _u32(c)
            at == UInt32(5) || error("load_gguf_tokenizer: $k elements are not int32")
            n = Int(_u64(c))
            arr = Vector{Int32}(undef, n)
            for i in 1:n
                arr[i] = reinterpret(Int32, _u32(c))
            end
            kv[k] = arr
        elseif t == UInt32(9)
            _skip_value!(c, t)
        else
            kv[k] = _read_value(c, t)
        end
    end

    model = get(kv, "tokenizer.ggml.model", nothing)
    model == "gpt2" || error("load_gguf_tokenizer: tokenizer.ggml.model is $(repr(model)), want \"gpt2\"")
    pre = get(kv, "tokenizer.ggml.pre", nothing)
    pre == "qwen35" || error("load_gguf_tokenizer: tokenizer.ggml.pre is $(repr(pre)), want \"qwen35\"")
    haskey(kv, "tokenizer.ggml.tokens") || error("load_gguf_tokenizer: no tokenizer.ggml.tokens in $path")
    haskey(kv, "tokenizer.ggml.token_type") || error("load_gguf_tokenizer: no tokenizer.ggml.token_type in $path")
    haskey(kv, "tokenizer.ggml.merges") || error("load_gguf_tokenizer: no tokenizer.ggml.merges in $path")

    tokens = kv["tokenizer.ggml.tokens"]
    types = kv["tokenizer.ggml.token_type"]
    length(types) == length(tokens) ||
        error("load_gguf_tokenizer: token_type has $(length(types)) entries, tokens has $(length(tokens))")
    isempty(tokens) && error("load_gguf_tokenizer: tokenizer.ggml.tokens is empty")

    vocab = Dict{String, Int}()
    sizehint!(vocab, length(tokens))
    for i in eachindex(tokens)
        t = tokens[i]
        j = get(vocab, t, nothing)
        if j === nothing
            vocab[t] = i - 1
        elseif types[j+1] == _GGUF_TOK_UNUSED && types[i] != _GGUF_TOK_UNUSED
            vocab[t] = i - 1
        elseif types[j+1] != _GGUF_TOK_UNUSED && types[i] != _GGUF_TOK_UNUSED
            error("load_gguf_tokenizer: duplicate token text $(repr(t)) at ids $j and $(i - 1)")
        end
    end

    byte_id = Vector{Int}(undef, 256)
    for b in 0:255
        s = string(_BYTES_TO_UNICODE[UInt8(b)])
        id = get(vocab, s, nothing)
        id === nothing && error("load_gguf_tokenizer: byte $b has no single-character token $(repr(s))")
        byte_id[b+1] = id
    end

    merge_rank = Dict{UInt64, Int}()
    merge_left = Int[]
    merge_right = Int[]
    merge_target = Int[]
    for line in kv["tokenizer.ggml.merges"]
        parts = split(line, ' ')
        length(parts) == 2 || error("load_gguf_tokenizer: malformed merge $(repr(line))")
        a_s, b_s = String(parts[1]), String(parts[2])
        a = get(vocab, a_s, nothing)
        b = get(vocab, b_s, nothing)
        m = get(vocab, a_s * b_s, nothing)
        (a === nothing || b === nothing || m === nothing) &&
            error("load_gguf_tokenizer: merge $(repr(line)) names a token missing from the vocabulary")
        key = (UInt64(a) << 32) | UInt64(b)
        haskey(merge_rank, key) && error("load_gguf_tokenizer: duplicate merge $(repr(line))")
        push!(merge_left, a)
        push!(merge_right, b)
        push!(merge_target, m)
        merge_rank[key] = length(merge_left)
    end

    specials = _GGUFSpecial[]
    for i in eachindex(tokens)
        ty = types[i]
        if ty == _GGUF_TOK_CONTROL || ty == _GGUF_TOK_USER || ty == _GGUF_TOK_UNKNOWN
            isempty(tokens[i]) && continue
            push!(specials, _GGUFSpecial(tokens[i], i - 1, ty != _GGUF_TOK_USER))
        end
    end
    sort!(specials; by=s -> (-ncodeunits(s.text), s.id))

    _special_id(key) = begin
        v = get(kv, key, nothing)
        v === nothing ? nothing : Int(v)
    end
    return GGUFBPE(
        tokens, types, vocab, byte_id,
        merge_rank, merge_left, merge_right, merge_target,
        specials,
        _special_id("tokenizer.ggml.bos_token_id"),
        _special_id("tokenizer.ggml.eos_token_id"),
        _special_id("tokenizer.ggml.padding_token_id"),
        get(kv, "tokenizer.chat_template", nothing),
    )
end

# --- qwen35 pre-tokenizer ------------------------------------------------------

function _q35_is_ws(c::UInt32)
    (0x09 <= c <= 0x0d) || c == 0x20 || c == 0x85 || c == 0xa0 || c == 0x1680 ||
        (0x2000 <= c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f ||
        c == 0x205f || c == 0x3000
end

function _q35_flags(c::UInt32)::UInt8
    f = _Q35_ANY
    _q35_is_ws(c) && (f |= _Q35_WS)
    cat = Base.Unicode.category_code(Char(c))
    if 1 <= cat <= 5
        f |= _Q35_L
    elseif 6 <= cat <= 8
        f |= _Q35_M
    elseif 9 <= cat <= 11
        f |= _Q35_N
    end
    return f
end

_q35_lower(c::UInt32) = (0x41 <= c <= 0x5a) ? c + 0x20 : c

# Port of unicode_regex_split_custom_qwen35 for one segment (the whole fragment).
# Returns half-open codepoint ranges (0-based) in order.
function _qwen35_split(cp::Vector{UInt32})
    n = length(cp)
    fl = Vector{UInt8}(undef, n)
    for i in 1:n
        fl[i] = _q35_flags(cp[i])
    end
    getc(p) = (0 <= p < n) ? cp[p+1] : _Q35_OUT
    getf(p) = (0 <= p < n) ? fl[p+1] : 0x00
    LM = _Q35_L | _Q35_M
    WLMN = _Q35_WS | _Q35_L | _Q35_M | _Q35_N
    out = Tuple{Int, Int}[]
    prev = 0
    pos = 0
    while pos < n
        c = getc(pos)
        f = getf(pos)

        # (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c == 0x27 && pos + 1 < n
            c1 = _q35_lower(getc(pos + 1))
            if c1 == 0x73 || c1 == 0x74 || c1 == 0x6d || c1 == 0x64   # s t m d
                pos += 2
                push!(out, (prev, pos)); prev = pos
                continue
            end
            if pos + 2 < n
                c2 = _q35_lower(getc(pos + 2))
                if ((c1 == 0x72 || c1 == 0x76) && c2 == 0x65) || (c1 == 0x6c && c2 == 0x6c)   # re ve ll
                    pos += 3
                    push!(out, (prev, pos)); prev = pos
                    continue
                end
            end
        end

        # [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if !(c == 0x0d || c == 0x0a || (f & _Q35_N) != 0)
            if (f & LM) != 0 || (getf(pos + 1) & LM) != 0
                pos += 1
                while (getf(pos) & LM) != 0
                    pos += 1
                end
                push!(out, (prev, pos)); prev = pos
                continue
            end
        end

        # \p{N}
        if (f & _Q35_N) != 0
            pos += 1
            push!(out, (prev, pos)); prev = pos
            continue
        end

        # ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`
        f2 = (c == 0x20) ? getf(pos + 1) : f
        if (f2 & WLMN) == 0 && f != 0
            if c == 0x20
                pos += 1
            end
            f2 = getf(pos)
            while (f2 & WLMN) == 0 && f2 != 0
                pos += 1
                f2 = getf(pos)
            end
            c2 = getc(pos)
            while c2 == 0x0d || c2 == 0x0a
                pos += 1
                c2 = getc(pos)
            end
            push!(out, (prev, pos)); prev = pos
            continue
        end

        # whitespace: `\s*[\r\n]+`, then `\s+(?!\S)`, then `\s+`
        nws = 0
        last_nl = 0
        while (getf(pos + nws) & _Q35_WS) != 0
            cc = getc(pos + nws)
            if cc == 0x0d || cc == 0x0a
                last_nl = pos + nws + 1
            end
            nws += 1
        end
        if last_nl > 0
            pos = last_nl
        elseif nws > 1 && getc(pos + nws) != _Q35_OUT
            pos += nws - 1
        elseif nws > 0
            pos += nws
        else
            pos += 1
        end
        push!(out, (prev, pos)); prev = pos
    end
    return out
end

# --- encode ---------------------------------------------------------------------

_pair_key(a::Int, b::Int) = (UInt64(a) << 32) | UInt64(b)

# Lowest-rank-first merge over one pre-token's symbol ids, in place.
function _merge_ids!(syms::Vector{Int}, tk::GGUFBPE)
    n = length(syms)
    while n > 1
        best = typemax(Int)
        for i in 1:(n - 1)
            r = get(tk.merge_rank, _pair_key(syms[i], syms[i+1]), typemax(Int))
            r < best && (best = r)
        end
        best == typemax(Int) && break
        a = tk.merge_left[best]
        b = tk.merge_right[best]
        t = tk.merge_target[best]
        w = 0
        i = 1
        while i <= n
            if i < n && syms[i] == a && syms[i+1] == b
                w += 1
                syms[w] = t
                i += 2
            else
                w += 1
                syms[w] = syms[i]
                i += 1
            end
        end
        resize!(syms, w)
        n = w
    end
    return syms
end

function _encode_raw!(ids::Vector{Int}, tk::GGUFBPE, s::String)
    cps = UInt32[]
    boff = Int[]
    for (i, ch) in pairs(s)
        push!(cps, UInt32(ch))
        push!(boff, i)
    end
    push!(boff, ncodeunits(s) + 1)
    cu = codeunits(s)
    for (a, b) in _qwen35_split(cps)
        syms = Int[]
        for k in boff[a+1]:(boff[b+1]-1)
            push!(syms, tk.byte_id[Int(cu[k]) + 1])
        end
        _merge_ids!(syms, tk)
        append!(ids, syms)
    end
    return ids
end

# Bytes a..b of s as a String. Match boundaries from findnext always fall on
# character boundaries, so the slice is valid UTF-8; string range indexing
# is avoided because it rejects an end byte that is mid-character.
_bytes_slice(s::String, a::Int, b::Int) =
    a > b ? "" : String(collect(view(codeunits(s), a:b)))

# Partition the text into raw fragments and special ids, in the order llama.cpp uses.
function _partition_specials(tk::GGUFBPE, s::String, parse_special::Bool)
    frags = Tuple{Int, String}[(-1, s)]
    for sp in tk.specials
        (!parse_special && sp.control) && continue
        nf = Tuple{Int, String}[]
        for (id, t) in frags
            if id >= 0 || isempty(t)
                push!(nf, (id, t))
                continue
            end
            pos = 1
            while true
                m = findnext(sp.text, t, pos)
                m === nothing && break
                if first(m) > pos
                    push!(nf, (-1, _bytes_slice(t, pos, first(m) - 1)))
                end
                push!(nf, (sp.id, ""))
                pos = last(m) + 1
            end
            pos <= ncodeunits(t) && push!(nf, (-1, _bytes_slice(t, pos, ncodeunits(t))))
        end
        frags = nf
    end
    return frags
end

"""
    encode(tk::GGUFBPE, text; parse_special=true) -> Vector{Int}

Encode `text` to 0-based ids the way llama.cpp does with add_special=false.
With `parse_special=true` (the server default), CONTROL and USER_DEFINED
tokens in the text become their ids. With `parse_special=false`, CONTROL
tokens are plain text; USER_DEFINED tokens are still recognised.
"""
function encode(tk::GGUFBPE, text::AbstractString; parse_special::Bool=true)
    ids = Int[]
    for (id, t) in _partition_specials(tk, String(text), parse_special)
        if id >= 0
            push!(ids, id)
        else
            _encode_raw!(ids, tk, t)
        end
    end
    return ids
end

# --- decode and streaming ---------------------------------------------------------

function _token_bytes!(buf::Vector{UInt8}, tk::GGUFBPE, id::Integer)
    0 <= id < length(tk.tokens) ||
        error("decode: token id $id is outside the vocabulary of $(length(tk.tokens)) tokens")
    i = Int(id) + 1
    ty = tk.token_type[i]
    text = tk.tokens[i]
    if ty == _GGUF_TOK_NORMAL
        for ch in text
            haskey(_UNICODE_TO_BYTE, ch) || error(
                "decode: token id $id maps to character $(repr(ch)), outside the bytes_to_unicode image",
            )
            push!(buf, _UNICODE_TO_BYTE[ch])
        end
    elseif ty == _GGUF_TOK_CONTROL || ty == _GGUF_TOK_USER || ty == _GGUF_TOK_UNKNOWN
        append!(buf, codeunits(text))
    else
        error("decode: token id $id has type $ty (unused), which has no text")
    end
    return buf
end

"""
    decode(tk::GGUFBPE, ids) -> String

Concatenate the text of `ids`. NORMAL tokens are mapped back to bytes through
bytes_to_unicode; special tokens emit their text verbatim.
"""
function decode(tk::GGUFBPE, ids::AbstractVector{<:Integer})
    buf = UInt8[]
    for id in ids
        _token_bytes!(buf, tk, id)
    end
    return String(buf)
end

tokenizer_metadata(tk::GGUFBPE) = (
    kind=:gguf_qwen35_bpe,
    vocab_size=length(tk.tokens),
    bos_token_id=tk.bos_token_id,
    eos_token_id=tk.eos_token_id,
    pad_token_id=tk.pad_token_id,
)

"""
    StreamDetokenizer(tk::GGUFBPE)

Turns ids into text as they arrive. `push!(dec, id)` returns the text that is
complete so far. A UTF-8 sequence split across ids is held back until its
last byte arrives. `finish!(dec)` returns whatever bytes are still held. The
concatenation of every push and the finish equals `decode` of the same ids.
"""
mutable struct StreamDetokenizer
    tk::GGUFBPE
    pending::Vector{UInt8}
end
StreamDetokenizer(tk::GGUFBPE) = StreamDetokenizer(tk, UInt8[])

# Length of the longest prefix of `buf` that ends on a UTF-8 character boundary.
# A stray continuation byte is treated as complete so it cannot stall the stream.
function _complete_utf8_prefix(buf::Vector{UInt8})
    n = length(buf)
    j = n
    while j >= 1 && n - j < 3 && (buf[j] & 0xc0) == 0x80
        j -= 1
    end
    j < 1 && return n
    lead = buf[j]
    (lead & 0xc0) == 0x80 && return n
    need = lead < 0xc0 ? 1 : lead < 0xe0 ? 2 : lead < 0xf0 ? 3 : lead < 0xf8 ? 4 : 1
    return (n - j + 1 >= need) ? n : j - 1
end

function Base.push!(dec::StreamDetokenizer, id::Integer)::String
    _token_bytes!(dec.pending, dec.tk, id)
    n = _complete_utf8_prefix(dec.pending)
    out = String(dec.pending[1:n])
    deleteat!(dec.pending, 1:n)
    return out
end

function finish!(dec::StreamDetokenizer)::String
    out = String(copy(dec.pending))
    empty!(dec.pending)
    return out
end

export GGUFBPE, load_gguf_tokenizer, StreamDetokenizer, finish!

load_gguf_tokenizer(path::AbstractString) =
    _load_boundary(() -> _load_gguf_tokenizer_impl(path), :load_gguf_tokenizer)
