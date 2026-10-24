# Provenance: ported from the GessoSymbolicDrafts package, commit 59553aa
# (v0.2.0). Logic is unchanged; locals named `first` were renamed so they no
# longer shadow Base.first.
#
# Include inside the Speculation module AFTER the seam types/functions exist.
# This file imports no package, model, tokenizer implementation or backend.
export SuffixSource,
    TemplateRule,
    TemplateSource,
    SymbolicSource,
    IntegerArithmetic,
    IntegerSequence,
    LastLineProcedure,
    complete_text,
    feedback_counts

abstract type HostDraftSource <: DraftSource end
mutable struct DraftFeedback
    windows::Int
    drafted::Int
    accepted::Int
    lock::ReentrantLock
end
DraftFeedback() = DraftFeedback(0, 0, 0, ReentrantLock())

function observe!(src::HostDraftSource, draft::Draft, accepted::Int)::Nothing
    draft.origin == src.origin || throw(ArgumentError("feedback origin mismatch"))
    0 <= accepted <= length(draft.ids) || throw(ArgumentError("invalid accepted prefix"))
    f = src.feedback
    lock(f.lock) do
        w = Base.Checked.checked_add(f.windows, 1)
        d = Base.Checked.checked_add(f.drafted, length(draft.ids))
        a = Base.Checked.checked_add(f.accepted, accepted)
        f.windows, f.drafted, f.accepted = w, d, a
    end
    nothing
end
function feedback_counts(src::HostDraftSource)
    f = src.feedback
    lock(f.lock) do
        (windows=f.windows, drafted=f.drafted, accepted=f.accepted)
    end
end

function _budget(max_tokens, cap)
    max_tokens >= 0 || throw(ArgumentError("negative draft budget"))
    min(max_tokens, cap)
end
function _valid_ids(ids, vocab_size)
    vocab_size > 0 || throw(ArgumentError("vocabulary must be positive"))
    all(id -> typeof(id) === Int && 0 <= id < vocab_size, ids) ||
        throw(ArgumentError("expected zero-based vocabulary IDs"))
    nothing
end
function _tail(ctx::DraftContext, count::Int)
    Base.require_one_based_indexing(ctx.ids)
    ids = Vector{Int}(@view ctx.ids[max(1, length(ctx.ids)-count+1):end])
    _valid_ids(ids, ctx.vocab_size)
    ids
end

struct SuffixSource <: HostDraftSource
    max_history::Int
    min_match::Int
    max_match::Int
    max_draft::Int
    origin::Symbol
    feedback::DraftFeedback
    # A suffix shorter than `min_match` proposes nothing: one shared token is
    # weak evidence, and a rejected draft still costs a full verify window.
    function SuffixSource(; max_history::Int=4096, min_match::Int=1, max_match::Int=16, max_draft::Int=8)
        1 <= max_history <= 65536 && 1 <= min_match <= max_match <= 128 && 1 <= max_draft <= 64 ||
            throw(ArgumentError("suffix budget outside bounds"))
        new(max_history, min_match, max_match, max_draft, :suffix, DraftFeedback())
    end
end
function propose(src::SuffixSource, ctx::DraftContext; max_tokens::Int)::Draft
    k = _budget(max_tokens, src.max_draft)
    k == 0 && return Draft(Int[], src.origin)
    ids = _tail(ctx, src.max_history)
    _suffix_proposal(ids, src.min_match, src.max_match, k, src.origin)
end
# Concrete host buffer forms a specialization boundary for the hot scan.
function _suffix_proposal(ids::Vector{Int}, min_match::Int, max_match::Int, k::Int, origin::Symbol)::Draft
    n = length(ids)
    # Longest suffix first; most recent earlier occurrence breaks ties.
    for width in min(max_match, n-1):-1:min_match
        start = n-width+1
        for j in (n-width):-1:1
            # A direct loop avoids per-candidate closure allocations on a
            # long no-match scan; retain exactly the same tie-breaking order.
            matches = true
            for t in 1:width
                if ids[j+t-1] != ids[start+t-1]
                    matches = false
                    break
                end
            end
            if matches
                cont = j+width
                return Draft(ids[cont:min(n, cont+k-1)], origin)
            end
        end
    end
    Draft(Int[], origin)
end

struct TemplateRule
    trigger::Tuple{Vararg{Int}}
    continuation::Tuple{Vararg{Int}}
    function TemplateRule(trigger::AbstractVector{Int}, continuation::AbstractVector{Int})
        1 <= length(trigger) <= 128 && 1 <= length(continuation) <= 256 ||
            throw(ArgumentError("template size outside bounds"))
        all(>=(0), trigger) && all(>=(0), continuation) ||
            throw(ArgumentError("negative template token ID"))
        new(Tuple(trigger), Tuple(continuation))
    end
end
struct TemplateSource <: HostDraftSource
    rules::Tuple{Vararg{TemplateRule}}
    max_draft::Int
    origin::Symbol
    feedback::DraftFeedback
    function TemplateSource(rules::AbstractVector{TemplateRule}; max_draft::Int=8)
        1 <= length(rules) <= 128 && 1 <= max_draft <= 64 ||
            throw(ArgumentError("template budget outside bounds"))
        length(unique(r.trigger for r in rules)) == length(rules) ||
            throw(ArgumentError("duplicate template trigger"))
        new(Tuple(rules), max_draft, :template, DraftFeedback())
    end
end
function propose(src::TemplateSource, ctx::DraftContext; max_tokens::Int)::Draft
    k = _budget(max_tokens, src.max_draft)
    k == 0 && return Draft(Int[], src.origin)
    ids = _tail(ctx, 384)
    best = nothing
    best_partial = 0
    best_width = 0
    for rule in src.rules
        _valid_ids(rule.trigger, ctx.vocab_size)
        _valid_ids(rule.continuation, ctx.vocab_size)
        a, b = length(rule.trigger), length(rule.continuation)
        for partial in min(b-1, length(ids)-a):-1:0
            width = a+partial
            width > best_width || continue
            start = length(ids)-width+1
            matches =
                all(t -> ids[start+t-1] == rule.trigger[t], 1:a) &&
                all(t -> ids[start+a+t-1] == rule.continuation[t], 1:partial)
            if matches
                best, best_partial, best_width = rule, partial, width
                break
            end
        end
    end
    best === nothing && return Draft(Int[], src.origin)
    stop = min(length(best.continuation), best_partial+k)
    Draft(Int[best.continuation[i] for i in (best_partial+1):stop], src.origin)
end

# Trusted Julia extension point. Return only the not-yet-emitted text, or nothing.
function complete_text end

# Explicit semantic contract: the wrapped procedure depends only on the last
# line. No inference about arbitrary procedure context requirements is made.
struct LastLineProcedure{P}
    procedure::P
end
complete_text(proc::LastLineProcedure, prefix::AbstractString) =
    complete_text(proc.procedure, last(split(prefix, '\n'; keepempty=true)))

function _selected_context(proc, history, detok, token_limit, byte_limit, vocab)
    length(history) <= token_limit || return nothing
    ids = Vector{Int}(history)
    _valid_ids(ids, vocab)
    text = IOBuffer()
    for id in ids
        part = detok(id)
        part isa AbstractString || throw(ArgumentError("detok must return token text"))
        position(text)+ncodeunits(part) <= byte_limit || return nothing
        write(text, part)
    end
    (ids, String(take!(text)))
end
function _selected_context(
    proc::LastLineProcedure,
    history,
    detok,
    token_limit,
    byte_limit,
    vocab,
)
    pieces = String[]
    count = 0
    found = false
    earliest = length(history)+1
    for i in length(history):-1:max(1, length(history)-token_limit+1)
        id = history[i]
        0 <= id < vocab || throw(ArgumentError("expected zero-based vocabulary IDs"))
        part = detok(id)
        part isa AbstractString || throw(ArgumentError("detok must return token text"))
        count += ncodeunits(part)
        count <= byte_limit || return nothing
        push!(pieces, String(part))
        earliest = i
        # Inspect raw bytes, since individual token pieces can split UTF-8.
        if 0x0a in codeunits(part)
            found = true
            break
        end
    end
    (found || earliest == 1 || isempty(history)) || return nothing
    ids = Vector{Int}(@view history[earliest:end])
    (ids, join(Iterators.reverse(pieces)))
end
struct SymbolicSource{P, E} <: HostDraftSource
    procedure::P
    encode::E
    max_context_tokens::Int
    max_text_bytes::Int
    max_draft::Int
    origin::Symbol
    feedback::DraftFeedback
    function SymbolicSource(
        procedure,
        encode;
        max_context_tokens::Int=4096,
        max_text_bytes::Int=32768,
        max_draft::Int=8,
        origin::Symbol=:symbolic,
    )
        1 <= max_context_tokens <= 65536 &&
        1 <= max_text_bytes <= 262144 &&
        1 <= max_draft <= 64 || throw(ArgumentError("symbolic budget outside bounds"))
        new{typeof(procedure), typeof(encode)}(
            procedure,
            encode,
            max_context_tokens,
            max_text_bytes,
            max_draft,
            origin,
            DraftFeedback(),
        )
    end
end
function propose(src::SymbolicSource, ctx::DraftContext; max_tokens::Int)::Draft
    k = _budget(max_tokens, src.max_draft)
    k == 0 && return Draft(Int[], src.origin)
    Base.require_one_based_indexing(ctx.ids)
    ctx.vocab_size > 0 || throw(ArgumentError("vocabulary must be positive"))
    _symbolic_proposal(src, ctx.ids, ctx.detok, ctx.vocab_size, k)
end
# Specialize the codec loop on the actual callable and concrete history type,
# rather than repeatedly dispatch through DraftContext's untyped detok field.
function _symbolic_proposal(
    src::SymbolicSource,
    history,
    detok::F,
    vocab::Int,
    k::Int,
)::Draft where {F}
    selected = _selected_context(
        src.procedure,
        history,
        detok,
        src.max_context_tokens,
        src.max_text_bytes,
        vocab,
    )
    selected === nothing && return Draft(Int[], src.origin)
    ids, prefix = selected
    isvalid(prefix) || return Draft(Int[], src.origin)
    suffix = complete_text(src.procedure, prefix)
    suffix === nothing && return Draft(Int[], src.origin)
    suffix isa AbstractString ||
        throw(ArgumentError("procedure must return text or nothing"))
    isempty(suffix) && return Draft(Int[], src.origin)
    ncodeunits(prefix)+ncodeunits(suffix) <= src.max_text_bytes ||
        return Draft(Int[], src.origin)
    # Encoding a suffix in isolation can change BPE boundaries. Refuse a rewrite
    # of an existing ID, and refuse a noncanonical/special-token prefix.
    canonical = src.encode(prefix)
    canonical isa AbstractVector{Int} ||
        throw(ArgumentError("encode must return integer IDs"))
    canonical == ids || return Draft(Int[], src.origin)
    combined = src.encode(prefix*String(suffix))
    combined isa AbstractVector{Int} ||
        throw(ArgumentError("encode must return integer IDs"))
    Base.require_one_based_indexing(combined)
    length(combined) <= src.max_context_tokens+src.max_text_bytes ||
        throw(ArgumentError("encoder output exceeds budget"))
    _valid_ids(combined, vocab)
    n = length(ids)
    length(combined) > n && all(i -> combined[i] == ids[i], 1:n) ||
        return Draft(Int[], src.origin)
    Draft(collect(@view combined[(n+1):min(length(combined), n+k)]), src.origin)
end

struct IntegerArithmetic
    max_expression_bytes::Int
    max_nodes::Int
    max_depth::Int
    max_bits::Int
    function IntegerArithmetic(;
        max_expression_bytes::Int=256,
        max_nodes::Int=128,
        max_depth::Int=16,
        max_bits::Int=256,
    )
        1 <= max_expression_bytes <= 4096 &&
        1 <= max_nodes <= 1024 &&
        1 <= max_depth <= 64 &&
        1 <= max_bits <= 4096 || throw(ArgumentError("arithmetic budget outside bounds"))
        new(max_expression_bytes, max_nodes, max_depth, max_bits)
    end
end
struct ArithmeticRefusal <: Exception end
mutable struct ArithmeticCursor
    bytes::Vector{UInt8}
    pos::Int
    nodes::Int
    limits::IntegerArithmetic
end
function _space!(c)
    while c.pos <= length(c.bytes) && c.bytes[c.pos] in (0x20, 0x09)
        c.pos += 1
    end
end
function _bounded(c, value::BigInt)
    ndigits(abs(value); base=2) <= c.limits.max_bits || throw(ArithmeticRefusal())
    value
end
function _atom!(c, depth)
    c.nodes += 1
    c.nodes <= c.limits.max_nodes && depth <= c.limits.max_depth ||
        throw(ArithmeticRefusal())
    _space!(c)
    c.pos <= length(c.bytes) || throw(ArithmeticRefusal())
    ch = c.bytes[c.pos]
    if ch in (0x2b, 0x2d)
        c.pos += 1
        value = _atom!(c, depth+1)
        return ch == 0x2d ? -value : value
    elseif ch == 0x28
        c.pos += 1
        value = _expression!(c, depth+1)
        _space!(c)
        c.pos <= length(c.bytes) && c.bytes[c.pos] == 0x29 || throw(ArithmeticRefusal())
        c.pos += 1
        return value
    elseif 0x30 <= ch <= 0x39
        value = BigInt(0)
        while c.pos <= length(c.bytes) && 0x30 <= c.bytes[c.pos] <= 0x39
            value = _bounded(c, value*10+Int(c.bytes[c.pos]-0x30))
            c.pos += 1
        end
        return value
    end
    throw(ArithmeticRefusal())
end
function _product!(c, depth)
    value = _atom!(c, depth)
    while true
        _space!(c)
        c.pos <= length(c.bytes) && c.bytes[c.pos] == 0x2a || return value
        c.pos += 1
        value = _bounded(c, value*_atom!(c, depth))
    end
end
function _expression!(c, depth)
    value = _product!(c, depth)
    while true
        _space!(c)
        c.pos <= length(c.bytes) && c.bytes[c.pos] in (0x2b, 0x2d) || return value
        op = c.bytes[c.pos]
        c.pos += 1
        rhs = _product!(c, depth)
        value = _bounded(c, op == 0x2b ? value+rhs : value-rhs)
    end
end
function complete_text(proc::IntegerArithmetic, prefix::AbstractString)
    line = last(split(prefix, '\n'; keepempty=true))
    m = match(r"^calc: (.+?) = ([0-9-]*)$", line)
    m === nothing && return nothing
    expression, emitted = m.captures
    ncodeunits(expression) <= proc.max_expression_bytes || return nothing
    c = ArithmeticCursor(collect(codeunits(expression)), 1, 0, proc)
    value = try
        result = _expression!(c, 1)
        _space!(c)
        c.pos > length(c.bytes) || throw(ArithmeticRefusal())
        result
    catch e
        e isa ArithmeticRefusal || rethrow()
        return nothing
    end
    answer = string(value)
    startswith(answer, emitted) || return nothing
    # Result grammar is ASCII; byte indexing is exact here.
    answer[(ncodeunits(emitted)+1):end]
end

# A bounded hypothesis source, not a proof of a sequence's unique next term.
struct IntegerSequence
    min_terms::Int
    max_terms::Int
    max_input_bytes::Int
    max_bits::Int
    function IntegerSequence(;
        min_terms::Int=4,
        max_terms::Int=32,
        max_input_bytes::Int=512,
        max_bits::Int=256,
    )
        4 <= min_terms <= max_terms <= 128 &&
        1 <= max_input_bytes <= 4096 &&
        1 <= max_bits <= 4096 || throw(ArgumentError("sequence budget outside bounds"))
        new(min_terms, max_terms, max_input_bytes, max_bits)
    end
end
function complete_text(proc::IntegerSequence, prefix::AbstractString)
    line = last(split(prefix, '\n'; keepempty=true))
    ncodeunits(line) <= proc.max_input_bytes || return nothing
    m = match(r"^seq: (.+?) -> ([0-9-]*)$", line)
    m === nothing && return nothing
    parts = split(m.captures[1], ',')
    proc.min_terms <= length(parts) <= proc.max_terms || return nothing
    terms = BigInt[]
    for part in parts
        token = strip(part)
        occursin(r"^[+-]?[0-9]+$", token) || return nothing
        value = parse(BigInt, token)
        ndigits(abs(value); base=2) <= proc.max_bits || return nothing
        push!(terms, value)
    end
    differences = diff(terms)
    all(x->ndigits(abs(x); base=2)<=proc.max_bits, differences) || return nothing
    candidate = if all(==(differences[1]), differences)
        terms[end]+differences[end]
    else
        second = diff(differences)
        all(x->ndigits(abs(x); base=2)<=proc.max_bits, second) || return nothing
        if all(==(second[1]), second)
            terms[end]+differences[end]+second[end]
        elseif length(terms)>=5 && all(i->terms[i]==terms[i-1]+terms[i-2], 3:length(terms))
            terms[end]+terms[end-1]
        else
            return nothing
        end
    end
    ndigits(abs(candidate); base=2) <= proc.max_bits || return nothing
    answer = string(candidate)
    emitted = m.captures[2]
    startswith(answer, emitted) || return nothing
    answer[(ncodeunits(emitted)+1):end]
end
