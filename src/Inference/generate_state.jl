# State one reference_generate call hands the next. Without it every turn of
# a conversation re-prefills the whole prompt: 100-130 s at 32k on the 35B.
#
# The GDN recurrent state folds every token it has seen into one matrix and
# cannot be rewound, so on a hybrid model a prompt that leaves `ids` early can
# resume only from a snapshot. One is taken where each call's prefill ends,
# which is the point a regenerate or an edit of the last reply goes back to.
# An attention-only model rewinds to any row: readers take K rows explicitly.

"""
    GenerateState(ctx)

Buffers for `reference_generate(...; state)` that outlive the call: the KV
cache, the GDN conv and recurrent states, and the newest hidden row, sized for
`ctx` tokens of prompt plus generation. Pass the whole token sequence on every
call; the prefix the state already holds is not prefilled again. A state is
bound to the model, backend and KV format of its first call. `close(state)`
frees its device memory.
"""
mutable struct GenerateState
    ctx::Int
    arena::Any        # _ZerosArena owning every buffer below; nothing until first use
    key::Any          # (embedding storage objectid, backend, KV format)
    kc::Any
    vc::Any
    h::Any            # hidden row of ids[end]
    gdn_conv::Any
    gdn_rec::Any
    gdn_scratch::Any
    cpu_scratch::Any
    ids::Vector{Int}  # tokens whose K/V rows and GDN state the buffers hold
    snap::Any         # (conv, rec, h) copies at snap_len
    snap_len::Int     # -1: no snapshot
end

GenerateState(ctx::Integer) = (ctx > 0 || throw(ArgumentError("GenerateState: ctx must be positive, got $ctx"));
    GenerateState(Int(ctx), nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, Int[], nothing, -1))

function Base.close(st::GenerateState)
    st.arena === nothing || _arena_release!(st.arena)
    st.arena = st.key = st.kc = st.vc = st.h = nothing
    st.gdn_conv = st.gdn_rec = st.gdn_scratch = st.cpu_scratch = st.snap = nothing
    empty!(st.ids)
    st.snap_len = -1
    return nothing
end

# A call that throws leaves the buffers half written.
function _generate_invalidate!(st::GenerateState)
    empty!(st.ids)
    st.snap_len = -1
    return nothing
end

_common_prefix(a, b) = something(findfirst(i -> a[i] != b[i], 1:min(length(a), length(b))),
    min(length(a), length(b)) + 1) - 1

function _gdn_copy!(dst, src)
    for (d, s) in zip(dst, src)
        d === nothing || copyto!(d, s)
    end
    return nothing
end

# Prompt rows the state already holds; prefill starts after them. When that
# is 0 the GDN states are zeroed, as a fresh generation's are.
function _generate_resume!(st::GenerateState, prompt::AbstractVector{Int}, hybrid::Bool)
    P = length(prompt)
    L = length(st.ids)
    n0 = if L <= P && view(prompt, 1:L) == st.ids
        L
    elseif !hybrid
        # at least one row re-prefilled, so h is the hidden row of prompt[n0]
        min(_common_prefix(st.ids, prompt), P - 1)
    elseif 0 < st.snap_len <= _common_prefix(st.ids, prompt)
        _gdn_copy!(st.gdn_conv, st.snap.conv)
        _gdn_copy!(st.gdn_rec, st.snap.rec)
        copyto!(st.h.storage, st.snap.h)
        st.snap_len
    else
        0
    end
    if n0 == 0
        for x in Iterators.flatten((st.gdn_conv, st.gdn_rec))
            x === nothing || fill!(x, 0)
        end
    end
    resize!(st.ids, n0)
    return n0
end

function _generate_snapshot!(st::GenerateState, P::Int)
    if st.snap === nothing
        like(xs) = [x === nothing ? nothing : st.arena(size(x)) for x in xs]
        st.snap = (conv=like(st.gdn_conv), rec=like(st.gdn_rec), h=st.arena(size(st.h.storage)))
    end
    _gdn_copy!(st.snap.conv, st.gdn_conv)
    _gdn_copy!(st.snap.rec, st.gdn_rec)
    copyto!(st.snap.h, st.h.storage)
    st.snap_len = P
    return nothing
end
