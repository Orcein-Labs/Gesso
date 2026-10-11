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
    pending::Any      # host payload of a loaded state, copied in by its first call
    fingerprint::Any  # model identity, stable across processes; nothing until bound
end

GenerateState(ctx::Integer) = (ctx > 0 || throw(ArgumentError("GenerateState: ctx must be positive, got $ctx"));
    GenerateState(Int(ctx), nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, Int[], nothing, -1, nothing, nothing))

function Base.close(st::GenerateState)
    st.arena === nothing || _arena_release!(st.arena)
    st.arena = st.key = st.kc = st.vc = st.h = nothing
    st.gdn_conv = st.gdn_rec = st.gdn_scratch = st.cpu_scratch = st.snap = nothing
    st.pending = st.fingerprint = nothing
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

function _generate_snapshot_buffers!(st::GenerateState)
    if st.snap === nothing
        like(xs) = [x === nothing ? nothing : st.arena(size(x)) for x in xs]
        st.snap = (conv=like(st.gdn_conv), rec=like(st.gdn_rec), h=st.arena(size(st.h.storage)))
    end
    return nothing
end

function _generate_snapshot!(st::GenerateState, P::Int)
    _generate_snapshot_buffers!(st)
    _gdn_copy!(st.snap.conv, st.gdn_conv)
    _gdn_copy!(st.snap.rec, st.gdn_rec)
    copyto!(st.snap.h, st.h.storage)
    st.snap_len = P
    return nothing
end

# Parked states are files. The model identity cannot be objectid (that changes
# every process), and hashing every weight of a 35B model on each first call is
# too slow, so the fingerprint covers every tensor's shape, element type and a
# fixed sample of its elements. A different seed or checkpoint changes the
# sample everywhere, which is the failure a load must refuse.
const _FNV_OFFSET = 0xcbf29ce484222325
const _FNV_PRIME = 0x100000001b3
const _FP_SAMPLE = 1024

function _fnv!(h::UInt64, bytes)
    for b in bytes
        h = (h ⊻ UInt64(b)) * _FNV_PRIME
    end
    return h
end

function _fp_array(h::UInt64, x::AbstractArray{<:Number})
    h = _fnv!(h, codeunits(string(eltype(x))))
    h = _fnv!(h, reinterpret(UInt8, Int64[size(x)...]))
    n = length(x)
    idx = n <= _FP_SAMPLE ? collect(1:n) : round.(Int, range(1, n; length=_FP_SAMPLE))
    return _fnv!(h, reinterpret(UInt8, vec(Array(vec(x)[idx]))))
end

function _fp_walk(h::UInt64, x)
    x === nothing && return _fnv!(h, codeunits("nothing"))
    x isa AbstractArray{<:Number} && return _fp_array(h, x)
    if x isa NamedTuple
        for (k, v) in pairs(x)
            h = _fp_walk(_fnv!(h, codeunits(string(k))), v)
        end
        return h
    elseif x isa Union{AbstractDict,AbstractSet}
        # Slot order follows object addresses, which differ between Julia
        # processes, so each entry is hashed alone and the sum is order-free.
        acc = UInt64(0)
        for e in x
            acc += _fp_walk(_FNV_OFFSET, e)
        end
        return _fnv!(h, reinterpret(UInt8, UInt64[length(x), acc]))
    elseif x isa AbstractArray
        h = _fnv!(h, reinterpret(UInt8, Int64[length(x)]))
        for i in eachindex(x)
            h = isassigned(x, i) ? _fp_walk(h, x[i]) : _fnv!(h, codeunits("#undef"))
        end
        return h
    elseif x isa Union{Bool,Number,AbstractString,Symbol}
        return _fnv!(h, codeunits(repr(x)))
    elseif x isa Union{Function,Type} || fieldcount(typeof(x)) == 0
        return _fnv!(h, codeunits(string(typeof(x))))
    end
    for f in fieldnames(typeof(x))
        h = _fnv!(h, codeunits(string(f)))
        h = isdefined(x, f) ? _fp_walk(h, getfield(x, f)) : _fnv!(h, codeunits("#undef"))
    end
    return h
end

_model_fingerprint(tensors) = _fp_walk(_FNV_OFFSET, tensors)

# A loaded state is bound to the model and KV format it was saved from; a fresh
# state takes the first call's model. Checked before any buffer is allocated,
# so a refused call leaves the state as it was.
function _bind_state!(st::GenerateState, tensors, kv_fmt::Symbol)
    fp = _model_fingerprint(tensors)
    st.fingerprint === nothing || st.fingerprint == fp || throw(gesso_error(ERR_INVALID_PLAN,
        "reference_generate: state was saved from another model"))
    st.pending === nothing || st.pending.fmt == kv_fmt || throw(gesso_error(ERR_INVALID_PLAN,
        "reference_generate: state was saved with KV format $(st.pending.fmt), this call uses $kv_fmt"))
    st.fingerprint = fp
    return nothing
end

# Element types a parked array may hold; the code is its position here.
const _STATE_DTYPES = (Float16, Float32, Float64, Int8, UInt8, Int32, Int64)
const _STATE_MAGIC = Vector{UInt8}("GESSOST1")
const _STATE_VERSION = UInt32(1)

function _state_put(io::IO, x::Nothing)
    write(io, UInt8(0))
    return nothing
end

function _state_put(io::IO, x::AbstractArray{<:Number})
    a = Array(x)
    code = findfirst(==(eltype(a)), _STATE_DTYPES)
    code === nothing && throw(gesso_error(ERR_INVALID_PLAN,
        "save_state: element type $(eltype(a)) is not a parked type"))
    write(io, UInt8(1), UInt8(code), UInt8(ndims(a)))
    for d in size(a)
        write(io, Int64(d))
    end
    write(io, a)
    return nothing
end

# The quantized cache keeps its codes and scales as arrays: the tag says the
# format, so a reader needs no header field to rebuild it.
function _state_put(io::IO, x::QuantKV{F}) where {F}
    write(io, F === :q8 ? UInt8(2) : UInt8(3))
    _state_put(io, x.codes)
    _state_put(io, x.scales)
    write(io, Int64(x.d))
    return nothing
end

function _state_list(io::IO, xs)
    write(io, Int64(length(xs)))
    for x in xs
        _state_put(io, x)
    end
    return nothing
end

_state_refuse(msg::AbstractString) = throw(gesso_error(ERR_INVALID_PLAN, "load_state: " * msg))

function _state_get(io::IO, limit::Integer)
    tag = read(io, UInt8)
    tag == 0x00 && return nothing
    if tag == 0x01
        code = Int(read(io, UInt8))
        nd = Int(read(io, UInt8))
        1 <= code <= length(_STATE_DTYPES) || _state_refuse("unknown element type code $code")
        T = _STATE_DTYPES[code]
        dims = Tuple(Int(read(io, Int64)) for _ in 1:nd)
        all(>=(0), dims) || _state_refuse("negative dimension")
        prod(dims) * sizeof(T) <= limit - position(io) || throw(EOFError())
        a = Array{T}(undef, dims)
        read!(io, a)
        return a
    elseif tag == 0x02 || tag == 0x03
        codes = _state_get(io, limit)
        scales = _state_get(io, limit)
        d = Int(read(io, Int64))
        codes isa AbstractArray && scales isa AbstractArray || _state_refuse("quantized cache without codes")
        return tag == 0x02 ? QuantKV{:q8}(codes, scales, d) : QuantKV{:q4h}(codes, scales, d)
    end
    _state_refuse("bad record tag $tag")
end

function _state_get_list(io::IO, limit::Integer)
    n = Int(read(io, Int64))
    0 <= n <= limit || _state_refuse("bad list length $n")
    return Any[_state_get(io, limit) for _ in 1:n]
end

_state_copy!(dst::QuantKV, src::QuantKV) = (copyto!(dst.codes, src.codes); copyto!(dst.scales, src.scales); nothing)
_state_copy!(dst, src) = (copyto!(dst, src); nothing)

# What a save writes: the live buffers, or the payload of a loaded state that no
# call has consumed yet, so a state can be saved again before it runs.
function _state_payload(st::GenerateState)
    st.pending === nothing || return merge((ctx=st.ctx, fingerprint=st.fingerprint), st.pending)
    st.key === nothing && throw(gesso_error(ERR_INVALID_PLAN,
        "save_state: the state holds no generation to save"))
    return (ctx=st.ctx, fingerprint=st.fingerprint, fmt=st.key[3], ids=st.ids,
        snap_len=st.snap_len, h=st.h.storage,
        kc=[a.storage for a in st.kc], vc=[a.storage for a in st.vc],
        gdn_conv=st.gdn_conv, gdn_rec=st.gdn_rec,
        snap=st.snap_len > 0 ? st.snap : nothing)
end

"""
    save_state(path, state)

Write the cache and recurrent state of `state` to `path`, so a later process can
resume the conversation with `load_state` instead of prefilling it again.
"""
function save_state(path::AbstractString, st::GenerateState)
    p = _state_payload(st)
    p.fingerprint === nothing && throw(gesso_error(ERR_INVALID_PLAN,
        "save_state: the state holds no generation to save"))
    tmp = string(path, ".tmp")
    open(tmp, "w") do io
        fmt = codeunits(string(p.fmt))
        write(io, _STATE_MAGIC, _STATE_VERSION, p.fingerprint, Int64(p.ctx))
        write(io, UInt8(length(fmt)), fmt)
        _state_put(io, p.ids)
        write(io, Int64(p.snap_len))
        _state_put(io, p.h)
        _state_list(io, p.kc)
        _state_list(io, p.vc)
        _state_list(io, p.gdn_conv)
        _state_list(io, p.gdn_rec)
        if p.snap === nothing
            write(io, UInt8(0))
        else
            write(io, UInt8(1))
            _state_list(io, p.snap.conv)
            _state_list(io, p.snap.rec)
            _state_put(io, p.snap.h)
        end
    end
    mv(tmp, path; force=true)
    return nothing
end

"""
    load_state(path, model, tensors; backend=CPUBackend()) -> GenerateState

Read a state written by `save_state`. The returned state holds the payload
without buffers; its first `reference_generate` call allocates them and copies
the payload in, so that call prefills only the tokens past the saved prefix.

Refuses, with `ERR_INVALID_PLAN`, a file from another model, a KV format other
than the one this backend would use, a bad magic or version, a truncated file,
and a file whose stored ctx does not fit its cache.
"""
function load_state(path::AbstractString, model, tensors; backend::AbstractGessoBackend=CPUBackend())
    fmt = _kv_format(_kv_default_format(backend_name(backend), _tensor_head_dim(model, tensors)))
    fp = _model_fingerprint(tensors)
    limit = filesize(path)
    try
        return open(path, "r") do io
            read(io, length(_STATE_MAGIC)) == _STATE_MAGIC ||
                _state_refuse("$path is not a Gesso state file")
            ver = read(io, UInt32)
            ver == _STATE_VERSION || _state_refuse("version $ver, expected $_STATE_VERSION")
            read(io, UInt64) == fp || _state_refuse("saved from another model")
            ctx = Int(read(io, Int64))
            stored_fmt = Symbol(String(read(io, Int(read(io, UInt8)))))
            stored_fmt == fmt || _state_refuse("saved with KV format $stored_fmt, this backend uses $fmt")
            ids = _state_get(io, limit)
            ids isa Vector{Int} || _state_refuse("token list missing")
            snap_len = Int(read(io, Int64))
            h = _state_get(io, limit)
            kc = _state_get_list(io, limit)
            vc = _state_get_list(io, limit)
            gdn_conv = _state_get_list(io, limit)
            gdn_rec = _state_get_list(io, limit)
            snap = if read(io, UInt8) == 0x00
                nothing
            else
                (conv=_state_get_list(io, limit), rec=_state_get_list(io, limit), h=_state_get(io, limit))
            end
            eof(io) || _state_refuse("trailing bytes after the state")

            ctx > 0 || _state_refuse("stored ctx $ctx is not positive")
            length(ids) <= ctx || _state_refuse("stored ctx $ctx holds fewer rows than the saved tokens")
            -1 <= snap_len <= length(ids) || _state_refuse("snapshot length $snap_len is out of range")
            (snap_len > 0) == (snap !== nothing) || _state_refuse("snapshot and its length disagree")
            nb = length(tensors.blocks)
            length(kc) == nb && length(vc) == nb && length(gdn_conv) == nb && length(gdn_rec) == nb ||
                _state_refuse("block count differs from the model")
            for (i, bt) in enumerate(tensors.blocks)
                rows = _block_mixer(bt) === :linear_attention ? 0 :
                    (fmt === :f32 ? ctx : cld(ctx, 4) * 4)
                _rows(x) = x isa AbstractArray ? size(x, 1) : -1
                _rows(kc[i]) == rows && _rows(vc[i]) == rows ||
                    _state_refuse("stored ctx $ctx does not fit block $i's cache rows")
            end

            st = GenerateState(ctx)
            st.fingerprint = fp
            st.pending = (fmt=fmt, ids=ids, snap_len=snap_len, h=h, kc=kc, vc=vc,
                gdn_conv=gdn_conv, gdn_rec=gdn_rec, snap=snap)
            return st
        end
    catch e
        e isa EOFError || rethrow()
        _state_refuse("$path is truncated")
    end
end

# Copies a loaded payload into the buffers the first call has just allocated.
# The payload is taken off the state before it is copied, so a copy that throws
# leaves no pending payload behind to be applied twice.
function _restore_pending!(st::GenerateState)
    p = st.pending
    st.pending = nothing
    for (a, s) in zip(st.kc, p.kc)
        _state_copy!(a.storage, s)
    end
    for (a, s) in zip(st.vc, p.vc)
        _state_copy!(a.storage, s)
    end
    _state_copy!(st.h.storage, p.h)
    _gdn_copy!(st.gdn_conv, p.gdn_conv)
    _gdn_copy!(st.gdn_rec, p.gdn_rec)
    if p.snap !== nothing
        _generate_snapshot_buffers!(st)
        _gdn_copy!(st.snap.conv, p.snap.conv)
        _gdn_copy!(st.snap.rec, p.snap.rec)
        _state_copy!(st.snap.h, p.snap.h)
    end
    empty!(st.ids)
    append!(st.ids, p.ids)
    st.snap_len = p.snap_len
    return nothing
end
