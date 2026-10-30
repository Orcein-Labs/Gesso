# GGUF transport for qwen35 (dense) and qwen35moe (Ornith-1.5-35B-A3B). Header
# parse and dequant are pure Julia (gguf_dequant.jl); libggml is loaded only
# as a test oracle. Experts stay mmap'd and dequant per selected expert. Large
# 2-D projections and the embedding/head stay quantized as `QuantMatrix`
# (gguf_quant_matrix.jl); small and 1-D tensors materialize as Float32 in
# Gesso `(out, in)` layout.

using Mmap
using Libdl
using LinearAlgebra

const _GGUF_MAGIC = (UInt8('G'), UInt8('G'), UInt8('U'), UInt8('F'))

"""
Quantized expert slab. GGUF stores ggml `(ne0, ne1, n_exp)` with `ne0`
innermost; Gesso's 3D expert layout is `(n_exp, ne1, ne0)` after the
reverse-perm applied to each expert's 2D matrix.
"""
mutable struct GgufQuantSlice
    data::Vector{UInt8}
    offset0::Int                 # 0-based byte offset into `data`
    ggml_type::Int32
    n_exp::Int
    ne0::Int                     # GGUF innermost
    ne1::Int
    bytes_per_expert::Int
    blck_size::Int64
    cache::Dict{Int, Matrix{Float32}}
    cache_order::Vector{Int}
    function GgufQuantSlice(
        data::Vector{UInt8},
        offset0::Int,
        ggml_type::Int32,
        n_exp::Int,
        ne0::Int,
        ne1::Int,
        bytes_per_expert::Int,
        blck_size::Int64,
    )
        return new(
            data,
            offset0,
            ggml_type,
            n_exp,
            ne0,
            ne1,
            bytes_per_expert,
            blck_size,
            Dict{Int, Matrix{Float32}}(),
            Int[],
        )
    end
end

Base.size(s::GgufQuantSlice) = (s.n_exp, s.ne1, s.ne0)
Base.size(s::GgufQuantSlice, d::Int) = size(s)[d]
Base.ndims(::GgufQuantSlice) = 3
Base.eltype(::GgufQuantSlice) = Float32
Base.length(s::GgufQuantSlice) = prod(size(s))

_is_host_float_storage(::GgufQuantSlice) = true

"""
Host-resident weight wrapper produced by `to_device` when a tensor does not
fit the GPU budget. Distinct from a raw `Array` so a forgotten transfer
still fails the CUDA storage check (§LXX). Experts in `GgufQuantSlice`
are already this kind of offload and stay unwrapped.
"""
struct HostOffload{T, N}
    data::Array{T, N}
end
HostOffload(data::AbstractArray{T, N}) where {T, N} =
    HostOffload{T, N}(convert(Array{T, N}, data))
Base.size(s::HostOffload) = size(s.data)
Base.size(s::HostOffload, d::Int) = size(s.data, d)
Base.ndims(s::HostOffload) = ndims(s.data)
Base.eltype(::HostOffload{T}) where {T} = T
Base.length(s::HostOffload) = length(s.data)
Base.sizeof(s::HostOffload) = sizeof(s.data)
_is_host_float_storage(s::HostOffload) = _is_host_float_storage(s.data)
_offload_host_array(s::HostOffload) = s.data
_offload_host_array(s::AbstractArray) = s

# libggml from a llama.cpp build, loaded only by the oracle tests that compare
# Gesso's kernels and dequant against ggml's (_ggml_lib_traits, _ggml_lib_dequant).
const _GGML_LIBS = Ref{Tuple{Ptr{Cvoid}, Ptr{Cvoid}}}((C_NULL, C_NULL))
const _GGML_GET_TRAITS = Ref{Ptr{Cvoid}}(C_NULL)

function _ggml_dir()
    dir = get(ENV, "GESSO_GGML_DIR", "")
    isempty(dir) && error(
        "gguf_import: GESSO_GGML_DIR is not set — point it at a directory " *
        "containing libggml-base.so (e.g. a llama.cpp build/bin)",
    )
    return dir
end

function _ensure_ggml!()
    _GGML_LIBS[][1] != C_NULL && return nothing
    dir = _ggml_dir()
    base = joinpath(dir, "libggml-base.so")
    cpu = joinpath(dir, "libggml-cpu.so")
    isfile(base) || error("gguf_import: GESSO_GGML_DIR has no libggml-base.so: $base")
    # cpu's DT_NEEDED is libggml-base.so.0; load base globally first.
    hbase = Libdl.dlopen(base, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    hcpu = C_NULL
    if isfile(cpu)
        hcpu = Libdl.dlopen(cpu, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
        init = Libdl.dlsym(hcpu, :ggml_cpu_init; throw_error=false)
        init != C_NULL && ccall(init, Cvoid, ())
    end
    traits = Libdl.dlsym(hbase, :ggml_get_type_traits)
    _GGML_LIBS[] = (hbase, hcpu)
    _GGML_GET_TRAITS[] = traits
    return nothing
end

# Block size and bytes per block of the ggml types Gesso reads, from ggml's own
# type table (the same sizes as _iq_block_bytes). Also the allow-list: a type
# missing here is refused, every type here has a decoder in gguf_dequant.jl,
# and a libggml whose traits struct no longer has the layout _ggml_lib_traits
# reads is caught against these sizes.
const _GGML_KNOWN_BLOCKS = Dict{Int32, Tuple{Int, Int}}(
    0 => (1, 4),        # F32
    1 => (1, 2),        # F16
    30 => (1, 2),       # BF16
    0 => (1, 4),        # F32
    1 => (1, 2),        # F16
    30 => (1, 2),       # BF16
    2 => (32, 18),      # Q4_0
    3 => (32, 20),      # Q4_1
    6 => (32, 22),      # Q5_0
    7 => (32, 24),      # Q5_1
    8 => (32, 34),      # Q8_0
    10 => (256, 84),    # Q2_K
    11 => (256, 110),   # Q3_K
    12 => (256, 144),   # Q4_K
    13 => (256, 176),   # Q5_K
    14 => (256, 210),   # Q6_K
    16 => (256, 66),    # IQ2_XXS
    17 => (256, 74),    # IQ2_XS
    18 => (256, 98),    # IQ3_XXS
    19 => (256, 50),    # IQ1_S
    20 => (32, 18),     # IQ4_NL
    21 => (256, 110),   # IQ3_S
    22 => (256, 82),    # IQ2_S
    23 => (256, 136),   # IQ4_XS
    29 => (256, 56),    # IQ1_M
    42 => (64, 18),     # Q2_0
)

# Throws unless the traits read for `ggml_type` equal Gesso's table. Wrong
# values here would size every row and dequant call wrongly, so they must not
# pass silently.
function _check_ggml_traits(ggml_type::Integer, blck::Integer, tsize::Integer)
    known = _GGML_KNOWN_BLOCKS[Int32(ggml_type)]
    (blck, tsize) == known && return nothing
    want = "block size $(known[1]) and type size $(known[2])"
    throw(
        gesso_error(
            ERR_RUNTIME,
            "gguf_import: libggml reports block size $blck and type size $tsize for ggml " *
            "type $ggml_type, expected $want. The ggml_type_traits layout read here " *
            "(blck_size@8, type_size@24, to_float@40) does not match the libggml in " *
            "GESSO_GGML_DIR; use a llama.cpp build this layout matches.";
            ggml_type=Int(ggml_type),
            blck_size=Int(blck),
            type_size=Int(tsize),
            ggml_dir=get(ENV, "GESSO_GGML_DIR", ""),
        ),
    )
end

function _ggml_traits(ggml_type::Integer; tensor=nothing)
    ggml_type in keys(_GGML_KNOWN_BLOCKS) || throw(
        gesso_error(
            ERR_RUNTIME,
            "gguf_import: ggml type $ggml_type" * (tensor === nothing ? "" : " (tensor $tensor)") *
            " has no reader in Gesso";
            reason=:unsupported_ggml_type,
            ggml_type=Int(ggml_type),
            tensor=something(tensor, ""),
        ),
    )
    blck, tsize = _GGML_KNOWN_BLOCKS[Int32(ggml_type)]
    return (blck_size=Int64(blck), type_size=tsize)
end

# libggml's traits for a type, checked against the table. Oracle tests only.
function _ggml_lib_traits(ggml_type::Integer)
    # ggml_get_type_traits indexes its table unchecked in release builds, so a
    # type outside Gesso's table is refused before it is asked.
    _ggml_traits(ggml_type)
    _ensure_ggml!()
    p = ccall(_GGML_GET_TRAITS[], Ptr{Cvoid}, (Cint,), Int32(ggml_type))
    p == C_NULL && error("gguf_import: ggml_get_type_traits($ggml_type) is null")
    # ggml_type_traits on x86-64: type_name@0, blck_size@8,
    # blck_size_interleave@16, type_size@24, is_quantized@32 + pad, to_float@40.
    # Nothing in the library states this layout, so the values are checked.
    blck = Int64(unsafe_load(Ptr{Int64}(p + 8)))
    tsize = Int(unsafe_load(Ptr{Csize_t}(p + 24)))
    _check_ggml_traits(ggml_type, blck, tsize)
    to_float = unsafe_load(Ptr{Ptr{Cvoid}}(p + 40))
    return (blck_size=blck, type_size=tsize, to_float=to_float)
end

# ggml's own dequant of n_elem values at 0-based offset0. Oracle tests only.
function _ggml_lib_dequant(data::AbstractVector{UInt8}, offset0::Int, ggml_type::Integer, n_elem::Int)
    tr = _ggml_lib_traits(ggml_type)
    tr.to_float == C_NULL && error("gguf_import: libggml has no to_float for ggml type $ggml_type")
    nbytes = cld(n_elem, Int(tr.blck_size)) * tr.type_size
    0 <= offset0 && offset0 + nbytes <= length(data) || error("ggml oracle: read past the end of data")
    y = Vector{Float32}(undef, n_elem)
    GC.@preserve data y ccall(
        tr.to_float,
        Cvoid,
        (Ptr{Cvoid}, Ptr{Float32}, Int64),
        pointer(data, offset0 + 1),
        pointer(y),
        Int64(n_elem),
    )
    return y
end

# --- tiny cursor --------------------------------------------------------------

mutable struct _Cur
    data::Vector{UInt8}
    pos::Int
end

# Raised by the cursor when a length or count read from the file cannot fit in
# what is left of it; gguf_header turns it into a GessoError naming the file.
struct _GgufOverrun <: Exception end

# A metadata value type GGUF does not define: the type field itself is damaged.
struct _GgufBadValueType <: Exception
    t::UInt32
end

_remaining(c::_Cur) = length(c.data) - c.pos + 1

# A length or element count read from the file, checked against the bytes left
# (each element takes at least `min_bytes`). Without this a corrupt count
# allocates, or loops, on a number the file never backs.
function _count(c::_Cur, min_bytes::Int=1)
    n = _u64(c)
    n <= UInt64(max(_remaining(c), 0) ÷ min_bytes) || throw(_GgufOverrun())
    return Int(n)
end

# Bytes per element of the fixed-width GGUF metadata value types; 0 for string
# (8) and array (9).
_gguf_value_width(t::UInt32) =
    t in (0, 1, 7) ? 1 : t in (2, 3) ? 2 : t in (4, 5, 6) ? 4 : t in (10, 11, 12) ? 8 : 0
_u8(c::_Cur) = (v=c.data[c.pos]; c.pos += 1; v)
function _u32(c::_Cur)
    p = c.pos
    v =
        UInt32(c.data[p]) | UInt32(c.data[p+1]) << 8 | UInt32(c.data[p+2]) << 16 |
        UInt32(c.data[p+3]) << 24
    c.pos = p + 4
    return v
end
function _u64(c::_Cur)
    p = c.pos
    v = UInt64(0)
    for i in 0:7
        v |= UInt64(c.data[p+i]) << (8 * i)
    end
    c.pos = p + 8
    return v
end
function _f32(c::_Cur)
    return reinterpret(Float32, _u32(c))
end
function _f64(c::_Cur)
    return reinterpret(Float64, _u64(c))
end
function _str(c::_Cur)
    n = _count(c)
    s = String(c.data[c.pos:(c.pos+n-1)])
    c.pos += n
    return s
end
function _read_u16!(c::_Cur)
    p = c.pos
    v = UInt16(c.data[p]) | UInt16(c.data[p+1]) << 8
    c.pos = p + 2
    return v
end

function _skip_value!(c::_Cur, t::UInt32)
    w = _gguf_value_width(t)
    if w > 0
        c.pos += w
    elseif t == 8
        n = _count(c)
        c.pos += n
    elseif t == 9
        at = _u32(c)
        aw = _gguf_value_width(at)
        if aw > 0
            n = _count(c, aw)
            c.pos += n * aw
        else
            for _ in 1:_count(c, 8)
                _skip_value!(c, at)
            end
        end
    else
        throw(_GgufBadValueType(t))
    end
    _remaining(c) >= 0 || throw(_GgufOverrun())
    return nothing
end

function _read_value(c::_Cur, t::UInt32)
    t == 0 && return UInt8(_u8(c))
    t == 1 && return Int8(_u8(c))
    t == 2 && return _read_u16!(c)
    t == 3 && return reinterpret(Int16, _read_u16!(c))
    t == 4 && return _u32(c)
    t == 5 && return reinterpret(Int32, _u32(c))
    t == 6 && return _f32(c)
    t == 7 && return _u8(c) != 0
    t == 8 && return _str(c)
    t == 10 && return _u64(c)
    t == 11 && return reinterpret(Int64, _u64(c))
    t == 12 && return _f64(c)
    if t == 9
        at = _u32(c)
        n = _count(c, max(_gguf_value_width(at), 1))
        return [_read_value(c, at) for _ in 1:n]
    end
    throw(_GgufBadValueType(t))
end

struct GgufTensorInfo
    name::String
    dims::Vector{Int}
    ggml_type::Int32
    offset::UInt64
end

# A key the architecture needs is absent: the bytes are fine, the export is not.
_missing_metadata(key::AbstractString) = gesso_error(
    ERR_RUNTIME,
    "spec_from_gguf_kv: GGUF metadata has no $key, which this architecture needs. " *
    "The file was not written by a converter Gesso supports.";
    reason=:missing_metadata,
    key=String(key),
)

function _malformed_gguf(path::AbstractString, msg::AbstractString; kw...)
    return gesso_error(
        ERR_RUNTIME,
        "gguf_header: $path: $msg. The file is truncated or corrupt; download it again.";
        reason=:malformed_gguf,
        path=String(path),
        kw...,
    )
end

"""
    gguf_header(path) -> (; data, kv, tensors, data_start)

Parse a GGUF header and check it against the file before anything reads
tensor bytes. A file that ends early, a count or length the file cannot hold,
a bad alignment, or a tensor whose bytes run past the end throws a
`GessoError` with `detail[:reason] == :malformed_gguf`. Tensors of a ggml type
missing from `_GGML_KNOWN_BLOCKS` are not sized here; loading one throws with
reason `:unsupported_ggml_type`.
"""
function gguf_header(path::AbstractString)
    isfile(path) || error("gguf_header: no such file: $path")
    data = open(path, "r") do io
        Mmap.mmap(io, Vector{UInt8}; grow=false, shared=false)
    end
    length(data) >= 24 || throw(_malformed_gguf(path, "$(length(data)) bytes is too short for a GGUF header"))
    hdr = try
        _gguf_header_parse(path, data)
    catch e
        e isa _GgufBadValueType &&
            throw(_malformed_gguf(path, "metadata value type $(e.t) is not a GGUF type"; value_type=Int(e.t)))
        e isa Union{_GgufOverrun, BoundsError, InexactError} || rethrow()
        throw(_malformed_gguf(path, "the metadata runs past the end of the file or holds an impossible length"))
    end
    _check_tensor_extents(path, hdr)
    return hdr
end

function _gguf_header_parse(path::AbstractString, data::Vector{UInt8})
    c = _Cur(data, 1)
    magic = (_u8(c), _u8(c), _u8(c), _u8(c))
    magic == _GGUF_MAGIC || error("gguf_header: not a GGUF file: $path")
    version = _u32(c)
    version >= 2 || error("gguf_header: GGUF version $version is too old")
    # A tensor info is at least 24 bytes and a key/value at least 13.
    n_tensors = _count(c, 24)
    n_kv = _count(c, 13)
    kv = Dict{String, Any}()
    for _ in 1:n_kv
        k = _str(c)
        t = _u32(c)
        # tokenizer arrays are hundreds of thousands of strings; keep counts.
        if t == UInt32(9) && startswith(k, "tokenizer.ggml.")
            at = _u32(c)
            n = _count(c, max(_gguf_value_width(at), 1))
            kv[k] = n
            for _ in 1:n
                _skip_value!(c, at)
            end
        elseif k == "tokenizer.chat_template"
            _skip_value!(c, t)
        else
            kv[k] = _read_value(c, t)
        end
    end
    infos = Vector{GgufTensorInfo}(undef, n_tensors)
    for i in 1:n_tensors
        name = _str(c)
        nd = Int(_u32(c))
        1 <= nd <= 4 || throw(_malformed_gguf(path, "tensor $name has $nd dimensions"; tensor=name))
        dims = Vector{Int}(undef, nd)
        for d in 1:nd
            dims[d] = Int(_u64(c))
        end
        t = Int32(_u32(c))
        off = _u64(c)
        infos[i] = GgufTensorInfo(name, dims, t, off)
    end
    alignment = Int(get(kv, "general.alignment", 32))
    alignment > 0 && ispow2(alignment) ||
        throw(_malformed_gguf(path, "general.alignment is $alignment"; alignment))
    pad = alignment - ((c.pos - 1) % alignment)
    pad == alignment && (pad = 0)
    data_start = (c.pos - 1) + pad
    return (data=data, kv=kv, tensors=infos, data_start=data_start, alignment=alignment, path=String(path))
end

# Bytes of a tensor of `ggml_type` with ggml dims `dims`, or `nothing` when the
# type is not in _GGML_KNOWN_BLOCKS. Throws on a row length that is not whole
# blocks or an element count that overflows.
function _known_tensor_nbytes(path, name, dims::Vector{Int}, ggml_type::Int32)
    blk = get(_GGML_KNOWN_BLOCKS, ggml_type, nothing)
    blk === nothing && return nothing
    blck, tsize = blk
    rem(dims[1], blck) == 0 || throw(
        _malformed_gguf(path, "tensor $name row length $(dims[1]) is not whole blocks of $blck"; tensor=name),
    )
    n = 1
    for d in dims
        n = Base.checked_mul(n, d)
    end
    return Base.checked_mul(n ÷ blck, tsize)
end

function _check_tensor_extents(path::AbstractString, hdr)
    len = length(hdr.data)
    hdr.data_start <= len ||
        throw(_malformed_gguf(path, "the tensor data would start at byte $(hdr.data_start) of $len"))
    for info in hdr.tensors
        name = info.name
        any(<(0), info.dims) && throw(_malformed_gguf(path, "tensor $name has dims $(info.dims)"; tensor=name))
        info.offset % UInt64(hdr.alignment) == 0 ||
            throw(_malformed_gguf(path, "tensor $name offset $(info.offset) is not aligned"; tensor=name))
        nbytes = try
            _known_tensor_nbytes(path, name, info.dims, info.ggml_type)
        catch e
            e isa OverflowError || rethrow()
            throw(_malformed_gguf(path, "tensor $name dims $(info.dims) overflow"; tensor=name))
        end
        nbytes === nothing && continue
        _check_extent(path, hdr, info, nbytes)
    end
    return nothing
end

# The one bounds check every raw tensor read goes through: `nbytes` from the
# tensor's start must lie inside the mapped file.
function _check_extent(path, hdr, info::GgufTensorInfo, nbytes::Integer)
    len = length(hdr.data)
    start = Int128(hdr.data_start) + Int128(info.offset)
    start + nbytes <= len && return nothing
    throw(
        _malformed_gguf(
            path,
            "tensor $(info.name) needs bytes $start:$(start + nbytes) but the file has $len";
            tensor=info.name,
            file_bytes=len,
        ),
    )
end

"""
    spec_from_gguf_kv(kv; tensor_names=nothing) -> ArchitectureSpec

Derive the spec from GGUF metadata. Eight arches are supported: qwen35, qwen35moe,
llama, spark2_5, qwen4exp, lfm2, lfm2moe and gemma4. Any other
`general.architecture` is an error. The two Qwen3.5 arches are described here;
the others have their own readers in this file:

* `qwen35`: dense Gated DeltaNet hybrid, `moe = nothing`, FFN width from
  `feed_forward_length`. `block_count` includes `nextn_predict_layers` MTP
  blocks at the end; those are not main layers, so `num_layers` excludes them.
* `qwen35moe`: the same hybrid mixer with routed experts (`expert_count` and
  friends are required).

Every head count, width and SSM dimension is read from the file. Nothing
defaults to a particular checkpoint's shape, except `full_attention_interval`
(4) and `rope.freq_base` (10000), which the GGUF may omit.
"""
function spec_from_gguf_kv(kv::AbstractDict; tensor_names=nothing)
    arch = String(get(kv, "general.architecture", "qwen35moe"))
    arch in ("qwen35", "qwen35moe", "llama", "spark2_5", "qwen4exp", "lfm2", "lfm2moe", "gemma4") || throw(
        gesso_error(
            ERR_RUNTIME,
            "spec_from_gguf_kv: unsupported general.architecture $(repr(arch)) " *
            "(supported: qwen35 dense, qwen35moe, llama, spark2_5, qwen4exp, lfm2, lfm2moe, gemma4)";
            reason=:unsupported_architecture,
            architecture=arch,
        ),
    )
    arch == "gemma4" && return _spec_from_gguf_gemma4(kv, tensor_names)
    arch == "llama" && return _spec_from_gguf_llama(kv, tensor_names)
    arch == "spark2_5" && return _spec_from_gguf_spark(kv, tensor_names)
    arch == "qwen4exp" && return _spec_from_gguf_qwen4exp(kv, tensor_names)
    arch in ("lfm2", "lfm2moe") && return _spec_from_gguf_lfm(kv, tensor_names)
    p(key, default=nothing) = get(kv, "$arch.$key", default)
    req(key) =
        (v=p(key); v === nothing ? throw(_missing_metadata("$arch.$key")) : v)
    nextn = Int(p("nextn_predict_layers", 0))
    block_count = Int(req("block_count"))
    n_layers = block_count - nextn
    n_layers > 0 || error(
        "spec_from_gguf_kv: block_count $block_count minus nextn_predict_layers $nextn leaves no main layers",
    )
    hidden = Int(req("embedding_length"))
    n_heads = Int(req("attention.head_count"))
    n_kv = Int(p("attention.head_count_kv", n_heads))
    head_dim = Int(p("attention.key_length", p("attention.head_dim", 0)))
    is_moe = arch == "qwen35moe"
    if is_moe
        n_exp = Int(req("expert_count"))
        n_used = Int(req("expert_used_count"))
        expert_ff = Int(req("expert_feed_forward_length"))
        shared_ff = Int(p("expert_shared_feed_forward_length", expert_ff))
        moe = MoeSpec(; n_experts=n_exp, n_expert_used=n_used, expert_ff, shared_ff)
        intermediate = expert_ff
        family = :qwen3_5_moe
    else
        p("expert_count") === nothing || error(
            "spec_from_gguf_kv: arch qwen35 is dense but the file carries expert keys",
        )
        moe = nothing
        intermediate = Int(req("feed_forward_length"))
        family = :qwen3_5
    end
    toks = get(kv, "tokenizer.ggml.tokens", nothing)
    vocab = if toks isa Integer
        Int(toks)
    elseif toks === nothing
        Int(p("vocab_size", 0))
    else
        length(toks)
    end
    vocab > 0 || throw(_missing_metadata("tokenizer.ggml.tokens"))
    interval = Int(p("full_attention_interval", 4))
    # llama.cpp: is_recr[i] = ((i+1) % interval != 0) with 0-based i, so
    # 1-based layers interval, 2*interval, … are full attention.
    layer_types =
        [(i % interval == 0 ? :full_attention : :linear_attention) for i in 1:n_layers]
    has_linear = any(!=(:full_attention), layer_types)
    mixer = nothing
    feats = Set{Symbol}([:qk_norm, :attn_output_gate, :mrope])
    if has_linear
        ssm_group = Int(req("ssm.group_count"))
        ssm_state = Int(req("ssm.state_size"))
        ssm_dt = Int(req("ssm.time_step_rank"))
        ssm_inner = Int(req("ssm.inner_size"))
        conv_k = Int(req("ssm.conv_kernel"))
        value_head_dim = ssm_state
        ssm_inner == ssm_dt * value_head_dim || error(
            "spec_from_gguf_kv: ssm.inner_size $ssm_inner ≠ time_step_rank $ssm_dt × state $ssm_state",
        )
        mixer = LinearMixerSpec(;
            layer_types,
            num_key_heads=ssm_group,
            key_head_dim=ssm_state,
            num_value_heads=ssm_dt,
            value_head_dim,
            conv_kernel_dim=conv_k,
            v_head_tiled=true,
        )
        push!(feats, :linear_attention)
    end
    theta = Float64(p("rope.freq_base", 10000.0))
    rotary_dim = Int(p("rope.dimension_count", 0))
    section = Int[Int(x) for x in p("rope.dimension_sections", Int[])]
    isempty(section) || (section = filter(>(0), section))
    tied = tensor_names !== nothing && !("output.weight" in tensor_names)
    # GGUF already bakes RMSNorm +1 except ssm_norm; do not set :rms_unit_offset.
    return ArchitectureSpec(;
        family,
        hidden_size=hidden,
        num_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        vocab_size=vocab,
        intermediate_size=intermediate,
        head_dim=head_dim > 0 ? head_dim : nothing,
        rope=RoPEPolicy(; theta, rotary_dim, interleaved=false, mrope_section=section),
        tie_word_embeddings=tied,
        features=feats,
        mixer,
        rms_norm_eps=Float64(p("attention.layer_norm_rms_epsilon", 1e-6)),
        moe,
    )
end

"""
`qwen4exp` (Flash-Next) GGUF spec. The spec records the shape and the
features (hyper-connection streams, per-layer embeddings) so the import report
can describe the file; the forward is `qwen4exp_forward!`, not the generic
block runner. MTP layers are excluded from `num_layers`.
"""
function _spec_from_gguf_qwen4exp(kv::AbstractDict, tensor_names)
    cfg = qwen4exp_config(kv)
    cfg.vocab > 0 || throw(_missing_metadata("tokenizer.ggml.tokens"))
    p(key) = kv["qwen4exp.$key"]
    layer_types = [a ? :full_attention : :linear_attention for a in cfg.attn_layer]
    mixer = LinearMixerSpec(;
        layer_types,
        num_key_heads=cfg.ssm_nk,
        key_head_dim=cfg.ssm_dk,
        num_value_heads=cfg.ssm_nv,
        value_head_dim=cfg.ssm_dv,
        conv_kernel_dim=cfg.conv_kernel,
        v_head_tiled=true,
    )
    expert_ff = Int(p("expert_feed_forward_length"))
    moe = MoeSpec(;
        n_experts=cfg.n_exp,
        n_expert_used=cfg.n_used,
        expert_ff,
        shared_ff=Int(p("expert_shared_feed_forward_length")),
    )
    section = filter(>(0), Int[Int(x) for x in p("rope.dimension_sections")])
    return ArchitectureSpec(;
        family=:qwen4exp,
        hidden_size=cfg.n_embd,
        num_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        n_kv_heads=cfg.n_head_kv,
        vocab_size=cfg.vocab,
        intermediate_size=expert_ff,
        head_dim=cfg.head_dim,
        rope=RoPEPolicy(;
            theta=cfg.rope_base, rotary_dim=cfg.n_rot, interleaved=false, mrope_section=section),
        tie_word_embeddings=tensor_names !== nothing && !("output.weight" in tensor_names),
        features=Set{Symbol}([:qk_norm, :attn_output_gate, :mrope, :linear_attention,
            :hyper_connections, :ple]),
        mixer,
        rms_norm_eps=cfg.eps,
        moe,
    )
end

"""
`llama` GGUF spec. Tied embeddings are read from the tensor list, not from
metadata: converters drop `output.weight` when the head is tied. Without
`tensor_names` the head is assumed present.

llama.cpp's converter permutes `attn_q` and `attn_k` rows into adjacent-pair
order for ggml's NORMAL rope. The weights are kept as stored and the rope is
declared `interleaved`, which gives the same attention scores as HF's
half-split rope on the unpermuted weights. Rope scaling beyond `linear`, and
the `rope_freqs.weight` tensor that carries llama3 scaling, are refused rather
than dropped.
"""
function _spec_from_gguf_llama(kv::AbstractDict, tensor_names)
    p(key, default=nothing) = get(kv, "llama.$key", default)
    req(key) =
        (v=p(key); v === nothing ? throw(_missing_metadata("llama.$key")) : v)
    hidden = Int(req("embedding_length"))
    n_heads = Int(req("attention.head_count"))
    head_dim = Int(p("attention.key_length", 0))
    toks = get(kv, "tokenizer.ggml.tokens", nothing)
    vocab =
        toks isa Integer ? Int(toks) :
        toks === nothing ? Int(p("vocab_size", 0)) : length(toks)
    vocab > 0 || throw(_missing_metadata("tokenizer.ggml.tokens"))
    scaling = String(p("rope.scaling.type", "none"))
    (kind, factor) = if scaling in ("none", "")
        (:none, 1.0)
    elseif scaling == "linear"
        (:linear, Float64(req("rope.scaling.factor")))
    else
        throw(LoweringNotImplemented(:rope_scaling, :llama))
    end
    tensor_names !== nothing &&
        "rope_freqs.weight" in tensor_names &&
        error(
            "spec_from_gguf_kv: llama rope_freqs.weight present (llama3 scaling baked into per-dim factors); not supported",
        )
    tied = tensor_names !== nothing && !("output.weight" in tensor_names)
    return ArchitectureSpec(;
        family=:llama,
        hidden_size=hidden,
        num_layers=Int(req("block_count")),
        n_heads,
        n_kv_heads=Int(p("attention.head_count_kv", n_heads)),
        vocab_size=vocab,
        intermediate_size=Int(req("feed_forward_length")),
        head_dim=head_dim > 0 ? head_dim : nothing,
        rope=RoPEPolicy(;
            theta=Float64(p("rope.freq_base", 10000.0)),
            kind,
            factor,
            rotary_dim=Int(p("rope.dimension_count", 0)),
            interleaved=true,
        ),
        tie_word_embeddings=tied,
        rms_norm_eps=Float64(p("attention.layer_norm_rms_epsilon", 1e-6)),
    )
end

"""
`spark2_5` GGUF spec (Spark-X2.6). Dense GQA attention with a fused Q|K|V
tensor, one sigmoid output gate per head, and a GELU-gated FFN. Layers follow
`attention.sliding_window_pattern` (true = windowed). Windowed layers rotate
the whole head at `rope.freq_base_swa`; full layers rotate the first
`rope.dimension_count` dims at `rope.freq_base`. Both use the half-split (NEOX)
layout, and llama.cpp does not permute Q/K for it.
"""
function _spec_from_gguf_spark(kv::AbstractDict, tensor_names)
    p(key, default=nothing) = get(kv, "spark2_5.$key", default)
    req(key) =
        (v=p(key); v === nothing ? throw(_missing_metadata("spark2_5.$key")) : v)
    n_layers = Int(req("block_count"))
    n_heads = Int(req("attention.head_count"))
    head_dim = Int(req("attention.key_length"))
    Int(p("attention.value_length", head_dim)) == head_dim ||
        error("spec_from_gguf_kv: spark2_5 key_length != value_length is not supported")
    toks = get(kv, "tokenizer.ggml.tokens", nothing)
    vocab =
        toks isa Integer ? Int(toks) :
        toks === nothing ? Int(p("vocab_size", 0)) : length(toks)
    vocab > 0 || throw(_missing_metadata("tokenizer.ggml.tokens"))
    pattern = collect(Bool, req("attention.sliding_window_pattern"))
    length(pattern) == n_layers || error(
        "spec_from_gguf_kv: spark2_5 sliding_window_pattern has $(length(pattern)) entries for $n_layers layers",
    )
    rot(key) = (r=Int(p(key, head_dim)); r == head_dim ? 0 : r)
    tied = tensor_names !== nothing && !("output.weight" in tensor_names)
    return ArchitectureSpec(;
        family=:spark2_5,
        hidden_size=Int(req("embedding_length")),
        num_layers=n_layers,
        n_heads,
        n_kv_heads=Int(p("attention.head_count_kv", n_heads)),
        vocab_size=vocab,
        intermediate_size=Int(req("feed_forward_length")),
        head_dim,
        activation_kind=:geglu,
        rope=RoPEPolicy(;
            theta=Float64(req("rope.freq_base")),
            rotary_dim=rot("rope.dimension_count"),
            interleaved=false,
        ),
        rope_swa=RoPEPolicy(;
            theta=Float64(req("rope.freq_base_swa")),
            rotary_dim=rot("rope.dimension_count_swa"),
            interleaved=false,
        ),
        sliding_window=Int(req("attention.sliding_window")),
        swa_layers=pattern,
        tie_word_embeddings=tied,
        features=Set([:fused_qkv, :attn_head_gate]),
        rms_norm_eps=Float64(p("attention.layer_norm_rms_epsilon", 1e-6)),
    )
end

function gguf_architecture_spec(path::AbstractString)
    hdr = gguf_header(path)
    return spec_from_gguf_kv(hdr.kv; tensor_names=Set(t.name for t in hdr.tensors))
end

function _dequant_f32(data::Vector{UInt8}, offset0::Int, ggml_type::Int32, n_elem::Int)
    traits = _ggml_traits(ggml_type)
    blck = Int(traits.blck_size)
    rem(n_elem, blck) == 0 || error("gguf_import: $n_elem values is not a whole number of $blck-value blocks")
    # Every caller passes header-checked offsets; this is the backstop for the
    # ones that compute their own.
    nbytes = div(n_elem, blck) * traits.type_size
    0 <= offset0 && offset0 + nbytes <= length(data) || throw(
        gesso_error(
            ERR_RUNTIME,
            "gguf_import: tensor read of $nbytes bytes at $offset0 runs past the $(length(data))-byte file. The file is truncated or corrupt; download it again.";
            reason=:malformed_gguf,
        ),
    )
    return _ggml_dequant!(Vector{Float32}(undef, n_elem), data, offset0, ggml_type, n_elem)
end

function _gguf_array(hdr, info::GgufTensorInfo)
    n_elem = prod(info.dims)
    if info.ggml_type == 0
        _check_extent(hdr.path, hdr, info, 4 * n_elem)
    else
        traits = _ggml_traits(info.ggml_type; tensor=info.name)
        rem(n_elem, Int(traits.blck_size)) == 0 || error(
            "gguf_import: $(info.name) n_elem $n_elem not divisible by blck_size $(traits.blck_size)",
        )
        _check_extent(hdr.path, hdr, info, div(n_elem, Int(traits.blck_size)) * traits.type_size)
    end
    off = Int(hdr.data_start) + Int(info.offset)
    raw = _dequant_f32(hdr.data, off, info.ggml_type, n_elem)
    nd = length(info.dims)
    nd == 1 && return raw
    A = reshape(raw, info.dims...)
    return permutedims(A, nd:-1:1)
end

function _is_expert_tensor(name::AbstractString)
    endswith(name, ".scale") && return false
    return occursin("ffn_gate_exps", name) ||
           occursin("ffn_up_exps", name) ||
           occursin("ffn_down_exps", name) ||
           occursin("ffn_gate_up_exps", name)
end

function _expert_slice(hdr, info::GgufTensorInfo)
    length(info.dims) == 3 || error("gguf_import: expert tensor $(info.name) is not 3D")
    ne0, ne1, n_exp = info.dims
    traits = _ggml_traits(info.ggml_type; tensor=info.name)
    n_elem = ne0 * ne1
    nbytes = if info.ggml_type == 0
        n_elem * 4
    else
        rem(n_elem, Int(traits.blck_size)) == 0 || error(
            "gguf_import: $(info.name) n_elem $n_elem not divisible by blck_size $(traits.blck_size)",
        )
        div(n_elem, Int(traits.blck_size)) * traits.type_size
    end
    _check_extent(hdr.path, hdr, info, nbytes * n_exp)
    return GgufQuantSlice(
        hdr.data,
        Int(hdr.data_start) + Int(info.offset),
        info.ggml_type,
        n_exp,
        ne0,
        ne1,
        nbytes,
        traits.blck_size,
    )
end

function _expert_cache_cap()
    raw = get(ENV, "GESSO_EXPERT_CACHE", "0")
    n = tryparse(Int, raw)
    return n === nothing ? 0 : max(0, n)
end

function _gguf_dequant_expert_into!(
    W::AbstractMatrix{Float32},
    raw::Vector{Float32},
    s::GgufQuantSlice,
    e::Int,
)
    n_elem = s.ne0 * s.ne1
    length(raw) >= n_elem || error("gguf_dequant_expert!: raw too small")
    size(W) == (s.ne1, s.ne0) ||
        error("gguf_dequant_expert!: W $(size(W)) ≠ ($(s.ne1), $(s.ne0))")
    off = s.offset0 + (e - 1) * s.bytes_per_expert
    _ggml_dequant!(raw, s.data, off, s.ggml_type, n_elem)
    A = reshape(view(raw, 1:n_elem), s.ne0, s.ne1)
    @inbounds for j in 1:s.ne1, i in 1:s.ne0
        W[j, i] = A[i, j]
    end
    return W
end

function _gguf_dequant_expert_uncached(s::GgufQuantSlice, e::Int)
    W = Matrix{Float32}(undef, s.ne1, s.ne0)
    raw = Vector{Float32}(undef, s.ne0 * s.ne1)
    return _gguf_dequant_expert_into!(W, raw, s, e)
end

function gguf_dequant_expert(s::GgufQuantSlice, e::Int)
    1 <= e <= s.n_exp || error("gguf_dequant_expert: expert $e out of 1:$(s.n_exp)")
    cap = _expert_cache_cap()
    cap == 0 && return _gguf_dequant_expert_uncached(s, e)
    cached = get(s.cache, e, nothing)
    cached !== nothing && return cached
    while length(s.cache) >= cap
        old = popfirst!(s.cache_order)
        delete!(s.cache, old)
    end
    W = _gguf_dequant_expert_uncached(s, e)
    s.cache[e] = W
    push!(s.cache_order, e)
    return W
end

function _maybe_T(A, ::Type{T}) where {T}
    return eltype(A) === T ? A : T.(A)
end

function _required_gguf_names(pmap::FamilyParamMap, spec::ArchitectureSpec)
    names = String[pmap.embedding_external]
    pmap.final_norm_external !== nothing && push!(names, pmap.final_norm_external)
    pmap.lm_head_external !== nothing && push!(names, pmap.lm_head_external)
    pmap.rope_freqs_external !== nothing && push!(names, pmap.rope_freqs_external)
    for i in 0:(spec.num_layers-1)
        refs = _layer_param_refs(pmap, spec, i)
        for ref in refs
            push!(names, external_name(pmap, ref, i))
        end
    end
    return unique(names)
end

"""
llama.cpp `_LinearAttentionVReorderBase` stores V heads in tiled order
(`[G0_v0, G1_v0, …]`) so ggml can broadcast. HuggingFace safetensors group
them (`[G0_v0..v{r-1}, G1_…]`). `_v_head_perm(nk, r, hd)` maps tiled →
grouped. The inverse is `_v_head_perm(r, nk, hd)` — an involution only
when `nk == r`. Gesso GDN on GGUF uses tiled pairing instead, so this
gather stays behind `GESSO_GGUF_UNREORDER_V=1`.
"""
function _v_head_perm(nk::Int, r::Int, hd::Int)
    n = nk * r * hd
    p = Vector{Int}(undef, n)
    for j in 0:(n-1)
        v = div(j, nk * hd)
        k = rem(div(j, hd), nk)
        d = rem(j, hd)
        i = k * (r * hd) + v * hd + d
        p[j+1] = i + 1
    end
    return p
end

function _gather_axis(A::AbstractArray, dim::Int, p::Vector{Int})
    if dim == 1
        ndims(A) == 1 && return A[p]
        rest = ntuple(_ -> Colon(), ndims(A) - 1)
        return A[p, rest...]
    elseif dim == 2
        rest = ntuple(_ -> Colon(), ndims(A) - 2)
        return A[:, p, rest...]
    end
    error("gguf_import: V-head unreorder dim $dim")
end

function _gguf_unreorder_linear(A, name::AbstractString, spec::ArchitectureSpec)
    mx = spec.mixer
    mx isa LinearMixerSpec || return A
    nk, nv = mx.num_key_heads, mx.num_value_heads
    nk == nv && return A
    r = div(nv, nk)
    r * nk == nv || return A
    dk, dv = mx.key_head_dim, mx.value_head_dim
    qk = 2 * nk * dk
    if endswith(name, ".attn_qkv.weight")
        size(A, 1) == qk + nv * dv || error("gguf_import: attn_qkv rows $(size(A,1))")
        p = _v_head_perm(nk, r, dv)
        v = _gather_axis(A[(qk+1):end, :], 1, p)
        return vcat(A[1:qk, :], v)
    elseif endswith(name, ".attn_gate.weight")
        return _gather_axis(A, 1, _v_head_perm(nk, r, dv))
    elseif endswith(name, ".ssm_alpha.weight") || endswith(name, ".ssm_beta.weight")
        return _gather_axis(A, 1, _v_head_perm(nk, r, 1))
    elseif endswith(name, ".ssm_a") || endswith(name, ".ssm_dt.bias")
        return _gather_axis(A, 1, _v_head_perm(nk, r, 1))
    elseif endswith(name, ".ssm_conv1d.weight")
        size(A, 1) == qk + nv * dv || error("gguf_import: conv1d channels $(size(A,1))")
        p = _v_head_perm(nk, r, dv)
        v = _gather_axis(A[(qk+1):end, :, :], 1, p)
        return vcat(A[1:qk, :, :], v)
    elseif endswith(name, ".ssm_out.weight")
        return _gather_axis(A, 2, _v_head_perm(nk, r, dv))
    end
    return A
end

"""
Smallest 2-D tensor (in elements) that stays quantized. Below this, the
tensor is materialized as F32: norms and gates are 1-D or tiny, the bytes
saved are negligible, and router-like values feed top-k decisions where
rounding changes routing. The same 2^20 line is the GPU F16 threshold, so
the two residency decisions agree.
"""
const _QUANT_RESIDENT_MIN_ELEMS = 1 << 20

"""
A 2-D non-F32 tensor at or above `_QUANT_RESIDENT_MIN_ELEMS` stays as quant
bytes (`QuantMatrix`). `GESSO_GGUF_UNREORDER_V=1` is the opt-in tiled→grouped
V gather, which rewrites F32 arrays, so it keeps the all-F32 load.
"""
function _quant_resident(info::GgufTensorInfo)
    length(info.dims) == 2 || return false
    info.ggml_type == GGML_TYPE_F32 && return false
    prod(info.dims) >= _QUANT_RESIDENT_MIN_ELEMS || return false
    return get(ENV, "GESSO_GGUF_UNREORDER_V", "") != "1"
end

function _quant_matrix(hdr, info::GgufTensorInfo)
    ne0, nrows = info.dims
    traits = _ggml_traits(info.ggml_type; tensor=info.name)
    rem(ne0, Int(traits.blck_size)) == 0 || error(
        "gguf_import: $(info.name) ne0 $ne0 not divisible by blck_size $(traits.blck_size)",
    )
    rb = div(ne0, Int(traits.blck_size)) * traits.type_size
    _check_extent(hdr.path, hdr, info, rb * nrows)
    return QuantMatrix(
        hdr.data,
        Int(hdr.data_start) + Int(info.offset),
        info.ggml_type,
        ne0,
        nrows,
        rb,
        traits.blck_size,
    )
end

function _materialize_gguf_tensor(
    hdr,
    info::GgufTensorInfo,
    spec::ArchitectureSpec,
    ::Type{T},
) where {T}
    if _is_expert_tensor(info.name)
        return _expert_slice(hdr, info)
    end
    _quant_resident(info) && return _quant_matrix(hdr, info)
    A = _gguf_array(hdr, info)
    if endswith(info.name, ".ssm_conv1d.weight") || info.name == "ssm_conv1d.weight"
        ndims(A) == 2 || error("gguf_import: conv1d $(info.name) not 2D")
        A = reshape(A, size(A, 1), 1, size(A, 2))
    elseif endswith(info.name, ".ssm_a") || info.name == "ssm_a"
        A = log.(-A)
    end
    # Unsloth qwen35moe GGUF is already llama.cpp tiled. Gesso GDN on this
    # path uses LinearMixerSpec.v_head_tiled, so leave weights as stored.
    # Opt in to a tiled→grouped gather with GESSO_GGUF_UNREORDER_V=1.
    if get(ENV, "GESSO_GGUF_UNREORDER_V", "") == "1"
        A = _gguf_unreorder_linear(A, info.name, spec)
    end
    return _maybe_T(A, T)
end

_gguf_param_map(spec::ArchitectureSpec) =
    spec.family === :qwen4exp ?
    error("qwen4exp GGUFs do not load through the generic block runner; use load_qwen4exp") :
    spec.family === :llama ? gguf_param_map(LlamaAdapter(), spec) :
    spec.family === :spark2_5 ? _gguf_spark_param_map(spec) :
    spec.family in (:lfm2, :lfm2moe) ? _gguf_lfm_param_map(spec) :
    spec.family === :gemma4 ? _gguf_gemma4_param_map(spec) :
    spec.moe === nothing ? gguf_param_map(Qwen35Adapter(), spec) :
    gguf_param_map(Qwen35MoeAdapter(), spec)

"""
llama.cpp `spark2_5` tensor spellings. `attn_qkv` packs Q, K and V
contiguously (rows Q | K | V); `attn_gate` is the per-head output gate. The
head is tied, so there is no `output.weight`.
"""
function _gguf_spark_param_map(spec::ArchitectureSpec)
    qd = spec.n_heads * spec.head_dim
    kvd = spec.n_kv_heads * spec.head_dim
    refs = ParamRef[
        ParamRef(SemanticParamId(ROLE_NORM_PRE_ATTN, 0, :frozen), "attn_norm.weight"),
        ParamRef(SemanticParamId(ROLE_Q, 0, :projection), "attn_qkv.weight", 1:qd),
        ParamRef(
            SemanticParamId(ROLE_K, 0, :projection),
            "attn_qkv.weight",
            (qd+1):(qd+kvd),
        ),
        ParamRef(
            SemanticParamId(ROLE_V, 0, :projection),
            "attn_qkv.weight",
            (qd+kvd+1):(qd+2kvd),
        ),
        ParamRef(SemanticParamId(ROLE_ATTN_HEAD_GATE, 0, :projection), "attn_gate.weight"),
        ParamRef(SemanticParamId(ROLE_O, 0, :projection), "attn_output.weight"),
        ParamRef(SemanticParamId(ROLE_NORM_POST_ATTN, 0, :frozen), "ffn_norm.weight"),
        ParamRef(SemanticParamId(ROLE_GATE, 0, :projection), "ffn_gate.weight"),
        ParamRef(SemanticParamId(ROLE_UP, 0, :projection), "ffn_up.weight"),
        ParamRef(SemanticParamId(ROLE_DOWN, 0, :projection), "ffn_down.weight"),
    ]
    return FamilyParamMap(
        :spark2_5,
        "blk.",
        refs,
        "token_embd.weight",
        "output_norm.weight",
        spec.tie_word_embeddings ? nothing : "output.weight",
    )
end

"""
Record the MTP (nextn) blocks that `spec_from_gguf_kv` excludes from
`num_layers`. They are not loaded and not executed. The record is what a
later draft-source path needs: the layer index, the count, and every tensor
name in those blocks. `nothing` when the file has no MTP blocks.
"""
function _gguf_mtp_record(hdr, spec::ArchitectureSpec)
    arch = String(get(hdr.kv, "general.architecture", "qwen35moe"))
    nextn = Int(get(hdr.kv, "$arch.nextn_predict_layers", 0))
    nextn == 0 && return nothing
    first_blk = spec.num_layers
    prefixes = ["blk.$(first_blk + k)." for k in 0:(nextn-1)]
    names =
        sort!([t.name for t in hdr.tensors if any(p -> startswith(t.name, p), prefixes)])
    isempty(names) &&
        error("load_gguf: nextn_predict_layers=$nextn but no tensors under $(prefixes)")
    return (layer=first_blk, n_predict=nextn, tensor_names=names)
end

"""
    load_gguf(path; T=Float32) -> (model, tensors, spec)

Load a GGUF of any architecture `spec_from_gguf_kv` accepts (qwen35,
qwen35moe, llama, spark2_5, qwen4exp, lfm2, lfm2moe, gemma4). Dense 2-D projection weights and
the embedding and output head stay as quant bytes (`QuantMatrix`), not F32.
Small and 1-D tensors materialize to `T` (Float32 by default; F64 cannot hold
these checkpoints). Routed experts stay mmap'd as `GgufQuantSlice` and dequant
one selected expert at a time. `tensors.mtp` records the MTP blocks (not run).
"""
function load_gguf(path::AbstractString; T::Type=Float32, verbose::Bool=false)
    hdr = gguf_header(path)
    spec = spec_from_gguf_kv(hdr.kv; tensor_names=Set(t.name for t in hdr.tensors))
    pmap = _gguf_param_map(spec)
    info_by = Dict{String, GgufTensorInfo}(info.name => info for info in hdr.tensors)
    required = _required_gguf_names(pmap, spec)
    by = Dict{String, Any}()
    nreq = length(required)
    for (i, name) in enumerate(required)
        haskey(info_by, name) || error("load_gguf: missing required tensor: $name")
        verbose && i % 50 == 0 && @info "load_gguf" i nreq name
        by[name] = _materialize_gguf_tensor(hdr, info_by[name], spec, T)
    end
    bound = materialize_architecture(spec, pmap, by)
    model = spec_to_model(spec)
    tensors = (
        embedding=bound.embedding,
        blocks=bound.blocks,
        lm_head=bound.lm_head,
        final_rms=bound.final_rms,
        rope=bound.rope,
        bindings=bound.bindings,
        spec=spec,
        gguf_path=String(path),
        mtp=_gguf_mtp_record(hdr, spec),
    )
    return (model, tensors, spec)
end

"""
Softmax top-k SwiGLU over mmap'd GGUF experts. Dequants only the `n_used`
selected experts per token (cached within the call).
"""
function _cpu_moe_topk_swiglu_gguf!(
    y::AbstractMatrix,
    x::AbstractMatrix,
    router::AbstractMatrix,
    gate_s,
    up_s,
    down_s,
    n_used::Int,
)
    seq, h = size(x)
    n_exp = size(router, 1)
    ff = gate_s.ne1
    0 < n_used <= n_exp ||
        error("_cpu_moe_topk_swiglu_gguf!: n_used $n_used out of 1..$n_exp")
    size(y) == (seq, h) ||
        error("_cpu_moe_topk_swiglu_gguf!: y shape $(size(y)) ≠ ($seq, $h)")
    size(router) == (n_exp, h) ||
        error("_cpu_moe_topk_swiglu_gguf!: router $(size(router)) ≠ ($n_exp, $h)")
    size(gate_s) == (n_exp, ff, h) ||
        error("_cpu_moe_topk_swiglu_gguf!: gate $(size(gate_s)) ≠ ($n_exp, $ff, $h)")
    size(up_s) == (n_exp, ff, h) ||
        error("_cpu_moe_topk_swiglu_gguf!: up $(size(up_s)) ≠ ($n_exp, $ff, $h)")
    size(down_s) == (n_exp, h, ff) ||
        error("_cpu_moe_topk_swiglu_gguf!: down $(size(down_s)) ≠ ($n_exp, $h, $ff)")
    T = eltype(y)
    fill!(y, zero(T))
    logits = Vector{T}(undef, n_exp)
    order = Vector{Int}(undef, n_exp)
    # K-quant slabs take the batched Q8_K kernels the CUDA host route uses (x
    # quantized once per token, every selected expert in one pool dispatch).
    # They make the same _dot_q8k calls and sum in k order, so the result is
    # bit-identical to the per-expert loop below. On the GSQ 35B the loop
    # below was about half of CPU decode, and Lava's host experts run here too.
    if T === Float32 &&
       gate_s isa GgufQuantSlice &&
       up_s isa GgufQuantSlice &&
       down_s isa GgufQuantSlice &&
       _moe_host_batchable(gate_s, up_s, down_s, h, ff)
        return _cpu_moe_topk_swiglu_batched!(
            y,
            x,
            router,
            gate_s,
            up_s,
            down_s,
            n_used,
            logits,
            order,
        )
    end
    # One scratch set per chunk of expert slots, indexed by chunk and not by
    # threadid(): a chunk is one @threads iteration, so no two tasks share a set.
    nchunk = min(Threads.nthreads(), n_used)
    gaccs = [Vector{Float32}(undef, ff) for _ in 1:nchunk]
    uaccs = [Vector{Float32}(undef, ff) for _ in 1:nchunk]
    hacts = [Vector{Float32}(undef, ff) for _ in 1:nchunk]
    xbufs = [Vector{Float32}(undef, h) for _ in 1:nchunk]
    ybufs = [zeros(Float32, h) for _ in 1:n_used]
    for t in 1:seq
        xrow = @view x[t, :]
        m = _cpu_router_logits!(logits, x, t, router)
        s = zero(T)
        for e in 1:n_exp
            v = exp(logits[e] - m)
            logits[e] = v
            s += v
        end
        invs = one(T) / s
        for e in 1:n_exp
            logits[e] *= invs
            order[e] = e
        end
        sort!(order; by=e -> (-logits[e], e))
        wsum = zero(T)
        for k in 1:n_used
            wsum += logits[order[k]]
        end
        wsum > 0 || continue
        invw = one(T) / wsum
        yrow = @view y[t, :]
        fill!(yrow, zero(T))
        if nchunk == 1
            xf = xbufs[1]
            @inbounds for i in 1:h
                xf[i] = Float32(xrow[i])
            end
            for k in 1:n_used
                e = order[k]
                w = Float32(logits[e] * invw)
                _qgemv_expert!(gaccs[1], xf, gate_s, e)
                _qgemv_expert!(uaccs[1], xf, up_s, e)
                @inbounds for j in 1:ff
                    hacts[1][j] = _cpu_silu(gaccs[1][j]) * uaccs[1][j]
                end
                _qgemv_expert!(ybufs[k], hacts[1], down_s, e; alpha=w, beta=0.0f0)
                @inbounds for i in 1:h
                    yrow[i] += ybufs[k][i]
                end
            end
        else
            xf0 = xbufs[1]
            @inbounds for i in 1:h
                xf0[i] = Float32(xrow[i])
            end
            # Chunk c owns scratch set c and the expert slots k = c, c+nchunk, ...
            # Each slot is written by exactly one chunk, and the sum below runs in
            # k order, so the result does not depend on scheduling. Scratch is
            # per chunk, not per threadid(), so it needs no :static pinning and
            # the loop may run nested or from concurrent tasks.
            Threads.@threads for c in 1:nchunk
                xft = xbufs[c]
                c == 1 || copyto!(xft, xf0)
                for k in c:nchunk:n_used
                    e = order[k]
                    w = Float32(logits[e] * invw)
                    _qgemv_expert!(gaccs[c], xft, gate_s, e)
                    _qgemv_expert!(uaccs[c], xft, up_s, e)
                    @inbounds for j in 1:ff
                        hacts[c][j] = _cpu_silu(gaccs[c][j]) * uaccs[c][j]
                    end
                    fill!(ybufs[k], 0)
                    _qgemv_expert!(ybufs[k], hacts[c], down_s, e; alpha=w, beta=0.0f0)
                end
            end
            for k in 1:n_used
                @inbounds for i in 1:h
                    yrow[i] += ybufs[k][i]
                end
            end
        end
    end
    return y
end

# One token's top-k routing: softmax over the router logits, then the top
# n_used experts renormalised. Shared by the batched host route and the device
# hot-expert route, so both select the same experts with the same weights.
# Returns false when the token routes nothing (weights sum to zero).
function _moe_route_token!(experts, wts, t::Int, x, router, logits, order, n_used::Int)
    n_exp = length(logits)
    m = _cpu_router_logits!(logits, x, t, router)
    s = 0.0f0
    for e in 1:n_exp
        v = exp(logits[e] - m)
        logits[e] = v
        s += v
    end
    invs = 1.0f0 / s
    for e in 1:n_exp
        logits[e] *= invs
        order[e] = e
    end
    sort!(order; by=e -> (-logits[e], e))
    wsum = 0.0f0
    for k in 1:n_used
        wsum += logits[order[k]]
    end
    wsum > 0 || return false
    invw = 1.0f0 / wsum
    for k in 1:n_used
        experts[k, t] = order[k]
        wts[k, t] = logits[order[k]] * invw
    end
    return true
end

# Routing is the per-expert loop's, op for op, so the selected experts and
# their weights match it exactly.
function _cpu_moe_topk_swiglu_batched!(
    y,
    x,
    router,
    gate_s,
    up_s,
    down_s,
    n_used,
    logits,
    order,
)
    seq, h = size(x)
    n_exp = length(logits)
    experts = Matrix{Int}(undef, n_used, seq)
    wts = Matrix{Float32}(undef, n_used, seq)
    live = fill(false, seq)
    for t in 1:seq
        live[t] = _moe_route_token!(experts, wts, t, x, router, logits, order, n_used)
    end
    if seq == 1
        live[1] || return y
        xf = Vector{Float32}(undef, h)
        yrow = Vector{Float32}(undef, h)
        @inbounds for i in 1:h
            xf[i] = x[1, i]
        end
        _moe_host_experts_auto!(
            yrow,
            xf,
            view(experts, :, 1),
            view(wts, :, 1),
            n_used,
            gate_s,
            up_s,
            down_s,
        )
        @inbounds for i in 1:h
            y[1, i] = yrow[i]
        end
        return y
    end
    X = x isa Matrix{Float32} ? x : Matrix{Float32}(x)
    Y = y isa Matrix{Float32} ? y : Matrix{Float32}(undef, seq, h)
    _moe_host_prefill!(Y, X, experts, wts, live, n_exp, gate_s, up_s, down_s)
    Y === y || copyto!(y, Y)
    return y
end

export gguf_header,
    gguf_architecture_spec,
    spec_from_gguf_kv,
    load_gguf,
    QuantMatrix,
    GgufQuantSlice,
    GpuQuantSlice,
    HostOffload,
    gguf_dequant_expert
