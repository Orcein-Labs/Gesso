# GGUF transport for qwen35moe (Ornith-1.5-35B-A3B). Header parse is pure
# Julia. Quantized payloads dequant through the local llama.cpp ggml
# `to_float` (same third-column binary the ID-gate already uses). Experts
# stay mmap'd and dequant per selected expert; everything else materializes
# as Float32 in Gesso `(out, in)` layout.

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
    to_float::Ptr{Cvoid}
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
        to_float::Ptr{Cvoid},
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
            to_float,
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
struct HostOffload{T,N}
    data::Array{T,N}
end
HostOffload(data::AbstractArray{T,N}) where {T,N} = HostOffload{T,N}(convert(Array{T,N}, data))
Base.size(s::HostOffload) = size(s.data)
Base.size(s::HostOffload, d::Int) = size(s.data, d)
Base.ndims(s::HostOffload) = ndims(s.data)
Base.eltype(::HostOffload{T}) where {T} = T
Base.length(s::HostOffload) = length(s.data)
Base.sizeof(s::HostOffload) = sizeof(s.data)
_is_host_float_storage(s::HostOffload) = _is_host_float_storage(s.data)
_offload_host_array(s::HostOffload) = s.data
_offload_host_array(s::AbstractArray) = s

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

function _ggml_traits(ggml_type::Integer)
    _ensure_ggml!()
    p = ccall(_GGML_GET_TRAITS[], Ptr{Cvoid}, (Cint,), Int32(ggml_type))
    p == C_NULL && error("gguf_import: ggml_get_type_traits($ggml_type) is null")
    # ggml_type_traits on x86-64: type_name@0, blck_size@8,
    # blck_size_interleave@16, type_size@24, is_quantized@32 + pad, to_float@40.
    blck = Int64(unsafe_load(Ptr{Int64}(p + 8)))
    tsize = Int(unsafe_load(Ptr{Csize_t}(p + 24)))
    to_float = unsafe_load(Ptr{Ptr{Cvoid}}(p + 40))
    return (blck_size=blck, type_size=tsize, to_float=to_float)
end

# --- tiny cursor --------------------------------------------------------------

mutable struct _Cur
    data::Vector{UInt8}
    pos::Int
end
_u8(c::_Cur) = (v = c.data[c.pos]; c.pos += 1; v)
function _u32(c::_Cur)
    p = c.pos
    v =
        UInt32(c.data[p]) |
        UInt32(c.data[p+1]) << 8 |
        UInt32(c.data[p+2]) << 16 |
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
    n = Int(_u64(c))
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
    if t == 0 || t == 1 || t == 7
        c.pos += 1
    elseif t == 2 || t == 3
        c.pos += 2
    elseif t == 4 || t == 5 || t == 6
        c.pos += 4
    elseif t == 8
        n = Int(_u64(c))
        c.pos += n
    elseif t == 10 || t == 11 || t == 12
        c.pos += 8
    elseif t == 9
        at = _u32(c)
        n = Int(_u64(c))
        for _ in 1:n
            _skip_value!(c, at)
        end
    else
        error("gguf_import: unknown value type $t")
    end
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
        n = Int(_u64(c))
        return [_read_value(c, at) for _ in 1:n]
    end
    error("gguf_import: unread value type $t")
end

struct GgufTensorInfo
    name::String
    dims::Vector{Int}
    ggml_type::Int32
    offset::UInt64
end

function gguf_header(path::AbstractString)
    isfile(path) || error("gguf_header: no such file: $path")
    data = open(path, "r") do io
        Mmap.mmap(io, Vector{UInt8}; grow=false, shared=false)
    end
    c = _Cur(data, 1)
    magic = (_u8(c), _u8(c), _u8(c), _u8(c))
    magic == _GGUF_MAGIC || error("gguf_header: not a GGUF file: $path")
    version = _u32(c)
    version >= 2 || error("gguf_header: GGUF version $version is too old")
    n_tensors = Int(_u64(c))
    n_kv = Int(_u64(c))
    kv = Dict{String, Any}()
    for _ in 1:n_kv
        k = _str(c)
        t = _u32(c)
        # tokenizer arrays are hundreds of thousands of strings; keep counts.
        if t == UInt32(9) && startswith(k, "tokenizer.ggml.")
            at = _u32(c)
            n = Int(_u64(c))
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
        dims = Vector{Int}(undef, nd)
        for d in 1:nd
            dims[d] = Int(_u64(c))
        end
        t = Int32(_u32(c))
        off = _u64(c)
        infos[i] = GgufTensorInfo(name, dims, t, off)
    end
    alignment = Int(get(kv, "general.alignment", 32))
    pad = alignment - ((c.pos - 1) % alignment)
    pad == alignment && (pad = 0)
    data_start = (c.pos - 1) + pad
    return (data=data, kv=kv, tensors=infos, data_start=data_start)
end

function spec_from_gguf_kv(kv::AbstractDict)
    arch = String(get(kv, "general.architecture", "qwen35moe"))
    p(key, default=nothing) = get(kv, "$arch.$key", default)
    n_layers = Int(p("block_count"))
    hidden = Int(p("embedding_length"))
    n_heads = Int(p("attention.head_count"))
    n_kv = Int(p("attention.head_count_kv", n_heads))
    n_exp = Int(p("expert_count"))
    n_used = Int(p("expert_used_count"))
    expert_ff = Int(p("expert_feed_forward_length"))
    shared_ff = Int(p("expert_shared_feed_forward_length", expert_ff))
    head_dim = Int(p("attention.key_length", p("attention.head_dim", 0)))
    if head_dim == 0
        head_dim = 256
    end
    toks = get(kv, "tokenizer.ggml.tokens", nothing)
    vocab = if toks isa Integer
        Int(toks)
    elseif toks === nothing
        Int(p("vocab_size", 0))
    else
        length(toks)
    end
    vocab > 0 || error("spec_from_gguf_kv: no vocabulary size in GGUF metadata")
    interval = Int(p("full_attention_interval", 4))
    # llama.cpp: is_recr[i] = ((i+1) % interval != 0) with 0-based i, so
    # 1-based layers interval, 2*interval, … are full attention.
    layer_types = [
        (i % interval == 0 ? :full_attention : :linear_attention) for i in 1:n_layers
    ]
    ssm_group = Int(p("ssm.group_count", 16))
    ssm_state = Int(p("ssm.state_size", 128))
    ssm_dt = Int(p("ssm.time_step_rank", 32))
    ssm_inner = Int(p("ssm.inner_size", 4096))
    conv_k = Int(p("ssm.conv_kernel", 4))
    value_head_dim = ssm_state
    ssm_inner == ssm_dt * value_head_dim || error(
        "spec_from_gguf_kv: ssm.inner_size $ssm_inner ≠ time_step_rank $ssm_dt × state $ssm_state",
    )
    theta = Float64(p("rope.freq_base", 10000.0))
    rotary_dim = Int(p("rope.dimension_count", 0))
    section = Int[Int(x) for x in p("rope.dimension_sections", Int[])]
    isempty(section) || (section = filter(>(0), section))
    mixer = LinearMixerSpec(;
        layer_types,
        num_key_heads=ssm_group,
        key_head_dim=ssm_state,
        num_value_heads=ssm_dt,
        value_head_dim,
        conv_kernel_dim=conv_k,
        v_head_tiled=true,
    )
    moe = MoeSpec(; n_experts=n_exp, n_expert_used=n_used, expert_ff, shared_ff)
    # GGUF already bakes RMSNorm +1 except ssm_norm; do not set :rms_unit_offset.
    feats = Set{Symbol}([:qk_norm, :linear_attention, :attn_output_gate, :mrope, :moe])
    return ArchitectureSpec(;
        family=:qwen3_5_moe,
        hidden_size=hidden,
        num_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        vocab_size=vocab,
        intermediate_size=expert_ff,
        head_dim,
        rope=RoPEPolicy(; theta, rotary_dim, interleaved=false, mrope_section=section),
        tie_word_embeddings=false,
        features=feats,
        mixer,
        rms_norm_eps=Float64(p("attention.layer_norm_rms_epsilon", 1e-6)),
        moe,
    )
end

gguf_architecture_spec(path::AbstractString) = spec_from_gguf_kv(gguf_header(path).kv)

function _dequant_f32(data::Vector{UInt8}, offset0::Int, ggml_type::Int32, n_elem::Int)
    traits = _ggml_traits(ggml_type)
    y = Vector{Float32}(undef, n_elem)
    if ggml_type == 0  # F32
        nbytes = n_elem * 4
        y .= reinterpret(Float32, view(data, (offset0+1):(offset0+nbytes)))
        return y
    end
    traits.to_float == C_NULL &&
        error("gguf_import: no to_float for ggml type $ggml_type")
    ccall(
        traits.to_float,
        Cvoid,
        (Ptr{Cvoid}, Ptr{Float32}, Int64),
        pointer(data, offset0 + 1),
        pointer(y),
        Int64(n_elem),
    )
    return y
end

function _gguf_array(hdr, info::GgufTensorInfo)
    n_elem = prod(info.dims)
    off = Int(hdr.data_start) + Int(info.offset)
    raw = _dequant_f32(hdr.data, off, info.ggml_type, n_elem)
    nd = length(info.dims)
    nd == 1 && return raw
    A = reshape(raw, info.dims...)
    return permutedims(A, nd:-1:1)
end

function _is_expert_tensor(name::AbstractString)
    return occursin("ffn_gate_exps", name) ||
           occursin("ffn_up_exps", name) ||
           occursin("ffn_down_exps", name)
end

function _expert_slice(hdr, info::GgufTensorInfo)
    length(info.dims) == 3 || error("gguf_import: expert tensor $(info.name) is not 3D")
    ne0, ne1, n_exp = info.dims
    traits = _ggml_traits(info.ggml_type)
    n_elem = ne0 * ne1
    nbytes = if info.ggml_type == 0
        n_elem * 4
    else
        rem(n_elem, Int(traits.blck_size)) == 0 || error(
            "gguf_import: $(info.name) n_elem $n_elem not divisible by blck_size $(traits.blck_size)",
        )
        div(n_elem, Int(traits.blck_size)) * traits.type_size
    end
    return GgufQuantSlice(
        hdr.data,
        Int(hdr.data_start) + Int(info.offset),
        info.ggml_type,
        n_exp,
        ne0,
        ne1,
        nbytes,
        traits.to_float,
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
    y = view(raw, 1:n_elem)
    if s.ggml_type == Int32(0)
        y .= reinterpret(Float32, view(s.data, (off + 1):(off + n_elem * 4)))
    else
        s.to_float == C_NULL &&
            error("gguf_dequant_expert: no to_float for ggml type $(s.ggml_type)")
        ccall(
            s.to_float,
            Cvoid,
            (Ptr{Cvoid}, Ptr{Float32}, Int64),
            pointer(s.data, off + 1),
            pointer(raw),
            Int64(n_elem),
        )
    end
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
    for i in 0:(spec.num_layers-1)
        mixer = layer_mixer(spec, i)
        refs =
            mixer === :linear_attention && !isempty(pmap.linear_refs) ? pmap.linear_refs :
            pmap.layer_refs
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
    for j in 0:(n - 1)
        v = div(j, nk * hd)
        k = rem(div(j, hd), nk)
        d = rem(j, hd)
        i = k * (r * hd) + v * hd + d
        p[j + 1] = i + 1
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
    mx === nothing && return A
    nk, nv = mx.num_key_heads, mx.num_value_heads
    nk == nv && return A
    r = div(nv, nk)
    r * nk == nv || return A
    dk, dv = mx.key_head_dim, mx.value_head_dim
    qk = 2 * nk * dk
    if endswith(name, ".attn_qkv.weight")
        size(A, 1) == qk + nv * dv || error("gguf_import: attn_qkv rows $(size(A,1))")
        p = _v_head_perm(nk, r, dv)
        v = _gather_axis(A[(qk + 1):end, :], 1, p)
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
        v = _gather_axis(A[(qk + 1):end, :, :], 1, p)
        return vcat(A[1:qk, :, :], v)
    elseif endswith(name, ".ssm_out.weight")
        return _gather_axis(A, 2, _v_head_perm(nk, r, dv))
    end
    return A
end

function _materialize_gguf_tensor(hdr, info::GgufTensorInfo, spec::ArchitectureSpec, ::Type{T}) where {T}
    if _is_expert_tensor(info.name)
        return _expert_slice(hdr, info)
    end
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

"""
    load_gguf(path; T=Float32) -> (model, tensors, spec)

Load a `qwen35moe` GGUF. Non-expert weights dequant to host `T` (Float32 by
default — F64 cannot hold this checkpoint). Routed experts stay mmap'd as
`GgufQuantSlice` and dequant one selected expert at a time.
"""
function load_gguf(path::AbstractString; T::Type=Float32, verbose::Bool=false)
    hdr = gguf_header(path)
    spec = spec_from_gguf_kv(hdr.kv)
    pmap = gguf_param_map(Qwen35MoeAdapter(), spec)
    info_by = Dict{String, GgufTensorInfo}(info.name => info for info in hdr.tensors)
    required = _required_gguf_names(pmap, spec)
    by = Dict{String, Any}()
    nreq = length(required)
    for (i, name) in enumerate(required)
        haskey(info_by, name) ||
            error("load_gguf: missing required tensor: $name")
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
    0 < n_used <= n_exp || error("_cpu_moe_topk_swiglu_gguf!: n_used $n_used out of 1..$n_exp")
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
    nthr = max(1, Threads.maxthreadid())
    nwork = Threads.nthreads()
    gaccs = [Vector{Float32}(undef, ff) for _ in 1:nthr]
    uaccs = [Vector{Float32}(undef, ff) for _ in 1:nthr]
    hacts = [Vector{Float32}(undef, ff) for _ in 1:nthr]
    xbufs = [Vector{Float32}(undef, h) for _ in 1:nthr]
    ybufs = [zeros(Float32, h) for _ in 1:n_used]
    for t in 1:seq
        xrow = @view x[t, :]
        m = typemin(T)
        for e in 1:n_exp
            acc = zero(T)
            @inbounds for i in 1:h
                acc += xrow[i] * router[e, i]
            end
            logits[e] = acc
            acc > m && (m = acc)
        end
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
        if nwork == 1
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
            Threads.@threads for k in 1:n_used
                tid = Threads.threadid()
                e = order[k]
                w = Float32(logits[e] * invw)
                xf = xbufs[tid]
                if tid != 1
                    copyto!(xf, xf0)
                end
                _qgemv_expert!(gaccs[tid], xf, gate_s, e)
                _qgemv_expert!(uaccs[tid], xf, up_s, e)
                @inbounds for j in 1:ff
                    hacts[tid][j] = _cpu_silu(gaccs[tid][j]) * uaccs[tid][j]
                end
                fill!(ybufs[k], 0)
                _qgemv_expert!(ybufs[k], hacts[tid], down_s, e; alpha=w, beta=0.0f0)
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

export gguf_header,
    gguf_architecture_spec,
    spec_from_gguf_kv,
    load_gguf,
    GgufQuantSlice,
    GpuQuantSlice,
    HostOffload,
    gguf_dequant_expert
