# Public loading failures retain useful source detail within Gesso's taxonomy.
function _load_boundary(f, name)
    try
        return f()
    catch err
        err isa Gesso.GessoException && rethrow()
        throw(
            gesso_error(
                err isa OutOfMemoryError ? Gesso.ERR_ALLOCATION : ERR_INVALID_PLAN,
                "$name: invalid checkpoint input";
                cause=sprint(showerror, err),
                cause_type=string(typeof(err)),
            ),
        )
    end
end
# Llama-family import (§LXXVI item B; §VIII: checkpoint format is transport).
#
# What lives here:
#   * load_llama_config      — config.json → validated NamedTuple
#   * config_to_model        — validated config → ModelIR.Model
#   * load_safetensors       — safetensors file → Dict{String, Array{Float64}}
#   * materialize_llama      — named tensors → the interpreter's tensor set
#   * load_llama             — the whole path: (model, tensors, cfg)
#
# Laws:
#   * Upcast at load: BF16/F16/F32/F64 → Float64 exactly (§LXXVI). Any other
#     dtype errors loudly — no silent representation change (§LXX).
#   * Unknown checkpoint keys error WITH THE KEY NAME; missing required keys
#     error WITH THE KEY NAME. No silent skips, ever.
#   * `tie_word_embeddings: true` with a present lm_head.weight that is not
#     byte-identical to embed_tokens is an error. Untied heads are out of
#     scope this sprint (§LXXVI).
#   * No network, no Hub, no tokenizer here — item B is config + tensors.
#     JSON parsing is the ONE sanctioned third-party dependency (§LXXVI);
#     Mmap (stdlib) reads the byte region without copying it.

using JSON
using Mmap

# Reject duplicate keys at every nesting level using JSON's existing parser.
# A private dictionary changes insertion policy, not JSON syntax/decoding.
struct _UniqueJSONDict <: AbstractDict{String, Any}
    data::Dict{String, Any}
end
_UniqueJSONDict() = _UniqueJSONDict(Dict{String, Any}())
Base.length(d::_UniqueJSONDict) = length(d.data)
Base.iterate(d::_UniqueJSONDict, args...) = iterate(d.data, args...)
Base.getindex(d::_UniqueJSONDict, k) = d.data[k]
Base.haskey(d::_UniqueJSONDict, k) = haskey(d.data, k)
Base.get(d::_UniqueJSONDict, k, default) = get(d.data, k, default)
function Base.setindex!(d::_UniqueJSONDict, v, k)
    haskey(d, k) && error("checkpoint JSON: duplicate key $(repr(k))")
    return d.data[k] = v
end
_strict_json(text) = JSON.parse(text; dicttype=_UniqueJSONDict)
_strict_jsonfile(path) = _strict_json(read(path, String))

function _checkpoint_int(value, label; minimum=0)
    value isa Integer && !(value isa Bool) && minimum <= value <= typemax(Int) ||
        error("checkpoint: $label must be an integer in $minimum..$(typemax(Int))")
    return Int(value)
end

# --- config ------------------------------------------------------------------

const _REQUIRED_CONFIG_KEYS = (
    "model_type",
    "hidden_act",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling",
    "rope_interleaved",
    "attention_bias",
    "tie_word_embeddings",
)

"""
    load_llama_config(path) -> NamedTuple

Read and VALIDATE a Llama `config.json`. Every refusal names the offending
field (§LXX). Returns a flat NamedTuple of the numbers this sprint uses.
"""
function _load_llama_config_impl(path::AbstractString)
    raw = _strict_jsonfile(path)
    missing_keys = [k for k in _REQUIRED_CONFIG_KEYS if !haskey(raw, k)]
    isempty(missing_keys) || error(
        "load_llama_config: config.json is missing required key(s): $(join(missing_keys, ", "))",
    )

    raw["model_type"] == "llama" || error(
        "load_llama_config: model_type $(repr(raw["model_type"])) is not \"llama\" — not Llama-shaped, refusing (§LXXVI)",
    )
    raw["hidden_act"] == "silu" || error(
        "load_llama_config: hidden_act $(repr(raw["hidden_act"])) is not \"silu\" — refusing (§LXXVI)",
    )
    raw["rope_scaling"] === nothing || error(
        "load_llama_config: rope_scaling must be null this sprint (got $(repr(raw["rope_scaling"])))",
    )
    raw["rope_interleaved"] === false || error(
        "load_llama_config: rope_interleaved must be false — interleaved RoPE is out of scope (§LXXVI)",
    )
    raw["attention_bias"] === false || error(
        "load_llama_config: attention_bias must be false — attention bias is out of scope (§LXXVI)",
    )
    # HuggingFace LlamaConfig defaults mlp_bias to false and SmolLM2-135M's
    # released config.json (transformers 4.40.1) omits the key. Missing ⇒
    # false. Present and true ⇒ refuse. Not a silent representation change.
    mlp_bias = get(raw, "mlp_bias", false)
    mlp_bias === false || error(
        "load_llama_config: mlp_bias must be false — MLP bias is out of scope (§LXXVI)",
    )
    raw["tie_word_embeddings"] === true || error(
        "load_llama_config: tie_word_embeddings must be true — untied heads are out of scope (§LXXVI)",
    )

    hidden = _checkpoint_int(raw["hidden_size"], "hidden_size"; minimum=1)
    n_heads = _checkpoint_int(raw["num_attention_heads"], "num_attention_heads"; minimum=1)
    n_kv = _checkpoint_int(raw["num_key_value_heads"], "num_key_value_heads"; minimum=1)
    hidden > 0 || error("load_llama_config: hidden_size must be positive")
    n_heads > 0 || error("load_llama_config: num_attention_heads must be positive")
    n_kv > 0 || error("load_llama_config: num_key_value_heads must be positive")
    hidden % n_heads == 0 || error(
        "load_llama_config: hidden_size $hidden is not divisible by num_attention_heads $n_heads",
    )
    n_heads % n_kv == 0 || error(
        "load_llama_config: num_attention_heads $n_heads is not divisible by num_key_value_heads $n_kv",
    )

    for key in ("rms_norm_eps", "rope_theta")
        value = raw[key]
        value isa Real && !(value isa Bool) && isfinite(value) && value > 0 ||
            error("load_llama_config: $key must be finite and positive")
    end
    return (
        model_type=String(raw["model_type"]),
        hidden_size=hidden,
        num_hidden_layers=_checkpoint_int(
            raw["num_hidden_layers"],
            "num_hidden_layers";
            minimum=1,
        ),
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        intermediate_size=_checkpoint_int(
            raw["intermediate_size"],
            "intermediate_size";
            minimum=1,
        ),
        vocab_size=_checkpoint_int(raw["vocab_size"], "vocab_size"; minimum=1),
        rms_norm_eps=Float64(raw["rms_norm_eps"]),
        rope_theta=Float64(raw["rope_theta"]),
        tie_word_embeddings=true,
    )
end

"""
    spec_from_llama_cfg(cfg) -> ArchitectureSpec

Lift the strict Llama config NamedTuple back into the canonical
`ArchitectureSpec` (BREADTH-0 Pass A) so the Llama path and the generic
doorway speak the SAME language downstream of validation.
"""
function spec_from_llama_cfg(cfg)
    return ArchitectureSpec(;
        family=Symbol(cfg.model_type),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_hidden_layers,
        n_heads=cfg.num_attention_heads,
        n_kv_heads=cfg.num_key_value_heads,
        vocab_size=cfg.vocab_size,
        intermediate_size=cfg.intermediate_size,
        norm_kind=:rms,
        activation_kind=:swiglu,
        rope=RoPEPolicy(; theta=cfg.rope_theta),
        tie_word_embeddings=cfg.tie_word_embeddings,
    )
end

"""
    llama_param_map() -> FamilyParamMap

The Llama family's external→canonical parameter map (BREADTH-0 Pass B).
The name map is now DATA behind an interface, not logic inside a loop.
"""
llama_param_map() = param_map(LlamaAdapter(), spec_from_llama_cfg(_LLAMA_MAP_PROBE_CFG))

const _LLAMA_MAP_PROBE_CFG = (
    model_type="llama",
    hidden_size=1,
    num_hidden_layers=1,
    num_attention_heads=1,
    num_key_value_heads=1,
    intermediate_size=1,
    vocab_size=1,
    rope_theta=10000.0,
    tie_word_embeddings=true,
)

"""
    config_to_model(cfg) -> Model

Compose a validated config into a `ModelIR.Model` (§VIII: a new architecture
is a new composition of existing primitives — no `LlamaModel` type).
"""
function config_to_model(cfg)
    blocks = ntuple(cfg.num_hidden_layers) do _
        Block(
            Attention(;
                n_heads=cfg.num_attention_heads,
                n_kv_heads=cfg.num_key_value_heads,
            ),
            SwiGLU(hidden=cfg.intermediate_size),
        )
    end
    return Model(;
        vocab_size=cfg.vocab_size,
        embedding=Embedding(dim=cfg.hidden_size),
        blocks=blocks,
    )
end

# --- safetensors reader -------------------------------------------------------

# header dtype → upcast plan. All four legal sources widen to Float64 exactly.
const _SAFETENSORS_DTYPES =
    Dict{String, Int}("BF16" => 2, "F16" => 2, "F32" => 4, "F64" => 8)

# exact BF16 → Float64 via bit manipulation (no Float16 round trip)
function _bf16_to_f64(bytes::Vector{UInt8}, off::Int)
    lo = UInt16(bytes[off])            # little-endian: low byte first
    hi = UInt16(bytes[off+1])
    bits = UInt32(hi) << 8 | lo
    f32bits = UInt32(bits) << 16       # bf16 is the top 16 bits of f32
    return Float64(reinterpret(Float32, f32bits))
end

function _f16_to_f64(bytes::Vector{UInt8}, off::Int)
    b0, b1 = UInt16(bytes[off]), UInt16(bytes[off+1])
    return Float64(reinterpret(Float16, b0 | b1 << 8))
end

_upcast(::Val{:BF16}, bytes, off) = _bf16_to_f64(bytes, off)
_upcast(::Val{:F16}, bytes, off) = _f16_to_f64(bytes, off)
function _upcast(::Val{:F32}, bytes, off)
    b =
        UInt32(bytes[off]) | UInt32(bytes[off+1]) << 8 | UInt32(bytes[off+2]) << 16 |
        UInt32(bytes[off+3]) << 24
    return Float64(reinterpret(Float32, b))
end
function _upcast(::Val{:F64}, bytes, off)
    b =
        UInt64(bytes[off]) | UInt64(bytes[off+1]) << 8 | UInt64(bytes[off+2]) << 16 |
        UInt64(bytes[off+3]) << 24 | UInt64(bytes[off+4]) << 32 |
        UInt64(bytes[off+5]) << 40 | UInt64(bytes[off+6]) << 48 | UInt64(bytes[off+7]) << 56
    return reinterpret(Float64, b)
end

"""
    load_safetensors(path) -> Dict{String, Array{Float64}}

Read a safetensors file: `uint64 header_len`, JSON header, raw bytes. Every
tensor is upcast to `Float64` exactly (BF16/F16/F32/F64 legal sources —
anything else errors with the dtype name). Offsets are relative to the start
of the raw region. Duplicate keys, overlapping/gapped ranges, invalid dimensions
and unindexed trailing bytes are rejected before any tensor allocation.
"""
function _load_safetensors_impl(path::AbstractString)
    isfile(path) || error("load_safetensors: no such file: $path")
    open(path, "r") do io
        file_size = filesize(io)
        file_size >= 8 || error("load_safetensors: truncated length prefix")
        hdr_len = ltoh(read(io, UInt64))
        2 <= hdr_len <= min(100_000_000, file_size - 8) ||
            error("load_safetensors: invalid/truncated header length $hdr_len")
        header_text = String(read(io, Int(hdr_len)))
        startswith(header_text, "{") ||
            error("load_safetensors: header must start with '{'")
        header = _strict_json(header_text)
        header isa AbstractDict || error("load_safetensors: header must be an object")
        data_size = file_size - 8 - Int(hdr_len)
        entries = []
        for (name, meta) in header
            if name == "__metadata__"
                meta isa AbstractDict && all(v isa AbstractString for v in values(meta)) ||
                    error("load_safetensors: __metadata__ must map strings to strings")
                continue
            end
            meta isa AbstractDict &&
            all(haskey(meta, k) for k in ("dtype", "shape", "data_offsets")) ||
                error("load_safetensors: entry $name lacks dtype/shape/data_offsets")
            dt = meta["dtype"]
            dt isa AbstractString && haskey(_SAFETENSORS_DTYPES, dt) ||
                error("load_safetensors: tensor $name has unsupported dtype $(repr(dt))")
            dims = meta["shape"]
            dims isa AbstractVector ||
                error("load_safetensors: tensor $name shape must be an array")
            shape = Tuple(_checkpoint_int(d, "$name shape") for d in dims)
            offsets = meta["data_offsets"]
            offsets isa AbstractVector && length(offsets) == 2 ||
                error("load_safetensors: tensor $name data_offsets must have two integers")
            b0, b1 = (_checkpoint_int(b, "$name offset") for b in offsets)
            0 <= b0 <= b1 <= data_size ||
                error("load_safetensors: tensor $name invalid data range [$b0, $b1)")
            # Checked arithmetic prevents a malicious shape wrapping into a small span.
            n =
                isempty(shape) ? 1 :
                (0 in shape ? 0 : foldl(Base.Checked.checked_mul, shape; init=1))
            esize = _SAFETENSORS_DTYPES[dt]
            Base.Checked.checked_mul(n, esize) == b1 - b0 || error(
                "load_safetensors: tensor $name byte span disagrees with shape and dtype",
            )
            push!(entries, (; name, shape, b0, b1, esize, dt))
        end
        sort!(entries; by=e -> (e.b0, e.b1))
        cursor = 0
        for e in entries
            e.b0 == cursor || error(
                "load_safetensors: overlapping or unindexed bytes at tensor $(e.name)",
            )
            cursor = e.b1
        end
        cursor == data_size || error("load_safetensors: unindexed trailing payload bytes")
        region = data_size == 0 ? UInt8[] : Mmap.mmap(io; grow=false, shared=false)
        tensors = Dict{String, Array{Float64}}()
        for e in entries
            tensors[e.name] =
                _safetensors_array(region, e.b0, e.shape, e.esize, Val(Symbol(e.dt)))
        end
        return tensors
    end
end

# Safetensors is C/row-major, Julia arrays are column-major. Decode the
# contiguous byte stream with reversed dimensions, then reverse the axes
# to preserve tensor coordinates. A function barrier specializes the dtype
# once, rather than dynamically dispatching for every checkpoint element.
function _safetensors_array(bytes, b0, shape, esize, tag::Val)
    raw = Array{Float64}(undef, reverse(shape))
    for i in eachindex(raw)
        raw[i] = _upcast(tag, bytes, b0 + (i - 1) * esize + 1)
    end
    length(shape) <= 1 && return raw
    return permutedims(raw, reverse(ntuple(identity, length(shape))))
end

# --- name map + materialization ------------------------------------------------

# HuggingFace Llama checkpoints often store reconstructed RoPE constants
# `*.rotary_emb.inv_freq` per layer. Gesso computes RoPE from `rope_theta`
# and does not consume inv_freq. This suffix is the ONLY known-ignored
# leftover; any other leftover key is an unknown-weight error (§LXX).
const _IGNORED_ROPE_INV_FREQ_SUFFIX = ".rotary_emb.inv_freq"

_is_ignored_rope_inv_freq(name::AbstractString) =
    endswith(name, _IGNORED_ROPE_INV_FREQ_SUFFIX)

# layer-local names → tensor-map fields, in the order the fixture walk uses.
const _LAYER_TENSOR_NAMES = (
    (name="self_attn.q_proj.weight", field=:wq),
    (name="self_attn.k_proj.weight", field=:wk),
    (name="self_attn.v_proj.weight", field=:wv),
    (name="self_attn.o_proj.weight", field=:wo),
    (name="mlp.gate_proj.weight", field=:wgate),
    (name="mlp.up_proj.weight", field=:wup),
    (name="mlp.down_proj.weight", field=:wdown),
    (name="input_layernorm.weight", field=:attn_rms),
    (name="post_attention_layernorm.weight", field=:ffn_rms),
)

"""
    materialize_llama(model, tensors_by_name, cfg) -> NamedTuple

Map checkpoint names onto the interpreter's tensor set
`(embedding, blocks, lm_head, final_rms)`. Semantic family per the §LXXVI
name map: weights are `ProjectionWeight`s (HF Linear is (out, in) — already
our convention), norms are `FrozenParameter`s, `model.norm.weight` becomes
`final_rms`, and the head is the embedding table itself (tied). Unknown keys
and missing keys are errors WITH THE KEY NAME (§LXX). The only known-ignored
leftover is reconstructed RoPE `*.rotary_emb.inv_freq` (not a weight).
"""
function materialize_llama(model, tensors_by_name::Dict{String, Array{Float64}}, cfg)
    # BREADTH-0 Pass B: the Llama path is now a THIN DELEGATION to the
    # family-agnostic binder. The nine hardcoded tensor spellings live in
    # `LlamaAdapter`'s `param_map` (data behind an interface), and the
    # binding walk is `materialize_architecture`. There is ONE name-map
    # implementation, not two — which is the meta-metric (§XVIII).
    bound = materialize_architecture(
        spec_from_llama_cfg(cfg),
        llama_param_map(),
        tensors_by_name,
    )
    return (
        embedding=bound.embedding,
        blocks=bound.blocks,
        lm_head=bound.lm_head,
        final_rms=bound.final_rms,
        rope=bound.rope,
    )
end


"""
    load_llama(dir) -> (model, tensors, cfg)

The whole item-B path: `config.json` + safetensors file(s) in `dir` →
composed ModelIR + materialized tensors the Phase 2/3 interpreter consumes.
Sharded checkpoints via `model.safetensors.index.json` `weight_map` are
followed; a single `model.safetensors` is the ordinary case. Tokenizer
files are NOT this function's concern (item C).
"""
function _load_llama_impl(dir::AbstractString)
    isdir(dir) || error("load_llama: no such directory: $dir")
    cfg = load_llama_config(joinpath(dir, "config.json"))
    model = config_to_model(cfg)

    index_path = joinpath(dir, "model.safetensors.index.json")
    tensors_by_name = Dict{String, Array{Float64}}()
    if isfile(index_path)
        index = _strict_jsonfile(index_path)
        weight_map = get(index, "weight_map", nothing)
        weight_map isa AbstractDict ||
            error("load_llama: model.safetensors.index.json lacks a weight_map object")
        expected = Set{String}()
        shard_paths = Set{String}()
        for (name, shard) in weight_map
            name in expected &&
                error("load_llama: tensor $name appears twice in weight_map")
            shard isa AbstractString &&
            !isempty(shard) &&
            !isabspath(shard) &&
            normpath(shard) == basename(shard) &&
            shard ∉ (".", "..") ||
                error("load_llama: shard must be a local filename (got $(repr(shard)))")
            push!(expected, String(name))
            push!(shard_paths, String(shard))
        end
        for shard in sort(collect(shard_paths))
            shard_tensors = load_safetensors(joinpath(dir, shard))
            for (name, arr) in shard_tensors
                name in expected || error(
                    "load_llama: shard $shard contains tensor $name not listed in weight_map",
                )
                weight_map[name] == shard || error(
                    "load_llama: tensor $name belongs to $(weight_map[name]), not $shard",
                )
                haskey(tensors_by_name, name) &&
                    error("load_llama: tensor $name loaded twice (shard $shard)")
                tensors_by_name[name] = arr
            end
        end
        unfulfilled = sort(collect(setdiff(expected, keys(tensors_by_name))))
        isempty(unfulfilled) || error(
            "load_llama: weight_map names missing tensor(s): $(join(unfulfilled, ", "))",
        )
    else
        merge!(tensors_by_name, load_safetensors(joinpath(dir, "model.safetensors")))
    end

    tensors = materialize_llama(model, tensors_by_name, cfg)
    return (model, tensors, cfg)
end

export load_llama_config, config_to_model, load_safetensors, materialize_llama, load_llama

load_llama_config(path::AbstractString) =
    _load_boundary(() -> _load_llama_config_impl(path), :load_llama_config)

load_safetensors(path::AbstractString) =
    _load_boundary(() -> _load_safetensors_impl(path), :load_safetensors)

load_llama(dir::AbstractString) = _load_boundary(() -> _load_llama_impl(dir), :load_llama)

@doc (@doc _load_llama_config_impl) load_llama_config
@doc (@doc _load_safetensors_impl) load_safetensors
@doc (@doc _load_llama_impl) load_llama
