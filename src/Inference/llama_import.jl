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
function load_llama_config(path::AbstractString)
    raw = JSON.parsefile(String(path))
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
    raw["rope_interleaved"] == false || error(
        "load_llama_config: rope_interleaved must be false — interleaved RoPE is out of scope (§LXXVI)",
    )
    raw["attention_bias"] == false || error(
        "load_llama_config: attention_bias must be false — attention bias is out of scope (§LXXVI)",
    )
    # HuggingFace LlamaConfig defaults mlp_bias to false and SmolLM2-135M's
    # released config.json (transformers 4.40.1) omits the key. Missing ⇒
    # false. Present and true ⇒ refuse. Not a silent representation change.
    mlp_bias = get(raw, "mlp_bias", false)
    mlp_bias == false || error(
        "load_llama_config: mlp_bias must be false — MLP bias is out of scope (§LXXVI)",
    )
    raw["tie_word_embeddings"] == true || error(
        "load_llama_config: tie_word_embeddings must be true — untied heads are out of scope (§LXXVI)",
    )

    hidden = Int(raw["hidden_size"])
    n_heads = Int(raw["num_attention_heads"])
    n_kv = Int(raw["num_key_value_heads"])
    hidden > 0 || error("load_llama_config: hidden_size must be positive")
    n_heads > 0 || error("load_llama_config: num_attention_heads must be positive")
    n_kv > 0 || error("load_llama_config: num_key_value_heads must be positive")
    hidden % n_heads == 0 || error(
        "load_llama_config: hidden_size $hidden is not divisible by num_attention_heads $n_heads",
    )
    n_heads % n_kv == 0 || error(
        "load_llama_config: num_attention_heads $n_heads is not divisible by num_key_value_heads $n_kv",
    )

    return (
        model_type=String(raw["model_type"]),
        hidden_size=hidden,
        num_hidden_layers=Int(raw["num_hidden_layers"]),
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        intermediate_size=Int(raw["intermediate_size"]),
        vocab_size=Int(raw["vocab_size"]),
        rms_norm_eps=Float64(raw["rms_norm_eps"]),
        rope_theta=Float64(raw["rope_theta"]),
        tie_word_embeddings=true,
    )
end

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
of the raw region. Duplicate header entries cannot survive JSON.parse (later
key wins), so the count audit below is the real duplicate guard.
"""
function load_safetensors(path::AbstractString)
    isfile(path) || error("load_safetensors: no such file: $path")
    io = open(path, "r")
    try
        hdr_len = read(io, UInt64)
        hdr_len > (1 << 30) &&
            error("load_safetensors: header length $hdr_len is implausible (> 1 GiB)")
        header = JSON.parse(String(read(io, hdr_len)))

        data_start = 8 + Int(hdr_len)
        # Mmap maps from the stream's CURRENT POSITION — i.e. the region IS
        # the raw data area, so header offsets index it directly (0-based).
        # The file-backed mapping stays valid for the life of the array; it
        # is released when the array is garbage-collected.
        mmap_region = Mmap.mmap(io; grow=false, shared=false)
        begin
            file_size = filesize(path)
            tensors = Dict{String, Array{Float64}}()
            for (name, meta) in header
                name == "__metadata__" && continue
                haskey(meta, "dtype") &&
                haskey(meta, "shape") &&
                haskey(meta, "data_offsets") ||
                    error("load_safetensors: entry $name lacks dtype/shape/data_offsets")
                dt = String(meta["dtype"])
                haskey(_SAFETENSORS_DTYPES, dt) || error(
                    "load_safetensors: tensor $name has unsupported dtype $dt (legal: BF16, F16, F32, F64)",
                )
                esize = _SAFETENSORS_DTYPES[dt]
                shape = Tuple(Int(d) for d in meta["shape"])
                b0, b1 = Int(meta["data_offsets"][1]), Int(meta["data_offsets"][2])
                n = prod(shape)
                n * esize == b1 - b0 || error(
                    "load_safetensors: tensor $name byte span $(b1 - b0) ≠ prod(shape) × sizeof($dt) = $(n * esize)",
                )
                b0 >= 0 && data_start + b1 <= file_size || error(
                    "load_safetensors: tensor $name data range [$b0, $b1) exceeds file size $file_size",
                )

                arr = Array{Float64}(undef, shape)
                tag = Symbol(dt)
                for (i, rel) in enumerate(b0:esize:(b1-1))
                    arr[i] = _upcast(Val{tag}(), mmap_region, rel + 1)
                end
                haskey(tensors, name) && error("load_safetensors: duplicate tensor $name")
                tensors[name] = arr
            end
            return tensors
        end
    finally
        close(io)
    end
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
    consumed = Set{String}()
    grab(name) = begin
        haskey(tensors_by_name, name) ||
            error("materialize_llama: missing required tensor: $name")
        push!(consumed, name)
        return tensors_by_name[name]
    end

    emb = EmbeddingTable(;
        shape=(cfg.vocab_size, cfg.hidden_size),
        storage=grab("model.embed_tokens.weight"),
    )

    # HuggingFace Llama is 0-based (`model.layers.0` … `n-1`). Gesso micro
    # fixtures were written 1-based (`model.layers.1` … `n`). Detect; refuse
    # a mix. Missing both origins names both keys.
    q0 = "model.layers.0.self_attn.q_proj.weight"
    q1 = "model.layers.1.self_attn.q_proj.weight"
    has0, has1 = haskey(tensors_by_name, q0), haskey(tensors_by_name, q1)
    layer_ids = if has0
        0:(cfg.num_hidden_layers-1)
    elseif has1
        1:cfg.num_hidden_layers
    else
        error("materialize_llama: missing required tensor: $q0 (HuggingFace) or $q1 (fixture)")
    end
    n = cfg.num_hidden_layers
    has0 &&
        has1 &&
        n > 1 &&
        haskey(tensors_by_name, "model.layers.$n.self_attn.q_proj.weight") &&
        error(
            "materialize_llama: checkpoint mixes HuggingFace 0-based layers with fixture 1-based layers",
        )

    blocks = map(layer_ids) do i
        fields = map(_LAYER_TENSOR_NAMES) do (suffix, field)
            name = "model.layers.$i.$suffix"
            arr = grab(name)
            if field === :attn_rms || field === :ffn_rms
                return field => FrozenParameter(; shape=size(arr), storage=arr)
            else
                return field => ProjectionWeight(; shape=size(arr), storage=arr)
            end
        end
        (; fields...)
    end

    final_rms = if haskey(tensors_by_name, "model.norm.weight")
        push!(consumed, "model.norm.weight")
        FrozenParameter(;
            shape=(cfg.hidden_size,),
            storage=tensors_by_name["model.norm.weight"],
        )
    else
        nothing
    end

    # tied head: a distinct lm_head must be byte-identical or it is an error
    if haskey(tensors_by_name, "lm_head.weight")
        lm = tensors_by_name["model.embed_tokens.weight"]
        lm_head_arr = tensors_by_name["lm_head.weight"]
        lm == lm_head_arr || error(
            "materialize_llama: lm_head.weight differs from embed_tokens.weight — untied heads are out of scope (§LXXVI)",
        )
        push!(consumed, "lm_head.weight")
    end

    leftover = sort(collect(setdiff(Set(keys(tensors_by_name)), consumed)))
    ignored_inv_freq = filter(_is_ignored_rope_inv_freq, leftover)
    unknown = filter(k -> !_is_ignored_rope_inv_freq(k), leftover)
    isempty(unknown) || error(
        "materialize_llama: unknown tensor(s) in checkpoint: $(join(unknown, ", "))" *
        (
            isempty(ignored_inv_freq) ? "" :
            " (ignored reconstructed RoPE $(_IGNORED_ROPE_INV_FREQ_SUFFIX): $(join(ignored_inv_freq, ", ")))"
        ),
    )

    return (embedding=emb, blocks=collect(blocks), lm_head=emb, final_rms=final_rms)
end

"""
    load_llama(dir) -> (model, tensors, cfg)

The whole item-B path: `config.json` + safetensors file(s) in `dir` →
composed ModelIR + materialized tensors the Phase 2/3 interpreter consumes.
Sharded checkpoints via `model.safetensors.index.json` `weight_map` are
followed; a single `model.safetensors` is the ordinary case. Tokenizer
files are NOT this function's concern (item C).
"""
function load_llama(dir::AbstractString)
    isdir(dir) || error("load_llama: no such directory: $dir")
    cfg = load_llama_config(joinpath(dir, "config.json"))
    model = config_to_model(cfg)

    index_path = joinpath(dir, "model.safetensors.index.json")
    tensors_by_name = Dict{String, Array{Float64}}()
    if isfile(index_path)
        index = JSON.parsefile(String(index_path))
        weight_map = get(index, "weight_map", nothing)
        weight_map isa AbstractDict ||
            error("load_llama: model.safetensors.index.json lacks a weight_map object")
        expected = Set{String}()
        shard_paths = Set{String}()
        for (name, shard) in weight_map
            name in expected &&
                error("load_llama: tensor $name appears twice in weight_map")
            push!(expected, String(name))
            push!(shard_paths, String(shard))
        end
        for shard in sort(collect(shard_paths))
            shard_tensors = load_safetensors(joinpath(dir, shard))
            for (name, arr) in shard_tensors
                name in expected || error(
                    "load_llama: shard $shard contains tensor $name not listed in weight_map",
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
