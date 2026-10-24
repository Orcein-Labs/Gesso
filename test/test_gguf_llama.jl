# GGUF `llama` import. Three checks:
#
# 1. Negative: an unknown architecture and a refused rope scaling are rejected
#    at the spec boundary, with no file needed.
# 2. Synthetic parity: a random micro llama is written twice, once as an HF
#    checkpoint and once as an F32 GGUF. The GGUF carries the llama.cpp Q/K row
#    permutation. The two must give the same prefill logits, and flipping the
#    GGUF rope to the HF half-split layout must visibly break them (red control).
# 3. Gated: a real llama GGUF named by GESSO_LLAMA_GGUF. Skipped when unset;
#    no llama-architecture GGUF is present on this machine.

using JSON
using Random: MersenneTwister, randn

const _LLAMA_GGUF = get(ENV, "GESSO_LLAMA_GGUF", "")

const _LM =
    (hidden=32, layers=2, heads=4, kv_heads=2, inter=64, vocab=32, eps=1e-5, theta=10000.0)

# --- minimal writers ----------------------------------------------------------

function _write_safetensors_f32(path, tensors::Dict{String, Array{Float32}})
    header = Dict{String, Any}()
    offset = 0
    names = sort(collect(keys(tensors)))
    chunks = Vector{UInt8}[]
    for name in names
        arr = tensors[name]
        # row-major bytes, matching the reader's C-order transport
        ordered = ndims(arr) <= 1 ? vec(arr) : vec(permutedims(arr, reverse(1:ndims(arr))))
        bytes = collect(reinterpret(UInt8, ordered))
        header[name] = Dict{String, Any}(
            "dtype" => "F32",
            "shape" => collect(size(arr)),
            "data_offsets" => [offset, offset + length(bytes)],
        )
        push!(chunks, bytes)
        offset += length(bytes)
    end
    hdr = JSON.json(header)
    hdr *= " "^((8 - ((8 + length(hdr)) % 8)) % 8)
    open(path, "w") do io
        write(io, UInt64(length(hdr)))
        write(io, hdr)
        foreach(b -> write(io, b), chunks)
    end
end

_gguf_str(io, s::AbstractString) = (write(io, UInt64(sizeof(s))); write(io, s))

function _gguf_kv(io, key::String, value)
    _gguf_str(io, key)
    if value isa AbstractString
        write(io, UInt32(8))
        _gguf_str(io, value)
    elseif value isa Float32
        write(io, UInt32(6))
        write(io, value)
    elseif value isa Integer
        write(io, UInt32(4))
        write(io, UInt32(value))
    elseif value isa AbstractVector{<:AbstractString}
        write(io, UInt32(9))
        write(io, UInt32(8))
        write(io, UInt64(length(value)))
        foreach(s -> _gguf_str(io, s), value)
    else
        error("test writer: unsupported value $(typeof(value))")
    end
end

# `tensors`: name => (Float32 array in HF (out, in) shape). Each is written
# with ggml dims reversed, so the bytes are the row-major HF weights.
function _write_gguf_f32(
    path,
    kv::Vector{Pair{String, Any}},
    tensors::Vector{Pair{String, Array{Float32}}},
)
    align = 32
    io = IOBuffer()
    write(io, UInt8['G', 'G', 'U', 'F'])
    write(io, UInt32(3))
    write(io, UInt64(length(tensors)))
    write(io, UInt64(length(kv)))
    for (k, v) in kv
        _gguf_kv(io, k, v)
    end
    data = IOBuffer()
    for (name, arr) in tensors
        pad = (align - (position(data) % align)) % align
        write(data, zeros(UInt8, pad))
        off = position(data)
        ordered = ndims(arr) <= 1 ? vec(arr) : vec(permutedims(arr, reverse(1:ndims(arr))))
        write(data, collect(reinterpret(UInt8, ordered)))
        _gguf_str(io, name)
        write(io, UInt32(ndims(arr)))
        for d in reverse(size(arr))
            write(io, UInt64(d))
        end
        write(io, UInt32(0))                  # ggml F32
        write(io, UInt64(off))
    end
    # info block ends here; pad to alignment before the data section
    pad = (align - (position(io) % align)) % align
    write(io, zeros(UInt8, pad))
    write(io, take!(data))
    open(path, "w") do f
        write(f, take!(io))
    end
end

# llama.cpp convert_hf_to_gguf: reshape(n_head, 2, hd/2, in).swapaxes(1, 2).
# New row 2b+a takes old row a*(hd/2)+b, which turns HF's half-split pairs
# (b, b+hd/2) into adjacent pairs (2b, 2b+1).
function _llamacpp_qk_permute(W::Matrix{Float32}, n_head::Int)
    rows = size(W, 1)
    hd = rows ÷ n_head
    idx = Int[]
    for h in 0:(n_head-1), j in 0:(hd-1)
        b, a = j ÷ 2, j % 2
        push!(idx, h * hd + a * (hd ÷ 2) + b + 1)
    end
    return W[idx, :]
end

# --- synthetic micro llama -----------------------------------------------------

function _micro_llama_weights(rng)
    d, nh, nkv, inter, V = _LM.hidden, _LM.heads, _LM.kv_heads, _LM.inter, _LM.vocab
    kdim = nkv * div(d, nh)
    w = Dict{String, Array{Float32}}(
        "model.embed_tokens.weight" => randn(rng, Float32, V, d) .* 0.5f0,
        "model.norm.weight" => 1.0f0 .+ 0.1f0 .* randn(rng, Float32, d),
    )
    for i in 0:(_LM.layers-1)
        w["model.layers.$i.self_attn.q_proj.weight"] =
            randn(rng, Float32, d, d) ./ sqrt(Float32(d))
        w["model.layers.$i.self_attn.k_proj.weight"] =
            randn(rng, Float32, kdim, d) ./ sqrt(Float32(d))
        w["model.layers.$i.self_attn.v_proj.weight"] =
            randn(rng, Float32, kdim, d) ./ sqrt(Float32(d))
        w["model.layers.$i.self_attn.o_proj.weight"] =
            randn(rng, Float32, d, d) ./ sqrt(Float32(d))
        w["model.layers.$i.mlp.gate_proj.weight"] =
            randn(rng, Float32, inter, d) ./ sqrt(Float32(d))
        w["model.layers.$i.mlp.up_proj.weight"] =
            randn(rng, Float32, inter, d) ./ sqrt(Float32(d))
        w["model.layers.$i.mlp.down_proj.weight"] =
            randn(rng, Float32, d, inter) ./ sqrt(Float32(inter))
        w["model.layers.$i.input_layernorm.weight"] =
            1.0f0 .+ 0.1f0 .* randn(rng, Float32, d)
        w["model.layers.$i.post_attention_layernorm.weight"] =
            1.0f0 .+ 0.1f0 .* randn(rng, Float32, d)
    end
    return w
end

function _write_micro_hf(dir, w)
    mkpath(dir)
    _write_safetensors_f32(joinpath(dir, "model.safetensors"), w)
    config = Dict{String, Any}(
        "model_type" => "llama",
        "hidden_act" => "silu",
        "hidden_size" => _LM.hidden,
        "num_hidden_layers" => _LM.layers,
        "num_attention_heads" => _LM.heads,
        "num_key_value_heads" => _LM.kv_heads,
        "intermediate_size" => _LM.inter,
        "vocab_size" => _LM.vocab,
        "rms_norm_eps" => _LM.eps,
        "rope_theta" => _LM.theta,
        "rope_scaling" => nothing,
        "rope_interleaved" => false,
        "attention_bias" => false,
        "mlp_bias" => false,
        "tie_word_embeddings" => true,
        "bos_token_id" => 0,
        "eos_token_id" => 0,
        "max_position_embeddings" => 512,
        "torch_dtype" => "float32",
    )
    open(joinpath(dir, "config.json"), "w") do io
        JSON.print(io, config)
    end
    return dir
end

# Writes the llama.cpp layout: GGUF names, Q/K permuted, output.weight dropped
# because the embedding is tied.
function _write_micro_gguf(path, w)
    map_name = Dict{String, String}(
        "model.embed_tokens.weight" => "token_embd.weight",
        "model.norm.weight" => "output_norm.weight",
    )
    kv = Pair{String, Any}[
        "general.architecture"=>"llama",
        "general.alignment"=>UInt32(32),
        "llama.block_count"=>_LM.layers,
        "llama.embedding_length"=>_LM.hidden,
        "llama.feed_forward_length"=>_LM.inter,
        "llama.attention.head_count"=>_LM.heads,
        "llama.attention.head_count_kv"=>_LM.kv_heads,
        "llama.attention.layer_norm_rms_epsilon"=>Float32(_LM.eps),
        "llama.rope.freq_base"=>Float32(_LM.theta),
        "tokenizer.ggml.tokens"=>["t$i" for i in 0:(_LM.vocab-1)],
    ]
    tensors = Pair{String, Array{Float32}}[]
    push!(tensors, "token_embd.weight" => w["model.embed_tokens.weight"])
    push!(tensors, "output_norm.weight" => w["model.norm.weight"])
    for i in 0:(_LM.layers-1)
        p = "model.layers.$i."
        q = _llamacpp_qk_permute(w[p*"self_attn.q_proj.weight"], _LM.heads)
        k = _llamacpp_qk_permute(w[p*"self_attn.k_proj.weight"], _LM.kv_heads)
        b = "blk.$i."
        push!(tensors, b * "attn_norm.weight" => w[p*"input_layernorm.weight"])
        push!(tensors, b * "attn_q.weight" => q)
        push!(tensors, b * "attn_k.weight" => k)
        push!(tensors, b * "attn_v.weight" => w[p*"self_attn.v_proj.weight"])
        push!(tensors, b * "attn_output.weight" => w[p*"self_attn.o_proj.weight"])
        push!(tensors, b * "ffn_norm.weight" => w[p*"post_attention_layernorm.weight"])
        push!(tensors, b * "ffn_gate.weight" => w[p*"mlp.gate_proj.weight"])
        push!(tensors, b * "ffn_up.weight" => w[p*"mlp.up_proj.weight"])
        push!(tensors, b * "ffn_down.weight" => w[p*"mlp.down_proj.weight"])
    end
    _write_gguf_f32(path, kv, tensors)
    return path
end

# --- tests --------------------------------------------------------------------

@testset "gguf llama: unknown architecture is refused" begin
    kv = Dict{String, Any}("general.architecture" => "mistral")
    @test_throws ErrorException Gesso.Inference.spec_from_gguf_kv(kv)
end

@testset "gguf llama: llama3 rope scaling is refused, not dropped" begin
    kv = Dict{String, Any}(
        "general.architecture" => "llama",
        "llama.block_count" => 2,
        "llama.embedding_length" => 32,
        "llama.feed_forward_length" => 64,
        "llama.attention.head_count" => 4,
        "llama.rope.scaling.type" => "yarn",
        "tokenizer.ggml.tokens" => 32,
    )
    @test_throws Gesso.LoweringNotImplemented Gesso.Inference.spec_from_gguf_kv(kv)
end

@testset "gguf llama: synthetic GGUF matches HF prefill logits" begin
    mktempdir() do dir
        w = _micro_llama_weights(MersenneTwister(0x6c6c616d))
        hf_dir = _write_micro_hf(joinpath(dir, "hf"), w)
        gguf_path = _write_micro_gguf(joinpath(dir, "micro-llama.gguf"), w)

        hf_model, hf_tensors, hf_cfg = Gesso.load_llama(hf_dir)
        g_model, g_tensors, g_spec = Gesso.load_gguf(gguf_path)

        @test g_spec.family === :llama
        @test g_spec.tie_word_embeddings
        @test g_spec.rope.interleaved
        @test g_tensors.lm_head === g_tensors.embedding
        @test g_tensors.blocks[1].wq.storage ==
              _llamacpp_qk_permute(w["model.layers.0.self_attn.q_proj.weight"], _LM.heads)

        toks = [0, 5, 9, 17, 3]
        ref = Gesso.reference_prefill(
            hf_model,
            hf_tensors,
            toks;
            eps=hf_cfg.rms_norm_eps,
            theta=hf_cfg.rope_theta,
        )
        got = Gesso.reference_prefill(
            g_model,
            g_tensors,
            toks;
            eps=g_spec.rms_norm_eps,
            theta=g_spec.rope.theta,
        )
        dmax = maximum(abs.(got .- ref))
        @info "gguf llama vs HF" max_abs_delta=dmax
        @test size(got) == size(ref)
        @test dmax < 1e-4

        # Red control: the same GGUF weights read with HF's half-split rope
        # must disagree. Without this the parity above proves nothing.
        flipped = merge(
            g_tensors,
            (rope=Gesso.RoPEPolicy(; theta=g_spec.rope.theta, interleaved=false),),
        )
        bad = Gesso.reference_prefill(
            g_model,
            flipped,
            toks;
            eps=g_spec.rms_norm_eps,
            theta=g_spec.rope.theta,
        )
        dbad = maximum(abs.(bad .- ref))
        @info "gguf llama red control (interleaved=false)" max_abs_delta=dbad
        @test dbad > 1e-2
    end
end

@testset "gguf llama: real GGUF (gated on GESSO_LLAMA_GGUF)" begin
    if isempty(_LLAMA_GGUF) || !isfile(_LLAMA_GGUF)
        @info "GESSO_LLAMA_GGUF unset or missing; real llama GGUF check skipped"
        @test_skip false
    else
        model, tensors, spec = Gesso.load_gguf(_LLAMA_GGUF)
        @test spec.family === :llama
        logits = Gesso.reference_prefill(
            model,
            tensors,
            [0, 1, 2];
            eps=spec.rms_norm_eps,
            theta=spec.rope.theta,
        )
        @test all(isfinite, logits)
    end
end
