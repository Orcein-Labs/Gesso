#!/usr/bin/env julia
# Writes the deepseek41 toy GGUF used by test/test_deepseek41.jl: random F32
# weights for every tensor the forward reads, with the metadata from
# test/deepseek41_toy.jl. Deterministic (fixed seed). Usage:
#
#     julia --project=. tools/make-deepseek41-toy-gguf.jl [out.gguf]
#
# Tensor shapes follow llama.cpp's load_arch_tensors (src/models/deepseek41.cpp)
# with dims in ggml order (ne0 first).

using Random

include(joinpath(@__DIR__, "..", "test", "deepseek41_toy.jl"))

const _GGUF_F32 = Int32(0)

function _kv_str!(io, s::AbstractString)
    write(io, UInt64(sizeof(s)))
    write(io, codeunits(s))
end

# Keys llama.cpp reads as signed int32 (GGUF type 5); everything else integer is uint32.
const _I32_KEYS = ("deepseek41.attention.indexer.candidate_source_layer",)
# Integer arrays llama.cpp reads as uint64 (std::vector<uint64_t> in the Engram hash setup).
const _U64_ARR_KEYS = ("deepseek41.ple.layer_multipliers", "deepseek41.ple.head_offsets",
                       "deepseek41.ple.head_vocab_sizes")

# GGUF value types: 4 uint32, 5 int32, 6 float32, 7 bool, 8 string, 9 array, 10 uint64.
function _kv!(io, key::AbstractString, val)
    _kv_str!(io, key)
    if key in _I32_KEYS
        write(io, UInt32(5)); write(io, Int32(val))
    elseif key in _U64_ARR_KEYS
        write(io, UInt32(9)); write(io, UInt32(10)); write(io, UInt64(length(val)))
        foreach(v -> write(io, UInt64(v)), val)
    elseif val isa AbstractString
        write(io, UInt32(8)); _kv_str!(io, val)
    elseif val isa Bool
        write(io, UInt32(7)); write(io, UInt8(val))
    elseif val isa Float64 || val isa Float32
        write(io, UInt32(6)); write(io, Float32(val))
    elseif val isa Integer
        write(io, UInt32(4)); write(io, UInt32(val))
    elseif val isa AbstractVector{<:AbstractString}
        write(io, UInt32(9)); write(io, UInt32(8)); write(io, UInt64(length(val)))
        foreach(v -> _kv_str!(io, v), val)
    elseif val isa AbstractVector{UInt64}
        write(io, UInt32(9)); write(io, UInt32(10)); write(io, UInt64(length(val)))
        foreach(v -> write(io, v), val)
    elseif val isa AbstractVector{<:Integer}
        write(io, UInt32(9)); write(io, UInt32(4)); write(io, UInt64(length(val)))
        foreach(v -> write(io, UInt32(v)), val)
    else
        error("make-deepseek41-toy-gguf: unsupported metadata value $(typeof(val)) for $key")
    end
end

# tensors: name => (dims in ne0-first order, values as Float32 array)
function _write_gguf(path::AbstractString, kvs, tensors)
    align = 32
    io = IOBuffer()
    write(io, b"GGUF"); write(io, UInt32(3))
    write(io, UInt64(length(tensors))); write(io, UInt64(length(kvs)))
    foreach(((k, v),) -> _kv!(io, k, v), kvs)
    offs = UInt64[]
    cur = 0
    for (_, (_, x)) in tensors
        cur = cld(cur, align) * align
        push!(offs, UInt64(cur))
        cur += 4 * length(x)
    end
    for ((name, (dims, _)), off) in zip(tensors, offs)
        _kv_str!(io, name)
        write(io, UInt32(length(dims)))
        foreach(d -> write(io, UInt64(d)), dims)
        write(io, UInt32(_GGUF_F32)); write(io, off)
    end
    hdr = take!(io)
    out = IOBuffer()
    write(out, hdr)
    write(out, zeros(UInt8, mod(-length(hdr), align)))
    base = 0
    for ((_, (_, x)), off) in zip(tensors, offs)
        Int(off) > base && write(out, zeros(UInt8, Int(off) - base))
        write(out, Float32.(vec(x)))
        base = Int(off) + 4 * length(x)
    end
    open(path, "w") do f
        write(f, take!(out))
    end
    return path
end

function make_deepseek41_toy(path::AbstractString; seed::Integer = 20261103)
    rng = Xoshiro(seed)
    kv = _ds41_toy_kv()
    cfg = (n_embd = 16, n_vocab = 40, n_layer = 6, n_head = 4, hd = 8, q_lora = 8,
           o_groups = 2, o_rank = 4, hc = 4, idx_head = 2, idx_dim = 4,
           n_expert = 4, ff = 6, ple_rows = 80, ple_dim = 4, engram_in = 16)
    hc_mix = (2 + cfg.hc) * cfg.hc
    hc_dim = cfg.hc * cfg.n_embd

    # matrices get std 1/sqrt(fan_in); norms near 1; scalars small
    mat(dims...) = randn(rng, Float32, dims...) ./ sqrt(Float32(dims[1]))
    norm(n) = 1f0 .+ 0.1f0 .* randn(rng, Float32, n)
    small(dims...) = 0.1f0 .* randn(rng, Float32, dims...)

    T = Pair{String, Tuple{Vector{Int}, Array{Float32}}}
    tensors = T[]
    add!(name, dims, x) = push!(tensors, name => (collect(Int, dims), Array{Float32}(x)))

    add!("token_embd.weight", (cfg.n_embd, cfg.n_vocab), mat(cfg.n_embd, cfg.n_vocab))
    add!("output_norm.weight", (cfg.n_embd,), norm(cfg.n_embd))
    add!("output.weight", (cfg.n_embd, cfg.n_vocab), mat(cfg.n_embd, cfg.n_vocab))
    add!("per_layer_token_embd.weight", (cfg.ple_dim, cfg.ple_rows),
         mat(cfg.ple_dim, cfg.ple_rows))

    kv_src = kv["deepseek41.attention.kv_source_layers"]
    idx_src = kv["deepseek41.attention.indexer.source_layers"]
    ratios = kv["deepseek41.attention.compress_ratios"]
    ple_layers = kv["deepseek41.ple.layers"]
    n_out = cfg.n_head * cfg.hd ÷ cfg.o_groups

    for il in 0:(cfg.n_layer - 1)
        p = "blk.$il."
        add!(p * "attn_norm.weight", (cfg.n_embd,), norm(cfg.n_embd))
        add!(p * "attn_sinks.weight", (cfg.n_head,), small(cfg.n_head))
        add!(p * "attn_q_a.weight", (cfg.n_embd, cfg.q_lora), mat(cfg.n_embd, cfg.q_lora))
        add!(p * "attn_q_a_norm.weight", (cfg.q_lora,), norm(cfg.q_lora))
        add!(p * "attn_q_b.weight", (cfg.q_lora, cfg.n_head * cfg.hd),
             mat(cfg.q_lora, cfg.n_head * cfg.hd))
        add!(p * "attn_kv.weight", (cfg.n_embd, cfg.hd), mat(cfg.n_embd, cfg.hd))
        add!(p * "attn_kv_a_norm.weight", (cfg.hd,), norm(cfg.hd))
        add!(p * "attn_output_a.weight", (n_out, cfg.o_rank, cfg.o_groups),
             mat(n_out, cfg.o_rank, cfg.o_groups))
        add!(p * "attn_output_b.weight", (cfg.o_groups * cfg.o_rank, cfg.n_embd),
             mat(cfg.o_groups * cfg.o_rank, cfg.n_embd))

        for part in ("attn", "ffn")
            add!(p * "hc_$(part)_fn.weight", (hc_dim, hc_mix), mat(hc_dim, hc_mix))
            add!(p * "hc_$(part)_base.weight", (hc_mix,), small(hc_mix))
            add!(p * "hc_$(part)_scale.weight", (3,), small(3))
        end

        if il in kv_src
            add!(p * "attn_compressor_kv.weight", (cfg.n_embd, cfg.hd), mat(cfg.n_embd, cfg.hd))
            add!(p * "attn_compressor_norm.weight", (cfg.hd,), norm(cfg.hd))
            ratios[il + 1] > 1 &&
                add!(p * "attn_compressor_gate.weight", (cfg.n_embd, cfg.hd),
                     mat(cfg.n_embd, cfg.hd))
            add!(p * "indexer.attn_k.weight", (cfg.hd, cfg.idx_dim), mat(cfg.hd, cfg.idx_dim))
            add!(p * "indexer.k_norm.weight", (cfg.idx_dim,), norm(cfg.idx_dim))
        end
        if il in idx_src
            add!(p * "indexer.attn_q_b.weight", (cfg.q_lora, cfg.idx_head * cfg.idx_dim),
                 mat(cfg.q_lora, cfg.idx_head * cfg.idx_dim))
            add!(p * "indexer.proj.weight", (cfg.n_embd, cfg.idx_head),
                 mat(cfg.n_embd, cfg.idx_head))
        end
        if il in ple_layers
            n_cols = (kv["deepseek41.ple.ngram_size"] - 1) * kv["deepseek41.ple.heads_per_ngram"]
            add!(p * "engram_kv.weight", (n_cols * cfg.ple_dim, cfg.n_embd * (cfg.hc + 1)),
                 mat(n_cols * cfg.ple_dim, cfg.n_embd * (cfg.hc + 1)))
            add!(p * "engram_gate.weight", (cfg.n_embd, cfg.hc), mat(cfg.n_embd, cfg.hc))
        end

        add!(p * "ffn_gate_inp.weight", (cfg.n_embd, cfg.n_expert), mat(cfg.n_embd, cfg.n_expert))
        add!(p * "exp_probs_b.bias", (cfg.n_expert,), small(cfg.n_expert))
        add!(p * "ffn_norm.weight", (cfg.n_embd,), norm(cfg.n_embd))
        add!(p * "ffn_gate_exps.weight", (cfg.n_embd, cfg.ff, cfg.n_expert),
             mat(cfg.n_embd, cfg.ff, cfg.n_expert))
        add!(p * "ffn_down_exps.weight", (cfg.ff, cfg.n_embd, cfg.n_expert),
             mat(cfg.ff, cfg.n_embd, cfg.n_expert))
        add!(p * "ffn_up_exps.weight", (cfg.n_embd, cfg.ff, cfg.n_expert),
             mat(cfg.n_embd, cfg.ff, cfg.n_expert))
        add!(p * "ffn_gate_shexp.weight", (cfg.n_embd, cfg.ff), mat(cfg.n_embd, cfg.ff))
        add!(p * "ffn_down_shexp.weight", (cfg.ff, cfg.n_embd), mat(cfg.ff, cfg.n_embd))
        add!(p * "ffn_up_shexp.weight", (cfg.n_embd, cfg.ff), mat(cfg.n_embd, cfg.ff))
    end

    _write_gguf(path, collect(kv), tensors)
    return path, tensors
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = length(ARGS) >= 1 ? ARGS[1] :
        joinpath(@__DIR__, "..", "test", "fixtures", "deepseek41_toy", "deepseek41-toy.gguf")
    path, tensors = make_deepseek41_toy(out)
    nparam = sum(length(x) for (_, (_, x)) in tensors)
    println("wrote $(path): $(length(tensors)) tensors, $nparam parameters, $(filesize(path)) bytes")
end
