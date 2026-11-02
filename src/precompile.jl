# Precompile workload. While the package image is built, tiny synthetic qwen35,
# qwen35moe, lfm2moe and gemma4 GGUFs run through the path `gesso serve` takes: load, tokenizer,
# chat render and two generations on the CPU. The code that path compiles is cached in the
# image, so a first reply does not wait on compilation.
#
# Deps are stdlibs plus JSON, so this is written by hand rather than with
# PrecompileTools. It runs only while generating output, and
# GESSO_PRECOMPILE_WORKLOAD=0 at build time skips it.
#
# The workload must not leave state in the image. Lazily cached settings (SIMD
# and pool modes start at -1 and read ENV on first use), scratch buffers, the
# receipt sink and the CPU pool would otherwise be frozen in as the build saw
# them. Every mutable const global under Gesso is snapshotted before the run
# and restored after it, and the pool is stopped and cleared.

# --- synthetic GGUF ------------------------------------------------------------

function _pc_gguf_str!(io, s::AbstractString)
    write(io, UInt64(sizeof(s)))
    write(io, codeunits(s))
    return nothing
end

function _pc_gguf_kv!(io, key::AbstractString, val)
    _pc_gguf_str!(io, key)
    if val isa AbstractString
        write(io, UInt32(8))
        _pc_gguf_str!(io, val)
    elseif val isa Float32
        write(io, UInt32(6))
        write(io, val)
    elseif val isa Float64
        write(io, UInt32(12))
        write(io, val)
    elseif val isa AbstractVector{<:AbstractString}
        write(io, UInt32(9))
        write(io, UInt32(8))
        write(io, UInt64(length(val)))
        foreach(v -> _pc_gguf_str!(io, v), val)
    elseif val isa AbstractVector{Bool}
        write(io, UInt32(9))
        write(io, UInt32(7))
        write(io, UInt64(length(val)))
        foreach(v -> write(io, UInt8(v)), val)
    elseif val isa AbstractVector{Int32}
        write(io, UInt32(9))
        write(io, UInt32(5))
        write(io, UInt64(length(val)))
        foreach(v -> write(io, v), val)
    elseif val isa AbstractVector{<:Integer}
        write(io, UInt32(9))
        write(io, UInt32(4))
        write(io, UInt64(length(val)))
        foreach(v -> write(io, UInt32(v)), val)
    elseif val isa Integer
        write(io, UInt32(4))
        write(io, UInt32(val))
    else
        error("precompile workload: unsupported kv value $(typeof(val))")
    end
    return nothing
end

# tensors: (name, dims ne0 first, ggml type, bytes)
function _pc_write_gguf(path::AbstractString, kvs, tensors)
    align = 32
    io = IOBuffer()
    write(io, b"GGUF")
    write(io, UInt32(3))
    write(io, UInt64(length(tensors)))
    write(io, UInt64(length(kvs)))
    for (k, v) in kvs
        _pc_gguf_kv!(io, k, v)
    end
    offs = Int[]
    cur = 0
    for t in tensors
        cur = cld(cur, align) * align
        push!(offs, cur)
        cur += length(t[4])
    end
    for (t, off) in zip(tensors, offs)
        _pc_gguf_str!(io, t[1])
        write(io, UInt32(length(t[2])))
        foreach(d -> write(io, UInt64(d)), t[2])
        write(io, UInt32(t[3]))
        write(io, UInt64(off))
    end
    write(io, zeros(UInt8, mod(-position(io), align)))
    base = position(io)
    for (t, off) in zip(tensors, offs)
        write(io, zeros(UInt8, base + off - position(io)))
        write(io, t[4])
    end
    write(path, take!(io))
    return path
end

# GGML ids and block sizes of the quant types the workload uses.
const _PC_Q8_0, _PC_Q4_K, _PC_Q5_K, _PC_Q6_K = Int32(8), Int32(12), Int32(13), Int32(14)
const _PC_Q5_1, _PC_IQ2_XS, _PC_IQ3_XXS = Int32(7), Int32(17), Int32(18)
const _PC_BLOCK = Dict(_PC_Q5_1 => (32, 24), _PC_Q8_0 => (32, 34), _PC_Q4_K => (256, 144), _PC_Q5_K => (256, 176),
    _PC_Q6_K => (256, 210), _PC_IQ2_XS => (256, 74), _PC_IQ3_XXS => (256, 98))

# Pseudo-random quant bytes with small fp16 block scales, so weights stay finite.
function _pc_quant(seed::UInt32, t::Int32, ne0::Int, nrows::Int)
    blk, bsz = _PC_BLOCK[t]
    nb = div(ne0, blk) * nrows
    b = Vector{UInt8}(undef, nb * bsz)
    s = seed
    for i in eachindex(b)
        s = s * 0x0019660d + 0x3c6ef35f
        b[i] = (s >> 24) % UInt8
    end
    scale = reinterpret(NTuple{2, UInt8}, Float16(1.0e-3))
    for k in 0:(nb-1)
        o = k * bsz
        if t == _PC_Q6_K
            b[o+209], b[o+210] = scale
        else
            b[o+1], b[o+2] = scale
            if t != _PC_Q8_0
                b[o+3], b[o+4] = scale
            end
        end
    end
    return b
end

_pc_f32(v::Vector{Float32}) = collect(reinterpret(UInt8, v))
_pc_norm(n) = _pc_f32(fill(1.0f0, n))

# Two layers, linear (GDN) then full attention, shaped and typed like
# Qwen3.5-2B-Q4_K_M at half width. Every projection has at least 2^20 elements,
# so it stays quant resident as in the real model.
function _pc_tensors()
    H, FF, V = 1024, 1024, 1024
    G, DK, NV, DV = 2, 128, 8, 128
    VD, CD = NV * DV, 2 * G * DK + NV * DV
    NH, NKV, HD = 8, 4, 256
    QD, KVD = NH * HD, NKV * HD
    seed = Ref(UInt32(1))
    q(t, ne0, nrows) = _pc_quant(seed[] += UInt32(1), t, ne0, nrows)
    tens = Any[
        ("token_embd.weight", [H, V], _PC_Q6_K, q(_PC_Q6_K, H, V)),
        ("output_norm.weight", [H], Int32(0), _pc_norm(H)),
    ]
    for L in 0:1
        p = "blk.$L."
        push!(tens, (p * "attn_norm.weight", [H], Int32(0), _pc_norm(H)))
        push!(tens, (p * "post_attention_norm.weight", [H], Int32(0), _pc_norm(H)))
        push!(tens, (p * "ffn_gate.weight", [H, FF], _PC_Q4_K, q(_PC_Q4_K, H, FF)))
        push!(tens, (p * "ffn_up.weight", [H, FF], _PC_Q4_K, q(_PC_Q4_K, H, FF)))
        down = L == 0 ? _PC_Q6_K : _PC_Q4_K
        push!(tens, (p * "ffn_down.weight", [FF, H], down, q(down, FF, H)))
        if L == 0
            push!(tens, (p * "attn_qkv.weight", [H, CD], _PC_Q5_K, q(_PC_Q5_K, H, CD)))
            push!(tens, (p * "attn_gate.weight", [H, VD], _PC_Q4_K, q(_PC_Q4_K, H, VD)))
            push!(tens, (p * "ssm_alpha.weight", [H, NV], _PC_Q8_0, q(_PC_Q8_0, H, NV)))
            push!(tens, (p * "ssm_beta.weight", [H, NV], _PC_Q8_0, q(_PC_Q8_0, H, NV)))
            push!(tens, (p * "ssm_conv1d.weight", [4, CD], Int32(0), _pc_f32(fill(0.1f0, 4 * CD))))
            push!(tens, (p * "ssm_a", [NV], Int32(0), _pc_f32(fill(-0.5f0, NV))))
            push!(tens, (p * "ssm_dt.bias", [NV], Int32(0), _pc_f32(fill(0.1f0, NV))))
            push!(tens, (p * "ssm_norm.weight", [DV], Int32(0), _pc_norm(DV)))
            push!(tens, (p * "ssm_out.weight", [VD, H], _PC_Q5_K, q(_PC_Q5_K, VD, H)))
        else
            push!(tens, (p * "attn_q.weight", [H, 2 * QD], _PC_Q4_K, q(_PC_Q4_K, H, 2 * QD)))
            push!(tens, (p * "attn_k.weight", [H, KVD], _PC_Q4_K, q(_PC_Q4_K, H, KVD)))
            push!(tens, (p * "attn_v.weight", [H, KVD], _PC_Q6_K, q(_PC_Q6_K, H, KVD)))
            push!(tens, (p * "attn_output.weight", [QD, H], _PC_Q4_K, q(_PC_Q4_K, QD, H)))
            push!(tens, (p * "attn_q_norm.weight", [HD], Int32(0), _pc_norm(HD)))
            push!(tens, (p * "attn_k_norm.weight", [HD], Int32(0), _pc_norm(HD)))
        end
    end
    return tens
end

# A byte-level vocabulary: the 256 byte characters, the chat specials, one merge,
# then filler up to the embedding's 1024 rows. `bos`, `eos` and `pad` name
# specials.
function _pc_tokenizer_kv(pre, specials, template; eos, pad, bos=nothing)
    tokens = [string(Inference._BYTES_TO_UNICODE[UInt8(b)]) for b in 0:255]
    types = fill(Int32(1), 256)
    for s in specials
        push!(tokens, s)
        push!(types, Int32(3))
    end
    push!(tokens, "hi")
    push!(types, Int32(1))
    while length(tokens) < 1024
        push!(tokens, "<|pc_$(length(tokens))|>")
        push!(types, Int32(5))
    end
    id(s) = findfirst(==(s), tokens) - 1
    kv = Pair{String, Any}[
        "tokenizer.ggml.model"=>"gpt2",
        "tokenizer.ggml.pre"=>pre,
        "tokenizer.ggml.tokens"=>tokens,
        "tokenizer.ggml.token_type"=>types,
        "tokenizer.ggml.merges"=>["h i"],
        "tokenizer.ggml.eos_token_id"=>id(eos),
        "tokenizer.ggml.padding_token_id"=>id(pad),
        # Rendering is native; the family is read from these markers.
        "tokenizer.chat_template"=>template,
    ]
    bos === nothing || push!(kv, "tokenizer.ggml.bos_token_id"=>id(bos))
    return kv
end

function _pc_kv()
    kv = Pair{String, Any}[
        "general.architecture"=>"qwen35",
        "qwen35.block_count"=>2,
        "qwen35.context_length"=>4096,
        "qwen35.embedding_length"=>1024,
        "qwen35.feed_forward_length"=>1024,
        "qwen35.attention.head_count"=>8,
        "qwen35.attention.head_count_kv"=>4,
        "qwen35.attention.key_length"=>256,
        "qwen35.attention.value_length"=>256,
        "qwen35.attention.layer_norm_rms_epsilon"=>Float32(1e-6),
        "qwen35.full_attention_interval"=>2,
        "qwen35.rope.dimension_count"=>64,
        "qwen35.rope.dimension_sections"=>Int32[11, 11, 10, 0],
        "qwen35.rope.freq_base"=>Float32(1.0e7),
        "qwen35.ssm.conv_kernel"=>4,
        "qwen35.ssm.group_count"=>2,
        "qwen35.ssm.state_size"=>128,
        "qwen35.ssm.time_step_rank"=>8,
        "qwen35.ssm.inner_size"=>1024,
    ]
    tok = _pc_tokenizer_kv("qwen35", ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"),
        "<function=…> <tool_response> <think>"; eos="<|im_end|>", pad="<|endoftext|>")
    return append!(kv, tok)
end

# Three lfm2moe blocks, shaped and typed like LFM2.5-8B-A1B-Q4_K_M with a
# smaller vocabulary, FFN and expert bank: a dense short-convolution block, an
# attention block and a short-convolution block, the last two with experts.
# Every projection has at least 2^20 elements, so it stays quant resident as in
# the real model; the router stays F32.
function _pc_lfm_tensors()
    H, FF, V = 2048, 1024, 1024
    NH, NKV, HD = 32, 8, 64
    E, EFF = 8, 256
    seed = Ref(UInt32(101))
    q(t, dims...) = _pc_quant(seed[] += UInt32(1), t, dims[1], prod(dims[2:end]))
    tens = Any[
        ("token_embd.weight", [H, V], _PC_Q8_0, q(_PC_Q8_0, H, V)),
        ("token_embd_norm.weight", [H], Int32(0), _pc_norm(H)),
    ]
    for L in 0:2
        p = "blk.$L."
        push!(tens, (p * "attn_norm.weight", [H], Int32(0), _pc_norm(H)))
        push!(tens, (p * "ffn_norm.weight", [H], Int32(0), _pc_norm(H)))
        if L == 1
            push!(tens, (p * "attn_q.weight", [H, NH * HD], _PC_Q8_0, q(_PC_Q8_0, H, NH * HD)))
            push!(tens, (p * "attn_k.weight", [H, NKV * HD], _PC_Q8_0, q(_PC_Q8_0, H, NKV * HD)))
            push!(tens, (p * "attn_v.weight", [H, NKV * HD], _PC_Q8_0, q(_PC_Q8_0, H, NKV * HD)))
            push!(tens, (p * "attn_output.weight", [NH * HD, H], _PC_Q8_0, q(_PC_Q8_0, NH * HD, H)))
            push!(tens, (p * "attn_q_norm.weight", [HD], Int32(0), _pc_norm(HD)))
            push!(tens, (p * "attn_k_norm.weight", [HD], Int32(0), _pc_norm(HD)))
        else
            push!(tens, (p * "shortconv.conv.weight", [3, H], Int32(0), _pc_f32(fill(0.1f0, 3 * H))))
            push!(tens, (p * "shortconv.in_proj.weight", [H, 3 * H], _PC_Q4_K, q(_PC_Q4_K, H, 3 * H)))
            push!(tens, (p * "shortconv.out_proj.weight", [H, H], _PC_Q4_K, q(_PC_Q4_K, H, H)))
        end
        if L == 0
            push!(tens, (p * "ffn_gate.weight", [H, FF], _PC_Q8_0, q(_PC_Q8_0, H, FF)))
            push!(tens, (p * "ffn_up.weight", [H, FF], _PC_Q8_0, q(_PC_Q8_0, H, FF)))
            push!(tens, (p * "ffn_down.weight", [FF, H], _PC_Q8_0, q(_PC_Q8_0, FF, H)))
        else
            push!(tens, (p * "exp_probs_b.bias", [E], Int32(0), _pc_f32(zeros(Float32, E))))
            router = Float32[0.01f0 * sin(Float32(i)) for i in 1:(H*E)]
            push!(tens, (p * "ffn_gate_inp.weight", [H, E], Int32(0), _pc_f32(router)))
            push!(tens, (p * "ffn_gate_exps.weight", [H, EFF, E], _PC_Q4_K, q(_PC_Q4_K, H, EFF, E)))
            push!(tens, (p * "ffn_up_exps.weight", [H, EFF, E], _PC_Q4_K, q(_PC_Q4_K, H, EFF, E)))
            push!(tens, (p * "ffn_down_exps.weight", [EFF, H, E], _PC_Q5_K, q(_PC_Q5_K, EFF, H, E)))
        end
    end
    return tens
end

function _pc_lfm_kv()
    kv = Pair{String, Any}[
        "general.architecture"=>"lfm2moe",
        "lfm2moe.block_count"=>3,
        "lfm2moe.context_length"=>4096,
        "lfm2moe.embedding_length"=>2048,
        "lfm2moe.feed_forward_length"=>1024,
        "lfm2moe.attention.head_count"=>32,
        "lfm2moe.attention.head_count_kv"=>Int32[0, 8, 0],
        "lfm2moe.rope.freq_base"=>Float32(5.0e6),
        "lfm2moe.attention.layer_norm_rms_epsilon"=>Float32(1e-5),
        "lfm2moe.expert_count"=>8,
        "lfm2moe.expert_used_count"=>4,
        "lfm2moe.expert_feed_forward_length"=>256,
        "lfm2moe.leading_dense_block_count"=>1,
        "lfm2moe.expert_gating_func"=>2,
        "lfm2moe.vocab_size"=>1024,
        "lfm2moe.shortconv.l_cache"=>3,
    ]
    tok = _pc_tokenizer_kv("lfm2", ("<|startoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"),
        "CONTINUE_FINAL_MESSAGE_TAG render_tool_calls \"<|im_start|>system\n\"";
        bos="<|startoftext|>", eos="<|im_end|>", pad="<|pad|>")
    return append!(kv, tok)
end

# Two qwen35moe layers, linear (GDN) then full attention, shaped and typed like
# Ornith-1.5-35B-A3B-UD-Q2_K_XL with fewer heads and experts: IQ2_XS gate/up
# and IQ3_XXS down experts, a shared expert and an untied output. Every 2-D
# projection has at least 2^20 elements, so it stays quant resident as in the
# real model; the routers and GDN alpha/beta stay F32.
function _pc_moe_tensors()
    H, V = 2048, 1024
    G, DK, NV, DV = 2, 128, 8, 128
    VD, CD = NV * DV, 2 * G * DK + NV * DV
    NH, NKV, HD = 8, 2, 256
    QD, KVD = NH * HD, NKV * HD
    E, EFF, SFF = 8, 256, 512
    seed = Ref(UInt32(201))
    q(t, dims...) = _pc_quant(seed[] += UInt32(1), t, dims[1], prod(dims[2:end]))
    f32(n, x) = _pc_f32(Float32[x * sin(Float32(i)) for i in 1:n])
    tens = Any[
        ("token_embd.weight", [H, V], _PC_Q5_K, q(_PC_Q5_K, H, V)),
        ("output.weight", [H, V], _PC_Q4_K, q(_PC_Q4_K, H, V)),
        ("output_norm.weight", [H], Int32(0), _pc_norm(H)),
    ]
    for L in 0:1
        p = "blk.$L."
        push!(tens, (p * "attn_norm.weight", [H], Int32(0), _pc_norm(H)))
        push!(tens, (p * "post_attention_norm.weight", [H], Int32(0), _pc_norm(H)))
        push!(tens, (p * "ffn_gate_inp.weight", [H, E], Int32(0), f32(H * E, 0.01f0)))
        push!(tens, (p * "ffn_gate_inp_shexp.weight", [H], Int32(0), f32(H, 0.01f0)))
        push!(tens, (p * "ffn_gate_exps.weight", [H, EFF, E], _PC_IQ2_XS, q(_PC_IQ2_XS, H, EFF, E)))
        push!(tens, (p * "ffn_up_exps.weight", [H, EFF, E], _PC_IQ2_XS, q(_PC_IQ2_XS, H, EFF, E)))
        push!(tens, (p * "ffn_down_exps.weight", [EFF, H, E], _PC_IQ3_XXS, q(_PC_IQ3_XXS, EFF, H, E)))
        push!(tens, (p * "ffn_gate_shexp.weight", [H, SFF], _PC_Q5_K, q(_PC_Q5_K, H, SFF)))
        push!(tens, (p * "ffn_up_shexp.weight", [H, SFF], _PC_Q5_K, q(_PC_Q5_K, H, SFF)))
        push!(tens, (p * "ffn_down_shexp.weight", [SFF, H], _PC_Q6_K, q(_PC_Q6_K, SFF, H)))
        if L == 0
            push!(tens, (p * "attn_qkv.weight", [H, CD], _PC_Q5_K, q(_PC_Q5_K, H, CD)))
            push!(tens, (p * "attn_gate.weight", [H, VD], _PC_Q5_K, q(_PC_Q5_K, H, VD)))
            push!(tens, (p * "ssm_alpha.weight", [H, NV], Int32(0), f32(H * NV, 0.01f0)))
            push!(tens, (p * "ssm_beta.weight", [H, NV], Int32(0), f32(H * NV, 0.01f0)))
            push!(tens, (p * "ssm_conv1d.weight", [4, CD], Int32(0), _pc_f32(fill(0.1f0, 4 * CD))))
            push!(tens, (p * "ssm_a", [NV], Int32(0), _pc_f32(fill(-0.5f0, NV))))
            push!(tens, (p * "ssm_dt.bias", [NV], Int32(0), _pc_f32(fill(0.1f0, NV))))
            push!(tens, (p * "ssm_norm.weight", [DV], Int32(0), _pc_norm(DV)))
            push!(tens, (p * "ssm_out.weight", [VD, H], _PC_Q6_K, q(_PC_Q6_K, VD, H)))
        else
            push!(tens, (p * "attn_q.weight", [H, 2 * QD], _PC_Q5_K, q(_PC_Q5_K, H, 2 * QD)))
            push!(tens, (p * "attn_k.weight", [H, KVD], _PC_Q6_K, q(_PC_Q6_K, H, KVD)))
            push!(tens, (p * "attn_v.weight", [H, KVD], _PC_Q6_K, q(_PC_Q6_K, H, KVD)))
            push!(tens, (p * "attn_output.weight", [QD, H], _PC_Q5_K, q(_PC_Q5_K, QD, H)))
            push!(tens, (p * "attn_q_norm.weight", [HD], Int32(0), _pc_norm(HD)))
            push!(tens, (p * "attn_k_norm.weight", [HD], Int32(0), _pc_norm(HD)))
        end
    end
    return tens
end

function _pc_moe_kv()
    kv = Pair{String, Any}[
        "general.architecture"=>"qwen35moe",
        "qwen35moe.block_count"=>2,
        "qwen35moe.context_length"=>4096,
        "qwen35moe.embedding_length"=>2048,
        "qwen35moe.attention.head_count"=>8,
        "qwen35moe.attention.head_count_kv"=>2,
        "qwen35moe.attention.key_length"=>256,
        "qwen35moe.attention.value_length"=>256,
        "qwen35moe.attention.layer_norm_rms_epsilon"=>Float32(1e-6),
        "qwen35moe.expert_count"=>8,
        "qwen35moe.expert_used_count"=>4,
        "qwen35moe.expert_feed_forward_length"=>256,
        "qwen35moe.expert_shared_feed_forward_length"=>512,
        "qwen35moe.full_attention_interval"=>2,
        "qwen35moe.rope.dimension_count"=>64,
        "qwen35moe.rope.dimension_sections"=>Int32[11, 11, 10, 0],
        "qwen35moe.rope.freq_base"=>Float32(1.0e7),
        "qwen35moe.ssm.conv_kernel"=>4,
        "qwen35moe.ssm.group_count"=>2,
        "qwen35moe.ssm.state_size"=>128,
        "qwen35moe.ssm.time_step_rank"=>8,
        "qwen35moe.ssm.inner_size"=>1024,
    ]
    tok = _pc_tokenizer_kv("qwen35", ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"),
        "<function=…> <tool_response> <think>"; eos="<|im_end|>", pad="<|endoftext|>")
    return append!(kv, tok)
end

# Two gemma4 layers, sliding-window then global, shaped and typed like
# gemma-4-26B-A4B-it-UD-Q4_K_M with a smaller width, vocabulary and expert bank:
# Q8_0 attention and dense FFN, Q4_K gate/up and Q5_1 down experts, and a
# k-only global layer with the wider head. Every 2-D projection has at least
# 2^20 elements, so it stays quant resident as in the real model; the router
# stays F32.
function _pc_g4_tensors()
    H, FF, V = 2048, 512, 1024
    NH, KV, HD = 8, (2, 1), (256, 512)
    E, EFF = 8, 256
    seed = Ref(UInt32(301))
    q(t, dims...) = _pc_quant(seed[] += UInt32(1), t, dims[1], prod(dims[2:end]))
    f32(n, x) = _pc_f32(fill(Float32(x), n))
    router = Float32[0.01f0 * sin(Float32(i)) for i in 1:(H*E)]
    tens = Any[
        ("token_embd.weight", [H, V], _PC_Q8_0, q(_PC_Q8_0, H, V)),
        ("output_norm.weight", [H], Int32(0), _pc_norm(H)),
        ("rope_freqs.weight", [HD[2] ÷ 2], Int32(0), f32(HD[2] ÷ 2, 1)),
    ]
    for L in 0:1
        p = "blk.$L."
        hd, kvd, qd = HD[L+1], KV[L+1] * HD[L+1], NH * HD[L+1]
        for n in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm", "post_ffw_norm_1",
            "post_ffw_norm_2", "pre_ffw_norm_2")
            push!(tens, (p * n * ".weight", [H], Int32(0), _pc_norm(H)))
        end
        push!(tens, (p * "attn_q.weight", [H, qd], _PC_Q8_0, q(_PC_Q8_0, H, qd)))
        push!(tens, (p * "attn_k.weight", [H, kvd], _PC_Q8_0, q(_PC_Q8_0, H, kvd)))
        L == 0 && push!(tens, (p * "attn_v.weight", [H, kvd], _PC_Q8_0, q(_PC_Q8_0, H, kvd)))
        push!(tens, (p * "attn_output.weight", [qd, H], _PC_Q8_0, q(_PC_Q8_0, qd, H)))
        push!(tens, (p * "attn_q_norm.weight", [hd], Int32(0), _pc_norm(hd)))
        push!(tens, (p * "attn_k_norm.weight", [hd], Int32(0), _pc_norm(hd)))
        push!(tens, (p * "ffn_gate.weight", [H, FF], _PC_Q8_0, q(_PC_Q8_0, H, FF)))
        push!(tens, (p * "ffn_up.weight", [H, FF], _PC_Q8_0, q(_PC_Q8_0, H, FF)))
        push!(tens, (p * "ffn_down.weight", [FF, H], _PC_Q8_0, q(_PC_Q8_0, FF, H)))
        push!(tens, (p * "ffn_gate_inp.weight", [H, E], Int32(0), _pc_f32(router)))
        push!(tens, (p * "ffn_gate_inp.scale", [H], Int32(0), f32(H, 1)))
        push!(tens, (p * "ffn_gate_up_exps.weight", [H, 2 * EFF, E], _PC_Q4_K, q(_PC_Q4_K, H, 2 * EFF, E)))
        push!(tens, (p * "ffn_down_exps.weight", [EFF, H, E], _PC_Q5_1, q(_PC_Q5_1, EFF, H, E)))
        push!(tens, (p * "ffn_down_exps.scale", [E], Int32(0), f32(E, 1)))
        push!(tens, (p * "layer_output_scale.weight", [1], Int32(0), f32(1, 1)))
    end
    return tens
end

# The gemma4 tokenizer: specials, the 256 byte tokens, single characters with a
# few merges, then normal filler up to the embedding's 1024 rows.
function _pc_g4_tokenizer_kv()
    specials = ("<pad>", "<eos>", "<bos>", "<unk>", "<|turn>", "<turn|>", "<|channel>", "<channel|>",
        "<|think|>", "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>", "<|tool_response>",
        "<tool_response|>")
    tokens = String[specials...]
    types = fill(Int32(3), length(tokens))
    append!(tokens, Inference._gemma4_byte_text(b) for b in 0:255)
    append!(types, fill(Int32(6), 256))
    for c in "\n▁abcdefghijklmnopqrstuvwxyz,."
        push!(tokens, string(c))
        push!(types, Int32(1))
    end
    merges = ["h i", "▁ h", "e r", "u s", "us er", "▁ t", "▁t h", "▁th e"]
    for m in merges
        push!(tokens, replace(m, " " => ""))
        push!(types, Int32(1))
    end
    while length(tokens) < 1024
        push!(tokens, "pc$(length(tokens))")
        push!(types, Int32(1))
    end
    id(s) = findfirst(==(s), tokens) - 1
    return Pair{String, Any}[
        "tokenizer.ggml.model"=>"gemma4",
        "tokenizer.ggml.tokens"=>tokens,
        "tokenizer.ggml.token_type"=>types,
        "tokenizer.ggml.merges"=>merges,
        "tokenizer.ggml.bos_token_id"=>id("<bos>"),
        "tokenizer.ggml.eos_token_id"=>id("<turn|>"),
        "tokenizer.ggml.padding_token_id"=>id("<pad>"),
        "tokenizer.chat_template"=>"<|turn> <turn|> <|channel>thought",
    ]
end

function _pc_g4_kv()
    kv = Pair{String, Any}[
        "general.architecture"=>"gemma4",
        "gemma4.block_count"=>2,
        "gemma4.context_length"=>4096,
        "gemma4.embedding_length"=>2048,
        "gemma4.feed_forward_length"=>512,
        "gemma4.attention.head_count"=>8,
        "gemma4.attention.head_count_kv"=>Int32[2, 1],
        "gemma4.rope.freq_base"=>Float32(1.0e6),
        "gemma4.rope.freq_base_swa"=>Float32(1.0e4),
        "gemma4.attention.layer_norm_rms_epsilon"=>Float32(1e-6),
        "gemma4.expert_count"=>8,
        "gemma4.expert_used_count"=>4,
        "gemma4.attention.key_length"=>512,
        "gemma4.attention.value_length"=>512,
        "gemma4.final_logit_softcapping"=>Float32(30.0),
        "gemma4.attention.sliding_window"=>1024,
        "gemma4.attention.shared_kv_layers"=>0,
        "gemma4.embedding_length_per_layer_input"=>0,
        "gemma4.attention.sliding_window_pattern"=>Bool[true, false],
        "gemma4.attention.key_length_swa"=>256,
        "gemma4.attention.value_length_swa"=>256,
        "gemma4.expert_feed_forward_length"=>256,
        "gemma4.rope.dimension_count"=>512,
        "gemma4.rope.dimension_count_swa"=>256,
    ]
    return append!(kv, _pc_g4_tokenizer_kv())
end

# --- the workload --------------------------------------------------------------

function _precompile_workload(dir::AbstractString)
    for (name, kv, tens) in (("qwen35", _pc_kv(), _pc_tensors()), ("qwen35moe", _pc_moe_kv(), _pc_moe_tensors()),
        ("lfm2moe", _pc_lfm_kv(), _pc_lfm_tensors()), ("gemma4", _pc_g4_kv(), _pc_g4_tensors()))
        path = _pc_write_gguf(joinpath(dir, "precompile-$name.gguf"), kv, tens)
        model, tensors, spec = Inference.load_gguf(path)
        tk = Inference.load_gguf_tokenizer(path)
        chat = gguf_chat_model(model, tensors, spec, tk; backend=CPUBackend(), n_ctx=512,
            model_id="precompile")
        for (text, n) in (("hi", 4), ("Name three rivers in Europe, and say why each matters.", 8))
            msgs = [Dict{String, Any}("role" => "user", "content" => text)]
            ids = chat.tokenize(chat.render(msgs, Any[], false))
            chat.generate(ids; max_new_tokens=n, on_token=_ -> true)
            chat.detokenize(ids)
        end
    end
    return nothing
end

# --- global state snapshot -----------------------------------------------------

function _pc_modules(m::Module, acc=Module[])
    push!(acc, m)
    for n in names(m; all=true, imported=false)
        isdefined(m, n) || continue
        v = getfield(m, n)
        v isa Module && v !== m && parentmodule(v) === m && !(v in acc) && _pc_modules(v, acc)
    end
    return acc
end

_pc_skip(v) =
    !ismutable(v) || v isa Union{Module, Function, Type, String, Symbol, Regex,
        Base.AbstractLock, Task, Base.GenericCondition, IO} || v === Runtime._CPU_POOL_SLOT

# Every mutable const global under Gesso with a deep copy of its value.
function _pc_snapshot()
    snap = Tuple{Any, Any}[]
    for m in _pc_modules(@__MODULE__), n in names(m; all=true, imported=false)
        startswith(String(n), "#") && continue
        (isdefined(m, n) && isconst(m, n)) || continue
        v = getfield(m, n)
        _pc_skip(v) || push!(snap, (v, deepcopy(v)))
    end
    return snap
end

_pc_restore!(x::Base.RefValue, s) = (x[] = s[]; nothing)
_pc_restore!(x::Threads.Atomic, s) = (x[] = s[]; nothing)
_pc_restore!(x::AbstractDict, s) = (empty!(x); merge!(x, s); nothing)
_pc_restore!(x::Vector, s) = (resize!(x, length(s)); copyto!(x, s); nothing)
# Unspecialized on purpose: with the type known at compile time, isconst and
# isfieldatomic fold the type's field-flag C pointers into the code, and this
# method is compiled into the image while precompiling, so a loaded session
# would read freed memory.
function _pc_restore!(@nospecialize(x), @nospecialize(s))
    T = typeof(x)
    for i in 1:fieldcount(T)
        isdefined(s, i) || continue
        f = getfield(x, i)
        if isconst(T, i)
            f isa Union{Vector, AbstractDict} && _pc_restore!(f, getfield(s, i))
        else
            order = Base.isfieldatomic(T, i) ? :sequentially_consistent : :not_atomic
            setfield!(x, i, getfield(s, i), order)
        end
    end
    return nothing
end

function _pc_stop_cpu_pool!()
    p = @atomic Runtime._CPU_POOL_SLOT.pool
    p === nothing && return nothing
    Runtime._cpu_pool_stop!(p)
    foreach(wait, p.workers)
    @atomic Runtime._CPU_POOL_SLOT.pool = nothing
    return nothing
end

# Runs the workload and leaves global state as it was, compiling every method
# inference touched so the image caches it.
function _run_precompile_workload()
    snap = _pc_snapshot()
    track = isdefined(Core.Compiler, :__set_measure_typeinf)
    try
        mktempdir() do dir
            Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                if track
                    Core.Compiler.Timings.reset_timings()
                    Core.Compiler.__set_measure_typeinf(true)
                end
                try
                    _precompile_workload(dir)
                finally
                    if track
                        Core.Compiler.__set_measure_typeinf(false)
                        Core.Compiler.Timings.close_current_timer()
                    end
                end
            end
        end
        if track
            for child in Core.Compiler.Timings._timings[1].children
                precompile(child.mi_info.mi.specTypes)
            end
        end
    finally
        _pc_stop_cpu_pool!()
        for (x, s) in snap
            _pc_restore!(x, s)
        end
    end
    return nothing
end

if ccall(:jl_generating_output, Cint, ()) == 1 && get(ENV, "GESSO_PRECOMPILE_WORKLOAD", "1") != "0"
    let
        Base.Experimental.@force_compile
        _run_precompile_workload()
    end
end
