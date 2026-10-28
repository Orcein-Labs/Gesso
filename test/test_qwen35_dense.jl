# Dense qwen35 GGUF: spec derivation, quantized-resident dense weights, and the
# CPU quant matmul against dequant + F32.
#
# Synthetic fixtures are written to a temp dir and need no environment. The
# 27B header test is gated on GESSO_QWEN38_27B_GGUF; the Ornith tensor check
# on GESSO_ORNITH35_GGUF. Unset means skip.

using Test
using Gesso
using Random

const _GI = Gesso.Inference
const _Q27_GGUF = get(ENV, "GESSO_QWEN38_27B_GGUF", "")
const _ORN_GGUF = get(ENV, "GESSO_ORNITH35_GGUF", "")

# --- GGML type ids and block geometry used by the fixtures ----------------------
const _T_Q2K = Int32(10)
const _T_IQ3XXS = Int32(18)
const _T_IQ4XS = Int32(23)
const _BLK_BYTES = Dict(_T_Q2K => 84, _T_IQ3XXS => 98, _T_IQ4XS => 136)

# Random quant bytes with a finite, small block scale, so weights stay O(0.1).
function _random_quant_bytes(rng, t::Int32, nblocks::Int)
    bsz = _BLK_BYTES[t]
    b = rand(rng, UInt8, nblocks * bsz)
    for k in 0:(nblocks-1)
        o = k * bsz
        if t == _T_Q2K
            b[(o+81):(o+82)] .= reinterpret(UInt8, [Float16(1.0e-3)])
            b[(o+83):(o+84)] .= reinterpret(UInt8, [Float16(1.0e-3)])
        else
            b[(o+1):(o+2)] .= reinterpret(UInt8, [Float16(1.0e-3)])
        end
    end
    return b
end

function _quant_fixture(rng, t::Int32, ne0::Int, nrows::Int)
    tr = _GI._ggml_traits(t)
    rb = div(ne0, Int(tr.blck_size)) * tr.type_size
    data = _random_quant_bytes(rng, t, nrows * div(ne0, 256))
    return _GI.QuantMatrix(data, 0, t, ne0, nrows, rb, tr.blck_size)
end

# --- minimal GGUF v3 writer for fixtures ----------------------------------------

function _gguf_str!(io, s::AbstractString)
    write(io, UInt64(sizeof(s)))
    write(io, codeunits(s))
    return nothing
end

function _gguf_kv!(io, key::AbstractString, val)
    _gguf_str!(io, key)
    if val isa AbstractString
        write(io, UInt32(8))
        _gguf_str!(io, val)
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
        for v in val
            _gguf_str!(io, v)
        end
    elseif val isa AbstractVector{<:Integer}
        write(io, UInt32(9))
        write(io, UInt32(4))
        write(io, UInt64(length(val)))
        for v in val
            write(io, UInt32(v))
        end
    elseif val isa Integer
        write(io, UInt32(4))
        write(io, UInt32(val))
    else
        error("fixture: unsupported kv value $(typeof(val))")
    end
    return nothing
end

# tensors: (name, dims (ne0 first), ggml type, raw bytes)
function _write_gguf(path::AbstractString, kvs::Vector{Pair{String, Any}}, tensors)
    align = 32
    io = IOBuffer()
    write(io, b"GGUF")
    write(io, UInt32(3))
    write(io, UInt64(length(tensors)))
    write(io, UInt64(length(kvs)))
    for (k, v) in kvs
        _gguf_kv!(io, k, v)
    end
    offs = UInt64[]
    cur = 0
    for (_, _, _, bytes) in tensors
        cur = cld(cur, align) * align
        push!(offs, UInt64(cur))
        cur += length(bytes)
    end
    for ((name, dims, t, _), off) in zip(tensors, offs)
        _gguf_str!(io, name)
        write(io, UInt32(length(dims)))
        for d in dims
            write(io, UInt64(d))
        end
        write(io, UInt32(t))
        write(io, off)
    end
    hdr = take!(io)
    out = IOBuffer()
    write(out, hdr)
    pad_h = mod(-length(hdr), align)
    write(out, zeros(UInt8, pad_h))
    base = 0
    for ((_, _, _, bytes), off) in zip(tensors, offs)
        pad = Int(off) - base
        pad > 0 && write(out, zeros(UInt8, pad))
        write(out, bytes)
        base = Int(off) + length(bytes)
    end
    open(path, "w") do f
        write(f, take!(out))
    end
    return path
end

# --- spec derivation -------------------------------------------------------------

function _dense_kv(; arch="qwen35", block_count=65, nextn=1)
    kv = Pair{String, Any}[
        "general.architecture"=>arch,
        "$arch.block_count"=>block_count,
        "$arch.nextn_predict_layers"=>nextn,
        "$arch.embedding_length"=>320,
        "$arch.feed_forward_length"=>960,
        "$arch.attention.head_count"=>6,
        "$arch.attention.head_count_kv"=>2,
        "$arch.attention.key_length"=>64,
        "$arch.attention.layer_norm_rms_epsilon"=>Float32(1e-6),
        "$arch.full_attention_interval"=>4,
        "$arch.rope.dimension_count"=>32,
        "$arch.rope.dimension_sections"=>[8, 8, 0, 0],
        "$arch.rope.freq_base"=>1.0e7,
        "$arch.ssm.conv_kernel"=>4,
        "$arch.ssm.group_count"=>2,
        "$arch.ssm.state_size"=>16,
        "$arch.ssm.time_step_rank"=>6,
        "$arch.ssm.inner_size"=>96,
        "tokenizer.ggml.tokens"=>["t$i" for i in 1:8],
    ]
    return kv
end

@testset "dense qwen35 spec: moe is nothing, layers exclude MTP" begin
    kv = Dict{String, Any}(_dense_kv())
    spec = _GI.spec_from_gguf_kv(kv)
    @test spec.family === :qwen3_5
    @test spec.moe === nothing
    @test :moe ∉ spec.features
    @test spec.num_layers == 64            # block_count 65 minus one MTP block
    @test spec.intermediate_size == 960    # feed_forward_length, not expert keys
    @test spec.hidden_size == 320
    @test spec.n_heads == 6
    @test spec.n_kv_heads == 2
    @test spec.head_dim == 64
    @test spec.mixer.num_value_heads == 6  # from ssm.time_step_rank, not a constant
    @test spec.mixer.num_key_heads == 2
    @test Gesso.layer_mixer(spec, 0) === :linear_attention
    @test Gesso.layer_mixer(spec, 3) === :full_attention
    @test count(==(:full_attention), spec.mixer.layer_types) == 16

    kv["qwen35.nextn_predict_layers"] = 0
    kv["qwen35.block_count"] = 64
    @test _GI.spec_from_gguf_kv(kv).num_layers == 64
end

@testset "spec: qwen35moe path unchanged, unknown arch refused" begin
    kv = Dict{String, Any}(
        "general.architecture" => "qwen35moe",
        "qwen35moe.block_count" => 4,
        "qwen35moe.embedding_length" => 320,
        "qwen35moe.attention.head_count" => 4,
        "qwen35moe.attention.head_count_kv" => 2,
        "qwen35moe.attention.key_length" => 80,
        "qwen35moe.full_attention_interval" => 4,
        "qwen35moe.expert_count" => 16,
        "qwen35moe.expert_used_count" => 4,
        "qwen35moe.expert_feed_forward_length" => 48,
        "qwen35moe.expert_shared_feed_forward_length" => 48,
        "qwen35moe.ssm.conv_kernel" => 4,
        "qwen35moe.ssm.group_count" => 2,
        "qwen35moe.ssm.state_size" => 16,
        "qwen35moe.ssm.time_step_rank" => 4,
        "qwen35moe.ssm.inner_size" => 64,
        "tokenizer.ggml.tokens" => ["a", "b"],
    )
    spec = _GI.spec_from_gguf_kv(kv)
    @test spec.family === :qwen3_5_moe
    @test spec.moe !== nothing
    @test spec.moe.n_experts == 16
    @test spec.intermediate_size == 48
    @test !spec.tie_word_embeddings

    # a file without output.weight has a tied head (Qwen3.5-0.8B ships this way)
    names = Set(["token_embd.weight", "output_norm.weight"])
    tied = _GI.spec_from_gguf_kv(kv; tensor_names=names)
    @test tied.tie_word_embeddings
    @test _GI._gguf_param_map(tied).lm_head_external === nothing
    @test _GI.spec_from_gguf_kv(kv; tensor_names=push!(copy(names), "output.weight")).tie_word_embeddings ==
          false

    @test spec.num_layers == 4

    kv["general.architecture"] = "kolibri1"
    @test_throws Gesso.GessoError _GI.spec_from_gguf_kv(kv)
end

@testset "real 27B header: dense spec and MTP block" begin
    if !isfile(_Q27_GGUF)
        @test_skip "set GESSO_QWEN38_27B_GGUF to the Qwen3.8-27B UD-IQ2_M GGUF"
    else
        spec = Gesso.gguf_architecture_spec(_Q27_GGUF)
        @test spec.family === :qwen3_5
        @test spec.moe === nothing
        @test spec.num_layers == 64
        @test spec.hidden_size == 5120
        @test spec.n_heads == 24
        @test spec.n_kv_heads == 4
        @test spec.head_dim == 256
        @test spec.intermediate_size == 17408
        @test spec.vocab_size == 248320
        @test spec.rope.rotary_dim == 64
        @test spec.rope.mrope_section == [11, 11, 10]
        @test spec.mixer.num_key_heads == 16
        @test spec.mixer.num_value_heads == 48
        @test spec.mixer.value_head_dim == 128
        @test count(==(:full_attention), spec.mixer.layer_types) == 16
        hdr = Gesso.gguf_header(_Q27_GGUF)
        mtp = _GI._gguf_mtp_record(hdr, spec)
        @test mtp !== nothing
        @test mtp.layer == 64
        @test mtp.n_predict == 1
        @test "blk.64.nextn.eh_proj.weight" in mtp.tensor_names
        @test "blk.64.attn_q.weight" in mtp.tensor_names
    end
end

# --- quantized storage: loader policy and row access -----------------------------

@testset "quant residency threshold and F32 small tensors" begin
    big = _GI.GgufTensorInfo("ffn_gate.weight", [1024, 1024], _T_Q2K, UInt64(0))
    small = _GI.GgufTensorInfo("ffn_gate.weight", [256, 512], _T_Q2K, UInt64(0))
    f32 = _GI.GgufTensorInfo("ffn_gate.weight", [1024, 1024], Int32(0), UInt64(0))
    vec1 = _GI.GgufTensorInfo("attn_norm.weight", [1 << 22], _T_Q2K, UInt64(0))
    @test _GI._quant_resident(big)
    @test !_GI._quant_resident(small)     # 131072 < 2^20: F32, as before
    @test !_GI._quant_resident(f32)       # F32 is never quantized
    @test !_GI._quant_resident(vec1)      # 1-D stays F32
end

@testset "row selection: view for runs, gather otherwise" begin
    rng = MersenneTwister(11)
    q = _quant_fixture(rng, _T_IQ4XS, 512, 40)
    full = _GI._quant_host_f32(q)
    run = q[5:9, :]
    @test size(run) == (5, 512)
    @test _GI._quant_host_f32(run) == full[5:9, :]
    pick = q[[12, 3, 30, 31], :]
    @test size(pick) == (4, 512)
    @test _GI._quant_host_f32(pick) == full[[12, 3, 30, 31], :]
end

@testset "quant matmul matches dequant + F32 (native and fallback)" begin
    rng = MersenneTwister(7)
    # All three run the native Q8_K integer kernels, so the activations are
    # quantized and the bound is the Q8_K one (measured about 0.7% of scale).
    for t in (_T_IQ4XS, _T_IQ3XXS, _T_Q2K)
        q = _quant_fixture(rng, t, 512, 96)
        W = _GI._quant_host_f32(q)
        tol = 2.0e-2
        for seq in (1, 3)
            x = randn(rng, Float32, seq, 512)
            dst = zeros(Float32, seq, 96)
            _GI._quant_matmul!(dst, x, q)
            ref = x * transpose(W)
            scale = maximum(abs, ref)
            @test maximum(abs, dst .- ref) <= tol * scale
        end
    end
end

@testset "cpu matmul! and embedding lookup dispatch to quant storage" begin
    rng = MersenneTwister(3)
    q = _quant_fixture(rng, _T_IQ4XS, 512, 64)
    W = _GI._quant_host_f32(q)
    x = randn(rng, Float32, 2, 512)
    w = Gesso.ProjectionWeight((64, 512), q)
    xa = Gesso.Activation(; shape=(2, 512), storage=x)
    da = Gesso.Activation(; shape=(2, 64), storage=zeros(Float32, 2, 64))
    Gesso.matmul!(Gesso.CPUBackend(), da, xa, w, Gesso.PrefillWorkload())
    # Q8_K-quantized activations (native kernel): the same bound as above.
    @test maximum(abs, da.storage .- x * transpose(W)) <=
          2.0e-2 * maximum(abs, x * transpose(W))

    tab = Gesso.EmbeddingTable((64, 512), q)
    dst = Gesso.Activation(; shape=(2, 512), storage=zeros(Float32, 2, 512))
    Gesso.embedding_lookup!(Gesso.CPUBackend(), dst, tab, [5, 0], Gesso.PrefillWorkload())
    @test dst.storage == W[[6, 1], :]
end

# --- end to end: quant-resident dense model == the same weights stored as F32 ----

# A two-layer dense qwen35 (linear then full attention) with an MTP block at
# index 2. Projections are 1024 x 1024 so they clear the 2^20 quant threshold.
function _tiny_dense_tensors(rng)
    H, FF, NV, DK, DV = 1024, 1024, 6, 16, 16
    VD = NV * DV
    CD = 2 * 2 * DK + VD
    QD = 8 * 128
    KVD = 2 * 128
    tens = Any[]
    push!(
        tens,
        (
            "token_embd.weight",
            [H, H],
            _T_Q2K,
            _random_quant_bytes(rng, _T_Q2K, H * div(H, 256)),
        ),
    )
    push!(
        tens,
        (
            "output_norm.weight",
            [H],
            Int32(0),
            reinterpret(UInt8, 1.0f0 .+ 0.05f0 .* randn(rng, Float32, H)),
        ),
    )
    push!(
        tens,
        (
            "output.weight",
            [H, H],
            _T_IQ4XS,
            _random_quant_bytes(rng, _T_IQ4XS, H * div(H, 256)),
        ),
    )
    fp(n, s=0.05f0) = reinterpret(UInt8, s .* randn(rng, Float32, n))
    for L in 0:1
        p = "blk.$L."
        push!(
            tens,
            (
                p * "attn_norm.weight",
                [H],
                Int32(0),
                reinterpret(UInt8, 1.0f0 .+ 0.05f0 .* randn(rng, Float32, H)),
            ),
        )
        push!(
            tens,
            (
                p * "post_attention_norm.weight",
                [H],
                Int32(0),
                reinterpret(UInt8, 1.0f0 .+ 0.05f0 .* randn(rng, Float32, H)),
            ),
        )
        push!(
            tens,
            (
                p * "ffn_gate.weight",
                [H, FF],
                _T_Q2K,
                _random_quant_bytes(rng, _T_Q2K, FF * div(H, 256)),
            ),
        )
        push!(
            tens,
            (
                p * "ffn_up.weight",
                [H, FF],
                _T_IQ3XXS,
                _random_quant_bytes(rng, _T_IQ3XXS, FF * div(H, 256)),
            ),
        )
        push!(
            tens,
            (
                p * "ffn_down.weight",
                [FF, H],
                _T_IQ4XS,
                _random_quant_bytes(rng, _T_IQ4XS, H * div(FF, 256)),
            ),
        )
        if L == 0
            push!(tens, (p * "attn_qkv.weight", [H, CD], Int32(0), fp(H * CD)))
            push!(tens, (p * "attn_gate.weight", [H, VD], Int32(0), fp(H * VD)))
            push!(tens, (p * "ssm_alpha.weight", [H, NV], Int32(0), fp(H * NV)))
            push!(tens, (p * "ssm_beta.weight", [H, NV], Int32(0), fp(H * NV)))
            push!(tens, (p * "ssm_conv1d.weight", [4, CD], Int32(0), fp(4 * CD)))
            push!(
                tens,
                (
                    p * "ssm_a",
                    [NV],
                    Int32(0),
                    reinterpret(UInt8, -(0.2f0 .+ rand(rng, Float32, NV))),
                ),
            )
            push!(tens, (p * "ssm_dt.bias", [NV], Int32(0), fp(NV)))
            push!(
                tens,
                (
                    p * "ssm_norm.weight",
                    [DV],
                    Int32(0),
                    reinterpret(UInt8, 1.0f0 .+ 0.05f0 .* randn(rng, Float32, DV)),
                ),
            )
            push!(tens, (p * "ssm_out.weight", [VD, H], Int32(0), fp(VD * H)))
        else
            push!(
                tens,
                (
                    p * "attn_q.weight",
                    [H, 2 * QD],
                    _T_IQ3XXS,
                    _random_quant_bytes(rng, _T_IQ3XXS, 2 * QD * div(H, 256)),
                ),
            )
            push!(
                tens,
                (
                    p * "attn_k.weight",
                    [H, KVD],
                    _T_IQ3XXS,
                    _random_quant_bytes(rng, _T_IQ3XXS, KVD * div(H, 256)),
                ),
            )
            push!(
                tens,
                (
                    p * "attn_v.weight",
                    [H, KVD],
                    _T_IQ3XXS,
                    _random_quant_bytes(rng, _T_IQ3XXS, KVD * div(H, 256)),
                ),
            )
            push!(
                tens,
                (
                    p * "attn_output.weight",
                    [QD, H],
                    _T_IQ4XS,
                    _random_quant_bytes(rng, _T_IQ4XS, H * div(QD, 256)),
                ),
            )
            push!(
                tens,
                (
                    p * "attn_q_norm.weight",
                    [128],
                    Int32(0),
                    reinterpret(UInt8, 1.0f0 .+ 0.05f0 .* randn(rng, Float32, 128)),
                ),
            )
            push!(
                tens,
                (
                    p * "attn_k_norm.weight",
                    [128],
                    Int32(0),
                    reinterpret(UInt8, 1.0f0 .+ 0.05f0 .* randn(rng, Float32, 128)),
                ),
            )
        end
    end
    # MTP block: must be excluded from the main layers and only recorded.
    push!(
        tens,
        (
            "blk.2.nextn.enorm.weight",
            [H],
            Int32(0),
            reinterpret(UInt8, 1.0f0 .+ 0.0f0 .* ones(Float32, H)),
        ),
    )
    return tens
end

function _tiny_kv()
    kv = Pair{String, Any}[
        "general.architecture"=>"qwen35",
        "qwen35.block_count"=>3,
        "qwen35.nextn_predict_layers"=>1,
        "qwen35.embedding_length"=>1024,
        "qwen35.feed_forward_length"=>1024,
        "qwen35.attention.head_count"=>8,
        "qwen35.attention.head_count_kv"=>2,
        "qwen35.attention.key_length"=>128,
        "qwen35.attention.layer_norm_rms_epsilon"=>Float32(1e-6),
        "qwen35.full_attention_interval"=>2,
        "qwen35.rope.dimension_count"=>64,
        "qwen35.rope.dimension_sections"=>[16, 16, 0, 0],
        "qwen35.rope.freq_base"=>1.0e4,
        "qwen35.ssm.conv_kernel"=>4,
        "qwen35.ssm.group_count"=>2,
        "qwen35.ssm.state_size"=>16,
        "qwen35.ssm.time_step_rank"=>6,
        "qwen35.ssm.inner_size"=>96,
        "tokenizer.ggml.tokens"=>["t$i" for i in 1:1024],
    ]
    return kv
end

# The F32 twin: every quant tensor replaced by its dequantized values in GGUF order.
function _f32_twin(tens)
    out = Any[]
    for (name, dims, t, bytes) in tens
        if t == Int32(0)
            push!(out, (name, dims, Int32(0), bytes))
        else
            n = prod(dims)
            vals = _GI._dequant_f32(collect(bytes), 0, t, n)
            push!(out, (name, dims, Int32(0), reinterpret(UInt8, vals)))
        end
    end
    return out
end

@testset "tiny dense qwen35: quant storage matches F32 end to end" begin
    mktempdir() do dir
        rng = MersenneTwister(2026)
        tens = _tiny_dense_tensors(rng)
        qpath = _write_gguf(joinpath(dir, "quant.gguf"), _tiny_kv(), tens)
        fpath = _write_gguf(joinpath(dir, "f32.gguf"), _tiny_kv(), _f32_twin(tens))

        model_q, tq, spec_q = Gesso.load_gguf(qpath)
        model_f, tf, spec_f = Gesso.load_gguf(fpath)
        @test spec_q.moe === nothing
        @test spec_q.num_layers == 2
        @test tq.mtp !== nothing && tq.mtp.layer == 2
        @test tq.mtp.tensor_names == ["blk.2.nextn.enorm.weight"]
        @test tf.mtp.layer == 2

        # Big projections stay quant; the F32 twin has none.
        @test tq.blocks[1].wgate.storage isa _GI.QuantMatrix
        @test tq.blocks[2].wq.storage isa _GI.QuantMatrix
        @test tq.lm_head.storage isa _GI.QuantMatrix
        @test tq.embedding.storage isa _GI.QuantMatrix
        @test !(tf.blocks[1].wgate.storage isa _GI.QuantMatrix)
        @test tq.blocks[1].wgdn_qkv.storage isa Matrix{Float32}   # small, F32

        ids_q = Gesso.reference_generate(
            model_q,
            tq,
            [5];
            max_new_tokens=4,
            eps=spec_q.rms_norm_eps,
        )
        ids_f = Gesso.reference_generate(
            model_f,
            tf,
            [5];
            max_new_tokens=4,
            eps=spec_f.rms_norm_eps,
        )
        @test length(ids_q) == 5   # prompt + 4 generated
        @test ids_q == ids_f

        # Logits, not just argmax ids: greedy ids can agree by accident on a
        # tiny model. A second weight draw is the negative control.
        prompt = [5, 17, 300]
        Lq = Gesso.reference_prefill(model_q, tq, prompt; eps=spec_q.rms_norm_eps)
        Lf = Gesso.reference_prefill(model_f, tf, prompt; eps=spec_f.rms_norm_eps)
        @test size(Lq) == (1024, 3)
        # Q2_K projections run the native Q8_K integer kernel, which quantizes the
        # activations (about 1% relative per matrix, measured 1.1% end to end on
        # this fixture). The IQ projections are still exact to F32 activations, so
        # the bound covers the Q8_K error, not a gap in the kernels.
        @test maximum(abs, Lq .- Lf) <= 3.0e-2 * maximum(abs, Lf)
        m2, t2, s2 = Gesso.load_gguf(
            _write_gguf(
                joinpath(dir, "other.gguf"),
                _tiny_kv(),
                _f32_twin(_tiny_dense_tensors(MersenneTwister(4051))),
            ),
        )
        L2 = Gesso.reference_prefill(m2, t2, prompt; eps=s2.rms_norm_eps)
        @test maximum(abs, L2 .- Lf) > 1.0e-1 * maximum(abs, Lf)
    end
end

@testset "real Ornith tensor: quant bytes dequant to the F32 loader's values" begin
    if !isfile(_ORN_GGUF)
        @test_skip "set GESSO_ORNITH35_GGUF"
    else
        hdr = Gesso.gguf_header(_ORN_GGUF)
        info = only(filter(t -> t.name == "blk.0.ffn_gate_shexp.weight", hdr.tensors))
        @test _GI._quant_resident(info)
        q = _GI._quant_matrix(hdr, info)
        @test size(q) == (512, 2048)
        @test _GI._quant_host_f32(q) == _GI._gguf_array(hdr, info)
    end
end
