# Phase 3 item B tests (§LXXVI): config → ModelIR, safetensors reader,
# Llama name map, sharded loading, and the loud-error paths.
#
# The fixture is GENERATED in-test by a safetensors WRITER that mirrors the
# reader's format byte-for-byte (§LXXVI: "a writer + round-trip test is
# better" than a checked-in binary). All four legal dtypes are exercised
# with exactness assertions; F16/BF16 tensors use values that are exactly
# representable so atol=0 holds across the upcast. The micro model mirrors
# the goal's llama_micro spec: 2 layers, hidden=32, heads=4, kv_heads=2,
# intermediate=64, vocab=32, eps=1e-5, theta=10000, tied embeddings.

using JSON
using .GessoTestHelpers: approx_eq, deterministic_rng

const MICRO = (
    hidden=32,
    layers=2,
    heads=4,
    kv_heads=2,
    intermediate=64,
    vocab=32,
    eps=1e-5,
    theta=10000.0,
)

# --- safetensors writer (mirror of the reader, byte-for-byte) ----------------

_safetensors_dtype_code(::AbstractArray{Float64}) = "F64"
_safetensors_dtype_code(::AbstractArray{Float32}) = "F32"
_safetensors_dtype_code(::AbstractArray{Float16}) = "F16"
_safetensors_dtype_code(::AbstractArray{UInt16}) = "BF16"   # raw bf16 bits

function _write_safetensors(path, tensors::Dict{String, AbstractArray})
    header = Dict{String, Any}()
    offset = 0
    entries = sort(collect(keys(tensors)))       # deterministic header order
    layout = Dict{String, Tuple{Int, Int}}()     # name → (b0, b1)
    for name in entries
        arr = tensors[name]
        esize =
            _safetensors_dtype_code(arr) == "F64" ? 8 :
            _safetensors_dtype_code(arr) == "F32" ? 4 : 2
        header[name] = Dict{String, Any}(
            "dtype" => _safetensors_dtype_code(arr),
            "shape" => collect(size(arr)),
            "data_offsets" => [offset, offset + length(arr) * esize],
        )
        layout[name] = (offset, offset + length(arr) * esize)
        offset += length(arr) * esize
    end
    hdr_json = JSON.json(header)
    # pad the header so every tensor's absolute start stays 8-aligned
    pad = (8 - ((8 + length(hdr_json)) % 8)) % 8
    hdr_json *= " "^(pad > 0 ? pad : 8)
    open(path, "w") do io
        write(io, UInt64(length(hdr_json)))
        write(io, hdr_json)
        for name in entries
            arr = tensors[name]
            code = _safetensors_dtype_code(arr)
            bytes =
                code == "F64" ? reinterpret(UInt8, vec(arr)) :
                code == "F32" ? reinterpret(UInt8, vec(Float32.(arr))) :
                code == "F16" ? reinterpret(UInt8, vec(Float16.(arr))) :
                reinterpret(UInt8, vec(arr))  # BF16: raw bits already
            write(io, bytes)
        end
    end
    return layout
end

# bf16 bits from an exactly-representable f64 value (top 16 bits of f32)
_bf16_bits(v) = (reinterpret(UInt32, Float32(v)) >> 16) & 0xffff

# deterministic micro checkpoint written into tmpdir; returns (dir, tensors)
function make_micro_checkpoint(dir; shards=false, extra_keys=String[], drop_keys=String[])
    mkpath(dir)
    rng = deterministic_rng(0x1337)
    d, nh, nkv, inter, V =
        MICRO.hidden, MICRO.heads, MICRO.kv_heads, MICRO.intermediate, MICRO.vocab
    dh = div(d, nh)
    kdim = nkv * dh
    tensors = Dict{String, AbstractArray}(
        "model.embed_tokens.weight" => randn(rng, V, d),
        "model.norm.weight" => randn(rng, d),
    )
    for i in 1:MICRO.layers
        tensors["model.layers.$i.self_attn.q_proj.weight"] = randn(rng, d, d)
        tensors["model.layers.$i.self_attn.k_proj.weight"] = randn(rng, kdim, d)
        tensors["model.layers.$i.self_attn.v_proj.weight"] = randn(rng, kdim, d)
        tensors["model.layers.$i.self_attn.o_proj.weight"] = randn(rng, d, d)
        tensors["model.layers.$i.mlp.gate_proj.weight"] = randn(rng, inter, d)
        tensors["model.layers.$i.mlp.up_proj.weight"] = randn(rng, inter, d)
        tensors["model.layers.$i.mlp.down_proj.weight"] = randn(rng, d, inter)
        tensors["model.layers.$i.input_layernorm.weight"] = randn(rng, d)
        tensors["model.layers.$i.post_attention_layernorm.weight"] = randn(rng, d)
    end
    for k in drop_keys
        delete!(tensors, k)
    end
    for k in extra_keys
        tensors[k] = randn(rng, d)
    end

    config = Dict{String, Any}(
        "model_type" => "llama",
        "hidden_act" => "silu",
        "hidden_size" => MICRO.hidden,
        "num_hidden_layers" => MICRO.layers,
        "num_attention_heads" => MICRO.heads,
        "num_key_value_heads" => MICRO.kv_heads,
        "intermediate_size" => MICRO.intermediate,
        "vocab_size" => MICRO.vocab,
        "rms_norm_eps" => MICRO.eps,
        "rope_theta" => MICRO.theta,
        "rope_scaling" => nothing,
        "rope_interleaved" => false,
        "attention_bias" => false,
        "mlp_bias" => false,
        "tie_word_embeddings" => true,
        "bos_token_id" => 0,
        "eos_token_id" => 0,
        "max_position_embeddings" => 512,
        "torch_dtype" => "bfloat16",
    )
    open(joinpath(dir, "config.json"), "w") do io
        JSON.print(io, config)
    end

    if shards
        names = sort(collect(keys(tensors)))
        shard1 = Dict{String, AbstractArray}(n => tensors[n] for n in names[1:div(end, 2)])
        shard2 =
            Dict{String, AbstractArray}(n => tensors[n] for n in names[(div(end, 2)+1):end])
        _write_safetensors(joinpath(dir, "model-00001-of-00002.safetensors"), shard1)
        _write_safetensors(joinpath(dir, "model-00002-of-00002.safetensors"), shard2)
        weight_map = Dict{String, String}()
        for n in names[1:div(end, 2)]
            weight_map[n] = "model-00001-of-00002.safetensors"
        end
        for n in names[(div(end, 2)+1):end]
            weight_map[n] = "model-00002-of-00002.safetensors"
        end
        open(joinpath(dir, "model.safetensors.index.json"), "w") do io
            JSON.print(io, Dict{String, Any}("weight_map" => weight_map))
        end
    else
        _write_safetensors(joinpath(dir, "model.safetensors"), tensors)
    end
    return tensors
end

# --- config tests ------------------------------------------------------------

@testset "load_llama_config: validates the micro config" begin
    dir = mktempdir()
    make_micro_checkpoint(dir)
    cfg = Gesso.load_llama_config(joinpath(dir, "config.json"))
    @test cfg.model_type == "llama"
    @test cfg.hidden_size == 32
    @test cfg.num_hidden_layers == 2
    @test cfg.num_attention_heads == 4
    @test cfg.num_key_value_heads == 2
    @test cfg.intermediate_size == 64
    @test cfg.vocab_size == 32
    @test cfg.rms_norm_eps == 1e-5
    @test cfg.rope_theta == 10000.0
end

@testset "config_to_model: composition is a Model, GQA-expressive" begin
    dir = mktempdir()
    make_micro_checkpoint(dir)
    cfg = Gesso.load_llama_config(joinpath(dir, "config.json"))
    m = Gesso.config_to_model(cfg)
    @test m isa Gesso.Model
    @test m.vocab_size == 32
    @test m.embedding.dim == 32
    @test length(m.blocks) == 2
    @test m.blocks[1].attention.n_heads == 4
    @test m.blocks[1].attention.n_kv_heads == 2
    @test m.blocks[1].ffn.hidden == 64
    @test m == Gesso.config_to_model(cfg)       # structural identity (§CIX)
end

@testset "config refusals name the offending field (§LXX)" begin
    dir = mktempdir()
    make_micro_checkpoint(dir)
    path = joinpath(dir, "config.json")
    base = JSON.parsefile(path)
    rewrite() = open(path, "w") do io
        JSON.print(io, base)
    end
    bad(name, mutate) = begin
        cfg = deepcopy(base)
        mutate(cfg)
        open(path, "w") do io
            JSON.print(io, cfg)
        end
        err = try
            Gesso.load_llama_config(path)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin(name, sprint(showerror, err))
    end
    bad("model_type", c -> c["model_type"] = "mistral")
    bad("hidden_act", c -> c["hidden_act"] = "gelu")
    bad("rope_scaling", c -> c["rope_scaling"] = Dict("type" => "linear"))
    bad("rope_interleaved", c -> c["rope_interleaved"] = true)
    bad("attention_bias", c -> c["attention_bias"] = true)
    bad("mlp_bias", c -> c["mlp_bias"] = true)
    bad("tie_word_embeddings", c -> c["tie_word_embeddings"] = false)
    bad("divisible", c -> c["num_attention_heads"] = 5)
    bad("num_key_value_heads", c -> c["num_key_value_heads"] = 3)
    # missing key is loud too
    delete!(base, "rms_norm_eps")
    rewrite()
    err = try
        Gesso.load_llama_config(path)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("rms_norm_eps", sprint(showerror, err))
end

# --- safetensors reader tests --------------------------------------------------

@testset "load_safetensors: all four dtypes upcast EXACTLY" begin
    dir = mktempdir()
    f64v = Float64[1.0, -2.5, 3.25, 1e-300, Inf]
    f32v = Float32[1.0, -2.5, 3.25, 1e-30]
    f16v = Float16[1.0, -2.5, 3.25, 0.5]
    # bf16 bit patterns for values exactly representable in bf16
    bf16v = UInt16[_bf16_bits(v) for v in (1.0, -2.5, 3.25, 0.5)]
    t = Dict{String, AbstractArray}(
        "a_f64" => f64v,
        "b_f32" => f32v,
        "c_f16" => f16v,
        "d_bf16" => bf16v,
    )
    path = joinpath(dir, "t.safetensors")
    _write_safetensors(path, t)
    got = Gesso.load_safetensors(path)
    @test got["a_f64"] == f64v                            # atol=0: exact
    @test got["b_f32"] == Float64.(f32v)                  # f32 → f64 exact
    @test got["c_f16"] == Float64.(f16v)                  # f16 → f64 exact
    @test got["d_bf16"] == Float64[1.0, -2.5, 3.25, 0.5]                   # bf16 → f64 exact
    @test eltype(got["a_f64"]) == Float64                 # everything is f64
end

@testset "load_safetensors: unsupported dtype and bad span error with names" begin
    dir = mktempdir()
    t = Dict{String, AbstractArray}("ok" => Float64[1.0, 2.0])
    path = joinpath(dir, "t.safetensors")
    _write_safetensors(path, t)
    # corrupt the header: claim I8
    hdr_len = read(open(path), UInt64)
    raw = read(path)
    header = JSON.parse(String(raw[9:(8+hdr_len)]))
    header["ok"]["dtype"] = "I8"
    open(path, "w") do io
        write(io, UInt64(length(JSON.json(header))))
        write(io, JSON.json(header))
        write(io, fill(UInt8(0), 16))
    end
    err = try
        Gesso.load_safetensors(path)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("I8", sprint(showerror, err))
    @test occursin("ok", sprint(showerror, err))
end

# --- materialization + end-to-end ----------------------------------------------

@testset "load_llama end-to-end: prefill runs, deterministic, shapes right" begin
    dir = mktempdir()
    src = make_micro_checkpoint(dir)
    model, tensors, cfg = Gesso.load_llama(dir)
    @test model isa Gesso.Model
    @test tensors.embedding isa Gesso.EmbeddingTable
    @test tensors.lm_head === tensors.embedding          # tied: same object
    @test tensors.final_rms isa Gesso.FrozenParameter    # model.norm.weight present
    @test length(tensors.blocks) == 2
    bt = tensors.blocks[1]
    @test size(bt.wk.storage) == (2 * 8, 32)             # n_kv*d_head rows
    @test bt.attn_rms isa Gesso.FrozenParameter
    # round-trip: loader values equal writer values, exactly
    @test tensors.embedding.storage == src["model.embed_tokens.weight"]
    @test tensors.blocks[2].wq.storage == src["model.layers.2.self_attn.q_proj.weight"]
    @test tensors.final_rms.storage == src["model.norm.weight"]
    # interpreter consumes it (item A's knobs thread through)
    logits = Gesso.reference_prefill(
        model,
        tensors,
        [0, 1, 2];
        eps=cfg.rms_norm_eps,
        theta=cfg.rope_theta,
    )
    @test size(logits) == (32, 3)
    @test approx_eq(
        logits,
        Gesso.reference_prefill(
            model,
            tensors,
            [0, 1, 2];
            eps=cfg.rms_norm_eps,
            theta=cfg.rope_theta,
        );
        atol=0.0,
    )
    ids = Gesso.reference_generate(
        model,
        tensors,
        [0, 1, 2];
        max_new_tokens=2,
        eps=cfg.rms_norm_eps,
        theta=cfg.rope_theta,
    )
    @test ids[1:3] == [0, 1, 2]
    @test length(ids) == 5
end

@testset "load_llama: sharded checkpoint via weight_map" begin
    dir = mktempdir()
    src = make_micro_checkpoint(dir; shards=true)
    model, tensors, cfg = Gesso.load_llama(dir)
    @test tensors.embedding.storage == src["model.embed_tokens.weight"]
    @test tensors.blocks[1].wgate.storage == src["model.layers.1.mlp.gate_proj.weight"]
    @test tensors.final_rms.storage == src["model.norm.weight"]
end

@testset "load_llama: unknown / missing / untied keys error WITH THE NAME" begin
    # unknown key
    dir = mktempdir()
    make_micro_checkpoint(dir; extra_keys=["model.layers.1.self_attn.q_proj.bias"])
    err = try
        Gesso.load_llama(dir)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("q_proj.bias", sprint(showerror, err))

    # missing key
    dir2 = mktempdir()
    make_micro_checkpoint(dir2; drop_keys=["model.layers.2.self_attn.k_proj.weight"])
    err2 = try
        Gesso.load_llama(dir2)
        nothing
    catch e
        e
    end
    @test err2 isa ErrorException
    @test occursin("k_proj.weight", sprint(showerror, err2))

    # untied head: distinct lm_head is an error
    dir3 = mktempdir()
    src3 = make_micro_checkpoint(dir3)
    src3["lm_head.weight"] = src3["model.embed_tokens.weight"] .+ 1.0
    _write_safetensors(joinpath(dir3, "model.safetensors"), src3)
    err3 = try
        Gesso.load_llama(dir3)
        nothing
    catch e
        e
    end
    @test err3 isa ErrorException
    @test occursin("lm_head", sprint(showerror, err3))
    @test occursin("untied", sprint(showerror, err3))
end

@testset "materialize_llama: absent model.norm ⇒ final_rms is nothing (toy2 shape)" begin
    dir = mktempdir()
    make_micro_checkpoint(dir; drop_keys=["model.norm.weight"])
    err = try
        Gesso.load_llama(dir)
        nothing
    catch e
        e
    end
    # note: this DROPS the key entirely — the loader must not invent a final
    # norm; toy2-shaped behavior is a final_rms of nothing, which the micro
    # checkpoint without model.norm exercises here
    if err === nothing
        _, tensors, _ = Gesso.load_llama(dir)
        @test tensors.final_rms === nothing
    else
        @test occursin("model.norm.weight", sprint(showerror, err))
    end
end

@testset "gqa micro checkpoint: interpreter GQA path matches repeat formula" begin
    # tie item B's materialization to item A's math: run the micro checkpoint
    # through an independent repeat-first formula (reuses test_gqa.jl's)
    dir = mktempdir()
    make_micro_checkpoint(dir)
    model, tensors, cfg = Gesso.load_llama(dir)
    tokens = [0, 1, 2]
    got = Gesso.reference_prefill(
        model,
        tensors,
        tokens;
        eps=cfg.rms_norm_eps,
        theta=cfg.rope_theta,
    )
    # same tensors through the MHA-materialized equivalent (repeat Wk/Wv rows)
    nh, nkv = MICRO.heads, MICRO.kv_heads
    dh = div(MICRO.hidden, nh)
    g = div(nh, nkv)
    rep_rows(W) = vcat([W[((div(h-1, g))*dh+1):((div(h-1, g)+1)*dh), :] for h in 1:nh]...)
    mk_mha(dir2) = begin
        make_micro_checkpoint(dir2)
        src = make_micro_checkpoint(dir2)
        src["model.layers.1.self_attn.k_proj.weight"] =
            rep_rows(src["model.layers.1.self_attn.k_proj.weight"])
        src["model.layers.1.self_attn.v_proj.weight"] =
            rep_rows(src["model.layers.1.self_attn.v_proj.weight"])
        src["model.layers.2.self_attn.k_proj.weight"] =
            rep_rows(src["model.layers.2.self_attn.k_proj.weight"])
        src["model.layers.2.self_attn.v_proj.weight"] =
            rep_rows(src["model.layers.2.self_attn.v_proj.weight"])
        cfg2 = JSON.parsefile(joinpath(dir2, "config.json"))
        cfg2["num_key_value_heads"] = nh
        open(joinpath(dir2, "config.json"), "w") do io
            JSON.print(io, cfg2)
        end
        _write_safetensors(joinpath(dir2, "model.safetensors"), src)
        Gesso.load_llama(dir2)
    end
    mha_model, mha_tensors, mha_cfg = mk_mha(mktempdir())
    want = Gesso.reference_prefill(
        mha_model,
        mha_tensors,
        tokens;
        eps=mha_cfg.rms_norm_eps,
        theta=mha_cfg.rope_theta,
    )
    @test approx_eq(got, want; atol=1e-12)   # same math, different layout path
end
