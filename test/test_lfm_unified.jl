# Unified LFM loader: lfm3 (LFM3-1.7B) feature declarations, loud failures for
# any metadata or tensor layout the feature table does not cover.

using Test
using Gesso

const _lfmu = Gesso.Inference
const _LFM3_GGUF = get(ENV, "GESSO_LFM3_GGUF", "")

# Four layers: conv, full attention with SWA, conv, full attention. Width 32 with
# four heads gives head_dim 8, so the rope dimension count is declared as 8.
function _lfm3_toy_kv(; swa=true)
    arch = "lfm3"
    d = Dict{String,Any}("general.architecture" => arch)
    for (k, v) in ("block_count" => 4, "embedding_length" => 32,
        "attention.head_count" => 4, "attention.head_count_kv" => [0, 2, 0, 2],
        "attention.layer_norm_rms_epsilon" => 1f-5, "feed_forward_length" => 48,
        "shortconv.l_cache" => 3, "vocab_size" => 40, "rope.freq_base" => 1f6,
        "rope.dimension_count" => 8)
        d["$arch.$k"] = v
    end
    if swa
        d["$arch.attention.sliding_window"] = 16
        d["$arch.attention.sliding_window_pattern"] = [false, true, false, true]
        d["$arch.rope.freq_base_swa"] = 1f4
    end
    return d
end

# Tensor names for the toy model, with the optional per-layer features switched on.
function _lfm3_toy_tensors(; gate=true, post=true, freqs=true, out_norm=true)
    names = String["token_embd.weight"]
    out_norm && push!(names, "output_norm.weight")
    freqs && push!(names, "rope_freqs.weight")
    out_norm || push!(names, "token_embd_norm.weight")
    for l in 0:3
        push!(names, "blk.$l.attn_norm.weight", "blk.$l.ffn_norm.weight",
            "blk.$l.ffn_gate.weight", "blk.$l.ffn_up.weight", "blk.$l.ffn_down.weight")
        post && push!(names, "blk.$l.post_attention_norm.weight", "blk.$l.post_ffw_norm.weight")
        if iseven(l)
            push!(names, "blk.$l.shortconv.in_proj.weight", "blk.$l.shortconv.conv.weight",
                "blk.$l.shortconv.out_proj.weight")
        else
            push!(names, "blk.$l.attn_q.weight", "blk.$l.attn_k.weight", "blk.$l.attn_v.weight",
                "blk.$l.attn_output.weight", "blk.$l.attn_q_norm.weight", "blk.$l.attn_k_norm.weight")
            gate && push!(names, "blk.$l.attn_gate.weight")
        end
    end
    return names
end

_lfm3_throws(kv; kw...) = try
    _lfmu.spec_from_gguf_kv(kv; kw...)
    false
catch err
    err isa ErrorException
end

@testset "LFM3 unified loader: feature declarations" begin
    kv = _lfm3_toy_kv()
    names = _lfm3_toy_tensors()
    spec = _lfmu.spec_from_gguf_kv(kv; tensor_names=names)
    @test spec.family === :lfm3
    @test spec.num_layers == 4
    @test spec.n_kv_heads == 2
    @test spec.head_dim == 8
    @test spec.sliding_window == 16
    @test spec.swa_layers == [false, true, false, true]
    @test spec.rope_swa !== nothing
    @test spec.rope_swa.theta == 1e4
    @test spec.rope.theta == 1e6
    @test spec.tie_word_embeddings
    @test Set([:qk_norm, :attn_head_gate, :sandwich_norm, :rope_freq_factors, :output_norm_final]) ⊆ spec.features
    @test isempty(_lfmu._missing_capabilities(Gesso.required_semantics(spec), Gesso.CPUBackend()))
    map = _lfmu._gguf_lfm_param_map(spec)
    attn_refs = _lfmu._layer_param_refs(map, spec, 1)
    @test any(r -> r.id.role === _lfmu.ROLE_ATTN_HEAD_GATE, attn_refs)
    @test any(r -> r.id.role === _lfmu.ROLE_ATTN_POST_NORM, attn_refs)
    @test any(r -> r.id.role === _lfmu.ROLE_FFN_POST_NORM, attn_refs)
    conv_refs = _lfmu._layer_param_refs(map, spec, 0)
    @test any(r -> r.id.role === _lfmu.ROLE_ATTN_POST_NORM, conv_refs)
    @test !any(r -> r.id.role === _lfmu.ROLE_ATTN_HEAD_GATE, conv_refs)
    @test map.final_norm_external == "output_norm.weight"
    @test map.rope_freqs_external == "rope_freqs.weight"
end

@testset "LFM3 unified loader: optional features are absent when undeclared" begin
    kv = _lfm3_toy_kv()
    spec = _lfmu.spec_from_gguf_kv(kv; tensor_names=_lfm3_toy_tensors(gate=false, post=false,
        freqs=false, out_norm=false))
    @test :attn_head_gate ∉ spec.features
    @test :sandwich_norm ∉ spec.features
    @test :rope_freq_factors ∉ spec.features
    @test :output_norm_final ∉ spec.features
    map = _lfmu._gguf_lfm_param_map(spec)
    @test map.final_norm_external == "token_embd_norm.weight"
    # metadata-only spec: no tensor-level feature is declared
    meta = _lfmu.spec_from_gguf_kv(kv)
    @test :sandwich_norm ∉ meta.features
    @test :attn_head_gate ∉ meta.features
end

@testset "LFM3 unified loader: loud failures" begin
    kv = _lfm3_toy_kv()
    names = _lfm3_toy_tensors()
    # no post norms on any layer is a valid plain model, not a failure
    @test !_lfm3_throws(kv; tensor_names=_lfm3_toy_tensors(post=false))
    one_post = filter(n -> !endswith(n, "post_ffw_norm.weight"), names)
    @test _lfm3_throws(kv; tensor_names=one_post)
    gate_partial = filter(n -> n != "blk.3.attn_gate.weight", names)
    @test _lfm3_throws(kv; tensor_names=gate_partial)
    bad = copy(kv)
    bad["lfm3.attention.new_mystery_key"] = 1
    err = try
        _lfmu.spec_from_gguf_kv(bad; tensor_names=names)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("lfm3.attention.new_mystery_key", err.msg)
    bad = copy(kv)
    bad["lfm3.attention.causal"] = false
    @test _lfm3_throws(bad)
    bad = copy(kv)
    delete!(bad, "lfm3.attention.sliding_window_pattern")
    @test _lfm3_throws(bad)
    bad = copy(kv)
    bad["lfm3.attention.sliding_window_pattern"] = [false, true, false]
    @test _lfm3_throws(bad)
    bad = copy(kv)
    delete!(bad, "lfm3.attention.sliding_window")
    delete!(bad, "lfm3.rope.freq_base_swa")
    @test _lfm3_throws(bad)
    bad = copy(kv)
    bad["lfm3.decision_block_count"] = 1
    @test _lfm3_throws(bad)
    @test _lfm3_throws(kv; tensor_names=[names; "dense_2_out.weight"])
end

@testset "LFM3 GGUF: header spec (set GESSO_LFM3_GGUF to the LFM3-1.7B file)" begin
    if !isfile(_LFM3_GGUF)
        @test_skip "set GESSO_LFM3_GGUF to the LFM3-1.7B GGUF"
        return
    end
    hdr = Gesso.gguf_header(_LFM3_GGUF)
    spec = _lfmu.spec_from_gguf_kv(hdr.kv; tensor_names=[t.name for t in hdr.tensors])
    @test spec.family === :lfm3
    @test spec.num_layers == 24
    @test spec.hidden_size == 2048
    @test spec.n_heads == 16
    @test spec.n_kv_heads == 4
    @test spec.head_dim == 128
    @test spec.intermediate_size == 7680
    @test spec.sliding_window == 256
    @test count(identity, spec.swa_layers) == 5
    @test spec.rope_swa.theta == 1e4
    @test spec.rope.theta == 1e6
    @test spec.mixer.layer_types[3] === :full_attention
    @test spec.mixer.layer_types[1] === :short_convolution
    @test Set([:attn_head_gate, :sandwich_norm, :rope_freq_factors, :output_norm_final]) ⊆ spec.features
    @test spec.tie_word_embeddings
end
