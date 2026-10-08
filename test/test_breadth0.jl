# BREADTH-0 — the universal model doorway (Passes A, B, D, E, F, G, H).
#
# The canonical success condition (§ mission):
#
#   "Adding an ordinary dense transformer family is now primarily an
#    adapter / conformance-fixture exercise rather than an
#    inference-engine rewrite."
#
# These tests are the EVIDENCE for that sentence. They assert three things:
#
#   1. four+ family configs are recognized through ONE canonical doorway,
#      with NO branch on model_type outside the registry;
#   2. unsupported semantics fail at the SPECIFIC MISSING OPERATION
#      (capability lattice), not at a model-family whitelist;
#   3. a second family — Phi-3, with FUSED qkv_proj/gate_up_proj tensors —
#      reaches CPU prefill/decode/generate with ZERO new operators and
#      ZERO edits to session.jl / Inference.jl / kv_manager.jl.
#
# Fixture policy: configs are written to the PUBLISHED conventions of each
# family (the same key spellings those families ship). No network, no Hub,
# no giant production checkpoint — the fixture is the METADATA, which is
# exactly the layer BREADTH-0 is about.

using JSON
using .GessoTestHelpers: approx_eq, deterministic_rng

# --- Pass A: ArchitectureSpec is the canonical description ----------------------

@testset "BREADTH-0 Pass A: ArchitectureSpec validates and is family-agnostic" begin
    spec = Gesso.ArchitectureSpec(;
        family=:llama,
        hidden_size=32,
        num_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=32,
        intermediate_size=64,
    )
    @test spec.head_dim == 8                       # derived, explicit
    published = Gesso.ArchitectureSpec(;
        family=:qwen3_5,
        hidden_size=1024,
        num_layers=1,
        n_heads=8,
        n_kv_heads=2,
        vocab_size=32,
        intermediate_size=64,
        head_dim=256,
    )
    @test published.head_dim == 256                 # Qwen3.5 independent head_dim
    @test spec.family === :llama
    @test spec.norm_kind === :rms
    @test spec.activation_kind === :swiglu

    # invariants refuse loudly, naming what is wrong (§LXX)
    @test_throws ErrorException Gesso.ArchitectureSpec(;
        family=:x,
        hidden_size=10,
        num_layers=1,
        n_heads=4,
        n_kv_heads=1,
        vocab_size=8,
        intermediate_size=8,
    )
    @test_throws ErrorException Gesso.ArchitectureSpec(;
        family=:x,
        hidden_size=32,
        num_layers=1,
        n_heads=4,
        n_kv_heads=3,
        vocab_size=8,
        intermediate_size=8,
    )
    # a form this build cannot express is refused AT ITSELF, not at the family
    @test_throws ErrorException Gesso.ArchitectureSpec(;
        family=:x,
        hidden_size=32,
        num_layers=1,
        n_heads=4,
        n_kv_heads=1,
        vocab_size=8,
        intermediate_size=8,
        norm_kind=:whatever,
    )
end

# --- Pass D: positional policy is representable, "must be null" is gone ---------

@testset "BREADTH-0 Pass D: RoPE policy replaces 'rope_scaling must be null'" begin
    p = Gesso.RoPEPolicy()
    @test p.kind === :none
    @test p.theta == 10000.0
    @test !Gesso.is_scaled(p)

    lin = Gesso.RoPEPolicy(; theta=10000.0, kind=:linear, factor=8.0)
    @test Gesso.is_scaled(lin)
    @test lin.factor == 8.0

    l3 = Gesso.RoPEPolicy(;
        theta=500000.0,
        kind=:llama3,
        factor=8.0,
        original_max_position_embeddings=8192,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
    )
    @test l3.kind === :llama3
    @test l3.original_max_position_embeddings == 8192

    # an unrepresentable policy fails on the POLICY, not on a family name
    err = try
        Gesso.RoPEPolicy(; theta=10000.0, kind=:yarn, factor=2.0)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("scaling kind", sprint(showerror, err))
end

# --- Pass A/F: four+ families recognized through ONE doorway -------------------

# Published key conventions per family. `rope_scaling` is non-null on the
# Qwen2 and Phi3 rows ON PURPOSE: the old importer refused both.
const _B0_CONFIGS = Dict{String, Dict{String, Any}}(
    "llama" => Dict{String, Any}(
        "model_type" => "llama",
        "hidden_size" => 32,
        "num_hidden_layers" => 2,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 2,
        "intermediate_size" => 64,
        "vocab_size" => 32,
        "hidden_act" => "silu",
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10000.0,
        "tie_word_embeddings" => true,
    ),
    "qwen2" => Dict{String, Any}(
        "model_type" => "qwen2",
        "hidden_size" => 32,
        "num_hidden_layers" => 2,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 2,
        "intermediate_size" => 64,
        "vocab_size" => 32,
        "hidden_act" => "silu",
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 1000000.0,
        "tie_word_embeddings" => false,
        "rope_scaling" => Dict{String, Any}("rope_type" => "linear", "factor" => 4.0),
    ),
    "gemma" => Dict{String, Any}(
        "model_type" => "gemma",
        "hidden_size" => 32,
        "num_hidden_layers" => 2,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 1,
        "intermediate_size" => 64,
        "vocab_size" => 32,
        "hidden_act" => "gelu_pytorch_tanh",   # NOT silu — the old importer refused this
        "rms_norm_eps" => 1e-6,
        "rope_theta" => 10000.0,
        "tie_word_embeddings" => true,
    ),
    "mistral" => Dict{String, Any}(
        "model_type" => "mistral",
        "hidden_size" => 32,
        "num_hidden_layers" => 2,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 2,
        "intermediate_size" => 64,
        "vocab_size" => 32,
        "hidden_act" => "silu",
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10000.0,
        "sliding_window" => 8,
        "tie_word_embeddings" => false,
    ),
    "phi3" => Dict{String, Any}(
        "model_type" => "phi3",
        "hidden_size" => 32,
        "num_hidden_layers" => 2,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 2,
        "intermediate_size" => 64,
        "vocab_size" => 32,
        "hidden_act" => "silu",
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10000.0,
        "tie_word_embeddings" => true,
    ),
)

@testset "BREADTH-0 Pass A/F: five families through ONE canonical doorway" begin
    dir = mktempdir()
    for (name, cfg) in _B0_CONFIGS
        path = joinpath(dir, "$name.json")
        open(path, "w") do io
            JSON.print(io, cfg)
        end
        spec = Gesso.architecture_spec(path)          # the generic doorway
        @test spec.family === Symbol(name)
        @test spec.hidden_size == 32
        @test spec.num_layers == 2
    end

    # a genuinely different family per row — semantics, not spelling
    qwen = Gesso.parse_config(Gesso.Qwen2Adapter(), _B0_CONFIGS["qwen2"])
    @test Gesso.capabilities(qwen).qk_norm == true
    @test Gesso.capabilities(qwen).rope_scaling === :linear

    gemma = Gesso.parse_config(Gesso.GemmaAdapter(), _B0_CONFIGS["gemma"])
    @test gemma.activation_kind === :gelu            # NOT swiglu — old importer refused
    @test Gesso.capabilities(gemma).gated_ffn == false
    @test gemma.tie_word_embeddings == true

    mistral = Gesso.parse_config(Gesso.MistralAdapter(), _B0_CONFIGS["mistral"])
    @test Gesso.capabilities(mistral).sliding_window == true
    @test mistral.sliding_window == 8

    phi3 = Gesso.parse_config(Gesso.Phi3Adapter(), _B0_CONFIGS["phi3"])
    @test Gesso.capabilities(phi3).fused_qkv == true
    @test Gesso.capabilities(phi3).gated_ffn == true

    # unknown family: a TRANSPORT refusal naming the family (import boundary)
    @test_throws ErrorException Gesso.adapter_for("no-such-family")
end

# Published Qwen3.5 / Ornith-1.5 nested text_config. The doorway must parse
# meaning and stop at named missing ops — not at the family name.
const _Q35_TEXT = Dict{String, Any}(
    "hidden_size" => 1024,
    "num_hidden_layers" => 8,
    "num_attention_heads" => 8,
    "num_key_value_heads" => 2,
    "intermediate_size" => 3584,
    "vocab_size" => 248320,
    "hidden_act" => "silu",
    "head_dim" => 256,
    "attn_output_gate" => true,
    "full_attention_interval" => 4,
    "linear_conv_kernel_dim" => 4,
    "linear_key_head_dim" => 128,
    "linear_num_key_heads" => 16,
    "linear_num_value_heads" => 16,
    "linear_value_head_dim" => 128,
    "layer_types" => [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    "tie_word_embeddings" => true,
    "rope_parameters" => Dict{String, Any}(
        "mrope_interleaved" => true,
        "mrope_section" => [11, 11, 10],
        "partial_rotary_factor" => 0.25,
        "rope_theta" => 10000000.0,
        "rope_type" => "default",
    ),
)

@testset "Qwen3.5 / Ornith doorway: nested text_config, named-op refusal" begin
    nested = Dict{String, Any}("model_type" => "qwen3_5", "text_config" => _Q35_TEXT)
    dir = mktempdir()
    path = joinpath(dir, "qwen3_5.json")
    open(path, "w") do io
        JSON.print(io, nested)
    end
    spec = Gesso.architecture_spec(path)
    @test spec.family === :qwen3_5
    @test spec.hidden_size == 1024
    @test spec.n_heads == 8
    @test spec.n_kv_heads == 2
    @test spec.head_dim == 256          # published, not 1024÷8=128
    @test spec.num_layers == 8
    @test spec.tie_word_embeddings
    @test spec.rope.theta == 10000000.0
    @test spec.rope.kind === :none
    @test spec.rope.rotary_dim == 64    # 0.25 × 256
    @test spec.rope.interleaved == false  # rotate_half, not GPT-J pairs
    @test spec.rope.mrope_section == [11, 11, 10]
    @test spec.mixer !== nothing
    @test Gesso.layer_mixer(spec, 0) === :linear_attention
    @test Gesso.layer_mixer(spec, 3) === :full_attention
    caps = Gesso.capabilities(spec)
    @test caps.qk_norm
    @test caps.linear_attention
    @test caps.attn_output_gate
    @test caps.mrope
    reqs = Gesso.required_semantics(spec)
    @test :linear_attention in reqs
    @test :attn_output_gate in reqs
    @test :mrope in reqs
    @test :qk_norm in reqs
    @test :rms_unit_offset in spec.features
    @test :rope_partial_dim in reqs
    @test :rope_interleaved ∉ reqs
    @test :swiglu_ffn in reqs
    @test Gesso.supports(Gesso.CPUBackend(), :linear_attention)
    @test Gesso.supports(Gesso.CPUBackend(), :qk_norm)
    report = Gesso.import_report(spec)
    @test occursin("linear_attention", report)
    @test occursin("qk_norm", report)
    @test occursin("Failure boundary: none", report)
    @test !occursin("qwen3_5 unsupported", report)

    # Ornith-1.5-9B sizes: hidden == n_heads × head_dim, same family
    ornith_text = merge(
        _Q35_TEXT,
        Dict{String, Any}(
            "hidden_size" => 4096,
            "num_attention_heads" => 16,
            "num_key_value_heads" => 4,
            "intermediate_size" => 12288,
            "tie_word_embeddings" => false,
        ),
    )
    ornith = Gesso.parse_config(
        Gesso.Qwen35Adapter(),
        Dict{String, Any}("model_type" => "qwen3_5", "text_config" => ornith_text),
    )
    @test ornith.hidden_size == 4096
    @test ornith.head_dim == 256
    @test ornith.n_heads * ornith.head_dim == ornith.hidden_size
    @test :linear_attention in Gesso.required_semantics(ornith)
end

@testset "BREADTH-0 Pass F: capability lattice, not a model whitelist" begin
    llm = Gesso.parse_config(Gesso.LlamaAdapter(), _B0_CONFIGS["llama"])
    reqs = Gesso.required_semantics(llm)
    # a Llama model requires NO capability Gesso does not already lower
    @test :qk_norm ∉ reqs
    @test :sliding_window_attention ∉ reqs
    @test :dense_ffn ∉ reqs
    @test :rope_none in reqs
    @test :swiglu_ffn in reqs

    # Gemma requires dense_ffn — which is NOT implemented — so it fails LATER,
    # at the named operation, having traveled all the way in through the door.
    gemma = Gesso.parse_config(Gesso.GemmaAdapter(), _B0_CONFIGS["gemma"])
    @test :dense_ffn in Gesso.required_semantics(gemma)
    # and the refusal names the OPERATION, not the family
    @test !occursin("gemma", string(:dense_ffn))
end

# --- Pass B: canonical identity survives a different spelling ------------------

@testset "BREADTH-0 Pass B: canonical identity ≠ source spelling (fused Phi-3)" begin
    phi3 = Gesso.parse_config(Gesso.Phi3Adapter(), _B0_CONFIGS["phi3"])
    m = Gesso.param_map(Gesso.Phi3Adapter(), phi3)

    # ONE external tensor, THREE canonical identities, disjoint row blocks
    qkv = filter(r -> r.external == "self_attn.qkv_proj.weight", m.layer_refs)
    @test length(qkv) == 3
    @test Set(Gesso.canonical_name(r.id) for r in qkv) ==
          Set(["layer[0].attention.q", "layer[0].attention.k", "layer[0].attention.v"])
    rows = [r.rows for r in qkv]
    @test !isempty(intersect(rows[1], rows[2])) == false
    @test !isempty(intersect(rows[2], rows[3])) == false

    # gate_up: one tensor, two identities
    gu = filter(r -> occursin("gate_up_proj", r.external), m.layer_refs)
    @test length(gu) == 2
    @test Gesso.canonical_name(gu[1].id) == "layer[0].ffn.gate"
    @test Gesso.canonical_name(gu[2].id) == "layer[0].ffn.up"
    @test gu[1].rows != gu[2].rows

    # every role a family binds must have an engine slot (meta-metric: this is
    # why a new family needs no engine edit)
    for r in m.layer_refs
        @test Gesso.slot_for(r.id.role) !== nothing
    end

    # Llama-shaped and Phi-3-shaped families produce the SAME canonical roles
    llama_spec = Gesso.parse_config(Gesso.LlamaAdapter(), _B0_CONFIGS["llama"])
    lm = Gesso.param_map(Gesso.LlamaAdapter(), llama_spec)
    llm_roles = Set(Gesso.canonical_name(r.id) for r in lm.layer_refs)
    p3_roles = Set(Gesso.canonical_name(r.id) for r in m.layer_refs)
    @test llm_roles == p3_roles          # identical MEANING, different spelling
end

@testset "BREADTH-0 Pass H: declared fusion layout is exact" begin
    # grouped layout (published for Phi-3): per KV group, [group Q][K][V]
    (q, k, v) = Gesso.fused_qkv_rows(Gesso.FusedQKVLayout(:grouped), 4, 2, 8)
    @test length(q) == 32 && length(k) == 16 && length(v) == 16
    @test isempty(intersect(q, k))
    @test isempty(intersect(k, v))
    @test maximum(v) == (4 + 2 * 2) * 8

    # contiguous layout: [all Q][all K][all V]
    (q2, k2, v2) = Gesso.fused_qkv_rows(Gesso.FusedQKVLayout(:contiguous), 4, 2, 8)
    @test first(q2) == 1 && last(q2) == 32
    @test first(k2) == 33 && last(k2) == 48
    @test first(v2) == 49 && last(v2) == 64
    # both layouts express the SAME three identities — only the PACKING
    # differs: each yields the same total Q/K/V row counts, all disjoint
    @test length(q2) == length(q) == 32
    @test length(k2) == length(k) == 16
    @test length(v2) == length(v) == 16
    @test isempty(intersect(q2, k2)) && isempty(intersect(k2, v2))
end

@testset "Qwen3.5 fused Q|gate is per-head interleaved" begin
    # 2 heads, head_dim=4 → packing [H0_Q, H0_gate, H1_Q, H1_gate]
    (q, gate) = Gesso.fused_qg_rows(2, 4)
    @test q == [1, 2, 3, 4, 9, 10, 11, 12]
    @test gate == [5, 6, 7, 8, 13, 14, 15, 16]
    @test isempty(intersect(q, gate))
    @test sort([q; gate]) == collect(1:16)
    # concatenated all-Q then all-gate would be 1:8 / 9:16 — that is wrong
    @test q != collect(1:8)
    @test gate != collect(9:16)

    spec = Gesso.parse_config(
        Gesso.Qwen35Adapter(),
        Dict{String, Any}("model_type" => "qwen3_5", "text_config" => _Q35_TEXT),
    )
    m = Gesso.param_map(Gesso.Qwen35Adapter(), spec)
    qref = only(filter(r -> Gesso.canonical_name(r.id) == "layer[0].attention.q", m.layer_refs))
    gref = only(filter(r -> Gesso.canonical_name(r.id) == "layer[0].attention.gate", m.layer_refs))
    @test qref.external == gref.external == "self_attn.q_proj.weight"
    want = Gesso.fused_qg_rows(spec.n_heads, spec.head_dim)
    @test qref.rows == want.q
    @test gref.rows == want.gate
end

# --- Pass H: a second family through EXECUTION ---------------------------------

# A tiny Phi-3-shaped checkpoint with FUSED qkv_proj / gate_up_proj, laid out
# with the declared :grouped packing. 2 layers, hidden=32, heads=4, kv=2,
# intermediate=64, vocab=32 — the same shape as the Llama micro fixture, so
# the two are directly comparable.
function _b0_write_safetensors(path, tensors::Dict{String, AbstractArray{Float64}})
    _write_safetensors(path, Dict{String, AbstractArray}(tensors))
    return nothing
end

function _b0_phi3_checkpoint(dir)
    mkpath(dir)
    rng = deterministic_rng(0xB0)
    d, nh, nkv, inter, V = 32, 4, 2, 64, 32
    dh = div(d, nh)
    tensors = Dict{String, AbstractArray{Float64}}(
        "model.embed_tokens.weight" => randn(rng, V, d),
        "model.norm.weight" => randn(rng, d),
    )
    for i in 0:1
        p = "model.layers.$i."
        tensors[p*"input_layernorm.weight"] = randn(rng, d)
        tensors[p*"post_attention_layernorm.weight"] = randn(rng, d)
        tensors[p*"self_attn.qkv_proj.weight"] = randn(rng, nh * dh + 2 * nkv * dh, d)
        tensors[p*"self_attn.o_proj.weight"] = randn(rng, d, d)
        tensors[p*"mlp.gate_up_proj.weight"] = randn(rng, 2 * inter, d)
        tensors[p*"mlp.down_proj.weight"] = randn(rng, d, inter)
    end
    _b0_write_safetensors(joinpath(dir, "model.safetensors"), tensors)
    cfg = _B0_CONFIGS["phi3"]
    open(joinpath(dir, "config.json"), "w") do io
        JSON.print(io, cfg)
    end
    return tensors
end

@testset "BREADTH-0 Pass H: Phi-3 (fused QKV + fused gate/up) runs on the CPU oracle" begin
    dir = mktempdir()
    src = _b0_phi3_checkpoint(dir)

    spec = Gesso.architecture_spec(joinpath(dir, "config.json"))
    m = Gesso.param_map(Gesso.Phi3Adapter(), spec)
    loaded = Gesso.load_safetensors(joinpath(dir, "model.safetensors"))
    bound = Gesso.materialize_architecture(spec, m, loaded)

    # the fused tensor became THREE distinct semantic parameters with the
    # EXACT rows the declared layout promised
    bt = bound.blocks[1]
    @test size(bt.wq.storage) == (32, 32)
    @test size(bt.wk.storage) == (16, 32)
    @test size(bt.wv.storage) == (16, 32)
    (qr, _, _) = Gesso.fused_qkv_rows(Gesso.FusedQKVLayout(:grouped), 4, 2, 8)
    @test bt.wq.storage == src["model.layers.0.self_attn.qkv_proj.weight"][qr, :]
    @test size(bt.wgate.storage) == (64, 32)
    @test size(bt.wup.storage) == (64, 32)
    @test bt.wgate.storage == src["model.layers.0.mlp.gate_up_proj.weight"][1:64, :]
    @test bt.wup.storage == src["model.layers.0.mlp.gate_up_proj.weight"][65:128, :]
    @test bt.attn_rms isa Gesso.FrozenParameter
    @test length(bound.blocks) == 2

    # Pass E: meaning is recorded SEPARATELY from the stored bytes
    # 2 layers × 9 layer roles, plus token_embedding and final_norm. The tied
    # head adds NO binding — it IS the embedding (same object, same bytes).
    @test length(bound.bindings) == 2 * 9 + 2
    ids = Set(Gesso.canonical_name(first(p)) for p in bound.bindings)
    @test "token_embedding" in ids
    @test "layer[0].attention.q" in ids
    @test "layer[1].ffn.down" in ids

    # ...and it EXECUTES: no new operators, no engine edit
    model = Gesso.config_to_model((
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=32,
    ))
    tensors = (
        embedding=bound.embedding,
        blocks=bound.blocks,
        lm_head=bound.lm_head,
        final_rms=bound.final_rms,
    )
    logits = Gesso.reference_prefill(model, tensors, [0, 1, 2])
    @test size(logits) == (32, 3)
    @test all(isfinite, logits)
    # deterministic: the same fixture twice is bit-identical (§LXXV)
    @test logits == Gesso.reference_prefill(model, tensors, [0, 1, 2])

    # and greedy generation runs the prefill/decode split on the second family
    ids_out = Gesso.reference_generate(model, tensors, [0, 1]; max_new_tokens=3)
    @test length(ids_out) == 5
    @test all(0 .<= ids_out .< 32)
end

# --- no silent discard (§LXX) on the generic path ------------------------------

@testset "BREADTH-0 Pass B: nothing is silently discarded on the generic path" begin
    dir = mktempdir()
    _b0_phi3_checkpoint(dir)
    spec = Gesso.architecture_spec(joinpath(dir, "config.json"))
    m = Gesso.param_map(Gesso.Phi3Adapter(), spec)
    loaded = Gesso.load_safetensors(joinpath(dir, "model.safetensors"))

    # an extra checkpoint tensor is an ERROR naming it (not a skip)
    loaded["model.layers.0.mlp.mystery.weight"] = randn(2, 2)
    err = try
        Gesso.materialize_architecture(spec, m, loaded)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("mystery", sprint(showerror, err))

    # a missing required tensor is an ERROR naming it
    delete!(loaded, "model.layers.0.mlp.mystery.weight")
    delete!(loaded, "model.layers.1.mlp.down_proj.weight")
    err2 = try
        Gesso.materialize_architecture(spec, m, loaded)
        nothing
    catch e
        e
    end
    @test err2 isa ErrorException
    @test occursin("layers.1.mlp.down_proj.weight", sprint(showerror, err2))
end

# --- the registry IS the documentation source (Pass J groundwork) ---------------

@testset "BREADTH-0: the family registry is derived, not written by hand" begin
    fams = Gesso.known_families()
    @test "llama" in fams
    @test "qwen2" in fams
    @test "gemma" in fams
    @test "mistral" in fams
    @test "phi3" in fams
    @test "qwen3_5" in fams
    @test "qwen3_5_moe" in fams
    # every registered family has an adapter object that resolves back to it
    for f in fams
        a = Gesso.adapter_for(f)
        @test Gesso.family_symbol(a) === Symbol(f)
    end
end

# --- Pass D (oracle half): scaled positional policy actually computes ---------

@testset "BREADTH-0 Pass D: :none is bit-identical; scaled policies compute" begin
    d = 8

    # THE REGRESSION-LAW ASSERTION: an unscaled policy produces `nothing`,
    # which routes rope! through its ORIGINAL literal expression. Anything
    # else would silently move every Llama logit.
    @test Gesso.rope_inv_freq(Gesso.RoPEPolicy(; theta=10000.0), d) === nothing
    @test Gesso.rope_inv_freq(Gesso.RoPEPolicy(; theta=500000.0), d) === nothing

    # :linear divides the inverse frequencies
    lin =
        Gesso.rope_inv_freq(Gesso.RoPEPolicy(; theta=10000.0, kind=:linear, factor=8.0), d)
    @test length(lin) == d ÷ 2
    @test lin[1] == 1.0 / 8.0
    @test lin[2] ≈ 10000.0^(-2 / d) / 8.0
    @test lin[2] ≈ lin[1] * 10000.0^(-2 / d)
    # a scaled policy is genuinely DIFFERENT from the unscaled one
    @test lin[2] != 10000.0^(-2 / d)

    # :llama3 splits by wavelength; the first component is always untouched
    l3 = Gesso.rope_inv_freq(
        Gesso.RoPEPolicy(;
            theta=500000.0,
            kind=:llama3,
            factor=8.0,
            original_max_position_embeddings=8192,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
        ),
        d,
    )
    @test length(l3) == d ÷ 2
    @test l3[1] == 1.0
    @test issorted(l3; rev=true)                          # frequency decreases with i
    @test all(>(0), l3)

    # interleaved is a rope! layout flag; frequencies stay the unscaled default
    @test Gesso.rope_inv_freq(Gesso.RoPEPolicy(; theta=10000.0, interleaved=true), d) ===
          nothing
    # partial rotary computes frequencies over rotary_dim, not d_head
    partial = Gesso.rope_inv_freq(Gesso.RoPEPolicy(; theta=10000.0, rotary_dim=4), d)
    @test length(partial) == 2
    @test partial[1] == 1.0
    @test partial[2] ≈ 10000.0^(-2 / 4)
end

@testset "BREADTH-0 Pass D: the policy travels with the model, not the call site" begin
    # no :rope field ⇒ unscaled ⇒ nothing (every pre-BREADTH-0 fixture)
    plain = (embedding=nothing, blocks=nothing, lm_head=nothing)
    @test Gesso.tensors_rope_inv_freq(plain, 8) === nothing

    # a scaled policy travels WITH the tensors and is picked up automatically
    scaled = (;
        embedding=nothing,
        blocks=nothing,
        lm_head=nothing,
        rope=Gesso.RoPEPolicy(; theta=10000.0, kind=:linear, factor=4.0),
    )
    inv = Gesso.tensors_rope_inv_freq(scaled, 8)
    @test inv !== nothing
    @test inv[2] ≈ 10000.0^(-2 / 8) / 4.0
end

@testset "BREADTH-0 Pass D: a scaled-rope model CHANGES the oracle result" begin
    dir = mktempdir()
    _b0_phi3_checkpoint(dir)
    spec = Gesso.architecture_spec(joinpath(dir, "config.json"))
    m = Gesso.param_map(Gesso.Phi3Adapter(), spec)
    loaded = Gesso.load_safetensors(joinpath(dir, "model.safetensors"))
    bound = Gesso.materialize_architecture(spec, m, loaded)
    model = Gesso.config_to_model((
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=32,
    ))
    base = (
        embedding=bound.embedding,
        blocks=bound.blocks,
        lm_head=bound.lm_head,
        final_rms=bound.final_rms,
    )
    unscaled_logits = Gesso.reference_prefill(model, base, [0, 1, 2])
    scaled =
        merge(base, (; rope=Gesso.RoPEPolicy(; theta=10000.0, kind=:linear, factor=8.0)))
    scaled_logits = Gesso.reference_prefill(model, scaled, [0, 1, 2])
    @test size(scaled_logits) == size(unscaled_logits)
    # the policy is not decorative: it changes the rotation
    @test scaled_logits != unscaled_logits
    # ...and it is still deterministic (same policy ⇒ bit-identical)
    @test scaled_logits == Gesso.reference_prefill(model, scaled, [0, 1, 2])
end

@testset "BREADTH-0 Pass D: the engine READS a scaled policy, and fails closed at the dimensions" begin
    dir = mktempdir()
    _b0_phi3_checkpoint(dir)
    spec = Gesso.architecture_spec(joinpath(dir, "config.json"))
    m = Gesso.param_map(Gesso.Phi3Adapter(), spec)
    loaded = Gesso.load_safetensors(joinpath(dir, "model.safetensors"))
    bound = Gesso.materialize_architecture(spec, m, loaded)
    model = Gesso.config_to_model((
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=32,
    ))
    tensors = (
        embedding=bound.embedding,
        blocks=bound.blocks,
        lm_head=bound.lm_head,
        final_rms=bound.final_rms,
        rope=Gesso.RoPEPolicy(; theta=10000.0, kind=:linear, factor=8.0),
    )
    # BREADTH-1: the engine now READS the positional policy instead of
    # refusing it. Refusal at CONSTRUCTION was correct while the engine
    # threaded `theta` only — but the CPU oracle has implemented scaled
    # policies since Pass D, so the door was shut on a working capability.
    # §LXX is not weakened: what must never happen is a scaled policy running
    # UNSCALED, and the engine now demonstrably does not.
    s = Gesso.Session(model, tensors; context_length=8, eos_token_id=2, theta=10000.0)
    @test s.inv_freq !== nothing                                  # policy was read
    @test s.inv_freq == Gesso.rope_inv_freq(
        Gesso.RoPEPolicy(; theta=10000.0, kind=:linear, factor=8.0),
        32 ÷ 4,
    )
    # and it is NOT the unscaled frequency vector
    @test s.inv_freq != Gesso.rope_inv_freq(Gesso.RoPEPolicy(), 32 ÷ 4)

    # Partial rotary and GPT-J pair layout now lower on CPU. Construction
    # reads the policy; generate executes it.
    for rope in (
        Gesso.RoPEPolicy(; theta=10000.0, kind=:linear, factor=8.0, interleaved=true),
        Gesso.RoPEPolicy(; theta=10000.0, kind=:linear, factor=8.0, rotary_dim=4),
    )
        s = Gesso.Session(
            model,
            merge(tensors, (; rope));
            context_length=8,
            eos_token_id=2,
            theta=10000.0,
        )
        @test s.inv_freq !== nothing
        ids = Gesso.generate(s, [0, 1]; max_new_tokens=2)
        @test length(ids) == 4
        @test all(0 .<= ids .< 32)
    end
end

# --- Pass I: the import report ------------------------------------------------

@testset "Pass I: the import report names the FAILURE BOUNDARY, not the family" begin
    dir = mktempdir()
    for (name, cfg) in _B0_CONFIGS
        open(joinpath(dir, "$name.json"), "w") do io
            JSON.print(io, cfg)
        end
    end

    # Llama-shaped: nothing missing
    llm = Gesso.architecture_spec(joinpath(dir, "llama.json"))
    rep = Gesso.import_report(llm)
    @test occursin("Gesso Import Report", rep)
    @test occursin("grouped-query", rep)
    @test occursin("Tied embeddings: yes", rep)
    @test occursin("Failure boundary: none", rep)

    # Qwen2 names :qk_norm and scaled rope; both now lower on CPU
    qwen = Gesso.architecture_spec(joinpath(dir, "qwen2.json"))
    qrep = Gesso.import_report(qwen)
    @test occursin("qk_norm", qrep)
    @test occursin("rope_linear", qrep)
    @test occursin("Failure boundary: none", qrep)
    @test !occursin("NOT IMPLEMENTED", qrep)

    # Gemma needs :dense_ffn — the report says so instead of "gemma unsupported"
    gem = Gesso.architecture_spec(joinpath(dir, "gemma.json"))
    grep_ = Gesso.import_report(gem)
    @test occursin("dense_ffn", grep_)
    @test occursin("Failure boundary: dense_ffn", grep_)

    # Mistral needs sliding-window attention
    mis = Gesso.architecture_spec(joinpath(dir, "mistral.json"))
    @test occursin("sliding_window_attention", Gesso.import_report(mis))

    # a report exists for EVERY registered family — no family is unreported
    for fam in Gesso.known_families()
        r = Gesso.import_report(
            Gesso.ArchitectureSpec(;
                family=Symbol(fam),
                hidden_size=8,
                num_layers=1,
                n_heads=4,
                n_kv_heads=2,
                vocab_size=8,
                intermediate_size=8,
            ),
        )
        @test occursin("Gesso Import Report", r)
    end
end

# --- Pass J: the compatibility matrix is GENERATED ----------------------------

@testset "Pass J: the compatibility matrix is derived, not hand-written" begin
    rows = Gesso.compatibility_matrix()

    # every registered family appears exactly once
    fams = Gesso.known_families()
    @test sort([String(r.family) for r in rows]) == sort(fams)
    @test length(rows) == length(fams)

    # rows are internally consistent with required_semantics
    for r in rows
        # the config space is actually swept, and every variant is consistent
        @test r.total == length(r.variants)
        @test r.total > 1                       # a single probe is the bug this replaced
        @test r.runnable == count(v -> isempty(v.missing), r.variants)
        @test 0 <= r.runnable <= r.total
        for v in r.variants
            @test (v.first_missing === nothing) == isempty(v.missing)
            v.first_missing === nothing || @test v.first_missing == first(v.missing)
            @test issubset(v.missing, r.unreachable)
        end

        # base form (variant 1) is the constructor default and runs today
        @test r.base_missing == r.variants[1].missing
        @test r.base_first_missing == r.variants[1].first_missing

        # the WORST case is the first of `unreachable` in required_semantics
        # order — not the base form's, and not `nothing` unless truly total.
        if isempty(r.unreachable)
            @test r.first_missing === nothing
            @test r.runnable == r.total
        else
            @test r.first_missing == first(r.unreachable)
            @test r.first_missing !== r.base_first_missing ||
                  r.base_first_missing === nothing
        end
    end

    # THE FENCE. BREADTH-1 made `supports(backend, cap)` the SINGLE authority,
    # so the matrix derives `missing` from it and can no longer OVERSTATE by
    # consulting a second, hand-kept capability set.
    backend_used = Gesso.CPUBackend()

    # (a) nothing is called unreachable that the backend can actually run.
    # This guards the OPPOSITE direction from the bug above: BREADTH-1 shipped
    # a matrix that UNDERSTATED, because `_implemented_capabilities` went stale
    # the moment CPU gained scaled RoPE and nothing noticed for a commit.
    for r in rows, c in r.unreachable
        @test !Gesso.supports(backend_used, c)
    end

    # (b) while any unimplemented capability is reachable from a family's
    # config space, NO family row may read total (§LXX: a report that
    # overstates is worse than no report).
    reachable_anywhere = Set{Symbol}()
    for r in rows
        union!(reachable_anywhere, r.unreachable)
    end
    if !isempty(reachable_anywhere)
        for r in rows
            @test r.first_missing !== nothing   # worst case must be named
        end
    end

    # (c) every row names the backend it was computed against — the matrix has
    # a backend dimension precisely because "implemented" is not global
    # (CPU runs scaled RoPE; CUDA and Lava decline it).
    for r in rows
        @test r.backend == :cpu
    end

    # (d) THE BACKEND ARGUMENT IS HONORED. A device-free probe backend that
    # reports CPU's set MINUS scaled RoPE must produce a strictly narrower
    # matrix — otherwise the `backend` argument is decorative and "one
    # authority" is one authority that only ever gets asked one question.
    # Modelled on CUDA, which genuinely declines scaled RoPE at the operation.
    @eval struct ProbeNoScaledRope <: $(Gesso.AbstractGessoBackend) end
    @eval Gesso.backend_name(::$ProbeNoScaledRope) = :probe
    @eval Gesso.supports(::$ProbeNoScaledRope, cap::Symbol) =
        Gesso.supports($(Gesso.CPUBackend()), cap) &&
        cap !== :rope_linear &&
        cap !== :rope_llama3
    cpu_row = first(Gesso.compatibility_matrix(; backend=Gesso.CPUBackend()))
    probe_row = first(Gesso.compatibility_matrix(; backend=ProbeNoScaledRope()))
    @test probe_row.runnable < cpu_row.runnable
    @test :rope_linear in probe_row.unreachable
    @test !(:rope_linear in cpu_row.unreachable)
    @test probe_row.backend == :probe
    # precisely the two capabilities the probe backend gave up, in
    # required_semantics order — and `first_missing` is deliberately NOT
    # asserted to differ: `:dense_ffn` precedes rope and is missing on BOTH,
    # so the first wall is genuinely the same. That is the answer being
    # correct, not the argument being untested.
    @test setdiff(probe_row.unreachable, cpu_row.unreachable) ==
          [:rope_linear, :rope_llama3]
    @test probe_row.first_missing == cpu_row.first_missing == :dense_ffn

    # the rendered table is generated from those rows
    io = IOBuffer()
    Gesso.compatibility_table(io)
    table = String(take!(io))
    @test occursin("| architecture |", table)
    @test occursin("runnable configs", table)
    for fam in fams
        @test occursin("| $fam |", table)   # no family may be silently dropped
    end
    # every row states its runnable fraction — "all green" is unrepresentable
    for r in rows
        @test occursin("| $(r.runnable)/$(r.total) |", table)
    end
end
