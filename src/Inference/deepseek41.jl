# DeepSeek-V4.1-Flash text CPU forward, after llama.cpp `src/models/deepseek41.cpp`
# and the reference `inference_model.py`. This file holds the metadata reader
# and the spec; the forward lands with `deepseek41_forward!`.
#
# Every layer reads and writes `hc` parallel residual streams of width `n_embd`
# (hyper-connections). Attention is multi-query with a learned sink per head:
# a sliding window of `attention.sliding_window` tokens, plus on compressed
# layers a pooled KV (ratio 1 or 2) and a top-k block selection from an indexer.
# Every layer routes experts by a sqrtsoftplus score with a selection bias
# (`exp_probs_b`); there is no token-id routing table. Layers listed in
# `ple.layers` add an Engram n-gram embedding.
#
# The oracle follows llama.cpp's F32 path: no activation quantization is
# simulated, unlike the fp8/fp4 steps in the reference `model.py`.

struct Deepseek41Config
    n_embd::Int
    n_layer::Int
    n_head::Int
    n_head_kv::Int
    head_dim::Int             # attention.key_length; the single KV head is this wide
    n_rot::Int                # rope.dimension_count, the RoPE tail of each head
    rope_base::Float64        # layers without compression
    compress_rope_base::Float64
    rope_factor::Float64      # YaRN factor on compressed layers
    rope_orig_ctx::Int
    eps::Float64
    q_lora::Int
    window::Int               # attention.sliding_window
    o_groups::Int
    o_lora::Int
    compress::Vector{Int}     # per layer (1-based index il+1): 0 SWA only, 1 or 2 compressed
    kv_src::Vector{Int}       # per layer: 0-based latest KV source at or before it, -1 if none
    idx_src::Vector{Int}      # per layer: 0-based latest indexer source, -1 if none
    idx_n_head::Int
    idx_head_dim::Int
    idx_top_k::Int
    cand_src::Int             # attention.indexer.candidate_source_layer, -1 if none
    cand_top_k_blocks::Int
    cand_block_size::Int
    hc::Int
    sinkhorn_iters::Int
    hc_eps::Float64
    n_exp::Int
    n_used::Int
    exp_ff::Int
    n_shared::Int
    route_scale::Float64
    route_norm::Bool
    clamp_exp::Vector{Float64}
    clamp_shexp::Vector{Float64}
    ple_layers::Vector{Int}   # 0-based Engram layers
    ple_ngram::Int
    ple_hpn::Int
    ple_head_dim::Int
    ple_mult::Vector{UInt64}  # per Engram module, ngram values
    ple_off::Vector{Int}      # per Engram module and head
    ple_voc::Vector{Int}      # per Engram module and head (bucket count)
    engram_pad::Int           # engram.pad_token_id, a compressed id
    token_map::Vector{Int}    # engram.token_map: token id -> compressed id
    vocab::Int
end

# SQRT_SOFTPLUS in llama.cpp's expert_gating_func enum.
const _DSV41_GATING_SQRT_SOFTPLUS = 4

"""
    deepseek41_config(kv) -> Deepseek41Config

Read the DeepSeek-V4.1 architecture from GGUF metadata. Every value comes from
the file; a missing key or inconsistent layout is an error. The checks mirror
llama.cpp's loader: compress ratios 0-2, each compressed layer matched to a KV
source and an indexer source of the same ratio, candidate selection pointing at
an indexer source, and Engram metadata of consistent size.
"""
function deepseek41_config(kv::AbstractDict)
    arch = String(get(kv, "general.architecture", ""))
    arch == "deepseek41" || error("deepseek41_config: general.architecture is $(repr(arch))")
    req(key) = (k = "deepseek41.$key"; haskey(kv, k) ? kv[k] : error("deepseek41_config: missing $k"))
    opt(key, default) = get(kv, "deepseek41.$key", default)
    ints(v) = Int[Int(x) for x in v]
    floats(v, n) = v isa AbstractVector ? Float64[Float64(x) for x in v] : fill(Float64(v), n)

    n_layer = Int(req("block_count"))
    n_layer > 0 || error("deepseek41_config: block_count must be > 0")
    n_embd = Int(req("embedding_length"))
    n_head = Int(req("attention.head_count"))
    head_dim = Int(req("attention.key_length"))
    n_rot = Int(req("rope.dimension_count"))
    n_rot <= head_dim && n_rot % 2 == 0 ||
        error("deepseek41_config: rope.dimension_count $n_rot must be even and at most key_length $head_dim")
    n_head_kv = Int(req("attention.head_count_kv"))
    n_head_kv == 1 || error("deepseek41_config: attention.head_count_kv must be 1 (multi-query), got $n_head_kv")

    compress = ints(req("attention.compress_ratios"))
    length(compress) >= n_layer || error("deepseek41_config: compress_ratios is shorter than block_count")
    compress = compress[1:n_layer]
    any(r -> r < 0 || r > 2, compress) &&
        error("deepseek41_config: compress ratios must be 0, 1 or 2, got $(compress)")

    kv_src_layers = ints(req("attention.kv_source_layers"))
    idx_src_layers = ints(req("attention.indexer.source_layers"))
    any(il -> il < 0 || il >= n_layer, kv_src_layers) &&
        error("deepseek41_config: kv source layer out of range")
    any(il -> il < 0 || il >= n_layer, idx_src_layers) &&
        error("deepseek41_config: indexer source layer out of range")
    kv_src = fill(-1, n_layer)
    idx_src = fill(-1, n_layer)
    last_kv = -1
    last_idx = -1
    for il in 0:(n_layer - 1)
        ratio = compress[il + 1]
        is_kv = il in kv_src_layers
        is_idx = il in idx_src_layers
        (is_kv || is_idx) && ratio == 0 &&
            error("deepseek41_config: source layer $il has compress ratio 0")
        last_kv = is_kv ? il : last_kv
        last_idx = is_idx ? il : last_idx
        ratio == 0 && continue
        (last_kv < 0 || last_idx < last_kv || compress[last_kv + 1] != ratio ||
            compress[last_idx + 1] != ratio) &&
            error("deepseek41_config: layer $il has compress ratio $ratio without a matching kv/indexer source")
        kv_src[il + 1] = last_kv
        idx_src[il + 1] = last_idx
    end

    idx_top_k = Int(req("attention.indexer.top_k"))
    idx_top_k > 0 || error("deepseek41_config: attention.indexer.top_k must be > 0")

    cand_src = Int(opt("attention.indexer.candidate_source_layer", -1))
    cand_top_k_blocks = 0
    cand_block_size = 0
    if cand_src >= 0
        cand_top_k_blocks = Int(req("attention.indexer.candidate_top_k_blocks"))
        cand_block_size = Int(req("attention.indexer.candidate_block_size"))
        (cand_src >= n_layer || idx_src[cand_src + 1] != cand_src || cand_block_size <= 0 ||
            cand_top_k_blocks <= 0) &&
            error("deepseek41_config: invalid candidate selection (the source must be an indexer source, " *
                  "block size and top-k blocks > 0)")
        for il in (cand_src + 1):(n_layer - 1)
            idx_src[il + 1] == il && compress[il + 1] != compress[cand_src + 1] &&
                error("deepseek41_config: candidate consumers must share the source's compress ratio")
        end
    end

    hc = Int(req("hyper_connection.count"))
    hc > 0 || error("deepseek41_config: hyper_connection.count must be > 0")
    sinkhorn_iters = Int(req("hyper_connection.sinkhorn_iterations"))
    hc_eps = Float64(req("hyper_connection.epsilon"))

    gating = Int(req("expert_gating_func"))
    gating == _DSV41_GATING_SQRT_SOFTPLUS ||
        error("deepseek41_config: expert_gating_func must be sqrtsoftplus ($_DSV41_GATING_SQRT_SOFTPLUS), got $gating")
    n_exp = Int(req("expert_count"))
    n_used = Int(req("expert_used_count"))
    n_used <= n_exp || error("deepseek41_config: expert_used_count $n_used exceeds expert_count $n_exp")
    exp_ff = Int(req("expert_feed_forward_length"))
    n_shared = Int(req("expert_shared_count"))
    route_scale = Float64(req("expert_weights_scale"))
    route_norm = Bool(req("expert_weights_norm"))
    clamp_exp = floats(req("swiglu_clamp_exp"), n_layer)
    clamp_shexp = haskey(kv, "deepseek41.swiglu_clamp_shexp") ?
        floats(kv["deepseek41.swiglu_clamp_shexp"], n_layer) : copy(clamp_exp)

    ple_layers = ints(opt("ple.layers", Int[]))
    ple_ngram = 0
    ple_hpn = 0
    ple_head_dim = 0
    ple_mult = UInt64[]
    ple_off = Int[]
    ple_voc = Int[]
    engram_pad = 0
    token_map = Int[]
    if !isempty(ple_layers)
        ple_layers == sort(unique(ple_layers)) && all(il -> 0 <= il < n_layer, ple_layers) ||
            error("deepseek41_config: ple.layers must be ascending layers in range")
        ple_ngram = Int(req("ple.ngram_size"))
        ple_hpn = Int(req("ple.heads_per_ngram"))
        ple_head_dim = Int(req("embedding_length_per_layer"))
        ple_mult = UInt64[UInt64(x) for x in req("ple.layer_multipliers")]
        ple_off = ints(req("ple.head_offsets"))
        ple_voc = ints(req("ple.head_vocab_sizes"))
        engram_pad = Int(req("engram.pad_token_id"))
        token_map = ints(req("engram.token_map"))
        n_e = length(ple_layers)
        n_cols = ple_ngram >= 2 ? (ple_ngram - 1) * ple_hpn : 0
        (n_cols == 0 || ple_head_dim == 0 || length(ple_mult) != n_e * ple_ngram ||
            length(ple_voc) != n_e * n_cols || length(ple_off) != n_e * n_cols ||
            engram_pad < 0 || engram_pad >= length(token_map)) &&
            error("deepseek41_config: engram metadata is inconsistent")
        for c in eachindex(ple_voc)
            (ple_voc[c] < 2 || ple_off[c] < 0 || ple_off[c] + ple_voc[c] > typemax(Int32)) &&
                error("deepseek41_config: engram head $c has a bucket range out of range")
        end
    end

    toks = get(kv, "tokenizer.ggml.tokens", nothing)
    vocab = toks isa Integer ? Int(toks) : toks === nothing ? 0 : length(toks)

    return Deepseek41Config(
        n_embd, n_layer, n_head, n_head_kv, head_dim, n_rot,
        Float64(req("rope.freq_base")),
        Float64(req("attention.compress_rope_freq_base")),
        Float64(req("rope.scaling.factor")),
        Int(req("rope.scaling.original_context_length")),
        Float64(req("attention.layer_norm_rms_epsilon")),
        Int(req("attention.q_lora_rank")),
        Int(req("attention.sliding_window")),
        Int(req("attention.output_group_count")),
        Int(req("attention.output_lora_rank")),
        compress, kv_src, idx_src,
        Int(req("attention.indexer.head_count")),
        Int(req("attention.indexer.key_length")),
        idx_top_k, cand_src, cand_top_k_blocks, cand_block_size,
        hc, sinkhorn_iters, hc_eps,
        n_exp, n_used, exp_ff, n_shared, route_scale, route_norm,
        clamp_exp, clamp_shexp,
        ple_layers, ple_ngram, ple_hpn, ple_head_dim, ple_mult, ple_off, ple_voc,
        engram_pad, token_map, vocab,
    )
end

"""
`deepseek41` (DeepSeek-V4.1-Flash) GGUF spec. It records the shape and the
features so the import report can describe the file. `moe` is left `nothing`:
`MoeSpec` only knows softmax and sigmoid gating, and the sqrtsoftplus router is
recorded as the `:sqrtsoftplus_moe` feature; the forward is
`deepseek41_forward!`, not the generic block runner. The DSpark / MTP draft
head is not part of the text model and is not read.
"""
function _spec_from_gguf_deepseek41(kv::AbstractDict, tensor_names)
    cfg = deepseek41_config(kv)
    cfg.vocab > 0 || throw(_missing_metadata("tokenizer.ggml.tokens"))
    features = Set{Symbol}([:hyper_connections, :sparse_attention, :compressed_kv,
        :sinks, :grouped_output, :sqrtsoftplus_moe])
    isempty(cfg.ple_layers) || push!(features, :ple)
    return ArchitectureSpec(;
        family=:deepseek41,
        hidden_size=cfg.n_embd,
        num_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        n_kv_heads=cfg.n_head_kv,
        vocab_size=cfg.vocab,
        intermediate_size=cfg.exp_ff,
        head_dim=cfg.head_dim,
        rope=RoPEPolicy(;
            theta=cfg.rope_base, rotary_dim=cfg.n_rot, interleaved=true),
        tie_word_embeddings=tensor_names !== nothing && !("output.weight" in tensor_names),
        features,
        rms_norm_eps=cfg.eps,
        moe=nothing,
    )
end
