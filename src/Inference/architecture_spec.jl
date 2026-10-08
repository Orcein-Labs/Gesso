# ArchitectureSpec — the canonical architecture-description boundary
# (BREADTH-0 Pass A; Gesso_Stack.md §VIII).
#
# BEFORE: `load_llama_config` answered "is this Llama-shaped?" by string
# equality on `model_type` plus a closed key list, and every refusal named the
# FAMILY. The engine knew about Llama because the importer told it about Llama.
#
# AFTER: `config.json` → ArchitectureAdapter → `ArchitectureSpec`. The spec
# carries MEANING (dimensions, norm form, activation form, positional
# policy, tie policy) and the engine consumes the meaning, never the raw HF
# `model_type` string. §VIII: "Checkpoint format is transport. Architecture
# semantics are execution information."
#
# Laws:
#   * GENERIC fields represent genuine common semantics. Where semantics
#     genuinely differ (positional policy, tie policy, attention variant) the
#     difference is an EXPLICIT field, never a forced false.
#   * Construction validates invariants (divisibility, positivity) so a spec
#     that exists is a spec that composed successfully.
#   * Nothing here lowers anything and nothing here executes (§XV: this is the
#     import boundary, not the engine).

"""
    RoPEPolicy

Positional-encoding SEMANTICS (§VIII positional policy; BREADTH-0 Pass D).

Replaces "rope_scaling must be null" as an architectural assumption. The
fields are the dimensions an actual encountered variant needs — not a
universal RoPE factory. Only three policies are expressible, and all three
are real, observed conventions:

    :none      inv_freq[i] = theta^(-2i/d)               (Llama / Qwen / Gemma default)
    :linear    the same inverse frequencies divided by factor      (linear RoPE scaling)
    :llama3    wavelength-split scaling with low/high frequency factors and an
               original context length                     (Llama-3 style)

`rotary_dim == 0` means "rotate the whole head dimension"; a partial rotary
dimension sets the number of rotated features explicitly. `interleaved` is
the layout policy (GPT-NeoX half-split vs GPT-J interleaved pairs).

The DEFAULT constructor reproduces the pre-BREADTH-0 behavior exactly:
`RoPEPolicy(; theta=10000.0)` is the unscaled policy that `rope!(…; theta=10000.0)`
computed, so the Llama path is bit-identical after this lands (regression law
§XIII).
"""
struct RoPEPolicy
    theta::Float64
    kind::Symbol
    factor::Float64
    original_max_position_embeddings::Int
    low_freq_factor::Float64
    high_freq_factor::Float64
    rotary_dim::Int
    interleaved::Bool

    function RoPEPolicy(;
        theta::Real=10000.0,
        kind::Symbol=:none,
        factor::Real=1.0,
        original_max_position_embeddings::Integer=0,
        low_freq_factor::Real=1.0,
        high_freq_factor::Real=4.0,
        rotary_dim::Integer=0,
        interleaved::Bool=false,
    )
        kind in (:none, :linear, :llama3) || error(
            "RoPEPolicy: unsupported scaling kind $(repr(kind)) — this build expresses :none, :linear, :llama3 (§LXX: no silent substitution)",
        )
        isfinite(theta) && theta > 0 ||
            error("RoPEPolicy: theta must be finite and positive")
        isfinite(factor) && factor > 0 ||
            error("RoPEPolicy: factor must be positive (got $factor)")
        isfinite(low_freq_factor) &&
        isfinite(high_freq_factor) &&
        low_freq_factor > 0 &&
        high_freq_factor > 0 || error("RoPEPolicy: frequency factors must be positive")
        rotary_dim >= 0 || error("RoPEPolicy: rotary_dim must be >= 0 (0 = full head dim)")
        if kind === :llama3
            high_freq_factor > low_freq_factor ||
                error("RoPEPolicy: high_freq_factor must exceed low_freq_factor")
            original_max_position_embeddings > 0 || error(
                "RoPEPolicy: :llama3 scaling requires original_max_position_embeddings > 0",
            )
        end
        return new(
            Float64(theta),
            kind,
            Float64(factor),
            Int(original_max_position_embeddings),
            Float64(low_freq_factor),
            Float64(high_freq_factor),
            Int(rotary_dim),
            interleaved,
        )
    end
end

is_scaled(p::RoPEPolicy) = p.kind !== :none

"""
    ArchitectureSpec

Canonical description of WHAT a model architecture IS (BREADTH-0 Pass A).

One canonical boundary replaces per-family config knowledge. Fields:

    family                the external family name, as a provenance label only
    hidden_size           model dimension
    num_layers            layer count
    n_heads / n_kv_heads  attention head count / KV head count
    head_dim              derived: hidden_size ÷ n_heads (explicit, validated)
    vocab_size            vocabulary size
    intermediate_size     FFN dimension
    norm_kind             :rms | :layer
    activation_kind       :swiglu | :gelu | :relu  (the FFN gating/activation form)
    rope                  positional-encoding SEMANTICS (a RoPEPolicy)
    tie_word_embeddings   the embedding IS the output head
    attention_bias        the attention projections carry bias
    mlp_bias              the FFN projections carry bias
    sliding_window        nothing, or the attention window size
    features              extra semantic flags as a Set (:qk_norm, :moe, …)

`family` is provenance — NOTHING in the engine may branch on it. Engine code
reads dimensions, forms and capabilities (§XXXIII Pass F). The only places
`family` is legal are an import report and a receipt.
"""
struct ArchitectureSpec
    family::Symbol
    hidden_size::Int
    num_layers::Int
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
    vocab_size::Int
    intermediate_size::Int
    norm_kind::Symbol
    activation_kind::Symbol
    rope::RoPEPolicy
    tie_word_embeddings::Bool
    attention_bias::Bool
    mlp_bias::Bool
    sliding_window::Union{Nothing, Int}
    features::Set{Symbol}
end

function ArchitectureSpec(;
    family::Symbol,
    hidden_size::Integer,
    num_layers::Integer,
    n_heads::Integer,
    n_kv_heads::Integer,
    vocab_size::Integer,
    intermediate_size::Integer,
    norm_kind::Symbol=:rms,
    activation_kind::Symbol=:swiglu,
    rope::RoPEPolicy=RoPEPolicy(),
    tie_word_embeddings::Bool=true,
    attention_bias::Bool=false,
    mlp_bias::Bool=false,
    sliding_window::Union{Nothing, Integer}=nothing,
    features=Set{Symbol}(),
)
    hidden_size > 0 || error("ArchitectureSpec: hidden_size must be positive")
    num_layers > 0 || error("ArchitectureSpec: num_layers must be positive")
    n_heads > 0 || error("ArchitectureSpec: n_heads must be positive")
    0 < n_kv_heads <= n_heads ||
        error("ArchitectureSpec: n_kv_heads must be in 1..n_heads (got $n_kv_heads)")
    vocab_size > 0 || error("ArchitectureSpec: vocab_size must be positive")
    intermediate_size > 0 || error("ArchitectureSpec: intermediate_size must be positive")
    norm_kind in (:rms, :layer) || error(
        "ArchitectureSpec: unsupported norm_kind $(repr(norm_kind)) — this build expresses :rms, :layer",
    )
    activation_kind in (:swiglu, :gelu, :relu) || error(
        "ArchitectureSpec: unsupported activation_kind $(repr(activation_kind)) — this build expresses :swiglu, :gelu, :relu",
    )
    hidden_size % n_heads == 0 || error(
        "ArchitectureSpec: hidden_size $hidden_size is not divisible by n_heads $n_heads",
    )
    n_heads % n_kv_heads == 0 || error(
        "ArchitectureSpec: n_heads $n_heads is not divisible by n_kv_heads $n_kv_heads",
    )
    sliding_window === nothing ||
        sliding_window > 0 ||
        error("ArchitectureSpec: sliding_window must be positive")
    head_dim = div(hidden_size, n_heads)
    return ArchitectureSpec(
        family,
        Int(hidden_size),
        Int(num_layers),
        Int(n_heads),
        Int(n_kv_heads),
        head_dim,
        Int(vocab_size),
        Int(intermediate_size),
        norm_kind,
        activation_kind,
        rope,
        tie_word_embeddings,
        attention_bias,
        mlp_bias,
        sliding_window === nothing ? nothing : Int(sliding_window),
        Set{Symbol}(features),
    )
end

Base.getproperty(s::ArchitectureSpec, name::Symbol) =
    name === :gqa ? getfield(s, :n_kv_heads) < getfield(s, :n_heads) :
    name === :has_qk_norm ? :qk_norm in getfield(s, :features) :
    name === :has_moe ? :moe in getfield(s, :features) : getfield(s, name)

"""
    ArchitectureCapabilities

WHAT SEMANTICS an architecture requires (BREADTH-0 Pass F).

This is the replacement for `if model_type == "llama"`. Execution asks
"can I lower these semantics?" instead of "is this Llama?". Capabilities are
derived from the spec's MEANING; there is exactly one derivation, so
capability detection can never be scattered type checks through the engine
(§VIII Pass F: no accidental detection).

    dense_attention / grouped_query_attention
    qk_norm               per-head QK normalization (Qwen-style)
    gated_ffn             SwiGLU-style gate/up/down
    fused_qkv             the checkpoint stores Q,K,V in ONE tensor
    moe                   mixture-of-experts routing
    tied_embeddings
    sliding_window
    rope_scaling          :none | :linear | :llama3
"""
struct ArchitectureCapabilities
    dense_attention::Bool
    grouped_query_attention::Bool
    qk_norm::Bool
    gated_ffn::Bool
    fused_qkv::Bool
    moe::Bool
    tied_embeddings::Bool
    sliding_window::Bool
    rope_scaling::Symbol
end

"""
    capabilities(spec) -> ArchitectureCapabilities

The single place capability detection happens. Everything else consumes this.
"""
function capabilities(spec::ArchitectureSpec)
    return ArchitectureCapabilities(
        spec.n_heads > 0,
        spec.n_kv_heads < spec.n_heads,
        :qk_norm in spec.features,
        spec.activation_kind === :swiglu,
        :fused_qkv in spec.features,
        :moe in spec.features,
        spec.tie_word_embeddings,
        spec.sliding_window !== nothing,
        spec.rope.kind,
    )
end

"""
    required_semantics(spec) -> Vector{Symbol}

The semantic OPERATIONS an architecture needs lowered, in a stable order.

This is the list execution checks against `supports(backend, cap)` (§XX). An
architecture whose semantics are all here lowers; one that needs something
else fails AT THAT OPERATION (Pass F) rather than at a family whitelist.
"""
function required_semantics(spec::ArchitectureSpec)
    caps = capabilities(spec)
    req = Symbol[:rmsnorm, :attention, :matmul]
    push!(req, caps.gated_ffn ? :swiglu_ffn : :dense_ffn)
    caps.qk_norm && push!(req, :qk_norm)
    caps.sliding_window && push!(req, :sliding_window_attention)
    caps.moe && push!(req, :moe_routing)
    push!(req, Symbol("rope_", caps.rope_scaling))
    return req
end

export RoPEPolicy,
    ArchitectureSpec, ArchitectureCapabilities, capabilities, required_semantics, is_scaled

# --- Pass D: the CPU-oracle reference for a positional policy ------------------

"""
    rope_inv_freq(policy::RoPEPolicy, d_head::Integer) -> Union{Nothing, Vector{Float64}}

The inverse frequencies a positional policy implies, or `nothing` when the
policy is the unscaled one.

`nothing` is a DELIBERATE signal, not a shortcut: `rope!` with `inv_freq ===
nothing` evaluates the literal `m * theta^(-2i/d)` expression the CPU oracle
has always evaluated, so every `:none` model — Llama included — stays
bit-identical (regression law §XIII). Only a genuinely scaled policy produces
a vector.

    :linear   inv_freq = base_inv_freq / factor
    :llama3   wavelength split against `original_max_position_embeddings`,
              with `low_freq_factor` / `high_freq_factor` easing the boundary
              bands (the published Llama-3 convention)

Unsupported policy DIMENSIONS fail closed at the dimension: a partial
rotary dimension or an interleaved layout is a real convention this build
does not implement, and saying so is the whole point of the capability
lattice (§II).
"""
function rope_inv_freq(policy::RoPEPolicy, d_head::Integer; theta=policy.theta)
    theta isa Real && isfinite(theta) && theta>0 || throw(
        gesso_error(ERR_INVALID_PLAN, "rope_inv_freq: theta must be finite and positive"),
    )
    policy.interleaved &&
        throw(LoweringNotImplemented(:rope_interleaved, Symbol("rope_", policy.kind)))
    policy.rotary_dim != 0 &&
        throw(LoweringNotImplemented(:rope_partial_dim, Symbol("rope_", policy.kind)))
    policy.kind === :none && return nothing
    d_head % 2 == 0 ||
        error("rope_inv_freq: d_head $d_head must be even for rotary embedding")
    n = d_head ÷ 2
    base = [theta^(-2i / d_head) for i in 0:(n-1)]

    policy.kind === :linear && return base ./ policy.factor

    # :llama3 — wavelength split. High-frequency components keep their
    # frequency; low-frequency components are divided by `factor`; the band
    # between is interpolated smoothly so the rotation stays continuous.
    old_ctx = Float64(policy.original_max_position_embeddings)
    low_wavelen = old_ctx / policy.low_freq_factor
    high_wavelen = old_ctx / policy.high_freq_factor
    out = similar(base)
    for i in 0:(n-1)
        wavelen = 2π / base[i+1]
        if wavelen < high_wavelen
            out[i+1] = base[i+1]
        elseif wavelen > low_wavelen
            out[i+1] = base[i+1] / policy.factor
        else
            smooth =
                (old_ctx / wavelen - policy.low_freq_factor) /
                (policy.high_freq_factor - policy.low_freq_factor)
            out[i+1] = (1 - smooth) * base[i+1] / policy.factor + smooth * base[i+1]
        end
    end
    return out
end

"""
    tensors_rope_inv_freq(tensors, d_head) -> Union{Nothing, Vector{Float64}}

Read the positional policy that TRAVELS WITH the model, if any, and turn it
into oracle frequencies. A tensor set without a `:rope` field is an unscaled
model and yields `nothing` — the bit-identical default path.

This is the Pass E seam in the positional dimension: the policy is a property
of the IMPORTED MODEL (meaning), not of the operator call site.
"""
function tensors_rope_inv_freq(tensors, d_head::Integer; theta=nothing)
    policy = (tensors isa NamedTuple && haskey(tensors, :rope)) ? tensors.rope : nothing
    policy === nothing && return nothing
    return rope_inv_freq(policy, d_head; theta=theta===nothing ? policy.theta : theta)
end

# Imported HF policies describe half-split layout. Legacy fixtures without
# policy metadata keep the original adjacent-pair operator convention.
_tensors_rope_interleaved(tensors) =
    haskey(tensors, :rope) && tensors.rope !== nothing ? tensors.rope.interleaved : true

export rope_inv_freq, tensors_rope_inv_freq

# A supplied override is authoritative; otherwise use the imported policy.
function _resolved_rope_theta(tensors, theta)
    if theta===nothing
        policy=haskey(tensors, :rope) ? tensors.rope : nothing
        theta=policy===nothing ? 10000.0 : policy.theta
    end
    theta isa Real && isfinite(theta) && theta>0 ||
        throw(gesso_error(ERR_INVALID_PLAN, "RoPE theta must be finite and positive"))
    return Float64(theta)
end
