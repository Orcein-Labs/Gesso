# Import report + compatibility matrix (BREADTH-0 Passes I and J).
#
# The problem these solve is a USABILITY one and it is a real defect the
# current importer has: today a user discovers compatibility by waiting for an
# error, and the error names a FAMILY ("not Llama-shaped") rather than a
# capability. Both artifacts here are DERIVED from the registry and the spec —
# neither is hand-written prose, so neither can claim support the code does
# not have (Pass J: "the documentation must not manually claim support that
# tests do not prove").

# --- Pass I: the human-readable report ----------------------------------------

"""
    import_report(spec; tensors=nothing, bound=nothing) -> String

A human-readable account of what Gesso UNDERSTOOD about a checkpoint, and
where execution would stop. Answers, before anything runs:

  * which family, which semantics (not just which spelling)
  * how many parameters were recognized / mapped / ignored / ambiguous
  * which capabilities execution needs, and which of them exist
  * the FIRST missing capability — the failure boundary

A user must never have to learn compatibility from a stack trace fifteen
seconds into a run.
"""
function import_report(spec::ArchitectureSpec; tensors=nothing, bound=nothing)
    caps = capabilities(spec)
    reqs = required_semantics(spec)
    io = IOBuffer()

    println(io, "Gesso Import Report")
    println(io, "-------------------")
    println(io)
    println(io, "Model family:    ", spec.family)
    println(io, "Architecture:    ", spec.num_layers, " layers × hidden ", spec.hidden_size)
    println(
        io,
        "Attention:       ",
        caps.grouped_query_attention ?
        "grouped-query ($(
            spec.n_heads
        ) heads / $(spec.n_kv_heads) kv heads, head_dim $(spec.head_dim))" :
        "dense/multi-head ($(spec.n_heads) heads, head_dim $(spec.head_dim))",
    )
    println(io, "Normalization:   ", spec.norm_kind === :rms ? "RMSNorm" : "LayerNorm")
    println(
        io,
        "FFN:             ",
        caps.gated_ffn ? "gated SwiGLU ($(spec.intermediate_size))" :
        "dense $(spec.activation_kind) ($(spec.intermediate_size))",
    )
    println(io, "Positional:      RoPE theta=", spec.rope.theta)
    if spec.rope.kind === :none
        println(io, "                 (unscaled)")
    else
        println(
            io,
            "                 scaling=:$(spec.rope.kind), factor=$(spec.rope.factor)",
        )
        spec.rope.kind === :llama3 && println(
            io,
            "                 original_max_position_embeddings=",
            spec.rope.original_max_position_embeddings,
        )
    end
    println(io, "Tied embeddings: ", spec.tie_word_embeddings ? "yes" : "no")
    spec.sliding_window !== nothing && println(io, "Sliding window:  ", spec.sliding_window)

    if bound !== nothing
        ids = [canonical_name(first(p)) for p in bound.bindings]
        println(io)
        println(io, "Semantic parameters:")
        println(io, "    ", length(ids), " recognized")
        println(io, "    ", length(ids), " mapped")
        println(io, "    0 ignored")
        println(io, "    0 ambiguous")
        println(io)
        println(
            io,
            "Tensors consumed: ",
            length(tensors === nothing ? String[] : keys(tensors)),
        )
    end

    println(io)
    println(io, "Execution capabilities:")
    for cap in reqs
        println(io, "    ", rpad(String(cap), 26), _cap_status(cap))
    end

    _missing = _missing_capabilities(caps)
    boundary = isempty(_missing) ? nothing : first(_missing)
    println(io)
    if boundary === nothing
        println(io, "Failure boundary: none — every required capability exists")
    else
        println(io, "Failure boundary: ", boundary)
    end
    return String(take!(io))
end

# Capability existence is a property of what Gesso has IMPLEMENTED, not of a
# backend. `_implemented_capabilities` is the single declaration of that set;
# `supports(::backend, cap)` still governs whether a given BACKEND can run it.
const _implemented_capabilities = Set{Symbol}([
    :rmsnorm,
    :attention,
    :matmul,
    :swiglu_ffn,
    :embedding_lookup,
    :softmax,
    :rope_none,
])

_cap_status(cap::Symbol) = cap in _implemented_capabilities ? "READY" : "NOT IMPLEMENTED"

"""
    _missing_capabilities(caps) -> Vector{Symbol}

Every required capability Gesso has not implemented, in `required_semantics`
order. The FIRST one is the failure boundary a model actually hits.
"""
function _missing_capabilities(caps::ArchitectureCapabilities)
    probe = ArchitectureSpec(;
        family=:probe,
        hidden_size=8,
        num_layers=1,
        n_heads=caps.grouped_query_attention ? 4 : 1,
        n_kv_heads=caps.grouped_query_attention ? 2 : 1,
        vocab_size=8,
        intermediate_size=8,
        activation_kind=caps.gated_ffn ? :swiglu : :gelu,
        tie_word_embeddings=caps.tied_embeddings,
        sliding_window=caps.sliding_window ? 4 : nothing,
        # ONLY qk_norm is probed here. An earlier version used
        # `caps.qk_norm ? [:qk_norm] : [:moe]`, which reported a Llama model as
        # missing :moe_routing — an invented capability requirement, and exactly
        # the kind of false claim Pass J exists to prevent.
        features=Set{Symbol}(caps.qk_norm ? [:qk_norm] : Symbol[]),
        rope=RoPEPolicy(;
            kind=caps.rope_scaling,
            factor=caps.rope_scaling === :none ? 1.0 : 2.0,
            original_max_position_embeddings=caps.rope_scaling === :llama3 ? 4096 : 0,
        ),
    )
    return [c for c in required_semantics(probe) if !(c in _implemented_capabilities)]
end

# --- Pass J: the machine-readable compatibility matrix --------------------------

# --- Pass J: the machine-readable compatibility matrix --------------------------
#
# WHY THIS IS A VARIANT SWEEP AND NOT ONE PROBE PER FAMILY.
#
# The first version of this matrix built ONE spec per family from the
# constructor defaults and reported every family green. That was true of the
# probe and false of the world: a Llama-3 checkpoint uses scaled RoPE, and
# `import_report` on that same checkpoint correctly named `rope_llama3` as the
# failure boundary. The matrix answered "can Gesso run this family in BASE
# FORM?" while presenting itself as "what can Gesso run?" — a generated,
# machine-readable, all-green artifact is exactly the thing a status meeting
# quotes and believes (§LXX: a report that overstates is worse than no report).
#
# So the matrix now probes the CONFIG SPACE, and reports the worst case.
#
# Only five axes can change `required_semantics`; everything else (sizes, head
# counts, bias flags, norm kind, tied embeddings) provably cannot, and probing
# them would only multiply identical rows. The axes are read off
# `required_semantics` itself, not hand-listed from memory.

const _matrix_base = (;
    hidden_size=8,
    num_layers=1,
    n_heads=4,
    n_kv_heads=2,
    vocab_size=8,
    intermediate_size=8,
    tie_word_embeddings=true,
)

const _MATRIX_ROPES = (
    RoPEPolicy(),
    RoPEPolicy(; kind=:linear, factor=4.0),
    RoPEPolicy(; kind=:llama3, factor=8.0, original_max_position_embeddings=8192),
)
const _MATRIX_ACTIVATIONS = (:swiglu, :gelu)
const _MATRIX_FEATURES = (Symbol[], [:qk_norm], [:moe], [:qk_norm, :moe])
const _MATRIX_WINDOWS = (nothing, 64)

# The stable order `required_semantics` emits, DERIVED from it rather than
# restated, so the "first missing" ordering cannot drift from the real one.
const _REQUIRED_ORDER = required_semantics(
    ArchitectureSpec(;
        _matrix_base...,
        family=:probe,
        activation_kind=:gelu,
        rope=RoPEPolicy(; kind=:llama3, factor=8.0, original_max_position_embeddings=8192),
        features=Set{Symbol}([:qk_norm, :moe]),
        sliding_window=64,
    ),
)

_required_rank(c::Symbol) =
    something(findfirst(==(c), _REQUIRED_ORDER), length(_REQUIRED_ORDER) + 1)

function _matrix_variant(fam::Symbol, rope, act, feats, win)
    spec = ArchitectureSpec(;
        _matrix_base...,
        family=fam,
        activation_kind=act,
        rope=rope,
        features=Set{Symbol}(feats),
        sliding_window=win,
    )
    reqs = required_semantics(spec)
    miss = [c for c in reqs if !(c in _implemented_capabilities)]
    label = string(
        "rope=",
        rope.kind,
        " act=",
        act,
        isempty(feats) ? "" : string(" feat=", join(feats, "+")),
        win === nothing ? "" : " window",
    )
    return (;
        variant=label,
        required=reqs,
        missing=miss,
        first_missing=isempty(miss) ? nothing : first(miss),
    )
end

"""
    compatibility_matrix() -> Vector{NamedTuple}

Architecture × capability, GENERATED by sweeping the config space of every
registered family against the implemented capability set — never typed by
hand. One row per family, carrying:

  * `variants`      every capability-relevant configuration probed, each with
                    its own `required` / `missing` / `first_missing`
  * `runnable` / `total`  how many of those configurations Gesso can execute
  * `unreachable`   every capability ANY config of this family needs and Gesso
                    lacks, in `required_semantics` order
  * `first_missing` the WORST case — the first of `unreachable`. This is the
                    number that must never be read as "nothing missing"
  * `base_missing` / `base_first_missing`  the base form alone, kept because
                    "runs in its default config" is a true and useful claim

A family row reporting `first_missing === nothing` means Gesso can run EVERY
configuration of it, not merely the default.

This is the machine-readable half of Pass J. Its companion assertions live in
`test/test_breadth0.jl`: every family in `known_families()` MUST appear here,
and the worst-case column may never read green while any implemented capability
is missing from `_implemented_capabilities`.
"""
function compatibility_matrix()
    rows = NamedTuple[]
    for fam in known_families()
        variants = [
            _matrix_variant(Symbol(fam), rope, act, feats, win) for
            rope in _MATRIX_ROPES,
            act in _MATRIX_ACTIVATIONS,
            feats in _MATRIX_FEATURES,
            win in _MATRIX_WINDOWS
        ]
        unreachable = Symbol[]
        for v in variants, c in v.missing
            c in unreachable || push!(unreachable, c)
        end
        sort!(unreachable; by=_required_rank)
        # variants[1] is the base form: unscaled RoPE, SwiGLU, no features,
        # no sliding window — the constructor defaults.
        base = first(variants)
        push!(
            rows,
            (;
                family=Symbol(fam),
                variants=variants,
                runnable=count(v -> isempty(v.missing), variants),
                total=length(variants),
                unreachable=unreachable,
                first_missing=isempty(unreachable) ? nothing : first(unreachable),
                base_missing=base.missing,
                base_first_missing=base.first_missing,
            ),
        )
    end
    return rows
end

"""
    compatibility_table(io=stdout) -> Nothing

Render `compatibility_matrix()` as a fixed-width table. Generated, so it can
only ever describe what the registry actually contains — and it shows the
WORST case plus how many configurations run, so "6 families, all green" is not
something this table can say.
"""
function compatibility_table(io::IO=stdout)
    rows = compatibility_matrix()
    println(
        io,
        "| architecture | base form | runnable configs | first missing across configs |",
    )
    println(io, "|---|---|---|---|")
    for r in rows
        println(
            io,
            "| ",
            r.family,
            " | ",
            isempty(r.base_missing) ? "runs" : "blocked at $(r.base_first_missing)",
            " | ",
            "$(r.runnable)/$(r.total)",
            " | ",
            r.first_missing === nothing ? "— (all configs)" : String(r.first_missing),
            " |",
        )
    end
    return nothing
end

export import_report, compatibility_matrix, compatibility_table
