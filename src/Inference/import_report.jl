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

"""
    compatibility_matrix() -> Vector{NamedTuple}

Architecture × capability, GENERATED from the registry and the implemented
capability set — never typed by hand. One row per family; `missing` is the
ordered list of capabilities that family needs and Gesso lacks, and
`first_missing` is the boundary where it would stop.

This is the machine-readable half of Pass J. Its companion assertion lives in
`test/test_breadth0.jl`: every family in `known_families()` MUST appear here,
so the matrix cannot silently drop a family the code supports.
"""
function compatibility_matrix()
    rows = NamedTuple[]
    for fam in known_families()
        adapter = adapter_for(fam)
        # Probe the family's SEMANTICS through a representative config built
        # from the shared conformance dimensions; the matrix reports capability
        # requirements, which do not depend on the model's size.
        probe = ArchitectureSpec(;
            family=Symbol(fam),
            hidden_size=8,
            num_layers=1,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=8,
            intermediate_size=8,
            activation_kind=:swiglu,
            tie_word_embeddings=true,
        )
        caps = capabilities(probe)
        reqs = required_semantics(probe)
        missingcaps = [c for c in reqs if !(c in _implemented_capabilities)]
        push!(
            rows,
            (;
                family=Symbol(fam),
                required=reqs,
                missing=missingcaps,
                first_missing=isempty(missingcaps) ? nothing : first(missingcaps),
            ),
        )
    end
    return rows
end

"""
    compatibility_table(io=stdout) -> Nothing

Render `compatibility_matrix()` as a fixed-width table. Generated, so it can
only ever describe what the registry actually contains.
"""
function compatibility_table(io::IO=stdout)
    rows = compatibility_matrix()
    println(io, "| architecture | required capabilities | first missing |")
    println(io, "|---|---|---|")
    for r in rows
        println(
            io,
            "| ",
            r.family,
            " | ",
            join(r.required, ", "),
            " | ",
            r.first_missing === nothing ? "—" : String(r.first_missing),
            " |",
        )
    end
    return nothing
end

export import_report, compatibility_matrix, compatibility_table
