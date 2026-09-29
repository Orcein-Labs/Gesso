# Toy fixture loader (laboratory material — see test/fixtures/toy/README.md).
#
# Loads the data-only fixture pack, enforces the documented contract, and
# exposes the deterministic weight-derivation protocol. This is test-side
# equipment: no Harpe types, no IR preview, no operator semantics. Block
# `kind` strings are data the Phase 2 oracle math will interpret.
#
# Every violation fails LOUDLY with a message that says which contract was
# broken — silent fixture drift is worse than no fixture.

module ToyFixtures

using TOML
using ..HarpeTestHelpers: deterministic_rng

export load_toy_fixture, toy_weights

const FIXTURE_DIR = joinpath(@__DIR__, "fixtures", "toy")
const MODEL_SCHEMA = "harpe-toy-fixture-v1"
const TOKENIZER_SCHEMA = "harpe-toy-tokenizer-v1"
const LOGITS_SCHEMA = "harpe-toy-expected-logits-v1"

# Fields each block kind carries. Unknown kinds and unknown keys are
# contract violations — a typo'd parameter must not be silently ignored by
# the future oracle math.
const BLOCK_FIELDS = Dict(:attention => (:n_heads,), :mlp => (:hidden,))

function _schema_ok(dict, expected, what)
    get(dict, "schema", "") == expected || error(
        "toy fixture: $what has schema $(repr(get(dict, "schema", nothing))), " *
        "expected $(repr(expected)) — loader and data must agree; bump both together",
    )
end

# FNV-1a over bytes: a tiny, fully specified, run-stable digest for the
# fixture seed fold (Symbol objectids are NOT run-stable; this is).
function _fnv1a(s::AbstractString)
    h = UInt64(0xcbf29ce484222325)
    for b in codeunits(s)
        h = (h ⊻ UInt64(b)) * UInt64(0x100000001b3)
    end
    return h
end

"""
    load_toy_fixture([dir]) -> NamedTuple

Load and validate the toy fixture pack from `dir` (default: the shipped
`test/fixtures/toy/`). Returns a plain NamedTuple of DATA:

    (name, vocab_size, dim, blocks, tokenizer, seed, expected_logits)

* `blocks` — ordered `NamedTuple`s `(kind, <kind-specific fields>)`
* `tokenizer` — `(pad, bos, eos, id_text)` with `id_text` a dense
  id → string Vector
* `seed` — the master fixture seed, derived deterministically from the
  architecture data (protocol in the fixtures README)
* `expected_logits` — `nothing` while the slot is empty; the loader
  REFUSES values until the Phase 2 CPU oracle exists to produce them
"""
function load_toy_fixture(dir=FIXTURE_DIR)
    model = TOML.parsefile(joinpath(dir, "model.toml"))
    tok = TOML.parsefile(joinpath(dir, "tokenizer.toml"))
    logits = TOML.parsefile(joinpath(dir, "expected_logits.toml"))
    _schema_ok(model, MODEL_SCHEMA, "model.toml")
    _schema_ok(tok, TOKENIZER_SCHEMA, "tokenizer.toml")
    _schema_ok(logits, LOGITS_SCHEMA, "expected_logits.toml")

    # --- architecture -------------------------------------------------------
    vocab_size = Int(model["vocab"]["size"])
    dim = Int(model["embedding"]["dim"])
    vocab_size > 0 || error("toy fixture: vocab size must be positive")
    dim > 0 || error("toy fixture: embedding dim must be positive")

    raw_blocks = get(model, "block", String[])
    isempty(raw_blocks) && error("toy fixture: model has no blocks")
    blocks = map(raw_blocks) do b
        raw_kind = get(b, "kind", "")
        kind = Symbol(raw_kind)
        haskey(BLOCK_FIELDS, kind) || error(
            "toy fixture: unknown block kind " *
            repr(raw_kind) *
            " — " *
            "fixture kinds are data; extend BLOCK_FIELDS and the README contract " *
            "together, never silently",
        )
        for k in keys(b)
            k in ("kind",) ||
                string(k) in string.(BLOCK_FIELDS[kind]) ||
                error(
                    "toy fixture: block field $(repr(k)) is not part of kind :$kind — " *
                    "typo'd parameters must not be silently ignored",
                )
        end
        params = NamedTuple{BLOCK_FIELDS[kind]}(
            tuple((Int(b[string(f)]) for f in BLOCK_FIELDS[kind])...),
        )
        all(>(0), values(params)) ||
            error("toy fixture: block parameters must be positive ($(params))")
        merge((kind=kind,), params)
    end

    # --- tokenizer ----------------------------------------------------------
    ids = get(tok, "ids", Dict{String, Any}())
    pad = Int(get(ids, "PAD", -1))
    bos = Int(get(ids, "BOS", -1))
    eos = Int(get(ids, "EOS", -1))
    (pad, bos, eos) == (0, 1, 2) || error(
        "toy fixture: control tokens must be PAD=0, BOS=1, EOS=2 (got $((pad, bos, eos)))",
    )

    entries = get(tok, "token", Any[])
    seen = Dict{Int, String}(pad => "PAD", bos => "BOS", eos => "EOS")
    for e in entries
        id = Int(e["id"])
        (pad <= id <= eos) && error(
            "toy fixture: [[token]] entry redefines control id $id (PAD/BOS/EOS own 0-2)",
        )
        text = String(e["text"])
        isempty(text) && error("toy fixture: token id $id has empty text")
        haskey(seen, id) && error(
            "toy fixture: duplicate token id $id ($(repr(seen[id])) vs $(repr(text)))",
        )
        seen[id] = text
    end
    expected_ids = 0:(vocab_size-1)
    Set(keys(seen)) == Set(expected_ids) || error(
        "toy fixture: token ids must be dense 0..$(vocab_size - 1); " *
        "missing = $(sort(collect(setdiff(Set(expected_ids), Set(keys(seen)))))), " *
        "extra = $(sort(collect(setdiff(Set(keys(seen)), expected_ids)))))",
    )
    id_text = String[seen[i] for i in expected_ids]

    # --- expected-logits slot ------------------------------------------------
    if haskey(logits, "value") || get(logits["provenance"], "oracle", "") != ""
        error(
            "toy fixture: expected_logits.toml carries values/oracle provenance — " *
            "filling it requires the Phase 2 CPU oracle to exist first " *
            "(see test/fixtures/toy/README.md); until then the slot stays empty",
        )
    end
    expected_logits = nothing

    # --- master seed (protocol: fold of the architecture data, FNV-1a) -------
    seed = UInt64(0x6f11406b13a90d0f)
    seed = seed * UInt64(0x100000001b3) + _fnv1a(String(model["name"]))
    seed = seed * UInt64(0x100000001b3) + UInt64(vocab_size)
    seed = seed * UInt64(0x100000001b3) + UInt64(dim)
    for b in blocks
        seed = seed * UInt64(0x100000001b3) + _fnv1a(String(b.kind))
        for f in BLOCK_FIELDS[b.kind]
            seed = seed * UInt64(0x100000001b3) + UInt64(getfield(b, f))
        end
    end

    return (
        name=String(model["name"]),
        vocab_size=vocab_size,
        dim=dim,
        blocks=blocks,
        tokenizer=(pad=pad, bos=bos, eos=eos, id_text=id_text),
        seed=seed,
        expected_logits=expected_logits,
    )
end

"""
    toy_weights(fixture, n) -> Vector{Float64}

The first `n` draws of the fixture's master stream — the row-major weight
order the derivation protocol in the fixtures README defines. The ONLY way
fixture tests may obtain randomness. Any change to what consumes the
stream, or in which order, is a fixture protocol break (bump schema tags).
"""
function toy_weights(fixture, n::Integer)
    rng = deterministic_rng(fixture.seed)
    return [randn(rng) for _ in 1:n]
end

end # module ToyFixtures
