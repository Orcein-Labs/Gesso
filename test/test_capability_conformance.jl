# CAPABILITY-0a — the capability declaration, tied to execution.
#
# `supports(backend, cap)` is a hand-written Set. BREADTH-2 collapsed three
# declarations of one fact into one, which is better and still a LIST — add
# `rope_linear` to CUDA's `rope!` and forget the Set, and the matrix lies
# again, silently, exactly as before. Registration would not help: a
# registration table is another declaration, maintained the same way.
#
# Evidence is the only thing that does not rot. So this file states the
# biconditional in five arms, each of which fails on its own:
#
#   I.   claimed  ⇒ proven       supports(b,cap)  ⇒ the operation executes
#   I′.  proven   ⇒ claimed      the operation executes ⇒ supports(b,cap)
#   II.  absent   ⇒ declared     !supports(b,cap) ⇒ import_report names it as
#                                     the failure boundary of a spec that needs it
#   III. the two compatibility surfaces agree with each other
#   IIIb. …per variant, matched on the required list rather than on a label
#
# Arm I is STRICT: claiming a capability with no probe is a FAILURE, not a
# `@test_broken`. This is mutation-verified — a `@test_broken` branch let a
# fabricated capability pass the suite green.
#
# Arm II is not hypothetical: a `moe` spec was declared fully runnable by
# `import_report` while the matrix correctly named `:moe_routing`, because
# `_missing_capabilities` rebuilt a probe that propagated `qk_norm` and
# dropped `moe`. Arm III is what makes that class of bug impossible.
#
# Every probe EXECUTES. None of them asserts a boolean the code already knows.

using .GessoTestHelpers: deterministic_rng

# The backend's declared set, named here so the arms below can iterate what is
# CLAIMED and not only what some spec happens to require.
const CPU_SUPPORTED_CAPS = Gesso.CPU_SUPPORTED_CAPS

# Run `f`; :proven if it completed, :refused if the lowering declined, and a
# hard error if anything ELSE went wrong — a probe that throws UndefVarError is
# a broken probe, not a missing capability, and must not be laundered.
function _conformance_run(f)
    try
        f()
        return :proven
    catch e
        e isa Gesso.LoweringNotImplemented && return :refused
        rethrow()
    end
end

_act(a) = Gesso.Activation(; shape=size(a), storage=a)
_froz(a) = Gesso.FrozenParameter(; shape=size(a), storage=a)
# `matmul!` computes `dst = x * transpose(W)`, so W is stored (out, in).
_pw(a) = Gesso.ProjectionWeight(; shape=size(a), storage=a)
_ws(a) = Gesso.TemporaryWorkspace(; shape=size(a), storage=a)

# `:attention` has NO lowering operator of its own — the engine composes it from
# matmul/softmax/matmul. That makes it the one capability whose "declared" and
# "implemented" facts are easiest to assert without proving anything, which is
# exactly the split-brain this file exists to close. So it gets a fixture that
# EXECUTES the composition through the backend's own operations, written out
# here rather than delegated to a session: a conformance fixture that calls the
# engine proves the engine agrees with itself, not that attention is supported.
#
# Inputs are passed IN rather than drawn inside, so the correctness arm below can
# re-derive the answer from the same numbers instead of trusting the fixture.
function _attention_case()
    P, H, D = 3, 2, 4
    rng = deterministic_rng(7)
    x = randn(rng, P, H * D)
    vs = [randn(rng, P, D) for _ in 1:H]
    return (; P, H, D, x, vs)
end

function _probe_attention(backend, wl, c=_attention_case())
    out = zeros(c.P, c.H * c.D)
    scores = zeros(c.P, c.P)
    probs = zeros(c.P, c.P)
    for h in 1:c.H
        cols = ((h-1)*c.D+1):(h*c.D)
        # Q_h (P,D) · K_hᵀ ⇒ scores (P,P), scaled by 1/√d as the engine does.
        # `matmul!` demands an Activation DESTINATION; only `softmax!` takes
        # workspace. Passing workspace to matmul! falls through to the lowering
        # stub and the probe reads `:refused` — which looks exactly like a
        # missing capability and is not one.
        Gesso.matmul!(
            backend,
            _act(scores),
            _act(@view(c.x[:, cols])),
            _pw(c.x[:, cols]),
            wl,
        )
        scores ./= sqrt(c.D)
        Gesso.softmax!(backend, _ws(probs), _ws(scores), wl)
        # probs (P,P) · V_h (P,D) ⇒ out (P,D), accumulated into the merge buffer
        Gesso.matmul!(
            backend,
            _act(@view(out[:, cols])),
            _act(probs),
            _pw(permutedims(c.vs[h])),   # V stored transposed: W is (out, in)
            wl,
        )
    end
    return out
end

# Plain-Julia causal attention over the same draws. Deliberately shares no code
# with the fixture above: if both call `softmax!`, a wrong softmax passes both.
function _attention_reference(c)
    out = zeros(c.P, c.H * c.D)
    for h in 1:c.H
        cols = ((h-1)*c.D+1):(h*c.D)
        q = c.x[:, cols]
        k = c.x[:, cols]
        s = [sum(q[t, j] * k[u, j] for j in 1:c.D) / sqrt(c.D) for t in 1:c.P, u in 1:c.P]
        for t in 1:c.P, u in (t+1):c.P
            s[t, u] = -Inf                       # causal mask
        end
        m = [maximum(@view s[t, :]) for t in 1:c.P]
        e = [exp(s[t, u] - m[t]) for t in 1:c.P, u in 1:c.P]
        p = [e[t, u] / sum(@view e[t, :]) for t in 1:c.P, u in 1:c.P]
        for t in 1:c.P, j in 1:c.D
            out[t, cols[j]] = sum(p[t, u] * c.vs[h][u, j] for u in 1:c.P)
        end
    end
    return out
end

# One probe per capability `required_semantics` can emit. A probe that returns
# :unsupported means "there is no operation to call for this yet" — those are
# covered by arm II instead, which is the honest way to say it.
function _probe(backend, cap::Symbol)
    wl = Gesso.PrefillWorkload()
    rng = deterministic_rng(7)
    if cap === :rmsnorm
        x = _act(randn(rng, 2, 4))
        return _conformance_run(
            () -> Gesso.rmsnorm!(backend, _act(zeros(2, 4)), x, _froz(ones(4)), wl),
        )
    elseif cap === :matmul
        x = _act(randn(rng, 3, 4))
        return _conformance_run(
            () -> Gesso.matmul!(backend, _act(zeros(3, 5)), x, _pw(randn(rng, 5, 4)), wl),
        )
    elseif cap === :softmax
        s = randn(rng, 2, 4)
        return _conformance_run(() -> Gesso.softmax!(backend, _ws(zeros(2, 4)), _ws(s), wl))
    elseif cap === :attention
        return _conformance_run(() -> _probe_attention(backend, wl))
    elseif cap === :embedding_lookup
        tbl = Gesso.EmbeddingTable(; shape=(8, 4), storage=randn(rng, 8, 4))
        return _conformance_run(
            () -> Gesso.embedding_lookup!(backend, _act(zeros(2, 4)), tbl, [1, 3], wl),
        )
    elseif cap === :swiglu_ffn
        g = _act(randn(rng, 2, 4))
        u = _act(randn(rng, 2, 4))
        return _conformance_run(() -> Gesso.swiglu!(backend, _act(zeros(2, 4)), g, u, wl))
    elseif cap in (:rope_none, :rope_linear, :rope_llama3)
        q = _act(reshape(randn(rng, 16), (2, 2, 4)))
        k = _act(reshape(randn(rng, 16), (2, 2, 4)))
        inv =
            cap === :rope_none ? nothing :
            Gesso.rope_inv_freq(
                cap === :rope_linear ? Gesso.RoPEPolicy(; kind=:linear, factor=4.0) :
                Gesso.RoPEPolicy(;
                    kind=:llama3,
                    factor=8.0,
                    original_max_position_embeddings=8192,
                ),
                4,
            )
        return _conformance_run(
            () -> Gesso.rope!(backend, q, k, [0, 1], wl; theta=10000.0, inv_freq=inv),
        )
    else
        return :unsupported
    end
end

# The capabilities a spec can be built to require. Derived from
# `required_semantics` itself via one probe per axis value, so a new capability
# cannot be added to the vocabulary without appearing here — and appearing here
# without a probe fails arm I.
const _CAP_PROBE_SPECS = [
    Gesso.ArchitectureSpec(;
        family=:probe,
        hidden_size=8,
        num_layers=1,
        n_heads=2,
        n_kv_heads=1,
        vocab_size=8,
        intermediate_size=16,
    ),
    Gesso.ArchitectureSpec(;
        family=:probe,
        hidden_size=8,
        num_layers=1,
        n_heads=2,
        n_kv_heads=1,
        vocab_size=8,
        intermediate_size=16,
        activation_kind=:gelu,
    ),
    Gesso.ArchitectureSpec(;
        family=:probe,
        hidden_size=8,
        num_layers=1,
        n_heads=2,
        n_kv_heads=1,
        vocab_size=8,
        intermediate_size=16,
        features=Set{Symbol}([:qk_norm]),
    ),
    Gesso.ArchitectureSpec(;
        family=:probe,
        hidden_size=8,
        num_layers=1,
        n_heads=2,
        n_kv_heads=1,
        vocab_size=8,
        intermediate_size=16,
        features=Set{Symbol}([:moe]),
    ),
    Gesso.ArchitectureSpec(;
        family=:probe,
        hidden_size=8,
        num_layers=1,
        n_heads=2,
        n_kv_heads=1,
        vocab_size=8,
        intermediate_size=16,
        sliding_window=4,
    ),
    Gesso.ArchitectureSpec(;
        family=:probe,
        hidden_size=8,
        num_layers=1,
        n_heads=2,
        n_kv_heads=1,
        vocab_size=8,
        intermediate_size=16,
        rope=Gesso.RoPEPolicy(; kind=:linear, factor=4.0),
    ),
    Gesso.ArchitectureSpec(;
        family=:probe,
        hidden_size=8,
        num_layers=1,
        n_heads=2,
        n_kv_heads=1,
        vocab_size=8,
        intermediate_size=16,
        rope=Gesso.RoPEPolicy(;
            kind=:llama3,
            factor=8.0,
            original_max_position_embeddings=8192,
        ),
    ),
]

const ALL_SEMANTIC_CAPS = sort!(
    unique(
        reduce(
            vcat,
            [Gesso.required_semantics(s) for s in _CAP_PROBE_SPECS];
            init=Symbol[],
        ),
    );
    by=String,
)

_report_boundary(spec; backend=Gesso.CPUBackend()) =
    match(r"Failure boundary: (\S+)", Gesso.import_report(spec; backend=backend)).captures[1]

# `findfirst` over a vector returns the INDEX. Asking for the capability needs
# the ELEMENT, and getting this wrong once already cost a debugging round.
function _first_unsupported(reqs, backend)
    for c in reqs
        Gesso.supports(backend, c) || return c
    end
    return nothing
end

# a spec that requires exactly `cap`, so arm II asks a question with one answer
function _spec_requiring(cap::Symbol)
    for s in _CAP_PROBE_SPECS
        cap in Gesso.required_semantics(s) && return s
    end
    error("no probe spec requires $(cap) — the vocabulary grew and this must too")
end

@testset "CAPABILITY-0a: the declaration is tied to execution" begin
    backend = Gesso.CPUBackend()

    @testset "arm I — claimed implies proven" begin
        # Iterate the UNION of what the backend claims and what a spec can
        # require. Iterating only the required set was a hole: `:softmax` and
        # `:embedding_lookup` are claimed by CPU and emitted by NO spec, so
        # neither of them was ever executed by this arm — a claimed capability
        # with no proof, which is the exact defect the arm exists to forbid.
        for cap in sort!(collect(union(CPU_SUPPORTED_CAPS, ALL_SEMANTIC_CAPS)); by=String)
            Gesso.supports(backend, cap) || continue
            outcome = _probe(backend, cap)
            # STRICT, deliberately. An earlier version branched to
            # `@test_broken false` here, and a mutation check proved why that
            # was worthless: adding a capability no operation implements left
            # the suite GREEN (Julia counts `broken` as a non-failure). A
            # declared capability with no proof is the defect, so it must be a
            # failure. There is no probe-free claimed capability to accommodate
            # — `:attention` used to be one, and now has a fixture.
            @test outcome === :proven
        end
    end

    @testset "arm I′ — proven implies claimed" begin
        # The other direction, and the one that was MISSING. Deleting
        # `:rope_llama3` from CPU_SUPPORTED_CAPS — while the CPU `rope!` plainly
        # executes it — left arms I–IIIb entirely green: arm I skips unclaimed
        # capabilities, and arm II is SATISFIED by a report that honestly says
        # "blocked at rope_llama3". Self-consistent, and still wrong: a model
        # that runs is refused, which is the false negative that made BREADTH-1
        # necessary in the first place. So evidence runs BOTH ways — anything
        # the backend can actually execute MUST be claimed.
        for cap in sort!(collect(union(CPU_SUPPORTED_CAPS, ALL_SEMANTIC_CAPS)); by=String)
            _probe(backend, cap) === :proven || continue
            @test Gesso.supports(backend, cap)
        end
    end

    @testset "the attention fixture computes attention, not merely a number" begin
        # "Did not throw" is a weak receipt. A backend that returned garbage
        # would satisfy arm I just as happily as one that returns the right
        # answer, so the composed probe is checked against a reference that
        # shares no operator with it.
        c = _attention_case()
        got = _probe_attention(backend, Gesso.PrefillWorkload(), c)
        want = _attention_reference(c)
        @test !all(iszero, got)
        @test maximum(abs.(got .- want)) <= 1e-12
    end

    @testset "arm II — absent implies declared" begin
        # A capability the backend cannot run must be NAMED as the failure
        # boundary of any spec that requires it. This is the arm that a `moe`
        # spec failed: import_report said "none" while the matrix said
        # :moe_routing, because the probe it rebuilt dropped `moe`.
        for cap in ALL_SEMANTIC_CAPS
            Gesso.supports(backend, cap) && continue
            spec = _spec_requiring(cap)
            @test _report_boundary(spec) == String(cap)
        end
    end

    @testset "arm III — the two compatibility surfaces agree" begin
        for spec in _CAP_PROBE_SPECS
            want = _first_unsupported(Gesso.required_semantics(spec), backend)
            @test _report_boundary(spec; backend=backend) ==
                  (want === nothing ? "none" : String(want))
        end
    end

    @testset "arm IIIb — report agrees with the matrix, per variant" begin
        # The matrix sweeps the config space from ITS OWN base spec; the report
        # answers about the probe specs, built independently. Two independently
        # constructed surfaces must agree wherever they describe the same
        # configuration — matched on the REQUIRED LIST, not on a label, because
        # a shared label is exactly the kind of coincidence that hides drift.
        m = first(Gesso.compatibility_matrix(; backend=backend))
        matched = 0
        for spec in _CAP_PROBE_SPECS
            reqs = Gesso.required_semantics(spec)
            idx = findfirst(v -> v.required == reqs, m.variants)
            @test idx !== nothing   # the sweep must actually cover this config
            idx === nothing && continue
            matched += 1
            v = m.variants[idx]
            expected = v.first_missing === nothing ? "none" : String(v.first_missing)
            @test _report_boundary(spec; backend=backend) == expected
        end
        @test matched == length(_CAP_PROBE_SPECS)   # not vacuously zero
    end

    @testset "every capability in the vocabulary is accounted for" begin
        # If required_semantics grows a capability, this file must grow with it
        # or arm I silently stops testing it. `_REQUIRED_RANK` is keyed BY symbol
        # and valued by rank, so `collect`ing it yields Pairs — take the keys.
        declared = Set(keys(Gesso.Inference._REQUIRED_RANK))
        @test issubset(ALL_SEMANTIC_CAPS, declared)
        # A claimed capability need not be required by any spec — `:softmax` and
        # `:embedding_lookup` are engine-internal — so it is NOT a defect for it
        # to be absent from `_REQUIRED_RANK`. Arm I covers those by execution.
        @test issubset(ALL_SEMANTIC_CAPS, union(declared, CPU_SUPPORTED_CAPS))
        probed = Set(c for c in ALL_SEMANTIC_CAPS if _probe(backend, c) !== :unsupported)
        @test issubset(probed, declared)
    end
end
