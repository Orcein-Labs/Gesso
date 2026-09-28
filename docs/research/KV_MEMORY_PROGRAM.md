# HARPE MEMORY PROGRAM
## Attention-Memory Rate–Distortion Compilation

    Internal codename:  Magenta Memory™
    Document class:     research program / phase-5–10 specification seed
    Status:             accepted direction, pre-data
                    (external paper: BLOCKED until first composition
                     experiment produces numbers — Harpe_Stack.md §XLII,
                     North Star §42: no paper before there is data)
    Derived from:       Harpe_musings.md (repo root — the dangerous
                        notebook — do not sanitize; promote from it,
                        never wholesale)
    Canon:              docs/Harpe_Stack.md remains canon; this document
                        extends §XXXI, §X, §LIX and feeds Phases 5/9/10.

---

## 0. ABSTRACT

Harpe's thesis — *representation is a lowering decision* — has now generated
the same decomposition four independent times: quantization (§XV), placement
and sharding, attention domain, and now working state (KV). This document
promotes the fourth instance from musing to program.

The central object is no longer `KVCache`. It is:

> **a logical attention-memory object that Harpe is free to physically
> realize in multiple ways, searched, verified, and cached like any other
> lowering.**

The central objective change: stop minimizing reconstruction error of stored
coordinates; start minimizing **damage to attention behavior** per unit of
memory and runtime cost. The central method: rate–distortion optimization
over **pipelines of lawful transformations**, executed by the same
bounded-search machinery the kernel autotuning project already specifies.

The boundary that keeps this sane:

> **NIRA decides what memory means.
> Harpe decides how that memory lives and participates in computation.**

---

## 1. THE INVARIANT

One sentence has now correctly predicted four subsystem decompositions:

    "X is a lowering decision."

    quantization      → §XV            (confirmed, planned Phase 10)
    placement         → §XIV/§XXX     (confirmed, planned Phase 11)
    sharding          → musings        (starred, parked — V1.5+)
    KV representation → this document  (Phase 5/10)

A thesis that keeps generating valid subsystem decompositions is probably
true. Working state is the highest-leverage instance because KV is the
dominant memory consumer at long context and the only major state that is
*continuously rewritten by execution*.

**Canonical invariant (goes in the architecture vocabulary):**

> Working-state representation is a lowering decision, subject to declared
> approximation contracts, searched under explicit budgets, and cached under
> conservative invalidation.

---

## 2. THE OBJECTIVE SWAP

### 2.1 The wrong objective

Conventional KV compression minimizes reconstruction error:

    minimize ||K − K̂||²   (and similarly for V)

### 2.2 What attention actually consumes

Attention computes scores `s = qᵀk`. If a stored key carries error `ε`:

    Δs = qᵀε
    E[Δs²] ≈ εᵀ Σ_q ε        (Σ_q = query covariance for that head)

Two key errors with identical Euclidean norm can differ by orders of
magnitude in attention damage. Errors perpendicular to the query
distribution are nearly free; errors aligned with it are catastrophic.

**V has different math.** Value errors are weighted by the attention mass
with which each value is actually read. There is no quadratic form; the
first oracle rung that sees V damage honestly is attention-output
distortion (§3). Any scheme that treats K and V with one metric is
suspect by construction.

### 2.3 The objective family

    minimize attention distortion(K̂, V̂ | query distribution, workload)
    subject to memory, latency, and quality budgets

### 2.4 Where Σ_q comes from — and why this is in scope

Estimating `Σ_q` (per layer, per head) is a **calibration pass**: a
read-only workload over the inference engine's own operator set. This is
exactly the activity §LIX preserved when training was excised:

> Calibration runs as a read-only workload over the inference engine's own
> operator set. It is measurement, not learning.

No AD machinery. No training dependency. The boundary work and this
research direction lock together by design.

### 2.5 Σ_q is an intermediate oracle, not the final one

The quadratic form captures second-order query sensitivity. Softmax is
nonlinear; values matter; tasks matter. Σ_q-weighted score distortion
ranks candidates cheaply — it does not certify them. Certification is the
oracle ladder's job (§3).

### 2.6 Calibration honesty

Σ_q estimated on one workload goes stale on another; misallocated bits
follow. Therefore:

    every research receipt records the calibration distribution
    (workload family, token budget, collection date) alongside the
    model/device/workload fingerprint (§8).

---

## 3. THE ORACLE LADDER

Quality gates are a ladder, not a single metric. Cheap rungs run inside
searcher inner loops; expensive rungs certify finalists.

    RUNG 0   byte/element reconstruction        (free, weak signal)
    RUNG 1   attention-score distortion         (Σ_q form; K-side only)
    RUNG 2   attention-output distortion        (first honest V signal)
    RUNG 3   logit / continuation divergence    (greedy decode parity)
    RUNG 4   task-level quality                 (the real oracle; expensive)

Rules:

    * Search proposes using rungs 0–2. Certification requires rungs 3–4.
    * No candidate is promoted across a rung boundary without passing it.
    * Rung 3 already exists in the plan: Phase 3's logit-parity harness
      IS rung 3. The ladder starts paying for itself three phases
      before Phase 10 touches KV.
    * Rung 4 needs a task suite (long-context recall, retrieval,
      generation quality) defined BEFORE the first KV experiment, not
      after the first embarrassing result.

**Oracle immaturity is a declared threat (§10), not a detail.**

---

## 4. THE PROGRAM: RATE–DISTORTION OVER PIPELINES

### 4.1 The optimization problem

    minimize    J(π) = M(π) + λ_L·L(π) + λ_C·C(π)
    subject to  D_attention(π) ≤ ε

where `π` is an entire physical KV policy (a pipeline), `M` memory,
`L` runtime cost, `D` attention distortion under the oracle ladder, and:

    C(π) = measured batchability / execution-irregularity cost.

**C is measured, never modeled.** The tuner already measures throughput;
`C` falls out as the delta between isolated-kernel time and batched
end-to-end throughput. A representation that saves 60% of KV memory and
wrecks GPU utilization is not a win, and only measurement knows.

### 4.2 We are building a codec

This formulation is the architecture of a modern video codec: Lagrangian
rate–distortion optimization per unit, trellis-style bit allocation under
a rate budget, mode decision over transform candidates, cheap proxy
metrics for search plus expensive verification for certification,
explicit modeling of decoder-side cost. Three decades of codec engineering
is prior art to raid, not reinvent. Trellis quantization in particular
maps almost directly onto per-head/token bit allocation (§5, Layer 7).

Codec lessons transfer, including the failure modes: transforms that flatter
one stage can destroy structure another stage exploits (§5.1).

### 4.3 Citation hygiene (binding for all Harpe documents)

In a written artifact, citation = claim of having read it.

    VERIFIED SET (cite freely — solid, widely-replicated pre-2025 work):
        FlashAttention · PagedAttention/vLLM · KIVI · KVQuant ·
        GEAR · H2O · StreamingLLM (attention sinks) · Quest ·
        SnapKV · MInference · product/residual VQ literature

    VERIFY-BEFORE-CITING SET (directions corroborated, papers unread):
        RotorQuant · Attention-Aware Transform Coding · CommVQ ·
        KV-COBRA

    Nothing from the second set enters canon docs, receipts, or papers
    until someone has actually read it. Propagating an unverified citation
    is exactly the failure mode the receipts law (§XLII) exists to prevent.

---

## 5. THE REALIZATION LATTICE (Phase 10 vocabulary)

Ten mostly-orthogonal layers, from quality-free to research-grade. These
become **named lowering choices** in the Phase 10 representation planner
and candidate axes for the search engine (§7).

    LAYER 1  SHARE / DEDUP          immutable prefix sharing, copy-on-write
                                    forks, cross-agent prefix dedup,
                                    page dedup. Nearly quality-free.
                                    Store one copy, not eight.

    LAYER 2  SELECT / CLASS         semantic memory classes
                                    (PERMANENT_PREFIX, TOOL_SCHEMA,
                                    TASK_STATE, EPHEMERAL_REASONING,
                                    RECONSTRUCTIBLE, DISCARDABLE).
                                    Retention becomes selective.
                                    Orthogonal to numerical compression.

    LAYER 3  SUBSPACE               head/page-wise low rank, cross-layer
                                    shared bases, grouped-head redundancy.
                                    Ask whether useful state occupies
                                    r ≪ d coordinates.

    LAYER 4  TRANSFORM              rotations/orthogonal bases conditioning
                                    the representation for what follows
                                    (Givens, isoclinic, learned bases,
                                    Stiefel-optimized). One stage of a
                                    pipeline, never the whole story.

    LAYER 5  VECTOR QUANTIZE        codebooks/lattices over vectors, not
                                    coordinates. Residual/additive VQ can
                                    pass below per-scalar bit floors.

    LAYER 6  SPARSE RESIDUAL        compressed base + small set of
                                    retained important errors. Keeps rare
                                    critical components out of the tiny
                                    bulk representation.

    LAYER 7  BIT ALLOCATION         unequal bits across layers/heads/
                                    tokens/K-vs-V/age. Compression as
                                    resource allocation (trellis-style).

    LAYER 8  RESOLUTION BY AGE      hot: token-level high fidelity.
                                    warm: token-level compressed.
                                    cold: pooled/grouped.
                                    archived: semantic summary.
                                    Reduces the number of elements
                                    attention considers — where KV
                                    compression merges into the
                                    subquadratic attention program (§9).

    LAYER 9  TIER / RESIDE          L0 ultra-hot GPU → L1 compressed GPU →
                                    L2 host-compressed → L3 seam (§8) →
                                    L4 reconstructible source.
                                    Promotion/demotion are lowering
                                    decisions. Eviction is a LATENCY risk
                                    (re-prefill on miss); compression is
                                    a QUALITY risk (compounds across all
                                    subsequent decode). Never blur them.

    LAYER 10 RECONSTRUCT / EVICT    recompute-on-miss policies, semantic
                                    eviction ordering. Folklore warning:
                                    recency eviction is WRONG in known
                                    ways — attention sinks are load-bearing
                                    (StreamingLLM). Semantic classes must
                                    be able to override recency. This is
                                    evidence semantic policy beats
                                    mechanical heuristics on real failure
                                    modes, not just marginal efficiency.

### 5.1 The unit of search is the PIPELINE

    SharePrefix → PageLowRank(48) → AttentionAwareRotation → VQ(1.7b) → SparseResidual(0.4%)

Interference is real: a rotation that produces gorgeous scalar-quant
statistics can flatten exactly the low-rank structure the previous stage
exploited. Therefore components are never benchmarked in isolation; the
composition experiment (§9) benchmarks the pipeline, incrementally, with
ablations. Expected result: some effects compose multiplicatively, some
interfere — and establishing which is the research.

### 5.2 Structural constraint: RoPE

Keys carry rotary positional structure. Transforms and codebooks on K must
respect it (the verify-before-citing set's "RoPE-commutative codebooks"
exist for this reason). RoPE-compatibility is a **named constraint** in the
search space, not an afterthought. V is easy; K is where the geometry bites.

### 5.3 Cost honesty: encoding is not free

Compression executes a pass over KV. Prefill-heavy workloads can pay more
than they save. Compress/decompress latency is first-class in every
measurement table (§9), never an afterthought.

---

## 6. APPROXIMATION CLASSES (correctness contract extension)

The musings stated the hard line; here it becomes contract vocabulary,
extending Harpe_Stack.md §X:

    ExactLowering
        byte-identical semantics to the reference operator.
        Certified by rung 0/logit parity as today.

    BoundedApproximation{metric, ε, oracle_tier}
        a DECLARED semantic contract: this lowering may deviate from
        exact execution by at most ε under `metric`, certified at
        `oracle_tier`. Approximate execution is never an undocumented
        optimization flag.

New failure taxonomy entry:

    APPROXIMATION_BUDGET_EXCEEDED
        a bounded-approximation lowering was measured exceeding its
        declared budget at its declared oracle tier. Treated like any
        VERIFY_MISMATCH: the candidate is disqualified and the failure
        recorded.

Planner law: approximate lowerings are only eligible when the active
policy admits their approximation class. §LXX (no silent anything)
applies verbatim.

---

## 7. THE SEARCH ENGINE (generalization, with a leash)

### 7.1 What changes

The North Star project specified a bounded-search/verify/measure/cache/
receipt machinery for kernel schedules. The framing generalizes:

    Logical semantics
        + constraints (legality, approximation class, budgets)
        + hardware/workload fingerprint
        ↓
    candidate realizations
        ↓
    legality oracle → quality oracle (ladder) → cost measurement (M, L, C)
        ↓
    winner + full receipt + conservative cache entry

Clients, eventually:

    kernel schedule search        (North Star, first client)
    quantization programs         (Phase 10)
    KV realization pipelines      (this document)
    memory placement / tiering    (Phase 11)
    sharding / placement          (parked)
    execution-plan selection      (Phase 7+)

**Deliberately absent: speculative plans.** Every real client has a
numeric or functional oracle over a fixed candidate. Speculation verifies
outcome equivalence over an uncertain future — a structurally different
oracle. Adding it to the client list now is the kind of free-looking
abstraction that costs a rewrite later. Revisit only from Phase 20.

### 7.2 The leash

The FRAMING is adopted now: keep the word "kernel" out of interface
vocabulary; the North Star's adapter contract (§27 there) is already
generic. The BUILD stays kernel-first. KV becomes the second client
exactly when Phase 10 earns the generalization — not before.

    Graveyard rule: universal interfaces designed around one client.
    Every layer earns the layer above it (§III).

---

## 8. THE OWNERSHIP SEAM

    Harpe/NIRA memory hierarchy:

        L0  current attention state         Harpe
        L1  recent session working context  Harpe
        L2  compressed session context      Harpe
        L3  semantic working memory         SEAM (contract only)
        L4  episodic / long-term memory     NOT HARPE
        L5  source artifacts                NOT HARPE

    NIRA decides what memory means.
    Harpe decides how that memory lives and participates in computation.

Harpe owns L0–L2 and the L2↔L3 contract. Semantic/episodic cognition above
the seam is NIRA's, exactly as training is somebody else's (§LVIII). If
Harpe ever builds L3+, it has quietly become an opinionated cognition
framework, and the project has failed its own boundary law (§XLIII).

The merge rule at the seam (from the musings, promoted): raw KV states of
diverged agents are NOT mathematically glueable. Merges happen in NIRA's
structured space; the result re-enters the machine the only sound way —

> **Prefill is the only sound compiler from meaning to KV.**

You cannot inject understanding as KV; it is computed from tokens through
the model. That is the neurosymbolic boundary with teeth.

---

## 9. EXPERIMENT LADDER

### 9.1 Composition experiment (the core method)

    Baseline
      ↓ + prefix sharing          (Layer 1)
      ↓ + page-wise low rank      (Layer 3)
      ↓ + attention-aware transform (Layer 4)
      ↓ + vector quantization     (Layer 5)
      ↓ + sparse residual         (Layer 6)
      ↓ + adaptive bit allocation (Layer 7)
      ↓ + cold-memory eviction    (Layer 10)

At every step record:

    bytes/token · VRAM · compression ratio ·
    compression latency · decompression latency ·
    decode tok/s · TTFT · prefill cost ·
    attention-score distortion (rung 1) ·
    logit parity (rung 3) · long-context task quality (rung 4) ·
    batched vs isolated throughput delta (C) ·
    agent concurrency · recovery under memory pressure

Then ablate. Multiplicative composition is a hypothesis to establish,
not assume (§5.1).

### 9.2 The jackpot experiment

    SAME checkpoint · SAME long-context benchmark · SAME hardware

        A  normal dense KV management
        B  sliding-window baseline
        C  Harpe semantic KV hierarchy

    measure: task accuracy · effective context retention ·
             decode latency · prefill cost · VRAM · attention work

    Success: C holds quality (rungs 3–4) while attention work grows
    closer to linear than quadratic in context.

### 9.3 The KV math that makes this worth it

For a 7B-class GQA model (8 KV heads, head_dim 128, BF16, 32 layers):
≈ 128 KiB of KV per token. A 20k-token shared doctrine prefix ≈ 2.5 GB.
Four agents independently: ~10 GB. Shared: 2.5 GB. Declared sharing
alone is the difference between a swarm fitting on one card or not.

### 9.4 Declared vs discovered sharing (the differentiator)

RadixAttention-style prefix caching DISCOVERS sharing after the fact by
exact token match. Harpe DECLARES sharing before the fact by identity —
agents share the system prefix because the runtime constructed them that
way. No n-gram matching, no cache warming, no misses from re-serialized
tool schemas. A generic server sees four requests; Harpe sees a tree.
This is a mechanism difference, not a rebrand, and it is the claim an
external paper would eventually hang on.

### 9.5 Sequencing (zero-research-risk first)

    1. Paged KV manager            (Phase 5 anyway; pure engineering)
    2. CoW fork                    (mechanics only; identical bytes
                                    until divergence)
    3. Identity-based prefix share (zero quality risk)
    4. Span classes                (needs span provenance, Phase 12)
    5. Tiered precision            (quality-gated; KIVI/KVQuant prior:
                                    8-bit safe, 4-bit marginal)
    6. Meaning-aware eviction      (attention-sink aware)
    7. Profile-guided KV remat     (closed loop on working state)

Steps 2–3 are first-semantic-win candidates in the Phase 7 class
(same model, same hardware, same correctness, measured advantage).

---

## 10. THREATS TO VALIDITY (declared, revisit per experiment)

    T1  DATA-DEPENDENT SHAPES.  Token-granularity retrieval breaks the
        static-shape assumption that made FlashAttention win; per-request
        divergent attention patterns wreck batching. Page/block
        granularity (Quest-class) is the deployable form. Semantic sparse
        attention is a KERNEL-FAMILY problem for the tuner (§7), with
        batchability measured as C — never a policy flip.
    T2  ORACLE IMMATURITY.       Rungs 2 and 4 do not exist yet. Until
        they do, all quality claims are provisional (§3).
    T3  ROPE STRUCTURE.          K-side transforms must respect rotary
        geometry (§5.2). Ignoring it invalidates key-compression results.
    T4  ENCODING COST.           Prefill-side compression cost can exceed
        savings on prefill-heavy workloads (§5.3).
    T5  CALIBRATION DRIFT.       Σ_q staleness across workloads (§2.6).
        Receipts record the calibration distribution.
    T6  ATTENTION SINKS.         Mechanical eviction folklore is wrong in
        known ways (§5.3, Layer 10). Any eviction experiment must sink-
        pin before it claims anything.
    T7  CITATION ROT.            The verify-before-citing set (§4.3) may
        not survive contact. The program stands on the verified set.
    T8  INTERFERENCE.            Layer composition may be sub-multiplicative
        or destructive (§5.1). That is a finding, not a failure.

---

## 11. WHAT THIS DOCUMENT IS NOT

    Not a Phase plan.        Phasing remains §LXXIII+. This seeds 5/9/10.
    Not an implementation.   No code exists. Phase 5 earns the first.
    Not a public paper.      Blocked until §9.1 produces numbers.
    Not canon-overriding.    Harpe_Stack.md remains canon; this extends
                             §XXXI/§X/§LIX and is subordinate to it.
    Not the notebook.        Harpe_musings.md (repo root) stays the
                             dangerous notebook. Promotions from it are
                             deliberate and recorded here.

## 12. IMMEDIATE ACTIONS (when Phases 5/10 open)

    1. Define the rung-4 task suite BEFORE the first KV experiment.
    2. Implement the paged KV manager with span provenance hooks.
    3. Land CoW fork + identity prefix sharing (win candidates §9.5).
    4. Add BoundedApproximation + APPROXIMATION_BUDGET_EXCEEDED to the
       correctness contract and failure taxonomy.
    5. Stand up calibration passes under the §LIX carve-out; record
       distributions in receipts.
    6. Wire KV pipelines into the Phase 10 planner vocabulary (§5) and
       the North Star search machinery as its second client (§7.2).
