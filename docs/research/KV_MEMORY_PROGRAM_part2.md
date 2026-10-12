# GESSO MEMORY PROGRAM — PART II
## Attention Topology, Retrieval, Residency & Execution Compilation

    Internal codename:  Magenta Memory™ — execution plane
    Document class:     research program / companion specification seed
    Status:             exploratory synthesis, pre-data
    Date:               2026-09-29
    Index:              docs/research/README.md
    Companion:          KV_MEMORY_PROGRAM.md (Part I)
    Scope:              Gesso inference memory R&D
    Canon:              docs/Gesso_Stack.md remains canon.
                        Part I defines the representation/quality half.
                        This document defines the topology/access/residency half.

    Research doctrine:
        Papers are R&D fuel, not implementation prescriptions.
        Extract mechanisms. Generalize the useful primitive. Recombine.
        Measure the composition. Keep receipts. Promote only what survives.

---

## 0. ABSTRACT

Part I made the decisive abstraction:

> **The central object is not `KVCache`. It is a logical attention-memory
> object that Gesso may physically realize in multiple ways.**

Part I then developed the **representation half** of that claim:

- share and deduplicate;
- classify spans;
- lower into subspaces;
- transform;
- vector-quantize;
- retain sparse residuals;
- allocate bits unequally;
- change resolution with age;
- tier;
- reconstruct or evict;
- verify every approximation against an oracle ladder.

This document develops the missing **execution half**.

A memory object is not fully described by its bytes. Long-context inference also
depends on:

1. **who produces attention state;**
2. **which layers own or reuse it;**
3. **which historical positions are consulted for a query;**
4. **where the selected state is physically resident;**
5. **when state moves between tiers;**
6. **how much of the model graph executes during prefill;**
7. **how those decisions interact with batching, branches, and agent traces.**

The new central object is therefore the composition:

> **LogicalAttentionMemory + AttentionMemoryPlan**

where the logical object declares identity, provenance, model semantics and
approximation constraints, while the plan declares representation, topology,
selection, residency, movement and execution schedule.

The target is not "a clever KV cache."

The target is:

> **an attention-memory compiler that lowers a model's lawful memory semantics
> into a verified physical execution plan for a particular device, workload and
> budget.**

That is the whole Magenta Memory program.

---

## 1. THE TWO HALVES

Part I and Part II are deliberately separable because they answer different
questions.

### 1.1 Part I — representation plane

    "How should this logical state be represented?"

    identity sharing
    copy-on-write
    span class
    low rank
    transform
    VQ
    residual
    bit allocation
    age resolution
    tier encoding
    rematerialization
    approximation contract

Part I's objective is primarily:

    retain useful attention behavior
    while reducing memory/runtime cost.

### 1.2 Part II — execution plane

    "How should this logical state participate in execution?"

    which layers produce memory
    which layers consume memory
    which layers reuse memory
    where prefill may stop
    which tokens/blocks/pages get read
    which selectors produce those reads
    what remains device-resident
    what moves from host
    what can be prefetched
    what access pattern the kernels see

Part II's objective is primarily:

    avoid doing or moving work
    that the current attention computation does not need.

### 1.3 The combined invariant

Neither half is sufficient alone.

A beautifully quantized cache still wastes time if every layer scans the entire
history.

A brilliant sparse selector still hits a capacity wall if every potentially
selectable KV record must remain in HBM.

A tiny device working set still loses if the host link thrashes.

A model with early-exit prefill still wastes memory if every surviving cache is
represented naively.

Therefore:

> **Representation, attention topology, selection, residency and movement are
> one joint lowering problem with different legality boundaries.**

---

## 2. R&D METHOD — MECHANISM MINING, NOT PAPER CARGO-CULTING

This program does not implement papers because they are new or impressive.

Every external mechanism enters Gesso through the same research funnel:

    PAPER OBSERVATION
        ↓
    What problem did it expose?
        ↓
    What mechanism produced the gain?
        ↓
    What is the most general useful primitive underneath it?
        ↓
    How could that primitive fit Gesso's architecture?
        ↓
    What new combinations become possible?
        ↓
    What is the cheapest experiment that can kill the idea?
        ↓
    evidence → receipt → promotion / graveyard

Research notes should distinguish four states:

    PAPER_OBSERVED
        directly supported by the cited work

    GESSO_HYPOTHESIS
        a proposed synthesis or extrapolation

    ENGINEERING_CANDIDATE
        sufficiently concrete to benchmark

    PROMOTED
        survived the required Gesso oracle + cost gates

No mechanism is canon merely because a paper reported a win.

No mechanism is rejected merely because Gesso would realize it differently.

The point of the source literature is to enlarge the design space.

---

## 3. PRIMARY MECHANISM ATLAS

The following papers matter because they expose different dimensions of the
same memory problem.

### 3.1 YOCO — prefill/decode dependency can be redesigned

**Source:** *You Only Cache Once: Decoder-Decoder Architectures for Language
Models* (Sun et al., 2024), arXiv:2405.05254.

YOCO separates the model into a self-decoder and a cross-decoder. The
self-decoder produces a global KV representation that later cross-decoder
layers reuse through cross-attention. The architecture therefore avoids
materializing a separate full-history global KV cache for every upper layer.

The important Gesso lesson is larger than "cache once."

> **Prefill has a dependency graph, and the full decode graph does not
> necessarily have to execute in order to produce every state decode needs.**

YOCO's execution dependency permits prefill to stop early while still
constructing the state required for later autoregressive decoding.

**Mechanism extracted for Gesso:**

    PREFILL_REACHABILITY

    Given a model topology:
        identify the minimum subgraph required to construct
        all durable decode-time memory products.

For ordinary Transformers that subgraph may be effectively the whole stack.

For architectures designed around reusable intermediate state, it may not be.

The compiler should represent the difference rather than assume one answer.

---

### 3.2 HySparse — let expensive layers become selectors and memory producers

**Source:** *HySparse: A Hybrid Sparse Attention Architecture with Oracle Token
Selection and KV Cache Sharing* (Gao et al., 2026), arXiv:2602.03560.

HySparse interleaves a full-attention layer with several sparse-attention
followers.

The full-attention layer does two jobs:

1. performs full attention;
2. exposes importance information that selects salient history for the
   following sparse layers.

Those sparse layers also reuse the preceding full layer's KV cache for their
global sparse branch.

This attacks two separate costs:

    COMPUTE:
        followers attend to selected history instead of all history

    MEMORY:
        followers reuse an existing KV representation instead of
        independently caching another complete one

**Mechanisms extracted for Gesso:**

    SELECTOR_AUTHORITY
        The component that already computes the strongest relevance signal
        may serve as the selector instead of adding a second predictor.

    CROSS_LAYER_SELECTION_REUSE
        Neighboring consumers may reuse one selection decision when the
        model semantics permit it.

    CROSS_LAYER_KV_REUSE
        A group of consumers may read one producer's KV representation
        rather than own independent copies.

The important whiteboard question is not "should Gesso implement HySparse?"

It is:

> **When can one piece of expensive neural work become a reusable information
> source for several cheaper consumers?**

That is a reusable systems primitive.

---

### 3.3 HySparse2 — sharing can happen at multiple levels

**Source:** *HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing*
(Wei et al., 2026), arXiv:2609.26368.

HySparse2 combines a YOCO-style self-decoder/cross-decoder split with
HySparse-style sparse followers.

It demonstrates two different sharing mechanisms that must remain distinct in
Gesso's vocabulary.

#### Outer level — KV Bridging

Cross-decoder full-attention layers derive their K/V from hidden states produced
by corresponding self-decoder full-attention layers.

The source hidden state is shared.

The target layers still have their own K/V projections.

Therefore:

    SHARED_SOURCE_STATE
        is not the same thing as
    IDENTICAL_KV_REUSE

This is a **bridge**.

#### Inner level — KV Reuse

Within a cross-decoder hybrid block, sparse followers reuse the full-attention
layer's KV cache and selection indices.

This is actual reuse of a produced attention-memory representation.

#### Selection refinement

HySparse2 moves from block-level selection to token-level selection for its
agent-oriented long-context workload, while forcing a recent local window into
the selected set.

The paper's model uses:

    1,024 selected global tokens
    + 128 always-selected recent tokens

The exact numbers are not Gesso doctrine. The transferable design is:

    GLOBAL_QUERY_DEPENDENT_SET
        +
    LOCAL_CONTINUITY_SET

The local set prevents global sparsity from accidentally deleting immediate
temporal continuity.

#### Prefill consequence

By removing the cross-decoder-local SWA dependency and constructing
cross-decoder KV from self-decoder states, the paper can stop cache-building
prefill after the self-decoder.

For its 80B-A3B configuration at 1M tokens, the paper reports:

    prefill FLOPs:
        2.92× lower than HySparse
        5.02× lower than Hybrid SWA

    FP8 KV storage:
        HySparse2    2.69 GB
        HySparse     6.72 GB
        Hybrid SWA  12.09 GB

Those measurements belong to that architecture and configuration; they are
evidence that the mechanism is worth studying, not promises for arbitrary
Gesso batteries.

**Mechanisms extracted for Gesso:**

    MEMORY_BRIDGE
        reusable source representation → target-specific KV projection

    KV_REUSE_GROUP
        one produced KV realization → several lawful consumers

    SELECTION_REUSE_GROUP
        one selector result → several lawful consumers

    FORCED_LOCAL_SET
        a mandatory recent-context component combined with global retrieval

    PREFILL_CUT
        a model-declared graph boundary after which durable decode state
        can be completed without executing the remaining prompt path

---

### 3.4 HiSparse — availability and residency are different properties

**Source:** *HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV
Cache Management* (Xie et al., 2026), arXiv:2608.07009.

HiSparse starts from an important systems observation:

A top-k sparse model may read only a few thousand KV records per decode step,
but conventional serving still keeps the entire selectable history in GPU HBM.

That means sparse attention may dramatically reduce **active reads** while not
reducing **resident capacity**.

HiSparse separates:

    LOGICAL AVAILABILITY
        "this KV position may legally be selected"

from:

    DEVICE RESIDENCY
        "this KV record is physically in HBM right now"

Its design keeps complete KV history in host memory while maintaining a bounded
GPU working set. The sparse selector emits logical positions; the runtime
resolves those positions against the GPU cache, fetches misses, updates
replacement metadata and presents physical slots to the sparse attention
kernel.

Because placement changes but the selected set and KV values do not, this path
is exact with respect to the sparse model.

HiSparse further exploits the fact that sparse selections have locality:

- nearby decode steps often reselect many of the same positions;
- some architectures share selection decisions across layers;
- therefore a relatively small device cache can have a high hit rate;
- shared future selections can enable exact prefetch.

**Mechanisms extracted for Gesso:**

    AVAILABILITY ≠ RESIDENCY

    ATTENTION_SELECTION_AS_DEMAND_SIGNAL

    BOUNDED_DEVICE_WORKING_SET

    LOGICAL_POSITION → PHYSICAL_SLOT RESOLUTION

    SELECTION_LOCALITY

    SELECTION_DRIVEN_PREFETCH

This is the missing runtime counterpart to Part I's tiering lattice.

Part I asks what representation should live in each tier.

Part II asks what should be physically present in the fast tier **right now**.

---

## 4. THE SYNTHESIS — MEMORY IS NOW FIVE THINGS

The old mental model:

    KV cache = tensor buffer

The Part I mental model:

    logical attention-memory
        ↓
    representation lowering

The complete model:

    LOGICAL ATTENTION MEMORY
        │
        ├── IDENTITY
        │     what logical state is this?
        │
        ├── REPRESENTATION
        │     how are its numbers encoded?
        │
        ├── TOPOLOGY
        │     who produces and consumes it?
        │
        ├── SELECTION
        │     which logical positions are read?
        │
        ├── RESIDENCY
        │     where do the required records live?
        │
        └── SCHEDULE
              when are they produced, moved, prefetched,
              decoded, compressed, promoted or evicted?

These dimensions are partially independent.

That is exactly why they belong in a compiler/search architecture rather than a
pile of feature flags.

---

## 5. CONCEPTUAL GESSO OBJECTS

These are vocabulary proposals, not frozen Julia APIs.

### 5.1 `LogicalAttentionMemory`

Represents semantics that must survive any physical lowering.

Conceptually:

    LogicalAttentionMemory
        identity
        model_identity
        token/span provenance
        logical positions
        semantic class constraints
        approximation contract
        reconstruction source
        sharing identity
        lifetime / branch ownership

It says **what exists**.

It does not decide whether that state is FP16 in HBM, quantized in host memory,
shared across layers, selected sparsely, or reconstructed later.

### 5.2 `AttentionMemoryPlan`

Represents one complete physical realization.

Conceptually:

    AttentionMemoryPlan
        representation_pipeline
        producer_graph
        consumer_graph
        sharing_graph
        bridge_graph
        selection_policy
        selection_granularity
        residency_policy
        movement_schedule
        prefill_products
        prefill_cut
        decode_working_set
        approximation_contract
        hardware/workload fingerprint

It says **how this logical memory participates in execution here**.

### 5.3 Why this split matters

It gives Gesso one place to represent:

    ordinary dense Transformer
    GQA / MQA
    MLA-like compressed latent memory
    cross-layer KV sharing
    YOCO-style reusable global caches
    HySparse-style selector/reuse groups
    HySparse2-style bridges
    training-free sparse selection
    hierarchical host/GPU residency
    future architectures we have not seen yet

without forcing Palette or NIRA to know the machinery.

---

## 6. THE SHARING TAXONOMY

"KV sharing" is too overloaded to remain one concept.

Gesso should distinguish at least five kinds.

### 6.1 Horizontal identity sharing

Across sessions, agents or branches:

    common prefix
         │
    ┌────┼────┐
    A    B    C

Implemented by:

    page sharing
    immutable prefix identity
    copy-on-write after divergence

This is primarily Part I Layer 1.

### 6.2 Vertical KV reuse

Across model layers:

    producer KV
      ├── follower layer
      ├── follower layer
      └── follower layer

The consumers read the same produced representation.

HySparse provides an existence proof for the architectural pattern.

### 6.3 Representation bridging

Across model regions:

    source hidden state
          │
          ├── target K/V projection A
          ├── target K/V projection B
          └── target K/V projection C

The source representation is reused, but target KV is not identical.

HySparse2's KV Bridging is the current motivating example.

### 6.4 Selection sharing

Across layers:

    selector result
       ├── consumer layer
       ├── consumer layer
       └── consumer layer

This can save selection work and can expose future access patterns early enough
for prefetch.

### 6.5 Physical deduplication

Independent logical references may point to one identical physical page.

This is a storage optimization and must not be confused with model-level
cross-layer sharing.

### 6.6 Why the taxonomy is useful

The cross-product is where Gesso becomes interesting.

A shared agent prefix may be:

    horizontally shared across four Cyan branches
    ×
    vertically reused by several model layers
    ×
    sparsely read by selected positions
    ×
    partially resident in HBM
    ×
    compressed differently by age

Those are not competing tricks.

They are different axes of one plan.

---

## 7. PREFILL BECOMES A COMPILATION PHASE

Part I established the boundary law:

> **Prefill is the only sound compiler from meaning to KV.**

Part II extends that idea.

Prefill need not produce only "the KV cache."

For a sufficiently expressive Gesso model/runtime pair, prefill may produce a
bundle of durable execution artifacts:

    KV pages
    shared-prefix identity
    hidden-state bridge sources
    sparse-selection metadata
    compact key/index summaries
    page statistics
    span provenance
    attention-memory class metadata
    host backing state
    initial device working set
    prefetch hints
    compression candidates
    reconstruction handles

Therefore the more general abstraction is:

    tokens / structured prompt
              │
              ▼
          PREFILL
              │
              ▼
      ATTENTION-MEMORY PRODUCTS
              │
              ▼
           DECODE

### 7.1 Prefill reachability

For each model topology, Gesso should be able to answer:

> What is the minimum lawful prompt-time execution required to construct all
> durable state needed by decode?

For a normal Transformer the answer may remain "all layers."

For a model-native architecture designed around reusable state, a shallower
prompt graph may be sufficient.

This is not an optimization guess.

It is a model dependency property.

### 7.2 Agentic observation ingestion

This matters especially for Cyan-class workloads:

    short action
        ↓
    long tool observation
        ↓
    short reasoning/action
        ↓
    long observation
        ↓
    repeat

The execution engine should treat "observation ingestion" as a first-class
workload shape.

The R&D question becomes:

> Can Gesso construct useful long-lived attention memory for large observations
> without paying uniform full-depth, full-history cost everywhere?

YOCO/HySparse2 demonstrate one model-native path.

Part I's compression/age/tiering program supplies runtime paths.

Gesso should study the composition.

---

## 8. SELECTION IS A FIRST-CLASS LOWERING AXIS

Long-context memory has two sizes:

    logical history size
    active read set size

Dense attention makes them approximately equal for each query.

Sparse attention decouples them.

Gesso therefore needs an explicit selection vocabulary.

### 8.1 Candidate granularities

    token
    block
    page
    segment
    hybrid

None is globally "best."

Token-level selection spends the budget precisely but creates irregular gather
patterns.

Block/page selection wastes some semantic budget on neighbors but maps more
naturally to memory systems and GPU kernels.

The correct granularity is hardware-, model-, batch- and workload-dependent.

Therefore:

> **selection granularity is a tuner/search dimension, not a philosophical
> commitment.**

### 8.2 Candidate selector sources

A selector may come from:

    full-attention scores
    model-native learned indexer
    compressed key/index state
    page bounds / summaries
    reused neighboring-layer selections
    fixed local window
    hybrid compositions

NIRA semantic classes may constrain selection:

    pin this span
    never discard this class
    this span is reconstructible

but NIRA should not pretend to be the model's attention mechanism.

### 8.3 Local continuity

The HySparse2 forced-window pattern suggests a strong generic form:

    selected_set(query)
        =
    mandatory_local_set
        ∪
    query_dependent_global_set

The exact window and global budget are model/workload parameters.

The architectural value is that local continuity and global retrieval do not
have to compete for the same failure mode.

### 8.4 Batchability remains a cost oracle

Part I already declared execution irregularity `C(π)` a measured cost.

Selection makes that clause even more important.

A selector that is algorithmically elegant but destroys batching is not a win.

Gesso should measure:

    selection kernel cost
    gather/scatter cost
    divergence across requests
    block/page coalescing
    batch throughput
    end-to-end decode latency

not just "attention FLOPs saved."

---

## 9. RESIDENCY IS NOT AVAILABILITY

This becomes a canonical vocabulary distinction.

    AVAILABLE
        the logical record may legally be referenced

    RESIDENT
        the physical record is immediately available in this memory tier

A one-million-token history may remain logically available while only a small
fraction is HBM-resident.

### 9.1 Proposed hierarchy

The exact tiers are hardware-dependent, but conceptually:

    R0  registers / shared / kernel-local staging
    R1  hot HBM attention working set
    R2  compressed HBM backing
    R3  host / unified-memory backing
    R4  reconstructible attention state
    R5  semantic seam → NIRA

Part I decides how a record may be represented in each legal tier.

Part II decides which records occupy the fast tiers now.

### 9.2 Selection-driven demand

Sparse selection gives the runtime an unusually clean signal:

    "these logical positions are about to be read"

That signal can drive:

    hit detection
    promotion
    replacement
    prefetch
    access-frequency tracking
    temperature updates

Dense attention does not expose the same narrow demand stream.

### 9.3 Backing store vs cache

A device working set should be understood as a cache over an authoritative or
reconstructible backing representation.

That avoids treating HBM occupancy as logical ownership.

### 9.4 Exact and approximate paths

Residency movement can be exact:

    same KV values
    different physical location

Compression can be approximate:

    changed representation
    quality contract required

These remain separate risk classes exactly as Part I requires.

---

## 10. NEW GESSO DIRECTION: KEY-FIRST, VALUE-LATE MEMORY

**Status: GESSO_HYPOTHESIS**

Part I already established that K and V have different error geometry.

Part II adds another asymmetry:

    K participates in deciding WHAT to read.
    V matters after the read set has been chosen.

That suggests a potentially powerful memory split.

### 10.1 Candidate architecture

Keep a compact selection representation of K hot:

    query
      ↓
    compact K / index state
      ↓
    selected logical positions
      ↓
    fetch/promote corresponding higher-fidelity K/V
      ↓
    attention

The compact selection state may be:

    model-native indexer keys
    page bounds
    low-rank key sketches
    quantized key summaries
    learned or non-learned selector state

Meanwhile bulky V records can live in a colder tier until selected.

### 10.2 Why this is interesting

HiSparse already demonstrates the general systems pattern:

    compact selection state stays resident
    bulky KV follows demand

Part I provides the missing Gesso research question:

> How aggressively can the selection representation itself be compressed while
> preserving the attention decisions that matter?

This creates a new target metric beyond coordinate reconstruction.

### 10.3 Selection-stability oracle

Candidate experimental metric:

    top-k overlap
    attention-mass recall
    rank inversions near the selection boundary
    score-margin violations
    selected-set divergence

Call this provisionally:

    RUNG 1S — selection stability

It does **not** replace Part I's oracle ladder.

It is an intermediate search oracle for selector representations.

Final promotion still requires logit and task-level certification.

---

## 11. NEW GESSO DIRECTION: ATTENTION HEAT

**Status: GESSO_HYPOTHESIS**

Part I's age hierarchy currently uses conceptual hot/warm/cold state.

Sparse-attention systems reveal a richer signal than wall-clock age:

> **actual read demand.**

A token from 30,000 turns ago may suddenly become hot because the current query
selects it repeatedly.

A token from 50 turns ago may never be consulted again.

Therefore Gesso should investigate a temperature function such as:

    heat(span/page/layer)
        =
    f(
        recency,
        selection frequency,
        attention mass,
        semantic class,
        branch sharing,
        reconstruction cost
    )

This can drive:

    residency
    precision
    compression effort
    retention
    prefetch priority

### 11.1 Promotion by use, not merely age

Example:

    cold host page
        ↓ selected
    warm device page
        ↓ repeatedly selected
    hot resident page

Likewise:

    hot page
        ↓ no demand
    warm
        ↓
    compressed / host

### 11.2 Semantic constraints remain above the physical heuristic

NIRA may declare:

    PERMANENT_PREFIX
    TOOL_SCHEMA
    TASK_STATE
    DISCARDABLE

Those classes can impose hard floors/ceilings on Gesso's physical policy.

But the heat model remains Gesso's runtime mechanism.

---

## 12. NEW GESSO DIRECTION: SELECTION-AWARE REPRESENTATION

**Status: GESSO_HYPOTHESIS**

Part I optimizes representation against attention damage.

Part II suggests representation can also be optimized against **selection
behavior**.

A compressed key that is numerically imperfect may still be excellent if:

    the same important positions are selected
    score ordering near top-k remains stable
    attention mass of retained positions is preserved

Conversely, tiny numeric error can be disastrous if it swaps one critical token
across the top-k boundary.

Therefore candidate KV codecs should eventually report two families of damage:

    REPRESENTATION DAMAGE
        score/output/logit/task distortion

    ACCESS DAMAGE
        changed selected set
        missed attention mass
        altered residency demand
        changed gather pattern

The second category matters only when a compressed representation participates
in selection.

This is exactly the kind of cross-half interaction Part II exists to expose.

---

## 13. NEW GESSO DIRECTION: DYNAMIC GRANULARITY

**Status: GESSO_HYPOTHESIS**

Token-, block- and page-level selection should not necessarily be global model
constants.

Possible policy:

    light load / ample bandwidth:
        finer token retrieval

    high batch pressure:
        page/block retrieval for regularity

    highly local query:
        recent window + small global budget

    retrieval-heavy query:
        larger global budget

    cold host tier:
        page-level prefetch, token-level consume

This suggests a two-stage retrieval plan:

    coarse physical fetch
        ↓
    fine logical selection inside fetched region

The semantic selection granularity and the movement granularity do not have to
match.

That may be the route to keeping token-level retrieval quality without forcing
random tiny PCIe transfers.

---

## 14. NEW GESSO DIRECTION: BRANCH-AWARE MEMORY EXECUTION

**Status: GESSO_HYPOTHESIS**

Part I already gives Cyan/NIRA cheap structural branching through identity
sharing and copy-on-write.

Part II can exploit branch structure during execution.

Suppose:

    shared prefix
      ├── branch A
      ├── branch B
      └── branch C

The branches may also share:

    hot prefix pages
    selection/index state
    prefill products
    immutable compressed backing
    residency hints

until their query streams diverge.

Potential benefit:

    tree-of-thought / red-team / planning branches

become not merely cheaper to STORE but cheaper to KEEP WARM.

The runtime sees a tree.

A generic request server sees unrelated sequences.

That remains one of Gesso's strongest system-level advantages to investigate.

---

## 15. MODEL-NATIVE VS RUNTIME-NATIVE MECHANISMS

This distinction matters because it determines legality, not because we care
about taxonomy for its own sake.

### Class A — exact runtime mechanisms

Can preserve model outputs without changing the checkpoint's learned
architecture.

Examples:

    paged KV
    prefix identity sharing
    copy-on-write
    exact residency movement
    exact rematerialization

### Class B — approximate runtime mechanisms

Can be attempted on an existing model under explicit approximation contracts.

Examples:

    KV quantization
    low-rank approximation
    pruning/eviction
    training-free sparse access
    approximate selector summaries

### Class C — model-native mechanisms

Require the model architecture/training to define them lawfully.

Examples:

    YOCO-style decoder-decoder topology
    HySparse-style architectural KV reuse
    HySparse2-style KV bridging
    native learned sparse indexers

### Class D — co-designed model/runtime mechanisms

The long-term frontier.

The model exposes topology or selection properties specifically so Gesso can
lower them efficiently.

Examples:

    model-declared reuse groups
    model-declared prefill cut points
    shared selector groups
    residency-aware sparse interfaces
    Gesso-native attention-memory layouts

### 15.1 BYOB remains intact

An ordinary Llama/Qwen/Gemma-like battery should still run.

Gesso reads what the model lawfully exposes and chooses from the legal subset of
the search space.

A future Gesso-native battery may expose much richer topology.

Same runtime philosophy.

Different lawful plans.

---

## 16. THE ATTENTION-MEMORY COMPILER

The complete conceptual pipeline becomes:

    MODEL
      +
    LOGICAL CONTEXT
      +
    NIRA MEMORY CONSTRAINTS
      +
    APPROXIMATION POLICY
      +
    HARDWARE / WORKLOAD FINGERPRINT
              │
              ▼
    ┌───────────────────────────────┐
    │      GESSO MEMORY COMPILER    │
    │                               │
    │  representation candidates   │
    │  sharing candidates          │
    │  topology declaration        │
    │  selection candidates        │
    │  residency candidates        │
    │  movement schedules          │
    │  prefill plan                │
    │  decode working-set plan     │
    └──────────────┬────────────────┘
                   │
                   ▼
            legality gates
                   │
                   ▼
            quality oracles
                   │
                   ▼
         measured execution cost
                   │
                   ▼
       winner + receipt + cache
                   │
                   ▼
                 LAVA
                   │
                   ▼
               HARDWARE

The planner does not invent model semantics.

It searches physical realizations of declared/legal semantics.

---

## 17. JOINT OBJECTIVE

Part I's objective already measures memory, runtime, quality and irregularity.

Part II suggests making movement and residency explicit.

A future planner objective may resemble:

    minimize J(π) =
        λ_H · HBM_bytes(π)
      + λ_R · backing_bytes(π)
      + λ_L · latency(π)
      + λ_P · prefill_cost(π)
      + λ_IO· bytes_moved(π)
      + λ_C · irregularity(π)

    subject to:
        attention_damage(π) ≤ ε
        task_quality(π)     ≥ Q_min
        legality(model, π)
        semantic_constraints(NIRA, π)

The exact scalarization is not canon.

The architectural point is:

> A memory plan that saves bytes by exploding PCIe traffic is not a win.

> A sparse plan that saves FLOPs by destroying batching is not a win.

> A codec that preserves vectors but changes important retrieval is not a win.

> A residency plan that is exact but stalls every decode layer is not a win.

Only end-to-end measurement can rank the composition.

---

## 18. AGENT TRACES ARE THE REAL WORKLOAD UNIT

Kernel microbenchmarks remain necessary.

They are not sufficient for this program.

Gesso's target workload includes long-lived agents where context structure is
highly nonuniform.

Representative trace families should include:

### 18.1 Engineer trace

    inspect repository
    run build
    ingest 20k–100k token logs
    patch
    run tests
    retrieve an error from many turns ago
    repeat

### 18.2 Research trace

    accumulate papers/notes
    branch hypotheses
    revisit early evidence
    synthesize across distant turns
    maintain stable doctrine prefix

### 18.3 Red-team trace

    fork many branches
    share common prefix
    diverge
    retain branch-local evidence
    compare results

### 18.4 WGE trace

    repeated generation receipts
    large tool observations
    world/asset state
    long build pipelines
    sparse retrieval of one old constraint

For each trace record:

    input tokens
    output tokens
    observation/action ratio
    selection traces if available
    branch structure
    prefix-sharing structure
    semantic classes
    context age distribution
    repeated-access locality

The memory planner should be calibrated on actual workload families, not a
single synthetic context length.

---

## 19. EXPERIMENT LADDER — PART II

The program should earn complexity in layers.

### E0 — dense baseline

Measure:

    normal full KV
    dense attention
    no offload
    no compression beyond model default

Collect:

    TTFT
    prefill time
    decode tok/s
    HBM
    host memory
    bytes moved
    batch throughput
    quality

### E1 — exact structural memory

Add:

    paged KV
    copy-on-write
    identity prefix sharing

No quality risk.

This is shared with Part I.

### E2 — exact hierarchical residency

On a sparse-capable path:

    full logical history backed outside HBM
    bounded GPU working set
    selection-driven fetch
    exact values

Goal:

    prove AVAILABILITY ≠ RESIDENCY in Gesso's own runtime.

### E3 — selection granularity sweep

Compare:

    token
    block
    page
    hybrid

under the same attention budget where possible.

Measure both:

    task quality
    batchability / gather cost

### E4 — selection locality characterization

Before inventing a replacement policy, record real traces:

    inter-step overlap
    cross-layer overlap
    page locality
    hot-set lifetime
    branch overlap

Then test:

    LRU
    frequency-aware
    attention-heat policies
    oracle upper bound

### E5 — K/V asymmetric residency

Test whether:

    compact K/index state stays hot
    selected V (and possibly high-fidelity K) moves on demand

Measure:

    selector fidelity
    traffic
    miss latency
    quality
    memory

### E6 — selection-aware compression

Compress selector state and/or backing KV.

Add experimental RUNG 1S metrics:

    top-k overlap
    attention-mass recall
    boundary rank inversions

Finalists still pass Part I rungs 3–4.

### E7 — age × demand × semantic class

Compare policies based on:

    recency only
    access heat only
    semantic class only
    combined policy

Look for:

    retention quality
    host traffic
    working-set stability
    pressure recovery

### E8 — joint Part I + Part II composition

Example candidate:

    SharedPrefix
      → PageLowRank
      → AttentionAwareTransform
      → VQ
      → SparseResidual
      → SelectorIndex
      → TieredBacking
      → BoundedHBMWorkingSet
      → DemandPrefetch

Ablate each stage.

The research result is the interaction matrix.

### E9 — model-native topology track

Using architecture-appropriate checkpoints:

    conventional dense model
    cross-layer-sharing model
    YOCO-class model
    HySparse-class model
    HySparse2-class model

Do **not** pretend these are the same checkpoint.

The question is not a pure runtime A/B.

The question is whether Gesso's topology/memory abstractions can faithfully and
efficiently express each architecture.

This is the torture test for the IR.

---

## 20. THE JACKPOT COMPOSITION EXPERIMENT

The long-term experiment should not ask whether one paper trick wins.

It should ask whether Gesso's **joint planner** finds a better operating point
than independent subsystems.

For a fixed model, hardware and agent trace where runtime techniques are legal:

    A  baseline memory/runtime

    B  representation-only optimization
       (Part I)

    C  access/residency-only optimization
       (Part II)

    D  jointly searched Gesso attention-memory plan
       (Part I × Part II)

Measure:

    task quality
    effective context retention
    TTFT
    prefill time
    decode throughput
    HBM
    host memory
    host↔device bytes
    attention work
    selector work
    branch concurrency
    batchability
    pressure recovery

The research thesis is not that D must win.

The thesis is that **joint optimization exposes interactions that isolated
feature tuning cannot see**.

If it does not, that is equally valuable evidence.

---

## 21. THREATS TO VALIDITY — PART II

### T1 — architecture dependence

YOCO/HySparse/HySparse2 mechanisms are model-architectural.

Runtime support must not silently reinterpret an arbitrary checkpoint.

### T2 — irregular selection

Token-level top-k can improve retrieval quality while producing hostile memory
access patterns.

Quality and kernel efficiency must be measured together.

### T3 — host-link saturation

Bounded HBM is not free memory.

Poor locality converts capacity savings into PCIe/NVLink/CXL stalls.

### T4 — cache thrash

A working set that changes faster than it can be serviced may perform worse
than full residency.

### T5 — selector damage

A compressed or approximate selector can retrieve the wrong state even when
the backing KV is perfect.

### T6 — representation/access interference

A transform that improves Part I compression may destroy the geometry used by a
Part II selector.

The reverse can also happen.

### T7 — stale heat

Past selection frequency may be a terrible predictor after a task shift.

Runtime heat is a hint, not semantic truth.

### T8 — agent-trace bias

A policy tuned on compiler logs may fail on document synthesis, game-world
state or code navigation.

Receipts must name the trace family.

### T9 — prefill/decode imbalance

A technique that makes decode beautiful may add unacceptable prefill work, or
vice versa.

Agent workloads can be heavily prefill-dominated.

### T10 — model-training confound

Model-native architecture comparisons conflate execution design with training
dynamics.

Do not turn them into unsupported runtime claims.

### T11 — unified-memory mirage

Hardware with coherent CPU/GPU memory can hide movement mechanics that become
painful on discrete GPUs.

Hardware fingerprints are mandatory.

### T12 — semantic leakage

Gesso may use NIRA-provided classes and constraints.

It must not quietly become the subsystem that decides what the user's memories
mean.

The ownership seam remains binding.

---

## 22. OWNERSHIP SEAM — UNCHANGED, NOW MORE USEFUL

The boundary from Part I remains:

    NIRA decides what memory means.
    Gesso decides how that memory lives and participates in computation.

Part II makes "participates in computation" concrete.

NIRA may say:

    this is doctrine
    this is task state
    this is reconstructible
    this is disposable
    these branches share this prefix

Gesso may decide:

    share these pages physically
    keep this class above a fidelity floor
    keep this selector state in HBM
    move these KV records to host
    prefetch these selected pages
    use this legal sparse kernel
    use this legal model-declared reuse group
    compress this cold backing representation
    rematerialize this state on miss

NIRA does not manipulate KV tensors.

Gesso does not invent semantic understanding.

And the merge rule remains:

> **Diverged semantic branches merge in NIRA's structured space, then re-enter
> Gesso through prefill.**

No raw-KV Frankenstein merges.

---

## 23. THE EMERGING FINAL FORM

The Magenta Memory stack now has a coherent shape.

```text
                       NIRA / CYAN
              meaning · policy · branch intent
                         │
                         ▼
                   semantic constraints
                         │
═════════════════════════╪═════════════════════════
                     GESSO SEAM
                         │
                         ▼
              LOGICAL ATTENTION MEMORY
                         │
          ┌──────────────┼───────────────┐
          │              │               │
          ▼              ▼               ▼
    REPRESENTATION    TOPOLOGY       SELECTION
      Part I           Part II        Part II
          │              │               │
          └───────┬──────┴───────┬───────┘
                  │              │
                  ▼              ▼
             RESIDENCY        SCHEDULE
               Part II          Part II
                  │              │
                  └──────┬───────┘
                         ▼
                ATTENTION MEMORY PLAN
                         │
               search · verify · measure
                         │
                         ▼
                        LAVA
                         │
                         ▼
                     HARDWARE
```

Part I answers:

> **What physical representation may carry this attention state?**

Part II answers:

> **Who needs that state, which pieces are read, where must they be, and when?**

The compiler answers both at once.

---

## 24. IMMEDIATE R&D DIRECTIONS

These are whiteboard directions, not implementation commitments.

### 24.1 Define the conceptual `AttentionMemoryPlan`

Before code, make sure the vocabulary can represent:

    dense per-layer KV
    GQA/MQA
    shared pages
    cross-layer KV reuse
    hidden-state bridges
    selection groups
    token/block/page selection
    forced local sets
    bounded device residency
    host backing
    prefetch
    representation pipelines
    approximation contracts

If one of the source architectures cannot be expressed without a special-case
escape hatch, the abstraction is not ready.

### 24.2 Add access-trace hooks early

When the paged KV manager lands, preserve the ability to record:

    logical reads
    page reads
    layer
    request/branch
    selector output
    hit/miss
    migration
    reuse distance

Those traces become the data source for later residency and heat research.

### 24.3 Build an agent-trace replay harness

The same recorded trace should be replayable against candidate:

    page sizes
    cache sizes
    replacement policies
    selection granularities
    prefetch policies

without rerunning the whole interactive agent every time.

### 24.4 Add movement to receipts

Every candidate receipt should eventually include:

    HBM bytes
    host bytes
    bytes promoted
    bytes demoted
    bytes reconstructed
    cache hit rate
    miss service time
    prefetch accuracy
    batch throughput

### 24.5 Prototype RUNG 1S

Before building exotic selectors, determine whether:

    selected-set overlap
    attention-mass recall
    score-margin violations

actually correlate with rungs 3–4 on representative models.

If they do not, kill the proxy.

### 24.6 Investigate K/V asymmetry

This is the strongest new synthesis from Part I × Part II:

> **selection state and value state may deserve different representations,
> residency policies and movement schedules.**

Test it.

### 24.7 Keep model-native work on a separate research rail

Gesso should support weird model topologies now at the IR level.

Actual Gesso-native model architecture can come later.

Do not make BYOB support wait for the research model.

---

## 25. PRIMARY SOURCE LEDGER

Primary sources inspected for this synthesis on 2026-09-29.

### YOCO

Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma,
Quanlu Zhang, Jianyong Wang, Furu Wei.

**You Only Cache Once: Decoder-Decoder Architectures for Language Models.**

arXiv:2405.05254

https://arxiv.org/abs/2405.05254

Mechanisms used here:

    decoder-decoder split
    shared global KV
    prefill early exit
    model-native inference memory reduction

### HySparse

Yizhao Gao, Jianyu Wei, Qihao Zhang, Yu Cheng, Shimao Chen, Zhengju Tang,
Zihan Jiang, Yifan Song, Hailin Zhang, Liang Zhao, Bo Yang, Gang Wang,
Shijie Cao, Fuli Luo.

**HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection
and KV Cache Sharing.**

arXiv:2602.03560

https://arxiv.org/abs/2602.03560

Mechanisms used here:

    full-attention layer as selector
    cross-layer salient-token reuse
    cross-layer KV reuse
    hybrid full/sparse architecture

### HiSparse

Zhiqiang Xie, Zhangheng Huang, Tingwei Huang, Ziyi Xu, Ruiyang Ma,
Christos Kozyrakis.

**HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache
Management.**

arXiv:2608.07009

https://arxiv.org/abs/2608.07009

Mechanisms used here:

    availability/residency separation
    bounded GPU working set
    host-backed full KV
    selector-driven demand
    selection locality
    exact prefetch / miss resolution

### HySparse2

Jianyu Wei, Yizhao Gao, Qihao Zhang, Shimao Chen, Zhengju Tang, Yu Cheng,
Shengjie Zhou, Zihan Jiang, Yifan Song, Hailin Zhang, Liang Zhao, Bo Yang,
Gang Wang, Shijie Cao, Fuli Luo.

**HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing.**

arXiv:2609.26368

https://arxiv.org/abs/2609.26368

Mechanisms used here:

    full-attention-only KV bridging
    KV reuse
    token-level sparse selection
    forced recent window
    self-decoder/cross-decoder split
    early-exit prefill
    agent-oriented long-context evaluation

---

## 26. WHAT THIS DOCUMENT IS NOT

    Not a demand to implement four papers.

    Not a claim that one attention architecture is Gesso's final form.

    Not permission to mutate arbitrary checkpoints into architectures
    they were not trained to be.

    Not a replacement for Part I's representation research.

    Not an excuse to optimize microbenchmarks while agent traces regress.

    Not canon-overriding.

This is the second half of the R&D map.

Its purpose is to make the design space explicit enough that Gesso can search,
measure and eventually discover its own final form.

---

## 27. ONE-SENTENCE PROGRAM

> **Gesso should compile logical attention memory into the cheapest verified
> combination of representation, sharing, topology, selection, residency and
> movement that the model, hardware, workload and declared quality contract
> lawfully permit.**
