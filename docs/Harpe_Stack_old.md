```text
HARPE — SEMANTIC JULIA ML STACK

MISSION
=======

Build a Julia-native machine-learning stack that can eventually compete with the conventional
PyTorch/CUDA ecosystem on ordinary modern ML workloads while exploiting architectural capabilities
that become difficult, brittle, or research-project-level work in conventional tensor-first stacks.

Harpe is NOT intended to begin as an exotic-model project.

The initial obligation is straightforward:

    If the conventional PyStack can load, train, infer, quantize, and serve an ordinary
    modern open-weight model, Harpe should have a principled route to doing the same.

Examples include contemporary transformer, MoE, hybrid-attention, recurrent/state-space, and
future architectures.

Only after this ordinary path is correct, complete, measurable, and fast should Harpe aggressively
exploit the deeper capabilities that motivate its existence.

The long-term thesis is larger than:

    "Julia can make PyTorch faster."

The emerging thesis is:

    Preserve semantic intent deeper into the ML stack so that representation, compilation,
    execution, memory layout, precision, kernel selection, and runtime policy can exploit
    information that conventional flattened tensor graphs frequently discard or must later
    rediscover.

In short:

    THE MODEL SHOULD NOT LOSE ITS MEANING BEFORE IT REACHES THE MACHINE.


=======================================================================
I. FOUNDATIONAL DESIGN LAW
=======================================================================

Never discard semantic information earlier than necessary.

Conventional ML systems often converge toward:

    model
      ↓
    tensor operations
      ↓
    graph
      ↓
    compiler
      ↓
    kernels

This is extremely successful, but by the time optimization happens much of the original meaning
may have been reduced to generic tensor operations.

Harpe should instead preserve a richer path:

    model semantics
          ↓
    semantic operators / parameters
          ↓
    representation planning
          ↓
    execution planning
          ↓
    specialization / lowering
          ↓
    generated or selected kernels
          ↓
    Lava / hardware backend

Meaning should survive as far down this pipeline as it remains useful.


=======================================================================
II. DEVELOPMENT ORDER
=======================================================================

The project should obey this progression:

    MAKE IT WORK
        ↓
    MAKE IT COMPLETE
        ↓
    MAKE IT MEASURABLE
        ↓
    MAKE IT FAST
        ↓
    MAKE IT WEIRD
        ↓
    MAKE THE WEIRDNESS FAST

Do not add exotic machinery before the ordinary path works.

However:

    Never design the ordinary path in a way that prevents exotic machinery later.

That distinction is critical.


=======================================================================
III. BASELINE COMPATIBILITY TARGET
=======================================================================

Harpe is an ML stack, not a transformer implementation.

Transformers are merely the first demanding compatibility target.

Do NOT build:

    QwenKernel
    LlamaKernel
    ModelXKernel

Instead, build a vocabulary of reusable architecture semantics and kernel families.

Conceptual architecture description:

    Model
      │
      ├── embedding policy
      ├── normalization family
      ├── positional encoding
      ├── attention family
      │      ├── MHA
      │      ├── MQA
      │      ├── GQA
      │      ├── sliding-window
      │      ├── local/global hybrids
      │      ├── compressed/latent attention
      │      └── future variants
      │
      ├── FFN family
      │      ├── dense
      │      ├── SwiGLU
      │      ├── gated variants
      │      └── MoE
      │
      ├── residual topology
      ├── routing policy
      ├── cache semantics
      ├── recurrent/state-space blocks
      └── output/head policy
               ↓
       Harpe semantic operator layer
               ↓
       implementation specialization
               ↓
              Lava

Supporting a new architecture should ideally mean describing how it composes existing semantics,
not recreating an entirely new runtime.


=======================================================================
IV. MODEL IMPORT / INTERCHANGE
=======================================================================

Weight format must remain independent from execution.

Target conceptual path:

    HF config
    tokenizer
    safetensors / equivalent checkpoint
            ↓
       architecture importer
            ↓
    canonical Harpe model semantics
            +
       parameter mapping
            ↓
       Harpe execution stack

The runtime should not fundamentally care whether parameters originated from:

    Hugging Face
    Lux
    Flux
    Harpe training
    another compatible exporter

Checkpoint format is transport.

Execution semantics are separate.


=======================================================================
V. CORRECTNESS BEFORE HEROICS
=======================================================================

Every primitive and architecture family needs a correctness oracle.

For supported models:

    original/reference model
            ↓
      reference logits
            ↕
      numerical comparison
            ↕
        Harpe logits
            ↓
      generation parity
            ↓
    performance benchmark

Primitive testing should include:

    CPU/reference oracle
    gradient checks
    dtype coverage
    strange shapes
    numerical tolerances
    deterministic regression
    invalid-input behavior
    serialization round trips
    forward/backward parity where applicable

Agents should not decide whether code "looks right."

The harness decides.


=======================================================================
VI. PERFORMANCE OBSERVABILITY
=======================================================================

Before aggressive optimization Harpe must make performance explainable.

Measure at minimum:

    kernel latency
    launch overhead
    occupancy
    memory bandwidth
    allocation count
    synchronization points
    compilation latency
    peak VRAM
    transfer cost
    tokens/sec
    time-to-first-token
    decode latency
    prefill latency
    training step time
    optimizer step time
    communication overhead

Every important performance gap should eventually be classifiable as something concrete:

    poor kernel
    bad memory layout
    launch overhead
    synchronization
    compiler issue
    runtime scheduling
    missing fusion
    bad algorithm
    poor device placement
    unnecessary representation conversion
    communication bottleneck

"No idea why it is slower" is not an acceptable steady state.


=======================================================================
VII. PYTHON/PYTORCH STACK PARITY CAMPAIGN
=======================================================================

For representative workloads compare:

    PyTorch eager
    torch.compile
    strong CUDA implementation
    Julia CUDA / Reactant control
    Harpe + Lava

CUDA should remain a control where appropriate.

The goal is to distinguish:

    Harpe architecture problem

from:

    immature Vulkan/Lava kernel

from:

    hardware disadvantage

from:

    algorithmic difference

Aspirationally, after serious kernel tuning and runtime maturation, Harpe should approach the
performance envelope of mature conventional implementations on workloads where hardware support
permits it.

Do not assume parity.

Measure it.


=======================================================================
VIII. LAVA ML KERNEL PROGRAM
=======================================================================

The ordinary ML hot path will eventually require strong implementations of families including:

    GEMM / GEMV
    reductions
    softmax
    RMSNorm / LayerNorm
    RoPE and positional transforms
    activation functions
    fused gated FFNs
    attention
    KV-cache operations
    quantize/dequantize
    sampling primitives
    routing primitives
    MoE token grouping
    expert GEMM
    aggregation
    embedding lookup
    optimizer kernels

Attention should be treated as a parameterized FAMILY, not a monolithic implementation.

Conceptually:

    attention(
        query_layout,
        kv_layout,
        causal_policy,
        window_policy,
        head_grouping,
        cache_policy,
        precision,
        workload,
        device
    )

Dispatch and/or code generation should select an appropriate implementation.


=======================================================================
IX. COMPLETE TRAINING SYSTEM
=======================================================================

Eventually support:

    FP32 / BF16 / FP16 where hardware permits
    mixed precision
    gradient accumulation
    appropriate loss scaling
    activation checkpointing
    optimizer states
    reproducibility
    deterministic options
    checkpoint save/resume
    distributed loading
    sharding
    collectives
    multi-device execution

A real training system must survive interruption and resume correctly.

"model(x) works" is not sufficient.


=======================================================================
X. COMPLETE INFERENCE SYSTEM
=======================================================================

Inference eventually needs:

    prefill/decode separation
    KV-cache management
    paged or equivalent cache strategies
    continuous batching
    prefix caching
    request scheduling
    streaming
    sampling
    quantized execution
    memory-pressure policies
    multi-device execution
    workload-specific specialization

Again:

    not just model(x)

but an actual serving runtime.


=======================================================================
XI. THE HEAVYWEIGHT IDEAS
=======================================================================

The most important conceptual discoveries from the current design discussion are three closely
related capabilities.

-----------------------------------------------------------------------
A. SEMANTIC MODEL MATERIALIZATION
-----------------------------------------------------------------------

The logical model is NOT identical to its physical representation.

Harpe should eventually be capable of answering:

    Given:
        this logical model
        this hardware
        this workload
        these latency/memory/error constraints

    how should this model physically exist?

The same logical model might be materialized differently for:

    training
    long-context prefill
    single-user decode
    high-throughput batched decode
    memory-constrained inference
    another GPU architecture

Possible decisions include:

    layout
    precision
    quantization
    packing
    sparsity
    fusion
    cache representation
    device placement
    parameter residency
    tiling
    kernel family
    routing strategy

Conceptual separation:

    logical model
          ≠
    physical model realization

This separation may become one of Harpe's defining architectural boundaries.


-----------------------------------------------------------------------
B. SEMANTIC EXECUTION SYNTHESIS
-----------------------------------------------------------------------

Harpe should not necessarily treat execution as one predetermined graph.

Given preserved model semantics, the system may eventually construct an execution strategy.

Possible decisions:

    fusion
    staging
    partial evaluation
    speculation
    scheduling
    asynchronous overlap
    cache strategy
    memory placement
    quantized path
    device partitioning
    numerical strategy

Conceptually:

    logical computation
           ↓
    set of legal execution plans
           ↓
    policy / compiler search
           ↓
    selected executable plan

This turns the stack from "a bag of kernels" into something closer to an ML compiler/runtime.


-----------------------------------------------------------------------
C. PROFILE-GUIDED REMATERIALIZATION
-----------------------------------------------------------------------

Execution can feed measurements back into compilation/materialization.

Conceptually:

    model semantics
          ↓
    current materialization
          ↓
    execution
          ↓
    profiling
          ↓
    identify stable bottlenecks / workload properties
          ↓
    construct improved realization
          ↓
    benchmark / verify
          ↓
    retain better realization

Examples:

    attention configuration dominates latency
        → specialize or generate a kernel

    certain weights are immutable during inference
        → repack them

    workload becomes predominantly long-context
        → change cache/layout policy

    expert remains cold
        → compress, evict, or alter residency

    hardware prefers a different tile
        → generate/select another implementation

This is NOT primarily the learned model changing itself.

The stack changes how the same logical model physically exists.

This concept is one of the most important discoveries so far.


=======================================================================
XII. GENERAL-PURPOSE SPECULATIVE EXECUTION / DECODING
=======================================================================

One research direction is to explore whether semantic execution synthesis can enable more general
speculative decoding or speculative execution strategies.

The important idea is NOT merely reproducing today's draft-model speculative decoding.

The question is:

    If Harpe understands model semantics and execution plans deeply enough,
    can speculation become a runtime/compiler concern rather than a special external architecture?

Potential territory includes:

    reusable speculative execution machinery
    workload-specific speculative plans
    branch/candidate execution
    confidence-aware execution paths
    partial evaluation
    specialized decode programs

This remains research territory.

Do not promise a specific speedup or assume a separate draft model can always be eliminated.

The strategic point is that Harpe's richer execution layer may expose speculative strategies that
are prohibitively awkward to implement generically elsewhere.


=======================================================================
XIII. SEMANTIC PARTIAL EVALUATION
=======================================================================

If facts about a model or workload are known and stable, Harpe may be able to collapse computation
before execution.

Examples of potentially stable facts:

    frozen parameters
    fixed head dimensions
    inference-only operators
    constant masks
    quantization format
    architecture topology
    static routing components
    device characteristics

The compiler can potentially turn:

    general model program

into:

    smaller residual executable program

for a particular execution regime.

This concept connects directly to:

    specialization
    kernel generation
    semantic execution synthesis
    quantization
    materialization


=======================================================================
XIV. WHOLE-MODEL MEMORY SYNTHESIS
=======================================================================

Memory placement should eventually be treated as a compiled plan rather than scattered heuristics.

Potential decisions:

    GPU residency
    host residency
    tiered CPU/GPU storage
    cache eviction
    expert residency
    quantized cold storage
    activation lifetime
    workspace reuse
    recomputation vs storage
    transfer overlap

Eventually Harpe could take a policy like:

    minimize latency under 16 GiB VRAM

or:

    maximize throughput under a given memory budget

and construct a legal materialization/execution plan.

This is a long-term target, not an initial requirement.


=======================================================================
XV. NUMERICAL STRATEGY AS COMPILATION
=======================================================================

Precision should not necessarily be permanently encoded into the model definition.

Potential future lowering decisions include:

    FP32
    BF16
    FP16
    FP8 where appropriate
    INT8
    INT4
    mixed precision
    approximations
    sparse representations

under explicit accuracy/error constraints.

In this framework:

    precision is a lowering decision
    quantization is a lowering decision
    sharding is a lowering decision
    kernel fusion is a lowering decision
    KV representation is a lowering decision
    device placement is a lowering decision

This is the conceptual connection that makes quantization especially interesting.


=======================================================================
XVI. QUANTIZATION AS A FIRST-CLASS REPRESENTATION PROBLEM
=======================================================================

Do NOT conceptualize quantization merely as:

    train model
        ↓
    convert checkpoint
        ↓
    hope operators support it

Instead:

    logical parameter
        │
        ├── semantic role
        ├── sensitivity constraints
        ├── update policy
        ├── error budget
        └── workload context
                ↓
        representation planner
                ↓
        BF16 / FP8 / INT8 / INT4 /
        sparse / packed / mixed
                ↓
        matching execution path

Potentially, different regions of one logical model may use different physical representations.

Representation may also differ between execution phases while preserving logical equivalence.

Example:

    training
        → BF16

    long-context prefill
        → another optimized representation

    single-token decode
        → aggressively packed/quantized realization

This makes quantization part of materialization rather than a detached conversion subsystem.


=======================================================================
XVII. SEMANTIC TENSORS
=======================================================================

Another major idea is to stop treating every tensor as merely:

    Tensor{Float16}

A Harpe tensor or parameter may preserve semantic identity.

Examples:

    Parameter
    FrozenParameter
    QuantizedWeight
    KVCache
    Activation
    GradientAccumulator
    EmbeddingTable
    MoEExpertWeight
    OptimizerState
    TemporaryWorkspace
    ProjectionWeight
    DecodeOnlyState

Important distinction:

    Array{Float16}
        = storage description

    KVCache{Float16, ...}
        = semantic description

The tensor can carry enough structured meaning for dispatch, compilation, representation planning,
and execution to make better decisions.

This should NOT mean stuffing arbitrary metadata into every scalar.

The primary goal is preserving meaningful semantics at the tensor/parameter/operator level.


=======================================================================
XVIII. TYPE-DIRECTED EXECUTION
=======================================================================

Julia's multiple dispatch is unusually well matched to ML execution because kernel choice is often
a function of interactions among many properties:

    operation
      × dtype
      × representation
      × layout
      × device
      × model role
      × workload
      × execution phase

Conceptually:

    mul!(y, A, x)

can resolve very differently when:

    A = frozen INT4 projection
    x = BF16 activation
    workload = batch-1 decode
    device = Vulkan GPU

versus:

    A = trainable BF16 matrix
    x = batched training activation
    workload = backward pass
    device = another architecture

The key advantage of multiple dispatch is that it naturally answers:

    "What algorithm is appropriate for this interaction between these things?"

instead of forcing a giant object hierarchy or one operator containing hundreds of branches.


=======================================================================
XIX. SEMANTIC OPERATOR DISPATCH
=======================================================================

Operators should also preserve meaning.

Conceptually:

    attention(
        q::QTensor,
        k::KTensor,
        v::VTensor,
        cache::PagedKV,
        ::GQA,
        ::Causal,
        ::Decode,
        ::VulkanDevice
    )

This does NOT imply that every property belongs in a concrete Julia type.

Rather, the operator system should expose enough structured information to choose or generate
specialized implementations.

This can potentially make model architecture semantics directly useful to kernel selection.


=======================================================================
XX. METAPROGRAMMING / GENERATED KERNEL FAMILIES
=======================================================================

Do not hand-write an endless matrix of:

    GEMM_INT8_GQA_DECODE_HEAD128
    GEMM_INT4_GQA_DECODE_HEAD128
    GEMM_BF16_GQA_PREFILL_HEAD128
    ...

Instead, long-term Harpe should explore generating implementation families from semantic traits.

Conceptually:

    semantic operator
          +
    representation
          +
    workload
          +
    hardware
          ↓
    implementation generator
          ↓
    candidate kernel(s)
          ↓
    benchmark/autotune
          ↓
    retained specialization

Potential pipeline:

    semantic tensors
          ↓
    multiple dispatch
          ↓
    operator traits
          ↓
    legal transformations
          ↓
    generated implementation
          ↓
    Lava kernel
          ↓
    autotuning
          ↓
    cached specialization


=======================================================================
XXI. LOGICAL VS PHYSICAL TENSOR REPRESENTATION
=======================================================================

One logical parameter may have multiple valid physical realizations.

Conceptually:

    materialize(weight, hardware, workload, policy)

might produce:

    DenseBF16
    PackedINT8
    PackedINT4
    sparse representation
    blocked FP8
    transposed packed storage
    GPU-resident form
    tiered CPU/GPU form

The logical model should not care which representation is chosen so long as the representation
satisfies its semantic contract.

This separation is potentially foundational to:

    quantization
    rematerialization
    hardware portability
    runtime specialization
    memory synthesis


=======================================================================
XXII. OPERATOR CAPABILITIES AND PLAN SEARCH
=======================================================================

Harpe should eventually be able to describe what operators/kernels support.

Conceptually:

    supports(
        FlashAttention(),
        PackedKV{Int8},
        DeviceFamily()
    )

Then compilation can search among LEGAL plans.

Example:

    Plan A:
        unpack INT4 → BF16
        run kernel A

    Plan B:
        run native INT4 kernel B

    Plan C:
        rematerialize to INT8
        run kernel C

Then:

    verify legality
    benchmark candidates
    select winner
    cache result

Multiple dispatch can therefore become more than a language convenience.

It can participate in defining a compiler search space.


=======================================================================
XXIII. SPECIALIZATION DISCIPLINE
=======================================================================

CRITICAL WARNING:

Do NOT encode every dynamic property in Julia's type system.

Doing so risks catastrophic specialization and compile-time explosion.

Harpe should distinguish approximately:

    TYPES
        stable structural semantics

    TRAITS
        optimization-relevant properties

    RUNTIME METADATA
        genuinely dynamic information

Potential examples:

TYPE-LEVEL:

    device family
    storage representation family
    tensor semantic class
    layout family

TRAIT-LEVEL:

    frozen
    quantized
    contiguous
    cooperative-matrix capable
    inference-only

RUNTIME:

    current sequence length
    active request count
    batch size
    available memory
    queue depth

A runtime fact should only become a specialization dimension when evidence suggests that
specializing on it is valuable.


=======================================================================
XXIV. PROFILE-GUIDED SPECIALIZATION
=======================================================================

The previous section creates another closed loop:

    observe workload
          ↓
    identify stable facts
          ↓
    decide which facts justify specialization
          ↓
    generate/select optimized implementation
          ↓
    benchmark
          ↓
    retain winner

This connects:

    semantic tensors
    multiple dispatch
    metaprogramming
    materialization
    execution synthesis
    profiling
    autotuning

into one coherent system.


=======================================================================
XXV. CLOSED-LOOP HARPE RUNTIME
=======================================================================

The nastiest long-term conceptual picture currently looks like:

    MODEL SEMANTICS
          ↓
    REPRESENTATION CHOICE
          ↓
    EXECUTION-PLAN SYNTHESIS
          ↓
    COMPILER SPECIALIZATION
          ↓
    KERNEL GENERATION / SELECTION
          ↓
    RUNTIME EXECUTION
          ↓
    MEASUREMENT
          ↓
    PROFILE-GUIDED REMATERIALIZATION
          └───────────────────────────────┐
                                          ↓
                                   repeat if useful

Again:

    the learned model need not change.

The system changes how the logical model is materialized and executed.

This may be the single broad concept that unifies most of the "weird Harpe advantages."


=======================================================================
XXVI. NEUROSYMBOLIC WEIGHT / PARAMETER DIRECTION
=======================================================================

A conceptual possibility emerged:

    Parameters may eventually carry richer semantics than numeric value alone.

This begins to resemble a neurosymbolic parameter system because symbolic meaning is not merely
sidecar documentation.

The semantics can actually affect compilation and execution.

However, be precise:

    The immediate Harpe idea is NOT "every scalar weight becomes a symbolic object."

The nearer-term design is:

    logical parameter
        +
    structured semantic properties
        ↓
    compiler/runtime decisions
        ↓
    physical tensor representation

This may later support richer research into genuinely stateful or semantic parameters.

For now, preserve the architectural possibility without turning Harpe v1 into a research-only
weight system.


=======================================================================
XXVII. FLUID / PLASTIC WEIGHTS NOTE
=======================================================================

Fluid/Plastic Weights is explicitly NOT the current priority.

Do not design Harpe around PlasticWeights.

Do not block Harpe v1 on it.

However, preserve extension points for future:

    state-aware parameters
    lifecycle-aware representations
    state-aware GEMM
    representation migration
    mixed arithmetic states
    dynamic packed formats

Research note:

    FUTURE: FluidGEMM / lifecycle-aware matrix execution.

    Preserve the seam.
    Do not implement it prematurely.
    Revisit when PlasticWeights semantics are sufficiently stable.

Potential future insight:

    consolidation may eventually become both a learning event and a compute event.

For example, a parameter region could potentially migrate from expensive mutable representation
toward cheaper packed/frozen representation as its lifecycle changes.

Again:

    FUTURE RESEARCH.
    NOT A V1 DEPENDENCY.


=======================================================================
XXVIII. MODELS MIXING NUMERICAL AND LEARNED MACHINERY
=======================================================================

Julia may provide another long-term advantage by allowing:

    neural modules
    ODE/PDE solvers
    iterative numerical algorithms
    symbolic transformations
    optimization routines
    simulation kernels
    learned components

to participate in one coherent program without forcing an artificial framework boundary.

Different components might use:

    reverse-mode AD
    forward-mode AD
    custom adjoints
    implicit differentiation
    no-grad execution
    surrogate gradients
    solver-aware differentiation

selected according to their actual semantics.

This should remain part of the longer-term research territory.


=======================================================================
XXIX. STRUCTURALLY DYNAMIC ARCHITECTURES
=======================================================================

Harpe should avoid assumptions that make dynamic architectures unnecessarily hostile.

Potential future systems may:

    add/remove experts
    alter routing structures
    freeze/thaw regions
    change representations
    redistribute capacity
    restructure sparse computation
    change device residency
    specialize around workload patterns

The architecture should permit such systems without requiring the entire framework to be rebuilt.

Do not prioritize this ahead of baseline compatibility.

Preserve the seam.


=======================================================================
XXX. HARDWARE-AWARE MODEL MATERIALIZATION
=======================================================================

One logical model may produce materially different executables for different devices.

Potentially varying:

    packing
    quantization
    tile shape
    memory layout
    fusion
    cache strategy
    precision
    dispatch strategy
    kernel family

This should be treated as expected behavior rather than a portability failure.

The model defines the logical computation.

Harpe defines its physical realization.


=======================================================================
XXXI. AUTOTUNED ARCHITECTURE FAMILIES
=======================================================================

Generated implementation families and multiple dispatch create the possibility of device- and
workload-specific autotuning.

Conceptual flow:

    operator semantics
          ↓
    generate legal candidate implementations
          ↓
    run microbenchmarks
          ↓
    validate correctness
          ↓
    choose best candidate
          ↓
    cache by hardware/workload signature

Autotuning should be measured, reproducible, and cacheable.

Never allow benchmark-selected code to silently violate numerical/correctness constraints.


=======================================================================
XXXII. POLICIES AS FIRST-CLASS INPUTS
=======================================================================

Long-term, Harpe could support whole-model execution objectives.

Examples:

    maximize throughput under 24 GiB VRAM

    minimize batch-1 decode latency

    keep selected layers above a specified precision

    minimize memory while maintaining a numerical tolerance

    maximize throughput while preserving deterministic execution

Then the stack could materialize a model/execution plan satisfying those constraints.

This is a long-term compiler/runtime goal.


=======================================================================
XXXIII. WHAT HARPE IS NOT
=======================================================================

Harpe is NOT:

    a thin Julia wrapper around PyTorch

    a collection of random custom Vulkan kernels

    a transformer-only framework

    a PlasticWeights runtime

    a speculative-decoding-only project

    a quantization library

    a benchmark stunt

It may contain or support all of those capabilities, but the intended architecture is broader.


=======================================================================
XXXIV. WHAT MAY DIFFERENTIATE HARPE
=======================================================================

The conventional ecosystem can implement many of these ideas individually.

The important comparison is NOT:

    "Can PyTorch theoretically express this?"

PyTorch is extremely general and extensible.

The meaningful question is:

    "Can Harpe make this an ordinary composition rule of the system where another stack
     would require extensive custom operators, compiler work, metadata plumbing, graph surgery,
     tensor subclasses, framework hooks, and bespoke runtime logic?"

The opportunity is ergonomic + architectural:

    make advanced execution strategies native

instead of:

    make advanced execution strategies heroic.


=======================================================================
XXXV. CURRENT NORTH-STAR STATEMENT
=======================================================================

Harpe is evolving conceptually toward:

    A SEMANTIC COMPILER AND RUNTIME FOR MACHINE LEARNING,
    implemented in Julia and backed by high-performance hardware execution,
    in which models retain enough meaning for the system to make informed decisions about
    representation, precision, memory, execution, specialization, and kernels.

The simplest useful version remains:

    "An ML stack where the model does not lose its meaning before it reaches the machine."


=======================================================================
XXXVI. CURRENT PRIORITY ORDER
=======================================================================

DO NOW:

    1. Define stack contracts and ownership boundaries.
    2. Glue together the required Julia ecosystem pieces.
    3. Establish end-to-end ordinary model execution.
    4. Build correctness/conformance harnesses.
    5. Build performance instrumentation.
    6. Build and tune conventional ML kernel families in Lava.
    7. Support real training.
    8. Support real inference.
    9. Expand architecture compatibility.
    10. Close measurable parity gaps.

PRESERVE ARCHITECTURAL SEAMS FOR:

    semantic tensors
    semantic parameters
    representation planners
    type/trait-directed execution
    generated operator families
    execution-plan synthesis
    quantization-as-lowering
    partial evaluation
    memory synthesis
    profile-guided specialization
    rematerialization
    policy-driven execution

RESEARCH AFTER BASELINE MATURITY:

    generalized speculative execution/decoding
    closed-loop materialization
    profile-guided rematerialization
    deeper semantic parameter systems
    structural model mutation
    FluidGEMM
    state-aware/lifecycle-aware compute


=======================================================================
XXXVII. RULE FOR THE LUNA / CODEX / CLAUDE SWARM
=======================================================================

Agents should receive narrow, testable pieces.

No agent should be trusted to "make it fast" without a benchmark.

No agent should be trusted to "support model X" without conformance tests.

No agent should be trusted to "optimize" by changing semantics.

Every meaningful change should produce receipts:

    what changed
    why
    correctness status
    numerical delta
    benchmark before
    benchmark after
    compile-time effect
    memory effect
    device tested
    workload tested

The harness, not agent confidence, determines success.


=======================================================================
XXXVIII. CENTRAL ARCHITECTURAL QUESTION
=======================================================================

For every abstraction added to Harpe, ask:

    Does this preserve useful model meaning?

    Can lower layers exploit that meaning?

    Does it keep logical semantics separate from physical representation?

    Does it permit multiple valid hardware realizations?

    Can it be measured?

    Can it be specialized without causing uncontrolled compilation?

    Does it preserve an escape hatch for future architecture classes?

If yes, it likely belongs.

If it prematurely flattens semantics into generic tensor operations, reconsider it.


=======================================================================
XXXIX. CURRENT CONCLUSION
=======================================================================

The project began as:

    "Can a Julia/Vulkan ML stack compete with the PyStack?"

The more important possibility now appears to be:

    "Can a Julia-native semantic ML compiler/runtime preserve enough model intent to unlock
     execution strategies that conventional tensor-first stacks make unusually difficult?"

Performance parity remains mandatory because the deeper features are irrelevant if the ordinary
stack is unusably slow or incomplete.

But parity is the floor.

The potential ceiling comes from combining:

    Julia multiple dispatch
    semantic tensors
    explicit logical/physical separation
    metaprogrammed implementation families
    Lava kernel generation
    quantization as lowering
    semantic model materialization
    execution-plan synthesis
    profile-guided specialization
    profile-guided rematerialization
    whole-model memory planning
    general speculative execution research

into one coherent architecture.

That combination is the current shape of Harpe.

Do not build all of it immediately.

Build the stack so that, when the ordinary path is mature, none of these ideas requires ripping
the foundation apart.
```
