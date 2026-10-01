/goal GESSO — SEMANTIC ML + AGENT RUNTIME

STATUS
======

Gesso is no longer being conceived as "a Julia replacement for PyTorch."

Gesso is a Julia-native semantic machine-learning compiler/runtime whose first practical mission is:

    TAKE EXISTING MODELS
    UNDERSTAND THEM BETTER
    MATERIALIZE THEM BETTER
    EXECUTE THEM BETTER
    ORCHESTRATE THEM BETTER

The primary and only target is inference.

    TRAINING IS NOT OUR PROBLEM.

Gesso is not a training stack, does not ship one, and does not reserve one. Training the
models Gesso serves is — by deliberate, explicit decision — somebody else's project.
See TRAINING BOUNDARY (Section LVIII).

Gesso should be able to consume models trained elsewhere, preserve their architectural meaning,
compile/materialize them for actual hardware and workloads, and expose an inference and
multi-agent execution runtime substantially more integrated than conventional Python-first stacks.

The long-term goal is not simply:

    "Julia, but PyTorch."

The long-term goal is:

    A SEMANTIC ML SYSTEM WHERE MODEL MEANING SURVIVES DEEP ENOUGH INTO THE MACHINE
    TO INFORM REPRESENTATION, PRECISION, MEMORY, SCHEDULING, SPECIALIZATION,
    QUANTIZATION, KERNELS, AND AGENT ORCHESTRATION.

Short version:

    THE MODEL SHOULD NOT LOSE ITS MEANING BEFORE IT REACHES THE MACHINE.

And now, additionally:

    THE RUNTIME SHOULD NOT LOSE SIGHT OF WHY THE MODEL IS BEING INVOKED.


===============================================================================
I. PROJECT IDENTITY
===============================================================================

Canonical project:

    Gesso.jl

Ecosystem shorthand:

    Gesso Stack™

Public names:

    Gesso     this package — Julia semantic ML compiler/runtime (formerly Harpe)
    Palette   operator surface (palette.jl; formerly NeuraJL)
    Cyan      harness / agent personality (internal policy codename: NIRA)
    Lava      portable Vulkan compute/graphics substrate

Gesso is the ground. Palette is the surface. Cyan is the hand. Lava is the kiln.

Conventional competing ecosystem shorthand used internally:

    PyStack™

Core practical pairing:

    using Gesso
    using Lava

Gesso:
    semantic ML compiler/runtime
    model representation
    inference runtime
    semantic scheduling
    multi-agent runtime
    representation planning
    materialization
    execution synthesis
    profiling/autotuning
    compatibility surfaces

Lava:
    portable machine/GPU substrate
    Vulkan-oriented hardware execution path
    major portability layer
    NOT authored by Gesso

CUDA.jl:
    mature NVIDIA execution path
    performance control
    strategic fast path
    proof that Gesso's semantic architecture can be developed independently of Lava maturity

Gesso should not demand ideological backend purity.

If CUDA is the strongest lowering for a workload:

    USE CUDA.

If Lava enables competitive execution on broader hardware:

    USE LAVA.

Gesso decides what should execute.

The backend determines how that decision reaches the machine.


===============================================================================
II. FIRST PRINCIPLE
===============================================================================

Never discard semantic information earlier than necessary.

Conventional execution frequently trends toward:

    model
      ↓
    generic tensor operations
      ↓
    graph
      ↓
    graph analysis
      ↓
    attempt to recover patterns
      ↓
    compiler optimization
      ↓
    backend

Gesso should prefer:

    logical model
        ↓
    semantic architecture
        ↓
    semantic parameters / tensors / operators
        ↓
    representation planning
        ↓
    execution planning
        ↓
    specialization
        ↓
    lowering
        ↓
    generated / selected kernels
        ↓
    hardware backend

The compiler should not need to rediscover information that Gesso possessed earlier.

Examples:

    This matrix is not merely [4096,4096] FP16.

It may be:

    frozen
    GQA K-projection
    inference-only
    repeatedly accessed during decode
    quantization tolerant under constraint ε
    resident across requests
    optimized primarily for batch 1–4

That additional information can influence:

    representation
    layout
    packing
    quantization
    cache behavior
    memory residency
    fusion
    specialization
    kernel choice
    scheduling


===============================================================================
III. DEVELOPMENT LAW
===============================================================================

The order remains:

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

But "complete" has now been redefined for V1.

V1 does NOT require:

    distributed training
    optimizer zoo
    arbitrary differentiation
    full framework parity
    PlasticWeights
    FluidGEMM
    every model architecture
    every accelerator backend

V1 DOES require a credible inference system.

There is no training stack to earn. Training is out of Gesso's scope permanently;
see TRAINING BOUNDARY (Section LVIII).


===============================================================================
IV. V1 PRODUCT DEFINITION
===============================================================================

Gesso V1 is:

    AN INFERENCE-FIRST SEMANTIC COMPILER/RUNTIME
    FOR EXISTING OPEN-WEIGHT MODELS.

Primary question:

    Can Gesso take a checkpoint that already exists and extract materially better
    practical execution from it because Gesso understands more about the model?

"Better" may include:

    lower latency
    higher throughput
    lower VRAM consumption
    better model fit
    longer practical context
    better batching
    better cache utilization
    phase-specific execution
    hardware-specific materialization
    better quantization decisions
    efficient multi-agent execution

Gesso cannot magically add learned knowledge to unchanged weights.

Therefore distinguish:

    intrinsic model capability

from:

    effective deployed capability.

Gesso targets the second.

A model that can run:

    faster
    with more context
    with more concurrent agents
    under less memory pressure
    on more hardware
    with more reasoning tokens per wall-clock second

is a more capable practical system even if its checkpoint is unchanged.


===============================================================================
V. V1 NON-GOALS
===============================================================================

Do NOT build, ever, as Gesso scope — these are the training stack's problem, not
deferred Gesso features (see TRAINING BOUNDARY):

    distributed training
    optimizer zoo
    gradient accumulation
    training checkpoint orchestration
    data pipelines
    FSDP equivalents
    training schedulers
    training-time fault tolerance
    pretraining infrastructure
    RLHF pipelines
    every AD mode
    Gesso-native model training

Do NOT begin by building:

    custom GPU kernels for every operation
    every quantization scheme
    every architecture
    PlasticWeights execution
    generalized speculative decoding
    autonomous architecture mutation

Do NOT begin by replacing:

    CUDA
    Lava
    JuliaGPU
    LLVM
    safetensors
    Hugging Face model metadata
    every Julia ML package

Reuse machinery where it is already good.

Gesso's value is architecture and semantics, not gratuitous reinvention.


===============================================================================
VI. HIGH-LEVEL STACK
===============================================================================

                        APPLICATIONS
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
       Cyan      Native Julia       C / C++ users
          │                 │                 │
          │                 │              libgesso
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                    GESSO PUBLIC API
                            │
                            ▼
                  SEMANTIC MODEL LAYER
                            │
             ┌──────────────┼──────────────┐
             │              │              │
       architecture     parameters     operators
       semantics        semantics      semantics
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                 REPRESENTATION PLANNER
                            │
                            ▼
                   EXECUTION PLANNER
                            │
             ┌──────────────┼──────────────┐
             │              │              │
          prefill         decode       agent/swarm
          planning       planning       planning
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                    SPECIALIZATION
                            │
                            ▼
                      LOWERING
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
           CUDA.jl                 Lava
              │                     │
              └──────────┬──────────┘
                         ▼
                     HARDWARE


===============================================================================
VII. PACKAGE PHILOSOPHY
===============================================================================

Gesso core should remain relatively lean.

Avoid importing the entire Julia ML ecosystem merely because it exists.

Initial Project.toml philosophy:

    Gesso earns every hard dependency.

Prefer:

    weak dependencies
    package extensions
    clean backend interfaces
    interoperability modules

Potential package organization:

    Gesso/
    ├── src/
    │   ├── Gesso.jl
    │   ├── Semantics/
    │   ├── ModelIR/
    │   ├── Parameters/
    │   ├── Operators/
    │   ├── Representation/
    │   ├── Planning/
    │   ├── Lowering/
    │   ├── Inference/
    │   ├── Agents/
    │   ├── Runtime/
    │   ├── Profiling/
    │   ├── Autotune/
    │   └── CAPI/
    │
    ├── ext/
    │   ├── GessoLavaExt.jl
    │   └── GessoCUDAExt.jl
    │
    ├── test/
    ├── benchmark/
    ├── examples/
    └── docs/

Exact boundaries remain subject to implementation discovery.

Do not prematurely split everything into separate repositories.


===============================================================================
VIII. MODEL IMPORT
===============================================================================

Gesso must accept existing models without requiring retraining.

Target:

    model config
    tokenizer definition
    safetensors / equivalent checkpoint
              ↓
        Gesso importer
              ↓
      canonical architecture
              +
      semantic parameter map
              ↓
       executable Gesso model

Checkpoint format is transport.

Architecture semantics are execution information.

Gesso should understand concepts such as:

    embeddings
    RMSNorm
    LayerNorm
    RoPE
    attention
    GQA
    MQA
    sliding attention
    local/global attention
    SwiGLU
    dense FFN
    MoE
    expert routing
    recurrent blocks
    state-space blocks
    cache semantics
    output heads

Do NOT implement:

    LlamaRuntime
    QwenRuntime
    MistralRuntime
    ModelFooRuntime

Prefer:

    reusable semantic primitives

A new architecture should usually require:

    configuration mapping
    parameter mapping
    composition of existing operators

Only genuinely novel computation should require a new semantic primitive.


===============================================================================
IX. MODEL ACCEPTANCE LADDER
===============================================================================

Model support should advance through explicit tiers.

TIER A — TOY / REFERENCE

    tiny internal transformer
    CPU/reference execution
    deterministic tests
    gradient irrelevant initially
    fast development cycle

TIER B — SMALL REAL MODEL

    open checkpoint
    fast enough for constant regression testing
    validates importer
    validates tokenizer
    validates generation

TIER C — MODERN MID-SIZED MODEL

    exercises:
        GQA
        current normalization
        modern positional encoding
        real KV behavior
        quantization
        longer context

TIER D — LARGE LOCAL MODEL

    validates:
        memory planning
        quantization
        multi-device pressure
        scheduler behavior
        real-world throughput

TIER E — ARCHITECTURAL VARIETY

    MoE
    hybrid models
    recurrent/state-space
    novel attention forms

For each accepted family:

    config acceptance
    parameter import
    logit parity
    generation parity
    quantized parity
    performance measurements


===============================================================================
X. CORRECTNESS CONTRACT
===============================================================================

No performance claim without correctness.

Gesso needs:

    CPU/reference implementation
    deterministic fixtures
    numerical tolerance policies
    dtype-specific tolerance
    odd-shape coverage
    boundary conditions
    invalid-input tests
    serialization tests
    tokenizer parity
    logit parity
    sampling parity where deterministic
    KV-cache consistency tests
    prefill/decode equivalence tests

Core invariant:

    optimized execution must preserve declared semantics.

Autotuning must NEVER select a faster implementation that silently violates:

    numerical contract
    memory contract
    determinism contract
    representation contract
    architecture semantics


===============================================================================
XI. SEMANTIC PARAMETERS AND TENSORS
===============================================================================

Gesso should distinguish storage from meaning.

Conventional:

    Tensor{Float16}

Gesso conceptually:

    ProjectionWeight
    KVCache
    EmbeddingTable
    ExpertWeight
    FrozenParameter
    QuantizedParameter
    Activation
    TemporaryWorkspace
    RoutingState
    DecodeState
    AdapterDelta

Important:

    DO NOT make every scalar symbolic.

Semantic richness belongs primarily at:

    parameter
    tensor
    operator
    region
    execution object

levels.

A parameter may carry:

    logical identity
    semantic role
    mutability
    lifecycle
    constraints
    precision policy
    quantization eligibility
    sparsity information
    locality
    execution phase relevance
    residency policy

Physical bytes are one realization of that logical object.

How those families are encoded as Julia objects is law under §CIX.
Phase 1 implements that encoding; it does not choose another.

Boundary notes:

    Gradient and OptimizerState are training-side concepts and are NOT Gesso semantic
    types. Gesso parameters are inference-resident objects.

    AdapterDelta (LoRA-style) IS a Gesso semantic type. Serving adapters is a
    materialization problem — base model plus hot-swappable deltas trained elsewhere.
    Consuming fine-tunes is in scope; producing them is not. The interchange point is
    weights-on-disk, which already exists and is standardized.


===============================================================================
XII. MULTIPLE DISPATCH AS EXECUTION MECHANISM
===============================================================================

Julia multiple dispatch is central.

Kernel/operator behavior is often a function of:

    operation
      × representation
      × dtype
      × layout
      × semantic role
      × device
      × workload
      × execution phase

Example conceptually:

    matmul!(
        ::Activation,
        ::FrozenQuantizedProjection,
        ::DecodeWorkload,
        ::LavaDevice
    )

may resolve differently from:

    matmul!(
        ::Activation,
        ::BF16Projection,
        ::PrefillWorkload,
        ::CUDADevice
    )

This avoids giant branch-heavy operators.

The semantic question becomes:

    What implementation is appropriate for this interaction among these objects?

An Operator is a function. Dispatch is the execution mechanism.
An operator is not a ModelIR node. Encoding: §CIX.


===============================================================================
XIII. SPECIALIZATION DISCIPLINE
===============================================================================

Do NOT encode every property in Julia types.

Otherwise:

    GESSO = compile-latency benchmark.

Use approximately:

    TYPES
        stable structural identity

    TRAITS
        optimization-relevant properties

    RUNTIME METADATA
        volatile information

Possible type-level information:

    tensor semantic family
    representation family
    backend family
    layout family

Possible traits:

    frozen
    quantized
    contiguous
    immutable
    decode-hot
    inference-only
    sparse
    cooperative-matrix capable

Runtime:

    batch size
    sequence length
    request count
    free memory
    queue pressure

Gesso should decide when runtime facts deserve promotion into specialization.

This promotion can eventually become profile-guided.

This three-way split is the allocation rule for the semantic core.
The Phase 1 object-model encoding (§CIX) is that rule applied to
SemanticTensor, ModelIR node, and Operator. Do not collapse the three
objects onto one encoding.


===============================================================================
XIV. SEMANTIC MODEL MATERIALIZATION
===============================================================================

CORE HEAVYWEIGHT IDEA #1

Logical model ≠ physical model.

Gesso should answer:

    Given:

        logical model
        hardware
        workload
        memory constraints
        latency objective
        numerical constraints

    how should this model physically exist?

Potential choices:

    dtype
    quantization
    layout
    packing
    sparsity
    tiling
    residency
    cache representation
    fusion
    kernel family
    backend
    device partitioning

Example:

    Same model:

        long prefill    → throughput-oriented layout
        batch-1 decode  → latency-oriented packed representation
        low-memory mode → more aggressive compression
        adapter serving → base + AdapterDelta realization

The logical model remains the same.


===============================================================================
XV. QUANTIZATION AS LOWERING
===============================================================================

Quantization is NOT merely:

    checkpoint
       ↓
    external quantizer
       ↓
    new checkpoint

Gesso:

    logical parameter
        +
    semantic role
        +
    sensitivity / error contract
        +
    workload
        +
    device capability
        ↓
    representation planner
        ↓
    selected physical representation
        ↓
    matching kernel

Potential result:

    embeddings      → INT8
    hot projection  → INT4
    sensitive norm  → BF16
    cold expert     → aggressive compression
    KV cache        → context/workload-specific representation

Representation selection should be explainable and benchmarked.

Long-term:

    the same logical checkpoint may not have one canonical quantization.


===============================================================================
XVI. PHASE-SPECIFIC MATERIALIZATION
===============================================================================

Prefill and decode are different workloads.

Do not pretend otherwise.

Gesso may choose distinct:

    layouts
    kernel families
    quantization
    cache strategies
    scheduling
    memory policies

for:

    prompt prefill
    batch-1 decode
    batched decode
    long-context execution
    structured/tool output
    swarm inference

This list is a taxonomy of CATEGORIES, not a frozen enum.

Prefill and decode are dispatch types (§CIX, Packet 1).
The finer names are documented tags for receipts and plans until a
lowering actually dispatches on them.

This is one of the earliest semantic advantages worth testing.


===============================================================================
XVII. SEMANTIC EXECUTION SYNTHESIS
===============================================================================

CORE HEAVYWEIGHT IDEA #2

Execution need not be one fixed tensor graph.

Gesso may synthesize among legal plans involving:

    fusion
    staging
    recomputation
    asynchronous overlap
    precision
    representation conversion
    memory placement
    backend choice
    cache policy
    speculation
    device partitioning

Concept:

    semantic computation
            ↓
    legal execution plans
            ↓
    constraint filtering
            ↓
    cost/performance evaluation
            ↓
    selected executable plan


===============================================================================
XVIII. SEMANTIC PARTIAL EVALUATION
===============================================================================

Gesso should eventually exploit known stable facts:

    frozen weights
    fixed architecture
    fixed head dimensions
    constant masks
    inference-only operators
    fixed routing elements
    device properties
    chosen quantization
    workload regime

Then transform:

    general model program

into:

    specialized residual program.

This can support:

    dead program elimination
    constant folding
    frozen-region folding
    specialization
    fusion
    smaller runtime surface

Gesso is the ground: partial evaluation leaves only what deployment still
needs for Palette to paint.


===============================================================================
XIX. WHOLE-MODEL MEMORY SYNTHESIS
===============================================================================

Memory management should eventually be a global planning problem.

Inputs:

    model
    hardware memory
    expected workload
    latency objective
    context requirements

Planner may determine:

    weight residency
    host/GPU tiering
    KV placement
    activation lifetime
    workspace reuse
    expert residency
    compression
    recomputation
    transfer scheduling

Long-term interface concept:

    materialize(
        model;
        objective = MinLatency(),
        memory_budget = GiB(16)
    )

Gesso should be able to explain why a plan fits.


===============================================================================
XX. HARDWARE CAPABILITY MODEL
===============================================================================

Gesso should target capabilities, not vendor names.

Represent capabilities such as:

    native BF16
    native FP16
    FP8 support
    INT8 dot products
    cooperative matrix operations
    subgroup width
    local/shared memory
    asynchronous copy behavior
    available VRAM
    relevant atomic operations
    matrix accelerator characteristics

Target question:

    What can this device legally and efficiently execute?

Not:

    Is this NVIDIA?


===============================================================================
XXI. EXECUTION TIERS
===============================================================================

Gesso should degrade gracefully.

TIER 0 — PORTABLE CORRECTNESS

    generic legal execution
    broad hardware support
    performance secondary

TIER 1 — OPTIMIZED GENERIC

    common GPU capabilities
    broadly tuned implementations

TIER 2 — ARCHITECTURE-SPECIALIZED

    hardware-family specialization
    NVIDIA-specific / AMD-specific etc.

TIER 3 — DEVICE + MODEL + WORKLOAD SPECIALIZATION

    actual model
    actual device
    actual workload regime
    profile-derived specialization

Every higher tier must retain a lower-tier fallback.


===============================================================================
XXII. CUDA.jl STRATEGY
===============================================================================

CUDA.jl is strategically valuable.

Early Gesso can prove:

    semantic architecture
    model import
    representation planning
    inference scheduling
    materialization
    quantization
    agent runtime

without simultaneously solving Vulkan kernel parity.

Path:

    Gesso
      ↓
    CUDA.jl
      ↓
    NVIDIA

This creates a mature control backend.

If Gesso + CUDA wins against conventional execution:

    Gesso architecture is doing useful work.

If Gesso + CUDA performs well but Lava lags:

    improve Lava/kernel path.

Do not confuse backend maturity with compiler architecture.


===============================================================================
XXIII. LAVA STRATEGY
===============================================================================

Lava provides the portable/native machine path.

Gesso should build against a capability-oriented backend contract.

Gesso should NOT force Lava to understand:

    model architecture
    agent roles
    quantization policy
    NIRA semantics

Lava should expose:

    device
    memory
    kernels
    synchronization
    execution
    low-level capabilities

Gesso owns ML semantics.

This separation prevents mutual contamination.


===============================================================================
XXIV. KERNEL PROGRAM
===============================================================================

Eventually required kernel families include:

    GEMM
    GEMV
    reductions
    softmax
    RMSNorm
    LayerNorm
    RoPE
    activations
    SwiGLU
    fused FFN
    attention
    KV transforms
    embedding lookup
    routing
    token grouping
    expert GEMM
    aggregation
    quantize
    dequantize
    sampling

But Gesso should not begin by rewriting every kernel.

Use available performant kernels where possible.

Generate/tune where Gesso has evidence of opportunity.


===============================================================================
XXV. METAPROGRAMMED IMPLEMENTATION FAMILIES
===============================================================================

Avoid handwritten explosion:

    INT4_GQA_DECODE_HEAD128_X
    INT8_GQA_DECODE_HEAD128_Y
    BF16_GQA_PREFILL_HEAD128_Z
    ...

Prefer:

    semantic operator
       +
    representation
       +
    hardware capability
       +
    workload
       ↓
    implementation generator
       ↓
    legal candidate family
       ↓
    correctness filter
       ↓
    benchmark
       ↓
    selected specialization


===============================================================================
XXVI. AUTOTUNING
===============================================================================

Autotuning is a first-class Gesso concern.

Gesso should survey and integrate existing Julia autotuning/kernel-generation work before
creating redundant infrastructure.

Generic loop:

    semantic operation
          ↓
    device capability set
          ↓
    workload signature
          ↓
    candidate generation
          ↓
    compile
          ↓
    correctness validation
          ↓
    benchmark
          ↓
    select winner
          ↓
    cache result

Cache key may eventually include:

    device identity
    driver/toolchain
    backend
    model architecture
    operator semantics
    representation
    shape regime
    workload signature

Autotuning must be:

    reproducible
    inspectable
    invalidatable
    numerically gated


===============================================================================
XXVII. PROFILE-GUIDED SPECIALIZATION
===============================================================================

Observe actual use.

Example:

    92% of decode traffic:
        batch 1–3
        head_dim 128
        context 8k–24k

Gesso may conclude:

    these properties are stable enough to justify specialized execution.

Loop:

    observe
      ↓
    identify stable facts
      ↓
    propose specialization
      ↓
    benchmark
      ↓
    validate
      ↓
    retain winner


===============================================================================
XXVIII. PROFILE-GUIDED REMATERIALIZATION
===============================================================================

CORE HEAVYWEIGHT IDEA #3

The learned weights do not need to change.

Gesso may alter their physical realization.

Examples:

    hot region
        → higher-performance packing

    cold expert
        → compress or move

    workload changes to long context
        → alter cache representation

    available memory changes
        → rematerialize under pressure

    decode dominates
        → choose decode-specific physical layout

Closed loop:

    semantics
       ↓
    representation
       ↓
    execution
       ↓
    profiling
       ↓
    improved materialization
       ↓
    validation
       ↓
    deployment


===============================================================================
XXIX. GESSO INFERENCE ENGINE
===============================================================================

Inference is NOT an adapter around llama.cpp.

Gesso requires a native inference stack.

Main components:

    model loader
    semantic model
    representation planner
    execution planner
    KV manager
    prefill engine
    decode engine
    scheduler
    continuous batching
    sampling
    prefix cache
    streaming
    memory-pressure handling
    backend lowering
    metrics
    profiling
    autotuning

Concept:

                    GESSO INFERENCE

     semantic model
          │
          ├── prefill plan
          ├── decode plan
          ├── batch plan
          └── long-context plan
          │
          ▼
        scheduler
          │
     ┌────┴────┐
     ▼         ▼
   CUDA       Lava


===============================================================================
XXX. PREFILL / DECODE SEPARATION
===============================================================================

Make prefill and decode explicit execution modes from the beginning.

Prefill:

    throughput-oriented
    large matrix work
    prompt ingestion
    different memory locality
    potentially different quantization/layout

Decode:

    latency-oriented
    small token increments
    KV dominated
    repeated weight access
    batch-sensitive
    scheduler-sensitive

Never force both through one "generic generate()" implementation internally.

PrefillWorkload and DecodeWorkload are therefore types in the operator
dispatch surface from Phase 1/2 (§CIX). They are not metadata.


===============================================================================
XXXI. KV CACHE AS A SEMANTIC OBJECT
===============================================================================

KV cache should not simply be "some tensors."

Gesso should understand:

    ownership
    request identity
    prefix sharing
    sequence position
    lifetime
    mutability
    compression
    representation
    residency
    eviction eligibility

Potential future optimizations:

    paged cache
    prefix reuse
    cross-agent system-prompt sharing
    tiering
    quantized KV
    context-pressure rematerialization


===============================================================================
XXXII. SCHEDULER
===============================================================================

Gesso requires its own inference-aware scheduler.

Inputs include:

    prefill requests
    decode requests
    request priority
    latency budget
    throughput objective
    memory state
    backend state
    agent dependencies

Scheduler can distinguish:

    long prefill
    short prefill
    interactive decode
    background decode
    speculative work
    blocked agents

This is necessary before Gesso can fully exploit multi-agent workloads.


===============================================================================
XXXIII. MULTI-AGENT ORCHESTRATION
===============================================================================

NEW MAJOR PILLAR

Gesso should treat multi-agent orchestration as a SYSTEMS RUNTIME problem.

Do NOT build:

    LangChain.jl

Do NOT reproduce:

    giant Python callback framework
    stringly typed agent objects
    serialization everywhere
    one inference API call per conceptual agent action

Gesso should expose a small set of composable primitives around:

    agents
    tasks
    dependencies
    channels
    capabilities
    tools
    inference intents
    budgets
    priorities
    cancellation
    deadlines
    receipts


===============================================================================
XXXIV. WHY ORCHESTRATION BELONGS IN GESSO
===============================================================================

Gesso can know BOTH:

    what the model computation means

and:

    why the model is currently being invoked.

This enables:

    cross-agent batching
    shared model residency
    shared kernels
    shared prefix state
    workload-aware scheduling
    GPU priority decisions
    memory-aware swarm control
    agent-aware inference planning

A detached application-level framework cannot easily optimize across those boundaries.


===============================================================================
XXXV. AGENT REPRESENTATION
===============================================================================

Potential conceptual structure:

    Agent
        model
        context
        tools
        policy
        priority
        budget
        capabilities

Julia multiple dispatch can naturally express:

    act!(::ResearchAgent, ::ResearchTask)
    act!(::CodeAgent, ::CodeTask)
    act!(::CriticAgent, ::ReviewTask)

Tool permissions:

    allowed(::ResearchAgent, ::WebSearch)
    allowed(::ResearchAgent, ::ShellExec)

Again:

    JSON may remain a wire format.

JSON should NOT become Gesso's ontology.


===============================================================================
XXXVI. SHARED MODEL EXECUTION
===============================================================================

Five agents using one model should not require five conceptual model installations.

Gesso should share when safe:

    model weights
    materialized representations
    compiled kernels
    tokenizer state
    common prefixes
    system-prompt prefixes
    immutable tool schemas

while maintaining:

    separate logical contexts
    separate task state
    separate mutable memories
    separate agent identity


===============================================================================
XXXVII. CROSS-AGENT BATCHING
===============================================================================

Example runtime state:

    Agent A → next-token decode
    Agent B → waiting for tool
    Agent C → next-token decode
    Agent D → 4k-token prefill
    Agent E → next-token decode
    Agent F → blocked
    Agent G → short prefill

Gesso may construct:

    decode batch:
        A, C, E

    prefill batch:
        D, G

    zero GPU time:
        B, F

No individual agent needs to understand batching.

The runtime does.


===============================================================================
XXXVIII. AGENT PRIORITIES
===============================================================================

Example:

    interactive user response       = highest priority
    tool-selection decision         = high priority
    coder continuation              = medium
    background critic               = low
    speculative research branch     = very low

Gesso can optimize global system latency rather than only per-request throughput.


===============================================================================
XXXIX. BUDGET-AWARE ORCHESTRATION
===============================================================================

Potential API direction:

    run!(
        swarm;
        token_budget = 50_000,
        memory_budget = GiB(24),
        latency_budget = Second(30),
        objective = MaximizeQuality()
    )

Gesso may decide:

    number of concurrent agents
    model assignment
    batch formation
    quantization
    speculative branch count
    branch termination
    inference priority
    memory reservations

Long-term policy stack:

    parameter policy
        ↓
    model policy
        ↓
    inference policy
        ↓
    agent policy
        ↓
    swarm policy


===============================================================================
XL. TOOL EXECUTION
===============================================================================

Tools should be semantic capabilities.

Examples:

    Search(query)
    RunTests(project)
    Compile(target)
    InspectArtifact(id)
    QueryMemory(key)

Model-facing serialization may remain JSON.

Runtime-facing representation should be structured.

This permits:

    type checking
    capability checking
    authorization
    validation
    scheduling
    receipts


===============================================================================
XLI. AGENT WORKFLOW COMPILATION
===============================================================================

Repeated workflows may expose stable structure.

Example:

    plan
      ↓
    inspect
      ↓
    edit
      ↓
    test
      ↓
    review
      ↓
    commit

Gesso should NOT compile away model reasoning.

But it may optimize the surrounding machinery:

    dependency scheduling
    tool prewarming
    likely schema loading
    model batching
    cache allocation
    process startup
    prefix preparation
    resource reservation

This is the agent analogue of semantic partial evaluation.


===============================================================================
XLII. RECEIPTS
===============================================================================

Gesso agent execution should be auditable.

Each significant action may record:

    agent
    task
    model
    materialization
    inference request
    tool request
    tool result
    parent dependency
    timing
    token usage
    memory usage
    failure
    retry
    cancellation
    output digest

Receipts matter for:

    debugging
    deterministic replay where possible
    benchmarking
    safety
    performance analysis
    swarm coordination

Identity scheme (§CIX, Packet 2):

    until receipts persist or cross process, id is a process-local UInt64
    parent_dependency is a receipt id in the same process
    `task` remains its own field

A globally unique id is minted at the first persistence or cross-process
boundary, by bumping RECEIPT_SCHEMA_VERSION. Never silently reinterpret
old ids (§LXIX).


===============================================================================
XLIII. GESSO / PALETTE / CYAN OWNERSHIP BOUNDARY
===============================================================================

This boundary should be explicit now.

GESSO
-----

Owns mechanism.

    agent execution primitives
    task graph
    inference scheduling
    cross-agent batching
    budgets
    priorities
    cancellation
    lifecycle
    shared model resources
    hardware resources
    receipts
    runtime semantics


PALETTE
-------

Owns expression.

Palette should expose a CONDENSED agent-friendly surface over Gesso.

Potential conceptual syntax:

    @agents begin
        planner  = agent(:planner)
        coder    = agent(:coder)
        reviewer = agent(:reviewer)

        planner --> coder
        coder   --> reviewer
    end

or:

    @parallel begin
        inspect(repo)
        benchmark(model)
        search(topic)
    end

Palette lowers concise intent into Gesso runtime structures.

Palette should NOT duplicate Gesso's scheduler.


CYAN
----

Owns policy and cognition (internal codename: NIRA).

    when to spawn agents
    roles
    modes
    cognitive architecture
    memory strategy
    tool policy
    domain mode selection
    planning behavior
    critic behavior

Rule:

    Gesso     = mechanism
    Palette   = expression
    Cyan      = policy and cognition (internal: NIRA)


===============================================================================
XLIV. CYAN AS FIRST GESSO-NATIVE HARNESS
===============================================================================

Cyan becomes Gesso's first demanding native consumer.

This is strategically excellent.

Cyan stresses:

    long-lived inference
    tool use
    structured outputs
    multi-agent execution
    repeated context
    agent modes
    background work
    tool latency
    scheduling
    shared model usage

Gesso should dogfood through Prime.

But Gesso must remain usable WITHOUT Prime.


===============================================================================
XLV. NIRA PYTHON REDUCTION
===============================================================================

Moving NIRA toward Gesso will likely justify removing Python orchestration where it provides no
unique value.

Target architecture:

    Julia
        Gesso
        Palette
        NIRA orchestration

    Rust
        existing authority/services where appropriate

    C ABI
        stable Julia ↔ Rust boundary where needed

Do NOT rewrite good Rust code merely to become "all Julia."

Rust can expose:

    extern "C"

Julia can call via:

    @ccall

Clang tooling may help generate Julia bindings for C APIs.

Goal:

    remove unnecessary Python hops,
    NOT conduct a language purity crusade.


===============================================================================
XLVI. C ABI
===============================================================================

Gesso should eventually expose a boring stable C interface.

This is an ADOPTION SURFACE.

Not the internal architecture.

Possible opaque types:

    gesso_runtime_t
    gesso_model_t
    gesso_context_t
    gesso_request_t

Possible operations:

    gesso_runtime_create
    gesso_model_load
    gesso_context_create
    gesso_tokenize
    gesso_prefill
    gesso_decode
    gesso_sample
    gesso_cancel
    gesso_context_destroy
    gesso_model_destroy

All semantic/compiler complexity stays behind opaque handles.


===============================================================================
XLVII. LLAMA.CPP COMPATIBILITY
===============================================================================

The C ABI exists for users who are stubbornly embedded in existing C/C++ inference software.

Compatibility direction:

    existing application
          ↓
    compatibility adapter
          ↓
    libgesso
          ↓
    Gesso native runtime

Do NOT architect Gesso around llama.cpp internals.

Rule:

    compatibility adapts TO Gesso.

Gesso does NOT adapt its ontology TO compatibility.

This gives a migration ladder:

    LEVEL 1
        keep existing app
        use compatibility bridge

    LEVEL 2
        use Gesso C ABI directly

    LEVEL 3
        native Julia / Gesso semantics


===============================================================================
XLVIII. BACKEND-INDEPENDENT PUBLIC API
===============================================================================

User-facing Gesso code should avoid hardcoding backend assumptions.

Concept:

    runtime = Gesso.Runtime(device)

    model = Gesso.load(path)

    deployed = Gesso.materialize(
        model,
        runtime;
        objective = MinLatency()
    )

Gesso decides backend-specific realization underneath.

Advanced users may override policies explicitly.


===============================================================================
XLIX. PERFORMANCE OBSERVABILITY
===============================================================================

Gesso needs first-class telemetry.

Track:

    kernel latency
    launch latency
    bandwidth
    occupancy
    allocation
    synchronization
    compilation time
    specialization count
    cache hit rate
    VRAM use
    host memory
    transfers
    prefill throughput
    TTFT
    decode tokens/sec
    per-token latency
    batching efficiency
    KV footprint
    scheduler queue time
    agent idle time
    tool wait time
    swarm GPU utilization

Gesso must explain performance.


===============================================================================
L. PERFORMANCE FAILURE TAXONOMY
===============================================================================

Every gap should be classifiable:

    kernel quality
    layout
    quantization
    memory transfer
    synchronization
    allocation
    compiler issue
    missing specialization
    excessive specialization
    scheduler
    batching
    cache policy
    backend
    hardware limitation
    algorithm
    semantic conversion overhead

"No idea why it is slow" is unacceptable.


===============================================================================
LI. BENCHMARK MATRIX
===============================================================================

Compare equivalent workloads across:

    reference Python execution
    compiled Python execution
    strong existing inference runtime
    Gesso + CUDA.jl
    Gesso + Lava

Where applicable:

    llama.cpp baseline
    other strong serving engines

Measure:

    correctness
    TTFT
    prefill tokens/sec
    decode tokens/sec
    latency distribution
    VRAM
    RAM
    power where available
    compile/startup time
    throughput under concurrency
    multi-agent throughput

Gesso + CUDA is particularly important for separating:

    Gesso architecture

from:

    Lava/kernel maturity.


===============================================================================
LII. MODEL-CAPABILITY BENCHMARKS
===============================================================================

Do not benchmark only raw token throughput.

Measure effective-system advantages.

Examples:

    maximum context under fixed VRAM
    largest model under fixed hardware
    agents served concurrently
    reasoning tokens / second
    throughput under mixed prefill+decode
    structured tool workloads
    memory-pressure behavior
    performance after profile-guided specialization


===============================================================================
LIII. MULTI-AGENT BENCHMARKS
===============================================================================

Compare:

    independent model requests

vs.

    Gesso semantic swarm execution.

Scenarios:

    2 agents
    4 agents
    8 agents
    16 agents

Mixed states:

    decode
    prefill
    blocked on tool
    background critic
    user-facing response

Measure:

    GPU occupancy
    queue delay
    tokens/sec aggregate
    user-facing latency
    VRAM
    shared-prefix savings
    batching efficiency


===============================================================================
LIV. FIRST RESEARCH WIN REQUIREMENT
===============================================================================

Gesso does not need to outperform everything immediately.

It needs ONE clean, reproducible semantic win.

Example candidates:

    phase-specific materialization
    semantic quantization choice
    cross-agent batching
    shared-prefix agent execution
    workload specialization
    memory-plan advantage
    profile-guided kernel choice

Requirement:

    same model
    same hardware
    same correctness
    measurable advantage
    reproducible benchmark

One such result validates the direction.


===============================================================================
LV. SPECULATIVE EXECUTION
===============================================================================

Longer-term research.

Question:

    Can speculation become a general runtime/compiler capability rather than a single
    draft-model trick?

Potential directions:

    candidate branch execution
    semantic verification
    confidence-aware paths
    partial computation
    decode-plan speculation
    workflow speculation
    agent-branch speculation

Do NOT promise:

    draft models are obsolete
    guaranteed acceleration
    solved speculative decoding

Research only until benchmarked.


===============================================================================
LVI. NEUROSYMBOLIC PARAMETER DIRECTION
===============================================================================

Future possibility:

    parameters carry richer semantic state than numeric value alone.

Near-term interpretation:

    logical parameter
        +
    semantic metadata/traits
        ↓
    representation and execution decisions

NOT:

    every individual scalar is a symbolic object.

Long-term this may become fertile ground for:

    stateful parameters
    lifecycle-aware parameters
    symbolic constraints
    semantic learning rules


===============================================================================
LVII. PLASTICWEIGHTS / FLUIDGEMM SEAM
===============================================================================

Gesso must NOT depend on PlasticWeights.

But Gesso should preserve extension points for:

    lifecycle-aware representation
    mutable/frozen transitions
    state-specific precision
    state-specific storage
    state-aware kernels
    FluidGEMM

Future possibility:

    learning consolidation event
            ↓
    compute representation transition

Example:

    plastic FP32/BF16 region
            ↓
    consolidated
            ↓
    packed/quantized/frozen execution form

This remains future research.


===============================================================================
LVIII. TRAINING BOUNDARY — NOT OUR PROBLEM
===============================================================================

    Gesso is an inference, execution, materialization, and orchestration stack.
    Training is somebody else's problem.

This is not a deferral. There is no training phase coming after the inference phases.
Nothing in Gesso — no AD mode, no optimizer, no gradient path, no training loop —
is being reserved for later. If a Julia-native training stack ever exists, Gesso will
consume its checkpoints exactly like every other checkpoint. The interchange point is
weights-on-disk, which is standardized and which Gesso already depends on.

The full training obligation list that is explicitly OUT of scope:

    distributed training
    optimizer zoo
    gradient accumulation
    training checkpoint orchestration
    data pipelines
    FSDP equivalents
    training schedulers
    training-time fault tolerance
    pretraining infrastructure
    RLHF pipelines
    forward/backward execution
    Enzyme / AD integration
    activation checkpointing

Only three inferences of training remain in Gesso, and they are consumption-side:

    fine-tune consumption   LoRA/adapter deltas are served materializations
                            (see AdapterDelta, Section XI). Trained elsewhere,
                            loaded like any weights.

    calibration passes      Activation statistics for representation planning
                            (Section XV). Read-only measurement, not learning.

    inference resiliency    Request recovery, KV durability under preemption,
                            scheduler crash-recovery. This is serving behavior,
                            not "training-time fault tolerance."

If the training problem is ever picked up — inside this ecosystem or outside it —
the natural Julia-native starting point is WGPU.jl, not Gesso.

One preserved rule for that future: if a training stack ever wants inference-time
co-design, it must REUSE Gesso semantic representations. Do NOT create
"Gesso Training Framework 2" beside Gesso, and do NOT bolt an unrelated training
architecture INTO Gesso.


===============================================================================
LIX. NUMERICS BESIDE THE MODEL (SEAM, NOT SCOPE)
===============================================================================

Long-term Julia advantage:

    ODE/PDE solvers
    symbolic transforms
    simulations
    iterative numerical methods

may coexist inside one program with model inference.

Gesso itself builds NO differentiation machinery. Gradient computation, optimizer
execution, and training of any kind belong to the training stack (Section LVIII).

The one in-scope numeric-adjacent activity is calibration:

    activation statistics passes used by representation planning
    (quantization scales, sensitivity estimates)

Calibration runs as a read-only workload over the inference engine's own operator set.
It is measurement, not learning.

If a future Gesso-native runtime feature ever legitimately requires differentiating
something, that requirement must be re-litigated explicitly at that time. It does not
quietly reopen training scope.


===============================================================================
LX. STRUCTURALLY DYNAMIC MODELS
===============================================================================

Preserve space for models that:

    add experts
    remove experts
    alter routing
    freeze/thaw regions
    change representation
    redistribute capacity
    change residency

Do not implement early.

Do not architect them out.


===============================================================================
LXI. POLICY-DRIVEN COMPILATION
===============================================================================

Long-term public interface may support:

    MinLatency()
    MaxThroughput()
    MinMemory()
    Deterministic()
    AccuracyBound(ε)

plus constraints:

    memory_budget
    precision_floor
    latency_budget
    device_set
    context_requirement

Gesso then constructs a legal plan satisfying policy.


===============================================================================
LXII. CLOSED-LOOP GESSO
===============================================================================

Long-term picture:

    MODEL SEMANTICS
         ↓
    PARAMETER SEMANTICS
         ↓
    WORKLOAD SEMANTICS
         ↓
    AGENT SEMANTICS
         ↓
    REPRESENTATION PLAN
         ↓
    EXECUTION PLAN
         ↓
    KERNEL SPECIALIZATION
         ↓
    RUNTIME EXECUTION
         ↓
    PROFILING
         ↓
    AUTOTUNING
         ↓
    REMATERIALIZATION
         └───────────────────┐
                             ↓
                           repeat

This is the conceptual unifier.


===============================================================================
LXIII. WHAT GESSO IS NOT
===============================================================================

Gesso is NOT:

    PyTorch rewritten in Julia
    llama.cpp rewritten in Julia
    LangChain rewritten in Julia
    a training stack (training is somebody else's problem — Section LVIII)
    Lava wrapper
    CUDA wrapper
    transformer-only runtime
    quantization tool
    PlasticWeights runtime
    NIRA-specific package
    kernel collection

Gesso may interact with all of those concerns.

It is broader:

    semantic ML + inference + execution runtime.


===============================================================================
LXIV. DIFFERENTIATION FROM PYSTACK™
===============================================================================

The useful question is NOT:

    "Can PyTorch theoretically implement this?"

PyTorch is extensible.

The question is:

    "Does Gesso make this a natural system composition rule where PyStack™ would require
     bespoke graph rewriting, tensor subclasses, compiler plugins, custom kernels,
     hooks, metadata plumbing, multiple runtimes, and application-level orchestration?"

Gesso advantage:

    advanced behavior should become ordinary.

Not heroic.


===============================================================================
LXV. THE SECOND SEMANTIC PRINCIPLE
===============================================================================

Original rule:

    Do not discard what the MODEL means.

New rule:

    Do not discard what the REQUEST means.

An inference request may be:

    interactive user decode
    bulk prefill
    critic pass
    tool-selection pass
    speculative branch
    low-priority background reasoning

That meaning should survive into scheduling.


===============================================================================
LXVI. THE THIRD SEMANTIC PRINCIPLE
===============================================================================

Do not discard what the AGENT COMPUTATION means.

A collection of model calls is not necessarily independent traffic.

It may be:

    planner → coder → reviewer

Gesso should understand dependencies.

This enables system-wide scheduling rather than blind request serving.


===============================================================================
LXVII. API DESIGN LAW
===============================================================================

Simple things must remain simple.

Ordinary use:

    using Gesso, Lava

    model = Gesso.load("model")
    session = Gesso.Session(model)
    generate(session, "Hello")

Advanced users may expose:

    policies
    representations
    device capabilities
    execution plans
    scheduling
    swarm execution

Do not require users to understand compiler internals merely to run a model.


===============================================================================
LXVIII. INTERNAL DEBUGGING
===============================================================================

Gesso must expose:

    plan inspection
    IR inspection
    representation inspection
    kernel selection
    materialization report
    scheduler trace
    agent trace
    memory plan
    autotune results

Potential developer UX:

    explain(model)
    explain(plan)
    explain(request)
    explain(materialization)

Gesso should make optimization decisions auditable.


===============================================================================
LXIX. DETERMINISM
===============================================================================

Where requested, deterministic behavior must be first-class.

Record:

    backend
    materialization
    kernel specialization
    seed
    sampling policy
    model digest
    configuration
    autotune cache version

Agent execution will include external nondeterminism, but Gesso should distinguish:

    deterministic internal computation

from:

    external side effects.


===============================================================================
LXX. FAILURE BEHAVIOR
===============================================================================

Gesso must fail explicitly.

No silent:

    representation downgrade
    backend switch
    quantization mismatch
    memory-plan violation
    kernel substitution

unless policy explicitly permits it.

If fallback occurs:

    record it.


===============================================================================
LXXI. SWARM IMPLEMENTATION RULE
===============================================================================

Gesso development will likely involve many coding agents.

Every task should be:

    bounded
    independently testable
    benchmarkable where relevant
    revertible

Agent work item format:

    objective
    permitted files
    interfaces
    invariants
    tests
    performance target
    expected artifact


===============================================================================
LXXII. ENGINEERING RECEIPTS
===============================================================================

Every meaningful optimization should report:

    what changed
    why
    tests
    numerical delta
    before benchmark
    after benchmark
    compile-time impact
    memory impact
    hardware
    workload
    model
    backend

Agent confidence is not evidence.

The harness decides.


===============================================================================
LXXIII. PHASE 0 — REPOSITORY FOUNDATION   [STATUS: COMPLETE 2026-09-28]
===============================================================================

GOAL:
    establish Gesso without inherited project debris.

Tasks:

    clean Project.toml
    package skeleton
    CI
    formatting
    test harness
    benchmark harness
    logging/telemetry conventions
    backend interface draft

Exit:

    package loads
    tests pass
    dependency graph intentional
    no unrelated NIRA / PlasticWeights / WGE dependencies


===============================================================================
LXXIV. PHASE 1 — SEMANTIC CORE
===============================================================================

Encoding is decided (§CIX). This phase implements that encoding.
It does not reopen the object-model question in code.

Build:

    model semantics
    parameter semantics
    operator semantics
    basic semantic IR
    traits
    backend capability abstraction

No GPU heroics required.

Exit:

    tiny reference model expressible entirely through semantic core.


===============================================================================
LXXV. PHASE 2 — REFERENCE EXECUTION
===============================================================================

Status: COMPLETE 2026-09-29 (prefill oracle + known logits + greedy KV
decode; receipt in the Phase 2 close note). Implementation goal:
`docs/goals/PHASE2_CPU_ORACLE.md`.

Build:

    CPU/reference operators
    deterministic model execution
    tokenizer/import fixtures
    logit oracle
    generation harness

Exit:

    toy model forward pass
    known logits
    deterministic generation


===============================================================================
LXXVI. PHASE 3 — FIRST REAL MODEL IMPORT
===============================================================================

Chosen model: HuggingFaceTB/SmolLM2-135M (LlamaForCausalLM).
Implementation goal: `docs/goals/PHASE3_FIRST_IMPORT.md`.

Build:

    config importer
    tokenizer path
    safetensors loading
    parameter mapping
    architecture composition

Exit:

    Gesso logits match reference within declared tolerance.


===============================================================================
LXXVII. PHASE 4 — CUDA.JL EXECUTION
===============================================================================

Status: COMPLETE 2026-09-29 (extension seam + CUDA ops + backend-generic
interpreter; toy2/llama_micro device-vs-oracle gates green on RTX 5060;
SmolLM2-CUDA and the micro-llama CUDA bench row are skip-or-green).
Implementation goal: `docs/goals/PHASE4_CUDA.md`.
CUDA.jl is a package extension, never a core dependency (§VII).

Build:

    device mapping
    parameter transfer
    backend lowering
    basic GPU inference

Exit:

    full generation on NVIDIA GPU
    reference correctness
    baseline performance measurements


===============================================================================
LXXVIII. PHASE 5 — NATIVE INFERENCE ENGINE
===============================================================================

Status: A/B/C/D LANDED 2026-09-30 (paged KV manager — Magenta §9.5 step 1;
Session engine `prefill!`/`decode!`/`generate` matching the oracle — ids
equal, CPU logits atol=0; streaming callback + string prompts + CUDA engine;
llama_micro/SmolLM2 gates skip-or-green). Greedy sampling only; scheduler /
continuous batching is a follow-on goal under this phase. Implementation
goal: `docs/goals/PHASE5_ENGINE.md`.

Build:

    prefill
    decode
    KV cache
    sampling
    scheduler
    streaming
    sessions

Exit:

    Gesso no longer relies on another inference engine.

Magenta step 1 lives here as a paged KV manager
(`docs/research/KV_MEMORY_PROGRAM.md`). Gauge-compiled weights do not
(`docs/research/REPRESENTATION_PROGRAM.md`). Index: `docs/research/README.md`.

Living split (`docs/research/ROADMAP_NOW.md`): 5a is the engine
(`docs/goals/PHASE5_ENGINE.md`). Scheduler / continuous batching is 5b,
a later goal under this phase.


===============================================================================
LXXIX. PHASE 6 — PERFORMANCE OBSERVABILITY
===============================================================================

Instrument everything.

Status: A/B/C/D LANDED 2026-09-30 (one receipt per Session call — prefill/
decode/TTFT timing, token usage, KV bytes derived from the page table,
failure records; Profiling renders stable machine-readable reports; warmed
TTFT/decode bench rows). Attribution, not speed — nothing got faster.
Implementation goal: `docs/goals/PHASE6_OBSERVABILITY.md`.
Do not invent a second result type.

Exit:

    every major latency component attributable
    memory accounting trustworthy
    profiler reports stable


===============================================================================
LXXX. PHASE 7 — FIRST SEMANTIC OPTIMIZATION
===============================================================================

Status: A/B/C/D LANDED 2026-09-30 — the pick is DECLARED CoW + identity
prefix share on the paged KV (Magenta §9.5 steps 2–3): `fork(s::Session)`
is the only share constructor (declaration, never token-match discovery);
`append_kv!` copies only the dirty shared page; full prefix pages stay
aliased forever; `Profiling.unique_kv_bytes` proves the win as a number of
bytes (llama_micro prefill pair: 2048 unique bytes vs 4096 isolated).
Two `generate` calls stay independent; token ids are unchanged.
Implementation goal: `docs/goals/PHASE7_PREFIX_SHARE.md`.

Choose ONE.

Recommended candidates:

    phase-specific prefill/decode realization

or:

    semantic quantization

or:

    workload specialization

Exit:

    reproducible win against equivalent generic execution.

Candidates and sequencing: `docs/research/README.md`.


===============================================================================
LXXXI. PHASE 8 — LAVA BACKEND
===============================================================================

Integrate Lava through backend contract.

Start portable.

Then tune.

Exit:

    same logical model
    same Gesso API
    Lava execution correct on supported Vulkan hardware

Status: COMPLETE 2026-09-30 (§LXXXI items A–D).
    A: Lava as weakdep + GessoLavaExt extension (never a core dep, §VII);
    Gesso.LavaBackend vs Lava.LavaBackend kept distinct (alias KALava);
    no-device construction throws ERR_RESOURCE_LIMIT — no silent fallback
    (§LXX). B: the six implemented ops on LavaArray{Float32} via GPUArrays
    broadcast + Lava mul! (no KA kernels, no SPIR-V); to_device copies
    F64→F32 explicitly; quantize!/dequantize! still decline. C:
    interpreter + Session on Lava match the CPU oracle (ids EXACT,
    logits atol 1e-3); fork aliases prefix pages and CoW works on
    LavaArray with no special casing; src/ untouched; one bench row
    accrued (post-warmup, schema 0.2.0). Exit verified: same logical
    model, same Gesso API, correct on supported Vulkan hardware;
    device-less machines skip by name and CI never requires a device.
    NOT claimed: tuned performance (Phase 9), any speed comparison.


===============================================================================
LXXXII. PHASE 9 — AUTOTUNING
===============================================================================

Build:

    candidate interface
    benchmark runner
    correctness gate
    result cache
    invalidation

Exit:

    at least one operator automatically selects device-specific winning implementation.

Status: COMPLETE 2026-10-01 (§LXXXII items A–D).
    A: mixed-backend to_device survives either load order
    (attach-or-own in GessoCUDAExt.__init__, mirroring the Lava
    pattern); Autotune filled with the generic loop — Candidate /
    TuneResult, register! (registration order is the contract),
    search! (correctness gate BEFORE any timing; compile + warmup
    outside the timed region, §XXXIII), select (a cache hit replays
    the SAME TuneResult), invalidate!, invalidate_all! — core
    machinery that imports neither CUDA nor Lava (§VII; extensions
    register candidates in __init__, never at precompile);
    AUTOTUNE_CACHE_VERSION = v"0.1.0" in src/versions.jl. B: two CUDA
    matmul! candidates — :cublas_mul (the Phase 4 CUBLAS path) and
    :generic_mul (broadcast outer-product accumulation) — registered
    under (:matmul!, :cuda), both gated against the CPU F64 oracle at
    the EXISTING CUDA op atol 1e-2 (no third atol invented); a
    gate-failing candidate is DISQUALIFIED (§LXX) and all-fail throws
    GessoError(ERR_VERIFY_MISMATCH) — never a silent keep, never a CPU
    fallback because search missed. C: the CUDA matmul! op methods
    consult Autotune — cache key (AUTOTUNE_CACHE_VERSION, device
    identity, backend, op, regime); regimes :toy2/:llama_micro are
    named (K, N) buckets, not a cartesian search — and dispatch to the
    winning candidate by name; the first call per key searches, later
    calls hit the cache; every selection emits ONE Receipt through the
    existing sink (task=:autotune_select; no new result type, no
    schema bump); toy2 + llama_micro greedy ids still match CPU
    exactly and logits stay inside the declared atol; one bench row
    accrued (post-warmup, schema 0.2.0). D: maps updated. Exit
    verified: one operator automatically selects a device-specific
    winning implementation and the engine uses it; device-less
    machines skip by name and CI never requires a device. NOT
    claimed: tuned performance, any speed comparison, occupancy
    search, foundry/kernel generation, disk-persisted cache, Lava
    autotuning (all later).


===============================================================================
LXXXIII. PHASE 10 — QUANTIZATION / REPRESENTATION PLANNER
===============================================================================

Build:

    logical/physical separation
    multiple representations
    planner
    conversion
    matching kernels

Exit:

    same logical model materializes into at least two valid execution representations.

Implementation seed: `docs/research/REPRESENTATION_PROGRAM.md`
(index: `docs/research/README.md`). Do not start this phase from musings.


===============================================================================
LXXXIV. PHASE 11 — MEMORY PLANNER
===============================================================================

Build:

    lifetimes
    residency
    workspace planning
    KV accounting
    pressure response

Exit:

    model fitting / memory behavior measurably improved versus naive allocation.


===============================================================================
LXXXV. PHASE 12 — GESSO AGENT RUNTIME
===============================================================================

Build minimal:

    Agent
    Task
    Dependency
    Channel
    Budget
    Priority
    Cancellation
    Receipt

Exit:

    several agents share one model runtime cleanly.


===============================================================================
LXXXVI. PHASE 13 — CROSS-AGENT INFERENCE SCHEDULING
===============================================================================

Integrate agent scheduler and inference scheduler.

Exit:

    mixed multi-agent workload
    automatic batching
    priority enforcement
    measurable utilization improvement


===============================================================================
LXXXVII. PHASE 14 — PALETTE SURFACE
===============================================================================

Add condensed orchestration language.

Palette must lower to Gesso.

Exit:

    simple agent graph expressed compactly
    execution identical to direct Gesso plan


===============================================================================
LXXXVIII. PHASE 15 — CYAN MIGRATION
===============================================================================

Make Cyan first native Gesso harness.

Tasks:

    replace unnecessary Python orchestration
    retain useful Rust services
    establish C ABI where required
    connect Palette
    connect Gesso inference
    benchmark real workloads

Exit:

    Cyan runs meaningfully on Gesso.


===============================================================================
LXXXIX. PHASE 16 — C ABI
===============================================================================

Stabilize libgesso.

Exit:

    external C/C++ program can:
        load model
        create context
        prefill
        decode
        sample
        destroy resources


===============================================================================
XC. PHASE 17 — LLAMA.CPP COMPATIBILITY ADAPTER
===============================================================================

Build adapter only after Gesso API is stable enough.

Exit:

    selected existing software can use Gesso without native rewrite.

Compatibility remains subordinate to Gesso architecture.


===============================================================================
XCI. PHASE 18 — PROFILE-GUIDED SPECIALIZATION
===============================================================================

Build:

    workload signatures
    profiling
    specialization proposal
    benchmark validation
    specialization cache

Exit:

    runtime can improve after observing stable workload behavior.


===============================================================================
XCII. PHASE 19 — PROFILE-GUIDED REMATERIALIZATION
===============================================================================

Allow physical model representation to adapt to observed use.

Exit:

    measurable workload-driven representation change with semantic equivalence.


===============================================================================
XCIII. PHASE 20 — GENERALIZED SPECULATION RESEARCH
===============================================================================

Only now.

Prototype.

Measure.

Kill bad ideas quickly.

No research program in `docs/research/` yet. Do not pull this into Phase 5.


===============================================================================
XCIV. PHASE 21 — CONTRIBUTOR APB: THE TRAINING STACK
===============================================================================

Gesso does not include training. This phase is an open call, not a work item.

    WANTED: A JULIA-NATIVE TRAINING STACK.

    Public position of the Gesso project: if you want to contribute to the Julia ML
    ecosystem and want a job with real scope, build the training stack. Gesso will load
    its checkpoints like everyone else's. The natural place to start is WGPU.jl.

The contract at the seam is already satisfied by standard formats:

    train elsewhere
        ↓
    standard checkpoint formats (safetensors etc.)
        ↓
    Gesso importer (already Gesso's job)
        ↓
    Gesso inference

    checkpoints carry the architecture semantics Gesso imports
    no Gesso-side training dependency, ever
    fine-tunes and LoRA adapters arrive as weights-on-disk deltas (AdapterDelta)

One standing rule for that future: it must REUSE Gesso semantic representations if it
wants inference-time co-design. Do NOT create "Gesso Training Framework 2" beside
Gesso, and do NOT bolt an unrelated training architecture INTO Gesso.


===============================================================================
XCV. PHASE 22 — PLASTIC / LIFECYCLE COMPUTE RESEARCH
===============================================================================

Revisit:

    PlasticWeights
    semantic parameter lifecycle
    FluidGEMM
    consolidation-aware representation
    adaptive storage


===============================================================================
XCVI. SUCCESS CRITERIA — V0
===============================================================================

Gesso can:

    load one real model
    preserve semantics
    run reference-correct inference
    execute through CUDA.jl
    generate text
    profile itself


===============================================================================
XCVII. SUCCESS CRITERIA — V1
===============================================================================

Gesso can:

    support several real model architectures
    run complete inference
    prefill/decode properly
    manage KV
    stream
    schedule
    quantize/materialize
    execute through CUDA and Lava paths
    expose C API
    demonstrate at least one semantic-performance advantage


===============================================================================
XCVIII. SUCCESS CRITERIA — V1.5
===============================================================================

Gesso can:

    orchestrate multiple agents
    batch across agents
    enforce budgets/priorities
    share model resources
    expose Palette orchestration surface
    run Cyan natively


===============================================================================
XCIX. SUCCESS CRITERIA — V2
===============================================================================

Gesso begins demonstrating:

    hardware-aware materialization
    automated kernel selection
    profile-guided specialization
    whole-model memory planning
    workload-specific representation
    materially differentiated execution from PyStack™


===============================================================================
C. NORTH-STAR EXPERIMENT
===============================================================================

Eventually run:

    SAME CHECKPOINT
    SAME PROMPT
    SAME HARDWARE
    SAME QUALITY REQUIREMENT

Compare:

    conventional execution

vs.

    Gesso generic execution

vs.

    Gesso specialized execution

vs.

    Gesso native multi-agent execution

Measure:

    latency
    memory
    throughput
    practical context
    concurrent agents
    total work completed per second


===============================================================================
CI. CENTRAL ARCHITECTURAL QUESTIONS
===============================================================================

For every new abstraction ask:

    Does this preserve meaning?

    Can lower layers exploit that meaning?

    Is logical identity separate from physical representation?

    Can multiple physical realizations exist?

    Is the property stable enough for specialization?

    Can this be measured?

    Can Gesso explain the decision?

    Does this unnecessarily bind Gesso to one hardware vendor?

    Does this unnecessarily bind Gesso to one model architecture?

    Does this unnecessarily bind Gesso to NIRA?

    Does this duplicate machinery Julia already gives us?


===============================================================================
CII. CENTRAL PRODUCT QUESTION
===============================================================================

At every milestone ask:

    "Does this help Gesso execute existing models more effectively?"

If not, determine whether it belongs in V1.


===============================================================================
CIII. CURRENT CORE THESIS
===============================================================================

Gesso is a semantic compiler/runtime for machine learning where:

    model meaning
    parameter meaning
    request meaning
    workload meaning
    agent meaning

survive deeply enough into execution to influence:

    representation
    precision
    memory
    scheduling
    batching
    quantization
    specialization
    kernels
    hardware placement


===============================================================================
CIV. CURRENT DIFFERENTIATOR
===============================================================================

The difference is NOT:

    Julia syntax.

The difference is NOT:

    Vulkan.

The difference is NOT:

    multiple dispatch by itself.

The difference is the COMBINATION:

    Julia multiple dispatch
    semantic model IR
    semantic tensors
    logical/physical separation
    generated specialization
    quantization as lowering
    hardware-aware materialization
    phase-specific inference
    execution-plan synthesis
    whole-model memory planning
    autotuning
    profile-guided rematerialization
    inference-aware agent scheduling
    native multi-agent orchestration

all operating inside one coherent runtime.


===============================================================================
CV. CURRENT ONE-SENTENCE DEFINITION
===============================================================================

    Gesso is a Julia-native semantic ML and agent execution runtime that loads existing
    models, preserves what they mean, and uses that information to determine how they
    should physically exist and execute on the hardware and workload actually present.


===============================================================================
CVI. THE STACK
===============================================================================

For ordinary users:

    using Gesso
    using Lava

That is the stack.

For NVIDIA:

    using Gesso
    using CUDA

Also valid.

For Cyan:

    Cyan
        ↓
    Palette
        ↓
    Gesso
        ↓
    CUDA / Lava


===============================================================================
CVII. FINAL DEVELOPMENT RULE
===============================================================================

DO NOT BUILD THE ENTIRE VISION AT ONCE.

Build the shortest vertical slice that proves the next architectural claim.

The intended progression is:

    RUN ONE MODEL
        ↓
    RUN IT CORRECTLY
        ↓
    UNDERSTAND IT SEMANTICALLY
        ↓
    MEASURE IT
        ↓
    EXECUTE IT WELL
        ↓
    MATERIALIZE IT DIFFERENTLY
        ↓
    PROVE ONE ADVANTAGE
        ↓
    GENERALIZE THE ADVANTAGE
        ↓
    RUN MANY MODELS
        ↓
    RUN MANY AGENTS
        ↓
    OPTIMIZE THE WHOLE MACHINE
        ↓
    THEN DO THE TERRIFYING SHIT


===============================================================================
CVIII. FINAL NORTH STAR
===============================================================================

PyStack™ largely asks:

    "How efficiently can we execute this graph?"

Gesso should ultimately ask:

    "Given what this model is,
     why it is being invoked,
     what hardware exists,
     what constraints matter,
     and what the surrounding agents are trying to accomplish...

     WHAT SHOULD THE MACHINE ACTUALLY DO?"

That is Gesso.


===============================================================================
CIX. OBJECT MODEL ENCODING
===============================================================================

Resolved 2026-09-29. Phase 1 gate. Closes the Foundation Hardening stop line
and decision packets 1 and 2.

The three objects of the semantic core are not one encoding.

Do not pick "immutable values" or "mutable graph handles" or
"trait/metadata layering" for the whole core. Each object uses the
encoding its job requires. §XIII is the allocation rule:

    TYPES      = stable structural identity that dispatch may see
    TRAITS     = optimization-relevant properties that dispatch may see
    METADATA   = volatile facts dispatch must not see


OPERATOR
--------

An Operator is a function. Multiple dispatch is the execution mechanism (§XII).

The operator vocabulary is the set of operation names
(rmsnorm!, rope!, matmul!, embedding_lookup!, ...). Adding an operator
means adding a function and methods, never a graph-node class.

An operator is not a ModelIR node. ModelIR refers to operators by composing
semantic primitives that lower to those functions.


MODELIR NODE
------------

A ModelIR node is an immutable value.

ModelIR is a semantic composition graph of architecture primitives
(embeddings, RMSNorm, RoPE, attention families, FFN families, MoE, ...),
NOT a tensor graph, NOT a per-family runtime (no LlamaRuntime, §VIII).

Identity is structural: same primitive, same children, same logical
parameter bindings ⇒ same node. Rewrites construct a new graph.
Runtime mutation (KV append, workspace fill) does not live on ModelIR nodes.

The importer parses config + parameter map INTO this graph (§VIII).
A new architecture is a new composition, usually not a new node type.


SEMANTICTENSOR / PARAMETER
--------------------------

Three layers, always.

    TYPE     semantic family, closed, slow-changing (§XI):
             ProjectionWeight, KVCache, EmbeddingTable, ExpertWeight,
             FrozenParameter, QuantizedParameter, Activation,
             TemporaryWorkspace, RoutingState, DecodeState, AdapterDelta
             Gradient and OptimizerState remain forbidden (§LVIII).

    TRAITS   optimization-relevant properties:
             frozen, quantized, layout family, representation family,
             decode-hot, sparse, ...
             Holy traits or equivalent. NOT stacked type parameters
             for every axis (that is the compile-latency failure §XIII forbids).

    METADATA runtime-volatile facts:
             batch, sequence length, request count, free memory,
             queue pressure, actual storage pointer, device residency.
             Fields or a small metadata struct. Never type parameters.

Physical bytes are a realization of the logical object (Representation,
Phase 10). Working-state representation is a lowering decision
(KV memory program).


WORKLOAD (Packet 1)
-------------------

Two levels.

    TYPE     PrefillWorkload, DecodeWorkload
             These participate in operator dispatch from Phase 1/2.
             Required by §XII examples and §XXX.

    TAG      §XVI finer categories, as documented symbols, for receipts
             and plans until a lowering actually dispatches on them:

                 prompt_prefill
                 batch1_decode
                 batched_decode
                 long_context
                 structured_tool_output
                 swarm_inference

Do not freeze the six §XVI names as an enum. Promotion of a tag to a
type is allowed when a real method signature needs it; that promotion
is a work item, not a drive-by. Tool/swarm kinds may become types in
their owning phases (12+).


RECEIPT IDENTITY (Packet 2)
---------------------------

Until receipts persist or cross process:

    id is a process-local UInt64, monotonic, atomic
    parent_dependency is a receipt id in the same process
    `task` remains its own field; parent_dependency is not a task id

When the first persistence or cross-process consumer lands:

    bump RECEIPT_SCHEMA_VERSION
    mint a globally unique, time-ordered id at that boundary
    (ULID / UUIDv7 / (host, session, counter) — chosen in that phase's
    work item; the escape hatch is the bump, not a silent reinterpret)

In-memory sinks keep the UInt64. Hybrid is the law: local now, global at
the serialization/swarm boundary. §LXIX: never silently reinterpret old ids.


WHAT PHASE 1 IMPLEMENTS
-----------------------

    semantic family types (§XI list)
    ModelIR immutable primitive nodes + composition
    operator functions for the existing lowering-stub vocabulary
    PrefillWorkload and DecodeWorkload types
    trait hooks sufficient for frozen vs not; no representation lattice yet

WHAT PHASE 1 DOES NOT IMPLEMENT
-------------------------------

    mutable IR handles
    Dict-ontology for semantic roles
    ExecutionPhase mega-enum
    global receipt ids
    LlamaRuntime-shaped types
    Gradient / OptimizerState
    storage as the meaning of a tensor

PHASE 1 EXIT (unchanged, now operational):

    a tiny reference model expressible entirely through this encoding.
