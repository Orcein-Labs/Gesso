# GESSO STACK — EXOTIC CAPABILITY MASTER LEDGER
## Product/Engineering synthesis for Gesso, Palette, and Cyan

    Document class:     living capability mine / product-engineering ledger
    Status:             first draft (parked). Not a Buffy /goal.
    Index:              docs/research/README.md
    Map:                docs/research/ROADMAP_NOW.md
    Speed gate:         docs/research/SPEED_FLOOR.md
                        + docs/goals/PHASE10_SPEED_FLOOR.md
    Promotion filter:   docs/research/CYAN_TRIAL_GESSO_FALLOUT.md
    This file is NOT:   permission to implement Time Machine, Foundry,
                        Magenta, cages, or a Julia fork. Strings, not sprints.

**Purpose:** collect “wait, Julia lets us do *what*?” capabilities without turning them into implementation commitments  
**Primary source basis:** `SPEED_FLOOR.md`, `ROADMAP_NOW.md`, `KV_MEMORY_PROGRAM_part2.md`  

---

# 0. EXECUTIVE SUMMARY

The greater stack should not be pitched as “an ML stack written in Julia.” That describes an implementation choice, not a product advantage.

The stronger product thesis is:

> **Gesso preserves meaning deep enough into execution that the stack can avoid rediscovering structure it already knows, specialize later than conventional systems, and turn repeated probabilistic work into verified reusable computation.**

That thesis appears at three levels:

- **Gesso** decides how model computation, attention memory, representation, residency, and schedule should physically execute.
- **Palette** decides how executable capabilities, artifacts, operators, and live computational state should exist and be reused.
- **Cyan** decides how long-running work, branches, project state, tools, and model effort should be orchestrated over time.

The recurring design pattern is the same:

```text
preserve identity and meaning
        ↓
delay unnecessary commitment
        ↓
materialize only what the current regime needs
        ↓
search among legal alternatives
        ↓
verify against an oracle / behavioral gate
        ↓
measure real cost
        ↓
retain the winner with receipts
```

This is the architectural connective tissue of the stack.

The current evidence base is intentionally narrower than the ambition:

- Gesso has landed identity-based prefix sharing with copy-on-write on the current fixture.
- Gesso has landed Autotune as **selection** among gated CUDA matmul candidates; this is not yet a speed claim.
- CUDA and Lava/Vulkan share the same high-level Session semantics, with Lava currently serving as a portable proof rather than the speed horse.
- The current board constraint explicitly blocks Phase 10+ exotic claims until Gesso has a named-model speed floor and preserves the semantic wins on the fast path.

That means this ledger is **a capability mine, not a roadmap override**.

---

# 1. TAXONOMY

Every idea in this ledger belongs to one of three categories.

## 1.1 Stack-wide meta-primitives

A **meta-primitive** is an architectural rule or reusable mechanism that can appear in Gesso, Palette, Cyan, or all three. It is not tied to one specific subsystem.

Examples:

- identity and provenance as execution information;
- availability separated from residency;
- branch structure as a first-class DAG;
- dynamic granularity;
- search under legality/correctness gates;
- demand-driven materialization;
- receipts and replayable provenance;
- hardware/workload specialization.

These are the deepest source of stack coherence.

## 1.2 Piece-specific primitives

A **piece-specific primitive** belongs primarily to one subsystem because that subsystem owns the relevant semantics.

Examples:

- Gesso owns `LogicalAttentionMemory`, `AttentionMemoryPlan`, native representation execution, KV residency, prefill compilation, and model/runtime lowering.
- Palette owns live executable artifacts, operator state, computational snapshots, capability lifecycle, and user-facing execution control.
- Cyan owns branch intent, project/task semantics, agent traces, workflow orchestration, shared model use, and long-horizon capability reuse.

A primitive may inspire analogues in another layer without literally moving code between layers.

## 1.3 Meta-tools

A **meta-tool** is a user-facing capability that exposes several lower-level primitives through one coherent interaction.

The current six lead candidates are:

1. **Time Machine**
2. **Execution X-Ray**
3. **Execution Contexts**
4. **Schedule Forge**
5. **Live Foundry**
6. **Compiler Laboratory**

These are the most promising “holy shit” demo surfaces because they convert deep runtime/compiler machinery into powers a user can understand.

---

# 2. STATUS LANGUAGE

This ledger uses four product maturity labels.

### `LAUNCH CANDIDATE`
The underlying mechanisms are tractable enough that a bounded, demonstrable version could plausibly ship near the first serious release if evidence supports it.

### `POST-LAUNCH R&D`
Technically credible and strategically interesting, but should not be allowed to destabilize the boring production path.

### `MOONSHOT`
A coherent long-term direction whose full form depends on substantial additional infrastructure, empirical validation, or compiler/runtime co-design.

### `SUPPORTING PRIMITIVE`
Important to the architecture, but not itself a headline user-facing feature.

These statuses are product sequencing labels, not scientific confidence scores.

---

# 3. STACK-WIDE META-PRIMITIVES

## MP-01 — Meaning Before Mechanism
**Status:** SUPPORTING PRIMITIVE  
**Applies to:** Gesso / Palette / Cyan

### Product idea
Keep the semantic fact around long enough that later layers can make better decisions instead of reconstructing intent from low-level artifacts.

### Engineering interpretation
A conventional stack often erases information early:

```text
high-level intent
→ generic graph / generic request / generic tool call
→ optimizer attempts to rediscover structure
```

The Gesso philosophy is closer to:

```text
high-level intent
→ explicit semantic object
→ legal physical alternatives
→ measured lowering
```

Examples already present in the research direction include GQA grouping, decode-hot vs prefill, `fork` identity, page aliasing, representation choice, selection policy, residency, and schedule.

### Why it is exotic
The unusual part is not that a compiler optimizes code. Compilers already do that. The unusual part is preserving domain-specific facts **as first-class execution information** long enough that they can choose structurally different realizations.

### User-facing effect
The user does not configure low-level implementation details. The stack simply behaves as though it “understands what kind of work this is.”

### Economic value
Less duplicated work, fewer unnecessary conversions, fewer generic fallbacks, and a larger search space for optimization without exposing that complexity to the user.

---

## MP-02 — Logical Identity ≠ Physical Representation
**Status:** SUPPORTING PRIMITIVE  
**Applies to:** Gesso / Palette / Cyan

### Product idea
The system should distinguish **what something is** from **how it happens to be stored or executed right now**.

### Gesso expression
`LogicalAttentionMemory` is not the same thing as FP16 KV in HBM. One logical memory object may be shared, compressed, paged, tiered, reconstructed, or moved while preserving its logical identity.

### Palette expression
A logical operator or artifact need not equal one loaded in-memory object. It may exist as source, compiled form, cached result, snapshot, remote artifact, or reconstructible state.

### Cyan expression
A task fact, project doctrine item, branch state, or tool observation can remain part of the project without being injected into the model’s active context.

### Why it is exotic
It creates room for aggressive physical optimization while keeping semantics stable.

### Product translation
> “Your world can stay large without forcing every byte, tool, or memory into the expensive active working set.”

---

## MP-03 — Identity and Provenance as Execution Information
**Status:** LAUNCH CANDIDATE / SUPPORTING PRIMITIVE  
**Applies to:** full stack

### Product idea
If the system already knows that two states share a common origin, do not throw that information away and later rediscover similarity heuristically.

### Gesso expression
`fork` constructs shared prefix identity directly. Copy-on-write begins only after divergence.

### Palette expression
Derived workspaces, experiment branches, and computational worlds can point back to one common ancestor.

### Cyan expression
Several agents or investigations can explicitly descend from the same project/task state.

### Why it is exotic
Generic systems often see several independent requests and later perform matching, caching, or deduplication. An identity-aware system sees the relationship **before** execution begins.

### Economic value
Reduced duplicated memory, reduced setup cost, cheaper branching, and potentially shared warm state.

### Current evidence
The present Gesso fixture measures 4096 unique KV bytes for isolated execution versus 2048 after `fork` for the pair. This is a real semantic storage win, but not yet a named-model board benchmark.

---

## MP-04 — Branches Are a Work DAG, Not Merely Copies
**Status:** POST-LAUNCH R&D  
**Applies to:** Gesso / Palette / Cyan

### Product idea
Once branch ancestry is explicit, the scheduler can treat a branch tree as a structured computation graph.

### Engineering interpretation
Instead of:

```text
request A
request B
request C
```

use:

```text
        shared prefix/state
          /      |      \
         A       B       C
```

Independent children can execute concurrently while shared parent state remains deduplicated.

### Potential composition
Branches may eventually share:

- hot prefix pages;
- selection/index state;
- prefill products;
- immutable compressed backing;
- residency hints;
- Palette artifact snapshots;
- Cyan project/tool state.

### User-facing effect
> “Try five approaches” becomes structurally cheaper than cloning five independent agents.

### Economic value
This directly attacks the cost curve of parallel agents, red-team branches, planning trees, and alternative implementation experiments.

---

## MP-05 — Availability ≠ Residency
**Status:** POST-LAUNCH R&D, with immediate architectural value  
**Applies to:** full stack

### Product idea
Something can remain legally available to the system without occupying the most expensive active memory tier.

### Gesso expression
A very large logical attention history may remain selectable while only a bounded working set sits in HBM.

### Palette expression
A large library of operators, datasets, artifacts, and compiled forms can remain available without all being loaded.

### Cyan expression
Months of project state can remain addressable while the model sees only the current working set.

### Why it is exotic
Most AI applications conflate “might need this” with “put this in prompt / memory / process now.” That scales poorly.

### User-facing effect
> “The system can know about far more than it has actively loaded.”

### Economic value
Potential reductions in VRAM pressure, context-token cost, startup cost, and active memory footprint.

---

## MP-06 — Demand-Driven Materialization
**Status:** POST-LAUNCH R&D  
**Applies to:** full stack

### Product idea
Keep cheap selection/index state hot; materialize expensive state only after the system knows it is needed.

### Gesso origin
The Magenta “key-first, value-late” hypothesis distinguishes state used to decide **what to read** from the heavier state needed **after selection**.

### Palette analogue
Keep compact operator metadata, names, signatures, dependency summaries, and indexes hot; load full implementations/artifacts only on demand.

### Cyan analogue
Keep lightweight summaries/indices of project history hot; retrieve full logs, observations, or branches only after relevance is established.

### Why it is exotic
This is not ordinary lazy loading. The selection stage itself becomes an explicitly optimized computational object with its own fidelity and cost model.

### Economic value
Lower memory use, lower token movement, lower tool-loading cost, and a path to huge logical workspaces with bounded active state.

---

## MP-07 — Heat / Use-Driven State Management
**Status:** POST-LAUNCH R&D  
**Applies to:** full stack

### Product idea
Manage state by actual demand, not only age.

### Engineering interpretation
A very old object can become hot again if repeatedly selected. A recent object can become cold if it is never reused.

### Gesso expression
Attention heat may drive residency, precision, compression effort, retention, and prefetch priority.

### Palette expression
Frequently reused operators/artifacts stay compiled and hot; rarely used ones demote.

### Cyan expression
Frequently consulted project facts or recurring procedures stay near the active working set.

### Product translation
> “The system keeps what you are actually using close, not merely what happened recently.”

---

## MP-08 — Dynamic Granularity
**Status:** POST-LAUNCH R&D  
**Applies to:** full stack

### Product idea
Do not force every decision to operate at one fixed unit size.

### Gesso examples
Token-, block-, page-, segment-, or hybrid selection may be appropriate under different workload and hardware conditions.

### Palette examples
Load or operate on a symbol, function, file, module, artifact group, or whole workspace depending on the task.

### Cyan examples
Retrieve one fact, one observation, one branch episode, one workflow summary, or an entire trace depending on the question.

### Why it is exotic
The system treats granularity itself as a tunable/searchable variable rather than a fixed architectural commitment.

### Economic value
Better tradeoffs between precision and batching, quality and overhead, or retrieval exactness and movement cost.

---

## MP-09 — Search Under Legality / Correctness Gates (RPDO)
**Status:** LAUNCH FOUNDATION; POST-LAUNCH expansion  
**Applies to:** full stack

### Product idea
The system may search among alternatives, but faster/cheaper never automatically means acceptable.

### Canonical loop

```text
candidate
→ legality / correctness gate
→ measurement
→ winner
→ cache
→ invalidate when regime changes
```

### Gesso today
Autotune already performs this loop for CUDA matmul candidates under a correctness gate. The next natural step is the fused decode step, then representations, schedules, and more complex plans.

### Palette future
Candidate operator implementations or generated computational routines can be tested against behavioral probes before becoming trusted reusable capabilities.

### Cyan future
Candidate workflows/playbooks can be compared on task success, cost, latency, and reliability before promotion.

### Why it is exotic
It enables a system to become structurally adaptive **without trusting unverified self-modification**.

### Product translation
> “The system can try better ways to do recurring work, but only keeps them if they pass tests.”

---

## MP-10 — Receipts as a First-Class Product Primitive
**Status:** LAUNCH CANDIDATE / FOUNDATION  
**Applies to:** full stack

### Product idea
Every optimization, capability promotion, replay, branch, or generated execution form should carry provenance sufficient to explain what happened.

### Gesso expression
Timing, memory, failure, candidate identity, backend, regime, and correctness information become attributable receipts.

### Palette expression
An artifact or operator can carry source, environment, dependencies, test evidence, and version identity.

### Cyan expression
A workflow can preserve evidence → interpretation → decision → implementation → validation lineage.

### Why it is exotic
Receipts turn aggressive adaptive behavior from “magic” into something inspectable, debuggable, and enterprise-credible.

### Economic value
Lower debugging cost, better reproducibility, stronger trust, and easier compliance/audit narratives.

---

## MP-11 — Hardware / Workload Fingerprints
**Status:** SUPPORTING PRIMITIVE  
**Applies to:** Gesso / Palette / Cyan

### Product idea
There is no universally optimal implementation. Cache decisions against the actual regime.

### Gesso regime dimensions
Device, backend, operator, model shape, representation, batch, share-tree, memory pressure, and execution phase.

### Palette/Cyan analogues
Project type, operator family, artifact size, concurrency, model choice, latency budget, local/remote execution, and task class.

### User-facing effect
The same high-level command can execute differently on a laptop, workstation, or accelerator without requiring the user to hand-tune it.

---

## MP-12 — Separate First-Use Cost from Steady-State Cost
**Status:** LAUNCH FOUNDATION  
**Applies to:** full stack

### Product idea
Never collapse compilation/setup latency and steady-state throughput into one misleading number.

### Gesso rule
Time-to-first-X is a compiler/load problem. Warm decode throughput is a kernel/fusion/transfer problem.

### Palette/Cyan analogue
Cold-starting a tool, loading an artifact, compiling an operator, or reviving a workspace is a different economic event from repeated hot execution.

### Product value
Makes optimization actionable and prevents misleading demos.

---

# 4. GESSO-SPECIFIC PRIMITIVES

## G-01 — `LogicalAttentionMemory + AttentionMemoryPlan`
**Status:** POST-LAUNCH R&D / core Magenta thesis

### What it does
Separates the identity and semantics of attention memory from one physical realization.

The logical object owns facts such as identity, provenance, model semantics, branch ownership, and approximation constraints.

The plan owns representation, producer/consumer topology, sharing, selection, residency, movement, prefill products, working set, and hardware/workload fingerprint.

### Why it matters
It turns “KV cache” from a tensor buffer into a compilation target.

### Exotic angle
The runtime may search over multiple lawful physical realizations of the same logical memory object and choose the cheapest verified one for a particular device and workload.

### Product/economic interpretation
A single checkpoint can potentially execute under different memory strategies without exposing that complexity to the user.

---

## G-02 — Native Representation Execution
**Status:** POST-LAUNCH R&D

### What it does
Allows kernels to consume Gesso-native representations directly instead of reconstructing conventional dense tensors merely because traditional kernels expect them.

### Why it is exotic
Representation search becomes valuable only if the execution stack can honor the winning representation rather than paying conversion tax back to the conventional form.

### Product effect
Users may get lower memory use or different performance/quality operating points while invoking the same high-level model interface.

### Proof requirement
A candidate representation must preserve the relevant oracle/quality contract and beat the conventional path end-to-end, not only in an isolated codec benchmark.

---

## G-03 — Prefill as Compilation
**Status:** POST-LAUNCH R&D

### What it does
Treats prefill as the phase that constructs durable decode-time execution products, not merely KV tensors.

Potential products include:

- shared-prefix identity;
- KV pages;
- hidden-state bridge sources;
- sparse-selection metadata;
- compact index state;
- page statistics;
- host backing state;
- initial working set;
- prefetch hints;
- reconstruction handles.

### Why it is exotic
Prompt processing becomes a compilation stage that can generate a reusable execution package for decode.

### Product effect
Long-running sessions and agent traces may amortize more work up front and avoid rebuilding equivalent state repeatedly.

---

## G-04 — Multi-Level Reuse
**Status:** POST-LAUNCH R&D

### What it does
Distinguishes several kinds of “sharing” that conventional discussions often collapse together:

- horizontal identity sharing;
- vertical KV reuse;
- representation bridging;
- selection sharing;
- physical deduplication.

### Why it is exotic
These mechanisms can compose rather than compete. One shared agent prefix could simultaneously be horizontally shared, vertically reused, sparsely selected, partially resident, and represented differently by age.

### Economic value
Compounding reductions in duplicate storage, selection work, data movement, and repeated projection/attention work.

---

## G-05 — Branch-Aware Memory Execution
**Status:** POST-LAUNCH R&D

### What it does
Uses known branch topology to coordinate working sets and execution rather than treating branches as unrelated requests.

### Potential advantage
Planning/red-team branches become cheaper not only to **store**, but potentially cheaper to **keep warm**.

### Product effect
Parallel agent exploration becomes more economically viable.

---

## G-06 — Shared Model Runtime / Shared Battery
**Status:** ROADMAP PRIMITIVE

### What it does
Allows many higher-level agents or sessions to consume one model/runtime infrastructure rather than cloning the battery per agent.

### Product effect
The mental model becomes:

> one expensive intelligence source, many lightweight structured processes around it.

### Economic value
Potentially much lower marginal cost per additional agent/process, especially when combined with prefix sharing and branch-aware memory.

---

## G-07 — Vendor-Neutral `Session`
**Status:** LANDED SEAM / SUPPORTING PRIMITIVE

### What it does
Keeps the high-level model/session contract stable across CUDA and Lava/Vulkan backends.

### Why it matters commercially
The product is not required to present itself as “NVIDIA or nothing,” even if CUDA remains the speed horse during the current optimization phase.

### Important limitation
Portability is currently a semantic seam, not evidence of equal performance.

---

## G-08 — Fail-Closed Numerical Optimization
**Status:** LANDED FOUNDATION

### What it does
Disqualifies a candidate that violates the numerical/semantic oracle even if it is faster.

### Why it is exotic
This allows aggressive search without making correctness negotiable by default.

### Product translation
> “Optimization is allowed to be inventive. It is not allowed to silently change the answer.”

---

## G-09 — Compile-vs-Execute Attribution
**Status:** SPEED-FLOOR REQUIREMENT

### What it does
Separates source/package load, inference, specialization, LLVM/native emit, GPUCompiler/SPIR-V/pipeline work, GPU execution, and data movement.

### Why it matters
It prevents “Julia is slow” from being treated as one undifferentiated problem and prevents the team from forking compilers to solve what is actually an application/kernel issue.

### Product value
Better engineering prioritization and more credible benchmark reporting.

---

## G-10 — Live Method Table over Live Session (state-preserving rewire)
**Status:** POST-LAUNCH R&D (Lane B). After speed floor. After fused decode
is a legal Autotune candidate. The model **proposes**; the oracle **installs**.
**Origin:** Julia hot-reload (Revise-class method overwrite, live dispatch
table, same address space as Session/KV). First working name:
RSI-adjacent — adjacent because a live install can *look* like the
runtime improving itself; the mechanism is hot-reload plus a gate,
not weight evolution.

### The idea

Julia keeps the compiler, the method table, and `Session` / paged KV in
**one address space**. A C++/Python inference server is a frozen graph
behind a process boundary: a better strategy stays in prompt text, or
you restart and deserialize. Here, a new method can land while the KV
pages stay resident.

Three real moves, in order:

1. **Self-specializing kernels.** Hardware counters + regime
   (shape, decode-hot, share-tree) → a new candidate for this session's
   actual memory shape → Autotune gate (ids / atol) → method table
   overwrite → rest of the session runs the winner. This is Schedule
   Forge / RPDO with a *live* install, not a process restart.
2. **Collapse repeated thought into a function.** When Cyan/Palette
   sees a structured procedure fire the same way (C-04), compile it to
   a typed Julia method and hot-swap the call site. The model stops
   re-deriving a deterministic loop in tokens. That is workflow
   promotion, not "the LLM becomes LLVM."
3. **State-preserving rewire.** Inner routing, token selection, even
   layout hooks can change **if** the new code honors the live
   storage contract. KV stays in VRAM. Layout-breaking rewires are
   illegal without a migration receipt.

### Why it is exotic

PyTorch/vLLM can autotune *offline* or recapture a graph. They do not
casually replace a method under a live KV without a restart story.
Julia's dispatch table is already a live, type-indexed rule engine
(Mordant). Gesso already refuses silent kernel substitution (§LXX).
The combination is: **hot-swap is legal, silent hot-swap is not.**

### What Gemini inflated

"Recursive self-improvement," "self-evolving cognitive engine,"
"bakes learned behaviors into the execution binary," "direct LLVM IR
injection mid-decode," the LLM rewriting architecture without a gate.
Those are the hype wrapper. Training/weight evolution is still out of
this package. A generated kernel that fails greedy ids is
`ERR_VERIFY_MISMATCH`, not an upgrade.

### Product translation

> "The session never restarts to get faster or smarter about *this*
> workload. New code arrives; the cache stays; the oracle still owns
> the answer."

User-facing, this **looks like systems-level RSI**: the running engine
rewrites its own execution layer mid-task and gets cheaper/sharper
without dropping context. It is not model-level RSI (weights do not
train). The honest pitch names both: what it looks like, and what
moved (methods, not parameters).

### Minimal credible demo (after G1∧G2∧G3)

Hold a CUDA Session with a warm KV. Install a second legal `matmul!`
or decode-step candidate mid-generate. Unique KV bytes unchanged.
Greedy ids unchanged. Subsequent tokens use the new method. Receipt
names the swap. That is the whole trick.

### Main risks

- layout-changing rewires that keep the old KV (silent corruption)
- specialization explosion (one method per prompt length)
- LLM-authored kernels skipping the oracle
- Revise/eval invalidating the world you meant to keep

---

# 5. PALETTE-SPECIFIC PRIMITIVES

> The items in this section are product/architecture syntheses from the capability discussion. They are not claims that the uploaded Gesso research documents already specify Palette APIs.

## P-01 — Persistent Executable World
**Status:** LAUNCH CANDIDATE

### What it does
Treats a Palette session/project as a persistent computational world rather than a disposable REPL/chat exchange.

The world can contain artifacts, operators, variables, compiled forms, probes, snapshots, and provenance.

### Why it is exotic
It gives AI work a durable substrate where repeated reasoning can turn into persistent executable structure.

### Product effect
The user returns to a live project rather than reconstructing context and tooling every session.

---

## P-02 — Capability Lifecycle
**Status:** LAUNCH CANDIDATE → POST-LAUNCH expansion

### What it does
Separates ephemeral, project-local, reusable/probationary, and canonical capabilities.

Generated or newly discovered operators do not immediately become permanent global tools.

### Why it matters
It creates a safe path from “the model improvised something useful” to “this is now a trusted system capability.”

### Product effect
Palette gradually becomes more capable around a project without becoming an uncurated junk drawer.

---

## P-03 — Demand-Loaded Operator Surface
**Status:** LAUNCH CANDIDATE

### What it does
Keeps lightweight descriptions/indexes of many tools and artifacts available while loading full implementations only when needed.

### Product effect
Palette can expose a very broad capability surface without flooding the model prompt or process memory with every possible tool.

### Economic value
Lower context/tool-description cost and better scalability of the toolbox.

---

## P-04 — Snapshot / Derived World Identity
**Status:** LAUNCH CANDIDATE

### What it does
Creates cheap derived computational worlds that preserve ancestry.

### Product effect
A user can fork an experiment, test a dangerous change, compare alternatives, and return to an earlier world without manually rebuilding state.

### This primitive underpins
Time Machine, Execution Contexts, Live Foundry, and branch-aware Cyan workflows.

---

# 6. CYAN-SPECIFIC PRIMITIVES

> As with Palette above, these are product/architecture syntheses built from the current stack direction and discussion.

## C-01 — Agent Trace as the Real Workload Unit
**Status:** LAUNCH CANDIDATE / R&D FOUNDATION

### What it does
Optimizes around real long-running task traces rather than isolated prompts or kernel microbenchmarks.

A useful trace includes:

- inputs/outputs;
- observation/action ratio;
- branch structure;
- shared prefixes;
- context age distribution;
- repeated-access locality;
- tool use;
- validation outcomes.

### Why it matters
Agent workloads are structurally nonuniform. Their cost is often dominated by long observations, repeated state retrieval, branching, tool outputs, and revisiting old evidence.

---

## C-02 — Semantic Constraints / Branch Intent
**Status:** LAUNCH CANDIDATE

### What it does
Cyan states what project memory or branch state **means**; Gesso decides how it physically lives.

Examples:

- doctrine;
- task state;
- reconstructible state;
- disposable observations;
- branches sharing a common prefix.

### Why it matters
This keeps semantic policy above the tensor/runtime layer while still allowing Gesso to exploit that policy physically.

---

## C-03 — Shared Battery Orchestration
**Status:** POST-LAUNCH R&D

### What it does
Runs many structured agents/processes over one shared model runtime where possible.

### Product effect
Multiple agents become more like lightweight processes sharing an expensive intelligence engine than fully independent cloned stacks.

### Economic value
Potentially reduces the marginal compute/memory cost of agent concurrency.

---

## C-04 — Workflow Promotion / RPDO for Work
**Status:** POST-LAUNCH R&D

### What it does
Repeated workflows can produce candidate playbooks or execution strategies; Cyan verifies them against task evidence and promotes only those that repeatedly work.

### Product translation
> “The system stops making the model rediscover the same deterministic procedure every time.”

### Economic value
Fewer model calls, fewer repeated tokens, lower latency, and higher consistency on recurring work.

---

## C-05 — Active Working Set Over Long-Horizon Project Memory
**Status:** POST-LAUNCH R&D

### What it does
Keeps the full project logically addressable while presenting the model only the subset required for the current decision.

### Why it matters
This is the Cyan-level version of availability ≠ residency.

### Product translation
> “The project can grow for months without every turn becoming more expensive.”

---

# 7. META-TOOLS — USER-FACING EXOTIC CAPABILITIES

# MT-01 — TIME MACHINE
**Primary layer:** Palette  
**Secondary consumers:** Cyan, Gesso debugging  
**Status:** LAUNCH CANDIDATE (bounded version)

## User power
> **“Take me back to exactly where this computation/project was before it went wrong.”**

Not merely undo text. Not merely git checkout. Not merely rerun a notebook.

The target experience is reconstructing a *computational world* with enough identity to answer:

- what source/artifacts existed;
- what operator versions existed;
- what model/runtime state mattered;
- what branch we were on;
- what inputs had arrived;
- what execution context was active;
- what receipts led to the current result.

## Why this is exotic
Most developer environments preserve source history but not a coherent history of the **live execution world**. Traditional debuggers can step through one run; version control can recover files; notebooks can preserve cells. Time Machine attempts to unify executable provenance at the project level.

Julia is relevant because the language/runtime already exposes unusually rich program identity, dispatch, methods, IR, and specialization surfaces. Palette adds the missing product semantics: snapshots, artifact identity, project worlds, and controlled restoration.

## Product-manager framing
This is not sold as “world age” or “method table replay.”

It is sold as:

> **Rewind the computational environment, not just the file.**

## Economic value
- lower debugging/reproduction time;
- fewer “works on my machine” state mysteries;
- cheaper agent recovery after a bad branch;
- safer experimentation;
- stronger auditability.

## Minimal credible demo
1. Start a live Palette project.
2. Load data and define operators.
3. Execute a successful analysis.
4. Introduce a breaking change and mutate the environment.
5. Select a prior checkpoint.
6. Restore enough of the world to rerun the earlier result identically.
7. Show receipts proving which code/artifact/operator identities were restored.

## Main risks
- mutable external resources cannot always be rewound;
- GPU/device state may need reconstruction rather than byte-perfect snapshots;
- package/environment drift;
- world-age semantics and invalidation complexity;
- storage explosion if snapshots are naive copies.

## Important ownership rule
The Gesso roadmap explicitly says **Julia-world snapshots do not belong inside Gesso**. That is a strong design clue: Time Machine belongs in Palette/Cyan above the inference substrate.

---

# MT-02 — EXECUTION X-RAY
**Primary layer:** Palette  
**Secondary layer:** Gesso  
**Status:** LAUNCH CANDIDATE

## User power
> **“Show me what the machine actually did, and why.”**

A user starts from a semantic operation and drills downward:

```text
user operation
→ Palette operator / Cyan action
→ Gesso semantic plan
→ dispatch choice
→ specialization
→ lowered IR
→ kernel(s)
→ memory movement
→ device execution
→ measured receipt
```

## Why this is exotic
Most profilers begin low in the stack and force users to infer the semantic reason for an event. Execution X-Ray goes both directions:

- **top-down:** why did this semantic operation produce these machine actions?
- **bottom-up:** what user/model operation caused this kernel, allocation, compilation, or transfer?

The exotic capability is **semantic observability across abstraction boundaries**.

## Product-manager framing
> “Click any expensive operation and trace it from intent to silicon.”

## Economic value
- dramatically faster performance diagnosis;
- fewer specialist hours required to understand GPU/runtime behavior;
- easier optimization prioritization;
- stronger customer trust in adaptive execution.

## Minimal credible demo
Run one inference/session operation and present:

- operator identity;
- selected implementation;
- compile time;
- execute time;
- GPU kernels;
- HBM/host movement;
- branch/share state;
- fallback status;
- correctness receipt.

Then click one kernel and navigate back to the semantic operation that caused it.

## Main risks
- instrumentation overhead;
- information volume;
- unstable compiler internals;
- exposing details without a coherent semantic hierarchy becomes profiler noise.

## Why it should probably ship early
Gesso already requires compile-vs-execute attribution and already has receipts/profiling as architectural foundations. X-Ray turns existing engineering instrumentation into a differentiated user product rather than creating an entirely separate research program.

---

# MT-03 — EXECUTION CONTEXTS
**Primary layer:** Palette  
**Secondary layer:** Cyan / Gesso experimental hooks  
**Status:** LAUNCH CANDIDATE → POST-LAUNCH expansion

## User power
> **“Run the same computation under a different set of execution laws without rewriting it.”**

Potential contexts include:

- trace everything;
- exact/reference mode;
- fault-injection mode;
- alternate backend;
- alternate operator implementation;
- memoized mode;
- resource-budgeted mode;
- deterministic replay mode;
- experimental scheduler;
- debugging arithmetic;
- safety/verification instrumentation.

## Why this is exotic
Instead of scattering flags and conditional branches throughout application code, execution policy becomes composable context around otherwise ordinary operations.

Julia’s dispatch and metaprogramming make this style unusually expressive; Palette gives it a stable user-facing abstraction.

## Product-manager framing
> “Change how the machine behaves without changing what your program means.”

## Economic value
- lower cost of experiments;
- easier A/B testing;
- safer debugging;
- reduced code duplication;
- one reproducible mechanism for testing hardware/backend/policy alternatives.

## Minimal credible demo
Run the same Palette operator under:

1. normal context;
2. reference/exact context;
3. fully traced context;
4. alternate backend or implementation context.

Show identical semantic outputs where required and different receipts/execution behavior underneath.

## Main risks
- context interactions can become combinatorial;
- hidden execution policy is dangerous if not visible in receipts;
- dispatch ambiguity / specialization explosion;
- users need a clear hierarchy when contexts conflict.

---

# MT-04 — SCHEDULE FORGE
**Primary layer:** Gesso engine, surfaced through Palette  
**Secondary consumer:** Cyan  
**Status:** POST-LAUNCH R&D

## User power
> **“The runtime can compile a different execution strategy because this workload is structurally different.”**

This is the user-facing form of the core exotic Gesso thesis.

## What it does
Given semantic facts such as:

- model topology;
- GQA grouping;
- prefill vs decode;
- representation;
- device;
- batch size;
- branch/share topology;
- memory pressure;
- attention-memory plan;

Gesso can choose or generate a structurally different schedule rather than merely selecting a different kernel parameter.

## Why this is exotic
Most autotuning is parameter search over an already chosen execution structure. Schedule Forge makes the **structure itself** part of the search space.

Possible candidates might differ in:

- fusion boundaries;
- page-table-aware vs gather-based attention;
- branch concurrency;
- stream layout;
- sparse/dense path;
- movement schedule;
- selection granularity;
- prefetch structure;
- representation-native kernels.

## Product-manager framing
> “Gesso doesn’t only tune kernels. It can specialize the execution plan for the actual model, hardware, and workload.”

## Economic value
Potential gains in latency, throughput, memory, host-device traffic, and agent branch concurrency.

## Minimal credible demo
Take two clearly different semantic regimes—e.g. single decode vs four-way shared-prefix branch—and show:

1. same high-level API;
2. different compiled schedules;
3. both pass oracle gates;
4. each wins in its intended regime;
5. receipts make the difference inspectable.

## Main risks
- compilation latency;
- schedule-space explosion;
- specialization bloat;
- difficult invalidation;
- measurement noise;
- the search can overfit to microbenchmarks rather than end-to-end traces.

## Sequencing rule
The current speed-floor document correctly puts this *after* fast ordinary execution. A clever schedule racing a slow generic baseline proves little.

---

# MT-05 — LIVE FOUNDRY
**Primary layer:** Palette + Gesso search/compiler surfaces  
**Secondary consumer:** Cyan  
**Status:** MOONSHOT

## User power
> **“Generate a better implementation for this recurring operation, prove it, and introduce it into the live system without destroying the project state.”**

## What it does
The foundry creates structurally different candidate implementations or schedules, runs them through legality/correctness gates, benchmarks them, retains Pareto winners, and publishes approved variants at controlled execution boundaries.

## Why this is exotic
This crosses a line from tuning existing implementations to **manufacturing new executable machinery**.

The important safety architecture is that the generator does not get unilateral authority:

```text
generate candidate
→ compile in isolation
→ oracle / behavioral tests
→ benchmark
→ inspect receipt
→ publish only at legal boundary
```

## Product-manager framing
> “The runtime can manufacture and certify specialized tools for the work it repeatedly encounters.”

## Economic value
If successful, recurring expensive model reasoning or generic code paths can become cheaper deterministic machinery.

This is where the stack-level thesis becomes particularly powerful:

> **Turn repeated inference into reusable computation.**

## Minimal credible demo
A bounded operator family is ideal:

1. begin with a generic correct implementation;
2. observe a recurring shape/regime;
3. generate two specialized variants;
4. reject one on correctness;
5. benchmark the survivor;
6. publish it to the persistent worker;
7. subsequent calls dispatch to it;
8. X-Ray shows the provenance and promotion receipt.

## Main risks
- compiler/runtime safety;
- code-size explosion;
- unbounded search cost;
- benchmark gaming;
- difficult rollback;
- generated code security;
- world-age and live-method publication semantics;
- users may overestimate “self-improvement” if the bounded scope is not communicated clearly.

---

# MT-06 — COMPILER LABORATORY
**Primary layer:** Palette power-user surface  
**Engine:** Julia/Gesso/Lava compiler stack  
**Status:** POST-LAUNCH R&D, likely internal-first

## User power
> **“Treat the compiler as an inspectable experimental instrument rather than an opaque implementation detail.”**

## What it does
Expose controlled views and experiments over:

- lowered Julia IR;
- inferred/typed IR;
- specialization decisions;
- LLVM/native output;
- GPUCompiler output;
- SPIR-V/Lava pipeline output;
- invalidation/compile attribution;
- candidate lowerings;
- experimental compiler passes.

## Why this is exotic
Julia already makes compiler/runtime structure unusually accessible to language-level tooling. The product opportunity is to turn that raw accessibility into a safe, coherent ML/systems workbench.

## Product-manager framing
> “A compiler workbench for people building new AI execution machinery.”

This is **not** a mainstream feature. It is a magnet for advanced systems, ML compiler, scientific computing, and performance users.

## Economic value
- accelerates internal R&D;
- creates a power-user moat;
- makes third-party experimentation around Gesso/Lava easier;
- can seed community-contributed optimizations.

## Minimal credible demo
Select one operator, inspect its path through Julia IR → GPU lowering → emitted device code, apply one bounded transformation, compile both variants, oracle-gate them, and benchmark the difference.

## Main risks
- unstable compiler APIs;
- very small expert user base;
- large QA surface;
- temptation to maintain a compiler fork before evidence warrants it.

## Sequencing rule
The speed-floor document is explicit: exhaust stock Julia, GPUCompiler, and Lava-emitter opportunities before maintaining a Julia compiler fork. Compiler Laboratory should reinforce that discipline, not undermine it.

---

# 8. ADJACENT PRODUCT CAPABILITIES THAT MAY BECOME META-TOOLS LATER

These are currently better treated as primitives or sub-features until evidence shows they deserve independent product surfaces.

## A-01 — Branch Lab
A visual/interactive surface for forking project/agent state, measuring shared cost, and comparing divergent outcomes.

Could eventually expose branch economics directly:

- shared bytes;
- shared context;
- unique bytes;
- duplicated compute avoided;
- concurrent execution;
- merge/re-entry cost.

## A-02 — Memory Planner Console
A power-user surface for inspecting logical vs resident state, hot/cold movement, selector behavior, and attention-memory plans.

Likely valuable for ML researchers before ordinary users.

## A-03 — Workflow Forge
The Cyan-level analogue of Schedule Forge: search among alternative workflow/team structures for recurring work.

Examples:

- serial investigator;
- parallel red-team branches;
- one researcher + one verifier;
- retrieve-first vs inspect-first;
- model-heavy vs tool-heavy execution.

This should remain a future idea until task-level evaluation is strong enough to gate it.

---

# 9. ECONOMIC MODEL OF THE STACK

The stack’s commercial thesis should be expressed in units buyers understand.

Traditional AI-product optimization tends to focus on:

- cheaper model;
- fewer tokens;
- faster inference;
- fewer model calls.

This stack potentially adds a fifth axis:

> **less repeated cognitive work.**

A recurring operation can move through a lifecycle:

```text
first encounter
→ model reasons through it

repeated encounter
→ system identifies structure

mature recurring case
→ verified operator / schedule / playbook handles more of it
→ model is reserved for the uncertain part
```

The economic metrics should therefore include more than tok/s:

- tokens per completed task;
- model calls per completed task;
- wall-clock per completed task;
- GPU-seconds per completed task;
- bytes moved;
- HBM peak;
- duplicated bytes avoided;
- duplicate computation avoided;
- cold-start vs warm execution cost;
- recovery/replay time;
- branch marginal cost;
- human debugging time;
- percentage of repeated work served by verified reusable machinery.

The strongest plain-English pitch remains:

> **Stop paying the model to rediscover what the system already knows.**

---

# 10. PRODUCT STORY BY LAYER

## Gesso
**User promise:** “Run the model correctly, then exploit every lawful fact about the model, memory, hardware, and workload to execute it more intelligently.”

Core exotic direction:

- semantic lowering;
- representation search;
- attention-memory compilation;
- branch-aware execution;
- schedule specialization;
- generated kernels/schedules;
- compiler/runtime co-optimization.

## Palette
**User promise:** “Your AI workspace is a persistent executable world, not a pile of prompts and disposable tool calls.”

Core exotic direction:

- snapshots;
- executable provenance;
- demand-loaded capabilities;
- operator lifecycle;
- execution contexts;
- Time Machine;
- Execution X-Ray;
- Live Foundry;
- Compiler Laboratory.

## Cyan
**User promise:** “Long-running AI work becomes an adaptive process that keeps project structure, branches, tools, and repeated procedures under control.”

Core exotic direction:

- shared battery;
- project working-set management;
- branch intent;
- agent-trace optimization;
- workflow promotion;
- demand-loaded history;
- eventually workflow-level execution search.

---

# 11. WHAT SHOULD BE MARKETED VS WHAT SHOULD STAY ENGINEER PORN

## Product-manager surface
Use language like:

- rewind the computational environment;
- trace intent to silicon;
- switch execution laws without rewriting code;
- keep huge projects available without loading everything;
- branch AI work without cloning the entire world;
- turn repeated inference into reusable computation;
- specialize execution to hardware and workload automatically.

## Engineer surface
Under the hood, talk about:

- multiple dispatch;
- specialization boundaries;
- Julia IR;
- GPUCompiler;
- SPIR-V;
- page aliasing;
- copy-on-write;
- share-trees;
- logical vs physical memory;
- selector reuse;
- demand-driven residency;
- world age;
- invalidation;
- generated execution forms;
- RPDO search spaces;
- legality and numerical oracles.

The deck should sell the first list and prove it with the second.

---

# 12. PRIORITY ORDER FOR DEEP-DIVE BRANCHES

The recommended investigation order remains:

1. **Time Machine**
2. **Execution X-Ray**
3. **Execution Contexts**
4. **Schedule Forge**
5. **Live Foundry**
6. **Compiler Laboratory**

Reasoning:

- The first three create observability, control, and safe state manipulation.
- Schedule Forge then exploits semantic specialization.
- Live Foundry requires those verification and replay surfaces to be credible.
- Compiler Laboratory is most useful once there are real problems and candidate transformations to inspect.

After those six, revisit the adjacent primitives and ask whether any deserves promotion into an independent meta-tool.

---

# 13. DEEP-DIVE TEMPLATE FOR EACH BRANCH

Every branch should answer the same questions before it earns an idea doc.

### 1. User power
What can a user do that feels materially different from a conventional ML/agent stack?

### 2. Julia leverage
Which Julia language/runtime/compiler capability makes this unusually tractable or expressive?

### 3. Stack leverage
What do Gesso, Palette, or Cyan contribute beyond raw Julia?

### 4. Minimum demo
What is the smallest demonstration that makes a competent engineer say, “okay, that is real”?

### 5. Economic metric
What unit improves: time, tokens, GPU memory, calls, branch cost, human effort, reproducibility, etc.?

### 6. Correctness / safety boundary
What must never silently change?

### 7. Failure modes
What would make the capability misleading, fragile, or too expensive?

### 8. Competitive substitutability
Can the same user power be reproduced easily in the conventional stack, or does it require substantial glue/architecture?

### 9. Product placement
Does it belong in Gesso, Palette, Cyan, or only as a shared meta-primitive?

### 10. Evidence status
Mark each major claim as one of:

- `MEASURED`
- `LANDED, NOT YET PERFORMANCE-VALIDATED`
- `ENGINEERING CANDIDATE`
- `PRODUCT SYNTHESIS`
- `MOONSHOT`

### 11. Promotion rule
What evidence would justify moving it from idea → experiment → benchmark candidate → product capability?

---

# 14. CURRENT EVIDENCE BOUNDARY

This first draft intentionally keeps the ambition high while preserving the grounding rules in the current Gesso documents.

### Measured / landed today

- correctness-oriented Gesso floor through the current phases;
- paged KV + Session;
- identity-based prefix sharing + copy-on-write on the current fixture;
- CUDA and Lava/Vulkan backend seams under the same Session semantics;
- receipts/profiling foundations;
- Autotune selection among two gated CUDA matmul candidates.

### Not yet justified as performance/product claims

- named-model speed competitiveness;
- fused decode performance;
- representation-planner wins;
- Magenta end-to-end memory wins;
- schedule specialization wins;
- branch-aware parallel execution wins;
- generated kernel/schedule wins;
- compiler-fork wins;
- Time Machine, Foundry, or Compiler Laboratory as shipped products.

That boundary is a feature, not a limitation of the document.

The purpose of this ledger is to make the design space explicit enough that we can investigate aggressively **without confusing possibility with proof**.

---

# 15. ONE-SLIDE PRODUCT THESIS

> ## Stop paying the model to rediscover what the system already knows.
>
> **Gesso** remembers how computation should execute.  
> **Palette** remembers how capability should execute.  
> **Cyan** remembers how work should execute.
>
> **Shared state. Demand-loaded knowledge. Verified reusable computation. Adaptive execution.**
>
> The model remains the intelligence source.  
> The stack stops wasting it.

---

# 16. LEDGER — FIRST PASS

| ID | Idea | Category | Primary Owner | Status | Next Action |
|---|---|---|---|---|---|
| MP-01 | Meaning Before Mechanism | Meta-primitive | Full stack | Foundation | Preserve |
| MP-02 | Logical ≠ Physical | Meta-primitive | Full stack | Foundation | Preserve |
| MP-03 | Identity + Provenance | Meta-primitive | Full stack | Launch candidate | Measure on named model / expose upward |
| MP-04 | Branch Work DAG | Meta-primitive | Full stack | Post-launch R&D | Deep dive after speed floor |
| MP-05 | Availability ≠ Residency | Meta-primitive | Full stack | Post-launch R&D | Derive Palette/Cyan analogues |
| MP-06 | Demand-Driven Materialization | Meta-primitive | Full stack | Post-launch R&D | Prototype selectors/index surfaces |
| MP-07 | Heat / Use-Driven State | Meta-primitive | Full stack | Post-launch R&D | Trace demand first |
| MP-08 | Dynamic Granularity | Meta-primitive | Full stack | Post-launch R&D | Add to search vocabulary |
| MP-09 | RPDO / Gated Search | Meta-primitive | Full stack | Foundation → R&D | Expand only with oracle coverage |
| MP-10 | Receipts | Meta-primitive | Full stack | Launch candidate | Surface in UX |
| MP-11 | Hardware/Workload Fingerprints | Meta-primitive | Full stack | Foundation | Preserve |
| MP-12 | Cold vs Warm Cost | Meta-primitive | Full stack | Foundation | Benchmark discipline |
| G-01 | LogicalAttentionMemory + Plan | Gesso primitive | Gesso | Post-launch R&D | After speed floor |
| G-02 | Native Representation Execution | Gesso primitive | Gesso | Post-launch R&D | After representation receipt |
| G-03 | Prefill as Compilation | Gesso primitive | Gesso | Post-launch R&D | Magenta experiments |
| G-04 | Multi-Level Reuse | Gesso primitive | Gesso | Post-launch R&D | Mechanism-by-mechanism validation |
| G-05 | Branch-Aware Memory Execution | Gesso primitive | Gesso | Post-launch R&D | After fast serial baseline |
| G-06 | Shared Model Runtime | Gesso primitive | Gesso/Cyan seam | Roadmap | Serving phase |
| G-07 | Vendor-Neutral Session | Gesso primitive | Gesso | Landed seam | Tune later |
| G-08 | Fail-Closed Numerical Optimization | Gesso primitive | Gesso | Landed foundation | Preserve |
| G-09 | Compile-vs-Execute Attribution | Gesso primitive | Gesso | Speed-floor requirement | Next recipe |
| G-10 | Live method table over live Session (hot-reload; RSI-adjacent as first name) | Gesso primitive | Gesso (+ Palette/Cyan propose) | Post-launch R&D | After fused decode; model proposes, oracle installs |
| P-01 | Persistent Executable World | Palette primitive | Palette | Launch candidate | Deep dive |
| P-02 | Capability Lifecycle | Palette primitive | Palette | Launch candidate | Deep dive |
| P-03 | Demand-Loaded Operator Surface | Palette primitive | Palette | Launch candidate | Deep dive |
| P-04 | Snapshot / Derived World Identity | Palette primitive | Palette | Launch candidate | Deep dive |
| C-01 | Agent Trace as Workload Unit | Cyan primitive | Cyan | Launch/R&D foundation | Instrument traces |
| C-02 | Semantic Constraints / Branch Intent | Cyan primitive | Cyan | Launch candidate | Formalize seam |
| C-03 | Shared Battery Orchestration | Cyan primitive | Cyan | Post-launch R&D | After shared runtime |
| C-04 | Workflow Promotion / RPDO | Cyan primitive | Cyan | Post-launch R&D | Build task oracles first |
| C-05 | Active Working Set over Long Memory | Cyan primitive | Cyan | Post-launch R&D | Derive from real traces |
| MT-01 | Time Machine | Meta-tool | Palette | Launch candidate | **Branch next** |
| MT-02 | Execution X-Ray | Meta-tool | Palette/Gesso | Launch candidate | Branch |
| MT-03 | Execution Contexts | Meta-tool | Palette | Launch candidate | Branch |
| MT-04 | Schedule Forge | Meta-tool | Gesso/Palette | Post-launch R&D | Branch after first 3 |
| MT-05 | Live Foundry | Meta-tool | Palette/Gesso | Moonshot | Branch after Forge |
| MT-06 | Compiler Laboratory | Meta-tool | Palette/Gesso | Post-launch R&D | Branch after concrete compiler evidence |
| A-01 | Branch Lab | Adjacent tool candidate | Palette/Cyan | Candidate | Reassess after branching work |
| A-02 | Memory Planner Console | Adjacent tool candidate | Gesso/Palette | Candidate | Reassess after Magenta data |
| A-03 | Workflow Forge | Adjacent tool candidate | Cyan | Moonshot | Reassess after task-level RPDO |

---

# 17. CLOSING RULE

This ledger should remain ambitious by design.

Its discipline comes from **status and evidence**, not from artificially shrinking the idea space.

The operating rule is:

> **Think as far ahead as the architecture permits. Implement only as far as the receipts justify.**

That keeps the capability mine wild and the production stack boring.
