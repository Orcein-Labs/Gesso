# GESSO SPEED FLOOR
## Board constraint on exotic work (Foreman investigation, 2026-10-01)

    Document class:     living investigation / sequencing amendment
    Status:             accepted as a gate on Phase 10, pre-recipe
    Index:              docs/research/README.md
    Map:                docs/research/ROADMAP_NOW.md
    Canon:              docs/Gesso_Stack.md remains canon; this file
                        does not rewrite phases. It says when Phase 10
                        is allowed to open.                        This file is NOT:   a speed claim, or a license to fill
                        Representation.jl. Speed-floor recipes:
                        `docs/goals/PHASE10_SPEED_FLOOR.md`,
                        `PHASE10B_SPEED_FLOOR_CLOSE.md`. Current
                        Buffy recipe:
                        `docs/goals/PHASE10C_NAMED_MODEL_HYGIENE.md`
                        (LANDED 2026-10-02: HF 0-based importer + frozen
                        `gesso-cpu` golden + SmolLM2 fork bytes).

---

## 0. WHY THIS FILE EXISTS

The product is an inference stack that loads currently available
open-weight models and then demonstrates capabilities PyStack does
not have. Training stays out (§LVIII). Runway is months, not years.

The board constraint, recorded 2026-10-01:

    Exotic work is the point.
    A board will not accept exotic work on an engine that is not
    even somewhat close in performance to the practical Python
    inference stack on a named model.

"Somewhat close" is a **gate**, not a slogan. This file says what
that gate is, what is actually slow on the tree today, which exotic
work already exists and must survive a fast path, and what the
greater plan does next.

RPD (the closed-recipe spiral, receipts, skip-or-green, fail-closed
oracles) is why Phases 0–9 exist as a floor instead of a swamp.
It is not a performance technology. RPDO — the same spiral aimed at
*search* — is Autotune (landed, selection only) and, later, the
representation planner. Do not advertise RPD as tok/s.

---

## 1. THE GATE (Phase 10 does not open until these are true)

Three conjuncts. All three. Receipts in this repo. Agent confidence
is not evidence.

### G1 — Named model

SmolLM2-135M (the Phase 3 pick) generates on this machine through
`Session`, CPU and CUDA, greedy ids matching `reference_*`. The
current skip-or-green becomes **green on the box that will be demoed**.
No download in CI. A local snapshot at `GESSO_SMOLLM2_DIR` is an
ops fact, not a code change.

Without G1, every timing row is llama_micro / toy2. Those are
correctness fixtures. They are not a product conversation.

### G2 — Same-box baseline, published as a factor

One warmed decode comparison, same GPU, same checkpoint, same
prompt, same `max_new_tokens`, batch 1:

    Gesso CUDA Session decode tok/s
    --------------------------------
    naive PyTorch eager generate tok/s
        (HF LlamaForCausalLM, no torch.compile, no vLLM)

Schema 0.2.0, post-warmup, compile outside the timed region.
The **factor** is the product. The first recipe that opens this
gate **measures**; it does not pick a marketing target in advance.

Publish **two** receipts, never one:

    first-token   (compile + load + first decode may be inside)
    warmed decode (compile outside the timed region — §XXXIII)

Those are different clocks. A board slide that mixes them is a
lie. Julia makes the split mandatory: TTFX is a compiler problem;
steady-state tok/s is a kernel/fusion/transfer problem.

A later recipe may add torch.compile and vLLM as *additional rows*.
Those rows are context. The gate itself is eager PyTorch, because
that is the stack a board member can reproduce in an afternoon.

### G3 — Fast path preserves the floor laws

The path that produces G2 still:

- matches CPU-oracle greedy ids on toy2 + llama_micro + SmolLM2
- fail-closes (no silent CPU fallback, no silent kernel swap)
- keeps `fork` + CoW byte-share (`unique_kv_bytes` still 2048 vs
  4096 on the llama_micro pair, or the analogous SmolLM2 figure —
  measured 2026-10-02 in 10C: CPU N = 1_474_560; fork == one
  session, two isolated == 2N)
- leaves CUDA and Lava as extensions, JSON as the only core
  third-party hard dep

If a fused kernel wins Autotune and breaks `fork`, it is not a
winner. Disqualify it.

**Phase 10 (representation planner, cages, ExactBits in `src/`)
opens after G1∧G2∧G3.** Magenta topology (Phase 11) waits on that
plus a real KV working-set on SmolLM2, not llama_micro.

---

## 2. WHAT IS ACTUALLY SLOW (tree, not folklore)

Read from `src/` on 2026-10-01. No tok/s claim is made here.

| # | Mechanism | Where | Why it costs |
|---|-----------|--------|--------------|
| 1 | Attention **gathers** paged KV into contiguous scratch every step | `kv_manager.jl` `gather_kv!`; `session.jl` | Pages are the cache; the manager is not a kernel. FlashAttention-class fusion is glue we have not written. |
| 2 | Prefill still D2H of the logits row; decode greedy is device-side | `session.jl` | CUDA `:argmax` — host gets one Int on **decode**. `prefill!` still returns `Array(logits)`. Lava: full-row host argmax. |
| 3 | Unfused op soup + gather then GEMM | `cpu.jl` / `cuda_ops.jl` / `lava_ops.jl` / `session.jl` | Layer ops still separate launches. CUDA attention is now device `mul!` over **gathered** scratch (`:attn_gemm`). Gather every step remains. Phase 9 Autotune still only selects among two **matmuls**. |
| 4 | Batch = 1, `generate` RESETS | `session.jl` | No continuous batching, no prefix-reuse across `generate` calls. `fork` is the share constructor; `generate` drops it. |
| 5 | Host-side interpreter loop | `Inference.jl` | Julia orchestrates layers in a serial for-loop. Fine for correctness. Death for decode tok/s until the inner step is one (or few) device graphs. |
| 6 | Lava path is GPUArrays broadcast + `mul!` | `lava_ops.jl` | Portable seam (Phase 8). Untuned. CUDA is the speed horse; Lava stays the portable proof. |
| 7 | No named model in the corpus | Phase 3 item D | llama_micro CUDA prefill median ~10 ms for **3 tokens** (`2026-10-01.tsv`). That number cannot enter a board slide. |
| 8 | `Lowering.jl` is empty | `src/Lowering/` | No fusion, no graph capture, no routing. Backends are reached by method dispatch on `*Backend`. |
| 9 | Julia compile / invalidation / GPUCompiler | runtime, not yet attributed | First-token and Lava load-from-source (~1 min on 1.12) are compiler-stack costs. See §2a. |

Phase 6 and 9 did their jobs: we can **attribute** engine calls, and we
can **select** among registered candidates. Neither moved the inner
decode. That is expected. MAKE IT FAST was never "pick CUBLAS."

---

## 2a. ONE LEVEL DEEPER — THE JULIA COMPILER

Fused CUDA candidates are necessary and insufficient. A Julia
inference stack that matches eager PyTorch on warmed decode and
then hitch-compiles the first token, the first shape, or a Lava
pipeline has not met the board constraint.

Two clocks, never one number (§2 of
`Julia Compiler Optimizations for Gesso/docs/the_ancient_texts.md`):

    compilation latency     TTFX, invalidation, GPUCompiler,
                            SPIR-V, VkPipeline
    steady-state throughput fusion, D2H, occupancy, launches

Gesso_Stack already states the architectural bet: the compiler
must not rediscover information the semantic core already had
(GQA K-projection, frozen, decode-hot). That is why this layer
exists *inside* Gesso's meaning pipeline, not as a generic
"make Julia faster" side quest.

### Exhaust this ladder, in order

1. **Attribute.** Split every G2 run into source load, package
   load, inference, specialization, LLVM, native emit, GPUCompiler,
   (Lava: SPIR-V + pipeline), GPU execute, D2H. A performance claim
   without this decomposition is not actionable. The ancient texts
   §3 are the instrument list. No Julia **compiler fork** begins
   until this table exists for SmolLM2 decode on the demo box.
2. **Win without modifying Julia** (ancient texts §4). Type
   stability on the decode step, bounded specialization, pinned
   boring deps in the shipping image, PrecompileTools /
   SnoopCompile invalidation control, persistent worker so
   restart is recovery not the hot path. Lava already does not
   precompile on 1.12 — that is a compiler-layer finding we
   already paid for.
3. **GPUCompiler / Lava emitter** as the portable-fast surface
   (ancient texts §11–13): Julia IR before GPUCompiler sees the
   kernel, then fusion, then SPIR-V quality. CUDA remains the
   G2 horse; this ladder is how Lava becomes more than a seam.
4. **A Julia compiler fork** only if (1)–(3) leave a measured
   hole that stock Julia cannot close, and WGE is demanding the
   same hole. The ancient texts' strategic bet: WGE + Gesso
   together make this shared infrastructure; Gesso alone does
   not justify maintaining a compiler.

RPDO applies here the same way it applies to matmul: propose a
compiler/kernel variant → compile → oracle gate → bench →
retain Pareto. Taste is not a lowering.

This layer is **weeks of QA**, same as G2. It is not a Buffy
sprint that lands a fork next week. Idle Buffys still take
application glue (fused decode, device argmax) and import.
Compiler attribution can be a slice of the speed-floor recipe
once SmolLM2 is green — counters, not a new Julia.

---

## 2b. THREE TUNERS, THEN DISPATCH-SHAPED PARALLELISM

The seriousness bet, recorded 2026-10-01:

    Stack kernel tuning
  + Julia compiler kernel tuning
  + Lava tuning
  = fairly close on the workloads we actually ship.

Fairly close is enough to be taken seriously, and then to be
*optimized*. It is not "beat vLLM on day one." It is G2 with a
factor a competent outsider will argue with rather than dismiss.

The exotic bet sits **after** that:

    Multiple dispatch + metaprogramming specialize *schedules*,
    not only kernels.

Gesso already knows facts a PyStack graph compiler has to
rediscover: GQA grouping, frozen projections, decode-hot vs
prefill, `fork` identity, page aliasing. Dispatch on those facts
can emit a different parallel form the same way it already emits
a different `matmul!`.

Two moves, in order:

1. **More and better parallelism where it already exists.**
   Heads, layers that are independent, expert/batch axes, CUDA
   streams vs Lava recordings. Specialize the *actual* parallel
   shape per (workload, representation, device, share-tree)
   instead of one generic launch.
2. **Parallelism where the stack currently serializes.**
   A `fork` tree is already a work DAG with shared prefix pages.
   Independent children can run as independent device work
   without a radix-cache lookup. Page-granular CoW is independent
   writers. Magenta pipelines and multi-Session prefix share are
   the same idea one layer up. This is undiscovered *for us* —
   it is not a claim of a paper. It waits on G1∧G2∧G3 so the
   parallel form is racing a fast serial form, not a gather loop.

Autotune is the consult site for both moves. New parallel forms
are **candidates** with the same oracle gate. A dispatch-specialized
schedule that fails ids or breaks `fork` is disqualified. The
foundry (ancient texts §26 — generate structurally different
kernels) consumes the tuner; the tuner does not wait on the
foundry.

Do not start (2) in a Buffy recipe while (the three tuners) are
still the open speed floor. Write it here so we do not forget
why Julia was the language.

---

## 2c. THE PROOF LADDER (where insanity becomes justified)

Chat's 2026-10-01 boil of this file, accepted as the rabbit hole.
Legal now is the top. Each rung is permission for the next.

```
0  Make Gesso correct.                         DONE (Phases 0–9)
1  Make individual operators fast.             speed floor (in)
2  Fuse the decode path.                       speed floor (in)
3  GPU understands Gesso native representations
   instead of converting them to conventional.   after G1∧G2∧G3
4  Semantic facts choose specialized schedules.  after 3
5  fork/share topology as a parallel work graph. after 4
6  Metaprogramming generates structurally
   different schedules/kernels.                  foundry; after tuner
7  RPDO searches that execution-design space.    Autotune grown
8  Compiler + Lava tuning co-adapt with
   generation, scheduling, representation.       after 7
9  Modify Julia itself, only if stock Julia
   is the remaining wall.                        last
```

The research thesis hiding in this file (Chat's sentence, kept):

    Julia may let Gesso preserve semantic knowledge far enough
    down the stack that multiple dispatch and metaprogramming
    can specialize entire execution schedules, while RPDO
    empirically determines which schedule should exist for a
    given semantic regime and hardware target.

If that works even modestly, Julia-native is an architecture
reason, not an implementation preference.

### Second lane

Two products, one engine:

    Lane A — boring.  Load a model. Session.generate. Ids match
                      the oracle. Factor vs eager PyTorch that a
                      board will argue with. Cyan consumes this.
    Lane B — exotic.  Power users and scientists (ML, data) who
                      can think in dispatch, representations, and
                      share-trees. "What, Julia would let us do
                      *what*?"  This lane is real only if Lane A
                      is reliably boring.

Lane B is not a license to skip Lane A. It is why we bother
keeping meaning all the way down. Collect those cases in musings
or this file as they appear; do not implement them from a
daydream. The ladder above is when each case is allowed to
become a candidate.

Corpus we do have (do not reuse as a PyStack comparison):

- llama_micro CUDA prefill, 3 tokens, F32, host readback: median
  ~1.00e7 ns (`micro_llama_cuda_prefill_012`, 2026-10-01)
- llama_micro Lava prefill, same shape: median ~1.38e7 ns
- unique KV bytes, llama_micro prefill pair: 4096 isolated / 2048
  after `fork` (the only measured semantic win)

---

## 3. EXOTIC: WHAT ALREADY EXISTS vs WHAT WAITS

### Already on the tree (must survive G3)

- **Declared identity prefix share.** `fork` aliases complete
  prefix pages; first write copies that page. vLLM-class systems
  *discover* prefix cache by token match. Gesso *constructs* a
  share tree. Measured on llama_micro AND on SmolLM2 (10C:
  CPU N = 1_474_560; fork == one session, isolated == 2N; CUDA
  repeats the identities at N = 737_280, F32 storage).
- **Fail-closed numerics.** Device work is illegal until greedy
  ids match the CPU F64 oracle. Autotune disqualifies a faster
  candidate that fails the gate.
- **Vendor-neutral Session.** CUDA and Vulkan are the same
  `prefill!` / `decode!` / `fork` API; backends are weakdeps.
  Speed floor rides CUDA. Lava stays green-or-skip.
- **Autotune as a loop.** One operator, two candidates, cache
  keyed by device × backend × op × regime. The next speed work
  **registers new candidates** (fused attention, device argmax).
  It does not build a second tuner.

### Parked until G1∧G2∧G3 (and usually Phase 10/11)

- Gauge-compiled / caged weights (`REPRESENTATION_PROGRAM.md`)
- Measurement 1 (`B1 < B0` on SmolLM2) — still requires the
  snapshot; still CPU-oracle legal; still not a Buffy sprint
  while the speed floor is open
- Magenta topology / residency / KV as a lowering beyond pages
- ExactBits codecs, cages in `src/`, foundry / kernel generation
- Generalized speculation (Phase 20)
- "Any model on earth" — that is an import *program*, not a phase

### "Any currently available model"

Honest near-term: **Llama-family HuggingFace safetensors** that
fit the existing name map, GQA, optional `final_rms`, GPT-2 BPE
or a second tokenizer after SmolLM2 is green.

Glue after G1 (not invention): `rope_scaling`, SentencePiece,
a second `model_type` (Mistral / Qwen / Gemma) **one architecture
at a time**, each with a conformance fixture and an oracle gate.
Do not promise a universal loader in the speed-floor sprint.

---

## 4. WHAT THE GREATER PLAN DOES NEXT

Canon order stays: 5b serving, 10 representation, 11 memory, 16
ABI, 18–20 specialize / rematerialize / speculate.

Living order, given the board constraint:

```
0–9         floor                         DONE
G1          SmolLM2 green on the demo box after ops places the snapshot
            (2026-10-01: protocol exists — test_session_smollm2.jl;
            ops-blocked on boxes without a local snapshot; skip-or-green)
G2+G3       speed floor                   LANDED 2026-10-01 (PHASE10_SPEED_FLOOR.md)
            fused decode step on CUDA,    rungs 1–2: device-side greedy DONE
            device-side greedy,           (one Int D2H per token); attention
            Autotune candidate(s) for the QKᵀ+PV is device GEMM over the
            step,                         gathered scratch (backend-dispatched
            eager-PyTorch factor published capability :attn_gemm, not a 2-D
            as TWO receipts (first-token / warmed) Autotune candidate — the
            plus a compile-vs-execute     contraction is head-structured);
            attribution                   still forks; G2 harness exists
            table (SPEED_FLOOR §2a).      (benchmark/compare_eager.py —
            No Julia fork.                rows land when snapshot + CUDA +
                                          torch exist; named skip otherwise);
                                          compile-vs-execute split is carried
                                          per-row (first-token vs warmed) in
                                          the G2 rows themselves
5b          serving fabric                after two concurrent Sessions
            on the fast path (fork is the
            concurrent primitive we already have)
10–11       representation + Magenta      AFTER G1∧G2∧G3
ABI         Cyan consumes Gesso           when Session is stable
18–20       the rest of the weirdness     after 10 has a receipt
```

Living Phase 10 (speed floor recipe) **LANDED 2026-10-01**, closed by
Phase 10B the same day (harness skip-or-land hardening: --probe dry
import, snapshot file gate, caps pinned by test, llama_micro generate
first-token + warmed rows on the GEMM path — fixture numbers, not the
board factor). Do not hand Buffy canon §LXXXIII. Ops delivered the
snapshot; G2 is measured (0.692×, `2026-10-02.tsv`); 10C landed
(HF importer, frozen `gesso-cpu` golden, named-model fork bytes).
Idle Buffys import glue.
Phase 9's Autotune loop is the consult site; fused work lands as
**candidates**, CUDA ext files, and a corpus row. `Lowering.jl`
stays empty until a recipe says routing/fusion lives there —
default: fusion is a candidate function, not a planner.

Lava is not the speed horse this quarter. Keep it skip-or-green.
A Lava fused candidate is a later portable-fast sprint (canon
still says start portable, then tune).

5b (scheduler / continuous batching) is the *other* way boards
measure "close" (throughput at batch > 1). Open it when G2 is
in-hand and two concurrent SmolLM2 Sessions are the workload.
`fork` is how those Sessions share; `generate` still RESETS —
5b will have to say what "a turn" is without dropping the share.

---

## 5. RPD / RPDO

RPD built the floor: bounded items, independent tests, receipts,
named skips, no silent fallback, oracle ids as law. That is why
a four-Buffy factory can land seams without semantic drift.

RPDO is the same spiral with a search objective:

    candidate → gate (oracle) → measure (post-warmup) →
    winner → cache → invalidate

Phase 9 is RPDO for **one matmul**. The speed floor is RPDO for
**the decode step**. Phase 10 is RPDO for **weight realizations**,
and it is illegal until the decode-step factor exists.

---

## 6. RISKS

- **Fusing attention against paged KV** is the actual engineering.
  Gather-to-scratch was the correctness encoding. A fused candidate
  that requires contiguous KV either copies (hides the pages) or
  speaks page tables (new kernel contract). Packet if the fused
  form cannot honor `fork`'s aliased pages.
- **Device-side argmax** must keep ties = first index, 0-based, no
  Random — the greedy law. A CUBLAS `argmax` with different tie
  breaks fails G3.
- **PyTorch baseline hygiene.** Same checkpoint bytes, same dtype
  story (Gesso CUDA is F32 compute; SmolLM2 weights may originate
  BF16). Stamp the arithmetic contract on the receipt. A factor
  that mixes F32 Gesso vs BF16-accelerated PyTorch is a lie.
- **SmolLM2 snapshot is ops.** The recipe can be skip-or-green in
  CI and **required green** on the demo box. Write that split
  explicitly so CI does not start downloading.
- **Opening Phase 10 in parallel "because Buffys are idle"**
  violates this file. Idle Buffys take import glue (second
  tokenizer, rope_scaling) or the speed-floor recipe.
- **Conflating TTFX with warmed tok/s.** Julia will look
  catastrophic on first token and merely unfused on token 32.
  Attribute before you fuse; fuse before you fork the compiler.
- **Forking Julia because Lava is slower than CUDA.** That is
  an application/vendor-library gap until the attribution table
  says otherwise. CUDA is the G2 horse.

---

## 7. WHAT THIS FILE IS NOT

- a tok/s claim
- a promise to beat vLLM this runway
- permission to add CUDA graphs, FlashAttention, or a Python
  baseline harness before a `docs/goals/` recipe names the files
- a rewrite of §LXXXIII (Phase 10 still exists; it waits)
- training, cages in `src/`, or "any model" as a single sprint
