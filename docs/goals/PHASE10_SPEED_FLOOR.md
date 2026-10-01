# /goal PHASE 10 — SPEED FLOOR (named model, factor, fast path)

**For:** Buffy (mechanical implementation)
**From:** Grok (encoding owner)
**Living name:** Phase 10 speed floor (proof-ladder rungs 1–2)
**Canon §LXXXIII (representation planner) is NOT this file.** That
phase stays parked until G1∧G2∧G3 exist. Do not fill
`src/Representation/`. Do not mark §LXXXIII COMPLETE.
**Canon touched:** §XXXIII (timing hygiene), §XLII (receipts),
§LXX (fail-closed), §XXVI / §LXXXII (Autotune consult), §LXXVI
(SmolLM2 skip-or-green).
**Map:** `docs/ARCHITECTURE.md`
**Research (read, then implement only what this file names):**
`docs/research/SPEED_FLOOR.md` (gates G1–G3, two clocks, §2a
compiler attribution, §2b three tuners). Companion seed:
`Julia Compiler Optimizations for Gesso/docs/the_ancient_texts.md`
§3 — instrument list; do not fork Julia.
**Depends on:** Phase 9 complete (`docs/goals/PHASE9_AUTOTUNE.md`)
**Packets:** 1 and 2 stay closed.

**Status:** OPEN. This is "publish a named-model factor vs eager
PyTorch, on a path that still `fork`s, with compile and execute
on different receipts." FlashAttention-class paged kernels,
foundry, Julia compiler fork, Lava tune, Magenta, cages — later.
Do **not** edit `libs/Lava`.

---

## Start condition

Phase 9 is on the tree you inherit (`master` at `4c01994` plus the
encoding-owner close of `PHASE9_AUTOTUNE.md`):

- Autotune loop in core; CUDA `matmul!` consults `select`; winner
  `:cublas_mul` on llama_micro (`micro_llama_cuda_prefill_012_autotuned`)
- 1483 pass / 3 named SmolLM2 skips; device-less CUDA 1300 / 10
- toy2 CUDA fingerprint print `0.00037607177004872483`; llama_micro
  `5.5006127839263286e-6`; greedy ids exact
- `fork` unique_kv_bytes 2048 vs 4096 on the llama_micro pair
- JSON only core third-party hard dep; CUDA and Lava weakdeps
- `Representation.jl` / `Planning.jl` / `Runtime.jl` still empty
- `generate` still RESETS; attention still gathers pages to scratch;
  greedy still host `argmax` after logits on the host

If Phase 9 is unfinished, stop.

## One-sentence objective

SmolLM2-135M generates on this box when a local snapshot is present;
CUDA decode publishes first-token and warmed receipts against naive
PyTorch eager on the same checkpoint; the path that produces those
numbers still matches oracle ids and still `fork`s.

CI never downloads weights, never requires a GPU, never requires
PyTorch. Demo-box green is ops + this recipe together.

## Why this phase exists

The board will not take exotic Gesso seriously on llama_micro.
`SPEED_FLOOR.md` G1∧G2∧G3 is the gate on canon Phase 10
(representation). This sprint *is* that gate's first implementation:
named model, published factor, fork-preserving fast-path glue
(device greedy + device attention over gathered scratch).

It is not "beat vLLM." It is a factor a competent outsider will
argue with. If the factor is ugly, the receipt still counts — then
the next recipe fuses harder. Do not hide a bad number.

## What this sprint is not

- canon §LXXXIII representation planner, cages, ExactBits, Magenta
- filling `Representation.jl` / `Planning.jl` / `Runtime.jl`
- a Julia compiler fork, a GPUCompiler patch, a Lava tune sprint
- FlashAttention / paged-KV kernel that skips gather (packet if
  you need it; default is device GEMM **over gathered scratch**)
- torch.compile / vLLM rows (later additional rows, not the gate)
- adding PyCall, PyTorch, SnoopCompile, or KernelAbstractions to
  core `[deps]`
- Lava Autotune consult
- changing `reference_*` signatures
- training
- editing `libs/Lava`
- a speed claim vs CUDA.jl or llama.cpp in prose

---

## Laws (pin this)

### G1 — snapshot is ops; tests are skip-or-green

`GESSO_SMOLLM2_DIR` already names the local HuggingFaceTB/SmolLM2-135M
tree. `make test` never downloads. Without the env var: named skip
(the three existing Broken records). **With** the env var on the
demo box: CPU and CUDA `Session.generate("Hello"; max_new_tokens=8)`
ids equal `reference_generate` / CPU Session (existing
`test_session_smollm2.jl` contract). Do not invent a new golden
for CUDA logits.

### G2 — two receipts, same checkpoint, same arithmetic stamp

When snapshot **and** CUDA.functional() **and** a Python with
`torch` + `transformers` exist:

    gesso CUDA Session warmed decode tok/s
    --------------------------------------
    HF LlamaForCausalLM eager generate tok/s
        (no torch.compile, batch 1, same prompt, same max_new_tokens)

Plus a **first-token** pair (compile/load allowed inside).

Schema 0.2.0, append-only TSV. Factor lives in `bench_note`.
Stamp dtype: Gesso CUDA is F32 compute. If the checkpoint is BF16
on disk, say so. Mixing silent casts is a lie.

Missing torch / missing snapshot / no GPU: named skip, no new row
required. The harness exists either way.

### G3 — fast path still the floor

Device greedy and device attention:

- greedy ids exact vs CPU on toy2, llama_micro, and SmolLM2-when-present
- ties = first index, 0-based, no Random
- `fork` unique_kv_bytes still 2048 vs 4096 on the llama_micro
  prefill pair (or an explicit new pair of BYTE rows if you re-bench)
- fail-closed: no silent CPU fallback
- Autotune remains the consult site for CUDA `matmul!`; do not
  rip it out

### Two clocks

Never one number. First-token ≠ warmed. Compiler attribution is
a coarse table in the bench note (load / first generate / subsequent
generate). Full SnoopCompile / invalidation trees are later; do not
add that package to core.

### Fusion bound for this sprint

The CUDA decode attention contraction in `session.jl` is a Julia
loop / broadcast over gathered scratch. Replace that, on CUDA, with
**batched device GEMM** (QKᵀ and PV) over the **same gathered
scratch**. Pages remain the cache. Gather remains legal. A kernel
that reads page tables directly is a packet.

Register the new contraction as an Autotune candidate if it is a
`matmul!`-shaped piece; if it lives in Session, keep it backend-
dispatched (`on_cpu` vs device) without a new core dep.

CPU path stays the bit-identical oracle loop.

---

## Work items (sequence)

### A — G1: SmolLM2 protocol is real on the demo box

**Objective.** Existing SmolLM2 tests remain skip-or-green in CI.
On a machine with `GESSO_SMOLLM2_DIR` set, they are **required
green** (CPU Session ids = reference; CUDA Session ids = CPU
Session). Document the demo-box command in the goal receipt.

**Permitted files**

```
test/test_smollm2.jl              # only if a hole appears
test/test_cuda_smollm2.jl
test/test_session_smollm2.jl
docs/goals/PHASE10_SPEED_FLOOR.md # receipt notes only at D; don't rewrite A
```

Prefer **zero code** if the existing tests already encode G1.
Then A's artifact is a receipt sentence: "ran with GESSO_SMOLLM2_DIR=…
CPU ids … CUDA ids …" plus CI still skip-or-green without it.

If the snapshot is **absent on the machine Buffy runs**, A is the
skip-law confirmation (already true) and G1 stays **ops-blocked** —
write that in the receipt. Do not download. Do not fake a golden.

---

### B — Device-side greedy

**Objective.** CUDA `decode!` / `prefill!` argmax runs on device.
Host receives **one integer id**, not the logits vector, on the
hot path. CPU greedy unchanged (host argmax, bit-identical).

**Permitted files**

```
src/Inference/session.jl
ext/cuda_ops.jl                   # only if argmax lives next to ops
ext/GessoCUDAExt.jl
test/test_session.jl
test/test_session_cuda.jl
test/test_session_fork.jl
test/runtests.jl
```

**Tests**

- toy2 + llama_micro greedy ids still equal the CPU oracle
- host `Array` + `CUDABackend()` still `ERR_INVALID_PLAN`
- CUDA fingerprint print stays inside the existing atol
- `fork` unique_kv_bytes 2048 vs 4096 on llama_micro prefill pair
- no CUDA: named skip
- ties = first index (construct a known-tie fixture or document
  how the existing toy2 case covers it)

**Artifact.** One D2H of an `Int` per token on CUDA decode.

---

### C — Device attention over gathered scratch

**Objective.** CUDA prefill/decode attention contraction uses
device batched GEMM (or equivalent `mul!`) over gathered K/V
scratch instead of the Julia per-head loop. Pages + gather stay.
CPU loop stays the oracle.

**Permitted files**

```
src/Inference/session.jl
src/Inference/Inference.jl        # only if a helper must move
ext/cuda_ops.jl
ext/GessoCUDAExt.jl
src/Autotune/Autotune.jl          # only if you register a new candidate
test/test_session_cuda.jl
test/test_cuda_inference.jl
test/runtests.jl
```

**Tests**

- generate ids still exact on toy2 + llama_micro (CUDA vs CPU)
- logits stay inside the existing CUDA atol
- `fork` share still aliases complete prefix pages (byte-identical
  CoW test still green)
- Lava path unchanged (still its own contraction; skip-or-green)
- no CUDA: named skip

**Packet and stop** if the only way to pass ids is a contiguous
KV cache that drops `fork` aliasing, or a new core dependency.

**Artifact.** CUDA decode is still gather-then-GEMM, and GEMM is
on device. Proof-ladder rung 1–2 started, not FlashAttention.

---

### D — G2 harness + attribution + maps

**Objective.** A bench path that, when snapshot + CUDA + torch
exist, appends first-token and warmed rows for Gesso and for
eager PyTorch, same prompt, same `max_new_tokens`, factor in the
note. Coarse compile-vs-execute split in the Gesso note. Maps
point here; §LXXXIII stays parked.

**Permitted files**

```
benchmark/runbenchmarks.jl
benchmark/compare_eager.py        # new; stdlib + torch/transformers
benchmark/results/                # append-only TSV
test/test_speed_floor_harness.jl  # skip-or-green: script exists, --help / dry
README.md
docs/ARCHITECTURE.md
docs/research/ROADMAP_NOW.md
docs/research/README.md
docs/research/SPEED_FLOOR.md      # G1/G2/G3 status sentences only
docs/Gesso_Stack.md               # a SPEED FLOOR status note under
                                  # §LXXXII or a short § pointer;
                                  # do NOT mark §LXXXIII COMPLETE
scripts/freeze.jl
```

PyTorch is an **external** binary (`python3` + `torch`). Probe
with a dry import; skip the torch rows if it fails. Never add it
to `Project.toml`.

Prompt / length for the comparison (pin these):

```
prompt            "Hello"
max_new_tokens    8
model             HuggingFaceTB/SmolLM2-135M via GESSO_SMOLLM2_DIR
Gesso             Session, CUDA, greedy
PyTorch           transformers LlamaForCausalLM.generate, greedy,
                  do_sample=False, batch 1, no torch.compile
```

If SmolLM2 is absent, you may still land the harness against
llama_micro for wiring tests, and skip the torch/SmolLM2 rows.
The **gate number** the board cares about is SmolLM2. llama_micro
is not that number.

**Maps.** Next open recipe after this file lands: either a harder
fusion sprint (if the factor is not yet "argue-with") **or**
canon Phase 10 representation if G1∧G2∧G3 are honestly green.
Encoding owner decides from the TSV, not from hope.

**Artifact.** Schema 0.2.0 rows on disk. Factor in the note.
No "faster than PyTorch" prose unless the numbers say so.

---

## Cross-item invariants

- toy2 CPU fingerprint bit-identical
- `reference_*` public signatures unchanged
- JSON remains the only core third-party hard dep
- Autotune.jl still imports neither CUDA nor Lava
- CPU and Lava paths do not silently start consulting Autotune
  for attention (CUDA `matmul!` consult stays)
- `libs/` untouched
- `Representation` / `Planning` / `Runtime` still empty
- no ExecutionResult, no receipt/bench schema bump
- no Julia compiler fork
- Receipt: what / why / tests / numerical delta (ids exact;
  CUDA atol held) / first-token vs warmed / factor vs eager
  if measured / compile-time / memory / hardware / workload
  Hello×8 / model SmolLM2-or-skip / backend cuda

## Escalation (packet, then stop)

- you need FlashAttention or a page-table kernel to pass ids
- you need to drop `fork` aliasing to go faster
- you need a new core dependency
- you want to fill Representation.jl or start Magenta
- you want to fork Julia or patch GPUCompiler
- both Gesso and torch rows cannot share a checkpoint/arithmetic
  stamp
- you want vLLM or torch.compile as the G2 gate (they are later
  additional rows)

## Exit checklist

- [x] A, B, C landed
- [x] D maps truthful; §LXXXIII still not COMPLETE
- [x] `make test` green without a CUDA device and without SmolLM2
      (named skips)
- [x] `make format` run
- [x] CPU fingerprint unchanged
- [x] Lava tests still skip-or-green
- [x] CUDA greedy ids still match CPU on toy2 + llama_micro
- [x] `fork` unique_kv_bytes still 2048 vs 4096 on llama_micro
- [x] device greedy: host does not receive the logits vector on
      the CUDA decode hot path
- [x] CUDA attention contraction is device GEMM over gathered scratch
- [x] G2 harness exists; torch/SmolLM2 rows skip-or-land
- [x] no silent fallback
- [x] `libs/` clean
- [x] no Representation fill, no foundry, no cages, no Julia fork
- [x] receipt

---

## Receipt (Phase 10 close, 2026-10-01)

**What changed.** A: G1 confirmed as the existing
`test_session_smollm2.jl` contract (CPU Session ids = oracle,
determinism, CUDA ids = CPU Session ids; skip-gated on
`GESSO_SMOLLM2_DIR`) — zero code; the snapshot is ABSENT on the box
Buffy runs, so G1 is OPS-BLOCKED here (no download, no invented
golden); demo-box command: `GESSO_SMOLLM2_DIR=<snapshot> make test`
(required green there, named skip in CI). B: CUDA `decode!` runs
argmax ON the device (`_device_greedy_id` — final RMSNorm + lm_head
matmul + `Base.argmax` over device storage; host receives ONE Int
per token; the (vocab,) logits row never crosses back). C: the
attention contraction (QKᵀ + PV) on CUDA is ONE flat device GEMM
per site (4 sites: prefill/decode × scores/PV) over the SAME
gathered scratch, via zero-copy `reshape(CuArray)` views; pages
remain the cache, gather remains legal, no page-table kernel (that
is the packet). Both fast paths sit behind DECLARED capability
probes — `Gesso.supports(backend, :argmax / :attn_gemm)` — so Lava
keeps its own contraction and the full-row host argmax (goal
sanction: "if it lives in Session, keep it backend-dispatched"); NOT registered as Autotune candidates (the head-structured 3-D
contraction does not fit the 2-D `(dst,x,w)` candidate contract;
every `matmul!` on the path still consults the search, §LXXXII
unchanged). D: `benchmark/compare_eager.py` (external python3;
local weights only via `local_files_only=True`, never downloads;
first/warmed clocks; eager = LlamaForCausalLM.generate greedy,
do_sample=False, batch 1, NO torch.compile) + the gated G2 block in
`runbenchmarks.jl` (ids gate BEFORE any row is recorded — fail
closed; eager rows recorded directly, never re-timed through
BenchmarkTools; factor is a post-loop DECLARATION row computed from
the run's own medians) + harness test `test_speed_floor_harness.jl`
+ maps (canon status note under §LXXXIII — NOT COMPLETE;
README/ARCHITECTURE/ROADMAP_NOW/research README/SPEED_FLOOR status
sentences; freeze list).

**Why.** SPEED_FLOOR G1∧G2∧G3 is the gate on canon §LXXXIII;
this sprint is that gate's first implementation: named-model
path, published-factor harness, fork-preserving fast-path glue.

**Tests.** `make test` 1497 pass / 3 named SmolLM2 skips (8m16s);
second runner `--check-bounds=yes` 1497/3 (8m21s); device-argmax
testset 4/4 (ids identity + tie law + capability intent); fork
suite 162/162 incl. device CoW aliasing; no-copy
`ERR_INVALID_PLAN` law intact; harness test 10/10 (torch-free);
bench wiring: 15 rows recorded, G2 skip names its missing gates.

**Numerical delta.** Fingerprints BIT-IDENTICAL to Phase 8/9:
toy2 cuda-vs-cpu max|Δlogit| = 0.00037607177004872483,
llama_micro = 5.5006127839263286e-6; greedy ids exact on toy2 +
llama_micro; `fork` unique_kv_bytes still 2048 vs 4096 (byte rows
unchanged). G2 FACTOR: NOT MEASURED on this box (no snapshot, no
torch) — the factor row records its own absence; no speed claim is
made anywhere.

**First-token vs warmed.** Both clocks implemented on BOTH sides
(Gesso first-token = one wall-clock generate on a fresh Session,
compile inside; warmed = untimed warmup then BenchmarkTools;
eager first-token = script single-shot; eager warmed = script
in-process warmup + median). Rows only land when all G2 gates hold.

**Compile-time / memory.** No new packages anywhere (LinearAlgebra
is stdlib; python3 + torch are external, never Project.toml deps —
asserted by test); no schema bump (receipts and bench rows stay
0.2.0); bench suite 14 → 15 rows (the factor declaration row).

**Hardware / workload / model / backend.** RTX 5060, CUDA 6.3.1,
Julia 1.12.6; workload `generate("Hello"; max_new_tokens=8)` on
SmolLM2-135M-when-present, toy2 + llama_micro gates otherwise;
backends: CUDA (fast paths), CPU (bit-identical oracle), Lava
(unchanged contraction, verified green).

**Known limits / unresolved.** G1 ops-blocked until ops places a
local SmolLM2 snapshot; the G2 gate number the board cares about is
SmolLM2 on the demo box — llama_micro is NOT that number; unknown
(K,N) shapes attribute to :toy2 in the Autotune cache (Phase 9
limit, unchanged); a page-table-reading kernel (skipping the
gather) remains a PACKET; §LXXXIII stays parked until G1∧G2∧G3 are
honestly green with a measured factor.
