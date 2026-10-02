# /goal PHASE 10C — NAMED-MODEL HYGIENE (importer + golden + SmolLM2 fork-bytes)

**For:** Buffy (mechanical implementation). **Gauntlet this.**
**From:** Grok (encoding owner)
**Depends on:** Phase 10 / 10B LANDED; demo-box snapshot + G2 already
measured (ops, not this sprint).
**Canon §LXXXIII stays PARKED.** Do not fill `Representation.jl`.
Do not mark representation COMPLETE. Do not fuse decode. Do not
edit `libs/Lava`. Do not fork Julia. Do not add torch to any
`Project.toml`. Packets 1 and 2 stay closed.
**This is not an optimization pass.** The 0.692× G2 factor stands.
Do not retune kernels to move it.

**Status:** COMPLETE 2026-10-02

---

## One-sentence objective

HuggingFace SmolLM2-135M loads under the real name map, the CPU
golden for `"Hello"` is frozen as a Gesso-CPU regression oracle,
and `fork` on that named model still shares pages as bytes.

## Why this sprint exists

Living Phase 10/10B built the speed-floor door. Ops placed
`snapshots/SmolLM2-135M` (gitignored) and published G2. Three
holes remain on the named-model path, all glue:

1. The released checkpoint is HuggingFace 0-based (`model.layers.0`
   … `n-1`), omits `mlp_bias` (HF default false), and *other* HF
   dumps sometimes carry reconstructed RoPE `*.rotary_emb.inv_freq`.
2. `test/fixtures/smollm2/expected_logits.toml` is still absent —
   `test_smollm2.jl` records Broken, not a fake file.
3. G3 share is proven on llama_micro (`unique_kv_bytes` 2048 vs
   4096). The analogous figure on SmolLM2 does not exist.

CI never downloads. Without `GESSO_SMOLLM2_DIR`, items B and C are
named skips; item A (micro fixtures) always runs.

## Start condition

You inherit the tree after 10B (`e01845d` class) plus whatever
uncommitted importer work is already in:

- `src/Inference/llama_import.jl`
- `test/test_import_llama.jl`

**Keep that work if it matches the laws below. Do not revert it.
If the tree is clean of it, implement the same laws.**

Demo box (do not commit these):

```
GESSO_SMOLLM2_DIR  →  <repo>/snapshots/SmolLM2-135M
                      (config.json, model.safetensors, vocab.json,
                       merges.txt, tokenizer.json)
GESSO_EAGER_PYTHON →  <repo>/snapshots/.venv/bin/python   # G2 only; unused here
```

`snapshots/` is gitignored. Never add weights or the venv.

G1 load/generate already works on this snapshot (13 pass / 1 Broken
golden). G2 is measured in `benchmark/results/2026-10-02.tsv`
(eager warmed 0.436 s / Gesso CUDA warmed 0.630 s = **0.692×**,
Gesso F32 vs eager bfloat16). This sprint does not republish that
factor.

If Phase 10/10B is unfinished, stop.

## What this sprint is not

- fused decode, page-table attention, FlashAttention
- Autotune new candidates, more operators, RPDO search
- `rope_scaling`, SentencePiece, a second `model_type`
- sampling beyond greedy, `generate` that does not reset, 5b serving
- compiler attribution table, PrecompileTools, a Julia fork
- Magenta, cages, ExactBits, `src/Representation/`
- torch.compile / vLLM extra rows
- committing `snapshots/` or `.venv`
- a G2 re-bench as the product of this sprint

---

## Laws (item A — importer)

Closed name map, fail-closed leftovers (§LXX / §LXXVI).

### `mlp_bias`

HuggingFace `LlamaConfig` defaults `mlp_bias` to false. SmolLM2-135M's
released `config.json` (transformers 4.40.1) **omits** the key.

- Missing key ⇒ `false`.
- Present and `false` ⇒ `false`.
- Present and `true` ⇒ error that names `mlp_bias`.

Not a silent representation change. Do not add `mlp_bias` to
`_REQUIRED_CONFIG_KEYS`.

### Layer origin

HuggingFace Llama is 0-based: `model.layers.0` … `n-1`.
Gesso micro fixtures were written 1-based: `model.layers.1` … `n`.

Detect from `q_proj.weight`:

```
q0 = "model.layers.0.self_attn.q_proj.weight"
q1 = "model.layers.1.self_attn.q_proj.weight"
```

- `q0` present ⇒ layer ids `0:(n-1)` (HuggingFace).
- else `q1` present ⇒ layer ids `1:n` (fixture).
- neither ⇒ error naming **both** keys.
- mix: `q0` and `q1` and `n > 1` **and**
  `model.layers.$n.self_attn.q_proj.weight` present ⇒ error that
  names the mix (0-based and 1-based). A real HF dump with `n>1`
  always has both `layers.0` and `layers.1`; the discriminator is
  the extra fixture layer `n`.

Materialize `model.layers.$i.$suffix` for `i` in the detected range.

### Leftover keys

Unknown checkpoint keys error WITH THE KEY NAME. The **only**
known-ignored leftover suffix is reconstructed RoPE:

```
*.rotary_emb.inv_freq
```

Gesso computes RoPE from `rope_theta` and does not consume
`inv_freq`. This snapshot (SmolLM2-135M, 272 BF16 tensors, layers
`0..29`) has **no** `inv_freq` and no `lm_head.weight`. The ignore
is for other HF dumps. Any other leftover (e.g. `q_proj.bias`)
still errors. If both an unknown weight and `inv_freq` exist, the
error names the unknown weight **and** the ignored suffix.

Do not ignore `lm_head.weight` — tied-head identity stays as it is.

---

## Work items (sequence)

### A — Land the importer laws

**Objective.** Micro fixtures (1-based) still load. An HF-shaped
micro (0-based) loads. A mix errors. Omitted `mlp_bias` loads.
`mlp_bias: true` errors. `*.rotary_emb.inv_freq` is ignored.
A leftover real weight still errors with the key name.

**Permitted files**

```
src/Inference/llama_import.jl
test/test_import_llama.jl
```

`make_micro_checkpoint` may take `layer_origin=:fixture|:hf`
(default `:fixture`) so the new tests can write both origins.
Do not change production `load_llama` signatures.

**Tests** (all always-on, no snapshot):

- omitted `mlp_bias` ⇒ config loads
- `mlp_bias: true` ⇒ error names `mlp_bias`
- `layer_origin=:hf` (n=2): blocks[1] is `layers.0`, blocks[2] is
  `layers.1`; no `layers.2`
- `layer_origin=:fixture`: blocks[1] is `layers.1`, blocks[2] is
  `layers.2`
- mix (`:hf` plus extra `layers.2.q_proj.weight`) ⇒ error contains
  `mixes`, `0-based`, `1-based`
- HF micro + `layers.{0,1}.self_attn.rotary_emb.inv_freq` extra
  keys ⇒ `load_llama` succeeds, 2 blocks
- fixture micro + leftover `q_proj.bias` ⇒ error contains
  `unknown tensor` and `q_proj.bias`
- HF micro + `inv_freq` + `q_proj.bias` ⇒ error contains
  `q_proj.bias` and `.rotary_emb.inv_freq`

Existing unknown/missing/untied tests stay green.

**Artifact.** Real HuggingFace Llama dumps load. Fixtures still load.
Fail-closed leftovers.

---

### B — Freeze the SmolLM2 CPU golden

**Objective.** `test/fixtures/smollm2/expected_logits.toml` exists,
provenance `oracle = "gesso-cpu"`, and `test_smollm2.jl` compares
last-position prefill logits of `"Hello"` to it at `atol=1e-2`,
`rtol=0`. The Broken skip for "golden not frozen yet" is **gone**
when the snapshot is present. Without the snapshot the whole file
is still one named skip.

**Permitted files**

```
test/freeze_smollm2_golden.jl          # writer; create
test/fixtures/smollm2/expected_logits.toml
test/test_smollm2.jl                  # reader; drop the unfrozen skip
test/test_cuda_smollm2.jl             # only if it assumed the old table shape
```

**Format** (compact TOML — 49152 tables of `{v=…}` is forbidden):

```toml
[provenance]
oracle = "gesso-cpu"
prompt = "Hello"
position = "last"
dtype = "Float64"
vocab_size = 49152
model = "HuggingFaceTB/SmolLM2-135M"
gesso_commit = "<git rev-parse --short HEAD>"
# snapshot identified by files present, never a Hub revision fetch
snapshot = "local GESSO_SMOLLM2_DIR (config.json + model.safetensors + vocab.json + merges.txt)"

values = [ … ]   # length 49152, Float64, last-position logits
```

Reader: `golden["values"]` as a vector of 49152 Float64s.
`provenance.oracle` must equal `"gesso-cpu"`. Do not invent a
second (HF) comparison this sprint.

**Writer.** `test/freeze_smollm2_golden.jl`, run under the **test**
project (TOML is a test dep, not core):

```
GESSO_SMOLLM2_DIR=/path/to/SmolLM2-135M \
  julia --project=test test/freeze_smollm2_golden.jl
```

- refuses if `GESSO_SMOLLM2_DIR` is unset or incomplete (never
  downloads)
- writes the path above
- refuses to overwrite unless `--force` (CI must not regenerate)
- uses `Gesso.reference_prefill` + snapshot tokenizer; last column
  of the logits matrix
- does not import torch

**Do not fake the file.** Produce it once on this snapshot from
Gesso CPU. Two prefills of the same prompt must match before you
commit (writer can assert that).

**Skip law unchanged:** no `GESSO_SMOLLM2_DIR` ⇒ named skip, CI
green. Golden may be committed; the skip happens before compare.

**Artifact.** Item D of Phase 3 is actually frozen. Demo-box
`GESSO_SMOLLM2_DIR=… make test` has **zero** SmolLM2 Broken for
the CPU golden.

---

### C — SmolLM2 `fork` unique_kv_bytes

**Objective.** On the named model, after `prefill!` of `"Hello"`
then `fork` and **before any decode**:

```
unique_kv_bytes(parent.mgr, child.mgr) == kv_footprint(parent.mgr)
```

Two independent prefills of the same prompt:

```
unique_kv_bytes(a.mgr, b.mgr) == kv_footprint(a.mgr) + kv_footprint(b.mgr)
```

CPU Session is the named-model share figure (F64 storage). CUDA,
when `CUDA.functional()`, repeats the **identities** (fork == one
session; isolated == sum). Do not require CUDA bytes == CPU bytes
(F32 vs F64).

**Permitted files**

```
test/test_session_smollm2.jl
benchmark/runbenchmarks.jl            # optional BYTE rows, skip-or-land
benchmark/results/                    # append-only if you add BYTE rows
```

**Pins**

```
prompt            "Hello"
max_new_tokens    unused (no decode before the byte check)
page_size         16
context_length    128
eos_token_id      0
backend           CPU (required); CUDA skip-or-green
```

Do **not** guess the integer in this file. Measure, then pin it in
the test (`@test uniq == N` and `@test isolated == 2N`) so silent
growth fails. Receipt records N (CPU) and, if measured, CUDA N.

Expected class (sanity, not law): one live page per cache, 30
layers × 2 (K,V) × 16 × 3 kv-heads × 64 d_head × 8 (F64) =
1_474_560 bytes for one CPU session if `"Hello"` fits in one page.
If the measured N disagrees with that class, say why in the receipt
(token count, extra pages) — do not "fix" sharing to hit the guess.

Without `GESSO_SMOLLM2_DIR`: named skip, same as the rest of the
file. llama_micro 2048 vs 4096 tests stay untouched.

Optional: schema 0.2.0 BYTE rows `smollm2_kv_bytes_prefill_fork_cpu`
and `smollm2_kv_bytes_two_isolated_prefill_cpu` when the snapshot
is present (ns fields 0, `bytes=N`). Named skip otherwise. Do not
time this.

**Artifact.** G3 has a SmolLM2 byte figure, not only llama_micro.

---

### D — maps; §LXXXIII still parked

**Objective.** Status sentences tell the truth: importer loads HF
0-based SmolLM2; golden frozen; named-model fork bytes recorded;
G2 factor still 0.692× from `2026-10-02.tsv`; representation still
parked.

**Permitted files**

```
README.md
docs/ARCHITECTURE.md
docs/research/ROADMAP_NOW.md
docs/research/README.md
docs/research/SPEED_FLOOR.md      # G1/G2/G3 status sentences only
docs/Gesso_Stack.md               # SPEED FLOOR status note under §LXXXIII;
                                  # do NOT mark §LXXXIII COMPLETE
docs/goals/PHASE10C_NAMED_MODEL_HYGIENE.md   # this file: Status COMPLETE + receipt
scripts/freeze.jl
```

Cite the CPU unique_kv_bytes integer. Cite golden provenance
`gesso-cpu`. Do not rewrite Phase 10/10B receipts. Do not claim
"faster than PyTorch."

**Do not** commit or rewrite: `AGENTS.md`, PHASE4/5/6/7/8/9 extras,
`KV_MEMORY_PROGRAM.md`, `EXOTIC_CAPABILITY_MASTER_LEDGER.md`,
`RPD_SOP.md`, `Julia Compiler Optimizations for Gesso/`, `1x`,
`snapshots/`, `libs/`.

---

## Cross-item invariants

- toy2 CPU fingerprint bit-identical
- llama_micro CUDA fingerprint class unchanged (cite the two floats
  if they reprint)
- greedy ids exact toy2 + llama_micro; SmolLM2 Session ids still
  equal `reference_generate` on `"Hello"` × 8
- `fork` unique_kv_bytes 2048 vs 4096 on llama_micro still holds
- `reference_*` signatures unchanged
- JSON only core third-party hard dep; TOML stays a **test** dep
- Autotune.jl imports neither CUDA nor Lava
- CUDA still `supports(:argmax)` and `:attn_gemm`; Lava still does not
- `libs/` porcelain 0
- `Representation` / `Planning` / `Runtime` still empty
- no receipt/bench schema bump (0.1.0 / 0.2.0)
- no `generate`-does-not-reset, no fusion, no page-table kernel
- `make test` without `GESSO_SMOLLM2_DIR` stays green (named skips)
- `make format` run
- never download; never talk to huggingface.co

## Escalation (stop and write a packet)

- snapshot requires `rope_scaling`, interleaved RoPE, attention
  bias, MLP bias, or a non-silu act
- leftover keys other than `*.rotary_emb.inv_freq` appear on
  **this** SmolLM2 dump and you want a second ignore
- two Gesso-CPU prefills of `"Hello"` disagree (oracle is not
  deterministic — stop)
- `fork` unique_kv_bytes(parent, child) ≠ kv_footprint(parent) on
  SmolLM2 after prefill (share is broken on the named model)
- golden file cannot live in git even as a compact `values` array
- you want torch, PyCall, or a Hub client to produce the golden
- you want to open §LXXXIII, Magenta, or a fusion sprint

## Performance target

N/A. Bytes, not nanoseconds. Do not treat a G2 re-run as success
of this sprint. If you happen to `make bench`, append-only TSV;
llama_micro is still not the board factor.

## Expected artifact

- HF 0-based Llama dumps load; fixtures still load
- committed `expected_logits.toml` with `oracle = "gesso-cpu"`
- SmolLM2 CPU `unique_kv_bytes` integer pinned by test and receipt
- maps truthful; §LXXXIII parked

## Exit checklist

- [x] A: importer laws + micro tests green without snapshot
- [x] B: golden frozen from Gesso CPU; demo-box `test_smollm2.jl`
      has no Broken
- [x] C: SmolLM2 fork identities green; N pinned; llama_micro
      2048 vs 4096 untouched
- [x] `GESSO_SMOLLM2_DIR=<snapshot> make test` green (1556 pass /
      0 Broken; CUDA green on the demo box)
- [x] `make test` green with the env **unset** (1526 pass / 3 Broken,
      baseline-identical)
- [x] `make format` / format-check
- [x] toy2 CPU fingerprint unchanged; CUDA max|Δlogit| reprint
      this run is 0.0004109930905542569 (inside atol=1e-3; 10B class
      print was 0.00037607177004872483 — F32 spread, not a kernel change)
- [x] `snapshots/` and `.venv` uncommitted
- [x] D: maps + this file Status COMPLETE + §LXXII receipt
- [x] no Representation fill, no fusion, no Julia fork

## Receipt (§LXXII — filled at close 2026-10-02)

- **what changed.** Item A: the inherited uncommitted importer work was
  verified against these laws and kept verbatim (HF 0-based detection from
  `q_proj.weight`, fixture 1-based still loads, mix refuses naming both
  bases, omitted `mlp_bias` ⇒ false / true ⇒ error, `*.rotary_emb.inv_freq`
  the ONLY ignored leftover, any other leftover fails closed with the key
  name). Item B: `test/freeze_smollm2_golden.jl` (writer: refuses without a
  complete local snapshot, refuses overwrite without `--force`, asserts two
  exact prefills before freezing) + the frozen golden. Both golden readers
  migrated to the compact `values` array (`test_smollm2.jl`,
  `test_cuda_smollm2.jl`). Item C: fork byte identities pinned in
  `test_session_smollm2.jl` (CPU N pinned; CUDA identities only). Item D:
  maps + this file.
- **why.** Item D of Phase 3 was an unfrozen skip; G3 had no named-model
  byte figure; the released HF checkpoint (0-based) could not load.
- **tests.** `test_import_llama.jl`: all green incl. the 8 new spec blocks
  (20 new asserts). Env **unset**: 1526 pass / 3 Broken — identical to the
  pre-sprint baseline (named skips preserved). Env **set** (demo snapshot):
  **1556 pass / 0 Broken / 0 errors**; `test_smollm2.jl` 15/15,
  `test_cuda_smollm2.jl` 8/8, `test_session_smollm2.jl` 10/10.
- **numerical delta.** CPU `unique_kv_bytes(parent, child) ==
  kv_footprint(parent) == N` with **N = 1_474_560**; two isolated prefills
  `== 2N = 2_949_120`. The measured N EQUALS the sanity class (30 layers ×
  2 × 16 × 3 kv-heads × 64 d-head × 8 F64 × 1 live page — "Hello" is ONE
  token, token id 19556), so the class guess needed no receipt apology.
  CUDA repeats the identities with N = 737_280 (F32 storage = F64/2,
  exactly). Golden: 49152 Float64 values; `smollm2 cuda-vs-cpu-golden
  max|Δlogit| = 2.34e-4` (atol=1e-2). toy2 CPU vs CPU remains
  bit-identical. toy2 CUDA max|Δlogit| reprinted this run as
  0.0004109930905542569 (inside atol=1e-3; 10B class print was
  0.00037607177004872483 — F32 spread, this sprint did not touch
  kernels). llama_micro 5.5006127839263286e-6 matches the 10B
  print. llama_micro fork 2048 vs 4096 untouched.
- **before benchmark.** G2 factor 0.692× stands (`2026-10-02.tsv`) — not
  re-claimed here; no kernel was touched.
- **after benchmark.** N/A. Bytes, not nanoseconds; no BYTE rows appended.
- **compile-time impact.** None — no new package code paths; importer edits
  are runtime branches in the load path.
- **memory impact.** Golden file 1_001_986 bytes committed;
  SmolLM2 CPU KV per session = 1_474_560 bytes under the pinned knobs.
- **hardware.** RTX 5060 (CUDA identity repeat); CPU F64 oracle host.
- **workload.** `"Hello"` prefill, page_size 16, context_length 128,
  eos_token_id 0; no decode before the byte check.
- **model.** HuggingFaceTB/SmolLM2-135M — local `GESSO_SMOLLM2_DIR`
  snapshot; never downloaded, never touched huggingface.co.
- **backend.** CPU (identities repeated on CUDA; CUDA bytes not required to
  equal CPU bytes — F32 vs F64).
- **deviations.** One, declared: the golden emits `values` BEFORE
  `[provenance]` — the spec's example order (table first, then `values`)
  parses `values` INTO the provenance table (TOML has no return to root),
  violating the reader law `golden["values"]`. The reader contract won.
- **known limitations.** The golden is Gesso-CPU-self-referential by law (a
  regression oracle, not an external reference); the writer refuses to
  regenerate without `--force` by law; §LXXXIII remains parked; CUDA N is
  printed, not pinned (device-storage asymmetry).
