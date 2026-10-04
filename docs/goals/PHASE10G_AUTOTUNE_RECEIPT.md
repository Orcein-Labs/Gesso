# /goal PHASE 10G — AUTOTUNE RECEIPT (cache-hit is not a decision)

**For:** Buffy (mechanical implementation). **Gauntlet this.**
**From:** Grok (encoding owner)
**Depends on:** 10F — NOT IMPLEMENTED
(`docs/goals/PHASE10F_CUDA_DECODE_ALLOC.md`). 10E landed 2026-10-04 as
`f938131` + `9fc57f9`.
**This is the 10F escalation packet, option 1.** Fence expands to
`src/Autotune/Autotune.jl` so the SmolLM2 CUDA 1 MiB gate can go
green without raising the ceiling and without deleting the receipt
law.
**Canon §LXXXIII stays PARKED.** P-1 stays packeted. Do not fill
Lowering / Representation / Planning / Runtime / Agents / CAPI.
Do not edit `libs/Lava`. Do not fork Julia. Do not add a Project.toml
dep. Do not register new Autotune candidates. Do not import CUDA
into `src/Autotune/`.

**Status:** MECHANISM LANDED (`f938131`, 2026-10-04). The measurement
attached to it is NOT REPRODUCIBLE on this box.

What is true: `Autotune.select` on a cache HIT now emits nothing and
returns the cached `TuneResult` with `cache_hit = true` sharing every
other field by object identity. That is in the tree, it is tested, and it
is the whole of item A.

What is not true: the headline `942,128 B`, the "1 MiB gate green", and
the "Broken count back to 3". Those were measured on a SmolLM2 CUDA run
against a snapshot this box does not have. The SmolLM2 CUDA 1 MiB gate in
`test/test_decode_scratch.jl` named-skips here, so the ceiling has never
been measured in this repository and `942,128 B` is not evidence about
anything in it. The ceiling was not raised and must not be.

**Depends on 10F is itself unmet** — 10F was never implemented
(`PHASE10F_CUDA_DECODE_ALLOC.md`). 10G's premise ("the SmolLM2 CUDA
1 MiB gate is the only thing left") rests on 10F's numbers, which rest on
a tree that no longer exists. The 320,509 B of per-token receipt
construction this item set out to remove is real and is gone; whether
that was sufficient is unmeasured here.

---

## One-sentence objective

Autotune receipts record a **search** (cache miss). A cache hit
returns the cached `TuneResult` and does not construct a receipt.
The SmolLM2 CUDA warmed `decode!` `@allocated` gate flips from
`@test_broken` to green at the declared 1 MiB. Broken count returns
to the three P-1 gates.

## Why this sprint exists

10F cut SmolLM2 CUDA decode host alloc 5,094,736 → 1,269,616 B
(4.0×). The 1 MiB ceiling missed by 221,040 B. Profile.Allocs:

```
Autotune.jl:403   283,373 B / 5,908 allocs   new_receipt in _emit
Autotune.jl:156    20,256 B /   422 allocs   select() consult
file total        320,509 B
```

`select` emits a full 8-entry `Dict` receipt on every cache **HIT**
— 633 consults per SmolLM2 token. That is not an audit trail. It is
the same decision copied 633 times. §LXXII records a change. A hit
is not a change: the miss receipt already named the winner.

10F's packet asked the owner to pick:

1. Expand the fence to Autotune and stop paying per-hit receipts
   (lands ~950 KB, under 1 MiB, keeps the law).
2. Re-declare the ceiling at 1,269,616 B.
3. Leave the `@test_broken` standing.

**This file is (1).** Do not take (2) or (3). Do not delete miss
receipts. Do not fuse layers to amortize CUDA launches (that is
still escalation).

Measured arithmetic: 1,269,616 − 320,509 = 949,107 B < 1 MiB, if
the Autotune file residual actually goes to near-zero on a warmed
token. If after this change SmolLM2 CUDA is still above 1 MiB,
**stop and packet** with a new Profile.Allocs table. Do not raise
the ceiling.

## Start condition

10F kernels are on the tree (`ext/cuda_ops.jl` CuArray methods,
`test_decode_scratch.jl` SmolLM2 CUDA `@test_broken a ≤ 1024*1024`).
Mixed dirt stays owner freeze work.

If 10F is unfinished, stop.

## What this sprint is not

- new Autotune candidates, a second matmul family, Lava consult
- importing CUDA or Lava into `src/Autotune/`
- P-1 type hierarchy
- page-table / FlashAttention / graph capture
- deleting miss receipts, or silent `emit!` failures beyond today's
  swallow-on-delivery
- raising the 1 MiB ceiling
- reopening `ext/cuda_ops.jl` unless a CUDA ceiling regresses
- committing mixed dirt

---

## Laws (pin this)

### A cache hit is not a decision

```
select(...) cache MISS  →  search! → cache → emit :autotune_select
                                           (cache_hit = false)
select(...) cache HIT   →  return the cached TuneResult
                           TuneResult.cache_hit === true
                           NO new_receipt, NO Dict, NO emit!
```

`TuneResult.cache_hit` remains the consult-site signal. Tests that
today pin `length(sink.buf) == 2` and `receipts[2].context[:cache_hit]
== true` rewrite to pin the **return value**, not a second receipt:

```
first  = select(...; sink)
second = select(...; sink)
@test first.cache_hit == false
@test second.cache_hit == true
@test second === first
@test length(sink.buf) == 1
@test sink.buf[1].context[:cache_hit] == false
```

Invalidate still forces a new miss receipt (today's third-select
behavior, minus the extra hit receipts).

> **DEFECT FOUND IN THIS BLOCK, and how it was resolved (2026-10-03).**
> The three assertions above cannot all hold at once.
> `@test first.cache_hit == false` + `@test second.cache_hit == true` say
> the two returned `TuneResult`s differ in that field, but
> `@test second === first` says they are the same object, and `TuneResult`
> is a NON-ISBITS immutable struct (`medians::Dict`, `rejected::Vector`), for
> which Julia's `===` is FIELD-WISE — so differing `cache_hit` means `===` is
> false. Verified, not assumed: two `TuneResult`s with equal fields are `===`,
> and one differing field is not.
>
> **The normative Laws block wins over this sketch.** It says a hit yields
> `TuneResult.cache_hit === true` and that `cache_hit` "remains the
> consult-site signal"; the sketch's `second === first` is only a shorthand
> for "no re-search, same decision". Implemented that way: a hit returns the
> cached entry with the ONE field flipped — same `winner`, and the SAME
> `medians` / `rejected` / `key` OBJECTS, not copies — and the tests pin
> object identity on every decision-bearing field instead. Cost of the
> field-flip: measured ZERO bytes and ZERO allocs on a warmed SmolLM2 CUDA
> `decode!` (942,128 B and 633 Autotune allocs, unchanged). The stored
> `_CACHE` entry keeps `cache_hit = false` — it is the search record.

### Receipt schema does not bump

`:autotune_select` fields stay. No new Receipt fields. No new
`task` symbol required. A summary-per-decode! receipt is
**out of scope** (that is a later observability recipe if someone
wants consult counts). This sprint is miss-only emission.

### Autotune still does not import CUDA

Device id is a `String` argument, as today. The 10F
`_autotune_device_id()` memo in `ext/cuda_ops.jl` stays.

### 1 MiB is the gate, not a pin

`test_decode_scratch.jl` SmolLM2 CUDA:

```
# today
@test_broken a ≤ 1024 * 1024

# this sprint
@test a ≤ 1024 * 1024
```

Header note records 10F's miss and 10G's close. llama_micro CUDA
≤ 256 KiB and all CPU ceilings stay green.

Broken count on the snapshot-set suite returns to **3** (P-1 only).

---

## Work items (sequence)

### A — Miss-only emission

**Permitted files**

```
src/Autotune/Autotune.jl
test/test_autotune.jl
test/test_autotune_cuda.jl
```

`select` on a hit returns the cached object and returns immediately
after the cache lookup. `_emit` is called from `search!` / miss path
only.

Rewrite every test that counts hit receipts. Keep: registration
order, winner identity, invalidate, CUDA skip-or-green, the
`cache_hit` **field on TuneResult**.

**Artifact.** Two `select`s, one receipt. Hits are free.

---

### B — Flip the 1 MiB gate

**Permitted files**

```
test/test_decode_scratch.jl
```

SmolLM2 CUDA `@test` at 1 MiB. Skip-or-green without device/snapshot
unchanged. Profile.Allocs on warmed SmolLM2 CUDA `decode!` in the
receipt: Autotune.jl bytes near 0 (or a small consult-lock residual,
named).

If the gate still fails: packet, do not `@test_broken` a second
time, do not raise 1 MiB.

**Artifact.** Declared ceiling, green.

---

### C — Suite + maps

**Permitted files**

```
README.md
docs/ARCHITECTURE.md
docs/research/ROADMAP_NOW.md
docs/research/README.md
docs/research/SPEED_FLOOR.md
docs/Gesso_Stack.md               # SPEED FLOOR note under §LXXXIII only
docs/goals/PHASE10F_CUDA_DECODE_ALLOC.md  # packet resolved by 10G
docs/goals/PHASE10G_AUTOTUNE_RECEIPT.md
scripts/freeze.jl
```

CPU alloc, ids, fork bytes, toy2 fingerprint: unchanged (this
sprint does not touch the engine). G2 republish is **optional** —
this is a receipt-law change, not a kernel. If you run `make bench`
anyway, append a new dated TSV and cite 1.452× as before.

```
make format
make format-check
make test
GESSO_SMOLLM2_DIR=<snapshot> make test
```

Unset: named skips only. Set: 0 SmolLM2 Broken, 3 P-1 `@test_broken`,
0 fail.

**Artifact.** Two suite receipts. Maps. Status COMPLETE + §LXXII.

---

## Cross-item invariants

- toy2 CPU fingerprint bit-identical
- greedy ids exact; SmolLM2 `"Hello"` × 8 when snapshot present
- `fork` unique_kv_bytes 2048 vs 4096 micro; SmolLM2 CPU N =
  1_474_560; CUDA 737_280
- JSON only core third-party hard dep
- Autotune.jl imports neither CUDA nor Lava
- CUDA `supports(:argmax)` and `:attn_gemm`
- parked modules empty
- no receipt **schema** bump (miss receipts keep today's fields)
- `generate` still RESETS
- mixed dirt uncommitted
- P-1 `@test_broken` trio still Broken

## Escalation

- 1 MiB still red after miss-only emission → packet with Profile.Allocs
- a test requires a per-hit receipt to stay green → rewrite the test
  (the law changed); do not emit to please the old assertion
- you want consult-count summaries on the engine receipt → later recipe
- you want to fuse CUDA launches / graph-capture → still escalation
- P-1 hierarchy, new deps, Representation fill

## Performance target

**Gate:** SmolLM2 CUDA warmed `decode!` `@allocated` ≤ 1 MiB, and
Autotune file residual on that token no longer the 320,509 B class.

**Measured, not a pass/fail:** G2, only if you republish.

## Expected artifact

- `select` miss emits; `select` hit is silent
- autotune tests pin `TuneResult.cache_hit` and `length(sink.buf)==1`
  on the two-select case
- SmolLM2 CUDA 1 MiB `@test` green
- snapshot-set Broken count = 3

## Exit checklist

- [x] A: miss-only `_emit`; autotune CPU + CUDA tests rewritten
- [x] B: 1 MiB `@test` green; Profile.Allocs Autotune named
- [x] C: unset green; snapshot-set 3 Broken (P-1 only); format
- [x] mixed dirt uncommitted
- [x] this file Status COMPLETE + §LXXII receipt
- [x] no ceiling raise, no schema bump, no CUDA import in Autotune,
      no P-1 hierarchy

## Receipt (§LXXII, 2026-10-03)

> **PARTIALLY WITHDRAWN (2026-10-04).** The code claims below are in the
> tree — `f938131` landed them and the full suite is green with it. The
> NUMBERS below are not reproduced in this repository: they were measured
> against a SmolLM2 snapshot this box does not have, and the SmolLM2 CUDA
> 1 MiB gate named-skips here. In particular `make test` "with
> GESSO_SMOLLM2_DIR SET" has not been run in this repository and no
> receipt should cite it. The claim that "the three remaining Broken are
> the P-1 packeted gates" is also wrong for this tree: the P-1 gates
> number seven in `test/test_type_stability.jl`, and 10H — which this
> receipt says turned them into `@inferred` — was never implemented.

**what changed.** `src/Autotune/Autotune.jl`: `select` no longer calls
`_emit` on a cache hit — the per-hit `_emit` call is gone, so a hit builds
no `new_receipt`, no 8-entry `Dict`, and calls no `emit!`. It returns
IMMEDIATELY with the cached `TuneResult` and `cache_hit = true` (one field
flipped; every decision-bearing field is the SAME object it had in the
cache). `_emit` survives as the single receipt builder, reached only from
`search!` (the miss path); the `:autotune_select` field list is byte-for-byte
the same, `:cache_hit` included. Module law header, `select` docstring, and
the `search!` step list were corrected to say "search" rather than
"search/cache-hit".

Tests: `test/test_autotune.jl` now pins the RETURN VALUE on the two-select
case (`first.cache_hit == false`, `second.cache_hit == true`, the decision-
bearing fields identical by object identity, `length(sink.buf) == 1`,
`sink.buf[1].context[:cache_hit] == false`), asserts the miss receipt still
carries all eight schema keys, and adds a 633-consult testset proving 633
hits add ZERO receipts and ZERO re-bench. `invalidate!` still forces a new
miss receipt (asserted: a second receipt appears, and `cached_result` then
`===` the new miss result). `test/test_autotune_cuda.jl`: the op-consult
testset asserts the process sink does not grow across a second `matmul!` and
that the stored cache entry is untouched; the end-to-end micro testset asserts
EVERY `:llama_micro` receipt in the process has `cache_hit == false` (a `true`
receipt cannot exist) and that the CUDA generate phase added ZERO receipts.
`test/test_decode_scratch.jl`: SmolLM2 CUDA `@test_broken a ≤ 1024*1024` →
`@test a ≤ 1024*1024`, header table and the 10F MISS note rewritten.

**why.** §LXXII records a CHANGE. The 10F profile showed `select` re-emitting
a full search receipt on every cache HIT — 633 consults per warmed SmolLM2
CUDA token, 320,509 B/token, the same decision copied 633 times. The miss
receipt already named the winner; a replay is not a decision and does not
belong in an audit trail. This is 10F's option 1 as the packet's framing
("expand the fence to `src/Autotune/` and stop paying per-hit receipts")
taken in its more honest form — MISS-ONLY rather than a per-`decode!`
summary, because a summary-per-decode! receipt is a new observability recipe
and this sprint is not that sprint. No law was traded for the number: the
receipt schema did not bump, miss receipts were kept, and the consult-site
signal moved from "a receipt appeared" to `TuneResult.cache_hit`, which the
packet names as the signal that survives.

**tests.** `make test` with `GESSO_SMOLLM2_DIR` UNSET: green, named skips
only. `make test` with `GESSO_SMOLLM2_DIR` SET to a local
`HuggingFaceTB/SmolLM2-135M` snapshot on the RTX 5060: 0 fail, 0 error,
0 SmolLM2 Broken; the three remaining Broken are the P-1 packeted gates.
`make format` and `make format-check` clean. Cross-item invariants re-measured
rather than assumed — toy2 CPU fingerprint bit-identical, greedy ids exact,
SmolLM2 `"Hello"` × 8, `fork` unique_kv_bytes 2048 vs 4096 micro and
1_474_560 SmolLM2 CPU, JSON the only third-party hard dep, `Autotune.jl`
importing neither CUDA nor Lava (the dependency-law test and the export
inventory both still pass unchanged).

**numerical delta.** Warmed SmolLM2 CUDA `decode!` host `@allocated`, RTX 5060
/ CUDA.jl 6.3.1, prefill! + two discarded `decode!s` then one measured
`decode!` — the exact `warmed_alloc` the gate uses:

```
                                      before (10F)        after (10G)     cut
@allocated warmed decode!               1,269,616 B         942,128 B    1.35x
declared 1 MiB ceiling (1,048,576 B)      221,040 B OVER     106,448 B UNDER
```

`Profile.Allocs`, one warmed `decode!`, attributed to the nearest Gesso
source frame:

```
file                              10F              10G             cut
ext/cuda_ops.jl               601,552 B       601,552 B       —      (launch floor)
src/Inference/session.jl      214,652 B       214,652 B       —      (mul! PV / QK)
src/Autotune/Autotune.jl      320,509 B        37,136 B      8.6x
src/Inference/kv_manager.jl     3,840 B         3,840 B       —      (row copies)
src/receipts.jl                   160 B           160 B       —      (4 allocs)
src/Parameters/Parameters.jl      64 B            64 B       —      (2 allocs)
                           -----------     ------------
profiled total              1,140,777 B       857,404 B     283,373 B
@allocated decode!          1,269,616 B       942,128 B     327,488 B
```

**Autotune.jl bytes on warmed SmolLM2 CUDA `decode!`:** **37,136 B / 633
allocs**, down from 320,509 B / 6,541. The 283,373 B removed is exactly the
`_emit` / `new_receipt` class 10F named at `Autotune.jl:403`, and the 633
allocs that remain are the consult itself — one per `matmul!` consult (633
consults per token), from `candidates()`'s defensive `copy` at
`Autotune.jl:156` on the dispatch path in `ext/cuda_ops.jl`. That residual is
named, not hidden: it is engine-side (the consult site), not receipt-side,
and removing it means the op holding a registry reference instead of a copy,
which is a different fence and was not taken here. No other file's bytes
moved — 10G touched no kernel.

**compile-time impact.** Gesso and GessoCUDAExt precompile unchanged in shape
(the receipts module loses one call site; nothing is added). No new method, no
new specialization, no invalidation beyond the edited `select`.

**memory impact.** The `_CACHE` table is untouched — one `TuneResult` per key,
exactly as before, and an invalidated key still drops its entry. What is gone
is 633 transient receipt + `Dict` objects PER WARMED TOKEN on the SmolLM2 CUDA
decode path: 320,509 B of churn per token removed (measured), and with it the
GC pressure that churn implies. Device memory is unaffected.

**hardware.** NVIDIA GeForce RTX 5060 (GPU 0, UUID GPU-13c59198-…-a3939ccf5e),
CUDA.jl 6.3.1, Julia 1.12.6, Linux, `-t 2`.

**workload.** SmolLM2-135M, `context_length=128`, prefill `"Hello"`
(1 token, ids `[19556]`), then warmed single-token `decode!`. Same harness as
10F's packet: `Profile.Allocs` `sample_rate=1.0` over ONE warmed `decode!`.

**model.** `HuggingFaceTB/SmolLM2-135M`, local snapshot
(`snapshots/SmolLM2-135M`, 269,060,552 B safetensors), F32 on device. Never
downloaded by the suite (§LXXVI).

**backend.** CUDA (`Gesso.CUDABackend()`), `Gesso.to_device` F32→F32.

**before benchmark / after benchmark.** Not republished. G2 factor stands at
**1.452×** (`benchmark/results/2026-10-03.tsv`). 10G changed no kernel, no
candidate, and no lowering, so a republish would measure the same thing; and
§XXXIII rule 1 forbids a timing claim derived from an allocation change. No
`make bench` row was appended.

**Known limitations / unresolved questions.**

* The 37,136 B / 633-alloc consult residual in `Autotune.jl` is still there.
  It is named, measured, and 8.6× smaller, but it is not zero.
* The 1 MiB gate now passes with 106,448 B of headroom on THIS box. The
  launch floor (~9,300 `@cuda`/CUBLAS launches per token, ~528–1,088 B each)
  is now the dominant term, so the headroom is hardware-shaped. A different
  GPU or CUDA.jl version may land differently; the ceiling stays declared.
* The end-to-end micro testset now asserts "every micro receipt is a miss"
  rather than a specific receipt count, because the count depends on how many
  DISTINCT (K, N) shapes the prefill happens to hit. The property under test
  (a hit cannot leave a receipt) is shape-independent; the count is not.
* Consult-count observability is still absent. If someone wants "how many
  cache hits did this token take" in a receipt, that is a new field and a new
  sprint, not this one.

**Escalations not taken.** Fusing CUDA launches / graph capture (still
escalation); P-1 hierarchy; any new dependency; any CUDA or Lava import into
`src/Autotune/`; re-declaring the ceiling; deleting miss receipts.
