# Saluki 27B verify-window oracle on CPU

Target: Underdog Saluki 27B 1.0 (GSQ-RCO IQ2 mix, 7.89 GB, qwen35 hybrid with 16
full-attention and 48 GDN blocks, no MTP). It does not fit the 8 GB card, so it runs on
the CPU backend only. The drafter is the Qwen3.5-0.8B bean, which has the same vocabulary.

Script: `scripts/saluki_window_oracle.jl`. Five chat prompts (thinking off), 96 new tokens
each, greedy. Xeon E5-2699 v3, 16 threads, page cache warm.

## Method

- Plain greedy decode gives the transcript and the time per step.
- For each window size T, the same decode runs with a source that always proposes the
  transcript's continuation. The ids must equal the transcript (all runs did). Time per
  round over time per plain step is W(T), the window's cost at full acceptance.
- One teacher-forced bean prefill over the transcript gives the bean's greedy id at each
  position. That gives the exact accepted length of a chained bean draft, so tokens per
  round follow without running the bean inside the window.
- Predicted speedup = plain time / (rounds × round time + drafted ids × bean step time).

## Chunking fix (953ab0b)

The first run had W(4) of 2.4 to 2.7. The multi-row quant matmul split rows into one
static chunk per thread, so each call waited on the slowest thread. It now uses the ticketed
split the decode GEMV uses. Output bits are unchanged and the full test suite passes.

| T | W(T) before | W(T) after |
|---|---|---|
| 2 | 1.9–2.0 | 1.34–1.63 |
| 4 | 2.4–2.7 | 1.97–2.27 |
| 8 | 3.8–4.9 | 3.44–4.79 |

## Results after the fix

Plain decode is 664 to 783 ms per step (1.3 to 1.5 tok/s). The bean is stored as BF16 and
widened to F32, so its step is 168 ms.

| Prompt | Bean top-1 | Tokens/round T=2 / 4 / 8 | Speedup T=2 | T=4 | T=8 |
|---|---|---|---|---|---|
| json | 0.964 | 1.87 / 3.50 / 5.60 | 0.99 | **1.28** | 1.12 |
| code | 0.905 | 1.90 / 3.39 / 5.28 | 1.07 | 1.13 | 0.82 |
| shell | 0.710 | 1.73 / 2.46 / 3.14 | 1.11 | 0.91 | 0.59 |
| math | 0.916 | 1.94 / 3.65 / 6.33 | 1.08 | **1.23** | 1.07 |
| prose | 0.495 | 1.48 / 1.83 / 1.94 | 0.86 | 0.68 | 0.36 |

Before the fix the best results were math T=4 at 1.16 and json T=4 at 1.11.

## Q8_0 bean and an 8-column tile (9bae144)

The BF16 bean's cost was a quarter to a third of each T=4 round. The bean is now
quantized to Q8_0 (774 MiB, 8.52 BPW, made with llama-quantize), and its step drops to
79–82 ms. The earlier estimate was 45 ms, so it is not purely bandwidth-bound. Its greedy
text differs slightly from the BF16 bean, but top-1 acceptance held or rose.

The multi-column quant tile is now 8 columns instead of 4, so T=8 unpacks each weight
block once. On the T=8 microbench the ratio against one GEMV fell 8–39% on every tensor.
Block 2's IQ1_S tensors stay near 5×. In the oracle, W(8) on code, shell, math and prose is
3.47–4.09, against 3.44–4.79 before. That is inside the run noise, so the end-to-end gain is
not proven.

Log: `.local/2026-10-23_saluki-window-oracle-tile8.log`, which is local only. The plain
step was 691–697 ms on shell, math and prose but 876 ms on code and 1559 ms on json, which
has only 28 decode steps. Those two baselines are inflated by noise. So the table also
gives each speedup normalized to a 690 ms plain step, computed as
tokens per round × 690 / (round time + bean time per round).

| Prompt | Bean top-1 | Tokens/round T=2 / 4 / 8 | Raw T=4 | Normalized T=2 | T=4 | T=8 |
|---|---|---|---|---|---|---|
| json | 0.964 | 1.87 / 3.50 / 5.60 | 3.56 (bad baseline) | 1.11 | **1.58** | 1.50 |
| code | 0.916 | 1.90 / 3.52 / 5.59 | 1.59 | 1.11 | **1.25** | 1.07 |
| shell | 0.725 | 1.73 / 2.46 / 3.14 | 0.93 | 1.05 | 0.92 | 0.64 |
| math | 0.916 | 1.94 / 3.52 / 5.94 | 1.46 | 1.17 | **1.46** | 1.33 |
| prose | 0.484 | 1.48 / 1.79 / 1.90 | 0.72 | 0.88 | 0.72 | 0.43 |

An earlier Q8 run at tile 4 gave raw T=4 speedups of 1.53 json, 1.51 code and 1.47 math.
So with a cheap bean, T=4 is worth about 1.25–1.6× on structured output.

## Prefill

- User-facing prefill goes through the window path (`_prefill_window_ok`) and runs at
  4.27 prompt tok/s on 128 tokens. Time to first token on a 128-token prompt is about 30 s.
- `reference_prefill`, the teacher-forced logits API this oracle uses for the bean, takes a
  separate per-row path. It runs at 0.89 tok/s on Saluki at either tile width. It only
  matters for scoring and oracle work, but it could be routed through the window path.

## What limits it

- Prose (0.48 top-1) and shell (0.73) lose at every T. Window size should follow the
  drafter's confidence or the content type, not a fixed T.
- A verify round at T=4 still costs 1.9–2.3 plain steps. Block 2's IQ1_S multi kernel is
  the worst tensor.
- These are oracle predictions. A real bean-in-the-loop draft source is needed to measure
  the actual speedup.

## Real n-gram drafter (SuffixSource)

Script: `scripts/saluki_draft_bench.jl`. This is an end-to-end run, not an oracle. The
source is `SuffixSource` (prompt lookup): the longest suffix of the context that occurred
earlier proposes the tokens that followed it. It runs on the host in microseconds, so the
measured time is the whole cost. Prefill and the first token are timed once and
subtracted. Every drafted run reproduced the plain ids. Each setting ran once, so
results within about ±0.05 of 1.0 are noise.

With `min_match=1` (the old behaviour), a single shared token triggers a draft. Most of
those drafts are rejected and each one still pays for a verify window:

| Prompt | Speedup k=3 | k=7 | Accept rate k=3 |
|---|---|---|---|
| json | 0.76 | 0.60 | 0.13 |
| code | **1.16** | 0.79 | 0.58 |
| shell | 0.83 | 0.56 | 0.15 |
| math | 0.84 | 0.64 | 0.25 |
| prose | 0.78 | 0.61 | 0.04 |

`SuffixSource` now takes `min_match` (74cd8c1). Speedup at k=3, with the best k=7 result:

| Prompt | min=2 | min=3 | min=4 | Best k=7 |
|---|---|---|---|---|
| code | 1.20 | **1.34** (accept 0.90) | 1.29 | 1.32 (min=4) |
| math | 1.09 | 1.04 | **1.10** | 1.02 |
| json | 0.81 | 1.02 | 1.00 | 1.01 |
| prose | 0.98 | 1.02 | 1.02 | 1.04 |
| shell | 0.85 | 0.92 | 0.94 | 0.87 |

- A higher `min_match` removes the losses because the drafter stops proposing. At min≥3 it
  proposes nothing on json and almost nothing on prose, so those are break-even, not gains.
- Code gains up to 1.34x, because the answer repeats the prompt's file.
- Shell still loses at every setting: its matches are long but the continuations differ.
- The n-gram drafter suits code (and other copy-heavy output) with min_match=3, k=3. The
  bean remains the drafter for other text. The default stays `min_match=1` until a caller
  picks a policy.

Logs (local only): `.local/2026-10-23_saluki-suffix-bench.log`,
`.local/2026-10-23_saluki-suffix-minmatch.log`.

## Real bean drafter (2026-10-24)

Script: `scripts/saluki_bean_bench.jl` (4c92a41). The Q8_0 bean runs as a `DraftSource`
inside the verify window, end to end, with 96 new tokens per prompt (json stops at 29). The
bean's prompt prefill is primed outside the timed decode and reported apart, like Saluki's.
Every drafted run reproduced the plain ids. Each setting ran once.

`GESSO_BEAN_TAU` gates the draft: drafting stops at the first token whose bean top-1
probability is below tau. `GESSO_BEAN_SELFTEST=1` runs the bean as its own target, which
must accept every self-draft. It passes at tau 0 and 0.6.

Speedup over plain decode (plain step 662–700 ms):

| Prompt | k=2 τ=0 | k=3 τ=0 | k=4 τ=0 | k=2 τ=0.5 | k=3 τ=0.5 | k=4 τ=0.5 | k=4 τ=0.7 |
|---|---|---|---|---|---|---|---|
| json | 1.32 | 1.48 | 1.49 | 1.36 | 1.50 | 1.58 | **1.61** |
| code | 1.32 | 1.32 | 1.31 | 1.28 | **1.35** | 1.29 | 1.33 |
| shell | 1.03 | 0.93 | 0.85 | 1.19 | 1.15 | **1.23** | 1.11 |
| math | 1.41 | 1.43 | 1.33 | 1.45 | **1.50** | 1.45 | 1.44 |
| prose | 0.89 | 0.76 | 0.67 | **0.99** | **0.99** | **0.99** | 0.96 |

Accept rate rises from 0.26–0.91 ungated to 0.81–0.98 at τ=0.5 and 0.94–1.0 at τ=0.7.

- The ungated bean matches the oracle on json, code and math (1.3–1.5×). As the oracle
  predicted, it loses on prose and about breaks even on shell.
- Gating at τ=0.5 removes the losses. Shell turns into a 1.15–1.23× gain. Prose is about
  break-even, because the bean still runs at every step to propose a token that is then
  gated away. That overhead is 13–14% of prose wall time.
- τ=0.9 gates too much: windows shrink to one or two tokens and the bean cost stays.
- k=3 or k=4 with τ=0.5 is a single setting that never loses. k=4 τ=0.5 averages 1.31×
  over the five prompts and k=3 τ=0.5 averages 1.30×.
- Code with the bean (1.35×) is level with the n-gram drafter at min_match=3 (1.34×). The
  n-gram drafter is free, so combining them (n-gram first, bean when it is silent) is the
  next drafter step.

Noise: two runs stalled machine-wide, with bean and Saluki time both inflated. In this run
they were code k=2 τ=0 at 0.48 and code k=2 τ=0.9 at 0.62. The previous ungated run had
the same kind of stall on json (0.05 / 0.73 / 0.34 at k=2/3/4), and those three settings
measure 1.32 / 1.48 / 1.49 here. The table uses this run, and the earlier ungated run for
code k=2 τ=0 (1.32). Code k=3 τ=0 also ran slow in the earlier run (1.07 against 1.32
here). The other ungated settings on code, shell, math and prose agree across the two runs
within ±0.05.

Logs (local only): `.local/2026-10-23_saluki-bean-bench.log` (ungated),
`.local/2026-10-24_saluki-bean-gated.log` (gated, with a `.vmstat` alongside).

## Combined drafter (2026-10-24)

Script: `scripts/saluki_combo_bench.jl` (43ae128). The n-gram `SuffixSource` (min_match 3)
proposes first, and the gated bean (τ=0.5) runs only when the n-gram is silent. An optional
back-off `after:skip` answers the next `skip` proposals empty after `after` consecutive gated
bean calls, so those steps run plain with no bean. Each prompt also reran the bean alone as a
same-session reference. Every run reproduced the plain ids. `GESSO_BENCH_ONLY` reruns a
subset of prompts.

Speedup over plain decode (plain step 661–692 ms):

| Prompt | Bean k=3 | Combo k=3 | +2:4 | +3:8 | Bean k=4 | Combo k=4 | +2:4 | +3:8 |
|---|---|---|---|---|---|---|---|---|
| json | 1.39 | 1.42 | 1.54 | 1.46 | 1.52 | 1.54 | 1.56 | 1.57 |
| code | **1.38** | 1.20 | 1.31 | 1.32 | 1.31 | **1.39** | 1.37 | 1.38 |
| shell | 1.16 | 1.05 | 1.01 | 1.03 | **1.20** | 1.10 | 1.01 | 1.03 |
| math | 1.48 | 1.43 | 1.42 | 1.43 | 1.46 | 1.50 | **1.52** | 1.50 |
| prose | **1.00** | **1.00** | 0.96 | 0.96 | 0.99 | 0.99 | 0.95 | 0.96 |
| mean | 1.28 | 1.22 | 1.25 | 1.24 | 1.30 | **1.30** | 1.28 | 1.29 |

The json column is the mean of two runs. In all eight json settings the n-gram never fired
and nothing was skipped, so the drafts were identical. Even so, single json runs ranged from
1.36 to 1.64. With only 29 new tokens, json differences below about 0.1 are noise. Two json
runs in the first pass stalled machine-wide (0.55 and 0.23, with the bean at 2214 ms per
window against the usual 280). A rerun replaced them. During the stall, I/O pressure
averaged 11% over a minute with 15 GB of swap in use, which points to the mmapped weights
being evicted and re-read from disk.

- **The combination does not beat the bean alone.** At k=4 the means are level (1.30). At
  k=3 the combo is lower (1.22 against 1.28).
- **Code** gains at k=4 (1.39 against 1.31) but loses at k=3 (1.20 against 1.38). The n-gram
  answers 12–18 of 23–29 windows, and its drafts are accepted less often than the bean's
  (0.82 against 0.91 at k=3).
- **Shell** loses at every setting (1.01–1.10 against 1.16–1.20). The n-gram fires on only 3
  to 5 windows, but almost none of those drafts are accepted: acceptance drops from 0.93 to
  0.62–0.72. This is the same pattern as the n-gram alone (long matches whose continuations
  differ).
- **Back-off never helps.** On prose it skips 32–40 bean calls and the bean's share of wall
  time falls from 0.15 to 0.10, but accepted tokens drop from 26 to 15–17, so the speedup
  falls to 0.95–0.96. On shell it skips 8–12 calls and loses 0.07–0.09. A run of gated bean
  calls does not predict that the next calls will be gated too.
- The best single setting is still the gated bean alone at k=4 τ=0.5 (1.30 here, 1.31 in the
  earlier run). Prose stays at break-even.

Next steps for the drafter, if it gets more time:

- Let the n-gram stop proposing after its drafts are rejected (a per-source acceptance
  record). That would keep the code gain without the shell loss. On these numbers it is
  worth about +0.02 on the mean.
- Gate the bean on Saluki's own confidence, which costs nothing, instead of the bean's
  history. Prose is where the bean's 14–15% share of wall time is spent proposing tokens that
  are then gated away.

Logs (local only): `.local/2026-10-24_saluki-combo-bench.log`,
`.local/2026-10-24_saluki-combo-json-rerun.log`, with a `.vmstat` alongside the first.

## Qwen3.5-2B as the drafter (2026-10-24)

The same script (`scripts/saluki_bean_bench.jl`) with `GESSO_BEAN_GGUF` pointed at
Qwen3.5-2B Q4_K_M instead of the 0.8B Q8_0 bean. The question was whether a bigger, more
accurate drafter pays for its extra cost. Every run reproduced the plain ids. Each setting ran
once.

Speedup over plain decode (plain step 663–720 ms), with the 0.8B bean from the combined run
for reference:

| Prompt | 2B k=3 τ=0.5 | 2B k=3 τ=0.7 | 2B k=4 τ=0.5 | 2B k=4 τ=0.7 | 0.8B k=3 τ=0.5 | 0.8B k=4 τ=0.5 |
|---|---|---|---|---|---|---|
| json | (1.06) | 1.52 | (0.67) | **1.60** | 1.39 | 1.52 |
| code | 1.18 | 1.10 | 1.20 | 1.28 | **1.38** | 1.31 |
| shell | 1.13 | 1.12 | 1.14 | 1.12 | 1.16 | **1.20** |
| math | 1.36 | 1.40 | 1.37 | **1.51** | 1.48 | 1.46 |
| prose | 0.95 | 0.93 | 0.92 | 0.95 | **1.00** | 0.99 |
| mean without json | 1.16 | 1.14 | 1.16 | 1.22 | **1.25** | 1.24 |

Values in parentheses stalled machine-wide: the 2B took 696 and 2369 ms per window there
against its usual 380–500, with swap-in running. The means leave json out, because of those
stalls and because json differences below about 0.1 are noise (29 tokens).

- **The 2B does not beat the 0.8B.** Its acceptance is only slightly higher (0.82–1.00
  against 0.81–0.98), but each window costs 380–640 ms against about 280. The extra cost eats
  the gain. Its best setting (k=4 τ=0.7, 1.22 without json) is level with or below the 0.8B
  at τ=0.5 (1.24–1.25).
- **Prose gets worse, not better** (0.92–0.95 against 0.99–1.00). At τ=0.5 the 2B accepts
  only 0.67–0.70 of its prose drafts. At τ=0.7, 50 of its proposals are gated away and it still takes
  17% of wall time.
- The 2B gains only where the 0.8B is already good (json, math) and only at τ=0.7.

Conclusion: a bigger off-the-shelf Qwen drafter is not the way forward on CPU. The next
candidates are drafters trained against Qwen3.8-27B itself (its MTP head and DFlash2), which
read Saluki's own hidden states instead of running a separate model.

Log (local only): `.local/2026-10-24_saluki-2b-drafter.log`.

## Published drafters through llama.cpp (2026-10-24)

Script: `scripts/saluki_llamacpp_drafter.sh`. Before porting a drafter into Gesso, this
measures the drafters trained for base Qwen3.8-27B on Saluki through llama.cpp's server
(CPU only, 16 threads, same prompts, greedy, 96 new tokens): the official MTP head
(`draft-mtp`, one extra layer reading Saluki's final-normed hidden state), z-lab's DFlash2
(`draft-dflash`, a block drafter reading five target layers) and the 0.8B bean as a plain
draft model (`draft-simple`). Both trained drafters are the Q4_0 files from
ggml-org/Qwen3.8-27B-GGUF. Saluki is a fine-tune of that model, so these read hidden
states they were not trained on. `pN` labels add llama.cpp's draft confidence gate
(`LLAMA_ARG_SPEC_DRAFT_P_MIN` = 0.N): drafting stops at the first token whose top
probability is below it.

Speedup over llama.cpp plain decode (plain 650–722 ms per token, the first run's numbers
for every column):

| Prompt | MTP 2 | MTP 3 | MTP 3 p50 | DFlash 4 | DFlash 7 | DFlash 7 p50 | DFlash 7 p70 | bean 4 |
|---|---|---|---|---|---|---|---|---|
| json | 1.64 | 1.45 | 1.44 | 1.50 | 1.83 | 1.77 | 2.04 | 1.28 |
| code | 1.65 | 1.82 | 1.77 | 1.73 | **2.38** | 2.21 | 2.15 | 1.38 |
| shell | 1.28 | 1.15 | 1.30 | 1.13 | 1.06 | 1.32 | **1.38** | 0.96 |
| math | 1.48 | 1.56 | 1.47 | 1.78 | 2.11 | **2.19** | **2.19** | 1.34 |
| prose | 1.02 | 0.91 | **1.15** | 0.79 | 0.59 | 0.99 | 1.08 | 0.69 |
| mean without json | 1.36 | 1.36 | 1.42 | 1.36 | 1.54 | 1.68 | **1.70** | 1.09 |
| texts equal to plain | 5/5 | 3/5 | 4/5 | 3/5 | 2/5 | 2/5 | 3/5 | 3/5 |

Draft acceptance, DFlash 7 p70: json 0.92, code 0.93, shell 0.68, math 0.90, prose 0.67.
MTP 3 p50: 0.70, 0.93, 0.63, 0.87, 0.68.

- **DFlash2 with a 0.7 gate is the fastest drafter measured on Saluki: 1.70x without
  json**, 2.15–2.19x on code and math. Ungated, its long drafts are mostly rejected on prose
  (acceptance 0.15, 0.59x). The gate fixes that: prose draft tokens drop from 321 to 60 and
  acceptance rises to 0.67.
- **The MTP head reaches 1.42x with a 0.5 gate** (k=3). The gate moved shell from 1.15 to
  1.30 and prose from 0.91 to 1.15. On code, math and json it changed almost nothing, because
  the head is already confident there.
- **Both trained drafters beat the bean** in every setting. The bean as a llama.cpp draft
  model reaches only 1.09x, below Gesso's own gated bean (1.24–1.25x on the same prompts).
- Both trained drafters work on Saluki even though they were trained for the base model:
  acceptance on code and math is 0.87–0.93.

**Text differences are llama.cpp numerics, not drafter errors.** Plain decode is repeatable
(a second plain run gave the same three texts). The speculative runs that differ from plain
diverge at a near-tie, where Saluki's top two tokens are within 0.008 nats (code, token 53),
0.042 (shell, token 12) and 0.103 (prose, token 16) of each other, checked with saved top-2
probabilities. llama.cpp's batched verify computes those logits slightly differently from
its single-token decode, and the near-tie flips. Gesso's batched verify is exact (every Gesso
drafter run above reproduced the plain ids), so a port into Gesso should keep all five texts.

What this means for Gesso:

- Port the MTP head first. It is one layer (about 0.95 GB of Q4_0 weights read per draft
  step, against 7.89 GB for a Saluki step) and needs only Saluki's final-normed hidden rows,
  which the verify window already computes.
- Then DFlash2 (five layers, block of 8, reads target layers 5, 19, 33, 47 and 61). It is
  the bigger prize (1.70x) but needs per-layer hidden rows from the target.
- Both need the confidence gate.

Logs (local only): `.local/2026-10-24_saluki-llamacpp-drafters.log`,
`.local/2026-10-24_saluki-llamacpp-drafters-2.log`, token dumps in
`.local/2026-10-24_lcpp-tokens/`.
