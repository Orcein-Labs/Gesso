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
