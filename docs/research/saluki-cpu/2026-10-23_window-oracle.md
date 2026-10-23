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
