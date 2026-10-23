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

## What limits it

- At T=4 the bean costs 505 ms per round against a 1.3 to 1.6 s verify round, a quarter
  to a third of the total. A quantized bean is the next lever: Q8_0 is about a quarter of
  the bytes, so its step should be roughly 45 ms if it stays bandwidth-bound. That is an
  estimate until measured.
- W(8) is still 3.4 to 4.8. The multi kernel tiles four rows at a time, so T=8 unpacks
  each weight block twice. A wider tile is the second lever.
- Prose acceptance (0.50 top-1) is too low for any window to pay. Window size should
  follow the drafter's confidence or the content type, not a fixed T.
