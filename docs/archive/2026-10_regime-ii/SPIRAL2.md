# RPDO spiral 2 — Lava leftover after frozen parity

Owner: Grok. Repo: Gesso. Layer: Gesso mechanism. Date: 2026-10-07.

Problem: spiral 1 hit frozen CUDA P≥0.90. Same-session CUDA fox
was 33.84 tok/s vs Lava 27.58 (P=0.815). Host alloc 6–8 MB/decode
vs CUDA 1 MB. Decode still D2Hs the vocab row for argmax.

## Load-bearing questions

| Q | Cheap experiment | Disposition |
| --- | --- | --- |
| Q1. Is the live-fox gap extra launches × larger K? | Batched decode attn: 9 heads × (scores+softmax+values) → 1 of each. | `BUILD` retain. Hello decode-only 31.70 tok/s. Three-prompt decode **36.48 / 36.70 / 38.68**. softmax_n 270/token → 30. IDs exact. |
| Q2. Does host argmax of 49k logits own leftover ms? | Device serial argmax, 4-byte D2H, logits `_all_finite` kept. | `REVERT`. Hello attribution 31.70→34.95; three-prompt generate did not move (36.35 / 35.66 / 36.22). Alloc unchanged. |
| Q3. Where is 6–8 MB/decode? | `@allocated decode!` after Q1. | Batched attn cut it to **3.45 / 3.71 / 3.94 MB**. CUDA is 1.06 MB. Leftover for next spiral. |
| Q4. Compiler fork? | Only if leftover is SPIR-V/compile. | `PARK` |

## Candidate 1 (RETAIN): batched decode attention

Workspace scores became `(n_heads, context_length)`. CUDA/CPU still
view row 1. Lava `_attention_heads!` for `L==1` launches three kernels
per layer. Prefill keeps the per-head loop.

Receipts: `s2-attn-hello.json`, `s2-attn-lava-three.json`.
Comparator: `s2-cuda-three.json` (same box, same night).

| Prompt | Lava | CUDA live | P live | CUDA frozen | P frozen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hello | 36.48 | 32.61 | **1.119** | 29.93 | 1.219 |
| The quick brown fox | 36.70 | 35.15 | **1.044** | 28.81 | 1.274 |
| Julia is a programming language. | 38.68 | 35.31 | **1.095** | 28.80 | 1.343 |

Median live P = **1.095**. All three P>1. IDs exact (27/27).

## Candidate 2 (REVERT): device argmax

`s2-argmax-hello.json` / `s2-final-lava-three.json`. Kernel was
correct (IDs exact, planted NaN true) and did not pay rent on the
generate harness. Reverted.

## Closed

Live-fox gap is closed. Host alloc 3.5–4 MB vs CUDA 1 MB is the
leftover. Compiler fork stays PARK.
