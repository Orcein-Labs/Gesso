# Verify window: bit-exact multi-token decode

Status: design, 2026-10-08. This is the runtime half of `SPECULATION_SEAM.md`. It fills in Phase 20
(§XCIII) and §LV of `docs/Gesso_Stack.md`, and it is the largest software lever for dense models on
bandwidth-bound hardware. Measured on this box (single-channel DDR4, about 14 GB/s), Qwen3.8-27B
IQ4_XS reads 14.4 GB of weights per token. A window of T tokens reads them once.

## Contract

Given accepted context ending at position `p0 - 1`, with last token `t0`, and a draft `d₁ … d_{T-1}`:

1. Run tokens `t0, d₁, …, d_{T-1}` at positions `p0 … p0+T-1` through every layer in one pass.
2. For each window position `i`, compute `a_i = greedy(logits_i)` using the SAME `_greedy_id`
   (first-max tie rule) that plain decode uses.
3. Accept `n` = the largest prefix with `a_{i-1} == d_i` for `i = 1 … n`. Emit `d₁ … d_n` and then
   `a_n`, which is the model's own next token. A window always emits `n + 1 ≥ 1` tokens.
4. Commit state so that it is BITWISE equal to what plain greedy decode holds after emitting those
   `n + 1` tokens.

**Bit-parity invariant:** `logits_i` from the window equals, bit for bit, the logits plain greedy
decode would produce at that position. Speculation therefore can never change output, and the
existing exact-id gates cover it with no new tolerance. Every op on the window path must
satisfy it.

## Per-op parity rules

| Op | Window form | Parity requirement |
|---|---|---|
| Embedding, RMSNorm, RoPE | per row / per position | already per-row; unchanged |
| Quant GEMV, CPU (Q8_K/Q8_0) | **multi-column**: decode each weight block once, then dot it with T activation vectors | each column's activation quantized exactly as in single-token decode; each column's dot keeps the single-column summation order |
| Quant GEMV, GPU (dp4a) | multi-column, as llama.cpp `mmvq` with `ncols_y > 1` | per-column reduction tree identical to `ncols = 1` |
| F16 GEMV, GPU (split-K) | multi-column | split-K partition and reduction order must not depend on T |
| Full attention | one query per window position, causal to its own position | per-query score/softmax/value order identical to decode at that K |
| GDN mixer | projections batched over T; **recurrence sequential over the T tokens** | same per-token inputs; same recurrence kernel per step |
| MoE | per-token routing and top-k; union of experts loaded once | per-token combine in fixed k order |
| lm_head + greedy | per position | same kernel and same tie rule |

A kernel that cannot meet its row is not used on the window path. Correctness outranks speed.

## State commit

- **Attention KV.** The window writes rows `p0 … p0+T-1`. Committing `n + 1` tokens sets the length
  to `p0 + n + 1`. Later rows are stale and get overwritten before any query can read them. The
  attention kernel must bound reads by length, never by buffer size. No copy is needed.
- **GDN conv and recurrent state** (27B: 48 layers × 48 heads × d_k × d_v F32). The window evolves
  a scratch copy of the state, made by one device copy at window start, and records each token's
  recurrence inputs (`q, k, v, g, β` after projection and conv). That record is small.
  - On full acceptance (`n + 1 == T`), the scratch state becomes the committed state.
  - On partial acceptance, keep the original state and replay the recurrence for the first `n + 1`
    tokens from the recorded inputs. Replay runs the recurrence only, with no projections, so it
    is cheap. Strata uses the same scheme (`include/strata/core/verify.hpp`, MIT).
  - Because replay uses the same inputs and the same kernel, the committed state is bitwise equal
    to sequential decode.
- **MTP draft layer** (Qwen3.8 `blk.64`, `nextn_predict_layers = 1`). It has its own KV cache and
  is committed the same way.

## Draft sources

- **MTP (model-native).** The draft forward is
  `eh_proj([enorm(embed(t)); hnorm(h)])` → the `blk.64` attention + FFN → `shared_head_norm` →
  the shared `lm_head` → greedy. It is chained for k drafts. Strata measured 2.4–3.2 tokens per pass
  with 3 drafts on Qwen3.8; that figure must be re-measured here.
- **Host sources** (`src/Speculation/`: suffix, template, symbolic). They are used when they trigger
  and pay. The per-origin pay-off rule from `SPECULATION_SEAM.md` applies.
- **Arbitration** is runtime policy, chosen by measured acceptance and cost per origin.

## Phases (each gated by exact ids and bitwise state equality)

1. **P1 — correctness skeleton (CPU).** A window driver that loops the existing single-token
   kernels over T (no speed gain yet), plus KV length commit, GDN record and replay, the acceptance
   rule, and receipts.
   - Tests: random and adversarial drafts (all wrong, partial, all right, EOS mid-window).
   - Window output equals sequential greedy, id for id.
   - Committed GDN/KV state is bitwise equal to sequential decode.
2. **P2 — multi-column CPU kernels.** Quant GEMV and attention per the table. Gate: P1's tests stay
   bitwise; the 27B CPU tokens/s at T = 2–4 with oracle drafts (replayed known-greedy ids) measures
   the ceiling.
3. **P3 — MTP draft forward** with its own KV and commit; measure acceptance on named prompts.
4. **P4 — GPU multi-column kernels** (dp4a, F16 split-K, fused GDN, attention).
5. **P5 — policy.** Source arbitration, per-origin receipts, T chosen by measured acceptance.

## Expected effect (estimate, to be measured)

Weight reads dominate decode on this box. A window costs about one token's weight read plus T
tokens' activation work. With mean acceptance `ā`, throughput is roughly `(ā + 1) / window_time`.
If `ā + 1 ≈ 2.5` and the window costs about 1.1 single-token steps, that is about 2.3x tokens/s on
the dense 27B on top of any kernel and placement gains.
