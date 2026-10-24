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

## Transaction rules

These rules refine the contract above. They come from Codex's RPDO handoff of 2026-10-08.

- **State is defined by consumed inputs**, not by emitted text. Decode holds a pending anchor
  token that has been emitted but not yet consumed. A window consumes `[anchor, d₁ … d_{T-1}]`.
  If `n` drafts agree, the commit is the state after consuming the anchor and those `n` drafts.
  The model's correction (or bonus token) becomes the next pending anchor. Prefill already
  consumed the prompt, so its last token is never fed twice.
- **EOS and output caps** can stop before the final emitted token is consumed. Match the baseline's
  state convention at those boundaries exactly.
- **Snapshot/commit covers** attention KV contents and valid lengths, conv history, GDN recurrent
  matrices, the last hidden/logit position, consumed and emitted counters, and EOS state. Derived
  scratch may be recomputed only if no rejected value can leak into the continuation. A KV-only
  rewind is not sufficient.
- **Required tests:**
  - forced rejection at every draft depth;
  - all accepted;
  - zero drafts;
  - EOS mid-window;
  - output cap mid-window;
  - cancellation;
  - after every commit, check that the NEXT ordinary decode step equals sequential decode.

## MoE caveat

A draft window can route each position to a different expert set. On MoE models the union of
experts touched per window can approach `T` single-token sets, which erodes weight reuse. Cost
each layer's actual expert union, weighted by encoded bytes and cache residency, before claiming
a gain. The window pays most clearly on dense models such as Qwen3.8-27B. Verifier routing is
never constrained to resident experts in the exact arm: that would change the model.

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
   - Done 2026-10-24 (`Inference.MTPDraft`, fed by the `on_rows` hook). Saluki on CPU: 1.38x
     without json at k = 3, p_min 0.5, ids equal to plain on every prompt. See
     `docs/research/saluki-cpu/2026-10-23_window-oracle.md`.
   - The DFlash2 block drafter (`Inference.DFlashDraft`, fed by `on_taps`) follows the same
     pattern: 1.33x without json at k = 7, p_min 0.7, ids equal to plain. Its acceptance matches
     llama.cpp's, and the 8-row verify window (about 3.5 plain steps) is now the limit.
4. **P4 — GPU multi-column kernels** (dp4a, F16 split-K, fused GDN, attention).
5. **P5 — policy.** Source arbitration, per-origin receipts, T chosen by measured acceptance.

## Expected effect (estimate, to be measured)

Weight reads dominate decode on this box. A window costs about one token's weight read plus T
tokens' activation work. With mean acceptance `ā`, throughput is roughly `(ā + 1) / window_time`.
If `ā + 1 ≈ 2.5` and the window costs about 1.1 single-token steps, that is about 2.3x tokens/s on
the dense 27B on top of any kernel and placement gains.

## Saluki 27B on CPU (2026-10-11)

Measured on the Xeon E5-2699 v3 at 16 threads with the Saluki IQ2-mix GGUF read from NVMe.

- **Decode is at the memory roof.** A step reads 7.61 GB of weights: the file is 7.89 GB, and the
  0.28 GB IQ1_M embedding is only gathered by row. The lm_head is IQ3_S and 0.55 GB. Gesso
  decodes in 649 ms (lane-s cfdb989), the same as llama.cpp tg32 at 650 ms. That is 11.7 GB/s,
  about 84% of the 14 GB/s figure above. Matvec is about 95% of the step. Kernel work alone can
  buy at most about 1.2x. The window is the remaining large lever.
- **No MTP layer.** The blocks end at `blk.63`, and the GGUF has no `nextn` keys. Drafts come
  from the host sources only.
- **Shape.** 64 blocks with full attention on every fourth: 16 attention layers with 24 query
  heads and 4 KV heads of 256, and 48 GDN layers. GDN has 16 key heads, 48 value heads, and
  `d_k = d_v = 128`, with a kernel-4 conv over 10240 channels. The recurrent state is
  48 × 48 × 128 × 128 F32 = **151 MB**. The conv history is 3 × 10240 × 48 F32 = 5.9 MB.

### Recurrent state: fused and out of place

The recurrence reads and writes all 151 MB once per token, about 22 ms at 14 GB/s (an estimate;
the non-matvec GDN time measured 40 ms in the 882 ms split). The scheme above runs it T times,
plus a copy at window start, so it pays (T + 1) state passes. At T = 4 that is about 17% of a step.
Two changes bring it down to 1 + P(reject) passes:

1. **Fuse the recurrence over the window.** `_gdn_delta_slab!` updates one `d_v` column `j` at a
   time. That slab is `nv × d_k` floats (24 KB), and no slab reads another. Apply the T token
   updates to one slab in order while it sits in L1/L2, then move to the next slab. Every element
   goes through the same operation sequence as in sequential decode, so the result is bitwise
   equal. The per-token `core` outputs come out of the same pass.
2. **Write out of place instead of copying.** The fused pass reads the committed state A and
   writes the window state B, so A is the snapshot and needs no copy.
   - On full acceptance, swap A and B.
   - On partial acceptance, run one fused replay pass from A over the `n + 1` consumed tokens
     into B, then swap.

   The record per token per GDN layer is `qn` and `kn` (`nv × d_k` each, after head repeat),
   `vn`, `decay`, `beta`, and the conv input row, about 115 KB. For 48 layers at T = 4 that is
   22 MB. The committed conv history is the last three consumed conv inputs, copied directly
   with no compute.

### Where it hooks in

Hybrid GGUF models decode through `reference_generate` and `GenerateState`
(`src/Inference/Inference.jl`, `generate_state.jl`), not through `Session`.
- Each loop iteration runs `_last_logits`, then `_greedy_id`, then `_decode_step!`. A window
  replaces one iteration.
- An empty draft takes the existing `_decode_step!` path unchanged, so T = 1 cannot regress.
- The window reuses `_decode_step!`'s stage code per row. Only the matvec groups
  (`_dec_matmul_group!` and the lm_head) take T rows.
- `reference_prefill` is not on the window path. Hybrid prefill is gated against decode by KL,
  not bit equality.
- `GenerateState`'s resume snapshot (`_generate_snapshot!`) is a separate facility. The window
  keeps its own A/B pair.

### Cost model for this box

`W(T) ≈ f_w · r(T) + f_tok · T + f_state · (1 + P_reject)`, in units of one decode step:
- `f_w ≈ 0.9` is weight streaming.
- `r(T)` is the multi-column matvec cost ratio. Measure it in place, streaming weights from DRAM.
  An L2-resident microbenchmark measures the unpack and dot compute, not the roof.
- `f_tok ≈ 0.03` is the per-token non-matvec work.
- `f_state ≈ 0.035` is one recurrent-state pass.

If the extra dot work per column hides under the weight stream, `r(4)` is near 1.15. Then
`W(4) ≈ 1.2` and `W(2) ≈ 1.1`, far below the Strata T = 4 figure of 3.05. That would make drafts
cheap: a one-token draft pays when it is right more than about 10% of the time. This depends on
`r(T)` and is unmeasured until P2.

### Acceptance oracle (run before P1 lands)

The window emits exactly the greedy stream, so a source's in-loop acceptance can be computed
offline from a greedy transcript without running the model. At window start `i`:
1. Call `propose(src, ctx(ids[1:i]); max_tokens = T - 1)`.
2. Let `n` be the draft's matching prefix against `transcript[i+1:]`.
3. Call `observe!(src, d, n)`.
4. Advance with `i += n + 1`.

This gives the per-origin windows, mean draft length and mean accepted for any T policy.
- Transcripts from Gesso CPU greedy are exact. llama.cpp greedy is a proxy that differs only at
  near-ties, and results from it must be labelled as proxy.
- Use Cyan-shaped workloads (tool-call JSON, code edits that restate a file, shell-output
  summaries, arithmetic) plus a free-prose control where the suffix source should fail.
- `E[n + 1]` from the oracle and `W(T)` from P2 give the predicted speedup before any policy work.
