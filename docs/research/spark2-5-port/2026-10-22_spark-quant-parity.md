# Spark quant parity against BF16 (2026-10-22)

Goal: a quantized Spark-X2.6-4B that keeps at least ~95% of BF16 quality. Each quant is
measured against BF16 logits with `llama-perplexity --kl-divergence`. These are parity
gates: no fresh task benchmarks were run.

## Method

- Reference: Spark-X2.6-4B BF16 GGUF. It is fully on the RTX 5060 for both eval sets.
- Wiki set: wikitext-2 `wiki.test.raw`, 60 chunks of 512 tokens. BF16 PPL is 19.24.
- Science set: the 198 GPQA Diamond questions and choices, each followed by its
  reasoning snippet from the earlier GSQ-RCO run. That gives 88 chunks of 512 tokens;
  BF16 PPL is 4.37. This set is only used for evaluation, never for calibration. Its
  snippets came from the GSQ-RCO model, so if anything it favours that quant.
- imatrix: built from BF16 over 300 chunks of `wiki.train.raw`.
- llama.cpp 0a1ef309c, CUDA build.

## Results

| Quant | Size GB | Wiki KLD | Wiki top-1 | Wiki PPL ratio | Sci KLD | Sci top-1 |
|---|---|---|---|---|---|---|
| GSQ-RCO (original schedule) | 2.14 | 0.967 | 59.6% | 2.49 | 0.231 | 82.6% |
| GSQ-RCO schedule + imatrix | 2.14 | 0.704 | 65.0% | 1.97 | — | — |
| Q3_K_S | 1.90 | 1.099 | 56.4% | 2.56 | — | — |
| Q3_K_M | 2.25 | 0.339 | 74.4% | 1.25 | 0.134 | 86.3% |
| Q4_K_M | 2.60 | 0.158 | 81.8% | 1.035 | — | — |
| Q4_K_M + imatrix | 2.60 | 0.190 | 81.1% | 1.17 | — | — |
| Q5_K_M + imatrix | 2.98 | 0.083 | 88.3% | 1.10 | — | — |
| Q6_K + imatrix | 3.38 | 0.022 | 93.6% | 1.012 | 0.0049 | 97.5% |
| **A8F6** (attn Q8_0, FFN Q6_K, imatrix) | **3.69** | **0.0107** | **95.5%** | **1.018** | **0.0032** | **98.3%** |
| A8F6 + ffn_down Q8_0 | 3.92 | 0.0093 | 95.9% | 1.021 | — | — |
| Q5_K_M + attn Q8_0 | 3.38 | 0.028 | 92.4% | 1.016 | — | — |
| Q5_K_M + attn Q6_K | 3.15 | 0.034 | 91.8% | 1.014 | — | — |
| Q8_0 | 4.38 | 0.0059 | 96.9% | 1.017 | 0.0010 | 98.9% |

## Per-class sensitivity

This ablation starts from Q8_0 and drops one tensor class to q4_k in every layer.

| Class at q4_k | Wiki KLD | Top-1 |
|---|---|---|
| attn_gate | 0.018 | 94.6% |
| attn_qkv | 0.066 | 88.8% |
| attn_output | 0.066 | 88.5% |
| ffn_gate | 0.035 | 91.5% |
| ffn_up | 0.044 | 90.4% |
| ffn_down | 0.047 | 90.3% |

## Findings

- The original GSQ-RCO schedule misses parity by a wide margin. It puts Q3_K on 27 of the
  36 layers, attention included. Spark is sensitive to quantization in every tensor class,
  and the attention projections are the most sensitive. An imatrix only partly recovers
  the loss.
- At about the same size, plain Q3_K_M beats the GSQ-RCO schedule.
- An imatrix does not help at Q4_K_M on this set. It helps at Q6_K and above.
- A8F6 is the knee. It sets attn_qkv, attn_output, attn_gate and token_embd to Q8_0,
  sets the FFN to Q6_K, and uses the imatrix.
  - On the wiki set it lands within 0.005 KLD of Q8_0, at 690 MB smaller.
  - On the science set it agrees with BF16 on top-1 for 98.3% of tokens.
  - Moving ffn_down up to Q8_0 as well adds 230 MB and gains only 0.0014 KLD.
- token_embd is also the output head, because Spark ties its embeddings. That is why it
  stays at Q8_0.

## Recipe

```
llama-quantize --imatrix imatrix-wikitext2-train-300.gguf \
  --tensor-type-file a8f6-tensor-types.txt --token-embedding-type q8_0 \
  Spark-X2.6-4B-BF16.gguf Spark-X2.6-4B-A8F6-imat.gguf Q6_K 16
```

`a8f6-tensor-types.txt` sets `blk.N.attn_qkv`, `blk.N.attn_output` and `blk.N.attn_gate` to
`q8_0` for N = 0..35. The quant, imatrix and types file sit beside the model, in the
Spark model directory and its `experiments/` folder.

## Gesso S1: A8F6 in the Gesso oracle

Gesso's `load_gguf` loads A8F6, which uses Q8_0 and Q6_K blocks. `reference_prefill` and
`reference_generate` ran on CPU with 8 threads. The reference was llama.cpp on CPU,
greedy, with 32 new tokens.

| Prompt | Tokens | Greedy match | Max Δlogprob, llama top-10 |
|---|---|---|---|
| p1 (short) | 5 | 2/32 | 0.143 |
| p2 (instruction) | 18 | 32/32 | 0.950 |
| long (prose) | 992 | 32/32 | 0.185 |

- p1 diverges at token 3. That step is a near-tie in llama.cpp: token 198 scores −1.431
  and token 614 scores −1.454, a gap of 0.023.
- p2's 0.95 gap is on its 4th-ranked token, 5248. Against a BF16 llama.cpp reference,
  Gesso is closer: BF16 gives it −2.785, Gesso −3.315 and llama.cpp A8F6 −4.265. Over the
  BF16 top-10 for p1 and p2, Gesso's error and llama.cpp A8F6's error are about the same
  size. Both engines carry the same quant noise, and Gesso adds no error of its own.
- Speed on CPU:
  - Decode is about 0.24 s per token at 1k context. llama.cpp is 0.27 s.
  - Prefill is 3.8 tok/s at P = 992. llama.cpp is 44 tok/s, so Gesso prefill is about
    12× slower. That gap is the S3 throughput target.
- The paged `Session` refuses Spark's ops: the attention gates, sliding windows,
  per-layer RoPE and GEGLU. Spark runs only through the oracle's cached decode.

## Open

- GPQA Diamond on A8F6 with a long thinking budget: `llama-server`, full GPU offload,
  thinking effort high, temperature 1.0, top-p 0.95, `max_tokens` 24000.
  - 113 of 198 correct, 57.1%. This is a lower bound: 60 items reached the 24k cap
    without an answer. On the 138 items that finished, 113 were correct (81.9%).
  - The vendor's per-item file (`orcein_spark_gpqa_diamond_calibrated_outputs.jsonl`)
    is not the BF16 77.1% or 69.9% run. It scores 40.4% with completions capped at
    about 3.6k tokens. So it is not a parity anchor, and the vendor's BF16 figure is
    still unverified.
  - BF16 parity for Spark rests on the KLD and top-1 gates above. A GPQA comparison needs
    a BF16 run under the same budget, or a budget large enough that nothing truncates.

## Gesso S3: CPU prefill (paused at 17 tok/s)

Timed with `reference_prefill` on 256 tokens of the long prompt, 8 threads.

| Change | Prefill s | tok/s |
|---|---|---|
| Baseline (one column per dot) | 59.8 | 4.3 |
| Multi-column Q6_K and Q8_0 tiles | 54.3 | 4.7 |
| Fused float step per tile | 44.2 | 5.8 |
| Threaded GEGLU for prefill-sized arrays | 37.0 | 6.9 |
| Q6_K pair-order kernel (pmaddubsw + pmaddwd, exact i32 lanes) | 34.9 | 7.3 |
| Q8_0 tile: phadd transpose and one vector float step | 28.0 | 9.2 |
| Attention reads only unmasked keys; exact inverse scale | 24.2 | 10.6 |
| Q6_K activations shuffled once per token; 32-row blocks | 23.2 | 11.0 |
| Activation quantize and shuffle split across threads | 20.8 | 12.3 |
| GELU erf and erfc from openlibm, with z's rounding corrected | 19.4 | 13.2 |
| Attention scores accumulate in Float32 when the scale is a power of two | 18.3 | 14.0 |
| Q6_K and Q8_0 tiles read their activation scales from one per-tile table | 18.3 | 14.0 |
| Attention in one threaded row pass; threaded rmsnorm, RoPE and logits transpose | 15.0 | 17.1 |

- llama.cpp baseline: `llama-bench` pp256, CPU only (`-ngl 0`), 8 threads, gives 48.0 ±
  0.3 tok/s, about 5.3 s for 256 tokens. At equal threads the gap is about 3.4×. The
  earlier "5.8 s, 3.2×" figure recorded no thread count. At 16 threads llama.cpp reaches
  86 tok/s.
- Each change is bit-equal to the single-column kernels, including a Q6_K extremes
  test. Greedy parity against llama.cpp is unchanged: p1 2/32, p2 32/32, long 32/32,
  with the same Δlogprob figures as in S1.
- `Profile` runs of this script with stdout redirected to a file segfault (3 of 3, a
  garbage program counter); through a pipe they finish, and the same prefill without
  the profiler has not crashed in 12 timed runs or the parity runs.
- The machine was shared during the last row (load 4 to 10), so its timing is the
  slower end of 22.0 to 23.4 s over 9 runs. 8cfaaa6 timed 23.9 to 24.3 s in the same
  session. The 992-token parity prompt went from 142 s to 127 s end to end.
- Q6_K step, measured alone on one 2560 × 10240 matrix with 256 tokens:
  - Row by row, the pre-shuffled activations are slower: 99 ms against 85 ms. They are
    twice the bytes (the shuffled values plus their offset words), and every weight row
    re-reads all 256 tokens.
  - In 32-row blocks the tile's tokens stay in cache. That gives 70 ms with the shuffle
    in the kernel, and 63 to 65 ms pre-shuffled.
  - The 10240-wide ffn_down shape gains less: 84 ms down to 76 ms.
  - An 8-token tile is no faster than 4.
- The activation quantize ran on one thread: about 45 ms per layer, 1.7 s per prefill.
  Split over 8 threads it takes 1.7 to 6.5 ms per matmul instead of 4 to 25 ms. Timed
  back to back at the same load, the commit before took 22.2 to 23.2 s.
- GELU's left tail ran a 120-step continued fraction, 970 ns per value against 18 ns
  elsewhere. openlibm's erfc at the rounded z = -x/√2, plus the exact first-order term
  for that rounding, takes 76 ns. It is within 1.4e-15 relative of MPFR over the whole
  tail. Greedy parity and every Δlogprob figure are unchanged, and the 992-token
  prefill went from 112 s to 90 s.
- Next target: the Q6_K tile still runs at about a third of its port-0 bound, and
  ffn_down's 32 rows no longer fit in L2.
- The attention scores at d_head = 256 scale by an exact 1/16. The old path widened each
  term to Float64 and rounded back. A Float32 sum rounded through Float64 equals the
  Float32 sum, because 53 ≥ 2·24 + 2, so the loop now stays in Float32 and vectorizes.
  The bit-equality tests and greedy parity are unchanged. The 256-token prefill went
  from 18.9–20.0 s to 18.2–18.4 s, though the second set ran at lower load (3 against
  5). The 992-token parity prompt fell from 142 s to 97 s end to end, with the erf change
  included.
- The tile kernels gathered each column's activation scale `d` separately, once per
  block: 4 loads for Q8_0, and 32 lane products for Q6_K. A per-prefill table now stores
  each tile's 4 `d` values side by side, so each block needs one load. The products are
  the same, so the bit-equality tests still pass. In one process, interleaved at load 10,
  a 2560 × 10240 Q6_K matmul over 256 tokens went from 59.8 to 58.1 ms median, and a
  2560 × 4096 Q8_0 matmul went from 32.2 to 30.8 ms. The end-to-end prefill timed
  18.3–21.1 s while load rose from 4 to 12, which is too noisy to show a gain that size.
  Greedy parity is unchanged.
- A main-thread profile of the 256-token prefill (894 samples at 20 ms) found about 3 s
  of serial work while the other 7 threads waited. The causal softmax ran on one thread
  between the score and value passes of every head, which cost 40 samples. RoPE and
  rmsnorm cost 30 each, and transposing the (seq, vocab) logits cost 21.
  - Prefill attention now does a row's scores, softmax and outputs for every head in one
    threaded pass over the rows. It keeps the per-head arithmetic element for element,
    and the bit-equality tests cover windowed rows and a chunk after cached keys
    (L < K). Prefill went from 18.3 s to 16.6 s (3 reps, load 3 to 7), and the
    992-token parity prompt fell from 97 s to 77 s.
  - rmsnorm, RoPE and the logits transpose now split their rows across threads at
    prefill sizes. That brought prefill to 14.7–15.3 s (load 9 to 14), and the parity
    prompt to 71 s.
  - A numerical failure inside a threaded task is rethrown as the same `GessoError`. Tests
    cover this for rmsnorm and for the softmax.
  - Greedy parity is unchanged after each step.
- Where the remaining time goes: a main-thread profile after the last row puts 84% of
  samples in the quant matmuls. Timed alone at 256 tokens and 8 threads, one block's
  seven matmuls take 257 ms and lm_head takes 1.0 s, about 10.2 s of the 15 s.
  - Q6_K (the FFN) runs at 0.068 to 0.071 ns per weight-token per thread, about 3× its
    in-L1 step rate of 0.023.
  - Q8_0 (attention, lm_head) runs at 0.093 to 0.103.
  - The step is memory bound. Each Q6_K block step reads 210 bytes of weights but
    8 × 512 bytes of prepared activations, and the tile's activations (40 KB at
    2560 inputs, 160 KB for ffn_down) do not fit in L1.
- Next levers, held until Ornith, Qwen and LFM run on Gesso:
  - A tile of 2 rows × 8 tokens, so each activation load feeds two weight rows.
  - Applying the Q6_K −32 offset through per-block sums instead of the per-pair `mo`
    words, which halves the activation bytes. It changes the int32 lane sums, so the
    decode kernel has to change with it to keep prefill and decode bit-equal.
