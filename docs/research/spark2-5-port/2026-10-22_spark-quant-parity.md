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

## Open

- Run GPQA Diamond on A8F6 with a long thinking budget and compare against the
  vendor's BF16 number. Earlier runs were capped at 1.1k–4k tokens and truncated many
  answers, so their scores are not comparable.
- Gesso S1: load A8F6 (Q8_0 + Q6_K blocks) in Gesso and check greedy parity against
  llama.cpp.
