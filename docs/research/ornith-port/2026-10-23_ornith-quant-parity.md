# Ornith 1.5 35B-A3B quant parity vs BF16

Target: a quant of `ornith-ai/Ornith-1.5-35B-A3B` (rev 10fbf86f, qwen35moe, 40 layers,
256 experts, 8 active) that keeps at least 95% of BF16 quality. Gates:

- MMLU-Pro (vendor 2000-question subset, conditional-likelihood scoring) ≥ 45.22,
  which is 0.95 × the vendor's BF16 47.6.
- Top-1 agreement with BF16 on science text ≥ 95%.
- Perplexity ratio vs BF16 ≤ about 1.05 on wikitext and science text.

## The vendor GSQ-RCO 3-bit fails the gate

The vendor's own results file gives GSQ-RCO-3 43.7 against BF16 47.6, a 91.8% ratio.
The file we hold matches the vendor's LFS hash, so this is the shipped artifact.

## Reproducing the vendor's MMLU-Pro harness

Every input matches the vendor's recorded hashes: the scorer source, task builder,
subset, task index and packed task file. A CPU build at the vendor's llama.cpp revision
(58367713) still disagrees with the vendor on per-question predictions (17/20 match on
the first 20 rows; mean logprob difference about 1.1 nats).

The scorer is deterministic: repeated runs and different thread counts are bit-identical.
The disagreement is between backends. CPU at two revisions, CPU vs CUDA at one revision,
and each of those vs the vendor all differ by 0.6 to 1.1 nats on average.

Root cause: after the `Answer:` prompt the model puts a median of 0.08% of its probability
on all answer letters combined (log −7.1), for both BF16 and the 3-bit quant. Ranking
choices that deep in the tail is numerically fragile, so the protocol carries backend
noise at the level of individual questions. It still carries signal (47% against a 10%
chance floor).

### Backend control

The vendor's GSQ-RCO-3 scored on all 2000 questions with our CUDA backend
(f32 KV, flash attention off, 2048 context):

| Run | Accuracy | Ratio vs vendor BF16 |
|---|---|---|
| Vendor BF16 (vendor run) | 47.60 | — |
| Vendor GSQ-RCO-3 (vendor run) | 43.70 | 91.8% |
| Vendor GSQ-RCO-3 (our run) | 42.60 | 89.5% |

Paired on the same questions, ours minus vendor is −1.10 pp ± 0.74 (1.5σ, 76.3%
prediction agreement, 122 lost and 100 won). That is not a significant offset, so the
vendor's BF16 47.6 stays the anchor. If our backend is slightly low, the 45.22 gate is
about 1 pp stricter than intended for candidates scored here. We do not correct for it.

BF16 MMLU-Pro cannot be rerun on this machine: the model does not fit in memory and
streaming it from disk would take about 40 hours.

## Candidates

The size budget is under 4 bits per weight, about 3.5 is fine, and up to 10% over the
vendor build being compared against.

All candidates are built from the vendor BF16 GGUF (sha256 044d7f8b…) with the bartowski
imatrix, Q4_K output and token embeddings (as the vendor uses), and Q8_0 for attention and
shared experts. Only the routed-expert types vary. BPW uses the vendor measure (file bytes
× 8 / 34.66 B parameters).

| Tag | Routed experts | BPW | Compare with |
|---|---|---|---|
| i3xxs | all IQ3_XXS | 3.43 | vendor GSQ-RCO-3 |
| d3 | gate/up Q2_K, down Q3_K | 3.27 | vendor GSQ-RCO-3 |
| i3mix | gate/up IQ3_XXS, down IQ3_S | 3.54 | vendor GSQ-RCO-3.5 |
| vl | vendor 3-bit layout, plain imatrix rounding | 3.20 | control for GSQ-RCO |
| d4 | gate/up Q2_K, down Q4_K | 3.60 | vendor GSQ-RCO-3.5 |
| i3s | all IQ3_S | 3.78 | |
| e3 | all Q3_K | 3.78 | |

Our files keep the MTP layer (blk.40, `nextn`), which the vendor stripped. That adds
about 0.5 GB at Q4_K, so the BPW above is slightly high against the vendor files. MTP
stays in the final quant for self-speculative decoding. It does not affect any score
here, because neither the KL-divergence runs nor the MMLU-Pro scorer use it.

The i-quants need imatrix data for every tensor they quantize. The bartowski imatrix
has none for the MTP experts, so the i-quant candidates (i3xxs, i3mix, i3s) set those to
Q4_K. The K-quant candidates give them the same type as the main layers.

## Results

MMLU-Pro on our CUDA backend (same settings as the backend control above). KL divergence,
top-1 agreement and perplexity ratio are against BF16 logits.

| Tag | BPW | MMLU-Pro | Wiki PPL ratio | Wiki KLD | Wiki top-1 | Sci PPL ratio | Sci KLD | Sci top-1 |
|---|---|---|---|---|---|---|---|---|
| d3 | 3.27 | 42.50 (850/2000) | 1.0736 | 0.162 | 83.09% | 1.0411 | 0.128 | 86.84% |
| i3mix | 3.54 | 38.30 (766/2000) | 1.0520 | 0.112 | 85.73% | 1.0512 | 0.096 | 88.77% |
| vl | 3.20 | failed (CUDA out of memory) | 1.0956 | 0.202 | 81.36% | 1.0430 | 0.155 | 85.45% |
| d4 | 3.60 | **45.75** (915/2000) | 1.0765 | 0.155 | 83.71% | 1.0412 | 0.120 | 87.35% |

- d4 (gate/up Q2_K, down Q4_K) clears the 45.22 MMLU-Pro gate on our backend. It also beats
  the vendor GSQ-RCO-3 (43.70 vendor run, 42.60 ours) at 3.60 BPW, a figure that includes
  the MTP layer.
- No candidate reaches the 95% top-1 gate. The best is i3mix at 88.8% on science text.
- MMLU-Pro does not track KL divergence here. i3mix has the lowest KLD of the four but the
  worst MMLU-Pro score. This fits the deep-tail fragility described above.
- i3xxs, i3s and e3 were not scored. The ladder was stopped when the work moved to
  Saluki 27B (2026-10-23). The candidate files are kept.

## Running real GSQ-RCO here is not feasible

From the IST-DASLab GSQ and RCO READMEs (both Apache-2.0, checked 2026-10-23):

- GSQ quantizes one layer at a time with offload, but the README lists 4× H200 for
  Qwen3-30B-A3B. Its default calibration is 4096 sequences of 4096 tokens. It has been
  tested on Hopper only, with Ada reported to work. Its GGUF route refines existing
  K-quant files and writes the same K-quant types back.
- RCO searches per-tensor bit widths over a database of GPTQ fake-quantized layers, one per
  bit width from 2 to 8. Its example builds that database on 8 GPUs, and its output is
  Hugging Face safetensors, not GGUF.

An 8 GB card cannot run either at the published settings. Shrinking the calibration enough
to fit would be an untested configuration. If plain imatrix quants fall short of the gate,
the realistic option is renting a multi-GPU node, not a local run. ISTA's own GGUF releases
use i-quant types (IQ2_S, IQ3_S), so the i-quant candidates above are in the same format
family.
