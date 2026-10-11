# Architecture gauntlet — first non-toy model

Owner: Grok. Layer: Gesso mechanism. Palette / Cyan / Luxel stay frozen.
Regime II Lava/CUDA parity on SmolLM2-135M is green (median P 1.22,
2026-10-08). This file is the next execution plan. It is not a speed
claim and it is not a claim that any model below already runs.

## Pick

**Campaign target: Ornith-1.5-9B** (`ornith-ai/Ornith-1.5-9B`).

Locked sequence (2026-10-08):

| # | Campaign | Why this slot |
| ---: | --- | --- |
| 1 | Ornith-1.5-9B | Dense Qwen3.5 hybrid. Architecture specimen: Qwen3.5-0.8B (F32-fits the 5060). |
| 2 | Ornith-1.5-35B-A3B | Same hybrid mixer + MoE (`qwen3_5_moe`: 40 layers, 256 experts / 8 active, hidden 2048, `head_dim` 256). |
| 3 | Xing4.0-29B-A4B | MLA + MoE + mHC + MTP + YaRN. |
| 4 | Gemma4-26B-A4B **at Q4** | Sliding+full hybrid, 128 experts / 8 active, local `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (15.78 GB). This campaign unparks representation / GGUF. |

## What is on disk today (2026-10-08)

No Hugging Face fetches. Weights already live under `/mnt/d/Code Projects`
(GGUF inventory) and `Gesso/snapshots/` (safetensors specimens). RTX 5060
8151 MiB, 31 GiB host RAM, `/mnt/d` ~768 G free.

Gesso architecture specimen: `snapshots/Qwen3.5-0.8B/` (safetensors BF16
shard `model.safetensors-00001-of-00001.safetensors`, ~1.75 GB). SmolLM2-135M
stays the Regime II oracle. Named non-toy files are llama.cpp GGUF quants
at `/mnt/d/Code Projects/` (third-column until campaign 4 unparks GGUF):

| File | Size | Role |
| --- | ---: | --- |
| `Ornith-1.5-9B-Q4_K_M.gguf` | 5.24 G | dense 9B, Qwen3.5 hybrid |
| `Unsloth-Ornith-1.5-35B-A3B-UD-Q2_K_XL.gguf` | 11.45 G | sparse MoE sibling |
| `Xing4.0-29B-A4B-Q4_K_M.gguf` | 17.65 G | MLA + MoE + mHC + MTP |
| `xing4_0-29b-IQ4_NL.gguf` | 0 B | empty, ignore |
| `Qwen3.5-0.8B-BF16.gguf` | 1.41 G | same hybrid family, F32-fittable size |
| `Qwen3.5-2B-Q4_K_M.gguf` / `Qwen3.5-4B-Q4_K_M.gguf` | 1.19 / 2.55 G | same family |
| `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` | 15.78 G | MoE |
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-IQ2_M.gguf` | 18.10 G | MoE |
| `Qwen3.8-27B-UD-IQ2_M.gguf` + Flash-Next 3-part | 9.61 G / huge | memory wall, last |

Gesso loads safetensors and upcasts to F64 on the host, F32 on device.
qwen35moe GGUF is unparked for Ornith-1.5-35B-A3B (mmap + ggml `to_float`,
lazy experts). Other GGUF files stay inventory until their campaign.

## Why Ornith, why not Xing first

Ornith-1.5-9B published config (`model_type: qwen3_5`, nested
`text_config`):

- 32 layers, hidden 4096, 16 Q heads / 4 KV heads, **explicit
  `head_dim` 256** (here `16 × 256 = 4096`, so it matches Gesso's
  derived head dim)
- FFN SwiGLU 12288, vocab 248320, RMSNorm, `attn_output_gate`
- Hybrid mixer: 24 `linear_attention` (Gated DeltaNet) + 8
  `full_attention`, interval 4
- Linear mixer: conv kernel 4, 16 key heads × 128, 32 value heads × 128
- Partial RoPE 0.25 (`rotary_dim` 64) + mRoPE sections `[11, 11, 10]`,
  `mrope_interleaved`
- MTP declared; vision tower present (text path only this campaign)
- BF16 checkpoint ~19 GB; F32 ~36 GB

Xing4.0-29B-A4B (`model_type: xing4_0`, custom `trust_remote_code`):

- 40 layers, hidden 3584, MLA (`q_lora_rank` 768, `kv_lora_rank` 512,
  `qk_nope_head_dim` 128, `qk_rope_head_dim` 64, `v_head_dim` 128)
- MoE: 64 routed / 4 active / 1 shared, `first_k_dense_replace` 2
- mHC (`hc_mult` 4, Sinkhorn)
- MTP, YaRN (`rope_scaling.type: yarn`, factor 64)
- BF16 ~58 GB; local Q4_K_M 17.65 G does not fit 8 GB without offload

Gesso named gaps before this campaign: `dense_ffn`, `qk_norm`,
`moe_routing`, `sliding_window_attention`. Ornith needs those plus
**new** named ops: `:linear_attention` (Gated DeltaNet),
`:attn_output_gate`, `:mrope`, and the already-refusing
`:rope_partial_dim` / `:rope_interleaved`. Xing needs those plus MLA,
MoE routing, mHC, and YaRN. One new family at a time.

## Hardware wall (accuracy vs throughput)

The 5060 has 8 GB. Ornith-1.5-9B cannot sit on-device in F32 or BF16.
Host RAM (31 GB, ~22 GB available) can hold the BF16 checkpoint for a
CPU quality oracle; F32 9B needs swap.

So the campaign splits:

1. **Architecture specimen that F32-fits the 5060:** Qwen3.5-0.8B
   (~1.75 GB safetensors; F32 weights ~3.2 GB). Same `qwen3_5` hybrid
   stack as Ornith. Watch the extra wrinkle: published `head_dim` 256
   with `hidden_size` 1024 and 8 heads (`8 × 256 = 2048 ≠ 1024`).
   Gesso today derives `head_dim = hidden / n_heads`.
2. **Quality on the real 9B:** greedy IDs vs independent Hugging Face
   on CPU (BF16 checkpoint, Gesso F32/F64). Same prompts, compile
   excluded, planted NaN still `ERR_NUMERICAL_INSTABILITY`.
3. **Throughput:** Gesso Lava vs Gesso CUDA on the specimen first
   (fits), then 9B only with a declared residency plan (layer offload
   or a later representation). llama.cpp Q4 is a third column, never
   the Gesso ID oracle.

Beating CUDA on 9B F32 on this GPU is not a legal plan. Beating CUDA
on the 0.8B hybrid specimen is. Quality of Ornith-1.5-9B is a CPU
(and later offload) measurement until representation lands.

## Execution order (MAKE IT WORK → COMPLETE → MEASURABLE → FAST)

1. Parse `qwen3_5` (nested `text_config`) through the adapter doorway.
   Fail at named missing ops, not at a family whitelist.
2. Independent `head_dim` on `ArchitectureSpec` (published Qwen3.5
   convention).
3. CPU Gated DeltaNet + gated full-attention + partial / mRoPE, with
   a synthetic then 0.8B ID oracle vs HF.
4. CUDA then Lava lowerings of those ops. Same three-prompt greedy
   harness, compile excluded.
5. Ornith-1.5-9B safetensors (BF16, ~19 GB) for CPU quality. No 19 GB
   fetch until step 3 is green.
6. Ornith-1.5-35B-A3B: MoE routing on the same hybrid mixer
   (`:moe_routing`, 256 experts / 8 active, shared expert).
7. Xing4.0-29B-A4B: MLA, mHC, YaRN, MTP on top of MoE.
8. Gemma4-26B-A4B at Q4: sliding-window attention, Gemma4 MoE, and a
   declared Q4 representation so the local 15.78 GB GGUF is a legal
   Gesso checkpoint. GGUF/Q4 stay parked until this campaign.

Vision, MTP decode, native FP16/BF16, and `libs/` stay parked unless a
measured receipt names them as the limiter. qwen35moe GGUF is unparked
for the 35B campaign. Other GGUF/Q4 stay parked until campaign 4,
except as llama.cpp third-column measurements.

## Oracle (unchanged)

Greedy token IDs exact against an independent Hugging Face reference
on the same checkpoint. Planted NaN still `ERR_NUMERICAL_INSTABILITY`.
No silent CPU fallback. No `libs/` edits. Receipt on every retain.

No local torch/transformers; the 0.8B ID-gate uses llama.cpp
(`llama-cpp-upstream`, `Qwen3.5-0.8B-BF16.gguf`) as the independent
third column on the same tokenizer id.

## Current receipt (2026-10-08)

Qwen3.5-0.8B CPU `reference_prefill` on prompt `Hello` (token `[9419]`):

| | greedy | top-5 ids |
| --- | ---: | --- |
| Gesso CPU F64 | 11 `,` | 11, 13, 0, 198, 4858 |
| llama.cpp BF16 GGUF | 11 `,` | 11, 13, 0, 198, 4858 |

Logit gaps vs greedy agree to ~0.001. Two binder fixes landed this
receipt: per-head interleaved Q\|gate (`fused_qg_rows`) and Gemma-style
RMSNorm `(1 + weight)` baked at materialize (`:rms_unit_offset`; GDN
gated-norm is already 1-centered and is not offset).

CUDA F32 hybrid extras (2026-10-08): `to_device` keeps `spec`/`rope`;
`rope!` accepts `inv_freq` + `rotary_dim`; GDN / causal conv1d+SiLU /
sigmoid-gate / silu-gate run as `@cuda` kernels. Qwen3.5-0.8B Hello
`reference_generate` on the 5060 matches the CPU ID-gate
`[11, 271, 40, 1044, 3133, 440, 264, 12654]`. CUDA now claims
`:linear_attention`, `:qk_norm`, `:attn_output_gate`, `:mrope`,
`:rope_partial_dim`, `:rope_interleaved`, `:rope_linear`, `:rope_llama3`.

CUDA MoE expert offload (2026-10-08): `to_device` uploads non-expert
weights small-first until `GESSO_GPU_BUDGET_MIB` (default 4800);
overflow is `HostOffload`; `GgufQuantSlice` experts stay mmap'd.
Active experts dequant into scratch, H2D, CUBLAS gemv. Same path for
any `(n_experts, n_used, expert_ff, hidden)` — Ornith 256/8, Xing 64/4,
Gemma4 128/8. CUDA claims `:moe_routing`.

35B Gesso CUDA Hello ID-gate on the 5060 (2026-10-08), same Unsloth
UD-Q2_K_XL GGUF, NVMe copy, `julia -t 8`, budget 6144 MiB:

Residency after `to_device`: 621 tensors on GPU (5459.7 MiB F32 layers),
embedding+lm_head `HostOffload` (3880 MiB), 120 expert slabs mmap
(10040 MiB Q2). 1528 MiB GPU free. Greedy ids exact vs llama.cpp CUDA
and Gesso CPU: `[9419, 11, 271, 40, 2688, 4313, 310, 958, 279]`.

Warmed decode (compile excluded, tokens 2–4 of the second generate):
FFN 10.4–11.0 s, GDN ~70 ms, attn ~20 ms, lm_head ~0.50 s → **0.09 tok/s**.
llama.cpp CUDA tg64 ngl99 ncmoe24 fa is **32.80 tok/s** on this card.
The gap is expert FFN: Gesso dequants 8×3 matrices per layer to F32 and
H2Ds them; llama.cpp runs quantized GEMM on-device. No local PyTorch;
llama.cpp CUDA is the 8 GB column. On-device Q2/Q4 expert GEMM is FAST.

Still open: Session mixer-aware path; Lava GDN / qk_norm / output-gate /
partial-RoPE; on-device quantized expert GEMM; Xing then Gemma4-Q4.

## Ornith-1.5-35B-A3B receipt (2026-10-08)

Local file `Unsloth-Ornith-1.5-35B-A3B-UD-Q2_K_XL.gguf` (11.44 GiB,
qwen35moe, 256 experts / 8 active). Gesso `load_gguf` mmaps experts and
dequants non-experts to host F32. GGUF V heads are llama.cpp tiled;
`LinearMixerSpec.v_head_tiled=true` on this path. Hello token `9419`.

Greedy 8-token continuation, Gesso CPU vs llama.cpp CUDA (`-ngl 99
-ncmoe 24 -fa on`):

`[9419, 11, 271, 40, 2688, 4313, 310, 958, 279]` — exact ID match.

Throughput (compile excluded after first decode; NVMe copy of the GGUF
under `~/.cache/gesso`, `julia -t 8`):

| | tok/s |
| --- | ---: |
| Gesso CPU F32, warmed decode (median tokens 2–6) | 0.22 |
| llama.cpp CUDA tg64 ngl99 ncmoe24 fa | 32.80 ± 3.20 |

The ~37 tok/s figure from Regime II is SmolLM2-135M Lava Hello, not 35B.
drvfs `/mnt/d` mmap made expert dequant ~5× slower than the same file on
NVMe. Expert F32 cache at cap 12 swapped the 31 GiB host and lost.
