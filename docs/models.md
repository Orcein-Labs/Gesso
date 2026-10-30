# Models

## GGUF architectures

`Gesso.load_gguf` reads `general.architecture` and accepts these values. A
missing key is treated as `qwen35moe`. Anything else fails with an error that
lists the supported names.

| `general.architecture` | Family |
| --- | --- |
| `qwen35` | Qwen3.5 dense, a gated-DeltaNet and attention hybrid |
| `qwen35moe` | Qwen3.5 mixture of experts, same hybrid layout |
| `llama` | Llama-style dense transformers |
| `spark2_5` | Spark |
| `qwen4exp` | Experimental Qwen hybrid: DeltaNet and attention layers, hyper-connections, per-layer embeddings, mixture of experts with a shared expert |
| `lfm2` | LFM2 and LFM2.5 dense |
| `lfm2moe` | LFM2 mixture of experts |
| `gemma4` | Gemma 4, mixture-of-experts variants only |

Multi-token-prediction blocks in Qwen3.5 files are recorded but not run.

Gemma 4 limits: files with per-layer input embeddings
(`embedding_length_per_layer_input` non-zero), with shared KV layers
(`attention.shared_kv_layers` non-zero), or with no experts are refused with
a message naming the unsupported field.

`load_gguf(path; T=Float32, verbose=false)` returns `(model, tensors, spec)`.
Routed expert weights stay memory-mapped from the file. Large 2-D weights
(projections, embedding, output head) stay in their quantized bytes. Small and
1-D tensors are converted to `Float32`. Loading does not need llama.cpp.

## GGUF tensor types

A tensor whose type is not in this table fails with
`gguf_import: ggml type N (tensor X) has no reader in Gesso`.

| Type | Block (elements / bytes) | Matmul path |
| --- | --- | --- |
| F32 | 1 / 4 | native |
| F16, BF16 | 1 / 2 | row dots widened to F32 |
| Q8_0 | 32 / 34 | integer kernel |
| Q4_0 | 32 / 18 | integer kernel |
| Q4_1 | 32 / 20 | dequantize per call, then F32 |
| Q5_0 | 32 / 22 | dequantize per call, then F32 |
| Q5_1 | 32 / 24 | native fused kernel |
| Q2_K | 256 / 84 | integer kernel |
| Q3_K | 256 / 110 | integer kernel |
| Q4_K | 256 / 144 | integer kernel |
| Q5_K | 256 / 176 | integer kernel |
| Q6_K | 256 / 210 | integer kernel |
| IQ1_S | 256 / 50 | integer kernel |
| IQ1_M | 256 / 56 | integer kernel |
| IQ2_XXS | 256 / 66 | integer kernel |
| IQ2_XS | 256 / 74 | native fused kernel |
| IQ2_S | 256 / 82 | integer kernel |
| IQ3_XXS | 256 / 98 | native fused kernel |
| IQ3_S | 256 / 110 | integer kernel |
| IQ4_NL | 32 / 18 | native fused kernel |
| IQ4_XS | 256 / 136 | native fused kernel |
| Q2_0 | 64 / 18 | native fused kernel |

A row length that is not a whole number of blocks, a misaligned tensor offset,
or a tensor that extends past the end of the file is reported as a malformed
GGUF. The table above describes the CPU matmul path; the GPU backends have
their own quantized kernels in `ext/`.

## Tokenizers and chat templates

The tokenizer is read from the GGUF (`Gesso.Inference.load_gguf_tokenizer`).
Supported pre-tokenizers are `qwen35`, `spark2_5` and `lfm2`; Gemma 4 needs
`tokenizer.ggml.pre` set to `gemma4`. The chat server additionally requires a
chat template in the file, and only serves the template families listed in
[serving.md](serving.md).

## Safetensors

`Gesso.load_llama(dir)` loads a Hugging Face directory (`config.json`,
`model.safetensors` or its shard index, `vocab.json`, `merges.txt`). The config
reader accepts `model_type == "llama"`. RoPE scaling `linear`, `llama3` and
none are implemented; other types fail with a `LoweringNotImplemented` error.
The architecture-adapter table also names `qwen2`, `qwen3_5`, `qwen3_5_moe`,
`gemma`, `mistral`, `phi3` and `phi`; those entries describe tensor-name
mappings and are not a promise that every such checkpoint runs end to end.
