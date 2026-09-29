# /goal PHASE 3 — FIRST REAL MODEL IMPORT

**Status:** LANDED (Buffy, 2026-09-29). Next: `docs/goals/PHASE4_CUDA.md`.
Real SmolLM2 snapshot remains skip-or-green (`GESSO_SMOLLM2_DIR`).

**For:** Buffy (mechanical implementation)
**From:** Grok (encoding owner)
**Canon:** `docs/Gesso_Stack.md` §LXXVI, §VIII, §XI, §CIX
**Map:** `docs/ARCHITECTURE.md`
**Depends on:** Phase 2 complete (`docs/goals/PHASE2_CPU_ORACLE.md`)
**Packets:** 1 and 2 stay closed.

---

## Start condition

Phase 2 is on the tree you inherit:

- `make test` / `make format-check` green
- `toy2` prefill + greedy generate on `CPUBackend`
- `expected_logits.toml` filled
- `quantize!` / `dequantize!` still decline

If Phase 2 is unfinished, stop.

## One-sentence objective

Gesso loads a Llama-family checkpoint (config + safetensors + tokenizer)
into the existing semantic core, and Gesso logits match a declared
reference within declared tolerance.

## The chosen model

**HuggingFaceTB/SmolLM2-135M** (base, Apache-2.0, ungated).

Released `config.json` (pin these numbers; do not invent others):

```
architectures            ["LlamaForCausalLM"]
model_type               llama
hidden_size              576
num_hidden_layers        30
num_attention_heads      9
num_key_value_heads      3
intermediate_size        1536
hidden_act               silu
rms_norm_eps             1e-5
rope_theta               100000
rope_scaling             null
rope_interleaved         false
attention_bias           false
tie_word_embeddings      true
vocab_size               49152
bos_token_id             0
eos_token_id             0
max_position_embeddings  8192
torch_dtype              bfloat16
```

`head_dim = hidden_size / num_attention_heads = 64`.
GQA repeat factor = `9 / 3 = 3`.

Tokenizer: `GPT2Tokenizer` (byte-level BPE), files `vocab.json` +
`merges.txt`. This is why this model is the first import: the
architecture is already our primitive list, and the tokenizer is
implementable without SentencePiece.

Do not switch to TinyLlama, Llama 3.2, Qwen, or Instruct variants.
Llama 3 rope-scaling and SentencePiece are later work items.

## What this sprint is not

- CUDA, Lava, quantization math
- downloading weights inside `make test` / CI
- HuggingFace Hub as a package dependency
- PyCall, Transformers.jl, Safetensors.jl, Tokenizers.jl
- MoE, attention bias, MLP bias, interleaved RoPE, rope_scaling ≠ null
- a new operator in the vocabulary
- changing §CIX family types
- training

CI must stay green on a machine that has never seen SmolLM2 weights.

---

## Laws (pin this)

### Interpreter extensions (item A) — toy2 must stay bit-identical

The Phase 2 recipe grows three knobs. Defaults preserve `toy2`.

1. **GQA.** `Attention.n_kv_heads` is already on ModelIR. Prefill and
   generate must:
   - project K/V at `(n_kv_heads * d_head, dim)`
   - reshape to `(seq, n_kv_heads, d_head)`
   - repeat each KV head `n_heads / n_kv_heads` times (integer; error
     if it does not divide) **for the score/value contraction only**
   - store the KV cache at `n_kv_heads` (not `n_heads`)
   - RoPE on the un-repeated K (and Q at full `n_heads`)
   toy2 is MHA (`n_kv_heads == n_heads`): repeat factor 1, cache shape
   unchanged, existing tests stay `atol=0` in-process.

2. **Final RMSNorm.** Llama has `model.norm` before the lm head. toy2
   has none. If `tensors` has a `final_rms::FrozenParameter`, apply
   `rmsnorm!` to `h` after the last block and before the tied head.
   If the field is absent, skip (toy2). Do not add a field to `Model`
   this sprint unless a packet says so.

3. **`eps` and `theta` are keyword defaults, not new operators.**

```
rmsnorm!(backend, dst, x, scale, workload; eps=1e-6)
rope!(backend, q, k, positions, workload; theta=10000.0)
```

toy2 tests call without keywords. SmolLM2 passes `eps=1e-5`,
`theta=100000.0`. Thread them from the importer through the
interpreter; do not hardcode SmolLM2 constants inside `cpu.jl`.

### Weight name map (Llama / SmolLM2)

HF Linear is `(out, in)` — already our `ProjectionWeight` convention.
Upcast every tensor to `Array{Float64}` at load (BF16 / F16 / F32 / F64
are legal source dtypes; anything else errors loudly).

```
model.embed_tokens.weight                         EmbeddingTable (vocab, hidden)
model.layers.{i}.self_attn.q_proj.weight          ProjectionWeight (hidden, hidden)
model.layers.{i}.self_attn.k_proj.weight          ProjectionWeight (n_kv*d_head, hidden)
model.layers.{i}.self_attn.v_proj.weight          ProjectionWeight (n_kv*d_head, hidden)
model.layers.{i}.self_attn.o_proj.weight          ProjectionWeight (hidden, hidden)
model.layers.{i}.mlp.gate_proj.weight             ProjectionWeight (intermediate, hidden)
model.layers.{i}.mlp.up_proj.weight               ProjectionWeight (intermediate, hidden)
model.layers.{i}.mlp.down_proj.weight             ProjectionWeight (hidden, intermediate)
model.layers.{i}.input_layernorm.weight           FrozenParameter (hidden,)
model.layers.{i}.post_attention_layernorm.weight  FrozenParameter (hidden,)
model.norm.weight                                 FrozenParameter (hidden,)   → final_rms
lm_head.weight                                    absent when tied; reuse embed
```

Unknown keys in the checkpoint: error with the key name (no silent skip).
Missing required keys: error with the key name.
`tie_word_embeddings: true` and a present `lm_head.weight` that is not
byte-identical to embed: error.

### Safetensors (no Safetensors.jl)

File layout (the published spec):

```
uint64 header_len
header_len bytes of JSON  (UTF-8)
raw tensor bytes
```

Header JSON: each tensor entry has `"dtype"`, `"shape"`, `"data_offsets":
[begin, end]`. Dtypes to accept: `BF16`, `F16`, `F32`, `F64`.
Offsets are relative to the start of the raw region (after the JSON).
Use `Mmap` (stdlib) for the byte region.

Multi-shard: if `model.safetensors.index.json` is present, follow
`weight_map` and load each listed file. Single `model.safetensors` is
the ordinary case.

### Config → ModelIR

`config.json` → `Gesso.Model`:

```
Embedding(dim=hidden_size)
for _ in 1:num_hidden_layers
    Block(Attention(n_heads=…, n_kv_heads=…), SwiGLU(hidden=intermediate_size))
end
vocab_size from config
```

Refuse configs that are not Llama-shaped for this sprint:

- `model_type` other than `"llama"`
- `hidden_act` other than `"silu"`
- `attention_bias` / `mlp_bias` true
- `rope_scaling` not `null`
- `rope_interleaved` true
- `num_attention_heads` does not divide `hidden_size`
- `num_key_value_heads` does not divide `num_attention_heads`

### Tokenizer (GPT-2 byte-level BPE)

Implement in Gesso (Inference or a file it includes). Algorithm:

1. `bytes_to_unicode` as in GPT-2 (the published 256-byte map).
2. Apply `merges.txt` in file order (pair ranks).
3. `vocab.json` maps token string → id.
4. Encode: UTF-8 bytes → unicode-mapped chars → greedy BPE by rank → ids.
5. Special tokens from tokenizer config: `bos_token_id`, `eos_token_id`.
   Do not apply a chat template this sprint (base model).

Do not add Tokenizers.jl. A tiny `vocab.json` + `merges.txt` fixture
covers CI. Real SmolLM2 tokenizer files are optional on disk (item D).

### JSON.jl

The one new core dependency this sprint. Add to `Project.toml` **and**
the dependency-law allowlist:

```
JSON = "config.json + safetensors header parsing (§LXXVI)"
```

UUID: `682c06a0-de6a-54ab-a142-c8b1cf79cde6`. No other new deps.
`Mmap` is stdlib; if you `using Mmap`, add it to the allowlist too
(`"safetensors byte region (§LXXVI)"`).

### CI vs real weights

`make test` never downloads. Never talks to huggingface.co.

| Surface | Always runs | Needs `GESSO_SMOLLM2_DIR` |
|---|---|---|
| GQA / eps / theta / final_rms on toy2 + micro-llama | yes | no |
| safetensors reader + name map | yes (micro fixture) | no |
| GPT-2 BPE | yes (tiny vocab fixture) | no |
| full SmolLM2 load + logit match | skip with `@test skip=…` | yes |

`GESSO_SMOLLM2_DIR` points at a local snapshot containing `config.json`,
`*.safetensors` (and index if sharded), `vocab.json`, `merges.txt`.
Document the env var in README.

---

## Work items (sequence; land A before B before C; D is the real-model gate)

### A — Interpreter: GQA, final RMSNorm, eps/theta

**Objective.** The CPU interpreter runs GQA and a final RMSNorm. toy2
numbers do not move.

**Permitted files**

```
src/Operators/cpu.jl
src/Operators/Operators.jl
src/Inference/Inference.jl
src/Harpe.jl                    # if the file is src/Gesso.jl, that one
src/Gesso.jl
test/test_cpu_ops.jl
test/test_reference_prefill.jl
test/test_reference_generate.jl
test/test_gqa.jl                # new
test/runtests.jl
docs/ARCHITECTURE.md
```

**Tests**

- toy2 prefill vs persisted logits: still `atol=1e-10`
- toy2 in-process determinism: still `atol=0`
- GQA micro: `n_heads=4`, `n_kv_heads=2`, `dim=16`, one block, seq=3.
  Independent formula in the test (repeat KV then scores). `atol=1e-12`
- `n_kv_heads` that does not divide `n_heads` errors
- `rmsnorm!(…; eps=1e-5)` differs from default on a fixture where it
  must; default still matches Phase 2
- `rope!(…; theta=100000.0)` differs from default; default matches Phase 2
- final_rms present vs absent: two-call fixture

**Performance.** N/A.

**Artifact.** Interpreter can express SmolLM2. toy2 still green.

---

### B — Config + safetensors + Llama name map

**Objective.** A Llama `config.json` + safetensors file become a
`Gesso.Model` plus materialized tensors that `reference_prefill` runs.

**Permitted files**

```
src/Inference/Inference.jl
src/Inference/*.jl              # split importer if the file grows
src/Gesso.jl
Project.toml                    # JSON, maybe Mmap
test/runtests.jl
test/test_import_llama.jl
test/fixtures/llama_micro/      # generated or checked-in tiny shard
test/toyfixtures.jl             # only if you reuse rng helpers
docs/ARCHITECTURE.md
README.md
scripts/freeze.jl
```

**Interfaces**

```
load_llama_config(path) -> NamedTuple   # validated numbers
config_to_model(cfg) -> Model
load_safetensors(path) -> Dict{String,Array}
materialize_llama(model, tensors_by_name, cfg) -> tensors named tuple
load_llama(dir) -> (model, tensors, cfg)
```

Live in `Inference`. Re-export from `Gesso` so `using Gesso` sees
`load_llama`.

**Micro fixture** (`test/fixtures/llama_micro/`):

- 2 layers, `hidden=32`, `heads=4`, `kv_heads=2`, `intermediate=64`,
  `vocab=32`, `eps=1e-5`, `theta=10000`, tied embeddings
- one safetensors file + `config.json` produced by a test helper from
  `deterministic_rng` (do not check in a 10MB binary if you can write
  the writer; a writer + round-trip test is better)
- `load_llama` → `reference_prefill` on tokens `[1, 2, 3]` returns
  shape `(32, 3)` and is deterministic in-process
- unknown key / missing `q_proj` / bad `model_type` each error with
  the name in the message

**Artifact.** Importer exists. No Hub, no 135M file in git.

---

### C — GPT-2 byte-level BPE

**Objective.** Encode a string to ids with the GPT-2 BPE algorithm.
A tiny fixture covers CI; the algorithm is the real SmolLM2 tokenizer.

**Permitted files**

```
src/Inference/Inference.jl
src/Inference/*.jl
src/Gesso.jl
test/test_tokenizer_gpt2.jl
test/fixtures/gpt2_tiny/        # small vocab.json + merges.txt
test/runtests.jl
docs/ARCHITECTURE.md
```

**Interfaces**

```
load_gpt2_tokenizer(dir) -> tokenizer
encode(tokenizer, text::AbstractString) -> Vector{Int}   # 0-based ids
```

**Tiny fixture.** Enough merges to encode `"ab"` and `"Hello"`
deterministically; golden ids live in the test. Round-trip is not
required this sprint (decode is nice-to-have, not the gate).

**Refuse:** empty vocab, merge pair not in vocab, unknown special-token
id in config.

**Artifact.** Tokenizer path exists. Chat templates do not.

---

### D — Real SmolLM2 logit match (the §LXXVI exit)

**Objective.** When `GESSO_SMOLLM2_DIR` is set to a local snapshot of
`HuggingFaceTB/SmolLM2-135M` (revision recorded in the receipt), Gesso
prefill logits for a pinned prompt match a checked-in reference within
declared tolerance.

**Permitted files**

```
test/test_smollm2.jl
test/fixtures/smollm2/expected_logits.toml   # filled when you have the snapshot
test/runtests.jl
README.md
docs/ARCHITECTURE.md
docs/Gesso_Stack.md                          # §LXXVI status line only
```

**Pinned prompt.** UTF-8 string `"Hello"` (no BOS prefix beyond what
the tokenizer itself emits). Encode with the snapshot's `vocab.json` +
`merges.txt`. Prefill. Compare the **last-position** logits (length
`vocab_size`) to the checked-in vector.

**Tolerance.** `atol=1e-2`, `rtol=0` on Float64 Gesso vs the reference
vector. The reference vector is produced **once** from Gesso itself on
the snapshot (provenance: `oracle = "gesso-cpu"`, revision, commit) —
then frozen. This is a regression oracle, not a second stack.

If you also have a transformers/numpy dump of the same prompt, you may
add a **second** comparison with a wider atol (`1e-1`) and a packet
note that BF16→F64 vs HF BF16 is not bit-identical. Do not fail the
sprint on HF disagreement; fail on Gesso-vs-frozen-Gesso drift.

**Skip.** If `GESSO_SMOLLM2_DIR` is unset or the directory is
incomplete, `@test skip=true` with a message that names the env var.
CI stays green.

**If you cannot obtain the snapshot.** Land A+B+C. Write a short note
in the receipt: item D is skip-only until a snapshot exists. Do not
fake the golden file.

**Artifact.** §LXXVI exit is met on a machine that has the snapshot;
CI proves the importer without it.

---

## Cross-item invariants

- toy2 tests remain green at the same tolerances
- `quantize!` / `dequantize!` still decline
- Operators still export nothing
- §CIX fence: ModelIR/Parameters/Semantics export lists unchanged
  unless item A truly requires a Model field — if so, stop and packet
- `libs/` untouched
- formatter: `make format` (it must not walk `libs/`)
- Receipt (§LXXII): what changed · why · tests · numerical delta
  (toy2 fingerprint must not move; micro-llama max|logit| recorded) ·
  compile-time · hardware CPU · workload prefill `"Hello"` / `[1,2,3]` ·
  model `toy2` + `llama_micro` + optional `SmolLM2-135M` · backend `cpu`

## Escalation (stop and write a packet)

- rope_scaling, interleaved RoPE, attention bias, or a non-silu act
  appear required for SmolLM2-135M as released (they should not)
- GQA cannot be done without mutating ModelIR
- JSON.jl is not enough and you want HuggingFace / Safetensors.jl
- Float64 CPU OOM on 135M at seq=4 (then shrink the D prompt, don't
  quantize)

## Exit checklist

- [ ] A, B, C landed
- [ ] D skip-or-green
- [ ] `make test` green without `GESSO_SMOLLM2_DIR`
- [ ] `make format` run
- [ ] toy2 logits fingerprint unchanged
- [ ] JSON.jl in Project.toml **and** the allowlist
- [ ] README documents `GESSO_SMOLLM2_DIR`
- [ ] ARCHITECTURE / §LXXVI status truthful
- [ ] receipt in the close note
- [ ] no CUDA, no Lava, no Hub client
