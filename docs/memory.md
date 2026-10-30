# Memory and context length

Context length costs memory in the KV cache. This page says how much, so you
can pick a context length that fits.

## KV formats

Set `GESSO_KV_FORMAT` to one of `f32`, `f16`, `q8`, `q4h` (case-insensitive).
When unset, `q8` is used on the CUDA and Lava backends and `f32` on the CPU
backend. If the head size is not a multiple of 32, the default is `f32`.
Any other value fails with `ERR_INVALID_PLAN`.

Bytes for one head's key (or value) row of `d` elements:

| Format | Bytes | Notes |
| --- | --- | --- |
| `f32` | `4d` | Exact. |
| `f16` | `2d` | Half precision. |
| `q8` | `d + 2*(d/32)` | Int8 codes plus one Float16 scale per 32 elements. Needs `d` divisible by 32. |
| `q4h` | `d/2 + 2*(d/32)` | 4-bit codes after a Hadamard rotation, Float16 scale per 32. Needs `d` divisible by 32 and a power of two. The CUDA backend implements it; Lava refuses it with `GESSO_KV_FORMAT=q4h is not implemented on Lava; use q8 or f32`. |

## Per-token cost

Only full-attention layers keep a per-token cache. Linear-attention
(DeltaNet) and short-convolution layers keep a fixed-size state that does not
grow with context. So:

```
kv_bytes_per_token = (full-attention layers) x 2 (K and V) x n_kv_heads x row_bytes(head_dim, format)
```

Gesso computes this for you:

```julia
spec = ...   # the third value returned by Gesso.load_gguf
per_token = Gesso.kv_bytes_per_token(spec, :q8)
```

For models whose layers differ in KV head count or head size (Gemma 4), the
function uses one uniform geometry and is only an estimate; the allocation
itself honors each layer's own sizes.

## Sizing a context

`memory_ceiling_tokens` turns a budget into a token count:

```julia
Gesso.memory_ceiling_tokens(; budget_bytes, resident_bytes,
                              scratch_bytes = 0, kv_bytes_per_token = per_token)
```

It returns `floor((budget - resident - scratch) / kv_bytes_per_token)`, or 0
when nothing fits. `resident_bytes` is your model weights that live in that
memory, and `budget_bytes` is the RAM or VRAM you can spare. Leave room for
the operating system and for the scratch buffers below.

Allocation rules to keep in mind:

- Caches are allocated up front at the full context length, not as the
  conversation grows. For a single `generate` call that is the prompt plus
  `max_new_tokens`. The chat server allocates once for `--n-ctx`, so a large
  `--n-ctx` costs its full size at startup.
- Quantized formats round the row count up to a multiple of 4.
- Models with DeltaNet layers also hold, per such layer, a convolution state
  of `(2*key_dim + value_dim) x (conv_kernel_dim - 1)` and a recurrent state
  of `num_value_heads x key_head_dim x value_head_dim` numbers. Short-conv
  layers hold `hidden_size x (conv_kernel_dim - 1)`.
- There is also one hidden row, decode scratch, and, for hybrid models,
  checkpoint slots for prefix reuse.

A request that does not fit the state fails with
`reference_generate: prompt P + max_new_tokens N exceed the state's ctx C`.

## GPU weight budget

On a GPU backend, weights are moved to the device up to a budget set by
`GESSO_GPU_BUDGET_MIB` (default 4800 MiB; an unparsable value falls back to
the default). Weights past the budget stay in host memory, and for
mixture-of-experts models the experts that do not fit are read from the host.
Raise the value on a larger card. Set it with the KV cache in mind, since the
cache also lives on the device. `GESSO_GPU_HOT_EXPERTS` and
`GESSO_GPU_HOT_EXPERTS_MIB` keep frequently used host experts in a device
pool; see [env-vars.md](env-vars.md) for which backends read which variables.

## Saving a session

`Gesso.save_state(path, state)` and `Gesso.load_state(path, model, tensors;
backend)` persist a generation state. Loading refuses a file for another model
or KV format, a bad magic or version, a truncated file, or a context mismatch.

## Context policy helpers

For long conversations the library also offers `ContextBudget`, `CostModel`,
`CompactionPolicy` and `decide_compaction`, which choose when to compact
history (`:rent_or_buy`, `:threshold` or `:never`). The chat server does not
use them.
