# Errors

## Typed errors

Library failures are `Gesso.GessoError` values with a code, a message and a
`detail` dictionary. They print as `GessoError(<code>): <message>` followed by
the detail.

| Code | Meaning | Where you will meet it |
| --- | --- | --- |
| `ERR_INVALID_PLAN` | The request or configuration is not valid. | Bad `GESSO_KV_FORMAT`; head size not divisible by 32 for `q8`/`q4h`; `q4h` on Lava; a prompt plus `max_new_tokens` larger than the state's context; a state built for another model, backend or KV format; a token ID outside the vocabulary; a malformed checkpoint input; a saved state that does not match. |
| `ERR_RESOURCE_LIMIT` | A resource the request needs is not there. | Asking for Lava with no Vulkan device, or CUDA with no functional NVIDIA device. Also raised when a `Session` runs out of context: `prefill!: prompt length P exceeds context_length C`, `decode!: context_length C exhausted`, and `append_kv!` past the cache size. |
| `ERR_ALLOCATION` | Out of memory. | An `OutOfMemoryError` while loading a checkpoint or running an engine operation is re-raised with this code. |
| `ERR_RUNTIME` | An operation failed while running or while reading a file. | Unsupported GGUF architecture or tensor type; unsupported Gemma 4 variant; other engine failures (the original error is in `detail.cause`). |
| `ERR_NUMERICAL_INSTABILITY` | A computation produced NaN or Inf, or had no finite value to work with. | `rmsnorm!: nonfinite normalization denominator`, `softmax!: no finite unmasked maximum`, `inference: nonfinite logits; no token may be emitted`. |
| `ERR_VERIFY_MISMATCH` | A checked kernel disagreed with its reference. | Autotuning and CUDA operator verification. |
| `ERR_INTERNAL` | A bug in Gesso. | Please report it with the message and detail. |

The codes `ERR_CONSTRAINT_REJECTED`, `ERR_COMPILE`, `ERR_LAUNCH`,
`ERR_TIMEOUT`, `ERR_CACHE`, `ERR_BENCHMARK` and
`ERR_APPROXIMATION_BUDGET_EXCEEDED` are defined, but the inference paths do
not raise them in this version.

`LoweringNotImplemented` is a separate error: `operation :OP has no lowering
for backend :B`. It means the backend has no implementation of that operation,
and Gesso deliberately does not substitute another one.

## Loading errors

Some load failures are plain messages rather than `GessoError`s:

- `load_gguf: missing required tensor: NAME`
- `load_gguf: nextn_predict_layers=...` when a multi-token-prediction layout is
  not understood
- `adapter_for: no architecture adapter for model_type ...`
- Gemma 4 files with unsupported fields (see [models.md](models.md))

Unsupported architectures and tensor types are `ERR_RUNTIME`:

- `spec_from_gguf_kv: unsupported general.architecture ...` followed by the
  supported list.
- `gguf_import: ggml type N (tensor X) has no reader in Gesso`.

## Server errors

The server does not translate `GessoError` codes into HTTP statuses. A request
that fails inside generation gets a `500` with the message `internal error`,
and the connection is closed. Errors the server detects itself use this body:

```json
{"error": {"message": "...", "type": "...", "code": "...", "param": "..."}}
```

| Status | `type` | When |
| --- | --- | --- |
| 400 | `invalid_request_error` | Body is not a JSON object; `messages` missing, empty, or not objects with a `role`; a field of the wrong type or outside its range (`temperature`, `top_p`, `seed`, `stop`, token limits, `stream`). |
| 400 | `invalid_request_error`, code `tools_unsupported`, param `tools` | `tools` sent to a model whose chat format has no tool support. |
| 400 | `invalid_request_error`, code `context_length_exceeded`, param `messages` | The prompt is at least `n_ctx` tokens: "This model's maximum context length is N tokens. However, your messages resulted in M tokens. Please reduce the length of the messages." |
| 404 | `not_found_error` | `no route for <path>`. |
| 405 | `invalid_request_error` | Wrong method: `<M> not allowed here, use <M2>`. |
| 413 | `invalid_request_error` | Body over 32 MiB. |
| 431 | `invalid_request_error` | Headers over 64 KiB. |
| 500 | `server_error` | `internal error`. |
| 503 | `server_error` | `too many open connections`. |
| 503 | `server_error`, header `Retry-After: 1` | `the generation queue is full, retry later`: `max_queue` chat requests are already waiting. |
| 503 | `server_error`, header `Retry-After: 1` | `waited too long for the generation queue, retry later`: the request waited `queue_timeout` seconds; the connection is closed. |

If a streamed response fails after it has started, the status line is already
sent, so the connection is simply closed.

## Launcher errors

`scripts/gesso-serve.jl` prints usage and exits with status 2 for:
`no GGUF given: pass --gguf <path> or set GESSO_ORNITH35_GGUF`,
`GGUF not found: <path>`, `unknown argument X`, and `X needs a value`.
An invalid `GESSO_BACKEND` raises `GESSO_BACKEND must be lava, cuda or cpu`.
The server also refuses to start for a chat family it cannot serve, or a
tokenizer with no end-of-turn token.
