# Serving

Gesso includes an OpenAI-compatible chat server. It serves one model and runs
one generation at a time. Up to 16 chat requests wait in a queue behind the
running one; past that, or after 300 seconds of waiting, a request gets `503`
with `Retry-After`.

## Start it

```bash
julia --project=benchmark scripts/gesso-serve.jl --gguf /path/to/model.gguf
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--gguf PATH` | `GESSO_ORNITH35_GGUF` | Model file. One of the two is required. |
| `--host H` | `127.0.0.1` | IPv4 address to bind. |
| `--port P` | `8080` | Port to listen on. |
| `--n-ctx N` | `65536` | Context length of the server's single session. |
| `--model-id ID` | `ornith-35b` | Name reported by `/v1/models` and `/props`. |
| `--cpu` | off | Run on the CPU backend. |
| `--check` | off | Start a fake engine on an ephemeral port, query `/health`, exit. |

`GESSO_BACKEND` selects the backend: `lava` (the default), `cuda` or `cpu`.
Any other value is an error. The launcher loads the GPU package itself, moves
the weights to the device, runs a short warm-up, and then prints
`gesso-serve ready on HOST:PORT n_ctx=N model=ID`. Bad arguments print usage
to stderr and exit with status 2.

Memory use grows with `--n-ctx`; see [memory.md](memory.md).

From Julia, `Gesso.serve(model; host="127.0.0.1", port=8080, block=true,
read_timeout=30, max_connections=64, max_queue=16, queue_timeout=300)` starts
the same server. `port=0` picks
a free port (read it from the returned server's `port`). `close(server)` stops
it. Connections past `max_connections` receive `503`. `max_queue` caps the
chat requests waiting behind the running generation (`0` refuses any request
that arrives while one runs), and `queue_timeout` is how many seconds one may
wait. A client that disconnects while queued leaves the queue at once.

## Chat families

The model needs a chat template in its GGUF. The server recognizes these
families from the template text and refuses any other:

| Family | Tools accepted |
| --- | --- |
| Qwen3.5 XML | yes |
| LFM2 | no |
| LFM2.5 | no |
| Gemma 4 | yes |

The server stops on the tokenizer's end-of-sequence token plus any of
`<|im_end|>`, `<|endoftext|>`, `<turn|>` and `<|tool_response>` that exist in
the vocabulary.

## Endpoints

### `GET /health`

`{"status":"ok"}`

### `GET /props`

```json
{"default_generation_settings":{"n_ctx":65536},"total_slots":1,"model_alias":"ornith-35b"}
```

### `GET /v1/models`

The OpenAI model list, with the single model under the `--model-id` name.

### `POST /v1/chat/completions`

Request body, a JSON object:

| Field | Notes |
| --- | --- |
| `messages` | Required, a non-empty array of objects each with a `role`. |
| `tools` | Array of tool definitions. Refused with `tools_unsupported` if the model's family does not accept tools. |
| `stream` | Boolean. |
| `stream_options` | `{"include_usage": true}` adds a final usage chunk to a stream. |
| `stop` | A string or an array of strings. |
| `max_completion_tokens`, `max_tokens` | Positive integer. `max_completion_tokens` wins if both are present. The limit is also capped at `n_ctx` minus the prompt length. |
| `temperature` | 0 to 2. 0 or absent means greedy. |
| `top_p` | In (0, 1]. |
| `seed` | Non-negative integer. |
| `chat_template_kwargs.enable_thinking`, `enable_thinking` | Turns thinking on or off. Default: on for Qwen and LFM, off for Gemma 4. |

Tool-call `arguments` inside assistant messages may be JSON strings; they are
parsed into objects before the template is applied.

Response:

```json
{
  "id": "chatcmpl-<16 hex digits>",
  "object": "chat.completion",
  "model": "<model id>",
  "choices": [{"index": 0,
               "message": {"role": "assistant", "content": "...", "reasoning_content": "..."},
               "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0}}
}
```

`finish_reason` is `stop`, `length` or `tool_calls`. `cached_tokens` counts
prompt tokens reused from the previous turn, because the server keeps its KV
cache and recurrent state between requests and reuses a matching prefix.

### Streaming

With `"stream": true` the reply is a chunked `text/event-stream`. The server
sends a role chunk, then deltas carrying `reasoning_content`, `content` or
`tool_calls`, then a chunk with the `finish_reason`, then (if requested) a
usage chunk, then `data: [DONE]`. If the client disconnects, generation is
cancelled. If generation fails after the stream has started, the connection is
closed without a final event.

## Errors

Errors use the OpenAI shape:

```json
{"error": {"message": "...", "type": "invalid_request_error", "code": "...", "param": "..."}}
```

`param` appears only when one field is to blame. The full list is in
[errors.md](errors.md).

## Limits

- Request headers: 64 KiB (`431` beyond that).
- Request body: 32 MiB (`413` beyond that). Chunked request bodies and
  `Expect: 100-continue` are supported; connections are kept alive.
- A prompt with `n_ctx` tokens or more is refused with
  `context_length_exceeded`.
- The launcher does not expose `read_timeout`, `max_connections`, `max_queue`
  or `queue_timeout`; use `Gesso.serve` from Julia to change them.
