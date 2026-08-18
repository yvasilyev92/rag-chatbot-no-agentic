# SSE streaming

`POST /v1/sessions/{id}/chat/completions` with `"stream": true` (default `false`). Helpers in [`gateway/app/main.py`](../app/main.py). No extra env vars.

RAG + token budget still run to completion first. Only the vLLM completion is streamed. `/v1/chat/completions` is 404.

## Path

1. `_stream_vllm_completion` — `httpx` SSE to vLLM, yield each line as `data: ...\n\n`. Mutates a shared `state` dict (`buffer`, `finish_reasons`, `last_usage`, `saw_done`, `error`) because an async generator's return value isn't reachable from `StreamingResponse`. Forwards vLLM bytes as-is (no re-serialize).
2. `_stream_and_persist` — yields those chunks to the client. In `finally` (clean end, disconnect, or error): if `state["buffer"][0]` is non-empty, persist user + assistant to DynamoDB.

Input-guard refusals skip vLLM and use `_stream_canned_refusal` (one content chunk + `[DONE]`), then persist as usual.

vLLM `usage` is requested via `stream_options.include_usage`.

## Disconnect

FastAPI raises `CancelledError` on client close; httpx then drops the vLLM connection (GPU slot released). `interrupted` is true if cancelled, vLLM errored, or `[DONE]` never arrived.

Two truncation signals, together:

- `metadata.interrupted=true` (plus `streamed=true`, `finish_reason`, `tokens`)
- `\n\n_[response interrupted]_` appended to persisted content

Empty buffer → persist nothing (same all-or-nothing as non-streaming). Only `choices[0]` is saved.

## Headers

```
Cache-Control: no-cache
X-Accel-Buffering: no
Connection: keep-alive
```

`X-Accel-Buffering: no` stops nginx/CloudFront from buffering the whole `text/event-stream` body.
