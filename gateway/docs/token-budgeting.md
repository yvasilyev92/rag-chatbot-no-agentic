# Token budgeting

vLLM truncates from the **front** of the message list when the prompt exceeds `--max-model-len`. That's where the system prompt lives, so an oversized turn silently drops RAG. [`_assemble_messages_for_budget`](../app/main.py) + [`build_rag_context`](../app/rag.py) share one per-turn budget so that doesn't happen.

`MODEL_CONTEXT_TOKENS` **must** equal vLLM's `--max-model-len` (both default `4096`). Mismatch = silent truncation.

Estimator is `len(text) // CHARS_PER_TOKEN` (`4`). Overestimates English a bit, which is the safer direction.

## Allocation order

1. Reserve completion: `max(request.max_tokens or 300, RESERVED_COMPLETION_TOKENS)`.
2. Size RAG body (highest-scored chunks first):

```
available  = MODEL_CONTEXT_TOKENS - completion - SYSTEM_PROMPT_OVERHEAD_TOKENS - new_user
rag_budget = max(0, min(RAG_MAX_CONTEXT_TOKENS, available // 2))
```

The `// 2` soft cap means RAG cannot wipe history. `build_rag_context` stops adding when the next chunk wouldn't fit. If nothing fits: log `RAG: no chunk fits within N-token budget; skipping injection`.

3. Trim history oldest-first to what's left. System prompt + new user turn always stay.

```
history_budget = MODEL_CONTEXT_TOKENS - completion - new_user - system_tokens
effective      = max(0, min(history_budget, MAX_HISTORY_TOKENS))
```

`system_tokens` is the full server prompt (base or base+RAG), not just the chunk body.

## Log

```
Token budget: ceiling=4096 completion=512 new_user=12 system=800 (rag_body=400) history=1200/3000 (kept 8/14 msgs)
```

`rag_body` is retrieved-chunk tokens only. Persistent `kept << M` → raise `MODEL_CONTEXT_TOKENS` (and vLLM `MAX_MODEL_LEN`) or lower `RAG_MAX_CONTEXT_TOKENS`.

## Knobs

| Knob | Default | Where | Notes |
| --- | --- | --- | --- |
| `MODEL_CONTEXT_TOKENS` | `4096` | env | Pair with `kubernetes/configmap.yaml` → `MAX_MODEL_LEN`. |
| `MAX_HISTORY_TOKENS` | `3000` | env | Hard cap on history, independent of the per-turn remainder. |
| `RESERVED_COMPLETION_TOKENS` | `512` | constant (`main.py`) | Floor even if the client sends a tiny `max_tokens`. |
| `RAG_MAX_CONTEXT_TOKENS` | `2500` | constant (`main.py`) | Hard cap on RAG body; runtime also applies `available // 2`. |
| `SYSTEM_PROMPT_OVERHEAD_TOKENS` | `500` | constant (`main.py`) | Template slack before RAG body is sized. Bump if you lengthen the prompt builders. |
| `CHARS_PER_TOKEN` | `4` | constant (`config.py`) | Llama-ish char/token guess. |
