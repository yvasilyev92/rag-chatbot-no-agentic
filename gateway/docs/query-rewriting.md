# Conversational query rewriting

Follow-ups like `"tell me more about that"` have no keywords. Before search, [`rewrite_query`](../app/rag.py) asks gpt-4o-mini (same OpenAI client/key as the input guard, not vLLM) to turn the latest message plus recent history into a standalone query.

The rewrite is **search-only**. DynamoDB, the chat LLM, and the stored transcript all keep the original message.

Always on under `RAG_ENABLED`. Skipped when there's no prior history, `is_chitchat` already gated the turn, or `OPENAI_API_KEY` is empty. On any failure, use the literal message.

We don't try to detect follow-ups in Python — the prompt says return the message unchanged if it's already standalone.

## Cache

In-process LRU keyed by `(session_id, history_length, latest_message)`. Hits skip the OpenAI call (retries / regenerations). Not in DynamoDB; a pod restart just re-rewrites once.

## Sanitize

[`_clean_rewrite_output`](../app/rag.py) falls back to the literal message if the model answers instead of rewriting:

- empty / whitespace
- starts with `"Sure,"` / `"Here is"` / `"Sorry,"` / `"I'm"` / `"Based on"` / etc.
- longer than 400 characters
- multi-line → first line only
- wrapping quotes / backticks / bullets → stripped, not rejected

## Logs

| Log | Means |
| --- | --- |
| `Rewrote query: 'tell me more about that' -> 'Soulfire Necklace details'` | Rewrite changed the search string. |
| `Query rewrite failed, using literal message: ...` | Timeout / 5xx / parse error; searching the original. |
| (silent) | Unchanged rewrite, skip (no history / no key / chit-chat), or cache hit. |

## Knobs

Constants in `rag.py` (coupled to `_QUERY_REWRITE_SYSTEM_PROMPT`). `OPENAI_API_KEY` is the only env var.

| Knob | Default | Notes |
| --- | --- | --- |
| `QUERY_REWRITE_MODEL` | `gpt-4o-mini` | |
| `QUERY_REWRITE_HISTORY_TURNS` | `4` | Prior messages in the rewrite prompt. |
| `QUERY_REWRITE_MAX_TOKENS` | `64` | Output cap. |
| `QUERY_REWRITE_TEMPERATURE` | `0.0` | Deterministic. |
| `QUERY_REWRITE_TIMEOUT_SECONDS` | `5.0` | Then fall through to literal. |
| `QUERY_REWRITE_CACHE_SIZE` | `512` | Per-pod LRU. |
