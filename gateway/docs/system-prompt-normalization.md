# System prompt normalization

vLLM gets **at most one** `role="system"` message, at index 0, and it is always server-controlled. [`_assemble_messages_for_budget`](../app/main.py).

Chat templates disagree on multiple system messages (first vs last vs concat vs error). We don't leave that to chance.

## What is sent

Every session turn starts with [`build_base_system_prompt`](../app/rag.py) (persona + scope + canned refusal). If retrieval returns chunks, [`build_rag_system_prompt`](../app/rag.py) replaces it with base + document context.

Client `role: "system"` on `POST /v1/sessions/{id}/chat/completions` → **400**. `/v1/chat/completions` is 404.

System rows in stored history are dropped (warning log). `add_message` only persists user/assistant, so this is defense in depth.

`_merge_system_content(rag_system_prompt, [])` is the production call — extras are unused. Other roles (`"moderator"`, etc.) still pass Pydantic and go to vLLM as-is.

## Budget

System tokens come out of the history slice:

```
history_budget = MODEL_CONTEXT_TOKENS - completion_reserve - new_user_tokens - system_tokens
```

Then oldest-first trim of user/assistant history. The system prompt and the new user turn always stay.

Log line: `Token budget: ... system=N (rag_body=B) history=H/cap ...`. `rag_body` is retrieved-chunk tokens only, not the whole system message.
