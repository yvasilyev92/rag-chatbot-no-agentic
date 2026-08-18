# RAG Chat API

Client guide for a chat UI. Four endpoints: create session, send a turn, get history, delete session.

The gateway owns history, RAG, persona, and off-topic refusal. You send **only the new user turn**. `POST /v1/chat/completions` is 404 — always use the session chat URL.

## Quick reference

| Field | Value |
| --- | --- |
| Base URL | `https://<nlb-hostname>` (TLS at the NLB, port 443) |
| Auth | `Authorization: Bearer <API_KEY>` except `GET /live`, `GET /health`, `GET /` |
| Model | Whatever vLLM is serving (default `meta-llama/Llama-3.1-8B-Instruct`). Forwarded as-is. |
| Session TTL | 24h sliding on **chat persist**, not on GET. Messages expire ~24h after write. |
| `max_tokens` | Default 300 |
| Streaming | `"stream": true` → SSE; JSON is the default |

Missing/invalid key → `401`. Store the key as `VLLM_API_KEY`; do not commit it. The NLB hostname changes if the cluster is rebuilt.

## Quickstart

```bash
export GATEWAY_URL="https://<nlb-hostname>"
export API_KEY="<API_KEY>"

SESSION_ID=$(curl -sS -X POST \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}' \
  "$GATEWAY_URL/v1/sessions" | jq -r '.session_id')

curl -sS -X POST \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What are relics?"}]
  }' \
  "$GATEWAY_URL/v1/sessions/$SESSION_ID/chat/completions" \
  | jq -r '.choices[0].message.content'

curl -sS -X POST \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Which is best for a fire build?"}]
  }' \
  "$GATEWAY_URL/v1/sessions/$SESSION_ID/chat/completions" \
  | jq -r '.choices[0].message.content'
```

The follow-up works because turn 1 is in DynamoDB; the rewriter uses it. Do not resend prior turns.

## Endpoints

### `GET /live`

Liveness. No auth. `{ "status": "ok" }`. Use for k8s liveness/startup, not readiness.

### `GET /health`

Readiness. No auth. Probes vLLM, DynamoDB, OpenSearch (if RAG is on). **200** when ready; **503** when a required dep is down.

### `POST /v1/sessions` — create session

**200:**

```json
{
  "session_id": "45447eb5-f85f-4151-9691-699d658b8d58",
  "created_at": "2026-05-26T19:06:28.056379Z",
  "expires_at": "2026-05-27T19:06:28Z"
}
```

Optional body: `{ "metadata": { "user_id": "abc-123" } }` (any JSON). Empty `{}` is fine. Persist `session_id`.

### `POST /v1/sessions/{session_id}/chat/completions` — send turn

OpenAI-shaped. **Only the new user turn(s)** in `messages`. Do not send `role: "system"` (persona is server-controlled) or prior assistant turns.

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [{ "role": "user", "content": "What is a good fire build?" }],
  "max_tokens": 300,
  "stream": false
}
```

| Field | Required | Default | Notes |
| --- | --- | --- | --- |
| `model` | yes | — | Passed through to vLLM. |
| `messages` | yes* | — | New user turn(s) only. `*`can be omitted but then there is no new user text. |
| `max_tokens` | no | `300` | Output cap. |
| `temperature` / `top_p` / `stop` | no | vLLM default | |
| `stream` | no | `false` | SSE when `true`. |
| `rag_filters` | no | — | See below. |

Answer: `choices[0].message.content`.

| Code | Meaning |
| --- | --- |
| 200 | Success |
| 400 | `role: system` in `messages` |
| 401 | Bad/missing API key |
| 404 | Session missing or expired |
| 422 | Body validation (unknown `rag_filters` keys, bad types) |
| 500 / 502 / 503 | Backend. Retry once; 503 can mean vLLM down |

Sending the full history doubles tokens, confuses the rewriter, and can duplicate stored turns.

#### Streaming

`"stream": true` → OpenAI SSE (`data: {...}\n\n`, ends with `data: [DONE]\n\n`). RAG + budget still run **before** the first byte. Parse `choices[0].delta.content`. Disconnect: persist partial with `metadata.interrupted: true` and an inline `_[response interrupted]_` marker. Use `curl -N` (or disable client buffering). Empty stream → persist nothing.

### `GET /v1/sessions/{session_id}` — history

**200:** `session_id`, `messages` (`role`, `content`, `created_at`), `message_count`, `created_at`, `expires_at`. **404** if gone. Does **not** slide TTL.

### `DELETE /v1/sessions/{session_id}` — delete

**200:** `{ "session_id", "deleted": true, "message": "..." }`. Optional — DynamoDB TTL also expires sessions. Use for "New chat".

## UI lifecycle

1. First open → `POST /v1/sessions`, store `session_id`.
2. Each message → session chat with **only** that user turn; render `choices[0].message.content`.
3. "New conversation" → `DELETE`, then create again.
4. `404` on chat → expired; create a new session.

## Assistant behavior

Canned refusal (also used by the input guard): `I can only help with {DESIRED_RAG_TOPIC} questions. What would you like to know?` Detect with substring `"only help with"`.

| User | Typical result |
| --- | --- |
| In-scope question | Direct answer from the knowledge base |
| Off-topic / jailbreak / "what model are you" | Canned refusal (guard first if `OPENAI_API_KEY` is set; else the system prompt) |
| `"hi"` / `"thanks"` | Short reply; RAG skipped |
| In-scope but not in the docs | Model is told to admit the gap |

## RAG filters (optional)

AND across fields, OR within a field. Strict. Keys: `category`, `topic`, `document_name`. Unknown keys → 422.

```json
{
  "rag_filters": {
    "category": "Equipment",
    "topic": ["Relics", "Weapons"]
  }
}
```

Skip for a normal chat box. Useful for filter chips.

## Smoke tests

```bash
curl -sS "$GATEWAY_URL/health" | jq

SESSION_ID=$(curl -sS -X POST -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" -d '{}' \
  "$GATEWAY_URL/v1/sessions" | jq -r .session_id)

# In-scope
curl -sS -X POST -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"What are relics?"}]}' \
  "$GATEWAY_URL/v1/sessions/$SESSION_ID/chat/completions" | jq -r '.choices[0].message.content'

# Off-topic → canned refusal
curl -sS -X POST -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"What is the weather today?"}]}' \
  "$GATEWAY_URL/v1/sessions/$SESSION_ID/chat/completions" | jq -r '.choices[0].message.content'

# Follow-up (history + rewriter)
curl -sS -X POST -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Which set is best for a fire build?"}]}' \
  "$GATEWAY_URL/v1/sessions/$SESSION_ID/chat/completions" | jq -r '.choices[0].message.content'
```

## Limits

- First SSE token waits on RAG + prompt assembly (often 1–4s). Show a typing indicator.
- 24h TTL is a hard cutoff on the **session metadata** unless a **chat** turn refreshes it. Persist transcripts on your side if you need longer threads.
- Gateway default is **2 replicas** (HPA 2–3). Retry with backoff across deploys.
