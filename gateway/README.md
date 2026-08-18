# API Gateway with Conversation Memory and RAG

FastAPI service between clients and vLLM. Adds session memory (DynamoDB), RAG over OpenSearch, and an OpenAI-compatible chat API.

Persona/scope come from `DESIRED_RAG_TOPIC`. Pipeline detail: [RAG_ARCHITECTURE.md](RAG_ARCHITECTURE.md). Client guide: [../CHAT_API.md](../CHAT_API.md). Component deep-dives: [docs/](docs/).

On each session chat turn the gateway: loads history → optional input guard → RAG (if not chit-chat) → token-budgeted prompt → vLLM (JSON or SSE) → persist. `/v1/chat/completions` is 404.

```
┌─────────────────────────────────────────────────────────────────┐
│                         EKS Cluster                              │
│                                                                  │
│   ┌──────────────────┐          ┌──────────────────────────┐   │
│   │   API Gateway    │          │      vLLM Server         │   │
│   │   (FastAPI)      │──────────│      (GPU)               │   │
│   │                  │          │                          │   │
│   │  - Sessions      │          │  - Chat completions      │   │
│   │  - RAG pipeline  │          │  - Model inference       │   │
│   │  - fastembed     │          │                          │   │
│   │    (embed+rerank)│          │                          │   │
│   └────────┬─────────┘          └──────────────────────────┘   │
│            │                                                     │
│   LoadBalancer (external)              ClusterIP (internal)     │
└────────────┼─────────────────────────────────────────────────────┘
             │
     ┌───────┴────────┐              ┌──────────────────────────┐
     │  AWS DynamoDB  │              │    AWS OpenSearch        │
     │  conversations │              │  Faiss kNN + BM25 text   │
     │  + doc metadata│              │  (RAG vector store)      │
     └────────────────┘              └──────────────────────────┘
```

Gateway pods use **IRSA** for DynamoDB and OpenSearch — no static AWS credentials. vLLM is ClusterIP-only.

## Endpoints (`app/main.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/sessions` | POST | Create conversation session |
| `/v1/sessions/{id}` | GET | Get conversation history |
| `/v1/sessions/{id}` | DELETE | Delete session |
| `/v1/sessions/{id}/chat/completions` | POST | Chat with memory + RAG (only chat path) |
| `/v1/documents` | POST | Upload document to knowledge base |
| `/v1/documents` | GET | List all documents |
| `/v1/documents/{id}` | GET | Get document status |
| `/v1/documents/{id}` | DELETE | Delete document and chunks |
| `/v1/models` | GET | List available models |
| `/live` | GET | Liveness — process up, no deps (no auth) |
| `/health` | GET | Readiness — vLLM + DynamoDB + OpenSearch; 503 if not ready (no auth) |
| `/` | GET | API info (no auth) |
| `/docs` | GET | Swagger UI |

Session chat supports `stream: true` (SSE), `rag_filters`, and `max_tokens` (default 300). Send only the new user turn — history is loaded from DynamoDB. Client `role: system` → 400.

## Components

**Session manager** (`app/session.py`) — create/delete sessions; store/retrieve messages. Metadata SK is `0000_metadata`. Token-aware history loader. Metadata TTL slides with activity; each message expires ~`SESSION_TTL_HOURS` after write.

**RAG** (`app/rag.py`) — chunk, embed, hybrid BM25+kNN, rerank, rewrite, caches, system prompts. See [RAG_ARCHITECTURE.md](RAG_ARCHITECTURE.md). Token budget: `len // 4`, RAG body `min(RAG_MAX_CONTEXT_TOKENS, available // 2)`, history trimmed oldest-first, `MAX_HISTORY_TOKENS` as a hard cap. `MODEL_CONTEXT_TOKENS` must equal vLLM `--max-model-len`.

**Documents** (`app/documents.py`) — PDF, TXT, MD, CSV (max 50 MB). Background: extract → chunk → embed → index. Metadata in DynamoDB `vllm-documents`. Raw files are not persisted. Cache bust on add/delete.

**Input guard** (`app/guard.py`) — optional gpt-4o-mini ALLOW/REFUSE (`INPUT_GUARD_ENABLED`). Same `OPENAI_API_KEY` as query rewrite; both skip when the key is empty.

## Configuration (`app/config.py`)

Also in `kubernetes/gateway-configmap.yaml`.

| Variable | Default | Notes |
|---|---|---|
| `RAG_ENABLED` | `true` | Master RAG kill-switch |
| `RAG_TOP_K` | `5` | Chunks kept after rerank |
| `RAG_MIN_SCORE` | `0.35` | Post-rerank relevance floor |
| `MODEL_CONTEXT_TOKENS` | `4096` | Must match vLLM `MAX_MODEL_LEN` |
| `OPENSEARCH_ENDPOINT` | — | Empty disables RAG |
| `SESSION_TTL_HOURS` | `24` | Session / message expiry |
| `MAX_HISTORY_TOKENS` | `3000` | Hard ceiling on history tokens |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 2000 / 200 | PDF/TXT and oversized markdown sections |
| `DESIRED_RAG_TOPIC` | `Desired RAG Topic` | Persona, scope, canned refusal |
| `INPUT_GUARD_ENABLED` | `true` | gpt-4o-mini classifier; no-ops without `OPENAI_API_KEY` |

Also: `AWS_REGION`, `DYNAMODB_TABLE`, `OPENSEARCH_INDEX`, `VLLM_INTERNAL_URL`, `VLLM_API_KEY`, `HOST`, `PORT`, `OPENAI_API_KEY`.

## Security

- `Authorization: Bearer <VLLM_API_KEY>` on all endpoints except `/live`, `/health`, and `/`
- HTTPS at the NLB (`ACM_CERT_ARN`)
- IRSA for DynamoDB and OpenSearch

## DynamoDB

**`vllm-conversations`**

| Row type | `message_id` (SK) | Description |
|---|---|---|
| Session metadata | `0000_metadata` | Created/expiry timestamps, optional client metadata |
| Messages | Timestamp-prefixed UUID | `role`, `content`, `created_at`, `expires_at` (TTL) |

**`vllm-documents`** — upload metadata (`document_id`, filename, status, chunk_count, etc.)

## Usage

```python
import requests

API_URL = "http://your-gateway-url"
API_KEY = "your-api-key"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

session_id = requests.post(f"{API_URL}/v1/sessions", headers=headers).json()["session_id"]

response = requests.post(
    f"{API_URL}/v1/sessions/{session_id}/chat/completions",
    headers=headers,
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "What are relics?"}],
    },
)
print(response.json()["choices"][0]["message"]["content"])

response = requests.post(
    f"{API_URL}/v1/sessions/{session_id}/chat/completions",
    headers=headers,
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "What's the best one for a fire build?"}],
    },
)
print(response.json()["choices"][0]["message"]["content"])
```

Upload docs with `POST /v1/documents`. After chunking or schema changes: delete the OpenSearch index, restart the gateway, re-upload with `scripts/upload-docs.sh`.

## Files

```
gateway/
├── Dockerfile              # Multi-stage build; pre-downloads embedder + reranker
├── requirements.txt
├── RAG_ARCHITECTURE.md     # RAG pipeline reference
├── docs/                   # Per-component deep dives
└── app/
    ├── main.py             # Routes, budget, streaming, RAG orchestration
    ├── session.py          # DynamoDB sessions
    ├── rag.py              # Chunk, search, rerank, rewrite, prompts
    ├── guard.py            # Input-guard classifier
    ├── documents.py        # Upload pipeline
    ├── models.py           # Pydantic (incl. RagFilters)
    └── config.py           # Env vars
```

## Cost

- **DynamoDB**: ~$0.25 per million read/write requests (on-demand)
- **OpenSearch**: ~$26/month (t3.small.search, single node)
- **Gateway pods**: CPU on existing nodes
- **fastembed**: local, no API cost
- **OpenAI gpt-4o-mini**: input guard + query rewrite (skipped without `OPENAI_API_KEY`)
