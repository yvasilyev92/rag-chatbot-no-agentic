# RAG caching

Two in-memory LRU caches in [`gateway/app/rag.py`](../app/rag.py), each behind a `threading.Lock`. They skip repeat embedding / OpenSearch / rerank work for identical retrieval inputs. Always on under `RAG_ENABLED`; no per-cache flag.

Pod-local only (not Redis). Retrieval only — the LLM call still runs every time.

## Embedding cache (`_embedding_cache`)

- **Key**: query string (post-rewrite, if rewriting fired).
- **Value**: 384-dim `fastembed` vector.
- **Size**: `EMBEDDING_CACHE_SIZE` (1024, ~1.5 MB).
- **Eviction**: LRU, no TTL. Embeddings are deterministic for a fixed model; restart the process to refresh.

## Search cache (`_search_cache`)

- **Key**: `(query_text, normalized_filters, top_k)`. `{"category": "Equipment"}` and `{"category": ["Equipment"]}` hit the same entry.
- **Value**: the list `OpenSearchRAG.search` would return (post-rerank, post min-score floor).
- **Size**: `SEARCH_CACHE_SIZE` (256).
- **Eviction**: LRU, plus `bust_search_cache()` on corpus change, plus a lazy TTL (`SEARCH_CACHE_TTL_SECONDS`, 600s) as a safety net.

Bust is global (every entry). Called from:

- `documents.process_document_background` — after indexing an upload
- `DocumentManager.delete_document` — after OpenSearch delete-by-query

Writes happen on the way out, so a miss stores exactly what the next caller would have computed.

## Knobs

Module-level constants in `rag.py`. Bounded pod-local memory; no ops reason to tune them.

| Knob | Default | Notes |
| --- | --- | --- |
| `EMBEDDING_CACHE_SIZE` | `1024` | ~1.5 MB. |
| `SEARCH_CACHE_SIZE` | `256` | |
| `SEARCH_CACHE_TTL_SECONDS` | `600` | Safety net. Corpus-change busts are the primary invalidation path. |
