# Metadata filters

Optional `rag_filters` on `POST /v1/sessions/{id}/chat/completions` narrows retrieval to matching chunks. Omit or `null` searches everything. No env vars; the client owns the filter.

[`RagFilters`](../app/models.py) + [`_build_filter_clauses`](../app/rag.py). Honored only on the session endpoint (`/v1/chat/completions` is 404).

## Filterable fields

Indexed from markdown frontmatter at upload. PDF/TXT only get `document_name` (from the filename), so `category`/`topic` filters miss them.

| Field | Source | Example |
| --- | --- | --- |
| `category` | frontmatter | `"Equipment"` |
| `topic` | frontmatter `topic:` or `type:` | `"Weapons"` |
| `document_name` | frontmatter `document:` | `"Equipment Catalog"` |

Not filterable: `section_title`, `last_updated`, anything else. Unknown keys → 422 (`extra="forbid"`). `_FILTERABLE_FIELDS` in `rag.py` is a second whitelist; unknown keys that somehow get past Pydantic are dropped.

## Semantics

AND across fields, OR within a field. Strict (excludes non-matches, does not boost).

```json
{
  "rag_filters": {
    "category": "Equipment",
    "topic": ["Weapons", "Armor"]
  }
}
```

Means `Equipment AND (Weapons OR Armor)`. Scalar → OpenSearch `term`; list → `terms`.

The client sets filters; we do not infer them from the user message.

## Engine

Both hybrid sides get the clauses (BM25 `bool.filter`, faiss kNN pre-filter). Same `msearch`, then RRF, then rerank. See [faiss-migration.md](faiss-migration.md) for why pre-filter matters on narrow topics.

A filter that matches nothing → empty retrieval → the model answers with no RAG context (same path as OpenSearch down).
