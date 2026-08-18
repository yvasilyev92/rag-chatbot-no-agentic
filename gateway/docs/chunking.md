# Markdown-aware chunking

[`chunk_markdown`](../app/rag.py) splits `.md` uploads at document structure instead of a sliding character window. PDF/TXT still use [`chunk_pages`](../app/rag.py) / [`chunk_text`](../app/rag.py). Orchestrated by `process_document_background` in [`documents.py`](../app/documents.py).

## Markdown path

1. **Records.** Split on `---` lines. Each record has YAML frontmatter (`document`, `category`, optional `topic`/`type`, `last_updated`) plus a markdown body.
2. **Sections.** Each `##` becomes its own chunk. Content before the first H2 uses the H1 title (or `"Overview"`).
3. **Oversized fallback.** Sections over `CHUNK_SIZE` are character-split with `CHUNK_OVERLAP`. Most sections fit in one chunk.
4. **Preamble.** Every chunk is prefixed before embed/index:

```
[Document: Equipment Catalog] [Category: Equipment] [Topic: Weapons] [Section: Weapon Types Overview]

(... chunk text ...)
```

The preamble is in the indexed `content` (so embeddings, BM25, and the reranker all see it). `raw_text` is stored without it for prompt display.

5. **Metadata fields.** `document_name`, `category`, `topic`, `section_title`, and `last_updated` are indexed separately. `category` / `topic` / `document_name` are the filterable fields on the chat API.

No overlap between H2 sections — they're independent topics. Overlap only applies inside the character-fallback path.

## PDF / TXT

Character-split with sentence/paragraph boundary preference, plus a `[Document: filename]` preamble. No frontmatter metadata. Same hybrid search + rerank path afterward.

## Knobs

| Env var | Default | Notes |
| --- | --- | --- |
| `CHUNK_SIZE` | `2000` | Character-fallback threshold (markdown oversized sections, and all PDF/TXT). |
| `CHUNK_OVERLAP` | `200` | Character-fallback only. Most markdown chunks have zero overlap. |

After a chunking or schema change: delete the OpenSearch index, restart the gateway, re-upload docs (`scripts/upload-docs.sh`). The gateway recreates the index on the next request.
