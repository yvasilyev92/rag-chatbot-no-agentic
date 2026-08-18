# Vector index: nmslib → faiss

The OpenSearch kNN mapping in [`create_index_if_not_exists`](../app/rag.py) uses HNSW via the **faiss** engine (not the old default, **nmslib**). Algorithm is unchanged; only the library implementing it changed.

```python
"method": {
    "name": "hnsw",
    "engine": "faiss",
    "space_type": "innerproduct",  # not cosinesimil
}
```

`innerproduct` because older OpenSearch + faiss combos don't always support `cosinesimil`. `BAAI/bge-small-en-v1.5` emits L2-normalized vectors, so inner product == cosine similarity.

## Why faiss

**Pre-filtering.** With `rag_filters` on the chat API, faiss applies the filter *during* HNSW traversal. nmslib only post-filters: search the whole graph, then drop non-matches. Narrow filters (a topic with 1–2 chunks) can starve under post-filter — top-K comes back with nothing from that topic.

**nmslib is deprecated.** OpenSearch is steering new kNN work toward faiss/lucene.

Hybrid search, rerank, and chunking don't care which engine is underneath.

## Reindex required

The engine is baked in at index create. No in-place swap:

1. Delete the OpenSearch index (`OPENSEARCH_INDEX`, default `vllm-documents`).
2. Restart the gateway — `create_index_if_not_exists` rebuilds it with faiss.
3. Re-upload docs.

To revert: flip the two mapping fields back to `nmslib` / `cosinesimil` and repeat.
