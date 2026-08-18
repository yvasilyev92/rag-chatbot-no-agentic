# Hybrid search (BM25 + kNN + RRF)

[`_hybrid_search`](../app/rag.py) runs BM25 and kNN in one OpenSearch `msearch`, then fuses with Reciprocal Rank Fusion in [`_rrf_fuse`](../app/rag.py). Always on under `RAG_ENABLED`. On any hybrid error, the same call falls back to pure kNN.

Neither method alone covers both named entities (BM25) and paraphrases (kNN). The fused list is the candidate pool for the [reranker](reranker.md), not the prompt.

## The two queries

| Side | Query | Good at | Weak at |
| --- | --- | --- | --- |
| BM25 | `match` on `content` | Exact names, SKUs, rare strings | Synonyms / paraphrase |
| kNN | HNSW over `embedding` | Conceptual / similar wording | Made-up proper nouns the embedder has never seen |

`rag_filters` attach to both sides (BM25 `bool.filter`, faiss kNN pre-filter). See [metadata-filtering.md](metadata-filtering.md).

## Fusion

Scores from the two engines aren't comparable, so we ignore them and use rank:

```
score(chunk) = Σ  1 / (RRF_K + rank_in_list)
```

A hit in both lists ranks above a hit in one. Display `score` is then rescaled so the top fused hit is `1.0` (raw RRF values are ~0.03).

RRF instead of a weighted average of scores: one constant that doesn't need per-corpus tuning, and a BM25 outlier can't dominate.

## Knobs

Constants in `rag.py`. No ops reason to tune them.

| Knob | Default | Notes |
| --- | --- | --- |
| `HYBRID_CANDIDATE_POOL` | `25` | Per-side fetch before fusion. |
| `RRF_K` | `60` | Cormack et al. smoothing constant. |

`search()` asks hybrid for `max(RERANK_POOL, top_k)` fused hits (`RERANK_POOL=20`), then the reranker cuts to `RAG_TOP_K`.
