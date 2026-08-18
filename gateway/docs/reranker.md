# Cross-encoder rerank

Hybrid search ranks chunks independently of the query (vectors + BM25). [`rerank`](../app/rag.py) then scores each `(query, chunk)` pair jointly with a cross-encoder and keeps the top `RAG_TOP_K`.

```
hybrid (pool of RERANK_POOL=20) → cross-encoder → top RAG_TOP_K → RAG_MIN_SCORE floor
```

Always on under `RAG_ENABLED`. Skipped when there's no query text or only one candidate. On any error, [`search()`](../app/rag.py) logs a warning and returns the un-reranked hybrid top-K.

## Model

`Xenova/ms-marco-MiniLM-L-6-v2` via `fastembed.TextCrossEncoder` (~22M params, ~80MB, ~10ms/pair on CPU). Pre-downloaded in [`gateway/Dockerfile`](../Dockerfile), pre-loaded in the FastAPI lifespan.

Swap = edit `RERANK_MODEL_NAME` **and** the matching `RUN python -c "...TextCrossEncoder(...)"` line, then rebuild.

Raw logits are sigmoid-normalized to `[0, 1]`. That scale is what [`_apply_min_score`](../app/rag.py) uses (`RAG_MIN_SCORE`, default `0.35`). RRF/kNN scores are not comparable, so the floor only runs after a successful rerank. See [retrieval-gating.md](retrieval-gating.md).

## Knobs

| Knob | Default | Where | Notes |
| --- | --- | --- | --- |
| `RERANK_MODEL_NAME` | `Xenova/ms-marco-MiniLM-L-6-v2` | constant (`rag.py`) | Coupled to the Docker pre-download. |
| `RERANK_POOL` | `20` | constant (`rag.py`) | Hybrid candidates fed to the reranker. Tune with `RAG_TOP_K`. |
| `RAG_TOP_K` | `5` | env | Chunks kept after rerank (before the score floor). |
| `RAG_MIN_SCORE` | `0.35` | env | Drop chunks below this sigmoid score. |
