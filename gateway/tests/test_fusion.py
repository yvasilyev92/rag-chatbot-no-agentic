"""Tests for Reciprocal Rank Fusion."""
from app.rag import OpenSearchRAG


def _hit(chunk_id: str, content: str, score: float = 1.0):
    return {
        "_id": chunk_id,
        "_score": score,
        "_source": {
            "content": content,
            "source_filename": "test.md",
            "page_number": 1,
            "document_id": "doc-1",
        },
    }


class TestRrfFuse:
    def test_overlap_accumulates_scores(self):
        bm25 = [_hit("a", "chunk a"), _hit("b", "chunk b")]
        knn = [_hit("b", "chunk b"), _hit("c", "chunk c")]
        results = OpenSearchRAG._rrf_fuse(bm25, knn, rrf_k=60, top_k=3)
        ids = [r["content"] for r in results]
        assert ids[0] == "chunk b"  # appears in both lists

    def test_top_score_normalized_to_one(self):
        bm25 = [_hit("a", "chunk a")]
        knn = [_hit("b", "chunk b")]
        results = OpenSearchRAG._rrf_fuse(bm25, knn, rrf_k=60, top_k=2)
        assert results[0]["score"] == 1.0

    def test_top_k_truncation(self):
        bm25 = [_hit("a", "a"), _hit("b", "b"), _hit("c", "c")]
        knn = []
        results = OpenSearchRAG._rrf_fuse(bm25, knn, rrf_k=60, top_k=2)
        assert len(results) == 2

    def test_missing_id_skipped(self):
        bm25 = [{"_source": {"content": "orphan"}}]
        knn = [_hit("a", "chunk a")]
        results = OpenSearchRAG._rrf_fuse(bm25, knn, rrf_k=60, top_k=5)
        assert len(results) == 1
        assert results[0]["content"] == "chunk a"

    def test_empty_inputs(self):
        assert OpenSearchRAG._rrf_fuse([], [], rrf_k=60, top_k=5) == []
