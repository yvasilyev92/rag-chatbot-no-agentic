"""Tests for in-memory LRU/TTL caches."""
import time
from unittest.mock import patch

from app import rag as rag_module
from app.rag import (
    _query_rewrite_cache,
    _query_rewrite_cache_lock,
    _rewrite_cache_get,
    _rewrite_cache_set,
    _search_cache,
    _search_cache_lock,
    _search_cache_get,
    _search_cache_set,
    bust_search_cache,
)


def _clear_rewrite_cache():
    with _query_rewrite_cache_lock:
        _query_rewrite_cache.clear()


def _clear_search_cache():
    with _search_cache_lock:
        _search_cache.clear()


class TestRewriteCache:
    def setup_method(self):
        _clear_rewrite_cache()

    def test_get_set_roundtrip(self):
        key = ("session-1", 2, "tell me more")
        _rewrite_cache_set(key, "Soulfire Necklace", max_size=10)
        assert _rewrite_cache_get(key) == "Soulfire Necklace"

    def test_lru_eviction(self):
        for i in range(5):
            _rewrite_cache_set(("s", i, f"msg{i}"), f"q{i}", max_size=3)
        assert len(_query_rewrite_cache) == 3
        assert _rewrite_cache_get(("s", 0, "msg0")) is None
        assert _rewrite_cache_get(("s", 4, "msg4")) == "q4"

    def test_move_to_end_on_hit(self):
        _rewrite_cache_set(("s", 1, "a"), "qa", max_size=3)
        _rewrite_cache_set(("s", 2, "b"), "qb", max_size=3)
        _rewrite_cache_set(("s", 3, "c"), "qc", max_size=3)
        _rewrite_cache_get(("s", 1, "a"))  # refresh "a"
        _rewrite_cache_set(("s", 4, "d"), "qd", max_size=3)
        assert _rewrite_cache_get(("s", 1, "a")) == "qa"  # not evicted
        assert _rewrite_cache_get(("s", 2, "b")) is None  # oldest untouched


class TestSearchCache:
    def setup_method(self):
        _clear_search_cache()

    def test_get_set_roundtrip(self):
        key = ("query", (), 5)
        results = [{"content": "chunk", "score": 0.9}]
        _search_cache_set(key, results, max_size=10)
        assert _search_cache_get(key, ttl_seconds=600) == results

    def test_ttl_expiry(self):
        key = ("query", (), 5)
        results = [{"content": "chunk"}]
        t0 = 1000.0
        with patch.object(time, "monotonic", side_effect=[t0, t0 + 601]):
            _search_cache_set(key, results, max_size=10)
            assert _search_cache_get(key, ttl_seconds=600) is None

    def test_bust_clears_all(self):
        _search_cache_set(("q1", (), 5), [{"content": "a"}], max_size=10)
        _search_cache_set(("q2", (), 5), [{"content": "b"}], max_size=10)
        cleared = bust_search_cache()
        assert cleared == 2
        assert _search_cache_get(("q1", (), 5), ttl_seconds=600) is None

    def test_size_bound(self):
        for i in range(5):
            _search_cache_set((f"q{i}", (), 5), [{"content": str(i)}], max_size=3)
        assert len(_search_cache) == 3
