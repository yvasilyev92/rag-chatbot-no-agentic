"""Tests for RAG metadata filter building and cache-key normalization."""
from app.rag import OpenSearchRAG, _normalize_filters_key


class TestBuildFilterClauses:
    def test_scalar_becomes_term(self):
        clauses = OpenSearchRAG._build_filter_clauses({"category": "Equipment"})
        assert clauses == [{"term": {"category": "Equipment"}}]

    def test_list_becomes_terms(self):
        clauses = OpenSearchRAG._build_filter_clauses({"topic": ["Weapons", "Armor"]})
        assert clauses == [{"terms": {"topic": ["Weapons", "Armor"]}}]

    def test_unknown_field_dropped(self):
        clauses = OpenSearchRAG._build_filter_clauses(
            {"category": "Equipment", "chunk_id": "evil", "tenant_id": "x"}
        )
        assert clauses == [{"term": {"category": "Equipment"}}]

    def test_empty_and_none_dropped(self):
        assert OpenSearchRAG._build_filter_clauses(None) == []
        assert OpenSearchRAG._build_filter_clauses({}) == []
        assert OpenSearchRAG._build_filter_clauses({"category": ""}) == []
        assert OpenSearchRAG._build_filter_clauses({"topic": [None, ""]}) == []

    def test_and_across_fields(self):
        clauses = OpenSearchRAG._build_filter_clauses(
            {"category": "Equipment", "topic": "Weapons"}
        )
        assert len(clauses) == 2


class TestNormalizeFiltersKey:
    def test_scalar_and_list_equivalent(self):
        key_scalar = _normalize_filters_key({"category": "Equipment"})
        key_list = _normalize_filters_key({"category": ["Equipment"]})
        assert key_scalar == key_list

    def test_field_order_independent(self):
        key1 = _normalize_filters_key({"category": "Equipment", "topic": "Weapons"})
        key2 = _normalize_filters_key({"topic": "Weapons", "category": "Equipment"})
        assert key1 == key2

    def test_empty_filters(self):
        assert _normalize_filters_key(None) == ()
        assert _normalize_filters_key({}) == ()
