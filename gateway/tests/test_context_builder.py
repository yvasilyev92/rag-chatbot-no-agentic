"""Tests for RAG context and system prompt builders."""
from app.guard import canned_refusal
from app.rag import (
    _format_rag_chunk,
    build_base_system_prompt,
    build_rag_context,
    build_rag_system_prompt,
)

_TOPIC = "Desired RAG Topic"


class TestFormatRagChunk:
    def test_markdown_metadata_header(self, fake_search_results):
        block = _format_rag_chunk(fake_search_results[0])
        assert block.startswith("[Source: equipment | Topic: Weapons | Section: Stats")
        assert "relevance: 0.92" in block
        assert "Soulfire Necklace stats" in block

    def test_prefers_raw_content(self, fake_search_results):
        block = _format_rag_chunk(fake_search_results[0])
        assert "[Document: equipment]" not in block

    def test_legacy_page_fallback(self, fake_search_results):
        block = _format_rag_chunk(fake_search_results[2])
        assert "Page 3" in block
        assert "Legacy chunk" in block


class TestBuildRagContext:
    def test_empty_results(self):
        context, tokens = build_rag_context([])
        assert context == ""
        assert tokens == 0

    def test_budget_stops_adding_chunks(self):
        results = [
            {"content": "aaa", "raw_content": "aaa", "document_name": "doc", "score": 0.9},
            {"content": "bbb", "raw_content": "bbb", "document_name": "doc", "score": 0.8},
            {"content": "ccc", "raw_content": "ccc", "document_name": "doc", "score": 0.7},
        ]
        # Budget fits the first formatted block (~40 chars) but not a second.
        context, tokens = build_rag_context(results, max_tokens=12, chars_per_token=4)
        assert "aaa" in context
        assert "bbb" not in context
        assert tokens <= 12

    def test_includes_all_when_no_budget(self, fake_search_results):
        context, tokens = build_rag_context(fake_search_results)
        assert "Soulfire Necklace stats" in context
        assert "Iron Helm defense" in context
        assert "Legacy chunk" in context
        assert tokens > 0


class TestSystemPrompts:
    def test_base_prompt_has_refusal(self):
        prompt = build_base_system_prompt(_TOPIC)
        assert canned_refusal(_TOPIC) in prompt
        assert _TOPIC in prompt

    def test_base_prompt_interpolates_custom_topic(self):
        prompt = build_base_system_prompt("Acme HR Policy")
        assert "Acme HR Policy" in prompt
        assert canned_refusal("Acme HR Policy") in prompt

    def test_rag_prompt_wraps_context(self):
        prompt = build_rag_system_prompt("chunk one\n\nchunk two", _TOPIC)
        assert "--- Document Context ---" in prompt
        assert "--- End Context ---" in prompt
        assert "chunk one" in prompt
        assert build_base_system_prompt(_TOPIC).split("\n")[0] in prompt
