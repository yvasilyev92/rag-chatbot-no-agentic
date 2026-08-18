"""Tests for query-rewrite sanitization helpers and the LangChain rewrite call."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import app.rag as rag_mod
from app.rag import (
    _build_rewrite_user_prompt,
    _clean_rewrite_output,
    _query_rewrite_cache,
    _query_rewrite_cache_lock,
    rewrite_query,
)


def _clear_rewrite_cache():
    with _query_rewrite_cache_lock:
        _query_rewrite_cache.clear()
    rag_mod._rewrite_chain = None
    rag_mod._rewrite_chain_api_key = None


def _mock_chain(content: str = "", side_effect=None) -> MagicMock:
    chain = MagicMock()
    if side_effect is not None:
        chain.ainvoke = AsyncMock(side_effect=side_effect)
    else:
        chain.ainvoke = AsyncMock(return_value=content)
    return chain


_HISTORY = [
    {"role": "user", "content": "What is Soulfire Necklace?"},
    {"role": "assistant", "content": "It is a fire relic."},
]


class TestCleanRewriteOutput:
    def test_empty_returns_original(self):
        assert _clean_rewrite_output("", "tell me more") == "tell me more"

    def test_unchanged_when_good(self):
        assert _clean_rewrite_output("Soulfire Necklace details", "tell me more") == (
            "Soulfire Necklace details"
        )

    def test_strips_wrapping_quotes(self):
        assert _clean_rewrite_output('"Soulfire Necklace"', "that") == "Soulfire Necklace"

    def test_strips_backticks_and_bullets(self):
        assert _clean_rewrite_output("`- fire build`", "that") == "fire build"

    def test_bad_prefix_falls_back_to_original(self):
        original = "tell me more"
        assert _clean_rewrite_output("Sure, here is the answer", original) == original
        assert _clean_rewrite_output("I'm sorry, I can't help", original) == original

    def test_multiline_collapses_to_first_line(self):
        assert _clean_rewrite_output("Soulfire Necklace\nextra narration", "that") == (
            "Soulfire Necklace"
        )

    def test_too_long_falls_back(self):
        original = "short"
        long_output = "x" * 401
        assert _clean_rewrite_output(long_output, original) == original


class TestBuildRewriteUserPrompt:
    def test_formats_history_and_latest(self):
        prompt = _build_rewrite_user_prompt(_HISTORY, "tell me more about that")
        assert "Prior turns:" in prompt
        assert "User: What is Soulfire Necklace?" in prompt
        assert "Assistant: It is a fire relic." in prompt
        assert "Latest user message: tell me more about that" in prompt
        assert "Rewritten search query:" in prompt

    def test_empty_history(self):
        prompt = _build_rewrite_user_prompt([], "hello")
        assert "(none)" in prompt


class TestRewriteQuery:
    def setup_method(self):
        _clear_rewrite_cache()

    def test_empty_history_skips_chain(self, monkeypatch):
        getter = MagicMock()
        monkeypatch.setattr(rag_mod, "_get_rewrite_chain", getter)
        result = asyncio.run(
            rewrite_query(
                [],
                "What are relics?",
                openai_api_key="sk-test",
            )
        )
        assert result == "What are relics?"
        getter.assert_not_called()

    def test_missing_key_skips_chain(self, monkeypatch):
        getter = MagicMock()
        monkeypatch.setattr(rag_mod, "_get_rewrite_chain", getter)
        result = asyncio.run(
            rewrite_query(
                _HISTORY,
                "tell me more about that",
                openai_api_key="",
            )
        )
        assert result == "tell me more about that"
        getter.assert_not_called()

    def test_rewrites_from_chain(self, monkeypatch):
        chain = _mock_chain("Soulfire Necklace details")
        monkeypatch.setattr(rag_mod, "_get_rewrite_chain", lambda api_key: chain)
        result = asyncio.run(
            rewrite_query(
                _HISTORY,
                "tell me more about that",
                openai_api_key="sk-test",
            )
        )
        assert result == "Soulfire Necklace details"
        chain.ainvoke.assert_called_once()
        payload = chain.ainvoke.call_args.args[0]
        assert "tell me more about that" in payload["user_prompt"]
        assert "Soulfire Necklace" in payload["user_prompt"]

    def test_timeout_falls_back_to_original(self, monkeypatch):
        chain = _mock_chain(side_effect=TimeoutError("slow"))
        monkeypatch.setattr(rag_mod, "_get_rewrite_chain", lambda api_key: chain)
        result = asyncio.run(
            rewrite_query(
                _HISTORY,
                "tell me more about that",
                openai_api_key="sk-test",
            )
        )
        assert result == "tell me more about that"

    def test_degenerate_output_falls_back_to_original(self, monkeypatch):
        chain = _mock_chain("Sure, here is the answer")
        monkeypatch.setattr(rag_mod, "_get_rewrite_chain", lambda api_key: chain)
        result = asyncio.run(
            rewrite_query(
                _HISTORY,
                "tell me more about that",
                openai_api_key="sk-test",
            )
        )
        assert result == "tell me more about that"

    def test_cache_hit_skips_second_call(self, monkeypatch):
        chain = _mock_chain("Soulfire Necklace details")
        monkeypatch.setattr(rag_mod, "_get_rewrite_chain", lambda api_key: chain)
        kwargs = dict(
            history_messages=_HISTORY,
            latest_message="tell me more about that",
            session_id="sess-1",
            openai_api_key="sk-test",
        )
        first = asyncio.run(rewrite_query(**kwargs))
        second = asyncio.run(rewrite_query(**kwargs))
        assert first == second == "Soulfire Necklace details"
        assert chain.ainvoke.call_count == 1


class TestGetRewriteChain:
    def setup_method(self):
        _clear_rewrite_cache()

    def test_builds_runnable_and_reuses_for_same_key(self):
        chain = rag_mod._get_rewrite_chain("sk-test")
        assert hasattr(chain, "ainvoke")
        assert rag_mod._get_rewrite_chain("sk-test") is chain

    def test_rebuilds_when_api_key_changes(self):
        first = rag_mod._get_rewrite_chain("sk-one")
        second = rag_mod._get_rewrite_chain("sk-two")
        assert first is not second
