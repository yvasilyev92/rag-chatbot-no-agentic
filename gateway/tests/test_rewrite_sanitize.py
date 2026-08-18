"""Tests for query-rewrite sanitization helpers and the OpenAI rewrite call."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx

from app.guard import OPENAI_CHAT_URL
from app.rag import (
    QUERY_REWRITE_MODEL,
    _build_rewrite_user_prompt,
    _clean_rewrite_output,
    _query_rewrite_cache,
    _query_rewrite_cache_lock,
    rewrite_query,
)


def _clear_rewrite_cache():
    with _query_rewrite_cache_lock:
        _query_rewrite_cache.clear()


def _mock_response(content: str, status_code: int = 200):
    response = MagicMock()
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err", request=MagicMock(), response=response
        )
    response.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return response


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

    def test_empty_history_skips_openai(self):
        client = AsyncMock()
        result = asyncio.run(
            rewrite_query(
                client,
                [],
                "What are relics?",
                openai_api_key="sk-test",
            )
        )
        assert result == "What are relics?"
        client.post.assert_not_called()

    def test_missing_key_skips_openai(self):
        client = AsyncMock()
        result = asyncio.run(
            rewrite_query(
                client,
                _HISTORY,
                "tell me more about that",
                openai_api_key="",
            )
        )
        assert result == "tell me more about that"
        client.post.assert_not_called()

    def test_rewrites_from_openai(self):
        client = AsyncMock()
        client.post = AsyncMock(
            return_value=_mock_response("Soulfire Necklace details")
        )
        result = asyncio.run(
            rewrite_query(
                client,
                _HISTORY,
                "tell me more about that",
                openai_api_key="sk-test",
            )
        )
        assert result == "Soulfire Necklace details"
        client.post.assert_called_once()
        args, kwargs = client.post.call_args
        assert args[0] == OPENAI_CHAT_URL
        assert kwargs["json"]["model"] == QUERY_REWRITE_MODEL

    def test_timeout_falls_back_to_original(self):
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.TimeoutException("slow"))
        result = asyncio.run(
            rewrite_query(
                client,
                _HISTORY,
                "tell me more about that",
                openai_api_key="sk-test",
            )
        )
        assert result == "tell me more about that"

    def test_degenerate_output_falls_back_to_original(self):
        client = AsyncMock()
        client.post = AsyncMock(
            return_value=_mock_response("Sure, here is the answer")
        )
        result = asyncio.run(
            rewrite_query(
                client,
                _HISTORY,
                "tell me more about that",
                openai_api_key="sk-test",
            )
        )
        assert result == "tell me more about that"

    def test_cache_hit_skips_second_call(self):
        client = AsyncMock()
        client.post = AsyncMock(
            return_value=_mock_response("Soulfire Necklace details")
        )
        kwargs = dict(
            history_messages=_HISTORY,
            latest_message="tell me more about that",
            session_id="sess-1",
            openai_api_key="sk-test",
        )
        first = asyncio.run(rewrite_query(client, **kwargs))
        second = asyncio.run(rewrite_query(client, **kwargs))
        assert first == second == "Soulfire Necklace details"
        assert client.post.call_count == 1
