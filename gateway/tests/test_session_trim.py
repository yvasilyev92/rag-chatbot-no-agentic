"""Tests for session history token trimming."""
from unittest.mock import MagicMock, patch

import pytest

from app.models import Message
from app.session import SessionManager


@pytest.fixture
def session_manager():
    with patch.object(SessionManager, "__init__", lambda self: None):
        mgr = SessionManager()
        mgr.max_history_tokens = 100
        mgr.chars_per_token = 4
        return mgr


class TestEstimateTokens:
    def test_chars_per_token_division(self, session_manager):
        assert session_manager._estimate_tokens("abcd") == 1
        assert session_manager._estimate_tokens("abcdefgh") == 2


class TestGetSessionHistoryWithTokenLimit:
    def _messages(self, contents):
        return [Message(role="user" if i % 2 == 0 else "assistant", content=c) for i, c in enumerate(contents)]

    def test_returns_all_when_under_limit(self, session_manager):
        msgs = self._messages(["short", "reply"])
        with patch.object(session_manager, "get_session_messages", return_value=msgs):
            result = session_manager.get_session_history_with_token_limit("sid")
        assert len(result) == 2

    def test_trims_oldest_first(self, session_manager):
        msgs = self._messages(["A" * 200, "B" * 200, "C" * 40])
        with patch.object(session_manager, "get_session_messages", return_value=msgs):
            result = session_manager.get_session_history_with_token_limit("sid")
        assert len(result) < 3
        assert result[0].content == "B" * 200

    def test_respects_min_of_max_tokens_and_max_history(self, session_manager):
        msgs = self._messages(["A" * 200, "B" * 200])
        with patch.object(session_manager, "get_session_messages", return_value=msgs):
            result = session_manager.get_session_history_with_token_limit("sid", max_tokens=20)
        # effective limit is min(20, 100) = 20 tokens -> 80 chars
        total = sum(len(m.content) for m in result)
        assert total <= 80 + 200  # may keep one oversized message

    def test_zero_or_negative_returns_empty(self, session_manager):
        msgs = self._messages(["hello"])
        with patch.object(session_manager, "get_session_messages", return_value=msgs):
            assert session_manager.get_session_history_with_token_limit("sid", max_tokens=0) == []
            assert session_manager.get_session_history_with_token_limit("sid", max_tokens=-5) == []

    def test_single_oversized_message_retained(self, session_manager):
        msgs = self._messages(["X" * 800])
        with patch.object(session_manager, "get_session_messages", return_value=msgs):
            result = session_manager.get_session_history_with_token_limit("sid")
        assert len(result) == 1


class TestRefreshSessionTtl:
    def test_updates_metadata_item_only(self, session_manager):
        session_manager.ttl_hours = 24
        table = MagicMock()
        session_manager.table = table

        assert session_manager.refresh_session_ttl("sid-1") is True

        table.query.assert_not_called()
        table.update_item.assert_called_once()
        kwargs = table.update_item.call_args.kwargs
        assert kwargs["Key"] == {"session_id": "sid-1", "message_id": "0000_metadata"}
        assert "expires_at" in kwargs["UpdateExpression"]
