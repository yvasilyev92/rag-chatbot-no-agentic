"""Tests for retrieval gating helpers."""
import pytest

from app.rag import _apply_min_score, is_chitchat


class TestIsChitchat:
    @pytest.mark.parametrize(
        "message",
        ["hi", "hello", "thanks", "thank you", "ok", "bye", "lol", "yes", "no"],
    )
    def test_phrase_list_matches(self, message):
        assert is_chitchat(message) is True

    def test_punctuation_stripped(self):
        assert is_chitchat("hi!") is True
        assert is_chitchat("ok.") is True

    def test_empty_input(self):
        assert is_chitchat("") is True
        assert is_chitchat("   ") is True

    def test_short_message_without_history(self):
        assert is_chitchat("more please") is True
        assert is_chitchat("ice build") is True

    def test_short_message_with_history_not_skipped(self):
        assert is_chitchat("more please", has_history=True) is False
        assert is_chitchat("what about ice", has_history=True) is False

    def test_chitchat_phrase_still_skipped_with_history(self):
        assert is_chitchat("thanks", has_history=True) is True

    def test_question_lead_rescues_short_message(self):
        assert is_chitchat("what types") is False
        assert is_chitchat("list pets") is False

    def test_question_mark_rescues_short_message(self):
        assert is_chitchat("ice?") is False

    def test_real_question_not_chitchat(self):
        assert is_chitchat("What are relics in the game?") is False


class TestApplyMinScore:
    def test_drops_below_floor(self):
        results = [
            {"content": "a", "score": 0.9},
            {"content": "b", "score": 0.34},
            {"content": "c", "score": 0.35},
        ]
        kept = _apply_min_score(results, 0.35)
        assert len(kept) == 2
        assert kept[0]["content"] == "a"
        assert kept[1]["content"] == "c"

    def test_empty_list(self):
        assert _apply_min_score([], 0.35) == []

    def test_keeps_all_above_floor(self):
        results = [{"content": "x", "score": 0.5}, {"content": "y", "score": 0.4}]
        assert len(_apply_min_score(results, 0.35)) == 2
