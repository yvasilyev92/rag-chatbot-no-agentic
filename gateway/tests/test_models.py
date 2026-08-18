"""Tests for Pydantic request/response models."""
import pytest
from pydantic import ValidationError

from app.models import ChatCompletionRequest, RagFilters


class TestRagFilters:
    def test_accepts_scalar_and_list(self):
        f1 = RagFilters(category="Equipment")
        f2 = RagFilters(topic=["Weapons", "Armor"])
        assert f1.category == "Equipment"
        assert f2.topic == ["Weapons", "Armor"]

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValidationError):
            RagFilters(category="Equipment", chunk_id="evil")


class TestChatCompletionRequest:
    def test_defaults(self):
        req = ChatCompletionRequest(model="meta-llama/Llama-3.1-8B-Instruct")
        assert req.stream is False
        assert req.messages is None
        assert req.rag_filters is None

    def test_accepts_rag_filters(self):
        req = ChatCompletionRequest(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": "hi"}],
            rag_filters={"category": "Equipment"},
        )
        assert req.rag_filters.category == "Equipment"
