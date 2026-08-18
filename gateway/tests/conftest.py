"""Shared fixtures for gateway unit tests."""
import os

import pytest

# Gateway refuses to start without an API key; set a dummy for all tests.
os.environ.setdefault("VLLM_API_KEY", "test-api-key-for-unit-tests")

from app.config import Settings


@pytest.fixture
def settings():
    """Minimal Settings for token-budget tests."""
    return Settings(
        model_context_tokens=4096,
        max_history_tokens=3000,
        chars_per_token=4,
    )


@pytest.fixture
def sample_markdown():
    """Multi-record markdown with frontmatter, H1/H2, and a separator."""
    return """\
---
document: equipment
category: Equipment
topic: Weapons
---

# Soulfire Necklace

A powerful fire relic worn by heroes.

## Stats

+50 fire damage and +10% crit rate.

---

---
document: equipment
category: Equipment
topic: Armor
---

# Iron Helm

Basic head protection.

## Defense

+20 armor rating.
"""


@pytest.fixture
def fake_search_results():
    """Search-result dicts shaped like reranker / OpenSearch output."""
    return [
        {
            "content": "[Document: equipment] Soulfire Necklace stats",
            "raw_content": "Soulfire Necklace stats",
            "source_filename": "equipment.md",
            "document_id": "doc-1",
            "document_name": "equipment",
            "topic": "Weapons",
            "section_title": "Stats",
            "page_number": 1,
            "score": 0.92,
        },
        {
            "content": "[Document: equipment] Iron Helm defense",
            "raw_content": "Iron Helm defense",
            "source_filename": "equipment.md",
            "document_id": "doc-2",
            "document_name": "equipment",
            "topic": "Armor",
            "section_title": "Defense",
            "page_number": 1,
            "score": 0.71,
        },
        {
            "content": "Legacy chunk without markdown metadata",
            "source_filename": "legacy.pdf",
            "document_id": "doc-3",
            "page_number": 3,
            "score": 0.55,
        },
    ]
