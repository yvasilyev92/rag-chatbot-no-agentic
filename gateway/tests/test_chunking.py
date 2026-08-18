"""Tests for text and markdown chunking."""
from app.rag import (
    _build_preamble,
    _extract_h1_and_sections,
    _filename_to_doc_name,
    _split_markdown_records,
    chunk_markdown,
    chunk_text,
)


class TestFilenameToDocName:
    def test_strips_path_and_extension(self):
        assert _filename_to_doc_name("docs/equipment.md") == "equipment"
        assert _filename_to_doc_name("equipment.md") == "equipment"


class TestBuildPreamble:
    def test_all_fields(self):
        preamble = _build_preamble("equipment", "Equipment", "Weapons", "Stats")
        assert "[Document: equipment]" in preamble
        assert "[Category: Equipment]" in preamble
        assert "[Topic: Weapons]" in preamble
        assert "[Section: Stats]" in preamble
        assert preamble.endswith("\n\n")

    def test_empty_returns_empty_string(self):
        assert _build_preamble(None, None, None, None) == ""


class TestSplitMarkdownRecords:
    def test_parses_frontmatter(self, sample_markdown):
        records = _split_markdown_records(sample_markdown)
        assert len(records) == 2
        assert records[0]["metadata"]["document"] == "equipment"
        assert records[0]["metadata"]["topic"] == "Weapons"
        assert "Soulfire Necklace" in records[0]["body"]

    def test_body_only_record(self):
        text = "# Title\n\nSome content without frontmatter."
        records = _split_markdown_records(text)
        assert len(records) == 1
        assert records[0]["metadata"] == {}
        assert "Title" in records[0]["body"]

    def test_whitespace_only_returns_empty(self):
        assert _split_markdown_records("") == []
        assert _split_markdown_records("   \n\n  ") == []


class TestExtractH1AndSections:
    def test_splits_h2_sections(self):
        body = "# Soulfire Necklace\n\nIntro text.\n\n## Stats\n\n+50 fire."
        h1, sections = _extract_h1_and_sections(body)
        assert h1 == "Soulfire Necklace"
        assert len(sections) == 2
        assert sections[0][0] == "Soulfire Necklace"
        assert "Intro text" in sections[0][1]
        assert sections[1][0] == "Stats"

    def test_no_h2_single_section(self):
        body = "# Iron Helm\n\nBasic protection."
        _, sections = _extract_h1_and_sections(body)
        assert len(sections) == 1
        assert sections[0][0] == "Iron Helm"

    def test_empty_body(self):
        h1, sections = _extract_h1_and_sections("")
        assert h1 == ""
        assert sections == []


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert chunk_text("hello world", chunk_size=100) == ["hello world"]

    def test_forward_progress_with_large_overlap(self):
        text = "word " * 200
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=45)
        assert len(chunks) > 1
        combined_len = sum(len(c) for c in chunks)
        assert combined_len >= len(text.strip())

    def test_prefers_paragraph_boundary(self):
        text = "A" * 80 + "\n\n" + "B" * 80
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 2


class TestChunkMarkdown:
    def test_end_to_end(self, sample_markdown):
        chunks = chunk_markdown(sample_markdown.encode("utf-8"), "equipment.md")
        assert len(chunks) >= 2

        first = chunks[0]
        assert first["text"].startswith("[Document: equipment]")
        assert first["raw_text"]
        assert first["document_name"] == "equipment"
        assert first["category"] == "Equipment"
        assert first["topic"] == "Weapons"
        assert first["section_title"]
        assert first["chunk_index"] == 0

    def test_empty_input(self):
        assert chunk_markdown(b"", "empty.md") == []
