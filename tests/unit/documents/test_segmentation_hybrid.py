"""Unit tests for Band D item 3 — structure-aware chunking without silent loss.

The live path split legal documents on a blank-line pattern and truncated to a
fixed prefix (`blocks[:200]`), discarding most of a long contract with no
warning. These tests pin the replacement: chunk boundaries come from the
Docling HybridChunker over a lifted document structure, every chunk carries its
heading path, clause sections that fit the budget are never split mid-clause,
and a long document is covered in full.
"""

from __future__ import annotations

from app.features.documents.classification import (
    ClassifiedDocument,
    ParsedDocument,
    classify_document,
    segment_chunks,
)


def _parsed(markdown: str, title: str = "Agreement") -> ParsedDocument:
    return ParsedDocument(title=title, markdown=markdown, page_count=1)


def _classified(kind: str = "legal_contract") -> ClassifiedDocument:
    return ClassifiedDocument(document_kind=kind, parties=["Acme", "Beta"])


async def test_legal_chunks_carry_their_heading_path() -> None:
    markdown = (
        "# Master Agreement\n\n"
        "## Indemnity\n\n"
        "The Supplier shall indemnify the Customer against third-party claims.\n\n"
        "## Termination\n\n"
        "Either party may terminate with thirty days notice.\n"
    )
    chunks, _warnings = await segment_chunks(
        parsed=_parsed(markdown), classified=_classified()
    )
    assert chunks, "a two-section document must produce chunks"
    paths = [" / ".join(chunk.metadata_["heading_path"]) for chunk in chunks]
    assert any("Indemnity" in path for path in paths)
    assert any("Termination" in path for path in paths)
    assert all(chunk.metadata_["tokenizer"] for chunk in chunks)


async def test_long_document_is_covered_in_full_not_truncated() -> None:
    sections = "".join(
        f"## Section {index}\n\nBody of section {index} marker-{index}.\n\n"
        for index in range(250)
    )
    chunks, _warnings = await segment_chunks(
        parsed=_parsed(f"# Contract\n\n{sections}"), classified=_classified()
    )
    covered = "\n".join(chunk.content for chunk in chunks)
    missing = [index for index in range(250) if f"marker-{index}" not in covered]
    assert missing == [], f"sections lost to truncation: {missing[:5]}"


async def test_peer_sections_merge_within_the_bound() -> None:
    markdown = "# Doc\n\n" + "".join(f"Tiny peer sentence {index}.\n\n" for index in range(6))
    chunks, _warnings = await segment_chunks(
        parsed=_parsed(markdown), classified=_classified()
    )
    assert 0 < len(chunks) < 6


async def test_clause_fitting_the_bound_is_not_split_mid_clause() -> None:
    clause = (
        "Section 12. Indemnity. The Supplier shall indemnify and hold harmless "
        "the Customer against all losses arising from breach of this agreement."
    )
    markdown = f"# Contract\n\n{clause}\n\nUnrelated closing paragraph here.\n"
    chunks, _warnings = await segment_chunks(
        parsed=_parsed(markdown), classified=_classified()
    )
    holders = [chunk for chunk in chunks if "hold harmless" in chunk.content]
    assert len(holders) == 1
    assert "Section 12" in holders[0].content
    assert holders[0].clause_type == "indemnity"


async def test_degenerate_parse_is_reported_not_silent() -> None:
    chunks, warnings = await segment_chunks(
        parsed=_parsed(""), classified=_classified()
    )
    assert chunks == []
    assert [warning.code for warning in warnings] == ["DEGENERATE_PARSE"]


async def test_generic_documents_use_the_same_structure_aware_path() -> None:
    markdown = "# Notes\n\n## First\n\nSome general prose here.\n\n## Second\n\nMore prose.\n"
    chunks, _warnings = await segment_chunks(
        parsed=_parsed(markdown), classified=_classified(kind="generic")
    )
    assert chunks
    assert all(chunk.custom_metadata["source"] == "hybrid" for chunk in chunks)
    assert all(chunk.chunk_kind == "generic" for chunk in chunks)


def test_classifier_still_routes_legal_documents() -> None:
    markdown = "This agreement between parties covers termination and governing law."
    assert (
        classify_document(markdown=markdown, filename="msa.pdf").document_kind
        == "legal_contract"
    )
