"""Plain-text parsing: the conditional fast path (no Docling conversion)."""

from __future__ import annotations

import pytest

from app.features.documents.application.ingestion.chunking.classification import ParsedDocument
from app.features.documents.application.ingestion.parsing import parser as parser_module
from app.features.documents.application.ingestion.parsing.parser import (
    DocumentExtractionBudgetError,
    parse_document,
)


async def test_plain_text_parses_without_docling() -> None:
    raw = b"# Note\n\nSome plain content.\n"
    parsed = await parse_document(
        raw_bytes=raw,
        filename="note.txt",
        content_type="text/plain",
    )
    assert isinstance(parsed, ParsedDocument)
    assert parsed.markdown == raw.decode()
    assert parsed.page_count == 1
    assert parsed.ocr_confidence is None
    assert parsed.ocr_source_coverage_ratio is None


async def test_form_feeds_count_as_pages() -> None:
    raw = b"page one\x0cpage two\x0cpage three"
    parsed = await parse_document(raw_bytes=raw, filename="multi.txt", content_type="text/plain")
    assert parsed.page_count == 3


async def test_markdown_extension_takes_the_fast_path() -> None:
    parsed = await parse_document(raw_bytes=b"# Title\n", filename="doc.md", content_type="")
    assert parsed.markdown == "# Title\n"


async def test_plain_text_page_budget_is_enforced() -> None:
    raw = b"\x0c" * 300
    with pytest.raises(DocumentExtractionBudgetError):
        await parse_document(raw_bytes=raw, filename="huge.txt", content_type="text/plain")


async def test_plain_text_output_budget_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    big = b"word " * 600_000  # ~2.9 MB of text
    with pytest.raises(DocumentExtractionBudgetError):
        await parse_document(
            raw_bytes=big,
            filename="big.txt",
            content_type="text/plain",
            max_output_bytes=1000,
        )


async def test_pdf_content_type_still_routes_to_docling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A text payload claiming to be a PDF is NOT the fast path's concern."""
    # The converter is cached behind a process-wide singleton; reset so this test's
    # throwing stub is the thing that would build it, not a leftover from another file.
    parser_module._reset_shared_converter_for_tests()
    monkeypatch.setattr(
        # the converter is imported into the parser's namespace — patch there
        "app.features.documents.application.ingestion.parsing.parser.create_document_converter",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Docling must not run for plain text")
        ),
    )
    # content_type application/pdf + .pdf filename → Docling path; the stub
    # converter raises, proving the route was taken.
    with pytest.raises(AssertionError, match="Docling must not run"):
        await parse_document(
            raw_bytes=b"text in a pdf costume",
            filename="costume.pdf",
            content_type="application/pdf",
        )


async def test_windows_1252_characters_survive_the_decode() -> None:
    """A cp1252 upload decodes losslessly — no silent replacement characters.

    Greptile wave-3 finding: ``errors="replace"`` rewrote ``\u00a3``/``\u00a7``
    into U+FFFD before any downstream check could see the loss, so accepted
    legal evidence changed silently. Latin-1 maps every byte, so the decode is
    total; cp1252 restores the legacy Windows punctuation bytes first.
    """
    raw = "Termination fee: \u00a3500 per \u00a712 breach.\n".encode("cp1252")
    parsed = await parse_document(raw_bytes=raw, filename="fees.txt", content_type="text/plain")
    assert "\ufffd" not in parsed.markdown
    assert "\u00a3500" in parsed.markdown
    assert "\u00a712" in parsed.markdown


async def test_valid_utf8_upload_still_decodes_exactly() -> None:
    raw = "Clause \u00a7 4.2 — \u0928\u093f\u0930\u094d\u0926\u0928\u094d\u0927.\n".encode("utf-8")
    parsed = await parse_document(raw_bytes=raw, filename="unicode.txt", content_type="text/plain")
    assert parsed.markdown == raw.decode("utf-8")


async def test_plain_text_uploads_carry_a_walkable_structural_tree() -> None:
    """The saved tree is non-empty and matches the navigator's expectations.

    Greptile wave-3 finding: the plain-text path left ``structural_tree``
    empty; ``fetch_structural_trees`` excludes empty trees, so
    ``navigate_structure`` and section hydration had no structure for these
    documents.
    """
    raw = (
        b"# Termination\n\nThe agreement ends on ninety days notice.\n\n"
        b"# Governing Law\n\nThe laws of India apply.\n"
    )
    parsed = await parse_document(raw_bytes=raw, filename="doc.md", content_type="text/markdown")
    tree = parsed.structural_tree
    assert tree, "plain text must persist a tree the navigator can walk"
    assert tree["name"]
    body = tree["body"]
    assert isinstance(body, dict)
    assert body["children"]
    labels = [child["label"] for child in body["children"]]
    assert labels == ["section_header", "section_header"], labels
    first = body["children"][0]
    assert first["text"] == "Termination"
    assert first["level"] == 1
    assert first["children"], "the section's paragraphs nest under their heading"
    assert first["children"][0]["label"] == "text"
    assert "ninety days notice" in first["children"][0]["text"]


async def test_headless_plain_text_gets_a_flat_paragraph_tree() -> None:
    raw = b"First paragraph of the contract.\n\nSecond paragraph follows.\n"
    parsed = await parse_document(raw_bytes=raw, filename="plain.txt", content_type="text/plain")
    body = parsed.structural_tree["body"]
    labels = [child["label"] for child in body["children"]]
    assert labels == ["text", "text"]
    assert body["children"][0]["text"] == "First paragraph of the contract."


async def test_the_plain_text_tree_drives_the_structural_navigator() -> None:
    """The built tree resolves a query to its section via navigate_tree."""
    from app.retrieval.structural_navigator import navigate_tree

    raw = (
        b"# Termination\n\nThe agreement ends on ninety days notice.\n\n"
        b"# Governing Law\n\nThe laws of India apply.\n"
    )
    parsed = await parse_document(raw_bytes=raw, filename="doc.md", content_type="text/markdown")
    paths = navigate_tree(parsed.structural_tree, "governing law")
    assert paths, "the tree must resolve matching text to a heading path"
    assert any("Governing Law" in str(path) for path in paths)
