"""Plain-text parsing: the conditional fast path (no Docling conversion)."""

from __future__ import annotations

import pytest

from app.features.documents.ingestion.chunking.classification import ParsedDocument
from app.features.documents.ingestion.parsing.parser import (
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
    monkeypatch.setattr(
        # the converter is imported into the parser's namespace — patch there
        "app.features.documents.ingestion.parsing.parser.create_document_converter",
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
