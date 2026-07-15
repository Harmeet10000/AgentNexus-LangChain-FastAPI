"""Document parsing helpers for unified ingestion."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

from docling.document_converter import DocumentConverter, DocumentStream

from app.shared.rag.document_processing import create_document_converter

from .classification import ParsedDocument

if TYPE_CHECKING:
    from docling.datamodel.document import ConversionResult
    from docling_core.types.doc.document import DoclingDocument


async def parse_document(*, raw_bytes: bytes, filename: str, content_type: str) -> ParsedDocument:
    del content_type
    if not raw_bytes:
        return ParsedDocument(title=filename or "uploaded-document", markdown="", page_count=0)

    converter: DocumentConverter = create_document_converter(gpu_available=False)
    result: ConversionResult = converter.convert(
        DocumentStream(name=filename, stream=BytesIO(raw_bytes))
    )
    document: DoclingDocument = result.document
    markdown = document.export_to_markdown()
    return ParsedDocument(
        title=_extract_title(markdown, filename),
        markdown=markdown,
        page_count=len(getattr(document, "pages", []) or []),
        tables=[],
    )


def _extract_title(markdown: str, filename: str) -> str:
    for line in markdown.splitlines():
        stripped: str = line.strip("# ").strip()
        if stripped:
            return stripped[:500]
    return filename[:500] or "uploaded-document"
