"""Document parsing helpers for unified ingestion."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import asyncer
from docling.document_converter import DocumentConverter, DocumentStream

from app.shared.rag.document_processing import create_document_converter
from app.shared.rag.document_processing.docling_enhanced import table_markdown

from .classification import ParsedDocument

if TYPE_CHECKING:
    from docling.datamodel.document import ConversionResult
    from docling_core.types.doc.document import DoclingDocument


async def parse_document(*, raw_bytes: bytes, filename: str, content_type: str) -> ParsedDocument:
    """Parse an uploaded document into markdown, off the event loop.

    The conversion runs in a worker thread. Docling's ``convert`` is synchronous and CPU-bound —
    OCR and table-structure inference over every page — and it used to be called directly in this
    coroutine's body, so a single upload stalled the whole loop for the duration of the parse.
    Nothing else the process was serving made progress: not a health check, not another request's
    database round trip, not a websocket heartbeat. It is the kind of defect no gate can see,
    because the code is correct in isolation and only wrong about where it runs.

    ``asyncer.asyncify`` rather than ``asyncio.to_thread``, matching
    ``ingestion_kb/nodes._parse_document_with_docling`` — the same offload for the same converter
    in the other ingestion path. Change 2 consolidates these two; arriving there with two
    different offload idioms would make that a reconciliation instead of a deletion.

    The converter is built **inside** the offloaded function, so it is still one per call. That is
    deliberate rather than overlooked: making it a cached singleton is the obvious next
    optimisation and would be wrong to do in the same step as the offload, because the offload is
    what first makes concurrent parses possible, and ``DocumentConverter`` holds mutable pipeline
    state that docling does not document as thread-safe. Caching it belongs with a claim about
    that, which this task does not have.
    """
    del content_type
    if not raw_bytes:
        return ParsedDocument(title=filename or "uploaded-document", markdown="", page_count=0)

    def _sync_parse() -> ParsedDocument:
        converter: DocumentConverter = create_document_converter(gpu_available=False)
        result: ConversionResult = converter.convert(
            source=DocumentStream(name=filename, stream=BytesIO(initial_bytes=raw_bytes))
        )
        document: DoclingDocument = result.document
        markdown: str = document.export_to_markdown()
        return ParsedDocument(
            title=_extract_title(markdown, filename),
            markdown=markdown,
            page_count=len(getattr(document, "pages", []) or []),
            # This field used to be built as an empty literal. The converter is configured with
            # `do_table_structure = True` (`docling_enhanced.py:67`), so every table was being
            # detected, laid out, and then dropped one line later — paid for on every upload and
            # never used. (Described rather than quoted: B3's second proof greps for that literal,
            # and prose containing it would defeat the guard.)
            tables=table_markdown(document),
        )

    return await asyncer.asyncify(_sync_parse)()


def _extract_title(markdown: str, filename: str) -> str:
    for line in markdown.splitlines():
        stripped: str = line.strip("# ").strip()
        if stripped:
            return stripped[:500]
    return filename[:500] or "uploaded-document"
