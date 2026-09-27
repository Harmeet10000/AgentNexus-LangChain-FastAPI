from __future__ import annotations

from typing import TYPE_CHECKING

import asyncer
from pydantic import BaseModel, Field
from returns.result import Failure, Success

from app.shared.rag.docling import create_document_converter
from app.shared.rag.errors import RagValidationError

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Annotated

    from app.shared.rag.errors import RagResult


class DoclingProcessingContext(BaseModel):
    """Narrow context for document preprocessing."""

    model_config = {"frozen": True}

    output_dir: Annotated[Path, Field(description="Temporary storage for parsed artifacts")]
    enable_tables: bool = True
    enable_figures: bool = False  # Legal docs rarely need figures


class CleanLegalDocument(BaseModel):
    """Structured output from preprocessing."""

    model_config = {"frozen": True}

    source_url: str
    markdown: str
    elements: list[dict[str, object]]  # Docling semantic elements for optional rich prompting
    page_count: int
    char_count: int


async def preprocess_legal_document(
    url: str,
    _ctx: DoclingProcessingContext,
) -> RagResult[CleanLegalDocument]:
    """Async wrapper around Docling (CPU-heavy)."""
    if not url.lower().endswith(".pdf"):
        msg = "Only PDF URLs supported for legal preprocessing"
        return Failure(
            RagValidationError(
                message=msg,
                source="docling_preprocessor",
                details={"url": url, "operation": "preprocess_legal_document"},
            )
        )

    # Run blocking Docling in thread pool
    def _sync_process() -> CleanLegalDocument:
        converter = create_document_converter(gpu_available=False)

        result = converter.convert(url)  # Docling handles remote URLs gracefully

        export = result.document.export_to_markdown()
        elements = [
            item.model_dump(mode="json") for item, _level in result.document.iterate_items()
        ]

        return CleanLegalDocument(
            source_url=url,
            markdown=export,
            elements=elements,
            page_count=len(result.document.pages),
            char_count=len(export),
        )

    return Success(await asyncer.asyncify(_sync_process)())
