from __future__ import annotations

from typing import TYPE_CHECKING

import asyncer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from pydantic import BaseModel, Field

from app.utils import ValidationException

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Annotated


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
    elements: list[dict]  # Docling semantic elements for optional rich prompting
    page_count: int
    char_count: int


async def preprocess_legal_document(
    url: str,
    ctx: DoclingProcessingContext,
) -> CleanLegalDocument:
    """Async wrapper around Docling (CPU-heavy)."""
    if not url.lower().endswith(".pdf"):
        raise ValidationException("Only PDF URLs supported for legal preprocessing")

    # Run blocking Docling in thread pool
    def _sync_process() -> CleanLegalDocument:
        converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(
                    pipeline=SimplePipeline(),
                    do_table_structure=True,
                    do_ocr=False,  # Set True only if scanned docs; expensive
                )
            }
        )

        result = converter.convert(url)  # Docling handles remote URLs gracefully

        export = result.document.export_to_markdown()
        elements = [e.to_dict() for e in result.document.iterate_items()]

        return CleanLegalDocument(
            source_url=url,
            markdown=export,
            elements=elements,
            page_count=len(result.document.pages),
            char_count=len(export),
        )

    return await asyncer.asyncify(_sync_process)()
