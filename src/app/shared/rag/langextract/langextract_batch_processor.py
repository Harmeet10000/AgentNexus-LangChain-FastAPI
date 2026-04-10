from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import langextract as lx
from pydantic import BaseModel

from app.utils import APIException, logger

from .docling_preprocessor import (
    preprocess_legal_document,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .docling_preprocessor import (
        CleanLegalDocument,
    )



class LangExtractBatchContext(BaseModel):
    """Orchestration context for batch extraction."""

    model_config = {"frozen": True}

    model_id: str = "gemini-2.5-flash"
    extraction_passes: int = 3
    max_workers: int = 12  # Tune based on your GPU/CPU quota
    vertex_batch_enabled: bool = False


class BatchExtractionResult(BaseModel):
    document_url: str
    extractions_count: int
    grounded_count: int
    status: str  # "success" | "partial" | "failed"


async def run_legal_extraction_batch(
    urls: Sequence[str],
    prompt_description: str,
    examples: list[lx.data.ExampleData],
    ctx: LangExtractBatchContext,
) -> list[BatchExtractionResult]:
    """Batch processor: Docling → LangExtract with proper grounding."""
    results: list[BatchExtractionResult] = []

    # Preprocess phase (parallel but bounded)
    preprocess_ctx = DoclingProcessingContext(output_dir=Path("/tmp/legal_parsed"))

    clean_docs: list[CleanLegalDocument] = []
    for url in urls:
        try:
            doc = await preprocess_legal_document(url, preprocess_ctx)
            clean_docs.append(doc)
            logger.info(f"Preprocessed {url} → {doc.char_count:,} chars")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Preprocessing failed for {url}", exc_info=True)
            results.append(
                BatchExtractionResult(
                    document_url=url, extractions_count=0, grounded_count=0, status="failed"
                )
            )

    if not clean_docs:
        return results

    # LangExtract phase
    try:
        language_model_params = (
            {"vertexai": True, "batch": {"enabled": ctx.vertex_batch_enabled}}
            if ctx.vertex_batch_enabled
            else None
        )

        raw_results = lx.extract(
            text_or_documents=[doc.markdown for doc in clean_docs],
            prompt_description=prompt_description,
            examples=examples,
            model_id=ctx.model_id,
            extraction_passes=ctx.extraction_passes,
            max_workers=ctx.max_workers,
            language_model_params=language_model_params,
        )

        # Normalize to list
        annotated_docs = raw_results if isinstance(raw_results, list) else [raw_results]

        for doc_url, ann_doc, clean_doc in zip(urls, annotated_docs, clean_docs):
            grounded = sum(1 for e in ann_doc.extractions if e.char_interval is not None)
            results.append(
                BatchExtractionResult(
                    document_url=doc_url,
                    extractions_count=len(ann_doc.extractions),
                    grounded_count=grounded,
                    status="success" if grounded > 0 else "partial",
                )
            )

    except Exception as e:
        logger.error("LangExtract batch failed", exc_info=True)
        raise APIException("Batch extraction failed") from e

    return results
