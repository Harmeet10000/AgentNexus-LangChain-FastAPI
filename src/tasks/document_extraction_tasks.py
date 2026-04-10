import asyncio

import langextract

from app.connections import celery_app
from app.shared.rag.langextract.langextract_batch_processor import (
    LangExtractBatchContext,
    run_legal_extraction_batch,
)


@celery_app.task(bind=True, name="document_extraction.legal_batch")
def legal_document_extraction_batch_task(
    urls: list[str],
    prompt_description: str,
    examples: list[dict],  # Serialized ExampleData
) -> dict:
    """Celery entrypoint — runs in dedicated worker pool."""
    # Re-hydrate examples (in real code you'd use model_validate)
    lx_examples = [langextract.data.ExampleData.model_validate(ex) for ex in examples]

    ctx = LangExtractBatchContext()

    result = asyncio.run(run_legal_extraction_batch(urls, prompt_description, lx_examples, ctx))

    return {"processed": len(result), "details": [r.model_dump() for r in result]}
