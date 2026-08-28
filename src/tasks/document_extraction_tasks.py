import asyncio
from typing import Any

import langextract

from app.connections.celery import CeleryTaskPayload, CeleryTaskRegistry, celery_app
from app.connections.celery_task_names import LEGAL_BATCH_EXTRACTION
from app.shared.rag.langextract.langextract_batch_processor import (
    LangExtractBatchContext,
    run_legal_extraction_batch,
)


class LegalBatchExtractionPayload(CeleryTaskPayload):
    """Typed payload for the batched legal-document extraction task."""

    urls: list[str]
    prompt_description: str
    examples: list[dict[str, Any]]


CeleryTaskRegistry.register(LEGAL_BATCH_EXTRACTION, LegalBatchExtractionPayload)


# `bind=True` was set here while the body takes no `self`, so Celery would have
# passed the task instance as `urls` and the batch would have iterated a Task.
# Nothing dispatches this name yet, which is why the mistake survived; registering
# the module is what makes it reachable, so it is corrected rather than registered
# broken.
@celery_app.task(name=LEGAL_BATCH_EXTRACTION)
def legal_document_extraction_batch_task(
    urls: list[str],
    prompt_description: str,
    examples: list[dict[str, Any]],  # Serialized ExampleData
) -> dict[str, Any]:
    """Celery entrypoint — runs in dedicated worker pool."""
    # Re-hydrate examples (in real code you'd use model_validate)
    lx_examples = [langextract.data.ExampleData.model_validate(ex) for ex in examples]

    ctx = LangExtractBatchContext()

    result = asyncio.run(run_legal_extraction_batch(urls, prompt_description, lx_examples, ctx))

    return {"processed": len(result), "details": [r.model_dump() for r in result]}
