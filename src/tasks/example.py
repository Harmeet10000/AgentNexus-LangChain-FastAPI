"""Example Celery tasks."""

from app.connections.celery import CeleryTaskPayload, CeleryTaskRegistry, ResilientTask, celery_app
from app.connections.celery_task_names import EXAMPLE_ADD, EXAMPLE_PROCESS_DOCUMENT
from app.utils import logger


class AddPayload(CeleryTaskPayload):
    """Typed payload for the arithmetic example task."""

    x: int
    y: int


class ProcessDocumentPayload(CeleryTaskPayload):
    """Typed payload for the document-processing example task."""

    document_id: str


CeleryTaskRegistry.register(EXAMPLE_ADD, AddPayload)
CeleryTaskRegistry.register(EXAMPLE_PROCESS_DOCUMENT, ProcessDocumentPayload)


@celery_app.task(name=EXAMPLE_ADD, base=ResilientTask)
def add(x: int, y: int) -> int:
    """Example task: Add two numbers."""
    result = x + y
    logger.info("Task executed", task="add", x=x, y=y, result=result)
    return result


@celery_app.task(
    name=EXAMPLE_PROCESS_DOCUMENT,
    bind=True,
    base=ResilientTask,
)
def process_document(self, document_id: str) -> dict[str, str]:
    """Example task: Process a document with resilient defaults."""
    logger.info("Processing document", task_id=self.request.id, document_id=document_id)
    # For real tasks:
    # 1. Acquire an idempotency key before external side effects.
    # 2. Call external dependencies behind the shared circuit breaker.
    # 3. Let ResilientTask handle transient failure retries with backoff+jitter.
    return {"status": "completed", "document_id": document_id}


# @celery_app.task(bind=True, base=ResilientTask)
# def charge_customer(self, payment_id: str, idempotency_key: str) -> None:
#     if idempotency_store.already_done(idempotency_key):
#         return

#     # do side effect once
#     payment_gateway.charge(payment_id, idempotency_key=idempotency_key)

#     idempotency_store.mark_done(idempotency_key)
