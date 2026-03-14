"""Example Celery tasks."""

from app.connections import celery_app
from app.connections.celery import ResilientTask
from app.utils import logger


@celery_app.task(name="tasks.add", base=ResilientTask)
def add(x: int, y: int) -> int:
    """Example task: Add two numbers."""
    result = x + y
    logger.info("Task executed", task="add", x=x, y=y, result=result)
    return result


@celery_app.task(
    name="tasks.process_document",
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
