"""Example Celery tasks."""

from app.connections import celery_app
from app.utils.logger import logger


@celery_app.task(name="tasks.add")
def add(x: int, y: int) -> int:
    """Example task: Add two numbers."""
    result = x + y
    logger.info("Task executed", task="add", x=x, y=y, result=result)
    return result


@celery_app.task(name="tasks.process_document", bind=True)
def process_document(self, document_id: str) -> dict[str, str]:
    """Example task: Process a document asynchronously."""
    logger.info("Processing document", task_id=self.request.id, document_id=document_id)
    # Add your document processing logic here
    return {"status": "completed", "document_id": document_id}
