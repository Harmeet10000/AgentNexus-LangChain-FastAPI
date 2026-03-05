"""Example router demonstrating Celery task usage."""

from fastapi import APIRouter, HTTPException

from app.tasks.example import add, process_document
from app.utils import logger

router = APIRouter(prefix="/api/v1/tasks", tags=["Tasks"])


@router.post("/add")
async def trigger_add_task(x: int, y: int) -> dict[str, str]:
    """Trigger async addition task."""
    try:
        task = add.delay(x, y)
        logger.info("Task queued", task_id=task.id, task_name="add")
        return {"task_id": task.id, "status": "queued"}
    except Exception as e:
        logger.error("Failed to queue task", error=str(e))
        raise HTTPException(status_code=503, detail="Task queue unavailable")


@router.post("/process")
async def trigger_process_task(document_id: str) -> dict[str, str]:
    """Trigger async document processing task."""
    try:
        task = process_document.delay(document_id)
        logger.info("Task queued", task_id=task.id, task_name="process_document")
        return {"task_id": task.id, "status": "queued"}
    except Exception as e:
        logger.error("Failed to queue task", error=str(e))
        raise HTTPException(status_code=503, detail="Task queue unavailable")


@router.get("/status/{task_id}")
async def get_task_status(task_id: str) -> dict:
    """Get task status and result."""
    from app.connections import celery_app

    try:
        task = celery_app.AsyncResult(task_id)
        return {
            "task_id": task_id,
            "status": task.state,
            "result": task.result if task.ready() else None,
            "info": task.info if task.state == "PROGRESS" else None,
        }
    except Exception as e:
        logger.error("Failed to get task status", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve task status")
