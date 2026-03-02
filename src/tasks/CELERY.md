# Celery Integration

## Overview

Celery is integrated with RabbitMQ for asynchronous task processing.

## Configuration

RabbitMQ settings are in `src/app/config/settings.py`:

```python
RABBITMQ_URL: str = Field(default="amqp://guest:guest@localhost:5672//")
RABBITMQ_DEFAULT_USER: str = Field(default="guest")
RABBITMQ_DEFAULT_PASS: str = Field(default="guest")
```

## Running Celery Worker

### Local Development

```bash
# Start RabbitMQ (if not using Docker)
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:4-management-alpine

# Run Celery worker
uv run celery -A celery_config worker --loglevel=info
```

### With Docker Compose

```bash
# Start all services including RabbitMQ and Celery worker
docker-compose up -d

# View Celery worker logs
docker-compose logs -f celery-worker

# Scale workers
docker-compose up -d --scale celery-worker=3
```

## Creating Tasks

Create tasks in `src/app/tasks/`:

```python
from app.connections import celery_app
from app.utils.logger import logger

@celery_app.task(name="tasks.my_task")
def my_task(arg1: str, arg2: int) -> dict:
    logger.info("Task started", arg1=arg1, arg2=arg2)
    # Your task logic here
    return {"status": "completed"}
```

## Using Tasks in FastAPI

```python
from fastapi import APIRouter
from app.tasks.example import process_document

router = APIRouter()

@router.post("/process")
async def trigger_task(document_id: str):
    # Async task execution
    task = process_document.delay(document_id)
    return {"task_id": task.id, "status": "processing"}

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    from app.connections import celery_app
    task = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.state,
        "result": task.result if task.ready() else None
    }
```

## RabbitMQ Management UI

Access at: http://localhost:15672
- Username: guest (or RABBITMQ_DEFAULT_USER)
- Password: guest (or RABBITMQ_DEFAULT_PASS)

## Monitoring

```bash
# Check active workers
uv run celery -A celery_config inspect active

# Check registered tasks
uv run celery -A celery_config inspect registered

# Check worker stats
uv run celery -A celery_config inspect stats
```
