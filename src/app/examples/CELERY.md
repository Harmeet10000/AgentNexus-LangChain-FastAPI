# Celery Usage

This project uses the Celery app defined in `src/app/connections/celery.py`.

The default setup already gives you:
- durable RabbitMQ queues
- DLQ routing
- late acknowledgements
- retry backoff with jitter
- task lifecycle logging
- Redis-backed idempotency helpers
- Redis-backed circuit breaker helpers

## Required Environment Variables

These settings are already supported by `src/app/config/settings.py`.

```env
RABBITMQ_URL=amqp://guest:guest@localhost:5672//
REDIS_URL=redis://localhost:6379

CELERY_DEFAULT_QUEUE=default
CELERY_DEFAULT_EXCHANGE=tasks
CELERY_DEFAULT_ROUTING_KEY=task.default

CELERY_DEAD_LETTER_EXCHANGE=tasks.dlx
CELERY_DEAD_LETTER_QUEUE=default.dlq
CELERY_DEAD_LETTER_ROUTING_KEY=task.default.dlq

CELERY_RETRY_MAX_RETRIES=5
CELERY_RETRY_BACKOFF_MAX=600
CELERY_DEFAULT_RETRY_DELAY=5

CELERY_TASK_SOFT_TIME_LIMIT=270
CELERY_TASK_TIME_LIMIT=300
CELERY_TASK_RESULT_EXPIRES=3600
CELERY_WORKER_MAX_TASKS_PER_CHILD=500

CELERY_IDEMPOTENCY_TTL_SECONDS=86400
CELERY_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CELERY_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60
```

## Queue Topology

The current worker config creates:
- main queue: `default`
- main exchange: `tasks`
- dead-letter exchange: `tasks.dlx`
- dead-letter queue: `default.dlq`

Normal flow:
1. FastAPI or another producer publishes to the main exchange.
2. Celery consumes from the main queue.
3. Transient failures are retried with backoff and jitter.
4. Messages that are rejected or dead-lettered land in the DLQ.

Important:
- The app is configured for at-least-once delivery, not exactly-once delivery.
- Because of that, idempotency is required for side-effecting tasks.

## Start The Worker

```bash
uv run celery -A app.connections.celery:celery_app worker --loglevel=info
```

Useful worker variants:

```bash
uv run celery -A app.connections.celery:celery_app worker --loglevel=info --concurrency=4
uv run celery -A app.connections.celery:celery_app inspect active
uv run celery -A app.connections.celery:celery_app inspect registered
uv run celery -A app.connections.celery:celery_app inspect stats
```

## What `ResilientTask` Adds

`ResilientTask` in `src/app/connections/celery.py` adds:
- `autoretry_for=(ConnectionError, TimeoutError, OSError)`
- exponential retry backoff
- retry jitter
- max retry limit from settings
- task lifecycle logs on publish, start, retry, failure, and finish
- Redis-backed helper methods for idempotency and circuit breaker use

This means you usually should not call `self.retry(...)` manually for transient infrastructure errors unless you need custom behavior.

## Basic Task

Use `ResilientTask` as the base task for normal background jobs.

```python
from app.connections import celery_app
from app.connections.celery import ResilientTask


@celery_app.task(
    name="tasks.add",
    bind=True,
    base=ResilientTask,
)
def add(self, x: int, y: int) -> int:
    return x + y
```

Use a basic task like this when:
- the task is simple
- duplicate execution is harmless
- there is no external side effect
- default retries are enough

## Idempotent Task

Use idempotency when a task can be retried or redelivered and must not repeat the side effect.

```python
from app.connections import celery_app
from app.connections.celery import ResilientTask


@celery_app.task(
    name="tasks.send_invoice_email",
    bind=True,
    base=ResilientTask,
)
def send_invoice_email(
    self,
    invoice_id: str,
    user_email: str,
    idempotency_key: str,
) -> dict[str, str]:
    acquired = self.acquire_idempotency_lock(
        idempotency_key,
        metadata={"invoice_id": invoice_id, "user_email": user_email},
    )
    if not acquired:
        return {"status": "duplicate-skipped", "invoice_id": invoice_id}

    try:
        # Put the real email provider call here.
        self.mark_idempotency_completed(
            idempotency_key,
            metadata={"invoice_id": invoice_id},
        )
        return {"status": "sent", "invoice_id": invoice_id}
    except ValueError:
        # Permanent failure: invalid payload, invalid recipient, etc.
        self.mark_idempotency_failed_permanently(
            idempotency_key,
            metadata={"invoice_id": invoice_id},
        )
        raise
    except Exception:
        # Transient failure: release lock so Celery retry can run again.
        self.release_idempotency_processing_lock(idempotency_key)
        raise
```

Use a business key for `idempotency_key`, not the Celery task id.

Good examples:
- `invoice:{invoice_id}:email`
- `payment:{payment_id}:capture`
- `user:{user_id}:welcome-email:v1`

Bad examples:
- Celery task id
- random UUID generated inside the task
- timestamp-only keys

## Retry Strategy

The current retry behavior is intended for transient failures:
- connection errors
- timeouts
- temporary network failures

Do retry:
- external API timeout
- temporary Redis outage
- RabbitMQ reconnect path
- rate-limited upstream that may recover

Do not retry:
- invalid payload
- missing required business entity
- bad user input
- permanent domain validation failure

For permanent failures, mark the idempotency key as permanently failed and raise.

## Circuit Breaker

Use the circuit breaker for flaky external dependencies so workers do not keep hammering an unhealthy service.

```python
from app.connections import celery_app
from app.connections.celery import ResilientTask


@celery_app.task(
    name="tasks.sync_customer_to_crm",
    bind=True,
    base=ResilientTask,
)
def sync_customer_to_crm(
    self,
    customer_id: str,
    idempotency_key: str,
) -> dict[str, str]:
    if not self.acquire_idempotency_lock(idempotency_key):
        return {"status": "duplicate-skipped", "customer_id": customer_id}

    try:
        def push_to_crm() -> dict[str, str]:
            # Replace with the real CRM client call.
            return {"status": "ok", "customer_id": customer_id}

        result = self.run_with_circuit_breaker("crm-api", push_to_crm)
        self.mark_idempotency_completed(idempotency_key)
        return result
    except Exception:
        self.release_idempotency_processing_lock(idempotency_key)
        raise
```

Use a stable dependency name for the breaker, for example:
- `crm-api`
- `payments-api`
- `email-provider`
- `search-indexer`

Do not use request-specific names like `crm-api:{customer_id}` because that defeats shared failure isolation.

## Trigger From FastAPI

```python
from fastapi import APIRouter

from tasks.example import process_document

router = APIRouter()


@router.post("/documents/{document_id}/process")
async def trigger_document_processing(document_id: str) -> dict[str, str]:
    task = process_document.delay(document_id=document_id)
    return {"task_id": task.id, "status": "queued"}
```

If you need explicit routing:

```python
task = process_document.apply_async(
    kwargs={"document_id": document_id},
    queue="default",
    routing_key="task.default",
)
```

## Fetch Task Status

```python
from app.connections.celery import celery_app


def get_task_status(task_id: str) -> dict[str, object]:
    result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "state": result.state,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else False,
        "result": result.result if result.ready() else None,
    }
```

## Current Helpers

`ResilientTask` in `src/app/connections/celery.py` exposes:
- `self.acquire_idempotency_lock(...)`
- `self.mark_idempotency_completed(...)`
- `self.mark_idempotency_failed_permanently(...)`
- `self.release_idempotency_processing_lock(...)`
- `self.run_with_circuit_breaker(...)`

The functional Redis helpers live in `src/app/shared/services/celery_reliability.py`.

If you need lower-level control outside a Celery task, you can call those functional helpers directly and pass the worker Redis client.

## Observability

The current setup emits logs for:
- task publish
- task start
- task retry
- task failure
- task completion

Recommended operational checks:

```bash
uv run celery -A app.connections.celery:celery_app inspect active
uv run celery -A app.connections.celery:celery_app inspect reserved
uv run celery -A app.connections.celery:celery_app inspect scheduled
uv run celery -A app.connections.celery:celery_app inspect stats
```

RabbitMQ management UI is also useful for:
- main queue depth
- DLQ depth
- consumer count
- message rates

## DLQ Handling

The guide is not complete without an operational rule for the DLQ.

Use the DLQ for:
- poison messages
- tasks that repeatedly fail after retry exhaustion
- payloads that need manual inspection

Recommended workflow:
1. inspect the payload and exception reason
2. fix the root cause
3. replay only safe tasks
4. never blindly replay non-idempotent tasks

## Important Caveats

- `task_acks_late=True` means a worker crash can cause a task to run again.
- That is why idempotency exists in this setup.
- `run_redis_call()` bridges sync Celery task methods with the async Redis client factory used by the app.
- Celery workers are separate processes from FastAPI, so they cannot use `get_redis(request)` directly.
- The result backend is `rpc://`, which is suitable for short-lived result retrieval but not a long-term audit store.
- If you need durable task history, store task outcomes in your own database.

## When To Use What

- Use plain `ResilientTask` for simple retryable work with no external side effect.
- Add idempotency for tasks that send emails, charge payments, write records, or call third-party APIs.
- Add circuit breaker when the task depends on a service that may become slow or unavailable.
- Use both idempotency and circuit breaker for expensive external operations.

## Suggested File Pattern

Keep real task modules small:

```python
from app.connections import celery_app
from app.connections.celery import ResilientTask


@celery_app.task(name="tasks.some_operation", bind=True, base=ResilientTask)
def some_operation(self, entity_id: str, idempotency_key: str) -> dict[str, str]:
    if not self.acquire_idempotency_lock(idempotency_key):
        return {"status": "duplicate-skipped", "entity_id": entity_id}

    try:
        result = self.run_with_circuit_breaker(
            "some-dependency",
            lambda: {"status": "ok", "entity_id": entity_id},
        )
        self.mark_idempotency_completed(idempotency_key)
        return result
    except ValueError:
        self.mark_idempotency_failed_permanently(idempotency_key)
        raise
    except Exception:
        self.release_idempotency_processing_lock(idempotency_key)
        raise
```

That pattern is the default for any side-effecting task in this repo.
