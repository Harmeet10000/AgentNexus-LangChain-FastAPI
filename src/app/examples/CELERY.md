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
- work queues: `default` and `ingestion`
- main exchange: `tasks`
- dead-letter exchange: `tasks.dlx`
- dead-letter queue: `default.dlq`, shared by both work queues

Normal flow:
1. FastAPI or another producer publishes to the main exchange.
2. A worker consumes from the work queue its `-Q` names.
3. Transient failures are retried with backoff and jitter.
4. Messages that are rejected or dead-lettered land in the DLQ.

Two work queues rather than one, and they are consumed by two separate worker
processes. Document ingestion is minutes of model work per message; the default
queue carries sub-second billing and transactional-email tasks. A single shared
pool makes those wait behind ingestion whenever every slot is busy, and
`worker_prefetch_multiplier=1` does not prevent it — prefetch stops one worker
hoarding messages off the broker and says nothing about head-of-line blocking
once every slot is already occupied. Disjoint queues with disjoint consumers is
what removes the coupling.

Which names go where is not decided in `task_routes` by hand. The names that run
for minutes are listed in `INGESTION_TASK_NAMES` in
`src/app/connections/celery_task_names.py`, and the routing table is derived from
that list, so the two cannot disagree.

Important:
- The app is configured for at-least-once delivery, not exactly-once delivery.
- Because of that, idempotency is required for side-effecting tasks.
- Every dispatchable name is routed explicitly. There is no glob and no default
  fallthrough, so routing a name that is not in the table is a publish-time
  failure rather than a message on a queue nobody expected.

## Start The Worker

Both workers and the scheduler are defined once, in the `Makefile`, and the
README and compose services are pinned to those definitions by a unit test.
Start them from there rather than by hand:

```bash
make celery            # the default queue, higher concurrency
make celery-ingestion  # the ingestion queue, low concurrency
make celery-beat       # the scheduler; publishes only, consumes nothing
make celery-command    # print all three without running them
```

`-Q` is mandatory and every one of those worker commands carries it. A worker
started without it consumes **every** queue the application declares, dead-letter
queue included, and so re-runs precisely the messages that were parked there for
a human to look at — while reporting itself perfectly healthy.

Inspection commands talk to whichever broker the environment configures, so check
that first if the environment may point at a shared or managed one:

```bash
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

## ReliabilitySystem

`ReliabilitySystem` is a unified base class that wraps circuit breaker and idempotency checks. It delegates to the functional helpers in `celery_reliability.py`.

```python
from app.connections.celery import ResilientTask
from app.connections.celery_reliability import ReliabilitySystem

@celery_app.task(name="tasks.sync_to_crm", bind=True, base=ResilientTask)
def sync_to_crm(self, customer_id: str) -> dict[str, str]:
    system = ReliabilitySystem(
        self.get_redis_client(),
        circuit_breaker_name="crm-api",
        failure_threshold=3,
        recovery_timeout_seconds=60,
    )

    system.check_circuit_breaker()

    try:
        result = push_to_crm(customer_id)
        system.record_success()
        return result
    except Exception:
        system.record_failure()
        raise
```

Check idempotency status before executing:

```python
status = system.get_idempotency_status("customer:123:sync")
if status == "completed":
    return {"status": "already-processed"}
```

## IdempotencyManager

`idempotency_manager` is an async context manager that automates lock acquisition, completion marking, and failure handling.

```python
import asyncio
from app.connections.celery_reliability import idempotency_manager

async def process_payment(payment_id: str, redis_client) -> dict[str, str]:
    async with idempotency_manager(
        redis_client,
        f"payment:{payment_id}:capture",
        task_id="task-123",
        retryable_exceptions=(TimeoutError, ConnectionError),
    ):
        # Task logic here
        await charge_payment(payment_id)
        return {"status": "charged", "payment_id": payment_id}
```

On normal exit: record marked as `completed`.
On retryable exception: processing lock released (Celery retry can re-acquire).
On non-retryable exception: record marked as `failed_permanent`.

## RateLimiter

`RateLimiter` provides sliding-window rate limiting with configuration embedded in Redis keys.

```python
import asyncio
from app.connections.celery_reliability import RateLimiter

async def rate_limited_task(redis_client) -> dict[str, str]:
    limiter = RateLimiter(
        redis_client,
        scope="api:process-document",
        rate=10,
        period_seconds=60,
        burst=15,
    )

    result = await limiter.check_and_increment(
        forwarded_for="203.0.113.50, 70.41.3.18",
        direct_ip="70.41.3.18",
    )

    if not result.allowed:
        raise Exception(f"Rate limit exceeded. Retry after {result.reset_at}")

    return {"status": "ok", "remaining": result.remaining}
```

The Redis key format is `celery:ratelimit:{scope}:rate={rate}:period={period}:burst={burst}`, so rate limit state is self-documenting in Redis.

IP-based rate limiting with proxy trust:

```python
limiter = RateLimiter(
    redis_client,
    scope="api:user-endpoints",
    rate=100,
    period_seconds=60,
)
# Client IP extracted from X-Forwarded-For using FASTAPI_GUARD_TRUSTED_PROXY_DEPTH
result = await limiter.check_and_increment(
    forwarded_for=request.headers.get("x-forwarded-for"),
    direct_ip=request.client.host,
)
```

## Combined Usage Example

```python
import asyncio
from app.connections.celery import ResilientTask
from app.connections.celery_reliability import (
    RateLimiter,
    ReliabilitySystem,
    idempotency_manager,
)


@celery_app.task(name="tasks.process_document", bind=True, base=ResilientTask)
def process_document(self, doc_id: str, idempotency_key: str) -> dict[str, str]:
    redis = self.get_redis_client()

    system = ReliabilitySystem(
        redis,
        circuit_breaker_name="document-processor",
    )
    system.check_circuit_breaker()

    loop = asyncio.new_event_loop()

    async def _run():
        async with idempotency_manager(
            redis,
            idempotency_key,
            task_id=self.request.id,
            retryable_exceptions=(TimeoutError, ConnectionError),
        ):
            limiter = RateLimiter(
                redis,
                scope=f"process_doc:{doc_id}",
                rate=5,
                period_seconds=60,
            )
            rate_result = await limiter.check_and_increment(direct_ip="127.0.0.1")
            if not rate_result.allowed:
                raise Exception("Rate limit exceeded for document processing")

            return {"status": "processed", "doc_id": doc_id}

    try:
        result = loop.run_until_complete(_run())
        system.record_success()
        return result
    except Exception:
        system.record_failure()
        raise
    finally:
        loop.close()
```

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
