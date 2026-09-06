# Celery Task Guidelines

This project has one Celery application: `app.connections.celery:celery_app`.

FastAPI publishes work; Celery workers execute it; Celery Beat publishes scheduled work. Do not register tasks from `src/app/server.py` or `src/app/main.py`. Worker discovery is explicit in `create_celery_app()` and typed dispatch can lazily import a declaring module through `TASK_DECLARING_MODULES`.

## Start Commands

Use the repository commands so worker and deployment configuration remain aligned:

```bash
make celery              # default queue
make celery-ingestion    # ingestion queue
make celery-beat         # scheduler; publishes, never consumes
make celery-command      # print the configured commands
```

The work queues are:

- `default`: short billing, credit, email, and general work.
- `ingestion`: long document/model-processing work.

Every dispatchable task name has an explicit route. Add long-running names to `INGESTION_TASK_NAMES` in `src/app/connections/celery_task_names.py`; all other names use the default route. Every worker must specify `-Q`. A worker without `-Q` also consumes the dead-letter queue.

## Declare A Task

Define the wire name once in `src/app/connections/celery_task_names.py` and import that constant in both producers and consumers. Keep the payload model, registry entry, task decorator, declaring-module mapping, and Celery `include` list synchronized.

```python
from app.connections.celery import (
    CeleryTaskPayload,
    CeleryTaskRegistry,
    ResilientTask,
    celery_app,
)
from app.connections.celery_task_names import TASK_NAME


class EntityTaskPayload(CeleryTaskPayload):
    entity_id: str
    idempotency_key: str


CeleryTaskRegistry.register(TASK_NAME, EntityTaskPayload)


@celery_app.task(name=TASK_NAME, bind=True, base=ResilientTask)
def process_entity(
    self: ResilientTask,
    *,
    entity_id: str,
    idempotency_key: str,
) -> dict[str, str]:
    ...
```

Use `CeleryTaskRegistry.typed_send(...)` for validated dispatch:

```python
CeleryTaskRegistry.typed_send(
    TASK_NAME,
    {"entity_id": entity_id, "idempotency_key": f"entity:{entity_id}:process"},
    queue="default",
)
```

Do not duplicate task-name strings, depend on `tasks/__init__.py` import side effects, or register payload models in unrelated modules.

## ResilientTask

Use `ResilientTask` for normal background jobs. It provides configured retries for `ConnectionError`, `TimeoutError`, and `OSError`, exponential backoff with jitter, retry limits, lifecycle logs/metrics, idempotency helpers, and circuit-breaker helpers.

Do not call `self.retry()` for ordinary transient infrastructure failures. Let the base policy handle them; use manual retry only for behavior the base cannot express.

## Result Handling

Services and repositories use typed `Result` values for expected outcomes. A `Failure` is data, not an exception. Inspect Results with `isinstance`, never `match` on `Success`/`Failure`.

```python
from returns.result import Failure


result = await service.process(entity_id)
if isinstance(result, Failure):
    error = result.failure()
    logger.bind(
        task=TASK_NAME,
        entity_id=entity_id,
        error=error.message,
        details=error.details,
    ).error("Entity processing failed")
    raise RuntimeError(error.message)

value = result.unwrap()
return {"status": "completed", "entity_id": value.entity_id}
```

Use feature-owned typed error unions. Do not raise a `FeatureError` directly and do not turn expected Result failures into raw `HTTPException` values. At the Celery boundary, raise an appropriate exception only when the task must retry or be marked failed; unexpected exceptions should reach Celery so `ResilientTask` and failure signals can observe them.

## Idempotency

Every side-effecting task needs a stable business idempotency key. Never use the Celery task ID or a random value generated inside the task.

```python
@celery_app.task(name=TASK_NAME, bind=True, base=ResilientTask)
def send_side_effect(
    self: ResilientTask,
    *,
    entity_id: str,
    idempotency_key: str,
) -> dict[str, str]:
    if not self.acquire_idempotency_lock(
        idempotency_key,
        metadata={"entity_id": entity_id},
    ):
        return {"status": "duplicate-skipped", "entity_id": entity_id}

    try:
        deliver_side_effect(entity_id)
        self.mark_idempotency_completed(
            idempotency_key,
            metadata={"entity_id": entity_id},
        )
        return {"status": "completed", "entity_id": entity_id}
    except ValueError as exc:
        exc.add_note(f"task={TASK_NAME}, entity_id={entity_id}")
        self.mark_idempotency_failed_permanently(
            idempotency_key,
            metadata={"entity_id": entity_id},
        )
        raise
    except Exception as exc:
        exc.add_note(f"task={TASK_NAME}, entity_id={entity_id}")
        self.release_idempotency_processing_lock(idempotency_key)
        raise
```

Use `idempotency_manager(...)` when an async context manager better expresses the lock lifecycle. Retryable failures release the processing lock; permanent failures record a terminal state.

## Circuit Breakers

Use one stable dependency name across all tasks calling the same dependency:

```python
@celery_app.task(name=TASK_NAME, bind=True, base=ResilientTask)
def call_provider(self: ResilientTask, *, entity_id: str) -> dict[str, str]:
    return self.run_with_circuit_breaker(
        "payments-api",
        lambda: provider_call(entity_id),
    )
```

Use names such as `payments-api`, `email-provider`, or `search-indexer`. Never include a user or entity ID; that defeats shared failure isolation.

## Logging

Use the project logger:

```python
from app.utils import logger

log = logger.bind(task=TASK_NAME, entity_id=entity_id)
log.info("Task started")
log.bind(error=str(exc)).exception("Task failed")
```

Rules:

- `.exception(...)` is for an active exception and records the traceback.
- `.error(...)` is for a known failure without a traceback requirement.
- `.warning(...)` is for degraded-but-continuing behavior.
- Use `exc.add_note(...)` when an exception crosses a task boundary with useful context.
- Keep messages static; put dynamic data in bound fields.
- Do not use stdlib logging or `from loguru import logger`.

## Async Work

Celery task bodies are synchronous by default. Run async services explicitly:

```python
@celery_app.task(name=TASK_NAME, bind=True, base=ResilientTask)
def run_async_task(self: ResilientTask, *, entity_id: str) -> dict[str, str]:
    return asyncio.run(process_async(entity_id))
```

Keep transaction ownership inside the service/repository layer. Commit or roll back in one place, and dispose independently created database engines in `finally` blocks.

## Operational Rules

- Keep task modules small: declaration, payload, registry entry, and thin orchestration.
- Put domain logic in feature services, not in the Celery decorator function.
- Keep task names in `celery_task_names.py`.
- Keep task modules in the Celery app `include` list.
- Treat delivery as at-least-once; design side effects for duplicate execution.
- Use `rpc://` only for short-lived result retrieval, not audit history.
- Inspect workers with:

```bash
uv run celery -A app.connections.celery:celery_app inspect active
uv run celery -A app.connections.celery:celery_app inspect registered
uv run celery -A app.connections.celery:celery_app inspect stats
```

Before adding a task, update the task-name constant, payload model, registry, declaring-module mapping, include list, and queue classification together. The registration tests are the guardrail that keeps those locations synchronized.
