# Typed Celery Task Registry Extension

## Scope

- `src/app/connections/celery_registry.py` — add `typed_send()` classmethod
- `src/tasks/document_tasks.py` — register typed payload
- `src/tasks/search_tasks.py` — register typed payload
- `src/app/shared/outbox/relay.py` — use `CeleryTaskRegistry.typed_send()`

## Problem

Outbox events dispatch Celery tasks via `celery_app.send_task(name, kwargs=...)` with bare string task names. There is no validation that:
1. The task name corresponds to a registered task
2. The kwargs match the task's expected parameters

This leads to silent failures when:
- A task is renamed but the outbox event_type string isn't updated
- A task signature changes (new required param) but all existing outbox events have old payloads
- A typo in the event_type string dispatches to a non-existent task (Celery silently discards)

The `CeleryTaskRegistry` and `TypedCeleryTask` already exist in `celery_registry.py`. The `auth_email_tasks_typed.py` file demonstrates the pattern for two email tasks. This spec extends the registry with a dispatch-side validation method.

## Solution

### 1. Add `typed_send()` to `CeleryTaskRegistry`

```python
class CeleryTaskRegistry:
    _registry: ClassVar[dict[str, type[CeleryTaskPayload]]] = {}

    @classmethod
    def typed_send(cls, task_name: str, kwargs: dict[str, object], **send_task_opts: object) -> object:
        """Validate kwargs against registered model, then send.

        Raises ValidationError if kwargs don't match the registered payload model.
        Falls back to LegacyTaskPayload (accepts any kwargs) if task is not registered.
        """
        cls.validate(task_name, kwargs)
        return celery_app.send_task(task_name, kwargs=kwargs, **send_task_opts)
```

### 2. Register document and search task payloads

In `src/tasks/document_tasks.py`:

```python
class DocumentIngestPayload(CeleryTaskPayload):
    document_id: str
    user_id: str
    filename: str
    content_type: str
    object_uri: str

CeleryTaskRegistry.register("tasks.documents_ingest", DocumentIngestPayload)
```

In `src/tasks/search_tasks.py`:

```python
class SearchIngestPayload(CeleryTaskPayload):
    document_id: str
    content: str
    content_hash: str
    title: str
    source_uri: str | None = None
    doc_metadata: dict[str, object] | None = None

CeleryTaskRegistry.register("tasks.search_ingest", SearchIngestPayload)
```

### 3. Update outbox relay dispatch

In `src/app/shared/outbox/relay.py`, replace:

```python
celery_app.send_task(event.event_type, kwargs=event.payload)
```

with:

```python
CeleryTaskRegistry.typed_send(event.event_type, kwargs=event.payload)
```

This validates kwargs against the registered Pydantic model *at outbox flush time*. If a payload is malformed, the error surfaces immediately in the health check logs rather than as a silent drop in the Celery worker.

## Edge Cases

| Case | Behaviour |
|------|-----------|
| Task name is registered | Validates kwargs against registered model before send |
| Task name is NOT registered | Falls back to `LegacyTaskPayload` (accepts anything); sends as before |
| Kwargs fail validation | `ValidationError` raised at outbox flush time; event goes to dead-letter after retries |
| Task renamed in code but not in registry | `LegacyTaskPayload` fallback; logged as warning by `validate()` |

## Verification

1. Unit: `CeleryTaskRegistry.validate("tasks.documents_ingest", {"document_id": "x", "user_id": "y"})` — passes
2. Unit: `CeleryTaskRegistry.validate("tasks.documents_ingest", {})` — raises ValidationError
3. Unit: `CeleryTaskRegistry.typed_send("nonexistent.task", {"a": 1})` — sends via fallback, logs warning
4. Integration: outbox event flushes → `typed_send` called → kwargs validated → task enqueued
