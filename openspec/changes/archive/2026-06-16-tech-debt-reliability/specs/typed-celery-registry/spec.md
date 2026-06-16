# Capability: typed-celery-registry

## Purpose
Replace string-based Celery task dispatch with a typed registry providing compile-time validation, Pydantic payload schemas, and IDE support.

## Requirements

### R1: Registry Module
- Location: `src/app/connections/celery_registry.py`
- `TaskRegistry` class mapping task names to Pydantic payload models
- `register_task(name, payload_model, retry_policy)` function
- `send_typed_task(registry, task_name, payload)` wrapper around `celery_app.send_task()`
- Payload validation on send (raises `ValidationError` if payload doesn't match schema)
- `LegacyTaskPayload` fallback for unmigrated tasks (accepts `**kwargs`)

### R2: Payload Models
- Location: `src/app/tasks/payloads.py`
- Pydantic models for each task:
  - `DocumentIngestPayload` (document_id, user_id, filename, content_type, object_uri)
  - `EmbedChunksPayload` (user_id, document_id, chunks, extra_warnings)
  - `SearchIndexPayload` (document_id, chunk_ids, action)
  - `MemoryDecayPayload` (user_id, scope, decay_config)
  - `PageIndexPayload` (document_id, urls, config)
  - `AuthEmailPayload` (user_id, template_id, variables)
  - `DocumentExtractionPayload` (document_id, raw_bytes_uri, content_type)
- All models use `frozen=True`, `extra="forbid"`

### R3: Retry Policies
- Per-task retry configuration as Pydantic models
- Default: `max_retries=5`, `backoff_max=600`, `retry_delay=5`
- Override per task: e.g., `DocumentExtractRetryPolicy(max_retries=3, backoff_max=120)`
- Retry policy sent as task header (Celery `retry_kwargs`)

### R4: Incremental Migration Path
- Phase 1: Create registry + payloads + `LegacyTaskPayload` fallback
- Phase 2: Migrate `documents_ingest` task (proof of concept)
- Phase 3: Migrate remaining 8 tasks one per PR
- Old `celery_app.send_task("tasks.xxx", kwargs={...})` calls remain supported via fallback
- Deprecated tasks emit `warnings.warn()` when called via string name

### R5: Task Signal Handlers
- `task_prerun`: inject `correlation_id` into task context (if present in headers)
- `task_postrun`: log task completion with timing
- `task_failure`: log failure with traceback
- Handlers registered in `src/app/connections/celery.py`

### R6: Type Checking Integration
- `ty` should not flag registry calls as errors
- `Annotated` type aliases for common payload patterns
- Registry dict typed as `dict[str, type[BaseModel]]`

## Acceptance Criteria
- [ ] `send_typed_task(registry, "documents.ingest", DocumentIngestPayload(...))` validates payload
- [ ] Invalid payload raises `ValidationError` with clear message
- [ ] Legacy `send_task("tasks.xxx", kwargs={...})` still works during migration
- [ ] `uv run ty check src/app/connections/celery_registry.py` passes
- [ ] First migrated task (`documents_ingest`) has integration test

## Non-Goals
- Replace Celery with another task queue
- Migrate all 9 tasks in one PR (incremental only)
- Add distributed tracing beyond correlation IDs
