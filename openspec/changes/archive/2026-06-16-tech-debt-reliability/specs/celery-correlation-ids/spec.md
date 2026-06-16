# Capability: celery-correlation-ids

## Purpose
Propagate correlation IDs from HTTP requests through Celery tasks to downstream services (Graphiti, Neo4j, Redis) for end-to-end distributed tracing.

## Requirements

### R1: Correlation ID Injection
- Use Celery `task_prerun` signal to inject `correlation_id` from task headers
- `correlation_id` is passed as a task header when dispatching: `celery_app.send_task(..., headers={"correlation_id": cid})`
- Inject into `contextvars.ContextVar` so `logger.bind(correlation_id=...)` works in worker

### R2: Correlation ID Propagation
- HTTP → Celery: via task headers (already supported by Celery protocol)
- Celery → Graphiti: via `source_description` field in episodes
- Celery → Neo4j: via session-level metadata
- Celery → Redis: via log context (no protocol-level propagation needed)

### R3: Logging Integration
- Celery worker logs include `correlation_id` in structured format
- Task start log: `{"event": "task_started", "task": "...", "correlation_id": "..."}`
- Task success log: `{"event": "task_completed", "task": "...", "correlation_id": "...", "duration_ms": N}`
- Task failure log: `{"event": "task_failed", "task": "...", "correlation_id": "...", "error": "..."}`

### R4: Signal Handlers
- Location: `src/app/connections/celery_signals.py`
- `task_prerun`: extract correlation_id from headers, set ContextVar, log task start
- `task_postrun`: clear ContextVar, log task completion with timing
- `task_failure`: log failure with traceback and correlation_id
- Register signals in `src/app/connections/celery.py` on worker init

### R5: Dispatch Side
- When `send_typed_task()` is called, auto-inject current `correlation_id` from ContextVar into task headers
- If no correlation_id in context, generate one (task-initiated)
- Legacy `send_task()` calls: manually pass `headers={"correlation_id": cid}`

### R6: Graphiti Integration
- `graphiti.add_episode()` receives `correlation_id` in metadata
- Stored as `source_description.correlation_id` for traceability
- Query results include correlation_id for debugging

## Acceptance Criteria
- [ ] HTTP request with `X-Correlation-ID: abc123` → Celery task logs show `correlation_id: abc123`
- [ ] Celery-initiated tasks generate their own correlation_id
- [ ] Graphiti episodes include correlation_id in metadata
- [ ] Task failure logs include correlation_id + traceback
- [ ] No performance overhead (ContextVar is essentially free)

## Non-Goals
- Full OpenTelemetry SDK integration (separate effort)
- Distributed trace context propagation (W3C Trace Context)
- Cross-service correlation (only within this app's Celery workers)
- Trace visualization (use LangSmith for that)
