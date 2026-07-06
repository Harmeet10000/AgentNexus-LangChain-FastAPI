## ADDED Requirements

### Requirement: Celery worker process calls setup_otel() before app creation
The Celery worker SHALL initialize the OTel pipeline during worker startup to enable auto-instrumentation of task execution.

#### Scenario: OTel initialized in Celery worker process
- **WHEN** the Celery worker process starts
- **THEN** `setup_otel()` SHALL be called before the Celery application is fully configured
- **AND** the `Resource` SHALL set `service.name` to `langchain-fastapi-celery` to distinguish Celery traces from FastAPI/MCP in SigNoz

```
# src/app/connections/celery.py — changes at the module level, before create_celery_app()
from app.shared.otel import setup_otel
from app.config import get_settings

_settings = get_settings()
if _settings.OTEL_ENABLED:
    setup_otel(service_name="langchain-fastapi-celery")
```

#### Scenario: Celery auto-instrumentation wraps task execution
- **WHEN** a Celery task runs
- **THEN** the `CeleryInstrumentor` SHALL create a root span for the task execution named `celery.{task_name}`
- **AND** the span SHALL include attributes for `celery.task.name`, `celery.task.id`, `celery.task.retries`

```
# The CeleryInstrumentor (registered in instrument.py) produces spans like:
# Span: "celery.tasks.example.add" — attributes:
#   celery.task.name: "tasks.example.add"
#   celery.task.id: "8e4f3c2a-..."
#   celery.task.retries: 0
```

#### Scenario: Producer-side spans for apply_async
- **WHEN** `tasks.example.add.delay(1, 2)` is called from FastAPI
- **THEN** the Celery instrumentor SHALL create a producer span that propagates trace context into the Celery message headers
- **AND** the worker span SHALL connect to the producer span as a child (trace context flows through message headers)

### Requirement: Task lifecycle logs include trace_id
The existing Celery signal handlers (`task_prerun`, `task_postrun`, `task_retry`, `task_failure`) SHALL enrich their log entries with the current OTel `trace_id` when available.

#### Scenario: trace_id attached to task prerun log
- **WHEN** a Celery task starts executing
- **THEN** `log_task_prerun` SHALL read `trace_id` from the current OTel span context
- **AND** SHALL include `trace_id` in the log bindings

```
# src/app/connections/celery.py — log_task_prerun enriched
@task_prerun.connect
def log_task_prerun(
    task_id: str | None = None,
    task: Task | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    **_: Any,
) -> None:
    # Extract trace_id from current OTel context
    trace_id = ""
    span = trace.get_current_span()
    if span is not None:
        span_context = span.get_span_context()
        if span_context.is_valid:
            trace_id = format(span_context.trace_id, "032x")

    logger.bind(
        task=task.name if task else None,
        task_id=task_id,
        trace_id=trace_id,                   # NEW
        args_count=len(args or ()),
        kwargs_keys=sorted((kwargs or {}).keys()),
    ).info("Celery task started")
```

#### Scenario: trace_id attached to task failure log
- **WHEN** a Celery task fails
- **THEN** `log_task_failure` SHALL include `trace_id` in the log bindings
- **AND** the error log SHALL reference the failing span

```
# log_task_failure enriched
@task_failure.connect
def log_task_failure(
    task_id: str | None = None,
    exception: BaseException | None = None,
    sender: Task | None = None,
    **_: Any,
) -> None:
    trace_id = ""
    span = trace.get_current_span()
    if span is not None:
        span_context = span.get_span_context()
        if span_context.is_valid:
            trace_id = format(span_context.trace_id, "032x")

    logger.bind(
        task=sender.name if sender else None,
        task_id=task_id,
        trace_id=trace_id,                   # NEW
    ).error(f"Celery task failed signal: {exception!s}")
```

### Requirement: ResilientTask lifecycle emits OTel metrics
The `ResilientTask` base class SHALL record task outcome metrics via the shared OTel meter.

#### Scenario: Task success increments success counter
- **WHEN** `ResilientTask.on_success()` is called
- **THEN** the system SHALL increment `celery.task.completed_total` counter with attributes `task_name`, `status="success"`
- **AND** SHALL record `celery.task.duration_seconds` histogram with the task duration

```
# src/app/connections/celery.py — ResilientTask changes
from opentelemetry import metrics

_otel_meter = metrics.get_meter("celery")
_celery_tasks_total = _otel_meter.create_counter(
    name="celery.task.completed_total",
    unit="1",
    description="Total completed Celery tasks by status",
)
_celery_task_duration = _otel_meter.create_histogram(
    name="celery.task.duration_seconds",
    unit="s",
    description="Celery task execution duration",
)
_celery_task_retries = _otel_meter.create_counter(
    name="celery.task.retries_total",
    unit="1",
    description="Total Celery task retries",
)

# These are declared at module level so all instances share them

class ResilientTask(Task):
    ...
    def on_success(self, retval: Any, task_id: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        _ = (retval, task_id, args, kwargs)
        logger.bind(
            task=self.name,
            task_id=task_id,
        ).info("Task completed successfully")

        # OTel metrics
        _celery_tasks_total.add(1, {"task_name": self.name, "status": "success"})
        # Duration comes from span timing — histogram from CeleryInstrumentor's span
        # If we want explicit tracking, store start time in on_prerun and compute here
```

#### Scenario: Task failure increments failure counter
- **WHEN** `ResilientTask.on_failure()` is called
- **THEN** the system SHALL increment `celery.task.completed_total` counter with attributes `task_name`, `status="failure"`
- **AND** SHALL record `celery.task.duration_seconds` histogram with the task duration

```
# in on_failure
def on_failure(self, exc: Any, task_id: str, args: tuple[Any, ...], kwargs: dict[str, Any], einfo: Any) -> None:
    _ = (args, kwargs, einfo)
    logger.bind(
        task=self.name,
        task_id=task_id,
    ).error(f"Task failed: {exc!s}")

    _celery_tasks_total.add(1, {"task_name": self.name, "status": "failure"})
    _celery_task_retries.add(1, {"task_name": self.name, "attempt": self.request.retries})
```

#### Scenario: Task retry increments retry counter
- **WHEN** `ResilientTask.on_retry()` is called
- **THEN** the system SHALL increment `celery.task.retries_total` counter with attributes `task_name`, `attempt`

```
# in on_retry
def on_retry(self, exc: Any, task_id: str, args: tuple[Any, ...], kwargs: dict[str, Any], einfo: Any) -> None:
    _ = (args, kwargs, einfo)
    logger.bind(
        task=self.name,
        task_id=task_id,
    ).warning(f"Task scheduled for retry")

    _celery_task_retries.add(1, {"task_name": self.name, "attempt": self.request.retries})
```

### Requirement: setup_otel() includes instrumentation.celery in instrument.py
The `setup_auto_instrumentation()` function SHALL register the Celery instrumentor when running in the Celery worker process.

#### Scenario: CeleryInstrumentor registered
- **WHEN** `setup_auto_instrumentation()` runs with Celery available
- **THEN** `CeleryInstrumentor().instrument()` SHALL be called
- **AND** the instrumentor SHALL create spans for task execution, including `apply_async` calls from the producer side

```
# src/app/shared/otel/instrument.py
try:
    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    CeleryInstrumentor().instrument()
except Exception:
    logger.warning("Celery auto-instrumentation failed — continuing without task tracing")
```

### Requirement: Celery worker task_track_started and task_send_sent_event remain enabled
The Celery configuration flags `task_track_started=True` and `task_send_sent_event=True` are required for accurate span capture and SHALL remain enabled.

#### Scenario: Existing Celery config flags preserved
- **WHEN** the OTel changes are applied
- **THEN** `task_track_started=True` SHALL remain in Celery config (needed for CeleryInstrumentor to capture task start)
- **AND** `task_send_sent_event=True` SHALL remain (needed for producer-side span creation)
