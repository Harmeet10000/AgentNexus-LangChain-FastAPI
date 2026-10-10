"""Worker execution telemetry and lifecycle hooks."""

from __future__ import annotations

from contextlib import ExitStack
from functools import cache
from typing import TYPE_CHECKING

import opentelemetry.trace as otel_trace
from celery.signals import (
    task_failure,
    task_postrun,
    task_prerun,
    task_retry,
    worker_process_init,
    worker_shutdown,
)
from opentelemetry import context as otel_context
from opentelemetry import metrics, propagate

from app.config import get_settings
from app.observability.logging import logger, setup_logging
from app.observability.logging import execution_path

if TYPE_CHECKING:
    from typing import Any

    from celery import Task


def initialize_celery_observability(**_: Any) -> None:
    """Initialize observability inside each Celery worker process."""
    settings = get_settings()
    setup_logging()
    if settings.OTEL_ENABLED:
        from app.observability import setup_otel  # noqa: PLC0415

        setup_otel(service_name=settings.OTEL_SERVICE_NAME)


def shutdown_celery_observability(**_: Any) -> None:
    """Flush worker telemetry when Celery is shutting down."""
    from app.observability import shutdown_otel  # noqa: PLC0415

    shutdown_otel()


worker_process_init.connect(initialize_celery_observability)
worker_shutdown.connect(shutdown_celery_observability)


_task_observability_contexts: dict[str, tuple[Any, Any, Any]] = {}


def _task_context_key(task_id: str | None, task: Task | None) -> str:
    return task_id or f"anonymous:{id(task)}"


@cache
def _celery_meters() -> tuple[Any, Any, Any]:
    """Create the Celery meters on first use and reuse them for this process."""
    # Keep this lazy: importing the module must not initialize OTel exporters.
    meter = metrics.get_meter("celery")
    return (
        meter.create_counter("celery.task.completed_total", unit="1"),
        meter.create_histogram("celery.task.duration_seconds", unit="s"),
        meter.create_counter("celery.task.retries_total", unit="1"),
    )


@task_prerun.connect
def log_task_prerun(
    task_id: str | None = None,
    task: Task | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    **_: Any,
) -> None:
    task_name = task.name if task else "unknown"
    key = _task_context_key(task_id, task)
    parent_context = propagate.extract(
        getattr(getattr(task, "request", None), "headers", None) or {}
    )
    context_token = otel_context.attach(parent_context)
    execution_token = execution_path.set([f"celery:{task_name}"])
    log_context = ExitStack()
    log_context.enter_context(
        logger.contextualize(
            task=task_name,
            task_id=task_id,
            layer="task",
            flow=f"celery:{task_name}",
        )
    )
    _task_observability_contexts[key] = (context_token, execution_token, log_context)

    span_ctx = otel_trace.get_current_span().get_span_context()
    trace_id = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
    extra = {
        "task": task_name,
        "task_id": task_id,
        "args_count": len(args or ()),
        "kwargs_keys": sorted((kwargs or {}).keys()),
    }
    if trace_id:
        extra["trace_id"] = trace_id
    logger.bind(**extra).info("Celery task started")


@task_postrun.connect
def log_task_postrun(
    task_id: str | None = None,
    task: Task | None = None,
    state: str | None = None,
    **_: Any,
) -> None:
    span_ctx = otel_trace.get_current_span().get_span_context()
    trace_id = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
    extra = {"task": task.name if task else None, "task_id": task_id, "state": state}
    if trace_id:
        extra["trace_id"] = trace_id
    try:
        logger.bind(**extra).info("Celery task finished")
    finally:
        context_data = _task_observability_contexts.pop(_task_context_key(task_id, task), None)
        if context_data is not None:
            context_token, execution_token, log_context = context_data
            log_context.close()
            execution_path.reset(execution_token)
            otel_context.detach(context_token)


@task_retry.connect
def log_task_retry(
    request: Any | None = None,
    reason: BaseException | None = None,
    **_: Any,
) -> None:
    span_ctx = otel_trace.get_current_span().get_span_context()
    trace_id = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
    extra = {
        "task": getattr(request, "task", None),
        "task_id": getattr(request, "id", None),
        "retry_count": getattr(request, "retries", None),
    }
    if trace_id:
        extra["trace_id"] = trace_id
    logger.bind(**extra, reason=str(reason)).warning("Celery task retry emitted")


@task_failure.connect
def log_task_failure(
    task_id: str | None = None,
    exception: BaseException | None = None,
    sender: Task | None = None,
    **_: Any,
) -> None:
    span_ctx = otel_trace.get_current_span().get_span_context()
    trace_id = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
    extra = {"task": sender.name if sender else None, "task_id": task_id}
    if trace_id:
        extra["trace_id"] = trace_id
    logger.bind(**extra, error=str(exception)).exception("Celery task failed signal")
