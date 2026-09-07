from types import SimpleNamespace

from opentelemetry import propagate, trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState

from app.connections.celery import (
    _inject_trace_context,
    log_task_postrun,
    log_task_prerun,
)
from app.utils import execution_path


def _span_context() -> SpanContext:
    return SpanContext(
        trace_id=0x1234567890ABCDEF1234567890ABCDEF,
        span_id=0x1234567890ABCDEF,
        is_remote=False,
        trace_flags=TraceFlags.SAMPLED,
        trace_state=TraceState(),
    )


def test_task_publish_injects_w3c_trace_context() -> None:
    carrier: dict[str, str] = {}
    context = trace.set_span_in_context(NonRecordingSpan(_span_context()))

    _inject_trace_context(carrier, context=context)

    extracted = propagate.extract(carrier)
    extracted_span = trace.get_current_span(extracted).get_span_context()
    assert extracted_span.trace_id == _span_context().trace_id
    assert extracted_span.span_id == _span_context().span_id


def test_task_signals_scope_trace_and_logging_context() -> None:
    headers: dict[str, str] = {}
    _inject_trace_context(
        headers,
        context=trace.set_span_in_context(NonRecordingSpan(_span_context())),
    )
    task = SimpleNamespace(
        name="tasks.example",
        request=SimpleNamespace(headers=headers),
    )

    log_task_prerun(task_id="task-1", task=task, args=(), kwargs={})
    try:
        current_span = trace.get_current_span().get_span_context()
        assert current_span.trace_id == _span_context().trace_id
        assert execution_path.get() == ["celery:tasks.example"]
    finally:
        log_task_postrun(task_id="task-1", task=task, state="SUCCESS")

    assert not trace.get_current_span().get_span_context().is_valid
    assert execution_path.get([]) == []


def test_celery_process_bootstrap_owns_logging_and_otel_lifecycle(monkeypatch) -> None:
    import importlib

    celery_module = importlib.import_module("app.connections.celery")
    calls: list[tuple[str, str | None]] = []
    settings = SimpleNamespace(OTEL_ENABLED=True, OTEL_SERVICE_NAME="celery-worker")

    monkeypatch.setattr(celery_module, "get_settings", lambda: settings)
    monkeypatch.setattr(celery_module, "setup_logging", lambda: calls.append(("logging", None)))
    monkeypatch.setattr(
        "app.shared.otel.setup_otel",
        lambda service_name: calls.append(("otel", service_name)),
    )
    monkeypatch.setattr("app.shared.otel.shutdown_otel", lambda: calls.append(("shutdown", None)))

    celery_module.initialize_celery_observability()
    celery_module.shutdown_celery_observability()

    assert calls == [
        ("logging", None),
        ("otel", "celery-worker"),
        ("shutdown", None),
    ]
