"""Regression tests for the logging standard.

Covers the two silent log-loss defects: brace-carrying messages/values
crashing ``console_format`` (KeyError → record dropped), and secret keys
escaping redaction.

Implementation note: these tests NEVER assert on captured stderr. Loguru
binds its sink stream object at import time (the real sys.stderr), so
per-test capsys/capfd assertions are unreliable here — output visibly
reaches the terminal yet ``readouterr()`` returns empty. Instead, brace
safety is tested at the exact crash site (``console_format`` output fed
through ``format_map``, replicating what loguru does with it), and
redaction/traceback through a temporary list sink.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING

from opentelemetry.trace import StatusCode
from returns.result import Failure

from app.utils import logger
from app.utils.logger import _InterceptHandler, console_format, trace_layer

if TYPE_CHECKING:
    from typing import Any


def _record(message: str, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "level": SimpleNamespace(name="INFO"),
        "time": datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC),
        "message": message,
        "extra": extra,
        "exception": None,
    }


def test_console_format_output_survives_format_map_with_braces() -> None:
    # console_format output is fed through format_map by loguru; any
    # leftover {placeholder} besides {exception} raises KeyError and the
    # record is lost. Reproduced pre-fix with KeyError 'oops'.
    out = console_format(
        _record("test {oops} path", {"details": {"doc_id": "x"}, "flow": "a -> b"})
    )
    formatted = out.format_map({"exception": ""})
    assert "{oops}" in formatted
    assert "doc_id" in formatted


def test_secret_keys_are_redacted() -> None:
    lines: list[str] = []
    handler_id = logger.add(lines.append, format="{message} | {extra}", level="DEBUG")
    try:
        logger.bind(password="hunter2", session_token="abc123", user_id=7).info(
            "Login attempt recorded"
        )
    finally:
        logger.remove(handler_id)
    assert lines, "expected the record to reach the sink"
    assert "*** REDACTED ***" in lines[0]
    assert "hunter2" not in lines[0]
    assert "abc123" not in lines[0]
    assert "user_id" in lines[0]


def test_structured_fields_are_bound_as_loguru_extra() -> None:
    records: list[dict[str, Any]] = []
    handler_id = logger.add(
        lambda message: records.append(message.record),
        format="{message}",
        level="DEBUG",
    )
    try:
        logger.bind(status="running", component="lifespan").info("Application ready")
    finally:
        logger.remove(handler_id)

    assert records
    assert records[0]["extra"]["status"] == "running"
    assert records[0]["extra"]["component"] == "lifespan"


def test_stdlib_records_are_forwarded_to_loguru() -> None:
    lines: list[str] = []
    handler_id = logger.add(lines.append, format="{message}", level="DEBUG")
    try:
        _InterceptHandler().emit(logging.LogRecord("uvicorn", logging.INFO, "", 0, "server ready", (), None))
    finally:
        logger.remove(handler_id)

    assert lines == ["server ready\n"]


def test_setup_logging_reads_level_from_settings(monkeypatch: Any) -> None:
    import importlib

    logger_module = importlib.import_module("app.utils.logger")

    class Settings:
        LOG_LEVEL = "WARNING"

    def fake_get_settings() -> Settings:
        return Settings()

    monkeypatch.setattr(logger_module, "get_settings", fake_get_settings)
    monkeypatch.setattr(logger_module.loguru_logger, "remove", lambda: None)
    captured: dict[str, Any] = {}

    def capture_add(*_args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(logger_module.loguru_logger, "add", capture_add)
    monkeypatch.setattr(logger_module, "_install_stdlib_bridge", lambda: None)

    logger_module.setup_logging()

    assert captured["level"] == "WARNING"


def test_setup_logging_uses_structured_stdout_for_json_format(monkeypatch: Any) -> None:
    import importlib

    logger_module = importlib.import_module("app.utils.logger")

    class Settings:
        LOG_LEVEL = "INFO"
        LOG_FORMAT = "json"

    def fake_get_settings() -> Settings:
        return Settings()

    captured: dict[str, Any] = {}
    monkeypatch.setattr(logger_module, "get_settings", fake_get_settings)
    monkeypatch.setattr(logger_module.loguru_logger, "remove", lambda: None)
    monkeypatch.setattr(
        logger_module.loguru_logger,
        "add",
        lambda *_args, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(logger_module, "_install_stdlib_bridge", lambda: None)

    logger_module.setup_logging()

    assert captured["serialize"] is True
    assert captured["colorize"] is False


def test_sensitive_values_are_redacted_recursively() -> None:
    lines: list[str] = []
    handler_id = logger.add(lines.append, format="{extra}", level="DEBUG")
    try:
        logger.bind(
            payload={
                "api_key": "key-value",
                "nested": [{"authorization": "Bearer secret-value"}],
            }
        ).info("Credentials received")
    finally:
        logger.remove(handler_id)

    assert lines
    assert "key-value" not in lines[0]
    assert "Bearer secret-value" not in lines[0]
    assert lines[0].count("*** REDACTED ***") >= 2


def _raise_boom() -> None:
    msg = "boom"
    raise ValueError(msg)


def test_exception_attaches_traceback() -> None:
    lines: list[str] = []
    handler_id = logger.add(lines.append, format="{message}\n{exception}", level="DEBUG")
    try:
        try:
            _raise_boom()
        except ValueError:
            logger.bind(operation="regression_probe").exception("Probe failed")
    finally:
        logger.remove(handler_id)
    assert lines, "expected the record to reach the sink"
    assert "Probe failed" in lines[0]
    assert "Traceback" in lines[0]
    assert "ValueError" in lines[0]


async def test_trace_layer_marks_raised_exceptions_as_span_errors() -> None:
    import importlib

    logger_module = importlib.import_module("app.utils.logger")

    class FakeSpan:
        def __init__(self) -> None:
            self.status_code = None
            self.name = ""
            self.attributes: dict[str, object] = {}

        def __enter__(self) -> FakeSpan:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def set_attribute(self, key: str, value: object) -> None:
            self.attributes[key] = value

        def record_exception(self, *_args: object) -> None:
            return None

        def set_status(self, status: object) -> None:
            self.status_code = getattr(status, "status_code", None)

    class FakeTracer:
        def __init__(self, span: FakeSpan) -> None:
            self.span = span

        def start_as_current_span(self, name: str, **_kwargs: object) -> FakeSpan:
            self.span.name = name
            return self.span

    span = FakeSpan()
    fake_tracer = FakeTracer(span)
    original_get_tracer = logger_module.otel_trace.get_tracer
    logger_module.otel_trace.get_tracer = lambda _name: fake_tracer

    @trace_layer("service")
    async def fail() -> None:
        message = "expected failure"
        raise ValueError(message)

    try:
        with suppress(ValueError):
            await fail()
    finally:
        logger_module.otel_trace.get_tracer = original_get_tracer

    assert span.status_code == StatusCode.ERROR
    assert span.name.endswith("test_logger_standard.fail")


async def test_trace_layer_marks_typed_failure_as_business_outcome() -> None:
    import importlib

    logger_module = importlib.import_module("app.utils.logger")

    class FakeSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}

        def __enter__(self) -> FakeSpan:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def set_attribute(self, key: str, value: object) -> None:
            self.attributes[key] = value

    class FakeTracer:
        def __init__(self, span: FakeSpan) -> None:
            self.span = span

        def start_as_current_span(self, _name: str, **_kwargs: object) -> FakeSpan:
            return self.span

    span = FakeSpan()
    original_get_tracer = logger_module.otel_trace.get_tracer
    logger_module.otel_trace.get_tracer = lambda _name: FakeTracer(span)

    @trace_layer("service")
    async def fail_result() -> Failure[str]:
        return Failure("expected business failure")

    try:
        await fail_result()
    finally:
        logger_module.otel_trace.get_tracer = original_get_tracer

    assert span.attributes["result.outcome"] == "failure"
