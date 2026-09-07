import traceback
from collections.abc import Mapping
from typing import Any

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import TraceFlags

from app.utils import logger

from .configuration import exporter_enabled
from .logs_compat import (
    BatchLogRecordProcessor,
    OTLPLogExporter,
    SeverityNumber,
)
from .logs_compat import (
    LoggerProvider as SDKLoggerProvider,
)
from .logs_compat import (
    LogRecord as OTelLogRecord,
)
from .logs_compat import (
    set_logger_provider as set_global_logger_provider,
)

_SEVERITY_MAP = {
    "TRACE": SeverityNumber.TRACE2,
    "DEBUG": SeverityNumber.DEBUG,
    "INFO": SeverityNumber.INFO,
    "SUCCESS": SeverityNumber.INFO,
    "WARNING": SeverityNumber.WARN,
    "ERROR": SeverityNumber.ERROR,
    "CRITICAL": SeverityNumber.FATAL,
}
_MAX_ATTRIBUTE_LENGTH = 4096


def _normalize_attribute(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        normalized = value
    elif isinstance(value, (list, tuple)):
        normalized = [str(item) for item in value]
    elif isinstance(value, Mapping):
        normalized = str(value)
    else:
        normalized = str(value)
    if isinstance(normalized, str):
        return normalized[:_MAX_ATTRIBUTE_LENGTH]
    return normalized


def normalize_otel_attributes(extra: Mapping[object, object]) -> dict[str, Any]:
    """Convert arbitrary Loguru metadata into OTEL-supported attributes."""
    return {str(key): _normalize_attribute(value) for key, value in extra.items()}


def _exception_attributes(exception: object) -> dict[str, str]:
    value = getattr(exception, "value", None)
    if value is None:
        return {}
    exception_type = type(value)
    return {
        "exception.type": exception_type.__name__,
        "exception.message": str(value)[:_MAX_ATTRIBUTE_LENGTH],
        "exception.stacktrace": "".join(
            traceback.format_exception(
                exception_type,
                value,
                getattr(exception, "traceback", None),
            )
        )[:_MAX_ATTRIBUTE_LENGTH],
    }


def build_otel_log_record(record: Any) -> OTelLogRecord:
    """Build an OTEL log record from a redacted Loguru record."""
    span = otel_trace.get_current_span()
    span_context = span.get_span_context() if span is not None else None
    trace_id = span_context.trace_id if span_context and span_context.is_valid else 0
    span_id = span_context.span_id if span_context and span_context.is_valid else 0
    trace_flags = (
        TraceFlags.SAMPLED
        if span_context and span_context.is_valid and span_context.trace_flags.sampled
        else TraceFlags.DEFAULT
    )
    attributes = normalize_otel_attributes(record.get("extra", {}))
    attributes.update(_exception_attributes(record.get("exception")))
    return OTelLogRecord(
        timestamp=int(record["time"].timestamp() * 1_000_000_000),
        trace_id=trace_id,
        span_id=span_id,
        trace_flags=trace_flags,  # ty: ignore[invalid-argument-type]
        severity_number=_SEVERITY_MAP.get(
            record["level"].name.upper(), SeverityNumber.INFO
        ),
        severity_text=record["level"].name,
        body=record["message"],
        attributes=attributes,
    )


def _setup_logger_provider(
    resource: Resource,
    *,
    exporter: str = "otlp",
    endpoint: str | None = None,
) -> SDKLoggerProvider | None:
    if not exporter_enabled(exporter):
        return None

    provider = SDKLoggerProvider(resource=resource)
    processor = BatchLogRecordProcessor(
        OTLPLogExporter(endpoint=endpoint),
        max_queue_size=2048,
        max_export_batch_size=512,
        schedule_delay_millis=5000,
    )
    provider.add_log_record_processor(processor)
    set_global_logger_provider(provider)
    return provider


def _patch_loguru_sink(logger_provider: SDKLoggerProvider) -> None:
    otel_logger = logger_provider.get_logger("loguru")

    def otel_sink(message) -> None:
        otel_logger.emit(build_otel_log_record(message.record))

    sink_id = getattr(_patch_loguru_sink, "_sink_id", None)
    if sink_id is not None:
        logger.remove(sink_id)

    new_sink_id = logger.add(
        otel_sink,
        level=0,
        format="{message}",
    )
    _patch_loguru_sink._sink_id = new_sink_id  # ty: ignore[unresolved-attribute]


def _remove_loguru_sink() -> None:
    sink_id = getattr(_patch_loguru_sink, "_sink_id", None)
    if sink_id is not None:
        logger.remove(sink_id)
        _patch_loguru_sink._sink_id = None  # ty: ignore[unresolved-attribute]
