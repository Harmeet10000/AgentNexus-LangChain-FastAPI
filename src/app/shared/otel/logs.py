from opentelemetry import trace as otel_trace
from opentelemetry._logs import SeverityNumber
from opentelemetry._logs import set_logger_provider as set_global_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider as SDKLoggerProvider
from opentelemetry.sdk._logs import LogRecord as OTelLogRecord  # type: ignore
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import TraceFlags

from app.utils import logger

_SEVERITY_MAP = {
    "TRACE": SeverityNumber.TRACE2,
    "DEBUG": SeverityNumber.DEBUG,
    "INFO": SeverityNumber.INFO,
    "SUCCESS": SeverityNumber.INFO,
    "WARNING": SeverityNumber.WARN,
    "ERROR": SeverityNumber.ERROR,
    "CRITICAL": SeverityNumber.FATAL,
}


def _setup_logger_provider(resource: Resource) -> SDKLoggerProvider | None:
    provider = SDKLoggerProvider(resource=resource)
    processor = BatchLogRecordProcessor(
        OTLPLogExporter(),
        max_queue_size=2048,
        max_export_batch_size=512,
        schedule_delay_millis=5000,
    )
    provider.add_log_record_processor(processor)
    set_global_logger_provider(provider)
    return provider


def _patch_loguru_sink(logger_provider: SDKLoggerProvider) -> None:
    def otel_sink(message) -> None:
        record = message.record
        span = otel_trace.get_current_span()
        span_context = span.get_span_context() if span is not None else None
        trace_id = span_context.trace_id if span_context and span_context.is_valid else 0
        span_id = span_context.span_id if span_context and span_context.is_valid else 0
        trace_flags = (
            TraceFlags.SAMPLED
            if span_context and span_context.is_valid and span_context.trace_flags.sampled
            else TraceFlags.DEFAULT
        )

        otel_record = OTelLogRecord(
            timestamp=int(record["time"].timestamp() * 1_000_000_000),
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=trace_flags,
            severity_number=_SEVERITY_MAP.get(record["level"].name.upper(), SeverityNumber.INFO),
            severity_text=record["level"].name,
            body=record["message"],
            resource=logger_provider.resource,
            attributes=dict(record.get("extra", {})),
        )
        logger_provider.emit(otel_record)  # type: ignore

    sink_id = getattr(_patch_loguru_sink, "_sink_id", None)
    if sink_id is not None:
        logger.remove(sink_id)

    new_sink_id = logger.add(
        otel_sink,
        level=0,
        format="{message}",
    )
    setattr(_patch_loguru_sink, "_sink_id", new_sink_id)
