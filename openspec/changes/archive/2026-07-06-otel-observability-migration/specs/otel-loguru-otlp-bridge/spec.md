## ADDED Requirements

### Requirement: loguru exports all log records to OTel via custom sink
The loguru logger SHALL have a custom sink that converts each log record to an OpenTelemetry `LogRecord` and exports it via `BatchLogRecordProcessor` + `OTLPLogExporter`.

#### Scenario: OTel LoggerProvider initialized in setup_otel()
- **WHEN** `setup_otel()` is called
- **THEN** a `LoggerProvider` SHALL be created with the shared `Resource` and `BatchLogRecordProcessor`
- **AND** the `LoggerProvider` SHALL use `OTLPLogExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT, insecure=True)`
- **AND** the `LoggerProvider` SHALL be set as the global logger provider via `logs.set_logger_provider()`
- **AND** if `OTEL_LOGS_EXPORTER` is set to `"none"`, no `LoggerProvider` SHALL be created

```
# src/app/shared/otel/bootstrap.py — LoggerProvider creation in _setup_logger_provider()
def _setup_logger_provider(resource: Resource, settings: Settings) -> LoggerProvider | None:
    if settings.OTEL_LOGS_EXPORTER == "none":
        return None

    from opentelemetry.exporter.otlp.proto.grpc.log_exporter import OTLPLogExporter
    from opentelemetry.sdk._logs import LoggerProvider as SDKLoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor

    exporter = OTLPLogExporter(
        endpoint=f"{settings.OTEL_EXPORTER_OTLP_ENDPOINT}/v1/logs",
        insecure=True,
    )

    provider = SDKLoggerProvider(resource=resource)
    # Batch processor: export every 5s or 512 records, whichever comes first
    provider.add_log_record_processor(
        BatchLogRecordProcessor(
            exporter,
            max_export_batch_size=512,
            schedule_delay_millis=5000,
            max_queue_size=2048,
        )
    )
    return provider
```

#### Scenario: loguru sink added via shared setup function
- **WHEN** `setup_otel()` runs
- **THEN** a helper function `_patch_loguru_sink()` SHALL be called to add a custom sink to the loguru logger
- **AND** the sink SHALL be added with `logger.add(sink_function, level=0)` to capture all levels including TRACE

```
# src/app/shared/otel/bootstrap.py — loguru OTel sink
from opentelemetry.sdk._logs import LogRecord as OTelLogRecord
from opentelemetry.sdk._logs.export import LoggerProvider
from opentelemetry.trace import SpanContext, TraceFlags

def _patch_loguru_sink(logger_provider: LoggerProvider) -> None:
    """Add a loguru sink that exports logs via OTel BatchLogRecordProcessor."""

    def otel_sink(message) -> None:
        record = message.record
        # Extract trace context from current OTel span
        span = trace.get_current_span()
        span_context = span.get_span_context() if span else None
        trace_id = span_context.trace_id if span_context and span_context.is_valid else 0
        span_id = span_context.span_id if span_context and span_context.is_valid else 0
        trace_flags = TraceFlags.SAMPLED if (span_context and span_context.is_valid and span_context.trace_flags.sampled) else TraceFlags.DEFAULT

        severity_map = {
            "TRACE": SeverityNumber.TRACE2,
            "DEBUG": SeverityNumber.DEBUG,
            "INFO": SeverityNumber.INFO,
            "SUCCESS": SeverityNumber.INFO,
            "WARNING": SeverityNumber.WARN,
            "ERROR": SeverityNumber.ERROR,
            "CRITICAL": SeverityNumber.FATAL,
        }

        otel_record = OTelLogRecord(
            timestamp=int(record["time"].timestamp() * 1_000_000_000),  # nanosecond
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=trace_flags,
            severity_number=severity_map.get(record["level"].name.upper(), SeverityNumber.INFO),
            severity_text=record["level"].name,
            body=record["message"],
            resource=logger_provider.resource,
            attributes=dict(record.get("extra", {})),
        )
        logger_provider.emit(otel_record)

    # Idempotent: remove the old sink first if it exists
    _LOGURU_OTEL_SINK_ID = getattr(_patch_loguru_sink, "_sink_id", None)
    if _LOGURU_OTEL_SINK_ID is not None:
        logger.remove(_LOGURU_OTEL_SINK_ID)

    sink_id = logger.add(
        otel_sink,
        level=0,  # TRACE — capture everything
        format="{message}",  # We handle all formatting in the sink
    )
    _patch_loguru_sink._sink_id = sink_id
```

#### Scenario: SUCCESS level mapped to INFO
- **WHEN** a loguru `SUCCESS` level record is emitted
- **THEN** the sink SHALL map it to `SeverityNumber.INFO (9)` (OTel has no SUCCESS equivalent)
- **AND** the `severity_text` SHALL remain `"SUCCESS"` for query-side identification

### Requirement: console logging output is preserved and enhanced
The existing console log output SHALL remain as-is, with the addition of `trace_id` display when a span is active.

#### Scenario: console formatter shows trace_id
- **WHEN** a log line is output to console during an active trace
- **THEN** the console format SHALL include `<green><b>trace_id</b>=<cyan>{trace_id}</cyan></green>` in the extra data section
- **AND** the format SHALL omit the `trace_id` field when no trace is active

```
# src/app/utils/logger.py — console_format() changes
def console_format(record: dict[str, Any]) -> str:
    extra = record.get("extra", {})
    trace_id_str = extra.get("trace_id", "")
    if record.get("exception") is not None:
        exception_str = f"<red>{record['exception']}</red>"
    else:
        exception_str = ""

    trace_part = ""
    if trace_id_str:
        trace_part = f" <green><b>trace_id</b>=<cyan>{trace_id_str}</cyan></green>"

    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
        f"{trace_part}"
        " - <level>{message}</level>"
        f" {exception_str}"
        "\n"
    )
```

#### Scenario: JSON file handler is removed
- **WHEN** the logging changes are applied
- **THEN** the commented-out JSON file handler code in `setup_logging()` SHALL be deleted (not just commented)
- **AND** no file-based log output SHALL be configured

### Requirement: redact_sensitive_data runs before OTel sink
The existing `redact_sensitive_data()` patcher SHALL continue to redact sensitive keys in log extra data before the OTel sink receives them.

#### Scenario: sensitive data redacted before OTel serialization
- **WHEN** a log record contains sensitive keys
- **THEN** the patcher SHALL replace values with `"*** REDACTED ***"` before the OTel sink processes the record
- **AND** the redacted values SHALL reach the OTel exporter (not the real values)

```
# src/app/utils/logger.py — unchanged redact_sensitive_data (already works)
# It wraps loguru's core and patches extra data before sinks see it.
# The OTel sink receives already-redacted data.
```

#### Scenario: nested dict redaction preserved
- **WHEN** a log record extra value is a dict containing sensitive keys
- **THEN** the nested keys SHALL also be replaced
- **AND** non-sensitive keys within the same dict SHALL be preserved

### Requirement: BatchLogRecordProcessor properly flushed on shutdown
During application shutdown, `LoggerProvider.force_flush()` SHALL be called to ensure buffered log records are exported.

#### Scenario: force_flush during shutdown
- **WHEN** `shutdown_otel()` is called (from lifespan handler)
- **THEN** `logger_provider.force_flush(timeout_millis=10000)` SHALL be called
- **AND** `logger_provider.shutdown()` SHALL be called after flush
- **AND** the loguru OTel sink SHALL NOT be removed (subsequent logs go to console only)

### Requirement: setup_logging() order preserved
The `setup_logging()` function SHALL continue to be called early in `main.py` before `setup_otel()`.

#### Scenario: Logger initialization order maintained
- **WHEN** the application starts
- **THEN** `setup_logging()` SHALL be called first (adds console sink, redact patcher)
- **AND** `setup_otel()` SHALL be called second (adds OTel sink on top of console)
- **AND** the console sink SHALL NOT be removed or duplicated

### Requirement: Multiprocessing safety
The loguru OTel sink SHALL be safe to use in multi-process Celery workers. Each worker process SHALL call `setup_otel()` independently.

#### Scenario: Each Celery worker creates its own LoggerProvider
- **WHEN** a Celery worker process starts
- **THEN** it SHALL call `setup_otel()` which creates a fresh `LoggerProvider` and OTel sink
- **AND** each process SHALL have its own `BatchLogRecordProcessor` instance
- **AND** the OTLP exporter SHALL handle concurrent exports from multiple workers
