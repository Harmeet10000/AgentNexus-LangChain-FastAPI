"""Compatibility boundary for the OpenTelemetry logging SDK internals."""

from opentelemetry._logs import (  # noqa: PLC2701
    LogRecord,
    SeverityNumber,
    set_logger_provider,
)
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (  # noqa: PLC2701
    OTLPLogExporter,
)
from opentelemetry.sdk._logs import LoggerProvider  # noqa: PLC2701
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor  # noqa: PLC2701

__all__ = [
    "BatchLogRecordProcessor",
    "LogRecord",
    "LoggerProvider",
    "OTLPLogExporter",
    "SeverityNumber",
    "set_logger_provider",
]
