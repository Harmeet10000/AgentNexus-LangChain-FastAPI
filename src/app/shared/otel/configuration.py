"""Single observability policy and OpenTelemetry exporter helpers."""

from dataclasses import dataclass
from typing import Any


def exporter_enabled(exporter: str) -> bool:
    """Return whether an OpenTelemetry signal exporter is enabled."""
    return exporter.strip().lower() not in {"", "none"}


@dataclass(frozen=True, slots=True)
class ObservabilityPolicy:
    """Immutable process-wide observability policy derived from Settings."""

    log_level: str
    structured: bool
    service_name: str
    endpoint: str
    sample_rate: float
    traces_enabled: bool
    metrics_enabled: bool
    logs_enabled: bool

    @classmethod
    def from_settings(cls, settings: Any) -> "ObservabilityPolicy":
        return cls(
            log_level=settings.LOG_LEVEL,
            structured=settings.LOG_FORMAT.lower() == "json",
            service_name=settings.OTEL_SERVICE_NAME,
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            sample_rate=settings.OTEL_SAMPLE_RATE,
            traces_enabled=exporter_enabled(settings.OTEL_TRACES_EXPORTER),
            metrics_enabled=exporter_enabled(settings.OTEL_METRICS_EXPORTER),
            logs_enabled=exporter_enabled(settings.OTEL_LOGS_EXPORTER),
        )
