"""Tests for OpenTelemetry exporter configuration policy."""

import pytest

from app.shared.otel.configuration import exporter_enabled
from app.shared.otel.logs import normalize_otel_attributes


def test_exporter_policy_disables_none_and_empty_values() -> None:
    assert exporter_enabled("none") is False
    assert exporter_enabled(" NONE ") is False
    assert exporter_enabled("") is False


def test_exporter_policy_enables_configured_exporters() -> None:
    assert exporter_enabled("otlp") is True
    assert exporter_enabled("prometheus") is True


def test_observability_policy_is_derived_from_settings() -> None:
    from types import SimpleNamespace

    from app.shared.otel.configuration import ObservabilityPolicy

    policy = ObservabilityPolicy.from_settings(
        SimpleNamespace(
            LOG_LEVEL="DEBUG",
            LOG_FORMAT="json",
            OTEL_ENABLED=True,
            OTEL_SERVICE_NAME="api",
            OTEL_EXPORTER_OTLP_ENDPOINT="http://collector:4317",
            OTEL_TRACES_EXPORTER="none",
            OTEL_METRICS_EXPORTER="otlp",
            OTEL_LOGS_EXPORTER="none",
            OTEL_SAMPLE_RATE=0.25,
        )
    )

    assert policy.log_level == "DEBUG"
    assert policy.structured is True
    assert policy.service_name == "api"
    assert policy.traces_enabled is False
    assert policy.metrics_enabled is True
    assert policy.logs_enabled is False
    assert policy.sample_rate == pytest.approx(0.25)


def test_otel_attributes_are_normalized_to_exportable_values() -> None:
    attributes = normalize_otel_attributes(
        {
            "path": {"id": 42},
            "items": ["one", 2],
            "object": object(),
        }
    )

    assert attributes["path"] == "{'id': 42}"
    assert attributes["items"] == ["one", "2"]
    assert isinstance(attributes["object"], str)


def test_shutdown_otel_clears_provider_references() -> None:
    import importlib

    otel_module = importlib.import_module("app.shared.otel")

    class Provider:
        def force_flush(self, *, timeout_millis: int) -> None:
            assert timeout_millis == 10000

        def shutdown(self) -> None:
            return None

    otel_module._otel_tracer_provider = Provider()
    otel_module._otel_meter_provider = Provider()
    otel_module._otel_logger_provider = None
    otel_module._otel_initialized = True

    otel_module.shutdown_otel()

    assert otel_module._otel_tracer_provider is None
    assert otel_module._otel_meter_provider is None
    assert otel_module._otel_logger_provider is None
    assert otel_module._otel_initialized is False


def test_shutdown_otel_continues_after_provider_flush_failure() -> None:
    import importlib

    otel_module = importlib.import_module("app.shared.otel")
    shutdowns: list[str] = []

    class BrokenProvider:
        def force_flush(self, *, timeout_millis: int) -> None:
            error = "flush failed"
            raise RuntimeError(error)

        def shutdown(self) -> None:
            shutdowns.append("broken")

    class HealthyProvider:
        def force_flush(self, *, timeout_millis: int) -> None:
            shutdowns.append("flushed")

        def shutdown(self) -> None:
            shutdowns.append("healthy")

    otel_module._otel_tracer_provider = BrokenProvider()
    otel_module._otel_meter_provider = HealthyProvider()
    otel_module._otel_logger_provider = None
    otel_module._otel_initialized = True

    otel_module.shutdown_otel()

    assert shutdowns == ["flushed", "healthy"]
    assert otel_module._otel_tracer_provider is None
    assert otel_module._otel_meter_provider is None
