from opentelemetry.sdk.resources import Resource

from app.config import get_settings
from app.utils import logger

from .configuration import ObservabilityPolicy

_otel_initialized: bool = False
_otel_tracer_provider = None
_otel_meter_provider = None
_otel_logger_provider = None


def get_otel_diagnostics() -> tuple[dict[str, str], ...]:
    """Return startup instrumentation status for health/debug tooling."""
    from .instrument import get_instrumentation_diagnostics

    return get_instrumentation_diagnostics()


def _build_resource(service_name: str | None = None) -> Resource:
    settings = get_settings()
    return Resource.create(
        {
            "service.name": service_name or settings.OTEL_SERVICE_NAME,
            "service.version": settings.APP_VERSION,
            "deployment.environment": settings.ENVIRONMENT,
        }
    )


def setup_otel(service_name: str | None = None) -> None:
    global _otel_initialized, _otel_tracer_provider, _otel_meter_provider, _otel_logger_provider  # noqa: PLW0603 — intentional module-level state for OTEL providers

    settings = get_settings()
    policy = ObservabilityPolicy.from_settings(settings)
    if not settings.OTEL_ENABLED or _otel_initialized:
        return

    resource = _build_resource(service_name)

    from .tracer import _setup_tracer_provider

    _otel_tracer_provider = _setup_tracer_provider(
        resource,
        settings.OTEL_SAMPLE_RATE,
        exporter=settings.OTEL_TRACES_EXPORTER,
        endpoint=policy.endpoint,
    )

    from .metrics import _setup_meter_provider

    _otel_meter_provider = _setup_meter_provider(
        resource,
        exporter=settings.OTEL_METRICS_EXPORTER,
        endpoint=policy.endpoint,
    )

    from .logs import (
        _patch_loguru_sink,
        _setup_logger_provider,
    )

    if policy.logs_enabled:
        logger_provider = _setup_logger_provider(
            resource,
            exporter=settings.OTEL_LOGS_EXPORTER,
            endpoint=policy.endpoint,
        )
        if logger_provider is not None:
            _patch_loguru_sink(logger_provider)
            _otel_logger_provider = logger_provider

    from .instrument import (
        _setup_auto_instrumentation,
        reset_instrumentation_diagnostics,
    )

    reset_instrumentation_diagnostics()
    _setup_auto_instrumentation()

    _otel_initialized = True


def shutdown_otel() -> None:
    global _otel_initialized, _otel_tracer_provider, _otel_meter_provider, _otel_logger_provider  # noqa: PLW0603 — intentional module-level state for OTEL providers

    if _otel_logger_provider is not None:
        from .logs import _remove_loguru_sink

        _remove_loguru_sink()

    providers = (
        ("tracer_provider", _otel_tracer_provider),
        ("meter_provider", _otel_meter_provider),
        ("logger_provider", _otel_logger_provider),
    )
    for component, provider in providers:
        if provider is None:
            continue
        try:
            provider.force_flush(timeout_millis=10000)
            provider.shutdown()
        except Exception:  # noqa: BLE001 — shutdown must not crash even if provider fails
            logger.bind(component=component).exception("OTel provider shutdown failed")

    _otel_tracer_provider = None
    _otel_meter_provider = None
    _otel_logger_provider = None
    _otel_initialized = False
