from opentelemetry.sdk.resources import Resource

from app.config import get_settings
from app.utils import logger

_otel_initialized: bool = False
_otel_tracer_provider = None
_otel_meter_provider = None
_otel_logger_provider = None


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
    global _otel_initialized, _otel_tracer_provider, _otel_meter_provider, _otel_logger_provider  # noqa: PLW0603

    settings = get_settings()
    if not settings.OTEL_ENABLED or _otel_initialized:
        return

    resource = _build_resource(service_name)

    from .tracer import _setup_tracer_provider  # noqa: PLC0415

    _otel_tracer_provider = _setup_tracer_provider(resource, settings.OTEL_SAMPLE_RATE)

    from .metrics import _setup_meter_provider  # noqa: PLC0415

    _otel_meter_provider = _setup_meter_provider(resource)

    from .logs import _patch_loguru_sink, _setup_logger_provider  # noqa: PLC0415

    if settings.OTEL_LOGS_EXPORTER != "none":
        logger_provider = _setup_logger_provider(resource)
        if logger_provider is not None:
            _patch_loguru_sink(logger_provider)
            _otel_logger_provider = logger_provider

    from .instrument import _setup_auto_instrumentation  # noqa: PLC0415

    _setup_auto_instrumentation()

    _otel_initialized = True


def shutdown_otel() -> None:
    global _otel_initialized  # noqa: PLW0603

    if _otel_tracer_provider is not None:
        try:
            _otel_tracer_provider.force_flush(timeout_millis=10000)
            _otel_tracer_provider.shutdown()
        except Exception:  # noqa: BLE001
            logger.warning("OTel tracer provider shutdown failed")

    if _otel_meter_provider is not None:
        try:
            _otel_meter_provider.force_flush(timeout_millis=10000)
            _otel_meter_provider.shutdown()
        except Exception:  # noqa: BLE001
            logger.warning("OTel meter provider shutdown failed")

    if _otel_logger_provider is not None:
        try:
            _otel_logger_provider.force_flush(timeout_millis=10000)
            _otel_logger_provider.shutdown()
        except Exception:  # noqa: BLE001
            logger.warning("OTel logger provider shutdown failed")

    _otel_initialized = False
