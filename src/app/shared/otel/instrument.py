"""Centralized optional OpenTelemetry instrumentation and diagnostics."""

from collections.abc import Callable
from importlib import import_module
from typing import Any

from app.utils import logger

_diagnostics: list[dict[str, str]] = []


def reset_instrumentation_diagnostics() -> None:
    _diagnostics.clear()


def get_instrumentation_diagnostics() -> tuple[dict[str, str], ...]:
    return tuple(dict(item) for item in _diagnostics)


def _instrument(component: str, factory: Callable[[], Any]) -> None:
    try:
        factory().instrument()
    except Exception as exc:  # noqa: BLE001 — optional instrumentation must not crash app
        _diagnostics.append(
            {"component": component, "status": "failed", "error": str(exc)}
        )
        logger.bind(component=component, error=str(exc)).exception(
            "OTel auto-instrumentation failed, continuing"
        )
    else:
        _diagnostics.append({"component": component, "status": "enabled"})


def _instrument_path(component: str, module_name: str, class_name: str) -> None:
    def load() -> Any:
        return getattr(import_module(module_name), class_name)()

    _instrument(component, load)


def _setup_auto_instrumentation() -> None:
    _instrument_path(
        "sqlalchemy", "opentelemetry.instrumentation.sqlalchemy", "SQLAlchemyInstrumentor"
    )
    _instrument_path("redis", "opentelemetry.instrumentation.redis", "RedisInstrumentor")
    _instrument_path("httpx", "opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor")
    _instrument_path("celery", "opentelemetry.instrumentation.celery", "CeleryInstrumentor")

    # ASGI instrumentation is deliberately owned by the FastAPI app factory.
